// Spikformer vision top (image classification).
//
//   image H x W x C0
//     -> SPS: conv3x3(C0->C1)+LIF -> 2x maxpool(stride2)
//             -> conv3x3(C1->C2)+LIF                (C2 == E, token width)
//     -> spiking_block_top (one block, DEPTH=1 in this cut)
//     -> per-channel sum over T*N tokens  (PS then FP32 Linear + argmax)
//
// Weight handling: SPS weights are held in on-chip ROMs because the SPS
// re-reads the conv2 tile window once per token (address-generator modulo).
// The block weight stream is stored in a ROM too and replayed into the block's
// loader, so the host can write everything once.
//
// Scope: replay-verified composition at the validated small geometry
// (H=8, C1=16, C2=32, T=2 -> 4 tokens x 32 channels). Native 32x32/384 needs
// the DDR streaming path used by the NLP platform top and is the next step.
module spikformer_vis_top #(
  parameter int H = 8,
  parameter int W_IN = 8,
  parameter int C0 = 3,
  parameter int C1 = 16,
  parameter int C2 = 32,
  parameter int T = 2,
  parameter int LANES = 8,
  parameter int TC2 = 4,
  parameter int E = 32,
  parameter int HEADS = 1,
  parameter int HEAD_DIM = 32,
  parameter int HIDDEN = 64,
  parameter int TL = 8,
  parameter int TC = 8,
  parameter int RW = 4,
  parameter int PSUM_SLICE = 32,
  parameter int SUM_W = 6,
  parameter int W = 8,
  parameter int SPS_ACC_W = 21,   // SPS accumulator width (golden packs biases at 21 bit)
  parameter int OW = 32,
  parameter int NTOK = (H/4)*(W_IN/4),
  parameter int W1B = (C1*C0*9+7)/8,
  parameter int BIB = (C1 > C2 ? C1 : C2),
  // conv2 tile set ROM must hold one tile window per token position: the SPS
  // re-reads the window for every (pr,pc,grp2,tile2) load, i.e. NTOK times the
  // per-position tile set, not a single position.
  parameter int W2B = NTOK*((C2*C1*9+7)/8),
  parameter int BLKW = (4*E*E/8) + (4*E/2) + (HIDDEN*E/8) + (HIDDEN/2) + (E*HIDDEN/8) + (E/2)
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic i_start,
  output logic o_done,
  input  logic [C0*8-1:0] i_pix_data,
  input  logic i_pix_valid,
  output logic o_pix_ready,
  input  logic [63:0] i_w_data,
  input  logic i_w_valid,
  output logic o_w_ready,
  input  logic signed [PSUM_SLICE-1:0] i_thr_1,
  input  logic signed [PSUM_SLICE-1:0] i_thr_2,
  input  logic signed [PSUM_SLICE-1:0] i_thr_q,
  input  logic signed [PSUM_SLICE-1:0] i_thr_k,
  input  logic signed [PSUM_SLICE-1:0] i_thr_v,
  input  logic signed [PSUM_SLICE-1:0] i_thr_proj,
  input  logic signed [PSUM_SLICE-1:0] i_thr_fc1,
  input  logic signed [PSUM_SLICE-1:0] i_thr_fc2,
  output logic [OW-1:0] o_sums [E],
  output logic o_sums_valid
);
  localparam int G2 = C2/LANES;
  localparam int G  = E/TL;
  localparam int SPSW = W1B + BIB + W2B;
  localparam int S_BW=0, S_W1=1, S_BI=2, S_PIX=3, S_TILE=4, S_TREP=5,
                     S_START=6, S_RUN=7, S_HEAD=8, S_DONE=9;

  logic [3:0] st;   // 10 states (S_BW..S_DONE), must not truncate S_HEAD/S_DONE
  logic [63:0] w1rom [W1B];
  logic [63:0] brom  [BIB];
  logic [63:0] w2rom [W2B];
  logic [63:0] blkrom[BLKW];
  logic [RW-1:0] tbuf [NTOK][E][T];

  // ---- SPS ----
  logic [LANES*T-1:0] sps_tok;
  logic sps_tvalid, sps_tready, sps_tlast;
  logic sps_wready, sps_pready;
  logic [63:0] sps_wdata;
  logic sps_wvalid;
  logic sps_pvalid;
  logic [C0*8-1:0] sps_pdata;
  assign sps_pvalid = i_pix_valid;
  assign sps_pdata  = i_pix_data;
  assign o_pix_ready = sps_pready;
  spiking_sps_top #(.H(H), .W_IN(W_IN), .C0(C0), .C1(C1), .C2(C2), .T(T),
                    .LANES(LANES), .TC2(TC2), .WT_W(W), .ACC_W(SPS_ACC_W),
                    .BUS_W(64)) u_sps (
    .s_clk(s_clk), .s_rst(s_rst),
    .i_in_ch_1(16'd0), .i_out_ch_1(16'd0), .i_thr_1(i_thr_1[15:0]),
    .i_in_ch_2(16'd0), .i_out_ch_2(16'd0), .i_thr_2(i_thr_2[15:0]),
    .i_weight_in(sps_wdata), .i_weight_valid(sps_wvalid), .o_weight_ready(sps_wready),
    .i_pix_valid(sps_pvalid), .o_pix_ready(sps_pready), .i_pix_data(sps_pdata),
    .o_tok_valid(sps_tvalid), .i_tok_ready(sps_tready),
    .o_tok_data(sps_tok), .o_tok_last(sps_tlast)
  );

  // ---- block ----
  logic blk_start, blk_done, blk_iv, blk_irdy, blk_ov, blk_ordy;
  logic [TL*T*RW-1:0] blk_idata, blk_odata;
  logic blk_olast;
  logic [63:0] blk_wdata;
  logic blk_wvalid, blk_wready;

  spiking_block_top #(.E(E), .HEADS(HEADS), .HEAD_DIM(HEAD_DIM), .HIDDEN(HIDDEN),
    .T(T), .TOKENS(NTOK), .TL(TL), .TC(TC), .RW(RW), .W(W),
    .PSUM_SLICE(PSUM_SLICE), .SUM_W(SUM_W)) u_blk (
    .s_clk(s_clk), .s_rst(s_rst), .i_start(blk_start), .o_done(blk_done),
    .i_in_valid(blk_iv), .o_in_ready(blk_irdy), .i_in_data(blk_idata),
    .o_out_valid(blk_ov), .i_out_ready(blk_ordy), .o_out_data(blk_odata),
    .o_out_last(blk_olast),
    .i_w_data(blk_wdata), .i_w_valid(blk_wvalid), .o_w_ready(blk_wready),
    .i_thr_q(i_thr_q), .i_thr_k(i_thr_k), .i_thr_v(i_thr_v),
    .i_thr_proj(i_thr_proj), .i_thr_fc1(i_thr_fc1), .i_thr_fc2(i_thr_fc2)
  );

  // ---- counters / bookkeeping ----
  logic [$clog2(BLKW+1)-1:0]     bcnt;      // block weight pointer
  logic [$clog2(W1B+1)-1:0]      c1cnt;     // conv1 weight pointer
  // bias pairs (BIB) and conv2 tiles (W2B) share this pointer: size it for
  // the larger of the two, otherwise it overflows and the phase never ends.
  logic [$clog2((BIB > W2B ? BIB : W2B)+1)-1:0] bicnt;
  logic [$clog2(NTOK*G2+1)-1:0]  tokb;      // SPS token beat index
  logic [$clog2(NTOK*G+1)-1:0]   runi;      // block input beat index
  logic [$clog2(NTOK*G)-1:0]     capi;      // block output beat index (TOKENS*G beats)
  logic [$clog2(NTOK)-1:0]       sumtok;
  logic [3:0] startd;
  logic wloaded;
  logic [OW-1:0] acc [E];
  integer a, k, t2;

  // The block weight stream is replayed from blkrom during S_RUN; without this
  // driver the block loader received X for every weight beat.
  assign blk_wdata = blkrom[bcnt];

  always_comb begin
    blk_idata = '0;
    for (int kk = 0; kk < TL; kk = kk + 1)
      for (int tt = 0; tt < T; tt = tt + 1)
        blk_idata[(kk*T+tt+1)*RW-1 -: RW] = tbuf[runi/G][(runi%G)*TL+kk][tt];
  end
  wire [$clog2(NTOK*G2+1)-1:0] tok_of  = tokb / G2;
  wire [$clog2(G2+1)-1:0]      grp_of  = tokb % G2;

  // Host-facing ready and the SPS weight valid are combinational so the
  // handshake cannot skew by a cycle (a registered ready silently drops beats).
  always_comb begin
    case (st)
      S_BW, S_TILE: o_w_ready = 1'b1;
      S_W1, S_BI:   o_w_ready = sps_wready;
      default:      o_w_ready = 1'b0;
    endcase
  end
  assign sps_wvalid = ((st == S_W1) || (st == S_BI)) ? i_w_valid :
                      (st == S_TREP) ? 1'b1 : 1'b0;

  // payload presented to the SPS on the weight bus, selected by host phase
  always_comb begin
    // host phases carry their own beat; the tile replay reads the ROM
    if (st == S_TREP) sps_wdata = w2rom[bicnt];
    else              sps_wdata = i_w_data;
  end

  // ---------------------------------------------------------------------
  // Host phases, in the exact order the SPS replay bench uses:
  //   S_BW  : block weight stream -> blkrom            (BLKW beats)
  //   S_W1  : conv1 weight tiles, passed straight to the SPS   (W1B beats)
  //   S_BI  : bias pairs, passed straight to the SPS           (BIB beats)
  //   S_PIX : image pixels, passed straight to the SPS   (H*W_IN beats)
  //   S_TILE: conv2 tile set -> w2rom (the SPS stalls here)    (W2B beats)
  //   S_TREP: replay the tile window once per token, capture tokens
  //   then start the block, feed tokens, run it, sum.
  // ---------------------------------------------------------------------
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin
      st <= S_BW; bcnt <= 0; c1cnt <= 0; bicnt <= 0; tokb <= 0; runi <= 0;
      capi <= 0; sumtok <= 0; startd <= 0; wloaded <= 1'b0;
      o_done <= 1'b0; o_sums_valid <= 1'b0; sps_tready <= 1'b0;
      blk_start <= 1'b0; blk_iv <= 1'b0; blk_ordy <= 1'b0; blk_wvalid <= 1'b0;
      for (a = 0; a < E; a = a + 1) acc[a] = 0;
    end else case (st)
      S_BW: begin
        if (i_w_valid) begin
          blkrom[bcnt] <= i_w_data;
          if (bcnt == BLKW-1) begin bcnt <= 0; st <= S_W1; end
          else bcnt <= bcnt + 1'b1;
        end
      end
      S_W1: begin
        if (i_w_valid && sps_wready) begin
          w1rom[c1cnt] <= i_w_data;
          if (c1cnt == W1B-1) begin c1cnt <= 0; st <= S_BI; end
          else c1cnt <= c1cnt + 1'b1;
        end
      end
      S_BI: begin
        if (i_w_valid && sps_wready) begin
          brom[bicnt] <= i_w_data;
          if (bicnt == BIB-1) begin bicnt <= 0; st <= S_PIX; end
          else bicnt <= bicnt + 1'b1;
        end
      end
      S_PIX: begin
        if (i_pix_valid && sps_pready) begin
          if (c1cnt == H*W_IN-1) begin c1cnt <= 0; st <= S_TILE; end
          else c1cnt <= c1cnt + 1'b1;
        end
      end
      S_TILE: begin
        if (i_w_valid) begin
          w2rom[bicnt] <= i_w_data;
          if (bicnt == W2B-1) begin bicnt <= 0; sps_tready <= 1'b1; st <= S_TREP; end
          else bicnt <= bicnt + 1'b1;
        end
      end
      S_TREP: begin
        // supply the tile window to the SPS, looping over it per token
        if (sps_wready) begin
          if (bicnt == W2B-1) bicnt <= 0; else bicnt <= bicnt + 1'b1;
        end
        if (sps_tvalid && sps_tready) begin
          for (k = 0; k < LANES; k = k + 1)
            for (t2 = 0; t2 < T; t2 = t2 + 1)
              tbuf[tok_of][grp_of*LANES+k][t2] <= sps_tok[k*T+t2];
          if (sps_tlast) begin
            sps_tready <= 1'b0; 
            blk_start <= 1'b1; startd <= 0; st <= S_START;
          end else tokb <= tokb + 1'b1;
        end
      end
      S_START: begin
        blk_start <= 1'b0;
        for (a = 0; a < E; a = a + 1) acc[a] = 0;
        startd <= startd + 1'b1;
        if (startd >= 4'd3) begin bcnt <= 0; blk_wvalid <= 1'b1; st <= S_RUN; end
      end
      S_RUN: begin
        if (!wloaded) begin
          // hold valid until the block accepts; advance the ROM pointer only on
          // a valid&&ready beat (advancing on ready alone skipped blkrom[0]).
          blk_wvalid <= 1'b1;
          if (blk_wvalid && blk_wready) begin
            if (bcnt == BLKW-1) begin bcnt <= 0; blk_wvalid <= 1'b0; wloaded <= 1'b1; runi <= 0; end
            else bcnt <= bcnt + 1'b1;
          end
        end else if (runi < NTOK*G) begin
          blk_iv <= 1'b1;
          if (blk_iv && blk_irdy) begin
            if (runi == NTOK*G-1) begin blk_iv <= 1'b0; blk_ordy <= 1'b1; capi <= 0; end
            else runi <= runi + 1'b1;
          end
        end
        if (blk_ov && blk_ordy) begin
          for (k = 0; k < TL; k = k + 1)
            for (t2 = 0; t2 < T; t2 = t2 + 1)
              tbuf[capi/G][(capi%G)*TL+k][t2] <= blk_odata[(k*T+t2+1)*RW-1 -: RW];
          if (blk_olast) begin blk_ordy <= 1'b0; capi <= 0; sumtok <= 0; st <= S_HEAD; end
          else capi <= capi + 1'b1;
        end
      end
      S_HEAD: begin
        for (a = 0; a < E; a = a + 1)
          for (t2 = 0; t2 < T; t2 = t2 + 1)
            acc[a] = acc[a] + {{(OW-RW){1'b0}}, tbuf[sumtok][a][t2]};
        if (sumtok == NTOK-1) begin
          for (a = 0; a < E; a = a + 1) o_sums[a] <= acc[a];
          o_sums_valid <= 1'b1; o_done <= 1'b1; st <= S_DONE;
        end else sumtok <= sumtok + 1'b1;
      end
      S_DONE: begin
        o_done <= 1'b1;
        if (!i_start) begin
          o_done <= 1'b0; o_sums_valid <= 1'b0; st <= S_BW; bcnt <= 0; 
        end
      end
      default: st <= S_BW;
    endcase
  end
endmodule
