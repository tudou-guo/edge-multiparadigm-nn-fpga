// spikformer_nlp_top - Spikformer NLP model-level top (Spikformer integer variant).
//
// Frozen external stream (contract hardware_abi.sample_sequence), per sample:
//   i_start (1-cycle pulse) ->
//   i_tab_*  : V*E/8 etab beats then N*E/8 ptab beats (64-bit words) -> u_emb
//   i_tok_*  : N beats of 16-bit token id (captured into top's token regs)
//   per block b=0..DEPTH-1:
//     4 descriptor beats on i_w_* :
//       beat0={thr_q,thr_k} beat1={thr_v,thr_proj} beat2={thr_fc1,thr_fc2}
//       beat3={32'h0,thr_e} (thr_e used for block 0 only)
//     then exactly WBEATS weight beats on i_w_* (frozen order), forwarded to
//     the shared spiking_block_top only after its i_start pulse has been given.
//     Block b then consumes TOKENS*E/TL captured spike beats produced here and
//     emits the same number of output beats; output beats of block b<DEPTH-1
//     become the input of block b+1.
//   after the last block: per-channel sum over T*N of the final residual value
//   -> o_sums[E] (OW-bit unsigned) + o_sums_valid + o_done.
//
// Handshake contract: every beat moves iff valid && ready in the SAME cycle;
// counters advance only then. Payloads are held stable while stalled. No
// ready-ahead-of-valid implicit contract. Second i_start reloads everything
// (tables, tokens, descriptors, weights) with no global reset.
//
// Embedding/threshold ordering detail: the embed frontend needs thr_e, which
// only arrives with block 0's descriptor beat 3 (frozen stream order), so the
// top buffers the N external token ids in regs during the token phase and
// feeds them into u_emb during block 0's weight-load phase, after thr_e has
// been latched. Block 0 is started only after BOTH its weights were accepted
// and the embed capture into tbuf completed; exactly one embed encode/sample.
module spikformer_nlp_top #(
  parameter int E = 128,
  parameter int HEADS = 4,
  parameter int HEAD_DIM = 32,
  parameter int HIDDEN = 512,
  parameter int T = 4,
  parameter int TOKENS = 32,
  parameter int DEPTH = 2,
  parameter int TL = 8,
  parameter int TC = 8,
  parameter int RW = 4,
  parameter int PSUM_SLICE = 32,
  parameter int SUM_W = 6,
  parameter int LANES = 8,
  parameter int VTAB = 4000,
  parameter int OW = 32
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic i_start,
  output logic o_done,
  input  logic [63:0] i_tab_data,
  input  logic i_tab_valid,
  output logic o_tab_ready,
  input  logic [15:0] i_tok_id,
  input  logic i_tok_valid,
  output logic o_tok_ready,
  input  logic [63:0] i_w_data,
  input  logic i_w_valid,
  output logic o_w_ready,
  output logic [OW-1:0] o_sums [E],
  output logic o_sums_valid
);
  localparam int G      = E/TL;        // block input/output groups per token
  localparam int EB     = E/LANES;     // embed beat groups per token
  localparam int NIN    = TOKENS*G;    // block in/out beats per block
  localparam int NBE    = TOKENS*EB;   // embed spike beats per sample
  localparam int WBEATS = 4*(E*E/8) + 4*(E/2) + (HIDDEN*E/8) + (HIDDEN/2)
                          + (E*HIDDEN/8) + (E/2);
  localparam int S_IDLE=0, S_TAB=1, S_TOK=2, S_DESC=3, S_WSTART=4, S_WLOAD=5,
                 S_WWAIT=6, S_IN=7, S_OUT=8, S_BDONE=9, S_SUMS=10, S_DONE=11;

  logic [3:0] st;

  // ---- captured sample state ----
  logic [15:0] tokbuf [TOKENS];
  // tbuf: single-write-port inferred synchronous-read RAM, one word per block
  // beat (TL channels x T timeslices x RW bits). Written either by the embed
  // capture during block 0 weight load, or by the block-out writeback that
  // feeds the next block; read by the block-input streamer.
  (* ram_style = "block" *)
  logic [TL*T*RW-1:0] tbuf [TOKENS*G];
  logic [OW-1:0] acc [E];
  logic signed [PSUM_SLICE-1:0] thr_q, thr_k, thr_v, thr_proj, thr_fc1, thr_fc2;
  logic signed [31:0] thr_e;

  // ---- counters ----
  logic [$clog2(TOKENS+1)-1:0] tcnt;
  logic [1:0] dcnt;
  logic [$clog2(WBEATS+1)-1:0] wcnt;
  logic [$clog2(NBE+1)-1:0] ebeat;
  logic [$clog2(TOKENS+1)-1:0] tokfeed;
  logic [$clog2(NIN+1)-1:0] iin, oout;   // linear block-in / block-out beat index
  logic [$clog2(G)-1:0] ogrp;            // output channel group (used for the final sum)
  logic [$clog2(DEPTH+1)-1:0] blkidx;
  logic in_data_vld;                     // block-in read data valid (1-cycle RAM latency)

  logic emb_start_r, blk_start_r, o_done_r, o_sums_valid_r;

  // ---- embed frontend ----
  logic emb_tab_ready, emb_ld_done, emb_tok_ready, emb_beat_valid, emb_beat_ready;
  logic emb_beat_last, emb_done;
  logic [LANES*T-1:0] emb_beat_data;
  logic emb_tok_valid;
  logic [15:0] emb_tok_id;

  // ---- shared block ----
  logic blk_done, blk_in_valid, blk_in_ready, blk_out_valid, blk_out_ready;
  logic [TL*T*RW-1:0] blk_in_data, blk_out_data;
  logic blk_out_last, blk_wvalid, blk_wready;
  logic [63:0] blk_wdata;

  integer a, k, tt2;   // sequential-block loop indices
  integer kw, tw;      // combinational tbuf write-payload loop indices
  integer bi;

  // tbuf single write port / single read port
  logic tb_we;
  logic [$clog2(NIN+1)-1:0] tb_waddr, tb_raddr;
  logic [TL*T*RW-1:0] tb_wdata, tb_rdata;
  logic tb_re;

  // ------------------------------------------------------------------
  // embedding frontend (table RAM load + token capture + LIF spike encode)
  // ------------------------------------------------------------------
  spiking_embed_top #(
    .V(VTAB), .N(TOKENS), .E(E), .T(T), .LANES(LANES), .ACC_W(PSUM_SLICE)
  ) u_emb (
    .s_clk(s_clk), .s_rst(s_rst), .i_start(emb_start_r),
    .i_tab_data(i_tab_data), .i_tab_valid(i_tab_valid), .o_tab_ready(emb_tab_ready),
    .o_ld_done(emb_ld_done),
    .i_tok_id(emb_tok_id), .i_tok_valid(emb_tok_valid), .o_tok_ready(emb_tok_ready),
    .i_thr_e(thr_e),
    .o_beat_valid(emb_beat_valid), .i_beat_ready(emb_beat_ready),
    .o_beat_data(emb_beat_data), .o_beat_last(emb_beat_last), .o_done(emb_done)
  );

  // ------------------------------------------------------------------
  // shared block datapath (weights + thresholds re-loaded for every block)
  // ------------------------------------------------------------------
  spiking_block_top #(
    .E(E), .HEADS(HEADS), .HEAD_DIM(HEAD_DIM), .HIDDEN(HIDDEN), .T(T),
    .TOKENS(TOKENS), .TL(TL), .TC(TC), .RW(RW), .W(8),
    .PSUM_SLICE(PSUM_SLICE), .SUM_W(SUM_W)
  ) u_blk (
    .s_clk(s_clk), .s_rst(s_rst), .i_start(blk_start_r), .o_done(blk_done),
    .i_in_valid(blk_in_valid), .o_in_ready(blk_in_ready), .i_in_data(blk_in_data),
    .o_out_valid(blk_out_valid), .i_out_ready(blk_out_ready),
    .o_out_data(blk_out_data), .o_out_last(blk_out_last),
    .i_w_data(blk_wdata), .i_w_valid(blk_wvalid), .o_w_ready(blk_wready),
    .i_thr_q(thr_q), .i_thr_k(thr_k), .i_thr_v(thr_v),
    .i_thr_proj(thr_proj), .i_thr_fc1(thr_fc1), .i_thr_fc2(thr_fc2)
  );

  // summed lane value of one 128-bit output beat (T timeslices of one lane)
  function automatic [OW-1:0] lane_sum(input [TL*T*RW-1:0] d, input integer lane);
    integer t2;
    begin
      lane_sum = {OW{1'b0}};
      for (t2 = 0; t2 < T; t2 = t2 + 1)
        lane_sum = lane_sum + d[(lane*T+t2+1)*RW-1 -: RW];
    end
  endfunction

  // ------------------------------------------------------------------
  // handshake outputs + combinational datapath
  // ------------------------------------------------------------------
  always_comb begin
    o_tab_ready = (st == S_TAB) && emb_tab_ready;
    o_tok_ready = (st == S_TOK);
    o_w_ready   = (st == S_DESC) || ((st == S_WLOAD) && blk_wready);

    // weight beat forwarded to the block only after its start pulse
    blk_wvalid = (st == S_WLOAD) && i_w_valid;
    blk_wdata  = i_w_data;

    // block input: captured tbuf word, read from the synchronous-read RAM
    blk_in_valid = (st == S_IN) && in_data_vld;
    blk_in_data  = tb_rdata;

    // block output beats are consumed in the output-collect state
    blk_out_ready = (st == S_OUT);

    // capture embed spike beats while block 0's weights are streaming
    emb_beat_ready = ((st == S_WLOAD) || (st == S_WWAIT)) && (blkidx == 0) && !emb_done;
    // feed buffered token ids into u_emb (after block 0 descriptor = thr_e known)
    emb_tok_valid = ((st == S_WLOAD) || (st == S_WWAIT)) && (blkidx == 0)
                    && (tokfeed < TOKENS);
    emb_tok_id    = (tokfeed < TOKENS) ? tokbuf[tokfeed] : 16'd0;

    // ---- tbuf write port: embed capture | block-out writeback ----
    tb_we    = 1'b0;
    tb_waddr = '0;
    tb_wdata = '0;
    if (((st == S_WLOAD) || (st == S_WWAIT)) && emb_beat_valid && emb_beat_ready) begin
      tb_we    = 1'b1;
      tb_waddr = ebeat;
      for (kw = 0; kw < LANES; kw = kw + 1)
        for (tw = 0; tw < T; tw = tw + 1)
          tb_wdata[(kw*T+tw+1)*RW-1 -: RW] = {{(RW-1){1'b0}}, emb_beat_data[kw*T+tw]};
    end else if ((st == S_OUT) && (blkidx != DEPTH-1) && blk_out_valid && blk_out_ready) begin
      tb_we    = 1'b1;
      tb_waddr = oout;
      tb_wdata = blk_out_data;
    end
    // ---- tbuf read port: current block-in word, prefetch next on handshake ----
    tb_re    = (st == S_IN);
    tb_raddr = iin;
    if ((st == S_IN) && in_data_vld && blk_in_ready && (iin != NIN-1))
      tb_raddr = iin + 1'b1;
  end

  assign o_done        = o_done_r;
  assign o_sums_valid  = o_sums_valid_r;

  // ------------------------------------------------------------------
  // sample sequencer
  // ------------------------------------------------------------------
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin
      st <= S_IDLE;
      tcnt <= 0; dcnt <= 0; wcnt <= 0; ebeat <= 0;
      tokfeed <= 0; iin <= 0; oout <= 0; ogrp <= 0; blkidx <= 0;
      in_data_vld <= 1'b0;
      emb_start_r <= 1'b0; blk_start_r <= 1'b0;
      o_done_r <= 1'b0; o_sums_valid_r <= 1'b0;
      thr_q <= '0; thr_k <= '0; thr_v <= '0; thr_proj <= '0;
      thr_fc1 <= '0; thr_fc2 <= '0; thr_e <= '0;
      for (a = 0; a < E; a = a + 1) acc[a] <= '0;
      for (a = 0; a < E; a = a + 1) o_sums[a] <= '0;
    end else begin
      emb_start_r <= 1'b0;
      blk_start_r <= 1'b0;
      case (st)
        S_IDLE: begin
          o_done_r <= 1'b0;
          o_sums_valid_r <= 1'b0;
          if (i_start) begin
            tcnt <= 0; dcnt <= 0; wcnt <= 0; ebeat <= 0;
            tokfeed <= 0; iin <= 0; oout <= 0; ogrp <= 0; blkidx <= 0;
            in_data_vld <= 1'b0;
            for (a = 0; a < E; a = a + 1) acc[a] <= '0;
            emb_start_r <= 1'b1;
            st <= S_TAB;
          end
        end
        S_TAB: begin
          if (emb_ld_done) begin
            st <= S_TOK;
            tcnt <= 0;
          end
        end
        S_TOK: begin
          if (i_tok_valid && o_tok_ready) begin
            tokbuf[tcnt] <= i_tok_id;
            if (tcnt == TOKENS-1) begin
              st <= S_DESC;
              dcnt <= 0;
            end else tcnt <= tcnt + 1'b1;
          end
        end
        S_DESC: begin
          // descriptor beats are consumed by the top itself (not forwarded)
          if (i_w_valid && o_w_ready) begin
            case (dcnt)
              2'd0: begin
                // beat0 = {thr_q[31:0], thr_k[31:0]}
                thr_q <= i_w_data[63:32];
                thr_k <= i_w_data[31:0];
              end
              2'd1: begin
                // beat1 = {thr_v[31:0], thr_proj[31:0]}
                thr_v <= i_w_data[63:32];
                thr_proj <= i_w_data[31:0];
              end
              2'd2: begin
                // beat2 = {thr_fc1[31:0], thr_fc2[31:0]}
                thr_fc1 <= i_w_data[63:32];
                thr_fc2 <= i_w_data[31:0];
              end
              default: begin
                // beat3: thr_e is meaningful for block 0 only
                if (blkidx == 0) thr_e <= i_w_data[31:0];
                blk_start_r <= 1'b1;
                st <= S_WSTART;
              end
            endcase
            if (dcnt != 2'd3) dcnt <= dcnt + 1'b1;
          end
        end
        S_WSTART: begin
          // 1-cycle block start pulse precedes the first forwarded weight beat
          st <= S_WLOAD;
        end
        S_WLOAD: begin
          // token feed to the embed frontend (block 0 only, exactly N beats)
          if (emb_tok_valid && emb_tok_ready) tokfeed <= tokfeed + 1'b1;
          // capture embed spike beats into tbuf (write is the combinational port)
          if (emb_beat_valid && emb_beat_ready) begin
            if (ebeat == NBE-1) ebeat <= 0;
            else ebeat <= ebeat + 1'b1;
          end
          // weight beats forwarded to the block, only valid&&ready consumed
          if (i_w_valid && o_w_ready) begin
            if (wcnt == WBEATS-1) begin
              wcnt <= 0;
              st <= S_WWAIT;
            end else wcnt <= wcnt + 1'b1;
          end
        end
        S_WWAIT: begin
          if (emb_tok_valid && emb_tok_ready) tokfeed <= tokfeed + 1'b1;
          if (emb_beat_valid && emb_beat_ready) begin
            if (ebeat == NBE-1) ebeat <= 0;
            else ebeat <= ebeat + 1'b1;
          end
          // block 0 runs only after its weights AND the embed capture are done
          if (((blkidx != 0) || emb_done) && (tokfeed == TOKENS)) begin
            st <= S_IN;
            iin <= 0;
            in_data_vld <= 1'b0;   // first tbuf word appears one cycle later
          end
        end
        S_IN: begin
          in_data_vld <= 1'b1;
          if (blk_in_valid && blk_in_ready) begin
            if (iin == NIN-1) begin
              iin <= 0;
              in_data_vld <= 1'b0;
              oout <= 0;
              ogrp <= 0;
              st <= S_OUT;
            end else iin <= iin + 1'b1;
          end
        end
        S_OUT: begin
          if (blk_out_valid && blk_out_ready) begin
            if (blkidx == DEPTH-1) begin
              // final residual: per-channel sum over T*N (o_sums)
              for (k = 0; k < TL; k = k + 1)
                acc[ogrp*TL+k] <= acc[ogrp*TL+k] + lane_sum(blk_out_data, k);
            end
            // non-last blocks: the combinational tbuf write port stores this beat
            if (blk_out_last) begin
              if (blkidx == DEPTH-1) st <= S_SUMS;
              else st <= S_BDONE;
            end else begin
              oout <= oout + 1'b1;
              if (ogrp == G-1) ogrp <= 0;
              else ogrp <= ogrp + 1'b1;
            end
          end
        end
        S_BDONE: begin
          if (blk_done) begin
            blkidx <= blkidx + 1'b1;
            dcnt <= 0;
            wcnt <= 0;
            st <= S_DESC;
          end
        end
        S_SUMS: begin
          for (a = 0; a < E; a = a + 1) o_sums[a] <= acc[a];
          o_sums_valid_r <= 1'b1;
          o_done_r <= 1'b1;
          st <= S_DONE;
        end
        S_DONE: begin
          // hold o_sums_valid/o_done until the next i_start (one assertion/sample)
          if (i_start) begin
            tcnt <= 0; dcnt <= 0; wcnt <= 0; ebeat <= 0;
            tokfeed <= 0; iin <= 0; oout <= 0; ogrp <= 0; blkidx <= 0;
            in_data_vld <= 1'b0;
            for (a = 0; a < E; a = a + 1) acc[a] <= '0;
            o_done_r <= 1'b0;
            o_sums_valid_r <= 1'b0;
            emb_start_r <= 1'b1;
            st <= S_TAB;
          end
        end
        default: st <= S_IDLE;
      endcase
    end
  end

  // tbuf storage: inferred synchronous-read, single-write-port RAM.
  // No per-element reset (contents are always written before being read).
  always_ff @(posedge s_clk) begin
    if (tb_we) tbuf[tb_waddr] <= tb_wdata;
  end
  always_ff @(posedge s_clk) begin
    if (tb_re) tb_rdata <= tbuf[tb_raddr];
  end

  // parameter sanity for the frozen stream ABI
  generate
    if (LANES != 8) begin : g_lanes_assert
      initial $error("spikformer_nlp_top requires LANES==8 for the frozen E/8-word table ABI");
    end
    if (E % TL != 0 || E % LANES != 0) begin : g_div_assert
      initial $error("spikformer_nlp_top requires E divisible by TL and LANES");
    end
    if (TL != LANES || (E/LANES) != G) begin : g_tbuf_assert
      initial $error("spikformer_nlp_top tbuf word layout requires TL==LANES and E/LANES==E/TL");
    end
    if (NBE != NIN) begin : g_beat_assert
      initial $error("spikformer_nlp_top requires TOKENS*E/LANES == TOKENS*E/TL");
    end
  endgenerate
endmodule