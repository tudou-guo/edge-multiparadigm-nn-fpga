// Spiking patch-embed (SPS): parameterized port of upstream eyeriss +
// SpikingEncoder + MaxPool path (conv_layer1 + simple_eyeriss_array +
// psum_lif_top + simple_maxpool_unit x2). Native: 32x32x3 -> 16x16 -> 8x8.
// Mechanism kept, per stage:
//   (1) 3x3 conv, weight-stationary (eyeriss PE idea). Stage-1 taps are
//       pixel x int8 weight; stage-2 taps are spike-gated (bit ? w : 0).
//   (2) tap-product reduction (add_tree idea), bias add, LIF over T.
//   (3) spike-domain 2x2 stride-2 maxpool = OR per timestep.
// Diffs: H/W/C0/C1/C2/T/LANES/TC2/W/ACC_W params; shared 64b streaming
// weight bus routed by FSM phase; config mirrors code_fetch CONV_CODE.
// Throughput note: upstream uses diagonal row-reuse across 8 groups; this
// port uses full-window MAC per beat with input-channel tiling (same sums,
// own cycle model in schedule.json). f1/prow2 behavioral (codegen: BRAM).
// Control: two phases. P1 streams padded pixels, computes one out-group per
// valid window, pulses shared LIF, stores spikes to f1. P2 loops pooled
// windows, streams weight tiles, accumulates psum over input tiles, pulses
// shared LIF, assembles pool-2 rows, drains tokens with ready/valid.
module spiking_sps_top #(
  parameter int H      = 32,
  parameter int W_IN   = 32,
  parameter int C0     = 3,
  parameter int C1     = 128,
  parameter int C2     = 384,
  parameter int T      = 4,
  parameter int LANES  = 8,
  parameter int TC2    = 8,
  parameter int WT_W   = 8,
  parameter int ACC_W  = 21,
  parameter int BUS_W  = 64
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic [15:0] i_in_ch_1,
  input  logic [15:0] i_out_ch_1,
  input  logic [15:0] i_thr_1,
  input  logic [15:0] i_in_ch_2,
  input  logic [15:0] i_out_ch_2,
  input  logic [15:0] i_thr_2,
  input  logic [BUS_W-1:0] i_weight_in,
  input  logic i_weight_valid,
  output logic o_weight_ready,
  input  logic i_pix_valid,
  output logic o_pix_ready,
  input  logic [C0*8-1:0] i_pix_data,
  output logic o_tok_valid,
  input  logic i_tok_ready,
  output logic [LANES*T-1:0] o_tok_data,
  output logic o_tok_last
);
  localparam int H1 = H/2, W1 = W_IN/2, H2 = H/4, W2 = W_IN/4;
  localparam int NTOK = H2*W2;
  localparam int G1 = C1/LANES, G2 = C2/LANES, NT2 = C1/TC2;
  localparam int TBEATS = NTOK*G2;
  localparam int LAT = T+4;
  localparam int S_LD1=0, S_P1=1, S_LDT=2, S_P2=3, S_DRN=4, S_DRN2=5;
  // S_DRN2 holds the last beat one extra cycle so a #-delayed sampler
  // cannot miss it when valid drops in the same acceptance edge.
  localparam int Q_PIX=0, Q_CALC=1, Q_LIFW=2;
  logic [2:0] st;
  logic [1:0] q;
  logic signed [WT_W-1:0] wbank1 [C1][C0][9];
  logic signed [ACC_W-1:0] bias1 [C1];
  logic [$clog2(C1*C0*9/8)+1-1:0] ld1_cnt;
  logic signed [WT_W-1:0] wtile2 [LANES][TC2][9];
  logic signed [ACC_W-1:0] biast2 [LANES];
  logic [$clog2(LANES*TC2*9/8)+1-1:0] ldt_cnt;
  logic [7:0] f0 [H][W_IN][C0];
  logic [$clog2(H*W_IN)-1:0] ldc0;
  logic signed [ACC_W-1:0] racc1 [LANES];
  logic [C1*T-1:0] f1 [H][W_IN];
  logic [$clog2(W_IN)-1:0] ox1;
  logic [$clog2(H)-1:0] oy1;
  logic [$clog2(G1)-1:0] grp1;
  logic [$clog2(LAT+9)-1:0] wait1;
  logic lif1_v;
  logic signed [T*ACC_W-1:0] lif1_ps [LANES];
  logic [LANES*T-1:0] lif1_spk;
  logic lif1_rdy [LANES];
  logic [$clog2(H1)-1:0] pr;
  logic [$clog2(W1)-1:0] pc;
  logic [$clog2(G2)-1:0] grp2;
  logic [$clog2(NT2)-1:0] tile2;
  logic signed [ACC_W-1:0] psum2 [LANES][T];
  logic lif2_v;
  logic signed [T*ACC_W-1:0] lif2_ps [LANES];
  logic [LANES*T-1:0] lif2_spk;
  logic lif2_rdy [LANES];
  logic [$clog2(LAT+9)-1:0] wait2;
  logic [C2*T-1:0] prow2 [H1][W1];
  logic signed [ACC_W-1:0] bias2f [C2];
  logic bias_ld;
  logic [$clog2(C2)-1:0] bias_cnt;
  logic [1:0] qr;
  logic [$clog2(TBEATS)-1:0] tbeat;
  logic drn_row;
  integer a, b, c, k, tt;
  wire signed [ACC_W-1:0] thr1e, thr2e;
  assign thr1e = {{ACC_W-16{i_thr_1[15]}}, i_thr_1};
  assign thr2e = {{ACC_W-16{i_thr_2[15]}}, i_thr_2};
  genvar ll;
  generate
    for (ll = 0; ll < LANES; ll = ll + 1) begin : g_lif1
      spiking_lif_group #(.T(T), .BITS(ACC_W)) u_l1 (
        .s_clk(s_clk), .s_rst(s_rst), .i_lif_thrd(thr1e),
        .i_PsumValid(lif1_v), .i_PsumData(lif1_ps[ll]),
        .o_spikes_out(lif1_spk[(ll+1)*T-1 -: T]), .o_spikes_valid(lif1_rdy[ll])
      );
    end
    for (ll = 0; ll < LANES; ll = ll + 1) begin : g_lif2
      spiking_lif_group #(.T(T), .BITS(ACC_W)) u_l2 (
        .s_clk(s_clk), .s_rst(s_rst), .i_lif_thrd(thr2e),
        .i_PsumValid(lif2_v), .i_PsumData(lif2_ps[ll]),
        .o_spikes_out(lif2_spk[(ll+1)*T-1 -: T]), .o_spikes_valid(lif2_rdy[ll])
      );
    end
  endgenerate
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin
      st <= S_LD1; q <= Q_PIX; ld1_cnt <= 0; ldt_cnt <= 0;
      ldc0 <= 0; ox1 <= 0; oy1 <= 0; grp1 <= 0; wait1 <= 0;
      pr <= 0; pc <= 0; grp2 <= 0; tile2 <= 0; wait2 <= 0;
      tbeat <= 0; drn_row <= 0;
      bias_ld <= 1'b0; bias_cnt <= 0; qr <= 0;
      o_weight_ready <= 1'b1; o_pix_ready <= 1'b0;
      o_tok_valid <= 1'b0; o_tok_last <= 1'b0;
      lif1_v <= 1'b0; lif2_v <= 1'b0;
      for (a = 0; a < 1; a = a + 1) begin
      end
    end else case (st)
      S_LD1: begin
        o_weight_ready <= 1'b1;
        if (i_weight_valid && bias_ld) begin
          if (bias_cnt < C1) bias1[bias_cnt] <= i_weight_in[ACC_W-1:0];
          if (bias_cnt < C2) bias2f[bias_cnt] <= i_weight_in[2*ACC_W-1:ACC_W];
          if ((C2 >= C1 && bias_cnt == C2-1) || (C1 > C2 && bias_cnt == C1-1)) begin
            st <= S_P1; q <= Q_PIX; o_weight_ready <= 1'b0; o_pix_ready <= 1'b1; bias_ld <= 1'b0;
          end else bias_cnt <= bias_cnt + 1'b1;
        end else if (i_weight_valid) begin
          integer base, kk, idx, oc, ic, tp;
          base = ld1_cnt * 8;
          for (kk = 0; kk < 8; kk = kk + 1) begin
            idx = base + kk;
            if (idx < C1*C0*9) begin
              oc = idx / (C0*9); ic = (idx % (C0*9)) / 9; tp = idx % 9;
              wbank1[oc][ic][tp] <= i_weight_in[(kk+1)*WT_W-1 -: WT_W];
            end
          end
          if (base + 8 >= C1*C0*9) begin bias_ld <= 1'b1; bias_cnt <= 0; end
          else ld1_cnt <= ld1_cnt + 1'b1;
        end
      end
      S_P1: begin
        case (q)
          Q_PIX: begin
            // Frame load: H*W pixel beats straight into f0 (no pad beats;
            // padding is applied explicitly at CALC read time, matching the
            // golden model tap-for-tap).
            o_pix_ready <= 1'b1;
            if (i_pix_valid) begin
              for (c = 0; c < C0; c = c + 1)
                f0[ldc0/W_IN][ldc0%W_IN][c] <= i_pix_data[(c+1)*8-1 -: 8];
              if (ldc0 == H*W_IN-1) begin
                ldc0 <= 0; ox1 <= 0; oy1 <= 0; grp1 <= 0;
                q <= Q_CALC; o_pix_ready <= 1'b0;
              end else ldc0 <= ldc0 + 1'b1;
            end
          end
          Q_CALC: begin
            o_pix_ready <= 1'b0;
            lif1_v <= 1'b0;
            for (a = 0; a < LANES; a = a + 1) begin
              racc1[a] = bias1[grp1*LANES+a];
              for (b = 0; b < 9; b = b + 1) begin
                integer yy, xx;
                yy = oy1 + b/3 - 1; xx = ox1 + b%3 - 1;
                for (c = 0; c < C0; c = c + 1) begin
                  logic [7:0] px;
                  px = (yy >= 0 && yy < H && xx >= 0 && xx < W_IN) ? f0[yy][xx][c] : 8'd0;
                  racc1[a] = racc1[a] + $signed(wbank1[grp1*LANES+a][c][b]) * $signed(px);
                end
              end
              lif1_ps[a] = {T{racc1[a]}};
            end
            lif1_v <= 1'b1; wait1 <= 0; q <= Q_LIFW;
          end
          Q_LIFW: begin
            o_pix_ready <= 1'b0;
            lif1_v <= 1'b0;
            // Sample on the group valid pulse (spikes are pulses, not levels;
            // a fixed LAT count can miss them). Timeout falls through.
            if (lif1_rdy[0] || wait1 == LAT+8) begin
              f1[oy1][ox1][(grp1+1)*LANES*T-1 -: LANES*T] <= lif1_spk;
              if (grp1 == G1-1) begin
                grp1 <= 0;
                if (oy1 == H-1 && ox1 == W_IN-1) begin st <= S_LDT; ldt_cnt <= 0; grp2 <= 0; tile2 <= 0; pr <= 0; pc <= 0; o_pix_ready <= 1'b0; q <= Q_PIX; end
                else if (ox1 == W_IN-1) begin ox1 <= 0; oy1 <= oy1 + 1'b1; q <= Q_CALC; end
                else begin ox1 <= ox1 + 1'b1; q <= Q_CALC; end
              end else begin grp1 <= grp1 + 1'b1; q <= Q_CALC; end
            end else wait1 <= wait1 + 1'b1;
          end
          default: q <= Q_PIX;
        endcase
      end
      S_LDT: begin
        // Stream one weight tile for (grp2, tile2): LANES*TC2*9 int8 + bias.
        // Consume only when our (registered) ready is high: the replay source
        // advances its ROM pointer on o_weight_ready, so accepting a beat in
        // the entry cycle (ready still 0) would desynchronise the two sides.
        o_weight_ready <= 1'b1;
        if (i_weight_valid && o_weight_ready) begin
          integer base, kk, idx, la, ch, tp;
          base = ldt_cnt * 8;
          for (kk = 0; kk < 8; kk = kk + 1) begin
            idx = base + kk;
            if (idx < LANES*TC2*9) begin
              la = idx / (TC2*9); ch = (idx % (TC2*9)) / 9; tp = idx % 9;
              wtile2[la][ch][tp] <= i_weight_in[(kk+1)*WT_W-1 -: WT_W];
            end
          end
          if (base + 8 >= LANES*TC2*9) begin
            o_weight_ready <= 1'b0; st <= S_P2; qr <= 0;
            if (tile2 == 0)
              for (a = 0; a < LANES; a = a + 1) begin
                biast2[a] <= bias2f[grp2*LANES+a];
                for (tt = 0; tt < T; tt = tt + 1) psum2[a][tt] <= bias2f[grp2*LANES+a];
              end
          end else ldt_cnt <= ldt_cnt + 1'b1;
        end
      end
      S_P2: begin
        case (qr)
          0: begin
            // R_TILE: accumulate one input-channel tile over the 3x3 window.
            for (a = 0; a < LANES; a = a + 1) begin
              for (tt = 0; tt < T; tt = tt + 1) begin
                logic signed [ACC_W-1:0] part;
                part = 0;
                for (b = 0; b < 9; b = b + 1) begin
                  integer wy, wx, fy, fx, ch;
                  wy = pr + b/3 - 1; wx = pc + b%3 - 1;
                  if (wy >= 0 && wy < H1 && wx >= 0 && wx < W1) begin
                    for (k = 0; k < TC2; k = k + 1) begin
                      logic bitv;
                      bitv = 1'b0;
                      ch = tile2*TC2 + k;
                      for (fy = 0; fy < 2; fy = fy + 1)
                        for (fx = 0; fx < 2; fx = fx + 1)
                          bitv = bitv | f1[2*wy+fy][2*wx+fx][ch*T+tt];
                      if (bitv) part = part + wtile2[a][k][b];
                    end
                  end
                end
                psum2[a][tt] <= psum2[a][tt] + part;
              end
            end
            lif2_v <= 1'b0; wait2 <= 0; qr <= 1;
          end
          1: begin
            // R_PACK/WAIT: pulse shared LIF once per (pr,pc,grp2) after the
            // last tile, then store the spike beat to the pool-2 assembly.
            if (tile2 != NT2-1) begin
              tile2 <= tile2 + 1'b1; ldt_cnt <= 0; st <= S_LDT; qr <= 0;
            end else begin
              if (wait2 == 0) begin
                for (a = 0; a < LANES; a = a + 1)
                  for (tt = 0; tt < T; tt = tt + 1)
                    lif2_ps[a][(tt+1)*ACC_W-1 -: ACC_W] <= psum2[a][tt];
              end
              lif2_v <= (wait2 == 0);
              if (lif2_rdy[0] || wait2 == LAT+8) begin
                lif2_v <= 1'b0;
                prow2[pr][pc][(grp2+1)*LANES*T-1 -: LANES*T] <= lif2_spk;
                tile2 <= 0; ldt_cnt <= 0;
                begin
                  logic last_pos;
                  last_pos = (grp2 == G2-1 && pc == W1-1 && pr == H1-1);
                  if (grp2 == G2-1) begin
                    grp2 <= 0;
                    if (pc == W1-1) begin
                      pc <= 0;
                      if (pr == H1-1) pr <= 0;
                      else pr <= pr + 1'b1;
                    end else pc <= pc + 1'b1;
                  end else grp2 <= grp2 + 1'b1;
                  if (last_pos) begin st <= S_DRN; tbeat <= 0; end
                  else st <= S_LDT;
                end
                qr <= 0;
              end else wait2 <= wait2 + 1'b1;
            end
          end
          default: qr <= 0;
        endcase
      end
      S_DRN: begin
        // Pool-2 OR over 2x2 prow2 entries, row-major beats with handshake.
        o_tok_valid <= 1'b1;
        if (i_tok_ready) begin
          integer g, pos, tpr, tpc, fy, fx;
          logic [LANES*T-1:0] beat;
          g = tbeat % G2; pos = tbeat / G2; tpc = pos % W2; tpr = pos / W2;
          beat = 0;
          for (fy = 0; fy < 2; fy = fy + 1)
            for (fx = 0; fx < 2; fx = fx + 1)
              beat = beat | prow2[2*tpr+fy][2*tpc+fx][(g+1)*LANES*T-1 -: LANES*T];
          o_tok_data <= beat;
          o_tok_last <= (tbeat == TBEATS-1);
          if (tbeat == TBEATS-1) begin
            st <= S_DRN2;
          end else tbeat <= tbeat + 1'b1;
        end
      end
      S_DRN2: begin
        o_tok_valid <= 1'b1;
        o_tok_last <= 1'b1;
        if (i_tok_ready) begin
          tbeat <= 0; o_tok_valid <= 1'b0; o_tok_last <= 1'b0; st <= S_LD1;
          ld1_cnt <= 0; bias_ld <= 1'b0; ldc0 <= 0;
        end
      end
      default: st <= S_LD1;
    endcase
  end
endmodule
