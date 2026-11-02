// Reference-style Eyeriss spike array (simple_eyeriss_array mechanism).
// Upstream: eyeriss_part/simple_eyeriss_array.v + simple_eyeriss_pe_unit.v
//   - LANES(8) output-channel lanes x TC2(8) input-channel PEs = 64 PE units.
//   - Each PE holds ONE kernel row (K=3 weights), spike shift reg with padding,
//     per-timestep spike-gated sums (spiking_eyeriss_pe, direct port of the
//     reference PE unit). Row reuse: 3 kernel rows sequenced explicitly.
//   - Input-channel grouping: 8 PEs per lane reduce via pipelined lane tree
//     (2 stages), rows accumulate in lane accumulator. No full-window
//     single-cycle dynamic array expression.
//   - Weight tile (LANES*TC2*9 B) streamed per (grp,tile); window spikes
//     (9*TC2*T bits) streamed per position by the controller from feature BRAM.
// Fixed-latency pipeline; tags (pos/grp/first/last) carried alongside.
module spiking_eyeriss_array #(
  parameter int LANES = 8,
  parameter int TC2 = 8,
  parameter int T = 4,
  parameter int W = 8,
  parameter int ACC_W = 21,
  parameter int POS_W = 8,
  parameter int GRP_W = 6
)(
  input  logic s_clk,
  input  logic s_rst,
  // tile weight stream: TILEB beats of 64b, little-endian byte pack
  input  logic [63:0] i_w_data,
  input  logic i_w_valid,
  output logic o_w_ready,
  // window spike stream: one beat per (pos,grp,tile)
  input  logic i_win_valid,
  output logic o_win_ready,
  input  logic [9*TC2*T-1:0] i_win_data,
  input  logic [POS_W-1:0] i_win_pos,
  input  logic [GRP_W-1:0] i_win_grp,
  input  logic i_win_first,
  input  logic i_win_last,
  // tile-result stream: per-lane accumulated gated sums for this tile
  output logic o_res_valid,
  input  logic i_res_ready,
  output logic [LANES*T*21-1:0] o_res_data,
  output logic [POS_W-1:0] o_res_pos,
  output logic [GRP_W-1:0] o_res_grp,
  output logic o_res_first,
  output logic o_res_last,
  // tile reuse: controller holds tile across many windows, releases after sweep
  input  logic i_tile_release
);
  localparam int TILEB = (LANES*TC2*9+7)/8;
  // Tile weight store as registers (reference PE r_weight style; 576B).
  logic [7:0] wtile [LANES][TC2][9];
  logic [$clog2(TILEB+1)-1:0] wcnt;
  logic tile_loaded;
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin wcnt <= 0; tile_loaded <= 1'b0; o_w_ready <= 1'b1; end
    else if (!tile_loaded) begin
      o_w_ready <= 1'b1;
      // accept only on fire (valid && ready); the loader counts the same
      // edge, so tile word counts cannot diverge across tile boundaries.
      if (i_w_valid && o_w_ready) begin
        integer base, k2, idx, a2, c2, b2;
        base = int'(wcnt) * 8;
        for (k2 = 0; k2 < 8; k2 = k2 + 1) begin
          idx = base + k2;
          if (idx < LANES*TC2*9) begin
            a2 = idx / (TC2*9); c2 = (idx % (TC2*9)) / 9; b2 = idx % 9;
            wtile[a2][c2][b2] <= i_w_data[(k2+1)*8-1 -: 8];
          end
        end
        if (base + 8 >= LANES*TC2*9) begin tile_loaded <= 1'b1; o_w_ready <= 1'b0; end
        else wcnt <= wcnt + 1'b1;
      end
    end else begin
      // released by FSM after window consumes tile (single-window per tile here)
      if (i_tile_release) begin tile_loaded <= 1'b0; wcnt <= 0; o_w_ready <= 1'b1; end
      else o_w_ready <= 1'b0;
    end
  end
  // ---- PE grid ----
  // pe_w[a][k]: 3 weights for current row; pe_s[a][k]: 3*T spike bits
  logic [3*8-1:0] pe_w [LANES][TC2];
  logic [3*T-1:0] pe_s [LANES][TC2];
  logic pe_wv, pe_sv, pe_cal;
  logic signed [T*10-1:0] pe_out [LANES][TC2];
  genvar ga, gk;
  generate
    for (ga = 0; ga < LANES; ga = ga + 1) begin : g_la
      for (gk = 0; gk < TC2; gk = gk + 1) begin : g_ch
        spiking_eyeriss_pe #(.K(3), .T(T), .W(8), .IMG_W(3), .SUM_W(10)) u_pe (
          .s_clk(s_clk), .s_rst(s_rst),
          .i_weight_valid(pe_wv), .i_weights(pe_w[ga][gk]),
          .i_spikes_valid(pe_sv), .i_spikes(pe_s[ga][gk]),
          .i_cal_start(pe_cal), .o_psum(pe_out[ga][gk])
        );
      end
    end
  endgenerate
  // window regs + tags
  logic win_bit [9][TC2][T];
  logic [POS_W-1:0] wpos; logic [GRP_W-1:0] wgrp;
  logic wfirst, wlast;
  // lane accumulators
  logic signed [ACC_W-1:0] lane_acc [LANES][T];
  // row sums (after lane tree, registered)
  logic signed [13:0] rowsum [LANES][T];
  logic signed [13:0] rowsum_q [LANES][T];
  // A_ACC registers the lane-row sum; A_ACC2 adds it to the tile accumulator.
  // (Merging both in one cycle would add the *previous* row's sum and make
  // the first window X.)
  typedef enum logic [3:0] {A_IDLE, A_LDW, A_LDS, A_CAL, A_CAP, A_ACC, A_ACC2, A_EMIT} astate_t;
  astate_t st;
  logic [1:0] row;
  logic win_done_pulse;
  // unpack window beat to bits
  always_ff @(posedge s_clk) begin
    if (st == A_IDLE && i_win_valid && tile_loaded) begin
      for (int b = 0; b < 9; b = b + 1)
        for (int k = 0; k < TC2; k = k + 1)
          for (int t = 0; t < T; t = t + 1)
            win_bit[b][k][t] <= i_win_data[(b*TC2*T + k*T + t)];
      wpos <= i_win_pos; wgrp <= i_win_grp; wfirst <= i_win_first; wlast <= i_win_last;
    end
  end
  // drive PE ports per state/row
  always_comb begin
    pe_wv = 1'b0; pe_sv = 1'b0; pe_cal = 1'b0;
    if (st == A_LDW) pe_wv = 1'b1;
    if (st == A_LDS) pe_sv = 1'b1;
    if (st == A_CAL) pe_cal = 1'b1;
    for (int a = 0; a < LANES; a = a + 1)
      for (int k = 0; k < TC2; k = k + 1) begin
        pe_w[a][k] = {wtile[a][k][row*3+2], wtile[a][k][row*3+1], wtile[a][k][row*3]};
        // spike row bits: cols of kernel row 'row'
        for (int cc = 0; cc < 3; cc = cc + 1)
          for (int t = 0; t < T; t = t + 1)
            pe_s[a][k][cc*T+t] = win_bit[row*3+cc][k][t];
      end
  end
  // lane tree: sum TC2 PE x 10b per (lane,t) -> 14b, 2 pipeline stages.
  // TC2=8 native, TC2=4 small; upper pairs tied zero when TC2==4.
  logic signed [11:0] tree_s1 [LANES][T][4];
  always_ff @(posedge s_clk) begin
    if (st == A_CAP) begin
      for (int a = 0; a < LANES; a = a + 1)
        for (int t = 0; t < T; t = t + 1) begin
          tree_s1[a][t][0] <= $signed(pe_out[a][0][(t+1)*10-1 -: 10]) + $signed(pe_out[a][1][(t+1)*10-1 -: 10]);
          tree_s1[a][t][1] <= $signed(pe_out[a][2][(t+1)*10-1 -: 10]) + $signed(pe_out[a][3][(t+1)*10-1 -: 10]);
          if (TC2 > 4) begin
            tree_s1[a][t][2] <= $signed(pe_out[a][4][(t+1)*10-1 -: 10]) + $signed(pe_out[a][5][(t+1)*10-1 -: 10]);
            tree_s1[a][t][3] <= $signed(pe_out[a][6][(t+1)*10-1 -: 10]) + $signed(pe_out[a][7][(t+1)*10-1 -: 10]);
          end else begin
            tree_s1[a][t][2] <= 12'sd0;
            tree_s1[a][t][3] <= 12'sd0;
          end
        end
    end
    if (st == A_ACC) begin
      for (int a = 0; a < LANES; a = a + 1)
        for (int t = 0; t < T; t = t + 1)
          rowsum_q[a][t] <= tree_s1[a][t][0] + tree_s1[a][t][1] + tree_s1[a][t][2] + tree_s1[a][t][3];
    end
  end
  assign o_win_ready = (st == A_IDLE) && tile_loaded;
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin
      st <= A_IDLE; row <= 0; win_done_pulse <= 1'b0;
      o_res_valid <= 1'b0; o_res_pos <= '0; o_res_grp <= '0;
      o_res_first <= 1'b0; o_res_last <= 1'b0;
      for (int a = 0; a < LANES; a = a + 1)
        for (int t = 0; t < T; t = t + 1) lane_acc[a][t] <= '0;
    end else begin
      win_done_pulse <= 1'b0;
      case (st)
        A_IDLE: begin
          if (i_win_valid && tile_loaded) begin
            for (int a = 0; a < LANES; a = a + 1)
              for (int t = 0; t < T; t = t + 1) lane_acc[a][t] <= '0;
            row <= 0; st <= A_LDW;
          end
        end
        A_LDW: st <= A_LDS;
        A_LDS: st <= A_CAL;
        A_CAL: st <= A_CAP;
        A_CAP: st <= A_ACC;
        A_ACC: st <= A_ACC2;
        A_ACC2: begin
          for (int a = 0; a < LANES; a = a + 1)
            for (int t = 0; t < T; t = t + 1)
              lane_acc[a][t] <= lane_acc[a][t] + rowsum_q[a][t];
          if (row == 2) st <= A_EMIT;
          else begin row <= row + 1'b1; st <= A_LDW; end
        end
        A_EMIT: begin
          if (!o_res_valid) begin
            for (int a = 0; a < LANES; a = a + 1)
              for (int t = 0; t < T; t = t + 1)
                o_res_data[(a*T+t+1)*21-1 -: 21] <= lane_acc[a][t][20:0];
            o_res_pos <= wpos; o_res_grp <= wgrp;
            o_res_first <= wfirst; o_res_last <= wlast;
            o_res_valid <= 1'b1;
          end else if (i_res_ready) begin
            o_res_valid <= 1'b0; win_done_pulse <= 1'b1; st <= A_IDLE;
          end
        end
      endcase
    end
  end
endmodule
