// Reference-style SPS top: RGB encoder + Eyeriss array + PSUM/LIF +
// dual 128x8192 feature banks + row-block controller + token adapter.
// Replaces spiking_sps_top's full-window single-cycle array expression and
// frame-load-then-tile behavioral memories with the actual array/controller/
// PSUM-RAM/dual-bank structure (A-round, task VISION_REFERENCE_ROUTE s4).
// Numerical contract unchanged: 3x3 pad=1, signed INT8, 21b bias pairs,
// T-step LIF arithmetic-shift/hard-reset, 2x2/s2 OR pools, token-major beats.
// Weight stream order (host/DDR): w1 (W1B) -> bias pairs (BMAX) -> w2 tiles
// in (out-group,in-tile) order; pixels on the separate pixel stream.
module spiking_sps_ref_top #(
  parameter int H = 32, parameter int W_IN = 32,
  parameter int C0 = 3, parameter int C1 = 96, parameter int C2 = 384,
  parameter int T = 4, parameter int LANES = 8, parameter int TC2 = 8,
  parameter int WT_W = 8, parameter int ACC_W = 21, parameter int RW = 4
)(
  input  logic s_clk, input  logic s_rst,
  input  logic [15:0] i_thr_1, input  logic [15:0] i_thr_2,
  input  logic [63:0] i_weight_in, input  logic i_weight_valid, output logic o_weight_ready,
  input  logic i_pix_valid, output logic o_pix_ready, input  logic [C0*8-1:0] i_pix_data,
  input  logic i_start, output logic o_done,
  output logic [LANES*T*RW-1:0] o_tok_data, output logic o_tok_valid,
  input  logic i_tok_ready, output logic o_tok_last
);
  localparam int CMAX = (C1>C2)?C1:C2;
  wire signed [ACC_W-1:0] thr1e, thr2e;
  assign thr1e = {{ACC_W-16{i_thr_1[15]}}, i_thr_1};
  assign thr2e = {{ACC_W-16{i_thr_2[15]}}, i_thr_2};
  // encoder links
  wire [63:0] ew_d; wire ew_v, ew_r;
  wire [C0*8-1:0] ep_d; wire ep_v, ep_r;
  wire enc_start, enc_done;
  wire [LANES*T-1:0] enc_beat; wire enc_bv, enc_rdy;
  spiking_rgb_encoder #(.H(H), .W_IN(W_IN), .C0(C0), .C1(C1), .T(T),
    .LANES(LANES), .WT_W(WT_W), .ACC_W(ACC_W), .C2MAX(CMAX)) u_enc (
    .s_clk(s_clk), .s_rst(s_rst),
    .i_w_data(ew_d), .i_w_valid(ew_v), .o_w_ready(ew_r),
    .i_pix_data(ep_d), .i_pix_valid(ep_v), .o_pix_ready(ep_r),
    .i_thr(thr1e), .i_start(enc_start), .o_done(enc_done),
    .o_beat_data(enc_beat), .o_beat_valid(enc_bv), .i_beat_ready(enc_rdy)
  );
  // array links
  wire [63:0] aw_d; wire aw_v, aw_r;
  wire arr_win_v, arr_win_r;
  wire [9*TC2*T-1:0] arr_win_d;
  wire [7:0] arr_win_pos; wire [5:0] arr_win_grp;
  wire arr_win_first, arr_win_last, arr_rel;
  wire arr_res_v, arr_res_r;
  wire [LANES*T*21-1:0] arr_res_d;
  wire [7:0] arr_res_pos; wire [5:0] arr_res_grp;
  wire arr_res_first, arr_res_last;
  spiking_eyeriss_array #(.LANES(LANES), .TC2(TC2), .T(T), .W(WT_W),
    .ACC_W(ACC_W)) u_arr (
    .s_clk(s_clk), .s_rst(s_rst),
    .i_w_data(aw_d), .i_w_valid(aw_v), .o_w_ready(aw_r),
    .i_win_valid(arr_win_v), .o_win_ready(arr_win_r),
    .i_win_data(arr_win_d), .i_win_pos(arr_win_pos), .i_win_grp(arr_win_grp),
    .i_win_first(arr_win_first), .i_win_last(arr_win_last),
    .o_res_valid(arr_res_v), .i_res_ready(arr_res_r),
    .o_res_data(arr_res_d), .o_res_pos(arr_res_pos), .o_res_grp(arr_res_grp),
    .o_res_first(arr_res_first), .o_res_last(arr_res_last),
    .i_tile_release(arr_rel)
  );
  // psum links
  wire [63:0] pw_d; wire pw_v, pw_r, pw_done;
  wire psum_res_v, psum_res_r;
  wire [LANES*T-1:0] psum_beat; wire psum_bv, psum_br;
  wire [7:0] psum_bpos; wire [5:0] psum_bgrp;
  spiking_sps_psum #(.LANES(LANES), .T(T), .ACC_W(ACC_W), .DEPTH(256),
    .C2(C2), .BIASDEPTH(CMAX)) u_psum (
    .s_clk(s_clk), .s_rst(s_rst),
    .i_w_data(pw_d), .i_w_valid(pw_v), .o_w_ready(pw_r), .i_w_done(pw_done),
    .i_res_valid(psum_res_v), .o_res_ready(psum_res_r),
    .i_res_data(arr_res_d), .i_res_pos(arr_res_pos), .i_res_grp(arr_res_grp),
    .i_res_first(arr_res_first), .i_res_last(arr_res_last),
    .i_thr(thr2e),
    .o_beat_data(psum_beat), .o_beat_valid(psum_bv), .i_beat_ready(psum_br),
    .o_beat_pos(psum_bpos), .o_beat_grp(psum_bgrp)
  );
  // banks
  wire b0_we, b1_we; wire [12:0] b0_wa, b0_ra, b1_wa, b1_ra;
  wire [127:0] b0_wd, b0_rd, b1_wd, b1_rd;
  spiking_feature_bram #(.WIDTH(128), .DEPTH(8192)) u_b0 (
    .s_clk(s_clk), .s_rst(s_rst),
    .i_we(b0_we), .i_waddr(b0_wa), .i_wdata(b0_wd),
    .i_raddr(b0_ra), .o_rdata(b0_rd), .o_rvalid()
  );
  spiking_feature_bram #(.WIDTH(128), .DEPTH(8192)) u_b1 (
    .s_clk(s_clk), .s_rst(s_rst),
    .i_we(b1_we), .i_waddr(b1_wa), .i_wdata(b1_wd),
    .i_raddr(b1_ra), .o_rdata(b1_rd), .o_rvalid()
  );
  // raw token beats from controller -> adapter
  wire [LANES*T-1:0] raw_tok; wire raw_tv, raw_tr, raw_last;
  spiking_sps_ref_ctrl #(.H(H), .W_IN(W_IN), .C1(C1), .C2(C2), .T(T),
    .LANES(LANES), .TC2(TC2), .ACC_W(ACC_W)) u_ctrl (
    .s_clk(s_clk), .s_rst(s_rst),
    .i_w_data(i_weight_in), .i_w_valid(i_weight_valid), .o_w_ready(o_weight_ready),
    .i_pix_data(i_pix_data), .i_pix_valid(i_pix_valid), .o_pix_ready(o_pix_ready),
    .i_start(i_start), .o_done(o_done),
    .o_enc_w_data(ew_d), .o_enc_w_valid(ew_v), .i_enc_w_ready(ew_r),
    .o_enc_pix_data(ep_d), .o_enc_pix_valid(ep_v), .i_enc_pix_ready(ep_r),
    .o_enc_start(enc_start), .i_enc_done(enc_done),
    .i_enc_beat(enc_beat), .i_enc_valid(enc_bv), .o_enc_ready(enc_rdy),
    .o_arr_w_data(aw_d), .o_arr_w_valid(aw_v), .i_arr_w_ready(aw_r),
    .o_arr_win_valid(arr_win_v), .i_arr_win_ready(arr_win_r),
    .o_arr_win_data(arr_win_d), .o_arr_win_pos(arr_win_pos), .o_arr_win_grp(arr_win_grp),
    .o_arr_win_first(arr_win_first), .o_arr_win_last(arr_win_last),
    .o_arr_tile_release(arr_rel),
    .i_arr_res_valid(arr_res_v), .o_arr_res_ready(arr_res_r),
    .i_arr_res_data(arr_res_d), .i_arr_res_pos(arr_res_pos), .i_arr_res_grp(arr_res_grp),
    .i_arr_res_first(arr_res_first), .i_arr_res_last(arr_res_last),
    .o_psum_w_data(pw_d), .o_psum_w_valid(pw_v), .i_psum_w_ready(pw_r),
    .o_psum_w_done(pw_done),
    .o_psum_res_valid(psum_res_v), .i_psum_res_ready(psum_res_r),
    .i_psum_beat(psum_beat), .i_psum_beat_valid(psum_bv), .o_psum_beat_ready(psum_br),
    .i_psum_beat_pos(psum_bpos), .i_psum_beat_grp(psum_bgrp),
    .o_b0_we(b0_we), .o_b0_waddr(b0_wa), .o_b0_wdata(b0_wd),
    .o_b0_raddr(b0_ra), .i_b0_rdata(b0_rd),
    .o_b1_we(b1_we), .o_b1_waddr(b1_wa), .o_b1_wdata(b1_wd),
    .o_b1_raddr(b1_ra), .i_b1_rdata(b1_rd),
    .o_tok_data(raw_tok), .o_tok_valid(raw_tv), .i_tok_ready(raw_tr), .o_tok_last(raw_last)
  );
  spiking_sps_token_adapter #(.LANES(LANES), .T(T), .RW(RW)) u_adp (
    .s_clk(s_clk), .s_rst(s_rst),
    .i_data(raw_tok), .i_valid(raw_tv), .o_ready(raw_tr), .i_last(raw_last),
    .o_data(o_tok_data), .o_valid(o_tok_valid), .i_ready(i_tok_ready), .o_last(o_tok_last)
  );
endmodule
