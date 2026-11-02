`timescale 1ns/1ps
// cfc_state_regs.sv -- 4 x int16 old state / new state staging, single commit.
// Contract section 3.1: the arrays below MUST keep the names h_cur / h_next so
// the testbenches can reference them hierarchically (tb_cfc_cell / tb_cfc_seq).
// All four new states are computed first (h_next comes from the cell core) and
// become visible together in h_cur on one commit pulse.
module cfc_state_regs (
  input  logic               clk,
  input  logic               rst,
  input  logic               clear,     // synchronous clear (idle CLEAR_STATE / pre-run)
  input  logic               commit,    // commit all four new states together
  input  logic signed [15:0] h_new_0,
  input  logic signed [15:0] h_new_1,
  input  logic signed [15:0] h_new_2,
  input  logic signed [15:0] h_new_3,
  output logic signed [15:0] h_cur_0,
  output logic signed [15:0] h_cur_1,
  output logic signed [15:0] h_cur_2,
  output logic signed [15:0] h_cur_3
);
  logic signed [15:0] h_cur  [0:3];
  logic signed [15:0] h_next [0:3];

  always_comb begin
    h_next[0] = h_new_0;
    h_next[1] = h_new_1;
    h_next[2] = h_new_2;
    h_next[3] = h_new_3;
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      for (int i = 0; i < 4; i++) h_cur[i] <= 16'sd0;
    end else if (clear) begin
      for (int i = 0; i < 4; i++) h_cur[i] <= 16'sd0;
    end else if (commit) begin
      for (int i = 0; i < 4; i++) h_cur[i] <= h_next[i];
    end
  end

  assign h_cur_0 = h_cur[0];
  assign h_cur_1 = h_cur[1];
  assign h_cur_2 = h_cur[2];
  assign h_cur_3 = h_cur[3];
endmodule
