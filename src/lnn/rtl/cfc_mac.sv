`timescale 1ns/1ps
// cfc_mac.sv -- 16x16 signed multiply + exact 48-bit accumulate (3-stage pipeline).
// Contract: docs/CFC_CONTRACT.md section 3.1 (rtl/cfc_mac.sv).
//
// One MAC beat per clock.  Pipeline (latency 3 cycles from input to acc):
//   cycle t   : a/b/c/valid/clear sampled by stage 1 registers
//   cycle t+1 : stage 1 (DSP input registers)
//   cycle t+2 : multiplier register
//   cycle t+3 : accumulator add, valid_out pulses, acc holds the new result
// clear flushes the accumulator before adding the current beat, so the first
// beat of a dot product may carry its bias in "c" to get exact bias+product
// accumulation (bias is int32 Q24, sign-extended into the 48-bit addend).
module cfc_mac (
  input  logic               clk,
  input  logic               rst,
  input  logic               valid_in,
  input  logic               clear,
  input  logic signed [15:0] a,
  input  logic signed [15:0] b,
  input  logic signed [47:0] c,
  output logic               valid_out,
  output logic signed [47:0] acc
);
  logic               v_s1, v_s2;
  logic               clr_s1, clr_s2;
  logic signed [15:0] a_s1, b_s1;
  logic signed [47:0] c_s1, c_s2;
  logic signed [31:0] prod_s2;

  always_ff @(posedge clk) begin
    if (rst) begin
      a_s1      <= '0;
      b_s1      <= '0;
      c_s1      <= '0;
      c_s2      <= '0;
      v_s1      <= 1'b0;
      v_s2      <= 1'b0;
      clr_s1    <= 1'b0;
      clr_s2    <= 1'b0;
      prod_s2   <= '0;
      acc       <= '0;
      valid_out <= 1'b0;
    end else begin
      // stage 1: register operands (DSP48 input registers)
      a_s1   <= a;
      b_s1   <= b;
      c_s1   <= c;
      v_s1   <= valid_in;
      clr_s1 <= clear;

      // stage 2: multiply (DSP48 M register)
      prod_s2 <= a_s1 * b_s1;
      c_s2    <= c_s1;
      v_s2    <= v_s1;
      clr_s2  <= clr_s1;

      // stage 3: accumulate (DSP48 P register)
      if (v_s2) begin
        acc       <= (clr_s2 ? 48'sd0 : acc) + {{16{prod_s2[31]}}, prod_s2} + c_s2;
        valid_out <= 1'b1;
      end else begin
        valid_out <= 1'b0;
      end
    end
  end
endmodule
