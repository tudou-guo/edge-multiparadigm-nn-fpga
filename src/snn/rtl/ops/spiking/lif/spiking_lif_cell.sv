// Single-step LIF cell - exact port/semantics of upstream proj_lif.
// Upstream: proj_lif/proj_lif.v : V = (delta + pre) >>> 1 (tau=2 on the sum),
//   spike iff valid && non-negative && >= THRESHOLD, hard reset to 0 on spike.
// Diffs: BITS is a parameter; no absolute include; async reset kept.
// 48h NLP contract fix (2026-10-29): delta+pre is sign-extended to BITS+1 bits
// BEFORE the add and the shift (BITS=32 must not wrap before >>>1).
module spiking_lif_cell #(
  parameter int BITS = 25
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic signed [BITS-1:0] THRESHOLD,
  input  logic signed [BITS-1:0] i_delta_mem,
  input  logic i_delta_mem_valid,
  input  logic signed [BITS-1:0] i_pre_mem,
  output logic o_spike,
  output logic o_delta_mem_valid,
  output logic signed [BITS-1:0] o_nxt_mem
);
  wire signed [BITS:0] w_sum;
  wire signed [BITS-1:0] w_mem;
  assign w_sum = {i_delta_mem[BITS-1], i_delta_mem} + {i_pre_mem[BITS-1], i_pre_mem};
  assign w_mem = w_sum[BITS:1];
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) o_spike <= 1'b0;
    else if (i_delta_mem_valid && !w_mem[BITS-1] && w_mem >= THRESHOLD)
      o_spike <= 1'b1;
    else o_spike <= 1'b0;
  end
  always_ff @(posedge s_clk) o_delta_mem_valid <= i_delta_mem_valid;
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) o_nxt_mem <= 0;
    else if (i_delta_mem_valid && (w_mem[BITS-1] || w_mem < THRESHOLD))
      o_nxt_mem <= w_mem;
    else o_nxt_mem <= 0;
  end
endmodule
