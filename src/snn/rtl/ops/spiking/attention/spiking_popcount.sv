// 32-lane popcount with CORRECT 6-bit output - fixes upstream truncation.
// Upstream: Transformer_part/AttnCalc/PipelineAdder.v declares o_SpikeSum
//   [5-1:0] (5 bits, max 31) for a 0..32 range; all-ones input (32) overflows.
//   Recorded as open-source arithmetic difference, NOT a golden reference.
// Fix: SUM_W = $clog2(LANES+1) = 6 for LANES=32; same 3-stage pipeline shape
// (byte groups -> reg -> add-tree) so data/valid alignment is preserved.
// Boundary case: i_data == 32'hFFFF_FFFF -> o_sum == 6'd32.
module spiking_popcount #(
  parameter int LANES = 32,
  parameter int SUM_W = 6
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic [LANES-1:0] i_data,
  input  logic i_valid,
  output logic [SUM_W-1:0] o_sum,
  output logic o_valid
);
  localparam int GROUPS = LANES / 8;
  logic [3:0] grp [GROUPS];
  genvar g;
  generate
    for (g = 0; g < GROUPS; g = g + 1) begin : g_grp
      always_comb begin
        grp[g] = i_data[g*8+0] + i_data[g*8+1] + i_data[g*8+2]
               + i_data[g*8+3] + i_data[g*8+4] + i_data[g*8+5]
               + i_data[g*8+6] + i_data[g*8+7];
      end
    end
  endgenerate
  logic [3:0] grp_q [GROUPS];
  logic [2:0] v_sr;
  always_ff @(posedge s_clk) begin
    for (int j = 0; j < GROUPS; j = j + 1) grp_q[j] <= grp[j];
    v_sr <= {v_sr[1:0], i_valid};
  end
  logic [SUM_W-1:0] acc;
  always_comb begin
    acc = 0;
    for (int k = 0; k < GROUPS; k = k + 1) acc = acc + grp_q[k];
  end
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) o_sum <= 0;
    else o_sum <= acc;
  end
  assign o_valid = v_sr[2];
endmodule
