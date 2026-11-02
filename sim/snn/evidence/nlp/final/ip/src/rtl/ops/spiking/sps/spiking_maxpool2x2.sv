// Spike-domain 2x2 stride-2 maxpool: OR over each 2x2 window per timestep.
// Upstream simple_maxpool_unit/row_unit compare spike planes; for binary
// spikes max == OR, which this implements directly (parameterized W/T).
// Keeps valid/ready stall semantics: no output beat is dropped on backpressure.
module spiking_maxpool2x2 #(
  parameter int W = 16,
  parameter int T = 4
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic i_valid,
  output logic o_ready,
  input  logic [W*T-1:0] i_row0,
  input  logic [W*T-1:0] i_row1,
  output logic o_valid,
  input  logic i_ready,
  output logic [(W/2)*T-1:0] o_data
);
  assign o_ready = !o_valid || i_ready;
  logic [(W/2)*T-1:0] w_or;
  genvar x, t;
  generate
    for (x = 0; x < W/2; x = x + 1) begin : g_x
      for (t = 0; t < T; t = t + 1) begin : g_t
        assign w_or[x*T+t] = i_row0[(2*x)*T+t] | i_row0[(2*x+1)*T+t]
                            | i_row1[(2*x)*T+t] | i_row1[(2*x+1)*T+t];
      end
    end
  endgenerate
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin o_valid <= 1'b0; o_data <= 0; end
    else if (o_ready) begin o_valid <= i_valid; o_data <= w_or; end
  end
endmodule
