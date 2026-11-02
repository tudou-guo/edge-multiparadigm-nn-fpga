// Attention QK dot-product PE - parameterized port of upstream Attention_PE.
// Upstream: Transformer_part/AttnCalc/Attention_PE.v (2*16 lanes x T=4 fixed,
//   5-bit per-T sums concatenated to 20 bits).
// Diffs: HEAD_DIM/LANES/T parameters; per-T width SUM_W=$clog2(HEAD_DIM+1)
// (6 bits at head_dim=32, total 24 bits) so value 32 is representable;
// AND of Q/K spikes then popcount per T, same valid semantics (t0 valid out).
module spiking_attention_pe #(
  parameter int HEAD_DIM = 32,
  parameter int T        = 4,
  parameter int SUM_W    = 6,
  parameter int OUT_W    = T*SUM_W
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic i_valid,
  input  logic [HEAD_DIM*T-1:0] i_q,
  input  logic [HEAD_DIM*T-1:0] i_k,
  output logic [OUT_W-1:0] o_data,
  output logic o_valid
);
  // Lane packing follows upstream convention: lane m owns bits [m*T +: T]
  // (time-minor), so packing.py and RTL agree on one layout.
  logic [HEAD_DIM*T-1:0] r_and;
  logic r_v;
  always_ff @(posedge s_clk) begin
    r_v <= i_valid;
    if (i_valid) r_and <= i_q & i_k;
  end
  logic [HEAD_DIM-1:0] lane_t [T];
  genvar m, tt;
  generate
    for (m = 0; m < HEAD_DIM; m = m + 1) begin : g_lane
      for (tt = 0; tt < T; tt = tt + 1) begin : g_t
        assign lane_t[tt][m] = r_and[m*T + tt];
      end
    end
    for (tt = 0; tt < T; tt = tt + 1) begin : g_pc
      logic v_t;
      spiking_popcount #(.LANES(HEAD_DIM), .SUM_W(SUM_W)) u_pc (
        .s_clk(s_clk), .s_rst(s_rst),
        .i_data(lane_t[tt]), .i_valid(r_v),
        .o_sum(o_data[(tt+1)*SUM_W-1 -: SUM_W]),
        .o_valid(v_t)
      );
      if (tt == 0) begin : g_v0
        assign o_valid = v_t;
      end
    end
  endgenerate
endmodule
