// Spiking systolic PE - parameterized port of upstream Systolic_pe.
// Upstream: Transformer_part/systolic_array/Systolic_pe.v (T=4 hard-coded,
//   2-bit spike slices, shift-add mult, weight ping-pong shared across T).
// Diffs: T/SPIKE_BITS/W/PSUM_SLICE are parameters; 2-bit keeps shift-add,
// wider slices use exact multiply; no absolute hyper_para.v include.
module spiking_systolic_pe #(
  parameter int T          = 4,
  parameter int SPIKE_BITS = 2,
  parameter int W          = 8,
  parameter int PSUM_SLICE = 20,
  parameter int DATA_W     = T*SPIKE_BITS,
  parameter int PSUM_W     = T*PSUM_SLICE
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic weight_LoadPtr,
  input  logic weight_CalcPtr,
  input  logic weight_valid,
  input  logic signed [W-1:0] weights,
  input  logic in_data_valid,
  input  logic [DATA_W-1:0] in_raw_data,
  output logic out_data_valid,
  output logic [DATA_W-1:0] out_raw_data,
  input  logic signed [PSUM_W-1:0] in_psum_data,
  output logic out_psum_valid,
  output logic signed [PSUM_W-1:0] out_psum_data
);
  localparam int MW = W + SPIKE_BITS;
  logic signed [W-1:0] r_w0, r_w1;
  logic signed [W-1:0] w_w;
  assign w_w = weight_CalcPtr ? r_w1 : r_w0;
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin r_w0 <= 0; r_w1 <= 0; end
    else if (weight_valid && !weight_LoadPtr) r_w0 <= weights;
    else if (weight_valid && weight_LoadPtr) r_w1 <= weights;
  end
  logic signed [MW-1:0] r_mult [T];
  logic signed [MW-1:0] w_mult [T];
  genvar t;
  generate
    for (t = 0; t < T; t = t + 1) begin : g_mult
      wire [SPIKE_BITS-1:0] slice;
      assign slice = in_raw_data[(t+1)*SPIKE_BITS-1 -: SPIKE_BITS];
      if (SPIKE_BITS == 2) begin : g_sft
        always_comb begin
          case (slice)
            2'd0: w_mult[t] = 0;
            2'd1: w_mult[t] = w_w;
            2'd2: w_mult[t] = w_w <<< 1;
            2'd3: w_mult[t] = $signed({w_w, 1'b0}) + $signed({{2{w_w[W-1]}}, w_w});
            default: w_mult[t] = 0;
          endcase
        end
      end else begin : g_mul
        always_comb w_mult[t] = $signed({1'b0, slice}) * w_w;
      end
      always_ff @(posedge s_clk) begin
        if (in_data_valid) r_mult[t] <= w_mult[t];
      end
    end
  endgenerate
  always_ff @(posedge s_clk) begin
    out_data_valid <= in_data_valid;
    out_raw_data   <= in_raw_data;
  end
  logic signed [PSUM_W-1:0] w_psum;
  genvar u;
  generate
    for (u = 0; u < T; u = u + 1) begin : g_psum
      assign w_psum[(u+1)*PSUM_SLICE-1 -: PSUM_SLICE] =
        in_psum_data[(u+1)*PSUM_SLICE-1 -: PSUM_SLICE] +
        r_mult[u][PSUM_SLICE-1:0];
    end
  endgenerate
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) out_psum_data <= 0;
    else if (out_data_valid) out_psum_data <= w_psum;
  end
  always_ff @(posedge s_clk) out_psum_valid <= out_data_valid;
endmodule
