// Row-stationary eyeriss PE: direct port of upstream simple_eyeriss_pe_unit.
// Mechanism kept: K weight regs (one kernel row); spike shift register with
// one zero-word padding; on cal_start the window shifts by T and each of the
// T timestep sums = sum over K taps of (spike_bit ? weight : 0).
// Diffs: K/T/W params (upstream K=3,T=4 fixed); sync active-high reset;
// edge-triggered weight load (valid rising edge) kept.
module spiking_eyeriss_pe #(
  parameter int K = 3,
  parameter int T = 4,
  parameter int W = 8,
  parameter int IMG_W = 32,
  parameter int SUM_W = W + 2
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic i_weight_valid,
  input  logic signed [K*W-1:0] i_weights,
  input  logic i_spikes_valid,
  input  logic [IMG_W*T-1:0] i_spikes,
  input  logic i_cal_start,
  output logic signed [T*SUM_W-1:0] o_psum
);
  logic signed [W-1:0] r_w [K];
  logic r_wv_q;
  logic [T*(IMG_W+1)-1:0] r_spikes;
  logic signed [SUM_W-1:0] r_sum [T];
  integer i;
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin
      r_wv_q <= 1'b0;
      for (i = 0; i < K; i = i + 1) r_w[i] <= 0;
    end else begin
      r_wv_q <= i_weight_valid;
      if (i_weight_valid && !r_wv_q)
        for (i = 0; i < K; i = i + 1) r_w[i] <= i_weights[(i+1)*W-1 -: W];
    end
  end
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) r_spikes <= 0;
    else if (i_cal_start) r_spikes <= r_spikes >> T;
    else if (i_spikes_valid) r_spikes <= {i_spikes, {T{1'b0}}};
  end
  logic [K*T-1:0] w_win;
  assign w_win = r_spikes[K*T-1:0];
  genvar t;
  generate
    for (t = 0; t < T; t = t + 1) begin : g_t
      logic signed [SUM_W-1:0] acc;
      integer k;
      always_comb begin
        acc = 0;
        for (k = 0; k < K; k = k + 1)
          if (w_win[k*T + t]) acc = acc + r_w[k];
      end
      always_ff @(posedge s_clk or posedge s_rst) begin
        if (s_rst) r_sum[t] <= 0;
        else if (i_cal_start) r_sum[t] <= acc;
      end
      assign o_psum[(t+1)*SUM_W-1 -: SUM_W] = r_sum[t];
    end
  endgenerate
endmodule
