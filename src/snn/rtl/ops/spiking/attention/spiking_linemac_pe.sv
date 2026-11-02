// Attn*V line accumulate PE - parameterized port of upstream LineMac_PE.
// Upstream: Transformer_part/AttnMultiplySpikes/LineMac_PE.v (attn slice
//   $clog2(32)=5 bits x T=4 fixed, 12-bit lane psums, MM_Calc_FIFO line buf).
// Diffs: SUM_W derived from $clog2(HEAD_DIM+1) (6 at head_dim=32), T/ACC_W
// parameters; value spikes gate attn rows per T; accumulate over tokens with
// a behavioral line buffer (codegen maps it to FIFO IP, depth = tokens).
module spiking_linemac_pe #(
  parameter int HEAD_DIM = 32,
  parameter int T        = 4,
  parameter int SUM_W    = 6,
  parameter int ACC_W    = 12,
  parameter int TOKENS   = 64
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic i_first_line_done,
  input  logic i_finish_once,
  input  logic i_send_valid,
  input  logic [T-1:0] i_value_spikes,
  input  logic [T*SUM_W-1:0] i_attn_data,
  input  logic i_final_valid,
  output logic [T*ACC_W-1:0] o_final_data,
  output logic o_final_valid
);
  logic [T*SUM_W-1:0] r_attn;
  logic [T-1:0] r_val;
  logic r_send, r_send_d0, r_acc_flag;
  always_ff @(posedge s_clk) begin
    r_send    <= i_send_valid;
    r_send_d0 <= r_send;
    if (i_send_valid) begin
      r_attn <= i_attn_data;
      r_val  <= i_value_spikes;
    end
  end
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst || i_finish_once) r_acc_flag <= 1'b0;
    else if (i_first_line_done) r_acc_flag <= 1'b1;
  end
  logic [T*ACC_W-1:0] fifo_dout;
  logic [T*ACC_W-1:0] w_add;
  genvar tt;
  generate
    for (tt = 0; tt < T; tt = tt + 1) begin : g_t
      wire [SUM_W-1:0] a_slice;
      assign a_slice = r_attn[(tt+1)*SUM_W-1 -: SUM_W];
      wire [SUM_W-1:0] gated;
      assign gated = r_val[tt] ? a_slice : 0;
      wire [ACC_W-1:0] fb;
      assign fb = r_acc_flag ? fifo_dout[(tt+1)*ACC_W-1 -: ACC_W] : 0;
      assign w_add[(tt+1)*ACC_W-1 -: ACC_W] = fb + gated;
    end
  endgenerate
  logic fifo_empty;
  // Behavioral line buffer, depth TOKENS; codegen retargets to FIFO IP.
  logic [T*ACC_W-1:0] mem [TOKENS];
  logic [$clog2(TOKENS):0] wptr, rptr;
  logic do_rd;
  assign do_rd = (r_send & r_acc_flag) | i_final_valid;
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin wptr <= 0; rptr <= 0; end
    else begin
      if (r_send_d0) begin mem[wptr[$clog2(TOKENS)-1:0]] <= w_add; wptr <= wptr + 1; end
      if (do_rd) rptr <= rptr + 1;
    end
  end
  assign fifo_dout = mem[rptr[$clog2(TOKENS)-1:0]];
  assign fifo_empty = (wptr == rptr);
  assign o_final_data  = fifo_dout;
  assign o_final_valid = do_rd & !fifo_empty;
endmodule
