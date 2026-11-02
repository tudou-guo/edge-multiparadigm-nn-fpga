// Token adapter: row/channel reorder boundary + 1-bit spike -> RW4 expand.
// Upstream order from ref_ctrl drain is already token-major (n,y,x then g);
// this stage provides the bounded rate-decoupling (2-deep skid, no full-frame
// tbuf) and expands each 1-bit spike to an RW nibble 0/1 (NOT 0/F).
// fire = valid && ready; data/valid/last held under backpressure.
module spiking_sps_token_adapter #(
  parameter int LANES = 8,
  parameter int T = 4,
  parameter int RW = 4
)(
  input  logic s_clk, input  logic s_rst,
  input  logic [LANES*T-1:0] i_data,
  input  logic i_valid, output logic o_ready, input  logic i_last,
  output logic [LANES*T*RW-1:0] o_data,
  output logic o_valid, input  logic i_ready, output logic o_last
);
  logic [LANES*T*RW-1:0] sk0, sk1;
  logic sk0v, sk1v, sk0l, sk1l;
  function automatic [LANES*T*RW-1:0] expand(input [LANES*T-1:0] d);
    for (int a = 0; a < LANES; a = a + 1)
      for (int t = 0; t < T; t = t + 1)
        expand[(a*T+t+1)*RW-1 -: RW] = {{(RW-1){1'b0}}, d[a*T+t]};
  endfunction
  assign o_ready = !sk0v || !sk1v || i_ready;
  assign o_valid = sk0v;
  assign o_data = sk0;
  assign o_last = sk0l;
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin sk0v <= 1'b0; sk1v <= 1'b0; sk0 <= '0; sk1 <= '0; sk0l <= 1'b0; sk1l <= 1'b0; end
    else begin
      // pop
      if (o_valid && i_ready) begin
        sk0 <= sk1; sk0v <= sk1v; sk0l <= sk1l;
        sk1v <= 1'b0;
      end
      // push
      if (i_valid && o_ready) begin
        if (!sk0v) begin sk0 <= expand(i_data); sk0v <= 1'b1; sk0l <= i_last; end
        else if (!sk1v) begin sk1 <= expand(i_data); sk1v <= 1'b1; sk1l <= i_last; end
        else begin // both full and popping: shift
          sk0 <= sk1; sk0l <= sk1l;
          sk1 <= expand(i_data); sk1l <= i_last;
        end
      end
    end
  end
endmodule
