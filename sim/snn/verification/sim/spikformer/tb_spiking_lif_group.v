`timescale 1ns/1ps
module tb_spiking_lif_group;
  reg s_clk = 0, s_rst = 1;
  reg signed [19:0] thr;
  reg v;
  reg signed [79:0] psum;
  wire [3:0] spikes;
  wire spkv;
  always #5 s_clk = ~s_clk;
  spiking_lif_group #(.T(4), .BITS(20)) dut (
    .s_clk(s_clk), .s_rst(s_rst), .i_lif_thrd(thr),
    .i_PsumValid(v), .i_PsumData(psum),
    .o_spikes_out(spikes), .o_spikes_valid(spkv)
  );
  integer fails = 0;
  reg [3:0] seen;
  integer n;
  initial begin
    v = 0; psum = 0; thr = 1000;
    repeat (4) @(posedge s_clk); s_rst = 0;
    @(posedge s_clk); psum = 0; v = 1;
    @(posedge s_clk); v = 0;
    repeat (12) @(posedge s_clk);
    if (spikes !== 4'b0000) begin $display("FAIL quiet"); fails = fails + 1; end
    else $display("PASS quiet");
    s_rst = 1; repeat (2) @(posedge s_clk); s_rst = 0;
    @(posedge s_clk); psum = {20'd5000, 20'd5000, 20'd5000, 20'd5000}; v = 1;
    @(posedge s_clk); v = 0;
    seen = 0;
    for (n = 0; n < 14; n = n + 1) begin
      @(posedge s_clk); #1;
      seen = seen | spikes;
    end
    if (seen === 4'b0000) begin $display("FAIL fires"); fails = fails + 1; end
    else $display("PASS fires seen=%b", seen);
    if (fails == 0) $display("ALL PASS tb_spiking_lif_group");
    else $display("FAILURES=%0d tb_spiking_lif_group", fails);
    $finish;
  end
endmodule
