`timescale 1ns/1ps
module tb_spiking_popcount;
  reg s_clk = 0, s_rst = 1;
  reg [31:0] i_data;
  reg i_valid;
  wire [5:0] o_sum;
  wire o_valid;
  always #5 s_clk = ~s_clk;
  spiking_popcount #(.LANES(32), .SUM_W(6)) dut (
    .s_clk(s_clk), .s_rst(s_rst), .i_data(i_data), .i_valid(i_valid),
    .o_sum(o_sum), .o_valid(o_valid)
  );
  integer fails = 0;
  initial begin
    i_valid = 0; i_data = 0;
    repeat (4) @(posedge s_clk); s_rst = 0;
    @(posedge s_clk); i_data = 32'hFFFFFFFF; i_valid = 1;
    @(posedge s_clk); i_valid = 0;
    repeat (4) @(posedge s_clk);
    if (o_sum !== 6'd32) begin $display("FAIL ones: %0d", o_sum); fails = fails + 1; end
    else $display("PASS ones = 32");
    @(posedge s_clk); i_data = 0; i_valid = 1;
    @(posedge s_clk); i_valid = 0;
    repeat (4) @(posedge s_clk);
    if (o_sum !== 6'd0) begin $display("FAIL zero: %0d", o_sum); fails = fails + 1; end
    else $display("PASS zero = 0");
    @(posedge s_clk); i_data = 32'hF0F0F0F0; i_valid = 1;
    @(posedge s_clk); i_valid = 0;
    repeat (4) @(posedge s_clk);
    if (o_sum !== 6'd16) begin $display("FAIL nibble: %0d", o_sum); fails = fails + 1; end
    else $display("PASS nibble = 16");
    if (fails == 0) $display("ALL PASS tb_spiking_popcount");
    else $display("FAILURES=%0d tb_spiking_popcount", fails);
    $finish;
  end
endmodule
