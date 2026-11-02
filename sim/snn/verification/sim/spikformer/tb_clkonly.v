`timescale 1ns/1ps
module tb_clkonly;
  reg s_clk = 0;
  always #5 s_clk = ~s_clk;
  initial begin
    $display("t=%0t clk=%b", $time, s_clk);
    #37;
    $display("t=%0t clk=%b", $time, s_clk);
    $finish;
  end
endmodule
