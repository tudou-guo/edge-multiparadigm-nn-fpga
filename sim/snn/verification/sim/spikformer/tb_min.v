`timescale 1ns/1ps
module tb_min;
  reg s_clk = 0, s_rst = 1;
  always #5 s_clk = ~s_clk;
  spiking_sps_top #(.H(8), .W_IN(8), .C0(3), .C1(16), .C2(32), .T(2),
    .LANES(8), .TC2(4)) dut (
    .s_clk(s_clk), .s_rst(s_rst),
    .i_in_ch_1(0), .i_out_ch_1(0), .i_thr_1(200),
    .i_in_ch_2(0), .i_out_ch_2(0), .i_thr_2(60),
    .i_weight_in(64'd0), .i_weight_valid(1'b0), .o_weight_ready(),
    .i_pix_valid(1'b0), .o_pix_ready(), .i_pix_data(24'd0),
    .o_tok_valid(), .i_tok_ready(1'b1),
    .o_tok_data(), .o_tok_last()
  );
  initial begin
    $display("min boot");
    repeat (4) @(posedge s_clk);
    s_rst = 0;
    $display("min reset done t=%0t", $time);
    repeat (10) @(posedge s_clk);
    $display("min end t=%0t", $time);
    $finish;
  end
endmodule
