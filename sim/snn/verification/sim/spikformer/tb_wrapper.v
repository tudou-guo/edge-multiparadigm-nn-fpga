`timescale 1ns/1ps
module tb_wrapper;
  reg s_clk = 0, s_rst_n = 0;
  reg [7:0] awaddr;
  reg [31:0] wdata;
  reg wvalid;
  wire wready;
  reg [7:0] araddr;
  reg arvalid;
  wire [31:0] rdata;
  wire rvalid;
  wire ddrr, busy, done, err;
  wire [63:0] ddraddr;
  wire [31:0] sums [0:3];
  wire hvalid;
  always #5 s_clk = ~s_clk;
  spikformer_platform_wrapper #(.T(2), .DEPTH(1), .CH(4), .HEADS(1)) dut (
    .s_clk(s_clk), .s_rst_n(s_rst_n),
    .s_axil_awaddr(awaddr), .s_axil_wdata(wdata), .s_axil_wvalid(wvalid),
    .s_axil_wready(wready), .s_axil_araddr(araddr), .s_axil_arvalid(arvalid),
    .s_axil_rdata(rdata), .s_axil_rvalid(rvalid),
    .o_ddr_req(ddrr), .i_ddr_gnt(1'b1), .o_ddr_addr(ddraddr),
    .o_head_sums(sums), .o_head_valid(hvalid),
    .o_busy(busy), .o_done(done), .o_error(err)
  );
  integer fails = 0;
  task wreg(input [7:0] a, input [31:0] d);
    begin
      @(posedge s_clk); awaddr = a; wdata = d; wvalid = 1;
      @(posedge s_clk); wvalid = 0;
    end
  endtask
  initial begin
    awaddr = 0; wdata = 0; wvalid = 0; araddr = 0; arvalid = 0;
    repeat (4) @(posedge s_clk); s_rst_n = 1;
    // ID/version read
    @(posedge s_clk); araddr = 8'h00; arvalid = 1;
    @(posedge s_clk); #1;
    if (rdata !== 32'h5350494B) begin $display("FAIL id=%h", rdata); fails = fails + 1; end
    else $display("PASS id");
    arvalid = 0;
    // start
    wreg(8'h14, 32'h00001000);
    wreg(8'h08, 32'h00000001);
    repeat (3) @(posedge s_clk);
    if (!busy) begin $display("FAIL not busy after start"); fails = fails + 1; end
    else $display("PASS busy after start");
    // double start while busy -> error (level must fall first: edge semantics)
    wreg(8'h08, 32'h00000000);
    wreg(8'h08, 32'h00000001);
    repeat (2) @(posedge s_clk);
    if (!err) begin $display("FAIL no error on double start"); fails = fails + 1; end
    else $display("PASS double-start rejected with error");
    // ack clears error (no done since engines tied)
    wreg(8'h30, 32'h00000000);
    wreg(8'h30, 32'h00000001);
    repeat (2) @(posedge s_clk);
    if (err) begin $display("FAIL ack did not clear"); fails = fails + 1; end
    else $display("PASS ack clears error");
    if (fails == 0) $display("ALL PASS tb_wrapper");
    else $display("FAILURES=%0d tb_wrapper", fails);
    $finish;
  end
  initial begin
    repeat (5000) @(posedge s_clk);
    $display("FAIL timeout");
    $finish;
  end
endmodule
