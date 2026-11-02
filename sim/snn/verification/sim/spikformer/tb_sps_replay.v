`timescale 1ns/1ps
module tb_sps_replay;
  reg s_clk = 0, s_rst = 1;
  reg [15:0] thr1 = 200, thr2 = 60;
  reg [63:0] wdata;
  reg wvalid;
  wire wready;
  reg pvalid;
  wire pready;
  reg [23:0] pdata;
  wire tvalid;
  reg tready;
  wire [15:0] tdata;
  wire tlast;
  always #5 s_clk = ~s_clk;
  spiking_sps_top #(.H(8), .W_IN(8), .C0(3), .C1(16), .C2(32), .T(2),
    .LANES(8), .TC2(4)) dut (
    .s_clk(s_clk), .s_rst(s_rst),
    .i_in_ch_1(0), .i_out_ch_1(0), .i_thr_1(thr1),
    .i_in_ch_2(0), .i_out_ch_2(0), .i_thr_2(thr2),
    .i_weight_in(wdata), .i_weight_valid(wvalid), .o_weight_ready(wready),
    .i_pix_valid(pvalid), .o_pix_ready(pready), .i_pix_data(pdata),
    .o_tok_valid(tvalid), .i_tok_ready(tready),
    .o_tok_data(tdata), .o_tok_last(tlast)
  );
  reg [63:0] w_ld1 [0:53];
  reg [63:0] b_ld [0:31];
  reg [23:0] px [0:63];
  reg [63:0] wt [0:9215];
  reg [31:0] expb [0:15];
  integer fails = 0;
  integer i, got;
  reg [3:0] gotlane;
  initial begin
    $readmemh("D:/dl_snn/build/spikformer/sps_replay/w_ld1.hex", w_ld1);
    $readmemh("D:/dl_snn/build/spikformer/sps_replay/b_ld.hex", b_ld);
    $readmemh("D:/dl_snn/build/spikformer/sps_replay/pix.hex", px);
    $readmemh("D:/dl_snn/build/spikformer/sps_replay/w_tiles.hex", wt);
    $readmemh("D:/dl_snn/build/spikformer/sps_replay/exp.hex", expb);
    wvalid = 0; pvalid = 0; tready = 0;
    $display("TB boot readmem ok w0=%h", w_ld1[0]);
    repeat (4) @(posedge s_clk); s_rst = 0;
    $display("TB reset done t=%0t", $time);
    // LD1 weights + bias
    $display("TB enter LD t=%0t wready=%b", $time, wready);
    for (i = 0; i < 54; i = i + 1) begin
      wait (wready); @(posedge s_clk); #1; wdata = w_ld1[i]; wvalid = 1;
      @(posedge s_clk); #1; wvalid = 0;
    end
    for (i = 0; i < 32; i = i + 1) begin
      wait (wready); @(posedge s_clk); #1; wdata = b_ld[i]; wvalid = 1;
      @(posedge s_clk); #1; wvalid = 0;
    end
    // pixels
    $display("TB enter PIX t=%0t", $time);
    for (i = 0; i < 64; i = i + 1) begin
      wait (pready); @(posedge s_clk); #1; pdata = px[i]; pvalid = 1;
      @(posedge s_clk); #1; pvalid = 0;
    end
    // stage-2 tiles
    $display("TB enter TILES t=%0t", $time);
    for (i = 0; i < 9216; i = i + 1) begin
      wait (wready); @(posedge s_clk); #1; wdata = wt[i]; wvalid = 1;
      @(posedge s_clk); #1; wvalid = 0;
    end
    // drain with one backpressure gap
    $display("TB enter DRAIN t=%0t", $time);
    got = 0;
    fork
      begin
        while (got < 16) begin
          @(posedge s_clk); #1;
          if (got == 5) begin tready = 0; repeat (3) @(posedge s_clk); #1; end
          tready = 1;
          if (tvalid && tready) begin
            for (i = 0; i < 8; i = i + 1) begin
              gotlane = expb[got] >> (4*(7-i));
              if (tdata[i*2 +: 2] !== gotlane[1:0]) begin
                $display("FAIL beat%0d lane%0d: got %b want %b", got, i, tdata[i*2 +: 2], gotlane[1:0]);
                fails = fails + 1;
              end
            end
            if (got == 15 && !tlast) begin $display("FAIL last flag"); fails = fails + 1; end
            got = got + 1;
          end
        end
      end
      begin
        repeat (300000) @(posedge s_clk);
        $display("FAIL timeout got=%0d t=%0t", got, $time);
        fails = fails + 1;
        $finish;
      end
    join_any
    disable fork;
    if (fails == 0) $display("ALL PASS tb_sps_replay (16 beats)");
    else $display("FAILURES=%0d tb_sps_replay", fails);
    $finish;
  end
endmodule
