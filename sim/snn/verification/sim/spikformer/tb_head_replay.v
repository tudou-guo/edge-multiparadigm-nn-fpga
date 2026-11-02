`timescale 1ns/1ps
module tb_head_replay;
  reg s_clk = 0, s_rst = 1;
  reg start = 0;
  reg bvalid;
  wire bready;
  reg [15:0] bdata;
  wire [31:0] hdout;
  wire hen;
  wire [3:0] hwe;
  wire [31:0] haddr;
  wire sums_v;
  wire intr;
  always #5 s_clk = ~s_clk;
  spiking_head_reduce #(.CH(32), .TOKENS(8), .T(2), .HL(8)) dut (
    .s_clk(s_clk), .s_rst(s_rst), .i_start(start),
    .i_beat_valid(bvalid), .o_beat_ready(bready), .i_beat_data(bdata),
    .head_dout(hdout), .head_en(hen), .head_we(hwe), .head_addr(haddr),
    .o_sums_valid(sums_v), .o_intr(intr)
  );
  reg [15:0] beats [0:31];
  reg [31:0] expsums [0:31];
  reg [31:0] gotsums [0:31];
  integer fails = 0;
  integer i, got;
  initial begin
    $readmemh("D:/dl_snn/build/spikformer/head_replay/beats.hex", beats);
    $readmemh("D:/dl_snn/build/spikformer/head_replay/exp.hex", expsums);
    bvalid = 0;
    repeat (4) @(posedge s_clk); s_rst = 0;
    @(posedge s_clk); start = 1;
    @(posedge s_clk); start = 0;
    for (i = 0; i < 32; i = i + 1) begin
      wait (bready); @(posedge s_clk); #1; bdata = beats[i]; bvalid = 1;
      @(posedge s_clk); #1; bvalid = 0;
    end
    $display("TB head beats done t=%0t", $time);
    got = 0;
    fork
      begin
        while (got < 32) begin
          @(posedge s_clk); #1;
          if (hen && hwe == 4'hf) begin
            gotsums[haddr[31:2]] = hdout;
            got = got + 1;
          end
        end
      end
      begin
        repeat (50000) @(posedge s_clk);
        $display("FAIL timeout got=%0d", got);
        fails = fails + 1;
        $finish;
      end
    join_any
    disable fork;
    for (i = 0; i < 32; i = i + 1) begin
      if (gotsums[i] !== expsums[i]) begin
        $display("FAIL ch%0d: got %0d want %0d", i, gotsums[i], expsums[i]);
        fails = fails + 1;
      end
    end
    if (fails == 0) $display("ALL PASS tb_head_replay (32 ch)");
    else $display("FAILURES=%0d tb_head_replay", fails);
    $finish;
  end
endmodule
