`timescale 1ns/1ps
module tb_mlp_replay;
  reg s_clk = 0, s_rst = 1;
  reg start = 0;
  reg [63:0] indata;
  reg invalid;
  wire inready;
  wire ovalid;
  reg oready;
  wire [63:0] odata;
  reg [63:0] wdata;
  reg wvalid;
  wire wready;
  wire odone;
  always #5 s_clk = ~s_clk;
  spiking_mlp_controller #(.C(32), .HIDDEN(64), .T(2), .RW(4)) dut (
    .s_clk(s_clk), .s_rst(s_rst), .i_start(start), .o_done(odone),
    .i_in_valid(invalid), .i_in_data(indata), .o_in_ready(inready),
    .o_out_valid(ovalid), .i_out_ready(oready), .o_out_data(odata),
    .i_w_data(wdata), .i_w_valid(wvalid), .o_w_ready(wready),
    .i_thr_hidden(60), .i_thr_out(0)
  );
  reg [63:0] inp [0:3];
  reg [63:0] wt1 [0:383];
  reg [63:0] wt2 [0:383];
  reg [63:0] expb [0:3];
  integer fdump;
  integer fails = 0;
  integer i, got, k, g;
  reg [3:0] want;
  initial begin
    fdump = $fopen("D:/dl_snn/build/spikformer/mlp_replay/got.hex", "w");
    $readmemh("D:/dl_snn/build/spikformer/mlp_replay/inp.hex", inp);
    $readmemh("D:/dl_snn/build/spikformer/mlp_replay/wt1.hex", wt1);
    $readmemh("D:/dl_snn/build/spikformer/mlp_replay/wt2.hex", wt2);
    $readmemh("D:/dl_snn/build/spikformer/mlp_replay/exp.hex", expb);
    invalid = 0; wvalid = 0; oready = 0;
    repeat (4) @(posedge s_clk); s_rst = 0;
    @(posedge s_clk); start = 1;
    @(posedge s_clk); start = 0;
    for (i = 0; i < 4; i = i + 1) begin
      wait (inready); @(posedge s_clk); #1; indata = inp[i]; invalid = 1;
      @(posedge s_clk); #1; invalid = 0;
    end
    $display("TB MLP input done t=%0t", $time);
    for (i = 0; i < 384; i = i + 1) begin
      wait (wready); @(posedge s_clk); #1; wdata = wt1[i]; wvalid = 1;
      @(posedge s_clk); #1; wvalid = 0;
    end
    $display("TB MLP wt1 done t=%0t", $time);
    got = 0;
    fork
      begin
        for (g = 0; g < 4; g = g + 1) begin
          for (i = 0; i < 96; i = i + 1) begin
            wait (wready); @(posedge s_clk); #1; wdata = wt2[g*96+i]; wvalid = 1;
            @(posedge s_clk); #1; wvalid = 0;
          end
          $display("TB g=%0d feed done t=%0t st=%0d", g, $time, dut.st);
          wait (ovalid);
          // Assert ready only after valid observed; sample pre-NBA so the
          // same-edge advance cannot hide the beat.
          oready = 1;
          // odata is (re)computed in the first S_RES cycle; sample one more
          // edge later so the beat is settled. DUT advances on that same
          // edge but pre-NBA values are still the presented beat.
          @(posedge s_clk);
          @(posedge s_clk);
          $display("TB g=%0d sample ovalid=%b odone=%b t=%0t", g, ovalid, odone, $time);
          for (k = 0; k < 16; k = k + 1) begin
            want = expb[got] >> (4*(15-k));
            if (odata[k*4 +: 4] !== want) begin
              $display("FAIL beat%0d pos%0d: got %h want %h", got, k, odata[k*4 +: 4], want);
              fails = fails + 1;
            end
          end
          $fwrite(fdump, "%h\n", odata);
          for (k = 0; k < 8; k = k + 1) $fwrite(fdump, "%h %h\n", dut.psum2[k][0], dut.psum2[k][1]);
          got = got + 1;
          // Hold ready one more cycle so the DUT observes ready&&ready_d.
          @(posedge s_clk); #1; oready = 0;
        end
      end
      begin
        repeat (300000) @(posedge s_clk);
        $display("FAIL timeout got=%0d", got);
        fails = fails + 1;
        $finish;
      end
    join_any
    disable fork;
    if (fails == 0) $display("ALL PASS tb_mlp_replay (4 beats)");
    else $display("FAILURES=%0d tb_mlp_replay", fails);
    $finish;
  end
endmodule
