// Block-level replay: 4 tokens through one Transformer block
// (QKV+LIF -> QK popcount attention -> AttnV -> proj -> residual ->
//  MLP+LIF+residual) against the Python golden from gen_block_replay.py.
// Load order matches the block's documented S_LDW sequence:
//   wq(128) wk(128) wv(128) wp(128) bq(16) bk(16) bv(16) bp(16)
//   wfc1(256) bf1(32) wfc2(256) bf2(16) = 1136 beats, then 16 token beats.
`timescale 1ns/1ps
module tb_block_replay;
  reg s_clk = 0, s_rst = 1;
  reg start = 0;
  reg [63:0] wdata;
  reg wvalid;
  wire wready;
  reg signed [31:0] thr;
  reg [63:0] indata;
  reg invalid;
  wire inready;
  wire ovalid;
  reg oready;
  wire [63:0] odata;
  wire olast;
  wire odone;
  always #5 s_clk = ~s_clk;
  spiking_block_top #(.E(32), .HEADS(1), .HEAD_DIM(32), .HIDDEN(64), .T(2),
    .TOKENS(4), .TL(8), .TC(8), .RW(4)) dut (
    .s_clk(s_clk), .s_rst(s_rst), .i_start(start), .o_done(odone),
    .i_in_valid(invalid), .o_in_ready(inready), .i_in_data(indata),
    .o_out_valid(ovalid), .i_out_ready(oready), .o_out_data(odata), .o_out_last(olast),
    .i_w_data(wdata), .i_w_valid(wvalid), .o_w_ready(wready),
    .i_thr_q(thr), .i_thr_k(thr), .i_thr_v(thr),
    .i_thr_proj(thr), .i_thr_fc1(thr), .i_thr_fc2(thr)
  );
  reg [63:0] ld [0:1135];
  reg [63:0] tok [0:15];
  reg [63:0] expb [0:15];
  integer fails = 0;
  integer i, got, k;
  reg [63:0] cat;
  initial begin
    $readmemh("D:/dl_snn/build/spikformer/block_replay/wq.hex", ld, 0, 127);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/wk.hex", ld, 128, 255);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/wv.hex", ld, 256, 383);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/wp.hex", ld, 384, 511);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/bq.hex", ld, 512, 527);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/bk.hex", ld, 528, 543);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/bv.hex", ld, 544, 559);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/bp.hex", ld, 560, 575);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/wfc1.hex", ld, 576, 831);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/bf1.hex", ld, 832, 863);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/wfc2.hex", ld, 864, 1119);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/bf2.hex", ld, 1120, 1135);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/tok.hex", tok);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/exp.hex", expb);
    wvalid = 0; invalid = 0; oready = 0; thr = 60;
    repeat (4) @(posedge s_clk); s_rst = 0;
    @(posedge s_clk); start = 1;
    @(posedge s_clk); start = 0;
    for (i = 0; i < 1136; i = i + 1) begin
      wait (wready); @(posedge s_clk); #1; wdata = ld[i]; wvalid = 1;
      @(posedge s_clk); #1; wvalid = 0;
    end
    for (i = 0; i < 16; i = i + 1) begin
      wait (inready); @(posedge s_clk); #1; indata = tok[i]; invalid = 1;
      @(posedge s_clk); #1; invalid = 0;
    end
    got = 0;
    fork
      begin
        while (got < 16) begin
          wait (ovalid);
          oready = 1;
          @(posedge s_clk);
          cat = odata;
          #1; oready = 0;
          @(posedge s_clk);
          begin
            for (k = 0; k < 16; k = k + 1) begin
              if (cat[k*4 +: 4] !== expb[got][k*4 +: 4]) begin
                $display("FAIL beat%0d pos%0d: got %h want %h", got, k, cat[k*4 +: 4], expb[got][k*4 +: 4]);
                fails = fails + 1;
              end
            end
            if (got == 15 && !olast) begin $display("FAIL last flag"); fails = fails + 1; end
            got = got + 1;
          end
        end
      end
      begin
        repeat (400000) @(posedge s_clk);
        $display("FAIL timeout got=%0d", got);
        fails = fails + 1;
        $finish;
      end
    join_any
    disable fork;
    if (fails == 0) $display("ALL PASS tb_block_replay (16 beats)");
    else $display("FAILURES=%0d tb_block_replay", fails);
    $finish;
  end
endmodule