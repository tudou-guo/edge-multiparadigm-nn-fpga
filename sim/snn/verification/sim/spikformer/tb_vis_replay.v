// Vision end-to-end replay: 8x8x3 image -> SPS -> block -> per-channel sums.
// Golden: gen_sps_replay (tokens) -> gen_block_replay TOKSRC (block out)
//         -> gen_vis_sums (sums).
`timescale 1ns/1ps
module tb_vis_replay;
  reg s_clk = 0, s_rst = 1, start = 0;
  reg [63:0] wdata; reg wvalid; wire wready;
  reg [23:0] pdata; reg pvalid; wire pready;
  reg signed [31:0] thr = 60;
  reg signed [31:0] thr1 = 200;
  reg signed [31:0] thr2 = 60;
  wire [31:0] sums [0:31];
  wire sums_valid, done;
  always #5 s_clk = ~s_clk;

  spikformer_vis_top #(.H(8), .W_IN(8), .C0(3), .C1(16), .C2(32), .T(2),
    .LANES(8), .TC2(4), .E(32), .HEADS(1), .HEAD_DIM(32), .HIDDEN(64),
    .TL(8), .TC(8), .RW(4), .NTOK(4), .W1B(54), .BIB(32), .W2B(9216), .BLKW(1136)
  ) dut (
    .s_clk(s_clk), .s_rst(s_rst), .i_start(start), .o_done(done),
    .i_pix_data(pdata), .i_pix_valid(pvalid), .o_pix_ready(pready),
    .i_w_data(wdata), .i_w_valid(wvalid), .o_w_ready(wready),
    .i_thr_1(thr1), .i_thr_2(thr2), .i_thr_q(thr), .i_thr_k(thr),
    .i_thr_v(thr), .i_thr_proj(thr), .i_thr_fc1(thr), .i_thr_fc2(thr),
    .o_sums(sums), .o_sums_valid(sums_valid)
  );

  reg [63:0] w_ld1 [0:53];
  reg [63:0] b_ld  [0:31];
  reg [63:0] w_tiles [0:9215];
  reg [63:0] blkw [0:1135];
  reg [23:0] pix [0:63];
  reg [31:0] exp_sums [0:31];
  reg [3:0] exp_tok [0:255];
  integer tokfail = 0, bwcnt = 0, dbg0 = 0, dbg1 = 0, dbg2 = 0, obeats = 0;
  integer aa, tt2, tk2;
  integer tsum [0:31];
  integer i, k, fails;
  integer ncy = 0;
  integer phase = 0;
  reg chk = 0;
  always @(posedge s_clk) begin
    if (!chk && dut.st == 6) begin   // S_START: tokens captured, block about to run
      for (i = 0; i < 4; i = i + 1)
        for (k = 0; k < 32; k = k + 1)
          for (integer tk = 0; tk < 2; tk = tk + 1)
            if (dut.tbuf[i][k][tk] !== exp_tok[i*64 + k*2 + tk]) begin
              if (tokfail < 6) $display("TOKFAIL n%0d c%0d t%0d got %0d want %0d", i, k, tk, dut.tbuf[i][k][tk], exp_tok[i*64 + k*2 + tk]);
              tokfail = tokfail + 1;
            end
      $display("TOKEN CHECK fails=%0d", tokfail);
      chk = 1;
    end
    ncy = ncy + 1;
    if (ncy % 20000 == 0)
      $display("MON cyc=%0d st=%0d sps=%0d blk=%0d bcnt=%0d wready=%b pready=%b phase=%0d",
        ncy, dut.st, dut.u_sps.st, dut.u_blk.st, dut.bcnt, wready, pready, phase);
  end
  always @(posedge s_clk) begin
    if (dut.blk_wvalid && dut.blk_wready) bwcnt = bwcnt + 1;
    if (dut.blk_ov && !dbg0) begin
      dbg0 = 1;
      $display("BLKOUT first beat=%h", dut.blk_odata);
    end
    if (dut.blk_ov && dut.blk_ordy) obeats = obeats + 1;
    if (dut.blk_done && !dbg1) begin
      dbg1 = 1;
      for (aa = 0; aa < 32; aa = aa + 1) begin
        tsum[aa] = 0;
        for (tk2 = 0; tk2 < 4; tk2 = tk2 + 1)
          for (tt2 = 0; tt2 < 2; tt2 = tt2 + 1)
            tsum[aa] = tsum[aa] + dut.tbuf[tk2][aa][tt2];
      end
      $display("BLKDONE bwcnt=%0d obeats=%0d blk_st=%0d", bwcnt, obeats, dut.u_blk.st);
      $display("TSUM %0d %0d %0d %0d %0d %0d %0d %0d",
        tsum[0], tsum[1], tsum[2], tsum[3], tsum[4], tsum[5], tsum[6], tsum[7]);
    end
    if (dut.st == 8 && !dbg2) begin
      dbg2 = 1;
      $display("TBOUT t0c0=%h t0c1=%h t1c0=%h", {dut.tbuf[0][0][0], dut.tbuf[0][0][1]},
        {dut.tbuf[0][1][0], dut.tbuf[0][1][1]}, {dut.tbuf[1][0][0], dut.tbuf[1][0][1]});
    end
  end

  initial begin
    $readmemh("D:/dl_snn/build/spikformer/sps_replay/w_ld1.hex", w_ld1);
    $readmemh("D:/dl_snn/build/spikformer/sps_replay/b_ld.hex", b_ld);
    $readmemh("D:/dl_snn/build/spikformer/sps_replay/w_tiles.hex", w_tiles);
    $readmemh("D:/dl_snn/build/spikformer/sps_replay/pix.hex", pix);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/wq.hex", blkw, 0, 127);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/wk.hex", blkw, 128, 255);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/wv.hex", blkw, 256, 383);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/wp.hex", blkw, 384, 511);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/bq.hex", blkw, 512, 527);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/bk.hex", blkw, 528, 543);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/bv.hex", blkw, 544, 559);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/bp.hex", blkw, 560, 575);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/wfc1.hex", blkw, 576, 831);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/bf1.hex", blkw, 832, 863);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/wfc2.hex", blkw, 864, 1119);
    $readmemh("D:/dl_snn/build/spikformer/block_replay/bf2.hex", blkw, 1120, 1135);
    $readmemh("D:/dl_snn/build/spikformer/vis_replay/exp_sums.hex", exp_sums);
    $readmemh("D:/dl_snn/build/spikformer/vis_replay/exp_tok.hex", exp_tok);
    wvalid = 0; pvalid = 0; fails = 0;
    repeat (4) @(posedge s_clk); s_rst = 0;
    @(posedge s_clk); start = 1; @(posedge s_clk); start = 0;
    // host order must match spikformer_vis_top FSM:
    //   S_BW block weights -> S_W1 conv1 -> S_BI bias -> S_PIX image
    //   -> S_TILE conv2 tile set (one window per token position)
    phase = 1;
    for (i = 0; i < 1136; i = i + 1) begin
      wait (wready); @(posedge s_clk); #1; wdata = blkw[i]; wvalid = 1;
      @(posedge s_clk); #1; wvalid = 0;
    end
    phase = 2;
    for (i = 0; i < 54; i = i + 1) begin
      wait (wready); @(posedge s_clk); #1; wdata = w_ld1[i]; wvalid = 1;
      @(posedge s_clk); #1; wvalid = 0;
    end
    phase = 3;
    for (i = 0; i < 32; i = i + 1) begin
      wait (wready); @(posedge s_clk); #1; wdata = b_ld[i]; wvalid = 1;
      @(posedge s_clk); #1; wvalid = 0;
    end
    phase = 4;
    for (i = 0; i < 64; i = i + 1) begin
      wait (pready); @(posedge s_clk); #1; pdata = pix[i]; pvalid = 1;
      @(posedge s_clk); #1; pvalid = 0;
    end
    phase = 5;
    for (i = 0; i < 9216; i = i + 1) begin
      wait (wready); @(posedge s_clk); #1; wdata = w_tiles[i]; wvalid = 1;
      @(posedge s_clk); #1; wvalid = 0;
    end
    // result
    fork : wait_sums
      begin
        wait (sums_valid);
        @(posedge s_clk); #1;
        for (k = 0; k < 32; k = k + 1) begin
          if (sums[k] !== exp_sums[k]) begin
            $display("FAIL ch%0d: got %0d want %0d", k, sums[k], exp_sums[k]);
            fails = fails + 1;
          end
        end
      end
      begin
        repeat (4000000) @(posedge s_clk);
        $display("FAIL timeout: sums_valid never asserted");
        fails = fails + 1;
      end
    join_any
    disable fork;
    if (fails == 0) $display("ALL PASS tb_vis_replay (32 channels)");
    else $display("FAILURES=%0d tb_vis_replay", fails);
    $finish;
  end
endmodule