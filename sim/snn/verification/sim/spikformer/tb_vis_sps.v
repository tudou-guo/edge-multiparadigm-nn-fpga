// SPS-only probe for spikformer_vis_top: streams the host stimulus in the
// same order the SPS replay bench uses and checks the raw SPS token beats.
`timescale 1ns/1ps
module tb_vis_sps;
  reg s_clk = 0, s_rst = 1, start = 0;
  reg [63:0] wdata; reg wvalid; wire wready;
  reg [23:0] pdata; reg pvalid; wire pready;
  reg signed [31:0] thr1 = 200, thr2 = 60, thr = 60;
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
  reg [31:0] expb [0:15];
  integer i, got, fails, ncy = 0, phase = 0, seen_tile = 0, sps_tiles = 0, tile_dbg = 0;
  integer fd_f1, yi, xi, ci, ti, wv;
  integer lane;
  reg pready_chk = 0;
  reg [15:0] gotbeat;

  always @(posedge s_clk) begin
    if (!s_rst && dut.sps_tvalid && dut.sps_tready) begin
      gotbeat = dut.sps_tok;
      if (got < 16) begin
        for (lane = 0; lane < 8; lane = lane + 1) begin
          if (gotbeat[lane*2 +: 2] !== ((expb[got] >> (4*(7-lane))) & 2'b11)) begin
            $display("SPSFAIL beat%0d lane%0d got %0d want %0d", got, lane,
                     gotbeat[lane*2 +: 2], (expb[got] >> (4*(7-lane))) & 2'b11);
            fails = fails + 1;
          end
        end
        got = got + 1;
      end
    end
    if (dut.st == 4 && dut.i_w_valid) seen_tile = seen_tile + 1;
    if (dut.st == 5 && dut.sps_wvalid && dut.sps_wready) sps_tiles = sps_tiles + 1;
    if (dut.st == 5 && dut.sps_wready && !pready_chk) begin
      if (tile_dbg < 12) begin
        $display("TILE c=%0d bi=%0d swr=1 sst=%0d tlos=%b tk=%0d", ncy, dut.bicnt,
                 dut.u_sps.st, dut.u_sps.o_tok_last, got);
        tile_dbg = tile_dbg + 1;
      end
    end
    ncy = ncy + 1;
    if (ncy % 4000 == 0)
      $display("seen_tile=%0d phase=%0d i=%0d MON c=%0d st=%0d c1=%0d ldc0=%0d q=%0d ox=%0d oy=%0d grp=%0d pready=%b pv=%b swr=%b spsst=%0d bi=%0d iv=%b",
        seen_tile, phase, i, ncy, dut.st, dut.c1cnt, dut.u_sps.ldc0, dut.u_sps.q, dut.u_sps.ox1, dut.u_sps.oy1,
        dut.u_sps.grp1, pready, pvalid, dut.sps_wready, dut.u_sps.st, dut.bicnt, dut.i_w_valid);
  end

  initial begin
    $readmemh("D:/dl_snn/build/spikformer/sps_replay/w_ld1.hex", w_ld1);
    $readmemh("D:/dl_snn/build/spikformer/sps_replay/b_ld.hex", b_ld);
    $readmemh("D:/dl_snn/build/spikformer/sps_replay/w_tiles.hex", w_tiles);
    $readmemh("D:/dl_snn/build/spikformer/sps_replay/pix.hex", pix);
    $readmemh("D:/dl_snn/build/spikformer/sps_replay/exp.hex", expb);
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
    wvalid = 0; pvalid = 0; fails = 0; got = 0;
    repeat (4) @(posedge s_clk); s_rst = 0;
    @(posedge s_clk); start = 1; @(posedge s_clk); start = 0;
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
    @(posedge s_clk);
    $display("F0 (0,0) = %02h %02h %02h  expect %02h %02h %02h",
      dut.u_sps.f0[0][0][0], dut.u_sps.f0[0][0][1], dut.u_sps.f0[0][0][2],
      pix[0][7:0], pix[0][15:8], pix[0][23:16]);
    $display("F0 (1,3) = %02h %02h %02h  expect %02h %02h %02h",
      dut.u_sps.f0[1][3][0], dut.u_sps.f0[1][3][1], dut.u_sps.f0[1][3][2],
      pix[11][7:0], pix[11][15:8], pix[11][23:16]);
    $display("WBANK1[0][0][0] = %0d  bias1[0] = %0d", dut.u_sps.wbank1[0][0][0], dut.u_sps.bias1[0]);
    $display("F1 g0(0,0)=%04h g0(1,1)=%04h g1(0,0)=%04h g0(7,7)=%04h",
      dut.u_sps.f1[0][0][15:0], dut.u_sps.f1[1][1][15:0],
      dut.u_sps.f1[0][0][31:16], dut.u_sps.f1[7][7][15:0]);
    $display("SPS tiles consumed=%0d  bias2f[0]=%0d bias2f[1]=%0d",
      sps_tiles, dut.u_sps.bias2f[0], dut.u_sps.bias2f[1]);

    repeat (100000) @(posedge s_clk);
    if (fails == 0 && got == 16) $display("ALL PASS tb_vis_sps (16 beats)");
    else $display("FAILURES=%0d got=%0d tb_vis_sps", fails, got);
    $finish;
  end
endmodule