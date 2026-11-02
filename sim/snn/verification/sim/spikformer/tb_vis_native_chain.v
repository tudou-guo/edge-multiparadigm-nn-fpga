// Native vision end-to-end chain TB (P2 gate) -- SKELETON, vectors/golden TBD.
// DUT: spikformer_vis_native_top (SPS -> block -> streaming sums), native params
//   H=W=32, C0=3, C1=96, C2=384, T=4, E=384, HEADS=12, HEAD_DIM=32,
//   HIDDEN=1536, TOKENS=64, TL=8, TC=8, RW=4, PSUM_SLICE=32, SUM_W=6, OW=32.
// Host ABI (derived from RTL, read-only):
//   per image: i_blk_w 222912 beats -> i_sps_w 324 (w1) + 384 (bias 21b pair)
//              -> i_pix 1024 -> i_sps_w 41472 (conv2 tile schedule) ; then
//   i_start; o_done / o_sums_valid sticky. Block reloads all weights per image
//   (native A-round, no on-chip block weight retention across starts).
// Required vectors (NOT PRESENT YET -> run reports NOT_RUN, see TODO at EOF):
//   vectors/block_native/blk_w.hex       222912 beats, S_LDW order
//   vectors/block_native/exp_out.hex     3072 block-out beats (128b nibble)
//   vectors/block_native/exp_sums_img1.hex / exp_sums_img2.hex  384 x 32b
//   vectors/block_native/meta.json       thr_q/k/v/proj/fc1/fc2, seed, order
//   vectors/sps_native_img2/pix.hex      1024 beats (second, different image)
//   (SPS weights reused from vectors/sps_native: same model, only input changes)
// Checks (when vectors exist): 3072 block-out beats, 384 sums, last/done/count,
// two consecutive images, optional host-stream bubbles (+BPS=1). Output-side
// sums backpressure hook does not exist in the A-round top (TODO list).
// RTL is read-only here: no force, no hierarchical writes.
`timescale 1ns/1ps
module tb_vis_native_chain;
  localparam int E = 384;
  localparam int TBEATS = 3072;
  localparam int BLKW = 222912;
  localparam int NW1 = 324, NBB = 384, NPIX = 1024, NSCH = 41472;
  reg s_clk = 0, s_rst = 1;
  always #5 s_clk = ~s_clk;
  reg start = 0;
  reg [63:0] blkw_d; reg blkw_v; wire blkw_r;
  reg [63:0] spsw_d; reg spsw_v; wire spsw_r;
  reg [23:0] pix_d;  reg pix_v;  wire pix_r;
  reg signed [31:0] thr1 = 200, thr2 = 60;
  reg signed [31:0] thr_q = 60, thr_k = 60, thr_v = 60, thr_proj = 60;
  reg signed [31:0] thr_fc1 = 60, thr_fc2 = 60;   // TODO: read from block meta.json
  wire [31:0] sums [0:E-1];
  wire sums_valid, done;

  spikformer_vis_native_top #(
    .H(32), .W_IN(32), .C0(3), .C1(96), .C2(384), .T(4), .LANES(8), .TC2(8),
    .E(E), .HEADS(12), .HEAD_DIM(32), .HIDDEN(1536),
    .TL(8), .TC(8), .RW(4), .PSUM_SLICE(32), .SUM_W(6), .W(8), .OW(32)
  ) dut (
    .s_clk(s_clk), .s_rst(s_rst), .i_start(start), .o_done(done),
    .i_blk_w_data(blkw_d), .i_blk_w_valid(blkw_v), .o_blk_w_ready(blkw_r),
    .i_sps_w_data(spsw_d), .i_sps_w_valid(spsw_v), .o_sps_w_ready(spsw_r),
    .i_pix_data(pix_d), .i_pix_valid(pix_v), .o_pix_ready(pix_r),
    .i_thr_1(thr1), .i_thr_2(thr2),
    .i_thr_q(thr_q), .i_thr_k(thr_k), .i_thr_v(thr_v), .i_thr_proj(thr_proj),
    .i_thr_fc1(thr_fc1), .i_thr_fc2(thr_fc2),
    .o_sums(sums), .o_sums_valid(sums_valid)
  );

  // SPS weight stream is split across files but driven as one 42180-beat stream.
  reg [63:0] w_ld1  [0:NW1-1];
  reg [63:0] b_ld   [0:NBB-1];
  reg [63:0] w_sched[0:NSCH-1];
  reg [63:0] blkw   [0:BLKW-1];
  reg [23:0] pix1   [0:NPIX-1];
  reg [23:0] pix2   [0:NPIX-1];
  reg [127:0] expo1 [0:TBEATS-1];
  reg [127:0] expo2 [0:TBEATS-1];
  reg [31:0] exps1  [0:E-1];
  reg [31:0] exps2  [0:E-1];

  integer i, k, tt, img, fails, ncy, bps, en2, have;
  integer nout, gotb, imgs_done;
  reg [31:0] sum_mismatch;

  function automatic int fexists(input string p);
    int fd;
    fd = $fopen(p, "r");
    if (fd != 0) $fclose(fd);
    return (fd != 0);
  endfunction

  always @(posedge s_clk) ncy = ncy + 1;

  // ---- one-image host stream, called from initial (task with timing) ----
  task automatic host_bubbles;
    begin
      if (bps != 0 && (($urandom % 7) == 0)) @(posedge s_clk);
    end
  endtask

  // drives the full host ABI for one image: block weights, SPS w1, SPS bias,
  // pixels, SPS conv2 schedule; optionally inserts bubbles (+BPS=1).
  task automatic drive_host(input integer use_img2);
    integer j;
    begin
      for (j = 0; j < BLKW; j = j + 1) begin
        host_bubbles;
        @(posedge s_clk); #1; blkw_d = blkw[j]; blkw_v = 1;
        do @(posedge s_clk); while (!(blkw_v && blkw_r));
        #1; blkw_v = 0;
      end
      for (j = 0; j < NW1; j = j + 1) begin
        host_bubbles;
        @(posedge s_clk); #1; spsw_d = w_ld1[j]; spsw_v = 1;
        do @(posedge s_clk); while (!(spsw_v && spsw_r));
        #1; spsw_v = 0;
      end
      for (j = 0; j < NBB; j = j + 1) begin
        host_bubbles;
        @(posedge s_clk); #1; spsw_d = b_ld[j]; spsw_v = 1;
        do @(posedge s_clk); while (!(spsw_v && spsw_r));
        #1; spsw_v = 0;
      end
      for (j = 0; j < NPIX; j = j + 1) begin
        host_bubbles;
        @(posedge s_clk); #1; pix_d = use_img2 ? pix2[j] : pix1[j]; pix_v = 1;
        do @(posedge s_clk); while (!(pix_v && pix_r));
        #1; pix_v = 0;
      end
      for (j = 0; j < NSCH; j = j + 1) begin
        host_bubbles;
        @(posedge s_clk); #1; spsw_d = w_sched[j]; spsw_v = 1;
        do @(posedge s_clk); while (!(spsw_v && spsw_r));
        #1; spsw_v = 0;
      end
    end
  endtask

  // block-out beat checker: TBEATS=3072 beats/image; 128b = TL(8) x T(4) x RW(4)
  // nibbles at (lane*T+t)*RW. Compared against block_native/exp_out_img{1,2}.hex.
  integer outfail = 0, kk, tt2;
  reg [127:0] expw;
  always @(posedge s_clk) begin
    if (dut.blk_ov && dut.blk_ordy) begin
      if (gotb < TBEATS) begin
        expw = (img == 1) ? expo2[gotb] : expo1[gotb];
        for (kk = 0; kk < 8; kk = kk + 1)
          for (tt2 = 0; tt2 < 4; tt2 = tt2 + 1) begin
            if (dut.blk_odata[((kk*4+tt2)+1)*4-1 -: 4] !== expw[((kk*4+tt2)+1)*4-1 -: 4]) begin
              outfail = outfail + 1;
              if (outfail < 12)
                $display("BLKFAIL img%0d beat=%0d lane=%0d t=%0d got=%h want=%h", img, gotb, kk, tt2,
                  dut.blk_odata[((kk*4+tt2)+1)*4-1 -: 4], expw[((kk*4+tt2)+1)*4-1 -: 4]);
            end
          end
        if (dut.blk_olast !== (gotb == TBEATS-1)) begin
          outfail = outfail + 1;
          $display("BLKLAST mismatch img%0d beat=%0d last=%b", img, gotb, dut.blk_olast);
        end
        gotb = gotb + 1;
      end
    end
  end

  always @(posedge s_clk) begin
    if ((ncy % 1000000) == 0)
      $display("MON cyc=%0d top=%0d blk=%0d sps=%0d sums_valid=%b done=%b blkw_r=%b spsw_r=%b pix_r=%b",
        ncy, dut.st, dut.u_blk.st, dut.u_sps.u_ctrl.st, sums_valid, done, blkw_r, spsw_r, pix_r);
  end

  initial begin
    fails = 0; ncy = 0; gotb = 0; bps = 0; en2 = 1; imgs_done = 0;
    start = 0; blkw_v = 0; spsw_v = 0; pix_v = 0;
    $display("TB_CHAIN ident params native H=W=32 C1=96 C2=384 T=4 E=384 TL=8 TOKENS=64");
    // 1) required golden/vector presence gate
    have = 1;
    if (!fexists("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/block_native/blk_w.hex")) begin
      $display("NOT_RUN missing vectors/block_native/blk_w.hex (222912 beats, S_LDW order)"); have = 0;
    end
    if (!fexists("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/block_native/exp_out_img1.hex")) begin
      $display("NOT_RUN missing vectors/block_native/exp_out_img1.hex (3072 block-out beats)"); have = 0;
    end
    if (!fexists("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/block_native/exp_sums_img1.hex")) begin
      $display("NOT_RUN missing vectors/block_native/exp_sums_img1.hex (384 sums)"); have = 0;
    end
    if (!fexists("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/sps_native_tc2_8/w_sched.hex")) begin
      $display("NOT_RUN missing vectors/sps_native_tc2_8/w_sched.hex (TC2=8 SPS schedule)"); have = 0;
    end
    if (!fexists("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/sps_native_img2/pix.hex")) begin
      $display("NOT_RUN missing vectors/sps_native_img2/pix.hex (second image)"); have = 0;
    end
    if (have == 0) begin
      $display("NOT_RUN tb_vis_native_chain: golden/vectors absent; skeleton compiles, run gated off");
      $display("TODO see docs/VISION_NATIVE_SIM_PROGRESS_20261031.md");
      $finish;
    end
    // 2) load vectors (all present)
    $readmemh("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/block_native/blk_w.hex", blkw);
    $readmemh("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/block_native/exp_out_img1.hex", expo1);
    $readmemh("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/block_native/exp_out_img2.hex", expo2);
    if (fexists("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/block_native/exp_sums_img2.hex"))
      $readmemh("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/block_native/exp_sums_img2.hex", exps2);
    $readmemh("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/block_native/exp_sums_img1.hex", exps1);
    $readmemh("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/sps_native_tc2_8/w_ld1.hex", w_ld1);
    $readmemh("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/sps_native_tc2_8/b_ld.hex", b_ld);
    $readmemh("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/sps_native_tc2_8/w_sched.hex", w_sched);
    $readmemh("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/sps_native_tc2_8/pix.hex", pix1);
    $readmemh("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/sps_native_img2/pix.hex", pix2);
    $display("VEC blk_w=%0d expo1[0]=%h exps1[0]=%0d pix2[0]=%h", BLKW, expo1[0], exps1[0], pix2[0]);
    if ($value$plusargs("BPS=%d", bps)) $display("BPS mode: host bubbles on (+BPS=%0d)", bps);
    if ($value$plusargs("IMG2=%d", en2)) ;
    repeat (4) @(posedge s_clk); s_rst = 0;
    fork : run
      begin : images
        for (img = 0; img < ((en2 != 0) ? 2 : 1); img = img + 1) begin
          gotb = 0; outfail = 0;
          $display("IMG%0d start (block weights + SPS stream/pixels)", img);
          @(posedge s_clk); #1; start = 1;
          @(posedge s_clk); #1; start = 0;
          drive_host(img);
          wait (sums_valid);
          @(posedge s_clk); #1;
          sum_mismatch = 0;
          for (k = 0; k < E; k = k + 1) begin
            if (sums[k] !== (img == 0 ? exps1[k] : exps2[k])) begin
              sum_mismatch = sum_mismatch + 1;
              if (sum_mismatch < 16)
                $display("SUMFAIL img%0d ch%0d got=%0d want=%0d", img, k, sums[k],
                  (img == 0 ? exps1[k] : exps2[k]));
            end
          end
          if (sum_mismatch == 0) $display("SUMS PASS img%0d (384 sums)", img);
          else begin $display("SUMS FAIL img%0d mismatches=%0d", img, sum_mismatch); fails = fails + 1; end
          if (!done) begin $display("DONE missing after sums_valid img%0d", img); fails = fails + 1; end
          if (gotb != TBEATS) begin $display("BLKOUT count img%0d got=%0d want=%0d", img, gotb, TBEATS); fails = fails + 1; end
          if (outfail == 0) $display("BLKOUT PASS img%0d (%0d beats bit-exact, last ok)", img, gotb);
          else begin $display("BLKOUT FAIL img%0d nibble mismatches=%0d", img, outfail); fails = fails + 1; end
          imgs_done = imgs_done + 1;
          repeat (20) @(posedge s_clk);
        end
      end
      begin
        repeat (120000000) @(posedge s_clk);
        $display("FAIL timeout img=%0d top=%0d blk=%0d sps=%0d sums_valid=%b done=%b",
          img, dut.st, dut.u_blk.st, dut.u_sps.u_ctrl.st, sums_valid, done);
        fails = fails + 1;
      end
    join_any
    disable fork;
    // NOTE: report the ACTUAL image count (en2=0 single-image runs must not
    // claim two images). Each finished image = TBEATS block-out + E sums checked.
    if (fails == 0)
      $display("ALL PASS tb_vis_native_chain (images=%0d, each: %0d block-out beats bit-exact + %0d sums; en2=%0d)",
        imgs_done, TBEATS, E, en2);
    else
      $display("FAILURES=%0d tb_vis_native_chain (images_done=%0d en2=%0d)", fails, imgs_done, en2);
    $finish;
  end
endmodule
// ---------------------------------------------------------------------------
// TODO (blocking P2 gate; negotiated with img-p1 / P2 golden owner):
//  1) Native block golden generator (extend gen_block_replay.py or new
//     gen_block_param.py): E=384 HEADS=12 HD=32 HIDDEN=1536 T=4 TOKENS=64
//     TL=8 TC=8 RW=4, same arithmetic as gen_block_replay.py; token source =
//     vectors/sps_native/exp.hex (token-major, 48 groups/beat); emit the exact
//     S_LDW host order: wq,wk,wv,wp (EEB=18432 beats ea) -> bq,bk,bv,bp
//     (E/2=192 ea, 2x signed PSUM_SLICE per beat, lo=even hi=odd) -> fc1
//     (FCD=73728) -> bf1 (HIDDEN/2=768) -> fc2 (FCD=73728) -> bf2 (192)
//     => 222912 beats; plus exp_out.hex (3072 x TL*T*RW nibble beats),
//     exp_sums_img1.hex (384 x 32b), meta.json (thresholds/seed/order).
//  2) Second image vectors: same SPS weights, new pixels; emit
//     vectors/sps_native_img2/{pix.hex,exp.hex} + block exp_sums_img2.hex
//     (reuse weights only; do NOT regenerate conv weights).
//  3) Output-side sums backpressure hook: the A-round native top hardwires
//     blk_ordy in N_RUN, so an external stall of block-out/sums is impossible
//     without an RTL test hook (not allowed this round). Input-side bubbles are
//     covered by +BPS=1. Record as ABI gap / P2 protocol TODO.
//  4) Compare exp_out.hex per beat, not only 384 sums (P2 gate requires all
//     3072 block-out beats); current skeleton counts beats + samples first 8.
// ---------------------------------------------------------------------------
