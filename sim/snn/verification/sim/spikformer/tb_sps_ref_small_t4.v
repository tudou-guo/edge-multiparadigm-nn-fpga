// P1 T=4 variant: same small geometry with T=4 (native timestep count).
// Checks the T-dependent paths: p2row 16*T layout, 128b flush zero pad,
// LANES*T=32b raw beats -> 128b RW4 expansion, 2x2 pool2 over T=4.
// Golden: build/spikformer/vision_ref_20261031/vectors/sps_small_t4_check/.
// Checks all 16 token beats (adapter-expanded nibbles 0/1) + last + done.
`timescale 1ns/1ps
module tb_sps_ref_small_t4;
  reg s_clk = 0, s_rst = 1, start = 0;
  always #5 s_clk = ~s_clk;
  reg [63:0] wdata; reg wvalid; wire wready;
  reg [23:0] pdata; reg pvalid; wire pready;
  reg [15:0] thr1 = 200, thr2 = 60;
  wire [8*4*4-1:0] tdata; wire tvalid, tlast; reg tready = 1;
  wire done;
  spiking_sps_ref_top #(.H(8), .W_IN(8), .C0(3), .C1(16), .C2(32),
    .T(4), .LANES(8), .TC2(4), .WT_W(8), .ACC_W(21), .RW(4)) dut (
    .s_clk(s_clk), .s_rst(s_rst), .i_thr_1(thr1), .i_thr_2(thr2),
    .i_weight_in(wdata), .i_weight_valid(wvalid), .o_weight_ready(wready),
    .i_pix_valid(pvalid), .o_pix_ready(pready), .i_pix_data(pdata),
    .i_start(start), .o_done(done),
    .o_tok_data(tdata), .o_tok_valid(tvalid), .i_tok_ready(tready), .o_tok_last(tlast)
  );
  reg [63:0] w_ld1 [0:53];
  reg [63:0] b_ld [0:31];
  reg [63:0] w_sched [0:575];
  reg [23:0] pix [0:63];
  reg [3:0] expb [0:15][0:7];
  integer i, k, fails = 0, got = 0;
  function automatic [3:0] hexv(input [7:0] ch);
    case (ch)
      "0": hexv = 0; "1": hexv = 1; "2": hexv = 2; "3": hexv = 3;
      "4": hexv = 4; "5": hexv = 5; "6": hexv = 6; "7": hexv = 7;
      "8": hexv = 8; "9": hexv = 9; "a", "A": hexv = 10; "b", "B": hexv = 11;
      "c", "C": hexv = 12; "d", "D": hexv = 13; "e", "E": hexv = 14; "f", "F": hexv = 15;
      default: hexv = 0;
    endcase
  endfunction
  initial begin
    $readmemh("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/sps_small_t4_check/w_ld1.hex", w_ld1);
    $readmemh("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/sps_small_t4_check/b_ld.hex", b_ld);
    $readmemh("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/sps_small_t4_check/w_sched.hex", w_sched);
    $readmemh("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/sps_small_t4_check/pix.hex", pix);
    // exp.hex: 16 lines of 8 chars; char a = lane a (T=2 bits)
    begin
      integer f, c;
      reg [8*8-1:0] line;
      f = $fopen("D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/sps_small_t4_check/exp.hex", "r");
      for (i = 0; i < 16; i = i + 1) begin
        c = $fscanf(f, "%s\n", line);
        for (k = 0; k < 8; k = k + 1) begin
          expb[i][k] = hexv(line[(8-k)*8-1 -: 8]);
        end
      end
      $fclose(f);
    end
    $display("EXPB0 %h %h %h %h %h %h %h %h", expb[0][0], expb[0][1], expb[0][2], expb[0][3], expb[0][4], expb[0][5], expb[0][6], expb[0][7]);
    wvalid = 0; pvalid = 0;
    repeat (4) @(posedge s_clk); s_rst = 0;
    @(posedge s_clk); start = 1; @(posedge s_clk); start = 0;
    // weight stream in controller schedule order
    for (i = 0; i < 54; i = i + 1) begin
      @(posedge s_clk); #1; wdata = w_ld1[i]; wvalid = 1;
      do @(posedge s_clk); while (!(wvalid && wready));
      #1; wvalid = 0;
    end
    for (i = 0; i < 32; i = i + 1) begin
      @(posedge s_clk); #1; wdata = b_ld[i]; wvalid = 1;
      do @(posedge s_clk); while (!(wvalid && wready));
      #1; wvalid = 0;
    end
    for (i = 0; i < 64; i = i + 1) begin
      @(posedge s_clk); #1; pdata = pix[i]; pvalid = 1;
      do @(posedge s_clk); while (!(pvalid && pready));
      #1; pvalid = 0;
    end
    for (i = 0; i < 576; i = i + 1) begin
      @(posedge s_clk); #1; wdata = w_sched[i]; wvalid = 1;
      do @(posedge s_clk); while (!(wvalid && wready));
      #1; wvalid = 0;
    end
  end
  // progress monitor
  always @(posedge s_clk) begin
    if ($time % 200000 == 0) $display("MON t=%0t enc_st=%0d arr_st=%0d psum_qv=%b ctrl_st=%0d wready=%b pready=%b", $time,
      dut.u_enc.st, dut.u_arr.st, dut.u_psum.q_v, dut.u_ctrl.st, wready, pready);
  end
  // array result monitor (first windows)
  integer mon_n = 0;
  always @(posedge s_clk) begin
    if (dut.u_arr.o_res_valid && dut.u_arr.i_res_ready && mon_n < 12) begin
      $display("ARRRES t=%0t pos=%0d grp=%0d first=%b last=%b data=%h", $time,
        dut.u_arr.o_res_pos, dut.u_arr.o_res_grp, dut.u_arr.o_res_first,
        dut.u_arr.o_res_last, dut.u_arr.o_res_data);
      mon_n = mon_n + 1;
    end
  end
  // full-beat dump for first 4 beats
  always @(posedge s_clk) begin
    if (tvalid && tready && got < 4) begin
      $display("BEATDUMP beat%0d got=%h", got, tdata);
    end
  end
  // encoder raw beat monitor
  integer enc_n = 0;
  always @(posedge s_clk) begin
    if (dut.u_ctrl.i_enc_valid && dut.u_ctrl.o_enc_ready && enc_n < 8) begin
      $display("ENCBEAT n=%0d data=%h", enc_n, dut.u_ctrl.i_enc_beat);
      enc_n = enc_n + 1;
    end
  end
  // psum beat + array window counters
  integer psum_n = 0, win_n = 0;
  always @(posedge s_clk) begin
    if (dut.u_ctrl.i_psum_beat_valid && dut.u_ctrl.o_psum_beat_ready) psum_n = psum_n + 1;
    if (dut.u_ctrl.o_arr_win_valid && dut.u_arr.o_win_ready) win_n = win_n + 1;
  end
  // capture trace: every p2row write
  always @(posedge s_clk) begin
    if ((dut.u_ctrl.st == 8 || dut.u_ctrl.st == 7 || dut.u_ctrl.st == 6 || dut.u_ctrl.st == 10)
        && dut.u_ctrl.i_psum_beat_valid && dut.u_ctrl.o_psum_beat_ready && dut.u_ctrl.p2cnt < 4) begin
      $display("CAP pos=%0d grp=%0d idx=%0d data=%h", dut.u_ctrl.i_psum_beat_pos,
        dut.u_ctrl.i_psum_beat_grp, dut.u_ctrl.p2cnt, dut.u_ctrl.i_psum_beat);
    end
  end
  // flush monitor: p2row at C_C2W entry (first cycle, o_b0_we low)
  always @(posedge s_clk) begin
    if (dut.u_ctrl.o_dbg_c2w && !dut.u_ctrl.o_b0_we && dut.u_ctrl.p2fl == 0) begin
      $display("C2W_ENTRY g=%0d fpr=%0d p2row0=%h p2row1=%h p2cnt=%0d", dut.u_ctrl.c_g2,
        dut.u_ctrl.c_fpr, dut.u_ctrl.p2row[0], dut.u_ctrl.p2row[1], dut.u_ctrl.p2cnt);
    end
  end
  // psum beat monitor (pos, grp, beat)
  integer pb_n = 0;
  always @(posedge s_clk) begin
    if (dut.u_ctrl.i_psum_beat_valid && dut.u_ctrl.o_psum_beat_ready && pb_n < 64) begin
      $display("PSUMBEAT pos=%0d grp=%0d data=%h", dut.u_ctrl.i_psum_beat_pos,
        dut.u_ctrl.i_psum_beat_grp, dut.u_ctrl.i_psum_beat);
      pb_n = pb_n + 1;
    end
  end
  // encoder psum monitor (pos, oc, psum)
  integer ps_n = 0;
  always @(posedge s_clk) begin
    if (dut.u_enc.st == 3 && ps_n < 20) begin // E_LIF=3
      $display("ENCPSUM pos=%0d oc=%0d psum=%0d", dut.u_enc.pos,
        dut.u_enc.grp * 8 + dut.u_enc.lane, $signed(dut.u_enc.psum));
      ps_n = ps_n + 1;
    end
  end
  // ---- P1 structural checks -------------------------------------------------
  // Independent of the golden compare: (1) every captured psum beat must follow
  // the exact (grp,pos) schedule order with no gap/drop; (2) at flush entry the
  // latched row identity must be the next row-major row and p2row must be full;
  // (3) totals must be exactly 64 captures / 16 flushes (H1*W1*G2, G2*H1).
  integer chk_fails = 0, cap_n = 0, flush_n = 0;
  always @(posedge s_clk) begin
    if (dut.u_ctrl.i_psum_beat_valid && dut.u_ctrl.o_psum_beat_ready) begin
      if ($isunknown(dut.u_ctrl.i_psum_beat)) begin
        chk_fails = chk_fails + 1; $display("CHKFAIL capture X n=%0d", cap_n);
      end
      if (dut.u_ctrl.i_psum_beat_pos !== (cap_n % 16) || dut.u_ctrl.i_psum_beat_grp !== (cap_n / 16)) begin
        chk_fails = chk_fails + 1;
        $display("CHKFAIL capture seq n=%0d pos=%0d want=%0d grp=%0d want=%0d", cap_n,
          dut.u_ctrl.i_psum_beat_pos, cap_n % 16, dut.u_ctrl.i_psum_beat_grp, cap_n / 16);
      end
      cap_n = cap_n + 1;
    end
    if (dut.u_ctrl.o_dbg_c2w && !dut.u_ctrl.o_b0_we && dut.u_ctrl.p2fl == 0) begin
      if (dut.u_ctrl.p2cnt !== 4) begin
        chk_fails = chk_fails + 1;
        $display("CHKFAIL flush entry p2cnt=%0d want=4 n=%0d", dut.u_ctrl.p2cnt, flush_n);
      end
      if (dut.u_ctrl.p2_g !== (flush_n / 4) || dut.u_ctrl.p2_y !== (flush_n % 4)) begin
        chk_fails = chk_fails + 1;
        $display("CHKFAIL flush rowid n=%0d g=%0d y=%0d want_g=%0d want_y=%0d", flush_n,
          dut.u_ctrl.p2_g, dut.u_ctrl.p2_y, flush_n / 4, flush_n % 4);
      end
      flush_n = flush_n + 1;
    end
  end
  // beat checker
  initial begin
    @(negedge s_rst);
    fork : watch
      begin
        while (got < 16) begin
          @(posedge s_clk);
          if (tvalid && tready) begin
            if ($isunknown(tdata)) begin $display("BEATFAIL beat%0d has X: %h", got, tdata); fails = fails + 1; end
            for (k = 0; k < 8; k = k + 1) begin
              for (integer tt = 0; tt < 4; tt = tt + 1) begin
                if (tdata[((k*4+tt)+1)*4-1 -: 4] !== {3'b0, expb[got][k][tt]}) begin
                  $display("BEATFAIL beat%0d lane%0d t%0d got=%h want=%b (full=%h)", got, k, tt,
                    tdata[((k*4+tt)+1)*4-1 -: 4], expb[got][k][tt], tdata);
                  fails = fails + 1;
                end
              end
            end
            if (got == 15 && !tlast) begin $display("LAST missing on final beat"); fails = fails + 1; end
            if (got < 15 && tlast) begin $display("LAST early on beat%0d", got); fails = fails + 1; end
            got = got + 1;
          end
        end
        wait (done);
        if (cap_n != 64) begin chk_fails = chk_fails + 1; $display("CHKFAIL cap_n=%0d want=64", cap_n); end
        if (flush_n != 16) begin chk_fails = chk_fails + 1; $display("CHKFAIL flush_n=%0d want=16", flush_n); end
        if (fails == 0 && chk_fails == 0)
          $display("ALL PASS tb_sps_ref_small_t4 (16 beats; capture=%0d flush=%0d structural ok)", cap_n, flush_n);
        else $display("FAILURES=%0d chkfail=%0d tb_sps_ref_small_t4", fails, chk_fails);
        $finish;
      end
      begin
        repeat (3000000) @(posedge s_clk);
        $display("FAIL timeout: got=%0d done=%b", got, done);
        $finish;
      end
    join_any
    disable fork;
  end
endmodule