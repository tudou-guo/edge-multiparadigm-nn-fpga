// tb_cfc_seq.sv -- AXI-Lite end-to-end sequence test (contract 3.5).
//
// Checks, all against the frozen V1.1 integer semantics:
//   * ID / VERSION / reset defaults / unimplemented offsets read 0
//   * 64-step run: OUTPUT and H_FINAL bit exact vs export/gen/y_expected.hex,
//     h_expected.hex and vs the embedded int_reference model, STEP_COUNT = 64,
//     BUSY=0 DONE=1, RUN_CYCLES > 0, irq sticky, CLEAR_STATUS clears them
//   * state-clear repeat run gives identical results / RUN_CYCLES
//   * 32 + 32 segmented run with state kept equals the continuous 64-step run
//   * MODE[1]=1 (clear h_old each step) matches the integer reference
`timescale 1ns/1ps
module tb_cfc_seq;

  logic clk;
  initial clk = 1'b0;
  always #2.5 clk = ~clk;

  logic rst;
  logic [11:0] s_awaddr;
  logic        s_awvalid;
  logic        s_awready;
  logic [31:0] s_wdata;
  logic [3:0]  s_wstrb;
  logic        s_wvalid;
  logic        s_wready;
  logic [1:0]  s_bresp;
  logic        s_bvalid;
  logic        s_bready;
  logic [11:0] s_araddr;
  logic        s_arvalid;
  logic        s_arready;
  logic [31:0] s_rdata;
  logic [1:0]  s_rresp;
  logic        s_rvalid;
  logic        s_rready;
  logic        irq;

  cfc_platform_wrapper u_dut (
    .clk           (clk),
    .rst           (rst),
    .s_axil_awaddr (s_awaddr),
    .s_axil_awvalid(s_awvalid),
    .s_axil_awready(s_awready),
    .s_axil_wdata  (s_wdata),
    .s_axil_wstrb  (s_wstrb),
    .s_axil_wvalid (s_wvalid),
    .s_axil_wready (s_wready),
    .s_axil_bresp  (s_bresp),
    .s_axil_bvalid (s_bvalid),
    .s_axil_bready (s_bready),
    .s_axil_araddr (s_araddr),
    .s_axil_arvalid(s_arvalid),
    .s_axil_arready(s_arready),
    .s_axil_rdata  (s_rdata),
    .s_axil_rresp  (s_rresp),
    .s_axil_rvalid (s_rvalid),
    .s_axil_rready (s_rready),
    .irq           (irq)
  );

  // reference parameters (same generated initialisation as the RTL ROMs)
  logic signed [15:0] w_rom       [0:83];
  logic signed [31:0] b_rom       [0:16];
  logic signed [15:0] tanh_rom    [0:1024];
  logic signed [15:0] sigmoid_rom [0:1024];
  `include "cfc_weight_init.svh"
  `include "cfc_lut_init.svh"
  `include "cfc_ref_model.svh"

  logic signed [15:0] x_vec  [0:63];
  logic signed [15:0] y_exp  [0:63];
  logic signed [15:0] h_exp  [0:255];

  logic signed [15:0] out_run1 [0:63];
  logic signed [15:0] out_run2 [0:63];
  logic signed [15:0] out_seg1 [0:31];
  logic signed [15:0] out_seg2 [0:31];
  logic signed [15:0] out_ce   [0:63];
  logic signed [15:0] hfin_run1 [0:3];
  logic signed [15:0] hfin_seg2 [0:3];
  logic signed [15:0] hfin_ce   [0:3];

  logic signed [15:0] ref_y    [0:63];
  logic signed [15:0] ref_hst  [0:255];
  logic signed [15:0] ref_y_ce [0:63];
  logic signed [15:0] ref_h_ce [0:255];

  int errors    = 0;
  int mism      = 0;
  int ncheck    = 0;
  int y_mism    = 0;
  int h_mism    = 0;
  int rep_mism  = 0;
  int seg_mism  = 0;
  int ce_mism   = 0;
  logic [31:0] rc_run1;

  // ------------------------------------------------------------------ AXI tasks
  task automatic axil_write (input logic [11:0] addr, input logic [31:0] data);
    begin
      @(posedge clk);
      s_awaddr  <= addr;
      s_wdata   <= data;
      s_wstrb   <= 4'hF;
      s_awvalid <= 1'b1;
      s_wvalid  <= 1'b1;
      @(posedge clk);
      s_awvalid <= 1'b0;
      s_wvalid  <= 1'b0;
      while (!s_bvalid) @(posedge clk);
      s_bready <= 1'b1;
      @(posedge clk);
      s_bready <= 1'b0;
    end
  endtask

  task automatic axil_read (input logic [11:0] addr, output logic [31:0] data);
    begin
      @(posedge clk);
      s_araddr  <= addr;
      s_arvalid <= 1'b1;
      @(posedge clk);
      s_arvalid <= 1'b0;
      while (!s_rvalid) @(posedge clk);
      data = s_rdata;
      s_rready <= 1'b1;
      @(posedge clk);
      s_rready <= 1'b0;
    end
  endtask

  // ------------------------------------------------------------- checks helpers
  task automatic cmp32 (input logic [31:0] got, input logic [31:0] exp, input string tag);
    begin
      ncheck = ncheck + 1;
      if (got !== exp) begin
        mism = mism + 1;
        if (mism <= 25) $display("CFC_TB_SEQ %s mismatch : got=0x%08h exp=0x%08h", tag, got, exp);
      end
    end
  endtask

  task automatic cmp16 (input logic signed [15:0] got, input logic signed [15:0] exp,
                        input string tag, input int idx, input int cat);
    begin
      ncheck = ncheck + 1;
      if (got !== exp) begin
        mism = mism + 1;
        case (cat)
          0: y_mism   = y_mism + 1;
          1: h_mism   = h_mism + 1;
          2: rep_mism = rep_mism + 1;
          3: seg_mism = seg_mism + 1;
          default: ce_mism = ce_mism + 1;
        endcase
        if (mism <= 25) $display("CFC_TB_SEQ %s[%0d] mismatch : got=%0d exp=%0d", tag, idx, got, exp);
      end
    end
  endtask

  // ------------------------------------------------------------ board-level ops
  task automatic read_out_word (input int idx, output logic signed [15:0] v);
    logic [31:0] w;
    begin
      axil_read(12'h200 + 12'd4*idx, w);
      v = $signed(w[15:0]);
      ncheck = ncheck + 1;
      if (w[31:16] !== {16{w[15]}}) begin
        mism = mism + 1;
        if (mism <= 25) $display("CFC_TB_SEQ OUTPUT[%0d] sign extension wrong : 0x%08h", idx, w);
      end
    end
  endtask

  task automatic read_hfinal_word (input int idx, output logic signed [15:0] v);
    logic [31:0] w;
    begin
      axil_read(12'h080 + 12'd4*idx, w);
      v = $signed(w[15:0]);
      ncheck = ncheck + 1;
      if (w[31:16] !== {16{w[15]}}) begin
        mism = mism + 1;
        if (mism <= 25) $display("CFC_TB_SEQ H_FINAL[%0d] sign extension wrong : 0x%08h", idx, w);
      end
    end
  endtask

  task automatic wait_run_end (input int max_polls, output logic ok, output logic err);
    logic [31:0] st;
    int p;
    begin
      ok = 1'b0;
      err = 1'b0;
      p = 0;
      while (p < max_polls) begin
        axil_read(12'h00C, st);
        if (st[1]) begin
          ok = 1'b1;
          p  = max_polls;
        end else if (st[2]) begin
          err = 1'b1;
          p   = max_polls;
        end else begin
          p = p + 1;
        end
      end
      if ((p == max_polls) && !ok && !err) errors = errors + 1;
    end
  endtask

  task automatic hw_run (input int n_steps, input logic [31:0] mode,
                         output logic ok, output logic err,
                         output logic [31:0] rc, output logic [31:0] sc, output logic [31:0] st);
    begin
      axil_write(12'h014, n_steps[31:0]);   // N_STEPS
      axil_write(12'h018, 32'd655);         // DELTA_T
      axil_write(12'h01C, mode);            // MODE
      axil_write(12'h008, 32'h0000_0001);   // CONTROL.START
      wait_run_end(40000, ok, err);
      axil_read(12'h00C, st);
      axil_read(12'h020, rc);
      axil_read(12'h024, sc);
    end
  endtask

  // ------------------------------------------------------- integer reference run
  task automatic ref_compute (input int n, input bit clear_each);
    logic signed [15:0] h0, h1, h2, h3, y, n0, n1, n2, n3;
    begin
      h0 = 16'sd0; h1 = 16'sd0; h2 = 16'sd0; h3 = 16'sd0;
      for (int t = 0; t < n; t = t + 1) begin
        if (clear_each)
          ref_step(x_vec[t], 16'sd0, 16'sd0, 16'sd0, 16'sd0, 16'sd655, y, n0, n1, n2, n3);
        else
          ref_step(x_vec[t], h0, h1, h2, h3, 16'sd655, y, n0, n1, n2, n3);
        h0 = n0; h1 = n1; h2 = n2; h3 = n3;
        if (clear_each) begin
          ref_y_ce[t]      = y;
          ref_h_ce[4*t+0]  = h0;
          ref_h_ce[4*t+1]  = h1;
          ref_h_ce[4*t+2]  = h2;
          ref_h_ce[4*t+3]  = h3;
        end else begin
          ref_y[t]        = y;
          ref_hst[4*t+0]  = h0;
          ref_hst[4*t+1]  = h1;
          ref_hst[4*t+2]  = h2;
          ref_hst[4*t+3]  = h3;
        end
      end
    end
  endtask

  // ------------------------------------------------------------------ stimulus
  int fh;
  logic [31:0] tmp, st, rc, sc;
  logic ok, err;

  initial begin : stimulus
    rst = 1'b1;
    s_awaddr = 12'd0; s_awvalid = 1'b0;
    s_wdata = 32'd0; s_wstrb = 4'd0; s_wvalid = 1'b0;
    s_bready = 1'b0;
    s_araddr = 12'd0; s_arvalid = 1'b0; s_rready = 1'b0;

    fh = $fopen("../export/gen/x_input.hex", "r");
    if (fh == 0) begin
      $display("CFC_TB_FAIL tb_cfc_seq : missing ../export/gen/x_input.hex");
      $fatal(1, "missing vector file");
    end
    $fclose(fh);

    repeat (8) @(posedge clk);
    #1 rst = 1'b0;
    repeat (4) @(posedge clk);

    $readmemh("../export/gen/x_input.hex", x_vec);
    $readmemh("../export/gen/y_expected.hex", y_exp);
    $readmemh("../export/gen/h_expected.hex", h_exp);

    // -------------------------------------------------- 1) identity / defaults
    axil_read(12'h000, tmp); cmp32(tmp, 32'h43464331, "ID");
    axil_read(12'h004, tmp); cmp32(tmp, 32'h00010000, "VERSION");
    axil_read(12'h02C, tmp); cmp32(tmp, 32'd300000000, "CLOCK_HZ");
    axil_read(12'h00C, tmp); cmp32(tmp, 32'd0, "STATUS_reset");
    axil_read(12'h010, tmp); cmp32(tmp, 32'd0, "ERROR_CODE_reset");
    axil_read(12'h014, tmp); cmp32(tmp, 32'd64, "N_STEPS_default");
    axil_read(12'h018, tmp); cmp32(tmp, 32'd655, "DELTA_T_default");
    axil_read(12'h01C, tmp); cmp32(tmp, 32'd1, "MODE_default");
    axil_read(12'h028, tmp); cmp32(tmp, 32'd1000000, "TIMEOUT_default");
    axil_read(12'h040, tmp); cmp32(tmp, 32'd0, "unimplemented_read_zero");
    if (irq !== 1'b0) begin errors = errors + 1; $display("CFC_TB_SEQ irq should be 0 at reset"); end

    // ------------------------------------------- 2) full 64-step run (MODE=1)
    for (int i = 0; i < 64; i = i + 1)
      axil_write(12'h100 + 12'd4*i, {16'd0, x_vec[i][15:0]});
    axil_read(12'h100, tmp); cmp32(tmp, {16'd0, x_vec[0][15:0]}, "INPUT_readback");
    axil_write(12'h008, 32'h0000_0002);   // CLEAR_STATE (idle)

    hw_run(64, 32'h0000_0001, ok, err, rc, sc, st);
    if (!ok) begin errors = errors + 1; $display("CFC_TB_SEQ run1 did not report DONE"); end
    if (err) begin errors = errors + 1; $display("CFC_TB_SEQ run1 reported ERROR (st=0x%08h)", st); end
    cmp32(st[0], 32'd0, "run1_STATUS_BUSY");
    cmp32(st[1], 32'd1, "run1_STATUS_DONE");
    cmp32(st[2], 32'd0, "run1_STATUS_ERROR");
    cmp32(sc, 32'd64, "run1_STEP_COUNT");
    if (irq !== 1'b1) begin errors = errors + 1; $display("CFC_TB_SEQ run1 irq should be 1"); end
    if ((rc == 0) || (rc > 32'd200000)) begin
      errors = errors + 1;
      $display("CFC_TB_SEQ run1 RUN_CYCLES out of range: %0d", rc);
    end
    rc_run1 = rc;
    $display("CFC_TB_SEQ run1 done, RUN_CYCLES=%0d STEP_COUNT=%0d", rc, sc);

    for (int i = 0; i < 64; i = i + 1) read_out_word(i, out_run1[i]);
    for (int k = 0; k < 4; k = k + 1) read_hfinal_word(k, hfin_run1[k]);

    ref_compute(64, 1'b0);
    for (int i = 0; i < 64; i = i + 1) begin
      cmp16(out_run1[i], y_exp[i], "y_run1_vs_export", i, 0);
      cmp16(out_run1[i], ref_y[i], "y_run1_vs_ref", i, 0);
    end
    for (int k = 0; k < 4; k = k + 1) begin
      cmp16(hfin_run1[k], h_exp[252+k], "hfinal_run1_vs_export", k, 1);
      cmp16(hfin_run1[k], ref_hst[252+k], "hfinal_run1_vs_ref", k, 1);
    end

    // CLEAR_STATUS: status/done/error/irq clear
    axil_write(12'h008, 32'h0000_0004);
    axil_read(12'h00C, tmp); cmp32(tmp, 32'd0, "STATUS_after_CLEAR_STATUS");
    axil_read(12'h010, tmp); cmp32(tmp, 32'd0, "ERROR_CODE_after_CLEAR_STATUS");
    if (irq !== 1'b0) begin errors = errors + 1; $display("CFC_TB_SEQ irq should clear on CLEAR_STATUS"); end

    // ------------------------------- 3) state-clear repeat run -> identical
    hw_run(64, 32'h0000_0001, ok, err, rc, sc, st);
    if (!ok || err) begin errors = errors + 1; $display("CFC_TB_SEQ repeat run not DONE cleanly"); end
    cmp32(sc, 32'd64, "repeat_STEP_COUNT");
    cmp32(rc, rc_run1, "repeat_RUN_CYCLES_equal");
    for (int i = 0; i < 64; i = i + 1) read_out_word(i, out_run2[i]);
    for (int i = 0; i < 64; i = i + 1) cmp16(out_run2[i], out_run1[i], "repeat_y", i, 2);

    // ------------------------------- 4) 32 + 32 segmented, state kept
    for (int i = 0; i < 32; i = i + 1)
      axil_write(12'h100 + 12'd4*i, {16'd0, x_vec[i][15:0]});
    hw_run(32, 32'h0000_0001, ok, err, rc, sc, st);   // clear before run
    if (!ok || err) begin errors = errors + 1; $display("CFC_TB_SEQ segment 1 not DONE cleanly"); end
    cmp32(sc, 32'd32, "seg1_STEP_COUNT");
    for (int i = 0; i < 32; i = i + 1) read_out_word(i, out_seg1[i]);

    for (int i = 0; i < 32; i = i + 1)
      axil_write(12'h100 + 12'd4*i, {16'd0, x_vec[32+i][15:0]});
    hw_run(32, 32'h0000_0000, ok, err, rc, sc, st);   // keep state
    if (!ok || err) begin errors = errors + 1; $display("CFC_TB_SEQ segment 2 not DONE cleanly"); end
    cmp32(sc, 32'd32, "seg2_STEP_COUNT");
    for (int i = 0; i < 32; i = i + 1) read_out_word(i, out_seg2[i]);
    for (int k = 0; k < 4; k = k + 1) read_hfinal_word(k, hfin_seg2[k]);

    for (int i = 0; i < 32; i = i + 1) begin
      cmp16(out_seg1[i], out_run1[i], "seg1_vs_cont", i, 3);
      cmp16(out_seg2[i], out_run1[32+i], "seg2_vs_cont", i, 3);
    end
    for (int k = 0; k < 4; k = k + 1) cmp16(hfin_seg2[k], hfin_run1[k], "seg_hfinal", k, 3);

    // ------------------------------- 5) MODE[1]=1: clear h_old every step
    for (int i = 0; i < 64; i = i + 1)
      axil_write(12'h100 + 12'd4*i, {16'd0, x_vec[i][15:0]});
    ref_compute(64, 1'b1);
    hw_run(64, 32'h0000_0003, ok, err, rc, sc, st);   // clear before run + clear each step
    if (!ok) begin errors = errors + 1; $display("CFC_TB_SEQ clear-each run did not report DONE"); end
    if (err) begin errors = errors + 1; $display("CFC_TB_SEQ clear-each run reported ERROR"); end
    cmp32(sc, 32'd64, "clear_each_STEP_COUNT");
    for (int i = 0; i < 64; i = i + 1) read_out_word(i, out_ce[i]);
    for (int k = 0; k < 4; k = k + 1) read_hfinal_word(k, hfin_ce[k]);
    for (int i = 0; i < 64; i = i + 1) cmp16(out_ce[i], ref_y_ce[i], "clear_each_y", i, 4);
    for (int k = 0; k < 4; k = k + 1) cmp16(hfin_ce[k], ref_h_ce[252+k], "clear_each_hfinal", k, 4);

    // ------------------------------------------------------------------ summary
    $display("CFC_TB_SEQ checks=%0d mismatches=%0d (y=%0d h=%0d repeat=%0d seg=%0d clear_each=%0d) errors=%0d",
             ncheck, mism, y_mism, h_mism, rep_mism, seg_mism, ce_mism, errors);
    if ((mism == 0) && (errors == 0)) begin
      $display("CFC_TB_PASS tb_cfc_seq mismatches=0 run_cycles=%0d", rc_run1);
      $finish;
    end else begin
      $display("CFC_TB_FAIL tb_cfc_seq mismatches=%0d errors=%0d", mism, errors);
      $fatal(1, "tb_cfc_seq failed");
    end
  end

  initial begin : watchdog
    #100_000_000;
    $display("CFC_TB_FAIL tb_cfc_seq watchdog timeout");
    $fatal(1, "tb_cfc_seq watchdog");
  end
endmodule