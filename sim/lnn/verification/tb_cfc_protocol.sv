// tb_cfc_protocol.sv -- AXI-Lite protocol + CSR behaviour matrix (contract 3.5).
//
// Covered:
//   * independent AW/W channels: same cycle, AW first, W first
//   * WSTRB byte writes; input window stores low 16 bits only
//   * B channel delay (bvalid held without bready), read back-pressure,
//     read latency 1..2 cycles, bresp/rresp = OKAY
//   * busy repeated START -> ERROR_CODE=2, run not restarted, config/input
//     writes while busy discarded
//   * illegal N_STEPS / DELTA_T -> ERROR_CODE=1 and no run
//   * run timeout -> ERROR_CODE=3, BUSY=0, DONE=0
//   * CLEAR_STATUS / CLEAR_STATE semantics, sticky done/error + irq
//   * unimplemented offsets read 0, writes ignored, CONTROL W1P reads 0
`timescale 1ns/1ps
module tb_cfc_protocol;

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

  logic signed [15:0] w_rom       [0:83];
  logic signed [31:0] b_rom       [0:16];
  logic signed [15:0] tanh_rom    [0:1024];
  logic signed [15:0] sigmoid_rom [0:1024];
  `include "cfc_weight_init.svh"
  `include "cfc_lut_init.svh"

  logic signed [15:0] x_vec [0:63];
  logic signed [15:0] y_exp [0:63];
  logic signed [15:0] h_exp [0:255];

  int ncheck = 0;
  int mism   = 0;
  int errs   = 0;

  // ---------------------------------------------------------------- helpers
  task automatic chk32 (input logic [31:0] got, input logic [31:0] exp, input string tag);
    begin
      ncheck = ncheck + 1;
      if (got !== exp) begin
        mism = mism + 1;
        if (mism <= 30) $display("CFC_TB_PROTO %s mismatch : got=0x%08h exp=0x%08h", tag, got, exp);
      end
    end
  endtask

  task automatic note_err (input string tag);
    begin
      errs = errs + 1;
      $display("CFC_TB_PROTO ERROR: %s", tag);
    end
  endtask

  // one AXI-Lite write; order: 0 = AW+W same cycle, 1 = AW first, 2 = W first
  task automatic axi_wr_ord (input logic [11:0] addr, input logic [31:0] data,
                             input logic [3:0] strb, input int ord, input int bdelay);
    int d;
    begin
      case (ord)
        0: begin
          @(posedge clk);
          s_awaddr  <= addr;
          s_wdata   <= data;
          s_wstrb   <= strb;
          s_awvalid <= 1'b1;
          s_wvalid  <= 1'b1;
          @(posedge clk);
          s_awvalid <= 1'b0;
          s_wvalid  <= 1'b0;
        end
        1: begin
          @(posedge clk);
          s_awaddr  <= addr;
          s_awvalid <= 1'b1;
          @(posedge clk);
          s_awvalid <= 1'b0;
          repeat (3) @(posedge clk);
          s_wdata  <= data;
          s_wstrb  <= strb;
          s_wvalid <= 1'b1;
          @(posedge clk);
          s_wvalid <= 1'b0;
        end
        default: begin
          @(posedge clk);
          s_wdata  <= data;
          s_wstrb  <= strb;
          s_wvalid <= 1'b1;
          @(posedge clk);
          s_wvalid <= 1'b0;
          repeat (3) @(posedge clk);
          s_awaddr  <= addr;
          s_awvalid <= 1'b1;
          @(posedge clk);
          s_awvalid <= 1'b0;
        end
      endcase
      while (!s_bvalid) @(posedge clk);
      if (s_bresp !== 2'b00) note_err("BRESP is not OKAY");
      for (d = 0; d < bdelay; d = d + 1) begin
        @(posedge clk);
        if (!s_bvalid) note_err("BVALID dropped before BREADY");
      end
      s_bready <= 1'b1;
      @(posedge clk);
      s_bready <= 1'b0;
    end
  endtask

  task automatic wr (input logic [11:0] addr, input logic [31:0] data);
    begin
      axi_wr_ord(addr, data, 4'hF, 0, 0);
    end
  endtask

  task automatic axi_rd_bp (input logic [11:0] addr, output logic [31:0] data, input int rdelay);
    logic [31:0] d0;
    int lat, k;
    begin
      @(posedge clk);
      s_araddr  <= addr;
      s_arvalid <= 1'b1;
      @(posedge clk);
      s_arvalid <= 1'b0;
      lat = 0;
      while ((s_rvalid !== 1'b1) && (lat < 20)) begin
        @(posedge clk);
        lat = lat + 1;
      end
      if (s_rvalid !== 1'b1) begin
        note_err("read timeout (no RVALID)");
        data = 32'hdead_beef;
      end else begin
        if ((lat < 1) || (lat > 2)) begin
          $display("CFC_TB_PROTO NOTE: read latency %0d cycles (contract allows 1..2)", lat);
          note_err("read latency out of 1..2");
        end
        if (s_rresp !== 2'b00) note_err("RRESP is not OKAY");
        d0 = s_rdata;
        for (k = 0; k < rdelay; k = k + 1) begin
          @(posedge clk);
          if ((s_rvalid !== 1'b1) || (s_rdata !== d0)) note_err("RDATA not stable under RREADY back-pressure");
        end
        data = s_rdata;
        s_rready <= 1'b1;
        @(posedge clk);
        s_rready <= 1'b0;
      end
    end
  endtask

  task automatic rd (input logic [11:0] addr, output logic [31:0] data);
    begin
      axi_rd_bp(addr, data, 0);
    end
  endtask

  task automatic chk_status (input logic eb, input logic ed, input logic ee, input string tag);
    logic [31:0] st;
    begin
      rd(12'h00C, st);
      ncheck = ncheck + 1;
      if ((st[0] !== eb) || (st[1] !== ed) || (st[2] !== ee)) begin
        mism = mism + 1;
        $display("CFC_TB_PROTO %s mismatch : STATUS=0x%08h exp busy=%0b done=%0b err=%0b (raw 0x%08h)",
                 tag, st, eb, ed, ee, st);
      end
    end
  endtask

  task automatic chk_ec (input logic [1:0] exp, input string tag);
    logic [31:0] ec;
    begin
      rd(12'h010, ec);
      chk32(ec, {30'd0, exp}, tag);
    end
  endtask

  task automatic clear_status;
    begin wr(12'h008, 32'h0000_0004); end
  endtask
  task automatic clear_state;
    begin wr(12'h008, 32'h0000_0002); end
  endtask
  task automatic start_it;
    begin wr(12'h008, 32'h0000_0001); end
  endtask
  task automatic cfg_n (input logic [31:0] n);  begin wr(12'h014, n); end  endtask
  task automatic cfg_dt(input logic [31:0] d);  begin wr(12'h018, d); end  endtask
  task automatic cfg_m (input logic [31:0] m);  begin wr(12'h01C, m); end  endtask
  task automatic cfg_t (input logic [31:0] t);  begin wr(12'h028, t); end  endtask

  task automatic wait_busy (input int maxp, output logic ok);
    logic [31:0] st;
    int p;
    begin
      ok = 1'b0;
      p  = 0;
      while ((p < maxp) && !ok) begin
        rd(12'h00C, st);
        ok = st[0];
        p  = p + 1;
      end
    end
  endtask

  // wait until the run has finished (BUSY drops; DONE/ERROR are sticky so they
  // cannot be used as an early exit, because a busy-start error sets ERROR
  // while the run keeps going)
  task automatic poll_end (input int maxp, output logic busy, output logic done, output logic err);
    logic [31:0] st;
    int p;
    begin
      busy = 1'b1;
      done = 1'b0;
      err  = 1'b0;
      p    = 0;
      while (p < maxp) begin
        rd(12'h00C, st);
        busy = st[0];
        done = st[1];
        err  = st[2];
        if (!busy) p = maxp;
        else p = p + 1;
      end
    end
  endtask

  task automatic chk_outputs (input int n, input string tag);
    logic [31:0] w;
    logic signed [15:0] v;
    begin
      for (int i = 0; i < n; i = i + 1) begin
        rd(12'h200 + 12'd4*i, w);
        v = $signed(w[15:0]);
        ncheck = ncheck + 1;
        if (v !== y_exp[i]) begin
          mism = mism + 1;
          if (mism <= 30) $display("CFC_TB_PROTO %s[%0d] mismatch : got=%0d exp=%0d", tag, i, v, y_exp[i]);
        end
      end
    end
  endtask

  task automatic chk_hfinal (input int tstep, input string tag);
    logic [31:0] w;
    logic signed [15:0] v;
    begin
      for (int k = 0; k < 4; k = k + 1) begin
        rd(12'h080 + 12'd4*k, w);
        v = $signed(w[15:0]);
        ncheck = ncheck + 1;
        if (v !== h_exp[4*tstep+k]) begin
          mism = mism + 1;
          if (mism <= 30) $display("CFC_TB_PROTO %s h[%0d] mismatch : got=%0d exp=%0d", tag, k, v, h_exp[4*tstep+k]);
        end
      end
    end
  endtask

  task automatic load_inputs;
    begin
      for (int i = 0; i < 64; i = i + 1)
        wr(12'h100 + 12'd4*i, {16'd0, x_vec[i][15:0]});
    end
  endtask

  // ------------------------------------------------------------------ stimulus
  logic [31:0] tmp, st, rc, sc;
  int fh;
  logic b_ok, b_busy, b_done, b_err;
  logic [31:0] ec;

  initial begin : stimulus
    rst = 1'b1;
    s_awaddr = 12'd0; s_awvalid = 1'b0;
    s_wdata = 32'd0; s_wstrb = 4'd0; s_wvalid = 1'b0;
    s_bready = 1'b0;
    s_araddr = 12'd0; s_arvalid = 1'b0; s_rready = 1'b0;

    fh = $fopen("../export/gen/x_input.hex", "r");
    if (fh == 0) begin
      $display("CFC_TB_FAIL tb_cfc_protocol : missing ../export/gen/x_input.hex");
      $fatal(1, "missing vector file");
    end
    $fclose(fh);
    $readmemh("../export/gen/x_input.hex", x_vec);
    $readmemh("../export/gen/y_expected.hex", y_exp);
    $readmemh("../export/gen/h_expected.hex", h_exp);

    repeat (8) @(posedge clk);
    #1 rst = 1'b0;
    repeat (4) @(posedge clk);

    // ============================ A) reset identity / defaults ============
    $display("CFC_TB_PROTO section A: identity / defaults");
    rd(12'h000, tmp); chk32(tmp, 32'h43464331, "ID");
    rd(12'h004, tmp); chk32(tmp, 32'h00010000, "VERSION");
    rd(12'h02C, tmp); chk32(tmp, 32'd300000000, "CLOCK_HZ");
    chk_status(1'b0, 1'b0, 1'b0, "STATUS_reset");
    chk_ec(2'd0, "ERROR_CODE_reset");
    rd(12'h014, tmp); chk32(tmp, 32'd64, "N_STEPS_default");
    rd(12'h018, tmp); chk32(tmp, 32'd655, "DELTA_T_default");
    rd(12'h01C, tmp); chk32(tmp, 32'd1, "MODE_default");
    rd(12'h028, tmp); chk32(tmp, 32'd1000000, "TIMEOUT_default");
    rd(12'h008, tmp); chk32(tmp, 32'd0, "CONTROL_reads_0");
    // W1P: writing 0 to CONTROL must not start anything
    wr(12'h008, 32'd0); chk_status(1'b0, 1'b0, 1'b0, "CONTROL_W1P_no_start");
    // unimplemented offsets read 0, writes ignored
    rd(12'h040, tmp); chk32(tmp, 32'd0, "unimpl_0x040_read0");
    wr(12'h040, 32'hDEAD_BEEF);
    rd(12'h040, tmp); chk32(tmp, 32'd0, "unimpl_0x040_ignored");
    rd(12'h0F0, tmp); chk32(tmp, 32'd0, "unimpl_0x0F0_read0");

    // ============================ B) AW/W order + WSTRB ===================
    $display("CFC_TB_PROTO section B: AW/W ordering + WSTRB");
    axi_wr_ord(12'h014, 32'd32, 4'hF, 0, 0);        // same cycle
    rd(12'h014, tmp); chk32(tmp, 32'd32, "same_cycle_write_N_STEPS");
    axi_wr_ord(12'h018, 32'd1000, 4'hF, 1, 0);      // AW first
    rd(12'h018, tmp); chk32(tmp, 32'd1000, "aw_first_write_DELTA_T");
    axi_wr_ord(12'h028, 32'd5000, 4'hF, 2, 0);      // W first
    rd(12'h028, tmp); chk32(tmp, 32'd5000, "w_first_write_TIMEOUT");
    axi_wr_ord(12'h01C, 32'd0, 4'h3, 0, 0);         // MODE = 0, lower 2 bytes
    rd(12'h01C, tmp); chk32(tmp, 32'd0, "mode_write0");
    wr(12'h01C, 32'd1);
    // WSTRB: only byte 1 written -> 0x0040 | 0x2000 = 0x2040
    axi_wr_ord(12'h014, 32'h0000_2040, 4'b0010, 0, 0);
    rd(12'h014, tmp); chk32(tmp, 32'h0000_2020, "wstrb_byte1_N_STEPS");
    axi_wr_ord(12'h014, 32'd64, 4'hF, 0, 0);        // restore
    // DELTA_T partial: 0xAB05 via byte0 then byte1
    axi_wr_ord(12'h018, 32'h0000_AB05, 4'hF, 0, 0);
    rd(12'h018, tmp); chk32(tmp, 32'h0000_AB05, "delta_t_partial_1");
    axi_wr_ord(12'h018, 32'h0000_1205, 4'b0010, 0, 0);   // byte1 = 0x12 -> 0x1205
    rd(12'h018, tmp); chk32(tmp, 32'h0000_1205, "delta_t_partial_2");
    axi_wr_ord(12'h018, 32'd655, 4'hF, 0, 0);       // restore
    // INPUT window: high 16 bits ignored, stored low16, read back high 16 = 0
    wr(12'h100, 32'hABCD_1234);
    rd(12'h100, tmp); chk32(tmp, 32'h0000_1234, "input_window_low16");
    axi_wr_ord(12'h100, 32'hFFFF_5678, 4'b1100, 0, 0);  // only upper bytes -> ignored
    rd(12'h100, tmp); chk32(tmp, 32'h0000_1234, "input_window_upper_strb_ignored");
    // restore N_STEPS / DELTA_T / TIMEOUT / MODE defaults
    wr(12'h014, 32'd64);
    wr(12'h018, 32'd655);
    wr(12'h028, 32'd1000000);
    wr(12'h01C, 32'd1);

    // ============================ C) B-channel delay ======================
    $display("CFC_TB_PROTO section C: B channel delay");
    axi_wr_ord(12'h014, 32'd48, 4'hF, 0, 6);        // hold BREADY low 6 cycles
    rd(12'h014, tmp); chk32(tmp, 32'd48, "b_delay_write_N_STEPS");
    wr(12'h014, 32'd64);

    // ============================ D) read back-pressure ===================
    $display("CFC_TB_PROTO section D: read back-pressure / latency");
    axi_rd_bp(12'h000, tmp, 6);
    chk32(tmp, 32'h43464331, "read_bp_ID");
    axi_rd_bp(12'h00C, tmp, 3);
    chk32(tmp, 32'd0, "read_bp_STATUS");

    // ============================ E) busy repeated START ==================
    $display("CFC_TB_PROTO section E: busy repeated START -> error 2");
    wr(12'h01C, 32'd1);            // MODE clear before run
    wr(12'h014, 32'd64);
    wr(12'h018, 32'd655);
    wr(12'h028, 32'd1000000);
    load_inputs();
    start_it();
    wait_busy(200, b_ok);
    if (!b_ok) note_err("BUSY never asserted after START");
    // repeated START while busy: error 2, no restart
    start_it();
    rd(12'h00C, st);
    ncheck = ncheck + 1;
    if (!(st[0] && st[2])) begin mism = mism + 1; $display("CFC_TB_PROTO busy_start STATUS=0x%08h (exp busy+err)", st); end
    chk_ec(2'd2, "busy_start_error_code");
    // config / input writes while busy are discarded
    cfg_n(16);
    rd(12'h014, tmp); chk32(tmp, 32'd64, "busy_config_write_discarded");
    wr(12'h100, 32'h0000_5555);     // input write while busy -> discarded
    clear_state();                  // CLEAR_STATE while busy -> ignored
    poll_end(40000, b_busy, b_done, b_err);
    if (!(b_done && !b_busy && b_err)) begin
      note_err("busy-start run did not finish as expected (expected done=1 busy=0 err=1)");
    end
    chk_status(1'b0, 1'b1, 1'b1, "busy_start_final_status");
    rd(12'h024, sc); chk32(sc, 32'd64, "busy_start_STEP_COUNT");
    chk_ec(2'd2, "busy_start_error_code_sticky");
    chk_outputs(64, "busy_start_y_vs_expected");
    chk_hfinal(63, "busy_start_hfinal");
    if (irq !== 1'b1) note_err("irq not set after done");
    clear_status();
    chk_status(1'b0, 1'b0, 1'b0, "STATUS_after_clear_status");
    chk_ec(2'd0, "ERROR_CODE_after_clear_status");
    if (irq !== 1'b0) note_err("irq not cleared by CLEAR_STATUS");

    // ============================ F) illegal configuration -> error 1 =====
    $display("CFC_TB_PROTO section F: illegal configuration -> error 1");
    wr(12'h018, 32'd655);
    wr(12'h028, 32'd1000000);
    wr(12'h01C, 32'd1);
    // N_STEPS = 0
    wr(12'h014, 32'd0);  start_it(); repeat (6) @(posedge clk);
    chk_status(1'b0, 1'b0, 1'b1, "cfg_err_n0_status");
    chk_ec(2'd1, "cfg_err_n0_code");
    clear_status();
    chk_status(1'b0, 1'b0, 1'b0, "cfg_err_n0_cleared");
    // N_STEPS = 65
    wr(12'h014, 32'd65); start_it(); repeat (6) @(posedge clk);
    chk_status(1'b0, 1'b0, 1'b1, "cfg_err_n65_status");
    chk_ec(2'd1, "cfg_err_n65_code");
    clear_status();
    // DELTA_T = 0
    wr(12'h014, 32'd64); wr(12'h018, 32'd0); start_it(); repeat (6) @(posedge clk);
    chk_status(1'b0, 1'b0, 1'b1, "cfg_err_dt0_status");
    chk_ec(2'd1, "cfg_err_dt0_code");
    clear_status();
    // DELTA_T = 32768
    wr(12'h018, 32'd32768); start_it(); repeat (6) @(posedge clk);
    chk_status(1'b0, 1'b0, 1'b1, "cfg_err_dt32768_status");
    chk_ec(2'd1, "cfg_err_dt32768_code");
    clear_status();
    chk_status(1'b0, 1'b0, 1'b0, "cfg_err_final_cleared");

    // ============================ G) run timeout -> error 3 ===============
    $display("CFC_TB_PROTO section G: run timeout -> error 3");
    wr(12'h014, 32'd64);
    wr(12'h018, 32'd655);
    wr(12'h01C, 32'd1);
    wr(12'h028, 32'd200);
    start_it();
    poll_end(40000, b_busy, b_done, b_err);
    if (!(b_err && !b_busy && !b_done)) begin
      note_err("timeout run status wrong (exp err=1 busy=0 done=0)");
    end
    chk_ec(2'd3, "timeout_error_code");
    rd(12'h024, tmp);
    ncheck = ncheck + 1;
    if (tmp >= 32'd64) begin mism = mism + 1; $display("CFC_TB_PROTO timeout STEP_COUNT=%0d (exp <64)", tmp); end
    clear_status();
    wr(12'h028, 32'd1000000);       // restore timeout

    // ============================ H) CLEAR_STATE + recovery ===============
    $display("CFC_TB_PROTO section H: CLEAR_STATE / recovery / repeat");
    // recovery run after timeout: 8 steps
    wr(12'h014, 32'd8);
    wr(12'h01C, 32'd1);
    start_it();
    poll_end(40000, b_busy, b_done, b_err);
    if (!(b_done && !b_busy && !b_err)) note_err("recovery 8-step run failed");
    rd(12'h024, sc); chk32(sc, 32'd8, "recovery_STEP_COUNT");
    chk_outputs(8, "recovery_y_vs_expected");
    chk_hfinal(7, "recovery_hfinal");
    if (irq !== 1'b1) note_err("irq not set after recovery run");
    // idle CLEAR_STATE clears the state registers
    clear_status();
    clear_state();
    repeat (2) @(posedge clk);
    for (int k = 0; k < 4; k = k + 1) begin
      rd(12'h080 + 12'd4*k, tmp);
      chk32(tmp, 32'd0, "hfinal_after_clear_state");
    end
    // run again from the cleared state (MODE=0: no pre-clear, state already 0)
    wr(12'h014, 32'd8);
    wr(12'h01C, 32'd0);
    start_it();
    poll_end(40000, b_busy, b_done, b_err);
    if (!(b_done && !b_busy && !b_err)) note_err("MODE=0 run from cleared state failed");
    chk_outputs(8, "cleared_repeat_y_vs_expected");
    chk_hfinal(7, "cleared_repeat_hfinal");
    // CLEAR_STATE while busy must be ignored
    clear_status();
    clear_state();
    wr(12'h014, 32'd8);
    wr(12'h01C, 32'd0);
    start_it();
    wait_busy(200, b_ok);
    if (!b_ok) note_err("BUSY never asserted in clear-state-while-busy test");
    clear_state();                  // ignored while busy
    poll_end(40000, b_busy, b_done, b_err);
    if (!(b_done && !b_busy && !b_err)) note_err("clear-state-while-busy run failed");
    chk_outputs(8, "clear_state_busy_ignored_y");
    chk_hfinal(7, "clear_state_busy_ignored_hfinal");

    // ------------------------------------------------------------- summary
    $display("CFC_TB_PROTO checks=%0d mismatches=%0d errors=%0d", ncheck, mism, errs);
    if ((mism == 0) && (errs == 0)) begin
      $display("CFC_TB_PASS tb_cfc_protocol checks=%0d", ncheck);
      $finish;
    end else begin
      $display("CFC_TB_FAIL tb_cfc_protocol mismatches=%0d errors=%0d", mism, errs);
      $fatal(1, "tb_cfc_protocol failed");
    end
  end

  initial begin : watchdog
    #100_000_000;
    $display("CFC_TB_FAIL tb_cfc_protocol watchdog timeout");
    $fatal(1, "tb_cfc_protocol watchdog");
  end
endmodule