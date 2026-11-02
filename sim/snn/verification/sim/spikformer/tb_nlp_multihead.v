// tb_nlp_multihead - multi-head short replay (M3 gate), block-level.
//
// Case: E64 / H2 / HD32 / seq4 / T2 / MLP64 / depth1, ONE spiking_block_top
// instance driven directly (contract block_ports_exact ABI). Two runs with
// head-swapped input spike patterns (head0/head1 activity exchanged) so any
// head swap or cross-head leakage changes the output; block output backpressure
// windows + input bubbles are driven deterministically. Consecutive runs without
// a global reset also check start/done cleanup.
//
// Golden: build/spikformer/nlp_48h_20261029_1915/replay/case_multihead/
//         (gen_nlp_multihead.py; generator asserts per-head Q/K/V/score patterns
//          are distinct; full traces cross-checked against integer_ref.py)
// Driver: scripts/spikformer/run_nlp_sim.py --case multihead
`timescale 1ns/1ps
module tb_nlp_multihead;
  // ---- frozen case parameters (must match gen_nlp_common.CASE_CFG["multihead"]) ----
  localparam int E = 64;
  localparam int HEADS = 2;
  localparam int HD = 32;
  localparam int HIDDEN = 64;
  localparam int T = 2;
  localparam int NTOK = 4;
  localparam int TL = 8;
  localparam int TC = 8;
  localparam int RW = 4;
  localparam int PSUM_SLICE = 32;
  localparam int SUM_W = 6;
  localparam int W = 8;
  localparam int NRUN = 2;                       // two consecutive block runs, no reset
  localparam int G = E / TL;
  localparam int NB = NTOK * G;                  // block in/out beats per run
  localparam int IW = TL * T * RW;               // 64-bit block beats at this size
  localparam int NW = 4 * E * E / 8 + 4 * E / 2 + HIDDEN * E / 8 + HIDDEN / 2
                     + E * HIDDEN / 8 + E / 2;   // weight beats per block run
  localparam longint WATCHDOG = 200000;

  reg s_clk = 0;
  reg s_rst = 1;
  reg i_start = 0;
  wire o_done;
  reg i_in_valid = 0;
  wire o_in_ready;
  reg [IW-1:0] i_in_data = 0;
  wire o_out_valid;
  reg i_out_ready = 0;
  wire [IW-1:0] o_out_data;
  wire o_out_last;
  reg [63:0] i_w_data = 0;
  reg i_w_valid = 0;
  wire o_w_ready;
  reg signed [PSUM_SLICE-1:0] i_thr_q = 0;
  reg signed [PSUM_SLICE-1:0] i_thr_k = 0;
  reg signed [PSUM_SLICE-1:0] i_thr_v = 0;
  reg signed [PSUM_SLICE-1:0] i_thr_proj = 0;
  reg signed [PSUM_SLICE-1:0] i_thr_fc1 = 0;
  reg signed [PSUM_SLICE-1:0] i_thr_fc2 = 0;

  always #5 s_clk = ~s_clk;

  spiking_block_top #(
    .E(E), .HEADS(HEADS), .HEAD_DIM(HD), .HIDDEN(HIDDEN), .T(T), .TOKENS(NTOK),
    .TL(TL), .TC(TC), .RW(RW), .W(W), .PSUM_SLICE(PSUM_SLICE), .SUM_W(SUM_W)
  ) dut (
    .s_clk(s_clk), .s_rst(s_rst), .i_start(i_start), .o_done(o_done),
    .i_in_valid(i_in_valid), .o_in_ready(o_in_ready), .i_in_data(i_in_data),
    .o_out_valid(o_out_valid), .i_out_ready(i_out_ready),
    .o_out_data(o_out_data), .o_out_last(o_out_last),
    .i_w_data(i_w_data), .i_w_valid(i_w_valid), .o_w_ready(o_w_ready),
    .i_thr_q(i_thr_q), .i_thr_k(i_thr_k), .i_thr_v(i_thr_v),
    .i_thr_proj(i_thr_proj), .i_thr_fc1(i_thr_fc1), .i_thr_fc2(i_thr_fc2)
  );

  // ---- expected vectors (+VECDIR) ----
  reg [63:0] desc [0:3];
  reg [63:0] wstream [0:NW-1];
  reg [IW-1:0] exp_in [0:NRUN*NB-1];
  reg [IW-1:0] exp_out [0:NRUN*NB-1];

  // ---- state ----
  integer cur_run = 0;
  integer in_cnt = 0;
  integer out_cnt = 0;
  integer cyc = 0;
  integer mm_cnt = 0;
  bit armed [0:NRUN-1];
  bit done_cap [0:NRUN-1];
  bit failed = 0;
  bit allow_mm = 0;
  bit trace_on = 0;
  bit done_prev = 0;
  integer i, r;
  string vecdir;
  reg [7:0] rdy_cnt = 0;

  // deterministic output backpressure: 2 cycles low / 4 cycles high windows
  always @(posedge s_clk) rdy_cnt <= rdy_cnt + 1'b1;
  always_comb i_out_ready = ((rdy_cnt % 6) >= 2);

  task automatic tb_fail(input string msg);
    begin
      failed = 1;
      $display("NLP_FAIL case=multihead cycle=%0d run=%0d %s", cyc, cur_run, msg);
      $display("  state: in_cnt=%0d/%0d out_cnt=%0d/%0d done=%b w_rdy=%b in_rdy=%b out_valid=%b",
               in_cnt, NRUN*NB, out_cnt, NRUN*NB, o_done, o_w_ready, o_in_ready, o_out_valid);
      $display("NLP_RESULT case=multihead pass=0 cycles=%0d", cyc);
      $finish;
    end
  endtask

  // ---- first-mismatch context helper (block ports only, no internals) ----
  function automatic string lane_vals(input [IW-1:0] d);
    string r;
    integer l2, t2;
    begin
      r = "";
      for (l2 = 0; l2 < TL; l2 = l2 + 1) begin
        if (l2 != 0) r = {r, "/"};
        for (t2 = 0; t2 < T; t2 = t2 + 1)
          r = {r, $sformatf("%0d", d[(l2*T+t2+1)*RW-1 -: RW])};
      end
      lane_vals = r;
    end
  endfunction

  // ---- watchdog ----
  initial begin
    forever begin
      @(posedge s_clk);
      if (!s_rst) begin
        cyc = cyc + 1;
        if (cyc > WATCHDOG) begin
          $display("NLP_FAIL case=multihead WATCHDOG timeout after %0d cycles", cyc);
          $display("  state: run=%0d in_cnt=%0d out_cnt=%0d done=%b w_rdy=%b in_rdy=%b out_valid=%b out_ready=%b",
                   cur_run, in_cnt, out_cnt, o_done, o_w_ready, o_in_ready,
                   o_out_valid, i_out_ready);
          $display("NLP_RESULT case=multihead pass=0 cycles=%0d", cyc);
          $finish;
        end
      end
    end
  end

  // ---- beat monitor: sample at posedge (pre-edge values determine the handshake) ----
  always @(posedge s_clk) begin
    if (!s_rst && !failed) begin
      if (i_in_valid === 1'b1 && o_in_ready === 1'b1) begin
        if (in_cnt >= NRUN*NB)
          tb_fail("extra block input beat");
        else if (i_in_data !== exp_in[in_cnt])
          tb_fail($sformatf("block-in beat %0d got=%h[%s] want=%h[%s]", in_cnt, i_in_data, lane_vals(i_in_data), exp_in[in_cnt], lane_vals(exp_in[in_cnt])));
        else in_cnt = in_cnt + 1;
      end
      if (o_out_valid === 1'b1 && i_out_ready === 1'b1) begin
        if (out_cnt >= NRUN*NB)
          tb_fail("extra block output beat");
        else if (o_out_data !== exp_out[out_cnt]) begin
          if (allow_mm) begin
            mm_cnt = mm_cnt + 1;
            $display("NLP_MM block-out beat=%0d cycle=%0d got=%h[%s] want=%h[%s]", out_cnt, cyc, o_out_data, lane_vals(o_out_data), exp_out[out_cnt], lane_vals(exp_out[out_cnt]));
            out_cnt = out_cnt + 1;
          end else
            tb_fail($sformatf("block-out beat %0d got=%h[%s] want=%h[%s]", out_cnt, o_out_data, lane_vals(o_out_data), exp_out[out_cnt], lane_vals(exp_out[out_cnt])));
        end else if (o_out_last !== (((out_cnt % NB) == NB-1) ? 1'b1 : 1'b0))
          tb_fail($sformatf("block-out beat %0d o_out_last=%b", out_cnt, o_out_last));
        else out_cnt = out_cnt + 1;
      end
      if (cur_run < NRUN && armed[cur_run]) begin
        if (o_done === 1'b1 && !done_cap[cur_run]) begin
          done_cap[cur_run] = 1;
          if (out_cnt != (cur_run+1)*NB)
            tb_fail($sformatf("o_done before all output beats (out_cnt=%0d)", out_cnt));
        end
      end
      if (trace_on && o_done === 1'b1 && !done_prev && (out_cnt % NB) == 0)
        $display("NLP_TR block_done run=%0d cycle=%0d out_beats=%0d",
                 (out_cnt / NB) - 1, cyc, out_cnt);
      done_prev = (o_done === 1'b1);
    end
  end

  // ---- protocol drivers ----
  task automatic send_w(input [63:0] d, input integer idx);
    begin
      @(negedge s_clk);
      i_w_data = d; i_w_valid = 1;
      while (o_w_ready !== 1'b1) @(negedge s_clk);
      @(posedge s_clk);
      @(negedge s_clk);
      i_w_valid = 0;
      if ((idx % 5) == 2) @(negedge s_clk);
      if ((idx % 11) == 7) begin @(negedge s_clk); @(negedge s_clk); end
    end
  endtask

  task automatic send_in(input [IW-1:0] d, input integer idx);
    begin
      @(negedge s_clk);
      i_in_data = d; i_in_valid = 1;
      while (o_in_ready !== 1'b1) @(negedge s_clk);
      @(posedge s_clk);
      @(negedge s_clk);
      i_in_valid = 0;
      if ((idx % 3) == 1) @(negedge s_clk);
      if ((idx % 7) == 5) begin @(negedge s_clk); end
    end
  endtask

  task automatic pulse_start();
    begin
      @(negedge s_clk);
      i_start = 1;
      @(posedge s_clk);
      @(negedge s_clk);
      i_start = 0;
    end
  endtask

  // ---- main ----
  initial begin
    allow_mm = $test$plusargs("NLP_ALLMISMATCH");
    trace_on = $test$plusargs("NLP_TRACE");
    if (!$value$plusargs("VECDIR=%s", vecdir)) begin
      $display("NLP_FAIL case=multihead missing +VECDIR plusarg");
      $display("NLP_RESULT case=multihead pass=0 cycles=0");
      $finish;
    end
    $readmemh($sformatf("%s/desc_b0.hex", vecdir), desc, 0, 3);
    $readmemh($sformatf("%s/stream_b0.hex", vecdir), wstream, 0, NW-1);
    for (r = 0; r < NRUN; r = r + 1) begin
      $readmemh($sformatf("%s/in_b%0d.hex", vecdir, r), exp_in, r*NB, (r+1)*NB-1);
      $readmemh($sformatf("%s/exp_out_b%0d.hex", vecdir, r), exp_out, r*NB, (r+1)*NB-1);
    end
    // thresholds come from the frozen 4-beat descriptor (Verilog concat order)
    i_thr_q = $signed(desc[0][63:32]);
    i_thr_k = $signed(desc[0][31:0]);
    i_thr_v = $signed(desc[1][63:32]);
    i_thr_proj = $signed(desc[1][31:0]);
    i_thr_fc1 = $signed(desc[2][63:32]);
    i_thr_fc2 = $signed(desc[2][31:0]);
    $display("NLP_TB case=multihead E=%0d H=%0d HD=%0d T=%0d N=%0d MLP=%0d runs=%0d",
             E, HEADS, HD, T, NTOK, HIDDEN, NRUN);
    $display("NLP_TB thresholds q=%0d k=%0d v=%0d proj=%0d fc1=%0d fc2=%0d thr_e=%0d",
             i_thr_q, i_thr_k, i_thr_v, i_thr_proj, i_thr_fc1, i_thr_fc2,
             $signed(desc[3][31:0]));
    $display("NLP_TB golden dir=%s watchdog=%0d", vecdir, WATCHDOG);
    repeat (4) @(posedge s_clk);
    s_rst = 0;
    repeat (2) @(posedge s_clk);
    for (r = 0; r < NRUN; r = r + 1) begin
      cur_run = r;
      armed[r] = 0;
      done_cap[r] = 0;
      in_cnt = r*NB;
      out_cnt = r*NB;
      pulse_start();
      repeat (2) @(negedge s_clk);
      if (o_done !== 1'b0)
        tb_fail($sformatf("run %0d: o_done not cleared by i_start", r));
      armed[r] = 1;
      $display("NLP_RUN %0d start cycle=%0d", r, cyc);
      for (i = 0; i < NW; i = i + 1) send_w(wstream[i], i);
      for (i = 0; i < NB; i = i + 1) send_in(exp_in[r*NB + i], i);
      wait (done_cap[r]);
      if (in_cnt != (r+1)*NB)
        tb_fail($sformatf("run %0d block-in count %0d != %0d", r, in_cnt, (r+1)*NB));
      if (out_cnt != (r+1)*NB)
        tb_fail($sformatf("run %0d block-out count %0d != %0d", r, out_cnt, (r+1)*NB));
      $display("NLP_RUN %0d done cycle=%0d out_beats=%0d", r, cyc, out_cnt);
    end
    if (!failed) begin
      $display("NLP_PASS case=multihead runs=%0d beats_per_run=%0d mismatches=%0d",
               NRUN, NB, mm_cnt);
      $display("NLP_RESULT case=multihead pass=1 cycles=%0d", cyc);
    end
    $finish;
  end
endmodule