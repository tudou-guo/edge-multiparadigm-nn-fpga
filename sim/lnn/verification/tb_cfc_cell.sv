// tb_cfc_cell.sv -- single-step cell core vs. bit-exact integer reference.
// Contract 3.5: feed random / boundary (x, h, dt) vectors and compare y and all
// four new states bit-exactly against the embedded integer reference model
// (docs/CFC_CONTRACT.md V1.1 section 1.3).  Uses the same generated weight/LUT
// initialisation the RTL ROMs use, so the comparison is exact.
`timescale 1ns/1ps
module tb_cfc_cell;

  logic clk;
  initial clk = 1'b0;
  always #2.5 clk = ~clk;

  logic               rst;
  logic               start;
  logic               abort;
  logic signed [15:0] x_in;
  logic signed [15:0] h0_in, h1_in, h2_in, h3_in;
  logic signed [15:0] dt_in;
  logic signed [15:0] y_out;
  logic signed [15:0] hn0, hn1, hn2, hn3;
  logic               done;
  logic               busy;

  // reference parameter copies (same generated initialisation as the RTL ROMs)
  logic signed [15:0] w_rom       [0:83];
  logic signed [31:0] b_rom       [0:16];
  logic signed [15:0] tanh_rom    [0:1024];
  logic signed [15:0] sigmoid_rom [0:1024];
  `include "cfc_weight_init.svh"
  `include "cfc_lut_init.svh"
  `include "cfc_ref_model.svh"

  cfc_cell_core u_dut (
    .clk     (clk),
    .rst     (rst),
    .start   (start),
    .abort   (abort),
    .x_in    (x_in),
    .h_old_0 (h0_in),
    .h_old_1 (h1_in),
    .h_old_2 (h2_in),
    .h_old_3 (h3_in),
    .delta_t (dt_in),
    .y_out   (y_out),
    .h_new_0 (hn0),
    .h_new_1 (hn1),
    .h_new_2 (hn2),
    .h_new_3 (hn3),
    .done    (done),
    .busy    (busy)
  );

  int          errors = 0;
  int          mism   = 0;
  int          ncase  = 0;
  logic [31:0] rng    = 32'h1234_5679;
  int          cyc_tot = 0;

  function automatic logic signed [15:0] rnd16;
    begin
      rng = {rng[30:0], rng[31] ^ rng[21] ^ rng[1] ^ rng[0]};
      rnd16 = $signed(rng[15:0]);
    end
  endfunction

  task automatic run_one (
    input int                  cid,
    input logic signed [15:0]  xv,
    input logic signed [15:0]  h0v,
    input logic signed [15:0]  h1v,
    input logic signed [15:0]  h2v,
    input logic signed [15:0]  h3v,
    input logic signed [15:0]  dtv
  );
    logic signed [15:0] yr, r0, r1, r2, r3;
    int cyc;
    logic seen;
    begin
      x_in  = xv;
      h0_in = h0v;
      h1_in = h1v;
      h2_in = h2v;
      h3_in = h3v;
      dt_in = dtv;
      @(posedge clk); #1;
      start = 1'b1;
      @(posedge clk); #1;
      start = 1'b0;
      seen = 1'b0;
      cyc  = 0;
      while (!seen && (cyc < 4000)) begin
        @(posedge clk); #1;
        seen = done;
        cyc  = cyc + 1;
      end
      ref_step(xv, h0v, h1v, h2v, h3v, dtv, yr, r0, r1, r2, r3);
      ncase   = ncase + 1;
      cyc_tot = cyc_tot + cyc;
      if (!seen) begin
        errors = errors + 1;
        $display("CFC_TB_CELL case %0d : TIMEOUT (no done)", cid);
      end else begin
        if (y_out !== yr) begin
          mism = mism + 1;
          $display("CFC_TB_CELL case %0d : y mismatch dut=%0d ref=%0d", cid, y_out, yr);
        end
        if (hn0 !== r0) begin mism = mism + 1; $display("CFC_TB_CELL case %0d : h0 mismatch dut=%0d ref=%0d", cid, hn0, r0); end
        if (hn1 !== r1) begin mism = mism + 1; $display("CFC_TB_CELL case %0d : h1 mismatch dut=%0d ref=%0d", cid, hn1, r1); end
        if (hn2 !== r2) begin mism = mism + 1; $display("CFC_TB_CELL case %0d : h2 mismatch dut=%0d ref=%0d", cid, hn2, r2); end
        if (hn3 !== r3) begin mism = mism + 1; $display("CFC_TB_CELL case %0d : h3 mismatch dut=%0d ref=%0d", cid, hn3, r3); end
        if ((y_out === yr) && (hn0 === r0) && (hn1 === r1) && (hn2 === r2) && (hn3 === r3))
          $display("CFC_TB_CELL case %0d : ok  x=%0d h=(%0d,%0d,%0d,%0d) dt=%0d -> y=%0d h=(%0d,%0d,%0d,%0d) cycles=%0d",
                   cid, xv, h0v, h1v, h2v, h3v, dtv, y_out, hn0, hn1, hn2, hn3, cyc);
      end
      repeat (2) @(posedge clk);
    end
  endtask

  initial begin : stimulus
    int i;
    rst    = 1'b1;
    start  = 1'b0;
    abort  = 1'b0;
    x_in   = 16'sd0;
    h0_in  = 16'sd0; h1_in = 16'sd0; h2_in = 16'sd0; h3_in = 16'sd0;
    dt_in  = 16'sd655;
    repeat (8) @(posedge clk);
    #1 rst = 1'b0;
    repeat (4) @(posedge clk);

    // boundary cases
    run_one(0, 16'sd0,     16'sd0,     16'sd0,     16'sd0,     16'sd0,     16'sd655);
    run_one(1, 16'sh7FFF,  16'sh7FFF,  16'sh7FFF,  16'sh7FFF,  16'sh7FFF,  16'sd655);
    run_one(2, -16'sd32768,-16'sd32768,-16'sd32768,-16'sd32768,-16'sd32768,16'sd655);
    run_one(3, 16'sh7FFF, -16'sd32768, 16'sh7FFF, -16'sd32768, 16'sh7FFF, 16'sd655);
    run_one(4, 16'sd4096,  16'sd4096, -16'sd4096,  16'sd2048, -16'sd2048, 16'sd655);
    // extreme delta_t
    run_one(5, 16'sd1234,  16'sd100,  -16'sd200,   16'sd300,  -16'sd400,  16'sd1);
    run_one(6, -16'sd1234, -16'sd100,  16'sd200,  -16'sd300,   16'sd400,  16'sd32767);
    // random cases
    for (i = 0; i < 10; i = i + 1) begin
      run_one(7 + i, rnd16(), rnd16(), rnd16(), rnd16(), rnd16(), 16'sd655);
    end
    // random with random delta_t in legal range
    for (i = 0; i < 4; i = i + 1) begin
      run_one(17 + i, rnd16(), rnd16(), rnd16(), rnd16(), rnd16(),
              $signed({1'b0, rng[14:0]}) | 16'sd1);
    end

    if (mism == 0 && errors == 0) begin
      $display("CFC_TB_PASS tb_cfc_cell cases=%0d mismatches=%0d avg_cycles=%0d",
               ncase, mism, cyc_tot / ncase);
      $finish;
    end else begin
      $display("CFC_TB_FAIL tb_cfc_cell cases=%0d mismatches=%0d errors=%0d",
               ncase, mism, errors);
      $fatal(1, "tb_cfc_cell failed");
    end
  end

  initial begin : watchdog
    #20_000_000;
    $display("CFC_TB_FAIL tb_cfc_cell watchdog timeout");
    $fatal(1, "tb_cfc_cell watchdog");
  end
endmodule
