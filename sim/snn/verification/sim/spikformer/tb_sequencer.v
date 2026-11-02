`timescale 1ns/1ps
module tb_sequencer;
  reg s_clk = 0, s_rst = 1;
  reg start = 0;
  wire done, err;
  wire [1:0] bidx;
  wire bfirst, blast;
  wire [31:0] wbase;
  reg [31:0] stride;
  reg sps_start_ack, blk_start_ack, head_start_ack;
  wire sps_st, blk_st, head_st;
  reg sps_done, blk_done, head_done;
  wire lif_rst, bank, busy;
  always #5 s_clk = ~s_clk;
  spikformer_block_sequencer #(.DEPTH(3)) dut (
    .s_clk(s_clk), .s_rst(s_rst),
    .i_sample_start(start), .o_sample_done(done), .o_sample_error(err),
    .o_block_idx(bidx), .o_block_first(bfirst), .o_block_last(blast),
    .o_block_weight_base(wbase), .i_block_weight_stride(stride),
    .o_sps_start(sps_st), .i_sps_done(sps_done),
    .o_blk_start(blk_st), .i_blk_done(blk_done),
    .o_head_start(head_st), .i_head_done(head_done),
    .o_lif_rst(lif_rst), .o_bank_sel(bank), .o_busy(busy)
  );
  integer fails = 0;
  reg [31:0] bases [0:2];
  reg banks [0:2];
  integer bi;
  initial begin
    sps_done = 0; blk_done = 0; head_done = 0; stride = 32'h100000;
    repeat (4) @(posedge s_clk); s_rst = 0;
    @(posedge s_clk); start = 1;
    @(posedge s_clk); start = 0;
    bi = 0;
    // SPS phase
    wait (sps_st); @(posedge s_clk); #1; sps_done = 1;
    @(posedge s_clk); #1; sps_done = 0;
    // 3 blocks with distinct weights
    repeat (3) begin
      wait (blk_st);
      bases[bi] = wbase; banks[bi] = bank;
      if (bi == 0 && !bfirst) begin $display("FAIL b0 not first"); fails = fails + 1; end
      if (bi == 2 && !blast) begin $display("FAIL b2 not last"); fails = fails + 1; end
      if (bi == 1 && (bfirst || blast)) begin $display("FAIL b1 flags"); fails = fails + 1; end
      @(posedge s_clk); #1; blk_done = 1;
      @(posedge s_clk); #1; blk_done = 0;
      bi = bi + 1;
    end
    // head only after last
    wait (head_st);
    if (bases[0] == bases[1] || bases[1] == bases[2]) begin
      $display("FAIL weight base reuse: %h %h %h", bases[0], bases[1], bases[2]);
      fails = fails + 1;
    end else $display("PASS distinct bases %h %h %h", bases[0], bases[1], bases[2]);
    if (banks[0] == banks[1] && banks[1] == banks[2]) begin
      $display("FAIL bank never toggles");
      fails = fails + 1;
    end else $display("PASS bank toggles %b %b %b", banks[0], banks[1], banks[2]);
    @(posedge s_clk); #1; head_done = 1;
    @(posedge s_clk); #1; head_done = 0;
    wait (done);
    // second run A->B->A lifecycle
    @(posedge s_clk); start = 1;
    @(posedge s_clk); start = 0;
    wait (sps_st);
    $display("PASS second start accepted");
    if (fails == 0) $display("ALL PASS tb_sequencer");
    else $display("FAILURES=%0d tb_sequencer", fails);
    $finish;
  end
  initial begin
    repeat (50000) @(posedge s_clk);
    $display("FAIL timeout bi=%0d", bi);
    $finish;
  end
endmodule
