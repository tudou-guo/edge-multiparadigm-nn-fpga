`timescale 1ns/1ps
module tb_embed_replay;
  reg s_clk = 0, s_rst = 1;
  reg [15:0] thr;
  reg [63:0] tdata;
  reg tvalid;
  wire tready;
  reg [15:0] tok;
  reg tokvalid;
  wire tokready;
  wire bvalid;
  reg bready;
  wire [15:0] bdata;
  wire blast;
  always #5 s_clk = ~s_clk;
  spiking_embed_top #(.V(64), .N(8), .E(16), .T(2), .LANES(8)) dut (
    .s_clk(s_clk), .s_rst(s_rst), .i_thr(thr),
    .i_tab_data(tdata), .i_tab_valid(tvalid), .o_tab_ready(tready),
    .i_tok_id(tok), .i_tok_valid(tokvalid), .o_tok_ready(tokready),
    .o_beat_valid(bvalid), .i_beat_ready(bready),
    .o_beat_data(bdata), .o_beat_last(blast)
  );
  reg [63:0] etab [0:127];
  reg [63:0] ptab [0:15];
  reg [15:0] toks [0:7];
  reg [31:0] expb [0:15];
  integer fails = 0;
  integer i, g, got, k;
  reg [3:0] want;
  initial begin
    $readmemh("D:/dl_snn/build/spikformer/embed_replay/etab.hex", etab);
    $readmemh("D:/dl_snn/build/spikformer/embed_replay/ptab.hex", ptab);
    $readmemh("D:/dl_snn/build/spikformer/embed_replay/toks.hex", toks);
    $readmemh("D:/dl_snn/build/spikformer/embed_replay/exp.hex", expb);
    tvalid = 0; tokvalid = 0; bready = 0; thr = 60;
    repeat (4) @(posedge s_clk); s_rst = 0;
    for (i = 0; i < 128; i = i + 1) begin
      wait (tready); @(posedge s_clk); #1; tdata = etab[i]; tvalid = 1;
      @(posedge s_clk); #1; tvalid = 0;
    end
    for (i = 0; i < 16; i = i + 1) begin
      wait (tready); @(posedge s_clk); #1; tdata = ptab[i]; tvalid = 1;
      @(posedge s_clk); #1; tvalid = 0;
    end
    $display("TB embed tables done t=%0t", $time);
    got = 0; bready = 0;
    fork
      begin
        for (g = 0; g < 8; g = g + 1) begin
          wait (tokready); @(posedge s_clk); #1; tok = toks[g]; tokvalid = 1;
          @(posedge s_clk); #1; tokvalid = 0;
          repeat (2) begin
            bready = 1;
            wait (bvalid);
            @(posedge s_clk); @(posedge s_clk);
            for (k = 0; k < 8; k = k + 1) begin
              want = expb[got] >> (4*(7-k));
              if (bdata[k*2 +: 2] !== want[1:0]) begin
                $display("FAIL beat%0d pos%0d: got %b want %b", got, k, bdata[k*2 +: 2], want[1:0]);
                fails = fails + 1;
              end
            end
            got = got + 1;
            @(posedge s_clk); #1; bready = 0;
          end
        end
      end
      begin
        repeat (300000) @(posedge s_clk);
        $display("FAIL timeout got=%0d", got);
        fails = fails + 1;
        $finish;
      end
    join_any
    disable fork;
    if (fails == 0) $display("ALL PASS tb_embed_replay (16 beats)");
    else $display("FAILURES=%0d tb_embed_replay", fails);
    $finish;
  end
endmodule
