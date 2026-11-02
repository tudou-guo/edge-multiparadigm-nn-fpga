// Head reduction: per-channel SUM over T*N spike samples (PL), Linear in PS.
// Upstream: Transformer_part/Head/Ram2Head.v comment says mean(2).mean(0)
// but hardware + head_tb.py implement SUM; PS post_proc does FP32 Linear.
// Contract (explicit): PL outputs per-channel uint32 SUM; the 1/(T*N)
// normalization is NOT divided in PL - either divided in PS runtime
// (head_norm=ps_div) or pre-absorbed into Head weights (head_norm=folded).
// Stream model: input beats of HL lanes x T spike bits, token-major,
// G=C/HL beats per token. acc[ch] += popcount(lane T-bits) per beat.
// After NTOK*G beats: sums written out via head_* BRAM port, intr raised.
// Diffs: CH/TOKENS/T/HL params replace 64/384/4 hard-codes + addr widths.
module spiking_head_reduce #(
  parameter int CH     = 384,
  parameter int TOKENS = 64,
  parameter int T      = 4,
  parameter int HL     = 8,
  parameter int SUM_W  = 32
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic i_start,
  input  logic i_beat_valid,
  output logic o_beat_ready,
  input  logic [HL*T-1:0] i_beat_data,
  output logic [SUM_W-1:0] head_dout,
  output logic head_en,
  output logic [3:0] head_we,
  output logic [31:0] head_addr,
  output logic o_sums_valid,
  output logic o_intr
);
  localparam int G = CH/HL;
  localparam int TOTAL = TOKENS*G;
  localparam int S_IDLE=0, S_SUM=1, S_OUT=2, S_OUT2=4, S_INTR=3;
  // S_OUT2 holds the last write an extra cycle so a #-delayed sampler
  // cannot miss it when we drops on the S_OUT->S_INTR edge.
  logic [1:0] st;
  logic [SUM_W-1:0] acc [CH];
  logic [$clog2(TOTAL)-1:0] bcnt;
  logic [$clog2(CH)-1:0] ocnt;
  integer l, tt;
  assign o_beat_ready = (st == S_SUM);
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin
      st <= S_IDLE; bcnt <= 0; ocnt <= 0;
      head_en <= 1'b0; head_we <= 4'b0; head_dout <= 0; head_addr <= 0;
      o_sums_valid <= 1'b0; o_intr <= 1'b0;
      for (l = 0; l < CH; l = l + 1) acc[l] <= 0;
    end else case (st)
      S_IDLE: begin
        if (i_start) begin
          st <= S_SUM; bcnt <= 0;
          for (l = 0; l < CH; l = l + 1) acc[l] <= 0;
        end
      end
      S_SUM: begin
        if (i_beat_valid) begin
          integer g;
          g = bcnt % G;
          for (l = 0; l < HL; l = l + 1) begin
            logic [7:0] pc;
            pc = 0;
            for (tt = 0; tt < T; tt = tt + 1)
              pc = pc + i_beat_data[l*T+tt];
            acc[g*HL+l] <= acc[g*HL+l] + pc;
          end
          if (bcnt == TOTAL-1) begin st <= S_OUT; ocnt <= 0; o_sums_valid <= 1'b1; end
          else bcnt <= bcnt + 1'b1;
        end
      end
      S_OUT: begin
        head_en <= 1'b1; head_we <= 4'hf;
        head_addr <= ocnt * 4;
        head_dout <= acc[ocnt];
        if (ocnt == CH-1) begin st <= S_OUT2; end
        else ocnt <= ocnt + 1'b1;
      end
      S_OUT2: begin
        st <= S_INTR; head_we <= 4'b0;
      end
      S_INTR: begin
        o_intr <= 1'b1; head_en <= 1'b0;
        if (!i_start) begin o_intr <= 1'b0; o_sums_valid <= 1'b0; st <= S_IDLE; end
      end
      default: st <= S_IDLE;
    endcase
  end
endmodule
