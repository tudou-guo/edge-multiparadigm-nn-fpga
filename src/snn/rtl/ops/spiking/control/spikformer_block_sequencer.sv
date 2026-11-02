// Multi-block sequencer: ONE shared Attention/MLP array, per-block params.
// Upstream has a single block (TOP_Transformer: MLP -> Ram2Head); depth=4
// exists only in python model.py, never in RTL. Diffs / rules enforced here:
// (1) SPS runs ONCE; blocks 0..D-1 loop over the same compute arrays.
// (2) Per block: switch Q/K/V/proj/fc1/fc2 weights + bias + scale + threshold;
//     never reuse block.0 params for block b>0 (block_base pointers).
// (3) Inter-block state rides ping-pong residual banks (bank_sel toggles);
//     local resets must not clear the bank the next block still needs.
// (4) Each LIF re-inits at t=0 (lif_rst per block+layer); sample_rst clears
//     everything including banks; engine_rst only FSMs.
// (5) Head reduce runs ONLY after the last block; mid-block done advances.
module spikformer_block_sequencer #(
  parameter int DEPTH = 1,
  parameter int CH    = 384
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic i_sample_start,
  output logic o_sample_done,
  output logic o_sample_error,
  // block context out
  output logic [$clog2(DEPTH)-1:0] o_block_idx,
  output logic o_block_first,
  output logic o_block_last,
  output logic [31:0] o_block_weight_base,
  input  logic [31:0] i_block_weight_stride,
  // engine control
  output logic o_sps_start,
  input  logic i_sps_done,
  output logic o_blk_start,
  input  logic i_blk_done,
  output logic o_head_start,
  input  logic i_head_done,
  // resets
  output logic o_lif_rst,
  output logic o_bank_sel,
  output logic o_busy
);
  localparam int S_IDLE=0, S_SPS=1, S_BLK=2, S_HEAD=3, S_DONE=4;
  logic [2:0] st;
  logic [$clog2(DEPTH)-1:0] blk;
  logic bank;
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin
      st <= S_IDLE; blk <= 0; bank <= 1'b0;
      o_sample_done <= 1'b0; o_sample_error <= 1'b0; o_busy <= 1'b0;
    end else case (st)
      S_IDLE: if (i_sample_start && !o_busy) begin
        st <= S_SPS; blk <= 0; bank <= 1'b0; o_busy <= 1'b1; o_sample_done <= 1'b0;
      end
      S_SPS: if (i_sps_done) st <= S_BLK;
      S_BLK: if (i_blk_done) begin
        if (blk == DEPTH-1) st <= S_HEAD;
        else begin blk <= blk + 1'b1; bank <= ~bank; end
      end
      S_HEAD: if (i_head_done) begin st <= S_DONE; end
      S_DONE: begin o_sample_done <= 1'b1; o_busy <= 1'b0; st <= S_IDLE; end
      default: st <= S_IDLE;
    endcase
  end
  assign o_block_idx = blk;
  assign o_block_first = (blk == 0);
  assign o_block_last  = (blk == DEPTH-1);
  assign o_block_weight_base = blk * i_block_weight_stride;
  assign o_sps_start = (st == S_SPS);
  assign o_blk_start = (st == S_BLK);
  assign o_head_start = (st == S_HEAD);
  assign o_lif_rst = (st == S_BLK) && i_blk_done;
  assign o_bank_sel = bank;
endmodule
