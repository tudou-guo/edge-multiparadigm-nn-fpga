// Native vision top, A-round (task VISION_REFERENCE_ROUTE s4/P2).
// DDR holds ONE copy of every weight (unique) + image; this top streams:
//   block weights  -> spiking_block_top loader (single on-chip copy, retained
//                     from the validated block; B-round will tile it)
//   SPS weights    -> spiking_sps_ref_top (w1, bias pairs, w2 tiles in
//                     (out-group,in-tile) schedule order; no w2rom)
//   pixels         -> SPS encoder; SPS tokens -> block; block beats -> sums.
// Deleted vs legacy spikformer_vis_top: w1rom/brom/w2rom/blkrom/tbuf.
// Token reorder uses the SPS feature banks + 2-deep adapter skid only.
// Block starts first and takes its weight stream, then waits in S_TOK for
// SPS tokens; sums accumulate on block-output handshake (order-independent).
module spikformer_vis_native_top #(
  parameter int H = 32, parameter int W_IN = 32,
  parameter int C0 = 3, parameter int C1 = 96, parameter int C2 = 384,
  parameter int T = 4, parameter int LANES = 8, parameter int TC2 = 8,
  parameter int E = 384, parameter int HEADS = 12, parameter int HEAD_DIM = 32,
  parameter int HIDDEN = 1536,
  parameter int TL = 8, parameter int TC = 8, parameter int RW = 4,
  parameter int PSUM_SLICE = 32, parameter int SUM_W = 6, parameter int W = 8,
  parameter int OW = 32
)(
  input  logic s_clk, input  logic s_rst,
  input  logic i_start, output logic o_done,
  // DDR streams (unique weights; schedule-ordered by DMA/host)
  input  logic [63:0] i_blk_w_data, input  logic i_blk_w_valid, output logic o_blk_w_ready,
  input  logic [63:0] i_sps_w_data, input  logic i_sps_w_valid, output logic o_sps_w_ready,
  input  logic [C0*8-1:0] i_pix_data, input  logic i_pix_valid, output logic o_pix_ready,
  input  logic signed [PSUM_SLICE-1:0] i_thr_1, input  logic signed [PSUM_SLICE-1:0] i_thr_2,
  input  logic signed [PSUM_SLICE-1:0] i_thr_q, input  logic signed [PSUM_SLICE-1:0] i_thr_k,
  input  logic signed [PSUM_SLICE-1:0] i_thr_v, input  logic signed [PSUM_SLICE-1:0] i_thr_proj,
  input  logic signed [PSUM_SLICE-1:0] i_thr_fc1, input  logic signed [PSUM_SLICE-1:0] i_thr_fc2,
  output logic [OW-1:0] o_sums [E],
  output logic o_sums_valid
);
  localparam int NTOK = (H/4)*(W_IN/4);
  localparam int G = E/TL;
  localparam int TBEATS = NTOK*G; // 3072 token beats = 3072 block-out beats
  // SPS
  wire [TL*T*RW-1:0] sps_tok; wire sps_tv, sps_tr, sps_last;
  wire sps_done;
  logic sps_start;
  spiking_sps_ref_top #(.H(H), .W_IN(W_IN), .C0(C0), .C1(C1), .C2(C2),
    .T(T), .LANES(LANES), .TC2(TC2), .WT_W(W), .ACC_W(21), .RW(RW)) u_sps (
    .s_clk(s_clk), .s_rst(s_rst),
    .i_thr_1(i_thr_1[15:0]), .i_thr_2(i_thr_2[15:0]),
    .i_weight_in(i_sps_w_data), .i_weight_valid(i_sps_w_valid), .o_weight_ready(o_sps_w_ready),
    .i_pix_valid(i_pix_valid), .o_pix_ready(o_pix_ready), .i_pix_data(i_pix_data),
    .i_start(sps_start), .o_done(sps_done),
    .o_tok_data(sps_tok), .o_tok_valid(sps_tv), .i_tok_ready(sps_tr), .o_tok_last(sps_last)
  );
  // block
  wire blk_done, blk_iv, blk_irdy, blk_ov, blk_olast;
  wire [TL*T*RW-1:0] blk_idata, blk_odata;
  logic blk_start, blk_ordy;
  spiking_block_top #(.E(E), .HEADS(HEADS), .HEAD_DIM(HEAD_DIM), .HIDDEN(HIDDEN),
    .T(T), .TOKENS(NTOK), .TL(TL), .TC(TC), .RW(RW), .W(W),
    .PSUM_SLICE(PSUM_SLICE), .SUM_W(SUM_W)) u_blk (
    .s_clk(s_clk), .s_rst(s_rst), .i_start(blk_start), .o_done(blk_done),
    .i_in_valid(blk_iv), .o_in_ready(blk_irdy), .i_in_data(blk_idata),
    .o_out_valid(blk_ov), .i_out_ready(blk_ordy), .o_out_data(blk_odata),
    .o_out_last(blk_olast),
    .i_w_data(i_blk_w_data), .i_w_valid(i_blk_w_valid), .o_w_ready(o_blk_w_ready),
    .i_thr_q(i_thr_q), .i_thr_k(i_thr_k), .i_thr_v(i_thr_v),
    .i_thr_proj(i_thr_proj), .i_thr_fc1(i_thr_fc1), .i_thr_fc2(i_thr_fc2)
  );
  // token path: SPS -> block (direct, both handshake; adapter skid inside SPS)
  assign blk_idata = sps_tok;
  assign blk_iv = sps_tv;
  assign sps_tr = blk_irdy;
  // top FSM
  typedef enum logic [2:0] {N_IDLE, N_BLKLD, N_RUN, N_DRAIN, N_DONE} nstate_t;
  nstate_t st;
  logic [$clog2(3072+1)-1:0] capi;
  logic [OW-1:0] acc [E];
  // P2 sums fix: a block-out beat carries TL lanes x T timesteps of RW-bit
  // samples. The old N_RUN code did T non-blocking adds to the same acc reg
  // inside a for loop (only the last one wins -> each beat added just t=T-1).
  // Sum the T samples of one beat combinationally first, then do ONE add.
  logic [OW-1:0] beat_add [TL];
  integer a;
  always_comb begin
    for (int a2 = 0; a2 < TL; a2 = a2 + 1) begin
      beat_add[a2] = '0;
      for (int t2 = 0; t2 < T; t2 = t2 + 1)
        beat_add[a2] = beat_add[a2]
          + {{(OW-RW){1'b0}}, blk_odata[(a2*T+t2+1)*RW-1 -: RW]};
    end
  end
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin
      st <= N_IDLE; o_done <= 1'b0; o_sums_valid <= 1'b0;
      blk_start <= 1'b0; sps_start <= 1'b0; blk_ordy <= 1'b0; capi <= 0;
      for (a = 0; a < E; a = a + 1) acc[a] <= 0;
    end else case (st)
      N_IDLE: begin
        if (i_start) begin
          o_done <= 1'b0; o_sums_valid <= 1'b0;
          for (a = 0; a < E; a = a + 1) acc[a] <= 0;
          capi <= 0; blk_start <= 1'b1; st <= N_BLKLD;
        end
      end
      N_BLKLD: begin
        blk_start <= 1'b0;
        // block loads weights autonomously after start; wait until it
        // opens its token port, then start SPS (weights stream on demand).
        if (blk_irdy) begin sps_start <= 1'b1; blk_ordy <= 1'b1; st <= N_RUN; end
      end
      N_RUN: begin
        sps_start <= 1'b0;
        if (blk_ov && blk_ordy) begin
          // one add per beat of its full T-sum (see beat_add above)
          for (a = 0; a < TL; a = a + 1)
            acc[(capi%G)*TL+a] <= acc[(capi%G)*TL+a] + beat_add[a];
          if (blk_olast) begin blk_ordy <= 1'b0; st <= N_DRAIN; end
          else capi <= capi + 1'b1;
        end
      end
      N_DRAIN: begin
        for (a = 0; a < E; a = a + 1) o_sums[a] <= acc[a];
        o_sums_valid <= 1'b1; o_done <= 1'b1; st <= N_DONE;
      end
      // o_done/sums sticky until next i_start (same race as SPS done).
      N_DONE: begin
        o_done <= 1'b1;
        if (i_start) begin o_done <= 1'b0; o_sums_valid <= 1'b0; st <= N_BLKLD; capi <= 0;
          for (a = 0; a < E; a = a + 1) acc[a] <= 0;
          blk_start <= 1'b1;
        end
        else if (!i_start) begin st <= N_IDLE; end
      end
      default: st <= N_IDLE;
    endcase
  end
endmodule
