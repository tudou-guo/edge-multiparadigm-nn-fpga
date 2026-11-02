// Reference-style PSUM + LIF stage (psum_callback + psum_lif_top mechanism).
// Upstream: eyeriss_part/PSUM_RAM/{psum_callback.v, psum_lif_top.v, psum_ram_top.v}
//   - Synchronous read-modify-write PSUM RAM, FIRST init / MID accumulate /
//     LAST finalize + LIF. Bias added exactly once (FIRST tile).
//   - Fixed 1-cycle read latency; result/addr/tags travel together.
//   - Same-address RMW hazard: controller sequences tile-inner sweeps so the
//     same (pos) is revisited only after the full array pipeline (~19 cycles),
//     well beyond the 2-cycle RMW window. No same-cycle bypass needed.
// Organization: 8 lane banks, each 256 deep x 84b (T=4 x 21b). Holds current
// output-group's 256 positions; reused across groups (bank swap by controller
// phase, never concurrent with array writes to the same bank).
module spiking_sps_psum #(
  parameter int LANES = 8,
  parameter int T = 4,
  parameter int ACC_W = 21,
  parameter int DEPTH = 256,
  parameter int POS_W = 8,
  parameter int GRP_W = 6,
  parameter int C2 = 384,
  parameter int BIASDEPTH = 384,
  parameter int C2MAX = 384
)(
  input  logic s_clk,
  input  logic s_rst,
  // bias-pair broadcast stream (hi field captured as bias2)
  input  logic [63:0] i_w_data,
  input  logic i_w_valid,
  output logic o_w_ready,
  input  logic i_w_done,   // top signals bias phase complete
  // tile-result stream from array
  input  logic i_res_valid,
  output logic o_res_ready,
  input  logic [LANES*T*21-1:0] i_res_data,
  input  logic [POS_W-1:0] i_res_pos,
  input  logic [GRP_W-1:0] i_res_grp,
  input  logic i_res_first,
  input  logic i_res_last,
  input  logic signed [ACC_W-1:0] i_thr,
  // finalized spike beats (one per (pos,grp) after LAST tile)
  output logic [LANES*T-1:0] o_beat_data,
  output logic o_beat_valid,
  input  logic i_beat_ready,
  output logic [POS_W-1:0] o_beat_pos,
  output logic [GRP_W-1:0] o_beat_grp
);
  logic signed [ACC_W-1:0] bias2 [C2];
  logic [$clog2(BIASDEPTH+1)-1:0] bcap;
  logic bias_done;
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin bcap <= 0; bias_done <= 1'b0; o_w_ready <= 1'b1; end
    else if (!bias_done) begin
      o_w_ready <= 1'b1;
      if (i_w_valid && o_w_ready) begin
        if (bcap < C2) bias2[bcap] <= i_w_data[2*ACC_W-1:ACC_W];
        bcap <= bcap + 1'b1;
        if (i_w_done) bias_done <= 1'b1;
      end
    end else o_w_ready <= 1'b0;
  end
  // 8 lane PSUM banks, sync read (BRAM: 84b x 256 per lane)
  (* ram_style = "block" *) logic [T*ACC_W-1:0] pram [LANES][DEPTH];
  // NOTE: no ram_style hint: 84b x 256 per lane infers BRAM cleanly in Vivado;
  // the hint is added at top-level per-bank if the OOC audit requests it.
  logic [POS_W-1:0] rd_addr [LANES];
  logic [T*ACC_W-1:0] rd_data [LANES];
  logic rd_valid;
  // request regs (stage 0)
  logic [POS_W-1:0] q_pos; logic [GRP_W-1:0] q_grp;
  logic q_first, q_last;
  logic signed [ACC_W-1:0] q_tile [LANES][T];
  logic q_v;
  logic can_advance;
  assign can_advance = !o_beat_valid || i_beat_ready;
  assign o_res_ready = !q_v || (q_v && rd_valid && can_advance);
  always_ff @(posedge s_clk) begin
    for (int a = 0; a < LANES; a = a + 1) rd_data[a] <= pram[a][rd_addr[a]];
  end
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin rd_valid <= 1'b0; q_v <= 1'b0; end
    else begin
      if (o_res_ready && i_res_valid) begin
        q_pos <= i_res_pos; q_grp <= i_res_grp;
        q_first <= i_res_first; q_last <= i_res_last;
        for (int a = 0; a < LANES; a = a + 1)
          for (int t = 0; t < T; t = t + 1)
            q_tile[a][t] <= $signed(i_res_data[(a*T+t+1)*21-1 -: 21]);
        for (int a = 0; a < LANES; a = a + 1) rd_addr[a] <= i_res_pos;
        q_v <= 1'b1; rd_valid <= 1'b0;
      end else if (q_v && !rd_valid) begin
        rd_valid <= 1'b1;
      end else if (q_v && rd_valid && can_advance) begin
        q_v <= 1'b0; rd_valid <= 1'b0;
      end
    end
  end
  // RMW + LIF
  logic lif_v;
  logic signed [T*ACC_W-1:0] lif_ps [LANES];
  logic [LANES*T-1:0] lif_spk;
  logic [LANES-1:0] lif_rdy;
  genvar gl;
  generate
    for (gl = 0; gl < LANES; gl = gl + 1) begin : g_lif
      spiking_lif_group #(.T(T), .BITS(ACC_W)) u_l (
        .s_clk(s_clk), .s_rst(s_rst), .i_lif_thrd(i_thr),
        .i_PsumValid(lif_v), .i_PsumData(lif_ps[gl]),
        .o_spikes_out(lif_spk[(gl+1)*T-1 -: T]), .o_spikes_valid(lif_rdy[gl])
      );
    end
  endgenerate
  typedef enum logic [1:0] {P_IDLE, P_LIFW} pstate_t;
  pstate_t pst;
  logic [LANES*T-1:0] beat_q;
  logic [POS_W-1:0] pos_q; logic [GRP_W-1:0] grp_q;
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin
      pst <= P_IDLE; lif_v <= 1'b0;
      o_beat_valid <= 1'b0; o_beat_pos <= '0; o_beat_grp <= '0;
    end else case (pst)
      P_IDLE: begin
        lif_v <= 1'b0;
        if (q_v && rd_valid && can_advance) begin
          // compute new psum per lane/t
          for (int a = 0; a < LANES; a = a + 1) begin
            logic signed [ACC_W-1:0] oldv [T];
            logic signed [ACC_W-1:0] newv [T];
            for (int t = 0; t < T; t = t + 1) begin
              oldv[t] = $signed(rd_data[a][(t+1)*ACC_W-1 -: ACC_W]);
              if (q_first)
                newv[t] = bias2[q_grp*LANES+a] + q_tile[a][t];
              else
                newv[t] = oldv[t] + q_tile[a][t];
              pram[a][q_pos][(t+1)*ACC_W-1 -: ACC_W] <= newv[t];
              lif_ps[a][(t+1)*ACC_W-1 -: ACC_W] <= newv[t];
            end
          end
          pos_q <= q_pos; grp_q <= q_grp;
          if (q_last) begin lif_v <= 1'b1; pst <= P_LIFW; end
        end
      end
      P_LIFW: begin
        lif_v <= 1'b0;
        if (lif_rdy[0]) begin
          beat_q <= lif_spk;
          o_beat_data <= lif_spk; o_beat_pos <= pos_q; o_beat_grp <= grp_q;
          o_beat_valid <= 1'b1; pst <= P_IDLE;
        end
      end
    endcase
    if (o_beat_valid && i_beat_ready) o_beat_valid <= 1'b0;
  end
endmodule
