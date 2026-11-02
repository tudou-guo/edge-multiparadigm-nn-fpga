// Reference-style RGB encoder (conv_layer1 + SpikingEncoder mechanism).
// Upstream: PE_dsp_part/conv_layer1.v + Multi_add_unit.v + SpikingEncoder.v
//   - 27 signed 8x8 multiplier units (C0=3 x 3x3 taps) per output channel.
//   - Row/window reuse: frame in 1 BRAM (1024x24b), shadowed to flop window
//     array for single-cycle 3x3 pad mux at 200MHz.
//   - Output-channel time division: one oc per pass through shared 27-mult
//     datapath + 5-stage pipelined add-tree + bias + shared LIF group.
// Weight bank: C1 x 216b BRAM, sync read. Bias1: local 21b regs from
// broadcast bias-pair stream (lo field). Pixels signed INT8.
// Out: raster-major, oc-group-minor beats (LANES*T bits) with ready/valid.
module spiking_rgb_encoder #(
  parameter int H = 32,
  parameter int W_IN = 32,
  parameter int C0 = 3,
  parameter int C1 = 96,
  parameter int T = 4,
  parameter int LANES = 8,
  parameter int WT_W = 8,
  parameter int ACC_W = 21,
  parameter int C2MAX = 384
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic [63:0] i_w_data,
  input  logic i_w_valid,
  output logic o_w_ready,
  input  logic [C0*8-1:0] i_pix_data,
  input  logic i_pix_valid,
  output logic o_pix_ready,
  input  logic signed [ACC_W-1:0] i_thr,
  input  logic i_start,
  output logic o_done,
  output logic [LANES*T-1:0] o_beat_data,
  output logic o_beat_valid,
  input  logic i_beat_ready,
  output logic o_beat_last
);
  localparam int G1 = C1 / LANES;
  localparam int W1B = (C1*C0*9+7)/8;
  localparam int NPOS = H*W_IN;
  localparam int BIASDEPTH = C2MAX;
  // Weight bank as registers (reference conv_layer1 r_weight_array style;
  // 96x216b = ~20kb; avoids a BRAM sync-read prefetch stage in this path).
  logic [27*8-1:0] wbank [C1];
  logic signed [ACC_W-1:0] bias1 [C1];
  // Frame store as plain registers (3KB; single-cycle 3x3 pad mux at 200MHz).
  // Written directly by the pixel stream; no BRAM sync-read hazard.
  logic [7:0] fwin [H][W_IN][C0];
  logic [$clog2(W1B+1)-1:0] ldcnt;
  logic [$clog2(BIASDEPTH+1)-1:0] bcap;
  logic [$clog2(NPOS+1)-1:0] pixcnt;
  logic w1_done, bias_done, pix_done, fwin_loaded;
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin
      ldcnt <= 0; bcap <= 0; pixcnt <= 0;
      w1_done <= 1'b0; bias_done <= 1'b0; pix_done <= 1'b0; fwin_loaded <= 1'b0;
      o_w_ready <= 1'b1; o_pix_ready <= 1'b0;
    end else if (!w1_done) begin
      o_w_ready <= 1'b1; o_pix_ready <= 1'b0;
      if (i_w_valid && o_w_ready) begin
        integer base, k2, idx, oc2, tp2;
        base = int'(ldcnt) * 8;
        for (k2 = 0; k2 < 8; k2 = k2 + 1) begin
          idx = base + k2;
          if (idx < C1*C0*9) begin
            oc2 = idx / (C0*9); tp2 = idx % (C0*9);
            wbank[oc2][(tp2+1)*8-1 -: 8] <= i_w_data[(k2+1)*8-1 -: 8];
          end
        end
        if (base + 8 >= C1*C0*9) w1_done <= 1'b1;
        else ldcnt <= ldcnt + 1'b1;
      end
    end else if (!bias_done) begin
      o_w_ready <= 1'b1; o_pix_ready <= 1'b0;
      if (i_w_valid && o_w_ready) begin
        if (bcap < C1) bias1[bcap] <= i_w_data[ACC_W-1:0];
        if (bcap == BIASDEPTH-1) bias_done <= 1'b1;
        bcap <= bcap + 1'b1;
      end
    end else if (!pix_done) begin
      o_w_ready <= 1'b0; o_pix_ready <= 1'b1;
      if (i_pix_valid && o_pix_ready) begin
        for (int cc = 0; cc < C0; cc = cc + 1)
          fwin[pixcnt/W_IN][pixcnt%W_IN][cc] <= i_pix_data[(cc+1)*8-1 -: 8];
        if (pixcnt == NPOS-1) begin pix_done <= 1'b1; o_pix_ready <= 1'b0; fwin_loaded <= 1'b1; end
        else pixcnt <= pixcnt + 1'b1;
      end
    end else begin o_w_ready <= 1'b0; o_pix_ready <= 1'b0; end
  end
  // fwin written directly by pixel stream; fwin_loaded set on last pixel.
  // ---- datapath regs ----
  typedef enum logic [2:0] {E_IDLE, E_PREF, E_WAIT, E_LIF, E_EMIT} estate_t;
  estate_t st;
  logic [$clog2(NPOS)-1:0] pos;
  logic [$clog2(G1)-1:0] grp;
  logic [$clog2(LANES)-1:0] lane;
  logic [$clog2(C1)-1:0] cur_oc;
  assign cur_oc = grp*LANES + lane;
  logic [27*8-1:0] wentry;
  logic signed [WT_W-1:0] wt [27];
  logic signed [7:0] px [27];
  logic signed [ACC_W-1:0] biassel;
  logic [3:0] waitcnt;
  logic signed [15:0] prod [27];
  logic [27*16-1:0] tree_in;
  logic [20:0] tree_out;
  logic signed [ACC_W-1:0] psum;
  logic lif_v;
  logic signed [T*ACC_W-1:0] lif_ps;
  logic [T-1:0] lif_spk;
  logic lif_rdy;
  logic [T-1:0] lane_spk [LANES];
  logic [LANES*T-1:0] beat_reg;
  always_ff @(posedge s_clk) begin
    if (st == E_PREF) begin
      integer yy0, xx0, ic, b, yy, xx;
      wentry <= wbank[cur_oc];
      biassel <= bias1[cur_oc];
      yy0 = int'(pos / W_IN); xx0 = int'(pos % W_IN);
      for (ic = 0; ic < C0; ic = ic + 1) begin
        for (b = 0; b < 9; b = b + 1) begin
          yy = yy0 + b/3 - 1; xx = xx0 + b%3 - 1;
          if (yy >= 0 && yy < H && xx >= 0 && xx < W_IN)
            px[ic*9+b] <= $signed(fwin[yy][xx][ic]);
          else
            px[ic*9+b] <= 8'sd0;
        end
      end
      for (int k = 0; k < 27; k = k + 1)
        wt[k] <= $signed(wbank[cur_oc][(k+1)*8-1 -: 8]);
    end
  end
  always_ff @(posedge s_clk) begin
    for (int k = 0; k < 27; k = k + 1)
      prod[k] <= $signed(wt[k]) * $signed(px[k]);
  end
  always_comb begin
    for (int k = 0; k < 27; k = k + 1) tree_in[(k+1)*16-1 -: 16] = prod[k];
  end
  spiking_adder_tree #(.N(27), .W(16), .STAGES(5), .NPOW(32), .OW(21)) u_tree (
    .s_clk(s_clk), .s_rst(s_rst), .idata(tree_in), .data_out(tree_out)
  );
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin lif_v <= 1'b0; lif_ps <= '0; psum <= '0; end
    else begin
      lif_v <= 1'b0;
      // Tree latency: prod@PREF+1, 5 tree stages -> full sum at PREF+6.
      // Sample psum at WAIT==6 (tree stable), pulse LIF at WAIT==7.
      // (Sampling at 5 uses the previous lane's sum.)
      if (st == E_WAIT && waitcnt == 4'd6) begin
        psum <= $signed(tree_out) + biassel;
      end
      if (st == E_WAIT && waitcnt == 4'd7) begin
        for (int t = 0; t < T; t = t + 1)
          lif_ps[(t+1)*ACC_W-1 -: ACC_W] <= psum;
        lif_v <= 1'b1;
      end
    end
  end
  spiking_lif_group #(.T(T), .BITS(ACC_W)) u_lif (
    .s_clk(s_clk), .s_rst(s_rst), .i_lif_thrd(i_thr),
    .i_PsumValid(lif_v), .i_PsumData(lif_ps),
    .o_spikes_out(lif_spk), .o_spikes_valid(lif_rdy)
  );
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin
      st <= E_IDLE; pos <= 0; grp <= 0; lane <= 0; waitcnt <= 0;
      o_done <= 1'b0; o_beat_valid <= 1'b0; o_beat_last <= 1'b0; beat_reg <= '0;
    end else case (st)
      E_IDLE: begin
        // o_done is sticky until next start (controller may be flushing
        // a row-block when the final beat lands; a 1-cycle pulse would
        // be missed). Cleared here on a new start.
        if (i_start && w1_done && bias_done && fwin_loaded) begin
          o_done <= 1'b0;
          pos <= 0; grp <= 0; lane <= 0; waitcnt <= 0; st <= E_PREF;
        end
      end
      E_PREF: begin waitcnt <= 0; st <= E_WAIT; end
      E_WAIT: begin
        if (waitcnt == 4'd7) begin
          if (lif_rdy) begin
            lane_spk[lane] <= lif_spk; st <= E_LIF;
          end
        end else waitcnt <= waitcnt + 1'b1;
      end
      E_LIF: begin
        // lif_spk already captured; advance lane
        if (lane == LANES-1) begin
          for (int a = 0; a < LANES; a = a + 1)
            beat_reg[(a+1)*T-1 -: T] <= (a == LANES-1) ? lif_spk : lane_spk[a];
          st <= E_EMIT;
        end else begin lane <= lane + 1'b1; st <= E_PREF; end
      end
      E_EMIT: begin
        if (!o_beat_valid) begin
          o_beat_data <= beat_reg; o_beat_valid <= 1'b1;
          o_beat_last <= (pos == NPOS-1 && grp == G1-1);
        end else if (i_beat_ready) begin
          o_beat_valid <= 1'b0;
          if (pos == NPOS-1 && grp == G1-1) begin o_done <= 1'b1; st <= E_IDLE; end
          else begin
            if (grp == G1-1) begin grp <= 0; pos <= pos + 1'b1; end
            else grp <= grp + 1'b1;
            lane <= 0; st <= E_PREF;
          end
        end
      end
    endcase
  end
endmodule
