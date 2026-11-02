// MLP block: fc1 (C->HIDDEN) + LIF(fc1) + fc2 (HIDDEN->C).
// 48h NLP integer contract (2026-10-29):
//  - The fc2 branch-output LIF lives in spiking_block_top, NOT here. This
//    controller outputs the FULL signed fc2 psum (TL*T*PSUM_SLICE bits, one
//    beat per output group): no residual add, no low-bit truncation.
//  - fc2 psum accumulator keeps the full PSUM_SLICE signed width.
//  - Weight stream per tile: NBW=TL*TC/8 weight beats (64-bit = 8 consecutive
//    int8 row-major [out][in]) + NBB=TL/2 bias beats (64-bit = two signed int32
//    words, [31:0] and [63:32]); 12 beats/tile at TL=TC=8.
//  - MAC: TL output lanes x TC input taps = 64 mults/cycle, T time slices
//    serialized over T cycles, balanced adder tree. LIF fires only after the
//    real deterministic T+1-cycle completion (never timeout-as-ready).
//
// Ports are frozen by contract hardware_abi.mlp_ports_exact.
module spiking_mlp_controller #(
  parameter int C = 128,
  parameter int HIDDEN = 512,
  parameter int T = 4,
  parameter int RW = 4,
  parameter int W = 8,
  parameter int PSUM_SLICE = 32,
  parameter int TL = 8,
  parameter int TC = 8
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic i_start,
  output logic o_done,
  input  logic i_in_valid,
  input  logic [TC*T*RW-1:0] i_in_data,
  output wire  o_in_ready,
  output wire  o_out_valid,
  input  logic i_out_ready,
  output logic [TL*T*PSUM_SLICE-1:0] o_out_data,
  input  logic [63:0] i_w_data,
  input  logic i_w_valid,
  output wire  o_w_ready,
  output wire  o_w_taken,
  input  logic signed [PSUM_SLICE-1:0] i_thr_hidden
);
  localparam int G1  = HIDDEN/TL;      // fc1 output groups
  localparam int NT  = C/TC;           // input tiles for fc1 (also input beats)
  localparam int G0  = C/TL;           // fc2/output groups
  localparam int NT2 = HIDDEN/TC;      // input tiles for fc2
  localparam int NBW = TL*TC/8;        // weight beats per tile
  localparam int NBB = TL/2;           // bias beats per tile
  // Narrow product width: signed range [-2^(RW+W), 2^(RW+W)-1] exactly covers
  // ((2^RW-1) * 2^(W-1)) = 1920 max magnitude at RW=4/W=8, so PRD_W=RW+W+1=13
  // is provably truncation-free for all parameter values (no 32x32 mult).
  localparam int PRD_W = RW+1+W;       // 13 at RW4/W8; see range note above
  localparam int TS_W  = $clog2(T+4);   // tsub 0..T+3 (5-stage MAC pipeline drain)
  localparam int S_IDLE=0, S_BUF=1, S_LD1T=2, S_FC1=3, S_LIFG=4,
                 S_LD2T=5, S_FC2=6, S_OUT0=7, S_OUT=8, S_DONE=9;

  logic [3:0] st;
  logic [TS_W-1:0] tsub;                 // time-slice counter 0..T-1
  logic [$clog2(NT+1)-1:0] bcnt;         // input beat counter
  logic [$clog2(NBW+1)-1:0] ldcnt;       // weight beat inside tile
  logic ldbias;
  logic [$clog2(NBB+1)-1:0] bcntb;       // bias beat inside tile
  logic [$clog2(G1+1)-1:0] g1;
  logic [$clog2(NT+1)-1:0] t1;
  logic [$clog2(G0+1)-1:0] g0;
  logic [$clog2(NT2+1)-1:0] t2;
  logic [TS_W-1:0] lifc;
  logic oov;

  logic signed [W-1:0] wt1 [TL][TC];
  logic signed [W-1:0] wt2 [TL][TC];
  logic signed [PSUM_SLICE-1:0] b1 [TL];
  logic signed [PSUM_SLICE-1:0] b2 [TL];
  logic [RW-1:0] inbuf [C][T];
  logic [T-1:0] hbuf [HIDDEN];
  logic signed [PSUM_SLICE-1:0] psum1 [TL][T];
  logic signed [PSUM_SLICE-1:0] psum2 [TL][T];

  // ---- fc1 LIF group bank (T+1 cycles) ----
  logic [TL-1:0] lifv;
  logic [TL-1:0] lif_rdy;
  logic [TL*T-1:0] lifspk_flat;
  logic signed [T*PSUM_SLICE-1:0] lifin [TL];
  always_comb begin
    for (int a = 0; a < TL; a = a + 1)
      for (int tt = 0; tt < T; tt = tt + 1)
        lifin[a][(tt+1)*PSUM_SLICE-1 -: PSUM_SLICE] = psum1[a][tt];
  end
  genvar ll;
  generate
    for (ll = 0; ll < TL; ll = ll + 1) begin : g_lif
      spiking_lif_group #(.T(T), .BITS(PSUM_SLICE)) u_l (
        .s_clk(s_clk), .s_rst(s_rst), .i_lif_thrd(i_thr_hidden),
        .i_PsumValid(lifv[ll]), .i_PsumData(lifin[ll]),
        .o_spikes_out(lifspk_flat[(ll+1)*T-1 -: T]),
        .o_spikes_valid(lif_rdy[ll])
      );
    end
  endgenerate

  // ---- MAC: TL lanes x TC taps, narrow signed operands, balanced tree ----
  logic signed [W-1:0] wsel [TL][TC];
  logic signed [PRD_W-1:0] mul_x [TC];
  logic signed [PRD_W-1:0] mul_w [TL][TC];
  // 5-stage MAC pipeline: S1 operands -> S2 products -> S3 half sums ->
  // S4 total -> S5 32-bit accumulate writeback (same structure as block_top).
  logic signed [PRD_W-1:0] xp_q [TC];
  logic signed [W-1:0]     wp_q [TL][TC];
  logic signed [PRD_W-1:0] pr_q [TL][TC];
  logic signed [31:0]      hf_q [TL][2];
  logic signed [31:0]      sm_q [TL];
  logic [3:0]              mpipe_v;
  logic [TS_W-1:0]         wb_tt;
  logic                    mac_iss;
  // ---- MAC 5-stage pipeline (200MHz timing fix) ---------------------------
  // Issue one time slice per cycle for T cycles, then drain the 4 register
  // stages. wb_tt = tsub - 4 identifies the slice being written back.
  assign wb_tt   = tsub - 4;
  assign mac_iss = ((st == S_FC1) || (st == S_FC2)) && (tsub < T);
  always_comb begin
    for (int k = 0; k < TC; k = k + 1)
      if (!mac_iss)
        mul_x[k] = '0;
      else if (st == S_FC2)
        mul_x[k] = {{(PRD_W-1){1'b0}}, hbuf[t2*TC+k][tsub]};       // 0/1 spike
      else
        mul_x[k] = {{(PRD_W-RW){1'b0}}, inbuf[t1*TC+k][tsub]};      // uint RW residual
    for (int a = 0; a < TL; a = a + 1)
      for (int k = 0; k < TC; k = k + 1) begin
        wsel[a][k] = (st == S_FC2) ? wt2[a][k] : wt1[a][k];
        mul_w[a][k] = mac_iss ? {{(PRD_W-W){wsel[a][k][W-1]}}, wsel[a][k]} : '0;
      end
  end
  always_ff @(posedge s_clk) begin
    if (s_rst) begin
      mpipe_v <= '0;
    end else begin
      if (mac_iss) begin
        for (int k = 0; k < TC; k = k + 1) xp_q[k] <= mul_x[k];
        for (int a = 0; a < TL; a = a + 1)
          for (int k = 0; k < TC; k = k + 1) wp_q[a][k] <= wsel[a][k];
      end
      for (int a = 0; a < TL; a = a + 1)
        for (int k = 0; k < TC; k = k + 1)
          pr_q[a][k] <= xp_q[k] * $signed({{(PRD_W-W){wp_q[a][k][W-1]}}, wp_q[a][k]});
      for (int a = 0; a < TL; a = a + 1) begin
        hf_q[a][0] <= (pr_q[a][0] + pr_q[a][1]) + (pr_q[a][2] + pr_q[a][3]);
        hf_q[a][1] <= (pr_q[a][4] + pr_q[a][5]) + (pr_q[a][6] + pr_q[a][7]);
      end
      for (int a = 0; a < TL; a = a + 1)
        sm_q[a] <= hf_q[a][0] + hf_q[a][1];
      mpipe_v <= {mpipe_v[2:0], mac_iss};
    end
  end

  assign o_in_ready = (st == S_BUF);
  assign o_w_ready  = (st == S_LD1T) || (st == S_LD2T);
  assign o_w_taken  = i_w_valid && o_w_ready;
  assign o_out_valid = oov;

`ifndef SYNTHESIS
  always_ff @(posedge s_clk) begin
    if (s_rst === 1'b0 && st == S_LIFG && lifc == T+1 && lif_rdy !== {TL{1'b1}})
      $display("ERROR %m: LIF completion mismatch at t=%0t (expected T+1=%0d cycles)", $time, T+1);
  end
`endif

  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin
      st <= S_IDLE; tsub <= 0; bcnt <= 0; ldcnt <= 0; ldbias <= 1'b0;
      bcntb <= 0; g1 <= 0; t1 <= 0; g0 <= 0; t2 <= 0; lifc <= 0; lifv <= '0;
      o_done <= 1'b0; oov <= 1'b0; o_out_data <= '0;
    end else case (st)
      S_IDLE: begin
        o_done <= 1'b0;
        if (i_start) begin
          bcnt<=0; ldcnt<=0; ldbias<=1'b0; bcntb<=0; tsub<=0;
          g1<=0; t1<=0; g0<=0; t2<=0; lifc<=0; lifv<='0; oov<=1'b0;
          st <= S_BUF;
        end
      end
      S_BUF: begin
        if (i_in_valid) begin
          for (int k = 0; k < TC; k = k + 1)
            for (int tt = 0; tt < T; tt = tt + 1)
              inbuf[bcnt*TC+k][tt] <= i_in_data[(k*T+tt+1)*RW-1 -: RW];
          if (bcnt == NT-1) begin
            bcnt <= 0; ldcnt <= 0; ldbias <= 1'b0; bcntb <= 0; t1 <= 0;
            st <= S_LD1T;
          end else bcnt <= bcnt + 1'b1;
        end
      end
      S_LD1T: begin
        if (i_w_valid) begin
          integer idx;
          if (!ldbias) begin
            for (int kk = 0; kk < 8; kk = kk + 1) begin
              idx = ldcnt*8 + kk;
              if (idx < TL*TC) wt1[idx/TC][idx%TC] <= $signed(i_w_data[(kk+1)*8-1 -: 8]);
            end
            if (ldcnt == NBW-1) begin ldbias <= 1'b1; bcntb <= 0; end
            else ldcnt <= ldcnt + 1'b1;
          end else begin
            if (bcntb*2 < TL) b1[bcntb*2] <= $signed(i_w_data[PSUM_SLICE-1:0]);
            if (bcntb*2+1 < TL) b1[bcntb*2+1] <= $signed(i_w_data[32+PSUM_SLICE-1:32]);
            if (bcntb == NBB-1) begin
              st <= S_FC1; tsub <= 0; ldcnt <= 0; ldbias <= 1'b0; bcntb <= 0;
            end else bcntb <= bcntb + 1'b1;
          end
        end
      end
      S_FC1: begin
        if (mpipe_v[3]) begin                       // S5 writeback
          for (int a = 0; a < TL; a = a + 1) begin
            logic signed [31:0] base;
            base = (t1 == 0) ? b1[a] : psum1[a][wb_tt];
            psum1[a][wb_tt] <= base + sm_q[a];
          end
        end
        if (tsub == T+3) begin                      // all T slices written back
          tsub <= 0;
          if (t1 == NT-1) begin
            t1 <= 0; lifc <= 0; lifv <= {TL{1'b1}}; st <= S_LIFG;
          end else begin
            t1 <= t1 + 1'b1; ldcnt <= 0; ldbias <= 1'b0; bcntb <= 0;
            st <= S_LD1T;
          end
        end else tsub <= tsub + 1'b1;
      end
      S_LIFG: begin
        lifv <= '0;
        lifc <= lifc + 1'b1;
        if (lifc == T+1) begin
          for (int a = 0; a < TL; a = a + 1)
            hbuf[g1*TL+a] <= lifspk_flat[(a+1)*T-1 -: T];
          lifc <= 0;
          if (g1 == G1-1) begin
            g1 <= 0; g0 <= 0; t2 <= 0; ldcnt <= 0; ldbias <= 1'b0; bcntb <= 0;
            st <= S_LD2T;
          end else begin
            g1 <= g1 + 1'b1; ldcnt <= 0; ldbias <= 1'b0; bcntb <= 0;
            st <= S_LD1T;
          end
        end
      end
      S_LD2T: begin
        if (i_w_valid) begin
          integer idx;
          if (!ldbias) begin
            for (int kk = 0; kk < 8; kk = kk + 1) begin
              idx = ldcnt*8 + kk;
              if (idx < TL*TC) wt2[idx/TC][idx%TC] <= $signed(i_w_data[(kk+1)*8-1 -: 8]);
            end
            if (ldcnt == NBW-1) begin ldbias <= 1'b1; bcntb <= 0; end
            else ldcnt <= ldcnt + 1'b1;
          end else begin
            if (bcntb*2 < TL) b2[bcntb*2] <= $signed(i_w_data[PSUM_SLICE-1:0]);
            if (bcntb*2+1 < TL) b2[bcntb*2+1] <= $signed(i_w_data[32+PSUM_SLICE-1:32]);
            if (bcntb == NBB-1) begin
              st <= S_FC2; tsub <= 0; ldcnt <= 0; ldbias <= 1'b0; bcntb <= 0;
            end else bcntb <= bcntb + 1'b1;
          end
        end
      end
      S_FC2: begin
        if (mpipe_v[3]) begin                       // S5 writeback
          for (int a = 0; a < TL; a = a + 1) begin
            logic signed [31:0] base;
            base = (t2 == 0) ? b2[a] : psum2[a][wb_tt];
            psum2[a][wb_tt] <= base + sm_q[a];
          end
        end
        if (tsub == T+3) begin                      // all T slices written back
          tsub <= 0;
          if (t2 == NT2-1) begin
            t2 <= 0; st <= S_OUT0;
          end else begin
            t2 <= t2 + 1'b1; ldcnt <= 0; ldbias <= 1'b0; bcntb <= 0;
            st <= S_LD2T;
          end
        end else tsub <= tsub + 1'b1;
      end
      S_OUT0: begin
        for (int a = 0; a < TL; a = a + 1)
          for (int tt = 0; tt < T; tt = tt + 1)
            o_out_data[(a*T+tt+1)*PSUM_SLICE-1 -: PSUM_SLICE] <= psum2[a][tt];
        oov <= 1'b1;
        st <= S_OUT;
      end
      S_OUT: begin
        if (i_out_ready) begin
          oov <= 1'b0;
          if (g0 == G0-1) begin
            g0 <= 0; o_done <= 1'b1; st <= S_DONE;
          end else begin
            g0 <= g0 + 1'b1; t2 <= 0; ldcnt <= 0; ldbias <= 1'b0; bcntb <= 0;
            st <= S_LD2T;
          end
        end
      end
      S_DONE: begin
        if (i_start) begin
          o_done <= 1'b0;
          bcnt<=0; ldcnt<=0; ldbias<=1'b0; bcntb<=0; tsub<=0;
          g1<=0; t1<=0; g0<=0; t2<=0; lifc<=0; lifv<='0; oov<=1'b0;
          st <= S_BUF;
        end
      end
      default: st <= S_IDLE;
    endcase
  end
endmodule
