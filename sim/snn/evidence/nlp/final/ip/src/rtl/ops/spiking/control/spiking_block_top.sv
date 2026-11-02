// Spikformer NLP single-block datapath (48h target E128/H4/HD32/HID512/T4/N32).
// 2026-10-29 contract rewrite:
//  - residual in -> Q/K/V linear + LIF(q/k/v) -> ONE shared QK popcount PE
//    reused sequentially over HEADS (channel offset h*HEAD_DIM, score never
//    stored across heads) -> unnormalized AV accumulated one source token per
//    cycle (one head's HEAD_DIM*T lanes per cycle, no /head_dim, no 32-token
//    combinational expansion) -> proj linear -> LIF(proj) -> residual add;
//    -> fc1/fc2 (spiking_mlp_controller, fc1 LIF inside) -> full fc2 psum beat
//    -> LIF(fc2) -> residual add -> out.
//  - MAC = TL output lanes x TC input taps = 64 mults/cycle, T time slices
//    serialized, balanced adder tree. Large weighting tables live in
//    synchronous-read RAM (8 banks per QKV/proj matrix, one 64-bit RAM each
//    for fc1/fc2); no per-element reset. Bias/threshold/control stay regs.
//  - Weight beats are consumed only in S_LDW after i_start; counters advance
//    only on valid && ready; last weight beat is accepted before o_in_ready
//    rises; o_w_ready is high only in the load phase. Consecutive samples/
//    blocks re-load all weights (no global reset needed).
//  - LIF completion is deterministic T+1 cycles (no timeout-as-ready).
//
// Ports are frozen by contract hardware_abi.block_ports_exact.
module spiking_block_top #(
  parameter int E = 128,
  parameter int HEADS = 4,
  parameter int HEAD_DIM = 32,
  parameter int HIDDEN = 512,
  parameter int T = 4,
  parameter int TOKENS = 32,
  parameter int TL = 8,
  parameter int TC = 8,
  parameter int RW = 4,
  parameter int W = 8,
  parameter int PSUM_SLICE = 32,
  parameter int SUM_W = 6
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic i_start,
  output logic o_done,
  input  logic i_in_valid,
  output logic o_in_ready,
  input  logic [TL*T*RW-1:0] i_in_data,
  output logic o_out_valid,
  input  logic i_out_ready,
  output logic [TL*T*RW-1:0] o_out_data,
  output logic o_out_last,
  input  logic [63:0] i_w_data,
  input  logic i_w_valid,
  output logic o_w_ready,
  input  logic signed [PSUM_SLICE-1:0] i_thr_q,
  input  logic signed [PSUM_SLICE-1:0] i_thr_k,
  input  logic signed [PSUM_SLICE-1:0] i_thr_v,
  input  logic signed [PSUM_SLICE-1:0] i_thr_proj,
  input  logic signed [PSUM_SLICE-1:0] i_thr_fc1,
  input  logic signed [PSUM_SLICE-1:0] i_thr_fc2
);
  localparam int G   = E/TL;                 // token beats / output groups
  localparam int NT  = E/TC;                 // input tiles for E-wide linear
  localparam int G1  = HIDDEN/TL;            // fc1 output groups
  localparam int NT2 = HIDDEN/TC;            // fc2 input tiles
  localparam int G0  = E/TL;                 // fc2 output groups
  localparam int AV_W  = $clog2(TOKENS*HEAD_DIM+1);   // 11 at target
  localparam int BANK_DEPTH = (E/TL)*(E/TC);          // entries per weight bank
  localparam int BANK_AW = $clog2(BANK_DEPTH+1);
  localparam int FCD  = HIDDEN*E/8;                   // fc1/fc2 RAM depth (beats)
  localparam int FCAW = $clog2(FCD+1);
  localparam int EEB  = E*E/8;
  localparam int LDM  = (EEB > FCD) ? EEB : FCD;      // longest load phase
  localparam int LD_W = $clog2(LDM+1);
  localparam int NBW  = TL*TC/8;                      // weight beats per MLP tile
  localparam int NBB  = TL/2;                         // bias beats per MLP tile
  localparam int NBT  = NBW+NBB;                      // 12 at TL=TC=8
  localparam int NF1  = NBT*G1*NT;                    // fc1 internal beats/token
  localparam int NF2  = NBT*G0*NT2;                   // fc2 internal beats/token
  localparam int NWT  = NF1+NF2;                      // MLP weight beats/token
  localparam int NWT_W = $clog2(NWT+1);
  localparam int WG_W = $clog2(((G1>G0)?G1:G0)+1);
  localparam int WT_W = $clog2(((NT>NT2)?NT:NT2)+1);
  localparam int QKV_PW = RW+1+W;
  localparam int PRJ_PW = AV_W+1+W;
  localparam int MAC_W  = (PRJ_PW > QKV_PW) ? PRJ_PW : QKV_PW;
  localparam int TS_W   = $clog2(T+5);   // tsub 0..T+4 (5-stage MAC pipeline drain)
  localparam int CAP_W  = $clog2(TOKENS*G+1);
  localparam int TOK_W  = $clog2(TOKENS+1);
  localparam int AI_W   = $clog2(TOKENS+4);
  localparam int BNKA_W = $clog2(TL+1);
  localparam int SB_HPT   = HEAD_DIM/TL;      // QKV groups per attention head
  localparam int SB_DEPTH = TOKENS*HEADS;     // entries per Q/K/V spike bank
  localparam int SB_AW    = $clog2(SB_DEPTH+1);
  localparam int SB_BW    = (SB_HPT > 1) ? $clog2(SB_HPT) : 1;
  localparam int S_IDLE=0, S_LDW=1, S_TOK=2, S_QKV=3, S_ATTN=4, S_PROJ=5, S_MLP=6, S_DONE=7;
  localparam int M_Q=0, M_K=1, M_V=2;
  localparam int MSG_W = 1;

  logic [2:0] st;
  // weight load
  logic [LD_W-1:0] ldcnt;
  logic [3:0] ldph;
  // token capture
  logic [CAP_W-1:0] capidx;
  // QKV + proj
  logic [TOK_W-1:0] ntok;
  logic [1:0] msel;
  logic [$clog2(G+1)-1:0] qg;
  logic [$clog2(NT+1)-1:0] qti;
  logic [TS_W-1:0] tsub;
  logic qlif;
  logic [TS_W-1:0] lifc;
  logic lifv;
  logic [$clog2(G+1)-1:0] pgrp;
  logic [$clog2(NT+1)-1:0] pti;
  logic plif;
  // attention
  logic [$clog2(HEADS+1)-1:0] ah;
  logic [AI_W-1:0] ai;
  logic [TOK_W-1:0] pj1, pj2, pj3;
  // spiking_attention_pe: o_data (popcount sum) is valid 3 cycles after the
  // i_valid cycle, but its o_valid carries one extra r_v pipeline stage (4
  // cycles), so o_valid lags o_data by one cycle. Keep a block-local data-valid
  // chain aligned to the 3-cycle data latency (att_dv3) and use that for avf.
  logic att_dv1, att_dv2, att_dv3;
  // MLP phase
  logic [1:0] mph;                     // 0 feed input beats, 1 run/collect
  logic [$clog2(G+1)-1:0] mfb;
  logic [1:0] mfs;                     // 0 wait psum, 1 LIF, 2 output, 3 wait mlp done
  logic [$clog2(G+1)-1:0] gcount;
  logic mlp_start;
  logic we_start;
  // MLP weight-stream engine
  logic mw_act;
  logic [NWT_W-1:0] mw_beat;
  logic mw_seg;
  logic [WG_W-1:0] mw_g;
  logic [WT_W-1:0] mw_t;
  logic [3:0] mw_b;
  logic [NWT_W-1:0] mw_qidx;
  logic mw_qvalid;

  // ---- data storage ----
  (* ram_style = "block" *) logic [TL*T*RW-1:0] tokram [TOKENS*G];  // residual beats
  logic [TL*T*RW-1:0] tokq;
  logic [CAP_W-1:0] tok_ra;
  logic [63:0] wrd [4][TL];                    // Q/K/V/proj bank read data
  logic [63:0] fc1q, fc2q;                     // fc1/fc2 weight RAM read data
  logic [FCAW-1:0] f1ra, f2ra;
  logic [BANK_AW-1:0] wram_ra;
  logic signed [PSUM_SLICE-1:0] psum [TL][T];
  logic signed [PSUM_SLICE-1:0] projf [TL][T];  // current output group scratch only (was E x T)
  logic signed [PSUM_SLICE-1:0] avf [E][T];
  // Q/K/V spike tables: 3 matrices x SB_HPT synchronous-read banks. Bank =
  // group-inside-head (qg % SB_HPT), entry = TL lanes x T bits (one LIF-group
  // result), depth = TOKENS*HEADS. This replaces the old 3 x TOKENS x E x T
  // bit-array registers whose dynamic-index writes/reads exploded area at the
  // target size (synth evidence: two attempts ballooned in block_top).
  logic [TL*T-1:0] sbrow [3][SB_HPT];           // registered read rows per matrix
  logic [SB_AW-1:0] sb_q_ra, sb_k_ra, sb_v_ra;  // combinational read addresses
  logic sb_we;
  logic [2:0] sb_wr_mat;
  logic [SB_BW-1:0] sb_wr_bank;
  logic [SB_AW-1:0] sb_wr_addr;
  logic [RW-1:0] ares [E][T];                  // in + LIF(proj) residual
  logic [TL*T*RW-1:0] tokres;
  logic [TL*T*PSUM_SLICE-1:0] fc2psum;
  logic signed [PSUM_SLICE-1:0] bq [E];
  logic signed [PSUM_SLICE-1:0] bk [E];
  logic signed [PSUM_SLICE-1:0] bv [E];
  logic signed [PSUM_SLICE-1:0] bp [E];
  logic signed [PSUM_SLICE-1:0] bf1 [HIDDEN];
  logic signed [PSUM_SLICE-1:0] bf2 [E];

  // ---- LIF bank shared by QKV / proj / fc2 phases (T+1 deterministic) ----
  logic [TL-1:0] lif_rdy;
  logic [TL*T-1:0] lifspk;
  logic signed [T*PSUM_SLICE-1:0] lifin [TL];
  logic signed [PSUM_SLICE-1:0] lif_thr;
  logic signed [PSUM_SLICE-1:0] mac_bias [TL];
  logic [$clog2(T+2)-1:0] tti;

  // ---- MAC combinational build ----
  logic signed [W-1:0] mac_w [TL][TC];
  logic signed [MAC_W-1:0] mul_x [TC];
  logic signed [MAC_W-1:0] mul_w [TL][TC];
  // 5-stage MAC pipeline: S1 operands -> S2 products -> S3 half sums ->
  // S4 total -> S5 32-bit accumulate writeback. Data registers need no reset;
  // only the valid chain is cleared so stale data can never write back.
  logic signed [MAC_W-1:0] xp_q [TC];
  logic signed [W-1:0]     wp_q [TL][TC];
  logic signed [MAC_W-1:0] pr_q [TL][TC];
  logic signed [31:0]      hf_q [TL][2];
  logic signed [31:0]      sm_q [TL];
  logic [3:0]              mpipe_v;
  logic [TS_W-1:0]         wb_tt;
  logic                    mac_iss;

  // ---- attention PE ----
  logic [HEAD_DIM*T-1:0] att_q, att_k;
  logic [T*SUM_W-1:0] att_out;
  logic att_outv;
  logic att_iv;

  // ---- MLP instance wires ----
  logic mlp_done, mlp_ov, mlp_inrdy, mlp_wready, mlp_wtaken;
  logic [TL*T*RW-1:0] mlp_ind;
  logic [TL*T*PSUM_SLICE-1:0] mlp_outd;
  logic mlp_iv, mlp_ordy;
  logic [63:0] mw_wdata;
  logic mw_wvalid;

  // ---- weight load write decode ----
  logic ld_we;
  logic [1:0] ld_mat;
  logic [BNKA_W-1:0] ld_bank;
  logic [BANK_AW-1:0] ld_addr;
  always_comb begin
    int base;
    int out_i;
    int in_off;
    base = int'(ldcnt) * 8;
    out_i = base / E;
    in_off = base % E;
    ld_mat = ldph[1:0];
    ld_bank = out_i % TL;
    ld_addr = (out_i / TL) * (E/TC) + (in_off / TC);
    ld_we = (st == S_LDW) && (ldph < 4) && (i_w_valid == 1'b1);
  end

  // 4 matrices x TL output-lane banks, synchronous read, no per-element reset.
  genvar gm, gb;
  generate
    for (gm = 0; gm < 4; gm = gm + 1) begin : g_wmat
      for (gb = 0; gb < TL; gb = gb + 1) begin : g_bank
        (* ram_style = "block" *) logic [63:0] mem [BANK_DEPTH];
        always_ff @(posedge s_clk) begin
          if (ld_we && (ld_mat == gm) && (ld_bank == gb))
            mem[ld_addr] <= i_w_data;
          wrd[gm][gb] <= mem[wram_ra];
        end
      end
    end
  endgenerate

  // fc1 / fc2 weight RAMs: one 64-bit synchronous-read RAM each, sequential
  // beat address from the tile stream (block reads them per MLP token run).
  (* ram_style = "block" *) logic [63:0] fc1mem [FCD];
  (* ram_style = "block" *) logic [63:0] fc2mem [FCD];
  always_ff @(posedge s_clk) begin
    if ((st == S_LDW) && (ldph == 8) && (i_w_valid == 1'b1))
      fc1mem[ldcnt] <= i_w_data;
    fc1q <= fc1mem[f1ra];
  end
  always_ff @(posedge s_clk) begin
    if ((st == S_LDW) && (ldph == 10) && (i_w_valid == 1'b1))
      fc2mem[ldcnt] <= i_w_data;
    fc2q <= fc2mem[f2ra];
  end

  // ---- token residual RAM (one 128-bit entry per token/group beat) ----
  always_comb begin
    if (st == S_QKV) begin
      tok_ra = ntok*G + qti;
    end else if (st == S_PROJ) begin
      tok_ra = ntok*G + pgrp;
    end else begin
      tok_ra = '0;
    end
  end
  always_ff @(posedge s_clk) begin
    if ((st == S_TOK) && (i_in_valid == 1'b1))
      tokram[capidx] <= i_in_data;
    tokq <= tokram[tok_ra];
  end
  assign tti = mac_iss ? (tsub - 1'b1) : '0;

  // ---- QKV/proj weight bank read address (common to 8 banks) ----
  always_comb begin
    if (st == S_QKV)
      wram_ra = qg*NT + qti;
    else if (st == S_PROJ)
      wram_ra = pgrp*NT + pti;
    else
      wram_ra = '0;
  end

  // ---- MAC 5-stage pipeline (200MHz timing fix) ---------------------------
  // Issue one time slice per cycle for T cycles, then drain the 4 register
  // stages. wb_tt = tsub - 5 identifies the slice being written back. S1
  // registers the operand mux; S2 registers the 64 products; S3/S4 register
  // the balanced sum; S5 does base+sum in the FSM.
  assign wb_tt   = tsub - 5;
  assign mac_iss = ((st == S_QKV) && !qlif && (tsub >= 1) && (tsub <= T)) ||
                   ((st == S_PROJ) && !plif && (tsub >= 1) && (tsub <= T));
  always_comb begin
    logic [63:0] w64;
    for (int k = 0; k < TC; k = k + 1) begin
      if (st == S_PROJ)
        mul_x[k] = mac_iss ? {{(MAC_W-AV_W){1'b0}}, avf[pti*TC+k][tti][AV_W-1:0]} : '0;
      else
        mul_x[k] = mac_iss ? {{(MAC_W-RW){1'b0}}, tokq[(k*T+tti+1)*RW-1 -: RW]} : '0;
    end
    for (int a = 0; a < TL; a = a + 1) begin
      if (st == S_PROJ)
        mac_bias[a] = bp[pgrp*TL+a];
      else if (msel == M_Q)
        mac_bias[a] = bq[qg*TL+a];
      else if (msel == M_K)
        mac_bias[a] = bk[qg*TL+a];
      else
        mac_bias[a] = bv[qg*TL+a];
      for (int k = 0; k < TC; k = k + 1) begin
        w64 = (st == S_PROJ) ? wrd[3][a] : wrd[msel][a];
        mac_w[a][k] = $signed(w64[(k+1)*8-1 -: 8]);
        mul_w[a][k] = mac_iss ? {{(MAC_W-W){mac_w[a][k][W-1]}}, mac_w[a][k]} : '0;
      end
    end
  end
  always_ff @(posedge s_clk) begin
    if (s_rst) begin
      mpipe_v <= '0;
    end else begin
      if (mac_iss) begin
        for (int k = 0; k < TC; k = k + 1) xp_q[k] <= mul_x[k];
        for (int a = 0; a < TL; a = a + 1)
          for (int k = 0; k < TC; k = k + 1) wp_q[a][k] <= mac_w[a][k];
      end
      for (int a = 0; a < TL; a = a + 1)
        for (int k = 0; k < TC; k = k + 1)
          pr_q[a][k] <= xp_q[k] * $signed({{(MAC_W-W){wp_q[a][k][W-1]}}, wp_q[a][k]});
      for (int a = 0; a < TL; a = a + 1) begin
        hf_q[a][0] <= (pr_q[a][0] + pr_q[a][1]) + (pr_q[a][2] + pr_q[a][3]);
        hf_q[a][1] <= (pr_q[a][4] + pr_q[a][5]) + (pr_q[a][6] + pr_q[a][7]);
      end
      for (int a = 0; a < TL; a = a + 1)
        sm_q[a] <= hf_q[a][0] + hf_q[a][1];
      mpipe_v <= {mpipe_v[2:0], mac_iss};
    end
  end

  // ---- shared LIF input / threshold mux ----
  always_comb begin
    for (int a = 0; a < TL; a = a + 1)
      for (int tt = 0; tt < T; tt = tt + 1) begin
        if (st == S_PROJ)
          lifin[a][(tt+1)*PSUM_SLICE-1 -: PSUM_SLICE] = projf[a][tt];
        else if (st == S_MLP)
          lifin[a][(tt+1)*PSUM_SLICE-1 -: PSUM_SLICE] = fc2psum[(a*T+tt+1)*PSUM_SLICE-1 -: PSUM_SLICE];
        else
          lifin[a][(tt+1)*PSUM_SLICE-1 -: PSUM_SLICE] = psum[a][tt];
      end
    if (st == S_PROJ)
      lif_thr = i_thr_proj;
    else if (st == S_MLP)
      lif_thr = i_thr_fc2;
    else if (msel == M_Q)
      lif_thr = i_thr_q;
    else if (msel == M_K)
      lif_thr = i_thr_k;
    else
      lif_thr = i_thr_v;
  end

  genvar ll;
  generate
    for (ll = 0; ll < TL; ll = ll + 1) begin : g_lif
      spiking_lif_group #(.T(T), .BITS(PSUM_SLICE)) u_l (
        .s_clk(s_clk), .s_rst(s_rst), .i_lif_thrd(lif_thr),
        .i_PsumValid(lifv), .i_PsumData(lifin[ll]),
        .o_spikes_out(lifspk[(ll+1)*T-1 -: T]),
        .o_spikes_valid(lif_rdy[ll])
      );
    end
  endgenerate

  // ---- Q/K/V spike banks: 3 matrices x SB_HPT sync-read block RAMs ----
  // One bank write per QKV LIF group (selected by qg%SB_HPT); read rows feed the
  // attention issue pipeline. No per-element reset (all entries are written by
  // QKV before the first attention read).
  always_comb begin
    sb_we      = (st == S_QKV) && qlif && (lifc == T+1);
    sb_wr_mat  = {1'b0, msel};
    sb_wr_bank = qg % SB_HPT;
    sb_wr_addr = ntok*HEADS + (qg / SB_HPT);
    sb_q_ra    = ntok*HEADS + ah;
    sb_k_ra    = (ai < TOKENS) ? (ai*HEADS + ah) : '0;
    sb_v_ra    = ((ai >= 3) && (ai <= TOKENS+2)) ? ((ai - 3)*HEADS + ah) : '0;
  end
  genvar gsm, gsb;
  generate
    for (gsm = 0; gsm < 3; gsm = gsm + 1) begin : g_sbm
      for (gsb = 0; gsb < SB_HPT; gsb = gsb + 1) begin : g_sb
        (* ram_style = "block" *) logic [TL*T-1:0] mem [SB_DEPTH];
        always_ff @(posedge s_clk) begin
          if (sb_we && (sb_wr_mat == gsm) && (sb_wr_bank == gsb))
            mem[sb_wr_addr] <= lifspk;
          sbrow[gsm][gsb] <= mem[(gsm == 0) ? sb_q_ra
                                : (gsm == 1) ? sb_k_ra : sb_v_ra];
        end
      end
    end
  endgenerate

  // ---- ONE shared QK popcount PE, reused sequentially over heads ----
  // Synchronous-read spike banks: the Q row is read at head start (available
  // from ai=1), the K row for source token k is read at cycle k (available at
  // its issue cycle k+1), the V row is read 3 cycles before its accumulation
  // cycle. PE data latency stays 3 cycles, so the att_dv1..3 chain and pj3
  // issue-index alignment below are unchanged; only the head cycle count grows
  // by two (exit at ai == TOKENS+3).
  assign att_iv = (st == S_ATTN) && (ai >= 1) && (ai <= TOKENS);
  always_comb begin
    for (int a = 0; a < HEAD_DIM; a = a + 1)
      for (int tt = 0; tt < T; tt = tt + 1) begin
        if ((st == S_ATTN) && (ai >= 1)) begin
          att_q[a*T+tt] = sbrow[0][a/TL][(a%TL)*T+tt];
          att_k[a*T+tt] = sbrow[1][a/TL][(a%TL)*T+tt];
        end else begin
          att_q[a*T+tt] = 1'b0;
          att_k[a*T+tt] = 1'b0;
        end
      end
  end
  spiking_attention_pe #(.HEAD_DIM(HEAD_DIM), .T(T), .SUM_W(SUM_W)) u_att (
    .s_clk(s_clk), .s_rst(s_rst), .i_valid(att_iv),
    .i_q(att_q), .i_k(att_k), .o_data(att_out), .o_valid(att_outv)
  );

  // ---- MLP controller (fc1 LIF inside; full fc2 psum out) ----
  assign mlp_iv   = (st == S_MLP) && (mph == 0);
  assign mlp_ordy = (st == S_MLP) && (mph == 1) && (mfs == 0);
  always_comb begin
    for (int k = 0; k < TL; k = k + 1)
      for (int tt = 0; tt < T; tt = tt + 1)
        mlp_ind[(k*T+tt+1)*RW-1 -: RW] = ares[mfb*TL+k][tt];
  end
  spiking_mlp_controller #(.C(E), .HIDDEN(HIDDEN), .T(T), .RW(RW), .W(W),
    .PSUM_SLICE(PSUM_SLICE), .TL(TL), .TC(TC)) u_mlp (
    .s_clk(s_clk), .s_rst(s_rst), .i_start(mlp_start), .o_done(mlp_done),
    .i_in_valid(mlp_iv), .i_in_data(mlp_ind), .o_in_ready(mlp_inrdy),
    .o_out_valid(mlp_ov), .i_out_ready(mlp_ordy), .o_out_data(mlp_outd),
    .i_w_data(mw_wdata), .i_w_valid(mw_wvalid), .o_w_ready(mlp_wready),
    .o_w_taken(mlp_wtaken), .i_thr_hidden(i_thr_fc1)
  );

  // ---- MLP weight-stream engine: internal tile order -> fc1/fc2 RAM beats ---
  logic mw_cur_bias;
  logic mw_nseg, mw_fseg, mw_fisw;
  logic [WG_W-1:0] mw_ng, mw_fg;
  logic [WT_W-1:0] mw_nt, mw_ft;
  logic [3:0] mw_nb, mw_fb;
  logic [NWT_W-1:0] mw_fidx;
  logic mw_fetch_valid;
  function automatic logic signed [31:0] sext32(input logic signed [PSUM_SLICE-1:0] v);
    sext32 = v;
  endfunction
  always_comb begin
    int a0;
    int a1;
    mw_cur_bias = (mw_b >= NBW);
    // counters after a taken beat
    mw_nseg = mw_seg; mw_ng = mw_g; mw_nt = mw_t; mw_nb = mw_b;
    if (mw_b != NBT-1) begin
      mw_nb = mw_b + 1'b1;
    end else begin
      mw_nb = '0;
      if (!mw_seg) begin
        if (mw_t != NT-1) begin
          mw_nt = mw_t + 1'b1;
        end else begin
          mw_nt = '0;
          if (mw_g != G1-1) mw_ng = mw_g + 1'b1;
          else begin mw_ng = '0; mw_nseg = 1'b1; end
        end
      end else begin
        if (mw_t != NT2-1) begin
          mw_nt = mw_t + 1'b1;
        end else begin
          mw_nt = '0;
          if (mw_g != G0-1) mw_ng = mw_g + 1'b1;
        end
      end
    end
    // fetch target (next beat when the current one is taken, else current)
    if (mlp_wtaken) begin
      mw_fseg = mw_nseg; mw_fg = mw_ng; mw_ft = mw_nt; mw_fb = mw_nb;
    end else begin
      mw_fseg = mw_seg; mw_fg = mw_g; mw_ft = mw_t; mw_fb = mw_b;
    end
    mw_fisw = (mw_fb < NBW);
    mw_fidx = mw_beat + (mlp_wtaken ? 1'b1 : 1'b0);
    mw_fetch_valid = mw_act && mw_fisw && (mw_fidx < NWT);
    f1ra = '0;
    f2ra = '0;
    if (mw_fetch_valid) begin
      if (!mw_fseg)
        f1ra = (mw_fg*TL + mw_fb) * (E/8) + mw_ft;
      else
        f2ra = (mw_fg*TL + mw_fb) * (HIDDEN/8) + mw_ft;
    end
    // weight/bias payload for the current beat
    mw_wdata = 64'd0;
    if (mw_cur_bias) begin
      a0 = (int'(mw_b) - NBW) * 2;
      a1 = a0 + 1;
      if (!mw_seg)
        mw_wdata = {sext32(bf1[mw_g*TL+a1]), sext32(bf1[mw_g*TL+a0])};
      else
        mw_wdata = {sext32(bf2[mw_g*TL+a1]), sext32(bf2[mw_g*TL+a0])};
    end else if (mw_qvalid && (mw_qidx == mw_beat)) begin
      mw_wdata = mw_seg ? fc2q : fc1q;
    end
    mw_wvalid = mw_act && (mw_cur_bias || (mw_qvalid && (mw_qidx == mw_beat)));
  end

  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin
      mw_act <= 1'b0; mw_beat <= '0; mw_seg <= 1'b0; mw_g <= '0; mw_t <= '0;
      mw_b <= '0; mw_qidx <= '0; mw_qvalid <= 1'b0;
    end else if (we_start) begin
      mw_act <= 1'b1; mw_beat <= '0; mw_seg <= 1'b0; mw_g <= '0; mw_t <= '0;
      mw_b <= '0; mw_qidx <= '0; mw_qvalid <= 1'b0;
    end else begin
      if (mlp_wtaken) begin
        mw_beat <= mw_beat + 1'b1;
        mw_seg <= mw_nseg; mw_g <= mw_ng; mw_t <= mw_nt; mw_b <= mw_nb;
        if (mw_beat == NWT-1) mw_act <= 1'b0;
      end
      if (mw_fetch_valid) begin
        mw_qidx <= mw_fidx;
        mw_qvalid <= 1'b1;
      end else begin
        mw_qvalid <= 1'b0;
      end
    end
  end

  // ---- main FSM ----
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin
      st <= S_IDLE; ldcnt <= '0; ldph <= '0; capidx <= '0; ntok <= '0; msel <= '0;
      qg <= '0; qti <= '0; tsub <= '0; qlif <= 1'b0; lifc <= '0; lifv <= 1'b0;
      pgrp <= '0; pti <= '0; plif <= 1'b0; ah <= '0; ai <= '0;
      pj1 <= '0; pj2 <= '0; pj3 <= '0; att_dv1 <= 1'b0; att_dv2 <= 1'b0; att_dv3 <= 1'b0;
      mph <= '0; mfb <= '0; mfs <= '0;
      gcount <= '0; mlp_start <= 1'b0; we_start <= 1'b0;
      o_done <= 1'b0; o_out_valid <= 1'b0; o_out_data <= '0; o_out_last <= 1'b0;
      o_w_ready <= 1'b0; o_in_ready <= 1'b0;
    end else begin
      lifv <= 1'b0;
      mlp_start <= 1'b0;
      we_start <= 1'b0;
      o_w_ready <= 1'b0;
      o_in_ready <= 1'b0;
      case (st)
        S_IDLE: begin
          o_done <= 1'b0;
          if (i_start) begin
            ldcnt <= '0; ldph <= '0; capidx <= '0; ntok <= '0; msel <= '0;
            qg <= '0; qti <= '0; tsub <= '0; qlif <= 1'b0; lifc <= '0;
            pgrp <= '0; pti <= '0; plif <= 1'b0; ah <= '0; ai <= '0;
            mph <= '0; mfb <= '0; mfs <= '0; gcount <= '0;
            o_out_valid <= 1'b0; o_out_last <= 1'b0;
            o_w_ready <= 1'b1;
            st <= S_LDW;
          end
        end
        S_LDW: begin
          o_w_ready <= 1'b1;
          if (i_w_valid) begin
            if (ldph < 4) begin
              if (ldcnt == EEB-1) begin ldcnt <= '0; ldph <= ldph + 1'b1; end
              else ldcnt <= ldcnt + 1'b1;
            end else if (ldph < 8) begin
              if (ldph == 4) begin
                bq[ldcnt*2] <= $signed(i_w_data[PSUM_SLICE-1:0]);
                bq[ldcnt*2+1] <= $signed(i_w_data[32+PSUM_SLICE-1:32]);
              end else if (ldph == 5) begin
                bk[ldcnt*2] <= $signed(i_w_data[PSUM_SLICE-1:0]);
                bk[ldcnt*2+1] <= $signed(i_w_data[32+PSUM_SLICE-1:32]);
              end else if (ldph == 6) begin
                bv[ldcnt*2] <= $signed(i_w_data[PSUM_SLICE-1:0]);
                bv[ldcnt*2+1] <= $signed(i_w_data[32+PSUM_SLICE-1:32]);
              end else begin
                bp[ldcnt*2] <= $signed(i_w_data[PSUM_SLICE-1:0]);
                bp[ldcnt*2+1] <= $signed(i_w_data[32+PSUM_SLICE-1:32]);
              end
              if (ldcnt == E/2-1) begin ldcnt <= '0; ldph <= ldph + 1'b1; end
              else ldcnt <= ldcnt + 1'b1;
            end else if (ldph == 8) begin
              if (ldcnt == FCD-1) begin ldcnt <= '0; ldph <= 9; end
              else ldcnt <= ldcnt + 1'b1;
            end else if (ldph == 9) begin
              bf1[ldcnt*2] <= $signed(i_w_data[PSUM_SLICE-1:0]);
              bf1[ldcnt*2+1] <= $signed(i_w_data[32+PSUM_SLICE-1:32]);
              if (ldcnt == HIDDEN/2-1) begin ldcnt <= '0; ldph <= 10; end
              else ldcnt <= ldcnt + 1'b1;
            end else if (ldph == 10) begin
              if (ldcnt == FCD-1) begin ldcnt <= '0; ldph <= 11; end
              else ldcnt <= ldcnt + 1'b1;
            end else begin
              bf2[ldcnt*2] <= $signed(i_w_data[PSUM_SLICE-1:0]);
              bf2[ldcnt*2+1] <= $signed(i_w_data[32+PSUM_SLICE-1:32]);
              if (ldcnt == E/2-1) begin
                ldcnt <= '0; ldph <= '0; o_w_ready <= 1'b0; o_in_ready <= 1'b1;
                capidx <= '0; st <= S_TOK;
              end else ldcnt <= ldcnt + 1'b1;
            end
          end
        end
        S_TOK: begin
          o_in_ready <= 1'b1;
          if (i_in_valid) begin
            if (capidx == TOKENS*G-1) begin
              capidx <= '0; o_in_ready <= 1'b0;
              ntok <= '0; msel <= '0; qg <= '0; qti <= '0; tsub <= '0;
              qlif <= 1'b0; lifc <= '0; st <= S_QKV;
            end else capidx <= capidx + 1'b1;
          end
        end
        S_QKV: begin
          if (qlif) begin
            lifc <= lifc + 1'b1;
            if (lifc == T+1) begin
              lifc <= '0; qlif <= 1'b0;   // spike banks written via sb_we/sb_wr_*
              qti <= '0; tsub <= '0;
              if (qg == G-1) begin
                qg <= '0;
                if (msel == M_V) begin
                  msel <= '0;
                  if (ntok == TOKENS-1) begin
                    ntok <= '0; ah <= '0; ai <= '0; pj1 <= '0; pj2 <= '0; pj3 <= '0;
                    att_dv1 <= 1'b0; att_dv2 <= 1'b0; att_dv3 <= 1'b0;
                    st <= S_ATTN;
                  end else ntok <= ntok + 1'b1;
                end else msel <= msel + 1'b1;
              end else qg <= qg + 1'b1;
            end
          end else if (tsub == 0) begin
            tsub <= 1;                        // synchronous RAM fetch cycle
          end else begin
            if (mpipe_v[3]) begin             // S5: write back one time slice
              for (int a = 0; a < TL; a = a + 1) begin
                logic signed [31:0] base;
                base = (qti == 0) ? mac_bias[a] : psum[a][wb_tt];
                psum[a][wb_tt] <= base + sm_q[a];
              end
            end
            if (tsub == T+4) begin            // all T slices written back
              tsub <= '0;
              if (qti == NT-1) begin
                qlif <= 1'b1; lifv <= 1'b1; lifc <= '0;
              end else qti <= qti + 1'b1;
            end else tsub <= tsub + 1'b1;
          end
        end
        S_ATTN: begin
          if ((ai == 0) && (ah == 0)) begin
            for (int c = 0; c < E; c = c + 1)
              for (int tt = 0; tt < T; tt = tt + 1) avf[c][tt] <= '0;
          end
          att_dv1 <= att_iv;
          att_dv2 <= att_dv1;
          att_dv3 <= att_dv2;
          if (att_iv) pj1 <= ai - 1'b1;
          pj2 <= pj1;
          pj3 <= pj2;
          if (att_dv3) begin
            for (int a = 0; a < HEAD_DIM; a = a + 1)
              for (int tt = 0; tt < T; tt = tt + 1)
                if (sbrow[2][a/TL][(a%TL)*T+tt])
                  avf[ah*HEAD_DIM+a][tt] <= avf[ah*HEAD_DIM+a][tt]
                      + $signed({1'b0, att_out[(tt+1)*SUM_W-1 -: SUM_W]});
          end
          if (ai == TOKENS+3) begin
            ai <= '0;
            if (ah == HEADS-1) begin
              ah <= '0; pgrp <= '0; pti <= '0; tsub <= '0; plif <= 1'b0; lifc <= '0;
              st <= S_PROJ;
            end else ah <= ah + 1'b1;
          end else ai <= ai + 1'b1;
        end
        S_PROJ: begin
          if (plif) begin
            lifc <= lifc + 1'b1;
            if (lifc == 1) tokres <= tokq;
            if (lifc == T+1) begin
              lifc <= '0; plif <= 1'b0;
              for (int a = 0; a < TL; a = a + 1)
                for (int tt = 0; tt < T; tt = tt + 1)
                  ares[pgrp*TL+a][tt] <= tokres[(a*T+tt+1)*RW-1 -: RW] + lifspk[a*T+tt];
              pti <= '0; tsub <= '0;
              if (pgrp == G-1) begin
                pgrp <= '0; mph <= '0; mfb <= '0; mfs <= '0; gcount <= '0;
                mlp_start <= 1'b1; we_start <= 1'b1;
                st <= S_MLP;
              end else pgrp <= pgrp + 1'b1;
            end
          end else if (tsub == 0) begin
            tsub <= 1;                        // synchronous RAM fetch cycle
          end else begin
            if (mpipe_v[3]) begin             // S5: write back one time slice
              for (int a = 0; a < TL; a = a + 1) begin
                logic signed [31:0] base;
                base = (pti == 0) ? mac_bias[a] : projf[a][wb_tt];
                projf[a][wb_tt] <= base + sm_q[a];
              end
            end
            if (tsub == T+4) begin            // all T slices written back
              tsub <= '0;
              if (pti == NT-1) begin
                plif <= 1'b1; lifv <= 1'b1; lifc <= '0;
              end else pti <= pti + 1'b1;
            end else tsub <= tsub + 1'b1;
          end
        end
        S_MLP: begin
          if (mph == 0) begin
            if (mlp_inrdy) begin
              if (mfb == G-1) begin mfb <= '0; mph <= 1; mfs <= '0; end
              else mfb <= mfb + 1'b1;
            end
          end else begin
            case (mfs)
              0: begin
                if (mlp_ov) begin
                  fc2psum <= mlp_outd;
                  lifv <= 1'b1; lifc <= '0;
                  mfs <= 1;
                end
              end
              1: begin
                lifc <= lifc + 1'b1;
                if (lifc == T+1) begin
                  for (int k = 0; k < TL; k = k + 1)
                    for (int tt = 0; tt < T; tt = tt + 1)
                      o_out_data[(k*T+tt+1)*RW-1 -: RW] <=
                          ares[gcount*TL+k][tt] + lifspk[k*T+tt];
                  o_out_valid <= 1'b1;
                  o_out_last <= (ntok == TOKENS-1) && (gcount == G-1);
                  mfs <= 2;
                end
              end
              2: begin
                if (i_out_ready) begin
                  o_out_valid <= 1'b0;
                  if (gcount == G-1) begin gcount <= '0; mfs <= 3; end
                  else begin gcount <= gcount + 1'b1; mfs <= 0; end
                end
              end
              default: begin
                if (mlp_done) begin
                  mph <= '0; mfb <= '0; mfs <= '0; gcount <= '0;
                  if (ntok == TOKENS-1) begin
                    o_done <= 1'b1; st <= S_DONE;
                  end else begin
                    ntok <= ntok + 1'b1; ah <= '0; ai <= '0;
                    pj1 <= '0; pj2 <= '0; pj3 <= '0;
                    att_dv1 <= 1'b0; att_dv2 <= 1'b0; att_dv3 <= 1'b0;
                    st <= S_ATTN;
                  end
                end
              end
            endcase
          end
        end
        S_DONE: begin
          o_done <= 1'b1;
          if (i_start) begin
            o_done <= 1'b0;
            ldcnt <= '0; ldph <= '0; capidx <= '0; ntok <= '0; msel <= '0;
            qg <= '0; qti <= '0; tsub <= '0; qlif <= 1'b0; lifc <= '0;
            pgrp <= '0; pti <= '0; plif <= 1'b0; ah <= '0; ai <= '0;
            mph <= '0; mfb <= '0; mfs <= '0; gcount <= '0;
            o_out_valid <= 1'b0; o_out_last <= 1'b0;
            o_w_ready <= 1'b1;
            st <= S_LDW;
          end
        end
        default: st <= S_IDLE;
      endcase
    end
  end

`ifndef SYNTHESIS
  initial begin
    if (TC != 8)
      $error("spiking_block_top: TC must be 8 (64-bit weight beat = 8 input taps)");
    if (TL != TC)
      $error("spiking_block_top: TL must equal TC for the token-beat packing");
    if (HEADS*HEAD_DIM != E)
      $error("spiking_block_top: HEADS*HEAD_DIM must equal E");
    if ((HEAD_DIM % TL) != 0)
      $error("spiking_block_top: HEAD_DIM must be a multiple of TL");
    if (SB_HPT < 2)
      $error("spiking_block_top: HEAD_DIM/TL must be >= 2");
    if ((E % TL) != 0 || (E % TC) != 0 || (HIDDEN % TL) != 0 || (HIDDEN % TC) != 0)
      $error("spiking_block_top: E/HIDDEN must be multiples of TL/TC");
    if ((32 - PSUM_SLICE) < 0)
      $error("spiking_block_top: PSUM_SLICE must be <= 32");
  end
`endif

endmodule
