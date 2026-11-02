// spiking_embed_top - Spikformer NLP token+position embedding spike frontend.
//
// Storage (inferred synchronous-read RAM, 64-bit words, NO per-element reset):
//   etab: V rows x E signed int8 = V*(E/8) words, row-major.
//   ptab: N rows x E signed int8 = N*(E/8) words, row-major.
// Table load stream: V*E/8 etab beats, then N*E/8 ptab beats. Each beat is one
// 64-bit word = 8 consecutive channels of one row.
// Encode: for each pos (N) and channel group grp (E/LANES), read
//   etab[tok_id[pos]*(E/8) + grp] and ptab[pos*(E/8) + grp]
// with one-cycle synchronous read latency; the next cycle splits the 8 signed
// bytes, adds etab+ptab per lane (signed, sign-extended to ACC_W), and pulses a
// bank of LANES spiking_lif_group cells (T identical rate-coded timesteps).
// Output beat: LANES lanes x T timeslices, lane-major
// {(lane*T+tt+1)*RW-1 -: RW} with RW hard-wired 1 bit here (first block input
// residual is 0/1). o_beat_last marks the final (token,group) beat.
//
// Handshake (frozen contract): a beat is consumed iff valid && ready in the
// SAME cycle. Table/token/beat payloads are held stable while stalled; no
// ready-ahead-of-valid implicit contract. i_start restarts a whole sample with
// no global reset (tables are reloaded; large RAMs are never reset).
// LIF completion is sampled by a deterministic counter (T+1 cycles after the
// valid pulse); no polling/timeout is used as data-ready.
module spiking_embed_top #(
  parameter int V     = 4000,
  parameter int N     = 32,
  parameter int E     = 128,
  parameter int T     = 4,
  parameter int LANES = 8,
  parameter int ACC_W = 32
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic i_start,
  input  logic [63:0] i_tab_data,
  input  logic i_tab_valid,
  output logic o_tab_ready,
  output logic o_ld_done,
  input  logic [15:0] i_tok_id,
  input  logic i_tok_valid,
  output logic o_tok_ready,
  input  logic signed [31:0] i_thr_e,
  output logic o_beat_valid,
  input  logic i_beat_ready,
  output logic [LANES*T-1:0] o_beat_data,
  output logic o_beat_last,
  output logic o_done
);
  localparam int AW    = E/8;       // 64-bit words per table row
  localparam int G     = E/LANES;   // output groups (channel beats) per row
  localparam int ETW   = V*AW;      // etab words
  localparam int PTW   = N*AW;      // ptab words
  localparam int S_IDLE=0, S_LDT=1, S_TOK=2, S_RD=3, S_DATA=4,
                 S_LIFP=5, S_LIFW=6, S_BEAT=7, S_DONE=8;

  logic [3:0] st;

  // ---- inferred synchronous-read table RAMs (never element-reset) ----
  logic [63:0] etab [ETW];
  logic [63:0] ptab [PTW];
  logic [$clog2(ETW+1)-1:0] ewr;
  logic [$clog2(PTW+1)-1:0] pwr;
  logic ld_ptab;
  logic rd_en;
  logic [$clog2(ETW+1)-1:0] etab_raddr;
  logic [$clog2(PTW+1)-1:0] ptab_raddr;
  logic [63:0] etab_q, ptab_q;

  // ---- token capture / encode position ----
  logic [15:0] tokbuf [N];
  logic [$clog2(N+1)-1:0] tcnt;
  logic [$clog2(N)-1:0] pos;
  logic [$clog2(G)-1:0] grp;
  logic [$clog2(T+4)-1:0] waitc;

  // ---- embedding LIF bank ----
  logic signed [ACC_W-1:0] racc [LANES];
  logic signed [T*ACC_W-1:0] lifps [LANES];
  logic [LANES*T-1:0] lifspk;
  logic [LANES-1:0] lif_rdy;
  wire lifv;
  wire signed [ACC_W-1:0] thr_e_i;
  logic [LANES*T-1:0] beat_hold;
  logic last_hold;

  integer a;    // sequential-block loop index
  integer ai;   // combinational-block loop index
  genvar ll;

  assign thr_e_i = i_thr_e[ACC_W-1:0];
  assign lifv    = (st == S_LIFP);

  // table load write port (synchronous write, no reset on the array)
  always_ff @(posedge s_clk) begin
    if (i_tab_valid && o_tab_ready) begin
      if (!ld_ptab) etab[ewr] <= i_tab_data;
      else          ptab[pwr] <= i_tab_data;
    end
  end

  // synchronous read port, one-cycle latency
  always_ff @(posedge s_clk) begin
    if (rd_en) begin
      etab_q <= etab[etab_raddr];
      ptab_q <= ptab[ptab_raddr];
    end
  end

  // output / handshake side (payload registered in beat_hold)
  always_comb begin
    o_tab_ready  = (st == S_LDT);
    o_tok_ready  = (st == S_TOK);
    // "load done" level: tables are loaded and the frontend is waiting for the
    // token burst. It is a function of state so it cannot appear stale at the
    // start of a new sample (no global reset between samples).
    o_ld_done    = (st == S_TOK);
    o_beat_valid = (st == S_BEAT);
    o_beat_data  = beat_hold;
    o_beat_last  = (st == S_BEAT) && last_hold;
  end

  // read addresses: token_id*(E/8)+grp for etab, pos*(E/8)+grp for ptab
  always_comb begin
    rd_en      = (st == S_RD);
    etab_raddr = '0;
    ptab_raddr = '0;
    if (st == S_RD) begin
      etab_raddr = tokbuf[pos] * AW + grp;
      ptab_raddr = pos * AW + grp;
    end
  end

  // rate-coded delta: same embedding value on every one of the T timesteps
  always_comb begin
    for (ai = 0; ai < LANES; ai = ai + 1) lifps[ai] = {T{racc[ai]}};
  end

  generate
    for (ll = 0; ll < LANES; ll = ll + 1) begin : g_lif
      spiking_lif_group #(.T(T), .BITS(ACC_W)) u_l (
        .s_clk(s_clk), .s_rst(s_rst),
        .i_lif_thrd(thr_e_i),
        .i_PsumValid(lifv),
        .i_PsumData(lifps[ll]),
        .o_spikes_out(lifspk[(ll+1)*T-1 -: T]),
        .o_spikes_valid(lif_rdy[ll])
      );
    end
  endgenerate

  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin
      st <= S_IDLE;
      ewr <= 0; pwr <= 0; ld_ptab <= 1'b0;
      tcnt <= 0; pos <= 0; grp <= 0; waitc <= 0;
      beat_hold <= 0; last_hold <= 1'b0; o_done <= 1'b0;
    end else if (i_start) begin
      // sample restart without global reset: reload tables, recapture tokens
      st <= S_LDT;
      ewr <= 0; pwr <= 0; ld_ptab <= 1'b0;
      tcnt <= 0; pos <= 0; grp <= 0; waitc <= 0;
      beat_hold <= 0; last_hold <= 1'b0; o_done <= 1'b0;
    end else case (st)
      S_IDLE: begin
        // wait for i_start
      end
      S_LDT: begin
        if (i_tab_valid && o_tab_ready) begin
          if (!ld_ptab) begin
            if (ewr == ETW-1) begin ld_ptab <= 1'b1; pwr <= 0; end
            else ewr <= ewr + 1'b1;
          end else begin
            if (pwr == PTW-1) begin st <= S_TOK; tcnt <= 0; end
            else pwr <= pwr + 1'b1;
          end
        end
      end
      S_TOK: begin
        if (i_tok_valid && o_tok_ready) begin
          tokbuf[tcnt] <= i_tok_id;
          if (tcnt == N-1) begin st <= S_RD; pos <= 0; grp <= 0; end
          else tcnt <= tcnt + 1'b1;
        end
      end
      S_RD: begin
        st <= S_DATA;
      end
      S_DATA: begin
        // one-cycle synchronous read result is valid in this cycle
        for (a = 0; a < LANES; a = a + 1)
          racc[a] <= $signed(etab_q[(a+1)*8-1 -: 8]) + $signed(ptab_q[(a+1)*8-1 -: 8]);
        st <= S_LIFP;
      end
      S_LIFP: begin
        // lifv is combinational in this state; LIF group samples the data now
        st <= S_LIFW;
        waitc <= 0;
      end
      S_LIFW: begin
        // deterministic T+1-cycle LIF latency: valid pulse in S_LIFP (cycle C),
        // o_spikes_out/o_spikes_valid are registered at cycle C+T+1.
        if (waitc == T) begin
          beat_hold <= lifspk;
          last_hold <= (pos == N-1) && (grp == G-1);
          if (grp == G-1) begin
            grp <= 0;
            if (pos == N-1) pos <= 0;
            else pos <= pos + 1'b1;
          end else grp <= grp + 1'b1;
          st <= S_BEAT;
        end else waitc <= waitc + 1'b1;
      end
      S_BEAT: begin
        if (i_beat_ready) begin
          if (last_hold) begin o_done <= 1'b1; st <= S_DONE; end
          else st <= S_RD;
        end
      end
      S_DONE: begin
        // hold o_done until the next i_start
      end
      default: st <= S_IDLE;
    endcase
  end

  generate
    if (LANES != 8) begin : g_lanes_assert
      initial $error("spiking_embed_top requires LANES==8 for E/8-word table addressing");
    end
  endgenerate
endmodule
