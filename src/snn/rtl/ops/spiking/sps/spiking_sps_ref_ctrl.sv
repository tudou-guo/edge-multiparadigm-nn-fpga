// Reference-style SPS controller (simple_eyeriss_Controller + code_fetch mechanism).
// Upstream: eyeriss_part/simple_eyeriss_Controller.v, code_fetch.v,
//           Organize_data_unit.v, simple_eyeriss_top.v bank rotation.
// Bank geometry mirrors the reference TmpRam: 128b word, 13b address.
//   f1    : word(g1,y,k)  = (g1*H+y)*8+k, bit(x*T+t), 32 pos used.
//   pooled: word(g1,y1,w) = (g1*H1+y1)*4+w, bit((i*8+k)*T+t), i=pos%4.
//   prow2 : word(g2,y2,k) = (g2*H1+y2)*8+k, bit(x*T+t), 16 pos used.
// bank0 holds f1 then prow2 (phases sequential); bank1 holds pooled.
// Row-block scheduling for conv2 with 3-row reuse buffers; tile weights
// streamed per (out-group,in-tile); positions swept inner.
// Pool1/Pool2: 2x2 OR per timestep (spike max == OR).
// Lmem=1 everywhere; tags travel with data; no full-frame top-level tbuf
// (row-block buffers only: f1row 12KB regs, c_rows 1.5Kb, p2row 512b).
module spiking_sps_ref_ctrl #(
  parameter int H = 32, parameter int W_IN = 32,
  parameter int C1 = 96, parameter int C2 = 384,
  parameter int T = 4, parameter int LANES = 8, parameter int TC2 = 8,
  parameter int ACC_W = 21
)(
  input  logic s_clk, input  logic s_rst,
  input  logic [63:0] i_w_data, input  logic i_w_valid, output logic o_w_ready,
  input  logic [23:0] i_pix_data, input  logic i_pix_valid, output logic o_pix_ready,
  input  logic i_start, output logic o_done,
  output logic [63:0] o_enc_w_data, output logic o_enc_w_valid, input  logic i_enc_w_ready,
  output logic [23:0] o_enc_pix_data, output logic o_enc_pix_valid, input  logic i_enc_pix_ready,
  output logic o_enc_start, input  logic i_enc_done,
  input  logic [LANES*T-1:0] i_enc_beat, input  logic i_enc_valid, output logic o_enc_ready,
  output logic [63:0] o_arr_w_data, output logic o_arr_w_valid, input  logic i_arr_w_ready,
  output logic o_arr_win_valid, input  logic i_arr_win_ready,
  output logic [9*TC2*T-1:0] o_arr_win_data,
  output logic [7:0] o_arr_win_pos, output logic [5:0] o_arr_win_grp,
  output logic o_arr_win_first, output logic o_arr_win_last, output logic o_arr_tile_release,
  input  logic i_arr_res_valid, output logic o_arr_res_ready,
  input  logic [LANES*T*21-1:0] i_arr_res_data,
  input  logic [7:0] i_arr_res_pos, input  logic [5:0] i_arr_res_grp,
  input  logic i_arr_res_first, input  logic i_arr_res_last,
  output logic [63:0] o_psum_w_data, output logic o_psum_w_valid, input  logic i_psum_w_ready,
  output logic o_psum_w_done,
  output logic o_psum_res_valid, input  logic i_psum_res_ready,
  input  logic [LANES*T-1:0] i_psum_beat, input  logic i_psum_beat_valid,
  output logic o_psum_beat_ready, input  logic [7:0] i_psum_beat_pos, input  logic [5:0] i_psum_beat_grp,
  output logic o_b0_we, output logic [12:0] o_b0_waddr, output logic [127:0] o_b0_wdata,
  output logic [12:0] o_b0_raddr, input  logic [127:0] i_b0_rdata,
  output logic o_b1_we, output logic [12:0] o_b1_waddr, output logic [127:0] o_b1_wdata,
  output logic [12:0] o_b1_raddr, input  logic [127:0] i_b1_rdata,
  output logic [LANES*T-1:0] o_tok_data, output logic o_tok_valid, input  logic i_tok_ready,
  output logic o_tok_last,
  output logic o_dbg_c2w
);
  localparam int G1 = C1/LANES, G2 = C2/LANES, NT2 = C1/TC2;
  localparam int NPOS = H*W_IN, H1 = H/2, W1 = W_IN/2;
  localparam int P1N = H1*W1, H2 = H/4, W2 = W_IN/4, NTOK = H2*W2;
  localparam int W1B = (C1*3*9+7)/8, BMAX = (C1>C2)?C1:C2;
  localparam int TILEB = (LANES*TC2*9+7)/8;
  localparam int TBEATS = NTOK*G2;
  localparam int SUBT = LANES/TC2;
  localparam int ROWW = W1; // prow2 beats per row (16 native, 4 small)
  localparam int BEATW = LANES*T; // spike-beat width (32 native, 16 small)
  typedef enum logic [3:0] {C_IDLE, C_LDW1, C_LDB, C_LDPIX, C_ENC,
    C_POOL1, C_C2TILE, C_C2ROWS, C_C2WIN, C_C2W, C_C2FL, C_DRAIN, C_DONE} cstate_t;
  cstate_t st;
  logic [31:0] cnt;
  // encoder row assembly
  logic [31:0] ey, ex; logic [15:0] eg;
  logic [127:0] f1row [16][8];
  logic f1row_v; logic [7:0] fl_cnt;
  logic enc_started;
  // pool1
  logic [15:0] p1g, p1py; logic [4:0] p1rk; logic [1:0] p1ph;
  logic [127:0] p1r0 [8], p1r1 [8];
  logic [2:0] p1wk;
  // conv2
  logic [15:0] c_g2, c_tile;
  logic [15:0] c_pr, c_pc;
  logic signed [15:0] c_lr;
  logic [5:0] c_ldn;
  logic [1:0] c_ldph;
  logic [LANES*T-1:0] c_rows [3][16];
  logic [3:0] c_wph;
  logic c_win_sent;
  // p2row holds a full prow2 row per channel (16 pos x T); flush word is
  // zero-padded to 128b (T=2 uses the low 32b).
  logic [16*T-1:0] p2row [8];
  logic [4:0] p2cnt; logic [3:0] p2fl;
  // Row identity is latched from the psum beat tags when the first beat of a
  // row is captured. The flush of a row can happen several cycles later (even
  // after c_g2 advanced / c_fpr was cleared: the LAST row of a group has its
  // final beat arrive 2+ cycles after the tile/g2 advance). Addressing the
  // flush from the running counters would therefore write the row to the next
  // group's row 0. Latched (p2_g,p2_y) is the only correct source.
  logic [5:0] p2_g;   // output group of the row assembled in p2row
  logic [15:0] p2_y;  // row index in group  (= beat pos / ROWW)
  // NOTE: C_C2ROWS is deliberately NOT in this set. A flush preempting the
  // b1 row-load 3-phase sequence would clobber the in-flight BRAM read: the
  // per-cycle default "o_b1_raddr <= 0" walks raddr_q to 0 during the flush, so
  // the ph2 sample would read mem[0] instead of the issued row word. Beats are
  // backpressured by o_psum_beat_ready while p2row is full, so the flush can
  // safely wait for the short (<=12 word) row load to finish and fires in the
  // next C_C2WIN/C_C2TILE.
  wire flush_now = ((st == C_C2WIN) || (st == C_C2TILE))
                   && (p2cnt == ROWW);
  // Direct row capture (no FIFO): psum beats land straight into p2row at
  // p2cnt; a full row triggers C_C2W, during which o_psum_beat_ready is 0
  // (clean backpressure — flush is 8 cycles, window period ~22, no deadlock).
  // This removes all head/tail pointer failure modes.
  // (beat FIFO removed; direct p2row capture above)
  // total psum beats expected this run; C_C2FL waits until all arrive
  // (array/psum pipeline in flight) before draining.
  localparam int TOTAL_BEATS = (H1*W1)*G2;
  logic [31:0] beat_pushed;
  logic [2:0] c_ret; // 0=WIN,1=FL(recheck),2=DRAIN,3=TILE(resume),4=ROWS(resume)
  logic [15:0] c_fpr;
  logic c_res_got;
  // drain
  logic [15:0] d_n, d_g; logic [4:0] d_rk; logic [1:0] d_ph;
  logic [127:0] dr0 [8], dr1 [8];
  logic [LANES*T-1:0] tok_q; logic tok_qv;
  assign o_dbg_c2w = (st == C_C2W);
  // Stream forwarding is COMBINATIONAL (same-edge fire for host, controller
  // counter and submodule). A registered forward would delay valid by one
  // cycle relative to the controller's own fire count and diverge word
  // counts across phase boundaries. fire = valid && ready everywhere.
  always_comb begin
    o_enc_w_data = i_w_data;
    o_enc_pix_data = i_pix_data[23:0];
    o_arr_w_data = i_w_data;
    o_psum_w_data = i_w_data;
    o_enc_w_valid = ((st == C_LDW1) || (st == C_LDB)) ? i_w_valid : 1'b0;
    o_enc_pix_valid = (st == C_LDPIX) ? i_pix_valid : 1'b0;
    // A flush preempts C_C2TILE in the same cycle; the weight handshake must
    // be closed then, else the host/array would consume a beat that cnt did
    // not count (tile stream would lose its last word).
    o_arr_w_valid = ((st == C_C2TILE) && !flush_now) ? i_w_valid : 1'b0;
    o_psum_w_valid = (st == C_LDB) ? i_w_valid : 1'b0;
    o_w_ready = ((st == C_LDW1) ? i_enc_w_ready : 1'b0)
              | ((st == C_LDB) ? (i_enc_w_ready & i_psum_w_ready) : 1'b0)
              | (((st == C_C2TILE) && !flush_now) ? i_arr_w_ready : 1'b0);
    o_pix_ready = (st == C_LDPIX) ? i_enc_pix_ready : 1'b0;
    o_arr_win_pos = c_pr*W1+c_pc; o_arr_win_grp = c_g2[5:0];
    o_arr_win_first = (c_tile == 0); o_arr_win_last = (c_tile == NT2-1);
    o_tok_data = tok_q; o_tok_valid = tok_qv;
    o_tok_last = (d_n == NTOK-1 && d_g == G2-1);
  end
  assign o_psum_res_valid = i_arr_res_valid;
  assign o_arr_res_ready = i_psum_res_ready;
  // Beat acceptance is combinational and gated by p2row room: a registered
  // ready that ignored p2cnt consumed psum beats with no capture slot and
  // silently dropped them (sequence then shifted by one per drop).
  assign o_psum_beat_ready = ((st == C_C2TILE) || (st == C_C2ROWS)
                           || (st == C_C2WIN)  || (st == C_C2FL))
                           && (p2cnt < ROWW);
  // window mux from row regs (pad-aware)
  always_comb begin
    for (int b = 0; b < 9; b = b + 1) begin
      int rr, cc;
      rr = b / 3; cc = b % 3;
      for (int k = 0; k < TC2; k = k + 1) begin
        int px, g1t, sub, lane;
        px = int'(c_pc) + cc - 1;
        g1t = int'(c_tile) / SUBT; sub = int'(c_tile) % SUBT;
        lane = sub*TC2 + k;
        for (int t = 0; t < T; t = t + 1) begin
          if (px >= 0 && px < W1)
            o_arr_win_data[(b*TC2+k)*T+t] = c_rows[rr][px][lane*T+t];
          else
            o_arr_win_data[(b*TC2+k)*T+t] = 1'b0;
        end
      end
    end
  end
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin
      st <= C_IDLE; cnt <= 0; o_done <= 1'b0;
      ey <= 0; ex <= 0; eg <= 0; f1row_v <= 1'b0; fl_cnt <= 0; enc_started <= 1'b0;
      p1g <= 0; p1py <= 0; p1rk <= 0; p1ph <= 0; p1wk <= 0;
      c_g2 <= 0; c_tile <= 0; c_pr <= 0; c_pc <= 0; c_lr <= 0; c_ldn <= 0; c_ldph <= 0;
      c_wph <= 0; c_win_sent <= 1'b0; p2cnt <= 0; p2fl <= 0;
      c_ret <= 2'b0;
      c_fpr <= 0; c_res_got <= 1'b0; beat_pushed <= 0; p2cnt <= 0;
      d_n <= 0; d_g <= 0; d_rk <= 0; d_ph <= 0; tok_q <= '0; tok_qv <= 1'b0;
      p2_g <= 0; p2_y <= 0;
      for (int k = 0; k < 8; k = k + 1) p2row[k] <= '0;
      for (int a = 0; a < 16; a = a + 1) for (int k = 0; k < 8; k = k + 1) f1row[a][k] <= '0;
    end else begin
      // clocked-output defaults (single driver; handshake pulses explicit).
      // Stream valid/ready are combinational above (same-edge fire).
      o_enc_start <= 1'b0; o_enc_ready <= 1'b0;
      o_arr_win_valid <= 1'b0; o_arr_tile_release <= 1'b0;
      o_psum_w_done <= 1'b0;
      // o_psum_beat_ready is combinational (see assign above), gated by room.
      o_b0_we <= 1'b0; o_b0_waddr <= '0; o_b0_wdata <= '0; o_b0_raddr <= '0;
      o_b1_we <= 1'b0; o_b1_waddr <= '0; o_b1_wdata <= '0; o_b1_raddr <= '0;
      if (i_psum_beat_valid && o_psum_beat_ready) begin
        // Latch the row identity from the beat tags on the first beat of the
        // row; every beat of a row carries the same (grp, pos/ROWW).
        if (p2cnt == 0) begin
          p2_g <= i_psum_beat_grp;
          p2_y <= i_psum_beat_pos / ROWW;
        end
        for (int k = 0; k < 8; k = k + 1)
          p2row[k][p2cnt*T +: T] <= i_psum_beat[k*T +: T];
        p2cnt <= p2cnt + 1'b1;
        beat_pushed <= beat_pushed + 1'b1;
      end
      // array-result accept flag (covers flush overlap)
      if ((st == C_C2WIN || st == C_C2W) && i_arr_res_valid && i_psum_res_ready)
        c_res_got <= 1'b1;
      // Row-flush trigger (flush_now) is applied by preempting the case body
      // of C_C2TILE/C_C2ROWS/C_C2WIN below. It must preempt (not be a separate
      // assignment before the case): the old trigger assigned st<=C_C2W and
      // then the same-cycle case body could assign st again and win, so the
      // flush was skipped while the state advanced underneath it.
      // c_ret records where to resume: 0=WIN,1=FL,2=DRAIN,3=TILE,4=ROWS.
      case (st)
      // o_done is sticky until the next i_start (a 1-cycle pulse would be
      // missed by a checker that just finished comparing the last beat).
      C_IDLE: begin if (i_start) begin o_done <= 1'b0; cnt <= 0;
        beat_pushed <= 0; p2cnt <= 0;
        st <= C_LDW1; end end
      C_LDW1: begin
        if (i_w_valid && i_enc_w_ready) begin
          if (cnt == W1B-1) begin cnt <= 0; st <= C_LDB; end else cnt <= cnt+1'b1;
        end
      end
      C_LDB: begin
        if (i_w_valid && i_enc_w_ready && i_psum_w_ready) begin
          if (cnt == BMAX-1) begin cnt <= 0; o_psum_w_done <= 1'b1; st <= C_LDPIX; end
          else cnt <= cnt+1'b1;
        end
      end
      C_LDPIX: begin
        if (i_pix_valid && i_enc_pix_ready) begin
          if (cnt == NPOS-1) begin cnt <= 0; ey <= 0; ex <= 0; eg <= 0; st <= C_ENC; end
          else cnt <= cnt+1'b1;
        end
      end
      C_ENC: begin
        if (!enc_started) o_enc_start <= 1'b1;
        if (i_enc_valid) enc_started <= 1'b1;
        o_enc_ready <= !f1row_v;
        if (i_enc_valid && !f1row_v) begin
          for (int k = 0; k < 8; k = k + 1)
            f1row[eg][k][ex*T +: T] <= i_enc_beat[k*T +: T];
          if (eg == G1-1) begin
            eg <= 0;
            if (ex == W_IN-1) begin ex <= 0; ey <= ey+1'b1; f1row_v <= 1'b1; fl_cnt <= 0; end
            else ex <= ex+1'b1;
          end else eg <= eg+1'b1;
        end
        if (f1row_v) begin
          o_b0_we <= 1'b1;
          o_b0_waddr <= ((fl_cnt/8)*H + (ey-1))*8 + fl_cnt[2:0];
          o_b0_wdata <= f1row[fl_cnt/8][fl_cnt[2:0]];
          if (fl_cnt == G1*8-1) begin
            fl_cnt <= 0; f1row_v <= 1'b0;
            for (int a = 0; a < 16; a = a + 1) for (int k = 0; k < 8; k = k + 1) f1row[a][k] <= '0;
            if (ey == H && i_enc_done) begin p1g <= 0; p1py <= 0; p1rk <= 0; p1ph <= 0; st <= C_POOL1; end
          end else fl_cnt <= fl_cnt+1'b1;
        end else if (ey == H && i_enc_done) begin
          p1g <= 0; p1py <= 0; p1rk <= 0; p1ph <= 0; st <= C_POOL1;
        end
      end
      C_POOL1: begin
        case (p1ph)
          0: begin
            if (p1rk < 8) o_b0_raddr <= (p1g*H + 2*p1py)*8 + p1rk;
            else o_b0_raddr <= (p1g*H + 2*p1py+1)*8 + (p1rk-8);
            p1ph <= 1;
          end
          1: p1ph <= 2;
          2: begin
            if (p1rk < 8) p1r0[p1rk] <= i_b0_rdata; else p1r1[p1rk-8] <= i_b0_rdata;
            if (p1rk == 15) begin p1rk <= 0; p1ph <= 3; p1wk <= 0; end
            else begin p1rk <= p1rk+1'b1; p1ph <= 0; end
          end
          3: begin
            // write pooled word p1wk (zero-init: T=2 leaves upper half 0)
            logic [127:0] wout;
            wout = '0;
            for (int i = 0; i < 4; i = i + 1)
              for (int k = 0; k < 8; k = k + 1)
                for (int t = 0; t < T; t = t + 1) begin
                  int x0;
                  x0 = p1wk*4+i;
                  wout[((i*8+k)*T)+t] = p1r0[k][((2*x0)*T)+t] | p1r0[k][((2*x0+1)*T)+t]
                    | p1r1[k][((2*x0)*T)+t] | p1r1[k][((2*x0+1)*T)+t];
                end
            o_b1_we <= 1'b1;
            o_b1_waddr <= (p1g*H1+p1py)*4+p1wk;
            o_b1_wdata <= wout;
            if (p1wk == 3) begin
              p1wk <= 0; p1ph <= 0; p1rk <= 0;
              if (p1g == G1-1 && p1py == H1-1) begin
                c_g2 <= 0; c_tile <= 0; c_pr <= 0; c_fpr <= 0; c_res_got <= 1'b0; st <= C_C2TILE;
              end else if (p1py == H1-1) begin p1py <= 0; p1g <= p1g+1'b1; end
              else p1py <= p1py+1'b1;
            end else p1wk <= p1wk+1'b1;
          end
        endcase
      end
      C_C2TILE: begin
        if (flush_now) begin
          // A completed p2row row preempts the tile stream (weight handshake is
          // gated by !flush_now in the comb block). Resume the same cnt after
          // the flush: no weight beat is consumed on this cycle.
          c_ret <= 3'd3; st <= C_C2W;
        end else begin
          // stream one tile's weights host -> array (comb forward)
          if (i_w_valid && i_arr_w_ready) begin
            if (cnt == TILEB-1) begin
              cnt <= 0; c_pr <= 0; c_lr <= -1; c_ldn <= 0; c_ldph <= 0; st <= C_C2ROWS;
            end else cnt <= cnt+1'b1;
          end
        end
      end
      C_C2ROWS: begin
        // Never preempted by the row flush (in-flight b1 read, see flush_now).
        // load pooled rows c_lr+1 .. ensure rows (c_pr-1,c_pr,c_pr+1) present.
        // simplified: load 3 rows fresh per pr (12 word reads).
        case (c_ldph)
          0: begin
            int py, g1t;
            py = int'(c_pr) + (c_ldn/4) - 1;
            g1t = int'(c_tile) / SUBT;
            if (py < 0 || py >= H1) begin
              // pad row: zeros, no read
              for (int i = 0; i < 4; i = i + 1)
                c_rows[(c_ldn/4)][c_ldn%4*4+i] <= '0;
              c_ldph <= 2; // skip to next
            end else begin
              o_b1_raddr <= (g1t*H1+py)*4 + (c_ldn%4);
              c_ldph <= 1;
            end
          end
          1: c_ldph <= 2;
          2: begin
            int py;
            py = int'(c_pr) + (c_ldn/4) - 1;
            if (!(py < 0 || py >= H1)) begin
              for (int i = 0; i < 4; i = i + 1)
                c_rows[(c_ldn/4)][(c_ldn%4)*4+i] <= i_b1_rdata[(i*BEATW)+BEATW-1 -: BEATW];
            end
            if (c_ldn == 11) begin c_ldn <= 0; c_ldph <= 0; c_pc <= 0; c_wph <= 0; c_win_sent <= 1'b0; st <= C_C2WIN; end
            else begin c_ldn <= c_ldn+1'b1; c_ldph <= 0; end
          end
        endcase
      end
      C_C2WIN: begin
        if (flush_now) begin
          c_ret <= 3'd0; st <= C_C2W;
        end else begin
        case (c_wph)
          0: begin
            o_arr_win_valid <= 1'b1;
            if (i_arr_win_ready) begin c_win_sent <= 1'b1; c_wph <= 1; end
          end
          1: begin
            if (c_res_got) begin
              c_res_got <= 1'b0;
              c_win_sent <= 1'b0;
              if (c_pc == W1-1) begin
                if (c_pr == H1-1) begin
                  o_arr_tile_release <= 1'b1;
                  c_wph <= 2;
                end else begin c_pr <= c_pr+1'b1; c_lr <= c_lr+1; c_ldn <= 0; c_ldph <= 0; st <= C_C2ROWS; end
              end else begin c_pc <= c_pc+1'b1; c_wph <= 0; end
            end
          end
          2: begin
            // tile done; advance tile/g2
            if (c_tile == NT2-1) begin
              if (c_g2 == G2-1) begin st <= C_C2FL; end
              else begin c_g2 <= c_g2+1'b1; c_tile <= 0; cnt <= 0; c_fpr <= 0; c_res_got <= 1'b0; st <= C_C2TILE; end
            end else begin c_tile <= c_tile+1'b1; cnt <= 0; st <= C_C2TILE; end
            c_wph <= 0;
          end
        endcase
        end
      end
      C_C2W: begin
        // Flush one prow2 row: 8 words of (p2_g,p2_y,k). The address MUST use
        // the identity latched with the captured row, not c_g2/c_fpr: the last
        // row of a group has its final psum beat arrive after the tile/g2
        // advance already ran (c_g2++, c_fpr<=0), so live counters are stale.
        o_b0_we <= 1'b1;
        o_b0_waddr <= (p2_g*H1 + p2_y)*8 + p2fl;
        o_b0_wdata <= {{(128-16*T){1'b0}}, p2row[p2fl]};
        if (p2fl == 7) begin
          p2fl <= 0; p2cnt <= 0; c_fpr <= c_fpr + 1'b1;
          if (c_ret == 3'd1) st <= C_C2FL;
          else if (c_ret == 3'd2) begin d_n <= 0; d_g <= 0; d_rk <= 0; d_ph <= 0; st <= C_DRAIN; end
          else if (c_ret == 3'd3) st <= C_C2TILE;
          else if (c_ret == 3'd4) st <= C_C2ROWS;
          else st <= C_C2WIN;
        end else p2fl <= p2fl+1'b1;
      end
      C_C2FL: begin
        // Sweep done: wait until every in-flight window produced its beat.
        // Rows flush as they complete; TOTAL is an exact multiple of ROWW,
        // so p2cnt==0 when the last beat lands. A nonzero residue would mean
        // lost beats: pad-and-flush it, then drain (never silently drop).
        if (p2cnt == ROWW) begin
          p2fl <= 0; c_ret <= 3'd1; st <= C_C2W;
        end else if (beat_pushed == TOTAL_BEATS && p2cnt == 0) begin
          d_n <= 0; d_g <= 0; d_rk <= 0; d_ph <= 0; st <= C_DRAIN;
        end else if (beat_pushed == TOTAL_BEATS && p2cnt > 0) begin
          for (int k = 0; k < 8; k = k + 1)
            for (int i = 0; i < 16; i = i + 1)
              if (i >= p2cnt) p2row[k][i*T +: T] <= '0;
          p2fl <= 0; c_ret <= 3'd2; st <= C_C2W;
        end
      end
      C_DRAIN: begin
        case (d_ph)
          0: begin
            // read prow2 rows 2ty (+k) - 16 words
            if (d_rk < 8) o_b0_raddr <= (d_g*H1 + 2*(d_n/W2))*8 + d_rk;
            else o_b0_raddr <= (d_g*H1 + 2*(d_n/W2)+1)*8 + (d_rk-8);
            d_ph <= 1;
          end
          1: d_ph <= 2;
          2: begin
            if (d_rk < 8) dr0[d_rk] <= i_b0_rdata; else dr1[d_rk-8] <= i_b0_rdata;
            if (d_rk == 15) begin d_rk <= 0; d_ph <= 3; end
            else begin d_rk <= d_rk+1'b1; d_ph <= 0; end
          end
          3: begin
            // OR 2x2 -> token beat; hold until accepted (blocking)
            if (!tok_qv) begin
              int tx;
              tx = d_n % W2;
              for (int k = 0; k < 8; k = k + 1)
                for (int t = 0; t < T; t = t + 1)
                  tok_q[k*T+t] <= dr0[k][((2*tx)*T)+t] | dr0[k][((2*tx+1)*T)+t]
                    | dr1[k][((2*tx)*T)+t] | dr1[k][((2*tx+1)*T)+t];
              tok_qv <= 1'b1;
            end else if (i_tok_ready) begin
              tok_qv <= 1'b0;
              if (d_n == NTOK-1 && d_g == G2-1) begin st <= C_DONE; end
              else if (d_g == G2-1) begin d_g <= 0; d_n <= d_n+1'b1; end
              else d_g <= d_g+1'b1;
              d_ph <= 0; d_rk <= 0;
            end
          end
        endcase
      end
      C_DONE: begin
        o_done <= 1'b1;
        tok_qv <= 1'b0;
        if (i_start) begin o_done <= 1'b0; cnt <= 0; st <= C_LDW1; end
        else begin st <= C_IDLE; cnt <= 0; end
      end
      default: st <= C_IDLE;
      endcase
    end
  end
endmodule
