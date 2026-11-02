`timescale 1ns/1ps
// cfc_cell_core.sv -- one CfC timestep datapath, bit exact with
// docs/CFC_CONTRACT.md V1.1 section 1.3.
//
// Fixed micro-sequence per timestep, all 16 multiplications groups share one
// time-multiplexed MAC (cfc_mac, 1 beat/clock, exact 48-bit accumulate):
//   S_LIN  : 16 linear branch outputs, 5 MAC beats each (bias in beat 0)
//            -> ROUND_Q12_SAT16 -> lin[0..15] = {W1,W2,Wa,Wb} x 4 outputs
//   S_TANH : 8 tanh LUT reads -> f1[0..3] = tanh(lin[0..3]),
//                               f2[0..3] = tanh(lin[4..7])
//   S_GATE : gate_pre[o] = ROUND( lin_a[o]*dt + (lin_b[o] << 12) )   (V1.1)
//   S_SIG  : 4 sigmoid LUT reads -> g[0..3]
//   S_HNEW : h_new[o] = ROUND( f1[o]*(4096-g[o]) + f2[o]*g[o] )
//   S_Y    : y = ROUND( bo + sum_i Wo[i]*h_new[i] )
// All four new states are produced from the same old-state snapshot; the
// sequence controller commits them together after "done".
//
// MAC operand/result pipeline inside this module:
//   issue cycle t (address/flags) -> d1 register (ROM data valid) ->
//   cfc_mac 3 stages -> valid_out (result) -> capture/round -> array write.
module cfc_cell_core (
  input  logic               clk,
  input  logic               rst,
  input  logic               start,
  input  logic               abort,
  input  logic signed [15:0] x_in,
  input  logic signed [15:0] h_old_0,
  input  logic signed [15:0] h_old_1,
  input  logic signed [15:0] h_old_2,
  input  logic signed [15:0] h_old_3,
  input  logic signed [15:0] delta_t,
  output logic signed [15:0] y_out,
  output logic signed [15:0] h_new_0,
  output logic signed [15:0] h_new_1,
  output logic signed [15:0] h_new_2,
  output logic signed [15:0] h_new_3,
  output logic               done,
  output logic               busy
);

  // ------------------------------------------------------------------ kinds
  localparam logic [1:0] K_LIN  = 2'd0;
  localparam logic [1:0] K_GATE = 2'd1;
  localparam logic [1:0] K_HNEW = 2'd2;
  localparam logic [1:0] K_Y    = 2'd3;

  localparam logic [1:0] AS_W     = 2'd0;
  localparam logic [1:0] AS_LIN_A = 2'd1;
  localparam logic [1:0] AS_F1    = 2'd2;
  localparam logic [1:0] AS_F2    = 2'd3;

  localparam logic [2:0] BS_Z      = 3'd0;
  localparam logic [2:0] BS_DT     = 3'd1;
  localparam logic [2:0] BS_M4096G = 3'd2;
  localparam logic [2:0] BS_G      = 3'd3;
  localparam logic [2:0] BS_HNEW   = 3'd4;

  localparam logic [1:0] CS_ZERO = 2'd0;
  localparam logic [1:0] CS_BIAS = 2'd1;
  localparam logic [1:0] CS_LINB = 2'd2;

  typedef enum logic [3:0] {
    S_IDLE, S_INIT, S_LIN, S_LIN_WAIT, S_TANH, S_GATE, S_GATE_WAIT,
    S_SIG, S_HNEW, S_HNEW_WAIT, S_Y, S_Y_WAIT
  } state_t;

  state_t state;

  // ------------------------------------------------------------ datapath regs
  logic signed [15:0] z    [0:4];   // old snapshot: z[0]=x, z[1..4]=h_old
  logic signed [15:0] dt_q;
  logic signed [15:0] lin  [0:15];
  logic signed [15:0] f1   [0:3];
  logic signed [15:0] f2   [0:3];
  logic signed [15:0] gpre [0:3];
  logic signed [15:0] g    [0:3];
  logic signed [15:0] hnew [0:3];

  assign h_new_0 = hnew[0];
  assign h_new_1 = hnew[1];
  assign h_new_2 = hnew[2];
  assign h_new_3 = hnew[3];

  // ---------------------------------------------------------------- counters
  logic [6:0] lin_idx;   // 0..79 linear MAC beats
  logic [2:0] zk;        // 0..4  z index inside a 5-beat group
  logic [4:0] grp;       // 0..15 linear output group
  logic [2:0] gcnt;      // 0..3  gate beats
  logic [3:0] hcnt;      // 0..7  h_new beats
  logic [2:0] ycnt;      // 0..3  readout beats
  logic [3:0] lut_cnt;   // tanh 0..8 / sigmoid 0..4
  logic [4:0] wr_cnt;    // committed result writes in the current phase

  // ------------------------------------------------------ issue stage signals
  logic        issue_en;
  logic        issue_clear;
  logic        issue_last;
  logic [1:0]  issue_kind;
  logic [1:0]  issue_a_sel;
  logic [2:0]  issue_b_sel;
  logic [1:0]  issue_c_sel;
  logic [4:0]  issue_ridx;
  logic [4:0]  issue_bidx;
  logic [2:0]  issue_zidx;
  logic [6:0]  w_addr;
  logic [4:0]  b_addr;

  // --------------------------------------------------------- d1 stage regs
  logic        d1_valid, d1_clear, d1_last;
  logic [1:0]  d1_kind, d1_a_sel, d1_c_sel;
  logic [2:0]  d1_b_sel;
  logic [4:0]  d1_ridx, d1_bidx;
  logic [2:0]  d1_zidx;

  // ------------------------------------------------------------- ROM / MAC
  logic signed [15:0] w_data;
  logic signed [31:0] b_data;
  logic signed [15:0] tanh_data, sig_data;
  logic [10:0]        tanh_addr, sig_addr;
  logic [3:0]         tanh_lin_idx;
  logic [1:0]         sig_lut_idx;

  logic               mac_valid_in;
  logic               mac_clear;
  logic signed [15:0] mac_a, mac_b;
  logic signed [47:0] mac_c;
  logic               mac_valid_out;
  logic signed [47:0] mac_acc;

  logic signed [15:0] lin_b_val;
  logic signed [15:0] d1_b_val;

  // ------------------------------------------- last/tag pipeline (align MAC)
  logic       lp1, lp2, lp3;
  logic [1:0] kp1, kp2, kp3;
  logic [4:0] rp1, rp2, rp3;

  // ------------------------------------------------ capture/round/write pipe
  logic               cap_valid;
  logic [1:0]         cap_kind;
  logic [4:0]         cap_ridx;
  logic signed [47:0] acc_cap;
  logic               wr_valid;
  logic [1:0]         wr_kind;
  logic [4:0]         wr_ridx;
  logic signed [15:0] wr_data;

  // -------------------------------------------------------------- functions
  // ROUND_Q12_SAT16 (contract 1.3): nearest, ties away from zero, sat to int16.
  function automatic logic signed [15:0] round_q12_sat16 (input logic signed [47:0] v);
    logic [48:0] mag;
    logic [48:0] q;
    logic signed [15:0] r;
    begin
      if (v >= 48'sd0) begin
        q = ({1'b0, v} + 49'd2048) >> 12;
        if (q > 49'd32767) r = 16'sh7FFF;
        else               r = q[15:0];
      end else begin
        mag = -{{1{v[47]}}, v};        // 49-bit magnitude of a negative int48
        q   = (mag + 49'd2048) >> 12;
        if (q >= 49'd32768) r = 16'sh8000;
        else                r = -$signed(q[15:0]);
      end
      return r;
    end
  endfunction

  // LUT index (contract 1.3): (v + 32768 + 32) >> 6, clamped to 0..1024.
  function automatic logic [10:0] lut_index (input logic signed [15:0] v);
    logic signed [18:0] t;
    logic [11:0]        idx;
    begin
      t   = {{3{v[15]}}, v} + 19'sd32800;
      idx = t[17:6];
      if (idx > 12'd1024) idx = 12'd1024;
      lut_index = idx[10:0];
    end
  endfunction

  // --------------------------------------------------------- child instances
  cfc_weight_rom u_wrom (
    .clk    (clk),
    .w_addr (w_addr),
    .w_data (w_data),
    .b_addr (b_addr),
    .b_data (b_data)
  );

  cfc_activation_rom u_arom (
    .clk       (clk),
    .tanh_addr (tanh_addr),
    .tanh_data (tanh_data),
    .sig_addr  (sig_addr),
    .sig_data  (sig_data)
  );

  cfc_mac u_mac (
    .clk       (clk),
    .rst       (rst),
    .valid_in  (mac_valid_in),
    .clear     (mac_clear),
    .a         (mac_a),
    .b         (mac_b),
    .c         (mac_c),
    .valid_out (mac_valid_out),
    .acc       (mac_acc)
  );

  // ------------------------------------------------------- LUT read addresses
  assign tanh_lin_idx = (lut_cnt <= 4'd7) ? lut_cnt[3:0] : 4'd7;
  assign sig_lut_idx  = (lut_cnt[2:0] <= 3'd3) ? lut_cnt[1:0] : 2'd3;
  assign tanh_addr    = lut_index(lin[tanh_lin_idx]);
  assign sig_addr     = lut_index(gpre[sig_lut_idx]);

  // -------------------------------------------------- issue stage sequencer
  always_comb begin
    issue_en     = 1'b0;
    issue_clear  = 1'b0;
    issue_last   = 1'b0;
    issue_kind   = K_LIN;
    issue_a_sel  = AS_W;
    issue_b_sel  = BS_Z;
    issue_c_sel  = CS_ZERO;
    issue_ridx   = 5'd0;
    issue_bidx   = 5'd0;
    issue_zidx   = 3'd0;
    w_addr       = 7'd0;
    b_addr       = 5'd0;
    case (state)
      S_LIN: begin
        issue_en    = 1'b1;
        issue_kind  = K_LIN;
        issue_ridx  = grp;
        issue_zidx  = zk;
        issue_clear = (zk == 3'd0);
        issue_last  = (zk == 3'd4);
        issue_c_sel = (zk == 3'd0) ? CS_BIAS : CS_ZERO;
        issue_a_sel = AS_W;
        issue_b_sel = BS_Z;
        w_addr      = lin_idx;
        b_addr      = (zk == 3'd0) ? {1'b0, grp[3:0]} : 5'd0;
      end
      S_GATE: begin
        issue_en    = 1'b1;
        issue_kind  = K_GATE;
        issue_ridx  = {3'b000, gcnt[1:0]};
        issue_clear = 1'b1;
        issue_last  = 1'b1;
        issue_a_sel = AS_LIN_A;
        issue_b_sel = BS_DT;
        issue_c_sel = CS_LINB;          // V1.1: + (lin_b << 12)
      end
      S_HNEW: begin
        issue_en    = 1'b1;
        issue_kind  = K_HNEW;
        issue_ridx  = {3'b000, hcnt[3:1]};
        issue_bidx  = {3'b000, hcnt[3:1]};
        issue_clear = ~hcnt[0];
        issue_last  = hcnt[0];
        issue_a_sel = hcnt[0] ? AS_F2 : AS_F1;
        issue_b_sel = hcnt[0] ? BS_G  : BS_M4096G;
        issue_c_sel = CS_ZERO;
      end
      S_Y: begin
        issue_en    = 1'b1;
        issue_kind  = K_Y;
        issue_ridx  = 5'd0;
        issue_bidx  = {3'b000, ycnt[1:0]};
        issue_clear = (ycnt == 3'd0);
        issue_last  = (ycnt == 3'd3);   // only the full readout sum is a result
        issue_a_sel = AS_W;
        issue_b_sel = BS_HNEW;
        issue_c_sel = (ycnt == 3'd0) ? CS_BIAS : CS_ZERO;
        w_addr      = 7'd80 + {4'b0000, ycnt[1:0]};
        b_addr      = (ycnt == 3'd0) ? 5'd16 : 5'd0;
      end
      default: begin
        issue_en = 1'b0;
      end
    endcase
  end

  // ----------------------------------------------------------- d1 pipeline
  always_ff @(posedge clk) begin
    if (rst) begin
      d1_valid <= 1'b0;
      d1_clear <= 1'b0;
      d1_last  <= 1'b0;
      d1_kind  <= 2'd0;
      d1_a_sel <= 2'd0;
      d1_b_sel <= 3'd0;
      d1_c_sel <= 2'd0;
      d1_ridx  <= 5'd0;
      d1_bidx  <= 5'd0;
      d1_zidx  <= 3'd0;
    end else if (abort) begin
      d1_valid <= 1'b0;
      d1_last  <= 1'b0;
    end else begin
      d1_valid <= issue_en;
      d1_clear <= issue_clear;
      d1_last  <= issue_last;
      d1_kind  <= issue_kind;
      d1_a_sel <= issue_a_sel;
      d1_b_sel <= issue_b_sel;
      d1_c_sel <= issue_c_sel;
      d1_ridx  <= issue_ridx;
      d1_bidx  <= issue_bidx;
      d1_zidx  <= issue_zidx;
    end
  end

  // ------------------------------------------------- MAC operand mux (d1)
  assign lin_b_val = lin[4'd12 + {2'b00, d1_ridx[1:0]}];

  always_comb begin
    mac_valid_in = d1_valid;
    mac_clear    = d1_clear;
    mac_a        = 16'sd0;
    mac_b        = 16'sd0;
    mac_c        = 48'sd0;
    case (d1_a_sel)
      AS_W:     mac_a = w_data;
      AS_LIN_A: mac_a = lin[4'd8 + {2'b00, d1_ridx[1:0]}];
      AS_F1:    mac_a = f1[d1_ridx[1:0]];
      default:  mac_a = f2[d1_ridx[1:0]];
    endcase
    case (d1_b_sel)
      BS_Z:     mac_b = z[d1_zidx];
      BS_DT:    mac_b = dt_q;
      BS_M4096G: mac_b = 16'sd4096 - g[d1_bidx[1:0]];
      BS_G:     mac_b = g[d1_bidx[1:0]];
      default:  mac_b = hnew[d1_bidx[1:0]];
    endcase
    case (d1_c_sel)
      CS_ZERO: mac_c = 48'sd0;
      CS_BIAS: mac_c = {{16{b_data[31]}}, b_data};
      default: mac_c = {{20{lin_b_val[15]}}, lin_b_val, 12'b0000_0000_0000};
    endcase
  end

  // ------------------------------------------- capture / round / write pipe
  always_ff @(posedge clk) begin
    if (rst) begin
      lp1 <= 1'b0; lp2 <= 1'b0; lp3 <= 1'b0;
      kp1 <= 2'd0; kp2 <= 2'd0; kp3 <= 2'd0;
      rp1 <= 5'd0; rp2 <= 5'd0; rp3 <= 5'd0;
      cap_valid <= 1'b0;
      cap_kind  <= 2'd0;
      cap_ridx  <= 5'd0;
      acc_cap   <= 48'sd0;
      wr_valid  <= 1'b0;
      wr_kind   <= 2'd0;
      wr_ridx   <= 5'd0;
      wr_data   <= 16'sd0;
      y_out     <= 16'sd0;
      for (int i = 0; i < 16; i++) lin[i] <= 16'sd0;
      for (int i = 0; i < 4; i++) begin
        // f1/f2/g are driven only by the main FSM block (S_TANH / S_SIG);
        // resetting them here as well would create a multi-driven register
        // (Vivado synth 8-6858/8-6859 critical warning).
        gpre[i] <= 16'sd0; hnew[i] <= 16'sd0;
      end
    end else if (abort) begin
      lp1 <= 1'b0; lp2 <= 1'b0; lp3 <= 1'b0;
      cap_valid <= 1'b0;
      wr_valid  <= 1'b0;
    end else begin
      // last / tag pipeline: aligned with cfc_mac valid_out
      lp1 <= d1_last; lp2 <= lp1; lp3 <= lp2;
      kp1 <= d1_kind; kp2 <= kp1; kp3 <= kp2;
      rp1 <= d1_ridx; rp2 <= rp1; rp3 <= rp2;

      // capture the final accumulator of each result group
      if (mac_valid_out && lp3) begin
        acc_cap   <= mac_acc;
        cap_kind  <= kp3;
        cap_ridx  <= rp3;
        cap_valid <= 1'b1;
      end else begin
        cap_valid <= 1'b0;
      end

      // round Q24 -> Q12 with saturation
      wr_valid <= cap_valid;
      if (cap_valid) begin
        wr_data <= round_q12_sat16(acc_cap);
        wr_kind <= cap_kind;
        wr_ridx <= cap_ridx;
      end

      // unified result write
      if (wr_valid) begin
        case (wr_kind)
          K_LIN:  lin[wr_ridx[3:0]]   <= wr_data;
          K_GATE: gpre[wr_ridx[1:0]]  <= wr_data;
          K_HNEW: hnew[wr_ridx[1:0]]  <= wr_data;
          default: y_out              <= wr_data;
        endcase
      end
    end
  end

  // ------------------------------------------------------------ main FSM
  always_ff @(posedge clk) begin
    if (rst) begin
      state   <= S_IDLE;
      busy    <= 1'b0;
      done    <= 1'b0;
      lin_idx <= 7'd0;
      zk      <= 3'd0;
      grp     <= 5'd0;
      gcnt    <= 3'd0;
      hcnt    <= 4'd0;
      ycnt    <= 3'd0;
      lut_cnt <= 4'd0;
      wr_cnt  <= 5'd0;
      for (int i = 0; i < 5; i++) z[i] <= 16'sd0;
      for (int i = 0; i < 4; i++) begin
        f1[i] <= 16'sd0; f2[i] <= 16'sd0; g[i] <= 16'sd0;
      end
      dt_q <= 16'sd0;
    end else begin
      done <= 1'b0;
      if (wr_valid) wr_cnt <= wr_cnt + 5'd1;

      if (abort) begin
        state  <= S_IDLE;
        busy   <= 1'b0;
        wr_cnt <= 5'd0;
      end else begin
        case (state)
          S_IDLE: begin
            busy <= 1'b0;
            if (start) begin
              z[0] <= x_in;
              z[1] <= h_old_0;
              z[2] <= h_old_1;
              z[3] <= h_old_2;
              z[4] <= h_old_3;
              dt_q <= delta_t;
              busy <= 1'b1;
              state <= S_INIT;
            end
          end
          S_INIT: begin
            lin_idx <= 7'd0;
            zk      <= 3'd0;
            grp     <= 5'd0;
            wr_cnt  <= 5'd0;
            state   <= S_LIN;
          end
          S_LIN: begin
            if (zk == 3'd4) begin
              zk  <= 3'd0;
              grp <= grp + 5'd1;
            end else begin
              zk <= zk + 3'd1;
            end
            if (lin_idx == 7'd79) state <= S_LIN_WAIT;
            else                  lin_idx <= lin_idx + 7'd1;
          end
          S_LIN_WAIT: begin
            if (wr_cnt >= 5'd16) begin
              lut_cnt <= 4'd0;
              state   <= S_TANH;
            end
          end
          S_TANH: begin
            if (lut_cnt != 4'd0) begin
              if (lut_cnt <= 4'd4) f1[lut_cnt[1:0] - 2'd1] <= tanh_data;
              else                 f2[lut_cnt[1:0] + 2'd3] <= tanh_data;
            end
            if (lut_cnt == 4'd8) begin
              state  <= S_GATE;
              gcnt   <= 3'd0;
              wr_cnt <= 5'd0;
            end else begin
              lut_cnt <= lut_cnt + 4'd1;
            end
          end
          S_GATE: begin
            if (gcnt == 3'd3) begin
              state  <= S_GATE_WAIT;
              wr_cnt <= 5'd0;
            end else begin
              gcnt <= gcnt + 3'd1;
            end
          end
          S_GATE_WAIT: begin
            if (wr_cnt >= 5'd4) begin
              lut_cnt <= 4'd0;
              state   <= S_SIG;
            end
          end
          S_SIG: begin
            if (lut_cnt != 4'd0) begin
              g[lut_cnt[1:0] - 2'd1] <= sig_data;
            end
            if (lut_cnt == 4'd4) begin
              state  <= S_HNEW;
              hcnt   <= 4'd0;
              wr_cnt <= 5'd0;
            end else begin
              lut_cnt <= lut_cnt + 4'd1;
            end
          end
          S_HNEW: begin
            if (hcnt == 4'd7) begin
              // NOTE: do not clear wr_cnt here; the first h_new result write
              // lands on this very edge and must be counted.
              state  <= S_HNEW_WAIT;
            end else begin
              hcnt <= hcnt + 4'd1;
            end
          end
          S_HNEW_WAIT: begin
            if (wr_cnt >= 5'd4) begin
              state  <= S_Y;
              ycnt   <= 3'd0;
              wr_cnt <= 5'd0;
            end
          end
          S_Y: begin
            if (ycnt == 3'd3) begin
              state  <= S_Y_WAIT;
              wr_cnt <= 5'd0;
            end else begin
              ycnt <= ycnt + 3'd1;
            end
          end
          S_Y_WAIT: begin
            if (wr_cnt >= 5'd1) begin
              done  <= 1'b1;
              busy  <= 1'b0;
              state <= S_IDLE;
            end
          end
          default: state <= S_IDLE;
        endcase
      end
    end
  end
endmodule