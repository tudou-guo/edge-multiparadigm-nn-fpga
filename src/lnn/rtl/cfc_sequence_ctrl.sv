`timescale 1ns/1ps
// cfc_sequence_ctrl.sv -- timestep loop, input/output RAM access, state commit,
// RUN_CYCLES counting and run timeout (contract section 3.1 / 3.4).
//
// One cell_core invocation per timestep.  Input sample i is read from the input
// RAM at address i, the output y of the cell is written at address i, and the
// four new states are committed together right after the cell reports done.
// The next timestep therefore always sees the state committed by the previous
// step only (all four branches of one step share one old-state snapshot).
// MODE[1] (cfg_clear_each) forces a zero h_old for the computation while the
// computed state is still committed (for H_FINAL observation).
module cfc_sequence_ctrl (
  input  logic               clk,
  input  logic               rst,
  // run request / configuration snapshot (from cfc_axi_lite_regs)
  input  logic               start_pulse,
  input  logic [6:0]         cfg_n_steps,
  input  logic [15:0]        cfg_delta_t,
  input  logic               cfg_clear_pre,
  input  logic               cfg_clear_each,
  input  logic [31:0]        cfg_timeout_cycles,
  input  logic               clear_state_cmd,
  // input vector RAM read port
  output logic [5:0]         in_ram_addr,
  input  logic signed [15:0] in_ram_data,
  // output vector RAM write port
  output logic               out_we,
  output logic [5:0]         out_addr,
  output logic signed [15:0] out_wdata,
  // state registers
  output logic               state_clear,
  output logic               state_commit,
  input  logic signed [15:0] h_cur_0,
  input  logic signed [15:0] h_cur_1,
  input  logic signed [15:0] h_cur_2,
  input  logic signed [15:0] h_cur_3,
  // single step cell core
  output logic               cell_start,
  output logic signed [15:0] cell_x,
  output logic signed [15:0] cell_h_old_0,
  output logic signed [15:0] cell_h_old_1,
  output logic signed [15:0] cell_h_old_2,
  output logic signed [15:0] cell_h_old_3,
  output logic signed [15:0] cell_delta_t,
  input  logic signed [15:0] cell_y,
  input  logic               cell_done,
  output logic               cell_abort,
  // status
  output logic               busy,
  output logic               run_done,
  output logic               run_error,
  output logic [1:0]         err_code,
  output logic [31:0]        run_cycles,
  output logic [6:0]         step_count
);
  typedef enum logic [2:0] {
    Q_IDLE, Q_PRECLEAR, Q_REQ, Q_START, Q_WAIT, Q_DONE, Q_ABORT, Q_ERR
  } qstate_t;

  qstate_t     state;
  logic [31:0] cycles;
  logic [6:0]  step_i;
  logic [6:0]  step_cnt;
  logic [31:0] run_cycles_q;
  logic signed [15:0] x_q;
  logic signed [15:0] dt_q;

  assign busy          = (state != Q_IDLE);
  assign in_ram_addr   = step_i[5:0];
  assign cell_start    = (state == Q_START);
  assign cell_x        = x_q;
  assign cell_h_old_0  = cfg_clear_each ? 16'sd0 : h_cur_0;
  assign cell_h_old_1  = cfg_clear_each ? 16'sd0 : h_cur_1;
  assign cell_h_old_2  = cfg_clear_each ? 16'sd0 : h_cur_2;
  assign cell_h_old_3  = cfg_clear_each ? 16'sd0 : h_cur_3;
  assign cell_delta_t  = dt_q;
  assign cell_abort    = (state == Q_ABORT);
  assign state_clear   = (state == Q_PRECLEAR) || ((state == Q_IDLE) && clear_state_cmd);
  assign state_commit  = (state == Q_WAIT) && cell_done;
  assign out_we        = (state == Q_WAIT) && cell_done;
  assign out_addr      = step_i[5:0];
  assign out_wdata     = cell_y;
  assign run_cycles    = run_cycles_q;
  assign step_count    = step_cnt;

  // ------------------------------------------------------------------- FSM
  always_ff @(posedge clk) begin
    if (rst) begin
      state        <= Q_IDLE;
      cycles       <= 32'd0;
      step_i       <= 7'd0;
      step_cnt     <= 7'd0;
      run_cycles_q <= 32'd0;
      x_q          <= 16'sd0;
      dt_q         <= 16'sd0;
      run_done     <= 1'b0;
      run_error    <= 1'b0;
      err_code     <= 2'd0;
    end else begin
      run_done  <= 1'b0;
      run_error <= 1'b0;

      // RUN_CYCLES: counted from accepted START to the last output commit
      if (state != Q_IDLE) cycles <= cycles + 32'd1;

      // run watchdog (contract 3.4: stop, ERROR, code 3, BUSY=0, outputs invalid)
      if ((state != Q_IDLE) && (state != Q_DONE) && (state != Q_ERR) &&
          (state != Q_ABORT) && (cycles >= cfg_timeout_cycles)) begin
        state <= Q_ABORT;
      end else begin
        case (state)
          Q_IDLE: begin
            if (start_pulse) begin
              cycles   <= 32'd0;
              step_i   <= 7'd0;
              step_cnt <= 7'd0;
              x_q      <= 16'sd0;
              dt_q     <= cfg_delta_t;
              state    <= cfg_clear_pre ? Q_PRECLEAR : Q_REQ;
            end
          end
          Q_PRECLEAR: state <= Q_REQ;
          Q_REQ: begin
            x_q   <= in_ram_data;
            state <= Q_START;
          end
          Q_START: state <= Q_WAIT;
          Q_WAIT: begin
            if (cell_done) begin
              step_cnt <= step_cnt + 7'd1;
              if (step_i == (cfg_n_steps - 7'd1)) begin
                run_cycles_q <= cycles;
                state        <= Q_DONE;
              end else begin
                step_i <= step_i + 7'd1;
                state  <= Q_REQ;
              end
            end
          end
          Q_DONE: begin
            run_done <= 1'b1;
            state    <= Q_IDLE;
          end
          Q_ABORT: state <= Q_ERR;
          Q_ERR: begin
            run_error <= 1'b1;
            err_code  <= 2'd3;
            state     <= Q_IDLE;
          end
          default: state <= Q_IDLE;
        endcase
      end
    end
  end
endmodule