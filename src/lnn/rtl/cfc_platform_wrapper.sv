`timescale 1ns/1ps
// cfc_platform_wrapper.sv -- frozen top level for the CfC PYNQ demo
// (contract section 3.3).  Single clock domain, active-high reset, AXI4-Lite
// slave + sticky done/error irq.  Internal instance names are frozen:
//   u_regs (cfc_axi_lite_regs), u_seq (cfc_sequence_ctrl),
//   u_cell (cfc_cell_core), u_state (cfc_state_regs).
module cfc_platform_wrapper (
  input  logic        clk,
  input  logic        rst,
  input  logic [11:0] s_axil_awaddr,
  input  logic        s_axil_awvalid,
  output logic        s_axil_awready,
  input  logic [31:0] s_axil_wdata,
  input  logic [3:0]  s_axil_wstrb,
  input  logic        s_axil_wvalid,
  output logic        s_axil_wready,
  output logic [1:0]  s_axil_bresp,
  output logic        s_axil_bvalid,
  input  logic        s_axil_bready,
  input  logic [11:0] s_axil_araddr,
  input  logic        s_axil_arvalid,
  output logic        s_axil_arready,
  output logic [31:0] s_axil_rdata,
  output logic [1:0]  s_axil_rresp,
  output logic        s_axil_rvalid,
  input  logic        s_axil_rready,
  output logic        irq
);
  // regs -> seq
  logic        start_pulse;
  logic [6:0]  cfg_n_steps;
  logic [15:0] cfg_delta_t;
  logic        cfg_clear_pre;
  logic        cfg_clear_each;
  logic [31:0] cfg_timeout_cycles;
  logic        clear_state_cmd;
  logic [5:0]  in_ram_addr;
  logic signed [15:0] in_ram_data;
  // seq -> regs / state
  logic        seq_done;
  logic        seq_error;
  logic [1:0]  seq_error_code;
  logic [31:0] seq_run_cycles;
  logic [6:0]  seq_step_count;
  logic        out_we;
  logic [5:0]  out_addr;
  logic signed [15:0] out_wdata;
  logic        state_clear;
  logic        state_commit;
  // state registers
  logic signed [15:0] h_cur_0, h_cur_1, h_cur_2, h_cur_3;
  logic signed [15:0] h_new_0, h_new_1, h_new_2, h_new_3;
  // seq <-> cell
  logic        cell_start;
  logic        cell_abort;
  logic        cell_done;
  logic        seq_busy;      // cfc_sequence_ctrl busy (status is kept in u_regs)
  logic        cell_busy;     // cfc_cell_core busy (observation only)
  logic signed [15:0] cell_x;
  logic signed [15:0] cell_h_old_0, cell_h_old_1, cell_h_old_2, cell_h_old_3;
  logic signed [15:0] cell_delta_t;
  logic signed [15:0] cell_y;

  cfc_axi_lite_regs u_regs (
    .clk                 (clk),
    .rst                 (rst),
    .s_axil_awaddr       (s_axil_awaddr),
    .s_axil_awvalid      (s_axil_awvalid),
    .s_axil_awready      (s_axil_awready),
    .s_axil_wdata        (s_axil_wdata),
    .s_axil_wstrb        (s_axil_wstrb),
    .s_axil_wvalid       (s_axil_wvalid),
    .s_axil_wready       (s_axil_wready),
    .s_axil_bresp        (s_axil_bresp),
    .s_axil_bvalid       (s_axil_bvalid),
    .s_axil_bready       (s_axil_bready),
    .s_axil_araddr       (s_axil_araddr),
    .s_axil_arvalid      (s_axil_arvalid),
    .s_axil_arready      (s_axil_arready),
    .s_axil_rdata        (s_axil_rdata),
    .s_axil_rresp        (s_axil_rresp),
    .s_axil_rvalid       (s_axil_rvalid),
    .s_axil_rready       (s_axil_rready),
    .irq                 (irq),
    .start_pulse         (start_pulse),
    .cfg_n_steps         (cfg_n_steps),
    .cfg_delta_t         (cfg_delta_t),
    .cfg_clear_pre       (cfg_clear_pre),
    .cfg_clear_each      (cfg_clear_each),
    .cfg_timeout_cycles  (cfg_timeout_cycles),
    .clear_state_cmd     (clear_state_cmd),
    .in_ram_addr         (in_ram_addr),
    .in_ram_data         (in_ram_data),
    .seq_done            (seq_done),
    .seq_error           (seq_error),
    .seq_error_code      (seq_error_code),
    .seq_run_cycles      (seq_run_cycles),
    .seq_step_count      (seq_step_count),
    .out_we              (out_we),
    .out_addr            (out_addr),
    .out_wdata           (out_wdata),
    .h_cur_0             (h_cur_0),
    .h_cur_1             (h_cur_1),
    .h_cur_2             (h_cur_2),
    .h_cur_3             (h_cur_3)
  );

  cfc_sequence_ctrl u_seq (
    .clk                 (clk),
    .rst                 (rst),
    .start_pulse         (start_pulse),
    .cfg_n_steps         (cfg_n_steps),
    .cfg_delta_t         (cfg_delta_t),
    .cfg_clear_pre       (cfg_clear_pre),
    .cfg_clear_each      (cfg_clear_each),
    .cfg_timeout_cycles  (cfg_timeout_cycles),
    .clear_state_cmd     (clear_state_cmd),
    .in_ram_addr         (in_ram_addr),
    .in_ram_data         (in_ram_data),
    .out_we              (out_we),
    .out_addr            (out_addr),
    .out_wdata           (out_wdata),
    .state_clear         (state_clear),
    .state_commit        (state_commit),
    .h_cur_0             (h_cur_0),
    .h_cur_1             (h_cur_1),
    .h_cur_2             (h_cur_2),
    .h_cur_3             (h_cur_3),
    .cell_start          (cell_start),
    .cell_x              (cell_x),
    .cell_h_old_0        (cell_h_old_0),
    .cell_h_old_1        (cell_h_old_1),
    .cell_h_old_2        (cell_h_old_2),
    .cell_h_old_3        (cell_h_old_3),
    .cell_delta_t        (cell_delta_t),
    .cell_y              (cell_y),
    .cell_done           (cell_done),
    .cell_abort          (cell_abort),
    .busy                (seq_busy),
    .run_done            (seq_done),
    .run_error           (seq_error),
    .err_code            (seq_error_code),
    .run_cycles          (seq_run_cycles),
    .step_count          (seq_step_count)
  );

  cfc_cell_core u_cell (
    .clk        (clk),
    .rst        (rst),
    .start      (cell_start),
    .abort      (cell_abort),
    .x_in       (cell_x),
    .h_old_0    (cell_h_old_0),
    .h_old_1    (cell_h_old_1),
    .h_old_2    (cell_h_old_2),
    .h_old_3    (cell_h_old_3),
    .delta_t    (cell_delta_t),
    .y_out      (cell_y),
    .h_new_0    (h_new_0),
    .h_new_1    (h_new_1),
    .h_new_2    (h_new_2),
    .h_new_3    (h_new_3),
    .done       (cell_done),
    .busy       (cell_busy)
  );

  cfc_state_regs u_state (
    .clk     (clk),
    .rst     (rst),
    .clear   (state_clear),
    .commit  (state_commit),
    .h_new_0 (h_new_0),
    .h_new_1 (h_new_1),
    .h_new_2 (h_new_2),
    .h_new_3 (h_new_3),
    .h_cur_0 (h_cur_0),
    .h_cur_1 (h_cur_1),
    .h_cur_2 (h_cur_2),
    .h_cur_3 (h_cur_3)
  );
endmodule