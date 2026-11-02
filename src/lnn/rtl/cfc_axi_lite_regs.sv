`timescale 1ns/1ps
// cfc_axi_lite_regs.sv -- AXI4-Lite slave, CFC1 register file, input/output
// vector RAMs (contract section 3.4).  Independent AW/W channels (any order or
// same cycle), WSTRB byte writes, 1-cycle read data, read back-pressure safe.
//
// Address map (12-bit byte address):
//   0x000 ID 0x43464331   0x004 VERSION 0x00010000   0x008 CONTROL (W1P)
//   0x00C STATUS          0x010 ERROR_CODE           0x014 N_STEPS
//   0x018 DELTA_T         0x01C MODE                 0x020 RUN_CYCLES
//   0x024 STEP_COUNT      0x028 TIMEOUT_CYCLES       0x02C CLOCK_HZ 300 MHz
//   0x080..0x08C H_FINAL (4 x sign-extended int16)
//   0x100..0x1FC INPUT  (64 x int16, low 16 bits)
//   0x200..0x2FC OUTPUT (64 x int16, sign-extended)
//
// NOTE: the generated parameter include is placed inside the module body;
// Vivado 2021.1 synthesis does not resolve file-scope `include through
// include_dirs (xsim does).  Same style as cfc_weight_rom.sv.

module cfc_axi_lite_regs (
  input  logic        clk,
  input  logic        rst,
  // AXI4-Lite slave (port list frozen by contract 3.3)
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
  output logic        irq,
  // to sequence controller (run configuration snapshot)
  output logic        start_pulse,
  output logic [6:0]  cfg_n_steps,
  output logic [15:0] cfg_delta_t,
  output logic        cfg_clear_pre,
  output logic        cfg_clear_each,
  output logic [31:0] cfg_timeout_cycles,
  output logic        clear_state_cmd,
  input  logic [5:0]  in_ram_addr,      // read address driven by cfc_sequence_ctrl
  output logic signed [15:0] in_ram_data, // registered input-RAM read data
  // from sequence controller
  input  logic        seq_done,
  input  logic        seq_error,
  input  logic [1:0]  seq_error_code,
  input  logic [31:0] seq_run_cycles,
  input  logic [6:0]  seq_step_count,
  input  logic        out_we,
  input  logic [5:0]  out_addr,
  input  logic signed [15:0] out_wdata,
  // current state registers (H_FINAL window)
  input  logic signed [15:0] h_cur_0,
  input  logic signed [15:0] h_cur_1,
  input  logic signed [15:0] h_cur_2,
  input  logic signed [15:0] h_cur_3
);
  // Generated parameter constants (must match export/gen and the manifest).
  `include "cfc_rom_params.svh"
  logic        aw_hs, w_hs;
  logic [11:0] awaddr_q;
  logic [31:0] wdata_q;
  logic [3:0]  wstrb_q;
  logic        bvalid_q;

  logic        rd_pending;
  logic [11:0] araddr_q;
  logic [31:0] rdata_q;
  logic        rvalid_q;

  logic [31:0] n_steps_q;
  logic [31:0] delta_t_q;
  logic [31:0] mode_q;
  logic [31:0] timeout_q;
  logic        busy_q, done_q, error_q;
  logic [1:0]  error_code_q;

  logic [6:0]  run_n_steps_r;
  logic [15:0] run_delta_t_r;
  logic        run_clear_pre_r, run_clear_each_r;
  logic [31:0] run_timeout_r;

  logic [15:0] in_ram  [0:63];
  logic [15:0] out_ram [0:63];

  logic        wr_commit;

  // --------------------------------------------------------------- outputs
  assign s_axil_awready = ~aw_hs & ~bvalid_q;
  assign s_axil_wready  = ~w_hs  & ~bvalid_q;
  assign s_axil_arready = ~rd_pending & ~rvalid_q;
  assign s_axil_bresp   = 2'b00;
  assign s_axil_rresp   = 2'b00;
  assign s_axil_rdata   = rdata_q;
  assign s_axil_bvalid  = bvalid_q;
  assign s_axil_rvalid  = rvalid_q;
  assign irq            = done_q | error_q;

  assign cfg_n_steps        = run_n_steps_r;
  assign cfg_delta_t        = run_delta_t_r;
  assign cfg_clear_pre      = run_clear_pre_r;
  assign cfg_clear_each     = run_clear_each_r;
  assign cfg_timeout_cycles = run_timeout_r;
  assign in_ram_data        = in_ram[in_ram_addr];

  assign wr_commit = aw_hs && w_hs && !bvalid_q;

  // ---------------------------------------------------- read data multiplexer
  function automatic logic [31:0] read_mux (input logic [11:0] a);
    logic [31:0] r;
    logic [15:0] hv;
    begin
      if (a[11:8] == 4'h1) begin
        r = {16'd0, in_ram[a[7:2]]};                                  // INPUT
      end else if (a[11:8] == 4'h2) begin
        r = {{16{out_ram[a[7:2]][15]}}, out_ram[a[7:2]]};             // OUTPUT
      end else if (a[11:4] == 8'h08) begin                            // H_FINAL
        case (a[3:2])
          2'd0:    hv = h_cur_0;
          2'd1:    hv = h_cur_1;
          2'd2:    hv = h_cur_2;
          default: hv = h_cur_3;
        endcase
        r = {{16{hv[15]}}, hv};
      end else begin
        case (a[11:2])
          10'h000: r = 32'h43464331;                 // ID "CFC1"
          10'h001: r = 32'h00010000;                 // VERSION
          10'h002: r = 32'h00000000;                 // CONTROL (W1P reads 0)
          10'h003: r = {29'd0, error_q, done_q, busy_q};
          10'h004: r = {30'd0, error_code_q};
          10'h005: r = n_steps_q;                 // RW 32-bit register
          10'h006: r = {16'd0, delta_t_q[15:0]};
          10'h007: r = {30'd0, mode_q[1:0]};
          10'h008: r = seq_run_cycles;
          10'h009: r = {25'd0, seq_step_count};
          10'h00A: r = timeout_q;
          10'h00B: r = 32'd300000000;                // CLOCK_HZ
          default: r = 32'h00000000;
        endcase
      end
      read_mux = r;
    end
  endfunction

  // -------------------------------------------------------- AXI handshaking
  always_ff @(posedge clk) begin
    if (rst) begin
      aw_hs     <= 1'b0;
      w_hs      <= 1'b0;
      awaddr_q  <= 12'd0;
      wdata_q   <= 32'd0;
      wstrb_q   <= 4'd0;
      bvalid_q  <= 1'b0;
      rd_pending<= 1'b0;
      araddr_q  <= 12'd0;
      rdata_q   <= 32'd0;
      rvalid_q  <= 1'b0;
    end else begin
      if (s_axil_awvalid && s_axil_awready) begin
        aw_hs    <= 1'b1;
        awaddr_q <= s_axil_awaddr;
      end
      if (s_axil_wvalid && s_axil_wready) begin
        w_hs    <= 1'b1;
        wdata_q <= s_axil_wdata;
        wstrb_q <= s_axil_wstrb;
      end
      if (wr_commit) begin
        aw_hs    <= 1'b0;
        w_hs     <= 1'b0;
        bvalid_q <= 1'b1;
      end else if (bvalid_q && s_axil_bready) begin
        bvalid_q <= 1'b0;
      end

      if (s_axil_arvalid && s_axil_arready) begin
        araddr_q   <= s_axil_araddr;
        rd_pending <= 1'b1;
      end
      if (rd_pending) begin
        rdata_q    <= read_mux(araddr_q);
        rvalid_q   <= 1'b1;
        rd_pending <= 1'b0;
      end else if (rvalid_q && s_axil_rready) begin
        rvalid_q <= 1'b0;
      end
    end
  end

  // --------------------------------------------- CSR state / data windows
  always_ff @(posedge clk) begin
    if (rst) begin
      n_steps_q      <= `CFC_N_STEPS_MAX;
      delta_t_q      <= `CFC_DT_INT;
      mode_q         <= 32'd1;
      timeout_q      <= 32'd1000000;
      busy_q         <= 1'b0;
      done_q         <= 1'b0;
      error_q        <= 1'b0;
      error_code_q   <= 2'd0;
      start_pulse    <= 1'b0;
      clear_state_cmd<= 1'b0;
      run_n_steps_r  <= `CFC_N_STEPS_MAX;
      run_delta_t_r  <= `CFC_DT_INT;
      run_clear_pre_r<= 1'b1;
      run_clear_each_r <= 1'b0;
      run_timeout_r  <= 32'd1000000;
      for (int i = 0; i < 64; i++) begin
        in_ram[i]  <= 16'd0;
        out_ram[i] <= 16'd0;
      end
    end else begin
      start_pulse     <= 1'b0;
      clear_state_cmd <= 1'b0;

      // output vector RAM write port (from the sequence controller)
      if (out_we) out_ram[out_addr] <= out_wdata;

      if (wr_commit) begin
        // ---------------- INPUT window: 0x100..0x1FC, low 16 bits, RW
        if (awaddr_q[11:8] == 4'h1) begin
          if (!busy_q) begin
            in_ram[awaddr_q[7:2]] <= {
              wstrb_q[1] ? wdata_q[15:8] : in_ram[awaddr_q[7:2]][15:8],
              wstrb_q[0] ? wdata_q[7:0]  : in_ram[awaddr_q[7:2]][7:0]
            };
          end
        end else begin
          case (awaddr_q[11:2])
            // ---------------- CONTROL W1P
            10'h002: begin
              if (wstrb_q[0]) begin
                if (wdata_q[2]) begin                    // CLEAR_STATUS
                  done_q       <= 1'b0;
                  error_q      <= 1'b0;
                  error_code_q <= 2'd0;
                end
                if (wdata_q[1]) begin                    // CLEAR_STATE (idle only)
                  if (!busy_q) clear_state_cmd <= 1'b1;
                end
                if (wdata_q[0]) begin                    // START
                  if (busy_q) begin
                    error_q      <= 1'b1;
                    error_code_q <= 2'd2;                // start while busy
                  end else if (!((n_steps_q >= 32'd1) && (n_steps_q <= `CFC_N_STEPS_MAX) &&
                                 (delta_t_q[15:0] >= 16'd1) &&
                                 (delta_t_q[15:0] <= 16'd32767))) begin
                    error_q      <= 1'b1;
                    error_code_q <= 2'd1;                // illegal configuration
                  end else begin
                    busy_q         <= 1'b1;
                    done_q         <= 1'b0;
                    error_q        <= 1'b0;
                    error_code_q   <= 2'd0;
                    start_pulse    <= 1'b1;
                    run_n_steps_r  <= n_steps_q[6:0];
                    run_delta_t_r  <= delta_t_q[15:0];
                    run_clear_pre_r<= mode_q[0];
                    run_clear_each_r <= mode_q[1];
                    run_timeout_r  <= timeout_q;
                  end
                end
              end
            end
            // ---------------- N_STEPS (legal 1..64)
            10'h005: if (!busy_q) begin
              if (wstrb_q[0]) n_steps_q[7:0]   <= wdata_q[7:0];
              if (wstrb_q[1]) n_steps_q[15:8]  <= wdata_q[15:8];
              if (wstrb_q[2]) n_steps_q[23:16] <= wdata_q[23:16];
              if (wstrb_q[3]) n_steps_q[31:24] <= wdata_q[31:24];
            end
            // ---------------- DELTA_T (low 16 bits, legal 1..32767)
            10'h006: if (!busy_q) begin
              if (wstrb_q[0]) delta_t_q[7:0]   <= wdata_q[7:0];
              if (wstrb_q[1]) delta_t_q[15:8]  <= wdata_q[15:8];
              if (wstrb_q[2]) delta_t_q[23:16] <= wdata_q[23:16];
              if (wstrb_q[3]) delta_t_q[31:24] <= wdata_q[31:24];
            end
            // ---------------- MODE
            10'h007: if (!busy_q) begin
              if (wstrb_q[0]) mode_q[7:0]   <= wdata_q[7:0];
              if (wstrb_q[1]) mode_q[15:8]  <= wdata_q[15:8];
              if (wstrb_q[2]) mode_q[23:16] <= wdata_q[23:16];
              if (wstrb_q[3]) mode_q[31:24] <= wdata_q[31:24];
            end
            // ---------------- TIMEOUT_CYCLES
            10'h00A: if (!busy_q) begin
              if (wstrb_q[0]) timeout_q[7:0]   <= wdata_q[7:0];
              if (wstrb_q[1]) timeout_q[15:8]  <= wdata_q[15:8];
              if (wstrb_q[2]) timeout_q[23:16] <= wdata_q[23:16];
              if (wstrb_q[3]) timeout_q[31:24] <= wdata_q[31:24];
            end
            default: ;   // writes to unimplemented offsets are ignored
          endcase
        end
      end

      // sequence controller run completion
      if (seq_done) begin
        busy_q <= 1'b0;
        done_q <= 1'b1;
      end
      if (seq_error) begin
        busy_q      <= 1'b0;
        error_q     <= 1'b1;
        error_code_q<= seq_error_code;
      end
    end
  end
endmodule