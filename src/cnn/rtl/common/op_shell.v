// -----------------------------------------------------------------------------
// op_shell.v
//
// ABI 1.0 fixed shell for parameterized CNN operators.
//
// T-OP2a status:
//   - AXI4-Lite CSR/control/error/timeout semantics are preserved.
//   - AXI4-Full read/write transport is implemented by
//     axi_burst_reader + axi_burst_writer, including 4 KB splitting.
//   - The normalized core memory service is wired through those engines.
//   - G0 identity is supplied by a separately instantiated identity_mover
//     test core; this shell does not claim to contain convolution math.
//   - Descriptor fields and complete CONV/QUANT/ACT words are captured into a
//     START-time snapshot. CSR writes during a run affect later starts only.
// -----------------------------------------------------------------------------

`timescale 1ns / 1ps
`include "op_abi_defs.vh"

module op_shell #(
    parameter integer AXIL_ADDR_WIDTH = 12,
    parameter integer AXI_ADDR_WIDTH  = 32,
    parameter integer AXI_DATA_WIDTH  = 128,
    parameter integer AXI_ID_WIDTH    = 4,
    parameter [31:0]  OP_ID           = 32'h4F504331,
    parameter [31:0]  ABI_VERSION     = 32'h00010000,
    parameter [31:0]  LAYOUT_ID       = 32'h00010101,
    parameter [31:0]  CORE_FEATURES   = 32'h00000001,
    parameter [31:0]  DEFAULT_TIMEOUT = 32'h01000000
) (
    input  wire                              clk,
    input  wire                              rst_n,

    // AXI4-Lite slave: control/status/configuration.
    input  wire [AXIL_ADDR_WIDTH-1:0]        s_axil_awaddr,
    input  wire                              s_axil_awvalid,
    output reg                               s_axil_awready,
    input  wire [31:0]                       s_axil_wdata,
    input  wire [3:0]                        s_axil_wstrb,
    input  wire                              s_axil_wvalid,
    output reg                               s_axil_wready,
    output reg  [1:0]                        s_axil_bresp,
    output reg                               s_axil_bvalid,
    input  wire                              s_axil_bready,
    input  wire [AXIL_ADDR_WIDTH-1:0]        s_axil_araddr,
    input  wire                              s_axil_arvalid,
    output reg                               s_axil_arready,
    output reg  [31:0]                       s_axil_rdata,
    output reg  [1:0]                        s_axil_rresp,
    output reg                               s_axil_rvalid,
    input  wire                              s_axil_rready,

    // AXI4-Full master: PS DDR read/write through burst engines.
    output wire [AXI_ID_WIDTH-1:0]           m_axi_awid,
    output wire [AXI_ADDR_WIDTH-1:0]         m_axi_awaddr,
    output wire [7:0]                        m_axi_awlen,
    output wire [2:0]                        m_axi_awsize,
    output wire [1:0]                        m_axi_awburst,
    output wire                              m_axi_awlock,
    output wire [3:0]                        m_axi_awcache,
    output wire [2:0]                        m_axi_awprot,
    output wire [3:0]                        m_axi_awqos,
    output wire [3:0]                        m_axi_awregion,
    output wire                              m_axi_awvalid,
    input  wire                              m_axi_awready,
    output wire [AXI_DATA_WIDTH-1:0]         m_axi_wdata,
    output wire [AXI_DATA_WIDTH/8-1:0]       m_axi_wstrb,
    output wire                              m_axi_wlast,
    output wire                              m_axi_wvalid,
    input  wire                              m_axi_wready,
    input  wire [AXI_ID_WIDTH-1:0]           m_axi_bid,
    input  wire [1:0]                        m_axi_bresp,
    input  wire                              m_axi_bvalid,
    output wire                              m_axi_bready,
    output wire [AXI_ID_WIDTH-1:0]           m_axi_arid,
    output wire [AXI_ADDR_WIDTH-1:0]         m_axi_araddr,
    output wire [7:0]                        m_axi_arlen,
    output wire [2:0]                        m_axi_arsize,
    output wire [1:0]                        m_axi_arburst,
    output wire                              m_axi_arlock,
    output wire [3:0]                        m_axi_arcache,
    output wire [2:0]                        m_axi_arprot,
    output wire [3:0]                        m_axi_arqos,
    output wire [3:0]                        m_axi_arregion,
    output wire                              m_axi_arvalid,
    input  wire                              m_axi_arready,
    input  wire [AXI_ID_WIDTH-1:0]           m_axi_rid,
    input  wire [AXI_DATA_WIDTH-1:0]         m_axi_rdata,
    input  wire [1:0]                        m_axi_rresp,
    input  wire                              m_axi_rlast,
    input  wire                              m_axi_rvalid,
    output wire                              m_axi_rready,

    // Locked configuration snapshot presented to op_core.
    output wire [31:0]                       cfg_in_addr_byte,
    output wire [31:0]                       cfg_weight_addr_byte,
    output wire [31:0]                       cfg_bias_addr_byte,
    output wire [31:0]                       cfg_quant_addr_byte,
    output wire [31:0]                       cfg_out_addr_byte,
    output wire [15:0]                       cfg_h,
    output wire [15:0]                       cfg_w,
    output wire [15:0]                       cfg_cin,
    output wire [15:0]                       cfg_cout,
    output wire [1:0]                        cfg_stride,
    output wire [1:0]                        cfg_pad,
    output wire [7:0]                        cfg_x_zero_point,
    output wire [31:0]                       cfg_in_row_stride_bytes,
    output wire [31:0]                       cfg_out_row_stride_bytes,
    output wire [31:0]                       cfg_in_bytes,
    output wire [31:0]                       cfg_weight_bytes,
    output wire [31:0]                       cfg_bias_bytes,
    output wire [31:0]                       cfg_quant_bytes,
    output wire [31:0]                       cfg_out_bytes,
    output wire [31:0]                       cfg_conv_cfg,
    output wire [31:0]                       cfg_quant_cfg,
    output wire [31:0]                       cfg_act_cfg,

    // op_core control/status boundary.
    output reg                               core_start,
    output reg                               core_soft_reset,
    output reg                               core_abort,
    input  wire                              core_busy,
    input  wire                              core_done,
    input  wire                              core_error,
    input  wire [15:0]                       core_error_code,

    // Normalized core memory service, backed by the T-OP2a AXI4-Full engines.
    input  wire                              core_rd_cmd_valid,
    output wire                              core_rd_cmd_ready,
    input  wire [31:0]                       core_rd_cmd_addr_byte,
    input  wire [15:0]                       core_rd_cmd_bytes,
    input  wire [7:0]                        core_rd_cmd_tag,
    output wire                              core_rd_data_valid,
    input  wire                              core_rd_data_ready,
    output wire [127:0]                      core_rd_data,
    output wire                              core_rd_data_last,
    output wire                              core_rd_data_error,
    output wire [15:0]                       core_rd_data_error_code,
    output wire [7:0]                        core_rd_data_tag,

    input  wire                              core_wr_cmd_valid,
    output wire                              core_wr_cmd_ready,
    input  wire [31:0]                       core_wr_cmd_addr_byte,
    input  wire [15:0]                       core_wr_cmd_bytes,
    input  wire [7:0]                        core_wr_cmd_tag,
    input  wire                              core_wr_data_valid,
    output wire                              core_wr_data_ready,
    input  wire [127:0]                      core_wr_data,
    input  wire                              core_wr_data_last,
    input  wire [7:0]                        core_wr_data_tag,
    output wire                              core_wr_rsp_valid,
    input  wire                              core_wr_rsp_ready,
    output wire                              core_wr_rsp_error,
    output wire                              core_wr_rsp_aborted,
    output wire [15:0]                       core_wr_rsp_error_code,
    output wire [7:0]                        core_wr_rsp_tag
);

    localparam [1:0] AXIL_RESP_OKAY   = 2'b00;
    localparam [1:0] AXIL_RESP_SLVERR = 2'b10;

    localparam [1:0] S_IDLE       = 2'd0;
    localparam [1:0] S_WAIT_BUSY  = 2'd1;
    localparam [1:0] S_RUN        = 2'd2;
    localparam [1:0] S_ABORT_WAIT = 2'd3;

    // Error encodings come from op_abi_defs.vh so the shell cannot drift.

    reg                    aw_hold_valid;
    reg [AXIL_ADDR_WIDTH-1:0] aw_hold_addr;
    reg                    w_hold_valid;
    reg [31:0]             w_hold_data;
    reg [3:0]              w_hold_strb;

    reg [31:0]             run_id;
    reg [31:0]             timeout_cycles;
    reg [31:0]             in_addr_byte;
    reg [31:0]             weight_addr_byte;
    reg [31:0]             bias_addr_byte;
    reg [31:0]             quant_addr_byte;
    reg [31:0]             out_addr_byte;
    reg [15:0]             cfg_h_reg;
    reg [15:0]             cfg_w_reg;
    reg [15:0]             cfg_cin_reg;
    reg [15:0]             cfg_cout_reg;
    reg [1:0]              cfg_stride_reg;
    reg [1:0]              cfg_pad_reg;
    reg [7:0]              cfg_x_zero_point_reg;
    reg [31:0]             in_row_stride_bytes_reg;
    reg [31:0]             out_row_stride_bytes_reg;
    reg [31:0]             in_bytes_reg;
    reg [31:0]             weight_bytes_reg;
    reg [31:0]             bias_bytes_reg;
    reg [31:0]             quant_bytes_reg;
    reg [31:0]             out_bytes_reg;
    reg [31:0]             conv_cfg_reg;
    reg [31:0]             quant_cfg_reg;
    reg [31:0]             act_cfg_reg;

    // START-time snapshot presented to op_core.  CSR writes during a run
    // update the live configuration registers but MUST NOT mutate this copy.
    reg [31:0]             snap_in_addr_byte;
    reg [31:0]             snap_weight_addr_byte;
    reg [31:0]             snap_bias_addr_byte;
    reg [31:0]             snap_quant_addr_byte;
    reg [31:0]             snap_out_addr_byte;
    reg [15:0]             snap_h;
    reg [15:0]             snap_w;
    reg [15:0]             snap_cin;
    reg [15:0]             snap_cout;
    reg [1:0]              snap_stride;
    reg [1:0]              snap_pad;
    reg [7:0]              snap_x_zero_point;
    reg [31:0]             snap_in_row_stride_bytes;
    reg [31:0]             snap_out_row_stride_bytes;
    reg [31:0]             snap_in_bytes;
    reg [31:0]             snap_weight_bytes;
    reg [31:0]             snap_bias_bytes;
    reg [31:0]             snap_quant_bytes;
    reg [31:0]             snap_out_bytes;
    reg [31:0]             snap_conv_cfg;
    reg [31:0]             snap_quant_cfg;
    reg [31:0]             snap_act_cfg;

    reg [3:0]              ctrl_cmd_pending;
    reg [1:0]              shell_state;
    reg [1:0]              soft_reset_timer;
    reg [31:0]             timeout_counter;
    reg                    status_done;
    reg                    status_error;
    reg                    status_timeout;
    reg [15:0]             error_code_reg;
    reg [31:0]             error_info_reg;
    reg [2:0]              irq_status;

    reg                    fault_pending;
    reg [15:0]             fault_code_pending;
    reg [31:0]             fault_info_pending;
    reg                    fault_timeout_pending;

    wire [31:0]            reader_stat_burst_count;
    wire [31:0]            writer_stat_burst_count;
    wire [31:0]            reader_stat_last_addr;
    wire [31:0]            writer_stat_last_addr;
    wire [1:0]             reader_stat_last_resp;
    wire [1:0]             writer_stat_last_resp;
    reg                    stat_clear;

    wire status_busy = (shell_state != S_IDLE) || core_busy;
    wire status_idle = (shell_state == S_IDLE) && !core_busy;
    wire status_ready = status_idle && !status_busy && !status_error && !status_timeout;
    wire [31:0] status_word = {25'b0, status_ready, 1'b0, status_timeout,
                               status_error, status_done, status_busy, status_idle};
    wire [31:0] debug_state_word = {22'b0, core_error_code[7:0], shell_state};
    wire ctrl_onehot = (w_hold_data[3:0] != 4'b0000) &&
                       ((w_hold_data[3:0] & (w_hold_data[3:0] - 4'b0001)) == 4'b0000);

    assign cfg_in_addr_byte          = snap_in_addr_byte;
    assign cfg_weight_addr_byte      = snap_weight_addr_byte;
    assign cfg_bias_addr_byte        = snap_bias_addr_byte;
    assign cfg_quant_addr_byte       = snap_quant_addr_byte;
    assign cfg_out_addr_byte         = snap_out_addr_byte;
    assign cfg_h                     = snap_h;
    assign cfg_w                     = snap_w;
    assign cfg_cin                   = snap_cin;
    assign cfg_cout                  = snap_cout;
    assign cfg_stride                = snap_stride;
    assign cfg_pad                   = snap_pad;
    assign cfg_x_zero_point          = snap_x_zero_point;
    assign cfg_in_row_stride_bytes   = snap_in_row_stride_bytes;
    assign cfg_out_row_stride_bytes  = snap_out_row_stride_bytes;
    assign cfg_in_bytes              = snap_in_bytes;
    assign cfg_weight_bytes          = snap_weight_bytes;
    assign cfg_bias_bytes            = snap_bias_bytes;
    assign cfg_quant_bytes           = snap_quant_bytes;
    assign cfg_out_bytes             = snap_out_bytes;
    assign cfg_conv_cfg              = snap_conv_cfg;
    assign cfg_quant_cfg             = snap_quant_cfg;
    assign cfg_act_cfg               = snap_act_cfg;

    // ---------------------------------------------------------------------
    // AXI4-Lite write/read state and CSR updates.
    // ---------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aw_hold_valid       <= 1'b0;
            aw_hold_addr        <= {AXIL_ADDR_WIDTH{1'b0}};
            w_hold_valid        <= 1'b0;
            w_hold_data         <= 32'b0;
            w_hold_strb         <= 4'b0;
            s_axil_bvalid       <= 1'b0;
            s_axil_bresp        <= AXIL_RESP_OKAY;
            s_axil_rvalid       <= 1'b0;
            s_axil_rdata        <= 32'b0;
            s_axil_rresp        <= AXIL_RESP_OKAY;

            run_id              <= 32'b0;
            timeout_cycles      <= DEFAULT_TIMEOUT;
            in_addr_byte        <= 32'b0;
            weight_addr_byte    <= 32'b0;
            bias_addr_byte      <= 32'b0;
            quant_addr_byte     <= 32'b0;
            out_addr_byte       <= 32'b0;
            cfg_h_reg           <= 16'b0;
            cfg_w_reg           <= 16'b0;
            cfg_cin_reg         <= 16'b0;
            cfg_cout_reg        <= 16'b0;
            cfg_stride_reg      <= 2'b00;
            cfg_pad_reg         <= 2'b00;
            cfg_x_zero_point_reg <= 8'b0;
            in_row_stride_bytes_reg  <= 32'b0;
            out_row_stride_bytes_reg <= 32'b0;
            in_bytes_reg        <= 32'b0;
            weight_bytes_reg    <= 32'b0;
            bias_bytes_reg      <= 32'b0;
            quant_bytes_reg     <= 32'b0;
            out_bytes_reg       <= 32'b0;
            conv_cfg_reg        <= 32'h00000400;
            quant_cfg_reg       <= 32'b0;
            act_cfg_reg         <= 32'h00000000;

            snap_in_addr_byte          <= 32'b0;
            snap_weight_addr_byte      <= 32'b0;
            snap_bias_addr_byte        <= 32'b0;
            snap_quant_addr_byte       <= 32'b0;
            snap_out_addr_byte         <= 32'b0;
            snap_h                     <= 16'b0;
            snap_w                     <= 16'b0;
            snap_cin                   <= 16'b0;
            snap_cout                  <= 16'b0;
            snap_stride                <= 2'b00;
            snap_pad                   <= 2'b00;
            snap_x_zero_point          <= 8'b0;
            snap_in_row_stride_bytes   <= 32'b0;
            snap_out_row_stride_bytes  <= 32'b0;
            snap_in_bytes              <= 32'b0;
            snap_weight_bytes          <= 32'b0;
            snap_bias_bytes            <= 32'b0;
            snap_quant_bytes           <= 32'b0;
            snap_out_bytes             <= 32'b0;
            snap_conv_cfg              <= 32'h00000400;
            snap_quant_cfg             <= 32'b0;
            snap_act_cfg               <= 32'b0;

            ctrl_cmd_pending    <= 4'b0;
            shell_state         <= S_IDLE;
            soft_reset_timer    <= 2'b0;
            timeout_counter     <= 32'b0;
            core_start          <= 1'b0;
            core_soft_reset     <= 1'b0;
            core_abort          <= 1'b0;
            status_done         <= 1'b0;
            status_error        <= 1'b0;
            status_timeout      <= 1'b0;
            error_code_reg      <= `OP_E_NONE;
            error_info_reg      <= 32'b0;
            irq_status          <= 3'b0;
            fault_pending       <= 1'b0;
            fault_code_pending  <= `OP_E_NONE;
            fault_info_pending  <= 32'b0;
            fault_timeout_pending <= 1'b0;
            stat_clear            <= 1'b0;
        end else begin
            ctrl_cmd_pending <= 4'b0;
            stat_clear       <= 1'b0;

            if (fault_pending && !status_error) begin
                status_error        <= 1'b1;
                status_timeout      <= status_timeout | fault_timeout_pending;
                error_code_reg      <= fault_code_pending;
                error_info_reg      <= fault_info_pending;
                irq_status[1]       <= 1'b1;
                if (fault_timeout_pending) begin
                    irq_status[2] <= 1'b1;
                end
                fault_pending       <= 1'b0;
                fault_timeout_pending <= 1'b0;
            end

            // AXI4-Lite write capture.
            if (s_axil_awready && s_axil_awvalid) begin
                aw_hold_valid <= 1'b1;
                aw_hold_addr  <= s_axil_awaddr;
            end
            if (s_axil_wready && s_axil_wvalid) begin
                w_hold_valid <= 1'b1;
                w_hold_data  <= s_axil_wdata;
                w_hold_strb  <= s_axil_wstrb;
            end

            if (aw_hold_valid && w_hold_valid && !s_axil_bvalid) begin
                s_axil_bvalid <= 1'b1;
                s_axil_bresp  <= AXIL_RESP_OKAY;
                aw_hold_valid <= 1'b0;
                w_hold_valid  <= 1'b0;

                if ((aw_hold_addr[1:0] != 2'b00) || (w_hold_strb != 4'b1111)) begin
                    s_axil_bresp <= AXIL_RESP_SLVERR;
                    if (!status_error && !fault_pending) begin
                        fault_pending      <= 1'b1;
                        fault_code_pending <= `OP_E_ILLEGAL_CTRL;
                        fault_info_pending <= {20'b0, aw_hold_addr};
                    end
                end else begin
                    case (aw_hold_addr)
                        12'h008: begin
                            if ((w_hold_data[31:4] != 28'b0) || !ctrl_onehot) begin
                                s_axil_bresp <= AXIL_RESP_SLVERR;
                                if (!status_error && !fault_pending) begin
                                    fault_pending      <= 1'b1;
                                    fault_code_pending <= `OP_E_ILLEGAL_CTRL;
                                    fault_info_pending <= w_hold_data;
                                end
                            end else begin
                                ctrl_cmd_pending <= w_hold_data[3:0];
                            end
                        end
                        12'h018: irq_status <= irq_status & ~w_hold_data[2:0];
                        12'h01C: timeout_cycles <= w_hold_data;
                        12'h020: run_id <= w_hold_data;
                        12'h024: in_addr_byte <= w_hold_data;
                        12'h028: weight_addr_byte <= w_hold_data;
                        12'h02C: bias_addr_byte <= w_hold_data;
                        12'h030: quant_addr_byte <= w_hold_data;
                        12'h034: out_addr_byte <= w_hold_data;
                        12'h038: begin
                            cfg_h_reg <= w_hold_data[15:0];
                            cfg_w_reg <= w_hold_data[31:16];
                        end
                        12'h03C: begin
                            cfg_cin_reg  <= w_hold_data[15:0];
                            cfg_cout_reg <= w_hold_data[31:16];
                        end
                        12'h040: begin
                            conv_cfg_reg   <= w_hold_data;
                            cfg_stride_reg <= w_hold_data[3:2];
                            cfg_pad_reg    <= w_hold_data[5:4];
                        end
                        12'h044: begin
                            quant_cfg_reg        <= w_hold_data;
                            cfg_x_zero_point_reg <= w_hold_data[7:0];
                        end
                        12'h048: act_cfg_reg <= w_hold_data;
                        12'h04C: in_row_stride_bytes_reg <= w_hold_data;
                        12'h050: out_row_stride_bytes_reg <= w_hold_data;
                        12'h054: in_bytes_reg <= w_hold_data;
                        12'h058: weight_bytes_reg <= w_hold_data;
                        12'h05C: bias_bytes_reg <= w_hold_data;
                        12'h060: quant_bytes_reg <= w_hold_data;
                        12'h064: out_bytes_reg <= w_hold_data;
                        default: begin
                            // Reserved/read-only write: ignore and return OKAY.
                        end
                    endcase
                end
            end

            if (s_axil_bvalid && s_axil_bready) begin
                s_axil_bvalid <= 1'b0;
            end

            // AXI4-Lite read acceptance.
            if (s_axil_arready && s_axil_arvalid) begin
                s_axil_rvalid <= 1'b1;
                s_axil_rresp <= AXIL_RESP_OKAY;
                if (s_axil_araddr[1:0] != 2'b00) begin
                    s_axil_rresp <= AXIL_RESP_SLVERR;
                    s_axil_rdata <= 32'b0;
                end else begin
                    case (s_axil_araddr)
                        12'h000: s_axil_rdata <= OP_ID;
                        12'h004: s_axil_rdata <= ABI_VERSION;
                        12'h008: s_axil_rdata <= 32'b0;
                        12'h00C: s_axil_rdata <= status_word;
                        12'h010: s_axil_rdata <= {16'b0, error_code_reg};
                        12'h014: s_axil_rdata <= error_info_reg;
                        12'h018: s_axil_rdata <= {29'b0, irq_status};
                        12'h01C: s_axil_rdata <= timeout_cycles;
                        12'h020: s_axil_rdata <= run_id;
                        12'h024: s_axil_rdata <= in_addr_byte;
                        12'h028: s_axil_rdata <= weight_addr_byte;
                        12'h02C: s_axil_rdata <= bias_addr_byte;
                        12'h030: s_axil_rdata <= quant_addr_byte;
                        12'h034: s_axil_rdata <= out_addr_byte;
                        12'h038: s_axil_rdata <= {cfg_w_reg, cfg_h_reg};
                        12'h03C: s_axil_rdata <= {cfg_cout_reg, cfg_cin_reg};
                        12'h040: s_axil_rdata <= conv_cfg_reg;
                        12'h044: s_axil_rdata <= quant_cfg_reg;
                        12'h048: s_axil_rdata <= act_cfg_reg;
                        12'h04C: s_axil_rdata <= in_row_stride_bytes_reg;
                        12'h050: s_axil_rdata <= out_row_stride_bytes_reg;
                        12'h054: s_axil_rdata <= in_bytes_reg;
                        12'h058: s_axil_rdata <= weight_bytes_reg;
                        12'h05C: s_axil_rdata <= bias_bytes_reg;
                        12'h060: s_axil_rdata <= quant_bytes_reg;
                        12'h064: s_axil_rdata <= out_bytes_reg;
                        12'h068: s_axil_rdata <= reader_stat_burst_count;
                        12'h06C: s_axil_rdata <= writer_stat_burst_count;
                        12'h070: s_axil_rdata <= reader_stat_last_addr;
                        12'h074: s_axil_rdata <= writer_stat_last_addr;
                        12'h078: s_axil_rdata <= {30'b0, writer_stat_last_resp,
                                                   reader_stat_last_resp};
                        12'h07C: s_axil_rdata <= debug_state_word;
                        12'h080: s_axil_rdata <= LAYOUT_ID;
                        12'h084: s_axil_rdata <= CORE_FEATURES;
                        default: s_axil_rdata <= 32'b0;
                    endcase
                end
            end

            if (s_axil_rvalid && s_axil_rready) begin
                s_axil_rvalid <= 1'b0;
            end

            // Control FSM, soft reset, clear commands and timeout.
            if (ctrl_cmd_pending[1]) begin
                shell_state         <= S_IDLE;
                soft_reset_timer    <= 2'd2;
                core_start          <= 1'b0;
                core_abort          <= 1'b0;
                core_soft_reset     <= 1'b1;
                status_done         <= 1'b0;
                status_error        <= 1'b0;
                status_timeout      <= 1'b0;
                error_code_reg      <= `OP_E_NONE;
                error_info_reg      <= 32'b0;
                irq_status          <= 3'b0;
                fault_pending       <= 1'b0;
                fault_timeout_pending <= 1'b0;
                timeout_counter     <= 32'b0;
                stat_clear          <= 1'b1;
            end else if (soft_reset_timer != 2'b0) begin
                if (soft_reset_timer == 2'd1) begin
                    soft_reset_timer <= 2'b0;
                    core_soft_reset  <= 1'b0;
                end else begin
                    soft_reset_timer <= soft_reset_timer - 2'd1;
                    core_soft_reset  <= 1'b1;
                end
            end else if (ctrl_cmd_pending[3]) begin
                if ((shell_state == S_IDLE) && !core_busy) begin
                    status_error    <= 1'b0;
                    status_timeout  <= 1'b0;
                    error_code_reg  <= `OP_E_NONE;
                    error_info_reg  <= 32'b0;
                    irq_status[1]   <= 1'b0;
                    irq_status[2]   <= 1'b0;
                    fault_pending   <= 1'b0;
                    fault_timeout_pending <= 1'b0;
                end else if (!status_error && !fault_pending) begin
                    fault_pending      <= 1'b1;
                    fault_code_pending <= `OP_E_ILLEGAL_CTRL;
                    fault_info_pending <= 32'h0000_0003;
                end
            end else if (ctrl_cmd_pending[2]) begin
                if ((shell_state == S_IDLE) && !core_busy) begin
                    status_done <= 1'b0;
                    irq_status[0] <= 1'b0;
                end else if (!status_error && !fault_pending) begin
                    fault_pending      <= 1'b1;
                    fault_code_pending <= `OP_E_ILLEGAL_CTRL;
                    fault_info_pending <= 32'h0000_0002;
                end
            end else if (ctrl_cmd_pending[0]) begin
                if (status_ready) begin
                    snap_in_addr_byte         <= in_addr_byte;
                    snap_weight_addr_byte     <= weight_addr_byte;
                    snap_bias_addr_byte       <= bias_addr_byte;
                    snap_quant_addr_byte      <= quant_addr_byte;
                    snap_out_addr_byte        <= out_addr_byte;
                    snap_h                     <= cfg_h_reg;
                    snap_w                     <= cfg_w_reg;
                    snap_cin                   <= cfg_cin_reg;
                    snap_cout                  <= cfg_cout_reg;
                    snap_stride                <= cfg_stride_reg;
                    snap_pad                   <= cfg_pad_reg;
                    snap_x_zero_point          <= cfg_x_zero_point_reg;
                    snap_in_row_stride_bytes   <= in_row_stride_bytes_reg;
                    snap_out_row_stride_bytes  <= out_row_stride_bytes_reg;
                    snap_in_bytes              <= in_bytes_reg;
                    snap_weight_bytes          <= weight_bytes_reg;
                    snap_bias_bytes            <= bias_bytes_reg;
                    snap_quant_bytes           <= quant_bytes_reg;
                    snap_out_bytes             <= out_bytes_reg;
                    snap_conv_cfg              <= conv_cfg_reg;
                    snap_quant_cfg             <= quant_cfg_reg;
                    snap_act_cfg               <= act_cfg_reg;
                    shell_state    <= S_WAIT_BUSY;
                    core_start     <= 1'b1;
                    core_abort     <= 1'b0;
                    status_done    <= 1'b0;
                    irq_status[0]  <= 1'b0;
                    timeout_counter <= 32'b0;
                    stat_clear      <= 1'b1;
                end else if (!status_error && !fault_pending) begin
                    fault_pending      <= 1'b1;
                    fault_code_pending <= `OP_E_ILLEGAL_START;
                    fault_info_pending <= {30'b0, shell_state};
                end
            end else begin
                case (shell_state)
                    S_IDLE: begin
                        core_start <= 1'b0;
                        core_abort <= 1'b0;
                        timeout_counter <= 32'b0;
                    end
                    S_WAIT_BUSY: begin
                        if (core_error) begin
                            core_start <= 1'b0;
                            shell_state <= S_IDLE;
                            if (!status_error && !fault_pending) begin
                                fault_pending      <= 1'b1;
                                fault_code_pending <= (core_error_code == 16'h0000) ? `OP_E_CORE_INTERNAL : core_error_code;
                                fault_info_pending <= 32'h0000_0004;
                            end
                        end else if (core_busy) begin
                            core_start <= 1'b0;
                            shell_state <= S_RUN;
                        end else if ((timeout_cycles != 32'b0) && (timeout_counter >= timeout_cycles)) begin
                            core_start <= 1'b0;
                            core_abort <= 1'b1;
                            shell_state <= S_ABORT_WAIT;
                            if (!status_error && !fault_pending) begin
                                fault_pending          <= 1'b1;
                                fault_code_pending     <= `OP_E_TIMEOUT;
                                fault_info_pending     <= timeout_counter;
                                fault_timeout_pending  <= 1'b1;
                            end
                        end else begin
                            timeout_counter <= timeout_counter + 32'b1;
                        end
                    end
                    S_RUN: begin
                        core_start <= 1'b0;
                        if (core_error) begin
                            core_abort <= 1'b1;
                            shell_state <= S_ABORT_WAIT;
                            if (!status_error && !fault_pending) begin
                                fault_pending      <= 1'b1;
                                fault_code_pending <= (core_error_code == 16'h0000) ? `OP_E_CORE_INTERNAL : core_error_code;
                                fault_info_pending <= 32'h0000_0005;
                            end
                        end else if (core_done) begin
                            status_done <= 1'b1;
                            irq_status[0] <= 1'b1;
                            shell_state <= S_IDLE;
                            core_abort  <= 1'b0;
                            timeout_counter <= 32'b0;
                        end else if ((timeout_cycles != 32'b0) && (timeout_counter >= timeout_cycles)) begin
                            core_abort <= 1'b1;
                            shell_state <= S_ABORT_WAIT;
                            if (!status_error && !fault_pending) begin
                                fault_pending          <= 1'b1;
                                fault_code_pending     <= `OP_E_TIMEOUT;
                                fault_info_pending     <= timeout_counter;
                                fault_timeout_pending  <= 1'b1;
                            end
                        end else begin
                            timeout_counter <= timeout_counter + 32'b1;
                        end
                    end
                    S_ABORT_WAIT: begin
                        core_start <= 1'b0;
                        if (!core_busy) begin
                            core_abort <= 1'b0;
                            shell_state <= S_IDLE;
                            timeout_counter <= 32'b0;
                        end
                    end
                    default: shell_state <= S_IDLE;
                endcase
            end
        end
    end

    // AXI4-Lite ready defaults.
    always @(*) begin
        if (!aw_hold_valid && !s_axil_bvalid) begin
            s_axil_awready = 1'b1;
        end else begin
            s_axil_awready = 1'b0;
        end
        if (!w_hold_valid && !s_axil_bvalid) begin
            s_axil_wready = 1'b1;
        end else begin
            s_axil_wready = 1'b0;
        end
        if (!s_axil_rvalid) begin
            s_axil_arready = 1'b1;
        end else begin
            s_axil_arready = 1'b0;
        end
    end

    // ---------------------------------------------------------------------
    // AXI4-Full transport engines.
    //
    // The reader and writer each own one AXI ID and one outstanding INCR
    // burst. Both reject non-16-byte-aligned requests and lengths before
    // issuing any AXI command. The splitter inside each engine enforces the
    // 16-beat/256-byte and 4 KB boundaries.
    // ---------------------------------------------------------------------
    wire reader_idle;
    wire writer_idle;
    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_engine_idle = reader_idle | writer_idle;
    /* verilator lint_on UNUSEDSIGNAL */

    axi_burst_reader #(
        .AXI_ADDR_WIDTH  (AXI_ADDR_WIDTH),
        .AXI_DATA_WIDTH  (AXI_DATA_WIDTH),
        .AXI_ID_WIDTH    (AXI_ID_WIDTH),
        .MAX_BURST_BEATS (`OP_MAX_BURST_BEATS),
        .M_ARID_VALUE    (`OP_AXI_RID_READ)
    ) u_axi_burst_reader (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .abort                 (core_abort),
        .stat_clear            (stat_clear),
        .cmd_valid             (core_rd_cmd_valid),
        .cmd_ready             (core_rd_cmd_ready),
        .cmd_addr_byte         (core_rd_cmd_addr_byte),
        .cmd_bytes             (core_rd_cmd_bytes),
        .cmd_tag               (core_rd_cmd_tag),
        .data_valid            (core_rd_data_valid),
        .data_ready            (core_rd_data_ready),
        .data                  (core_rd_data),
        .data_last             (core_rd_data_last),
        .data_error            (core_rd_data_error),
        .data_error_code       (core_rd_data_error_code),
        .data_tag              (core_rd_data_tag),
        .m_arid                (m_axi_arid),
        .m_araddr              (m_axi_araddr),
        .m_arlen               (m_axi_arlen),
        .m_arsize              (m_axi_arsize),
        .m_arburst             (m_axi_arburst),
        .m_arlock              (m_axi_arlock),
        .m_arcache             (m_axi_arcache),
        .m_arprot              (m_axi_arprot),
        .m_arqos               (m_axi_arqos),
        .m_arregion            (m_axi_arregion),
        .m_arvalid             (m_axi_arvalid),
        .m_arready             (m_axi_arready),
        .m_rid                 (m_axi_rid),
        .m_rdata               (m_axi_rdata),
        .m_rresp               (m_axi_rresp),
        .m_rlast               (m_axi_rlast),
        .m_rvalid              (m_axi_rvalid),
        .m_rready              (m_axi_rready),
        .stat_burst_count      (reader_stat_burst_count),
        .stat_last_addr        (reader_stat_last_addr),
        .stat_last_resp        (reader_stat_last_resp),
        .idle                  (reader_idle)
    );

    axi_burst_writer #(
        .AXI_ADDR_WIDTH  (AXI_ADDR_WIDTH),
        .AXI_DATA_WIDTH  (AXI_DATA_WIDTH),
        .AXI_ID_WIDTH    (AXI_ID_WIDTH),
        .MAX_BURST_BEATS (`OP_MAX_BURST_BEATS),
        .M_AWID_VALUE    (`OP_AXI_ID_WRITE)
    ) u_axi_burst_writer (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .abort                 (1'b0),
        .stat_clear            (stat_clear),
        .cmd_valid             (core_wr_cmd_valid),
        .cmd_ready             (core_wr_cmd_ready),
        .cmd_addr_byte         (core_wr_cmd_addr_byte),
        .cmd_bytes             (core_wr_cmd_bytes),
        .cmd_tag               (core_wr_cmd_tag),
        .data_valid            (core_wr_data_valid),
        .data_ready            (core_wr_data_ready),
        .data                  (core_wr_data),
        .data_last             (core_wr_data_last),
        .data_tag              (core_wr_data_tag),
        .rsp_valid             (core_wr_rsp_valid),
        .rsp_ready             (core_wr_rsp_ready),
        .rsp_error             (core_wr_rsp_error),
        .rsp_aborted           (core_wr_rsp_aborted),
        .rsp_error_code        (core_wr_rsp_error_code),
        .rsp_tag               (core_wr_rsp_tag),
        .m_awid                (m_axi_awid),
        .m_awaddr              (m_axi_awaddr),
        .m_awlen               (m_axi_awlen),
        .m_awsize              (m_axi_awsize),
        .m_awburst             (m_axi_awburst),
        .m_awlock              (m_axi_awlock),
        .m_awcache             (m_axi_awcache),
        .m_awprot              (m_axi_awprot),
        .m_awqos               (m_axi_awqos),
        .m_awregion            (m_axi_awregion),
        .m_awvalid             (m_axi_awvalid),
        .m_awready             (m_axi_awready),
        .m_wdata               (m_axi_wdata),
        .m_wstrb               (m_axi_wstrb),
        .m_wlast               (m_axi_wlast),
        .m_wvalid              (m_axi_wvalid),
        .m_wready              (m_axi_wready),
        .m_bid                 (m_axi_bid),
        .m_bresp               (m_axi_bresp),
        .m_bvalid              (m_axi_bvalid),
        .m_bready              (m_axi_bready),
        .stat_burst_count      (writer_stat_burst_count),
        .stat_last_addr        (writer_stat_last_addr),
        .stat_last_resp        (writer_stat_last_resp),
        .idle                  (writer_idle)
    );

endmodule

