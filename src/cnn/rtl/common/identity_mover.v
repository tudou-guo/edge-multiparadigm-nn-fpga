// -----------------------------------------------------------------------------
// identity_mover.v
//
// G0 test core: copies a DDR region from cfg_in_addr_byte to cfg_out_addr_byte
// using ONLY the frozen normalized core memory interface from docs/op_abi.md 2.3.
//
// T-OP2a scope:
//   - one read command and one write command per start;
//   - cfg_in_bytes == cfg_out_bytes, non-zero, 16-byte aligned, <= 65535 bytes;
//   - read-before-write gating: the write command is not accepted until the first
//     valid read beat has arrived. This guarantees that a rejected read address
//     produces no output write transaction;
//   - pass-through streaming after the first beat, so all byte lanes are copied
//     unchanged and no arithmetic is performed;
//   - cancellation drains the accepted read burst. A late read error is still
//     forwarded as data so an accepted write burst can terminate cleanly.
//
// TODO(T-OP2b): replace the 16-bit single-command limit with deterministic
// chunking and add the full AXI backpressure/long-run stress BFM to G0.
//
// This module is NOT conv3x3 and MUST NOT be reported as a convolution result.
// -----------------------------------------------------------------------------

`timescale 1ns / 1ps
`include "op_abi_defs.vh"

module identity_mover #(
    parameter integer DATA_WIDTH = 128
) (
    input  wire        clk,
    input  wire        rst_n,

    // Locked configuration snapshot from op_shell.
    input  wire [31:0] cfg_in_addr_byte,
    input  wire [31:0] cfg_out_addr_byte,
    input  wire [31:0] cfg_in_bytes,
    input  wire [31:0] cfg_out_bytes,

    // Shell control/status boundary.
    input  wire        core_start,
    output reg         core_busy,
    output reg         core_done,
    output reg         core_error,
    output reg  [15:0] core_error_code,
    input  wire        core_soft_reset,
    input  wire        core_abort,

    // Normalized read command/data service.
    output reg         rd_cmd_valid,
    input  wire        rd_cmd_ready,
    output reg  [31:0] rd_cmd_addr_byte,
    output reg  [15:0] rd_cmd_bytes,
    output reg  [7:0]  rd_cmd_tag,
    input  wire        rd_data_valid,
    output reg         rd_data_ready,
    input  wire [DATA_WIDTH-1:0] rd_data,
    input  wire        rd_data_last,
    input  wire        rd_data_error,
    input  wire [15:0] rd_data_error_code,
    input  wire [7:0]  rd_data_tag,

    // Normalized write command/data service.
    output reg         wr_cmd_valid,
    input  wire        wr_cmd_ready,
    output reg  [31:0] wr_cmd_addr_byte,
    output reg  [15:0] wr_cmd_bytes,
    output reg  [7:0]  wr_cmd_tag,
    output reg         wr_data_valid,
    input  wire        wr_data_ready,
    output reg  [DATA_WIDTH-1:0] wr_data,
    output reg         wr_data_last,
    output reg  [7:0]  wr_data_tag,
    input  wire        wr_rsp_valid,
    output reg         wr_rsp_ready,
    input  wire        wr_rsp_error,
    input  wire        wr_rsp_aborted,
    input  wire [15:0] wr_rsp_error_code,
    input  wire [7:0]  wr_rsp_tag
);

    localparam [2:0] I_IDLE     = 3'd0;
    localparam [2:0] I_RD_ISSUE = 3'd1;
    localparam [2:0] I_FIRST    = 3'd2;
    localparam [2:0] I_WR_ISSUE = 3'd3;
    localparam [2:0] I_STREAM   = 3'd4;
    localparam [2:0] I_DRAIN    = 3'd5;
    localparam [2:0] I_WAIT     = 3'd6;
    localparam [2:0] I_ERROR    = 3'd7;

    reg [2:0]  state;
    reg [15:0] xfer_bytes;
    reg        rd_cmd_done;
    reg        wr_cmd_done;
    reg        abort_seen;

    wire rd_cmd_fire = rd_cmd_valid && rd_cmd_ready;
    wire wr_cmd_fire = wr_cmd_valid && wr_cmd_ready;
    wire rd_beat     = rd_data_valid && rd_data_ready;
    wire wr_beat     = wr_data_valid && wr_data_ready;
    wire error_beat  = rd_data_valid && rd_data_error;

    wire cfg_len_ok   = (cfg_in_bytes != 32'b0) && (cfg_in_bytes == cfg_out_bytes);
    wire cfg_align_ok = (cfg_in_addr_byte[3:0] == 4'b0) &&
                        (cfg_out_addr_byte[3:0] == 4'b0) &&
                        (cfg_in_bytes[3:0] == 4'b0);
    wire cfg_fits     = (cfg_in_bytes <= 32'd65535);

    // Read is issued first. Write is deliberately held until the first read
    // beat is known to be valid.
    always @(*) begin
        rd_cmd_valid     = (state == I_RD_ISSUE) && !rd_cmd_done;
        rd_cmd_addr_byte = cfg_in_addr_byte;
        rd_cmd_bytes     = xfer_bytes;
        rd_cmd_tag       = 8'h01;

        wr_cmd_valid     = (state == I_WR_ISSUE) && !wr_cmd_done;
        wr_cmd_addr_byte = cfg_out_addr_byte;
        wr_cmd_bytes     = xfer_bytes;
        wr_cmd_tag       = 8'h02;

        rd_data_ready = 1'b0;
        wr_data_valid = 1'b0;
        wr_data       = rd_data;
        wr_data_last  = rd_data_last;
        wr_data_tag   = rd_data_tag;
        wr_rsp_ready  = 1'b0;

        if (state == I_FIRST) begin
            // Hold the first read beat until the write command is accepted.
            // If the read failed or the shell is cancelling, consume exactly
            // this terminal beat and do not create an output transaction.
            if (core_abort || rd_data_error) begin
                rd_data_ready = 1'b1;
            end
        end else if (state == I_WR_ISSUE) begin
            // Keep holding the first beat. If cancellation wins before the
            // write command is accepted, drain it instead of issuing a write.
            rd_data_ready = core_abort;
        end else if (state == I_STREAM) begin
            rd_data_ready = wr_data_ready;
            wr_data_valid = rd_data_valid;
        end else if (state == I_DRAIN) begin
            rd_data_ready = 1'b1;
        end else if (state == I_WAIT) begin
            wr_rsp_ready = 1'b1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= I_IDLE;
            xfer_bytes      <= 16'b0;
            rd_cmd_done     <= 1'b0;
            wr_cmd_done     <= 1'b0;
            abort_seen      <= 1'b0;
            core_busy       <= 1'b0;
            core_done       <= 1'b0;
            core_error      <= 1'b0;
            core_error_code <= `OP_E_NONE;
        end else if (core_soft_reset) begin
            state           <= I_IDLE;
            rd_cmd_done     <= 1'b0;
            wr_cmd_done     <= 1'b0;
            abort_seen      <= 1'b0;
            core_busy       <= 1'b0;
            core_done       <= 1'b0;
            core_error      <= 1'b0;
            core_error_code <= `OP_E_NONE;
        end else begin
            core_done <= 1'b0;

            case (state)
                I_IDLE: begin
                    core_busy   <= 1'b0;
                    core_error  <= 1'b0;
                    rd_cmd_done <= 1'b0;
                    wr_cmd_done <= 1'b0;
                    abort_seen  <= 1'b0;
                    if (core_start) begin
                        xfer_bytes <= cfg_in_bytes[15:0];
                        if (!cfg_len_ok) begin
                            core_error      <= 1'b1;
                            core_error_code <= `OP_E_CFG;
                            core_busy       <= 1'b0;
                        end else if (!cfg_align_ok) begin
                            core_error      <= 1'b1;
                            core_error_code <= `OP_E_ADDR_ALIGN;
                            core_busy       <= 1'b0;
                        end else if (!cfg_fits) begin
                            core_error      <= 1'b1;
                            core_error_code <= `OP_E_UNSUPPORTED;
                            core_busy       <= 1'b0;
                        end else begin
                            core_error <= 1'b0;
                            core_busy  <= 1'b1;
                            state      <= I_RD_ISSUE;
                        end
                    end
                end

                I_RD_ISSUE: begin
                    if (rd_cmd_fire) begin
                        rd_cmd_done <= 1'b1;
                        state       <= I_FIRST;
                    end else if (core_abort) begin
                        core_busy <= 1'b0;
                        state     <= I_ERROR;
                    end
                end

                I_FIRST: begin
                    if (rd_data_valid && core_abort) begin
                        core_busy <= 1'b0;
                        state     <= I_ERROR;
                    end else if (rd_data_valid && rd_data_error) begin
                        core_error      <= 1'b1;
                        core_error_code <= (rd_data_error_code == `OP_E_NONE)
                                           ? `OP_E_AXI_RESP : rd_data_error_code;
                        core_busy       <= 1'b0;
                        state           <= I_ERROR;
                    end else if (rd_data_valid) begin
                        state <= I_WR_ISSUE;
                    end
                end

                I_WR_ISSUE: begin
                    if (wr_cmd_fire) begin
                        wr_cmd_done <= 1'b1;
                        state       <= I_STREAM;
                    end else if (core_abort) begin
                        state <= I_DRAIN;
                    end
                end

                I_STREAM: begin
                    if (error_beat) begin
                        core_error      <= 1'b1;
                        core_error_code <= (rd_data_error_code == `OP_E_NONE)
                                           ? `OP_E_AXI_RESP : rd_data_error_code;
                    end

                    if (wr_rsp_valid && (wr_rsp_error || wr_rsp_aborted)) begin
                        if (wr_rsp_error) begin
                            core_error      <= 1'b1;
                            core_error_code <= (wr_rsp_error_code == `OP_E_NONE)
                                               ? `OP_E_AXI_RESP : wr_rsp_error_code;
                        end
                        state <= (rd_beat && rd_data_last) ? I_WAIT : I_DRAIN;
                    end else if (rd_beat && rd_data_last) begin
                        state <= I_WAIT;
                    end
                end

                I_DRAIN: begin
                    if (rd_beat && rd_data_last) begin
                        state <= I_WAIT;
                    end
                end

                I_WAIT: begin
                    if (wr_rsp_valid) begin
                        core_busy <= 1'b0;
                        if (wr_rsp_error) begin
                            core_error      <= 1'b1;
                            core_error_code <= (wr_rsp_error_code == `OP_E_NONE)
                                               ? `OP_E_AXI_RESP : wr_rsp_error_code;
                        end
                        if (core_error || wr_rsp_error || wr_rsp_aborted) begin
                            state <= I_ERROR;
                        end else begin
                            core_done <= 1'b1;
                            state     <= I_IDLE;
                        end
                    end
                end

                I_ERROR: begin
                    core_busy <= 1'b0;
                    if (core_abort) begin
                        abort_seen <= 1'b1;
                    end else if (abort_seen) begin
                        core_error      <= 1'b0;
                        core_error_code <= `OP_E_NONE;
                        state           <= I_IDLE;
                    end
                end

                default: state <= I_IDLE;
            endcase
        end
    end

    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_rd_tag = |rd_data_tag;
    wire unused_wr_tag = |wr_rsp_tag;
    wire unused_wr_beat = wr_beat;
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
