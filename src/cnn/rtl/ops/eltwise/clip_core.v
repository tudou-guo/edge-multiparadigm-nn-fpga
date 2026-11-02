// clip_core.v -- E1-P2 INT8 Clip transform core
`timescale 1ns / 1ps
`include "op_abi_defs.vh"

module clip_core #(
    parameter integer DATA_WIDTH = 128,
    parameter integer CHUNK_BYTES = 16,
    parameter integer MAX_CH = 16
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] cfg_in_addr_byte,
    input  wire [31:0] cfg_out_addr_byte,
    input  wire [31:0] cfg_in_bytes,
    input  wire [31:0] cfg_out_bytes,
    input  wire [31:0] cfg_in_row_stride_bytes,
    input  wire [31:0] cfg_out_row_stride_bytes,
    input  wire [15:0] cfg_h,
    input  wire [15:0] cfg_w,
    input  wire [15:0] cfg_cin,
    input  wire [15:0] cfg_calloc,
    input  wire signed [7:0] cfg_qmin,
    input  wire signed [7:0] cfg_qmax,
    input  wire        core_start,
    output reg         core_busy,
    output reg         core_done,
    output reg         core_error,
    output reg  [15:0] core_error_code,
    input  wire        core_soft_reset,
    input  wire        core_abort,
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

    localparam [3:0] S_IDLE = 4'd0;
    localparam [3:0] S_RD_CMD = 4'd1;
    localparam [3:0] S_RD_DATA = 4'd2;
    localparam [3:0] S_WR_CMD = 4'd3;
    localparam [3:0] S_WR_DATA = 4'd4;
    localparam [3:0] S_WR_RSP = 4'd5;
    localparam [3:0] S_DONE = 4'd6;
    localparam [3:0] S_ERROR = 4'd7;
    localparam [3:0] S_ABORT = 4'd8;
    localparam [3:0] S_CHK   = 4'd9;   // revision6 PF4: one-cycle cfg product check
    localparam [7:0] TAG_RD = 8'h50;
    localparam [7:0] TAG_WR = 8'h60;

    reg [3:0] state;
    reg [15:0] row_idx;
    reg [31:0] chunk_off;
    reg [31:0] row_stride;
    reg [31:0] rd_row_base, wr_row_base;
    reg [DATA_WIDTH-1:0] out_beat;
    reg rd_outstanding, wr_outstanding, abort_seen;
    reg [63:0] chk_in_q, chk_out_q;   // h*stride products registered in S_IDLE

    wire rd_beat = rd_data_valid && rd_data_ready;
    wire wr_beat = wr_data_valid && wr_data_ready;
    // revision6 PF4: the two 64-bit h*stride products are registered in S_IDLE,
    // so the S_CHK state only compares the registered products with the byte
    // counts.  This removes the h*stride multiply cone from the state/busy CE
    // path; the only behavioural difference is that clip_core asserts busy/error
    // one cycle later than before (Clip is not used by the M4 95-node model).
    wire [63:0] chk_in_now  = {32'd0, cfg_h} * {32'd0, cfg_in_row_stride_bytes};
    wire [63:0] chk_out_now = {32'd0, cfg_h} * {32'd0, cfg_out_row_stride_bytes};
    wire cfg_ok_cheap = (cfg_h != 0) && (cfg_w != 0) && (cfg_cin != 0) &&
                  (cfg_cin <= MAX_CH) && (cfg_calloc <= MAX_CH) &&
                  ((cfg_calloc == 16'd8) || (cfg_calloc == 16'd16)) &&
                  (cfg_cin <= cfg_calloc) &&
                  ((cfg_in_row_stride_bytes & 32'hF) == 0) &&
                  ((cfg_out_row_stride_bytes & 32'hF) == 0) &&
                  (cfg_in_row_stride_bytes == cfg_out_row_stride_bytes) &&
                  (cfg_in_bytes != 0) && (cfg_out_bytes == cfg_in_bytes) &&
                  (cfg_in_addr_byte[3:0] == 0) && (cfg_out_addr_byte[3:0] == 0) &&
                  (cfg_qmin <= cfg_qmax);
    wire cfg_ok = cfg_ok_cheap &&
                  ({32'd0, cfg_in_bytes} == chk_in_q) &&
                  ({32'd0, cfg_out_bytes} == chk_out_q);
    wire [31:0] rd_addr = rd_row_base + chunk_off;
    wire [31:0] wr_addr = wr_row_base + chunk_off;    wire [DATA_WIDTH-1:0] transformed_beat;
    genvar gl;
    generate
        for (gl = 0; gl < DATA_WIDTH/8; gl = gl + 1) begin : G_CLIP_LANE
            wire [31:0] lane_off = chunk_off + gl;
            wire [15:0] lane_pixel = (cfg_calloc == 16'd8) ? lane_off[31:3] : lane_off[31:4];
            wire [15:0] lane_ch = (cfg_calloc == 16'd8) ? {13'd0, lane_off[2:0]} :
                                  {12'd0, lane_off[3:0]};
            wire lane_active = (lane_pixel < cfg_w) && (lane_ch < cfg_cin);
            wire signed [7:0] raw_byte = rd_data[gl*8 +: 8];
            wire [7:0] clamped = (raw_byte < cfg_qmin) ? cfg_qmin :
                                 (raw_byte > cfg_qmax) ? cfg_qmax : raw_byte;
            assign transformed_beat[gl*8 +: 8] = lane_active ? clamped : 8'h00;
        end
    endgenerate

    always @(*) begin
        rd_cmd_valid = (state == S_RD_CMD) && !rd_outstanding;
        rd_cmd_addr_byte = rd_addr;
        rd_cmd_bytes = CHUNK_BYTES;
        rd_cmd_tag = TAG_RD;
        rd_data_ready = (state == S_RD_DATA) || (state == S_ABORT && rd_outstanding);
        wr_cmd_valid = (state == S_WR_CMD) && !wr_outstanding;
        wr_cmd_addr_byte = wr_addr;
        wr_cmd_bytes = CHUNK_BYTES;
        wr_cmd_tag = TAG_WR;
        wr_data_valid = (state == S_WR_DATA);
        wr_data = out_beat;
        wr_data_last = 1'b1;
        wr_data_tag = TAG_WR;
        wr_rsp_ready = (state == S_WR_RSP) || (state == S_ABORT && wr_outstanding);
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; row_idx <= 0; chunk_off <= 0; row_stride <= 0;
            rd_row_base <= 0; wr_row_base <= 0;
            out_beat <= 0; rd_outstanding <= 0; wr_outstanding <= 0; abort_seen <= 0;
            chk_in_q <= 0; chk_out_q <= 0;
            core_busy <= 0; core_done <= 0; core_error <= 0; core_error_code <= `OP_E_NONE;
        end else if (core_soft_reset) begin
            state <= S_IDLE; row_idx <= 0; chunk_off <= 0; row_stride <= 0;
            rd_row_base <= 0; wr_row_base <= 0;
            out_beat <= 0; rd_outstanding <= 0; wr_outstanding <= 0; abort_seen <= 0;
            chk_in_q <= 0; chk_out_q <= 0;
            core_busy <= 0; core_done <= 0; core_error <= 0; core_error_code <= `OP_E_NONE;
        end else begin
            core_done <= 1'b0;
            case (state)
                S_IDLE: begin
                    core_busy <= 0; core_error <= 0; core_error_code <= `OP_E_NONE;
                    rd_outstanding <= 0; wr_outstanding <= 0; abort_seen <= 0;
                    if (core_start) begin
                        row_idx <= 0; chunk_off <= 0; out_beat <= 0;
                        chk_in_q  <= chk_in_now;
                        chk_out_q <= chk_out_now;
                        state <= S_CHK;
                    end
                end
                S_CHK: begin
                    if (!cfg_ok) begin
                        core_error <= 1'b1; core_error_code <= `OP_E_CFG;
                        state <= S_IDLE;
                    end else begin
                        row_stride <= cfg_in_row_stride_bytes;
                        rd_row_base <= cfg_in_addr_byte;
                        wr_row_base <= cfg_out_addr_byte;
                        core_busy <= 1'b1;
                        state <= S_RD_CMD;
                    end
                end
                S_RD_CMD: begin
                    if (core_abort) begin abort_seen <= 1'b1; state <= S_ABORT; end
                    else if (rd_cmd_valid && rd_cmd_ready) begin
                        rd_outstanding <= 1'b1; state <= S_RD_DATA;
                    end
                end
                S_RD_DATA: begin
                    if (rd_beat) begin
                        rd_outstanding <= 1'b0;
                        if (rd_data_error) begin
                            core_error <= 1'b1;
                            core_error_code <= (rd_data_error_code == `OP_E_NONE) ?
                                               `OP_E_AXI_RESP : rd_data_error_code;
                            state <= S_ERROR;
                        end else if (abort_seen || core_abort) begin
                            abort_seen <= 1'b1; state <= S_ABORT;
                        end else begin
                            out_beat <= transformed_beat;
                            state <= S_WR_CMD;
                        end
                    end
                end
                S_WR_CMD: begin
                    if (core_abort) begin abort_seen <= 1'b1; state <= S_ABORT; end
                    else if (wr_cmd_valid && wr_cmd_ready) begin
                        wr_outstanding <= 1'b1; state <= S_WR_DATA;
                    end
                end
                S_WR_DATA: begin
                    if (core_abort) abort_seen <= 1'b1;
                    if (wr_beat) state <= S_WR_RSP;
                end
                S_WR_RSP: begin
                    if (wr_rsp_valid) begin
                        wr_outstanding <= 1'b0;
                        if (wr_rsp_error || wr_rsp_aborted) begin
                            core_error <= 1'b1;
                            core_error_code <= (wr_rsp_error_code == `OP_E_NONE) ?
                                               `OP_E_AXI_RESP : wr_rsp_error_code;
                            state <= S_ERROR;
                        end else if (abort_seen || core_abort) begin
                            abort_seen <= 1'b1; state <= S_ABORT;
                        end else if (chunk_off + CHUNK_BYTES >= row_stride) begin
                            chunk_off <= 0;
                            if (row_idx + 1 >= cfg_h) state <= S_DONE;
                            else begin
                                row_idx <= row_idx + 1;
                                rd_row_base <= rd_row_base + row_stride;
                                wr_row_base <= wr_row_base + row_stride;
                                state <= S_RD_CMD;
                            end
                        end else begin
                            chunk_off <= chunk_off + CHUNK_BYTES; state <= S_RD_CMD;
                        end
                    end
                end
                S_DONE: begin core_busy <= 0; core_done <= 1'b1; state <= S_IDLE; end
                S_ABORT: begin
                    if (!rd_outstanding && !wr_outstanding) begin
                        core_busy <= 0; state <= S_IDLE;
                    end
                end
                S_ERROR: begin core_busy <= 0; core_error <= 1'b1; end
                default: state <= S_IDLE;
            endcase
        end
    end

    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_tags = |{rd_data_last, rd_data_tag, wr_rsp_tag};
    /* verilator lint_on UNUSEDSIGNAL */
endmodule