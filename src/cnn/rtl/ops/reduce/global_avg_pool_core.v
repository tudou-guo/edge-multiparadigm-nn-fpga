// global_avg_pool_core.v -- E1-P2 exact rounded INT8 channel mean
`timescale 1ns / 1ps
`include "op_abi_defs.vh"

module global_avg_pool_core #(
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
    input  wire [15:0] cfg_h,
    input  wire [15:0] cfg_w,
    input  wire [15:0] cfg_cin,
    input  wire [15:0] cfg_calloc,
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
    localparam [3:0] S_PROC = 4'd3;
    localparam [3:0] S_DIV_INIT = 4'd4;
    localparam [3:0] S_DIV_STEP = 4'd5;
    localparam [3:0] S_DIV_FINISH = 4'd6;
    localparam [3:0] S_WR_CMD = 4'd7;
    localparam [3:0] S_WR_DATA = 4'd8;
    localparam [3:0] S_WR_RSP = 4'd9;
    localparam [3:0] S_DONE = 4'd10;
    localparam [3:0] S_ERROR = 4'd11;
    localparam [3:0] S_ABORT = 4'd12;
    localparam [7:0] TAG_RD = 8'h70;
    localparam [7:0] TAG_WR = 8'h80;

    reg [3:0] state;
    reg [15:0] row_idx;
    reg [31:0] chunk_off;
    reg [31:0] row_stride;
    reg [31:0] rd_row_base;
    reg [4:0] lane_idx;
    reg signed [31:0] sums [0:MAX_CH-1];
    reg [7:0] out_vals [0:MAX_CH-1];
    reg [DATA_WIDTH-1:0] beat_q;
    reg [4:0] div_ch;
    reg [5:0] div_bit;
    reg [31:0] dividend_abs, divisor, rem, quot;
    reg dividend_negative;
    reg rd_outstanding, wr_outstanding, abort_seen;
    integer init_i;

    wire rd_beat = rd_data_valid && rd_data_ready;
    wire wr_beat = wr_data_valid && wr_data_ready;
    wire [63:0] expected_in = {32'd0, cfg_h} * {32'd0, cfg_in_row_stride_bytes};
    wire cfg_ok = (cfg_h != 0) && (cfg_w != 0) && (cfg_cin != 0) &&
                  (cfg_cin <= MAX_CH) && (cfg_calloc <= MAX_CH) &&
                  ((cfg_calloc == 16'd8) || (cfg_calloc == 16'd16)) &&
                  (cfg_cin <= cfg_calloc) &&
                  ((cfg_in_row_stride_bytes & 32'hF) == 0) &&
                  (cfg_in_bytes != 0) && ({32'd0, cfg_in_bytes} == expected_in) &&
                  (cfg_out_bytes == 32'd16) &&
                  (cfg_in_addr_byte[3:0] == 0) && (cfg_out_addr_byte[3:0] == 0);
    wire [31:0] rd_addr = rd_row_base + chunk_off;
    wire [31:0] lane_off = chunk_off + lane_idx;
    wire [15:0] lane_pixel = (cfg_calloc == 16'd8) ? lane_off[31:3] : lane_off[31:4];
    wire [15:0] lane_ch = (cfg_calloc == 16'd8) ? {13'd0, lane_off[2:0]} :
                          {12'd0, lane_off[3:0]};
    wire lane_active = (lane_pixel < cfg_w) && (lane_ch < cfg_cin);
    wire signed [7:0] lane_byte = beat_q[lane_idx*8 +: 8];
    wire [31:0] rem_shift = ({rem[30:0], 1'b0}) | ((dividend_abs >> div_bit) & 32'd1);
    wire [31:0] target_divisor = {16'd0, cfg_h} * {16'd0, cfg_w};
    wire [31:0] round_twice = {rem[30:0], 1'b0};
    wire [127:0] out_beat;    genvar gl;
    generate
        for (gl = 0; gl < DATA_WIDTH/8; gl = gl + 1) begin : G_OUT_LANE
            assign out_beat[gl*8 +: 8] = (gl < cfg_cin) ? out_vals[gl] : 8'h00;
        end
    endgenerate

    always @(*) begin
        rd_cmd_valid = (state == S_RD_CMD) && !rd_outstanding;
        rd_cmd_addr_byte = rd_addr;
        rd_cmd_bytes = CHUNK_BYTES;
        rd_cmd_tag = TAG_RD;
        rd_data_ready = (state == S_RD_DATA) || (state == S_ABORT && rd_outstanding);
        wr_cmd_valid = (state == S_WR_CMD) && !wr_outstanding;
        wr_cmd_addr_byte = cfg_out_addr_byte;
        wr_cmd_bytes = 16'd16;
        wr_cmd_tag = TAG_WR;
        wr_data_valid = (state == S_WR_DATA);
        wr_data = out_beat;
        wr_data_last = 1'b1;
        wr_data_tag = TAG_WR;
        wr_rsp_ready = (state == S_WR_RSP) || (state == S_ABORT && wr_outstanding);
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; row_idx <= 0; chunk_off <= 0; row_stride <= 0; lane_idx <= 0;
            rd_row_base <= 0;
            div_ch <= 0; div_bit <= 0; dividend_abs <= 0; divisor <= 0; rem <= 0; quot <= 0;
            dividend_negative <= 0; beat_q <= 0; rd_outstanding <= 0; wr_outstanding <= 0;
            abort_seen <= 0; core_busy <= 0; core_done <= 0; core_error <= 0;
            core_error_code <= `OP_E_NONE;
            for (init_i = 0; init_i < MAX_CH; init_i = init_i + 1) begin
                sums[init_i] <= 0; out_vals[init_i] <= 0;
            end
        end else if (core_soft_reset) begin
            state <= S_IDLE; row_idx <= 0; chunk_off <= 0; row_stride <= 0; lane_idx <= 0;
            div_ch <= 0; div_bit <= 0; dividend_abs <= 0; divisor <= 0; rem <= 0; quot <= 0;
            dividend_negative <= 0; beat_q <= 0; rd_outstanding <= 0; wr_outstanding <= 0;
            abort_seen <= 0; core_busy <= 0; core_done <= 0; core_error <= 0;
            core_error_code <= `OP_E_NONE;
            for (init_i = 0; init_i < MAX_CH; init_i = init_i + 1) begin
                sums[init_i] <= 0; out_vals[init_i] <= 0;
            end
        end else begin
            core_done <= 1'b0;
            case (state)
                S_IDLE: begin
                    core_busy <= 0; core_error <= 0; core_error_code <= `OP_E_NONE;
                    rd_outstanding <= 0; wr_outstanding <= 0; abort_seen <= 0;
                    if (core_start) begin
                        row_idx <= 0; chunk_off <= 0; lane_idx <= 0; div_ch <= 0;
                        rd_row_base <= cfg_in_addr_byte;
                        div_bit <= 0; rem <= 0; quot <= 0; dividend_abs <= 0;
                        dividend_negative <= 0; row_stride <= cfg_in_row_stride_bytes;
                        for (init_i = 0; init_i < MAX_CH; init_i = init_i + 1) begin
                            sums[init_i] <= 0; out_vals[init_i] <= 0;
                        end
                        if (!cfg_ok) begin
                            core_error <= 1'b1; core_error_code <= `OP_E_CFG;
                        end else begin
                            core_busy <= 1'b1; state <= S_RD_CMD;
                        end
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
                            beat_q <= rd_data; lane_idx <= 0; state <= S_PROC;
                        end
                    end
                end
                S_PROC: begin
                    if (lane_active) begin
                        sums[lane_ch] <= sums[lane_ch] + {{24{lane_byte[7]}}, lane_byte};
                    end
                    if (lane_idx == 5'd15) begin
                        if (chunk_off + CHUNK_BYTES >= row_stride) begin
                            chunk_off <= 0;
                            if (row_idx + 1 >= cfg_h) begin
                                div_ch <= 0; div_bit <= 31; rem <= 0; quot <= 0;
                                if (sums[0] < 0) begin
                                    dividend_abs <= -sums[0]; dividend_negative <= 1'b1;
                                end else begin
                                    dividend_abs <= sums[0]; dividend_negative <= 1'b0;
                                end
                                divisor <= target_divisor; state <= S_DIV_STEP;
                            end else begin
                                row_idx <= row_idx + 1;
                                rd_row_base <= rd_row_base + row_stride;
                                state <= S_RD_CMD;
                            end
                        end else begin
                            chunk_off <= chunk_off + CHUNK_BYTES; state <= S_RD_CMD;
                        end
                    end else begin
                        lane_idx <= lane_idx + 5'd1;
                    end
                end
                S_DIV_STEP: begin
                    if (rem_shift >= divisor) begin
                        rem <= rem_shift - divisor; quot[div_bit] <= 1'b1;
                    end else begin
                        rem <= rem_shift; quot[div_bit] <= 1'b0;
                    end
                    if (div_bit == 6'd0) state <= S_DIV_FINISH;
                    else div_bit <= div_bit - 6'd1;
                end
                S_DIV_FINISH: begin
                    if (round_twice >= divisor) quot <= quot + 32'd1;
                    if (dividend_negative) begin
                        if ((quot + (round_twice >= divisor ? 32'd1 : 32'd0)) > 32'd128)
                            out_vals[div_ch] <= 8'h80;
                        else
                            out_vals[div_ch] <= -(quot + (round_twice >= divisor ? 32'd1 : 32'd0));
                    end else begin
                        if ((quot + (round_twice >= divisor ? 32'd1 : 32'd0)) > 32'd127)
                            out_vals[div_ch] <= 8'h7f;
                        else
                            out_vals[div_ch] <= quot + (round_twice >= divisor ? 32'd1 : 32'd0);
                    end
                    if (div_ch + 1 >= cfg_cin) begin
                        state <= S_WR_CMD;
                    end else begin
                        div_ch <= div_ch + 5'd1; div_bit <= 31; rem <= 0; quot <= 0;
                        if (sums[div_ch + 1] < 0) begin
                            dividend_abs <= -sums[div_ch + 1]; dividend_negative <= 1'b1;
                        end else begin
                            dividend_abs <= sums[div_ch + 1]; dividend_negative <= 1'b0;
                        end
                        state <= S_DIV_STEP;
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
                        end else begin
                            state <= S_DONE;
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