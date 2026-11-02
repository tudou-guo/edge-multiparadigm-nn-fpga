// -----------------------------------------------------------------------------
// eltwise_core.v
//
// W4 elementwise core for E1:
//   mode 1 = Relu (signed INT8 clamp min 0)
//   mode 2 = same-shape, same-scale INT8 Add with signed saturation
//   mode 2 + cfg_flags==2 = affine-int8-v1 Add: two per-input M0 multipliers,
//     one final ties-away rounding in the output domain, + z_out and saturate.
//     The rescale is a two-stage pipeline (products, then round/shift/sat), never
//     one large combinational multiply-shift-饱和 path.
//
// The core consumes the frozen normalized op_shell read/write service.  It
// processes one 16-byte chunk at a time, issuing one read command per input,
// one write command per output, and one write response per chunk.  This keeps
// the first correctness implementation independent of a large staging RAM.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps
`include "op_abi_defs.vh"

module eltwise_core #(
    parameter integer DATA_WIDTH = 128,
    parameter integer CHUNK_BYTES = 16,
    parameter [1:0]   MODE_RELU = 2'd1,
    parameter [1:0]   MODE_ADD  = 2'd2
) (
    input  wire        clk,
    input  wire        rst_n,

    // Locked per-record configuration.
    input  wire [1:0]  cfg_mode,
    input  wire [31:0] cfg_in0_addr_byte,
    input  wire [31:0] cfg_in1_addr_byte,
    input  wire [31:0] cfg_out_addr_byte,
    input  wire [31:0] cfg_in0_bytes,
    input  wire [31:0] cfg_in1_bytes,
    input  wire [31:0] cfg_out_bytes,
    input  wire [31:0] cfg_flags,
    // affine-int8-v1 Add rescale fields (cfg_flags == 2 selects this path).
    input  wire [31:0] cfg_m0_a,
    input  wire [31:0] cfg_m0_b,
    // [7:0] shift, [15:8] zp_a, [23:16] zp_b, [31:24] zp_out (all signed int8)
    input  wire [31:0] cfg_rescale_cfg,

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

    localparam [3:0] S_IDLE    = 4'd0;
    localparam [3:0] S_RD0     = 4'd1;
    localparam [3:0] S_RD0_W   = 4'd2;
    localparam [3:0] S_RD1     = 4'd3;
    localparam [3:0] S_RD1_W   = 4'd4;
    localparam [3:0] S_WR      = 4'd5;
    localparam [3:0] S_WR_DATA = 4'd6;
    localparam [3:0] S_WR_RSP  = 4'd7;
    localparam [3:0] S_DONE    = 4'd8;
    localparam [3:0] S_ERROR   = 4'd9;
    localparam [3:0] S_ABORT   = 4'd10;
    localparam [3:0] S_RS_MUL  = 4'd11;
    localparam [3:0] S_RS_SHIFT= 4'd12;

    localparam [7:0] TAG_RD0 = 8'h10;
    localparam [7:0] TAG_RD1 = 8'h11;
    localparam [7:0] TAG_WR  = 8'h20;

    reg [3:0]  state;
    reg [1:0]  mode_q;
    reg [31:0] remain_bytes;
    reg [31:0] cur_in0;
    reg [31:0] cur_in1;
    reg [31:0] cur_out;
    reg [127:0] a_beat;
    reg [127:0] b_beat;
    reg        rd0_done, rd1_done, wr_done;
    reg        read_outstanding, write_outstanding;
    reg        abort_seen;
    reg [31:0] flags_q, m0a_q, m0b_q, rescale_q;
    reg signed [63:0] rs_sum [0:15];
    reg        rs_negative [0:15];
    // revision5 PF2: for signed-int8 inputs and unsigned 32-bit M0,
    // |sum| < 2^41, so abs+bias+shift is exact in 42 bits (doc proof).
    reg [41:0] rs_magnitude [0:15];
    reg [127:0] rs_out;
    reg        rs_shift_large_q;   // cfg shift >= 42 -> magnitude is exactly 0
    reg [5:0]  rs_shift_small_q;   // cfg shift[5:0], used only when !large
    reg [41:0] rs_round_bias_q;    // (shift==0 || shift>=42) ? 0 : 1<<(shift-1)
    reg signed [7:0] rs_zp_out;

    wire add_mode = (mode_q == MODE_ADD);
    wire relu_mode = (mode_q == MODE_RELU);
    wire rescale_mode = add_mode && (flags_q == 32'h00000002);
    wire cmd_fire = rd_cmd_valid && rd_cmd_ready;
    wire wr_cmd_fire = wr_cmd_valid && wr_cmd_ready;
    wire wr_beat = wr_data_valid && wr_data_ready;
    wire rd_beat = rd_data_valid && rd_data_ready;

    wire cfg_flags_ok = (cfg_flags == 32'h00000001) ||
                        ((cfg_flags == 32'h00000002) && (cfg_mode == MODE_ADD));
    wire cfg_mode_ok  = (cfg_mode == MODE_RELU) || (cfg_mode == MODE_ADD);
    wire cfg_rescale_ok = (cfg_flags != 32'h00000002) ||
                          ((cfg_m0_a != 32'b0) && (cfg_m0_b != 32'b0));
    wire cfg_len_ok   = (cfg_in0_bytes != 0) && (cfg_out_bytes == cfg_in0_bytes) &&
                        ((cfg_in0_bytes & 32'hF) == 0);
    wire cfg_start_add = (cfg_mode == MODE_ADD);
    wire cfg_add_len_ok = !cfg_start_add || (cfg_in1_bytes == cfg_in0_bytes);
    wire cfg_addr_ok = (cfg_in0_addr_byte[3:0] == 0) &&
                       (cfg_out_addr_byte[3:0] == 0) &&
                       (!cfg_start_add || (cfg_in1_addr_byte[3:0] == 0));

    // Elementwise lane transform.
    wire [127:0] ctrl_result;
    genvar lane;
    integer li;
    generate
        for (lane = 0; lane < DATA_WIDTH/8; lane = lane + 1) begin : G_LANE
            wire signed [7:0] av = a_beat[lane*8 +: 8];
            wire signed [7:0] bv = b_beat[lane*8 +: 8];
            wire signed [8:0] sum = $signed(av) + $signed(bv);
            wire [7:0] add_out = (sum > 9'sd127) ? 8'sd127 :
                                 (sum < -9'sd128) ? -8'sd128 : sum[7:0];
            wire [7:0] relu_out = (av < 0) ? 8'sd0 : av;
            assign ctrl_result[lane*8 +: 8] = add_mode ? add_out : relu_out;
        end
    endgenerate

    // affine Add result (registered by the two-stage rescale pipeline).
    wire [127:0] wr_result = rescale_mode ? rs_out : ctrl_result;

    always @(*) begin
        rd_cmd_valid     = (state == S_RD0) && !rd0_done ||
                           (state == S_RD1) && !rd1_done;
        rd_cmd_addr_byte = (state == S_RD1) ? cur_in1 : cur_in0;
        rd_cmd_bytes     = CHUNK_BYTES;
        rd_cmd_tag       = (state == S_RD1) ? TAG_RD1 : TAG_RD0;

        rd_data_ready = (state == S_RD0_W) || (state == S_RD1_W) ||
                        (state == S_ABORT && read_outstanding);

        wr_cmd_valid     = (state == S_WR) && !wr_done;
        wr_cmd_addr_byte = cur_out;
        wr_cmd_bytes     = CHUNK_BYTES;
        wr_cmd_tag       = TAG_WR;

        wr_data_valid = (state == S_WR_DATA);
        wr_data       = wr_result;
        wr_data_last  = 1'b1;
        wr_data_tag   = TAG_WR;
        wr_rsp_ready  = (state == S_WR_RSP) ||
                        (state == S_ABORT && write_outstanding);
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state               <= S_IDLE;
            mode_q              <= MODE_RELU;
            flags_q             <= 32'b0;
            m0a_q               <= 32'b0;
            m0b_q               <= 32'b0;
            rescale_q           <= 32'b0;
            rs_out              <= 128'b0;
            rs_zp_out           <= 8'sd0;
            remain_bytes        <= 32'b0;
            cur_in0             <= 32'b0;
            cur_in1             <= 32'b0;
            cur_out             <= 32'b0;
            a_beat              <= 128'b0;
            b_beat              <= 128'b0;
            rd0_done            <= 1'b0;
            rd1_done            <= 1'b0;
            wr_done             <= 1'b0;
            read_outstanding    <= 1'b0;
            write_outstanding   <= 1'b0;
            abort_seen          <= 1'b0;
            core_busy           <= 1'b0;
            core_done           <= 1'b0;
            core_error          <= 1'b0;
            core_error_code     <= `OP_E_NONE;
        end else if (core_soft_reset) begin
            state               <= S_IDLE;
            remain_bytes        <= 32'b0;
            rd0_done            <= 1'b0;
            rd1_done            <= 1'b0;
            wr_done             <= 1'b0;
            read_outstanding    <= 1'b0;
            write_outstanding   <= 1'b0;
            abort_seen          <= 1'b0;
            core_busy           <= 1'b0;
            core_done           <= 1'b0;
            core_error          <= 1'b0;
            core_error_code     <= `OP_E_NONE;
        end else begin
            core_done <= 1'b0;

            case (state)
                S_IDLE: begin
                    core_busy <= 1'b0;
                    core_error <= 1'b0;
                    core_error_code <= `OP_E_NONE;
                    rd0_done <= 1'b0;
                    rd1_done <= 1'b0;
                    wr_done <= 1'b0;
                    read_outstanding <= 1'b0;
                    write_outstanding <= 1'b0;
                    abort_seen <= 1'b0;
                    if (core_start) begin
                        mode_q <= cfg_mode;
                        flags_q <= cfg_flags;
                        m0a_q <= cfg_m0_a;
                        m0b_q <= cfg_m0_b;
                        rescale_q <= cfg_rescale_cfg;
                        rs_shift_large_q <= (cfg_rescale_cfg[7:0] >= 8'd42);
                        rs_shift_small_q <= cfg_rescale_cfg[5:0];
                        rs_round_bias_q <= ((cfg_rescale_cfg[7:0] == 8'd0) ||
                                            (cfg_rescale_cfg[7:0] >= 8'd42))
                                           ? 42'd0
                                           : (42'd1 << (cfg_rescale_cfg[7:0] - 8'd1));
                        rs_zp_out <= cfg_rescale_cfg[31:24];
                        cur_in0 <= cfg_in0_addr_byte;
                        cur_in1 <= cfg_in1_addr_byte;
                        cur_out <= cfg_out_addr_byte;
                        remain_bytes <= cfg_in0_bytes;
                        if (!cfg_mode_ok || !cfg_flags_ok || !cfg_rescale_ok) begin
                            core_error <= 1'b1;
                            core_error_code <= `OP_E_CFG;
                        end else if (!cfg_len_ok || !cfg_add_len_ok) begin
                            core_error <= 1'b1;
                            core_error_code <= `OP_E_CFG;
                        end else if (!cfg_addr_ok) begin
                            core_error <= 1'b1;
                            core_error_code <= `OP_E_ADDR_ALIGN;
                        end else begin
                            core_busy <= 1'b1;
                            state <= S_RD0;
                        end
                    end
                end

                S_RD0: begin
                    if (cmd_fire) begin
                        rd0_done <= 1'b1;
                        read_outstanding <= 1'b1;
                        if (abort_seen || core_abort) begin
                            abort_seen <= 1'b1;
                            state <= S_ABORT;
                        end else begin
                            state <= S_RD0_W;
                        end
                    end else if (core_abort) begin
                        abort_seen <= 1'b1;
                        state <= S_ABORT;
                    end
                end

                S_RD0_W: begin
                    if (rd_beat) begin
                        if (rd_data_error) begin
                            core_error <= 1'b1;
                            core_error_code <= (rd_data_error_code == `OP_E_NONE) ?
                                               `OP_E_AXI_RESP : rd_data_error_code;
                            read_outstanding <= 1'b0;
                            state <= S_ERROR;
                        end else begin
                            a_beat <= rd_data;
                            read_outstanding <= 1'b0;
                            if (abort_seen || core_abort) begin
                                abort_seen <= 1'b1;
                                state <= S_ABORT;
                            end else if (add_mode) begin
                                rd1_done <= 1'b0;
                                state <= S_RD1;
                            end else begin
                                wr_done <= 1'b0;
                                state <= S_WR;
                            end
                        end
                    end
                end

                S_RD1: begin
                    if (cmd_fire) begin
                        rd1_done <= 1'b1;
                        read_outstanding <= 1'b1;
                        if (abort_seen || core_abort) begin
                            abort_seen <= 1'b1;
                            state <= S_ABORT;
                        end else begin
                            state <= S_RD1_W;
                        end
                    end else if (core_abort) begin
                        abort_seen <= 1'b1;
                        state <= S_ABORT;
                    end
                end

                S_RD1_W: begin
                    if (rd_beat) begin
                        if (rd_data_error) begin
                            core_error <= 1'b1;
                            core_error_code <= (rd_data_error_code == `OP_E_NONE) ?
                                               `OP_E_AXI_RESP : rd_data_error_code;
                            read_outstanding <= 1'b0;
                            state <= S_ERROR;
                        end else begin
                            b_beat <= rd_data;
                            read_outstanding <= 1'b0;
                            if (abort_seen || core_abort) begin
                                abort_seen <= 1'b1;
                                state <= S_ABORT;
                            end else if (rescale_mode) begin
                                wr_done <= 1'b0;
                                state <= S_RS_MUL;
                            end else begin
                                wr_done <= 1'b0;
                                state <= S_WR;
                            end
                        end
                    end
                end

                S_RS_MUL: begin
                    for (li = 0; li < DATA_WIDTH/8; li = li + 1) begin
                        reg signed [8:0] ad;
                        reg signed [8:0] bd;
                        ad = $signed(a_beat[li*8 +: 8]) - $signed(rescale_q[15:8]);
                        bd = $signed(b_beat[li*8 +: 8]) - $signed(rescale_q[23:16]);
                        rs_sum[li] <= ad * $signed({1'b0, m0a_q}) +
                                      bd * $signed({1'b0, m0b_q});
                    end
                    state <= S_RS_SHIFT;
                end

                S_RS_SHIFT: begin
                    // revision5 PF2 stage 1: abs(sum) + precomputed bias, then shift.
                    // Exact in 42 bits because |sum| < 2^41; for cfg shift >= 42 the
                    // old 64-bit expression also produces a scaled magnitude of 0.
                    for (li = 0; li < DATA_WIDTH/8; li = li + 1) begin
                        reg [41:0] mag;
                        reg [41:0] rounded;
                        reg [41:0] shifted;
                        mag = rs_sum[li][63] ? ((~rs_sum[li][41:0]) + 42'd1)
                                             : rs_sum[li][41:0];
                        if (rs_shift_large_q) begin
                            shifted = 42'd0;
                        end else begin
                            rounded = mag + rs_round_bias_q;
                            shifted = (rs_shift_small_q == 6'd0)
                                      ? rounded
                                      : (rounded >> rs_shift_small_q);
                        end
                        rs_magnitude[li] <= shifted;
                        rs_negative[li]  <= rs_sum[li][63];
                    end
                    wr_done <= 1'b0;
                    state <= S_WR;
                end

                S_WR: begin
                    // revision5 PF2 stage 2: sign restore + z_out + saturation in
                    // the existing S_WR cycle.  wr_data_valid only rises in the
                    // following S_WR_DATA cycle, so the first possible data beat
                    // already sees the new rs_out.
                    for (li = 0; li < DATA_WIDTH/8; li = li + 1) begin
                        reg signed [42:0] sval;
                        reg signed [42:0] biased;
                        sval = rs_negative[li]
                               ? -$signed({1'b0, rs_magnitude[li]})
                               :  $signed({1'b0, rs_magnitude[li]});
                        biased = sval + $signed({{35{rs_zp_out[7]}}, rs_zp_out});
                        if (biased > 43'sd127)
                            rs_out[li*8 +: 8] <= 8'h7f;
                        else if (biased < -43'sd128)
                            rs_out[li*8 +: 8] <= 8'h80;
                        else
                            rs_out[li*8 +: 8] <= biased[7:0];
                    end
                    if (wr_cmd_fire) begin
                        wr_done <= 1'b1;
                        write_outstanding <= 1'b1;
                        if (abort_seen || core_abort) begin
                            abort_seen <= 1'b1;
                        end
                        state <= S_WR_DATA;
                    end else if (core_abort) begin
                        abort_seen <= 1'b1;
                        state <= S_ABORT;
                    end
                end

                S_WR_DATA: begin
                    if (core_abort) begin
                        abort_seen <= 1'b1;
                    end
                    if (wr_beat) begin
                        state <= S_WR_RSP;
                    end
                end

                S_WR_RSP: begin
                    if (wr_rsp_valid) begin
                        write_outstanding <= 1'b0;
                        if (wr_rsp_error || wr_rsp_aborted) begin
                            core_error <= 1'b1;
                            core_error_code <= (wr_rsp_error_code == `OP_E_NONE) ?
                                               `OP_E_AXI_RESP : wr_rsp_error_code;
                            state <= S_ERROR;
                        end else if (abort_seen || core_abort) begin
                            abort_seen <= 1'b1;
                            state <= S_ABORT;
                        end else if (remain_bytes <= CHUNK_BYTES) begin
                            state <= S_DONE;
                        end else begin
                            remain_bytes <= remain_bytes - CHUNK_BYTES;
                            cur_in0 <= cur_in0 + CHUNK_BYTES;
                            cur_in1 <= cur_in1 + CHUNK_BYTES;
                            cur_out <= cur_out + CHUNK_BYTES;
                            rd0_done <= 1'b0;
                            rd1_done <= 1'b0;
                            wr_done <= 1'b0;
                            state <= S_RD0;
                        end
                    end
                end

                S_DONE: begin
                    core_busy <= 1'b0;
                    core_done <= 1'b1;
                    state <= S_IDLE;
                end

                S_ABORT: begin
                    if (!read_outstanding && !write_outstanding) begin
                        core_busy <= 1'b0;
                        state <= S_IDLE;
                    end
                end

                S_ERROR: begin
                    core_busy <= 1'b0;
                    core_error <= 1'b1;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_rd_tag = |{rd_data_tag, rd_data_last};
    wire unused_wr_tag = |{wr_rsp_tag};
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
