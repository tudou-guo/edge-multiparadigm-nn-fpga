// -----------------------------------------------------------------------------
// int8_mul_core.v -- E1-P3 restricted same-shape INT8 multiply
//
//   S   = x * y                          (signed 8x8 -> exact 16-bit)
//   out = sat8(round_away(S * M0 / 2^SHIFT))
//
// The product S fits INT32 trivially, but S*M0 does not: this core keeps the
// full signed 40-bit product and a 48-bit rounding temporary so no
// intermediate can wrap.  Rounding is ties-away-from-zero, identical to the
// integer reference (_requant / exact_quant.quantize_product).
//
// Contract enforced here (compile time already rejects the others):
//   0 < M0 <= 0xFFFFFF, 0 <= SHIFT <= 31, zero_point = 0,
//   identical byte counts for both inputs and the output.
//
// The core consumes the frozen normalized op_shell read/write service one
// 16-byte chunk at a time and is a drop-in peer of eltwise_core.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps
`include "op_abi_defs.vh"

module int8_mul_core #(
    parameter integer DATA_WIDTH = 128,
    parameter integer CHUNK_BYTES = 16
) (
    input  wire        clk,
    input  wire        rst_n,

    // Locked per-record configuration.
    input  wire [31:0] cfg_in0_addr_byte,
    input  wire [31:0] cfg_in1_addr_byte,
    input  wire [31:0] cfg_out_addr_byte,
    input  wire [31:0] cfg_in0_bytes,
    input  wire [31:0] cfg_in1_bytes,
    input  wire [31:0] cfg_out_bytes,
    input  wire [23:0] cfg_m0,
    input  wire [4:0]  cfg_shift,

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
    localparam [3:0] S_LANE1   = 4'd11;
    localparam [3:0] S_LANE2   = 4'd12;
    localparam [3:0] S_WR      = 4'd5;
    localparam [3:0] S_WR_DATA = 4'd6;
    localparam [3:0] S_WR_RSP  = 4'd7;
    localparam [3:0] S_DONE    = 4'd8;
    localparam [3:0] S_ERROR   = 4'd9;
    localparam [3:0] S_ABORT   = 4'd10;

    localparam [7:0] TAG_RD0 = 8'h10;
    localparam [7:0] TAG_RD1 = 8'h11;
    localparam [7:0] TAG_WR  = 8'h20;

    reg [3:0]  state;
    reg [31:0] remain_bytes;
    reg [31:0] cur_in0;
    reg [31:0] cur_in1;
    reg [31:0] cur_out;
    reg [127:0] a_beat;
    reg [127:0] b_beat;
    reg [23:0] m0_q;
    reg [4:0]  shift_q;
    reg        rd0_done, rd1_done, wr_done;
    reg        read_outstanding, write_outstanding;
    reg        abort_seen;

    wire cmd_fire = rd_cmd_valid && rd_cmd_ready;
    wire wr_cmd_fire = wr_cmd_valid && wr_cmd_ready;
    wire wr_beat = wr_data_valid && wr_data_ready;
    wire rd_beat = rd_data_valid && rd_data_ready;

    wire cfg_len_ok = (cfg_in0_bytes != 0) && (cfg_out_bytes == cfg_in0_bytes) &&
                      (cfg_in1_bytes == cfg_in0_bytes) &&
                      ((cfg_in0_bytes & 32'hF) == 0);
    wire cfg_quant_ok = (cfg_m0 != 24'd0) && (cfg_shift <= 5'd31);
    wire cfg_addr_ok = (cfg_in0_addr_byte[3:0] == 0) &&
                       (cfg_in1_addr_byte[3:0] == 0) &&
                       (cfg_out_addr_byte[3:0] == 0);

    // ---- 16-lane 3-stage requant pipeline ----------------------------------
    // The 5ns implementation failed with a single-cycle multiply + 48-bit
    // rounding cone (WNS -2.674ns, 34 logic levels).  The lane is therefore
    // split into registered product, registered rounding and a registered
    // output; the FSM adds S_LANE1/S_LANE2 so the write data is stable.
    reg  signed [39:0] lane_wide_q [0:15];
    reg  signed [7:0]  lane_out_q  [0:15];

    genvar lane;
    generate
        for (lane = 0; lane < DATA_WIDTH/8; lane = lane + 1) begin : G_LANE
            wire signed [7:0]  av = a_beat[lane*8 +: 8];
            // In the read-completion cycle the second beat is still arriving on
            // rd_data, so the product must use that value (b_beat updates with
            // a nonblocking assignment at the same edge).
            wire signed [7:0]  bv = (state == S_RD1_W) ? rd_data[lane*8 +: 8]
                                                       : b_beat[lane*8 +: 8];
            wire signed [15:0] product = av * bv;
            wire signed [39:0] wide_now = product * $signed({16'd0, m0_q});
            wire signed [39:0] wide_q = lane_wide_q[lane];
            wire [39:0] magnitude = wide_q[39] ? (~wide_q + 40'd1) : wide_q;
            wire [39:0] half = (shift_q == 5'd0) ? 40'd0
                                                : (40'd1 << (shift_q - 5'd1));
            wire [39:0] rounded_abs = (magnitude + half) >> shift_q;
            wire signed [39:0] rounded = wide_q[39] ? -$signed(rounded_abs)
                                                   : $signed(rounded_abs);
            always @(posedge clk) begin
                if (state == S_RD1_W && rd_beat)
                    lane_wide_q[lane] <= wide_now;
                if (state == S_LANE1)
                    lane_out_q[lane] <= (rounded > 40'sd127) ? 8'sd127 :
                                        (rounded < -40'sd128) ? -8'sd128 :
                                        rounded[7:0];
            end
        end
    endgenerate

    reg [DATA_WIDTH-1:0] wr_data_q;
    integer lane_i;
    always @(*) begin
        wr_data_q = {DATA_WIDTH{1'b0}};
        for (lane_i = 0; lane_i < DATA_WIDTH/8; lane_i = lane_i + 1)
            wr_data_q[lane_i*8 +: 8] = lane_out_q[lane_i];
    end

    always @(*) begin
        rd_cmd_valid     = ((state == S_RD0) && !rd0_done) ||
                           ((state == S_RD1) && !rd1_done);
        rd_cmd_addr_byte = (state == S_RD1) ? cur_in1 : cur_in0;
        rd_cmd_bytes     = CHUNK_BYTES;
        rd_cmd_tag       = (state == S_RD1) ? TAG_RD1 : TAG_RD0;

        rd_data_ready = (state == S_RD0_W) || (state == S_RD1_W) ||
                        ((state == S_ABORT) && read_outstanding);

        wr_cmd_valid     = (state == S_WR) && !wr_done;
        wr_cmd_addr_byte = cur_out;
        wr_cmd_bytes     = CHUNK_BYTES;
        wr_cmd_tag       = TAG_WR;

        wr_data_valid = (state == S_WR_DATA);
        wr_data       = (state == S_WR_DATA) ? wr_data_q : {DATA_WIDTH{1'b0}};
        wr_data_last  = 1'b1;
        wr_data_tag   = TAG_WR;
        wr_rsp_ready  = (state == S_WR_RSP) ||
                        ((state == S_ABORT) && write_outstanding);
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state               <= S_IDLE;
            remain_bytes        <= 32'b0;
            cur_in0             <= 32'b0;
            cur_in1             <= 32'b0;
            cur_out             <= 32'b0;
            a_beat              <= 128'b0;
            b_beat              <= 128'b0;
            m0_q                <= 24'b0;
            shift_q             <= 5'b0;
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
        end else if (core_abort) begin
            // Immediate abort, matching the pool core: the operator stops and
            // releases busy in the same cycle.  The shell owns the in-flight
            // AXI cleanup; a new START re-arms this core from S_IDLE.
            // wr_data_valid is a pure function of state (combinational); the
            // abort only needs to leave S_WR_DATA, which deasserts it.
            state               <= S_IDLE;
            core_busy           <= 1'b0;
            core_done           <= 1'b0;
            read_outstanding    <= 1'b0;
            write_outstanding   <= 1'b0;
            abort_seen          <= 1'b0;
            rd0_done            <= 1'b0;
            rd1_done            <= 1'b0;
            wr_done             <= 1'b0;
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
                        cur_in0 <= cfg_in0_addr_byte;
                        cur_in1 <= cfg_in1_addr_byte;
                        cur_out <= cfg_out_addr_byte;
                        remain_bytes <= cfg_in0_bytes;
                        m0_q <= cfg_m0;
                        shift_q <= cfg_shift;
                        if (!cfg_len_ok) begin
                            core_error <= 1'b1;
                            core_error_code <= `OP_E_CFG;
                        end else if (!cfg_quant_ok) begin
                            core_error <= 1'b1;
                            core_error_code <= `OP_E_QUANT;
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
                            end else begin
                                rd1_done <= 1'b0;
                                state <= S_RD1;
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
                            end else begin
                                wr_done <= 1'b0;
                                state <= S_LANE1;
                            end
                        end
                    end
                end

                S_LANE1: begin
                    if (core_abort) begin
                        abort_seen <= 1'b1;
                        state <= S_ABORT;
                    end else begin
                        state <= S_LANE2;
                    end
                end

                S_LANE2: begin
                    if (core_abort) begin
                        abort_seen <= 1'b1;
                        state <= S_ABORT;
                    end else begin
                        state <= S_WR;
                    end
                end

                S_WR: begin
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