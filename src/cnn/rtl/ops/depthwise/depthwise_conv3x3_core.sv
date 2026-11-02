// -----------------------------------------------------------------------------
// depthwise_conv3x3_core.sv -- parameterized, decoupled DepthwiseConv3x3 core
//
// Batch: depthwise performance (2026-10-24).  Replaces the frozen B0 serial FSM
// (parent sha256 5055814f...) with a three-stage decoupled pipeline:
//
//   READ ENGINE (param load + row prefetch)  ->  4-row cache
//   MAC ENGINE  (DW_PC lanes, 1 tap/cycle)   ->  2 acc banks + group metadata
//   RQ ENGINE   (DW_RQ_LANES lanes/cycle)    ->  output row assembly
//   DRAIN ENGINE (1 x 128-bit beat/cycle)    ->  AXI write service
//
// Contract (frozen before coding):
//   build/onnx_e1/depthwise_performance_20261024_201800_a1/contracts/
//     benchmark_contract.json, storage_ownership_contract.json, dw_profiles.json
//
// Numerics / protocol / layout are unchanged from the frozen Depthwise contract:
//   acc[c] = int32_wrap(bias[c] + sum int8*int8)
//   mag    = (|acc*M0| + (SHIFT==0?0:1<<(SHIFT-1))) >> SHIFT
//   y      = sat8(neg ? -mag : +mag)                     (ties-away, zp=0)
//   NHWC8_A16, ABI tags/errors, DONE only after the last write response,
//   abort/reset convergence.
//
// Parameters:
//   DW_PC          : MAC logical channels per group.  Legal 4/8/16 (default 8).
//   DW_TAP_PARALLEL: taps per channel per cycle.  Implemented 1 (9-cycle group
//                    issue); 3 is assessed in the architecture card and not
//                    implemented in this batch (no third synthesis group).
//   DW_RQ_LANES    : requant lanes per cycle.  Legal 1/2; PC16 pairs with 2 so
//                    ceil(PC/RQ)=8 <= 9.  PC16 with RQ1 is legal/configurable
//                    but its RQ rate bounds II to 16.
//   MAX_CH / MAX_W : capacity only; independent of C and DW_PC.
//
// Steady state (rows ready, banks free, downstream accepting):
//   * group issue interval 9 cycles; MAC accumulate continuous; RQ consumes a
//     group in ceil(PC/RQ) <= 8 cycles and overlaps the next MAC group;
//   * pointer/output LIFO/QUEUE ownership is explicit (see storage contract):
//       - row cache slot: 4 tags + in-flight kill; writable only when its tag
//         is outside the in-range window still needed;
//       - acc banks: MAC writes group n%2; RQ consumes in order, releases after
//         the last lane read; MAC starts group n+2 only when n is released;
//       - output banks: compute writes row oy%2; drain reads (oy-1)%2; row oy
//         starts only after row oy-2 was drained (rows_drained >= oy-1).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module depthwise_conv3x3_core #(
    parameter integer DATA_WIDTH      = 128,
    parameter integer CHUNK_BYTES     = 16,
    parameter integer MAX_CH          = 16,
    parameter integer MAX_W           = 32,
    parameter integer DW_PC           = 8,
    parameter integer DW_TAP_PARALLEL = 1,
    parameter integer DW_RQ_LANES     = 1
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire [31:0] cfg_in_addr_byte,
    input  wire [31:0] cfg_weight_addr_byte,
    input  wire [31:0] cfg_bias_addr_byte,
    input  wire [31:0] cfg_quant_addr_byte,
    input  wire [31:0] cfg_out_addr_byte,
    input  wire [15:0] cfg_h,
    input  wire [15:0] cfg_w,
    input  wire [15:0] cfg_cin,
    input  wire [15:0] cfg_cout,
    input  wire [15:0] cfg_calloc,
    input  wire [1:0]  cfg_stride,
    input  wire [1:0]  cfg_pad,
    input  wire [31:0] cfg_in_row_stride_bytes,
    input  wire [31:0] cfg_out_row_stride_bytes,
    input  wire [31:0] cfg_in_bytes,
    input  wire [31:0] cfg_weight_bytes,
    input  wire [31:0] cfg_bias_bytes,
    input  wire [31:0] cfg_quant_bytes,
    input  wire [31:0] cfg_out_bytes,

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

    // ------------------------------------------------------------------
    // Frozen ABI error codes (op_abi_defs.vh values; no macro dependency)
    // ------------------------------------------------------------------
    localparam [15:0] OP_E_NONE          = 16'h0000;
    localparam [15:0] OP_E_CFG           = 16'h0001;
    localparam [15:0] OP_E_ADDR_ALIGN    = 16'h0003;
    localparam [15:0] OP_E_ADDR_RANGE    = 16'h0004;
    localparam [15:0] OP_E_STRIDE        = 16'h0005;
    localparam [15:0] OP_E_QUANT         = 16'h0006;
    localparam [15:0] OP_E_AXI_RESP      = 16'h0008;
    localparam [15:0] OP_E_AXI_PROTOCOL  = 16'h0009;
    localparam [15:0] OP_E_CORE_INTERNAL = 16'h000A;

    // ------------------------------------------------------------------
    // Derived constants
    // ------------------------------------------------------------------
    localparam integer MAX_H        = 32;
    localparam integer PB           = DW_PC;
    localparam integer PR           = DW_RQ_LANES;
    localparam integer MAX_GROUPS   = (MAX_CH + PB - 1) / PB;
    localparam integer PB_SHIFT     = (PB == 4) ? 2 : (PB == 8) ? 3 : 4;
    localparam integer MAX_ROW_WORDS= ((MAX_W * MAX_CH) + 15) / 16;
    localparam integer CACHE_WORDS  = 4 * MAX_ROW_WORDS;
    localparam integer CACHE_AW     = (CACHE_WORDS <= 2) ? 1 : $clog2(CACHE_WORDS);
    localparam integer OB_WORDS     = 2 * MAX_ROW_WORDS;
    localparam integer OB_AW        = (OB_WORDS <= 2) ? 1 : $clog2(OB_WORDS);
    localparam integer OB_ADDR_W    = (MAX_ROW_WORDS <= 2) ? 1 : $clog2(MAX_ROW_WORDS) + 1;

    localparam [7:0] TAG_INPUT  = 8'h01;
    localparam [7:0] TAG_WEIGHT = 8'h02;
    localparam [7:0] TAG_BIAS   = 8'h03;
    localparam [7:0] TAG_QUANT  = 8'h04;
    localparam [7:0] TAG_OUTPUT = 8'h05;

    localparam [1:0] K_W = 2'd0, K_B = 2'd1, K_Q = 2'd2, K_ROW = 2'd3;

    localparam [3:0] L_IDLE      = 4'd0;
    localparam [3:0] L_VALIDATE  = 4'd1;
    localparam [3:0] L_VALIDATE2 = 4'd2;
    localparam [3:0] L_VALIDATE3 = 4'd3;
    localparam [3:0] L_VALIDATE4 = 4'd4;
    localparam [3:0] L_PREQ      = 4'd5;
    localparam [3:0] L_PWAIT     = 4'd6;
    localparam [3:0] L_RUN       = 4'd7;
    localparam [3:0] L_DONE      = 4'd8;
    localparam [3:0] L_ERROR     = 4'd9;
    localparam [3:0] L_ABORT     = 4'd10;

    localparam [2:0] RD_IDLE = 3'd0, RD_CMD = 3'd1, RD_DATA = 3'd2;
    localparam [1:0] OD_IDLE = 2'd0, OD_CMD = 2'd1, OD_DATA = 2'd2, OD_RSP = 2'd3;
    localparam [1:0] RQ_IDLE = 2'd0, RQ_ACT = 2'd1;

    // ------------------------------------------------------------------
    // Simulation-only parameter legality check (toolchain rejects builds too)
    // ------------------------------------------------------------------
    // synthesis translate_off
    initial begin
        if ((DATA_WIDTH != 128) || (CHUNK_BYTES != 16)) begin
            $display("DW_FATAL: DATA_WIDTH/CHUNK_BYTES must be 128/16"); $finish;
        end
        if ((DW_PC != 4) && (DW_PC != 8) && (DW_PC != 16)) begin
            $display("DW_FATAL: unsupported DW_PC=%0d (legal 4/8/16)", DW_PC); $finish;
        end
        if (DW_TAP_PARALLEL != 1) begin
            $display("DW_FATAL: DW_TAP_PARALLEL=%0d assessed-but-not-implemented", DW_TAP_PARALLEL); $finish;
        end
        if ((DW_RQ_LANES != 1) && (DW_RQ_LANES != 2)) begin
            $display("DW_FATAL: unsupported DW_RQ_LANES=%0d", DW_RQ_LANES); $finish;
        end
        if (DW_RQ_LANES > DW_PC) begin
            $display("DW_FATAL: DW_RQ_LANES > DW_PC"); $finish;
        end
        if (DW_PC > MAX_CH) begin
            $display("DW_FATAL: DW_PC=%0d exceeds MAX_CH=%0d", DW_PC, MAX_CH); $finish;
        end
        if (MAX_CH > 16) begin
            $display("DW_FATAL: MAX_CH=%0d exceeds frozen capacity 16", MAX_CH); $finish;
        end
    end
    // synthesis translate_on

    // ------------------------------------------------------------------
    // Launch configuration / geometry
    // ------------------------------------------------------------------
    reg [31:0] cf_in_addr, cf_w_addr, cf_b_addr, cf_q_addr, cf_out_addr;
    reg [15:0] cf_h, cf_w, cf_cin, cf_cout, cf_calloc;
    reg [1:0]  cf_stride, cf_pad;
    reg [31:0] cf_in_stride, cf_out_stride;
    reg [31:0] cf_in_bytes, cf_w_bytes, cf_b_bytes, cf_q_bytes, cf_out_bytes;

    reg [15:0] g_hout, g_wout;
    reg [31:0] g_in_stride, g_out_stride;
    reg [15:0] g_in_beats, g_out_words;
    reg [2:0]  g_ngroups;
    reg        g_geom_ok;
    reg [31:0] g_exp_in_bytes, g_exp_out_bytes, g_exp_w_bytes, g_exp_b_bytes, g_exp_q_bytes;
    reg [15:0] g_exp_calloc;
    reg [15:0] g_fail_code;

    wire g_calloc16 = (cf_calloc == 16'd16);

    wire signed [17:0] h_ext_s = $signed({2'b0, cf_h}) + $signed({16'b0, cf_pad, 1'b0}) - 18'sd3;
    wire signed [17:0] w_ext_s = $signed({2'b0, cf_w}) + $signed({16'b0, cf_pad, 1'b0}) - 18'sd3;
    wire signed [17:0] hout_s  = (cf_stride == 2'd2) ? ((h_ext_s >>> 1) + 18'sd1) : (h_ext_s + 18'sd1);
    wire signed [17:0] wout_s  = (cf_stride == 2'd2) ? ((w_ext_s >>> 1) + 18'sd1) : (w_ext_s + 18'sd1);
    wire [31:0] row_bytes_nom  = (cf_calloc == 16'd16) ? ({16'b0, cf_w} << 4)
                                                       : ({16'b0, cf_w} << 3);
    wire [31:0] in_stride_ext  = (row_bytes_nom + 32'd15) & 32'hFFFF_FFF0;
    wire        wout_positive  = (wout_s >= 18'sd1);
    wire [31:0] out_bytes_nom  = (cf_calloc == 16'd16) ? ({16'b0, wout_s[15:0]} << 4)
                                                       : ({16'b0, wout_s[15:0]} << 3);
    wire [31:0] out_stride_ext = wout_positive
        ? ((out_bytes_nom + 32'd15) & 32'hFFFF_FFF0)
        : 32'd0;
    wire [16:0] ngroups_ext = ({1'b0, cf_cin} + (PB - 1)) >> PB_SHIFT;
    // narrowed expected-byte products (h,w <= 32, row stride <= 512)
    wire [15:0] exp_in_w  = cf_h[5:0]  * g_in_stride[9:0];
    wire [15:0] exp_out_w = g_hout[5:0] * g_out_stride[9:0];

    wire [32:0] end_in  = {1'b0, cf_in_addr}  + {1'b0, cf_in_bytes};
    wire [32:0] end_w   = {1'b0, cf_w_addr}   + {1'b0, cf_w_bytes};
    wire [32:0] end_b   = {1'b0, cf_b_addr}   + {1'b0, cf_b_bytes};
    wire [32:0] end_q   = {1'b0, cf_q_addr}   + {1'b0, cf_q_bytes};
    wire [32:0] end_out = {1'b0, cf_out_addr} + {1'b0, cf_out_bytes};

    wire v_attr_bad  = !((cf_stride == 2'd1) || (cf_stride == 2'd2)) ||
                       ((cf_pad != 2'd0) && (cf_pad != 2'd1));
    wire v_shape_bad = (cf_h == 16'd0) || (cf_w == 16'd0) ||
                       (cf_h > MAX_H) || (cf_w > MAX_W) ||
                       (cf_cin == 16'd0) || (cf_cin > MAX_CH) ||
                       (cf_cin != cf_cout) ||
                       (cf_calloc != g_exp_calloc) ||
                       ((cf_calloc != 16'd8) && (cf_calloc != 16'd16)) ||
                       (ngroups_ext > MAX_GROUPS) || (ngroups_ext == 17'd0);
    wire v_geom_bad  = !g_geom_ok;
    wire v_stride_bad = (cf_in_stride != g_in_stride) ||
                        (cf_out_stride != g_out_stride) ||
                        (cf_in_bytes != g_exp_in_bytes) ||
                        (cf_out_bytes != g_exp_out_bytes) ||
                        (cf_w_bytes != g_exp_w_bytes) ||
                        (cf_b_bytes != g_exp_b_bytes);
    wire v_quant_bad = (cf_q_bytes != g_exp_q_bytes);
    wire v_align_bad = (cf_in_addr[3:0] != 4'b0) || (cf_w_addr[3:0] != 4'b0) ||
                       (cf_b_addr[3:0] != 4'b0) || (cf_q_addr[3:0] != 4'b0) ||
                       (cf_out_addr[3:0] != 4'b0);
    wire v_wrap_bad  = end_in[32] || end_w[32] || end_b[32] || end_q[32] || end_out[32];
    wire v_zero_bad  = (cf_in_addr == 32'b0) || (cf_w_addr == 32'b0) ||
                       (cf_b_addr == 32'b0) || (cf_q_addr == 32'b0) ||
                       (cf_out_addr == 32'b0);
    wire v_overlap_bad =
        ((cf_in_addr  < end_w)  && (cf_w_addr   < end_in))  ||
        ((cf_in_addr  < end_b)  && (cf_b_addr   < end_in))  ||
        ((cf_in_addr  < end_q)  && (cf_q_addr   < end_in))  ||
        ((cf_in_addr  < end_out)&& (cf_out_addr < end_in))  ||
        ((cf_w_addr   < end_b)  && (cf_b_addr   < end_w))   ||
        ((cf_w_addr   < end_q)  && (cf_q_addr   < end_w))   ||
        ((cf_w_addr   < end_out)&& (cf_out_addr < end_w))   ||
        ((cf_b_addr   < end_q)  && (cf_q_addr   < end_b))   ||
        ((cf_b_addr   < end_out)&& (cf_out_addr < end_b))   ||
        ((cf_q_addr   < end_out)&& (cf_out_addr < end_q));
    wire v_range_bad = v_wrap_bad || v_zero_bad || v_overlap_bad;
    wire [15:0] v_fail_code_w =
        (v_attr_bad || v_shape_bad || v_geom_bad) ? OP_E_CFG :
        v_stride_bad ? OP_E_STRIDE :
        v_align_bad  ? OP_E_ADDR_ALIGN :
        v_range_bad  ? OP_E_ADDR_RANGE :
        v_quant_bad  ? OP_E_QUANT : OP_E_NONE;

    // ------------------------------------------------------------------
    // Lifecycle / global control
    // ------------------------------------------------------------------
    reg [3:0]  state;
    reg [1:0]  start_pipe;
    reg [1:0]  ld_kind;
    reg [15:0] fatal_code;
    reg        abort_req;
    reg        launch_tick;

    reg        rd_err_pulse;
    reg [15:0] rd_err_code;
    reg        od_err_pulse;
    reg [15:0] od_err_code;

    wire fatal_flag = (fatal_code != 16'h0);
    wire soft_rst   = core_soft_reset;
    wire stop_new   = abort_req || fatal_flag || soft_rst;
    wire run_enable = (state == L_RUN) && !stop_new;

    // request bus (combinational mux, single driver)
    wire        req_valid;
    wire [1:0]  req_kind;
    wire [31:0] req_addr;
    wire [15:0] req_bytes;
    wire [15:0] req_row_y;
    wire [1:0]  req_row_slot;
    wire        req_ready;
    reg         req_done;

    // row fetch request (row scheduler -> arbiter)
    wire        fetch_req_valid;
    wire [31:0] fetch_req_addr;
    wire [15:0] fetch_req_bytes;
    wire [15:0] fetch_req_y;
    wire [1:0]  fetch_req_slot;

    // cross-engine RQ state used by the MAC bank-release handshake
    reg [1:0]  rq_phase;
    reg [1:0]  rq_bank;
    reg [15:0] rq_lane_idx;

    // cross-engine group counters (MAC owns issued/done, RQ owns consumed)
    reg [31:0] grp_issued;
    reg [31:0] grp_done;
    reg [31:0] grp_consumed;

    wire rd_busy_w;
    wire od_busy_w;
    wire [15:0] rows_computed_w;
    wire [15:0] rows_drained_w;
    wire row_last_word;
    wire [15:0] row_begin_oy;
    wire signed [17:0] row_begin_yb;
    wire        row_begin;

    // ------------------------------------------------------------------
    // Main lifecycle FSM
    // ------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= L_IDLE;
            start_pipe      <= 2'b0;
            launch_tick     <= 1'b0;
            cf_in_addr      <= 32'b0;  cf_w_addr   <= 32'b0;
            cf_b_addr       <= 32'b0;  cf_q_addr   <= 32'b0;
            cf_out_addr     <= 32'b0;
            cf_h            <= 16'd0;  cf_w        <= 16'd0;
            cf_cin          <= 16'd0;  cf_cout     <= 16'd0;
            cf_calloc       <= 16'd0;
            cf_stride       <= 2'd0;   cf_pad      <= 2'd0;
            cf_in_stride    <= 32'd0;  cf_out_stride <= 32'd0;
            cf_in_bytes     <= 32'd0;  cf_w_bytes  <= 32'd0;
            cf_b_bytes      <= 32'd0;  cf_q_bytes  <= 32'd0;
            cf_out_bytes    <= 32'd0;
            g_hout          <= 16'd0;  g_wout      <= 16'd0;
            g_in_stride     <= 32'd0;  g_out_stride<= 32'd0;
            g_in_beats      <= 16'd0;  g_out_words <= 16'd0;
            g_ngroups       <= 3'd0;   g_geom_ok   <= 1'b0;
            g_exp_in_bytes  <= 32'd0;  g_exp_out_bytes <= 32'd0;
            g_exp_w_bytes   <= 32'd0;  g_exp_b_bytes  <= 32'd0;
            g_exp_q_bytes   <= 32'd0;  g_exp_calloc   <= 16'd0;
            g_fail_code     <= OP_E_NONE;
            ld_kind         <= 2'd0;
            fatal_code      <= 16'd0;
            abort_req       <= 1'b0;
            core_busy       <= 1'b0;
            core_done       <= 1'b0;
            core_error      <= 1'b0;
            core_error_code <= OP_E_NONE;
        end else if (core_soft_reset) begin
            state           <= L_IDLE;
            launch_tick     <= 1'b0;
            core_busy       <= 1'b0;
            core_done       <= 1'b0;
            core_error      <= 1'b0;
            core_error_code <= OP_E_NONE;
            abort_req       <= 1'b0;
            fatal_code      <= 16'd0;
        end else begin
            start_pipe  <= {start_pipe[0], core_start};
            launch_tick <= 1'b0;

            if (core_abort && (state != L_IDLE))
                abort_req <= 1'b1;

            if (rd_err_pulse)
                fatal_code <= rd_err_code;
            else if (od_err_pulse)
                fatal_code <= od_err_code;

            if (fatal_flag && (state != L_ERROR) && (state != L_IDLE)) begin
                state <= L_ERROR;
            end else begin
                case (state)
                    L_IDLE: begin
                        core_busy       <= 1'b0;
                        core_done       <= 1'b0;
                        core_error      <= 1'b0;
                        core_error_code <= OP_E_NONE;
                        abort_req       <= 1'b0;
                        if (start_pipe[0]) begin
                            cf_in_addr   <= cfg_in_addr_byte;
                            cf_w_addr    <= cfg_weight_addr_byte;
                            cf_b_addr    <= cfg_bias_addr_byte;
                            cf_q_addr    <= cfg_quant_addr_byte;
                            cf_out_addr  <= cfg_out_addr_byte;
                            cf_h         <= cfg_h;      cf_w      <= cfg_w;
                            cf_cin       <= cfg_cin;    cf_cout   <= cfg_cout;
                            cf_calloc    <= cfg_calloc;
                            cf_stride    <= cfg_stride; cf_pad    <= cfg_pad;
                            cf_in_stride <= cfg_in_row_stride_bytes;
                            cf_out_stride<= cfg_out_row_stride_bytes;
                            cf_in_bytes  <= cfg_in_bytes;
                            cf_w_bytes   <= cfg_weight_bytes;
                            cf_b_bytes   <= cfg_bias_bytes;
                            cf_q_bytes   <= cfg_quant_bytes;
                            cf_out_bytes <= cfg_out_bytes;
                            fatal_code   <= 16'd0;
                            abort_req    <= 1'b0;
                            launch_tick  <= 1'b1;
                            core_busy    <= 1'b1;
                            core_done    <= 1'b0;
                            core_error   <= 1'b0;
                            core_error_code <= OP_E_NONE;
                            state        <= L_VALIDATE;
                        end
                    end

                    L_VALIDATE: begin
                        g_hout       <= hout_s[15:0];
                        g_wout       <= wout_s[15:0];
                        g_in_stride  <= in_stride_ext;
                        g_in_beats   <= in_stride_ext[15:4];
                        g_out_stride <= out_stride_ext;
                        g_out_words  <= out_stride_ext[15:4];
                        g_ngroups    <= ngroups_ext[2:0];
                        g_geom_ok    <= (hout_s >= 18'sd1) && (wout_s >= 18'sd1) &&
                                        (ngroups_ext != 17'd0) && (ngroups_ext <= MAX_GROUPS);
                        state        <= L_VALIDATE2;
                    end

                    L_VALIDATE2: begin
                        g_exp_in_bytes  <= {16'b0, exp_in_w};
                        g_exp_out_bytes <= {16'b0, exp_out_w};
                        g_exp_w_bytes   <= {16'b0, cf_cin} << 4;
                        g_exp_b_bytes   <= (({16'b0, cf_cin} << 2) + 32'd15) & 32'hFFFF_FFF0;
                        g_exp_q_bytes   <= (({16'b0, cf_cin} << 3) + 32'd15) & 32'hFFFF_FFF0;
                        g_exp_calloc    <= ({16'b0, cf_cin} + 16'd7) & 16'hFFF8;
                        state           <= L_VALIDATE3;
                    end

                    L_VALIDATE3: begin
                        g_fail_code <= v_fail_code_w;
                        state       <= L_VALIDATE4;
                    end

                    L_VALIDATE4: begin
                        if (g_fail_code != OP_E_NONE) begin
                            state           <= L_ERROR;
                            core_error      <= 1'b1;
                            core_error_code <= g_fail_code;
                            core_busy       <= 1'b0;
                        end else begin
                            ld_kind <= 2'd0;
                            state   <= L_PREQ;
                        end
                    end

                    L_PREQ: begin
                        if (req_ready) begin
                            state <= L_PWAIT;
                        end else if (abort_req) begin
                            state <= L_ABORT;
                        end
                    end

                    L_PWAIT: begin
                        if (req_done) begin
                            if (ld_kind == 2'd2) begin
                                state <= L_RUN;
                            end else begin
                                ld_kind <= ld_kind + 2'd1;
                                state   <= L_PREQ;
                            end
                        end else if (abort_req) begin
                            state <= L_ABORT;
                        end
                    end

                    L_RUN: begin
                        if (abort_req) begin
                            state <= L_ABORT;
                        end else if ((rows_drained_w == g_hout) &&
                                     (rows_computed_w == g_hout) &&
                                     !od_busy_w && !rd_busy_w) begin
                            state     <= L_DONE;
                            core_done <= 1'b1;
                            core_busy <= 1'b0;
                        end
                    end

                    L_DONE: begin
                        core_busy <= 1'b0;
                        core_done <= 1'b1;
                        state     <= L_IDLE;
                    end

                    L_ERROR: begin
                        core_busy       <= 1'b0;
                        core_done       <= 1'b0;
                        core_error      <= 1'b1;
                        core_error_code <= fatal_flag ? fatal_code : g_fail_code;
                        state           <= L_IDLE;
                    end

                    L_ABORT: begin
                        core_busy       <= 1'b0;
                        core_done       <= 1'b0;
                        core_error      <= 1'b0;
                        core_error_code <= OP_E_NONE;
                        if (!rd_busy_w && !od_busy_w) begin
                            abort_req <= 1'b0;
                            state     <= L_IDLE;
                        end
                    end

                    default: begin
                        state           <= L_IDLE;
                        core_error      <= 1'b1;
                        core_error_code <= OP_E_CORE_INTERNAL;
                        core_busy       <= 1'b0;
                    end
                endcase
            end
        end
    end

    // request mux (combinational, no storage)
    assign req_valid    = (state == L_RUN) ? fetch_req_valid :
                          (state == L_PREQ) ? 1'b1 : 1'b0;
    assign req_kind     = (state == L_RUN) ? K_ROW :
                          (ld_kind == 2'd0) ? K_W :
                          (ld_kind == 2'd1) ? K_B : K_Q;
    assign req_addr     = (state == L_RUN) ? fetch_req_addr :
                          (ld_kind == 2'd0) ? cf_w_addr :
                          (ld_kind == 2'd1) ? cf_b_addr : cf_q_addr;
    assign req_bytes    = (state == L_RUN) ? fetch_req_bytes :
                          (ld_kind == 2'd0) ? cf_w_bytes[15:0] :
                          (ld_kind == 2'd1) ? cf_b_bytes[15:0] : cf_q_bytes[15:0];
    assign req_row_y    = (state == L_RUN) ? fetch_req_y : 16'd0;
    assign req_row_slot = (state == L_RUN) ? fetch_req_slot : 2'd0;

    // ------------------------------------------------------------------
    // Parameter storage
    // ------------------------------------------------------------------
    reg [MAX_CH*8-1:0] wt_t [0:8];
    reg signed [31:0]  bias_flat [0:MAX_CH-1];
    reg        [31:0]  m0_flat   [0:MAX_CH-1];
    reg        [7:0]   shift_flat[0:MAX_CH-1];

    // ------------------------------------------------------------------
    // Row cache / output row buffer
    // ------------------------------------------------------------------
    reg        cache_valid [0:3];
    reg [15:0] cache_tag   [0:3];
    reg        cache_wen;
    reg [CACHE_AW-1:0] cache_waddr;
    reg [DATA_WIDTH-1:0] cache_wdata;
    wire [DATA_WIDTH-1:0] cache_rdata;
    reg        mac_cache_ren;
    reg [CACHE_AW-1:0] mac_cache_raddr;

    reg        ob_wen;
    reg [OB_AW-1:0] ob_waddr;
    reg [DATA_WIDTH-1:0] ob_wdata;
    wire [DATA_WIDTH-1:0] ob_rdata;
    reg        ob_ren;
    reg [OB_AW-1:0] ob_raddr;

    spram #(.DP(CACHE_WORDS), .DW(DATA_WIDTH), .PIPE(1)) u_row_cache (
        .clk(clk), .wdata(cache_wdata), .wen(cache_wen), .waddr(cache_waddr),
        .ren(mac_cache_ren), .raddr(mac_cache_raddr), .rdata(cache_rdata));

    spram #(.DP(OB_WORDS), .DW(DATA_WIDTH), .PIPE(1)) u_out_rowbuf (
        .clk(clk), .wdata(ob_wdata), .wen(ob_wen), .waddr(ob_waddr),
        .ren(ob_ren), .raddr(ob_raddr), .rdata(ob_rdata));

    // ------------------------------------------------------------------
    // READ ENGINE: parameter load + row prefetch; one burst in flight
    // ------------------------------------------------------------------
    reg [2:0]  rd_state;
    reg [1:0]  rd_kind;
    reg [31:0] rd_addr;
    reg [15:0] rd_beats;
    reg [15:0] rd_beat;
    reg [15:0] rd_row_y;
    reg [1:0]  rd_row_slot;
    reg        rd_aborting;

    wire rd_idle = (rd_state == RD_IDLE);
    assign req_ready = rd_idle && !fatal_flag && !soft_rst;
    assign rd_busy_w = (rd_state != RD_IDLE);

    wire [15:0] rd_ch_base = {1'b0, rd_beat[3:0]} << 2;
    wire [15:0] rd_q_base  = {1'b0, rd_beat[3:0]} << 1;

    integer di;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_state      <= RD_IDLE;
            rd_kind       <= K_W;
            rd_addr       <= 32'd0;
            rd_beats      <= 16'd0;
            rd_beat       <= 16'd0;
            rd_row_y      <= 16'd0;
            rd_row_slot   <= 2'd0;
            rd_aborting   <= 1'b0;
            rd_cmd_valid  <= 1'b0;
            rd_cmd_addr_byte <= 32'd0;
            rd_cmd_bytes  <= 16'd0;
            rd_cmd_tag    <= TAG_INPUT;
            rd_data_ready <= 1'b0;
            cache_wen     <= 1'b0;
            cache_waddr   <= {CACHE_AW{1'b0}};
            cache_wdata   <= {DATA_WIDTH{1'b0}};
            for (di = 0; di < 4; di = di + 1) begin
                cache_valid[di] <= 1'b0;
                cache_tag[di]   <= 16'd0;
            end
            for (di = 0; di < 9; di = di + 1)
                wt_t[di] <= {(MAX_CH*8){1'b0}};
            for (di = 0; di < MAX_CH; di = di + 1) begin
                bias_flat[di]  <= 32'sd0;
                m0_flat[di]    <= 32'd0;
                shift_flat[di] <= 8'd0;
            end
        end else if (core_soft_reset) begin
            rd_state      <= RD_IDLE;
            rd_cmd_valid  <= 1'b0;
            rd_data_ready <= 1'b0;
            cache_wen     <= 1'b0;
            rd_aborting   <= 1'b0;
            req_done      <= 1'b0;
        end else begin
            cache_wen     <= 1'b0;
            rd_cmd_valid  <= 1'b0;
            rd_data_ready <= 1'b0;
            rd_err_pulse  <= 1'b0;
            req_done      <= 1'b0;

            if (launch_tick) begin
                rd_state      <= RD_IDLE;
                rd_cmd_valid  <= 1'b0;
                rd_data_ready <= 1'b0;
                cache_wen     <= 1'b0;
                rd_aborting   <= 1'b0;
                rd_beat       <= 16'd0;
                rd_err_pulse  <= 1'b0;
                req_done      <= 1'b0;
                for (di = 0; di < 4; di = di + 1) begin
                    cache_valid[di] <= 1'b0;
                    cache_tag[di]   <= 16'd0;
                end
                for (di = 0; di < 9; di = di + 1)
                    wt_t[di] <= {(MAX_CH*8){1'b0}};
                for (di = 0; di < MAX_CH; di = di + 1) begin
                    bias_flat[di]  <= 32'sd0;
                    m0_flat[di]    <= 32'd0;
                    shift_flat[di] <= 8'd0;
                end
            end else if (fatal_flag) begin
                rd_state      <= RD_IDLE;
                rd_cmd_valid  <= 1'b0;
                rd_data_ready <= 1'b0;
                rd_err_pulse  <= 1'b0;
                req_done      <= 1'b0;
            end else begin
                case (rd_state)
                    RD_IDLE: begin
                        if (req_valid && req_ready) begin
                            rd_kind      <= req_kind;
                            rd_addr      <= req_addr;
                            rd_beats     <= {1'b0, req_bytes[15:4]};
                            rd_beat      <= 16'd0;
                            rd_row_y     <= req_row_y;
                            rd_row_slot  <= req_row_slot;
                            rd_aborting  <= 1'b0;
                            rd_cmd_valid <= 1'b1;
                            rd_cmd_addr_byte <= req_addr;
                            rd_cmd_bytes <= req_bytes;
                            rd_cmd_tag   <= (req_kind == K_W) ? TAG_WEIGHT :
                                            (req_kind == K_B) ? TAG_BIAS :
                                            (req_kind == K_Q) ? TAG_QUANT : TAG_INPUT;
                            if (req_kind == K_ROW) begin
                                cache_valid[req_row_slot] <= 1'b0;
                            end
                            rd_state <= RD_CMD;
                        end
                    end
                    RD_CMD: begin
                        rd_cmd_valid <= 1'b1;
                        if (rd_cmd_valid && rd_cmd_ready) begin
                            rd_cmd_valid <= 1'b0;
                            rd_beat      <= 16'd0;
                            rd_state     <= RD_DATA;
                        end else if (abort_req) begin
                            rd_cmd_valid <= 1'b0;
                            rd_state     <= RD_IDLE;
                            req_done     <= 1'b1;
                        end
                    end
                    RD_DATA: begin
                        rd_data_ready <= 1'b1;
                        if (rd_data_valid && rd_data_ready) begin
                            if (rd_aborting) begin
                                if (rd_data_last || rd_data_error) begin
                                    rd_data_ready<= 1'b0;
                                    rd_aborting  <= 1'b0;
                                    rd_state     <= RD_IDLE;
                                    req_done     <= 1'b1;
                                end
                            end else if (abort_req) begin
                                rd_aborting <= 1'b1;
                                if (rd_data_last || rd_data_error) begin
                                    rd_data_ready<= 1'b0;
                                    rd_aborting  <= 1'b0;
                                    rd_state     <= RD_IDLE;
                                    req_done     <= 1'b1;
                                end
                            end else if (rd_data_error) begin
                                rd_data_ready <= 1'b0;
                                rd_state      <= RD_IDLE;
                                rd_err_code   <= (rd_data_error_code == 16'h0) ?
                                                 OP_E_AXI_RESP : rd_data_error_code;
                                rd_err_pulse  <= 1'b1;
                            end else if (rd_data_tag != ((rd_kind == K_W) ? TAG_WEIGHT :
                                                          (rd_kind == K_B) ? TAG_BIAS :
                                                          (rd_kind == K_Q) ? TAG_QUANT : TAG_INPUT)) begin
                                rd_data_ready <= 1'b0;
                                rd_state      <= RD_IDLE;
                                rd_err_code   <= OP_E_AXI_PROTOCOL;
                                rd_err_pulse  <= 1'b1;
                            end else if (rd_beat >= rd_beats) begin
                                rd_data_ready <= 1'b0;
                                rd_state      <= RD_IDLE;
                                rd_err_code   <= OP_E_AXI_PROTOCOL;
                                rd_err_pulse  <= 1'b1;
                            end else if (rd_data_last != (rd_beat == (rd_beats - 16'd1))) begin
                                rd_data_ready <= 1'b0;
                                rd_state      <= RD_IDLE;
                                rd_err_code   <= OP_E_AXI_PROTOCOL;
                                rd_err_pulse  <= 1'b1;
                            end else begin
                                case (rd_kind)
                                    K_W: begin
                                        if (rd_data[127:72] != 56'b0) begin
                                            rd_data_ready <= 1'b0;
                                            rd_state      <= RD_IDLE;
                                            rd_err_code   <= OP_E_CFG;
                                            rd_err_pulse  <= 1'b1;
                                        end else begin
                                            wt_t[0][rd_beat[3:0]*8 +: 8] <= rd_data[7:0];
                                            wt_t[1][rd_beat[3:0]*8 +: 8] <= rd_data[15:8];
                                            wt_t[2][rd_beat[3:0]*8 +: 8] <= rd_data[23:16];
                                            wt_t[3][rd_beat[3:0]*8 +: 8] <= rd_data[31:24];
                                            wt_t[4][rd_beat[3:0]*8 +: 8] <= rd_data[39:32];
                                            wt_t[5][rd_beat[3:0]*8 +: 8] <= rd_data[47:40];
                                            wt_t[6][rd_beat[3:0]*8 +: 8] <= rd_data[55:48];
                                            wt_t[7][rd_beat[3:0]*8 +: 8] <= rd_data[63:56];
                                            wt_t[8][rd_beat[3:0]*8 +: 8] <= rd_data[71:64];
                                        end
                                    end
                                    K_B: begin
                                        if ((rd_ch_base + 16'd0 >= cf_cin) && (rd_data[31:0]  != 32'b0)) begin
                                            rd_data_ready <= 1'b0; rd_state <= RD_IDLE;
                                            rd_err_code <= OP_E_CFG; rd_err_pulse <= 1'b1;
                                        end else if ((rd_ch_base + 16'd1 >= cf_cin) && (rd_data[63:32] != 32'b0)) begin
                                            rd_data_ready <= 1'b0; rd_state <= RD_IDLE;
                                            rd_err_code <= OP_E_CFG; rd_err_pulse <= 1'b1;
                                        end else if ((rd_ch_base + 16'd2 >= cf_cin) && (rd_data[95:64] != 32'b0)) begin
                                            rd_data_ready <= 1'b0; rd_state <= RD_IDLE;
                                            rd_err_code <= OP_E_CFG; rd_err_pulse <= 1'b1;
                                        end else if ((rd_ch_base + 16'd3 >= cf_cin) && (rd_data[127:96] != 32'b0)) begin
                                            rd_data_ready <= 1'b0; rd_state <= RD_IDLE;
                                            rd_err_code <= OP_E_CFG; rd_err_pulse <= 1'b1;
                                        end else begin
                                            if ((rd_ch_base + 16'd0) < cf_cin) bias_flat[rd_ch_base[3:0] + 4'd0] <= rd_data[31:0];
                                            if ((rd_ch_base + 16'd1) < cf_cin) bias_flat[rd_ch_base[3:0] + 4'd1] <= rd_data[63:32];
                                            if ((rd_ch_base + 16'd2) < cf_cin) bias_flat[rd_ch_base[3:0] + 4'd2] <= rd_data[95:64];
                                            if ((rd_ch_base + 16'd3) < cf_cin) bias_flat[rd_ch_base[3:0] + 4'd3] <= rd_data[127:96];
                                        end
                                    end
                                    K_Q: begin
                                        if (((rd_q_base + 16'd0) >= cf_cin) && (rd_data[63:0] != 64'b0)) begin
                                            rd_data_ready <= 1'b0; rd_state <= RD_IDLE;
                                            rd_err_code <= OP_E_QUANT; rd_err_pulse <= 1'b1;
                                        end else if (rd_data[47:40] != 8'b0) begin
                                            rd_data_ready <= 1'b0; rd_state <= RD_IDLE;
                                            rd_err_code <= OP_E_QUANT; rd_err_pulse <= 1'b1;
                                        end else if (((rd_q_base + 16'd0) < cf_cin) && (rd_data[63:48] != 16'b0)) begin
                                            rd_data_ready <= 1'b0; rd_state <= RD_IDLE;
                                            rd_err_code <= OP_E_QUANT; rd_err_pulse <= 1'b1;
                                        end else if (((rd_q_base + 16'd1) >= cf_cin) && (rd_data[127:64] != 64'b0)) begin
                                            rd_data_ready <= 1'b0; rd_state <= RD_IDLE;
                                            rd_err_code <= OP_E_QUANT; rd_err_pulse <= 1'b1;
                                        end else if (((rd_q_base + 16'd1) < cf_cin) && (rd_data[111:104] != 8'b0)) begin
                                            rd_data_ready <= 1'b0; rd_state <= RD_IDLE;
                                            rd_err_code <= OP_E_QUANT; rd_err_pulse <= 1'b1;
                                        end else if (((rd_q_base + 16'd1) < cf_cin) && (rd_data[127:112] != 16'b0)) begin
                                            rd_data_ready <= 1'b0; rd_state <= RD_IDLE;
                                            rd_err_code <= OP_E_QUANT; rd_err_pulse <= 1'b1;
                                        end else begin
                                            if ((rd_q_base + 16'd0) < cf_cin) begin
                                                m0_flat[rd_q_base[3:0] + 4'd0]    <= rd_data[31:0];
                                                shift_flat[rd_q_base[3:0] + 4'd0] <= rd_data[39:32];
                                            end
                                            if ((rd_q_base + 16'd1) < cf_cin) begin
                                                m0_flat[rd_q_base[3:0] + 4'd1]    <= rd_data[95:64];
                                                shift_flat[rd_q_base[3:0] + 4'd1] <= rd_data[103:96];
                                            end
                                        end
                                    end
                                    default: begin // K_ROW
                                        cache_wen   <= 1'b1;
                                        cache_waddr <= (rd_row_slot * MAX_ROW_WORDS) +
                                                       rd_beat[CACHE_AW-1:0];
                                        cache_wdata <= rd_data;
                                        if (rd_data_last) begin
                                            cache_valid[rd_row_slot] <= 1'b1;
                                            cache_tag[rd_row_slot]   <= rd_row_y;
                                        end
                                    end
                                endcase
                                if (rd_err_pulse) begin
                                    // error path already stopped the engine
                                end else if (rd_data_last) begin
                                    rd_data_ready <= 1'b0;
                                    rd_state      <= RD_IDLE;
                                    req_done      <= 1'b1;
                                end else begin
                                    rd_beat <= rd_beat + 16'd1;
                                end
                            end
                        end
                    end
                    default: rd_state <= RD_IDLE;
                endcase
            end
        end
    end

    // ------------------------------------------------------------------
    // ROW SCHEDULER + PREFETCH
    // ------------------------------------------------------------------
    reg [1:0]  rs_phase;
    reg [15:0] rs_oy;
    reg signed [17:0] rs_yb;
    reg [15:0] rs_rows_computed;
    reg        row_begin_r;
    reg        fetch_busy;

    assign rows_computed_w = rs_rows_computed;
    assign row_begin       = row_begin_r;
    assign row_begin_oy    = rs_oy;
    wire signed [17:0] rs_yb_next = (rs_oy * {16'b0, cf_stride}) - $signed({16'b0, cf_pad});
    assign row_begin_yb    = rs_yb_next;

    wire signed [17:0] cs0 = rs_yb;
    wire signed [17:0] cs1 = rs_yb + 18'sd1;
    wire signed [17:0] cs2 = rs_yb + 18'sd2;
    wire signed [17:0] ns0 = rs_yb + $signed({16'b0, cf_stride});
    wire signed [17:0] ns1 = ns0 + 18'sd1;
    wire signed [17:0] ns2 = ns0 + 18'sd2;

    function automatic integer in_row_range(input signed [17:0] y);
        in_row_range = (y >= 18'sd0) && (y < $signed({2'b0, cf_h}));
    endfunction

    function automatic integer row_present(input signed [17:0] y);
        reg [1:0] sl;
        begin
            sl = y[1:0];
            row_present = in_row_range(y) && cache_valid[sl] && (cache_tag[sl] == y[15:0]);
        end
    endfunction

    function automatic integer slot_replaceable(input signed [17:0] y);
        reg [1:0] sl;
        reg busy;
        begin
            sl   = y[1:0];
            busy = 1'b0;
            if (cache_valid[sl]) begin
                if (in_row_range(cs0) && (cache_tag[sl] == cs0[15:0])) busy = 1'b1;
                if (in_row_range(cs1) && (cache_tag[sl] == cs1[15:0])) busy = 1'b1;
                if (in_row_range(cs2) && (cache_tag[sl] == cs2[15:0])) busy = 1'b1;
                if (in_row_range(ns0) && (ns0 != y) && (cache_tag[sl] == ns0[15:0])) busy = 1'b1;
                if (in_row_range(ns1) && (ns1 != y) && (cache_tag[sl] == ns1[15:0])) busy = 1'b1;
                if (in_row_range(ns2) && (ns2 != y) && (cache_tag[sl] == ns2[15:0])) busy = 1'b1;
            end
            slot_replaceable = !busy;
        end
    endfunction

    reg        cand_valid;
    reg [15:0] cand_y;
    integer    ci;
    reg signed [17:0] cand_s;
    reg        cand_found;

    always @(rs_yb or cf_stride or cf_h or cf_in_addr or g_in_stride or run_enable or
             fetch_busy or cache_valid[0] or cache_valid[1] or cache_valid[2] or cache_valid[3] or
             cache_tag[0] or cache_tag[1] or cache_tag[2] or cache_tag[3]) begin
        cand_valid = 1'b0;
        cand_y     = 16'd0;
        cand_found = 1'b0;
        if (run_enable && !fetch_busy) begin
            for (ci = 0; ci < 3; ci = ci + 1) begin
                cand_s = rs_yb + ci;
                if (!cand_found && in_row_range(cand_s) &&
                    !row_present(cand_s) && slot_replaceable(cand_s)) begin
                    cand_valid = 1'b1;
                    cand_y     = cand_s[15:0];
                    cand_found = 1'b1;
                end
            end
            for (ci = 0; ci < 3; ci = ci + 1) begin
                cand_s = rs_yb + $signed({16'b0, cf_stride}) + ci;
                if (!cand_found && in_row_range(cand_s) &&
                    !row_present(cand_s) && slot_replaceable(cand_s)) begin
                    cand_valid = 1'b1;
                    cand_y     = cand_s[15:0];
                    cand_found = 1'b1;
                end
            end
        end
    end

    assign fetch_req_valid = cand_valid && !fetch_busy;
    assign fetch_req_addr  = cf_in_addr + (cand_y[4:0] * g_in_stride[9:0]);
    assign fetch_req_bytes = g_in_stride[15:0];
    assign fetch_req_y     = cand_y;
    assign fetch_req_slot  = cand_y[1:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rs_phase         <= 2'd0;
            rs_oy            <= 16'd0;
            rs_yb            <= 18'sd0;
            rs_rows_computed <= 16'd0;
            row_begin_r      <= 1'b0;
            fetch_busy       <= 1'b0;
        end else if (core_soft_reset) begin
            rs_phase         <= 2'd0;
            rs_rows_computed <= 16'd0;
            row_begin_r      <= 1'b0;
            fetch_busy       <= 1'b0;
        end else if (launch_tick) begin
            rs_phase         <= 2'd0;
            rs_oy            <= 16'd0;
            rs_yb            <= 18'sd0;
            rs_rows_computed <= 16'd0;
            row_begin_r      <= 1'b0;
            fetch_busy       <= 1'b0;
        end else begin
            row_begin_r <= 1'b0;
            if (stop_new) begin
                fetch_busy  <= 1'b0;
                row_begin_r <= 1'b0;
            end else if (!run_enable) begin
                fetch_busy <= 1'b0;
            end else begin
                if (fetch_req_valid && req_ready) begin
                    fetch_busy <= 1'b1;
                end
                if (fetch_busy && req_done) begin
                    fetch_busy <= 1'b0;
                end
                if (row_last_word) begin
                    rs_rows_computed <= rs_rows_computed + 16'd1;
                end
                case (rs_phase)
                    2'd0: begin
                        if ((rs_oy < g_hout) &&
                            ((rs_oy <= 16'd1) ||
                             (rows_drained_w >= (rs_oy - 16'd1)))) begin
                            rs_yb       <= rs_yb_next;
                            row_begin_r <= 1'b1;
                            rs_phase    <= 2'd1;
                        end
                    end
                    2'd1: begin
                        if (row_last_word) begin
                            rs_phase <= 2'd0;
                            rs_oy    <= rs_oy + 16'd1;
                        end
                    end
                    default: rs_phase <= 2'd0;
                endcase
            end
        end
    end

    // ------------------------------------------------------------------
    // MAC ENGINE: 9 taps per group, 1 tap/cycle issue, 2-deep read pipeline
    //
    // The row cache is a synchronous RAM (read data one cycle after the
    // registered read enable), so a tap issued in cycle c is accumulated in
    // cycle c+2 through a two-entry pending pipeline (p1 -> p2).  Taps are
    // still issued every cycle, which keeps the steady group issue interval
    // at 9 cycles while MAC and RQ overlap through the two acc banks.
    // ------------------------------------------------------------------
    reg        mac_active;
    reg        mac_issue_active;
    reg        mac_ng;
    reg [3:0]  mac_tap;
    reg [15:0] mac_ox;
    reg [2:0]  mac_g;
    reg signed [17:0] mac_yb;
    reg [15:0] mac_oy;
    reg        mac_last_in_row;

    // currently-issuing group context
    reg [15:0] cur_ox_q;
    reg [3:0]  cur_gbase_q;
    reg [15:0] cur_lane_count_q;
    reg [15:0] cur_oy_q;
    reg [1:0]  mac_cur_bank;

    // group metadata per acc bank (written at group start, read by RQ)
    reg [15:0] bank_ox         [0:1];
    reg [15:0] bank_oy         [0:1];
    reg [3:0]  bank_gbase      [0:1];
    reg        bank_word_end   [0:1];

    reg signed [31:0] acc_flat [0:2*PB-1];
    wire signed [15:0] lane_prod [0:PB-1];

    // pending read pipeline: p1 = issued, p2 = data valid this cycle
    reg        p1_valid; reg [3:0] p1_tap; reg [1:0] p1_bank; reg [15:0] p1_lc;
    reg [127:0] p1_wt;  reg [1:0] p1_base; reg p1_zero;
    reg        p2_valid; reg [3:0] p2_tap; reg [1:0] p2_bank; reg [15:0] p2_lc;
    reg [127:0] p2_wt;  reg [1:0] p2_base; reg p2_zero;

    // issued geometry
    wire [15:0] mac_gbase_w  = {12'b0, mac_g} * PB;
    wire [15:0] mac_rem      = cf_cin - mac_gbase_w;
    wire [15:0] mac_lane_cnt = (mac_rem > PB) ? PB : mac_rem;
    wire [3:0]  iss_gbase    = mac_ng ? mac_gbase_w[3:0] : cur_gbase_q;
    wire [15:0] iss_ox       = mac_ng ? mac_ox : cur_ox_q;
    wire [3:0]  iss_tap      = mac_ng ? 4'd0 : mac_tap;
    wire [2:0]  iss_ky       = (iss_tap >= 4'd6) ? 3'd2 : (iss_tap >= 4'd3) ? 3'd1 : 3'd0;
    wire [3:0]  iss_kx       = iss_tap - ((iss_tap >= 4'd6) ? 4'd6 : (iss_tap >= 4'd3) ? 4'd3 : 4'd0);
    wire signed [31:0] iss_iy = $signed({{14{mac_yb[17]}}, mac_yb}) + $signed({29'b0, iss_ky});
    wire signed [31:0] iss_ix = ($signed({16'b0, iss_ox}) * $signed({30'b0, cf_stride}))
                                - $signed({30'b0, cf_pad}) + $signed({28'b0, iss_kx});
    wire iss_row_ok = (iss_iy >= 32'sd0) && (iss_iy < $signed({16'b0, cf_h}));
    wire iss_col_ok = (iss_ix >= 32'sd0) && (iss_ix < $signed({16'b0, cf_w}));
    wire iss_zero   = !iss_row_ok || !iss_col_ok;
    wire [1:0] iss_slot = iss_iy[1:0];
    wire [4:0] iss_word = (cf_calloc == 16'd16) ? iss_ix[4:0] : iss_ix[4:1];
    wire iss_row_ready = iss_row_ok ? (cache_valid[iss_slot] &&
                                       (cache_tag[iss_slot] == iss_iy[15:0])) : 1'b1;

    wire [3:0]  issue_data_base = ((cf_calloc == 16'd16) ? 4'd0 :
                                   (iss_ox[0] ? 4'd8 : 4'd0)) + iss_gbase;
    wire [1:0]  issue_data_sel  = issue_data_base[3:2];
    wire [1:0]  wt_sel          = iss_gbase[3:2];

    wire [127:0] wt_tap_word = wt_t[iss_tap];
    wire [127:0] wt_shifted =
        (wt_sel == 2'd0) ? wt_tap_word :
        (wt_sel == 2'd1) ? {32'b0, wt_tap_word[127:32]} :
        (wt_sel == 2'd2) ? {64'b0, wt_tap_word[127:64]} :
                           {96'b0, wt_tap_word[127:96]};

    // p2 datapath (registered metadata selects the same lane bytes)
    wire [127:0] mac_data_raw = p2_zero ? 128'b0 : cache_rdata;
    wire [127:0] mac_data_shifted =
        (p2_base == 2'd0) ? mac_data_raw :
        (p2_base == 2'd1) ? {32'b0, mac_data_raw[127:32]} :
        (p2_base == 2'd2) ? {64'b0, mac_data_raw[127:64]} :
                            {96'b0, mac_data_raw[127:96]};

    genvar gi;
    generate
        for (gi = 0; gi < PB; gi = gi + 1) begin : G_MAC_LANE
            wire signed [7:0] lane_d = $signed(mac_data_shifted[gi*8 +: 8]);
            wire signed [7:0] lane_w = $signed(p2_wt[gi*8 +: 8]);
            assign lane_prod[gi] = lane_d * lane_w;
        end
    endgenerate

    wire        last_group_w   = (mac_g == (g_ngroups - 3'd1));
    wire        last_pixel_w   = (mac_ox == (g_wout - 16'd1));
    wire        last_row_grp_w = last_group_w && last_pixel_w;
    wire        word_end_w     = last_group_w &&
                                 ((cf_calloc == 16'd16) || mac_ox[0] || last_pixel_w);
    wire [31:0] grp_inflight   = grp_issued - grp_consumed;
    // RQ reads its last lane of the previous same-bank group at this edge, so
    // the MAC may safely start the next group into that bank in the same cycle
    // (the RQ stage-1 register samples the pre-edge acc value).
    wire        bank_release_now = (rq_phase == RQ_ACT) &&
                                   ((rq_lane_idx + PR) >= PB) &&
                                   (rq_bank == grp_issued[0]);
    wire        mac_can_start  = mac_active && ((grp_inflight < 32'd2) || bank_release_now);
    wire        issue_permitted= mac_issue_active && !stop_new &&
                                 (mac_ng ? mac_can_start : 1'b1);
    wire        issue_fire     = issue_permitted && iss_row_ready;
    wire        mac_issue_stall= mac_issue_active && !stop_new && !issue_fire;
    wire        mac_busy_w     = mac_issue_active || p1_valid || p2_valid;
    wire        grp_launch_pulse = mac_ng && mac_issue_active && mac_can_start && iss_row_ready;

    integer ai;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mac_active       <= 1'b0;
            mac_issue_active <= 1'b0;
            mac_ng           <= 1'b1;
            mac_tap          <= 4'd0;
            mac_ox           <= 16'd0;
            mac_g            <= 3'd0;
            mac_yb           <= 18'sd0;
            mac_oy           <= 16'd0;
            mac_last_in_row  <= 1'b0;
            cur_ox_q         <= 16'd0;
            cur_gbase_q      <= 4'd0;
            cur_lane_count_q <= 16'd0;
            cur_oy_q         <= 16'd0;
            mac_cur_bank     <= 2'd0;
            p1_valid <= 1'b0; p1_tap <= 4'd0; p1_bank <= 2'd0; p1_lc <= 16'd0;
            p1_wt <= 128'b0; p1_base <= 2'd0; p1_zero <= 1'b0;
            p2_valid <= 1'b0; p2_tap <= 4'd0; p2_bank <= 2'd0; p2_lc <= 16'd0;
            p2_wt <= 128'b0; p2_base <= 2'd0; p2_zero <= 1'b0;
            grp_issued       <= 32'd0;
            grp_done         <= 32'd0;
            mac_cache_ren    <= 1'b0;
            mac_cache_raddr  <= {CACHE_AW{1'b0}};
            bank_ox[0]       <= 16'd0;  bank_ox[1]       <= 16'd0;
            bank_oy[0]       <= 16'd0;  bank_oy[1]       <= 16'd0;
            bank_gbase[0]    <= 4'd0;   bank_gbase[1]    <= 4'd0;
            bank_word_end[0] <= 1'b0;   bank_word_end[1] <= 1'b0;
            for (ai = 0; ai < 2*PB; ai = ai + 1)
                acc_flat[ai] <= 32'sd0;
        end else if (core_soft_reset || launch_tick) begin
            mac_active       <= 1'b0;
            mac_issue_active <= 1'b0;
            mac_ng           <= 1'b1;
            mac_tap          <= 4'd0;
            mac_last_in_row  <= 1'b0;
            p1_valid         <= 1'b0;
            p2_valid         <= 1'b0;
            grp_issued       <= 32'd0;
            grp_done         <= 32'd0;
            mac_cache_ren    <= 1'b0;
            for (ai = 0; ai < 2*PB; ai = ai + 1)
                acc_flat[ai] <= 32'sd0;
        end else if (stop_new) begin
            mac_active       <= 1'b0;
            mac_issue_active <= 1'b0;
            p1_valid         <= 1'b0;
            p2_valid         <= 1'b0;
            mac_cache_ren    <= 1'b0;
        end else begin
            mac_cache_ren <= 1'b0;

            if (row_begin) begin
                mac_active       <= 1'b1;
                mac_issue_active <= 1'b1;
                mac_ng           <= 1'b1;
                mac_tap          <= 4'd0;
                mac_ox           <= 16'd0;
                mac_g            <= 3'd0;
                mac_yb           <= row_begin_yb;
                mac_oy           <= row_begin_oy;
                p1_valid         <= 1'b0;
                p2_valid         <= 1'b0;
            end

            // 1) accumulate the p2 tap (its read data is on cache_rdata now)
            if (p2_valid) begin
                for (ai = 0; ai < PB; ai = ai + 1)
                    acc_flat[p2_bank*PB + ai] <=
                        acc_flat[p2_bank*PB + ai] +
                        {{16{lane_prod[ai][15]}}, lane_prod[ai]};
                if (p2_tap == 4'd8)
                    grp_done <= grp_done + 32'd1;
            end

            // 2) shift p1 -> p2
            p2_valid <= p1_valid;
            p2_tap   <= p1_tap;
            p2_bank  <= p1_bank;
            p2_lc    <= p1_lc;
            p2_wt    <= p1_wt;
            p2_base  <= p1_base;
            p2_zero  <= p1_zero;

            // 3) issue the next tap into p1
            p1_valid <= 1'b0;
            if (mac_issue_active) begin
                if (mac_ng) begin
                    if (mac_can_start && iss_row_ready) begin
                        mac_cur_bank <= grp_issued[0];
                        for (ai = 0; ai < PB; ai = ai + 1)
                            acc_flat[grp_issued[0]*PB + ai] <= 32'sd0;
                        cur_ox_q         <= mac_ox;
                        cur_gbase_q      <= mac_gbase_w[3:0];
                        cur_lane_count_q <= mac_lane_cnt;
                        cur_oy_q         <= mac_oy;
                        bank_ox[grp_issued[0]]       <= mac_ox;
                        bank_oy[grp_issued[0]]       <= mac_oy;
                        bank_gbase[grp_issued[0]]    <= mac_gbase_w[3:0];
                        bank_word_end[grp_issued[0]] <= word_end_w;
                        grp_issued       <= grp_issued + 32'd1;
                        mac_ng           <= 1'b0;
                        mac_tap          <= 4'd1;
                        mac_last_in_row  <= last_row_grp_w;
                        p1_valid         <= 1'b1;
                        p1_tap           <= 4'd0;
                        p1_bank          <= grp_issued[0];
                        p1_lc            <= mac_lane_cnt;
                        p1_wt            <= wt_shifted;
                        p1_base          <= issue_data_sel;
                        p1_zero          <= iss_zero;
                        mac_cache_ren    <= !iss_zero;
                        mac_cache_raddr  <= (iss_slot * MAX_ROW_WORDS) + iss_word;
                        if (last_group_w) begin
                            mac_g  <= 3'd0;
                            mac_ox <= mac_ox + 16'd1;
                        end else begin
                            mac_g  <= mac_g + 3'd1;
                        end
                    end
                end else if (iss_row_ready) begin
                    p1_valid        <= 1'b1;
                    p1_tap          <= iss_tap;
                    p1_bank         <= mac_cur_bank;
                    p1_lc           <= cur_lane_count_q;
                    p1_wt           <= wt_shifted;
                    p1_base         <= issue_data_sel;
                    p1_zero         <= iss_zero;
                    mac_cache_ren   <= !iss_zero;
                    mac_cache_raddr <= (iss_slot * MAX_ROW_WORDS) + iss_word;
                    if (iss_tap == 4'd8) begin
                        mac_ng <= 1'b1;
                        if (mac_last_in_row)
                            mac_issue_active <= 1'b0;
                    end else begin
                        mac_tap <= iss_tap + 4'd1;
                    end
                end
            end
        end
    end

    // ------------------------------------------------------------------
    // RQ ENGINE + output word assembler
    //
    // Requant bytes leave the pipeline in ascending channel order and are
    // shifted into a 128-bit word register PR bytes per cycle (pure wiring).
    // At the word-end lane the completed word is written to the output row
    // bank with its byte order reversed (also pure wiring).  No dynamic
    // byte-index mux and no combinational block over the pipeline arrays.
    // flags layout: [8]=word_end [7]=lane_last [6]=channel_valid [5:0]=ob_addr
    // ------------------------------------------------------------------
    reg [15:0] rq_ox;
    reg [15:0] rq_oy;
    reg [3:0]  rq_gbase;
    reg        rq_word_end;
    reg [15:0] rq_word_cnt;
    reg [127:0] pix_shift;
    reg        row_last_word_r;

    assign row_last_word = row_last_word_r;

    reg        s1v             [0:PR-1];
    reg signed [31:0] s1_sum    [0:PR-1];
    reg        [31:0] s1_m0     [0:PR-1];
    reg        [7:0]  s1_sh     [0:PR-1];
    reg        [8:0]  s1_flags  [0:PR-1];
    reg        s2v             [0:PR-1];
    reg        [63:0] s2_prod   [0:PR-1];
    reg               s2_neg    [0:PR-1];
    reg        [7:0]  s2_sh     [0:PR-1];
    reg        [8:0]  s2_flags  [0:PR-1];
    reg        s3v             [0:PR-1];
    reg        [63:0] s3_abs    [0:PR-1];
    reg               s3_neg    [0:PR-1];
    reg        [7:0]  s3_sh     [0:PR-1];
    reg        [8:0]  s3_flags  [0:PR-1];
    reg        s4v             [0:PR-1];
    reg        [63:0] s4_mag    [0:PR-1];
    reg               s4_neg    [0:PR-1];
    reg        [8:0]  s4_flags  [0:PR-1];
    reg        s5v             [0:PR-1];
    reg        [7:0]  s5_byte   [0:PR-1];
    reg        [8:0]  s5_flags  [0:PR-1];

    wire [4:0] lane_ch_idx = rq_gbase + rq_lane_idx[3:0];
    wire [5:0] lane_ob_addr = (rq_oy[0] ? MAX_ROW_WORDS : 6'd0) +
                              ((cf_calloc == 16'd16) ? rq_ox[4:0] : rq_ox[4:1]);

    // The group completing this cycle is already stable in its acc bank; allowing
    // RQ to start on it removes the completion->start bubble and keeps the
    // steady group issue interval at 9 cycles.
    wire rq_ready_now = (grp_consumed < grp_done) ||
                        (p2_valid && (p2_tap == 4'd8));
    wire rq_start = (rq_phase == RQ_IDLE) && rq_ready_now &&
                    run_enable && !stop_new;
    wire [1:0] rq_bank_w = grp_consumed[0];

    function automatic [127:0] dw_shift_in(input [127:0] cur, input [PR*8-1:0] bytes);
        begin
            dw_shift_in = {cur[127-PR*8:0], bytes};
        end
    endfunction

    function automatic [127:0] dw_reverse_bytes(input [127:0] w);
        integer k;
        begin
            for (k = 0; k < 16; k = k + 1)
                dw_reverse_bytes[k*8 +: 8] = w[(15-k)*8 +: 8];
        end
    endfunction

    integer rk;
    integer rk2;
    reg [PR*8-1:0] rq_cur_bytes;
    reg [127:0]    rq_pix_next;
    reg [127:0]    rq_word_out;
    reg            rq_wr_event;
    reg [5:0]      rq_ev_addr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rq_phase        <= RQ_IDLE;
            rq_bank         <= 2'd0;
            rq_ox           <= 16'd0;
            rq_oy           <= 16'd0;
            rq_gbase        <= 4'd0;
            rq_word_end     <= 1'b0;
            rq_lane_idx     <= 16'd0;
            grp_consumed    <= 32'd0;
            rq_word_cnt     <= 16'd0;
            pix_shift       <= 128'b0;
            row_last_word_r <= 1'b0;
            ob_wen          <= 1'b0;
            ob_waddr        <= {OB_AW{1'b0}};
            ob_wdata        <= {DATA_WIDTH{1'b0}};
            for (rk = 0; rk < PR; rk = rk + 1) begin
                s1v[rk] <= 1'b0; s2v[rk] <= 1'b0; s3v[rk] <= 1'b0;
                s4v[rk] <= 1'b0; s5v[rk] <= 1'b0;
                s1_flags[rk] <= 9'd0; s2_flags[rk] <= 9'd0;
                s3_flags[rk] <= 9'd0; s4_flags[rk] <= 9'd0;
                s5_flags[rk] <= 9'd0;
            end
        end else if (core_soft_reset) begin
            rq_phase        <= RQ_IDLE;
            ob_wen          <= 1'b0;
            pix_shift       <= 128'b0;
            row_last_word_r <= 1'b0;
            for (rk = 0; rk < PR; rk = rk + 1) begin
                s1v[rk] <= 1'b0; s2v[rk] <= 1'b0; s3v[rk] <= 1'b0;
                s4v[rk] <= 1'b0; s5v[rk] <= 1'b0;
            end
        end else if (launch_tick) begin
            rq_phase        <= RQ_IDLE;
            rq_lane_idx     <= 16'd0;
            grp_consumed    <= 32'd0;
            rq_word_cnt     <= 16'd0;
            pix_shift       <= 128'b0;
            row_last_word_r <= 1'b0;
            ob_wen          <= 1'b0;
            for (rk = 0; rk < PR; rk = rk + 1) begin
                s1v[rk] <= 1'b0; s2v[rk] <= 1'b0; s3v[rk] <= 1'b0;
                s4v[rk] <= 1'b0; s5v[rk] <= 1'b0;
            end
        end else begin
            ob_wen          <= 1'b0;
            row_last_word_r <= 1'b0;

            if (row_begin) begin
                rq_word_cnt <= 16'd0;
                pix_shift   <= 128'b0;
            end

            if (stop_new) begin
                rq_phase <= RQ_IDLE;
                for (rk = 0; rk < PR; rk = rk + 1) begin
                    s1v[rk] <= 1'b0; s2v[rk] <= 1'b0; s3v[rk] <= 1'b0;
                    s4v[rk] <= 1'b0; s5v[rk] <= 1'b0;
                end
            end else begin
                // stage 5 <- stage 4 (sat8)
                for (rk = 0; rk < PR; rk = rk + 1) begin
                    s5v[rk]     <= s4v[rk];
                    s5_flags[rk]<= s4_flags[rk];
                    if (|s4_mag[rk][63:8]) begin
                        s5_byte[rk] <= s4_neg[rk] ? 8'h80 : 8'h7f;
                    end else if (s4_neg[rk]) begin
                        s5_byte[rk] <= (s4_mag[rk][7:0] > 8'd128) ? 8'h80 :
                                       (~s4_mag[rk][7:0] + 8'd1);
                    end else begin
                        s5_byte[rk] <= (s4_mag[rk][7:0] > 8'd127) ? 8'h7f :
                                       s4_mag[rk][7:0];
                    end
                end
                // stage 4 <- stage 3 (round + shift)
                for (rk = 0; rk < PR; rk = rk + 1) begin
                    s4v[rk]     <= s3v[rk];
                    s4_neg[rk]  <= s3_neg[rk];
                    s4_flags[rk]<= s3_flags[rk];
                    s4_mag[rk]  <= (s3_abs[rk] +
                                    ((s3_sh[rk] == 8'd0) ? 64'd0 :
                                     (64'd1 << (s3_sh[rk] - 8'd1)))) >> s3_sh[rk];
                end
                // stage 3 <- stage 2 (abs)
                for (rk = 0; rk < PR; rk = rk + 1) begin
                    s3v[rk]     <= s2v[rk];
                    s3_neg[rk]  <= s2_neg[rk];
                    s3_sh[rk]   <= s2_sh[rk];
                    s3_flags[rk]<= s2_flags[rk];
                    s3_abs[rk]  <= s2_neg[rk] ? (~s2_prod[rk] + 64'd1) : s2_prod[rk];
                end
                // stage 2 <- stage 1 (64-bit product, unsigned M0)
                for (rk = 0; rk < PR; rk = rk + 1) begin
                    s2v[rk]     <= s1v[rk];
                    s2_prod[rk] <= $signed(s1_sum[rk]) * $signed({1'b0, s1_m0[rk]});
                    s2_neg[rk]  <= s1_sum[rk][31];
                    s2_sh[rk]   <= s1_sh[rk];
                    s2_flags[rk]<= s1_flags[rk];
                end
                // stage 1 <- issue
                for (rk = 0; rk < PR; rk = rk + 1) begin
                    if (rq_phase == RQ_ACT) begin
                        s1v[rk]     <= 1'b1;
                        s1_sum[rk]  <= acc_flat[rq_bank*PB + (rq_lane_idx + rk)] +
                                       bias_flat[lane_ch_idx + rk];
                        s1_m0[rk]   <= m0_flat[lane_ch_idx + rk];
                        s1_sh[rk]   <= shift_flat[lane_ch_idx + rk];
                        s1_flags[rk]<= {rq_word_end,
                                        ((rq_lane_idx + rk) == (PB-1)),
                                        ((lane_ch_idx + rk) < cf_cin),
                                        lane_ob_addr};
                    end else begin
                        s1v[rk] <= 1'b0;
                    end
                end

                // shift-register word assembly + row-bank word write
                rq_cur_bytes = {PR*8{1'b0}};
                rq_wr_event  = 1'b0;
                rq_ev_addr   = 6'd0;
                for (rk2 = 0; rk2 < PR; rk2 = rk2 + 1) begin
                    if (s5v[rk2] && s5_flags[rk2][6])
                        rq_cur_bytes[rk2*8 +: 8] = s5_byte[rk2];
                    if (s5v[rk2] && s5_flags[rk2][8] && s5_flags[rk2][7]) begin
                        rq_wr_event = 1'b1;
                        rq_ev_addr  = s5_flags[rk2][5:0];
                    end
                end
                rq_pix_next = dw_shift_in(pix_shift, rq_cur_bytes);
                rq_word_out = dw_reverse_bytes(rq_pix_next);

                if (rq_wr_event) begin
                    ob_wen      <= 1'b1;
                    ob_waddr    <= rq_ev_addr[OB_AW-1:0];
                    ob_wdata    <= rq_word_out;
                    pix_shift   <= 128'b0;
                    rq_word_cnt <= rq_word_cnt + 16'd1;
                    if (rq_word_cnt == (g_out_words - 16'd1))
                        row_last_word_r <= 1'b1;
                end else if (s5v[0]) begin
                    pix_shift <= rq_pix_next;
                end

                // RQ sequencer
                case (rq_phase)
                    RQ_IDLE: begin
                        if (rq_start) begin
                            rq_phase    <= RQ_ACT;
                            rq_bank     <= rq_bank_w;
                            rq_ox       <= bank_ox[rq_bank_w];
                            rq_oy       <= bank_oy[rq_bank_w];
                            rq_gbase    <= bank_gbase[rq_bank_w];
                            rq_word_end <= bank_word_end[rq_bank_w];
                            rq_lane_idx <= 16'd0;
                        end
                    end
                    RQ_ACT: begin
                        if ((rq_lane_idx + PR) >= PB) begin
                            rq_phase     <= RQ_IDLE;
                            grp_consumed <= grp_consumed + 32'd1;
                        end else begin
                            rq_lane_idx <= rq_lane_idx + PR;
                        end
                    end
                    default: rq_phase <= RQ_IDLE;
                endcase
            end
        end
    end

    // ------------------------------------------------------------------
    // DRAIN ENGINE: one output-row burst at a time, 1 accepted beat/cycle.
    // A 2-deep read FIFO absorbs the synchronous RAM latency; a separate
    // output register presents wr_data/valid/last with no bubbles.
    // ------------------------------------------------------------------
    reg [1:0]  od_state;
    reg [15:0] od_oy;
    reg [15:0] od_beats;
    reg [15:0] od_rows_drained;
    reg [15:0] od_issued;
    reg [15:0] od_arrived;
    reg [1:0]  od_rdv;
    reg [1:0]  od_fifo_cnt;
    reg [127:0] od_fifo0, od_fifo1;
    reg        od_out_valid;
    reg        od_out_last;
    reg [127:0] od_out_data;
    reg [15:0] od_out_beat;

    assign rows_drained_w = od_rows_drained;
    assign od_busy_w = (od_state != OD_IDLE);

    wire od_issue_fire = (od_state == OD_DATA) &&
                         (od_issued < od_beats) &&
                         ((od_issued - od_arrived) < 16'd2);
    wire od_out_take   = (od_state == OD_DATA) && (od_fifo_cnt != 2'd0) &&
                         (!od_out_valid || wr_data_ready);
    wire od_out_consume= (od_state == OD_DATA) && od_out_valid && wr_data_ready;

    always @* begin
        wr_data       = od_out_data;
        wr_data_valid = od_out_valid;
        wr_data_last  = od_out_last;
        wr_data_tag   = TAG_OUTPUT;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            od_state        <= OD_IDLE;
            od_oy           <= 16'd0;
            od_beats        <= 16'd0;
            od_rows_drained <= 16'd0;
            od_issued       <= 16'd0;
            od_arrived      <= 16'd0;
            od_rdv          <= 2'd0;
            od_fifo_cnt     <= 2'd0;
            od_fifo0        <= 128'b0;
            od_fifo1        <= 128'b0;
            od_out_valid    <= 1'b0;
            od_out_last     <= 1'b0;
            od_out_data     <= 128'b0;
            od_out_beat     <= 16'd0;
            wr_cmd_valid    <= 1'b0;
            wr_cmd_addr_byte<= 32'd0;
            wr_cmd_bytes    <= 16'd0;
            wr_cmd_tag      <= TAG_OUTPUT;
            wr_rsp_ready    <= 1'b0;
            ob_ren          <= 1'b0;
            ob_raddr        <= {OB_AW{1'b0}};
            od_err_pulse    <= 1'b0;
            od_err_code     <= 16'd0;
        end else if (core_soft_reset || launch_tick) begin
            od_state        <= OD_IDLE;
            od_rows_drained <= 16'd0;
            od_issued       <= 16'd0;
            od_arrived      <= 16'd0;
            od_rdv          <= 2'd0;
            od_fifo_cnt     <= 2'd0;
            od_out_valid    <= 1'b0;
            od_out_last     <= 1'b0;
            od_out_beat     <= 16'd0;
            wr_cmd_valid    <= 1'b0;
            wr_rsp_ready    <= 1'b0;
            ob_ren          <= 1'b0;
            od_err_pulse    <= 1'b0;
        end else if (fatal_flag) begin
            od_state        <= OD_IDLE;
            wr_cmd_valid    <= 1'b0;
            od_out_valid    <= 1'b0;
            wr_rsp_ready    <= 1'b0;
            ob_ren          <= 1'b0;
            od_err_pulse    <= 1'b0;
        end else begin
            ob_ren       <= 1'b0;
            od_err_pulse <= 1'b0;

            case (od_state)
                OD_IDLE: begin
                    if (run_enable && (od_rows_drained < rs_rows_computed)) begin
                        od_oy           <= od_rows_drained;
                        od_beats        <= g_out_words;
                        od_issued       <= 16'd0;
                        od_arrived      <= 16'd0;
                        od_rdv          <= 2'd0;
                        od_fifo_cnt     <= 2'd0;
                        od_out_valid    <= 1'b0;
                        od_out_last     <= 1'b0;
                        od_out_beat     <= 16'd0;
                        wr_cmd_valid    <= 1'b1;
                        wr_cmd_addr_byte<= cf_out_addr +
                                           (od_rows_drained[4:0] * g_out_stride[9:0]);
                        wr_cmd_bytes    <= g_out_stride[15:0];
                        wr_cmd_tag      <= TAG_OUTPUT;
                        od_state        <= OD_CMD;
                    end
                end
                OD_CMD: begin
                    if (wr_cmd_valid && wr_cmd_ready) begin
                        wr_cmd_valid <= 1'b0;
                        od_state     <= OD_DATA;
                    end else if (abort_req) begin
                        wr_cmd_valid <= 1'b0;
                        od_state     <= OD_IDLE;
                    end
                end
                OD_DATA: begin
                    // 1) consume the presented beat
                    if (od_out_consume) begin
                        od_out_valid <= 1'b0;
                        if (od_out_last) begin
                            od_state    <= OD_RSP;
                            od_fifo_cnt <= 2'd0;
                        end
                    end
                    // 2) read issue (registered enable -> data 2 cycles later)
                    if (od_issue_fire) begin
                        ob_ren    <= 1'b1;
                        ob_raddr  <= (od_oy[0] * MAX_ROW_WORDS) + od_issued[OB_AW-1:0];
                        od_issued <= od_issued + 16'd1;
                    end
                    od_rdv <= {od_rdv[0], od_issue_fire};
                    // 3) FIFO push/pop and output-register load (all cases explicit)
                    if (od_out_take && od_rdv[1]) begin
                        od_arrived   <= od_arrived + 16'd1;
                        od_out_valid <= 1'b1;
                        od_out_data  <= od_fifo0;
                        od_out_last  <= (od_out_beat == (od_beats - 16'd1));
                        od_out_beat  <= od_out_beat + 16'd1;
                        if (od_fifo_cnt == 2'd1) begin
                            od_fifo0    <= ob_rdata;
                            od_fifo1    <= 128'b0;
                            od_fifo_cnt <= 2'd1;
                        end else begin
                            od_fifo0    <= od_fifo1;
                            od_fifo1    <= ob_rdata;
                            od_fifo_cnt <= 2'd2;
                        end
                    end else if (od_out_take) begin
                        od_out_valid <= 1'b1;
                        od_out_data  <= od_fifo0;
                        od_out_last  <= (od_out_beat == (od_beats - 16'd1));
                        od_out_beat  <= od_out_beat + 16'd1;
                        if (od_fifo_cnt == 2'd1) begin
                            od_fifo0    <= 128'b0;
                            od_fifo1    <= 128'b0;
                            od_fifo_cnt <= 2'd0;
                        end else begin
                            od_fifo0    <= od_fifo1;
                            od_fifo1    <= 128'b0;
                            od_fifo_cnt <= 2'd1;
                        end
                    end else if (od_rdv[1]) begin
                        od_arrived <= od_arrived + 16'd1;
                        if (od_fifo_cnt == 2'd0) begin
                            od_fifo0    <= ob_rdata;
                            od_fifo_cnt <= 2'd1;
                        end else begin
                            od_fifo1    <= ob_rdata;
                            od_fifo_cnt <= 2'd2;
                        end
                    end
                end
                                OD_RSP: begin
                    wr_rsp_ready <= 1'b1;
                    if (wr_rsp_valid && wr_rsp_ready) begin
                        wr_rsp_ready <= 1'b0;
                        if (wr_rsp_error || wr_rsp_aborted) begin
                            od_err_code  <= (wr_rsp_error_code == 16'h0) ?
                                            OP_E_AXI_RESP : wr_rsp_error_code;
                            od_err_pulse <= 1'b1;
                            od_state     <= OD_IDLE;
                        end else begin
                            od_rows_drained <= od_rows_drained + 16'd1;
                            od_state        <= OD_IDLE;
                        end
                    end
                end
                default: od_state <= OD_IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------------
    // Performance counters: simulation-only instrumentation for the
    // verification bench (hierarchical access).  Excluded from synthesis so
    // the OOC resource gate measures the functional datapath only, matching
    // the treatment the frozen B0 counters received.
    // ------------------------------------------------------------------
    // synthesis translate_off
    // ------------------------------------------------------------------
    wire out_beat_accept_pulse = wr_data_valid && wr_data_ready;
    reg [31:0] total_cycles;
    reg [31:0] input_read_cycles;
    reg [31:0] weight_load_cycles;
    reg [31:0] compute_cycles;
    reg [31:0] wr_wait_cycles;
    reg [31:0] valid_mac_count;
    reg [31:0] mac_active_cycles;
    reg [31:0] rq_active_cycles;
    reg [31:0] mac_rq_overlap_cycles;
    reg [31:0] read_commands;
    reg [31:0] read_beats;
    reg [31:0] write_commands;
    reg [31:0] write_beats;
    reg [31:0] group_count;
    reg [31:0] row_stall_cycles;
    reg [31:0] out_stall_cycles;
    reg [31:0] grp_start_ts;
    reg [31:0] grp_start_delta;

    wire rq_any_active = (rq_phase == RQ_ACT) ||
                         s1v[0] || s2v[0] || s3v[0] || s4v[0] || s5v[0];
    wire rq_group_launch_pulse = rq_start;
    wire mac_rq_overlap_pulse  = mac_busy_w && rq_any_active;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || core_soft_reset || launch_tick) begin
            total_cycles <= 32'd0; input_read_cycles <= 32'd0;
            weight_load_cycles <= 32'd0; compute_cycles <= 32'd0;
            wr_wait_cycles <= 32'd0; valid_mac_count <= 32'd0;
            mac_active_cycles <= 32'd0; rq_active_cycles <= 32'd0;
            mac_rq_overlap_cycles <= 32'd0;
            read_commands <= 32'd0; read_beats <= 32'd0;
            write_commands <= 32'd0; write_beats <= 32'd0;
            group_count <= 32'd0; row_stall_cycles <= 32'd0;
            out_stall_cycles <= 32'd0;
            grp_start_ts <= 32'd0; grp_start_delta <= 32'd0;
        end else begin
            if (state != L_IDLE) total_cycles <= total_cycles + 32'd1;
            if ((rd_state != RD_IDLE) && (rd_kind == K_ROW) && !fatal_flag)
                input_read_cycles <= input_read_cycles + 32'd1;
            if ((rd_state != RD_IDLE) && (rd_kind != K_ROW) && !fatal_flag)
                weight_load_cycles <= weight_load_cycles + 32'd1;
            if (mac_busy_w && !fatal_flag)
                compute_cycles <= compute_cycles + 32'd1;
            if ((od_state != OD_IDLE) && !fatal_flag)
                wr_wait_cycles <= wr_wait_cycles + 32'd1;
            if (p2_valid && !fatal_flag)
                valid_mac_count <= valid_mac_count + {16'b0, p2_lc};
            if (mac_busy_w && !fatal_flag) mac_active_cycles <= mac_active_cycles + 32'd1;
            if (rq_any_active && !fatal_flag) rq_active_cycles <= rq_active_cycles + 32'd1;
            if (mac_rq_overlap_pulse && !fatal_flag)
                mac_rq_overlap_cycles <= mac_rq_overlap_cycles + 32'd1;
            if (rd_cmd_valid && rd_cmd_ready) read_commands <= read_commands + 32'd1;
            if (rd_data_valid && rd_data_ready) read_beats <= read_beats + 32'd1;
            if (wr_cmd_valid && wr_cmd_ready) write_commands <= write_commands + 32'd1;
            if (out_beat_accept_pulse) write_beats <= write_beats + 32'd1;
            if (p2_valid && (p2_tap == 4'd8) && !fatal_flag)
                group_count <= group_count + 32'd1;
            if (mac_issue_stall) row_stall_cycles <= row_stall_cycles + 32'd1;
            if (wr_data_valid && !wr_data_ready)
                out_stall_cycles <= out_stall_cycles + 32'd1;
            if (grp_launch_pulse) begin
                grp_start_delta <= total_cycles - grp_start_ts;
                grp_start_ts    <= total_cycles;
            end
        end
    end


    // synthesis translate_on

    // unused-signal sink (keeps lint quiet without changing behavior)
    wire unused_ok = |{rd_data[0], wr_rsp_tag[0], g_exp_in_bytes[0],
                       g_exp_out_bytes[0], g_exp_w_bytes[0], g_exp_b_bytes[0],
                       g_exp_q_bytes[0], g_exp_calloc[0], cf_cout[0]};

endmodule
