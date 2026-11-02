`include "npu_backend_contract.vh"
`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// conv3x3_core.v
//
// Parameterized ABI 1.0.0 INT8 dense 3x3 convolution core.
//
// Architecture reuse from the verified D:/yolov8/src/cnn/conv project:
//   * three-row activation window / row fill order: conv33_data_in_ctrl.v
//   * nine weight faces consumed as 64-byte ABI blocks: conv_weight.v
//   * 8 input x 8 output outer-product PE and adder tree: conv_calc.v
//   * row-oriented output staging: conv_data_out_buf_top.v
//
// The ABI weight packing and requantization rules are deliberately taken from
// D:/dl_v0.1/docs/op_abi.md; the reference project's weight packing and scale
// format are not used.
// -----------------------------------------------------------------------------
module conv3x3_core #(
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH  = 32,
    parameter integer MAX_CIN    = 64,
    parameter integer MAX_COUT   = 64,
    parameter integer MAX_W      = 256,
    // v0.2 affine capacity overrides.  A zero keeps the historical derivation
    // from MAX_C*/MAX_W byte-for-byte; an affine build passes the *real* largest
    // row span / tile product of the graph instead of MAX_C*MAX_W (GAP-18/22).
    parameter integer MAX_LINE_ROW_BYTES = 0,
    parameter integer MAX_OUT_ROW_BYTES  = 0,
    parameter integer MAX_WEIGHT_TILES   = 0,
    // Read-port count of the weight face RAM: 0 keeps the historical four
    // replicated ports; a single-face profile (Tn<=8/Tm<=8) passes 1.
    parameter integer WEIGHT_READ_PORTS  = 0,
    // Build-time legal feature-map height (v0.2 spatial capacity).  MAX_H only
    // gates legality: 0 keeps the legacy behaviour (no height check at all, so
    // every pre-v0.2 build/elaboration is unchanged).  A positive value rejects
    // cfg_h > MAX_H with E_UNSUPPORTED in validate_config_header.  The row
    // buffers are sized from MAX_W and never from MAX_H, so raising the height
    // range does not enlarge any RAM depth.
    parameter integer MAX_H      = 0,
    parameter integer Tn         = 16,
    parameter integer Tm         = 16,
    parameter integer WEIGHT_MODE = 0,
    parameter integer MAC_IMPL = 0,
    parameter signed [7:0] LOCAL_W0 = 0, parameter signed [7:0] LOCAL_W1 = 0,
    parameter signed [7:0] LOCAL_W2 = 0, parameter signed [7:0] LOCAL_W3 = 0,
    parameter signed [7:0] LOCAL_W4 = 0, parameter signed [7:0] LOCAL_W5 = 0,
    parameter signed [7:0] LOCAL_W6 = 0, parameter signed [7:0] LOCAL_W7 = 0,
    parameter signed [7:0] LOCAL_W8 = 0
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
    input  wire [1:0]  cfg_stride,
    input  wire [1:0]  cfg_pad,
    input  wire [7:0]  cfg_x_zero_point,
    input  wire [31:0] cfg_in_row_stride_bytes,
    input  wire [31:0] cfg_out_row_stride_bytes,
    input  wire [31:0] cfg_in_bytes,
    input  wire [31:0] cfg_weight_bytes,
    input  wire [31:0] cfg_bias_bytes,
    input  wire [31:0] cfg_quant_bytes,
    input  wire [31:0] cfg_out_bytes,
    input  wire [31:0] cfg_conv_cfg,
    input  wire [31:0] cfg_quant_cfg,
    input  wire [31:0] cfg_act_cfg,

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
    input  wire [127:0] rd_data,
    input  wire        rd_data_last,
    input  wire        rd_data_error,
    input  wire [15:0] rd_data_error_code,
    input  wire [7:0]  rd_data_tag,

    output wire        wr_cmd_valid,
    input  wire        wr_cmd_ready,
    output wire [31:0] wr_cmd_addr_byte,
    output wire [15:0] wr_cmd_bytes,
    output wire [7:0]  wr_cmd_tag,
    output wire        wr_data_valid,
    input  wire        wr_data_ready,
    output wire [127:0] wr_data,
    output wire        wr_data_last,
    output wire [7:0]  wr_data_tag,
    input  wire        wr_rsp_valid,
    output wire        wr_rsp_ready,
    input  wire        wr_rsp_error,
    input  wire        wr_rsp_aborted,
    input  wire [15:0] wr_rsp_error_code,
    input  wire [7:0]  wr_rsp_tag
);

    // Capacity-bound scheduling: a build limited to <=8 channels cannot use
    // the upper ABI block. Do not issue dummy 16-channel microtiles and then
    // synthesize a universal 16x16 gather network for an 8x8 physical engine.
    function integer core_ceil_log2;
        input integer value;
        integer p, levels;
        begin p=1;levels=0;while(p<value)begin p=p<<1;levels=levels+1;end core_ceil_log2=levels;end
    endfunction
    // A non-straddling microtile consumes one 8x8 ABI weight block per issue.
    // Arbitrary straddling lane sizes retain the four-face reference path.
    // DSP legacy adapter is unchanged here; its separate integration gate stays
    // outside this balanced/legacy resource prototype.
    localparam integer SINGLE_WEIGHT_FACE = (MAC_IMPL != 1) && (Tn <= 8) && (Tm <= 8) &&
                                            ((8 % Tn) == 0) && ((8 % Tm) == 0);
    localparam integer IN_GROUP_SPAN = (MAX_CIN <= 8) ? 8 : 16;
    localparam integer OUT_GROUP_SPAN = (MAX_COUT <= 8) ? 8 : 16;

    // Static weight/activation gather (plan section 5).  Only the MAC3/DDR
    // backend with an aligned 4/8/16 tile that divides its ABI group width may
    // select the constant-candidate network; every other path keeps the
    // original runtime dynamic gather.  The calculator re-checks the same rule.
    localparam integer OPT_TILE_LEGAL =
        ((Tn == 4) || (Tn == 8) || (Tn == 16)) &&
        ((Tm == 4) || (Tm == 8) || (Tm == 16)) &&
        (Tn <= IN_GROUP_SPAN) && (Tm <= OUT_GROUP_SPAN) &&
        ((IN_GROUP_SPAN % Tn) == 0) && ((OUT_GROUP_SPAN % Tm) == 0);
    localparam integer STATIC_DDR_GATHER =
        (MAC_IMPL == `NPU_MAC_BALANCED_DSP) && (WEIGHT_MODE == 0) && OPT_TILE_LEGAL;
    // Step B keeps the current physical read-port organisation.  The span is
    // the current 8x8 single-face window when that path is taken, else the
    // full 16x16 ABI group.
    localparam integer GATHER_IN_SPAN_B  = SINGLE_WEIGHT_FACE ? 8 : 16;
    localparam integer GATHER_OUT_SPAN_B = SINGLE_WEIGHT_FACE ? 8 : 16;

    // ------------------------------------------------------------------
    // Step C: one physical read port per required ABI face (plan section 6).
    // The physical port index p = oh*I_HALVES + ih.  The calculator only ever
    // sees a local 8/8 or 16/16 window, so the mixed profiles need two ports
    // while the public 16x16 profile keeps its four ports.
    // ------------------------------------------------------------------
    localparam integer LOCAL_I_SPAN = (Tn <= 8) ? 8 : 16;
    localparam integer LOCAL_O_SPAN = (Tm <= 8) ? 8 : 16;
    localparam integer I_HALVES = (Tn == 16) ? 2 : 1;
    localparam integer O_HALVES = (Tm == 16) ? 2 : 1;
    localparam integer FACE_PORTS = I_HALVES * O_HALVES;   // 1, 2 or 4
    // Only the four mixed one-axis-16 profiles need the two-port adapter.  The
    // 1-port (both axes <= 8) and 4-port (both axes 16) profiles keep the
    // candidate-B wiring exactly, so candidate C cannot perturb their netlist.
    localparam integer TWO_PORT_ADAPTER = STATIC_DDR_GATHER && (FACE_PORTS == 2);
    // W2 is selected only for the fixed E1 MAX16 / 8x8 / DDR candidate.
    // Other capacities/backends/local weights retain the original schedule.
    // A partial ABI block still issues every one of its eight physical lanes.
    localparam integer OPT_RUNTIME_TILE_SKIP = (MAX_CIN == 16) && (MAX_COUT == 16) &&
        (MAX_W == 32) && (Tn == 8) && (Tm == 8) && (MAC_IMPL == 3) && (WEIGHT_MODE == 0);
    localparam integer TN_CURSOR_W = core_ceil_log2(IN_GROUP_SPAN);
    localparam integer TM_CURSOR_W = core_ceil_log2(OUT_GROUP_SPAN);
    localparam integer PE_TOKEN_LATENCY = `NPU_CALC_LATENCY(Tn, MAC_IMPL);
    localparam integer MAX_ICB       = (MAX_CIN + 7) / 8;
    localparam integer MAX_OCB       = (MAX_COUT + 7) / 8;
    // Output block-pair cursor width (one hand-off group covers two 8-channel
    // blocks); used by the hand-off advance slice.
    localparam integer OBC_W         = core_ceil_log2(MAX_OCB + 1);
    // spram uses a minimum two-entry address space (AW=$clog2(DP)); keep the physical face RAM valid for a single logical block.
    localparam integer MAX_WTILES    = (MAX_WEIGHT_TILES > 0) ? MAX_WEIGHT_TILES :
                                       ((MAX_ICB * MAX_OCB < 2) ? 2 : (MAX_ICB * MAX_OCB));
    localparam integer LINE_MAX_ROW_BYTES =
        (MAX_LINE_ROW_BYTES > 0) ? MAX_LINE_ROW_BYTES :
        ((((MAX_CIN + 7) / 8) * 8) * MAX_W + 15) / 16 * 16;
    localparam integer LINE_MAX_ROW_BEATS = LINE_MAX_ROW_BYTES / 16;
    localparam integer OUT_MAX_ROW_BYTES =
        (MAX_OUT_ROW_BYTES > 0) ? MAX_OUT_ROW_BYTES :
        ((((MAX_COUT + 7) / 8) * 8) * MAX_W + 15) / 16 * 16;
    localparam integer OUT_MAX_ROW_BEATS = OUT_MAX_ROW_BYTES / 16;
    localparam integer WEIGHT_PORT_COUNT = (WEIGHT_READ_PORTS > 0) ? WEIGHT_READ_PORTS : 4;

    // ------------------------------------------------------------------
    // v0.2 spatial capacity: build-derived row budgets and width contract.
    //
    // Only MAX_W and the channel capacities scale the storage; MAX_H is a
    // legality gate and never sizes a RAM.  For the spatial_B build
    // (MAX_CIN=MAX_COUT=16, MAX_W=640):
    //     LINE_MAX_ROW_BYTES = align16(align8(16)*640) = 10240 B
    //     LINE_MAX_ROW_BEATS = 640
    //     OUT_MAX_ROW_BYTES  = 10240 B, OUT_MAX_ROW_BEATS = 640
    // ST_CFG_ROW_CHECK rejects any runtime row stride above these budgets and
    // any stride > 65535 B before the load FSM starts, so:
    //   * in_row_beats/out_row_beats stay inside their 16-bit lanes
    //     (<= 4095 beats for the largest legal 65535-byte stride);
    //   * the [15:0] slices of cfg_*_row_stride_bytes cannot truncate;
    //   * conv3x3_linebuf / conv3x3_out_buf keep RAM depth == MAX_ROW_BEATS
    //     with AW = $clog2(MAX_ROW_BEATS); every write address is gated by
    //     wr_beat < MAX_ROW_BEATS and every read address by beats_safe*, so a
    //     non-power-of-two depth never aliases;
    //   * conv3x3_out_fifo depth == OUT_MAX_ROW_BEATS, i.e. exactly one full
    //     row, which is what the reference hand-off requires.
    // The 7x16-bit row products (cfg_in_row_raw/cfg_out_row_raw) are exact for
    // any MAX_C <= 64 and any 16-bit W (max 64*65535 < 2^23), so no clamp or
    // wider field is needed inside the ABI 1.0 capacity envelope.
    // ------------------------------------------------------------------

    localparam [4:0] ST_IDLE        = 5'd0;
    localparam [4:0] ST_W_CMD       = 5'd1;
    localparam [4:0] ST_W_DATA      = 5'd2;
    localparam [4:0] ST_B_CMD       = 5'd3;
    localparam [4:0] ST_B_DATA      = 5'd4;
    localparam [4:0] ST_Q_CMD       = 5'd5;
    localparam [4:0] ST_Q_DATA      = 5'd6;
    localparam [4:0] ST_IN_CMD      = 5'd7;
    localparam [4:0] ST_IN_DATA     = 5'd8;
    localparam [4:0] ST_ROW_PREP    = 5'd9;
    localparam [4:0] ST_CALC_INIT   = 5'd10;
    localparam [4:0] ST_CALC_TAP    = 5'd11;
    localparam [4:0] ST_CALC_WRITE  = 5'd12;
    localparam [4:0] ST_ERROR       = 5'd16;
    localparam [4:0] ST_CALC_PUSH   = 5'd17;
    localparam [4:0] ST_OUT_WAIT    = 5'd18;
    localparam [4:0] ST_CALC_RQ      = 5'd19;
    localparam [4:0] ST_CALC_RQ_ABS  = 5'd24;
    localparam [4:0] ST_CALC_RQ_ADD  = 5'd25;
    localparam [4:0] ST_CALC_RQ_SHIFT= 5'd26;
    localparam [4:0] ST_CALC_RQ_SAT  = 5'd27;
    localparam [4:0] ST_CFG_GEOM     = 5'd20;
    localparam [4:0] ST_CFG_ROW      = 5'd21;
    localparam [4:0] ST_CFG_BYTES    = 5'd22;
    localparam [4:0] ST_CFG_CHECK    = 5'd23;
    localparam [4:0] ST_CFG_ROW_CHECK= 5'd28;
    localparam [4:0] ST_OUT_SELECT   = 5'd29;
    localparam [4:0] ST_OUT_CORRECT  = 5'd30;
    localparam [4:0] ST_OUT_ACC      = 5'd31;
    // M3 output overlap: the result handoff is a compute state, while requant +
    // write-back run in a parallel state machine (out_state) so a finished
    // group's epilogue overlaps the next group's issue.  The former in-FSM
    // epilogue state numbers 5'd12/17/18/19/24..27/29..31 stay reserved (never
    // entered by the compute FSM) so the frozen output accounting in the graph
    // testbench remains readable.
    localparam [4:0] ST_CALC_HANDOFF = 5'd13;
    localparam [3:0] O_IDLE          = 4'd0;
    localparam [3:0] O_SELECT        = 4'd1;
    localparam [3:0] O_CORRECT       = 4'd2;
    localparam [3:0] O_ACC           = 4'd3;
    localparam [3:0] O_RQ            = 4'd4;
    localparam [3:0] O_RQ_ABS        = 4'd5;
    localparam [3:0] O_RQ_ADD        = 4'd6;
    localparam [3:0] O_RQ_SHIFT      = 4'd7;
    localparam [3:0] O_RQ_SAT        = 4'd8;
    localparam [3:0] O_PUSH          = 4'd9;

    localparam [7:0] TAG_INPUT  = 8'h01;
    localparam [7:0] TAG_WEIGHT = 8'h02;
    localparam [7:0] TAG_BIAS   = 8'h03;
    localparam [7:0] TAG_QUANT  = 8'h04;
    localparam [7:0] TAG_OUTPUT = 8'h05;

    localparam [15:0] E_NONE         = 16'h0000;
    localparam [15:0] E_CFG          = 16'h0001;
    localparam [15:0] E_UNSUPPORTED   = 16'h0002;
    localparam [15:0] E_ADDR_ALIGN   = 16'h0003;
    localparam [15:0] E_ADDR_RANGE   = 16'h0004;
    localparam [15:0] E_STRIDE       = 16'h0005;
    localparam [15:0] E_QUANT        = 16'h0006;
    localparam [15:0] E_AXI_RESP     = 16'h0008;
    localparam [15:0] E_AXI_PROTOCOL = 16'h0009;
    localparam [15:0] E_CORE_INTERNAL = 16'h000A;
    localparam [31:0] READ_CHUNK_BYTES = 32'd4096;

    reg [4:0] state;

    // Core-local start snapshot. The header validator retains the external
    // inputs so rejection priority/cycle is unchanged. Every executed command
    // and RAM/PE control thereafter uses this immutable snapshot.
    // Capture on an IDLE start sample; rejected headers never execute the data.
    reg [31:0] cfg_core_in_addr_byte_q;
    reg [31:0] cfg_core_weight_addr_byte_q;
    reg [31:0] cfg_core_bias_addr_byte_q;
    reg [31:0] cfg_core_quant_addr_byte_q;
    reg [31:0] cfg_core_out_addr_byte_q;
    reg [15:0] cfg_core_h_q;
    reg [15:0] cfg_core_w_q;
    reg [15:0] cfg_core_cin_q;
    reg [15:0] cfg_core_cout_q;
    reg [1:0] cfg_core_stride_q;
    (* max_fanout = 32 *) reg [1:0] cfg_core_pad_q;
    reg [7:0] cfg_core_x_zero_point_q;
    reg [31:0] cfg_core_in_row_stride_bytes_q;
    reg [31:0] cfg_core_out_row_stride_bytes_q;
    reg [31:0] cfg_core_in_bytes_q;
    reg [31:0] cfg_core_weight_bytes_q;
    reg [31:0] cfg_core_bias_bytes_q;
    reg [31:0] cfg_core_quant_bytes_q;
    reg [31:0] cfg_core_out_bytes_q;
    reg [31:0] cfg_core_conv_cfg_q;
    reg [31:0] cfg_core_quant_cfg_q;
    reg [31:0] cfg_core_act_cfg_q;
    reg [31:0] cfg_core_cin_alloc_q, cfg_core_cout_alloc_q;
    (* max_fanout = 64 *) reg cfg_core_k1_reuse_q;
    wire [31:0] cfg_cin_alloc_next = (({16'b0, cfg_cin} + 32'd7) >> 3) << 3;
    wire cfg_k1_reuse_next;
    reg [core_ceil_log2(MAX_ICB+1)-1:0] cfg_core_icb_count_q;
    reg [core_ceil_log2(MAX_OCB+1)-1:0] cfg_core_ocb_count_q;
    reg [1:0] cfg_core_stride_eff_q;

    // Synchronous config-data reset avoids adding async-reset datapath loads.
    always @(posedge clk) begin
        if (!rst_n) begin
            cfg_core_in_addr_byte_q <= 0;
            cfg_core_weight_addr_byte_q <= 0;
            cfg_core_bias_addr_byte_q <= 0;
            cfg_core_quant_addr_byte_q <= 0;
            cfg_core_out_addr_byte_q <= 0;
            cfg_core_h_q <= 0;
            cfg_core_w_q <= 0;
            cfg_core_cin_q <= 0;
            cfg_core_cout_q <= 0;
            cfg_core_stride_q <= 0;
            cfg_core_pad_q <= 0;
            cfg_core_x_zero_point_q <= 0;
            cfg_core_in_row_stride_bytes_q <= 0;
            cfg_core_out_row_stride_bytes_q <= 0;
            cfg_core_in_bytes_q <= 0;
            cfg_core_weight_bytes_q <= 0;
            cfg_core_bias_bytes_q <= 0;
            cfg_core_quant_bytes_q <= 0;
            cfg_core_out_bytes_q <= 0;
            cfg_core_conv_cfg_q <= 0;
            cfg_core_quant_cfg_q <= 0;
            cfg_core_act_cfg_q <= 0;
            cfg_core_cin_alloc_q <= 0; cfg_core_cout_alloc_q <= 0;
            cfg_core_k1_reuse_q <= 1'b0;
            cfg_core_icb_count_q <= 0; cfg_core_ocb_count_q <= 0;
            cfg_core_stride_eff_q <= 1;
        end else if (!core_soft_reset && !core_abort &&
                     (state == ST_IDLE) && core_start) begin
            cfg_core_in_addr_byte_q <= cfg_in_addr_byte;
            cfg_core_weight_addr_byte_q <= cfg_weight_addr_byte;
            cfg_core_bias_addr_byte_q <= cfg_bias_addr_byte;
            cfg_core_quant_addr_byte_q <= cfg_quant_addr_byte;
            cfg_core_out_addr_byte_q <= cfg_out_addr_byte;
            cfg_core_h_q <= cfg_h;
            cfg_core_w_q <= cfg_w;
            cfg_core_cin_q <= cfg_cin;
            cfg_core_cout_q <= cfg_cout;
            cfg_core_stride_q <= cfg_stride;
            cfg_core_pad_q <= cfg_pad;
            cfg_core_x_zero_point_q <= cfg_x_zero_point;
            cfg_core_in_row_stride_bytes_q <= cfg_in_row_stride_bytes;
            cfg_core_out_row_stride_bytes_q <= cfg_out_row_stride_bytes;
            cfg_core_in_bytes_q <= cfg_in_bytes;
            cfg_core_weight_bytes_q <= cfg_weight_bytes;
            cfg_core_bias_bytes_q <= cfg_bias_bytes;
            cfg_core_quant_bytes_q <= cfg_quant_bytes;
            cfg_core_out_bytes_q <= cfg_out_bytes;
            cfg_core_conv_cfg_q <= cfg_conv_cfg;
            cfg_core_quant_cfg_q <= cfg_quant_cfg;
            cfg_core_act_cfg_q <= cfg_act_cfg;
            cfg_core_cin_alloc_q <= cfg_cin_alloc_next;
            cfg_core_k1_reuse_q <= cfg_k1_reuse_next;
            cfg_core_cout_alloc_q <= (({16'b0, cfg_cout} + 32'd7) >> 3) << 3;
            cfg_core_icb_count_q <= (cfg_cin + 15'd7) >> 3;
            cfg_core_ocb_count_q <= (cfg_cout + 15'd7) >> 3;
            cfg_core_stride_eff_q <= (cfg_stride == 2'd0) ? 2'd1 : 2'd2;
        end
    end

    // Native K1 weight format (conv_cfg[11]): the DDR weight buffer holds only
    // the center-tap 64-byte group per (ocb, icb) block instead of the frozen
    // nine-tap layout, i.e. cfg_weight_bytes = icb*ocb*64.  Bit 11 is reserved
    // (must be zero) in every legacy record, so legacy builds are unchanged.
    // The lowered 1x1 Conv keeps its pad/stride geometry; with the standard
    // center-tap lowering (pads=1) tap 4 samples exactly the 1x1 input pixel,
    // so only the center-tap weight face may reach the PE.
    wire k1_native_weight = cfg_core_conv_cfg_q[11];

    // ------------------------------------------------------------------
    // M4: K1 eight-group reuse (design contract section 2/3/5).
    //
    // Native K1 with the square 8/8 or 16/16 profile stops treating the nine
    // physical tap planes as 3x3 spatial taps: tap g = 2*i + j carries the
    // product group of input subgroup i (Tn channels) and output subgroup j
    // (Tm channels), so one accepted token is a (4*Tn) x (2*Tm) channel block
    // and the ninth tap is masked.  The other legal K1 profiles (tn4/tm4 and
    // the mixed 8/16 profiles) keep the M3 center-tap path; the reuse
    // condition is an elaboration-time structural property, not a layer name.
    // ------------------------------------------------------------------
    localparam integer K1_PROFILE_OK =
        (((MAC_IMPL == `NPU_MAC_BALANCED) || (MAC_IMPL == `NPU_MAC_BALANCED_DSP)) &&
         (Tn == Tm) && ((Tn == 8) || (Tn == 16))) ? 1 : 0;
    // ABI blocks per input/output group (Tn/2 and Tm/4 are integers for the
    // legal 8/16 profiles) and K1 ABI faces per group block axis.
    localparam integer K1_BLK_I = (K1_PROFILE_OK != 0) ? (Tn / 2) : 1;
    localparam integer K1_BLK_O = (K1_PROFILE_OK != 0) ? (Tm / 4) : 1;
    localparam integer K1_BPI   = (K1_PROFILE_OK != 0) ? (Tn / 8) : 1;
    localparam integer K1_BPO   = (K1_PROFILE_OK != 0) ? (Tm / 8) : 1;
    // Weight rows per token: 4*Tn*2*Tm bytes = 8 tap planes x 64 B.
    localparam integer K1_ROWS  = (K1_PROFILE_OK != 0) ? ((Tn * Tm) / 64) : 1;
    // Capacity groups are rounded UP: a capacity ABI block count that is not a
    // multiple of the group block count (e.g. MAX_COUT=20 on the 8/8 profile)
    // still needs a distinct row slot for its trailing group.  The floor form
    // aliased the trailing output group onto the first rows of the next input
    // group; that is a real weight-bank collision, so the reuse legality also
    // requires the row footprint to fit MAX_WTILES.
    localparam integer K1_OC_GROUPS  = (MAX_OCB + K1_BLK_O - 1) / K1_BLK_O;
    localparam integer K1_IN_GROUPS_C = (MAX_ICB + K1_BLK_I - 1) / K1_BLK_I;
    localparam integer K1_ROW_STRIDE = K1_OC_GROUPS * K1_ROWS;
    localparam integer K1_ROW_CAP    = K1_ROW_STRIDE * K1_IN_GROUPS_C;
    localparam integer K1_STORAGE_OK = (MAX_WTILES >= K1_ROW_CAP) ? 1 : 0;
    localparam integer K1_REUSE_LEGAL = ((K1_PROFILE_OK != 0) && (K1_STORAGE_OK != 0)) ? 1 : 0;
    assign cfg_k1_reuse_next = cfg_conv_cfg[11] && (K1_REUSE_LEGAL != 0) &&
                               !cfg_cin_alloc_next[3];
    // Accumulator/output lane count: 2*Tm for B (32), otherwise the historical
    // 16-lane group bank.
    localparam integer K1_ACC_LANES = (K1_PROFILE_OK != 0) ? ((2 * Tm > 16) ? (2 * Tm) : 16) : 16;
    // Runtime mode: a native-K1 record on a reuse-legal build whose allocated
    // input row is 16-byte aligned.  cin_alloc is align8(Cin), so bit3 == 0
    // means every (4*Tn)-channel token window starts on a 16-byte boundary;
    // the few remaining legal Cin values keep the M3 center-tap path.
    wire k1_reuse = cfg_core_k1_reuse_q;

    // ABI-derived logical and physical dimensions.
    wire [31:0] cin_alloc32  = cfg_core_cin_alloc_q;
    wire [31:0] cout_alloc32 = cfg_core_cout_alloc_q;
    wire [15:0] cin_alloc16  = cin_alloc32[15:0];
    wire [15:0] cout_alloc16 = cout_alloc32[15:0];
    wire [15:0] icb_count   = cfg_core_icb_count_q;
    wire [15:0] ocb_count   = cfg_core_ocb_count_q;
    wire [31:0] pad32       = {30'b0, cfg_core_pad_q};
    // ABI CONV_CFG stores stride-1 in the 2-bit stride field; address and
    // shape arithmetic must use the decoded numeric stride, not the CSR code.
    wire [1:0]  stride_eff   = cfg_core_stride_eff_q;
    wire [31:0] stride32    = {30'b0, stride_eff};
    wire [31:0] hpad32 = {16'b0, cfg_core_h_q} + (cfg_core_pad_q == 2'd1 ? 32'd2 : 32'd0);
    wire [31:0] wpad32 = {16'b0, cfg_core_w_q} + (cfg_core_pad_q == 2'd1 ? 32'd2 : 32'd0);
    // No variable divide: stride is 1 or 2 (see stride_eff), so /stride
    // is a shift/mux, matching the reference conv_ctrl row_num[9:1] handling.
    wire [31:0] hout32 = (hpad32 >= 32'd3) ? (((stride32 == 32'd2) ? ((hpad32 - 32'd3) >> 1) : (hpad32 - 32'd3)) + 32'd1) : 32'd0;
    wire [31:0] wout32 = (wpad32 >= 32'd3) ? (((stride32 == 32'd2) ? ((wpad32 - 32'd3) >> 1) : (wpad32 - 32'd3)) + 32'd1) : 32'd0;
    wire [15:0] in_row_beats  = cfg_core_in_row_stride_bytes_q[15:0] >> 4;
    wire [15:0] out_row_beats = cfg_core_out_row_stride_bytes_q[15:0] >> 4;

    // NOTE (XSim 2021.1): a Verilog function called from a continuous
    // assignment only re-evaluates when its *arguments* change; module-level
    // references inside the function body (stride_eff, cfg_core_pad_q) are not added
    // to the sensitivity list.  The shell drives those config nets after
    // reset, so the tap-window assign stayed X forever when it called this
    // function.  The same formula is therefore written inline at the window
    // assign below, where all operands are explicit.  This function is kept
    // as the readable specification of the coordinate mapping and is safe only
    // in procedural (clocked) context.
    function signed [31:0] image_coord;
        input [15:0] out_coord;
        input [15:0] kernel_coord;
        integer value;
        begin
            value = ((stride_eff == 2'd2) ? ({16'b0, out_coord} << 1) : {16'b0, out_coord}) + kernel_coord - cfg_core_pad_q;
            image_coord = value;
        end
    endfunction

    function signed [31:0] need_bottom;
        input [15:0] out_y;
        integer value;
        begin
            value = ((stride_eff == 2'd2) ? ({16'b0, out_y} << 1) : {16'b0, out_y}) + 2 - cfg_core_pad_q;
            if (value < 0) need_bottom = -32'sd1;
            else if (value >= cfg_core_h_q) need_bottom = $signed({16'b0, cfg_core_h_q}) - 32'sd1;
            else need_bottom = value;
        end
    endfunction

    // Cheap, non-arithmetic checks run before the shape/size pipeline.  Keep
    // the error priority identical to the ABI contract; wide byte-count
    // arithmetic is deliberately split across later FSM states.
    function [15:0] validate_config_header;
        begin
            validate_config_header = E_NONE;
            if ((cfg_conv_cfg[1:0] != 2'b00) ||
                (cfg_conv_cfg[7:6] != 2'b00) ||
                (cfg_conv_cfg[9:8] != 2'b00) ||
                (cfg_stride > 2'd1) || (cfg_pad > 2'd1)) begin
                validate_config_header = E_UNSUPPORTED;
            end else if ((cfg_conv_cfg[3:2] != cfg_stride) ||
                         (cfg_conv_cfg[5:4] != cfg_pad) ||
                         (cfg_conv_cfg[10] != 1'b1) ||
                         (cfg_conv_cfg[31:12] != 20'b0)) begin
                validate_config_header = E_CFG;
            end else if (cfg_conv_cfg[11] &&
                         ((MAC_IMPL == `NPU_MAC_LEGACY) ||
                          (MAC_IMPL == `NPU_MAC_DSP_ADAPTER) ||
                          (WEIGHT_MODE != 0))) begin
                // Native K1 needs the tap-masked balanced calculator; the
                // legacy/adapter calculators (and any local-weight build)
                // reject the format instead of silently reading nine taps.
                validate_config_header = E_UNSUPPORTED;
            end else if ((cfg_h == 16'b0) || (cfg_w == 16'b0) ||
                         (cfg_cin == 16'b0) || (cfg_cout == 16'b0)) begin
                validate_config_header = E_CFG;
            end else if ((cfg_cin > MAX_CIN) || (cfg_cout > MAX_COUT) ||
                         (cfg_w > MAX_W) ||
                         // v0.2 spatial capacity: MAX_H=0 keeps the legacy
                         // "no height limit" behaviour; a positive build limit
                         // rejects an over-height command with the same
                         // E_UNSUPPORTED priority as the other capacity bounds.
                         ((MAX_H > 0) && (cfg_h > MAX_H))) begin
                validate_config_header = E_UNSUPPORTED;
            end else if (cfg_act_cfg[31:1] != 31'b0) begin
                validate_config_header = E_UNSUPPORTED;
            end else if (cfg_quant_cfg[7:0] != cfg_x_zero_point) begin
                validate_config_header = E_CFG;
            end else if (cfg_quant_cfg[15:8] != 8'b0) begin
                validate_config_header = E_CFG;
            end else if (cfg_quant_cfg[17:16] != 2'b00) begin
                validate_config_header = E_UNSUPPORTED;
            end else if (cfg_quant_cfg[31:18] != 14'b0) begin
                validate_config_header = E_QUANT;
            end
        end
    endfunction

    // Match the ABI's truncation-toward-zero output-shape formula for the
    // supported stride encodings (0=>1, 1=>2).
    function integer conv_output_dim;
        input [15:0] dim;
        input [1:0] stride_cfg;
        input [1:0] pad_cfg;
        integer dim_i, numerator_i, quotient_i;
        begin
            dim_i = dim;
            numerator_i = dim_i + (2 * pad_cfg) - 3;
            if (numerator_i >= 0)
                quotient_i = (stride_cfg == 2'd1) ? (numerator_i >> 1) : numerator_i;
            else
                quotient_i = (stride_cfg == 2'd1) ? -((-numerator_i) >> 1) : numerator_i;
            conv_output_dim = quotient_i + 1;
        end
    endfunction

    // On-chip weight faces.  Each entry is one ABI 64-byte block:
    // [tap][ocb * ICB + icb][ic_inner*8 + oc_inner].
    // Nine independent on-chip weight faces, mirroring the reference conv_weight
    // spram_1..spram_9 bank structure.
    wire [511:0] weight_faces [0:3][0:8];
    // Native K1 tap mask: the compact buffer cannot make every stale plane safe
    // by itself, because the other eight faces may still hold the weights of an
    // earlier 3x3 command on this shared engine.
    //   * M4 K1 reuse: taps 0..7 are the eight product groups and are kept; the
    //     unused ninth tap is masked.
    //   * M3 center-tap K1 (non-reuse legal profiles): only tap 4 is kept.
    //   * 3x3 records: all nine taps pass through untouched.
    wire [511:0] pe_weight_faces [0:3][0:8];
    genvar gk1, gk1p;
    generate
        for (gk1 = 0; gk1 < 9; gk1 = gk1 + 1) begin : GEN_K1_TAP_MASK
            wire tap_keep = !k1_native_weight ? 1'b1 :
                            ((K1_REUSE_LEGAL != 0) && k1_reuse) ? (gk1 != 8) :
                            (gk1 == 4);
            for (gk1p = 0; gk1p < 4; gk1p = gk1p + 1) begin : GEN_K1_FACE
                assign pe_weight_faces[gk1p][gk1] =
                    tap_keep ? weight_faces[gk1p][gk1] : 512'b0;
            end
        end
    endgenerate
    wire [15:0]  calc_rd_index [0:3];
    wire         calc_rd_zero  [0:3];
    reg signed [31:0] bias_mem [0:MAX_COUT-1];
    reg [31:0] q_m0 [0:MAX_COUT-1];
    reg [7:0]  q_shift [0:MAX_COUT-1];
    reg signed [7:0] q_zp [0:MAX_COUT-1];
    reg               out_wr_en;
    reg [15:0]        out_wr_beat;
    reg [3:0]         out_wr_byte;
    reg [63:0]        out_wr_data;
    reg               out_wr_last;
    reg               out_wr_zero_high;


    reg [15:0] fault_code_r;

    reg [31:0] load_remaining;
    reg [31:0] load_offset;
    reg [15:0] load_chunk_left;
    reg [15:0] w_ocb;
    reg [15:0] w_tap;
    reg [15:0] w_icb;
    reg [15:0] w_beat4;
    reg [15:0] bias_idx;
    reg [15:0] quant_idx;

    reg [31:0] input_remaining;
    reg [31:0] input_offset;
    reg [15:0] input_row;
    reg [15:0] input_beat;
    reg [15:0] input_rows_loaded;
    reg [15:0] input_chunk_left;
    reg        input_cmd_active;

    reg [15:0] calc_oy;
    reg [15:0] calc_ox;
    reg [core_ceil_log2(MAX_OCB+1)-1:0] calc_ocb;
    reg [core_ceil_log2(MAX_ICB+1)-1:0] calc_icb;
    reg signed [31:0] calc_acc [0:K1_ACC_LANES-1];
    reg signed [31:0] calc_wsum [0:K1_ACC_LANES-1];
    reg               calc_issue_done;
    reg [TN_CURSOR_W-1:0] calc_tn_offset;
    reg [TM_CURSOR_W-1:0] calc_tm_offset;
    reg [3:0]         pe_tn_offset_d;
    reg [3:0]         pe_tm_offset_d;
    reg [3:0]         pe_tm_offset_pipe [0:PE_TOKEN_LATENCY];
    reg [7:0]         pe_inflight;
    reg               pe_addr_valid_d;
    reg               issue_done_d;
    reg [15:0]        w_base;
    reg signed [63:0] acc_corr_calc;
    // M3 output overlap: one handed-off 16-channel group snapshot.  The compute
    // FSM fills these once per group (one cycle) and the parallel output pipeline
    // consumes them; out_busy is the only hand-off back-pressure.  Bias is
    // already included in calc_acc, so it is never added here a second time.
    // Stage A: lane/quant select (from ostg_*), B: registered zero-point
    // product, C: accumulator subtraction, D: 8-lane requant, E: 8-byte push.
    reg [3:0]         out_state;
    reg signed [31:0] ostg_acc [0:K1_ACC_LANES-1];
    reg signed [31:0] ostg_wsum [0:K1_ACC_LANES-1];
    reg [15:0]        ostg_ocb_q, ostg_ox_q, ostg_oy_q, ostg_cout_q, ostg_hout_q;
    reg [31:0]        ostg_pixel_base_q;
    reg [7:0]         ostg_x_bias_q;
    reg [1:0]         o_pass_q;      // 8-lane output pass inside one group
    reg [2:0]         ostg_passes_q;  // number of 8-lane passes of the handoff group
    // Compute-side strobe: one cycle after the snapshot is written.  The output
    // pipeline only samples it in O_IDLE, so no acknowledge path is required.
    reg               out_start;
    wire              out_busy = (out_state != O_IDLE);
    reg signed [31:0] out_sel_acc_q [0:7];
    reg signed [31:0] out_sel_wsum_q [0:7];
    reg signed [63:0] out_corr_product_q [0:7];
    // Write-back pipeline registers (reference-style staging: one lane
    // group per stage instead of one giant combinational cone).
    reg signed [31:0] wr_acc [0:7];
    reg [31:0] wr_m0 [0:7];
    reg [7:0]  wr_shift [0:7];
    reg signed [7:0] wr_zp [0:7];
    reg signed [63:0] rq_product [0:7];
    reg [63:0] rq_abs [0:7];
    reg [63:0] rq_round_bias [0:7];
    reg [63:0] rq_round_sum [0:7];
    reg [63:0] rq_magnitude [0:7];
    reg rq_negative [0:7];
    reg signed [9:0] rq_y_calc;
    reg [15:0] cfg_check_tmp;
    reg signed [31:0] cfg_v_hout;
    reg signed [31:0] cfg_v_wout;
    reg [31:0] cfg_v_in_stride;
    reg [31:0] cfg_v_out_stride;
    reg [31:0] cfg_v_in_bytes;
    reg [31:0] cfg_v_out_bytes;
    reg [31:0] cfg_v_weight_bytes;
    reg [31:0] cfg_v_bias_bytes;
    reg [31:0] cfg_v_quant_bytes;
    reg [11:0] cfg_v_weight_blocks;

    wire [15:0] cfg_cin_alloc = (cfg_core_cin_q + 16'd7) & 16'hfff8;
    wire [15:0] cfg_cout_alloc = (cfg_core_cout_q + 16'd7) & 16'hfff8;
    wire signed [31:0] cfg_calc_hout = conv_output_dim(cfg_core_h_q, cfg_core_stride_q, cfg_core_pad_q);
    wire signed [31:0] cfg_calc_wout = conv_output_dim(cfg_core_w_q, cfg_core_stride_q, cfg_core_pad_q);
    // Cin_alloc is limited by the Conv core to 64 lanes and W is a 16-bit
    // ABI field. Keep the input stride product at its true 7x16-bit width.
    // 10-bit allocated channel count x 16-bit width: exact for C_alloc <= 1024
    // (the affine-int8-v1 envelope is 512), so no truncation can wrap a legal row
    // (GAP-19).
    wire [25:0] cfg_in_row_raw = cfg_cin_alloc[9:0] * cfg_core_w_q[15:0];
    wire [26:0] cfg_in_row_aligned =
        ({1'b0, cfg_in_row_raw} + 27'd15) & 27'h7fffff0;
    wire [31:0] cfg_calc_in_stride = {5'b0, cfg_in_row_aligned};
    // Wout is bounded by the 16-bit ABI geometry fields. Restrict the
    // stride product to the actual 7x16-bit operand widths; the prior 32x32
    // expression inferred a multi-DSP cone in the configuration FSM.
    wire [25:0] cfg_out_row_raw = cfg_cout_alloc[9:0] * cfg_v_wout[15:0];
    wire [26:0] cfg_out_row_aligned =
        ({1'b0, cfg_out_row_raw} + 27'd15) & 27'h7fffff0;
    wire [31:0] cfg_calc_out_stride = {5'b0, cfg_out_row_aligned};
    // up to 128 x 64 ABI blocks (C_alloc 1024 x 512): 12 bits, never truncated.
    wire [11:0] cfg_calc_weight_blocks = cfg_cin_alloc[9:3] * cfg_cout_alloc[9:3];
    reg [15:0] calc_rd_c;
    reg [7:0]  q_byte_calc;
    reg [31:0] out_off_calc;

    integer i;
    integer j;
    integer li;
    integer oc_lane;
    integer lane_addr;
    // Dedicated procedural variables for the parallel output pipeline: sharing
    // i/li/lane_addr across two always blocks would infer multi-driven nets.
    integer oi;
    integer o_lane_addr;
    integer o_acc_lane;
    genvar gtap;

    // Activation window from the reference five-bank data-control structure.
    wire        line_wr_en;
    wire [15:0] line_wr_row;
    wire [15:0] line_wr_beat;
    wire [127:0] line_wr_data;
    wire [63:0] line_rd_data [0:8];
    wire [63:0] line_rd_data_b [0:8];
    wire signed [31:0] tap_rd_y [0:8];
    wire signed [31:0] tap_rd_x [0:8];
    wire [63:0] pe_act_data [0:8];
    wire [63:0] pe_act_data_b [0:8];

    // The reference multiplier consumes an unsigned 8-bit activation.  ABI
    // activations are signed INT8, so issue (x + 128) to the copied data path
    // and remove (128 + x_zero_point) * sum(weight) once per output pixel.
    wire [7:0]  pe_x_bias_byte = cfg_core_x_zero_point_q ^ 8'h80;
    wire        pe_addr_valid = (state == ST_CALC_TAP) && !calc_issue_done;
    wire        pe_in_valid   = pe_addr_valid_d;
    wire [511:0] pe_sum_packed;
    wire [255:0] pe_wsum_packed;
    wire [511:0] pe_sum_packed_hi;
    wire [255:0] pe_wsum_packed_hi;
    wire         pe_out_valid;
    // M4 K1 reuse: per-token input-block validity reaches the calculator with
    // the same two-stage latency as the weight/activation data.  bit[2*i+a] is
    // cleared when ABI block a of input subgroup i is beyond the real Cin.
    wire [7:0]   k1_grp_valid;
    reg  [7:0]   k1_valid_q1, k1_valid_q2;
    // ABI 1.0 has no writable local-weight CSR. Local-register mode therefore
    // initializes this bank from build parameters at reset and keeps it stable
    // during a run; constant mode uses the same build parameters directly.
    reg signed [7:0] local_weight_reg [0:8];
    wire signed [7:0] pe_local_weight [0:8];
    assign pe_local_weight[0] = (WEIGHT_MODE == 1) ? local_weight_reg[0] : LOCAL_W0;
    assign pe_local_weight[1] = (WEIGHT_MODE == 1) ? local_weight_reg[1] : LOCAL_W1;
    assign pe_local_weight[2] = (WEIGHT_MODE == 1) ? local_weight_reg[2] : LOCAL_W2;
    assign pe_local_weight[3] = (WEIGHT_MODE == 1) ? local_weight_reg[3] : LOCAL_W3;
    assign pe_local_weight[4] = (WEIGHT_MODE == 1) ? local_weight_reg[4] : LOCAL_W4;
    assign pe_local_weight[5] = (WEIGHT_MODE == 1) ? local_weight_reg[5] : LOCAL_W5;
    assign pe_local_weight[6] = (WEIGHT_MODE == 1) ? local_weight_reg[6] : LOCAL_W6;
    assign pe_local_weight[7] = (WEIGHT_MODE == 1) ? local_weight_reg[7] : LOCAL_W7;
    assign pe_local_weight[8] = (WEIGHT_MODE == 1) ? local_weight_reg[8] : LOCAL_W8;
    wire [7:0]  pe_inflight_next = pe_inflight +
        (pe_in_valid ? 8'd1 : 8'd0) - (pe_out_valid ? 8'd1 : 8'd0);

    assign line_wr_en   = (state == ST_IN_DATA) && rd_data_valid && rd_data_ready;
    assign line_wr_row  = input_row;
    assign line_wr_beat = input_beat;
    assign line_wr_data = rd_data;

    wire [15:0] calc_icb_next = calc_icb + 16'd1;
    // *8 channel addressing is a free concat (reference precomputed-stride
    // style: no index multipliers).
    wire [15:0] ocb_base8 = calc_ocb << 3;
    wire [15:0] calc_ocb_next = calc_ocb + 16'd1;
    // Test the remaining blocks of this group, not whole-layer Cin/Cout.
    // This keeps input and output tails independent of one another.
    wire [4:0] calc_in_group_span = OPT_RUNTIME_TILE_SKIP ?
        ((calc_icb_next < icb_count) ? 5'd16 : 5'd8) : IN_GROUP_SPAN;
    wire [4:0] calc_out_group_span = OPT_RUNTIME_TILE_SKIP ?
        ((calc_ocb_next < ocb_count) ? 5'd16 : 5'd8) : OUT_GROUP_SPAN;

    // Hand-off side coordinate of the *next* group.  The snapshot and the bias
    // re-init inside ST_CALC_HANDOFF must agree on the same next position, so
    // the advance is computed once here rather than in two places.
    // Channel-block step of one compute group: the historical 16-channel pair
    // (2 ABI blocks) for 3x3 / center-tap K1, or 2*Tm channels (Tm/4 blocks)
    // for the M4 K1 eight-group token group.
    wire [15:0] calc_group_step = k1_reuse ? K1_BLK_O : 16'd2;
    wire [15:0] hand_next_ocb   = ((calc_ocb + calc_group_step) < ocb_count) ?
                                  (calc_ocb + calc_group_step) : 16'd0;
    wire [15:0] hand_next_ox    = ((calc_ocb + calc_group_step) < ocb_count) ?
                                  calc_ox : (calc_ox + 16'd1);
    wire [15:0] hand_next_base8 = hand_next_ocb << 3;

    // revision5 PF3: precomputed last-group thresholds for the output-row done
    // flag.  They only depend on snapshot config, so they are registered here
    // instead of being recomputed from cfg_pad/cfg_w/cfg_stride inside the
    // accumulator CE cone (the new A-OOC critical path).  The comparisons are
    // exactly equivalent to (calc_ocb + step >= ocb_count) and
    // (calc_ox + 1 >= wout32); the first calc phase is many cycles after the
    // config snapshot, so the two-cycle register latency is never observed.
    reg [OBC_W-1:0] hand_last_ocb_q;
    reg [15:0]      hand_last_ox_q;
    always @(posedge clk) begin
        if (!rst_n) begin
            hand_last_ocb_q <= {OBC_W{1'b0}};
            hand_last_ox_q  <= 16'd0;
        end else begin
            hand_last_ocb_q <= (cfg_core_ocb_count_q > calc_group_step)
                               ? (cfg_core_ocb_count_q - calc_group_step[OBC_W-1:0])
                               : {OBC_W{1'b0}};
            hand_last_ox_q  <= (wout32[15:0] == 16'd0) ? 16'd0 : (wout32[15:0] - 16'd1);
        end
    end
    // 8-lane output passes of the group that is being handed off: K1 covers
    // 2*Tm lanes (2 for A, 4 for B), every other mode covers the 16-channel
    // ABI group (2 passes).
    wire [2:0]  calc_group_passes = k1_reuse ? (2 * Tm / 8) : 3'd2;
    // Last group of this output ROW: no further channel group can be issued
    // (the group either covers the final channels or is the 8-channel tail of
    // an odd block count) and this is the final pixel column.
    wire        hand_row_done   = (calc_ocb >= hand_last_ocb_q) &&
                                  (calc_ox >= hand_last_ox_q);

    wire [15:0] calc_rd_c_b;

    assign calc_rd_c   = calc_icb << 3;
    assign calc_rd_c_b = calc_rd_c + 16'd8;

    // Four independent reads form one 16x16 issue group:
    //   0 = (icb,ocb), 1 = (icb+1,ocb),
    //   2 = (icb,ocb+1), 3 = (icb+1,ocb+1).
    // The masked ports force both address and data to zero for channel tails.
    assign calc_rd_index[0] = calc_ocb * icb_count + calc_icb;
    assign calc_rd_index[1] = calc_ocb * icb_count + calc_icb_next;
    assign calc_rd_index[2] = calc_ocb_next * icb_count + calc_icb;
    assign calc_rd_index[3] = calc_ocb_next * icb_count + calc_icb_next;

    assign calc_rd_zero[0] = (calc_icb >= icb_count) || (calc_ocb >= ocb_count);
    assign calc_rd_zero[1] = (calc_icb_next >= icb_count) || (calc_ocb >= ocb_count);
    assign calc_rd_zero[2] = (calc_icb >= icb_count) || (calc_ocb_next >= ocb_count);
    assign calc_rd_zero[3] = (calc_icb_next >= icb_count) || (calc_ocb_next >= ocb_count);

    // ABI weight blocks are accepted in four 128-bit beats.  The write side
    // maps them to the same [ocb][icb] index on each independent reference tap
    // face; the calculation side selects one face per tap.
    wire weight_wr_en = (state == ST_W_DATA) &&
                        rd_data_valid && rd_data_ready &&
                        !rd_data_error && (rd_data_tag == TAG_WEIGHT) &&
                        (load_chunk_left >= 16'd16) &&
                        (load_remaining >= 32'd16);
    // M4 K1 reuse: the compact blob is written once into the internal bank
    // layout of contract section 2 -- ABI block (ocb, icb) -> (row, tap g) --
    // without copying the blob or widening the external weight format.
    wire [15:0] k1w_in_group = w_icb / K1_BLK_I;
    wire [15:0] k1w_i_block  = w_icb - (k1w_in_group * K1_BLK_I);
    wire [15:0] k1w_oc_group = w_ocb / K1_BLK_O;
    wire [15:0] k1w_j_block  = w_ocb - (k1w_oc_group * K1_BLK_O);
    wire [15:0] k1w_i        = k1w_i_block / K1_BPI;
    wire [15:0] k1w_a        = k1w_i_block - (k1w_i * K1_BPI);
    wire [15:0] k1w_j        = k1w_j_block / K1_BPO;
    wire [15:0] k1w_b        = k1w_j_block - (k1w_j * K1_BPO);
    wire [15:0] k1w_dense    = (k1w_a * K1_BPO) + k1w_b;
    wire [15:0] k1w_row      = (k1w_in_group * K1_ROW_STRIDE) +
                               (k1w_oc_group * K1_ROWS) + k1w_dense;
    wire [3:0]  k1w_tap        = (k1w_i << 1) + k1w_j;
    wire [15:0] weight_wr_index = k1_reuse ? k1w_row : (w_base + w_icb);
    // Native K1 without reuse stores its single group in the center-tap face
    // (tap 4).  M4 reuse writes group g = 2*i + j to tap g; the unused ninth
    // tap is masked before the PE.
    wire [3:0]  weight_wr_tap = k1_reuse ? k1w_tap :
                                (k1_native_weight ? 4'd4 : w_tap[3:0]);

    wire [3:0] weight_issue_in_cursor = calc_tn_offset;
    wire [3:0] weight_issue_out_cursor = calc_tm_offset;
    wire [1:0] weight_issue_face = {weight_issue_out_cursor[3], weight_issue_in_cursor[3]};
    wire [15:0] weight_bank_rd_index [0:3];
    wire weight_bank_rd_zero [0:3];
    // M4 K1 reuse token weight rows: row = in_group*K1_ROW_STRIDE +
    // oc_group*K1_ROWS + DENSE(face).  Port p reads ABI face p of the group so
    // the calculator's existing face/byte formula sees the token's weights and
    // the ninth tap stays masked.  Single-face builds (A) only use port 0.
    wire [15:0] k1_rd_oc_group = calc_ocb / K1_BLK_O;
    wire [15:0] k1_rd_base     = (calc_icb * K1_ROW_STRIDE) +
                                 (k1_rd_oc_group * K1_ROWS);
    wire [15:0] k1_in_groups   = (icb_count + (K1_BLK_I - 1)) / K1_BLK_I;
    wire [15:0] k1_bank_rd_index [0:3];
    genvar pk1;
    generate
        for (pk1 = 0; pk1 < 4; pk1 = pk1 + 1) begin : GEN_K1_WEIGHT_FACE
            localparam integer K1_DENSE = ((pk1 % 2) * (Tm / 8)) + (pk1 / 2);
            assign k1_bank_rd_index[pk1] = k1_rd_base + K1_DENSE;
        end
    endgenerate
    wire [3:0] pe_calc_in_base = SINGLE_WEIGHT_FACE ? {1'b0,pe_tn_offset_d[2:0]} : pe_tn_offset_d;
    wire [3:0] pe_calc_out_base = SINGLE_WEIGHT_FACE ? {1'b0,pe_tm_offset_d[2:0]} : pe_tm_offset_d;
    wire [63:0] pe_calc_act [0:8], pe_calc_act_b [0:8];
    wire signed [7:0] pe_calc_local_weight [0:8];
    genvar pa;

    // Step C local calculator bases.  The raw ABI cursor (pe_tn/pe_tm_offset_d)
    // is never rewritten: it still drives read addresses and the accumulation
    // metadata; only the base handed to the calculator is normalised.
    wire [3:0] pe_calc_in_base_c =
        TWO_PORT_ADAPTER ? ((Tn == 16) ? 4'd0 : {1'b0, pe_tn_offset_d[2:0]}) : pe_calc_in_base;
    wire [3:0] pe_calc_out_base_c =
        TWO_PORT_ADAPTER ? ((Tm == 16) ? 4'd0 : {1'b0, pe_tm_offset_d[2:0]}) : pe_calc_out_base;

    // Physical read-port addressing for the static path.  Addresses and tail
    // masks are sampled at the existing read-command register boundary
    // (read_cmd_*_q); the RAM read latency and the accumulation metadata are
    // untouched.
    wire [15:0] c_rd_index [0:3];
    wire        c_rd_zero  [0:3];
    genvar pc;
    generate
        for (pc = 0; pc < 4; pc = pc + 1) begin : GEN_C_READ_PORT
            if (pc < FACE_PORTS) begin : GEN_C_PORT
                localparam integer IH = (I_HALVES == 2) ? (pc % I_HALVES) : 0;
                localparam integer OH = (O_HALVES == 2) ? (pc / I_HALVES) : 0;
                wire [15:0] sel_icb = (I_HALVES == 2) ? (calc_icb + IH[15:0])
                                                      : (calc_icb + {15'b0, calc_tn_offset[3]});
                wire [15:0] sel_ocb = (O_HALVES == 2) ? (calc_ocb + OH[15:0])
                                                      : (calc_ocb + {15'b0, calc_tm_offset[3]});
                assign c_rd_index[pc] = sel_ocb * icb_count + sel_icb;
                assign c_rd_zero[pc] = (sel_icb >= icb_count) || (sel_ocb >= ocb_count);
            end else begin : GEN_C_PORT_MASKED
                assign c_rd_index[pc] = 16'd0;
                assign c_rd_zero[pc]  = 1'b1;
            end
        end
    endgenerate

    // Step-C activation adapter: an 8-wide local input window takes the
    // addressed 64-bit half into act_a and zeroes act_b; a 16-wide window
    // forwards both halves unchanged.
    wire [63:0] c_act [0:8], c_act_b [0:8];
    genvar ca;
    generate
        for (ca = 0; ca < 9; ca = ca + 1) begin : GEN_C_ACT
            if (TWO_PORT_ADAPTER) begin : GEN_C_ACT_STATIC
                if (Tn <= 8) begin : ACT_HALF
                    assign c_act[ca]   = pe_tn_offset_d[3] ? pe_act_data_b[ca] : pe_act_data[ca];
                    assign c_act_b[ca] = 64'b0;
                end else begin : ACT_FULL
                    assign c_act[ca]   = pe_act_data[ca];
                    assign c_act_b[ca] = pe_act_data_b[ca];
                end
            end else begin : GEN_C_ACT_BYPASS
                assign c_act[ca]   = pe_calc_act[ca];
                assign c_act_b[ca] = pe_calc_act_b[ca];
            end
        end
    endgenerate

    // Step-C physical-port -> logical ABI face adapter.  Physical port numbering
    // is NOT ABI face numbering: for 4x16/8x16 the second physical port is
    // logical face 2, while for 16x4/16x8 it is logical face 1.
    wire [511:0] c_calc_weight [0:3][0:8];
    genvar cwt;
    generate
        for (cwt = 0; cwt < 9; cwt = cwt + 1) begin : GEN_C_WEIGHT_FACE
            if (TWO_PORT_ADAPTER) begin : GEN_C_WEIGHT_STATIC
                if (I_HALVES == 1 && O_HALVES == 1) begin : BOTH_8
                    assign c_calc_weight[0][cwt] = pe_weight_faces[0][cwt];
                    assign c_calc_weight[1][cwt] = 512'b0;
                    assign c_calc_weight[2][cwt] = 512'b0;
                    assign c_calc_weight[3][cwt] = 512'b0;
                end else if (I_HALVES == 1 && O_HALVES == 2) begin : OUT_16
                    assign c_calc_weight[0][cwt] = pe_weight_faces[0][cwt];
                    assign c_calc_weight[1][cwt] = 512'b0;
                    assign c_calc_weight[2][cwt] = pe_weight_faces[1][cwt];
                    assign c_calc_weight[3][cwt] = 512'b0;
                end else if (I_HALVES == 2 && O_HALVES == 1) begin : IN_16
                    assign c_calc_weight[0][cwt] = pe_weight_faces[0][cwt];
                    assign c_calc_weight[1][cwt] = pe_weight_faces[1][cwt];
                    assign c_calc_weight[2][cwt] = 512'b0;
                    assign c_calc_weight[3][cwt] = 512'b0;
                end else begin : BOTH_16
                    assign c_calc_weight[0][cwt] = pe_weight_faces[0][cwt];
                    assign c_calc_weight[1][cwt] = pe_weight_faces[1][cwt];
                    assign c_calc_weight[2][cwt] = pe_weight_faces[2][cwt];
                    assign c_calc_weight[3][cwt] = pe_weight_faces[3][cwt];
                end
            end else begin : GEN_C_WEIGHT_BYPASS
                assign c_calc_weight[0][cwt] = pe_weight_faces[0][cwt];
                assign c_calc_weight[1][cwt] = pe_weight_faces[1][cwt];
                assign c_calc_weight[2][cwt] = pe_weight_faces[2][cwt];
                assign c_calc_weight[3][cwt] = pe_weight_faces[3][cwt];
            end
        end
    endgenerate

    generate
        for(pa=0;pa<4;pa=pa+1)begin : GEN_WEIGHT_BANK_ADDRESS
            wire [15:0] legacy_rd_index;
            wire        legacy_rd_zero;
            if(TWO_PORT_ADAPTER)begin
                // Step C: two physical read ports for the mixed one-axis-16 tiles.
                assign legacy_rd_index = c_rd_index[pa];
                assign legacy_rd_zero  = c_rd_zero[pa];
            end else if(pa==0)begin
                assign legacy_rd_index = SINGLE_WEIGHT_FACE ? calc_rd_index[weight_issue_face] : calc_rd_index[pa];
                assign legacy_rd_zero  = SINGLE_WEIGHT_FACE ? calc_rd_zero[weight_issue_face] : calc_rd_zero[pa];
            end else begin
                assign legacy_rd_index = calc_rd_index[pa];
                assign legacy_rd_zero  = calc_rd_zero[pa];
            end
            if(K1_REUSE_LEGAL != 0)begin : GEN_K1_REUSE_MUX
                // M4: one token reads all eight groups; per-token tail masking
                // is done by k1_grp_valid inside the calculator, so the port is
                // never zeroed here.
                assign weight_bank_rd_index[pa] = k1_reuse ? k1_bank_rd_index[pa] : legacy_rd_index;
                assign weight_bank_rd_zero[pa]  = k1_reuse ? 1'b0 : legacy_rd_zero;
            end else begin : GEN_LEGACY_ONLY
                assign weight_bank_rd_index[pa] = legacy_rd_index;
                assign weight_bank_rd_zero[pa]  = legacy_rd_zero;
            end
        end
        for(pa=0;pa<9;pa=pa+1)begin : GEN_SINGLE_FACE_PE_ADAPTER
            assign pe_calc_act[pa] = (SINGLE_WEIGHT_FACE && pe_tn_offset_d[3]) ? pe_act_data_b[pa] : pe_act_data[pa];
            assign pe_calc_act_b[pa] = SINGLE_WEIGHT_FACE ? 64'b0 : pe_act_data_b[pa];
            // Local mode represents only logical IC=OC=0, not lane zero of
            // every physical quadrant after normalization.
            assign pe_calc_local_weight[pa] = (SINGLE_WEIGHT_FACE && (pe_tn_offset_d[3] || pe_tm_offset_d[3])) ? 8'sd0 : pe_local_weight[pa];
        end
    endgenerate
    reg read_cmd_valid_q;
    reg [3:0] read_cmd_tn_q,read_cmd_tm_q;
    reg [15:0] weight_read_index_q [0:3];
    reg weight_read_zero_q [0:3];
    integer read_cmd_i;
    always @(posedge clk)begin
      if(!rst_n || core_soft_reset || core_abort)begin
        read_cmd_valid_q<=0;read_cmd_tn_q<=0;read_cmd_tm_q<=0;
        for(read_cmd_i=0;read_cmd_i<4;read_cmd_i=read_cmd_i+1)begin
          weight_read_index_q[read_cmd_i]<=0;weight_read_zero_q[read_cmd_i]<=1;
        end
      end else begin
        read_cmd_valid_q<=pe_addr_valid;
        if(pe_addr_valid)begin
          read_cmd_tn_q<=calc_tn_offset;read_cmd_tm_q<=calc_tm_offset;
          for(read_cmd_i=0;read_cmd_i<4;read_cmd_i=read_cmd_i+1)begin
            weight_read_index_q[read_cmd_i]<=weight_bank_rd_index[read_cmd_i];
            weight_read_zero_q[read_cmd_i]<=weight_bank_rd_zero[read_cmd_i];
          end
        end
      end
    end
    // Break the w_icb/K1 address-decode -> URAM write-enable path.  Weight
    // loading precedes bias/quant/input loading, so this pending write is
    // committed before the first weight read without adding a compute cycle.
    reg weight_wr_valid_q;
    reg [3:0] weight_wr_tap_q;
    reg [15:0] weight_wr_index_q;
    reg [1:0] weight_wr_word_q;
    reg [127:0] weight_wr_data_q;
    always @(posedge clk) begin
        if (!rst_n || core_soft_reset || core_abort) begin
            weight_wr_valid_q <= 1'b0;
        end else begin
            weight_wr_valid_q <= weight_wr_en;
            if (weight_wr_en) begin
                weight_wr_tap_q <= weight_wr_tap;
                weight_wr_index_q <= weight_wr_index;
                weight_wr_word_q <= w_beat4[1:0];
                weight_wr_data_q <= rd_data;
            end
        end
    end
    conv3x3_weight_faces #(
        .MAX_WTILES (MAX_WTILES),
        .READ_PORTS (TWO_PORT_ADAPTER ? FACE_PORTS : (SINGLE_WEIGHT_FACE ? 1 : 4))
    ) u_weight_faces (
        .clk       (clk),
        .rst_n     (rst_n),
        .wr_en     (weight_wr_valid_q),
        .wr_tap    (weight_wr_tap_q),
        .wr_index  (weight_wr_index_q),
        .wr_word   (weight_wr_word_q),
        .wr_data   (weight_wr_data_q),
        .rd_index  (weight_read_index_q),
        .rd_zero   (weight_read_zero_q),
        .weight    (weight_faces)
    );

    // Present the complete verified 3x3 window in one cycle, matching the
    // reference conv33_data_in_ctrl -> conv_calc ownership boundary.
    // Stride multiply by {1,2} is a shift/mux (gtap/3 and gtap%3 are elab
    // constants).  No variable multipliers remain in the window path.
    wire signed [31:0] calc_oy_scaled =
        (stride_eff == 2'd2) ? ($signed({16'b0, calc_oy}) <<< 1) : $signed({16'b0, calc_oy});
    wire signed [31:0] calc_ox_scaled =
        (stride_eff == 2'd2) ? ($signed({16'b0, calc_ox}) <<< 1) : $signed({16'b0, calc_ox});
    // M4 K1 reuse reads the same input pixel through all nine tap slots (the
    // lowered 1x1 geometry is pad=1, so the center tap of the 3x3 window is the
    // source pixel); the eight groups only differ in their channel window.
    wire signed [31:0] k1_center_y =
        calc_oy_scaled + 32'sd1 - $signed({30'b0, cfg_core_pad_q});
    wire signed [31:0] k1_center_x =
        calc_ox_scaled + 32'sd1 - $signed({30'b0, cfg_core_pad_q});
    // K1 channel-window bases of one token: base[s] = ic_base + s*Tn
    // (A: ic_base+0/8/16/24, B: ic_base+0/16/32/48).
    wire [15:0] k1_ic_base     = calc_icb * (4 * Tn);
    wire [15:0] line_rd_c_k1   = k1_ic_base;
    wire [15:0] line_rd_c_b_k1 = k1_ic_base + Tn;
    wire [15:0] line_rd_c_c    = k1_ic_base + 2 * Tn;
    wire [15:0] line_rd_c_d    = k1_ic_base + 3 * Tn;
    wire [15:0] line_rd_c_mux   = k1_reuse ? line_rd_c_k1   : calc_rd_c;
    wire [15:0] line_rd_c_b_mux = k1_reuse ? line_rd_c_b_k1 : calc_rd_c_b;
    generate
        for (gtap = 0; gtap < 9; gtap = gtap + 1) begin : GEN_TAP_WINDOW
            assign tap_rd_y[gtap] =
                k1_reuse ? k1_center_y :
                (calc_oy_scaled
                + (gtap / 3) - $signed({30'b0, cfg_core_pad_q}));
            assign tap_rd_x[gtap] =
                k1_reuse ? k1_center_x :
                (calc_ox_scaled
                + (gtap % 3) - $signed({30'b0, cfg_core_pad_q}));
            assign pe_act_data[gtap]   = line_rd_data[gtap]   ^ {8{8'h80}};
            assign pe_act_data_b[gtap] = line_rd_data_b[gtap] ^ {8{8'h80}};
        end
    endgenerate

    conv3x3_linebuf #(
        .MAX_W         (MAX_W),
        .MAX_CIN       (MAX_CIN),
        .MAX_ROW_BYTES (LINE_MAX_ROW_BYTES),
        .READ_CMD_PIPE (1),
        .K1_TN         (Tn)
    ) u_linebuf (
        .clk        (clk),
        .rst_n      (rst_n),
        .wr_en      (line_wr_en),
        .wr_row     (line_wr_row),
        .wr_beat    (line_wr_beat),
        .wr_data    (line_wr_data),
        .h          (cfg_core_h_q),
        .w          (cfg_core_w_q),
        .cin        (cfg_core_cin_q),
        .cin_alloc  (cin_alloc16),
        .x_zero_point($signed(cfg_core_x_zero_point_q)),
        .rd_y       (tap_rd_y),
        .rd_x       (tap_rd_x),
        .k1_mode    (k1_reuse),
        .rd_c       (line_rd_c_mux),
        .rd_data    (line_rd_data),
        .rd_c_b     (line_rd_c_b_mux),
        .rd_data_b  (line_rd_data_b),
        .rd_c_c     (line_rd_c_c),
        .rd_c_d     (line_rd_c_d)
    );

    wire              out_wr_ready;
    wire              out_pipe_done;
    wire              out_pipe_error;
    wire [15:0]       out_pipe_error_code;

    // The staging controller is flushed on every accepted start and on every
    // soft reset / abort, so no stale page, FIFO beat or B response can leak
    // into a later run.
    wire out_pipe_flush = ((state == ST_IDLE) && core_start) ||
                          core_soft_reset || core_abort;
    wire out_pipe_active = (state != ST_IDLE) && (state != ST_ERROR);

    // Reference-derived output row handoff path: independent w_ping/r_ping row
    // pages plus the 129-bit {last,data} beat FIFO, ported from
    // D:/yolov8/src/cnn/conv/conv_data_out_buf_top.v and
    // D:/yolov8/src/cnn/conv/conv_data_out_buf_ctrl.v.  Row N is drained into
    // the AXI write engine while row N+1 is staged in the opposite page.
    conv3x3_out_ctrl #(
        .MAX_ROW_BEATS (OUT_MAX_ROW_BEATS)
    ) u_out_ctrl (
        .clk                      (clk),
        .rst_n                    (rst_n),
        .flush                    (out_pipe_flush),
        .cfg_out_addr_byte        (cfg_core_out_addr_byte_q),
        .cfg_out_row_stride_bytes (cfg_core_out_row_stride_bytes_q),
        .cfg_hout                 (hout32[15:0]),
        .s_valid                  (out_wr_en),
        .s_ready                  (out_wr_ready),
        .s_beat                   (out_wr_beat),
        .s_byte                   (out_wr_byte),
        .s_data                   (out_wr_data),
        .s_last_group             (out_wr_last),
        .s_zero_high              (out_wr_zero_high),
        .wr_cmd_valid             (wr_cmd_valid),
        .wr_cmd_ready             (wr_cmd_ready),
        .wr_cmd_addr_byte         (wr_cmd_addr_byte),
        .wr_cmd_bytes             (wr_cmd_bytes),
        .wr_cmd_tag               (wr_cmd_tag),
        .wr_data_valid            (wr_data_valid),
        .wr_data_ready            (wr_data_ready),
        .wr_data                  (wr_data),
        .wr_data_last             (wr_data_last),
        .wr_data_tag              (wr_data_tag),
        .wr_rsp_valid             (wr_rsp_valid),
        .wr_rsp_ready             (wr_rsp_ready),
        .wr_rsp_error             (wr_rsp_error),
        .wr_rsp_aborted           (wr_rsp_aborted),
        .wr_rsp_error_code        (wr_rsp_error_code),
        .wr_rsp_tag               (wr_rsp_tag),
        .out_done                 (out_pipe_done),
        .out_error                (out_pipe_error),
        .out_error_code           (out_pipe_error_code)
    );


    // Calculator-local synchronous flush: core soft-reset/abort must clear the
    // balanced tree, DSP valid pipe and weight-sum delay, not only the router.
    wire calc_rst_n = rst_n && !core_soft_reset && !core_abort;

    // M4 K1 reuse: per-token input-block validity.  ABI block a of input
    // subgroup i exists only while (token_in_base + i*Tn/8 + a) < icb_count, so
    // the last token of a Cin that is not a multiple of 4*Tn can carry stale
    // weight rows; the calculator zeroes exactly those weight lanes.
    genvar pv;
    generate
        for (pv = 0; pv < 4; pv = pv + 1) begin : GEN_K1_GRP_VALID
            assign k1_grp_valid[2*pv] =
                (((calc_icb * K1_BLK_I) + (pv * K1_BPI)) < icb_count);
            if (K1_BPI == 2) begin : GEN_K1_GRP_VALID_HI
                assign k1_grp_valid[2*pv+1] =
                    ((((calc_icb * K1_BLK_I) + (pv * K1_BPI)) + 16'd1) < icb_count);
            end else begin : GEN_K1_GRP_VALID_NOHI
                assign k1_grp_valid[2*pv+1] = 1'b0;
            end
        end
    endgenerate
    always @(posedge clk) begin
        if (!rst_n || core_soft_reset || core_abort) begin
            k1_valid_q1 <= 8'b0;
            k1_valid_q2 <= 8'b0;
        end else begin
            if (pe_addr_valid) k1_valid_q1 <= k1_grp_valid;
            k1_valid_q2 <= k1_valid_q1;
        end
    end

    // MAC implementation selector. MAC_IMPL=0 preserves the legacy
    // runtime-base calculator. MAC_IMPL=1 uses a fixed physical 8x8 DSP tile;
    // the small adapter selects the runtime 0/8 tile bank before the fixed
    // lane calculator. This keeps runtime Cin/Cout separate from PE geometry.
    generate
        if (MAC_IMPL == 0) begin : GEN_LEGACY_CALC
            conv3x3_calc #(.Tn(Tn), .Tm(Tm), .WEIGHT_MODE(WEIGHT_MODE)) u_calc (
                .clk             (clk),
                .rst_n           (calc_rst_n),
                .in_valid        (pe_in_valid),
                .in_lane_base    (pe_calc_in_base_c),
                .out_lane_base   (pe_calc_out_base_c),
                .act_data_0      (pe_calc_act[0]),
                .act_data_1      (pe_calc_act[1]),
                .act_data_2      (pe_calc_act[2]),
                .act_data_3      (pe_calc_act[3]),
                .act_data_4      (pe_calc_act[4]),
                .act_data_5      (pe_calc_act[5]),
                .act_data_6      (pe_calc_act[6]),
                .act_data_7      (pe_calc_act[7]),
                .act_data_8      (pe_calc_act[8]),
                .act_data_b_0    (pe_calc_act_b[0]),
                .act_data_b_1    (pe_calc_act_b[1]),
                .act_data_b_2    (pe_calc_act_b[2]),
                .act_data_b_3    (pe_calc_act_b[3]),
                .act_data_b_4    (pe_calc_act_b[4]),
                .act_data_b_5    (pe_calc_act_b[5]),
                .act_data_b_6    (pe_calc_act_b[6]),
                .act_data_b_7    (pe_calc_act_b[7]),
                .act_data_b_8    (pe_calc_act_b[8]),
                .weight          (pe_weight_faces),
                .local_weight    (pe_calc_local_weight),
                .sum_packed      (pe_sum_packed),
                .wsum_packed     (pe_wsum_packed),
                .out_valid       (pe_out_valid)
            );
            assign pe_sum_packed_hi  = 512'b0;
            assign pe_wsum_packed_hi = 256'b0;
        end else if (MAC_IMPL == 1) begin : GEN_DSP_CALC
            wire [63:0] dsp_act_data [0:8];
            wire [63:0] dsp_act_data_b [0:8];
            wire [511:0] dsp_weight [0:3][0:8];
            wire [1:0] dsp_weight_face = {pe_tm_offset_d[3], pe_tn_offset_d[3]};
            genvar gd;
            for (gd = 0; gd < 9; gd = gd + 1) begin : GEN_DSP_ADAPTER
                assign dsp_act_data[gd] = pe_tn_offset_d[3] ? pe_act_data_b[gd] : pe_act_data[gd];
                assign dsp_act_data_b[gd] = 64'b0;
                assign dsp_weight[0][gd] = pe_weight_faces[dsp_weight_face][gd];
                assign dsp_weight[1][gd] = 512'b0;
                assign dsp_weight[2][gd] = 512'b0;
                assign dsp_weight[3][gd] = 512'b0;
            end
            conv3x3_calc_dsp #(
                .Tn              (Tn),
                .Tm              (Tm),
                .WEIGHT_MODE    (WEIGHT_MODE),
                .FIXED_LANES    (1),
                .IN_LANE_BASE   (0),
                .OUT_LANE_BASE  (0)
            ) u_calc_dsp (
                .clk             (clk),
                .rst_n           (calc_rst_n),
                .in_valid        (pe_in_valid),
                .in_lane_base   (4'b0),
                .out_lane_base  (4'b0),
                .act_data_0     (dsp_act_data[0]),
                .act_data_1     (dsp_act_data[1]),
                .act_data_2     (dsp_act_data[2]),
                .act_data_3     (dsp_act_data[3]),
                .act_data_4     (dsp_act_data[4]),
                .act_data_5     (dsp_act_data[5]),
                .act_data_6     (dsp_act_data[6]),
                .act_data_7     (dsp_act_data[7]),
                .act_data_8     (dsp_act_data[8]),
                .act_data_b_0   (dsp_act_data_b[0]),
                .act_data_b_1   (dsp_act_data_b[1]),
                .act_data_b_2   (dsp_act_data_b[2]),
                .act_data_b_3   (dsp_act_data_b[3]),
                .act_data_b_4   (dsp_act_data_b[4]),
                .act_data_b_5   (dsp_act_data_b[5]),
                .act_data_b_6   (dsp_act_data_b[6]),
                .act_data_b_7   (dsp_act_data_b[7]),
                .act_data_b_8   (dsp_act_data_b[8]),
                .weight         (dsp_weight),
                .local_weight   (pe_local_weight),
                .sum_packed     (pe_sum_packed),
                .wsum_packed    (pe_wsum_packed),
                .out_valid      (pe_out_valid)
            );
            assign pe_sum_packed_hi  = 512'b0;
            assign pe_wsum_packed_hi = 256'b0;
        end else begin : GEN_BALANCED_CALC
            conv3x3_calc_balanced #(
                .Tn               (Tn),
                .Tm               (Tm),
                .WEIGHT_MODE      (WEIGHT_MODE),
                .PRODUCT_IMPL     (MAC_IMPL == `NPU_MAC_BALANCED_DSP),
                // Internal static gather: constant-candidate weight/activation
                // supply for all nine aligned 4/8/16 tile combinations.
                .STATIC_DDR_GATHER (STATIC_DDR_GATHER),
                .GATHER_IN_SPAN    (TWO_PORT_ADAPTER ? LOCAL_I_SPAN : GATHER_IN_SPAN_B),
                .GATHER_OUT_SPAN   (TWO_PORT_ADAPTER ? LOCAL_O_SPAN : GATHER_OUT_SPAN_B)
            ) u_calc_balanced (
                .clk             (clk),
                .rst_n           (calc_rst_n),
                .in_valid        (pe_in_valid),
                .in_lane_base    (pe_calc_in_base_c),
                .out_lane_base   (pe_calc_out_base_c),
                .act_data_0      (c_act[0]), .act_data_1      (c_act[1]),
                .act_data_2      (c_act[2]), .act_data_3      (c_act[3]),
                .act_data_4      (c_act[4]), .act_data_5      (c_act[5]),
                .act_data_6      (c_act[6]), .act_data_7      (c_act[7]),
                .act_data_8      (c_act[8]),
                .act_data_b_0    (c_act_b[0]), .act_data_b_1    (c_act_b[1]),
                .act_data_b_2    (c_act_b[2]), .act_data_b_3    (c_act_b[3]),
                .act_data_b_4    (c_act_b[4]), .act_data_b_5    (c_act_b[5]),
                .act_data_b_6    (c_act_b[6]), .act_data_b_7    (c_act_b[7]),
                .act_data_b_8    (c_act_b[8]),
                .weight          (TWO_PORT_ADAPTER ? c_calc_weight : pe_weight_faces),
                .local_weight    (pe_calc_local_weight),
                .sum_packed      (pe_sum_packed),
                .wsum_packed     (pe_wsum_packed),
                .sum_packed_hi   (pe_sum_packed_hi),
                .wsum_packed_hi  (pe_wsum_packed_hi),
                .k1_mode         (k1_reuse),
                .k1_grp_valid    (k1_valid_q2),
                .out_valid       (pe_out_valid)
            );
        end
    endgenerate
    wire [15:0] weight_cmd_bytes =
        (load_remaining > READ_CHUNK_BYTES) ? 16'd4096 : load_remaining[15:0];
    wire [15:0] bias_cmd_bytes =
        (load_remaining > READ_CHUNK_BYTES) ? 16'd4096 : load_remaining[15:0];
    wire [15:0] quant_cmd_bytes =
        (load_remaining > READ_CHUNK_BYTES) ? 16'd4096 : load_remaining[15:0];
    wire [15:0] input_cmd_bytes =
        (input_remaining > READ_CHUNK_BYTES) ? 16'd4096 : input_remaining[15:0];

    wire input_row_end = (input_beat == (in_row_beats - 16'd1));
    wire compute_after_row = input_row_end &&
                             (({16'b0, input_row} + 32'd1) > need_bottom(calc_oy));
    wire [31:0] input_remaining_after = input_remaining - 32'd16;
    wire [15:0] input_chunk_after = input_chunk_left - 16'd16;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state               <= ST_IDLE;
            core_busy           <= 1'b0;
            core_done           <= 1'b0;
            core_error          <= 1'b0;
            core_error_code     <= E_NONE;
            fault_code_r        <= E_NONE;
            rd_cmd_valid        <= 1'b0;
            rd_cmd_addr_byte    <= 32'b0;
            rd_cmd_bytes        <= 16'b0;
            rd_cmd_tag          <= 8'b0;
            rd_data_ready       <= 1'b0;
            load_remaining      <= 32'b0;
            load_offset         <= 32'b0;
            load_chunk_left     <= 16'b0;
            w_ocb               <= 16'b0;
            w_tap               <= 16'b0;
            w_icb               <= 16'b0;
            w_beat4             <= 16'b0;
            local_weight_reg[0] <= LOCAL_W0; local_weight_reg[1] <= LOCAL_W1;
            local_weight_reg[2] <= LOCAL_W2; local_weight_reg[3] <= LOCAL_W3;
            local_weight_reg[4] <= LOCAL_W4; local_weight_reg[5] <= LOCAL_W5;
            local_weight_reg[6] <= LOCAL_W6; local_weight_reg[7] <= LOCAL_W7;
            local_weight_reg[8] <= LOCAL_W8;
            bias_idx            <= 16'b0;
            quant_idx           <= 16'b0;
            input_remaining     <= 32'b0;
            input_offset        <= 32'b0;
            input_row           <= 16'b0;
            input_beat          <= 16'b0;
            input_rows_loaded   <= 16'b0;
            input_chunk_left    <= 16'b0;
            input_cmd_active    <= 1'b0;
            calc_oy             <= 16'b0;
            calc_ox             <= 16'b0;
            calc_ocb            <= 16'b0;
            calc_icb            <= 16'b0;
            calc_tn_offset      <= 5'b0;
            calc_tm_offset      <= 5'b0;
            pe_tn_offset_d      <= 4'b0;
            pe_tm_offset_d      <= 4'b0;
            for (i = 0; i <= PE_TOKEN_LATENCY; i = i + 1) pe_tm_offset_pipe[i] <= 4'b0;

            for (i = 0; i < MAX_COUT; i = i + 1) begin
                bias_mem[i] <= 32'sd0;
                q_m0[i] <= 32'b0;
                q_shift[i] <= 8'b0;
                q_zp[i] <= 8'sd0;
            end
            out_start          <= 1'b0;
            ostg_passes_q      <= 3'd2;
            ostg_ocb_q        <= 16'b0;
            ostg_ox_q         <= 16'b0;
            ostg_oy_q         <= 16'b0;
            ostg_cout_q       <= 16'b0;
            ostg_hout_q       <= 16'b0;
            ostg_pixel_base_q <= 32'b0;
            ostg_x_bias_q     <= 8'b0;
            for (i = 0; i < K1_ACC_LANES; i = i + 1) begin
                calc_acc[i]  <= 32'sd0;
                calc_wsum[i] <= 32'sd0;
                ostg_acc[i]  <= 32'sd0;
                ostg_wsum[i] <= 32'sd0;
            end
            calc_issue_done <= 1'b0;
            pe_inflight     <= 8'b0;
            pe_addr_valid_d <= 1'b0;
            issue_done_d    <= 1'b0;
            w_base          <= 16'b0;
            cfg_check_tmp   <= 16'b0;
            cfg_v_hout <= 32'sd0;
            cfg_v_wout <= 32'sd0;
            cfg_v_in_stride <= 32'b0;
            cfg_v_out_stride <= 32'b0;
            cfg_v_in_bytes <= 32'b0;
            cfg_v_out_bytes <= 32'b0;
            cfg_v_weight_bytes <= 32'b0;
            cfg_v_bias_bytes <= 32'b0;
            cfg_v_quant_bytes <= 32'b0;
            cfg_v_weight_blocks <= 8'b0;
        end else if (core_soft_reset) begin
            state           <= ST_IDLE;
            core_busy       <= 1'b0;
            core_done       <= 1'b0;
            core_error      <= 1'b0;
            core_error_code <= E_NONE;
            fault_code_r    <= E_NONE;
            rd_cmd_valid    <= 1'b0;
            rd_data_ready   <= 1'b0;
            pe_addr_valid_d <= 1'b0;
            issue_done_d    <= 1'b0;
        end else if (core_abort) begin
            state           <= ST_IDLE;
            core_busy       <= 1'b0;
            core_done       <= 1'b0;
            rd_cmd_valid    <= 1'b0;
            rd_data_ready   <= 1'b0;
            pe_addr_valid_d <= 1'b0;
            issue_done_d    <= 1'b0;
        end else begin
            core_done    <= 1'b0;
            core_error   <= 1'b0;
            out_start    <= 1'b0;
            rd_cmd_valid <= 1'b0;
            rd_data_ready<= 1'b0;

            // A fault inside the output pipeline (bad AXI write response, bad
            // tag, ...) must stop the core even while it is still computing a
            // later row; otherwise the page back-pressure would deadlock.
            if (out_pipe_active && out_pipe_error) begin
                fault_code_r <= out_pipe_error_code;
                state        <= ST_ERROR;
            end
            else begin
            case (state)
                ST_IDLE: begin
                    core_busy <= 1'b0;
                    if (core_start) begin
                        cfg_check_tmp = validate_config_header();
                        if (cfg_check_tmp != E_NONE) begin
                            fault_code_r    <= cfg_check_tmp;
                            core_error_code <= cfg_check_tmp;
                            core_error      <= 1'b1;
                            state           <= ST_IDLE;
                        end else begin
                            core_busy       <= 1'b1;
                            core_error      <= 1'b0;
                            core_error_code <= E_NONE;
                            state           <= ST_CFG_GEOM;
                        end
                    end
                end

                ST_CFG_GEOM: begin
                    if ((cfg_calc_hout <= 0) || (cfg_calc_wout <= 0)) begin
                        fault_code_r    <= E_CFG;
                        core_error_code <= E_CFG;
                        core_error      <= 1'b1;
                        core_busy       <= 1'b0;
                        state           <= ST_IDLE;
                    end else begin
                        cfg_v_hout <= cfg_calc_hout;
                        cfg_v_wout <= cfg_calc_wout;
                        state      <= ST_CFG_ROW;
                    end
                end

                ST_CFG_ROW: begin
                    // Register both bounded row-stride products independently
                    // from range/error decode. This cuts the arithmetic cone
                    // off before configuration failure/next-state logic.
                    cfg_v_in_stride  <= cfg_calc_in_stride;
                    cfg_v_out_stride <= cfg_calc_out_stride;
                    state            <= ST_CFG_ROW_CHECK;
                end

                ST_CFG_ROW_CHECK: begin
                    if ((cfg_v_in_stride > 32'd65535) ||
                        (cfg_v_out_stride > 32'd65535)) begin
                        fault_code_r    <= E_UNSUPPORTED;
                        core_error_code <= E_UNSUPPORTED;
                        core_error      <= 1'b1;
                        core_busy       <= 1'b0;
                        state           <= ST_IDLE;
                    end else if (((cfg_v_in_stride >> 4) > LINE_MAX_ROW_BEATS) ||
                                 ((cfg_v_out_stride >> 4) > OUT_MAX_ROW_BEATS)) begin
                        fault_code_r    <= E_UNSUPPORTED;
                        core_error_code <= E_UNSUPPORTED;
                        core_error      <= 1'b1;
                        core_busy       <= 1'b0;
                        state           <= ST_IDLE;
                    end else begin
                        cfg_v_weight_blocks <= cfg_calc_weight_blocks;
                        state               <= ST_CFG_BYTES;
                    end
                end

                ST_CFG_BYTES: begin
                    cfg_v_in_bytes     <= cfg_v_in_stride * {16'b0, cfg_core_h_q};
                    cfg_v_out_bytes    <= cfg_v_out_stride * cfg_v_hout[31:0];
                    // Native K1: one 64-byte center-tap group per block;
                    // legacy: nine groups (512+64 bytes per block).
                    cfg_v_weight_bytes <= k1_native_weight ?
                                          ({24'b0, cfg_v_weight_blocks} << 6) :
                                          (({24'b0, cfg_v_weight_blocks} << 9) +
                                           ({24'b0, cfg_v_weight_blocks} << 6));
                    cfg_v_bias_bytes   <= (({16'b0, cfg_cout_alloc} << 2) + 32'd15) & 32'hfffffff0;
                    cfg_v_quant_bytes  <= (({16'b0, cfg_cout_alloc} << 3) + 32'd15) & 32'hfffffff0;
                    state              <= ST_CFG_CHECK;
                end

                ST_CFG_CHECK: begin
                    if ((cfg_core_in_row_stride_bytes_q != cfg_v_in_stride) ||
                        (cfg_core_out_row_stride_bytes_q != cfg_v_out_stride) ||
                        (cfg_core_in_bytes_q != cfg_v_in_bytes) ||
                        ((WEIGHT_MODE == 0) && (cfg_core_weight_bytes_q != cfg_v_weight_bytes)) ||
                        (cfg_core_bias_bytes_q != cfg_v_bias_bytes) ||
                        (cfg_core_quant_bytes_q != cfg_v_quant_bytes) ||
                        (cfg_core_out_bytes_q != cfg_v_out_bytes)) begin
                        fault_code_r    <= E_STRIDE;
                        core_error_code <= E_STRIDE;
                        core_error      <= 1'b1;
                        core_busy       <= 1'b0;
                        state           <= ST_IDLE;
                    end else if ((cfg_core_in_addr_byte_q == 32'b0) ||
                                 ((WEIGHT_MODE == 0) && (cfg_core_weight_addr_byte_q == 32'b0)) ||
                                 (cfg_core_bias_addr_byte_q == 32'b0) ||
                                 (cfg_core_quant_addr_byte_q == 32'b0) ||
                                 (cfg_core_out_addr_byte_q == 32'b0)) begin
                        fault_code_r    <= E_ADDR_RANGE;
                        core_error_code <= E_ADDR_RANGE;
                        core_error      <= 1'b1;
                        core_busy       <= 1'b0;
                        state           <= ST_IDLE;
                    end else if ((cfg_core_in_addr_byte_q[3:0] != 4'b0) ||
                                 ((WEIGHT_MODE == 0) && (cfg_core_weight_addr_byte_q[3:0] != 4'b0)) ||
                                 (cfg_core_bias_addr_byte_q[3:0] != 4'b0) ||
                                 (cfg_core_quant_addr_byte_q[3:0] != 4'b0) ||
                                 (cfg_core_out_addr_byte_q[3:0] != 4'b0)) begin
                        fault_code_r    <= E_ADDR_ALIGN;
                        core_error_code <= E_ADDR_ALIGN;
                        core_error      <= 1'b1;
                        core_busy       <= 1'b0;
                        state           <= ST_IDLE;
                    end else if (({32'b0, cfg_core_in_addr_byte_q} + {32'b0, cfg_core_in_bytes_q} > 64'h1_0000_0000) ||
                                 ((WEIGHT_MODE == 0) && ({32'b0, cfg_core_weight_addr_byte_q} + {32'b0, cfg_core_weight_bytes_q} > 64'h1_0000_0000)) ||
                                 ({32'b0, cfg_core_bias_addr_byte_q} + {32'b0, cfg_core_bias_bytes_q} > 64'h1_0000_0000) ||
                                 ({32'b0, cfg_core_quant_addr_byte_q} + {32'b0, cfg_core_quant_bytes_q} > 64'h1_0000_0000) ||
                                 ({32'b0, cfg_core_out_addr_byte_q} + {32'b0, cfg_core_out_bytes_q} > 64'h1_0000_0000)) begin
                        fault_code_r    <= E_ADDR_RANGE;
                        core_error_code <= E_ADDR_RANGE;
                        core_error      <= 1'b1;
                        core_busy       <= 1'b0;
                        state           <= ST_IDLE;
                    end else begin
                        if (WEIGHT_MODE == 0) begin
                            load_remaining <= cfg_core_weight_bytes_q;
                            load_offset    <= 32'b0;
                            w_ocb          <= 16'b0;
                            w_base         <= 16'b0;
                            w_tap          <= 16'b0;
                            w_icb          <= 16'b0;
                            w_beat4        <= 16'b0;
                            state          <= ST_W_CMD;
                        end else begin
                            load_remaining <= cfg_core_bias_bytes_q;
                            load_offset    <= 32'b0;
                            bias_idx       <= 16'b0;
                            state          <= ST_B_CMD;
                        end
                    end
                end

                ST_W_CMD: begin
                    rd_cmd_valid     <= 1'b1;
                    rd_cmd_addr_byte <= cfg_core_weight_addr_byte_q + load_offset;
                    rd_cmd_bytes     <= weight_cmd_bytes;
                    rd_cmd_tag       <= TAG_WEIGHT;
                    if (rd_cmd_valid && rd_cmd_ready) begin
                        load_chunk_left <= weight_cmd_bytes;
                        state           <= ST_W_DATA;
                    end
                end

                ST_W_DATA: begin
                    rd_data_ready <= 1'b1;
                    if (rd_data_valid && rd_data_ready) begin
                        if (rd_data_error || (rd_data_tag != TAG_WEIGHT)) begin
                            fault_code_r <= rd_data_error ? rd_data_error_code : E_AXI_PROTOCOL;
                            state        <= ST_ERROR;
                        end else if ((load_chunk_left < 16'd16) || (load_remaining < 32'd16)) begin
                            fault_code_r <= E_AXI_PROTOCOL;
                            state        <= ST_ERROR;
                        end else begin
                            load_offset     <= load_offset + 32'd16;
                            load_remaining  <= load_remaining - 32'd16;
                            load_chunk_left <= load_chunk_left - 16'd16;
                            if (w_beat4 == 16'd3) begin
                                w_beat4 <= 16'b0;
                                if ((w_icb + 16'd1) == icb_count) begin
                                    w_icb <= 16'b0;
                                    // K1 has a single (center) tap group: advance
                                    // straight to the next output-channel block.
                                    if (k1_native_weight || (w_tap == 16'd8)) begin
                                        w_tap <= 16'b0;
                                        w_ocb <= w_ocb + 16'd1;
                                        w_base <= w_base + icb_count;
                                    end else begin
                                        w_tap <= w_tap + 16'd1;
                                    end
                                end else begin
                                    w_icb <= w_icb + 16'd1;
                                end
                            end else begin
                                w_beat4 <= w_beat4 + 16'd1;
                            end

                            if (rd_data_last) begin
                                if (load_chunk_left != 16'd16) begin
                                    fault_code_r <= E_AXI_PROTOCOL;
                                    state        <= ST_ERROR;
                                end else if (load_remaining > 32'd16) begin
                                    state <= ST_W_CMD;
                                end else begin
                                    load_remaining <= cfg_core_bias_bytes_q;
                                    load_offset    <= 32'b0;
                                    bias_idx       <= 16'b0;
                                    state          <= ST_B_CMD;
                                end
                            end else if (load_chunk_left == 16'd16) begin
                                fault_code_r <= E_AXI_PROTOCOL;
                                state        <= ST_ERROR;
                            end
                        end
                    end
                end

                ST_B_CMD: begin
                    rd_cmd_valid     <= 1'b1;
                    rd_cmd_addr_byte <= cfg_core_bias_addr_byte_q + load_offset;
                    rd_cmd_bytes     <= bias_cmd_bytes;
                    rd_cmd_tag       <= TAG_BIAS;
                    if (rd_cmd_valid && rd_cmd_ready) begin
                        load_chunk_left <= bias_cmd_bytes;
                        state           <= ST_B_DATA;
                    end
                end

                ST_B_DATA: begin
                    rd_data_ready <= 1'b1;
                    if (rd_data_valid && rd_data_ready) begin
                        if (rd_data_error || (rd_data_tag != TAG_BIAS)) begin
                            fault_code_r <= rd_data_error ? rd_data_error_code : E_AXI_PROTOCOL;
                            state        <= ST_ERROR;
                        end else if ((load_chunk_left < 16'd16) || (load_remaining < 32'd16)) begin
                            fault_code_r <= E_AXI_PROTOCOL;
                            state        <= ST_ERROR;
                        end else begin
                            for (i = 0; i < 4; i = i + 1) begin
                                if ((bias_idx + i) < (MAX_COUT)) begin
                                    bias_mem[bias_idx + i] <= rd_data[i*32 +: 32];
                                end
                            end
                            bias_idx        <= bias_idx + 16'd4;
                            load_offset     <= load_offset + 32'd16;
                            load_remaining  <= load_remaining - 32'd16;
                            load_chunk_left <= load_chunk_left - 16'd16;
                            if (rd_data_last) begin
                                if (load_chunk_left != 16'd16) begin
                                    fault_code_r <= E_AXI_PROTOCOL;
                                    state        <= ST_ERROR;
                                end else if (load_remaining > 32'd16) begin
                                    state <= ST_B_CMD;
                                end else begin
                                    load_remaining <= cfg_core_quant_bytes_q;
                                    load_offset    <= 32'b0;
                                    quant_idx      <= 16'b0;
                                    state          <= ST_Q_CMD;
                                end
                            end else if (load_chunk_left == 16'd16) begin
                                fault_code_r <= E_AXI_PROTOCOL;
                                state        <= ST_ERROR;
                            end
                        end
                    end
                end

                ST_Q_CMD: begin
                    rd_cmd_valid     <= 1'b1;
                    rd_cmd_addr_byte <= cfg_core_quant_addr_byte_q + load_offset;
                    rd_cmd_bytes     <= quant_cmd_bytes;
                    rd_cmd_tag       <= TAG_QUANT;
                    if (rd_cmd_valid && rd_cmd_ready) begin
                        load_chunk_left <= quant_cmd_bytes;
                        state           <= ST_Q_DATA;
                    end
                end

                ST_Q_DATA: begin
                    rd_data_ready <= 1'b1;
                    if (rd_data_valid && rd_data_ready) begin
                        if (rd_data_error || (rd_data_tag != TAG_QUANT)) begin
                            fault_code_r <= rd_data_error ? rd_data_error_code : E_AXI_PROTOCOL;
                            state        <= ST_ERROR;
                        end else if ((load_chunk_left < 16'd16) || (load_remaining < 32'd16)) begin
                            fault_code_r <= E_AXI_PROTOCOL;
                            state        <= ST_ERROR;
                        end else begin
                            q_m0[quant_idx]       <= rd_data[31:0];
                            q_shift[quant_idx]    <= rd_data[39:32];
                            q_zp[quant_idx]       <= rd_data[47:40];
                            q_m0[quant_idx + 1]   <= rd_data[95:64];
                            q_shift[quant_idx + 1]<= rd_data[103:96];
                            q_zp[quant_idx + 1]   <= rd_data[111:104];
                            quant_idx       <= quant_idx + 16'd2;
                            load_offset     <= load_offset + 32'd16;
                            load_remaining  <= load_remaining - 32'd16;
                            load_chunk_left <= load_chunk_left - 16'd16;
                            if (rd_data_last) begin
                                if (load_chunk_left != 16'd16) begin
                                    fault_code_r <= E_AXI_PROTOCOL;
                                    state        <= ST_ERROR;
                                end else if (load_remaining > 32'd16) begin
                                    state <= ST_Q_CMD;
                                end else begin
                                    input_remaining   <= cfg_core_in_bytes_q;
                                    input_offset      <= 32'b0;
                                    input_row         <= 16'b0;
                                    input_beat        <= 16'b0;
                                    input_rows_loaded <= 16'b0;
                                    input_chunk_left  <= 16'b0;
                                    input_cmd_active  <= 1'b0;
                                    calc_oy           <= 16'b0;
                                    calc_ox           <= 16'b0;
                                    calc_ocb          <= 16'b0;
                                    calc_icb          <= 16'b0;
                                    state             <= ST_ROW_PREP;
                                end
                            end else if (load_chunk_left == 16'd16) begin
                                fault_code_r <= E_AXI_PROTOCOL;
                                state        <= ST_ERROR;
                            end
                        end
                    end
                end

                ST_IN_CMD: begin
                    rd_cmd_valid     <= 1'b1;
                    rd_cmd_addr_byte <= cfg_core_in_addr_byte_q + input_offset;
                    rd_cmd_bytes     <= input_cmd_bytes;
                    rd_cmd_tag       <= TAG_INPUT;
                    if (rd_cmd_valid && rd_cmd_ready) begin
                        input_chunk_left <= input_cmd_bytes;
                        input_cmd_active <= 1'b1;
                        state            <= ST_IN_DATA;
                    end
                end

                ST_IN_DATA: begin
                    rd_data_ready <= 1'b1;
                    if (rd_data_valid && rd_data_ready) begin
                        if (rd_data_error || (rd_data_tag != TAG_INPUT)) begin
                            fault_code_r <= rd_data_error ? rd_data_error_code : E_AXI_PROTOCOL;
                            state        <= ST_ERROR;
                        end else if ((input_chunk_left < 16'd16) || (input_remaining < 32'd16)) begin
                            fault_code_r <= E_AXI_PROTOCOL;
                            state        <= ST_ERROR;
                        end else begin
                            input_offset    <= input_offset + 32'd16;
                            input_remaining <= input_remaining - 32'd16;
                            input_chunk_left<= input_chunk_left - 16'd16;
                            if (input_row_end) begin
                                input_row         <= input_row + 16'd1;
                                input_beat        <= 16'b0;
                                input_rows_loaded <= input_row + 16'd1;
                            end else begin
                                input_beat <= input_beat + 16'd1;
                            end

                            if (rd_data_last) begin
                                if (input_chunk_left != 16'd16) begin
                                    fault_code_r <= E_AXI_PROTOCOL;
                                    state        <= ST_ERROR;
                                end else begin
                                    input_cmd_active <= 1'b0;
                                    if (compute_after_row) begin
                                        rd_data_ready <= 1'b0;
                                        state <= ST_CALC_INIT;
                                    end else if (input_remaining > 32'd16) begin
                                        state <= ST_IN_CMD;
                                    end else if (compute_after_row == 1'b0) begin
                                        fault_code_r <= E_CORE_INTERNAL;
                                        state        <= ST_ERROR;
                                    end
                                end
                            end else if (input_chunk_left == 16'd16) begin
                                fault_code_r <= E_AXI_PROTOCOL;
                                state        <= ST_ERROR;
                            end else if (compute_after_row) begin
                                rd_data_ready <= 1'b0;
                                state <= ST_CALC_INIT;
                            end
                        end
                    end
                end

                ST_ROW_PREP: begin
                    // Wait until the three-row activation window that this
                    // output row needs (need_bottom) has been loaded.  Row
                    // staging happens in conv3x3_out_ctrl, so no page clearing
                    // is required here: every physical row byte is written by
                    // the producer, including the zeroed half-beat tail.
                    if ((need_bottom(calc_oy) < 0) ||
                        ($signed({16'b0, input_rows_loaded}) > need_bottom(calc_oy))) begin
                        calc_ox  <= 16'b0;
                        calc_ocb <= 16'b0;
                        calc_icb <= 16'b0;
                        state    <= ST_CALC_INIT;
                    end else if (input_remaining == 32'b0) begin
                        fault_code_r <= E_CORE_INTERNAL;
                        state        <= ST_ERROR;
                    end else if ((input_chunk_left != 16'b0) && input_cmd_active) begin
                        state <= ST_IN_DATA;
                    end else begin
                        state <= ST_IN_CMD;
                    end
                end

                ST_CALC_INIT: begin
                    // A 16x16 issue group spans two consecutive 8-channel ABI
                    // blocks in both input and output channel directions; an M4
                    // K1 reuse group spans 2*Tm lanes (16 for A, 32 for B).
                    for (i = 0; i < K1_ACC_LANES; i = i + 1) begin
                        lane_addr = ocb_base8 + i;
                        if (lane_addr < cfg_core_cout_q) begin
                            calc_acc[i] <= bias_mem[lane_addr];
                        end else begin
                            calc_acc[i] <= 32'sd0;
                        end
                        calc_wsum[i] <= 32'sd0;
                    end
                    calc_icb         <= 16'b0;
                    calc_tn_offset    <= 5'b0;
                    calc_tm_offset    <= 5'b0;
                    calc_issue_done  <= 1'b0;
                    pe_inflight      <= 8'b0;
                    pe_addr_valid_d  <= 1'b0;
                    issue_done_d     <= 1'b0;
                    state            <= ST_CALC_TAP;
                end

                ST_CALC_TAP: begin
                    if (k1_reuse) begin
                        // M4 K1 eight-group reuse: one token per cycle covers all
                        // eight product groups, so the token cursor is calc_icb
                        // and no Tn/Tm micro-tile cursors are used.
                        if (!calc_issue_done) begin
                            if ((calc_icb + 16'd1) >= k1_in_groups) begin
                                calc_icb        <= 16'b0;
                                calc_issue_done <= 1'b1;
                            end else begin
                                calc_icb <= calc_icb + 16'd1;
                            end
                        end
                        if (pe_out_valid) begin
                            for (i = 0; i < K1_ACC_LANES; i = i + 1) begin
                                if (i < 16) begin
                                    calc_acc[i]  <= calc_acc[i]  +
                                        $signed(pe_sum_packed[i*32 +: 32]);
                                    calc_wsum[i] <= calc_wsum[i] +
                                        $signed(pe_wsum_packed[i*16 +: 16]);
                                end else begin
                                    calc_acc[i]  <= calc_acc[i]  +
                                        $signed(pe_sum_packed_hi[(i-16)*32 +: 32]);
                                    calc_wsum[i] <= calc_wsum[i] +
                                        $signed(pe_wsum_packed_hi[(i-16)*16 +: 16]);
                                end
                            end
                        end
                    end else begin
                    // Micro-tile a 16x16 ABI group in Tn x Tm physical PE issues.
                    if (!calc_issue_done) begin
                        if ((calc_tm_offset + Tm) >= calc_out_group_span) begin
                            calc_tm_offset <= 0;
                            if ((calc_tn_offset + Tn) >= calc_in_group_span) begin
                                calc_tn_offset <= 0;
                                if ((calc_icb + 16'd2) >= icb_count) begin calc_icb <= 0; calc_issue_done <= 1'b1; end
                                else calc_icb <= calc_icb + 16'd2;
                            end else calc_tn_offset <= calc_tn_offset + Tn;
                        end else calc_tm_offset <= calc_tm_offset + Tm;
                    end
                    if (pe_out_valid) for (i = 0; i < Tm; i = i + 1)
                        if ((pe_tm_offset_pipe[PE_TOKEN_LATENCY] + i) < 16) begin
                            calc_acc[pe_tm_offset_pipe[PE_TOKEN_LATENCY] + i] <= calc_acc[pe_tm_offset_pipe[PE_TOKEN_LATENCY] + i] + $signed(pe_sum_packed[i*32 +: 32]);
                            calc_wsum[pe_tm_offset_pipe[PE_TOKEN_LATENCY] + i] <= calc_wsum[pe_tm_offset_pipe[PE_TOKEN_LATENCY] + i] + $signed(pe_wsum_packed[i*16 +: 16]);
                        end
                    end
                    pe_addr_valid_d <= read_cmd_valid_q;
                    pe_tn_offset_d <= read_cmd_tn_q; pe_tm_offset_d <= read_cmd_tm_q;
                    pe_tm_offset_pipe[0] <= pe_tm_offset_d;
                    for (i = 1; i <= PE_TOKEN_LATENCY; i = i + 1) pe_tm_offset_pipe[i] <= pe_tm_offset_pipe[i-1];
                    issue_done_d <= calc_issue_done; pe_inflight <= pe_inflight_next;
                    // One cycle after the last partial sum commits the bank is
                    // complete; the hand-off state takes the snapshot then.
                    if (calc_issue_done && !read_cmd_valid_q && !pe_addr_valid_d && (pe_inflight_next == 0))
                        state <= ST_CALC_HANDOFF;
                end

                ST_CALC_HANDOFF: begin
                    // M3 output overlap: the accumulator bank is complete and
                    // the PE pipe has drained.  Snapshot the finished group for
                    // the parallel requant/output pipeline and re-init the bank
                    // for the next group in the same cycle, so the epilogue no
                    // longer serialises with issue.  Only this stage stalls when
                    // the output pipeline still owns the snapshot registers:
                    // calc_acc is never overwritten before its result has been
                    // taken, and no other state is forced to wait.
                    if (!out_busy) begin
                        for (i = 0; i < K1_ACC_LANES; i = i + 1) begin
                            ostg_acc[i]  <= calc_acc[i];
                            ostg_wsum[i] <= calc_wsum[i];
                        end
                        ostg_passes_q     <= calc_group_passes;
                        ostg_ocb_q        <= calc_ocb;
                        ostg_ox_q         <= calc_ox;
                        ostg_oy_q         <= calc_oy;
                        ostg_cout_q       <= cfg_core_cout_q;
                        ostg_hout_q       <= hout32[15:0];
                        ostg_pixel_base_q <= calc_ox * cout_alloc16;
                        ostg_x_bias_q     <= pe_x_bias_byte;
                        out_start         <= 1'b1;
                        calc_icb          <= 16'b0;
                        calc_tn_offset    <= 5'b0;
                        calc_tm_offset    <= 5'b0;
                        calc_issue_done   <= 1'b0;
                        pe_addr_valid_d   <= 1'b0;
                        issue_done_d      <= 1'b0;
                        if (hand_row_done) begin
                            calc_ocb <= 16'b0;
                            calc_ox  <= 16'b0;
                            if ((calc_oy + 16'd1) < hout32[15:0]) begin
                                calc_oy <= calc_oy + 16'd1;
                                state   <= ST_ROW_PREP;
                            end else begin
                                state <= ST_OUT_WAIT;
                            end
                        end else begin
                            calc_ocb <= hand_next_ocb[OBC_W-1:0];
                            calc_ox  <= hand_next_ox;
                            // Bias re-init for the next group uses the same
                            // cycle as the hand-off, so a steady-state group is
                            // exactly issue + drain + this hand-off cycle.
                            for (i = 0; i < K1_ACC_LANES; i = i + 1) begin
                                if ((hand_next_base8 + i) < cfg_core_cout_q)
                                    calc_acc[i] <= bias_mem[hand_next_base8 + i];
                                else
                                    calc_acc[i] <= 32'sd0;
                                calc_wsum[i] <= 32'sd0;
                            end
                            state <= ST_CALC_TAP;
                        end
                    end
                end

                ST_OUT_WAIT: begin
                    // The last group has been handed to the parallel output
                    // pipeline; report done only after that pipeline is idle and
                    // the write controller has seen every AXI write response.
                    if (out_pipe_error) begin
                        fault_code_r <= out_pipe_error_code;
                        state        <= ST_ERROR;
                    end else if (out_pipe_done && (out_state == O_IDLE)) begin
                        core_busy <= 1'b0;
                        core_done <= 1'b1;
                        state     <= ST_IDLE;
                    end
                end


                ST_ERROR: begin
                    core_busy       <= 1'b0;
                    core_error      <= 1'b1;
                    core_error_code <= fault_code_r;
                    state           <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
            end
        end
    end

    // -------------------------------------------------------------------------
    // M3 overlapped requant / output pipeline.
    //
    // The compute FSM hands one completed 16-channel group per out_start strobe
    // (the ostg_* snapshot) and keeps issuing the next group.  This pipeline
    // walks that snapshot exactly like the former in-FSM epilogue: two 8-lane
    // passes (low half, then the high half when it is a real group), each pass
    // staged as select -> correct -> subtract -> multiply -> abs/round -> add ->
    // shift -> saturate -> push.  The arithmetic stays split across register
    // stages, so no combinational multiply/abs/shift/saturate cone is created.
    //
    // Back-pressure from the row-buffer page only stalls O_PUSH; that keeps
    // out_busy high only for as long as the write path is actually blocked, and
    // out_busy is the only signal the compute hand-off stage waits on.
    // -------------------------------------------------------------------------
    wire out_fsm_kill = core_soft_reset || core_abort || (state == ST_ERROR) ||
                        ((state == ST_IDLE) && core_start);
    // Per-pass address/flag decode from the handed-off group snapshot.
    wire [15:0] o_chan_base = (ostg_ocb_q + {14'b0, o_pass_q}) << 3;
    wire [31:0] o_off       = ostg_pixel_base_q + {16'b0, o_chan_base};
    wire        o_zero_high = ((ostg_ocb_q + {14'b0, o_pass_q} + 16'd1) >= ocb_count);
    // last is the final group of this output ROW, not EOF.
    wire        o_last      = (((ostg_ocb_q + {14'b0, o_pass_q} + 16'd1) >= ocb_count) &&
                               ((ostg_ox_q + 16'd1) >= wout32[15:0]));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_state           <= O_IDLE;
            o_pass_q            <= 2'b0;
            out_off_calc      <= 32'b0;
            out_wr_en         <= 1'b0;
            out_wr_beat       <= 16'b0;
            out_wr_byte       <= 4'b0;
            out_wr_data       <= 64'b0;
            out_wr_last       <= 1'b0;
            out_wr_zero_high  <= 1'b0;
            for (oi = 0; oi < 8; oi = oi + 1) begin
                out_sel_acc_q[oi]  <= 32'sd0;
                out_sel_wsum_q[oi] <= 32'sd0;
                out_corr_product_q[oi] <= 64'sd0;
                wr_acc[oi]   <= 32'sd0;
                wr_m0[oi]    <= 32'b0;
                wr_shift[oi] <= 8'b0;
                wr_zp[oi]    <= 8'sd0;
                rq_product[oi]  <= 64'sd0;
                rq_abs[oi]      <= 64'd0;
                rq_round_bias[oi] <= 64'd0;
                rq_round_sum[oi] <= 64'd0;
                rq_magnitude[oi] <= 64'd0;
                rq_negative[oi] <= 1'b0;
            end
        end else if (out_fsm_kill) begin
            // Soft reset / abort / output-path fault: drop the epilogue state and
            // the push handshake, exactly like the former in-FSM clears.
            out_state        <= O_IDLE;
            out_wr_en        <= 1'b0;
            out_wr_last      <= 1'b0;
            out_wr_zero_high <= 1'b0;
        end else begin
            out_wr_en <= 1'b0;
            case (out_state)
                O_IDLE: begin
                    if (out_start) begin
                        o_pass_q  <= 2'b0;
                        out_state <= O_SELECT;
                    end
                end

                O_SELECT: begin
                    // Stage A (was ST_OUT_SELECT): constant lane indices plus a
                    // two-way half mux; no correction multiply/subtract here.
                    out_off_calc <= o_off;
                    for (li = 0; li < 8; li = li + 1) begin
                        o_lane_addr = {16'b0, o_chan_base} + li;
                        o_acc_lane  = ({3'b0, o_pass_q} << 3) + li;
                        if (o_lane_addr < ostg_cout_q) begin
                            out_sel_acc_q[li]  <= ostg_acc[o_acc_lane];
                            out_sel_wsum_q[li] <= ostg_wsum[o_acc_lane];
                            wr_m0[li]    <= q_m0[o_lane_addr];
                            wr_shift[li] <= q_shift[o_lane_addr];
                            wr_zp[li]    <= q_zp[o_lane_addr];
                        end else begin
                            // Padded channels retain the exact zero contract.
                            out_sel_acc_q[li]  <= 32'sd0;
                            out_sel_wsum_q[li] <= 32'sd0;
                            wr_m0[li]    <= 32'b0;
                            wr_shift[li] <= 8'b0;
                            wr_zp[li]    <= 8'sd0;
                        end
                    end
                    out_state <= O_CORRECT;
                end

                O_CORRECT: begin
                    // Stage B: preserve the original signed 64-bit product.
                    for (li = 0; li < 8; li = li + 1)
                        out_corr_product_q[li] <= $signed(out_sel_wsum_q[li]) *
                                                 $signed({1'b0, ostg_x_bias_q});
                    out_state <= O_ACC;
                end

                O_ACC: begin
                    // Stage C: subtract only, then the original low-32-bit
                    // accumulator wrap. Requantization below is unchanged.
                    for (li = 0; li < 8; li = li + 1) begin
                        acc_corr_calc = $signed(out_sel_acc_q[li]) -
                                        $signed(out_corr_product_q[li]);
                        wr_acc[li] <= acc_corr_calc[31:0];
                    end
                    out_state <= O_RQ;
                end

                O_RQ: begin
                    for (li = 0; li < 8; li = li + 1)
                        rq_product[li] <= $signed(wr_acc[li]) *
                                          $signed({1'b0, wr_m0[li]});
                    out_state <= O_RQ_ABS;
                end

                O_RQ_ABS: begin
                    for (li = 0; li < 8; li = li + 1) begin
                        rq_negative[li] <= rq_product[li][63];
                        if (rq_product[li][63])
                            rq_abs[li] <= (~rq_product[li]) + 64'd1;
                        else
                            rq_abs[li] <= rq_product[li];
                        if (wr_shift[li] == 8'd0)
                            rq_round_bias[li] <= 64'd0;
                        else
                            rq_round_bias[li] <= 64'd1 << (wr_shift[li] - 8'd1);
                    end
                    out_state <= O_RQ_ADD;
                end

                O_RQ_ADD: begin
                    for (li = 0; li < 8; li = li + 1)
                        rq_round_sum[li] <= rq_abs[li] + rq_round_bias[li];
                    out_state <= O_RQ_SHIFT;
                end

                O_RQ_SHIFT: begin
                    for (li = 0; li < 8; li = li + 1)
                        rq_magnitude[li] <= rq_round_sum[li] >> wr_shift[li];
                    out_state <= O_RQ_SAT;
                end

                O_RQ_SAT: begin
                    for (li = 0; li < 8; li = li + 1) begin
                        if (|rq_magnitude[li][63:8]) begin
                            q_byte_calc = rq_negative[li] ? 8'h80 : 8'h7f;
                        end else begin
                            if (rq_negative[li])
                                rq_y_calc = $signed({{2{wr_zp[li][7]}}, wr_zp[li]}) -
                                            $signed({2'b00, rq_magnitude[li][7:0]});
                            else
                                rq_y_calc = $signed({{2{wr_zp[li][7]}}, wr_zp[li]}) +
                                            $signed({2'b00, rq_magnitude[li][7:0]});
                            if (rq_y_calc > 10'sd127)
                                q_byte_calc = 8'h7f;
                            else if (rq_y_calc < -10'sd128)
                                q_byte_calc = 8'h80;
                            else
                                q_byte_calc = rq_y_calc[7:0];
                        end
                        // affine-int8-v1 fused activation (act_cfg[0]): clamp the
                        // saturated result at the output zero point zs_out.  This
                        // is exactly the contract's relu threshold and adds no
                        // second pass; legacy builds never set act_cfg and are
                        // byte-identical.
                        if (cfg_core_act_cfg_q[0] &&
                            ($signed(q_byte_calc) < $signed(wr_zp[li])))
                            q_byte_calc = wr_zp[li];
                        out_wr_data[li*8 +: 8] <= q_byte_calc;
                    end
                    out_wr_beat      <= out_off_calc >> 4;
                    out_wr_byte      <= out_off_calc[3:0];
                    out_wr_zero_high <= (out_off_calc[3:0] == 4'd0) && o_zero_high;
                    out_wr_last      <= o_last;
                    out_wr_en        <= 1'b1;
                    out_state        <= O_PUSH;
                end

                O_PUSH: begin
                    // Hold each 8-byte lane group until the output controller
                    // accepts it; the second pass covers the high 8 channels
                    // only when they are a real group of this command.
                    if (out_wr_en && out_wr_ready) begin
                        if ((({1'b0, o_pass_q} + 3'd1) < ostg_passes_q) &&
                            ((ostg_ocb_q + {14'b0, o_pass_q} + 16'd1) < ocb_count)) begin
                            o_pass_q  <= o_pass_q + 2'd1;
                            out_state <= O_SELECT;
                        end else begin
                            out_state <= O_IDLE;
                        end
                    end else begin
                        // A ready/valid group is consumed on exactly one edge.
                        // Keep valid high only while the write path is
                        // back-pressuring; nothing else in the core stalls.
                        out_wr_en <= 1'b1;
                    end
                end

                default: out_state <= O_IDLE;
            endcase
        end
    end

endmodule






