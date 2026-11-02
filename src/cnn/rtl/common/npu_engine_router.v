// -----------------------------------------------------------------------------
// npu_engine_router.v
//
// Mixed-engine operator router for project_3.
//
// It owns the reusable, model-agnostic compute cores
//   * conv3x3_core      (opcode 1)
//   * maxpool2x2_core   (opcode 2)
//   * eltwise_core      (opcode 0x10 Relu / 0x11 Add)
//   * copy_transform_core / clip_core / global_avg_pool_core /
//     avgpool2x2_core / int8_mul_core  (opcodes 0x12..0x18)
//   * resize_nearest2x_core (opcode 0x19)
//   * maxpool5x5_core       (opcode 0x1A)
// and time-multiplexes them onto the single normalized read/write memory service
// exported by the unchanged ABI 1.0 op_shell.  Selection is explicit (op_sel),
// never inferred from weight_bytes: a 1x1 or zero-weight conv must never be
// mistaken for a pool.
//
// Per-operator configuration is broadcast from the sequencer (ROM record).
// cfg_stride/cfg_pad/x_zero_point are decoded from CONV_CFG/QUANT_CFG exactly
// the way op_shell decodes its own CSR snapshot, so both cores see the same
// ABI encoding they were verified with.
//
// This file adds no model-specific logic: it is a static opcode mux + config fanout.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module npu_engine_router #(
    parameter integer MAX_CIN       = 64,
    parameter integer MAX_COUT      = 64,
    parameter integer MAX_W         = 256,
    parameter integer DENSE_LINE_ROW_BYTES   = 0,
    parameter integer DENSE_OUT_ROW_BYTES    = 0,
    parameter integer DENSE_MAX_WEIGHT_TILES = 0,
    parameter integer DENSE_WEIGHT_READ_PORTS = 0,
    parameter integer Tn            = 16,
    parameter integer Tm            = 16,
    parameter integer WEIGHT_MODE   = 0,
    parameter integer MAC_IMPL      = 0,
    parameter signed [7:0] LOCAL_W0 = 0, parameter signed [7:0] LOCAL_W1 = 0,
    parameter signed [7:0] LOCAL_W2 = 0, parameter signed [7:0] LOCAL_W3 = 0,
    parameter signed [7:0] LOCAL_W4 = 0, parameter signed [7:0] LOCAL_W5 = 0,
    parameter signed [7:0] LOCAL_W6 = 0, parameter signed [7:0] LOCAL_W7 = 0,
    parameter signed [7:0] LOCAL_W8 = 0,
    parameter integer MAX_CH        = 64,
    // v0.2 spatial capacity.  MAX_W/MAX_H are the allocated feature bounds of
    // the *used* engines; a per-engine override of 0 inherits them, which keeps
    // a legacy build (MAX_H=0, every override 0) byte-identical.
    parameter integer MAX_H         = 0,
    parameter integer POOL_MAX_W    = 0,
    parameter integer POOL_MAX_H    = 0,
    parameter integer RESIZE_MAX_W  = 0,
    parameter integer RESIZE_MAX_H  = 0,
    // v0.2 M3: real per-engine channel / row capacity.  0 keeps the historical
    // inheritance (MAX_CH / align16(align8(MAX_CH)*MAX_W)), so legacy tops are
    // unchanged; an affine build passes the real engine maxima.
    parameter integer RESIZE_MAX_CH = 0,
    parameter integer RESIZE_MAX_IN_ROW_BYTES  = 0,
    parameter integer RESIZE_MAX_OUT_ROW_BYTES = 0,
    parameter integer POOL5_MAX_W   = 0,
    parameter integer POOL5_MAX_H   = 0,
    parameter integer POOL5_MAX_CH  = 0,
    parameter integer POOL2D_MAX_IN_ROW_BYTES = 1024,
    parameter integer POOL2D_MAX_OUT_ROW_BYTES = 1024,
    parameter integer POOL2D_MAX_EH = 25,
    parameter integer POOL2D_CACHE_ROWS = 32,
    parameter integer ACT_MAX_W     = 0,
    parameter integer ACT_MAX_H     = 0,
    parameter integer ACT_MAX_CH    = 0,
    parameter integer COPY_MAX_ROW_BYTES = 0,
    parameter integer COPY_MAX_H    = 0,
    parameter integer COPY_MAX_W    = 0,
    parameter integer COPY_MAX_CH   = 0,
    parameter integer DW_MAX_W      = 0,
    // v0.2 per-engine instantiation gate.  1 (default) = historical netlist with
    // every engine elaborated.  0 = the depthwise core is NOT instantiated (it
    // is recorded instantiated=false in the manifest) because the compiled graph
    // never executes opcode 0x001C; the opcode mux can then never select it.
    // Used only for spatial builds whose capability table reports
    // depthwise.used_by_graph=false, so an unused, not-yet-closed DW core cannot
    // block a large graph's timing or inflate its resource budget.
    parameter integer ENABLE_DEPTHWISE = 1,
    // v0.3 physical-instance mask.  bit map: 0 dense, 1 maxpool2, 2 eltwise,
    // 3 copy, 4 clip, 5 gap, 6 avgpool2, 7 mul, 8 resize, 9 pool5, 10 activation,
    // 11 depthwise.  A cleared bit elaborates that core away; the opcode mux
    // can then never select it (the compiler rejects such records too).
    // 12'hFFF keeps the historical full-function netlist.
    parameter [12:0] PHYSICAL_ENGINE_MASK = 13'h0FFF,
    // Depthwise engine hardware configuration (frozen DW core interface).  The
    // router only forwards these to the 0x001C core; all datapath scaling lives
    // in depthwise_conv3x3_core.
    parameter integer DW_PC           = 8,
    parameter integer DW_TAP_PARALLEL = 1,
    parameter integer DW_RQ_LANES     = 1,
    parameter integer CLAMP_MIN_EN  = 1
) (
    input  wire        clk,
    input  wire        rst_n,

    // explicit operator select + launch + lifecycle
    input  wire [7:0]  op_sel,          // 1/2 = Conv/Pool, 0x10/0x11 = Relu/Add
    input  wire        launch,          // 1-cycle start pulse
    input  wire        soft_reset,
    input  wire        abort,

    // per-record configuration (from npu_instr_seq)
    input  wire [31:0] cfg_in_addr,
    input  wire [31:0] cfg_in1_addr,
    input  wire [31:0] cfg_weight_addr,
    input  wire [31:0] cfg_bias_addr,
    input  wire [31:0] cfg_quant_addr,
    input  wire [31:0] cfg_out_addr,
    input  wire [15:0] cfg_h,
    input  wire [15:0] cfg_w,
    input  wire [15:0] cfg_cin,
    input  wire [15:0] cfg_cout,
    input  wire [31:0] cfg_conv_cfg,
    input  wire [31:0] cfg_quant_cfg,
    input  wire [31:0] cfg_act_cfg,
    input  wire [31:0] cfg_in_row_stride_bytes,
    input  wire [31:0] cfg_out_row_stride_bytes,
    input  wire [31:0] cfg_in_bytes,
    input  wire [31:0] cfg_in1_bytes,
    input  wire [31:0] cfg_elem_cfg,
    input  wire [31:0] cfg_rescale_a,
    input  wire [31:0] cfg_rescale_b,
    input  wire [31:0] cfg_rescale_cfg,
    input  wire [31:0] cfg_weight_bytes,
    input  wire [31:0] cfg_bias_bytes,
    input  wire [31:0] cfg_quant_bytes,
    input  wire [31:0] cfg_out_bytes,
    input  wire        copy_cfg_we,
    input  wire [5:0]  copy_cfg_addr,
    input  wire [31:0] copy_cfg_data,
    input  wire        pool_cfg_we,
    input  wire [3:0]  pool_cfg_addr,
    input  wire [31:0] pool_cfg_data,

    // selected-core status (to npu_instr_seq)
    output wire        active_core_busy,
    output wire        active_core_done,
    output wire        active_core_error,
    output wire [15:0] active_core_error_code,

    // normalized read service (to op_shell)
    output wire        shell_rd_cmd_valid,
    input  wire        shell_rd_cmd_ready,
    output wire [31:0] shell_rd_cmd_addr_byte,
    output wire [15:0] shell_rd_cmd_bytes,
    output wire [7:0]  shell_rd_cmd_tag,
    input  wire        shell_rd_data_valid,
    output wire        shell_rd_data_ready,
    input  wire [127:0] shell_rd_data,
    input  wire        shell_rd_data_last,
    input  wire        shell_rd_data_error,
    input  wire [15:0] shell_rd_data_error_code,
    input  wire [7:0]  shell_rd_data_tag,

    // normalized write service (to op_shell)
    output wire        shell_wr_cmd_valid,
    input  wire        shell_wr_cmd_ready,
    output wire [31:0] shell_wr_cmd_addr_byte,
    output wire [15:0] shell_wr_cmd_bytes,
    output wire [7:0]  shell_wr_cmd_tag,
    output wire        shell_wr_data_valid,
    input  wire        shell_wr_data_ready,
    output wire [127:0] shell_wr_data,
    output wire        shell_wr_data_last,
    output wire [7:0]  shell_wr_data_tag,
    input  wire        shell_wr_rsp_valid,
    output wire        shell_wr_rsp_ready,
    input  wire        shell_wr_rsp_error,
    input  wire        shell_wr_rsp_aborted,
    input  wire [15:0] shell_wr_rsp_error_code,
    input  wire [7:0]  shell_wr_rsp_tag
);

    // Launch-time configuration boundary: capture one instruction record and
    // start the selected reusable core one cycle later. This cuts the direct
    // sequencer-to-core high-fanout timing path without changing the ABI.
    reg        router_launch_q;
    reg [7:0]  router_op_sel_q;
    reg [31:0] router_cfg_in_addr_q, router_cfg_in1_addr_q, router_cfg_weight_addr_q, router_cfg_bias_addr_q, router_cfg_quant_addr_q, router_cfg_out_addr_q;
    reg [15:0] router_cfg_h_q, router_cfg_w_q, router_cfg_cin_q, router_cfg_cout_q;
    reg [31:0] router_cfg_conv_cfg_q, router_cfg_quant_cfg_q, router_cfg_act_cfg_q;
    reg [31:0] router_cfg_in_row_stride_bytes_q, router_cfg_out_row_stride_bytes_q, router_cfg_in_bytes_q, router_cfg_in1_bytes_q, router_cfg_weight_bytes_q, router_cfg_bias_bytes_q, router_cfg_quant_bytes_q, router_cfg_out_bytes_q, router_cfg_elem_cfg_q;
    reg [31:0] router_rescale_a_q, router_rescale_b_q, router_rescale_cfg_q;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            router_launch_q <= 1'b0; router_op_sel_q <= 8'd0;
            router_cfg_in_addr_q <= 0; router_cfg_in1_addr_q <= 0; router_cfg_weight_addr_q <= 0; router_cfg_bias_addr_q <= 0; router_cfg_quant_addr_q <= 0; router_cfg_out_addr_q <= 0;
            router_cfg_h_q <= 0; router_cfg_w_q <= 0; router_cfg_cin_q <= 0; router_cfg_cout_q <= 0;
            router_cfg_conv_cfg_q <= 0; router_cfg_quant_cfg_q <= 0; router_cfg_act_cfg_q <= 0;
            router_cfg_in_row_stride_bytes_q <= 0; router_cfg_out_row_stride_bytes_q <= 0; router_cfg_in_bytes_q <= 0; router_cfg_in1_bytes_q <= 0; router_cfg_elem_cfg_q <= 0; router_cfg_weight_bytes_q <= 0; router_cfg_bias_bytes_q <= 0; router_cfg_quant_bytes_q <= 0; router_cfg_out_bytes_q <= 0;
            router_rescale_a_q <= 0; router_rescale_b_q <= 0; router_rescale_cfg_q <= 0;
        end else begin
            router_launch_q <= launch;
            if (launch) begin
                router_op_sel_q <= op_sel;
                router_cfg_in_addr_q <= cfg_in_addr; router_cfg_in1_addr_q <= cfg_in1_addr; router_cfg_weight_addr_q <= cfg_weight_addr; router_cfg_bias_addr_q <= cfg_bias_addr; router_cfg_quant_addr_q <= cfg_quant_addr; router_cfg_out_addr_q <= cfg_out_addr;
                router_cfg_h_q <= cfg_h; router_cfg_w_q <= cfg_w; router_cfg_cin_q <= cfg_cin; router_cfg_cout_q <= cfg_cout;
                router_cfg_conv_cfg_q <= cfg_conv_cfg; router_cfg_quant_cfg_q <= cfg_quant_cfg; router_cfg_act_cfg_q <= cfg_act_cfg;
                router_cfg_in_row_stride_bytes_q <= cfg_in_row_stride_bytes; router_cfg_out_row_stride_bytes_q <= cfg_out_row_stride_bytes; router_cfg_in_bytes_q <= cfg_in_bytes; router_cfg_in1_bytes_q <= cfg_in1_bytes; router_cfg_elem_cfg_q <= cfg_elem_cfg; router_cfg_weight_bytes_q <= cfg_weight_bytes; router_cfg_bias_bytes_q <= cfg_bias_bytes; router_cfg_quant_bytes_q <= cfg_quant_bytes; router_cfg_out_bytes_q <= cfg_out_bytes;
                router_rescale_a_q <= cfg_rescale_a; router_rescale_b_q <= cfg_rescale_b; router_rescale_cfg_q <= cfg_rescale_cfg;
            end
        end
    end
    wire [1:0] cfg_stride = router_cfg_conv_cfg_q[3:2];
    wire [1:0] cfg_pad    = router_cfg_conv_cfg_q[5:4];
    // DepthwiseConv3x3 payload word 7 encodes stride-1 in [3:2] (SPEC 1.2) while
    // the frozen core port takes the literal stride value 1/2 (SPEC 3.1), so the
    // router decodes [3:2]+1; pad [5:4] carries the literal 0/1 value.
    wire [1:0] dw_cfg_stride = router_cfg_conv_cfg_q[3:2] + 2'd1;
    wire [1:0] dw_cfg_pad    = router_cfg_conv_cfg_q[5:4];
    // C_alloc = align8(C) is a core-side contract value (SPEC 3.2); the payload
    // carries C in both halves (Cin==Cout==C for depthwise), so derive it here
    // instead of forwarding the raw channel count.
    wire [15:0] dw_cfg_calloc = (router_cfg_cout_q + 16'd7) & 16'hFFF8;
    wire [7:0] cfg_xz     = router_cfg_quant_cfg_q[7:0];

    wire is_conv = (router_op_sel_q == 8'h01);
    wire is_pool = (router_op_sel_q == 8'h02);
    wire is_relu = (router_op_sel_q == 8'h10);
    wire is_add  = (router_op_sel_q == 8'h11);
    wire is_pad  = (router_op_sel_q == 8'h12);
    wire is_slice = (router_op_sel_q == 8'h13);
    wire is_concat = (router_op_sel_q == 8'h14);
    wire is_eltwise = is_relu | is_add;
    wire is_copy = is_pad | is_slice | is_concat;
    wire is_clip = (router_op_sel_q == 8'h15);
    wire is_gap  = (router_op_sel_q == 8'h16);
    wire is_avg  = (router_op_sel_q == 8'h17);
    wire is_mul  = (router_op_sel_q == 8'h18);
    wire is_resize = (router_op_sel_q == 8'h19);
    wire is_pool5  = (router_op_sel_q == 8'h1A);
    wire is_lut    = (router_op_sel_q == 8'h1B);
    wire is_dw     = (router_op_sel_q == 8'h1C);
    wire is_pool2d = (router_op_sel_q == 8'h1D);

    wire conv_launch = router_launch_q & is_conv;
    wire pool_launch = router_launch_q & is_pool & PHYSICAL_ENGINE_MASK[1];
    wire ew_launch   = router_launch_q & is_eltwise;
    wire copy_launch = router_launch_q & is_copy;
    wire clip_launch = router_launch_q & is_clip & PHYSICAL_ENGINE_MASK[4];
    wire gap_launch  = router_launch_q & is_gap & PHYSICAL_ENGINE_MASK[5];
    wire avg_launch  = router_launch_q & is_avg & PHYSICAL_ENGINE_MASK[6];
    wire mul_launch  = router_launch_q & is_mul & PHYSICAL_ENGINE_MASK[7];
    wire resize_launch = router_launch_q & is_resize;
    wire pool5_launch  = router_launch_q & is_pool5;
    wire lut_launch    = router_launch_q & is_lut & PHYSICAL_ENGINE_MASK[10];
    wire dw_launch     = router_launch_q & is_dw & PHYSICAL_ENGINE_MASK[11] &
                          (ENABLE_DEPTHWISE != 0);
    wire pool2d_launch = router_launch_q & is_pool2d & PHYSICAL_ENGINE_MASK[12];

    // v0.2 per-engine capacity resolution.  A zero override means "inherit the
    // global MAX_W/MAX_H", i.e. exactly the pre-v0.2 netlist for legacy builds.
    // Declared before the first core instantiation because every instance below
    // consumes one of these elaborated constants.
    localparam integer EFF_MAX_H        = (MAX_H <= 0) ? 0 : MAX_H;
    // Legacy builds (MAX_H=0) inherit MAX_W as the vertical override: the legacy
    // capacity derivation folds max(H,W) into MAX_W, so this is exactly the
    // pre-v0.2 netlist for every legacy graph.
    localparam integer LEGACY_H_INHERIT  = (MAX_H > 0) ? MAX_H : MAX_W;
    localparam integer EFF_POOL_MAX_W   = (POOL_MAX_W <= 0) ? MAX_W : POOL_MAX_W;
    localparam integer EFF_POOL_MAX_H   = (POOL_MAX_H <= 0) ? EFF_MAX_H : POOL_MAX_H;
    localparam integer EFF_RESIZE_MAX_W = (RESIZE_MAX_W <= 0) ? MAX_W : RESIZE_MAX_W;
    localparam integer EFF_RESIZE_MAX_H = (RESIZE_MAX_H <= 0) ? LEGACY_H_INHERIT : RESIZE_MAX_H;
    localparam integer EFF_RESIZE_MAX_CH = (RESIZE_MAX_CH <= 0) ? MAX_CH : RESIZE_MAX_CH;
    localparam integer EFF_RESIZE_IN_ROW  = (RESIZE_MAX_IN_ROW_BYTES <= 0) ? 0 :
                                            RESIZE_MAX_IN_ROW_BYTES;
    localparam integer EFF_RESIZE_OUT_ROW = (RESIZE_MAX_OUT_ROW_BYTES <= 0) ? 0 :
                                            RESIZE_MAX_OUT_ROW_BYTES;
    localparam integer EFF_POOL5_MAX_W  = (POOL5_MAX_W <= 0) ? MAX_W : POOL5_MAX_W;
    localparam integer EFF_POOL5_MAX_H  = (POOL5_MAX_H <= 0) ? LEGACY_H_INHERIT : POOL5_MAX_H;
    localparam integer EFF_POOL5_MAX_CH = (POOL5_MAX_CH <= 0) ? MAX_CH : POOL5_MAX_CH;
    localparam integer EFF_ACT_MAX_W    = (ACT_MAX_W <= 0) ? MAX_W : ACT_MAX_W;
    localparam integer EFF_ACT_MAX_H    = (ACT_MAX_H <= 0) ? LEGACY_H_INHERIT : ACT_MAX_H;
    localparam integer EFF_ACT_MAX_CH   = (ACT_MAX_CH <= 0) ? MAX_CH : ACT_MAX_CH;
    localparam integer EFF_COPY_ROW_BYTES = (COPY_MAX_ROW_BYTES <= 0) ? 16384 : COPY_MAX_ROW_BYTES;
    localparam integer EFF_COPY_MAX_W   = (COPY_MAX_W <= 0) ? MAX_W : COPY_MAX_W;
    localparam integer EFF_COPY_MAX_H   = (COPY_MAX_H <= 0) ? LEGACY_H_INHERIT : COPY_MAX_H;
    localparam integer EFF_COPY_MAX_CH  = (COPY_MAX_CH <= 0) ? MAX_CH : COPY_MAX_CH;
    localparam integer EFF_DW_MAX_W     = (DW_MAX_W <= 0) ? MAX_W : DW_MAX_W;

    // ---- conv3x3_core ------------------------------------------------------
    wire        c_busy, c_done, c_error;
    wire [15:0] c_error_code;
    wire        c_rd_cmd_valid, c_rd_cmd_ready, c_rd_data_ready;
    wire [31:0] c_rd_cmd_addr_byte;
    wire [15:0] c_rd_cmd_bytes;
    wire [7:0]  c_rd_cmd_tag;
    wire        c_rd_data_valid, c_rd_data_last, c_rd_data_error;
    wire [127:0] c_rd_data;
    wire [15:0] c_rd_data_error_code;
    wire [7:0]  c_rd_data_tag;
    wire        c_wr_cmd_valid, c_wr_cmd_ready, c_wr_data_valid, c_wr_data_ready;
    wire [31:0] c_wr_cmd_addr_byte;
    wire [15:0] c_wr_cmd_bytes;
    wire [7:0]  c_wr_cmd_tag;
    wire [127:0] c_wr_data;
    wire        c_wr_data_last;
    wire [7:0]  c_wr_data_tag;
    wire        c_wr_rsp_valid, c_wr_rsp_ready, c_wr_rsp_error, c_wr_rsp_aborted;
    wire [15:0] c_wr_rsp_error_code;
    wire [7:0]  c_wr_rsp_tag;

    generate if (PHYSICAL_ENGINE_MASK[0]) begin : GEN_CONV
    conv3x3_core #(
        .MAX_CIN      (MAX_CIN),
        .MAX_COUT     (MAX_COUT),
        .MAX_W        (MAX_W),
        .MAX_H        (EFF_MAX_H),
        .MAX_LINE_ROW_BYTES (DENSE_LINE_ROW_BYTES),
        .MAX_OUT_ROW_BYTES  (DENSE_OUT_ROW_BYTES),
        .MAX_WEIGHT_TILES   (DENSE_MAX_WEIGHT_TILES),
        .WEIGHT_READ_PORTS  (DENSE_WEIGHT_READ_PORTS),
        .Tn           (Tn),
        .Tm           (Tm),
        .WEIGHT_MODE  (WEIGHT_MODE),
        .MAC_IMPL     (MAC_IMPL),
        .LOCAL_W0     (LOCAL_W0), .LOCAL_W1 (LOCAL_W1), .LOCAL_W2 (LOCAL_W2),
        .LOCAL_W3     (LOCAL_W3), .LOCAL_W4 (LOCAL_W4), .LOCAL_W5 (LOCAL_W5),
        .LOCAL_W6     (LOCAL_W6), .LOCAL_W7 (LOCAL_W7), .LOCAL_W8 (LOCAL_W8)
    ) u_conv3x3_core (
        .clk                      (clk),
        .rst_n                    (rst_n),
        .cfg_in_addr_byte         (cfg_in_addr),
        .cfg_weight_addr_byte     (cfg_weight_addr),
        .cfg_bias_addr_byte       (cfg_bias_addr),
        .cfg_quant_addr_byte      (cfg_quant_addr),
        .cfg_out_addr_byte        (cfg_out_addr),
        .cfg_h                    (router_cfg_h_q),
        .cfg_w                    (router_cfg_w_q),
        .cfg_cin                  (router_cfg_cin_q),
        .cfg_cout                 (router_cfg_cout_q),
        .cfg_stride               (cfg_stride),
        .cfg_pad                  (cfg_pad),
        .cfg_x_zero_point         (cfg_xz),
        .cfg_in_row_stride_bytes  (router_cfg_in_row_stride_bytes_q),
        .cfg_out_row_stride_bytes (router_cfg_out_row_stride_bytes_q),
        .cfg_in_bytes             (router_cfg_in_bytes_q),
        .cfg_weight_bytes         (router_cfg_weight_bytes_q),
        .cfg_bias_bytes           (router_cfg_bias_bytes_q),
        .cfg_quant_bytes          (router_cfg_quant_bytes_q),
        .cfg_out_bytes            (router_cfg_out_bytes_q),
        .cfg_conv_cfg             (router_cfg_conv_cfg_q),
        .cfg_quant_cfg            (router_cfg_quant_cfg_q),
        .cfg_act_cfg              (router_cfg_act_cfg_q),
        .core_start               (conv_launch),
        .core_busy                (c_busy),
        .core_done                (c_done),
        .core_error               (c_error),
        .core_error_code          (c_error_code),
        .core_soft_reset          (soft_reset),
        .core_abort               (abort),
        .rd_cmd_valid             (c_rd_cmd_valid),
        .rd_cmd_ready             (c_rd_cmd_ready),
        .rd_cmd_addr_byte         (c_rd_cmd_addr_byte),
        .rd_cmd_bytes             (c_rd_cmd_bytes),
        .rd_cmd_tag               (c_rd_cmd_tag),
        .rd_data_valid            (c_rd_data_valid),
        .rd_data_ready            (c_rd_data_ready),
        .rd_data                  (c_rd_data),
        .rd_data_last             (c_rd_data_last),
        .rd_data_error            (c_rd_data_error),
        .rd_data_error_code       (c_rd_data_error_code),
        .rd_data_tag              (c_rd_data_tag),
        .wr_cmd_valid             (c_wr_cmd_valid),
        .wr_cmd_ready             (c_wr_cmd_ready),
        .wr_cmd_addr_byte         (c_wr_cmd_addr_byte),
        .wr_cmd_bytes             (c_wr_cmd_bytes),
        .wr_cmd_tag               (c_wr_cmd_tag),
        .wr_data_valid            (c_wr_data_valid),
        .wr_data_ready            (c_wr_data_ready),
        .wr_data                  (c_wr_data),
        .wr_data_last             (c_wr_data_last),
        .wr_data_tag              (c_wr_data_tag),
        .wr_rsp_valid             (c_wr_rsp_valid),
        .wr_rsp_ready             (c_wr_rsp_ready),
        .wr_rsp_error             (c_wr_rsp_error),
        .wr_rsp_aborted           (c_wr_rsp_aborted),
        .wr_rsp_error_code        (c_wr_rsp_error_code),
        .wr_rsp_tag               (c_wr_rsp_tag)
    );
    end else begin : GEN_NO_CONV
        assign c_busy = 0; assign c_done = 0;
        assign c_error = is_conv; assign c_error_code = 16'h0002;
        assign c_rd_cmd_valid = 0; assign c_rd_cmd_addr_byte = 0;
        assign c_rd_cmd_bytes = 0; assign c_rd_cmd_tag = 0; assign c_rd_data_ready = 0;
        assign c_wr_cmd_valid = 0; assign c_wr_cmd_addr_byte = 0;
        assign c_wr_cmd_bytes = 0; assign c_wr_cmd_tag = 0;
        assign c_wr_data_valid = 0; assign c_wr_data = 0;
        assign c_wr_data_last = 0; assign c_wr_data_tag = 0; assign c_wr_rsp_ready = 0;
    end endgenerate

    // ---- maxpool2x2_core ---------------------------------------------------
    wire        p_busy, p_done, p_error;
    wire [15:0] p_error_code;
    wire        p_rd_cmd_valid, p_rd_cmd_ready, p_rd_data_ready;
    wire [31:0] p_rd_cmd_addr_byte;
    wire [15:0] p_rd_cmd_bytes;
    wire [7:0]  p_rd_cmd_tag;
    wire        p_rd_data_valid, p_rd_data_last, p_rd_data_error;
    wire [127:0] p_rd_data;
    wire [15:0] p_rd_data_error_code;
    wire [7:0]  p_rd_data_tag;
    wire        p_wr_cmd_valid, p_wr_cmd_ready, p_wr_data_valid, p_wr_data_ready;
    wire [31:0] p_wr_cmd_addr_byte;
    wire [15:0] p_wr_cmd_bytes;
    wire [7:0]  p_wr_cmd_tag;
    wire [127:0] p_wr_data;
    wire        p_wr_data_last;
    wire [7:0]  p_wr_data_tag;
    wire        p_wr_rsp_valid, p_wr_rsp_ready, p_wr_rsp_error, p_wr_rsp_aborted;
    wire [15:0] p_wr_rsp_error_code;
    wire [7:0]  p_wr_rsp_tag;

    generate
    if (PHYSICAL_ENGINE_MASK[1] != 0) begin : GEN_MAXPOOL2
    maxpool2x2_core #(
        .MAX_CH       (MAX_CH),
        .MAX_W        (EFF_POOL_MAX_W),
        .MAX_H        (EFF_POOL_MAX_H),
        .CLAMP_MIN_EN (CLAMP_MIN_EN)
    ) u_maxpool2x2_core (
        .clk                      (clk),
        .rst_n                    (rst_n),
        .cfg_in_addr_byte         (cfg_in_addr),
        .cfg_weight_addr_byte     (cfg_weight_addr),
        .cfg_bias_addr_byte       (cfg_bias_addr),
        .cfg_quant_addr_byte      (cfg_quant_addr),
        .cfg_out_addr_byte        (cfg_out_addr),
        .cfg_h                    (router_cfg_h_q),
        .cfg_w                    (router_cfg_w_q),
        .cfg_cin                  (router_cfg_cin_q),
        .cfg_cout                 (router_cfg_cout_q),
        .cfg_stride               (cfg_stride),
        .cfg_pad                  (cfg_pad),
        .cfg_x_zero_point         (cfg_xz),
        .cfg_in_row_stride_bytes  (router_cfg_in_row_stride_bytes_q),
        .cfg_out_row_stride_bytes (router_cfg_out_row_stride_bytes_q),
        .cfg_in_bytes             (router_cfg_in_bytes_q),
        .cfg_weight_bytes         (router_cfg_weight_bytes_q),
        .cfg_bias_bytes           (router_cfg_bias_bytes_q),
        .cfg_quant_bytes          (router_cfg_quant_bytes_q),
        .cfg_out_bytes            (router_cfg_out_bytes_q),
        .cfg_conv_cfg             (router_cfg_conv_cfg_q),
        .cfg_quant_cfg            (router_cfg_quant_cfg_q),
        .cfg_act_cfg              (router_cfg_act_cfg_q),
        .core_start               (pool_launch),
        .core_busy                (p_busy),
        .core_done                (p_done),
        .core_error               (p_error),
        .core_error_code          (p_error_code),
        .core_soft_reset          (soft_reset),
        .core_abort               (abort),
        .rd_cmd_valid             (p_rd_cmd_valid),
        .rd_cmd_ready             (p_rd_cmd_ready),
        .rd_cmd_addr_byte         (p_rd_cmd_addr_byte),
        .rd_cmd_bytes             (p_rd_cmd_bytes),
        .rd_cmd_tag               (p_rd_cmd_tag),
        .rd_data_valid            (p_rd_data_valid),
        .rd_data_ready            (p_rd_data_ready),
        .rd_data                  (p_rd_data),
        .rd_data_last             (p_rd_data_last),
        .rd_data_error            (p_rd_data_error),
        .rd_data_error_code       (p_rd_data_error_code),
        .rd_data_tag              (p_rd_data_tag),
        .wr_cmd_valid             (p_wr_cmd_valid),
        .wr_cmd_ready             (p_wr_cmd_ready),
        .wr_cmd_addr_byte         (p_wr_cmd_addr_byte),
        .wr_cmd_bytes             (p_wr_cmd_bytes),
        .wr_cmd_tag               (p_wr_cmd_tag),
        .wr_data_valid            (p_wr_data_valid),
        .wr_data_ready            (p_wr_data_ready),
        .wr_data                  (p_wr_data),
        .wr_data_last             (p_wr_data_last),
        .wr_data_tag              (p_wr_data_tag),
        .wr_rsp_valid             (p_wr_rsp_valid),
        .wr_rsp_ready             (p_wr_rsp_ready),
        .wr_rsp_error             (p_wr_rsp_error),
        .wr_rsp_aborted           (p_wr_rsp_aborted),
        .wr_rsp_error_code        (p_wr_rsp_error_code),
        .wr_rsp_tag               (p_wr_rsp_tag)
    );
    end else begin : GEN_MAXPOOL2_OFF
        assign p_busy = 0;
        assign p_done = 0;
        assign p_error = 0;
        assign p_error_code = 0;
        assign p_rd_cmd_addr_byte = 0;
        assign p_rd_cmd_bytes = 0;
        assign p_rd_cmd_tag = 0;
        assign p_rd_cmd_valid = 0;
        assign p_rd_data_ready = 0;
        assign p_wr_cmd_addr_byte = 0;
        assign p_wr_cmd_bytes = 0;
        assign p_wr_cmd_tag = 0;
        assign p_wr_cmd_valid = 0;
        assign p_wr_data = 0;
        assign p_wr_data_last = 0;
        assign p_wr_data_tag = 0;
        assign p_wr_data_valid = 0;
        assign p_wr_rsp_ready = 0;
    end
    endgenerate

    // ---- eltwise_core ------------------------------------------------------
    wire        e_busy, e_done, e_error;
    wire [15:0] e_error_code;
    wire        e_rd_cmd_valid, e_rd_cmd_ready, e_rd_data_ready;
    wire [31:0] e_rd_cmd_addr_byte;
    wire [15:0] e_rd_cmd_bytes;
    wire [7:0]  e_rd_cmd_tag;
    wire        e_rd_data_valid, e_rd_data_last, e_rd_data_error;
    wire [127:0] e_rd_data;
    wire [15:0] e_rd_data_error_code;
    wire [7:0]  e_rd_data_tag;
    wire        e_wr_cmd_valid, e_wr_cmd_ready, e_wr_data_valid, e_wr_data_ready;
    wire [31:0] e_wr_cmd_addr_byte;
    wire [15:0] e_wr_cmd_bytes;
    wire [7:0]  e_wr_cmd_tag;
    wire [127:0] e_wr_data;
    wire        e_wr_data_last;
    wire [7:0]  e_wr_data_tag;
    wire        e_wr_rsp_valid, e_wr_rsp_ready, e_wr_rsp_error, e_wr_rsp_aborted;
    wire [15:0] e_wr_rsp_error_code;
    wire [7:0]  e_wr_rsp_tag;

    wire [1:0] ew_mode = is_add ? 2'd2 : 2'd1;
    eltwise_core u_eltwise_core (
        .clk                      (clk),
        .rst_n                    (rst_n),
        .cfg_mode                 (ew_mode),
        .cfg_in0_addr_byte        (router_cfg_in_addr_q),
        .cfg_in1_addr_byte        (router_cfg_in1_addr_q),
        .cfg_out_addr_byte        (router_cfg_out_addr_q),
        .cfg_in0_bytes            (router_cfg_in_bytes_q),
        .cfg_in1_bytes            (router_cfg_in1_bytes_q),
        .cfg_out_bytes            (router_cfg_out_bytes_q),
        .cfg_flags                (router_cfg_elem_cfg_q),
        .cfg_m0_a                 (router_rescale_a_q),
        .cfg_m0_b                 (router_rescale_b_q),
        .cfg_rescale_cfg          (router_rescale_cfg_q),
        .core_start               (ew_launch),
        .core_busy                (e_busy),
        .core_done                (e_done),
        .core_error               (e_error),
        .core_error_code          (e_error_code),
        .core_soft_reset          (soft_reset),
        .core_abort               (abort),
        .rd_cmd_valid             (e_rd_cmd_valid),
        .rd_cmd_ready             (e_rd_cmd_ready),
        .rd_cmd_addr_byte         (e_rd_cmd_addr_byte),
        .rd_cmd_bytes             (e_rd_cmd_bytes),
        .rd_cmd_tag               (e_rd_cmd_tag),
        .rd_data_valid            (e_rd_data_valid),
        .rd_data_ready            (e_rd_data_ready),
        .rd_data                  (e_rd_data),
        .rd_data_last             (e_rd_data_last),
        .rd_data_error            (e_rd_data_error),
        .rd_data_error_code       (e_rd_data_error_code),
        .rd_data_tag              (e_rd_data_tag),
        .wr_cmd_valid             (e_wr_cmd_valid),
        .wr_cmd_ready             (e_wr_cmd_ready),
        .wr_cmd_addr_byte         (e_wr_cmd_addr_byte),
        .wr_cmd_bytes             (e_wr_cmd_bytes),
        .wr_cmd_tag               (e_wr_cmd_tag),
        .wr_data_valid            (e_wr_data_valid),
        .wr_data_ready            (e_wr_data_ready),
        .wr_data                  (e_wr_data),
        .wr_data_last             (e_wr_data_last),
        .wr_data_tag              (e_wr_data_tag),
        .wr_rsp_valid             (e_wr_rsp_valid),
        .wr_rsp_ready             (e_wr_rsp_ready),
        .wr_rsp_error             (e_wr_rsp_error),
        .wr_rsp_aborted           (e_wr_rsp_aborted),
        .wr_rsp_error_code        (e_wr_rsp_error_code),
        .wr_rsp_tag               (e_wr_rsp_tag)
    );

    // ---- copy_transform_core (Pad / Slice / Concat) -----------------------
    wire        x_busy, x_done, x_error;
    wire [15:0] x_error_code;
    wire        x_rd_cmd_valid, x_rd_cmd_ready, x_rd_data_ready;
    wire [31:0] x_rd_cmd_addr_byte;
    wire [15:0] x_rd_cmd_bytes;
    wire [7:0]  x_rd_cmd_tag;
    wire        x_rd_data_valid, x_rd_data_last, x_rd_data_error;
    wire [127:0] x_rd_data;
    wire [15:0] x_rd_data_error_code;
    wire [7:0]  x_rd_data_tag;
    wire        x_wr_cmd_valid, x_wr_cmd_ready, x_wr_data_valid, x_wr_data_ready;
    wire [31:0] x_wr_cmd_addr_byte;
    wire [15:0] x_wr_cmd_bytes;
    wire [7:0]  x_wr_cmd_tag;
    wire [127:0] x_wr_data;
    wire        x_wr_data_last;
    wire [7:0]  x_wr_data_tag;
    wire        x_wr_rsp_valid, x_wr_rsp_ready, x_wr_rsp_error, x_wr_rsp_aborted;
    wire [15:0] x_wr_rsp_error_code;
    wire [7:0]  x_wr_rsp_tag;

    copy_transform_core #(
        .DATA_WIDTH          (128),
        .CHUNK_BYTES         (16),
        .MAX_ROW_BYTES       (EFF_COPY_ROW_BYTES),
        .MAX_H               (EFF_COPY_MAX_H),
        .MAX_W               (EFF_COPY_MAX_W),
        .MAX_CH              (EFF_COPY_MAX_CH),
        .CFG_WORDS           (8'd64)
    ) u_copy_transform_core (
        .clk                      (clk),
        .rst_n                    (rst_n),
        .copy_cfg_we              (copy_cfg_we),
        .copy_cfg_addr            (copy_cfg_addr),
        .copy_cfg_data            (copy_cfg_data),
        .core_start               (copy_launch),
        .core_busy                (x_busy),
        .core_done                (x_done),
        .core_error               (x_error),
        .core_error_code          (x_error_code),
        .core_soft_reset          (soft_reset),
        .core_abort               (abort),
        .rd_cmd_valid             (x_rd_cmd_valid),
        .rd_cmd_ready             (x_rd_cmd_ready),
        .rd_cmd_addr_byte         (x_rd_cmd_addr_byte),
        .rd_cmd_bytes             (x_rd_cmd_bytes),
        .rd_cmd_tag               (x_rd_cmd_tag),
        .rd_data_valid            (x_rd_data_valid),
        .rd_data_ready            (x_rd_data_ready),
        .rd_data                  (x_rd_data),
        .rd_data_last             (x_rd_data_last),
        .rd_data_error            (x_rd_data_error),
        .rd_data_error_code       (x_rd_data_error_code),
        .rd_data_tag              (x_rd_data_tag),
        .wr_cmd_valid             (x_wr_cmd_valid),
        .wr_cmd_ready             (x_wr_cmd_ready),
        .wr_cmd_addr_byte         (x_wr_cmd_addr_byte),
        .wr_cmd_bytes             (x_wr_cmd_bytes),
        .wr_cmd_tag               (x_wr_cmd_tag),
        .wr_data_valid            (x_wr_data_valid),
        .wr_data_ready            (x_wr_data_ready),
        .wr_data                  (x_wr_data),
        .wr_data_last             (x_wr_data_last),
        .wr_data_tag              (x_wr_data_tag),
        .wr_rsp_valid             (x_wr_rsp_valid),
        .wr_rsp_ready             (x_wr_rsp_ready),
        .wr_rsp_error             (x_wr_rsp_error),
        .wr_rsp_aborted           (x_wr_rsp_aborted),
        .wr_rsp_error_code        (x_wr_rsp_error_code),
        .wr_rsp_tag               (x_wr_rsp_tag)
    );

    // ---- E1-P2 Clip core ----------------------------------------------------
    wire        cl_busy, cl_done, cl_error;
    wire [15:0] cl_error_code;
    wire        cl_rd_cmd_valid, cl_rd_cmd_ready, cl_rd_data_ready;
    wire [31:0] cl_rd_cmd_addr_byte;
    wire [15:0] cl_rd_cmd_bytes;
    wire [7:0]  cl_rd_cmd_tag;
    wire        cl_rd_data_valid, cl_rd_data_last, cl_rd_data_error;
    wire [127:0] cl_rd_data;
    wire [15:0] cl_rd_data_error_code;
    wire [7:0]  cl_rd_data_tag;
    wire        cl_wr_cmd_valid, cl_wr_cmd_ready, cl_wr_data_valid, cl_wr_data_ready;
    wire [31:0] cl_wr_cmd_addr_byte;
    wire [15:0] cl_wr_cmd_bytes;
    wire [7:0]  cl_wr_cmd_tag;
    wire [127:0] cl_wr_data;
    wire        cl_wr_data_last;
    wire [7:0]  cl_wr_data_tag;
    wire        cl_wr_rsp_valid, cl_wr_rsp_ready, cl_wr_rsp_error, cl_wr_rsp_aborted;
    wire [15:0] cl_wr_rsp_error_code;
    wire [7:0]  cl_wr_rsp_tag;
    generate
    if (PHYSICAL_ENGINE_MASK[4] != 0) begin : GEN_CLIP
    clip_core #(.MAX_CH(MAX_CH)) u_clip_core (
        .clk(clk), .rst_n(rst_n),
        .cfg_in_addr_byte(router_cfg_in_addr_q), .cfg_out_addr_byte(router_cfg_out_addr_q),
        .cfg_in_bytes(router_cfg_in_bytes_q), .cfg_out_bytes(router_cfg_out_bytes_q),
        .cfg_in_row_stride_bytes(router_cfg_in_row_stride_bytes_q),
        .cfg_out_row_stride_bytes(router_cfg_out_row_stride_bytes_q),
        .cfg_h(router_cfg_h_q), .cfg_w(router_cfg_w_q), .cfg_cin(router_cfg_cin_q),
        .cfg_calloc(router_cfg_cout_q), .cfg_qmin(router_cfg_elem_cfg_q[7:0]),
        .cfg_qmax(router_cfg_elem_cfg_q[15:8]),
        .core_start(clip_launch), .core_busy(cl_busy), .core_done(cl_done),
        .core_error(cl_error), .core_error_code(cl_error_code),
        .core_soft_reset(soft_reset), .core_abort(abort),
        .rd_cmd_valid(cl_rd_cmd_valid), .rd_cmd_ready(cl_rd_cmd_ready),
        .rd_cmd_addr_byte(cl_rd_cmd_addr_byte), .rd_cmd_bytes(cl_rd_cmd_bytes),
        .rd_cmd_tag(cl_rd_cmd_tag), .rd_data_valid(cl_rd_data_valid),
        .rd_data_ready(cl_rd_data_ready), .rd_data(cl_rd_data), .rd_data_last(cl_rd_data_last),
        .rd_data_error(cl_rd_data_error), .rd_data_error_code(cl_rd_data_error_code),
        .rd_data_tag(cl_rd_data_tag), .wr_cmd_valid(cl_wr_cmd_valid),
        .wr_cmd_ready(cl_wr_cmd_ready), .wr_cmd_addr_byte(cl_wr_cmd_addr_byte),
        .wr_cmd_bytes(cl_wr_cmd_bytes), .wr_cmd_tag(cl_wr_cmd_tag),
        .wr_data_valid(cl_wr_data_valid), .wr_data_ready(cl_wr_data_ready),
        .wr_data(cl_wr_data), .wr_data_last(cl_wr_data_last), .wr_data_tag(cl_wr_data_tag),
        .wr_rsp_valid(cl_wr_rsp_valid), .wr_rsp_ready(cl_wr_rsp_ready),
        .wr_rsp_error(cl_wr_rsp_error), .wr_rsp_aborted(cl_wr_rsp_aborted),
        .wr_rsp_error_code(cl_wr_rsp_error_code), .wr_rsp_tag(cl_wr_rsp_tag)
    );
    end else begin : GEN_CLIP_OFF
        assign cl_busy = 0;
        assign cl_done = 0;
        assign cl_error = 0;
        assign cl_error_code = 0;
        assign cl_rd_cmd_addr_byte = 0;
        assign cl_rd_cmd_bytes = 0;
        assign cl_rd_cmd_tag = 0;
        assign cl_rd_cmd_valid = 0;
        assign cl_rd_data_ready = 0;
        assign cl_wr_cmd_addr_byte = 0;
        assign cl_wr_cmd_bytes = 0;
        assign cl_wr_cmd_tag = 0;
        assign cl_wr_cmd_valid = 0;
        assign cl_wr_data = 0;
        assign cl_wr_data_last = 0;
        assign cl_wr_data_tag = 0;
        assign cl_wr_data_valid = 0;
        assign cl_wr_rsp_ready = 0;
    end
    endgenerate

    // ---- E1-P2 GlobalAveragePool core --------------------------------------
    wire        gp_busy, gp_done, gp_error;
    wire        ap_busy, ap_done, ap_error;
    wire        mu_busy, mu_done, mu_error;
    wire [15:0] gp_error_code;
    wire [15:0] ap_error_code;
    wire [15:0] mu_error_code;
    wire        gp_rd_cmd_valid, gp_rd_cmd_ready, gp_rd_data_ready;
    wire        ap_rd_cmd_valid, ap_rd_cmd_ready, ap_rd_data_ready;
    wire        mu_rd_cmd_valid, mu_rd_cmd_ready, mu_rd_data_ready;
    wire [31:0] gp_rd_cmd_addr_byte;
    wire [31:0] ap_rd_cmd_addr_byte;
    wire [31:0] mu_rd_cmd_addr_byte;
    wire [15:0] gp_rd_cmd_bytes;
    wire [15:0] ap_rd_cmd_bytes;
    wire [15:0] mu_rd_cmd_bytes;
    wire [7:0]  gp_rd_cmd_tag;
    wire [7:0]  ap_rd_cmd_tag;
    wire [7:0]  mu_rd_cmd_tag;
    wire        gp_rd_data_valid, gp_rd_data_last, gp_rd_data_error;
    wire        ap_rd_data_valid, ap_rd_data_last, ap_rd_data_error;
    wire        mu_rd_data_valid, mu_rd_data_last, mu_rd_data_error;
    wire [127:0] gp_rd_data;
    wire [127:0] ap_rd_data;
    wire [127:0] mu_rd_data;
    wire [15:0] gp_rd_data_error_code;
    wire [15:0] ap_rd_data_error_code;
    wire [15:0] mu_rd_data_error_code;
    wire [7:0]  gp_rd_data_tag;
    wire [7:0]  ap_rd_data_tag;
    wire [7:0]  mu_rd_data_tag;
    wire        gp_wr_cmd_valid, gp_wr_cmd_ready, gp_wr_data_valid, gp_wr_data_ready;
    wire        ap_wr_cmd_valid, ap_wr_cmd_ready, ap_wr_data_valid, ap_wr_data_ready;
    wire        mu_wr_cmd_valid, mu_wr_cmd_ready, mu_wr_data_valid, mu_wr_data_ready;
    wire [31:0] gp_wr_cmd_addr_byte;
    wire [31:0] ap_wr_cmd_addr_byte;
    wire [31:0] mu_wr_cmd_addr_byte;
    wire [15:0] gp_wr_cmd_bytes;
    wire [15:0] ap_wr_cmd_bytes;
    wire [15:0] mu_wr_cmd_bytes;
    wire [7:0]  gp_wr_cmd_tag;
    wire [7:0]  ap_wr_cmd_tag;
    wire [7:0]  mu_wr_cmd_tag;
    wire [127:0] gp_wr_data;
    wire [127:0] ap_wr_data;
    wire [127:0] mu_wr_data;
    wire        gp_wr_data_last;
    wire        ap_wr_data_last;
    wire        mu_wr_data_last;
    wire [7:0]  gp_wr_data_tag;
    wire [7:0]  ap_wr_data_tag;
    wire [7:0]  mu_wr_data_tag;
    wire        gp_wr_rsp_valid, gp_wr_rsp_ready, gp_wr_rsp_error, gp_wr_rsp_aborted;
    wire        ap_wr_rsp_valid, ap_wr_rsp_ready, ap_wr_rsp_error, ap_wr_rsp_aborted;
    wire        mu_wr_rsp_valid, mu_wr_rsp_ready, mu_wr_rsp_error, mu_wr_rsp_aborted;
    wire [15:0] gp_wr_rsp_error_code;
    wire [15:0] ap_wr_rsp_error_code;
    wire [15:0] mu_wr_rsp_error_code;
    wire [7:0]  gp_wr_rsp_tag;
    wire [7:0]  ap_wr_rsp_tag;
    wire [7:0]  mu_wr_rsp_tag;
    generate
    if (PHYSICAL_ENGINE_MASK[5] != 0) begin : GEN_GAP
    global_avg_pool_core #(.MAX_CH(MAX_CH)) u_global_avg_pool_core (
        .clk(clk), .rst_n(rst_n),
        .cfg_in_addr_byte(router_cfg_in_addr_q), .cfg_out_addr_byte(router_cfg_out_addr_q),
        .cfg_in_bytes(router_cfg_in_bytes_q), .cfg_out_bytes(router_cfg_out_bytes_q),
        .cfg_in_row_stride_bytes(router_cfg_in_row_stride_bytes_q),
        .cfg_h(router_cfg_h_q), .cfg_w(router_cfg_w_q), .cfg_cin(router_cfg_cin_q),
        .cfg_calloc(router_cfg_cout_q),
        .core_start(gap_launch), .core_busy(gp_busy), .core_done(gp_done),
        .core_error(gp_error), .core_error_code(gp_error_code),
        .core_soft_reset(soft_reset), .core_abort(abort),
        .rd_cmd_valid(gp_rd_cmd_valid), .rd_cmd_ready(gp_rd_cmd_ready),
        .rd_cmd_addr_byte(gp_rd_cmd_addr_byte), .rd_cmd_bytes(gp_rd_cmd_bytes),
        .rd_cmd_tag(gp_rd_cmd_tag), .rd_data_valid(gp_rd_data_valid),
        .rd_data_ready(gp_rd_data_ready), .rd_data(gp_rd_data), .rd_data_last(gp_rd_data_last),
        .rd_data_error(gp_rd_data_error), .rd_data_error_code(gp_rd_data_error_code),
        .rd_data_tag(gp_rd_data_tag), .wr_cmd_valid(gp_wr_cmd_valid),
        .wr_cmd_ready(gp_wr_cmd_ready), .wr_cmd_addr_byte(gp_wr_cmd_addr_byte),
        .wr_cmd_bytes(gp_wr_cmd_bytes), .wr_cmd_tag(gp_wr_cmd_tag),
        .wr_data_valid(gp_wr_data_valid), .wr_data_ready(gp_wr_data_ready),
        .wr_data(gp_wr_data), .wr_data_last(gp_wr_data_last), .wr_data_tag(gp_wr_data_tag),
        .wr_rsp_valid(gp_wr_rsp_valid), .wr_rsp_ready(gp_wr_rsp_ready),
        .wr_rsp_error(gp_wr_rsp_error), .wr_rsp_aborted(gp_wr_rsp_aborted),
        .wr_rsp_error_code(gp_wr_rsp_error_code), .wr_rsp_tag(gp_wr_rsp_tag)
    );
    end else begin : GEN_GAP_OFF
        assign gp_busy = 0;
        assign gp_done = 0;
        assign gp_error = 0;
        assign gp_error_code = 0;
        assign gp_rd_cmd_addr_byte = 0;
        assign gp_rd_cmd_bytes = 0;
        assign gp_rd_cmd_tag = 0;
        assign gp_rd_cmd_valid = 0;
        assign gp_rd_data_ready = 0;
        assign gp_wr_cmd_addr_byte = 0;
        assign gp_wr_cmd_bytes = 0;
        assign gp_wr_cmd_tag = 0;
        assign gp_wr_cmd_valid = 0;
        assign gp_wr_data = 0;
        assign gp_wr_data_last = 0;
        assign gp_wr_data_tag = 0;
        assign gp_wr_data_valid = 0;
        assign gp_wr_rsp_ready = 0;
    end
    endgenerate


    // ---- E1-P3 AveragePool core -------------------------------------------
    generate
    if (PHYSICAL_ENGINE_MASK[6] != 0) begin : GEN_AVGPOOL2
    avgpool2x2_core #(
        .MAX_CH       (MAX_CH),
        .MAX_W        (EFF_POOL_MAX_W),
        .MAX_H        (EFF_POOL_MAX_H)
    ) u_avgpool2x2_core (
        .clk                      (clk),
        .rst_n                    (rst_n),
        .cfg_in_addr_byte         (cfg_in_addr),
        .cfg_weight_addr_byte     (cfg_weight_addr),
        .cfg_bias_addr_byte       (cfg_bias_addr),
        .cfg_quant_addr_byte      (cfg_quant_addr),
        .cfg_out_addr_byte        (cfg_out_addr),
        .cfg_h                    (router_cfg_h_q),
        .cfg_w                    (router_cfg_w_q),
        .cfg_cin                  (router_cfg_cin_q),
        .cfg_cout                 (router_cfg_cout_q),
        .cfg_stride               (cfg_stride),
        .cfg_pad                  (cfg_pad),
        .cfg_x_zero_point         (cfg_xz),
        .cfg_in_row_stride_bytes  (router_cfg_in_row_stride_bytes_q),
        .cfg_out_row_stride_bytes (router_cfg_out_row_stride_bytes_q),
        .cfg_in_bytes             (router_cfg_in_bytes_q),
        .cfg_weight_bytes         (router_cfg_weight_bytes_q),
        .cfg_bias_bytes           (router_cfg_bias_bytes_q),
        .cfg_quant_bytes          (router_cfg_quant_bytes_q),
        .cfg_out_bytes            (router_cfg_out_bytes_q),
        .cfg_conv_cfg             (router_cfg_conv_cfg_q),
        .cfg_quant_cfg            (router_cfg_quant_cfg_q),
        .cfg_act_cfg              (router_cfg_act_cfg_q),
        .core_start               (avg_launch),
        .core_busy                (ap_busy),
        .core_done                (ap_done),
        .core_error               (ap_error),
        .core_error_code          (ap_error_code),
        .core_soft_reset          (soft_reset),
        .core_abort               (abort),
        .rd_cmd_valid             (ap_rd_cmd_valid),
        .rd_cmd_ready             (ap_rd_cmd_ready),
        .rd_cmd_addr_byte         (ap_rd_cmd_addr_byte),
        .rd_cmd_bytes             (ap_rd_cmd_bytes),
        .rd_cmd_tag               (ap_rd_cmd_tag),
        .rd_data_valid            (ap_rd_data_valid),
        .rd_data_ready            (ap_rd_data_ready),
        .rd_data                  (ap_rd_data),
        .rd_data_last             (ap_rd_data_last),
        .rd_data_error            (ap_rd_data_error),
        .rd_data_error_code       (ap_rd_data_error_code),
        .rd_data_tag              (ap_rd_data_tag),
        .wr_cmd_valid             (ap_wr_cmd_valid),
        .wr_cmd_ready             (ap_wr_cmd_ready),
        .wr_cmd_addr_byte         (ap_wr_cmd_addr_byte),
        .wr_cmd_bytes             (ap_wr_cmd_bytes),
        .wr_cmd_tag               (ap_wr_cmd_tag),
        .wr_data_valid            (ap_wr_data_valid),
        .wr_data_ready            (ap_wr_data_ready),
        .wr_data                  (ap_wr_data),
        .wr_data_last             (ap_wr_data_last),
        .wr_data_tag              (ap_wr_data_tag),
        .wr_rsp_valid             (ap_wr_rsp_valid),
        .wr_rsp_ready             (ap_wr_rsp_ready),
        .wr_rsp_error             (ap_wr_rsp_error),
        .wr_rsp_aborted           (ap_wr_rsp_aborted),
        .wr_rsp_error_code        (ap_wr_rsp_error_code),
        .wr_rsp_tag               (ap_wr_rsp_tag)
    );
    end else begin : GEN_AVGPOOL2_OFF
        assign ap_busy = 0;
        assign ap_done = 0;
        assign ap_error = 0;
        assign ap_error_code = 0;
        assign ap_rd_cmd_addr_byte = 0;
        assign ap_rd_cmd_bytes = 0;
        assign ap_rd_cmd_tag = 0;
        assign ap_rd_cmd_valid = 0;
        assign ap_rd_data_ready = 0;
        assign ap_wr_cmd_addr_byte = 0;
        assign ap_wr_cmd_bytes = 0;
        assign ap_wr_cmd_tag = 0;
        assign ap_wr_cmd_valid = 0;
        assign ap_wr_data = 0;
        assign ap_wr_data_last = 0;
        assign ap_wr_data_tag = 0;
        assign ap_wr_data_valid = 0;
        assign ap_wr_rsp_ready = 0;
    end
    endgenerate

    // ---- E1-P3 restricted INT8 Mul core ------------------------------------
    generate
    if (PHYSICAL_ENGINE_MASK[7] != 0) begin : GEN_MUL
    int8_mul_core u_int8_mul_core (
        .clk                      (clk),
        .rst_n                    (rst_n),
        .cfg_in0_addr_byte        (router_cfg_in_addr_q),
        .cfg_in1_addr_byte        (router_cfg_in1_addr_q),
        .cfg_out_addr_byte        (router_cfg_out_addr_q),
        .cfg_in0_bytes            (router_cfg_in_bytes_q),
        .cfg_in1_bytes            (router_cfg_in1_bytes_q),
        .cfg_out_bytes            (router_cfg_out_bytes_q),
        .cfg_m0                   (router_cfg_elem_cfg_q[23:0]),
        .cfg_shift                (router_cfg_elem_cfg_q[28:24]),
        .core_start               (mul_launch),
        .core_busy                (mu_busy),
        .core_done                (mu_done),
        .core_error               (mu_error),
        .core_error_code          (mu_error_code),
        .core_soft_reset          (soft_reset),
        .core_abort               (abort),
        .rd_cmd_valid             (mu_rd_cmd_valid),
        .rd_cmd_ready             (mu_rd_cmd_ready),
        .rd_cmd_addr_byte         (mu_rd_cmd_addr_byte),
        .rd_cmd_bytes             (mu_rd_cmd_bytes),
        .rd_cmd_tag               (mu_rd_cmd_tag),
        .rd_data_valid            (mu_rd_data_valid),
        .rd_data_ready            (mu_rd_data_ready),
        .rd_data                  (mu_rd_data),
        .rd_data_last             (mu_rd_data_last),
        .rd_data_error            (mu_rd_data_error),
        .rd_data_error_code       (mu_rd_data_error_code),
        .rd_data_tag              (mu_rd_data_tag),
        .wr_cmd_valid             (mu_wr_cmd_valid),
        .wr_cmd_ready             (mu_wr_cmd_ready),
        .wr_cmd_addr_byte         (mu_wr_cmd_addr_byte),
        .wr_cmd_bytes             (mu_wr_cmd_bytes),
        .wr_cmd_tag               (mu_wr_cmd_tag),
        .wr_data_valid            (mu_wr_data_valid),
        .wr_data_ready            (mu_wr_data_ready),
        .wr_data                  (mu_wr_data),
        .wr_data_last             (mu_wr_data_last),
        .wr_data_tag              (mu_wr_data_tag),
        .wr_rsp_valid             (mu_wr_rsp_valid),
        .wr_rsp_ready             (mu_wr_rsp_ready),
        .wr_rsp_error             (mu_wr_rsp_error),
        .wr_rsp_aborted           (mu_wr_rsp_aborted),
        .wr_rsp_error_code        (mu_wr_rsp_error_code),
        .wr_rsp_tag               (mu_wr_rsp_tag)
    );
    end else begin : GEN_MUL_OFF
        assign mu_busy = 0;
        assign mu_done = 0;
        assign mu_error = 0;
        assign mu_error_code = 0;
        assign mu_rd_cmd_addr_byte = 0;
        assign mu_rd_cmd_bytes = 0;
        assign mu_rd_cmd_tag = 0;
        assign mu_rd_cmd_valid = 0;
        assign mu_rd_data_ready = 0;
        assign mu_wr_cmd_addr_byte = 0;
        assign mu_wr_cmd_bytes = 0;
        assign mu_wr_cmd_tag = 0;
        assign mu_wr_cmd_valid = 0;
        assign mu_wr_data = 0;
        assign mu_wr_data_last = 0;
        assign mu_wr_data_tag = 0;
        assign mu_wr_data_valid = 0;
        assign mu_wr_rsp_ready = 0;
    end
    endgenerate

    // ---- E1 direct-op Resize nearest x2 core (opcode 0x0019) ---------------
    wire        rs_busy, rs_done, rs_error;
    wire [15:0] rs_error_code;
    wire        rs_rd_cmd_valid, rs_rd_cmd_ready, rs_rd_data_ready;
    wire [31:0] rs_rd_cmd_addr_byte;
    wire [15:0] rs_rd_cmd_bytes;
    wire [7:0]  rs_rd_cmd_tag;
    wire        rs_rd_data_valid, rs_rd_data_last, rs_rd_data_error;
    wire [127:0] rs_rd_data;
    wire [15:0] rs_rd_data_error_code;
    wire [7:0]  rs_rd_data_tag;
    wire        rs_wr_cmd_valid, rs_wr_cmd_ready, rs_wr_data_valid, rs_wr_data_ready;
    wire [31:0] rs_wr_cmd_addr_byte;
    wire [15:0] rs_wr_cmd_bytes;
    wire [7:0]  rs_wr_cmd_tag;
    wire [127:0] rs_wr_data;
    wire        rs_wr_data_last;
    wire [7:0]  rs_wr_data_tag;
    wire        rs_wr_rsp_valid, rs_wr_rsp_ready, rs_wr_rsp_error, rs_wr_rsp_aborted;
    wire [15:0] rs_wr_rsp_error_code;
    wire [7:0]  rs_wr_rsp_tag;

    resize_nearest2x_core #(
        .MAX_CH       (EFF_RESIZE_MAX_CH),
        .MAX_W        (EFF_RESIZE_MAX_W),
        .MAX_H        (EFF_RESIZE_MAX_H),
        .MAX_IN_ROW_BYTES  (EFF_RESIZE_IN_ROW),
        .MAX_OUT_ROW_BYTES (EFF_RESIZE_OUT_ROW)
    ) u_resize_nearest2x_core (
        .clk                      (clk),
        .rst_n                    (rst_n),
        .cfg_in_addr_byte         (router_cfg_in_addr_q),
        .cfg_weight_addr_byte     (router_cfg_weight_addr_q),
        .cfg_bias_addr_byte       (router_cfg_bias_addr_q),
        .cfg_quant_addr_byte      (router_cfg_quant_addr_q),
        .cfg_out_addr_byte        (router_cfg_out_addr_q),
        .cfg_h                    (router_cfg_h_q),
        .cfg_w                    (router_cfg_w_q),
        .cfg_cin                  (router_cfg_cin_q),
        .cfg_cout                 (router_cfg_cout_q),
        .cfg_stride               (cfg_stride),
        .cfg_pad                  (cfg_pad),
        .cfg_x_zero_point         (cfg_xz),
        .cfg_in_row_stride_bytes  (router_cfg_in_row_stride_bytes_q),
        .cfg_out_row_stride_bytes (router_cfg_out_row_stride_bytes_q),
        .cfg_in_bytes             (router_cfg_in_bytes_q),
        .cfg_weight_bytes         (router_cfg_weight_bytes_q),
        .cfg_bias_bytes           (router_cfg_bias_bytes_q),
        .cfg_quant_bytes          (router_cfg_quant_bytes_q),
        .cfg_out_bytes            (router_cfg_out_bytes_q),
        .cfg_conv_cfg             (router_cfg_conv_cfg_q),
        .cfg_quant_cfg            (router_cfg_quant_cfg_q),
        .cfg_act_cfg              (router_cfg_act_cfg_q),
        .core_start               (resize_launch),
        .core_busy                (rs_busy),
        .core_done                (rs_done),
        .core_error               (rs_error),
        .core_error_code          (rs_error_code),
        .core_soft_reset          (soft_reset),
        .core_abort               (abort),
        .rd_cmd_valid             (rs_rd_cmd_valid),
        .rd_cmd_ready             (rs_rd_cmd_ready),
        .rd_cmd_addr_byte         (rs_rd_cmd_addr_byte),
        .rd_cmd_bytes             (rs_rd_cmd_bytes),
        .rd_cmd_tag               (rs_rd_cmd_tag),
        .rd_data_valid            (rs_rd_data_valid),
        .rd_data_ready            (rs_rd_data_ready),
        .rd_data                  (rs_rd_data),
        .rd_data_last             (rs_rd_data_last),
        .rd_data_error            (rs_rd_data_error),
        .rd_data_error_code       (rs_rd_data_error_code),
        .rd_data_tag              (rs_rd_data_tag),
        .wr_cmd_valid             (rs_wr_cmd_valid),
        .wr_cmd_ready             (rs_wr_cmd_ready),
        .wr_cmd_addr_byte         (rs_wr_cmd_addr_byte),
        .wr_cmd_bytes             (rs_wr_cmd_bytes),
        .wr_cmd_tag               (rs_wr_cmd_tag),
        .wr_data_valid            (rs_wr_data_valid),
        .wr_data_ready            (rs_wr_data_ready),
        .wr_data                  (rs_wr_data),
        .wr_data_last             (rs_wr_data_last),
        .wr_data_tag              (rs_wr_data_tag),
        .wr_rsp_valid             (rs_wr_rsp_valid),
        .wr_rsp_ready             (rs_wr_rsp_ready),
        .wr_rsp_error             (rs_wr_rsp_error),
        .wr_rsp_aborted           (rs_wr_rsp_aborted),
        .wr_rsp_error_code        (rs_wr_rsp_error_code),
        .wr_rsp_tag               (rs_wr_rsp_tag)
    );

    // ---- E1 direct-op MaxPool 5x5/s1/pad2 core (opcode 0x001A) -------------
    wire        p5_busy, p5_done, p5_error;
    wire [15:0] p5_error_code;
    wire        p5_rd_cmd_valid, p5_rd_cmd_ready, p5_rd_data_ready;
    wire [31:0] p5_rd_cmd_addr_byte;
    wire [15:0] p5_rd_cmd_bytes;
    wire [7:0]  p5_rd_cmd_tag;
    wire        p5_rd_data_valid, p5_rd_data_last, p5_rd_data_error;
    wire [127:0] p5_rd_data;
    wire [15:0] p5_rd_data_error_code;
    wire [7:0]  p5_rd_data_tag;
    wire        p5_wr_cmd_valid, p5_wr_cmd_ready, p5_wr_data_valid, p5_wr_data_ready;
    wire [31:0] p5_wr_cmd_addr_byte;
    wire [15:0] p5_wr_cmd_bytes;
    wire [7:0]  p5_wr_cmd_tag;
    wire [127:0] p5_wr_data;
    wire        p5_wr_data_last;
    wire [7:0]  p5_wr_data_tag;
    wire        p5_wr_rsp_valid, p5_wr_rsp_ready, p5_wr_rsp_error, p5_wr_rsp_aborted;
    wire [15:0] p5_wr_rsp_error_code;
    wire [7:0]  p5_wr_rsp_tag;

    generate if (PHYSICAL_ENGINE_MASK[9]) begin : GEN_POOL5
    maxpool5x5_core #(
        .MAX_CH       (EFF_POOL5_MAX_CH),
        .MAX_W        (EFF_POOL5_MAX_W),
        .MAX_H        (EFF_POOL5_MAX_H)
    ) u_maxpool5x5_core (
        .clk                      (clk),
        .rst_n                    (rst_n),
        .cfg_in_addr_byte         (router_cfg_in_addr_q),
        .cfg_weight_addr_byte     (router_cfg_weight_addr_q),
        .cfg_bias_addr_byte       (router_cfg_bias_addr_q),
        .cfg_quant_addr_byte      (router_cfg_quant_addr_q),
        .cfg_out_addr_byte        (router_cfg_out_addr_q),
        .cfg_h                    (router_cfg_h_q),
        .cfg_w                    (router_cfg_w_q),
        .cfg_cin                  (router_cfg_cin_q),
        .cfg_cout                 (router_cfg_cout_q),
        .cfg_stride               (cfg_stride),
        .cfg_pad                  (cfg_pad),
        .cfg_x_zero_point         (cfg_xz),
        .cfg_in_row_stride_bytes  (router_cfg_in_row_stride_bytes_q),
        .cfg_out_row_stride_bytes (router_cfg_out_row_stride_bytes_q),
        .cfg_in_bytes             (router_cfg_in_bytes_q),
        .cfg_weight_bytes         (router_cfg_weight_bytes_q),
        .cfg_bias_bytes           (router_cfg_bias_bytes_q),
        .cfg_quant_bytes          (router_cfg_quant_bytes_q),
        .cfg_out_bytes            (router_cfg_out_bytes_q),
        .cfg_conv_cfg             (router_cfg_conv_cfg_q),
        .cfg_quant_cfg            (router_cfg_quant_cfg_q),
        .cfg_act_cfg              (router_cfg_act_cfg_q),
        .core_start               (pool5_launch),
        .core_busy                (p5_busy),
        .core_done                (p5_done),
        .core_error               (p5_error),
        .core_error_code          (p5_error_code),
        .core_soft_reset          (soft_reset),
        .core_abort               (abort),
        .rd_cmd_valid             (p5_rd_cmd_valid),
        .rd_cmd_ready             (p5_rd_cmd_ready),
        .rd_cmd_addr_byte         (p5_rd_cmd_addr_byte),
        .rd_cmd_bytes             (p5_rd_cmd_bytes),
        .rd_cmd_tag               (p5_rd_cmd_tag),
        .rd_data_valid            (p5_rd_data_valid),
        .rd_data_ready            (p5_rd_data_ready),
        .rd_data                  (p5_rd_data),
        .rd_data_last             (p5_rd_data_last),
        .rd_data_error            (p5_rd_data_error),
        .rd_data_error_code       (p5_rd_data_error_code),
        .rd_data_tag              (p5_rd_data_tag),
        .wr_cmd_valid             (p5_wr_cmd_valid),
        .wr_cmd_ready             (p5_wr_cmd_ready),
        .wr_cmd_addr_byte         (p5_wr_cmd_addr_byte),
        .wr_cmd_bytes             (p5_wr_cmd_bytes),
        .wr_cmd_tag               (p5_wr_cmd_tag),
        .wr_data_valid            (p5_wr_data_valid),
        .wr_data_ready            (p5_wr_data_ready),
        .wr_data                  (p5_wr_data),
        .wr_data_last             (p5_wr_data_last),
        .wr_data_tag              (p5_wr_data_tag),
        .wr_rsp_valid             (p5_wr_rsp_valid),
        .wr_rsp_ready             (p5_wr_rsp_ready),
        .wr_rsp_error             (p5_wr_rsp_error),
        .wr_rsp_aborted           (p5_wr_rsp_aborted),
        .wr_rsp_error_code        (p5_wr_rsp_error_code),
        .wr_rsp_tag               (p5_wr_rsp_tag)
    );
    end else begin : GEN_NO_POOL5
        assign p5_busy = 0; assign p5_done = 0;
        assign p5_error = is_pool5; assign p5_error_code = 16'h0002;
        assign p5_rd_cmd_valid = 0; assign p5_rd_cmd_addr_byte = 0;
        assign p5_rd_cmd_bytes = 0; assign p5_rd_cmd_tag = 0; assign p5_rd_data_ready = 0;
        assign p5_wr_cmd_valid = 0; assign p5_wr_cmd_addr_byte = 0;
        assign p5_wr_cmd_bytes = 0; assign p5_wr_cmd_tag = 0;
        assign p5_wr_data_valid = 0; assign p5_wr_data = 0;
        assign p5_wr_data_last = 0; assign p5_wr_data_tag = 0; assign p5_wr_rsp_ready = 0;
    end endgenerate

    // ---- E1 S3 ActivationLUT core (opcode 0x001B) -------------------------
    // Frozen ports: contracts/S3_INTERFACE_FREEZE_20261024.md section 2.
    // cfg_lut_addr_byte <= cfg_quant_addr_q (payload word 3) and
    // cfg_lut_bytes <= cfg_quant_bytes_q (payload word 15) per section 1.1.
    wire        al_busy, al_done, al_error;
    wire [15:0] al_error_code;
    wire        al_rd_cmd_valid, al_rd_cmd_ready;
    wire [31:0] al_rd_cmd_addr_byte;
    wire [15:0] al_rd_cmd_bytes;
    wire [7:0]  al_rd_cmd_tag;
    wire        al_rd_data_valid, al_rd_data_ready, al_rd_data_last, al_rd_data_error;
    wire [127:0] al_rd_data;
    wire [15:0] al_rd_data_error_code;
    wire [7:0]  al_rd_data_tag;
    wire        al_wr_cmd_valid, al_wr_cmd_ready;
    wire [31:0] al_wr_cmd_addr_byte;
    wire [15:0] al_wr_cmd_bytes;
    wire [7:0]  al_wr_cmd_tag;
    wire        al_wr_data_valid, al_wr_data_ready, al_wr_data_last;
    wire [127:0] al_wr_data;
    wire [7:0]  al_wr_data_tag;
    wire        al_wr_rsp_valid, al_wr_rsp_ready, al_wr_rsp_error, al_wr_rsp_aborted;
    wire [15:0] al_wr_rsp_error_code;
    wire [7:0]  al_wr_rsp_tag;

    generate
    if (PHYSICAL_ENGINE_MASK[10] != 0) begin : GEN_ACTIVATION
    activation_lut_core #(
        .DATA_WIDTH  (128),
        .CHUNK_BYTES (16),
        .MAX_CH      (EFF_ACT_MAX_CH),
        .MAX_H       (EFF_ACT_MAX_H),
        .MAX_W       (EFF_ACT_MAX_W)
    ) u_activation_lut_core (
        .clk                            (clk),
        .rst_n                          (rst_n),
        .cfg_in_addr_byte               (router_cfg_in_addr_q),
        .cfg_out_addr_byte              (router_cfg_out_addr_q),
        .cfg_lut_addr_byte              (router_cfg_quant_addr_q),
        .cfg_in_bytes                   (router_cfg_in_bytes_q),
        .cfg_out_bytes                  (router_cfg_out_bytes_q),
        .cfg_in_row_stride_bytes        (router_cfg_in_row_stride_bytes_q),
        .cfg_out_row_stride_bytes       (router_cfg_out_row_stride_bytes_q),
        .cfg_h                          (router_cfg_h_q),
        .cfg_w                          (router_cfg_w_q),
        .cfg_cin                        (router_cfg_cin_q),
        .cfg_calloc                     (router_cfg_cout_q),
        .cfg_lut_bytes                  (router_cfg_quant_bytes_q),
        .core_start                     (lut_launch),
        .core_busy                      (al_busy),
        .core_done                      (al_done),
        .core_error                     (al_error),
        .core_error_code                (al_error_code),
        .core_soft_reset                (soft_reset),
        .core_abort                     (abort),
        .rd_cmd_valid                   (al_rd_cmd_valid),
        .rd_cmd_ready                   (al_rd_cmd_ready),
        .rd_cmd_addr_byte               (al_rd_cmd_addr_byte),
        .rd_cmd_bytes                   (al_rd_cmd_bytes),
        .rd_cmd_tag                     (al_rd_cmd_tag),
        .rd_data_valid                  (al_rd_data_valid),
        .rd_data_ready                  (al_rd_data_ready),
        .rd_data                        (al_rd_data),
        .rd_data_last                   (al_rd_data_last),
        .rd_data_error                  (al_rd_data_error),
        .rd_data_error_code             (al_rd_data_error_code),
        .rd_data_tag                    (al_rd_data_tag),
        .wr_cmd_valid                   (al_wr_cmd_valid),
        .wr_cmd_ready                   (al_wr_cmd_ready),
        .wr_cmd_addr_byte               (al_wr_cmd_addr_byte),
        .wr_cmd_bytes                   (al_wr_cmd_bytes),
        .wr_cmd_tag                     (al_wr_cmd_tag),
        .wr_data_valid                  (al_wr_data_valid),
        .wr_data_ready                  (al_wr_data_ready),
        .wr_data                        (al_wr_data),
        .wr_data_last                   (al_wr_data_last),
        .wr_data_tag                    (al_wr_data_tag),
        .wr_rsp_valid                   (al_wr_rsp_valid),
        .wr_rsp_ready                   (al_wr_rsp_ready),
        .wr_rsp_error                   (al_wr_rsp_error),
        .wr_rsp_aborted                 (al_wr_rsp_aborted),
        .wr_rsp_error_code              (al_wr_rsp_error_code),
        .wr_rsp_tag                     (al_wr_rsp_tag)
    );
    end else begin : GEN_ACTIVATION_OFF
        assign al_busy = 0;
        assign al_done = 0;
        assign al_error = 0;
        assign al_error_code = 0;
        assign al_rd_cmd_addr_byte = 0;
        assign al_rd_cmd_bytes = 0;
        assign al_rd_cmd_tag = 0;
        assign al_rd_cmd_valid = 0;
        assign al_rd_data_ready = 0;
        assign al_wr_cmd_addr_byte = 0;
        assign al_wr_cmd_bytes = 0;
        assign al_wr_cmd_tag = 0;
        assign al_wr_cmd_valid = 0;
        assign al_wr_data = 0;
        assign al_wr_data_last = 0;
        assign al_wr_data_tag = 0;
        assign al_wr_data_valid = 0;
        assign al_wr_rsp_ready = 0;
    end
    endgenerate

    // ---- E1 S3 DepthwiseConv3x3 core (opcode 0x001C) ----------------------
    // Frozen ports: contracts/S3_INTERFACE_FREEZE_20261024.md section 3.
    // stride/pad decode from cfg_conv_cfg_q[3:2]/[5:4] (payload word 7);
    // MAX_W sizes internal row buffers only (frozen W limit 32).
    wire        dw_busy, dw_done, dw_error;
    wire [15:0] dw_error_code;
    wire        dw_rd_cmd_valid, dw_rd_cmd_ready;
    wire [31:0] dw_rd_cmd_addr_byte;
    wire [15:0] dw_rd_cmd_bytes;
    wire [7:0]  dw_rd_cmd_tag;
    wire        dw_rd_data_valid, dw_rd_data_ready, dw_rd_data_last, dw_rd_data_error;
    wire [127:0] dw_rd_data;
    wire [15:0] dw_rd_data_error_code;
    wire [7:0]  dw_rd_data_tag;
    wire        dw_wr_cmd_valid, dw_wr_cmd_ready;
    wire [31:0] dw_wr_cmd_addr_byte;
    wire [15:0] dw_wr_cmd_bytes;
    wire [7:0]  dw_wr_cmd_tag;
    wire        dw_wr_data_valid, dw_wr_data_ready, dw_wr_data_last;
    wire [127:0] dw_wr_data;
    wire [7:0]  dw_wr_data_tag;
    wire        dw_wr_rsp_valid, dw_wr_rsp_ready, dw_wr_rsp_error, dw_wr_rsp_aborted;
    wire [15:0] dw_wr_rsp_error_code;
    wire [7:0]  dw_wr_rsp_tag;

    generate
    if ((PHYSICAL_ENGINE_MASK[11] != 0) && (ENABLE_DEPTHWISE != 0)) begin : GEN_DEPTHWISE
    depthwise_conv3x3_core #(
        .DATA_WIDTH      (128),
        .CHUNK_BYTES     (16),
        .MAX_CH          (MAX_CH),
        .MAX_W           (EFF_DW_MAX_W),
        .DW_PC           (DW_PC),
        .DW_TAP_PARALLEL (DW_TAP_PARALLEL),
        .DW_RQ_LANES     (DW_RQ_LANES)
    ) u_depthwise_conv3x3_core (
        .clk                            (clk),
        .rst_n                          (rst_n),
        .cfg_in_addr_byte               (router_cfg_in_addr_q),
        .cfg_weight_addr_byte           (router_cfg_weight_addr_q),
        .cfg_bias_addr_byte             (router_cfg_bias_addr_q),
        .cfg_quant_addr_byte            (router_cfg_quant_addr_q),
        .cfg_out_addr_byte              (router_cfg_out_addr_q),
        .cfg_h                          (router_cfg_h_q),
        .cfg_w                          (router_cfg_w_q),
        .cfg_cin                        (router_cfg_cin_q),
        .cfg_cout                       (router_cfg_cout_q),
        .cfg_calloc                     (dw_cfg_calloc),
        .cfg_stride                     (dw_cfg_stride),
        .cfg_pad                        (dw_cfg_pad),
        .cfg_in_row_stride_bytes        (router_cfg_in_row_stride_bytes_q),
        .cfg_out_row_stride_bytes       (router_cfg_out_row_stride_bytes_q),
        .cfg_in_bytes                   (router_cfg_in_bytes_q),
        .cfg_weight_bytes               (router_cfg_weight_bytes_q),
        .cfg_bias_bytes                 (router_cfg_bias_bytes_q),
        .cfg_quant_bytes                (router_cfg_quant_bytes_q),
        .cfg_out_bytes                  (router_cfg_out_bytes_q),
        .core_start                     (dw_launch),
        .core_busy                      (dw_busy),
        .core_done                      (dw_done),
        .core_error                     (dw_error),
        .core_error_code                (dw_error_code),
        .core_soft_reset                (soft_reset),
        .core_abort                     (abort),
        .rd_cmd_valid                   (dw_rd_cmd_valid),
        .rd_cmd_ready                   (dw_rd_cmd_ready),
        .rd_cmd_addr_byte               (dw_rd_cmd_addr_byte),
        .rd_cmd_bytes                   (dw_rd_cmd_bytes),
        .rd_cmd_tag                     (dw_rd_cmd_tag),
        .rd_data_valid                  (dw_rd_data_valid),
        .rd_data_ready                  (dw_rd_data_ready),
        .rd_data                        (dw_rd_data),
        .rd_data_last                   (dw_rd_data_last),
        .rd_data_error                  (dw_rd_data_error),
        .rd_data_error_code             (dw_rd_data_error_code),
        .rd_data_tag                    (dw_rd_data_tag),
        .wr_cmd_valid                   (dw_wr_cmd_valid),
        .wr_cmd_ready                   (dw_wr_cmd_ready),
        .wr_cmd_addr_byte               (dw_wr_cmd_addr_byte),
        .wr_cmd_bytes                   (dw_wr_cmd_bytes),
        .wr_cmd_tag                     (dw_wr_cmd_tag),
        .wr_data_valid                  (dw_wr_data_valid),
        .wr_data_ready                  (dw_wr_data_ready),
        .wr_data                        (dw_wr_data),
        .wr_data_last                   (dw_wr_data_last),
        .wr_data_tag                    (dw_wr_data_tag),
        .wr_rsp_valid                   (dw_wr_rsp_valid),
        .wr_rsp_ready                   (dw_wr_rsp_ready),
        .wr_rsp_error                   (dw_wr_rsp_error),
        .wr_rsp_aborted                 (dw_wr_rsp_aborted),
        .wr_rsp_error_code              (dw_wr_rsp_error_code),
        .wr_rsp_tag                     (dw_wr_rsp_tag)
    );
    end else begin : GEN_DEPTHWISE_OFF
        // Not instantiated: tie every core-driven output to a benign idle value.
        // The router only selects this engine for opcode 0x001C, which the
        // compiler rejects for a graph that does not contain DepthwiseConv.
        assign dw_busy                = 1'b0;
        assign dw_done                = 1'b0;
        assign dw_error               = 1'b0;
        assign dw_error_code          = 16'h0000;
        assign dw_rd_cmd_valid        = 1'b0;
        assign dw_rd_cmd_ready        = 1'b0;
        assign dw_rd_cmd_addr_byte    = 32'h00000000;
        assign dw_rd_cmd_bytes        = 16'h0000;
        assign dw_rd_cmd_tag          = 8'h00;
        assign dw_rd_data_ready       = 1'b0;
        assign dw_wr_cmd_valid        = 1'b0;
        assign dw_wr_cmd_ready        = 1'b0;
        assign dw_wr_cmd_addr_byte    = 32'h00000000;
        assign dw_wr_cmd_bytes        = 16'h0000;
        assign dw_wr_cmd_tag          = 8'h00;
        assign dw_wr_data_valid       = 1'b0;
        assign dw_wr_data_ready       = 1'b0;
        assign dw_wr_data             = 128'h0;
        assign dw_wr_data_last        = 1'b0;
        assign dw_wr_data_tag         = 8'h00;
        assign dw_wr_rsp_ready        = 1'b0;
    end
    endgenerate

    // Generic MaxPool has its own descriptor bank and is physically absent
    // unless bit 12 is set. Its service uses the same selected-core boundary.
    wire g2_busy, g2_done, g2_error;
    wire [15:0] g2_error_code;
    wire g2_rd_cmd_valid, g2_rd_data_ready;
    wire [31:0] g2_rd_cmd_addr_byte;
    wire [15:0] g2_rd_cmd_bytes;
    wire [7:0] g2_rd_cmd_tag;
    wire g2_wr_cmd_valid, g2_wr_data_valid, g2_wr_data_last, g2_wr_rsp_ready;
    wire [31:0] g2_wr_cmd_addr_byte;
    wire [15:0] g2_wr_cmd_bytes;
    wire [7:0] g2_wr_cmd_tag, g2_wr_data_tag;
    wire [127:0] g2_wr_data;
    // Declared before the generate scope so these connections cannot become
    // implicit, undriven local nets in tools that process declarations in order.
    reg rdq_valid;
    reg [127:0] rdq_data;
    reg rdq_last, rdq_error;
    reg [15:0] rdq_error_code;
    reg [7:0] rdq_tag;
    generate if (PHYSICAL_ENGINE_MASK[12]) begin : GEN_POOL2D
        maxpool2d_core #(
            .MAX_IN_ROW_BYTES(POOL2D_MAX_IN_ROW_BYTES),
            .MAX_OUT_ROW_BYTES(POOL2D_MAX_OUT_ROW_BYTES),
            .MAX_EH(POOL2D_MAX_EH), .CACHE_ROWS(POOL2D_CACHE_ROWS)
        ) u_pool2d (
            .clk(clk), .rst_n(rst_n), .pool_cfg_we(pool_cfg_we),
            .pool_cfg_addr(pool_cfg_addr), .pool_cfg_data(pool_cfg_data),
            .core_start(pool2d_launch), .core_busy(g2_busy), .core_done(g2_done),
            .core_error(g2_error), .core_error_code(g2_error_code),
            .core_soft_reset(soft_reset), .core_abort(abort),
            .rd_cmd_valid(g2_rd_cmd_valid), .rd_cmd_ready(is_pool2d && shell_rd_cmd_ready),
            .rd_cmd_addr_byte(g2_rd_cmd_addr_byte), .rd_cmd_bytes(g2_rd_cmd_bytes), .rd_cmd_tag(g2_rd_cmd_tag),
            .rd_data_valid(is_pool2d && rdq_valid), .rd_data_ready(g2_rd_data_ready),
            .rd_data(rdq_data), .rd_data_last(rdq_last), .rd_data_error(rdq_error),
            .rd_data_error_code(rdq_error_code), .rd_data_tag(rdq_tag),
            .wr_cmd_valid(g2_wr_cmd_valid), .wr_cmd_ready(is_pool2d && shell_wr_cmd_ready),
            .wr_cmd_addr_byte(g2_wr_cmd_addr_byte), .wr_cmd_bytes(g2_wr_cmd_bytes), .wr_cmd_tag(g2_wr_cmd_tag),
            .wr_data_valid(g2_wr_data_valid), .wr_data_ready(is_pool2d && shell_wr_data_ready),
            .wr_data(g2_wr_data), .wr_data_last(g2_wr_data_last), .wr_data_tag(g2_wr_data_tag),
            .wr_rsp_valid(is_pool2d && shell_wr_rsp_valid), .wr_rsp_ready(g2_wr_rsp_ready),
            .wr_rsp_error(shell_wr_rsp_error), .wr_rsp_aborted(shell_wr_rsp_aborted),
            .wr_rsp_error_code(shell_wr_rsp_error_code), .wr_rsp_tag(shell_wr_rsp_tag)
        );
    end else begin : GEN_NO_POOL2D
        assign g2_busy = 0; assign g2_done = 0;
        assign g2_error = is_pool2d; assign g2_error_code = 16'h0002;
        assign g2_rd_cmd_valid = 0; assign g2_rd_cmd_addr_byte = 0;
        assign g2_rd_cmd_bytes = 0; assign g2_rd_cmd_tag = 0; assign g2_rd_data_ready = 0;
        assign g2_wr_cmd_valid = 0; assign g2_wr_cmd_addr_byte = 0;
        assign g2_wr_cmd_bytes = 0; assign g2_wr_cmd_tag = 0;
        assign g2_wr_data_valid = 0; assign g2_wr_data = 0;
        assign g2_wr_data_last = 0; assign g2_wr_data_tag = 0; assign g2_wr_rsp_ready = 0;
    end endgenerate

    // ---- selected-core status ---------------------------------------------
    assign active_core_busy        = is_pool2d ? g2_busy : is_lut ? al_busy : is_dw ? dw_busy : is_pool ? p_busy :
                                     is_conv ? c_busy :
                                     is_eltwise ? e_busy :
                                     is_copy ? x_busy :
                                     is_clip ? cl_busy :
                                     is_gap ? gp_busy : is_avg ? ap_busy : is_mul ? mu_busy :
                                     is_resize ? rs_busy : is_pool5 ? p5_busy : 1'b0;
    assign active_core_done        = is_pool2d ? g2_done : is_lut ? al_done : is_dw ? dw_done : is_pool ? p_done :
                                     is_conv ? c_done :
                                     is_eltwise ? e_done :
                                     is_copy ? x_done :
                                     is_clip ? cl_done :
                                     is_gap ? gp_done : is_avg ? ap_done : is_mul ? mu_done :
                                     is_resize ? rs_done : is_pool5 ? p5_done : 1'b0;
    assign active_core_error       = is_pool2d ? g2_error : is_lut ? al_error : is_dw ? dw_error : is_pool ? p_error :
                                     is_conv ? c_error :
                                     is_eltwise ? e_error :
                                     is_copy ? x_error :
                                     is_clip ? cl_error :
                                     is_gap ? gp_error : is_avg ? ap_error : is_mul ? mu_error :
                                     is_resize ? rs_error : is_pool5 ? p5_error : 1'b0;
    assign active_core_error_code  = is_pool2d ? g2_error_code : is_lut ? al_error_code : is_dw ? dw_error_code : is_pool ? p_error_code :
                                     is_conv ? c_error_code :
                                     is_eltwise ? e_error_code :
                                     is_copy ? x_error_code :
                                     is_clip ? cl_error_code : is_avg ? ap_error_code : is_mul ? mu_error_code :
                                     is_resize ? rs_error_code : is_pool5 ? p5_error_code : gp_error_code;

    // ---- read service mux (selected core <-> shell) ------------------------
    assign shell_rd_cmd_valid     = is_pool2d ? g2_rd_cmd_valid : is_lut ? al_rd_cmd_valid : is_dw ? dw_rd_cmd_valid : is_pool ? p_rd_cmd_valid :
                                    is_conv ? c_rd_cmd_valid :
                                    is_eltwise ? e_rd_cmd_valid :
                                    is_copy ? x_rd_cmd_valid :
                                    is_clip ? cl_rd_cmd_valid :
                                    is_gap ? gp_rd_cmd_valid : is_avg ? ap_rd_cmd_valid : is_mul ? mu_rd_cmd_valid :
                                    is_resize ? rs_rd_cmd_valid : is_pool5 ? p5_rd_cmd_valid : 1'b0;
    assign shell_rd_cmd_addr_byte = is_pool2d ? g2_rd_cmd_addr_byte : is_lut ? al_rd_cmd_addr_byte : is_dw ? dw_rd_cmd_addr_byte : is_pool ? p_rd_cmd_addr_byte :
                                    is_conv ? c_rd_cmd_addr_byte :
                                    is_eltwise ? e_rd_cmd_addr_byte :
                                    is_copy ? x_rd_cmd_addr_byte :
                                    is_clip ? cl_rd_cmd_addr_byte : is_avg ? ap_rd_cmd_addr_byte : is_mul ? mu_rd_cmd_addr_byte :
                                    is_resize ? rs_rd_cmd_addr_byte : is_pool5 ? p5_rd_cmd_addr_byte : gp_rd_cmd_addr_byte;
    assign shell_rd_cmd_bytes     = is_pool2d ? g2_rd_cmd_bytes : is_lut ? al_rd_cmd_bytes : is_dw ? dw_rd_cmd_bytes : is_pool ? p_rd_cmd_bytes :
                                    is_conv ? c_rd_cmd_bytes :
                                    is_eltwise ? e_rd_cmd_bytes :
                                    is_copy ? x_rd_cmd_bytes :
                                    is_clip ? cl_rd_cmd_bytes : is_avg ? ap_rd_cmd_bytes : is_mul ? mu_rd_cmd_bytes :
                                    is_resize ? rs_rd_cmd_bytes : is_pool5 ? p5_rd_cmd_bytes : gp_rd_cmd_bytes;
    assign shell_rd_cmd_tag       = is_pool2d ? g2_rd_cmd_tag : is_lut ? al_rd_cmd_tag : is_dw ? dw_rd_cmd_tag : is_pool ? p_rd_cmd_tag :
                                    is_conv ? c_rd_cmd_tag :
                                    is_eltwise ? e_rd_cmd_tag :
                                    is_copy ? x_rd_cmd_tag :
                                    is_clip ? cl_rd_cmd_tag : is_avg ? ap_rd_cmd_tag : is_mul ? mu_rd_cmd_tag :
                                    is_resize ? rs_rd_cmd_tag : is_pool5 ? p5_rd_cmd_tag : gp_rd_cmd_tag;
    assign c_rd_cmd_ready         = is_conv ? shell_rd_cmd_ready : 1'b0;
    assign p_rd_cmd_ready         = is_pool ? shell_rd_cmd_ready : 1'b0;
    assign e_rd_cmd_ready         = is_eltwise ? shell_rd_cmd_ready : 1'b0;
    assign x_rd_cmd_ready         = is_copy ? shell_rd_cmd_ready : 1'b0;
    assign cl_rd_cmd_ready        = is_clip ? shell_rd_cmd_ready : 1'b0;
    assign gp_rd_cmd_ready        = is_gap ? shell_rd_cmd_ready : 1'b0;
    assign ap_rd_cmd_ready        = is_avg ? shell_rd_cmd_ready : 1'b0;
    assign mu_rd_cmd_ready        = is_mul ? shell_rd_cmd_ready : 1'b0;
    assign rs_rd_cmd_ready        = is_resize ? shell_rd_cmd_ready : 1'b0;
    assign p5_rd_cmd_ready        = is_pool5 ? shell_rd_cmd_ready : 1'b0;
    assign al_rd_cmd_ready        = is_lut ? shell_rd_cmd_ready : 1'b0;
    assign dw_rd_cmd_ready        = is_dw ? shell_rd_cmd_ready : 1'b0;

    // Registered one-entry read-data boundary shared by all cores.
    wire rdq_core_ready = is_pool2d ? g2_rd_data_ready : is_lut ? al_rd_data_ready : is_dw ? dw_rd_data_ready : is_pool ? p_rd_data_ready :
                          is_conv ? c_rd_data_ready :
                          is_eltwise ? e_rd_data_ready :
                          is_copy ? x_rd_data_ready :
                          is_clip ? cl_rd_data_ready :
                          is_gap ? gp_rd_data_ready : is_avg ? ap_rd_data_ready : is_mul ? mu_rd_data_ready :
                          is_resize ? rs_rd_data_ready : is_pool5 ? p5_rd_data_ready : 1'b0;
    wire rdq_pop = rdq_valid && rdq_core_ready;
    assign shell_rd_data_ready = !rdq_valid || rdq_core_ready;
    wire rdq_push = shell_rd_data_valid && shell_rd_data_ready;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rdq_valid <= 1'b0;
            rdq_data <= 128'b0;
            rdq_last <= 1'b0;
            rdq_error <= 1'b0;
            rdq_error_code <= 16'b0;
            rdq_tag <= 8'b0;
        end else begin
            if (rdq_push) begin
                rdq_valid <= 1'b1;
                rdq_data <= shell_rd_data;
                rdq_last <= shell_rd_data_last;
                rdq_error <= shell_rd_data_error;
                rdq_error_code <= shell_rd_data_error_code;
                rdq_tag <= shell_rd_data_tag;
            end else if (rdq_pop) begin
                rdq_valid <= 1'b0;
            end
        end
    end
    assign c_rd_data_valid = is_conv ? rdq_valid : 1'b0;
    assign p_rd_data_valid = is_pool ? rdq_valid : 1'b0;
    assign e_rd_data_valid = is_eltwise ? rdq_valid : 1'b0;
    assign x_rd_data_valid = is_copy ? rdq_valid : 1'b0;
    assign cl_rd_data_valid = is_clip ? rdq_valid : 1'b0;
    assign gp_rd_data_valid = is_gap ? rdq_valid : 1'b0;
    assign ap_rd_data_valid = is_avg ? rdq_valid : 1'b0;
    assign mu_rd_data_valid = is_mul ? rdq_valid : 1'b0;
    assign rs_rd_data_valid = is_resize ? rdq_valid : 1'b0;
    assign p5_rd_data_valid = is_pool5 ? rdq_valid : 1'b0;
    assign c_rd_data = rdq_data;
    assign p_rd_data = rdq_data;
    assign e_rd_data = rdq_data;
    assign x_rd_data = rdq_data;
    assign cl_rd_data = rdq_data;
    assign gp_rd_data = rdq_data;
    assign ap_rd_data = rdq_data;
    assign mu_rd_data = rdq_data;
    assign rs_rd_data = rdq_data;
    assign p5_rd_data = rdq_data;
    assign c_rd_data_last = rdq_last;
    assign p_rd_data_last = rdq_last;
    assign e_rd_data_last = rdq_last;
    assign x_rd_data_last = rdq_last;
    assign cl_rd_data_last = rdq_last;
    assign gp_rd_data_last = rdq_last;
    assign ap_rd_data_last = rdq_last;
    assign mu_rd_data_last = rdq_last;
    assign rs_rd_data_last = rdq_last;
    assign p5_rd_data_last = rdq_last;
    assign c_rd_data_error = rdq_error;
    assign p_rd_data_error = rdq_error;
    assign e_rd_data_error = rdq_error;
    assign x_rd_data_error = rdq_error;
    assign cl_rd_data_error = rdq_error;
    assign gp_rd_data_error = rdq_error;
    assign ap_rd_data_error = rdq_error;
    assign mu_rd_data_error = rdq_error;
    assign rs_rd_data_error = rdq_error;
    assign p5_rd_data_error = rdq_error;
    assign c_rd_data_error_code = rdq_error_code;
    assign p_rd_data_error_code = rdq_error_code;
    assign e_rd_data_error_code = rdq_error_code;
    assign x_rd_data_error_code = rdq_error_code;
    assign cl_rd_data_error_code = rdq_error_code;
    assign gp_rd_data_error_code = rdq_error_code;
    assign ap_rd_data_error_code = rdq_error_code;
    assign mu_rd_data_error_code = rdq_error_code;
    assign rs_rd_data_error_code = rdq_error_code;
    assign p5_rd_data_error_code = rdq_error_code;
    assign c_rd_data_tag = rdq_tag;
    assign p_rd_data_tag = rdq_tag;
    assign e_rd_data_tag = rdq_tag;
    assign x_rd_data_tag = rdq_tag;
    assign cl_rd_data_tag = rdq_tag;
    assign gp_rd_data_tag = rdq_tag;
    assign ap_rd_data_tag = rdq_tag;
    assign mu_rd_data_tag = rdq_tag;
    assign rs_rd_data_tag = rdq_tag;
    assign p5_rd_data_tag = rdq_tag;
    assign al_rd_data_valid = is_lut ? rdq_valid : 1'b0;
    assign dw_rd_data_valid = is_dw ? rdq_valid : 1'b0;
    assign al_rd_data = rdq_data;
    assign dw_rd_data = rdq_data;
    assign al_rd_data_last = rdq_last;
    assign dw_rd_data_last = rdq_last;
    assign al_rd_data_error = rdq_error;
    assign dw_rd_data_error = rdq_error;
    assign al_rd_data_error_code = rdq_error_code;
    assign dw_rd_data_error_code = rdq_error_code;
    assign al_rd_data_tag = rdq_tag;
    assign dw_rd_data_tag = rdq_tag;

    // ---- write service mux (selected core <-> shell) -----------------------
    assign shell_wr_cmd_valid = is_pool2d ? g2_wr_cmd_valid : is_lut ? al_wr_cmd_valid : is_dw ? dw_wr_cmd_valid : is_pool ? p_wr_cmd_valid :
                                is_conv ? c_wr_cmd_valid :
                                is_eltwise ? e_wr_cmd_valid :
                                is_copy ? x_wr_cmd_valid :
                                is_clip ? cl_wr_cmd_valid :
                                is_gap ? gp_wr_cmd_valid : is_avg ? ap_wr_cmd_valid : is_mul ? mu_wr_cmd_valid :
                                is_resize ? rs_wr_cmd_valid : is_pool5 ? p5_wr_cmd_valid : 1'b0;
    assign shell_wr_cmd_addr_byte = is_pool2d ? g2_wr_cmd_addr_byte : is_lut ? al_wr_cmd_addr_byte : is_dw ? dw_wr_cmd_addr_byte : is_pool ? p_wr_cmd_addr_byte :
                                     is_conv ? c_wr_cmd_addr_byte :
                                     is_eltwise ? e_wr_cmd_addr_byte :
                                     is_copy ? x_wr_cmd_addr_byte :
                                     is_clip ? cl_wr_cmd_addr_byte : is_avg ? ap_wr_cmd_addr_byte : is_mul ? mu_wr_cmd_addr_byte :
                                     is_resize ? rs_wr_cmd_addr_byte : is_pool5 ? p5_wr_cmd_addr_byte : gp_wr_cmd_addr_byte;
    assign shell_wr_cmd_bytes = is_pool2d ? g2_wr_cmd_bytes : is_lut ? al_wr_cmd_bytes : is_dw ? dw_wr_cmd_bytes : is_pool ? p_wr_cmd_bytes :
                                is_conv ? c_wr_cmd_bytes :
                                is_eltwise ? e_wr_cmd_bytes :
                                is_copy ? x_wr_cmd_bytes :
                                is_clip ? cl_wr_cmd_bytes : is_avg ? ap_wr_cmd_bytes : is_mul ? mu_wr_cmd_bytes :
                                is_resize ? rs_wr_cmd_bytes : is_pool5 ? p5_wr_cmd_bytes : gp_wr_cmd_bytes;
    assign shell_wr_cmd_tag = is_pool2d ? g2_wr_cmd_tag : is_lut ? al_wr_cmd_tag : is_dw ? dw_wr_cmd_tag : is_pool ? p_wr_cmd_tag :
                              is_conv ? c_wr_cmd_tag :
                              is_eltwise ? e_wr_cmd_tag :
                              is_copy ? x_wr_cmd_tag :
                              is_clip ? cl_wr_cmd_tag : is_avg ? ap_wr_cmd_tag : is_mul ? mu_wr_cmd_tag :
                              is_resize ? rs_wr_cmd_tag : is_pool5 ? p5_wr_cmd_tag : gp_wr_cmd_tag;
    assign shell_wr_data_valid = is_pool2d ? g2_wr_data_valid : is_lut ? al_wr_data_valid : is_dw ? dw_wr_data_valid : is_pool ? p_wr_data_valid :
                                 is_conv ? c_wr_data_valid :
                                 is_eltwise ? e_wr_data_valid :
                                 is_copy ? x_wr_data_valid :
                                 is_clip ? cl_wr_data_valid :
                                 is_gap ? gp_wr_data_valid : is_avg ? ap_wr_data_valid : is_mul ? mu_wr_data_valid :
                                 is_resize ? rs_wr_data_valid : is_pool5 ? p5_wr_data_valid : 1'b0;
    assign shell_wr_data = is_pool2d ? g2_wr_data : is_lut ? al_wr_data : is_dw ? dw_wr_data : is_pool ? p_wr_data :
                           is_conv ? c_wr_data :
                           is_eltwise ? e_wr_data :
                           is_copy ? x_wr_data :
                           is_clip ? cl_wr_data : is_avg ? ap_wr_data : is_mul ? mu_wr_data :
                           is_resize ? rs_wr_data : is_pool5 ? p5_wr_data : gp_wr_data;
    assign shell_wr_data_last = is_pool2d ? g2_wr_data_last : is_lut ? al_wr_data_last : is_dw ? dw_wr_data_last : is_pool ? p_wr_data_last :
                                is_conv ? c_wr_data_last :
                                is_eltwise ? e_wr_data_last :
                                is_copy ? x_wr_data_last :
                                is_clip ? cl_wr_data_last : is_avg ? ap_wr_data_last : is_mul ? mu_wr_data_last :
                                is_resize ? rs_wr_data_last : is_pool5 ? p5_wr_data_last : gp_wr_data_last;
    assign shell_wr_data_tag = is_pool2d ? g2_wr_data_tag : is_lut ? al_wr_data_tag : is_dw ? dw_wr_data_tag : is_pool ? p_wr_data_tag :
                               is_conv ? c_wr_data_tag :
                               is_eltwise ? e_wr_data_tag :
                               is_copy ? x_wr_data_tag :
                               is_clip ? cl_wr_data_tag : is_avg ? ap_wr_data_tag : is_mul ? mu_wr_data_tag :
                               is_resize ? rs_wr_data_tag : is_pool5 ? p5_wr_data_tag : gp_wr_data_tag;
    assign shell_wr_rsp_ready = is_pool2d ? g2_wr_rsp_ready : is_lut ? al_wr_rsp_ready : is_dw ? dw_wr_rsp_ready : is_pool ? p_wr_rsp_ready :
                                is_conv ? c_wr_rsp_ready :
                                is_eltwise ? e_wr_rsp_ready :
                                is_copy ? x_wr_rsp_ready :
                                is_clip ? cl_wr_rsp_ready :
                                is_gap ? gp_wr_rsp_ready : is_avg ? ap_wr_rsp_ready : is_mul ? mu_wr_rsp_ready :
                                is_resize ? rs_wr_rsp_ready : is_pool5 ? p5_wr_rsp_ready : 1'b0;

    assign c_wr_cmd_ready = is_conv ? shell_wr_cmd_ready : 1'b0;
    assign p_wr_cmd_ready = is_pool ? shell_wr_cmd_ready : 1'b0;
    assign e_wr_cmd_ready = is_eltwise ? shell_wr_cmd_ready : 1'b0;
    assign x_wr_cmd_ready = is_copy ? shell_wr_cmd_ready : 1'b0;
    assign cl_wr_cmd_ready = is_clip ? shell_wr_cmd_ready : 1'b0;
    assign gp_wr_cmd_ready = is_gap ? shell_wr_cmd_ready : 1'b0;
    assign ap_wr_cmd_ready = is_avg ? shell_wr_cmd_ready : 1'b0;
    assign mu_wr_cmd_ready = is_mul ? shell_wr_cmd_ready : 1'b0;
    assign rs_wr_cmd_ready = is_resize ? shell_wr_cmd_ready : 1'b0;
    assign p5_wr_cmd_ready = is_pool5 ? shell_wr_cmd_ready : 1'b0;
    assign c_wr_data_ready = is_conv ? shell_wr_data_ready : 1'b0;
    assign p_wr_data_ready = is_pool ? shell_wr_data_ready : 1'b0;
    assign e_wr_data_ready = is_eltwise ? shell_wr_data_ready : 1'b0;
    assign x_wr_data_ready = is_copy ? shell_wr_data_ready : 1'b0;
    assign cl_wr_data_ready = is_clip ? shell_wr_data_ready : 1'b0;
    assign gp_wr_data_ready = is_gap ? shell_wr_data_ready : 1'b0;
    assign ap_wr_data_ready = is_avg ? shell_wr_data_ready : 1'b0;
    assign mu_wr_data_ready = is_mul ? shell_wr_data_ready : 1'b0;
    assign rs_wr_data_ready = is_resize ? shell_wr_data_ready : 1'b0;
    assign p5_wr_data_ready = is_pool5 ? shell_wr_data_ready : 1'b0;
    assign al_wr_cmd_ready = is_lut ? shell_wr_cmd_ready : 1'b0;
    assign dw_wr_cmd_ready = is_dw ? shell_wr_cmd_ready : 1'b0;
    assign al_wr_data_ready = is_lut ? shell_wr_data_ready : 1'b0;
    assign dw_wr_data_ready = is_dw ? shell_wr_data_ready : 1'b0;

    assign c_wr_rsp_valid = is_conv ? shell_wr_rsp_valid : 1'b0;
    assign p_wr_rsp_valid = is_pool ? shell_wr_rsp_valid : 1'b0;
    assign e_wr_rsp_valid = is_eltwise ? shell_wr_rsp_valid : 1'b0;
    assign x_wr_rsp_valid = is_copy ? shell_wr_rsp_valid : 1'b0;
    assign cl_wr_rsp_valid = is_clip ? shell_wr_rsp_valid : 1'b0;
    assign gp_wr_rsp_valid = is_gap ? shell_wr_rsp_valid : 1'b0;
    assign ap_wr_rsp_valid = is_avg ? shell_wr_rsp_valid : 1'b0;
    assign mu_wr_rsp_valid = is_mul ? shell_wr_rsp_valid : 1'b0;
    assign rs_wr_rsp_valid = is_resize ? shell_wr_rsp_valid : 1'b0;
    assign p5_wr_rsp_valid = is_pool5 ? shell_wr_rsp_valid : 1'b0;
    assign al_wr_rsp_valid = is_lut ? shell_wr_rsp_valid : 1'b0;
    assign dw_wr_rsp_valid = is_dw ? shell_wr_rsp_valid : 1'b0;
    assign c_wr_rsp_error = shell_wr_rsp_error;
    assign p_wr_rsp_error = shell_wr_rsp_error;
    assign e_wr_rsp_error = shell_wr_rsp_error;
    assign x_wr_rsp_error = shell_wr_rsp_error;
    assign cl_wr_rsp_error = shell_wr_rsp_error;
    assign gp_wr_rsp_error = shell_wr_rsp_error;
    assign ap_wr_rsp_error = shell_wr_rsp_error;
    assign mu_wr_rsp_error = shell_wr_rsp_error;
    assign rs_wr_rsp_error = shell_wr_rsp_error;
    assign p5_wr_rsp_error = shell_wr_rsp_error;
    assign c_wr_rsp_aborted = shell_wr_rsp_aborted;
    assign p_wr_rsp_aborted = shell_wr_rsp_aborted;
    assign e_wr_rsp_aborted = shell_wr_rsp_aborted;
    assign x_wr_rsp_aborted = shell_wr_rsp_aborted;
    assign cl_wr_rsp_aborted = shell_wr_rsp_aborted;
    assign gp_wr_rsp_aborted = shell_wr_rsp_aborted;
    assign ap_wr_rsp_aborted = shell_wr_rsp_aborted;
    assign mu_wr_rsp_aborted = shell_wr_rsp_aborted;
    assign rs_wr_rsp_aborted = shell_wr_rsp_aborted;
    assign p5_wr_rsp_aborted = shell_wr_rsp_aborted;
    assign c_wr_rsp_error_code = shell_wr_rsp_error_code;
    assign p_wr_rsp_error_code = shell_wr_rsp_error_code;
    assign e_wr_rsp_error_code = shell_wr_rsp_error_code;
    assign x_wr_rsp_error_code = shell_wr_rsp_error_code;
    assign cl_wr_rsp_error_code = shell_wr_rsp_error_code;
    assign gp_wr_rsp_error_code = shell_wr_rsp_error_code;
    assign ap_wr_rsp_error_code = shell_wr_rsp_error_code;
    assign mu_wr_rsp_error_code = shell_wr_rsp_error_code;
    assign rs_wr_rsp_error_code = shell_wr_rsp_error_code;
    assign p5_wr_rsp_error_code = shell_wr_rsp_error_code;
    assign c_wr_rsp_tag = shell_wr_rsp_tag;
    assign p_wr_rsp_tag = shell_wr_rsp_tag;
    assign e_wr_rsp_tag = shell_wr_rsp_tag;
    assign x_wr_rsp_tag = shell_wr_rsp_tag;
    assign cl_wr_rsp_tag = shell_wr_rsp_tag;
    assign gp_wr_rsp_tag = shell_wr_rsp_tag;
    assign ap_wr_rsp_tag = shell_wr_rsp_tag;
    assign mu_wr_rsp_tag = shell_wr_rsp_tag;
    assign rs_wr_rsp_tag = shell_wr_rsp_tag;
    assign p5_wr_rsp_tag = shell_wr_rsp_tag;
    assign al_wr_rsp_error = shell_wr_rsp_error;
    assign dw_wr_rsp_error = shell_wr_rsp_error;
    assign al_wr_rsp_aborted = shell_wr_rsp_aborted;
    assign dw_wr_rsp_aborted = shell_wr_rsp_aborted;
    assign al_wr_rsp_error_code = shell_wr_rsp_error_code;
    assign dw_wr_rsp_error_code = shell_wr_rsp_error_code;
    assign al_wr_rsp_tag = shell_wr_rsp_tag;
    assign dw_wr_rsp_tag = shell_wr_rsp_tag;

endmodule



