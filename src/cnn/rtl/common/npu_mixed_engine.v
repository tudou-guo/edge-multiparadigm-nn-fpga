// -----------------------------------------------------------------------------
// npu_mixed_engine.v
//
// Reusable, *model-agnostic* mixed conv/pool NPU engine for project_3.
//
// Layer stack (all reuse-only, no new ABI):
//   op_shell            : frozen ABI 1.0 CSR + AXI4-Full DDR service (unchanged)
//   npu_instr_rom       : compile-time 32-bit instruction word stream
//   npu_instr_seq       : continuous variable-length word-stream sequencer
//   npu_engine_router   : explicit op mux onto conv3x3 / maxpool2x2 / eltwise cores
//
// Framing follows the reference project (continuous 32-bit instruction word
// stream: 0xFFFFFFFF marker, word count, then variable-length records).  The
// stream is baked into ROM at generation time instead of being pushed through
// runtime CSRs, because ABI 1.0 offsets are frozen and capacity is compile-time
// metadata (capacity_is_runtime_csr = false).  No CSR offset/layout changes.
//
// The PS still sees exactly one ABI START / one DONE for the whole chain.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps
`include "op_abi_defs.vh"

module npu_mixed_engine #(
    parameter integer AXIL_ADDR_WIDTH = `OP_AXIL_ADDR_WIDTH,
    parameter integer AXI_ADDR_WIDTH  = `OP_AXI_ADDR_WIDTH,
    parameter integer AXI_DATA_WIDTH  = `OP_AXI_DATA_WIDTH,
    parameter integer AXI_ID_WIDTH    = `OP_AXI_ID_WIDTH,
    parameter [31:0]  DEFAULT_TIMEOUT = `OP_DEFAULT_TIMEOUT,
    parameter integer MAX_CIN         = 64,
    parameter integer MAX_COUT        = 64,
    parameter integer MAX_CH          = 64,
    parameter integer MAX_W           = 256,
    parameter integer DENSE_LINE_ROW_BYTES   = 0,
    parameter integer DENSE_OUT_ROW_BYTES    = 0,
    parameter integer DENSE_MAX_WEIGHT_TILES = 0,
    parameter integer DENSE_WEIGHT_READ_PORTS = 0,
    // v0.2 spatial capacity: per-engine width/height overrides.  0 keeps the
    // historical "inherit MAX_W/MAX_H" behaviour, so a legacy build's generated
    // parameters and elaborated netlist are unchanged.
    parameter integer MAX_H           = 0,
    parameter integer POOL_MAX_W      = 0,
    parameter integer POOL_MAX_H      = 0,
    parameter integer RESIZE_MAX_W    = 0,
    parameter integer RESIZE_MAX_H    = 0,
    parameter integer RESIZE_MAX_CH   = 0,
    parameter integer RESIZE_MAX_IN_ROW_BYTES  = 0,
    parameter integer RESIZE_MAX_OUT_ROW_BYTES = 0,
    parameter integer POOL5_MAX_W     = 0,
    parameter integer POOL5_MAX_H     = 0,
    parameter integer POOL5_MAX_CH    = 0,
    parameter integer POOL2D_MAX_IN_ROW_BYTES = 1024,
    parameter integer POOL2D_MAX_OUT_ROW_BYTES = 1024,
    parameter integer POOL2D_MAX_EH = 25,
    parameter integer POOL2D_CACHE_ROWS = 32,
    parameter integer ACT_MAX_W       = 0,
    parameter integer ACT_MAX_H       = 0,
    parameter integer ACT_MAX_CH      = 0,
    parameter integer COPY_MAX_ROW_BYTES = 0,
    parameter integer COPY_MAX_H      = 0,
    parameter integer COPY_MAX_W      = 0,
    parameter integer COPY_MAX_CH     = 0,
    parameter integer DW_MAX_W        = 0,
    // v0.2 per-engine instantiation gate; 1 = historical netlist (unchanged).
    parameter integer ENABLE_DEPTHWISE = 1,
    // v0.3 physical-instance mask (see npu_engine_router.v).
    parameter [12:0] PHYSICAL_ENGINE_MASK = 13'h0FFF,
    // Depthwise engine hardware configuration (frozen DW core interface).  The
    // values are pass-through only: this engine does not reinterpret them.
    parameter integer DW_PC           = 8,
    parameter integer DW_TAP_PARALLEL = 1,
    parameter integer DW_RQ_LANES     = 1,
    parameter integer Tn              = 16,
    parameter integer Tm              = 16,
    parameter integer WEIGHT_MODE     = 0,
    parameter integer MAC_IMPL        = 0,
    parameter signed [7:0] LOCAL_W0 = 0, parameter signed [7:0] LOCAL_W1 = 0,
    parameter signed [7:0] LOCAL_W2 = 0, parameter signed [7:0] LOCAL_W3 = 0,
    parameter signed [7:0] LOCAL_W4 = 0, parameter signed [7:0] LOCAL_W5 = 0,
    parameter signed [7:0] LOCAL_W6 = 0, parameter signed [7:0] LOCAL_W7 = 0,
    parameter signed [7:0] LOCAL_W8 = 0,
    parameter integer CLAMP_MIN_EN    = 1,
    parameter integer INSTR_WORDS     = 256,
    parameter integer INSTR_ADDR_W    = 8,
    parameter string   INSTR_INIT     = "instr_rom.mem"
) (
    input  wire                              clk,
    input  wire                              rst_n,

    input  wire [AXIL_ADDR_WIDTH-1:0]        s_axil_awaddr,
    input  wire                              s_axil_awvalid,
    output wire                              s_axil_awready,
    input  wire [31:0]                       s_axil_wdata,
    input  wire [3:0]                        s_axil_wstrb,
    input  wire                              s_axil_wvalid,
    output wire                              s_axil_wready,
    output wire [1:0]                        s_axil_bresp,
    output wire                              s_axil_bvalid,
    input  wire                              s_axil_bready,
    input  wire [AXIL_ADDR_WIDTH-1:0]        s_axil_araddr,
    input  wire                              s_axil_arvalid,
    output wire                              s_axil_arready,
    output wire [31:0]                       s_axil_rdata,
    output wire [1:0]                        s_axil_rresp,
    output wire                              s_axil_rvalid,
    input  wire                              s_axil_rready,

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

    // Read-only observation taps at the shell <-> engine boundary.
    output wire                              dbg_core_start,
    output wire                              dbg_core_soft_reset,
    output wire                              dbg_core_abort,
    output wire                              dbg_core_busy,
    output wire                              dbg_core_done,
    output wire                              dbg_core_error,
    output wire [15:0]                       dbg_core_error_code,
    output wire                              dbg_core_rd_cmd_valid,
    output wire                              dbg_core_rd_cmd_ready,
    output wire [7:0]                        dbg_core_rd_cmd_tag,
    output wire [15:0]                       dbg_core_rd_cmd_bytes,
    output wire [31:0]                       dbg_core_rd_cmd_addr_byte,
    output wire                              dbg_core_rd_data_valid,
    output wire                              dbg_core_rd_data_ready,
    output wire                              dbg_core_rd_data_last,
    output wire [7:0]                        dbg_core_rd_data_tag,
    output wire                              dbg_core_wr_cmd_valid,
    output wire                              dbg_core_wr_cmd_ready,
    output wire [7:0]                        dbg_core_wr_cmd_tag,
    output wire [15:0]                       dbg_core_wr_cmd_bytes,
    output wire [31:0]                       dbg_core_wr_cmd_addr_byte,
    output wire                              dbg_core_wr_data_valid,
    output wire                              dbg_core_wr_data_ready,
    output wire                              dbg_core_wr_data_last,
    output wire [7:0]                        dbg_core_wr_data_tag,
    output wire                              dbg_core_wr_rsp_valid,
    output wire                              dbg_core_wr_rsp_ready,
    output wire [7:0]                        dbg_core_wr_rsp_tag,

    // Read-only sequencer observation (optional; harmless when unused).
    output wire [7:0]                        dbg_op_sel,
    output wire                              dbg_seq_error,
    output wire [15:0]                       dbg_seq_error_code,
    output wire                              dbg_record_done,
    output wire [15:0]                       dbg_record_index
);

    // ---- shell CSR snapshot (ignored by the mixed scheduler) --------------
    wire [31:0] cfg_in_addr_byte;
    wire [31:0] cfg_in1_addr;
    wire [31:0] cfg_weight_addr_byte;
    wire [31:0] cfg_bias_addr_byte;
    wire [31:0] cfg_quant_addr_byte;
    wire [31:0] cfg_out_addr_byte;
    wire [15:0] cfg_h;
    wire [15:0] cfg_w;
    wire [15:0] cfg_cin;
    wire [15:0] cfg_cout;
    wire [1:0]  cfg_stride;
    wire [1:0]  cfg_pad;
    wire [7:0]  cfg_x_zero_point;
    wire [31:0] cfg_in_row_stride_bytes;
    wire [31:0] cfg_out_row_stride_bytes;
    wire [31:0] cfg_in_bytes;
    wire [31:0] cfg_in1_bytes;
    wire [31:0] cfg_elem_cfg;
    wire [31:0] cfg_weight_bytes;
    wire [31:0] cfg_bias_bytes;
    wire [31:0] cfg_quant_bytes;
    wire [31:0] cfg_out_bytes;
    wire [31:0] cfg_conv_cfg;
    wire [31:0] cfg_quant_cfg;
    wire [31:0] cfg_act_cfg;

    wire        core_start;
    wire        core_soft_reset;
    wire        core_abort;
    wire        core_busy;
    wire        core_done;
    wire        core_error;
    wire [15:0] core_error_code;

    wire         core_rd_cmd_valid;
    wire         core_rd_cmd_ready;
    wire [31:0]  core_rd_cmd_addr_byte;
    wire [15:0]  core_rd_cmd_bytes;
    wire [7:0]   core_rd_cmd_tag;
    wire         core_rd_data_valid;
    wire         core_rd_data_ready;
    wire [127:0] core_rd_data;
    wire         core_rd_data_last;
    wire         core_rd_data_error;
    wire [15:0]  core_rd_data_error_code;
    wire [7:0]   core_rd_data_tag;

    wire         core_wr_cmd_valid;
    wire         core_wr_cmd_ready;
    wire [31:0]  core_wr_cmd_addr_byte;
    wire [15:0]  core_wr_cmd_bytes;
    wire [7:0]   core_wr_cmd_tag;
    wire         core_wr_data_valid;
    wire         core_wr_data_ready;
    wire [127:0] core_wr_data;
    wire         core_wr_data_last;
    wire [7:0]   core_wr_data_tag;
    wire         core_wr_rsp_valid;
    wire         core_wr_rsp_ready;
    wire         core_wr_rsp_error;
    wire         core_wr_rsp_aborted;
    wire [15:0]  core_wr_rsp_error_code;
    wire [7:0]   core_wr_rsp_tag;

    // ---- shell ------------------------------------------------------------
    op_shell #(
        .AXIL_ADDR_WIDTH (AXIL_ADDR_WIDTH),
        .AXI_ADDR_WIDTH  (AXI_ADDR_WIDTH),
        .AXI_DATA_WIDTH  (AXI_DATA_WIDTH),
        .AXI_ID_WIDTH    (AXI_ID_WIDTH),
        .DEFAULT_TIMEOUT (DEFAULT_TIMEOUT)
    ) u_shell (
        .clk                      (clk),
        .rst_n                    (rst_n),
        .s_axil_awaddr            (s_axil_awaddr),
        .s_axil_awvalid           (s_axil_awvalid),
        .s_axil_awready           (s_axil_awready),
        .s_axil_wdata             (s_axil_wdata),
        .s_axil_wstrb             (s_axil_wstrb),
        .s_axil_wvalid            (s_axil_wvalid),
        .s_axil_wready            (s_axil_wready),
        .s_axil_bresp             (s_axil_bresp),
        .s_axil_bvalid            (s_axil_bvalid),
        .s_axil_bready            (s_axil_bready),
        .s_axil_araddr            (s_axil_araddr),
        .s_axil_arvalid           (s_axil_arvalid),
        .s_axil_arready           (s_axil_arready),
        .s_axil_rdata             (s_axil_rdata),
        .s_axil_rresp             (s_axil_rresp),
        .s_axil_rvalid            (s_axil_rvalid),
        .s_axil_rready            (s_axil_rready),
        .m_axi_awid               (m_axi_awid),
        .m_axi_awaddr             (m_axi_awaddr),
        .m_axi_awlen              (m_axi_awlen),
        .m_axi_awsize             (m_axi_awsize),
        .m_axi_awburst            (m_axi_awburst),
        .m_axi_awlock             (m_axi_awlock),
        .m_axi_awcache            (m_axi_awcache),
        .m_axi_awprot             (m_axi_awprot),
        .m_axi_awqos              (m_axi_awqos),
        .m_axi_awregion           (m_axi_awregion),
        .m_axi_awvalid            (m_axi_awvalid),
        .m_axi_awready            (m_axi_awready),
        .m_axi_wdata              (m_axi_wdata),
        .m_axi_wstrb              (m_axi_wstrb),
        .m_axi_wlast              (m_axi_wlast),
        .m_axi_wvalid             (m_axi_wvalid),
        .m_axi_wready             (m_axi_wready),
        .m_axi_bid                (m_axi_bid),
        .m_axi_bresp              (m_axi_bresp),
        .m_axi_bvalid             (m_axi_bvalid),
        .m_axi_bready             (m_axi_bready),
        .m_axi_arid               (m_axi_arid),
        .m_axi_araddr             (m_axi_araddr),
        .m_axi_arlen              (m_axi_arlen),
        .m_axi_arsize             (m_axi_arsize),
        .m_axi_arburst            (m_axi_arburst),
        .m_axi_arlock             (m_axi_arlock),
        .m_axi_arcache            (m_axi_arcache),
        .m_axi_arprot             (m_axi_arprot),
        .m_axi_arqos              (m_axi_arqos),
        .m_axi_arregion           (m_axi_arregion),
        .m_axi_arvalid            (m_axi_arvalid),
        .m_axi_arready            (m_axi_arready),
        .m_axi_rid                (m_axi_rid),
        .m_axi_rdata              (m_axi_rdata),
        .m_axi_rresp              (m_axi_rresp),
        .m_axi_rlast              (m_axi_rlast),
        .m_axi_rvalid             (m_axi_rvalid),
        .m_axi_rready             (m_axi_rready),
        .cfg_in_addr_byte         (cfg_in_addr_byte),
        .cfg_weight_addr_byte     (cfg_weight_addr_byte),
        .cfg_bias_addr_byte       (cfg_bias_addr_byte),
        .cfg_quant_addr_byte      (cfg_quant_addr_byte),
        .cfg_out_addr_byte        (cfg_out_addr_byte),
        .cfg_h                    (cfg_h),
        .cfg_w                    (cfg_w),
        .cfg_cin                  (cfg_cin),
        .cfg_cout                 (cfg_cout),
        .cfg_stride               (cfg_stride),
        .cfg_pad                  (cfg_pad),
        .cfg_x_zero_point         (cfg_x_zero_point),
        .cfg_in_row_stride_bytes  (cfg_in_row_stride_bytes),
        .cfg_out_row_stride_bytes (cfg_out_row_stride_bytes),
        .cfg_in_bytes             (cfg_in_bytes),
        .cfg_weight_bytes         (cfg_weight_bytes),
        .cfg_bias_bytes           (cfg_bias_bytes),
        .cfg_quant_bytes          (cfg_quant_bytes),
        .cfg_out_bytes            (cfg_out_bytes),
        .cfg_conv_cfg             (cfg_conv_cfg),
        .cfg_quant_cfg            (cfg_quant_cfg),
        .cfg_act_cfg              (cfg_act_cfg),
        .core_start               (core_start),
        .core_soft_reset          (core_soft_reset),
        .core_abort               (core_abort),
        .core_busy                (core_busy),
        .core_done                (core_done),
        .core_error               (core_error),
        .core_error_code          (core_error_code),
        .core_rd_cmd_valid        (core_rd_cmd_valid),
        .core_rd_cmd_ready        (core_rd_cmd_ready),
        .core_rd_cmd_addr_byte    (core_rd_cmd_addr_byte),
        .core_rd_cmd_bytes        (core_rd_cmd_bytes),
        .core_rd_cmd_tag          (core_rd_cmd_tag),
        .core_rd_data_valid       (core_rd_data_valid),
        .core_rd_data_ready       (core_rd_data_ready),
        .core_rd_data             (core_rd_data),
        .core_rd_data_last        (core_rd_data_last),
        .core_rd_data_error       (core_rd_data_error),
        .core_rd_data_error_code  (core_rd_data_error_code),
        .core_rd_data_tag         (core_rd_data_tag),
        .core_wr_cmd_valid        (core_wr_cmd_valid),
        .core_wr_cmd_ready        (core_wr_cmd_ready),
        .core_wr_cmd_addr_byte    (core_wr_cmd_addr_byte),
        .core_wr_cmd_bytes        (core_wr_cmd_bytes),
        .core_wr_cmd_tag          (core_wr_cmd_tag),
        .core_wr_data_valid       (core_wr_data_valid),
        .core_wr_data_ready       (core_wr_data_ready),
        .core_wr_data             (core_wr_data),
        .core_wr_data_last        (core_wr_data_last),
        .core_wr_data_tag         (core_wr_data_tag),
        .core_wr_rsp_valid        (core_wr_rsp_valid),
        .core_wr_rsp_ready        (core_wr_rsp_ready),
        .core_wr_rsp_error        (core_wr_rsp_error),
        .core_wr_rsp_aborted      (core_wr_rsp_aborted),
        .core_wr_rsp_error_code   (core_wr_rsp_error_code),
        .core_wr_rsp_tag          (core_wr_rsp_tag)
    );

    // ---- explicit operator router + reusable compute cores ----------------
    wire                    seq_error;
    wire [15:0]             seq_error_code;
    wire active_core_busy;
    wire active_core_done;
    wire active_core_error;
    wire [15:0] active_core_error_code;
    // ---- instruction ROM + sequencer --------------------------------------
    wire [INSTR_ADDR_W-1:0] rom_addr;
    wire [31:0]             rom_data;

    npu_instr_rom #(
        .INSTR_WORDS (INSTR_WORDS),
        .ADDR_W      (INSTR_ADDR_W),
        .INIT_FILE   (INSTR_INIT)
    ) u_instr_rom (
        .addr (rom_addr),
        .dout (rom_data)
    );

    wire [7:0]  op_sel;
    wire        launch;

    wire [31:0] seq_in_addr;
    wire [31:0] seq_weight_addr;
    wire [31:0] seq_bias_addr;
    wire [31:0] seq_quant_addr;
    wire [31:0] seq_out_addr;
    wire [15:0] seq_h;
    wire [15:0] seq_w;
    wire [15:0] seq_cin;
    wire [15:0] seq_cout;
    wire [31:0] seq_conv_cfg;
    wire [31:0] seq_quant_cfg;
    wire [31:0] seq_act_cfg;
    wire [31:0] seq_in_row_stride_bytes;
    wire [31:0] seq_out_row_stride_bytes;
    wire [31:0] seq_in_bytes;
    wire [31:0] seq_weight_bytes;
    wire [31:0] seq_bias_bytes;
    wire [31:0] seq_quant_bytes;
    wire [31:0] seq_out_bytes;
    wire [31:0] seq_elem_cfg;
    wire [31:0] seq_rescale_a;
    wire [31:0] seq_rescale_b;
    wire [31:0] seq_rescale_cfg;
    wire        copy_cfg_we;
    wire [5:0]  copy_cfg_addr;
    wire [31:0] copy_cfg_data;
    wire        pool_cfg_we;
    wire [3:0]  pool_cfg_addr;
    wire [31:0] pool_cfg_data;
    wire        seq_record_done;
    wire [15:0] seq_record_index;

    wire [31:0] chain_runtime_base = (cfg_in_addr_byte == 32'b0) ? 32'h1000_0000 : cfg_in_addr_byte;

    npu_instr_seq #(
        .ADDR_W (INSTR_ADDR_W),
        .ROM_BASE_ADDR (32'h1000_0000)
    ) u_instr_seq (
        .clk                  (clk),
        .rst_n                (rst_n),
        .core_start           (core_start),
        .runtime_base_addr    (chain_runtime_base),
        .core_busy            (core_busy),
        .core_done            (core_done),
        .seq_error            (seq_error),
        .seq_error_code       (seq_error_code),
        .core_soft_reset      (core_soft_reset),
        .core_abort           (core_abort),
        .active_core_busy     (active_core_busy),
        .active_core_done     (active_core_done),
        .active_core_error    (active_core_error),
        .active_core_error_code(active_core_error_code),
        .record_index         (seq_record_index),
        .record_done          (seq_record_done),
        .rom_addr             (rom_addr),
        .rom_data             (rom_data),
        .op_sel               (op_sel),
        .launch               (launch),
        .cfg_in_addr          (seq_in_addr),
        .cfg_in1_addr         (cfg_in1_addr),
        .cfg_weight_addr      (seq_weight_addr),
        .cfg_bias_addr        (seq_bias_addr),
        .cfg_quant_addr       (seq_quant_addr),
        .cfg_out_addr         (seq_out_addr),
        .cfg_h                (seq_h),
        .cfg_w                (seq_w),
        .cfg_cin              (seq_cin),
        .cfg_cout             (seq_cout),
        .cfg_conv_cfg         (seq_conv_cfg),
        .cfg_quant_cfg        (seq_quant_cfg),
        .cfg_act_cfg          (seq_act_cfg),
        .cfg_in_row_stride_bytes (seq_in_row_stride_bytes),
        .cfg_out_row_stride_bytes(seq_out_row_stride_bytes),
        .cfg_in_bytes         (seq_in_bytes),
        .cfg_in1_bytes        (cfg_in1_bytes),
        .cfg_weight_bytes     (seq_weight_bytes),
        .cfg_bias_bytes       (seq_bias_bytes),
        .cfg_quant_bytes      (seq_quant_bytes),
        .cfg_out_bytes        (seq_out_bytes),
        .cfg_elem_cfg         (cfg_elem_cfg),
        .cfg_rescale_a        (seq_rescale_a),
        .cfg_rescale_b        (seq_rescale_b),
        .cfg_rescale_cfg      (seq_rescale_cfg),
        .copy_cfg_we          (copy_cfg_we),
        .copy_cfg_addr        (copy_cfg_addr),
        .copy_cfg_data        (copy_cfg_data),
        .pool_cfg_we          (pool_cfg_we),
        .pool_cfg_addr        (pool_cfg_addr),
        .pool_cfg_data        (pool_cfg_data)
    );

    assign dbg_op_sel = op_sel;


    npu_engine_router #(
        .MAX_CIN      (MAX_CIN),
        .MAX_COUT     (MAX_COUT),
        .MAX_W        (MAX_W),
        .DENSE_LINE_ROW_BYTES   (DENSE_LINE_ROW_BYTES),
        .DENSE_OUT_ROW_BYTES    (DENSE_OUT_ROW_BYTES),
        .DENSE_MAX_WEIGHT_TILES (DENSE_MAX_WEIGHT_TILES),
        .DENSE_WEIGHT_READ_PORTS (DENSE_WEIGHT_READ_PORTS),
        .MAX_H        (MAX_H),
        .POOL_MAX_W   (POOL_MAX_W),
        .POOL_MAX_H   (POOL_MAX_H),
        .RESIZE_MAX_W (RESIZE_MAX_W),
        .RESIZE_MAX_H (RESIZE_MAX_H),
        .RESIZE_MAX_CH (RESIZE_MAX_CH),
        .RESIZE_MAX_IN_ROW_BYTES  (RESIZE_MAX_IN_ROW_BYTES),
        .RESIZE_MAX_OUT_ROW_BYTES (RESIZE_MAX_OUT_ROW_BYTES),
        .POOL5_MAX_W  (POOL5_MAX_W),
        .POOL5_MAX_H  (POOL5_MAX_H),
        .POOL5_MAX_CH (POOL5_MAX_CH),
        .POOL2D_MAX_IN_ROW_BYTES (POOL2D_MAX_IN_ROW_BYTES),
        .POOL2D_MAX_OUT_ROW_BYTES (POOL2D_MAX_OUT_ROW_BYTES),
        .POOL2D_MAX_EH (POOL2D_MAX_EH),
        .POOL2D_CACHE_ROWS (POOL2D_CACHE_ROWS),
        .ACT_MAX_W    (ACT_MAX_W),
        .ACT_MAX_H    (ACT_MAX_H),
        .ACT_MAX_CH   (ACT_MAX_CH),
        .COPY_MAX_ROW_BYTES (COPY_MAX_ROW_BYTES),
        .COPY_MAX_H   (COPY_MAX_H),
        .COPY_MAX_W   (COPY_MAX_W),
        .COPY_MAX_CH  (COPY_MAX_CH),
        .DW_MAX_W     (DW_MAX_W),
        .ENABLE_DEPTHWISE (ENABLE_DEPTHWISE),
        .PHYSICAL_ENGINE_MASK (PHYSICAL_ENGINE_MASK),
        .Tn           (Tn),
        .Tm           (Tm),
        .WEIGHT_MODE  (WEIGHT_MODE),
        .MAC_IMPL     (MAC_IMPL),
        .LOCAL_W0     (LOCAL_W0), .LOCAL_W1 (LOCAL_W1), .LOCAL_W2 (LOCAL_W2),
        .LOCAL_W3     (LOCAL_W3), .LOCAL_W4 (LOCAL_W4), .LOCAL_W5 (LOCAL_W5),
        .LOCAL_W6     (LOCAL_W6), .LOCAL_W7 (LOCAL_W7), .LOCAL_W8 (LOCAL_W8),
        .MAX_CH       (MAX_CH),
        .DW_PC           (DW_PC),
        .DW_TAP_PARALLEL (DW_TAP_PARALLEL),
        .DW_RQ_LANES     (DW_RQ_LANES),
        .CLAMP_MIN_EN (CLAMP_MIN_EN)
    ) u_router (
        .clk                      (clk),
        .rst_n                    (rst_n),
        .op_sel                   (op_sel),
        .launch                   (launch),
        .soft_reset               (core_soft_reset),
        .abort                    (core_abort),
        .cfg_in_addr              (seq_in_addr),
        .cfg_in1_addr             (cfg_in1_addr),
        .cfg_weight_addr          (seq_weight_addr),
        .cfg_bias_addr            (seq_bias_addr),
        .cfg_quant_addr           (seq_quant_addr),
        .cfg_out_addr             (seq_out_addr),
        .cfg_h                    (seq_h),
        .cfg_w                    (seq_w),
        .cfg_cin                  (seq_cin),
        .cfg_cout                 (seq_cout),
        .cfg_conv_cfg             (seq_conv_cfg),
        .cfg_quant_cfg            (seq_quant_cfg),
        .cfg_act_cfg              (seq_act_cfg),
        .cfg_in_row_stride_bytes  (seq_in_row_stride_bytes),
        .cfg_out_row_stride_bytes (seq_out_row_stride_bytes),
        .cfg_in_bytes             (seq_in_bytes),
        .cfg_in1_bytes            (cfg_in1_bytes),
        .cfg_elem_cfg             (cfg_elem_cfg),
        .cfg_rescale_a            (seq_rescale_a),
        .cfg_rescale_b            (seq_rescale_b),
        .cfg_rescale_cfg          (seq_rescale_cfg),
        .cfg_weight_bytes         (seq_weight_bytes),
        .cfg_bias_bytes           (seq_bias_bytes),
        .cfg_quant_bytes          (seq_quant_bytes),
        .cfg_out_bytes            (seq_out_bytes),
        .copy_cfg_we              (copy_cfg_we),
        .copy_cfg_addr            (copy_cfg_addr),
        .copy_cfg_data            (copy_cfg_data),
        .pool_cfg_we              (pool_cfg_we),
        .pool_cfg_addr            (pool_cfg_addr),
        .pool_cfg_data            (pool_cfg_data),
        .active_core_busy         (active_core_busy),
        .active_core_done         (active_core_done),
        .active_core_error        (active_core_error),
        .active_core_error_code   (active_core_error_code),
        .shell_rd_cmd_valid       (core_rd_cmd_valid),
        .shell_rd_cmd_ready       (core_rd_cmd_ready),
        .shell_rd_cmd_addr_byte   (core_rd_cmd_addr_byte),
        .shell_rd_cmd_bytes       (core_rd_cmd_bytes),
        .shell_rd_cmd_tag         (core_rd_cmd_tag),
        .shell_rd_data_valid      (core_rd_data_valid),
        .shell_rd_data_ready      (core_rd_data_ready),
        .shell_rd_data            (core_rd_data),
        .shell_rd_data_last       (core_rd_data_last),
        .shell_rd_data_error      (core_rd_data_error),
        .shell_rd_data_error_code (core_rd_data_error_code),
        .shell_rd_data_tag        (core_rd_data_tag),
        .shell_wr_cmd_valid       (core_wr_cmd_valid),
        .shell_wr_cmd_ready       (core_wr_cmd_ready),
        .shell_wr_cmd_addr_byte   (core_wr_cmd_addr_byte),
        .shell_wr_cmd_bytes       (core_wr_cmd_bytes),
        .shell_wr_cmd_tag         (core_wr_cmd_tag),
        .shell_wr_data_valid      (core_wr_data_valid),
        .shell_wr_data_ready      (core_wr_data_ready),
        .shell_wr_data            (core_wr_data),
        .shell_wr_data_last       (core_wr_data_last),
        .shell_wr_data_tag        (core_wr_data_tag),
        .shell_wr_rsp_valid       (core_wr_rsp_valid),
        .shell_wr_rsp_ready       (core_wr_rsp_ready),
        .shell_wr_rsp_error       (core_wr_rsp_error),
        .shell_wr_rsp_aborted     (core_wr_rsp_aborted),
        .shell_wr_rsp_error_code  (core_wr_rsp_error_code),
        .shell_wr_rsp_tag         (core_wr_rsp_tag)
    );

    // mixed-engine fault reporting: sequencer faults must reach the frozen ABI status/error CSRs.
    assign core_error                 = seq_error;
    assign core_error_code            = seq_error_code;
    assign dbg_seq_error              = seq_error;
    assign dbg_seq_error_code         = seq_error_code;
    assign dbg_record_done            = seq_record_done;
    assign dbg_record_index           = seq_record_index;

    assign dbg_core_start            = core_start;
    assign dbg_core_soft_reset       = core_soft_reset;
    assign dbg_core_abort            = core_abort;
    assign dbg_core_busy             = core_busy;
    assign dbg_core_done             = core_done;
    assign dbg_core_error            = core_error;
    assign dbg_core_error_code       = core_error_code;
    assign dbg_core_rd_cmd_valid     = core_rd_cmd_valid;
    assign dbg_core_rd_cmd_ready     = core_rd_cmd_ready;
    assign dbg_core_rd_cmd_tag       = core_rd_cmd_tag;
    assign dbg_core_rd_cmd_bytes     = core_rd_cmd_bytes;
    assign dbg_core_rd_cmd_addr_byte = core_rd_cmd_addr_byte;
    assign dbg_core_rd_data_valid    = core_rd_data_valid;
    assign dbg_core_rd_data_ready    = core_rd_data_ready;
    assign dbg_core_rd_data_last     = core_rd_data_last;
    assign dbg_core_rd_data_tag      = core_rd_data_tag;
    assign dbg_core_wr_cmd_valid     = core_wr_cmd_valid;
    assign dbg_core_wr_cmd_ready     = core_wr_cmd_ready;
    assign dbg_core_wr_cmd_tag       = core_wr_cmd_tag;
    assign dbg_core_wr_cmd_bytes     = core_wr_cmd_bytes;
    assign dbg_core_wr_cmd_addr_byte = core_wr_cmd_addr_byte;
    assign dbg_core_wr_data_valid    = core_wr_data_valid;
    assign dbg_core_wr_data_ready    = core_wr_data_ready;
    assign dbg_core_wr_data_last     = core_wr_data_last;
    assign dbg_core_wr_data_tag      = core_wr_data_tag;
    assign dbg_core_wr_rsp_valid     = core_wr_rsp_valid;
    assign dbg_core_wr_rsp_ready     = core_wr_rsp_ready;
    assign dbg_core_wr_rsp_tag       = core_wr_rsp_tag;

    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_shell_cfg = |{cfg_in_addr_byte, cfg_weight_addr_byte, cfg_bias_addr_byte,
                              cfg_quant_addr_byte, cfg_out_addr_byte, cfg_h, cfg_w, cfg_cin,
                              cfg_cout, cfg_stride, cfg_pad, cfg_x_zero_point,
                              cfg_in_row_stride_bytes, cfg_out_row_stride_bytes, cfg_in_bytes,
                              cfg_weight_bytes, cfg_bias_bytes, cfg_quant_bytes, cfg_out_bytes,
                              cfg_conv_cfg, cfg_quant_cfg, cfg_act_cfg,
                              core_rd_data, core_wr_data, core_wr_rsp_error,
                              core_wr_rsp_aborted, core_wr_rsp_error_code};
    /* verilator lint_on UNUSEDSIGNAL */

endmodule



