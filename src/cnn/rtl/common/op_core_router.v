// -----------------------------------------------------------------------------
// op_core_router.v
//
// Generic ABI 1.0 core router for the project_3 mixed NPU engine.
//
// The frozen op_shell exposes exactly one "op_core" boundary:
//   * control/status: core_start / core_busy / core_done / core_error
//   * one normalized memory service (row read commands + row write commands).
//
// A generated network may contain more than one *reusable* operator core (for
// project_3: conv3x3_core and maxpool2x2_core).  This router lets a single,
// frozen op_shell drive either core one command at a time, so the PS keeps the
// same CSR map and still issues exactly one START per tile/layer, and each core
// keeps its "one row, one command" boundary.
//
// It does NOT touch ABI 1.0:
//   * no new / changed CSR offsets, layout, quantization semantics or codes,
//   * no capacity-as-runtime-CSR (capacity stays compile-time metadata),
//   * the 4 KB page / burst split remains entirely in the transport layer
//     (rtl/common/axi4_4kb_splitter.v inside the shell's read/write engines).
//
// Selection (latched on core_start, held until the next core_start):
//   pool <=> cfg_weight_bytes == 0 && cfg_bias_bytes == 0 && cfg_quant_bytes == 0
//   conv <=> otherwise
//
// A maxpool2x2 layer has no weight/bias/quant blob at all, while a DDR-weight
// conv always has weight_bytes >= 9*8.  A mixed conv+pool build therefore
// requires WEIGHT_MODE=ddr for its conv layers; npu_gen enforces that when a
// net contains pool layers, so local-register weights cannot alias the pool
// selector.
//
// Core A is the conv core, core B is the pool core.  Naming below is written
// from the *core's* point of view: "a_rd_cmd_valid" is an output of core A and
// therefore an input of this router.
// -----------------------------------------------------------------------------

module op_core_router (
    input  wire        clk,
    input  wire        rst_n,

    // ---- START-time configuration snapshot (from op_shell) -----------------
    input  wire [31:0] cfg_weight_bytes,
    input  wire [31:0] cfg_bias_bytes,
    input  wire [31:0] cfg_quant_bytes,
    output wire        sel_pool,

    // ---- op_shell -> core: control/status ---------------------------------
    input  wire        core_start,
    input  wire        core_soft_reset,
    input  wire        core_abort,
    output wire        core_busy,
    output wire        core_done,
    output wire        core_error,
    output wire [15:0] core_error_code,

    // ---- op_shell -> core: normalized read service ------------------------
    input  wire        rd_cmd_ready,
    output wire        rd_cmd_valid,
    output wire [31:0] rd_cmd_addr_byte,
    output wire [15:0] rd_cmd_bytes,
    output wire [7:0]  rd_cmd_tag,
    input  wire        rd_data_valid,
    input  wire [127:0] rd_data,
    input  wire        rd_data_last,
    input  wire        rd_data_error,
    input  wire [15:0] rd_data_error_code,
    input  wire [7:0]  rd_data_tag,
    output wire        rd_data_ready,

    // ---- op_shell -> core: normalized write service -----------------------
    input  wire        wr_cmd_ready,
    output wire        wr_cmd_valid,
    output wire [31:0] wr_cmd_addr_byte,
    output wire [15:0] wr_cmd_bytes,
    output wire [7:0]  wr_cmd_tag,
    input  wire        wr_data_ready,
    output wire        wr_data_valid,
    output wire [127:0] wr_data,
    output wire        wr_data_last,
    output wire [7:0]  wr_data_tag,
    input  wire        wr_rsp_valid,
    input  wire        wr_rsp_error,
    input  wire        wr_rsp_aborted,
    input  wire [15:0] wr_rsp_error_code,
    input  wire [7:0]  wr_rsp_tag,
    output wire        wr_rsp_ready,

    // ---- core A (conv) boundary -------------------------------------------
    output wire        a_start,
    output wire        a_soft_reset,
    output wire        a_abort,
    input  wire        a_busy,
    input  wire        a_done,
    input  wire        a_error,
    input  wire [15:0] a_error_code,

    output wire        a_rd_cmd_ready,
    input  wire        a_rd_cmd_valid,
    input  wire [31:0] a_rd_cmd_addr_byte,
    input  wire [15:0] a_rd_cmd_bytes,
    input  wire [7:0]  a_rd_cmd_tag,
    output wire        a_rd_data_valid,
    output wire [127:0] a_rd_data,
    output wire        a_rd_data_last,
    output wire        a_rd_data_error,
    output wire [15:0] a_rd_data_error_code,
    output wire [7:0]  a_rd_data_tag,
    input  wire        a_rd_data_ready,

    output wire        a_wr_cmd_ready,
    input  wire        a_wr_cmd_valid,
    input  wire [31:0] a_wr_cmd_addr_byte,
    input  wire [15:0] a_wr_cmd_bytes,
    input  wire [7:0]  a_wr_cmd_tag,
    output wire        a_wr_data_ready,
    input  wire        a_wr_data_valid,
    input  wire [127:0] a_wr_data,
    input  wire        a_wr_data_last,
    input  wire [7:0]  a_wr_data_tag,
    output wire        a_wr_rsp_valid,
    output wire        a_wr_rsp_error,
    output wire        a_wr_rsp_aborted,
    output wire [15:0] a_wr_rsp_error_code,
    output wire [7:0]  a_wr_rsp_tag,
    input  wire        a_wr_rsp_ready,

    // ---- core B (pool) boundary (identical shape) -------------------------
    output wire        b_start,
    output wire        b_soft_reset,
    output wire        b_abort,
    input  wire        b_busy,
    input  wire        b_done,
    input  wire        b_error,
    input  wire [15:0] b_error_code,

    output wire        b_rd_cmd_ready,
    input  wire        b_rd_cmd_valid,
    input  wire [31:0] b_rd_cmd_addr_byte,
    input  wire [15:0] b_rd_cmd_bytes,
    input  wire [7:0]  b_rd_cmd_tag,
    output wire        b_rd_data_valid,
    output wire [127:0] b_rd_data,
    output wire        b_rd_data_last,
    output wire        b_rd_data_error,
    output wire [15:0] b_rd_data_error_code,
    output wire [7:0]  b_rd_data_tag,
    input  wire        b_rd_data_ready,

    output wire        b_wr_cmd_ready,
    input  wire        b_wr_cmd_valid,
    input  wire [31:0] b_wr_cmd_addr_byte,
    input  wire [15:0] b_wr_cmd_bytes,
    input  wire [7:0]  b_wr_cmd_tag,
    output wire        b_wr_data_ready,
    input  wire        b_wr_data_valid,
    input  wire [127:0] b_wr_data,
    input  wire        b_wr_data_last,
    input  wire [7:0]  b_wr_data_tag,
    output wire        b_wr_rsp_valid,
    output wire        b_wr_rsp_error,
    output wire        b_wr_rsp_aborted,
    output wire [15:0] b_wr_rsp_error_code,
    output wire [7:0]  b_wr_rsp_tag,
    input  wire        b_wr_rsp_ready
);

    // -----------------------------------------------------------------------
    // Selection: no weight/bias/quant blob => pool.  Latched at core_start so
    // the choice is stable for the whole run and cannot change mid-command.
    // -----------------------------------------------------------------------
    wire sel_pool_comb = (cfg_weight_bytes == 32'd0) &&
                         (cfg_bias_bytes   == 32'd0) &&
                         (cfg_quant_bytes  == 32'd0);

    reg sel_pool_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sel_pool_r <= 1'b0;
        end else if (core_start) begin
            sel_pool_r <= sel_pool_comb;
        end
    end
    assign sel_pool = sel_pool_r;

    // -----------------------------------------------------------------------
    // Control / status.
    // -----------------------------------------------------------------------
    assign a_start      = core_start & ~sel_pool_r;
    assign b_start      = core_start &  sel_pool_r;
    assign a_soft_reset = core_soft_reset;   // both cores idle at reset time
    assign b_soft_reset = core_soft_reset;
    assign a_abort      = core_abort & ~sel_pool_r;
    assign b_abort      = core_abort &  sel_pool_r;

    assign core_busy       = sel_pool_r ? b_busy       : a_busy;
    assign core_done       = sel_pool_r ? b_done       : a_done;
    assign core_error      = sel_pool_r ? b_error      : a_error;
    assign core_error_code = sel_pool_r ? b_error_code : a_error_code;

    // -----------------------------------------------------------------------
    // Read service mux.  Ready is broadcast (only the running core can assert
    // a valid command); data/last/error/tag are delivered to the selected core
    // only, so an idle core can never see a stale response.
    // -----------------------------------------------------------------------
    assign rd_cmd_valid        = sel_pool_r ? b_rd_cmd_valid        : a_rd_cmd_valid;
    assign rd_cmd_addr_byte    = sel_pool_r ? b_rd_cmd_addr_byte    : a_rd_cmd_addr_byte;
    assign rd_cmd_bytes        = sel_pool_r ? b_rd_cmd_bytes        : a_rd_cmd_bytes;
    assign rd_cmd_tag          = sel_pool_r ? b_rd_cmd_tag          : a_rd_cmd_tag;
    assign rd_data_ready       = sel_pool_r ? b_rd_data_ready       : a_rd_data_ready;

    assign a_rd_cmd_ready      = rd_cmd_ready;
    assign b_rd_cmd_ready      = rd_cmd_ready;

    assign a_rd_data_valid     = rd_data_valid & ~sel_pool_r;
    assign a_rd_data           = rd_data;
    assign a_rd_data_last      = rd_data_last;
    assign a_rd_data_error     = rd_data_error;
    assign a_rd_data_error_code= rd_data_error_code;
    assign a_rd_data_tag       = rd_data_tag;

    assign b_rd_data_valid     = rd_data_valid &  sel_pool_r;
    assign b_rd_data           = rd_data;
    assign b_rd_data_last      = rd_data_last;
    assign b_rd_data_error     = rd_data_error;
    assign b_rd_data_error_code= rd_data_error_code;
    assign b_rd_data_tag       = rd_data_tag;

    // -----------------------------------------------------------------------
    // Write service mux (command, data and completion response).
    // -----------------------------------------------------------------------
    assign wr_cmd_valid        = sel_pool_r ? b_wr_cmd_valid        : a_wr_cmd_valid;
    assign wr_cmd_addr_byte    = sel_pool_r ? b_wr_cmd_addr_byte    : a_wr_cmd_addr_byte;
    assign wr_cmd_bytes        = sel_pool_r ? b_wr_cmd_bytes        : a_wr_cmd_bytes;
    assign wr_cmd_tag          = sel_pool_r ? b_wr_cmd_tag          : a_wr_cmd_tag;

    assign wr_data_valid       = sel_pool_r ? b_wr_data_valid       : a_wr_data_valid;
    assign wr_data             = sel_pool_r ? b_wr_data             : a_wr_data;
    assign wr_data_last        = sel_pool_r ? b_wr_data_last        : a_wr_data_last;
    assign wr_data_tag         = sel_pool_r ? b_wr_data_tag         : a_wr_data_tag;

    assign wr_rsp_ready        = sel_pool_r ? b_wr_rsp_ready        : a_wr_rsp_ready;

    assign a_wr_cmd_ready      = wr_cmd_ready;
    assign b_wr_cmd_ready      = wr_cmd_ready;
    assign a_wr_data_ready     = wr_data_ready;
    assign b_wr_data_ready     = wr_data_ready;

    assign a_wr_rsp_valid      = wr_rsp_valid & ~sel_pool_r;
    assign a_wr_rsp_error      = wr_rsp_error;
    assign a_wr_rsp_aborted    = wr_rsp_aborted;
    assign a_wr_rsp_error_code = wr_rsp_error_code;
    assign a_wr_rsp_tag        = wr_rsp_tag;

    assign b_wr_rsp_valid      = wr_rsp_valid &  sel_pool_r;
    assign b_wr_rsp_error      = wr_rsp_error;
    assign b_wr_rsp_aborted    = wr_rsp_aborted;
    assign b_wr_rsp_error_code = wr_rsp_error_code;
    assign b_wr_rsp_tag        = wr_rsp_tag;

endmodule
