
// -----------------------------------------------------------------------------
// copy_transform_core.v
//
// ABI 1.0 Pad / Slice / Concat transform core (opcodes 0x0012/0x0013/0x0014).
//
// v0.2 second batch (B2) rewrite.  The first-batch core staged the complete
// input tensor and the complete output tensor in two 16 KiB block RAMs and
// walked them one byte per cycle.  This revision streams one output row at a
// time:
//
//   * every (output row, pass) issues exactly ONE normalized read command for
//     the 16-byte aligned span of the input row that the pass really needs;
//   * the returned 128-bit beats feed an 8 x 128-bit sliding window register
//     bank that can present any 16 consecutive row bytes;
//   * a compose datapath produces one 128-bit output word per clock into an
//     output row buffer of MAX_ROW_BYTES/16 words with CHUNK_BYTES byte write
//     enables (Pad fills zero words locally, no whole-tensor clear);
//   * every output row is written exactly once with ONE normalized write
//     command (out_words beats, last on the final beat, one response).
//
// There is no whole-tensor staging, no whole-tensor clear and no byte-serial
// scan of the valid region.  H only extends the row loop / address range.
//
// On-chip storage (bounded by elaboration parameters):
//   u_out_row : MAX_ROW_BYTES/16 x DATA_WIDTH, 1 write port (CHUNK_BYTES byte
//               enables) + 1 synchronous read port (row drain), 1 clk latency.
//   u_win     : 8 x DATA_WIDTH register bank (asynchronous two-word read).
//   Peak simultaneously live rows = 1 (the output row being assembled).
//
// Configuration: the frozen 27-word payload is decoded exactly as in the first
// batch (same fields, same widths, START-time snapshot).
//
// Modes:
//   1 = Pad    (constant zero fill + spatial H/W copy)
//   2 = Slice  (static step=1 H/W/C gather, canonical NHWC8_A16 output)
//   3 = Concat (channel-axis gather from 2..4 inputs, logical C only)
//
// Error/reset/abort semantics follow the ABI 1.0 contract (contracts/copy.md
// sections 5/6).  Three debug counters (compose cycles, compose steps, compose
// read stalls) are exposed hierarchically for the unit TB; they are not part of
// the ABI and do not affect behaviour.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps
`include "op_abi_defs.vh"

module copy_transform_core #(
    parameter integer DATA_WIDTH    = 128,
    parameter integer CHUNK_BYTES   = 16,
    parameter integer MAX_ROW_BYTES = 16384,
    parameter integer MAX_H         = 0,
    parameter integer MAX_W         = 0,
    parameter integer MAX_CH        = 0,
    // v2 affine-int8-v1 records are 40 words; the table is sized to the 64-word
    // decode space so the wider copy_cfg_addr never silently wraps a field into
    // an earlier word.
    parameter [7:0]   CFG_WORDS     = 8'd64
) (
    input  wire        clk,
    input  wire        rst_n,

    // Configuration write port from the sequencer (via router).
    input  wire        copy_cfg_we,
    input  wire [5:0]  copy_cfg_addr,
    input  wire [31:0] copy_cfg_data,

    // Core lifecycle.
    input  wire        core_start,
    output reg         core_busy,
    output reg         core_done,
    output reg         core_error,
    output reg  [15:0] core_error_code,
    input  wire        core_soft_reset,
    input  wire        core_abort,

    // Normalized read service.
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

    // Normalized write service.
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

    localparam [7:0] MODE_PAD    = 8'd1;
    localparam [7:0] MODE_SLICE  = 8'd2;
    localparam [7:0] MODE_CONCAT = 8'd3;

    localparam [15:0] E_NONE          = `OP_E_NONE;
    localparam [15:0] E_CFG           = `OP_E_CFG;
    localparam [15:0] E_UNSUPPORTED   = `OP_E_UNSUPPORTED;
    localparam [15:0] E_ADDR_ALIGN    = `OP_E_ADDR_ALIGN;
    localparam [15:0] E_ADDR_RANGE    = `OP_E_ADDR_RANGE;
    localparam [15:0] E_STRIDE        = `OP_E_STRIDE;
    localparam [15:0] E_AXI_PROTOCOL  = `OP_E_AXI_PROTOCOL;
    localparam [15:0] E_CORE_INTERNAL = `OP_E_CORE_INTERNAL;

    localparam integer WORDS_MAX = (MAX_ROW_BYTES / CHUNK_BYTES);
    localparam integer AW        = (WORDS_MAX <= 2) ? 1 : $clog2(WORDS_MAX);
    localparam integer WIN_DEPTH = 8;

    localparam [7:0] TAG_RD = 8'h30;
    localparam [7:0] TAG_WR = 8'h40;

    // ------------------------------------------------------------------ FSM --
    localparam [4:0] S_IDLE       = 5'd0;
    localparam [4:0] S_BASE       = 5'd1;
    localparam [4:0] S_CHECK0     = 5'd2;
    localparam [4:0] S_CHECK1     = 5'd3;
    localparam [4:0] S_CHECK2     = 5'd4;
    localparam [4:0] S_CHECK3     = 5'd5;
    localparam [4:0] S_ROW_SETUP  = 5'd6;
    localparam [4:0] S_PASS_SETUP = 5'd7;
    localparam [4:0] S_RD_CMD     = 5'd8;
    localparam [4:0] S_STREAM     = 5'd9;
    localparam [4:0] S_PASS_DRAIN = 5'd10;
    localparam [4:0] S_NEXT_PASS  = 5'd11;
    localparam [4:0] S_WR_CMD     = 5'd12;
    localparam [4:0] S_WR_PRE     = 5'd13;
    localparam [4:0] S_WR_DATA    = 5'd14;
    localparam [4:0] S_WR_RSP     = 5'd15;
    localparam [4:0] S_NEXT_ROW   = 5'd16;
    localparam [4:0] S_DONE       = 5'd17;
    localparam [4:0] S_ERROR      = 5'd18;
    localparam [4:0] S_ABORT_RD   = 5'd19;
    localparam [4:0] S_ABORT_WAIT = 5'd20;
    localparam [4:0] S_ABORT_WD   = 5'd21;
    localparam [4:0] S_ABORT_RSP  = 5'd22;
    // affine-int8-v1 copy/transform v2 compose pipeline (versioned 40-word
    // record, word 23 bit16).  Descriptor, window, product, round/shift stages
    // are registered so no multiply/shift/saturate path is combinational with
    // the compose loop control.
    localparam [4:0] S_V2_D1      = 5'd23;
    localparam [4:0] S_V2_D2      = 5'd24;
    localparam [4:0] S_V2_D3      = 5'd25;
    localparam [4:0] S_V2_D4      = 5'd26;
    localparam [4:0] S_V2_D5      = 5'd27;

    // -------------------------------------------------- configuration words --
    reg [31:0] cfg [0:63];

    always @(posedge clk) begin
        if (copy_cfg_we && (copy_cfg_addr < CFG_WORDS))
            cfg[copy_cfg_addr] <= copy_cfg_data;
    end

    reg [7:0]  mode_q;
    reg [7:0]  n_inputs_q;
    reg [31:0] in_addr_q  [0:3];
    reg [31:0] out_addr_q;
    reg [31:0] in_bytes_q [0:3];
    reg [31:0] out_bytes_q;
    reg [31:0] in_row_q   [0:3];
    reg [31:0] out_row_q;
    reg [15:0] in_h_q, in_w_q, out_h_q, out_w_q;
    reg [15:0] in_c_q, out_c_q, in_ca_q, out_ca_q;
    reg [15:0] src_h_q, src_w_q, src_c_q;
    reg [15:0] dst_h_q, dst_w_q, dst_c_q;
    reg [15:0] ch_off_q  [0:3];
    reg [15:0] in_c_i_q  [0:3];
    reg [15:0] in_ca_i_q [0:3];
    reg [7:0]  zero_point_q;

    // ------------------------------------------------ affine-int8-v1 copy v2 --
    reg        fmt_v2_q;
    reg [15:0] in_ps_q    [0:3];   // input pixel stride (bytes)
    reg [15:0] out_ps_q;           // output pixel stride (bytes)
    reg signed [7:0] zs_out_q;     // output domain signed zero point
    reg signed [7:0] fill_q;       // encoded-zero fill code for C_alloc padding
    reg [31:0] rs_m0_q    [0:3];   // per-input requant multiplier
    reg [7:0]  rs_shift_q [0:3];   // per-input requant SHIFT
    reg signed [7:0] rs_zp_q [0:3];// per-input signed zero point
    reg        rs_en_q    [0:3];   // 1 = this input pass is re-scaled
    reg        rs_bypass_q[0:3];   // 1 = same-domain passthrough
    reg [15:0] wpp_q;              // 16B words per output pixel
    reg [5:0]  wip_q;              // word-in-pixel index
    reg [5:0]  wpp_m1_q;
    reg signed [31:0] pxbase_q;    // source offset of current pixel (rel. read base)
    reg [31:0] v2_rd_base_q;       // 16B-aligned read start of the pass row
    reg [15:0] v2_rd_span_q;       // read span in bytes (multiple of 16)
    reg [31:0] v2_span_al_q;       // registered align16 span for the deferred
                                   // MAX_ROW_BYTES check (timing: the check is
                                   // evaluated in the next state, not in the
                                   // pass-setup cone)
    reg [15:0] v2_px_stride_q;     // source pixel stride of the running pass
    reg        v2_prefill_q;       // this pass pre-fills the whole row with fill_q
    // ------------------------------------------- revision3 pass snapshots --
    // Captured in S_ROW_SETUP (explicit pass 0) / S_NEXT_PASS (explicit next
    // pass) so no v2 setup/runtime logic selects per-pass config arrays with
    // the live pass_q.  Derived constants are computed at capture time (one
    // cycle before S_PASS_SETUP) so S_PASS_SETUP can consume them without a
    // same-cycle non-blocking read.
    reg [31:0] sel_in_addr_q;
    reg [31:0] sel_in_row_q;
    reg [31:0] sel_in_bytes_q;
    reg [15:0] sel_in_ps_q;
    reg [15:0] sel_ci_q;
    reg [15:0] sel_ch_off_q;
    reg [16:0] sel_ch_end_q;
    reg [31:0] sel_rs_m0_q;
    reg [7:0]  sel_rs_shift_q;
    reg signed [7:0] sel_rs_zp_q;
    reg        sel_rs_en_q;
    reg        sel_rs_bypass_q;
    reg [32:0] sel_rd_limit_q;
    reg [63:0] sel_round_bias_q;
    reg        sel_rescale_q;
    // stage-1 registered compose descriptor
    reg [15:0] s1_mask_q;          // contribution bytes inside the output word
    reg [15:0] s1_wmask_q;         // bytes of the output word actually written
    reg [3:0]  s1_shift_q;         // left shift (bytes) into the output word
    reg signed [31:0] s1_src_q;    // source window offset (rel. read base)
    reg [4:0]  s1_n_q;             // contribution length
    reg [31:0] s1_obaddr_q;        // row-buffer word address
    reg [11:0] s1_ow_q;            // output word counter
    reg [127:0] s1_data_q;         // spread window bytes for the contribution
    // stage-2/3 rescale results
    reg signed [43:0] v2_prod_lo [0:7];
    reg signed [43:0] v2_prod_hi [0:7];
    reg [63:0] v2_res_lo;
    reg [63:0] v2_res_hi;

    // ------------------------------------------------------ derived geometry --
    reg [15:0] out_words_q;
    reg [16:0] qend_q;
    reg [16:0] lo_q;
    reg [16:0] hi_q;
    reg [16:0] span_pad_q;
    reg [16:0] span_slice_q;
    reg [16:0] ns_slice_q;
    reg [16:0] span_off_slice_q;
    // Staged configuration-check operands (D-timing r2): the span/align
    // arithmetic and the range/alignment comparisons are split across the
    // existing S_CHECK0 (stage A) and S_CHECK1 (stage B) cycles so the
    // cfg -> check_input -> core_error_code[*]/CE path is no longer one
    // 20-level combinational chain.  Each register is written once, in exactly
    // one state, before any read; no reset is required and there is no second
    // driver.  Widths stay 32 bit so legality/error behaviour is unchanged.
    reg [31:0] stage_a_pad_q;
    reg [31:0] stage_a_ns_q;
    reg [31:0] stage_a_prod_q;
    reg [31:0] stage_a_ci_q [0:3];
    reg [31:0] stage_b_pad_q;
    reg [31:0] stage_b_slice_q;
    reg [31:0] stage_b_ci_q [0:3];
    // rev3 S_STREAM per-compose-step descriptor registers (see
    // diagnostics/copy_timing/REV3_STREAM_DESCRIPTOR_PLAN.md).  Written once per
    // state in the main FSM (S_PASS_SETUP / S_RD_CMD load, S_STREAM advance),
    // never reset, single driver.
    reg        d_anymask;
    reg [15:0] d_comp_mask;
    reg [15:0] d_comp_wr_mask;
    reg [3:0]  d_comp_shift;
    reg [16:0] d_comp_w;
    reg [12:0] d_q0_word;
    reg [12:0] d_q1_word;

    // ------------------------------------------------------ run-time counters --
    reg [15:0] y_q;
    reg [31:0] out_row_addr_q;
    reg [31:0] in_cur_addr_q [0:3];
    reg [1:0]  pass_q;
    reg [2:0]  n_passes_q;   // 1..4 input passes (Concat 4 needs 3 bits)
    reg        row_data_q;

    reg [1:0]  ps_in_q;
    reg [16:0] ps_span_q;
    reg [16:0] ps_span_off_q;
    reg [12:0] ps_span_beats_q;
    reg [15:0] ps_chm_q;
    reg [15:0] ps_zl_q;
    reg [3:0]  ps_shift_q;
    reg [4:0]  ps_wstride_q;
    reg        ps_two_steps_q;

    reg [11:0] o_q;
    reg        step_q;
    reg [16:0] qbase_q;
    reg [16:0] w_q;

    reg [DATA_WIDTH-1:0] win_mem [0:WIN_DEPTH-1];
    reg [12:0] wq_wr;
    reg [12:0] wq_rd;
    reg [12:0] rd_beat_cnt;
    reg        rd_cmd_out;
    reg        wr_cmd_out;
    reg        abort_seen;
    reg [4:0]  state_q;
    reg [15:0] concat_sum_q;

    // --------------------------------------------------------- shift-add unit --
    reg [4:0]  bm_cnt;
    reg [15:0] bm_a;
    reg [31:0] bm_b;
    reg [31:0] bm_acc;

    // ---------------------------------------------------------- output row RAM --
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] out_row_mem [0:WORDS_MAX-1];
    reg [AW-1:0]          ob_waddr;
    reg [DATA_WIDTH-1:0]  ob_wdata;
    reg [CHUNK_BYTES-1:0] ob_we;
    reg [AW-1:0]          ob_raddr;
    reg [DATA_WIDTH-1:0]  ob_rdata;

    // ------------------------------------------------------- debug counters ----
`ifdef COPY_DEBUG
    reg [15:0] dbg_beat_q;
    always @(posedge clk) begin
        if (!rst_n) dbg_beat_q <= 16'd0;
        else begin
            dbg_beat_q <= dbg_beat_q + 16'd1;
            if ((dbg_beat_q[7:0] == 8'd0) && (state_q != S_IDLE))
                $display("COPYDBG_CYC t=%0t st=%0d y=%0d pass=%0d o=%0d wip=%0d wqrd=%0d wqwr=%0d rdo=%b rdcnt=%0d last=%b span=%h",
                         $time, state_q, y_q, pass_q, o_q, wip_q, wq_rd, wq_wr,
                         rd_cmd_out, rd_beat_cnt, rd_data_last, v2_rd_span_q);
        end
    end
`endif
    reg [31:0] dbg_compose_cycles;
    reg [31:0] dbg_compose_steps;
    reg [31:0] dbg_read_stall;

    // ----------------------------------------------------------- helpers ------
    function [15:0] low16;
        input [4:0] n;
        begin
            low16 = 16'hFFFF >> (5'd16 - n);
        end
    endfunction

    function [7:0] low8;
        input [3:0] n;
        begin
            low8 = 8'hFF >> (4'd8 - n);
        end
    endfunction

    function [31:0] shl_ca;
        input [15:0] v;
        input [15:0] ca;
        begin
            shl_ca = (ca == 16'd16) ? ({16'd0, v} << 4) : ({16'd0, v} << 3);
        end
    endfunction

    function [15:0] align8_c;
        input [15:0] c;
        reg [15:0] t;
        begin
            t = c + 16'd7;
            align8_c = {t[15:3], 3'b000};
        end
    endfunction

    function [31:0] align16f;
        input [31:0] v;
        begin
            align16f = (v + 32'd15) & 32'hFFFFFFF0;
        end
    endfunction

    function [3:0] prio16;
        input [15:0] m;
        reg [7:0] upper8;
        reg [3:0] upper4;
        reg [1:0] upper2;
        begin
            // A balanced leading-one tree keeps the mask-dependent path
            // logarithmic.  The empty mask retains the legacy result zero.
            prio16[3] = |m[15:8];
            upper8 = prio16[3] ? m[15:8] : m[7:0];
            prio16[2] = |upper8[7:4];
            upper4 = prio16[2] ? upper8[7:4] : upper8[3:0];
            prio16[1] = |upper4[3:2];
            upper2 = prio16[1] ? upper4[3:2] : upper4[1:0];
            prio16[0] = upper2[1];
        end
    endfunction

    function [DATA_WIDTH-1:0] expand_mask;
        input [CHUNK_BYTES-1:0] m;
        integer k;
        begin
            expand_mask = {DATA_WIDTH{1'b0}};
            for (k = 0; k < CHUNK_BYTES; k = k + 1)
                expand_mask[k*8 +: 8] = {8{m[k]}};
        end
    endfunction

    function [15:0] check_input;
        input [15:0] ca;
        input [15:0] c;
        input [31:0] a;
        input [31:0] row;
        input [31:0] bytes;
        input [31:0] span;
        reg [32:0] sum;
        begin
            check_input = E_NONE;
            sum = {1'b0, a} + {1'b0, bytes};
            if ((ca != 16'd8) && (ca != 16'd16))             check_input = E_CFG;
            else if ((ca < c) || (ca != align8_c(c)))        check_input = E_CFG;
            else if (a == 32'd0)                             check_input = E_ADDR_RANGE;
            else if (sum[32])                                check_input = E_ADDR_RANGE;
            else if ((a[3:0] != 4'd0) || (row[3:0] != 4'd0) ||
                     (bytes[3:0] != 4'd0))                    check_input = E_ADDR_ALIGN;
            else if ((row == 32'd0) || (bytes == 32'd0))     check_input = E_STRIDE;
            else if (span == 32'd0)                          check_input = E_CFG;
            else if (span > row)                             check_input = E_STRIDE;
            else if ((span > MAX_ROW_BYTES) ||
                     (row > MAX_ROW_BYTES))                  check_input = E_UNSUPPORTED;
        end
    endfunction

    // ------------------------------------------------- derived combinational --
    wire [31:0] v2_qend_w     = out_w_q * out_ps_q;
    wire [31:0] v2_out_row_al = align16f(v2_qend_w);
    wire [32:0] v2_rd_end_w   = {1'b0, v2_rd_base_q} + {1'b0, v2_rd_span_q};
    wire        v2_rd_range_bad = (v2_rd_end_w > sel_rd_limit_q);
    wire [32:0] out_end_w    = {1'b0, out_addr_q} + {1'b0, out_bytes_q};
    wire [31:0] qend_w       = shl_ca(out_w_q, out_ca_q);
    wire [31:0] out_row_al_w = align16f(qend_w);
    wire [31:0] lo_w         = shl_ca(dst_w_q, out_ca_q);
    wire [31:0] hi_w         = lo_w + shl_ca(in_w_q, out_ca_q);
    // D-timing r2: stage A (registered in S_CHECK0) - shifted/additive bases.
    // These are algebraically the same terms the unregistered spans used; only
    // the pipeline boundary moves.  32-bit widths are preserved.
    localparam [31:0] ALIGN15      = 32'd15;
    localparam [31:0] ALIGN16_MASK = 32'hFFFFFFF0;
    wire [31:0] stage_a_pad_w  = shl_ca(in_w_q - 16'd1, in_ca_q) + {16'd0, in_c_q};
    wire [31:0] stage_a_ns_w   = shl_ca(src_w_q, in_ca_q) + {16'd0, src_c_q};
    wire [31:0] stage_a_prod_w = shl_ca(out_w_q - 16'd1, in_ca_q) + {16'd0, out_c_q};
    wire [31:0] stage_a_ci_w0  = shl_ca(out_w_q - 16'd1, {8'd0, in_ca_i_q[0]}) + {16'd0, in_c_i_q[0]};
    wire [31:0] stage_a_ci_w1  = shl_ca(out_w_q - 16'd1, {8'd0, in_ca_i_q[1]}) + {16'd0, in_c_i_q[1]};
    wire [31:0] stage_a_ci_w2  = shl_ca(out_w_q - 16'd1, {8'd0, in_ca_i_q[2]}) + {16'd0, in_c_i_q[2]};
    wire [31:0] stage_a_ci_w3  = shl_ca(out_w_q - 16'd1, {8'd0, in_ca_i_q[3]}) + {16'd0, in_c_i_q[3]};

    // D-timing r2: stage B (registered in S_CHECK1) - final aligned spans.
    // span_pad = align16f(ne_pad) = (ne_pad + 15) & ~15.
    wire [31:0] stage_b_pad_w = (stage_a_pad_q + ALIGN15) & ALIGN16_MASK;
    // span_slice = align16f(ns + prod + out_c) - (ns & ~15)
    //            = ((ns[3:0] + (prod + out_c) + 15) & ~15); ns & ~15 is 16-B
    // aligned, so subtracting it from the aligned sum is the same rectangle.
    wire [4:0]  stage_b_ns_lo15_w = {1'b0, stage_a_ns_q[3:0]} + 5'd15;
    wire [31:0] stage_b_slice_w   = ({27'd0, stage_b_ns_lo15_w} + stage_a_prod_q) &
                                   ALIGN16_MASK;
    wire [31:0] stage_b_ci_w0 = (stage_a_ci_q[0] + ALIGN15) & ALIGN16_MASK;
    wire [31:0] stage_b_ci_w1 = (stage_a_ci_q[1] + ALIGN15) & ALIGN16_MASK;
    wire [31:0] stage_b_ci_w2 = (stage_a_ci_q[2] + ALIGN15) & ALIGN16_MASK;
    wire [31:0] stage_b_ci_w3 = (stage_a_ci_q[3] + ALIGN15) & ALIGN16_MASK;

    // ------------------------------------------------------------------------
    // rev3: S_STREAM per-compose-step descriptor generator.
    //
    // r2 spatial_A OOC worst path (5.739 ns / 22 levels) was the per-word stream
    // chain:  qbase_q -> range/qmask (qend-qbase, lo/hi clip) -> comp_mask ->
    // prio16 -> maxidx -> w_end -> q1_word -> win_ok -> o_q/qbase/state CE.
    // The arithmetic below is IDENTICAL to the previous unregistered version; it
    // now produces the descriptor of the NEXT compose step (S_STREAM: next
    // qbase/w_q/step from the values the FSM is about to load) or of the first
    // step (S_RD_CMD / S_PASS_SETUP: current word-0 values).  The descriptor is
    // registered; compose/write and the loop control then use only registered
    // values.  q1_word (d_q1_word) keeps the exact old win condition, so the
    // cycle-by-cycle behaviour is unchanged (1 word/cycle, identical cycles).
    // ------------------------------------------------------------------------
    wire        dg_in_stream = (state_q == S_STREAM);
    wire        dg_first     = (state_q == S_PASS_SETUP);

    // next base values (only meaningful when the current step is consumed)
    wire        dg_step = dg_in_stream ?
                          (d_anymask ? (ps_two_steps_q ? ~step_q : 1'b0) : step_q) :
                          (dg_first ? 1'b0 : step_q);
    wire [16:0] dg_qbase = dg_in_stream ?
                           ((d_anymask && ps_two_steps_q && !step_q) ?
                            qbase_q : (qbase_q + 17'd16)) :
                           (dg_first ? 17'd0 : qbase_q);
    wire [16:0] dg_w = dg_in_stream ?
                       (((d_anymask || (mode_q != MODE_PAD))) ?
                        (w_q + {12'd0, ps_wstride_q}) : w_q) :
                       (dg_first ?
                        ((mode_q == MODE_SLICE) ?
                         (ns_slice_q - span_off_slice_q) : 17'd0) : w_q);
    wire        dg_row_data = dg_first ?
                              ((mode_q == MODE_PAD) ? row_data_q : 1'b1) :
                              row_data_q;

    wire        dg_ge_qend  = ({1'b0, qend_q} >= {1'b0, dg_qbase});
    wire [17:0] dg_rem_qend = {1'b0, qend_q} - {1'b0, dg_qbase};
    wire [4:0]  dg_qm_n     = (!dg_ge_qend) ? 5'd0 :
                              ((dg_rem_qend >= 18'd16) ? 5'd16 : dg_rem_qend[4:0]);
    wire [15:0] dg_qmask    = low16(dg_qm_n);

    wire        dg_lo_below = ({1'b0, lo_q} > {1'b0, dg_qbase});
    wire [17:0] dg_lo_rem   = {1'b0, lo_q} - {1'b0, dg_qbase};
    wire [4:0]  dg_lo_clip  = (!dg_lo_below) ? 5'd0 :
                              ((dg_lo_rem >= 18'd16) ? 5'd16 : dg_lo_rem[4:0]);
    wire        dg_lt_hi    = ({1'b0, dg_qbase} < {1'b0, hi_q});
    wire [17:0] dg_hi_rem   = {1'b0, hi_q} - {1'b0, dg_qbase};
    wire [4:0]  dg_hi_clip  = (!dg_lt_hi) ? 5'd0 :
                              ((dg_hi_rem >= 18'd16) ? 5'd16 : dg_hi_rem[4:0]);
    wire [15:0] dg_range_mask = low16(dg_hi_clip) & ~low16(dg_lo_clip);

    wire        dg_pad_left    = ({1'b0, lo_q} > {1'b0, dg_qbase});
    wire [17:0] dg_pad_shift_w = dg_pad_left ?
                                 ({1'b0, lo_q} - {1'b0, dg_qbase}) : 18'd0;
    wire [16:0] dg_pad_w       = dg_pad_left ? 17'd0 :
                                 ({1'b0, dg_qbase} - {1'b0, lo_q});
    wire [3:0]  dg_pad_shift   = dg_pad_shift_w[3:0];

    wire [15:0] dg_half_half_chm = dg_step ? {ps_chm_q[7:0], 8'h00}
                                            : {8'h00, ps_chm_q[7:0]};
    wire [15:0] dg_half_half_all = dg_step ? 16'hFF00 : 16'h00FF;

    reg  [15:0] dg_comp_mask;
    reg  [15:0] dg_comp_wr_mask;
    reg  [3:0]  dg_comp_shift;
    reg  [16:0] dg_comp_w;
    always @* begin
        dg_comp_mask    = 16'd0;
        dg_comp_wr_mask = 16'd0;
        dg_comp_shift   = 4'd0;
        dg_comp_w       = 17'd0;
        case (mode_q)
            MODE_PAD: begin
                dg_comp_mask    = (dg_row_data ? dg_range_mask : 16'd0) & ps_chm_q;
                dg_comp_wr_mask = {CHUNK_BYTES{1'b1}};
                dg_comp_shift   = dg_pad_shift;
                dg_comp_w       = dg_pad_w;
            end
            MODE_SLICE: begin
                if (ps_two_steps_q) begin
                    dg_comp_mask    = dg_qmask & dg_half_half_chm;
                    dg_comp_wr_mask = dg_qmask & dg_half_half_all;
                    dg_comp_shift   = dg_step ? 4'd8 : 4'd0;
                end else begin
                    dg_comp_mask    = dg_qmask & ps_chm_q;
                    dg_comp_wr_mask = dg_qmask;
                    dg_comp_shift   = 4'd0;
                end
                dg_comp_w = dg_w;
            end
            MODE_CONCAT: begin
                dg_comp_mask    = dg_qmask & ps_chm_q;
                dg_comp_wr_mask = dg_qmask & (ps_chm_q |
                                              ((pass_q == 2'd0) ? ps_zl_q : 16'd0));
                dg_comp_shift   = ps_shift_q;
                dg_comp_w       = dg_w;
            end
            default: begin
            end
        endcase
    end

    wire        dg_anymask = |dg_comp_mask;
    wire [3:0]  dg_ihi     = prio16(dg_comp_mask);
    wire [4:0]  dg_maxidx  = (dg_anymask &&
                              ({1'b0, dg_ihi} >= {1'b0, dg_comp_shift})) ?
                             ({1'b0, dg_ihi} - {1'b0, dg_comp_shift}) : 5'd0;
    wire [12:0] dg_q0_word = dg_comp_w[16:4];
    // dg_maxidx is in [0,15].  Speculate the high-word increment in parallel
    // with mask/priority evaluation: the late mask-dependent carry selects
    // between two already-computed word addresses instead of traversing a
    // further 13-bit carry chain.  Truncation at bit 17 is unchanged.
    wire [4:0] dg_end_low = {1'b0, dg_comp_w[3:0]} + dg_maxidx;
    wire [12:0] dg_q0_next = dg_q0_word + 13'd1;
    wire [12:0] dg_q1_word = dg_end_low[4] ? dg_q0_next : dg_q0_word;

    // Signed-safe: the v2 descriptor may name a beat that has not arrived yet
    // (the source window is fed in order), so wq_rd can temporarily lead wq_wr.
    // An unguarded subtraction would wrap and permanently close the read gate.
    wire [12:0] win_used  = (wq_wr >= wq_rd) ? (wq_wr - wq_rd) : 13'd0;
    wire        win_space = (win_used < (WIN_DEPTH - 2));
    // exact previous win condition, now against the registered descriptor
    wire        win_ok    = (d_q1_word < wq_wr);

    // compose window/data path: registered descriptor only
    wire [DATA_WIDTH-1:0]   d_win_lo = win_mem[d_q0_word[2:0]];
    wire [DATA_WIDTH-1:0]   d_win_hi = win_mem[(d_q0_word + 13'd1) & 13'h7];
    wire [2*DATA_WIDTH-1:0] d_win_cat = {d_win_hi, d_win_lo};
    wire [2*DATA_WIDTH-1:0] d_win_sh  = d_win_cat >> {d_comp_w[3:0], 3'b000};
    wire [DATA_WIDTH-1:0]   d_window  = d_win_sh[DATA_WIDTH-1:0];
    wire [DATA_WIDTH-1:0]   d_win_spread = d_window << {d_comp_shift, 3'b000};
    wire [DATA_WIDTH-1:0]   d_comp_data  = d_win_spread & expand_mask(d_comp_mask);

    // --------------------------------------------------------- row tail mask --
    wire [4:0]  tail_n      = (qend_q[3:0] == 4'd0) ? 5'd16 : {1'b0, qend_q[3:0]};
    wire [15:0] tail_mask16 = low16(tail_n);
    wire [15:0] row_tail_mask = (out_words_q != 16'd0) &&
                                (o_q == out_words_q - 16'd1) ?
                                tail_mask16 : 16'hFFFF;


    // ---------------------------------------------- v2 per-word descriptor --
    // State carried across the compose pipeline.
    reg [15:0] px_q;               // output pixel index inside the row (PAD window)
    reg [4:0]  v2_start;           // contribution start byte inside the word
    reg [4:0]  v2_n;               // contribution length in bytes
    reg [31:0] v2_src;             // source window offset relative to read base
    reg        v2_contrib;         // a source contribution exists in this word
    reg [16:0] v2_dend;            // scratch: exclusive end
    reg [16:0] v2_tmp;

    wire [16:0] v2_woff   = {1'b0, o_q[11:0], 4'b0000};
    wire [16:0] v2_within = {1'b0, wip_q, 4'b0000};
    wire        v2_word_in = (v2_woff < qend_q);
    wire [15:0] v2_mask_full = (v2_n == 5'd0) ? 16'd0 : (low16(v2_n) << v2_start);
    wire        v2_any = |v2_mask_full;
    wire [17:0] v2_src_last = {1'b0, v2_src[16:0]} + {13'd0, v2_n} - 18'd1;
    wire [12:0] v2_cur_need = v2_src_last[16:4];

    always @* begin
        v2_start   = 5'd0;
        v2_n       = 5'd0;
        v2_src     = 32'd0;
        v2_contrib = 1'b0;
        v2_dend    = 17'd0;
        v2_tmp     = 17'd0;
        case (mode_q)
            MODE_SLICE: begin
                if (v2_word_in && ({1'b0, v2_within} < {1'b0, out_c_q})) begin
                    v2_tmp     = {1'b0, out_c_q} - v2_within;
                    v2_n       = (v2_tmp > 17'd16) ? 5'd16 : v2_tmp[4:0];
                    v2_start   = 5'd0;
                    v2_src     = pxbase_q + {15'd0, v2_within[15:0]};
                    v2_contrib = 1'b1;
                end
            end
            MODE_CONCAT: begin
                if (v2_word_in &&
                    ({1'b0, v2_within} < sel_ch_end_q) &&
                    ((v2_within + 17'd16) > {1'b0, sel_ch_off_q})) begin
                    if (v2_within >= {1'b0, sel_ch_off_q}) begin
                        v2_start = 5'd0;
                        v2_dend  = sel_ch_end_q - v2_within;
                        v2_src   = pxbase_q + (v2_within - {1'b0, sel_ch_off_q});
                    end else begin
                        v2_tmp   = {1'b0, sel_ch_off_q} - v2_within;
                        v2_start = v2_tmp[4:0];
                        v2_dend  = {1'b0, sel_ci_q};
                        v2_src   = pxbase_q;
                    end
                    if (v2_dend > 17'd16) begin
                        v2_n = 5'd16 - v2_start;
                    end else begin
                        v2_n = v2_dend[4:0] - v2_start;
                    end
                    v2_contrib = (v2_n != 5'd0);
                end
            end
            MODE_PAD: begin
                if (v2_word_in && row_data_q && (px_q >= dst_w_q) &&
                    (px_q < dst_w_q + in_w_q) &&
                    ({1'b0, v2_within} < {1'b0, in_c_q})) begin
                    v2_tmp     = {1'b0, in_c_q} - v2_within;
                    v2_n       = (v2_tmp > 17'd16) ? 5'd16 : v2_tmp[4:0];
                    v2_start   = 5'd0;
                    v2_src     = pxbase_q + {15'd0, v2_within[15:0]};
                    v2_contrib = 1'b1;
                end
            end
            default: begin
            end
        endcase
    end

    wire [15:0] v2_wmask = v2_prefill_q ? 16'hFFFF : v2_mask_full;
    wire        v2_write = v2_prefill_q || v2_any;

    // Window read for the current descriptor (only valid in S_V2_D1).
    wire [12:0]  v2_q0w    = s1_src_q[16:4];
    wire [3:0]   v2_boff   = s1_src_q[3:0];
    wire [127:0] v2_wlo    = win_mem[v2_q0w[2:0]];
    wire [127:0] v2_whi    = win_mem[(v2_q0w + 13'd1) & 13'h7];
    wire [255:0] v2_wcat   = {v2_whi, v2_wlo};
    wire [127:0] v2_wsel   = v2_wcat >> {v2_boff, 3'b000};
    wire [127:0] v2_wspread= v2_wsel << {s1_shift_q, 3'b000};
    wire [127:0] v2_fillv  = {16{fill_q}};
    wire [127:0] v2_bypass_word = (v2_wspread & expand_mask(s1_mask_q)) |
                                  (v2_fillv & expand_mask(s1_wmask_q & ~s1_mask_q));
    wire [17:0]  v2_src_end = {1'b0, s1_src_q[16:0]} + {13'd0, s1_n_q} - 18'd1;
    wire [12:0]  v2_need_beat = v2_src_end[16:4];
    wire         v2_win_ok = (s1_n_q == 5'd0) || (v2_need_beat < wq_wr);
    wire         v2_last_word = (s1_ow_q == out_words_q[11:0] - 12'd1);

    // Byte-lane requantisation (single rounding, ties away from zero).
    function signed [43:0] v2_product;
        input [7:0] q;
        input signed [7:0] zp;
        input [31:0] m0;
        reg signed [9:0] diff;
        begin
            // both operands are signed INT8 codes: sign-extend the byte, never
            // zero-extend it (0x80 must be -128, not +128).
            diff = $signed(q) - $signed(zp);
            v2_product = diff * $signed({1'b0, m0});
        end
    endfunction

    // revision4: the arithmetic is exactly representable in 45 bits for a
    // 44-bit signed product (|prod| < 2^43): the rounding bias for shift<=44
    // is < 2^44, mag+bias < 2^45, and for shift>=45 the quotient is exactly
    // 0 (proved by exhaustive equivalence: shift 0..255, prod/zp extremes).
    // Narrowing the carry chains 64->45 bits shortens the v2_res_hi round
    // path without changing a single output bit.
    function [7:0] v2_round_sat;
        input signed [43:0] prod;
        input [7:0] shift;
        input [63:0] bias;   // revision3: precomputed rounding bias (shift==0 -> 0)
        input signed [7:0] zpout;
        reg signed [44:0] p45;
        reg [44:0] mag, rounded, shifted;
        reg signed [44:0] sval, biased;
        begin
            p45 = {prod[43], prod};
            mag = p45[44] ? ((~p45) + 45'd1) : p45;
            if (shift == 8'd0) begin
                shifted = mag;
            end else if (shift >= 8'd45) begin
                shifted = 45'd0;
            end else begin
                rounded = mag + bias[44:0];
                shifted = rounded >> shift;
            end
            sval = p45[44] ? (45'sd0 - $signed({1'b0, shifted[43:0]}))
                           : $signed({1'b0, shifted[43:0]});
            biased = sval + $signed({{37{zpout[7]}}, zpout});
            if (biased > 45'sd127)      v2_round_sat = 8'h7F;
            else if (biased < -45'sd128) v2_round_sat = 8'h80;
            else                         v2_round_sat = biased[7:0];
        end
    endfunction

    // Stage-3/4 result word: re-scaled contribution bytes + encoded-zero fill.
    // high-half rounding is combinational on the registered products so the D4
    // cycle can write the finished word; the low half was rounded in D3.
    wire [63:0] v2_hi_comb = {
        v2_round_sat(v2_prod_hi[7], sel_rs_shift_q, sel_round_bias_q, zs_out_q),
        v2_round_sat(v2_prod_hi[6], sel_rs_shift_q, sel_round_bias_q, zs_out_q),
        v2_round_sat(v2_prod_hi[5], sel_rs_shift_q, sel_round_bias_q, zs_out_q),
        v2_round_sat(v2_prod_hi[4], sel_rs_shift_q, sel_round_bias_q, zs_out_q),
        v2_round_sat(v2_prod_hi[3], sel_rs_shift_q, sel_round_bias_q, zs_out_q),
        v2_round_sat(v2_prod_hi[2], sel_rs_shift_q, sel_round_bias_q, zs_out_q),
        v2_round_sat(v2_prod_hi[1], sel_rs_shift_q, sel_round_bias_q, zs_out_q),
        v2_round_sat(v2_prod_hi[0], sel_rs_shift_q, sel_round_bias_q, zs_out_q)};
    wire [127:0] v2_res_word = {v2_res_hi, v2_res_lo};
    wire [127:0] v2_d4_word = (v2_res_word & expand_mask(s1_mask_q)) |
                              (v2_fillv & expand_mask(s1_wmask_q & ~s1_mask_q));
    wire        v2_rescale_pass = sel_rescale_q;

    // ------------------------------------------------------ compose write port -
    always @* begin
        ob_waddr = o_q[AW-1:0];
        ob_we    = {CHUNK_BYTES{1'b0}};
        ob_wdata = {DATA_WIDTH{1'b0}};
        if (fmt_v2_q) begin
            // v2 compose writes one full output word from the registered
            // descriptor; only the descriptor write mask reaches the row RAM,
            // so a later pass never overwrites an earlier pass's contribution.
            if ((state_q == S_V2_D1) && !v2_rescale_pass) begin
                ob_waddr = s1_obaddr_q[AW-1:0];
                ob_we    = s1_wmask_q[CHUNK_BYTES-1:0];
                ob_wdata = v2_bypass_word;
            end else if (state_q == S_V2_D5) begin
                ob_waddr = s1_obaddr_q[AW-1:0];
                ob_we    = s1_wmask_q[CHUNK_BYTES-1:0];
                ob_wdata = v2_d4_word;
            end
        end else if (state_q == S_STREAM) begin
            if (d_anymask) begin
                ob_we    = d_comp_wr_mask;
                ob_wdata = d_comp_data;
            end else if (mode_q == MODE_PAD) begin
                ob_we    = {CHUNK_BYTES{1'b1}};
                ob_wdata = {DATA_WIDTH{1'b0}};
            end
        end
    end

    // --------------------------------------------------------- row RAM --------
    always @(posedge clk) begin
        if (ob_we[0])  out_row_mem[ob_waddr][0*8  +: 8] <= ob_wdata[0*8  +: 8];
        if (ob_we[1])  out_row_mem[ob_waddr][1*8  +: 8] <= ob_wdata[1*8  +: 8];
        if (ob_we[2])  out_row_mem[ob_waddr][2*8  +: 8] <= ob_wdata[2*8  +: 8];
        if (ob_we[3])  out_row_mem[ob_waddr][3*8  +: 8] <= ob_wdata[3*8  +: 8];
        if (ob_we[4])  out_row_mem[ob_waddr][4*8  +: 8] <= ob_wdata[4*8  +: 8];
        if (ob_we[5])  out_row_mem[ob_waddr][5*8  +: 8] <= ob_wdata[5*8  +: 8];
        if (ob_we[6])  out_row_mem[ob_waddr][6*8  +: 8] <= ob_wdata[6*8  +: 8];
        if (ob_we[7])  out_row_mem[ob_waddr][7*8  +: 8] <= ob_wdata[7*8  +: 8];
        if (ob_we[8])  out_row_mem[ob_waddr][8*8  +: 8] <= ob_wdata[8*8  +: 8];
        if (ob_we[9])  out_row_mem[ob_waddr][9*8  +: 8] <= ob_wdata[9*8  +: 8];
        if (ob_we[10]) out_row_mem[ob_waddr][10*8 +: 8] <= ob_wdata[10*8 +: 8];
        if (ob_we[11]) out_row_mem[ob_waddr][11*8 +: 8] <= ob_wdata[11*8 +: 8];
        if (ob_we[12]) out_row_mem[ob_waddr][12*8 +: 8] <= ob_wdata[12*8 +: 8];
        if (ob_we[13]) out_row_mem[ob_waddr][13*8 +: 8] <= ob_wdata[13*8 +: 8];
        if (ob_we[14]) out_row_mem[ob_waddr][14*8 +: 8] <= ob_wdata[14*8 +: 8];
        if (ob_we[15]) out_row_mem[ob_waddr][15*8 +: 8] <= ob_wdata[15*8 +: 8];
        ob_rdata <= out_row_mem[ob_raddr];
    end

    // -------------------------------------------------- row RAM read address --
    // Word o_q is prefetched one cycle before it is presented; the read address
    // therefore tracks o_q plus one when the current beat is accepted.
    always @* begin
        ob_raddr = {AW{1'b0}};
        if ((state_q == S_WR_DATA) || (state_q == S_ABORT_WD)) begin
            ob_raddr = wr_data_ready ? (o_q[AW-1:0] + 1) : o_q[AW-1:0];
        end
    end

    // ------------------------------------------------------- read beat checks -
    wire rd_fire    = rd_data_valid && rd_data_ready;
    // v2 passes carry their read span in v2_rd_span_q (bytes, multiple of 16);
    // the v1 ps_* beat registers are not written in that path.
    wire [12:0] rd_beats_w = fmt_v2_q ? {1'b0, v2_rd_span_q[15:4]} :
                                         ps_span_beats_q;
    wire rd_bad     = rd_fire && rd_cmd_out &&
                      (rd_data_error || (rd_data_tag != TAG_RD) ||
                       (rd_beat_cnt >= rd_beats_w) ||
                       (rd_data_last != ((rd_beat_cnt + 13'd1) ==
                                         rd_beats_w)));
    wire [15:0] rd_bad_code = rd_data_error ?
        ((rd_data_error_code == 16'd0) ? E_CORE_INTERNAL : rd_data_error_code) :
        E_AXI_PROTOCOL;

    // ------------------------------------------------------ service outputs ---
    wire [32:0] rd_span_end_w = {1'b0, in_cur_addr_q[ps_in_q]} +
                                {1'b0, ps_span_off_q} + {1'b0, ps_span_q};
    wire rd_range_bad = fmt_v2_q ? v2_rd_range_bad :
                        (rd_span_end_w > ({1'b0, in_addr_q[ps_in_q]} +
                                          {1'b0, in_bytes_q[ps_in_q]}));
    wire wr_range_bad = ({1'b0, out_row_addr_q} + {1'b0, out_row_q}) >
                        ({1'b0, out_addr_q} + {1'b0, out_bytes_q});

    always @* begin
        rd_cmd_valid     = 1'b0;
        rd_cmd_addr_byte = fmt_v2_q ? v2_rd_base_q :
                           (in_cur_addr_q[ps_in_q] + ps_span_off_q);
        rd_cmd_bytes     = fmt_v2_q ? v2_rd_span_q : ps_span_q[15:0];
        rd_cmd_tag       = TAG_RD;
        rd_data_ready    = 1'b0;
        wr_cmd_valid     = 1'b0;
        wr_cmd_addr_byte = out_row_addr_q;
        wr_cmd_bytes     = out_row_q[15:0];
        wr_cmd_tag       = TAG_WR;
        wr_data_valid    = 1'b0;
        wr_data          = ob_rdata & expand_mask(row_tail_mask);
        wr_data_last     = (o_q == out_words_q - 16'd1);
        wr_data_tag      = TAG_WR;
        wr_rsp_ready     = 1'b0;
        case (state_q)
            S_RD_CMD: begin
                if (!rd_range_bad) rd_cmd_valid = 1'b1;
            end
            S_STREAM:     rd_data_ready = rd_cmd_out && win_space;
            S_V2_D1, S_V2_D2, S_V2_D3, S_V2_D4:
                          rd_data_ready = rd_cmd_out && win_space;
            S_PASS_DRAIN: rd_data_ready = rd_cmd_out;
            S_ABORT_RD:   rd_data_ready = rd_cmd_out;
            S_WR_CMD: begin
                if (!wr_range_bad) wr_cmd_valid = 1'b1;
            end
            S_WR_DATA, S_ABORT_WD: begin
                wr_data_valid = 1'b1;
            end
            S_WR_RSP, S_ABORT_RSP: begin
                wr_rsp_ready = 1'b1;
            end
            default: begin
            end
        endcase
    end

    // ------------------------------------------------------------ debug/timing -
    always @(posedge clk) begin
        if (!rst_n) begin
            dbg_compose_cycles <= 32'd0;
            dbg_compose_steps  <= 32'd0;
            dbg_read_stall     <= 32'd0;
        end else if (state_q == S_STREAM) begin
            dbg_compose_cycles <= dbg_compose_cycles + 32'd1;
            if (d_anymask && !win_ok) dbg_read_stall <= dbg_read_stall + 32'd1;
            else                    dbg_compose_steps <= dbg_compose_steps + 32'd1;
        end
    end

    // ---------------------------------------------------------------- tasks ---
    task automatic raise_error;
        input [15:0] code;
        begin
`ifdef COPY_DEBUG
            $display("COPYDBG_ERR t=%0t state=%0d code=%04h fmt=%b span=%h beats=%h cnt=%0d last=%b tag=%h",
                     $time, state_q, code, fmt_v2_q, v2_rd_span_q, rd_beats_w,
                     rd_beat_cnt, rd_data_last, rd_data_tag);
`endif
            core_error      <= 1'b1;
            core_error_code <= code;
            core_busy       <= 1'b0;
            state_q         <= S_ERROR;
        end
    endtask

    // ------------------------------------------------------ main FSM ----------
    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= S_IDLE;
            core_busy <= 1'b0; core_done <= 1'b0;
            core_error <= 1'b0; core_error_code <= E_NONE;
            mode_q <= 8'd0; n_inputs_q <= 8'd0;
            out_addr_q <= 32'd0; out_bytes_q <= 32'd0; out_row_q <= 32'd0;
            in_h_q <= 16'd0; in_w_q <= 16'd0; out_h_q <= 16'd0; out_w_q <= 16'd0;
            in_c_q <= 16'd0; out_c_q <= 16'd0; in_ca_q <= 16'd0; out_ca_q <= 16'd0;
            src_h_q <= 16'd0; src_w_q <= 16'd0; src_c_q <= 16'd0;
            dst_h_q <= 16'd0; dst_w_q <= 16'd0; dst_c_q <= 16'd0;
            zero_point_q <= 8'd0;
            fmt_v2_q <= 1'b0;
            out_ps_q <= 16'd0;
            zs_out_q <= 8'sd0;
            fill_q   <= 8'sd0;
            wpp_q <= 16'd0;
            wpp_m1_q <= 16'd0;
            wip_q <= 6'd0;
            px_q <= 16'd0;
            pxbase_q <= 32'd0;
            v2_rd_base_q <= 32'd0;
            v2_rd_span_q <= 16'd0;
            v2_span_al_q <= 32'd0;
            v2_px_stride_q <= 16'd0;
            v2_prefill_q <= 1'b0;
            sel_in_addr_q <= 32'd0;
            sel_in_row_q <= 32'd0;
            sel_in_bytes_q <= 32'd0;
            sel_in_ps_q <= 16'd0;
            sel_ci_q <= 16'd0;
            sel_ch_off_q <= 16'd0;
            sel_ch_end_q <= 17'd0;
            sel_rs_m0_q <= 32'd0;
            sel_rs_shift_q <= 8'd0;
            sel_rs_zp_q <= 8'sd0;
            sel_rs_en_q <= 1'b0;
            sel_rs_bypass_q <= 1'b1;
            sel_rd_limit_q <= 33'd0;
            sel_round_bias_q <= 64'd0;
            sel_rescale_q <= 1'b0;
            s1_mask_q <= 16'd0;
            s1_wmask_q <= 16'd0;
            s1_shift_q <= 4'd0;
            s1_src_q <= 32'd0;
            s1_n_q <= 5'd0;
            s1_obaddr_q <= 32'd0;
            s1_ow_q <= 12'd0;
            s1_data_q <= 128'd0;
            v2_res_lo <= 64'd0;
            v2_res_hi <= 64'd0;
            for (k = 0; k < 4; k = k + 1) begin
                in_addr_q[k] <= 32'd0;
                in_bytes_q[k] <= 32'd0;
                in_row_q[k] <= 32'd0;
                ch_off_q[k] <= 16'd0;
                in_c_i_q[k] <= 16'd0;
                in_ca_i_q[k] <= 16'd0;
                in_ps_q[k] <= 16'd0;
                rs_m0_q[k] <= 32'd0;
                rs_shift_q[k] <= 8'd0;
                rs_zp_q[k] <= 8'sd0;
                rs_en_q[k] <= 1'b0;
                rs_bypass_q[k] <= 1'b1;
                in_cur_addr_q[k] <= 32'd0;
            end
            for (k = 0; k < 8; k = k + 1) begin
                v2_prod_lo[k] <= 44'sd0;
                v2_prod_hi[k] <= 44'sd0;
            end
            out_words_q <= 16'd0; qend_q <= 17'd0;
            lo_q <= 17'd0; hi_q <= 17'd0; span_pad_q <= 17'd0;
            span_slice_q <= 17'd0; ns_slice_q <= 17'd0; span_off_slice_q <= 17'd0;
            y_q <= 16'd0; out_row_addr_q <= 32'd0;
            pass_q <= 2'd0; n_passes_q <= 3'd0; row_data_q <= 1'b0;
            ps_in_q <= 2'd0; ps_span_q <= 17'd0; ps_span_off_q <= 17'd0;
            ps_span_beats_q <= 13'd0; ps_chm_q <= 16'd0; ps_zl_q <= 16'd0;
            ps_shift_q <= 4'd0; ps_wstride_q <= 5'd16; ps_two_steps_q <= 1'b0;
            o_q <= 12'd0; step_q <= 1'b0; qbase_q <= 17'd0; w_q <= 17'd0;
            win_mem[0] <= {DATA_WIDTH{1'b0}};
            win_mem[1] <= {DATA_WIDTH{1'b0}};
            win_mem[2] <= {DATA_WIDTH{1'b0}};
            win_mem[3] <= {DATA_WIDTH{1'b0}};
            win_mem[4] <= {DATA_WIDTH{1'b0}};
            win_mem[5] <= {DATA_WIDTH{1'b0}};
            win_mem[6] <= {DATA_WIDTH{1'b0}};
            win_mem[7] <= {DATA_WIDTH{1'b0}};
            wq_wr <= 13'd0; wq_rd <= 13'd0; rd_beat_cnt <= 13'd0;
            rd_cmd_out <= 1'b0; wr_cmd_out <= 1'b0; abort_seen <= 1'b0;
            concat_sum_q <= 16'd0;
            bm_cnt <= 5'd0; bm_a <= 16'd0; bm_b <= 32'd0; bm_acc <= 32'd0;
            // ob_raddr is a combinational variable driven by its own always @*
            // block below; ob_rdata is written unconditionally by the out_row_mem
            // read block.  Do NOT drive either net from this reset branch: doing so
            // made both multi-driven (Vivado preserves the constant GND driver and
            // drops the functional driver: Synth 8-6858/8-6859, copy RAM not
            // inferred, see predecessor run defects_found.md D1).
        end else begin
            core_done <= 1'b0;

            if (core_soft_reset) begin
                // soft reset: clear lifecycle state and drain any outstanding
                // normalized command (ABI 9.4); never raises an error.
                core_busy       <= 1'b0;
                core_done       <= 1'b0;
                core_error      <= 1'b0;
                core_error_code <= E_NONE;
                abort_seen      <= 1'b1;
                if (rd_cmd_out) begin
                    state_q <= S_ABORT_RD;
                end else if (wr_cmd_out) begin
                    state_q <= S_ABORT_WAIT;
                end else begin
                    state_q <= S_IDLE;
                end
            end else if (core_abort && state_q != S_IDLE && state_q != S_ERROR &&
                         state_q != S_ABORT_RD && state_q != S_ABORT_WAIT &&
                         state_q != S_ABORT_WD && state_q != S_ABORT_RSP) begin
                abort_seen <= 1'b1;
                core_busy  <= 1'b0;
                if (rd_cmd_out)      state_q <= S_ABORT_RD;
                else if (wr_cmd_out) state_q <= S_ABORT_WAIT;
                else                 state_q <= S_IDLE;
            end else begin
                // Common read-beat capture: valid for the v1 stream state and
                // for the v2 compose pipeline states (D1..D5), which keep the
                // read service flowing while a word is being re-scaled.
                if (rd_fire && rd_cmd_out) begin
                    win_mem[wq_wr[2:0]] <= rd_data;
                    wq_wr       <= wq_wr + 13'd1;
                    rd_beat_cnt <= rd_beat_cnt + 13'd1;
                    if (rd_data_last) rd_cmd_out <= 1'b0;
                end
                case (state_q)

                    S_IDLE: begin
                        core_busy <= 1'b0;
                        core_error <= 1'b0;
                        core_error_code <= E_NONE;
                        rd_cmd_out <= 1'b0;
                        wr_cmd_out <= 1'b0;
                        abort_seen <= 1'b0;
                        if (core_start) begin
`ifdef COPY_DEBUG
                            $display("COPYDBG_CFG c23=%h c31=%h c36=%h c37=%h c38=%h c39=%h c32=%h c18=%h",
                                     cfg[23], cfg[31], cfg[36], cfg[37], cfg[38], cfg[39], cfg[32], cfg[18]);
`endif
`ifdef COPY_DEBUG
                            $display("COPYDBG_START t=%0t mode=%0d n_in=%0d fmt=%b out_addr=%h out_hw=%h/%h out_c/ca=%0d/%0d ps=%0d inc/ca=%0d/%0d",
                                     $time, cfg[23][7:0], cfg[23][15:8], cfg[23][16],
                                     cfg[4], cfg[16][31:16], cfg[16][15:0],
                                     cfg[17][31:16], cfg[18][31:16], cfg[31][15:0],
                                     cfg[17][15:0], cfg[18][15:0]);
`endif
                            fmt_v2_q    <= cfg[23][16];
                            if (cfg[23][16]) begin
                                // versioned affine-int8-v1 40-word record
                                mode_q      <= cfg[23][7:0];
                                n_inputs_q  <= cfg[23][15:8];
                                in_addr_q[0] <= cfg[0]; in_addr_q[1] <= cfg[1];
                                in_addr_q[2] <= cfg[2]; in_addr_q[3] <= cfg[3];
                                out_addr_q  <= cfg[4];
                                in_bytes_q[0] <= cfg[5]; in_bytes_q[1] <= cfg[6];
                                in_bytes_q[2] <= cfg[7]; in_bytes_q[3] <= cfg[8];
                                out_bytes_q <= cfg[9];
                                in_row_q[0] <= cfg[10]; in_row_q[1] <= cfg[11];
                                in_row_q[2] <= cfg[12]; in_row_q[3] <= cfg[13];
                                out_row_q   <= cfg[14];
                                in_h_q      <= cfg[15][31:16];
                                in_w_q      <= cfg[15][15:0];
                                out_h_q     <= cfg[16][31:16];
                                out_w_q     <= cfg[16][15:0];
                                in_c_q      <= cfg[17][15:0];
                                out_c_q     <= cfg[17][31:16];
                                in_ca_q     <= cfg[18][15:0];
                                out_ca_q    <= cfg[18][31:16];
                                src_h_q     <= cfg[19][31:16];
                                src_w_q     <= cfg[19][15:0];
                                src_c_q     <= cfg[20][15:0];
                                dst_h_q     <= cfg[20][31:16];
                                dst_w_q     <= cfg[21][15:0];
                                dst_c_q     <= cfg[21][31:16];
                                ch_off_q[0] <= cfg[22][15:0];
                                ch_off_q[1] <= cfg[22][31:16];
                                ch_off_q[2] <= cfg[24][15:0];
                                ch_off_q[3] <= cfg[24][31:16];
                                in_c_i_q[0] <= cfg[25][15:0];
                                in_c_i_q[1] <= cfg[25][31:16];
                                in_c_i_q[2] <= cfg[26][15:0];
                                in_c_i_q[3] <= cfg[26][31:16];
                                in_ca_i_q[0] <= cfg[27][15:0];
                                in_ca_i_q[1] <= cfg[27][31:16];
                                in_ca_i_q[2] <= cfg[28][15:0];
                                in_ca_i_q[3] <= cfg[28][31:16];
                                in_ps_q[0]  <= cfg[29][15:0];
                                in_ps_q[1]  <= cfg[29][31:16];
                                in_ps_q[2]  <= cfg[30][15:0];
                                in_ps_q[3]  <= cfg[30][31:16];
                                out_ps_q    <= cfg[31][15:0];
                                zs_out_q    <= cfg[31][23:16];
                                fill_q      <= cfg[31][31:24];
                                rs_m0_q[0]  <= cfg[32];
                                rs_m0_q[1]  <= cfg[33];
                                rs_m0_q[2]  <= cfg[34];
                                rs_m0_q[3]  <= cfg[35];
                                rs_shift_q[0]  <= cfg[36][7:0];
                                rs_zp_q[0]     <= cfg[36][15:8];
                                rs_en_q[0]     <= cfg[36][16];
                                rs_bypass_q[0] <= cfg[36][17];
                                rs_shift_q[1]  <= cfg[37][7:0];
                                rs_zp_q[1]     <= cfg[37][15:8];
                                rs_en_q[1]     <= cfg[37][16];
                                rs_bypass_q[1] <= cfg[37][17];
                                rs_shift_q[2]  <= cfg[38][7:0];
                                rs_zp_q[2]     <= cfg[38][15:8];
                                rs_en_q[2]     <= cfg[38][16];
                                rs_bypass_q[2] <= cfg[38][17];
                                rs_shift_q[3]  <= cfg[39][7:0];
                                rs_zp_q[3]     <= cfg[39][15:8];
                                rs_en_q[3]     <= cfg[39][16];
                                rs_bypass_q[3] <= cfg[39][17];
                                zero_point_q <= cfg[26][7:0];
                            end else begin
                            fmt_v2_q    <= 1'b0;
                            mode_q      <= cfg[23][7:0];
                            n_inputs_q  <= cfg[23][15:8];
                            in_addr_q[0] <= cfg[0]; in_addr_q[1] <= cfg[1];
                            in_addr_q[2] <= cfg[2]; in_addr_q[3] <= cfg[3];
                            out_addr_q  <= cfg[4];
                            in_bytes_q[0] <= cfg[5]; in_bytes_q[1] <= cfg[6];
                            in_bytes_q[2] <= cfg[7]; in_bytes_q[3] <= cfg[8];
                            out_bytes_q <= cfg[9];
                            in_row_q[0] <= cfg[10]; in_row_q[1] <= cfg[11];
                            in_row_q[2] <= cfg[12]; in_row_q[3] <= cfg[13];
                            out_row_q   <= cfg[14];
                            in_h_q      <= cfg[15][31:16];
                            in_w_q      <= cfg[15][15:0];
                            out_h_q     <= cfg[16][31:16];
                            out_w_q     <= cfg[16][15:0];
                            in_c_q      <= cfg[17][15:0];
                            out_c_q     <= cfg[17][31:16];
                            in_ca_q     <= cfg[18][15:0];
                            out_ca_q    <= cfg[18][31:16];
                            src_h_q     <= cfg[19][31:16];
                            src_w_q     <= cfg[19][15:0];
                            src_c_q     <= cfg[20][15:0];
                            dst_h_q     <= cfg[20][31:16];
                            dst_w_q     <= cfg[21][15:0];
                            dst_c_q     <= cfg[21][31:16];
                            ch_off_q[0] <= cfg[22][7:0];
                            ch_off_q[1] <= cfg[22][15:8];
                            ch_off_q[2] <= cfg[22][23:16];
                            ch_off_q[3] <= cfg[22][31:24];
                            in_c_i_q[0] <= cfg[24][7:0];
                            in_c_i_q[1] <= cfg[24][15:8];
                            in_c_i_q[2] <= cfg[24][23:16];
                            in_c_i_q[3] <= cfg[24][31:24];
                            in_ca_i_q[0] <= cfg[25][7:0];
                            in_ca_i_q[1] <= cfg[25][15:8];
                            in_ca_i_q[2] <= cfg[25][23:16];
                            in_ca_i_q[3] <= cfg[25][31:24];
                            zero_point_q <= cfg[26][7:0];
                            // v1 has no explicit pixel-stride/rescale fields:
                            // derive them from C_alloc and keep passthrough.
                            in_ps_q[0]  <= {8'd0, cfg[25][7:0]};
                            in_ps_q[1]  <= (cfg[23][15:8] > 8'd1) ? {8'd0, cfg[25][15:8]} : 16'd0;
                            in_ps_q[2]  <= (cfg[23][15:8] > 8'd2) ? {8'd0, cfg[25][23:16]} : 16'd0;
                            in_ps_q[3]  <= (cfg[23][15:8] > 8'd3) ? {8'd0, cfg[25][31:24]} : 16'd0;
                            out_ps_q    <= cfg[18][31:16];
                            zs_out_q    <= 8'sd0;
                            fill_q      <= 8'sd0;
                            rs_m0_q[0]  <= 32'd0;
                            rs_m0_q[1]  <= 32'd0;
                            rs_m0_q[2]  <= 32'd0;
                            rs_m0_q[3]  <= 32'd0;
                            rs_shift_q[0]  <= 8'd0;
                            rs_shift_q[1]  <= 8'd0;
                            rs_shift_q[2]  <= 8'd0;
                            rs_shift_q[3]  <= 8'd0;
                            rs_zp_q[0]  <= 8'sd0;
                            rs_zp_q[1]  <= 8'sd0;
                            rs_zp_q[2]  <= 8'sd0;
                            rs_zp_q[3]  <= 8'sd0;
                            rs_en_q[0]  <= 1'b0;
                            rs_en_q[1]  <= 1'b0;
                            rs_en_q[2]  <= 1'b0;
                            rs_en_q[3]  <= 1'b0;
                            rs_bypass_q[0] <= 1'b1;
                            rs_bypass_q[1] <= 1'b1;
                            rs_bypass_q[2] <= 1'b1;
                            rs_bypass_q[3] <= 1'b1;
                            end
                            core_busy <= 1'b1;
                            y_q             <= 16'd0;
                            out_row_addr_q  <= cfg[4];
                            o_q             <= 12'd0;
                            qbase_q         <= 17'd0;
                            step_q          <= 1'b0;
                            concat_sum_q    <= 16'd0;
                            bm_cnt <= 5'd0; bm_acc <= 32'd0;
                            bm_a <= (cfg[23][7:0] == MODE_SLICE) ?
                                    cfg[19][31:16] : 16'd0;
                            bm_b <= cfg[10];
                            concat_sum_q <= 16'd0;
                            state_q <= S_BASE;
                        end
                    end

                    S_BASE: begin
                        if (bm_cnt == 5'd16) begin
                            in_cur_addr_q[0] <= in_addr_q[0] + bm_acc;
                            in_cur_addr_q[1] <= in_addr_q[1];
                            in_cur_addr_q[2] <= in_addr_q[2];
                            in_cur_addr_q[3] <= in_addr_q[3];
                            state_q <= S_CHECK0;
                        end else begin
                            if (bm_a[0]) bm_acc <= bm_acc + bm_b;
                            bm_a <= bm_a >> 1;
                            bm_b <= bm_b << 1;
                            bm_cnt <= bm_cnt + 5'd1;
                        end
                    end

                    S_CHECK0: begin
                        // stage A: register the span/align arithmetic bases (all
                        // inputs are START-time cfg snapshot regs; one cycle total).
                        stage_a_pad_q   <= stage_a_pad_w;
                        stage_a_ns_q    <= stage_a_ns_w;
                        stage_a_prod_q  <= stage_a_prod_w;
                        stage_a_ci_q[0] <= stage_a_ci_w0;
                        stage_a_ci_q[1] <= stage_a_ci_w1;
                        stage_a_ci_q[2] <= stage_a_ci_w2;
                        stage_a_ci_q[3] <= stage_a_ci_w3;
                        if (fmt_v2_q) begin
                            if ((mode_q != MODE_PAD) && (mode_q != MODE_SLICE) &&
                                (mode_q != MODE_CONCAT)) begin
                                raise_error(E_CFG);
                            end else if (mode_q == MODE_CONCAT) begin
                                if ((n_inputs_q < 8'd2) || (n_inputs_q > 8'd4))
                                    raise_error(E_CFG);
                                else if ((in_h_q != out_h_q) ||
                                         (in_w_q != out_w_q))
                                    raise_error(E_CFG);
                                else if ((out_ps_q == 16'd0) ||
                                         (out_ps_q[3:0] != 4'd0))
                                    raise_error(E_CFG);
                                else if (out_ps_q != out_ca_q)
                                    raise_error(E_CFG);
                                else
                                    state_q <= S_CHECK1;
                            end else if (n_inputs_q != 8'd1) begin
                                raise_error(E_CFG);
                            end else if ((in_h_q == 16'd0) || (in_w_q == 16'd0) ||
                                         (out_h_q == 16'd0) || (out_w_q == 16'd0) ||
                                         (in_c_q == 16'd0) || (out_c_q == 16'd0)) begin
                                raise_error(E_CFG);
                            end else if ((in_ca_q < in_c_q) ||
                                         (out_ca_q < out_c_q)) begin
                                raise_error(E_CFG);
                            end else if ((in_ca_q != align8_c(in_c_q)) ||
                                         (out_ca_q != align8_c(out_c_q))) begin
                                raise_error(E_CFG);
                            end else if ((out_ps_q == 16'd0) ||
                                         (out_ps_q[3:0] != 4'd0) ||
                                         (out_ps_q != out_ca_q) ||
                                         (in_ps_q[0] != in_ca_q)) begin
                                raise_error(E_CFG);
                            end else if (mode_q == MODE_PAD) begin
                                if ((in_c_q != out_c_q) ||
                                    (in_ca_q != out_ca_q))
                                    raise_error(E_UNSUPPORTED);
                                else
                                    state_q <= S_CHECK1;
                            end else begin
                                state_q <= S_CHECK1;
                            end
                        end else if (mode_q < MODE_PAD || mode_q > MODE_CONCAT) begin
                            raise_error(E_CFG);
                        end else if (mode_q == MODE_CONCAT) begin
                            if (n_inputs_q < 8'd2 || n_inputs_q > 8'd4)
                                raise_error(E_CFG);
                            else
                                state_q <= S_CHECK1;
                        end else if (n_inputs_q != 8'd1) begin
                            raise_error(E_CFG);
                        end else if (in_h_q == 16'd0 || in_w_q == 16'd0 ||
                                     out_h_q == 16'd0 || out_w_q == 16'd0 ||
                                     in_c_q == 16'd0 || out_c_q == 16'd0) begin
                            raise_error(E_CFG);
                        end else if (in_ca_q < in_c_q || out_ca_q < out_c_q) begin
                            raise_error(E_CFG);
                        end else if ((in_ca_q != 16'd8 && in_ca_q != 16'd16) ||
                                     (out_ca_q != 16'd8 && out_ca_q != 16'd16)) begin
                            raise_error(E_CFG);
                        end else if ((in_ca_q != align8_c(in_c_q)) ||
                                     (out_ca_q != align8_c(out_c_q))) begin
                            raise_error(E_CFG);
                        end else if (zero_point_q != 8'd0) begin
                            raise_error(E_CFG);
                        end else if (mode_q == MODE_PAD && (in_c_q != out_c_q)) begin
                            raise_error(E_CFG);
                        end else if (mode_q == MODE_CONCAT &&
                                     ((in_h_q != out_h_q) || (in_w_q != out_w_q))) begin
                            raise_error(E_CFG);
                        end else begin
                            state_q <= S_CHECK1;
                        end
                    end

                    S_CHECK1: begin
                        // stage B: register the final aligned spans; S_CHECK2 /
                        // S_CHECK3 compare these registers only.
                        stage_b_pad_q   <= stage_b_pad_w;
                        stage_b_slice_q <= stage_b_slice_w;
                        stage_b_ci_q[0] <= stage_b_ci_w0;
                        stage_b_ci_q[1] <= stage_b_ci_w1;
                        stage_b_ci_q[2] <= stage_b_ci_w2;
                        stage_b_ci_q[3] <= stage_b_ci_w3;
                        if (fmt_v2_q) begin
                            if (MAX_H != 0 && (in_h_q > MAX_H || out_h_q > MAX_H)) begin
                                raise_error(E_UNSUPPORTED);
                            end else if (MAX_W != 0 && (in_w_q > MAX_W || out_w_q > MAX_W)) begin
                                raise_error(E_UNSUPPORTED);
                            end else if (MAX_CH != 0 &&
                                         (in_c_q > MAX_CH || in_ca_q > MAX_CH ||
                                          out_c_q > MAX_CH || out_ca_q > MAX_CH)) begin
                                raise_error(E_UNSUPPORTED);
                            end else if ((out_addr_q == 32'd0) ||
                                         (out_addr_q[3:0] != 4'd0)) begin
                                raise_error(E_ADDR_RANGE);
                            end else if ((out_row_q == 32'd0) ||
                                         (out_row_q[3:0] != 4'd0) ||
                                         (out_bytes_q == 32'd0) ||
                                         (out_bytes_q[3:0] != 4'd0)) begin
                                raise_error(E_STRIDE);
                            end else if (out_end_w[32]) begin
                                raise_error(E_ADDR_RANGE);
                            end else if ((out_row_q > MAX_ROW_BYTES) ||
                                         (out_row_q != v2_out_row_al)) begin
                                raise_error(E_STRIDE);
                            end else begin
                                state_q <= S_CHECK2;
                            end
                        end else if (MAX_H != 0 && (in_h_q > MAX_H || out_h_q > MAX_H)) begin
                            raise_error(E_UNSUPPORTED);
                        end else if (MAX_W != 0 && (in_w_q > MAX_W || out_w_q > MAX_W)) begin
                            raise_error(E_UNSUPPORTED);
                        end else if (MAX_CH != 0 &&
                                     (in_c_q > MAX_CH || in_ca_q > MAX_CH ||
                                      out_c_q > MAX_CH || out_ca_q > MAX_CH)) begin
                            raise_error(E_UNSUPPORTED);
                        end else if (out_addr_q == 32'd0) begin
                            raise_error(E_ADDR_RANGE);
                        end else if ((out_addr_q[3:0] != 4'd0) ||
                                     (out_row_q[3:0] != 4'd0)) begin
                            raise_error(E_ADDR_ALIGN);
                        end else if (out_row_q == 32'd0 || out_bytes_q == 32'd0 ||
                                     (out_bytes_q[3:0] != 4'd0)) begin
                            raise_error(E_STRIDE);
                        end else if (out_end_w[32]) begin
                            raise_error(E_ADDR_RANGE);
                        end else if (out_row_q > MAX_ROW_BYTES) begin
                            raise_error(E_UNSUPPORTED);
                        end else if (out_row_q != out_row_al_w) begin
                            raise_error(E_STRIDE);
                        end else begin
                            state_q <= S_CHECK2;
                        end
                    end

                    S_CHECK2: begin
                        if (fmt_v2_q) begin
                            if (mode_q == MODE_SLICE) begin : v2_slice_chk
                                if ((src_h_q + out_h_q > in_h_q) ||
                                    (src_w_q + out_w_q > in_w_q) ||
                                    (src_c_q + out_c_q > in_c_q)) begin
                                    raise_error(E_CFG);
                                end else begin
                                    qend_q      <= v2_qend_w[16:0];
                                    out_words_q <= out_row_q[15:4];
                                    state_q     <= S_ROW_SETUP;
                                end
                            end else if (mode_q == MODE_PAD) begin : v2_pad_chk
                                if ((dst_h_q + in_h_q > out_h_q) ||
                                    (dst_w_q + in_w_q > out_w_q)) begin
                                    raise_error(E_CFG);
                                end else begin
                                    qend_q      <= v2_qend_w[16:0];
                                    out_words_q <= out_row_q[15:4];
                                    state_q     <= S_ROW_SETUP;
                                end
                            end else begin : v2_cat_chk
                                reg [15:0] sumc;
                                reg        bad;
                                begin
                                    sumc = 16'd0;
                                    bad  = 1'b0;
                                    for (k = 0; k < 4; k = k + 1) begin
                                        if (k < n_inputs_q) begin
                                            if ((in_c_i_q[k] == 16'd0) ||
                                                (in_c_i_q[k] > in_ca_i_q[k]) ||
                                                (in_ps_q[k] != in_ca_i_q[k]) ||
                                                ((ch_off_q[k] + in_c_i_q[k]) >
                                                 out_c_q)) begin
                                                bad = 1'b1;
                                            end
                                            sumc = sumc + in_c_i_q[k];
                                        end
                                    end
                                    if (bad) begin
                                        raise_error(E_CFG);
                                    end else if (ch_off_q[0] != 16'd0) begin
                                        raise_error(E_CFG);
                                    end else if (sumc != out_c_q) begin
                                        raise_error(E_CFG);
                                    end else begin
                                        qend_q      <= v2_qend_w[16:0];
                                        out_words_q <= out_row_q[15:4];
                                        state_q     <= S_ROW_SETUP;
                                    end
                                end
                            end
                        end else if (mode_q == MODE_PAD) begin
                            if (dst_h_q + in_h_q > out_h_q ||
                                dst_w_q + in_w_q > out_w_q) begin
                                raise_error(E_CFG);
                            end else begin : pad_chk
                                reg [15:0] code;
                                code = check_input(in_ca_q, in_c_q, in_addr_q[0],
                                                   in_row_q[0], in_bytes_q[0],
                                                   stage_b_pad_q);
                                if (code != E_NONE) begin
                                    raise_error(code);
                                end else begin
                                    span_pad_q  <= stage_b_pad_q[16:0];
                                    qend_q      <= qend_w[16:0];
                                    out_words_q <= out_row_q[15:4];
                                    lo_q        <= lo_w[16:0];
                                    hi_q        <= hi_w[16:0];
                                    state_q     <= S_ROW_SETUP;
                                end
                            end
                        end else if (mode_q == MODE_SLICE) begin
                            if (src_h_q + out_h_q > in_h_q ||
                                src_w_q + out_w_q > in_w_q ||
                                src_c_q + out_c_q > in_c_q) begin
                                raise_error(E_CFG);
                            end else begin : slice_chk
                                reg [15:0] code;
                                code = check_input(in_ca_q, in_c_q, in_addr_q[0],
                                                   in_row_q[0], in_bytes_q[0],
                                                   stage_b_slice_q);
                                if (code != E_NONE) begin
                                    raise_error(code);
                                end else begin
                                    span_slice_q     <= stage_b_slice_q[16:0];
                                    ns_slice_q       <= stage_a_ns_q[16:0];
                                    span_off_slice_q <= stage_a_ns_q[16:0] &
                                                        17'h1FFF0;
                                    qend_q           <= qend_w[16:0];
                                    out_words_q      <= out_row_q[15:4];
                                    state_q          <= S_ROW_SETUP;
                                end
                            end
                        end else begin
                            qend_q      <= qend_w[16:0];
                            out_words_q <= out_row_q[15:4];
                            bm_cnt      <= 5'd0;
                            concat_sum_q<= 16'd0;
                            state_q     <= S_CHECK3;
                        end
                    end

                    S_CHECK3: begin
                        if (bm_cnt == 5'd4 || bm_cnt >= {3'd0, n_inputs_q[2:0]}) begin
                            if (concat_sum_q != out_c_q) begin
                                raise_error(E_CFG);
                            end else begin
                                state_q <= S_ROW_SETUP;
                            end
                        end else begin : concat_chk
                            reg [15:0] code;
                            reg [15:0] ca_j;
                            reg [15:0] c_j;
                            reg [31:0] a_j, row_j, bytes_j, span_j;
                            begin
                                ca_j    = {8'd0, in_ca_i_q[bm_cnt[1:0]]};
                                c_j     = {8'd0, in_c_i_q[bm_cnt[1:0]]};
                                a_j     = in_addr_q[bm_cnt[1:0]];
                                row_j   = in_row_q[bm_cnt[1:0]];
                                bytes_j = in_bytes_q[bm_cnt[1:0]];
                                span_j = stage_b_ci_q[bm_cnt[1:0]];
                                code = check_input(ca_j, c_j, a_j, row_j,
                                                   bytes_j, span_j);
                                if (code != E_NONE) begin
                                    raise_error(code);
                                end else if (MAX_CH != 0 &&
                                             (c_j > MAX_CH || ca_j > MAX_CH)) begin
                                    raise_error(E_UNSUPPORTED);
                                end else if ({8'd0, ch_off_q[bm_cnt[1:0]]} +
                                             {8'd0, in_c_i_q[bm_cnt[1:0]]} >
                                             out_c_q) begin
                                    raise_error(E_CFG);
                                end else if (bm_cnt == 5'd0 &&
                                             ch_off_q[0] != 8'd0) begin
                                    raise_error(E_CFG);
                                end else if (bm_cnt != 5'd0 &&
                                             ({8'd0, ch_off_q[bm_cnt[1:0]]} !=
                                              {8'd0, ch_off_q[(bm_cnt - 5'd1) & 5'd3]} +
                                              {8'd0, in_c_i_q[(bm_cnt - 5'd1) & 5'd3]})) begin
                                    raise_error(E_CFG);
                                end else begin
                                    concat_sum_q <= concat_sum_q + c_j;
                                    bm_cnt       <= bm_cnt + 5'd1;
                                end
                            end
                        end
                    end

                    S_ROW_SETUP: begin
                        if (y_q == out_h_q) begin
                            state_q <= S_DONE;
                        end else begin
                            pass_q <= 2'd0;
                            o_q    <= 12'd0;
                            // revision3: snapshot the pass-0 config explicitly
                            sel_in_addr_q    <= in_addr_q[2'd0];
                            sel_in_row_q     <= in_row_q[2'd0];
                            sel_in_bytes_q   <= in_bytes_q[2'd0];
                            sel_in_ps_q      <= in_ps_q[2'd0];
                            sel_ci_q         <= in_c_i_q[2'd0];
                            sel_ch_off_q     <= ch_off_q[2'd0];
                            sel_ch_end_q     <= {1'b0, ch_off_q[2'd0]} +
                                                {1'b0, in_c_i_q[2'd0]};
                            sel_rs_m0_q      <= rs_m0_q[2'd0];
                            sel_rs_shift_q   <= rs_shift_q[2'd0];
                            sel_rs_zp_q      <= rs_zp_q[2'd0];
                            sel_rs_en_q      <= rs_en_q[2'd0];
                            sel_rs_bypass_q  <= rs_bypass_q[2'd0];
                            sel_rd_limit_q   <= {1'b0, in_addr_q[2'd0]} +
                                                {1'b0, in_bytes_q[2'd0]};
                            sel_round_bias_q <= (rs_shift_q[2'd0] == 8'd0) ? 64'd0 :
                                                (64'd1 << (rs_shift_q[2'd0] - 8'd1));
                            sel_rescale_q    <= rs_en_q[2'd0] && !rs_bypass_q[2'd0];
                            if (mode_q == MODE_PAD) begin
                                row_data_q <= (y_q >= dst_h_q) &&
                                              (y_q < dst_h_q + in_h_q);
                                n_passes_q <= 3'd1;
                            end else if (mode_q == MODE_SLICE) begin
                                row_data_q <= 1'b1;
                                n_passes_q <= 3'd1;
                            end else begin
                                row_data_q <= 1'b1;
                                n_passes_q <= {1'b0, n_inputs_q[2:0]};
                            end
                            state_q <= S_PASS_SETUP;
                        end
                    end

                    S_PASS_SETUP: begin
                        if (fmt_v2_q) begin : v2_pass_setup
                            reg [31:0] base_w;
                            reg [31:0] span_w;
                            reg [31:0] span_al_w;
                            reg [15:0] ps_p;
                            reg [15:0] c_need;
                            reg [15:0] npix;
                            begin
                                ps_p   = sel_in_ps_q;
                                npix   = out_w_q;
                                c_need = sel_ci_q;
                                if (mode_q == MODE_CONCAT) begin
                                    base_w = sel_in_addr_q +
                                             (y_q * sel_in_row_q);
                                end else if (mode_q == MODE_SLICE) begin
                                    base_w = in_addr_q[0] +
                                             ((y_q + src_h_q) * in_row_q[0]) +
                                             (src_w_q * in_ps_q[0]) + src_c_q;
                                    c_need = out_c_q;
                                end else begin
                                    base_w = row_data_q ?
                                             (in_addr_q[0] +
                                              ((y_q - dst_h_q) * in_row_q[0])) :
                                             32'd0;
                                    c_need = in_c_q;
                                end
                                span_w = (({16'd0, npix} - 32'd1) *
                                          {16'd0, ps_p}) + {16'd0, c_need} +
                                         {28'd0, base_w[3:0]};
                                span_al_w = align16f(span_w);
                                v2_rd_base_q   <= base_w & 32'hFFFFFFF0;
                                v2_rd_span_q   <= span_al_w[15:0];
                                v2_span_al_q   <= span_al_w;
                                v2_px_stride_q <= ps_p;
                                v2_prefill_q   <= (pass_q == 2'd0);
                                if (mode_q == MODE_PAD) begin
                                    pxbase_q <= {28'd0, base_w[3:0]} -
                                                (dst_w_q * in_ps_q[0]);
                                end else begin
                                    pxbase_q <= {28'd0, base_w[3:0]};
                                end
                                wpp_q    <= out_ps_q >> 4;
                                wpp_m1_q <= (out_ps_q >> 4) - 16'd1;
                                wip_q    <= 6'd0;
                                px_q     <= 16'd0;
                                o_q      <= 12'd0;
                                rd_beat_cnt <= 13'd0;
                                wq_wr    <= 13'd0;
                                wq_rd    <= 13'd0;
`ifdef COPY_DEBUG
                                $display("COPYDBG_PASS t=%0t y=%0d pass=%0d base=%h span=%h pxbase=%0d ps=%0d",
                                         $time, y_q, pass_q, base_w, span_al_w,
                                         (mode_q == MODE_PAD) ?
                                         ({28'd0, base_w[3:0]} - (dst_w_q * in_ps_q[0])) :
                                         {28'd0, base_w[3:0]}, ps_p);
`endif
                                // The MAX_ROW_BYTES rejection is checked from
                                // the registered v2_span_al_q in the next state
                                // (S_RD_CMD / S_STREAM).  Keeping the wide
                                // align16 span arithmetic out of this cycle's
                                // next-state/CE cone removes the only failing
                                // OOC path from pass_q[1]; valid-config timing
                                // and the state sequence are unchanged.
                                state_q <= (mode_q == MODE_PAD) ?
                                           (row_data_q ? S_RD_CMD : S_STREAM) :
                                           S_RD_CMD;
                            end
                        end else begin
                        begin : pass_derive
                            reg [15:0] ca_p;
                            reg [15:0] c_p;
                            reg [31:0] ne_w;
                            reg [31:0] span_w;
                            reg [15:0] chm_w;
                            reg [15:0] zl_w;
                            reg [16:0] span_use;
                            reg [16:0] off_use;
                            reg        rd_en;
                            begin
                                ca_p = (mode_q == MODE_CONCAT) ?
                                       {8'd0, in_ca_i_q[pass_q]} : in_ca_q;
                                c_p  = (mode_q == MODE_CONCAT) ?
                                       {8'd0, in_c_i_q[pass_q]} :
                                       ((mode_q == MODE_SLICE) ? out_c_q : in_c_q);
                                ne_w   = shl_ca(out_w_q - 16'd1, ca_p) + {16'd0, c_p};
                                span_w = align16f(ne_w);

                                chm_w = 16'd0;
                                zl_w  = 16'd0;
                                if (mode_q == MODE_CONCAT) begin
                                    if (out_ca_q == 16'd16) begin
                                        chm_w = low16({1'b0, ch_off_q[pass_q][3:0]} +
                                                      {1'b0, c_p[4:0]}) &
                                                ~low16({1'b0, ch_off_q[pass_q][3:0]});
                                        zl_w  = ~low16(out_c_q[4:0]);
                                    end else begin
                                        chm_w = { (low8({1'b0, ch_off_q[pass_q][3:0]} +
                                                        {1'b0, c_p[3:0]}) &
                                                   ~low8({1'b0, ch_off_q[pass_q][3:0]})),
                                                  (low8({1'b0, ch_off_q[pass_q][3:0]} +
                                                        {1'b0, c_p[3:0]}) &
                                                   ~low8({1'b0, ch_off_q[pass_q][3:0]})) };
                                        zl_w  = {~low8(out_c_q[3:0]),
                                                 ~low8(out_c_q[3:0])};
                                    end
                                end else if (mode_q == MODE_SLICE) begin
                                    if (out_ca_q == 16'd16)
                                        chm_w = low16(out_c_q[4:0]);
                                    else
                                        chm_w = {low8(out_c_q[3:0]),
                                                 low8(out_c_q[3:0])};
                                end else begin
                                    if (out_ca_q == 16'd16)
                                        chm_w = low16(in_c_q[4:0]);
                                    else
                                        chm_w = {low8(in_c_q[3:0]),
                                                 low8(in_c_q[3:0])};
                                end
                                span_use = (mode_q == MODE_PAD) ? span_pad_q :
                                           (mode_q == MODE_SLICE) ? span_slice_q :
                                           span_w[16:0];
                                off_use  = (mode_q == MODE_SLICE) ?
                                           span_off_slice_q : 17'd0;
                                rd_en    = (mode_q == MODE_PAD) ? row_data_q :
                                           1'b1;

                                ps_in_q         <= (mode_q == MODE_CONCAT) ?
                                                   pass_q : 2'd0;
                                ps_span_q       <= span_use;
                                ps_span_off_q   <= off_use;
                                ps_span_beats_q <= span_use[16:4];
                                ps_chm_q        <= chm_w;
                                ps_zl_q         <= zl_w;
                                ps_shift_q      <= (mode_q == MODE_CONCAT) ?
                                                   ch_off_q[pass_q][3:0] : 4'd0;
                                ps_two_steps_q  <= (mode_q == MODE_SLICE) &&
                                                   (in_ca_q != out_ca_q);
                                ps_wstride_q    <= 5'd16;
                                if (mode_q == MODE_CONCAT) begin
                                    ps_wstride_q <= (out_ca_q == 16'd16) ?
                                                    {1'b0, ca_p[4:0]} : 5'd16;
                                end
                                w_q     <= (mode_q == MODE_SLICE) ?
                                           (ns_slice_q - span_off_slice_q) : 17'd0;
                                o_q     <= 12'd0;
                                step_q  <= 1'b0;
                                qbase_q <= 17'd0;
                                rd_beat_cnt <= 13'd0;
                                // load the first-compose-step descriptor (word 0).
                                // For rd_en=1 this is overwritten in S_RD_CMD once
                                // ps_* are visible; for the PAD no-read row it is
                                // final and d_anymask is forced 0 by row_data_q.
                                d_anymask      <= dg_anymask;
                                d_comp_mask    <= dg_comp_mask;
                                d_comp_wr_mask <= dg_comp_wr_mask;
                                d_comp_shift   <= dg_comp_shift;
                                d_comp_w        <= dg_comp_w;
                                d_q0_word      <= dg_q0_word;
                                d_q1_word      <= dg_q1_word;
                                state_q <= rd_en ? S_RD_CMD : S_STREAM;
                            end
                        end
                        end
                    end

                    S_RD_CMD: begin
                        // first-compose-step descriptor (word 0; ps_* already valid)
                        if (!fmt_v2_q) begin
                            d_anymask      <= dg_anymask;
                            d_comp_mask    <= dg_comp_mask;
                            d_comp_wr_mask <= dg_comp_wr_mask;
                            d_comp_shift   <= dg_comp_shift;
                            d_comp_w        <= dg_comp_w;
                            d_q0_word      <= dg_q0_word;
                            d_q1_word      <= dg_q1_word;
                        end
                        if (fmt_v2_q && (v2_span_al_q > MAX_ROW_BYTES)) begin
                            raise_error(E_UNSUPPORTED);
                        end else if (rd_range_bad) begin
                            raise_error(E_STRIDE);
                        end else if (rd_cmd_valid && rd_cmd_ready) begin
                            rd_cmd_out  <= 1'b1;
                            rd_beat_cnt <= 13'd0;
                            wq_wr       <= 13'd0;
                            wq_rd       <= 13'd0;
                            state_q     <= S_STREAM;
                        end
                    end

                    S_STREAM: begin
                        if (fmt_v2_q) begin
                            if (v2_span_al_q > MAX_ROW_BYTES) begin
                                raise_error(E_UNSUPPORTED);
                            end else if (rd_bad) begin
                                raise_error(rd_bad_code);
                            end else if (!v2_write) begin
                                // no contribution in this word: advance only
                                if (wip_q == wpp_m1_q) begin
                                    wip_q    <= 6'd0;
                                    pxbase_q <= pxbase_q +
                                                $signed(v2_px_stride_q);
                                    px_q     <= px_q + 16'd1;
                                end else begin
                                    wip_q <= wip_q + 6'd1;
                                end
                                o_q <= o_q + 12'd1;
                                if (o_q == out_words_q - 16'd1)
                                    state_q <= S_PASS_DRAIN;
                            end else begin
                                s1_mask_q   <= v2_mask_full;
                                s1_wmask_q  <= v2_wmask;
                                s1_shift_q  <= v2_start;
                                s1_src_q    <= v2_src;
                                s1_n_q      <= v2_n;
                                s1_obaddr_q <= {20'd0, o_q};
                                s1_ow_q     <= o_q;
                                if (v2_any)
                                    wq_rd <= v2_cur_need;
                                if (wip_q == wpp_m1_q) begin
                                    wip_q    <= 6'd0;
                                    pxbase_q <= pxbase_q +
                                                $signed(v2_px_stride_q);
                                    px_q     <= px_q + 16'd1;
                                end else begin
                                    wip_q <= wip_q + 6'd1;
                                end
                                o_q <= o_q + 12'd1;
                                state_q <= S_V2_D1;
                            end
                        end else if (rd_bad) begin
                            raise_error(rd_bad_code);
                        end else begin
                            if (d_anymask && win_ok) begin
                                wq_rd <= d_q0_word;
                                w_q   <= w_q + {12'd0, ps_wstride_q};
                                if (ps_two_steps_q && !step_q) begin
                                    step_q <= 1'b1;
                                end else begin
                                    step_q  <= 1'b0;
                                    o_q     <= o_q + 12'd1;
                                    qbase_q <= qbase_q + 17'd16;
                                    if (o_q == out_words_q - 16'd1)
                                        state_q <= S_PASS_DRAIN;
                                end
                                // the generator already holds the next step
                                d_anymask      <= dg_anymask;
                                d_comp_mask    <= dg_comp_mask;
                                d_comp_wr_mask <= dg_comp_wr_mask;
                                d_comp_shift   <= dg_comp_shift;
                                d_comp_w        <= dg_comp_w;
                                d_q0_word      <= dg_q0_word;
                                d_q1_word      <= dg_q1_word;
                            end else if (!d_anymask) begin
                                // Pad zero words, or defensive skip for a word
                                // outside the logical rows (cannot happen for
                                // validated Slice/Concat configurations).
                                o_q     <= o_q + 12'd1;
                                qbase_q <= qbase_q + 17'd16;
                                if (mode_q != MODE_PAD)
                                    w_q <= w_q + {12'd0, ps_wstride_q};
                                if (o_q == out_words_q - 16'd1)
                                    state_q <= S_PASS_DRAIN;
                                d_anymask      <= dg_anymask;
                                d_comp_mask    <= dg_comp_mask;
                                d_comp_wr_mask <= dg_comp_wr_mask;
                                d_comp_shift   <= dg_comp_shift;
                                d_comp_w        <= dg_comp_w;
                                d_q0_word      <= dg_q0_word;
                                d_q1_word      <= dg_q1_word;
                            end
                        end
                    end

                    S_V2_D1: begin
                        if (rd_bad) begin
                            raise_error(rd_bad_code);
                        end else if (v2_win_ok) begin
`ifdef COPY_DEBUG
                            if (s1_ow_q < 12'd3)
                                $display("COPYDBG_D1 ow=%0d src=%0d n=%0d shift=%0d mask=%h wmask=%h data=%h wqrd=%0d wqwr=%0d",
                                         s1_ow_q, s1_src_q, s1_n_q, s1_shift_q,
                                         s1_mask_q, s1_wmask_q, v2_wspread, wq_rd, wq_wr);
`endif
                            s1_data_q <= v2_wspread;
                            if (!v2_rescale_pass) begin
                                // passthrough pass: word already assembled
                                if (v2_last_word) state_q <= S_PASS_DRAIN;
                                else              state_q <= S_STREAM;
                            end else begin
                                state_q <= S_V2_D2;
                            end
                        end
                    end

                    S_V2_D2: begin
                        if (rd_bad) begin
                            raise_error(rd_bad_code);
                        end else begin
                            v2_prod_lo[0] <= v2_product(s1_data_q[7:0],
                                sel_rs_zp_q, sel_rs_m0_q);
                            v2_prod_lo[1] <= v2_product(s1_data_q[15:8],
                                sel_rs_zp_q, sel_rs_m0_q);
                            v2_prod_lo[2] <= v2_product(s1_data_q[23:16],
                                sel_rs_zp_q, sel_rs_m0_q);
                            v2_prod_lo[3] <= v2_product(s1_data_q[31:24],
                                sel_rs_zp_q, sel_rs_m0_q);
                            v2_prod_lo[4] <= v2_product(s1_data_q[39:32],
                                sel_rs_zp_q, sel_rs_m0_q);
                            v2_prod_lo[5] <= v2_product(s1_data_q[47:40],
                                sel_rs_zp_q, sel_rs_m0_q);
                            v2_prod_lo[6] <= v2_product(s1_data_q[55:48],
                                sel_rs_zp_q, sel_rs_m0_q);
                            v2_prod_lo[7] <= v2_product(s1_data_q[63:56],
                                sel_rs_zp_q, sel_rs_m0_q);
                            state_q <= S_V2_D3;
                        end
                    end

                    S_V2_D3: begin
                        if (rd_bad) begin
                            raise_error(rd_bad_code);
                        end else begin
                            v2_res_lo[7:0]   <= v2_round_sat(v2_prod_lo[0],
                                sel_rs_shift_q, sel_round_bias_q, zs_out_q);
                            v2_res_lo[15:8]  <= v2_round_sat(v2_prod_lo[1],
                                sel_rs_shift_q, sel_round_bias_q, zs_out_q);
                            v2_res_lo[23:16] <= v2_round_sat(v2_prod_lo[2],
                                sel_rs_shift_q, sel_round_bias_q, zs_out_q);
                            v2_res_lo[31:24] <= v2_round_sat(v2_prod_lo[3],
                                sel_rs_shift_q, sel_round_bias_q, zs_out_q);
                            v2_res_lo[39:32] <= v2_round_sat(v2_prod_lo[4],
                                sel_rs_shift_q, sel_round_bias_q, zs_out_q);
                            v2_res_lo[47:40] <= v2_round_sat(v2_prod_lo[5],
                                sel_rs_shift_q, sel_round_bias_q, zs_out_q);
                            v2_res_lo[55:48] <= v2_round_sat(v2_prod_lo[6],
                                sel_rs_shift_q, sel_round_bias_q, zs_out_q);
                            v2_res_lo[63:56] <= v2_round_sat(v2_prod_lo[7],
                                sel_rs_shift_q, sel_round_bias_q, zs_out_q);
                            v2_prod_hi[0] <= v2_product(s1_data_q[71:64],
                                sel_rs_zp_q, sel_rs_m0_q);
                            v2_prod_hi[1] <= v2_product(s1_data_q[79:72],
                                sel_rs_zp_q, sel_rs_m0_q);
                            v2_prod_hi[2] <= v2_product(s1_data_q[87:80],
                                sel_rs_zp_q, sel_rs_m0_q);
                            v2_prod_hi[3] <= v2_product(s1_data_q[95:88],
                                sel_rs_zp_q, sel_rs_m0_q);
                            v2_prod_hi[4] <= v2_product(s1_data_q[103:96],
                                sel_rs_zp_q, sel_rs_m0_q);
                            v2_prod_hi[5] <= v2_product(s1_data_q[111:104],
                                sel_rs_zp_q, sel_rs_m0_q);
                            v2_prod_hi[6] <= v2_product(s1_data_q[119:112],
                                sel_rs_zp_q, sel_rs_m0_q);
                            v2_prod_hi[7] <= v2_product(s1_data_q[127:120],
                                sel_rs_zp_q, sel_rs_m0_q);
                            state_q <= S_V2_D4;
                        end
                    end

                    S_V2_D4: begin
                        if (rd_bad) begin
                            raise_error(rd_bad_code);
                        end else begin
                            v2_res_hi <= v2_hi_comb;
                            state_q   <= S_V2_D5;
                        end
                    end

                    S_V2_D5: begin
`ifdef COPY_DEBUG
                        if (s1_ow_q < 12'd3)
                            $display("COPYDBG_D5 ow=%0d reslo=%h reshi=%h word=%h mask=%h wmask=%h",
                                     s1_ow_q, v2_res_lo, v2_res_hi, v2_d4_word,
                                     s1_mask_q, s1_wmask_q);
`endif
                        if (v2_last_word) state_q <= S_PASS_DRAIN;
                        else              state_q <= S_STREAM;
                    end

                    S_PASS_DRAIN: begin
                        if (!rd_cmd_out) begin
                            state_q <= S_NEXT_PASS;
                        end else if (rd_bad) begin
                            raise_error(rd_bad_code);
                        end else if (rd_fire && rd_cmd_out) begin
                            rd_beat_cnt <= rd_beat_cnt + 13'd1;
                            if (rd_data_last) rd_cmd_out <= 1'b0;
                        end
                    end

                    S_NEXT_PASS: begin
                        if ({1'b0, pass_q} + 3'd1 < n_passes_q) begin
                            // revision3: snapshot the explicit next-pass config
                            sel_in_addr_q    <= in_addr_q[pass_q + 2'd1];
                            sel_in_row_q     <= in_row_q[pass_q + 2'd1];
                            sel_in_bytes_q   <= in_bytes_q[pass_q + 2'd1];
                            sel_in_ps_q      <= in_ps_q[pass_q + 2'd1];
                            sel_ci_q         <= in_c_i_q[pass_q + 2'd1];
                            sel_ch_off_q     <= ch_off_q[pass_q + 2'd1];
                            sel_ch_end_q     <= {1'b0, ch_off_q[pass_q + 2'd1]} +
                                                {1'b0, in_c_i_q[pass_q + 2'd1]};
                            sel_rs_m0_q      <= rs_m0_q[pass_q + 2'd1];
                            sel_rs_shift_q   <= rs_shift_q[pass_q + 2'd1];
                            sel_rs_zp_q      <= rs_zp_q[pass_q + 2'd1];
                            sel_rs_en_q      <= rs_en_q[pass_q + 2'd1];
                            sel_rs_bypass_q  <= rs_bypass_q[pass_q + 2'd1];
                            sel_rd_limit_q   <= {1'b0, in_addr_q[pass_q + 2'd1]} +
                                                {1'b0, in_bytes_q[pass_q + 2'd1]};
                            sel_round_bias_q <= (rs_shift_q[pass_q + 2'd1] == 8'd0) ? 64'd0 :
                                                (64'd1 << (rs_shift_q[pass_q + 2'd1] - 8'd1));
                            sel_rescale_q    <= rs_en_q[pass_q + 2'd1] &&
                                                !rs_bypass_q[pass_q + 2'd1];
                            pass_q  <= pass_q + 2'd1;
                            state_q <= S_PASS_SETUP;
                        end else begin
                            o_q     <= 12'd0;
                            state_q <= S_WR_CMD;
                        end
                    end

                    S_WR_CMD: begin
                        if (wr_range_bad) begin
                            raise_error(E_STRIDE);
                        end else if (wr_cmd_valid && wr_cmd_ready) begin
                            wr_cmd_out <= 1'b1;
                            o_q        <= 12'd0;
                            state_q    <= S_WR_PRE;
                        end
                    end

                    S_WR_PRE: begin
                        state_q <= S_WR_DATA;
                    end

                    S_WR_DATA: begin
                        if (wr_data_valid && wr_data_ready) begin
                            if (o_q == out_words_q - 16'd1) begin
                                state_q <= S_WR_RSP;
                            end else begin
                                o_q <= o_q + 12'd1;
                            end
                        end
                    end

                    S_WR_RSP: begin
                        if (wr_rsp_valid) begin
                            wr_cmd_out <= 1'b0;
                            if (wr_rsp_tag != TAG_WR) begin
                                raise_error(E_AXI_PROTOCOL);
                            end else if (wr_rsp_error) begin
                                raise_error((wr_rsp_error_code == 16'd0) ?
                                            E_CORE_INTERNAL : wr_rsp_error_code);
                            end else if (wr_rsp_aborted) begin
                                raise_error(E_AXI_PROTOCOL);
                            end else begin
                                state_q <= S_NEXT_ROW;
                            end
                        end
                    end

                    S_NEXT_ROW: begin
                        y_q            <= y_q + 16'd1;
                        out_row_addr_q <= out_row_addr_q + out_row_q;
                        for (k = 0; k < 4; k = k + 1) begin
                            if ((k < n_inputs_q) &&
                                ((mode_q != MODE_PAD) || row_data_q))
                                in_cur_addr_q[k] <= in_cur_addr_q[k] + in_row_q[k];
                        end
                        state_q <= S_ROW_SETUP;
                    end

                    S_DONE: begin
`ifdef COPY_DEBUG
                        $display("COPYDBG_DONE t=%0t y=%0d pass=%0d", $time, y_q, pass_q);
`endif
                        core_busy <= 1'b0;
                        core_done <= 1'b1;
                        state_q   <= S_IDLE;
                    end

                    S_ERROR: begin
                        core_busy  <= 1'b0;
                        core_error <= 1'b1;
                        state_q    <= S_IDLE;
                    end

                    S_ABORT_RD: begin
                        if (rd_fire && rd_cmd_out && rd_data_last) begin
                            rd_cmd_out <= 1'b0;
                            if (wr_cmd_out) state_q <= S_ABORT_WAIT;
                            else            state_q <= S_IDLE;
                        end
                    end

                    S_ABORT_WAIT: begin
                        state_q <= S_ABORT_WD;
                    end

                    S_ABORT_WD: begin
                        if (wr_data_valid && wr_data_ready) begin
                            if (o_q == out_words_q - 16'd1) begin
                                state_q <= S_ABORT_RSP;
                            end else begin
                                o_q <= o_q + 12'd1;
                            end
                        end
                    end

                    S_ABORT_RSP: begin
                        if (wr_rsp_valid) begin
                            wr_cmd_out <= 1'b0;
                            state_q    <= S_IDLE;
                        end
                    end

                    default: state_q <= S_IDLE;
                endcase
            end
        end
    end

endmodule
