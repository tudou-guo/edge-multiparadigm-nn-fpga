// -----------------------------------------------------------------------------
// activation_lut_core.sv -- E1 ActivationLUT core (opcode 0x001B)
//
// One generic INT8 activation engine shared by SiLU / Sigmoid / Tanh /
// LeakyRelu.  The numeric function never enters RTL: every record loads one
// 256-byte table from DDR (16 x 128-bit beats), then the engine maps
//   y[byte] = table[x[byte] + 128]      (two's-complement int8 -> 8-bit value)
// for every logical activation element of a NHWC8_A16 tensor.
//
// Structure
//   * 8 synchronous table banks (activation_lut_bank: 8 x 256 x 8 LUTRAM),
//     broadcast table write (one byte/cycle into all 8 banks), 1-cycle read.
//     A 16-byte input beat is split into two 8-byte lookup groups; each group
//     reads its 8 table entries from the 8 banks in one cycle (8 lanes/cycle).
//     Steady state is 2 clk per 16B beat: the high-half issue cycle prefetches
//     the next beat from the input FIFO into the beat register, so there is no
//     per-beat refill bubble (v0.2 batch 2 removed the legacy 3 clk/beat gap).
//   * Small elastic input beat FIFO and output beat FIFO absorb read/write
//     channel backpressure.
//   * Per row exactly one normalized read command (bytes = in row stride,
//     <= 10240 B) and one normalized write command (bytes = out row stride);
//     the shell burst engines split them into legal AXI bursts.  The 256 B
//     table load (16 beats) is a separate command and is counted apart from
//     the activation traffic.
//   * Invalid lanes (pixel >= W or channel >= C, i.e. NHWC8_A16 tail/row pad)
//     are forced to 8'h00 on the output, never to table[Sigmoid(0)].
//
// Capacity (v0.2 batch 2): H/W/C are elaboration parameters, not fixed 32.
//   MAX_H / MAX_W : 0 = inherit legacy 32; else the checked H/W limit (<=640).
//   MAX_CH        : C_alloc limit (1..16, default 16).
//   Capacity is a validation bound only: the working set is constant
//   (2 KiB table + four-beat in/out FIFOs) and does not grow with H/W.
//   640-wide rows (row stride 5120 B @ C8 / 10240 B @ C16) stream through the
//   same 4-entry FIFOs.
//
// State machine: IDLE / VALIDATE / LOAD_TABLE / PROCESS / DRAIN /
// WAIT_WRITE_RESPONSE / DONE / ERROR (+ CANCEL convergence for abort and
// soft reset).  core_done is asserted exactly once, only after the final write
// response of the final row has been observed successfully.
//
// Error semantics: any fault sets core_error/core_error_code and never sets
// core_done.  In-flight read beats are drained and in-flight write beats are
// completed with zero data (the shell writer has no abort input, so dropping
// W beats would deadlock the interconnect); once the channels are drained the
// core stays in ERROR until core_soft_reset.  core_abort / core_soft_reset
// converge to IDLE through CANCEL without issuing new commands.
//
// Reports internal performance counters (registers, hierarchical TB access,
// not ports): table_load_cycles, table_load_beats, activation_rd_beats,
// activation_wr_beats, input_read_cycles, lookup_active_cycles,
// lookup_issue_cycles, process_cycles, wait_cycles, total_cycles,
// valid_lane_count.  Table-load and activation-transfer counters are kept
// separate (performance contract contracts/perf_lut.json).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps
`include "op_abi_defs.vh"

module activation_lut_core #(
    parameter integer DATA_WIDTH = 128,
    parameter integer CHUNK_BYTES = 16,
    parameter integer MAX_CH = 16,
    // v0.2 batch 2 capacity parameters (0 = inherit the legacy 32 limit, so a
    // router/top that does not pass them elaborates exactly the legacy core).
    parameter integer MAX_H = 0,
    parameter integer MAX_W = 0
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire [31:0] cfg_in_addr_byte,
    input  wire [31:0] cfg_out_addr_byte,
    input  wire [31:0] cfg_lut_addr_byte,
    input  wire [31:0] cfg_in_bytes,
    input  wire [31:0] cfg_out_bytes,
    input  wire [31:0] cfg_in_row_stride_bytes,
    input  wire [31:0] cfg_out_row_stride_bytes,
    input  wire [15:0] cfg_h,
    input  wire [15:0] cfg_w,
    input  wire [15:0] cfg_cin,
    input  wire [15:0] cfg_calloc,
    input  wire [31:0] cfg_lut_bytes,

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
    // Structural constants
    // ------------------------------------------------------------------
    localparam integer GROUP_BYTES     = 8;                          // 8 lanes / banks
    localparam integer BEAT_BYTES      = DATA_WIDTH / 8;             // 16
    localparam integer LANES           = GROUP_BYTES;                // table banks
    localparam integer IN_DEPTH        = 4;
    localparam integer OUT_DEPTH       = 4;
    localparam integer LOAD_BEATS      = 16;                         // 256 B / 16 B
    // Capacity limits driven by elaboration parameters.  0 means "inherit the
    // legacy 32x32 limit" so existing instantiations keep their exact behaviour.
    localparam integer EFF_MAX_H       = (MAX_H == 0) ? 32 : MAX_H;
    localparam integer EFF_MAX_W       = (MAX_W == 0) ? 32 : MAX_W;
    localparam integer EFF_MAX_CH      = (MAX_CH == 0) ? 16 : MAX_CH;
    localparam [15:0]  EFF_MAX_H_U16   = EFF_MAX_H[15:0];
    localparam [15:0]  EFF_MAX_W_U16   = EFF_MAX_W[15:0];
    localparam [15:0]  EFF_MAX_CH_U16  = EFF_MAX_CH[15:0];
    localparam [31:0]  TABLE_BYTES     = 32'd256;

    // ------------------------------------------------------------------
    // Error codes (op_abi_defs.vh)
    // ------------------------------------------------------------------
    localparam [15:0] E_NONE         = `OP_E_NONE;
    localparam [15:0] E_CFG          = `OP_E_CFG;
    localparam [15:0] E_ADDR_ALIGN   = `OP_E_ADDR_ALIGN;
    localparam [15:0] E_ADDR_RANGE   = `OP_E_ADDR_RANGE;
    localparam [15:0] E_STRIDE       = `OP_E_STRIDE;
    localparam [15:0] E_AXI_RESP     = `OP_E_AXI_RESP;
    localparam [15:0] E_AXI_PROTOCOL = `OP_E_AXI_PROTOCOL;

    // ------------------------------------------------------------------
    // FSM states
    // ------------------------------------------------------------------
    localparam [3:0] S_IDLE     = 4'd0;
    localparam [3:0] S_VALIDATE = 4'd1;
    localparam [3:0] S_LOAD_CMD = 4'd2;
    localparam [3:0] S_LOAD_DATA= 4'd3;
    localparam [3:0] S_ROW_START= 4'd4;
    localparam [3:0] S_PROCESS  = 4'd5;
    localparam [3:0] S_DRAIN    = 4'd6;
    localparam [3:0] S_WAIT_RSP = 4'd7;
    localparam [3:0] S_DONE     = 4'd8;
    localparam [3:0] S_ERROR    = 4'd9;
    localparam [3:0] S_CANCEL   = 4'd10;

    // Normalized-channel tags.
    localparam [7:0] TAG_LUT = 8'hA1;
    localparam [7:0] TAG_RD  = 8'hB2;
    localparam [7:0] TAG_WR  = 8'hC3;

    // ------------------------------------------------------------------
    // Helpers (shift/add only: no DSP multiplier is inferred anywhere)
    // ------------------------------------------------------------------
    function [31:0] mul_h_stride;
        input [15:0] hh;
        input [31:0] stride;
        integer i;
        reg [31:0] acc;
        begin
            acc = 32'd0;
            for (i = 0; i < 16; i = i + 1) begin
                if (hh[i]) acc = acc + (stride << i);
            end
            mul_h_stride = acc;
        end
    endfunction

    function [3:0] popcount8;
        input [7:0] v;
        integer i;
        begin
            popcount8 = 4'd0;
            for (i = 0; i < 8; i = i + 1) popcount8 = popcount8 + v[i];
        end
    endfunction

    // ------------------------------------------------------------------
    // Config snapshot (registered once per operation)
    // ------------------------------------------------------------------
    reg [31:0] in_addr_q, out_addr_q, lut_addr_q;
    reg [31:0] in_bytes_q, out_bytes_q, lut_bytes_q;
    reg [31:0] in_stride_q, out_stride_q;
    reg [15:0] h_q, w_q, cin_q, calloc_q;
    reg [15:0] row_beats_q, groups_per_row_q;

    // ------------------------------------------------------------------
    // Operation progress
    // ------------------------------------------------------------------
    reg [15:0] row_idx_q;
    reg [31:0] in_row_base_q, out_row_base_q;
    reg        rd_inflight_q;
    reg [15:0] rd_beats_expected_q, rd_beats_seen_q;
    reg [15:0] grp_issued_q;         // lookup groups issued in the current row
    reg [15:0] beats_pushed_q;       // output beats pushed in the current row
    reg        wr_inflight_q;
    reg [15:0] wr_beats_sent_q;
    reg        wr_last_sent_q;
    reg        rsp_seen_q, rsp_error_q, rsp_aborted_q;

    // ------------------------------------------------------------------
    // Table load
    // ------------------------------------------------------------------
    reg        table_valid_q;
    reg        load_beat_valid_q, load_done_q;
    reg [DATA_WIDTH-1:0] load_beat_q;
    reg [15:0] load_base_q;
    reg [15:0] load_beats_written_q;
    reg [7:0]  load_wr_idx_q;

    // ------------------------------------------------------------------
    // Elastic input beat FIFO (128b x 4) and output beat FIFO (128b x 4)
    // ------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] in_fifo  [0:IN_DEPTH-1];
    // Pointers are 2-bit (one per FIFO slot); only the fill counter is 3-bit.
    reg [1:0] in_wr, in_rd;
    reg [2:0] in_cnt;
    reg [DATA_WIDTH-1:0] out_fifo [0:OUT_DEPTH-1];
    reg [1:0] out_wr, out_rd;
    reg [2:0] out_cnt;

    // ------------------------------------------------------------------
    // Group / beat pipeline
    // ------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] cur_beat_q;
    reg        cur_beat_valid_q;
    reg        grp_valid_q;      // bank read result valid this cycle
    reg        grp_phase_q;      // 0 = low 64b half, 1 = high 64b half
    reg [7:0]  grp_mask_q;       // per-lane validity captured with the lookup
    reg [63:0] out_lo_q;         // low 64b half of the beat under assembly
    reg        out_lo_valid_q;

    // ------------------------------------------------------------------
    // Cancellation / performance
    // ------------------------------------------------------------------
    reg        cancel_soft_q;
    reg        perf_active_q;
    reg [31:0] table_load_cycles;
    reg [31:0] table_load_beats;
    reg [31:0] activation_rd_beats;
    reg [31:0] activation_wr_beats;
    reg [31:0] input_read_cycles;
    reg [31:0] lookup_active_cycles;
    reg [31:0] lookup_issue_cycles;
    reg [31:0] process_cycles;
    reg [31:0] wait_cycles;
    reg [31:0] total_cycles;
    reg [31:0] valid_lane_count;

    reg [3:0]  state;

    // ------------------------------------------------------------------
    // Table banks: 8 x 256 x 8, broadcast write, individual read address
    // ------------------------------------------------------------------
    reg         bank_we;
    reg  [7:0]  bank_waddr;
    reg  [7:0]  bank_wdata;
    wire        bank_re;
    wire [LANES*8-1:0] bank_raddr;
    wire [LANES*8-1:0] bank_rdata;

    genvar gb;
    generate
        for (gb = 0; gb < LANES; gb = gb + 1) begin : GEN_BANK
            activation_lut_bank u_bank (
                .clk   (clk),
                .we    (bank_we),
                .waddr (bank_waddr),
                .wdata (bank_wdata),
                .re    (bank_re),
                .raddr (bank_raddr[gb*8 +: 8]),
                .rdata (bank_rdata[gb*8 +: 8])
            );
        end
    endgenerate

    // ------------------------------------------------------------------
    // Lookup issue stage
    // ------------------------------------------------------------------
    wire        half_now    = grp_issued_q[0];
    wire [63:0] grp_bytes_c = half_now ? cur_beat_q[127:64] : cur_beat_q[63:0];
    wire [15:0] grp_off_c   = {grp_issued_q[12:0], 3'b000};
    wire        out_space_c = (out_cnt < OUT_DEPTH[2:0]);

    wire [7:0] grp_mask_c;
    genvar gl;
    generate
        for (gl = 0; gl < LANES; gl = gl + 1) begin : GEN_MASK
            wire [15:0] lane_off;
            wire [15:0] lane_pix;
            wire [15:0] lane_ch;
            assign lane_off = grp_off_c + gl[15:0];
            assign lane_pix = (calloc_q == 16'd8) ? (lane_off >> 3) : (lane_off >> 4);
            assign lane_ch  = (calloc_q == 16'd8) ? {13'b0, lane_off[2:0]} :
                              {12'b0, lane_off[3:0]};
            assign grp_mask_c[gl] = (lane_pix < w_q) && (lane_ch < cin_q);
        end
    endgenerate

    // Bank read addresses: table index = signed byte + 128 (bit 7 inverted).
    generate
        for (gb = 0; gb < LANES; gb = gb + 1) begin : GEN_RADDR
            assign bank_raddr[gb*8 +: 8] = grp_bytes_c[gb*8 +: 8] ^ 8'h80;
        end
    endgenerate

    // Issue one lookup group per cycle while the process stage is streaming and
    // the output FIFO can hold the beat that the high half will complete.
    wire issue_run = (state == S_PROCESS) && cur_beat_valid_q &&
                     (!half_now || out_space_c);
    assign bank_re = issue_run;

    wire [63:0] masked_c;
    generate
        for (gb = 0; gb < LANES; gb = gb + 1) begin : GEN_MASKED
            assign masked_c[gb*8 +: 8] = grp_mask_q[gb] ? bank_rdata[gb*8 +: 8] : 8'h00;
        end
    endgenerate

    // ------------------------------------------------------------------
    // Configuration validation (evaluated in S_VALIDATE, before any DDR
    // command and without any truncation of the 32-bit address fields)
    // ------------------------------------------------------------------
    wire [31:0] w_alloc_c       = (calloc_q == 16'd8) ? ({16'd0, w_q} << 3)
                                                      : ({16'd0, w_q} << 4);
    wire [31:0] stride_expect_c = (w_alloc_c + 32'd15) & 32'hFFFF_FFF0;
    wire [31:0] in_bytes_expect_c  = mul_h_stride(h_q, in_stride_q);
    wire [31:0] out_bytes_expect_c = mul_h_stride(h_q, out_stride_q);

    wire [32:0] in_start_c  = {1'b0, in_addr_q};
    wire [32:0] in_end_c    = in_start_c  + {1'b0, in_bytes_q};
    wire [32:0] out_start_c = {1'b0, out_addr_q};
    wire [32:0] out_end_c   = out_start_c + {1'b0, out_bytes_q};
    wire [32:0] lut_start_c = {1'b0, lut_addr_q};
    wire [32:0] lut_end_c   = lut_start_c + {1'b0, TABLE_BYTES};

    wire range_wrap_c = (in_end_c > 33'h1_0000_0000) ||
                        (out_end_c > 33'h1_0000_0000) ||
                        (lut_end_c > 33'h1_0000_0000);
    wire overlap_c = ((in_start_c  < out_end_c)  && (out_start_c < in_end_c))  ||
                     ((in_start_c  < lut_end_c)  && (lut_start_c < in_end_c))  ||
                     ((out_start_c < lut_end_c)  && (lut_start_c < out_end_c));

    reg [15:0] validate_code_c;
    always @(*) begin
        validate_code_c = E_NONE;
        if ((h_q == 16'd0) || (h_q > EFF_MAX_H_U16) ||
            (w_q == 16'd0) || (w_q > EFF_MAX_W_U16) ||
            (cin_q == 16'd0) || (cin_q > EFF_MAX_CH_U16)) begin
            validate_code_c = E_CFG;
        end else if ((calloc_q != 16'd8) && (calloc_q != 16'd16)) begin
            validate_code_c = E_CFG;
        end else if ((((cin_q + 16'd7) & 16'hFFF8) != calloc_q) || (cin_q > calloc_q)) begin
            validate_code_c = E_CFG;
        end else if (lut_bytes_q != TABLE_BYTES) begin
            validate_code_c = E_CFG;
        end else if ((in_addr_q[3:0] != 4'b0) || (out_addr_q[3:0] != 4'b0) ||
                     (lut_addr_q[3:0] != 4'b0)) begin
            validate_code_c = E_ADDR_ALIGN;
        end else if (stride_expect_c > 32'h0000_FFFF) begin
            // row bytes are carried in the 16-bit normalized command field
            validate_code_c = E_CFG;
        end else if ((in_stride_q != stride_expect_c) ||
                     (out_stride_q != stride_expect_c)) begin
            validate_code_c = E_STRIDE;
        end else if ((in_bytes_q != in_bytes_expect_c) ||
                     (out_bytes_q != out_bytes_expect_c)) begin
            validate_code_c = E_STRIDE;
        end else if (range_wrap_c) begin
            validate_code_c = E_ADDR_RANGE;
        end else if (overlap_c) begin
            validate_code_c = E_ADDR_RANGE;
        end
    end

    // ------------------------------------------------------------------
    // Combinational channel drivers
    // ------------------------------------------------------------------
    always @(*) begin
        rd_cmd_valid     = 1'b0;
        rd_cmd_addr_byte = in_row_base_q;
        rd_cmd_bytes     = in_stride_q[15:0];
        rd_cmd_tag       = TAG_RD;
        rd_data_ready    = 1'b0;

        wr_cmd_valid     = 1'b0;
        wr_cmd_addr_byte = out_row_base_q;
        wr_cmd_bytes     = out_stride_q[15:0];
        wr_cmd_tag       = TAG_WR;

        wr_data_valid    = 1'b0;
        wr_data          = out_fifo[out_rd];
        wr_data_last     = 1'b0;
        wr_data_tag      = TAG_WR;
        wr_rsp_ready     = 1'b0;

        bank_we          = 1'b0;
        bank_waddr       = 8'd0;
        bank_wdata       = 8'd0;

        case (state)
            S_LOAD_CMD: begin
                rd_cmd_valid     = 1'b1;
                rd_cmd_addr_byte = lut_addr_q;
                rd_cmd_bytes     = TABLE_BYTES[15:0];
                rd_cmd_tag       = TAG_LUT;
            end
            S_LOAD_DATA: begin
                rd_data_ready = !load_beat_valid_q;
                if (load_beat_valid_q) begin
                    bank_we    = 1'b1;
                    bank_waddr = load_base_q[7:0] + load_wr_idx_q;
                    bank_wdata = load_beat_q[{load_wr_idx_q, 3'b000} +: 8];
                end
            end
            S_ROW_START: begin
                if (!rd_inflight_q) begin
                    rd_cmd_valid     = 1'b1;
                    rd_cmd_addr_byte = in_row_base_q;
                    rd_cmd_bytes     = in_stride_q[15:0];
                    rd_cmd_tag       = TAG_RD;
                end
                if (!wr_inflight_q) begin
                    wr_cmd_valid     = 1'b1;
                    wr_cmd_addr_byte = out_row_base_q;
                    wr_cmd_bytes     = out_stride_q[15:0];
                    wr_cmd_tag       = TAG_WR;
                end
            end
            S_PROCESS: begin
                // The row's read and write commands are issued once per row in
                // S_ROW_START; this state only streams data (issuing another
                // command here would fetch a phantom next row).
                rd_data_ready = rd_inflight_q && (in_cnt < IN_DEPTH[2:0]);
                if (wr_inflight_q && (out_cnt != 3'd0)) begin
                    wr_data_valid = 1'b1;
                    wr_data       = out_fifo[out_rd];
                    wr_data_last  = (wr_beats_sent_q == (row_beats_q - 16'd1));
                end
            end
            S_DRAIN: begin
                if (out_cnt != 3'd0) begin
                    wr_data_valid = 1'b1;
                    wr_data       = out_fifo[out_rd];
                    wr_data_last  = (wr_beats_sent_q == (row_beats_q - 16'd1));
                end
                if (rd_inflight_q) rd_data_ready = 1'b1;
            end
            S_WAIT_RSP: begin
                wr_rsp_ready = 1'b1;
            end
            S_ERROR, S_CANCEL: begin
                if (rd_inflight_q) rd_data_ready = 1'b1;
                if (wr_inflight_q && !wr_last_sent_q) begin
                    wr_data_valid = 1'b1;
                    wr_data       = {DATA_WIDTH{1'b0}};
                    wr_data_last  = (wr_beats_sent_q == (row_beats_q - 16'd1));
                end
            end
            default: begin
                rd_data_ready = 1'b0;
            end
        endcase

        // A write response is always consumed while a write command is in
        // flight, so a response pulse can never be missed.
        if (wr_inflight_q) wr_rsp_ready = 1'b1;
    end

    // ------------------------------------------------------------------
    // Fault entry: sticky error, no DONE, drop all staged data.  In-flight
    // channels are drained by S_ERROR/S_CANCEL.
    // ------------------------------------------------------------------
    task automatic fail_op;
        input [15:0] code;
        begin
            core_error        <= 1'b1;
            core_error_code   <= code;
            core_busy         <= 1'b0;
            core_done         <= 1'b0;
            table_valid_q     <= 1'b0;
            grp_valid_q       <= 1'b0;
            cur_beat_valid_q  <= 1'b0;
            out_lo_valid_q    <= 1'b0;
            load_beat_valid_q <= 1'b0;
            in_wr <= 2'd0; in_rd <= 2'd0; in_cnt <= 3'd0;
            out_wr <= 2'd0; out_rd <= 2'd0; out_cnt <= 3'd0;
            state <= S_ERROR;
        end
    endtask

    // ------------------------------------------------------------------
    // Sequential engine
    // ------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state             <= S_IDLE;
            core_busy         <= 1'b0;
            core_done         <= 1'b0;
            core_error        <= 1'b0;
            core_error_code   <= E_NONE;
            in_addr_q         <= 32'd0;
            out_addr_q        <= 32'd0;
            lut_addr_q        <= 32'd0;
            in_bytes_q        <= 32'd0;
            out_bytes_q       <= 32'd0;
            lut_bytes_q       <= 32'd0;
            in_stride_q       <= 32'd0;
            out_stride_q      <= 32'd0;
            h_q               <= 16'd0;
            w_q               <= 16'd0;
            cin_q             <= 16'd0;
            calloc_q          <= 16'd0;
            row_beats_q       <= 16'd0;
            groups_per_row_q  <= 16'd0;
            row_idx_q         <= 16'd0;
            in_row_base_q     <= 32'd0;
            out_row_base_q    <= 32'd0;
            rd_inflight_q     <= 1'b0;
            rd_beats_expected_q <= 16'd0;
            rd_beats_seen_q   <= 16'd0;
            grp_issued_q      <= 16'd0;
            beats_pushed_q    <= 16'd0;
            wr_inflight_q     <= 1'b0;
            wr_beats_sent_q   <= 16'd0;
            wr_last_sent_q    <= 1'b0;
            rsp_seen_q        <= 1'b0;
            rsp_error_q       <= 1'b0;
            rsp_aborted_q     <= 1'b0;
            table_valid_q     <= 1'b0;
            load_beat_valid_q <= 1'b0;
            load_done_q       <= 1'b0;
            load_beat_q       <= {DATA_WIDTH{1'b0}};
            load_base_q       <= 16'd0;
            load_beats_written_q <= 16'd0;
            load_wr_idx_q     <= 8'd0;
            in_wr <= 3'd0; in_rd <= 3'd0; in_cnt <= 3'd0;
            out_wr <= 3'd0; out_rd <= 3'd0; out_cnt <= 3'd0;
            cur_beat_q        <= {DATA_WIDTH{1'b0}};
            cur_beat_valid_q  <= 1'b0;
            grp_valid_q       <= 1'b0;
            grp_phase_q       <= 1'b0;
            grp_mask_q        <= 8'd0;
            out_lo_q          <= 64'd0;
            out_lo_valid_q    <= 1'b0;
            cancel_soft_q     <= 1'b0;
            perf_active_q     <= 1'b0;
            table_load_cycles <= 32'd0;
            table_load_beats  <= 32'd0;
            activation_rd_beats <= 32'd0;
            activation_wr_beats <= 32'd0;
            input_read_cycles <= 32'd0;
            lookup_active_cycles <= 32'd0;
            lookup_issue_cycles  <= 32'd0;
            process_cycles    <= 32'd0;
            wait_cycles       <= 32'd0;
            total_cycles      <= 32'd0;
            valid_lane_count  <= 32'd0;
        end else begin
            // Global defaults.
            core_done <= 1'b0;

            // Performance counters (all-zero outside an operation).
            if (perf_active_q) begin
                total_cycles <= total_cycles + 32'd1;
            end
            if (state == S_LOAD_CMD || state == S_LOAD_DATA) begin
                table_load_cycles <= table_load_cycles + 32'd1;
            end
            if (rd_inflight_q && (state == S_ROW_START || state == S_PROCESS)) begin
                input_read_cycles <= input_read_cycles + 32'd1;
            end
            if (state == S_PROCESS) begin
                process_cycles <= process_cycles + 32'd1;
            end
            if ((state == S_WAIT_RSP) ||
                ((state == S_PROCESS || state == S_DRAIN) && wr_data_valid && !wr_data_ready) ||
                ((state == S_ROW_START) &&
                 ((rd_cmd_valid && !rd_cmd_ready) || (wr_cmd_valid && !wr_cmd_ready))) ||
                ((state == S_PROCESS) && cur_beat_valid_q && half_now && !out_space_c)) begin
                wait_cycles <= wait_cycles + 32'd1;
            end

            // Write response capture: always consumed while a write command is
            // in flight so a response pulse cannot be missed.
            if (wr_rsp_valid && wr_rsp_ready) begin
                rsp_seen_q    <= 1'b1;
                rsp_error_q   <= wr_rsp_error;
                rsp_aborted_q <= wr_rsp_aborted;
            end

            if (core_soft_reset || core_abort) begin
                core_busy <= 1'b0;
                core_done <= 1'b0;
                table_valid_q <= 1'b0;
                grp_valid_q <= 1'b0;
                cur_beat_valid_q <= 1'b0;
                out_lo_valid_q <= 1'b0;
                load_beat_valid_q <= 1'b0;
                if (core_soft_reset) begin
                    core_error <= 1'b0;
                    core_error_code <= E_NONE;
                end
                if (state != S_IDLE) begin
                    cancel_soft_q <= core_soft_reset;
                    in_wr <= 2'd0; in_rd <= 2'd0; in_cnt <= 3'd0;
                    out_wr <= 2'd0; out_rd <= 2'd0; out_cnt <= 3'd0;
                    state <= S_CANCEL;
                end
            end else begin
                case (state)
                    // ----------------------------------------------
                    S_IDLE: begin
                        core_busy       <= 1'b0;
                        core_error      <= 1'b0;
                        core_error_code <= E_NONE;
                        table_valid_q   <= 1'b0;
                        if (core_start) begin
                            in_addr_q       <= cfg_in_addr_byte;
                            out_addr_q      <= cfg_out_addr_byte;
                            lut_addr_q      <= cfg_lut_addr_byte;
                            in_bytes_q      <= cfg_in_bytes;
                            out_bytes_q     <= cfg_out_bytes;
                            lut_bytes_q     <= cfg_lut_bytes;
                            in_stride_q     <= cfg_in_row_stride_bytes;
                            out_stride_q    <= cfg_out_row_stride_bytes;
                            h_q             <= cfg_h;
                            w_q             <= cfg_w;
                            cin_q           <= cfg_cin;
                            calloc_q        <= cfg_calloc;
                            row_beats_q     <= {4'b0, cfg_in_row_stride_bytes[15:4]};
                            groups_per_row_q<= {3'b0, cfg_in_row_stride_bytes[15:3]};
                            row_idx_q       <= 16'd0;
                            rd_inflight_q   <= 1'b0;
                            wr_inflight_q   <= 1'b0;
                            rsp_seen_q      <= 1'b0;
                            wr_last_sent_q  <= 1'b0;
                            table_load_cycles <= 32'd0;
                            table_load_beats  <= 32'd0;
                            activation_rd_beats <= 32'd0;
                            activation_wr_beats <= 32'd0;
                            input_read_cycles <= 32'd0;
                            lookup_active_cycles <= 32'd0;
                            lookup_issue_cycles  <= 32'd0;
                            process_cycles    <= 32'd0;
                            wait_cycles       <= 32'd0;
                            total_cycles      <= 32'd0;
                            valid_lane_count  <= 32'd0;
                            perf_active_q   <= 1'b1;
                            core_busy       <= 1'b1;
                            core_error      <= 1'b0;
                            core_error_code <= E_NONE;
                            state           <= S_VALIDATE;
                        end
                    end

                    // ----------------------------------------------
                    S_VALIDATE: begin
                        if (validate_code_c != E_NONE) begin
                            core_error      <= 1'b1;
                            core_error_code <= validate_code_c;
                            core_busy       <= 1'b0;
                            perf_active_q   <= 1'b0;
                            state           <= S_ERROR;
                        end else begin
                            in_row_base_q  <= in_addr_q;
                            out_row_base_q <= out_addr_q;
                            row_idx_q      <= 16'd0;
                            state          <= S_LOAD_CMD;
                        end
                    end

                    // ----------------------------------------------
                    S_LOAD_CMD: begin
                        if (rd_cmd_ready) begin
                            rd_inflight_q       <= 1'b1;
                            rd_beats_expected_q <= LOAD_BEATS[15:0];
                            rd_beats_seen_q     <= 16'd0;
                            load_beats_written_q<= 16'd0;
                            load_wr_idx_q       <= 8'd0;
                            load_beat_valid_q   <= 1'b0;
                            load_done_q         <= 1'b0;
                            state               <= S_LOAD_DATA;
                        end
                    end

                    // ----------------------------------------------
                    S_LOAD_DATA: begin
                        // Write the buffered 16-byte beat into all 8 banks,
                        // one table index (byte) per cycle.
                        if (load_beat_valid_q) begin
                            if (load_wr_idx_q == 8'd15) begin
                                load_beat_valid_q <= 1'b0;
                                load_beats_written_q <= load_beats_written_q + 16'd1;
                                if (load_beats_written_q == 16'd15) load_done_q <= 1'b1;
                            end else begin
                                load_wr_idx_q <= load_wr_idx_q + 8'd1;
                            end
                        end
                        if (rd_data_valid && rd_data_ready) begin
                            if (rd_data_error) begin
                                fail_op((rd_data_error_code == E_NONE) ? E_AXI_RESP
                                                                      : rd_data_error_code);
                                rd_beats_seen_q <= rd_beats_seen_q + 16'd1;
                                if (rd_data_last ||
                                    ((rd_beats_seen_q + 16'd1) >= rd_beats_expected_q))
                                    rd_inflight_q <= 1'b0;
                            end else if (rd_data_tag != TAG_LUT) begin
                                fail_op(E_AXI_PROTOCOL);
                                rd_beats_seen_q <= rd_beats_seen_q + 16'd1;
                                if (rd_data_last ||
                                    ((rd_beats_seen_q + 16'd1) >= rd_beats_expected_q))
                                    rd_inflight_q <= 1'b0;
                            end else if (rd_beats_seen_q >= LOAD_BEATS[15:0]) begin
                                fail_op(E_AXI_PROTOCOL);
                                rd_inflight_q <= 1'b0;
                            end else if (rd_data_last !=
                                         (rd_beats_seen_q == (LOAD_BEATS[15:0] - 16'd1))) begin
                                fail_op(E_AXI_PROTOCOL);
                                rd_beats_seen_q <= rd_beats_seen_q + 16'd1;
                                if (rd_data_last ||
                                    ((rd_beats_seen_q + 16'd1) >= rd_beats_expected_q))
                                    rd_inflight_q <= 1'b0;
                            end else begin
                                load_beat_q  <= rd_data;
                                load_base_q  <= {rd_beats_seen_q[11:0], 4'b0000};
                                load_wr_idx_q<= 8'd0;
                                load_beat_valid_q <= 1'b1;
                                table_load_beats <= table_load_beats + 32'd1;
                                rd_beats_seen_q <= rd_beats_seen_q + 16'd1;
                                if (rd_data_last) rd_inflight_q <= 1'b0;
                            end
                        end
                        if (load_done_q && !rd_inflight_q) begin
                            table_valid_q  <= 1'b1;
                            in_row_base_q  <= in_addr_q;
                            out_row_base_q <= out_addr_q;
                            row_idx_q      <= 16'd0;
                            state          <= S_ROW_START;
                        end
                    end

                    // ----------------------------------------------
                    S_ROW_START: begin
                        if (!rd_inflight_q && rd_cmd_ready) begin
                            rd_inflight_q       <= 1'b1;
                            rd_beats_expected_q <= row_beats_q;
                            rd_beats_seen_q     <= 16'd0;
                        end
                        if (!wr_inflight_q && wr_cmd_ready) begin
                            wr_inflight_q  <= 1'b1;
                            wr_beats_sent_q<= 16'd0;
                            wr_last_sent_q <= 1'b0;
                            rsp_seen_q     <= 1'b0;
                        end
                        if (rd_inflight_q && wr_inflight_q) begin
                            grp_issued_q     <= 16'd0;
                            beats_pushed_q   <= 16'd0;
                            cur_beat_valid_q <= 1'b0;
                            out_lo_valid_q   <= 1'b0;
                            state            <= S_PROCESS;
                        end
                    end

                    // ----------------------------------------------
                    S_PROCESS: begin : proc_stage
                        reg rx_push, beat_pop, out_push, out_pop;
                        begin
                            rx_push  = 1'b0;
                            beat_pop = 1'b0;
                            out_push = 1'b0;
                            out_pop  = 1'b0;

                            // (a) lookup issue into the 8 banks.  The high-half
                            // cycle also pops the next beat from the input FIFO,
                            // so the group splitter never has an empty beat slot
                            // between two consecutive beats (2 clk/beat steady
                            // state instead of the legacy 3 clk/beat).
                            if (issue_run) begin
                                grp_valid_q   <= 1'b1;
                                grp_phase_q   <= half_now;
                                grp_mask_q    <= grp_mask_c;
                                grp_issued_q  <= grp_issued_q + 16'd1;
                                lookup_issue_cycles <= lookup_issue_cycles + 32'd1;
                                if (half_now) begin
                                    if (in_cnt != 3'd0) begin
                                        cur_beat_q       <= in_fifo[in_rd];
                                        cur_beat_valid_q <= 1'b1;
                                        beat_pop         = 1'b1;
                                    end else begin
                                        cur_beat_valid_q <= 1'b0;
                                    end
                                end
                            end else begin
                                grp_valid_q <= 1'b0;
                            end

                            // (b) bank read results: mask, then half-beat assembly
                            if (grp_valid_q) begin
                                lookup_active_cycles <= lookup_active_cycles + 32'd1;
                                valid_lane_count <= valid_lane_count + {28'd0, popcount8(grp_mask_q)};
                                if (grp_phase_q == 1'b0) begin
                                    out_lo_q       <= masked_c;
                                    out_lo_valid_q <= 1'b1;
                                end else begin
                                    out_fifo[out_wr] <= {masked_c, out_lo_q};
                                    out_push = 1'b1;
                                    out_lo_valid_q <= 1'b0;
                                    beats_pushed_q <= beats_pushed_q + 16'd1;
                                end
                            end

                            // (c) fetch the next input beat into the group splitter
                            if (!cur_beat_valid_q && (in_cnt != 3'd0)) begin
                                cur_beat_q       <= in_fifo[in_rd];
                                cur_beat_valid_q <= 1'b1;
                                beat_pop = 1'b1;
                            end

                            // (d) accept a read beat into the elastic input FIFO
                            if (rd_data_valid && rd_data_ready) begin
                                if (rd_data_error) begin
                                    fail_op((rd_data_error_code == E_NONE) ? E_AXI_RESP
                                                                          : rd_data_error_code);
                                    rd_beats_seen_q <= rd_beats_seen_q + 16'd1;
                                    if (rd_data_last ||
                                        ((rd_beats_seen_q + 16'd1) >= rd_beats_expected_q))
                                        rd_inflight_q <= 1'b0;
                                end else if (rd_data_tag != TAG_RD) begin
                                    fail_op(E_AXI_PROTOCOL);
                                    rd_beats_seen_q <= rd_beats_seen_q + 16'd1;
                                    if (rd_data_last ||
                                        ((rd_beats_seen_q + 16'd1) >= rd_beats_expected_q))
                                        rd_inflight_q <= 1'b0;
                                end else if (rd_beats_seen_q >= row_beats_q) begin
                                    fail_op(E_AXI_PROTOCOL);
                                    rd_inflight_q <= 1'b0;
                                end else if (rd_data_last !=
                                             (rd_beats_seen_q == (row_beats_q - 16'd1))) begin
                                    fail_op(E_AXI_PROTOCOL);
                                    rd_beats_seen_q <= rd_beats_seen_q + 16'd1;
                                    if (rd_data_last ||
                                        ((rd_beats_seen_q + 16'd1) >= rd_beats_expected_q))
                                        rd_inflight_q <= 1'b0;
                                end else begin
                                    in_fifo[in_wr] <= rd_data;
                                    rx_push = 1'b1;
                                    activation_rd_beats <= activation_rd_beats + 32'd1;
                                    rd_beats_seen_q <= rd_beats_seen_q + 16'd1;
                                    if (rd_data_last) rd_inflight_q <= 1'b0;
                                end
                            end

                            // (e) write data drain (output FIFO -> shell writer)
                            if (wr_inflight_q && (out_cnt != 3'd0) && wr_data_ready) begin
                                out_pop = 1'b1;
                                activation_wr_beats <= activation_wr_beats + 32'd1;
                                wr_beats_sent_q <= wr_beats_sent_q + 16'd1;
                                if (wr_beats_sent_q == (row_beats_q - 16'd1))
                                    wr_last_sent_q <= 1'b1;
                            end

                            // (f) FIFO pointers
                            in_cnt  <= in_cnt  + {2'b0, rx_push} - {2'b0, beat_pop};
                            in_wr   <= in_wr   + rx_push;
                            in_rd   <= in_rd   + beat_pop;
                            out_cnt <= out_cnt + {2'b0, out_push} - {2'b0, out_pop};
                            out_wr  <= out_wr  + out_push;
                            out_rd  <= out_rd  + out_pop;

                            // (g) row finished: read drained, all groups issued, all
                            // output beats pushed.
                            if (!rd_inflight_q && (in_cnt == 3'd0) && !cur_beat_valid_q &&
                                (grp_issued_q == groups_per_row_q) &&
                                (beats_pushed_q == row_beats_q) && !grp_valid_q) begin
                                state <= S_DRAIN;
                            end
                        end
                    end

                    // ----------------------------------------------
                    S_DRAIN: begin
                        if (wr_inflight_q && (out_cnt != 3'd0) && wr_data_ready) begin
                            out_rd  <= out_rd + 2'd1;
                            out_cnt <= out_cnt - 3'd1;
                            activation_wr_beats <= activation_wr_beats + 32'd1;
                            wr_beats_sent_q <= wr_beats_sent_q + 16'd1;
                            if (wr_beats_sent_q == (row_beats_q - 16'd1))
                                wr_last_sent_q <= 1'b1;
                        end
                        if ((out_cnt == 3'd0) && wr_last_sent_q) begin
                            state <= S_WAIT_RSP;
                        end
                    end

                    // ----------------------------------------------
                    S_WAIT_RSP: begin
                        if (rsp_seen_q) begin
                            rsp_seen_q   <= 1'b0;
                            wr_inflight_q<= 1'b0;
                            if (rsp_error_q || rsp_aborted_q) begin
                                fail_op(E_AXI_RESP);
                            end else if ((row_idx_q + 16'd1) >= h_q) begin
                                state <= S_DONE;
                            end else begin
                                row_idx_q      <= row_idx_q + 16'd1;
                                in_row_base_q  <= in_row_base_q + in_stride_q;
                                out_row_base_q <= out_row_base_q + out_stride_q;
                                state          <= S_ROW_START;
                            end
                        end
                    end

                    // ----------------------------------------------
                    S_DONE: begin
                        core_busy     <= 1'b0;
                        core_done     <= 1'b1;
                        perf_active_q <= 1'b0;
                        state         <= S_IDLE;
                    end

                    // ----------------------------------------------
                    S_ERROR: begin
                        core_busy       <= 1'b0;
                        core_error      <= 1'b1;
                        core_done       <= 1'b0;
                        table_valid_q   <= 1'b0;
                        // Drain an in-flight read command (the burst reader always
                        // terminates with one final beat).
                        if (rd_data_valid && rd_data_ready) begin
                            rd_beats_seen_q <= rd_beats_seen_q + 16'd1;
                            if (rd_data_last ||
                                ((rd_beats_seen_q + 16'd1) >= rd_beats_expected_q))
                                rd_inflight_q <= 1'b0;
                        end
                        // Complete an in-flight write command with zero beats: the
                        // shell writer has no abort input and must never be left
                        // waiting for W data.
                        if (wr_inflight_q && !wr_last_sent_q &&
                            wr_data_valid && wr_data_ready) begin
                            wr_beats_sent_q <= wr_beats_sent_q + 16'd1;
                            if (wr_beats_sent_q == (row_beats_q - 16'd1))
                                wr_last_sent_q <= 1'b1;
                        end
                        if (wr_rsp_valid && wr_rsp_ready) wr_inflight_q <= 1'b0;
                        // A response captured before the fault/cancel must still
                        // release the write command (it may never pulse again).
                        if (rsp_seen_q) begin
                            rsp_seen_q    <= 1'b0;
                            wr_inflight_q <= 1'b0;
                        end
                    end

                    // ----------------------------------------------
                    S_CANCEL: begin
                        core_busy     <= 1'b0;
                        core_done     <= 1'b0;
                        table_valid_q <= 1'b0;
                        if (rd_data_valid && rd_data_ready) begin
                            rd_beats_seen_q <= rd_beats_seen_q + 16'd1;
                            if (rd_data_last ||
                                ((rd_beats_seen_q + 16'd1) >= rd_beats_expected_q))
                                rd_inflight_q <= 1'b0;
                        end
                        if (wr_inflight_q && !wr_last_sent_q &&
                            wr_data_valid && wr_data_ready) begin
                            wr_beats_sent_q <= wr_beats_sent_q + 16'd1;
                            if (wr_beats_sent_q == (row_beats_q - 16'd1))
                                wr_last_sent_q <= 1'b1;
                        end
                        if (wr_rsp_valid && wr_rsp_ready) begin
                            wr_inflight_q <= 1'b0;
                            rsp_seen_q    <= 1'b0;
                        end
                        // A response captured just before the cancel (e.g. in
                        // DRAIN) releases the in-flight write command here.
                        if (rsp_seen_q) begin
                            rsp_seen_q    <= 1'b0;
                            wr_inflight_q <= 1'b0;
                        end
                        if (!rd_inflight_q && !wr_inflight_q) begin
                            if (cancel_soft_q) begin
                                core_error      <= 1'b0;
                                core_error_code <= E_NONE;
                            end
                            cancel_soft_q <= 1'b0;
                            state         <= S_IDLE;
                        end
                    end

                    default: begin
                        fail_op(E_AXI_PROTOCOL);
                    end
                endcase
            end
        end
    end

    /* verilator lint_off UNUSEDSIGNAL */
    // table_load_beats / activation_rd_beats / activation_wr_beats are read-only
    // evidence counters (hierarchical TB access); keep them referenced so lint
    // sees an intentional sink rather than dead logic.
    wire unused_cfg = |{table_valid_q, wr_rsp_error_code, wr_rsp_tag,
                        rd_data_error_code, out_lo_valid_q,
                        table_load_beats, activation_rd_beats, activation_wr_beats,
                        process_cycles, wait_cycles, total_cycles};
    /* verilator lint_on UNUSEDSIGNAL */

endmodule