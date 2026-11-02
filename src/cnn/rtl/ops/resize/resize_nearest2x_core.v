`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// resize_nearest2x_core.v
//
// Independent ABI 1.0 signed-INT8 nearest-neighbour x2 resize core.
//
// Logical operation (ONNX Resize with mode=nearest,
// coordinate_transformation_mode=asymmetric, nearest_mode=floor, N/C identity):
//
//     Y[c, 2*y + dy, 2*x + dx] = X[c, y, x]      for dy, dx in {0, 1}
//
// Input/output are NHWC8_A16 INT8 feature maps with
//   channel_alloc = align8(C)              (K = channel_alloc / 8, 1 or 2)
//   row_stride    = align16(channel_alloc * W)
// so a 64-bit chunk is one (pixel x, channel group g) and a 128-bit beat is two
// consecutive chunks of the row byte stream (little endian: [63:0] first).
//
// Capacity (v0.2 M3): the kernel parameters are *input* limits derived from the
// real graph, and the storage is a real per-node row extent, never a
// MAX_CH*MAX_W product:
//   MAX_CH = channel_alloc capacity (align8(C)); K = align8(C)/8 channel groups.
//            The real model needs C=256 (K=32) for up0, so the old K<=2 clamp is
//            gone.  Supported group counts are K=1 (one 64-bit chunk per pixel)
//            and even K>=2 (integral 128-bit beats); an odd K>1 has no integral
//            beat stream and is rejected with E_CFG at launch.
//   MAX_W  = input width limit, legal range 1..320.
//   MAX_H  = input height limit, legal range 1..320; 0 = inherit MAX_W
//            (legacy heredity, matching the router's RESIZE_MAX_H=0 rule; the
//            generated top always passes the resolved positive value).
//   MAX_IN_ROW_BYTES / MAX_OUT_ROW_BYTES = optional explicit real input/output
//            row capacities from the graph (0 = derive the legacy
//            align16(align8(MAX_CH)*MAX_W) / 2x form, which is byte-identical to
//            the second-batch capacities).  The line buffer depth is the input
//            row capacity, i.e. align8(C)*W bytes of one real input row.
// The output is always (2H, 2W) so it is 2..640 on both axes by construction;
// input 321 would imply output 642 and is rejected as E_UNSUPPORTED.  For the
// real up0 fragment (C=256, W=20) the elaborated capacity is exactly 5,120 B
// input row (= 640 chunks) and 10,240 B output row (= 640 beats).
//
// Data flow (one input row at a time, no whole-image buffer, no per-pixel DDR
// command):
//   1. One rd_cmd per input row:
//        addr = in_addr + y*in_row_stride, bytes = in_row_stride, tag = 1.
//      The slave returns in_row_stride/16 beats with rd_data_last on the final
//      beat.  Bad beat count / LAST / tag is E_AXI_PROTOCOL.  in_row_stride
//      <= align16(16*320) = 5,120 B, so the 16-bit command byte field is enough.
//   2. Each accepted beat holds two chunks.  Chunks are written one per cycle
//      into a 64-bit line buffer of W*K chunks (K <= 2, W <= 320 -> depth
//      <= 640 chunks = 5,120 B) with a single write port.  Only chunk indices
//      < W*K are stored, which drops the 8-byte row tail pad chunk of the
//      K=1 / odd-W case (that padding is *not* a pixel and is never repeated
//      horizontally).  rd_data_ready therefore alternates:
//      accept beat, write lo, write hi, ready again.
//   3. For dy = 0, 1 an output row oy = 2*y + dy is written with one wr_cmd
//      (addr = out_addr + oy*out_row_stride, bytes = out_row_stride, tag = 2)
//      followed by out_row_stride/16 beats (wr_data_last on the final one,
//      tag = 2) and then one wr_rsp is awaited (wr_rsp_error/wr_rsp_aborted is
//      a failure).  out_row_stride <= align16(16*640) = 10,240 B (640 beats),
//      so the 16-bit command byte field is enough.  The same buffered input
//      row feeds *both* output rows: the second output row never re-reads DDR.
//      Output beats use the floor mapping ix = ox >> 1:
//        K>=2 (even): each input pixel is K chunks = K/2 128-bit beats; every
//            beat is emitted twice (output pixels 2*ix and 2*ix+1) and both
//            chunks of a beat are adjacent channel groups of that pixel, so no
//            lane is ever half-swapped.  The stream is W*K beats per row.
//        K=1 (channel_alloc 8) : beat b = {chunk_b, chunk_b}; one beat covers
//            the two output pixels 2b and 2b+1.
//      wr_data / wr_data_last / wr_data_tag are held stable while
//      wr_data_ready is low; indices only advance on a real handshake.
//   4. Every input row is read exactly once (rd_cmd count == H) and every
//      output row is written exactly once (wr_cmd count == 2H).  The line
//      buffer is reused for input row y+1 only after *both* output rows of
//      row y have their write response.  core_done is a 1-cycle pulse after
//      the final write response; input-read completion alone never ends the
//      operation.  Output rows are already 16-byte aligned by construction
//      (out_row = 2*channel_alloc*W), so no output padding beat is ever
//      generated (a mismatch is rejected).
//
// Error codes (docs/op_abi.md 9.1):
//   0x0005 E_STRIDE        cfg_stride/cfg_pad not 0/0, or row stride / byte
//                          counts do not match the NHWC8_A16 geometry.
//   0x0001 E_CFG           H/W/C is 0, C above MAX_CH, channel_alloc not
//                          align8(C)/not a legal 8*K group count (odd K>1 has
//                          no integral beat stream), cfg_cout inconsistent with
//                          C/alloc, or the chunk/beat count exceeds the real
//                          row capacity.
//   0x0002 E_UNSUPPORTED   input H/W above MAX_H/MAX_W capacity (e.g. W=321
//                          would imply output 642), or C above MAX_CH.
//   0x0003 E_ADDR_ALIGN    base address or row stride not 16-byte aligned.
//   0x0004 E_ADDR_RANGE    null base address, or addr+bytes wraps 32 bits.
//   0x0009 E_AXI_PROTOCOL  wrong beat count / LAST / tag or write response tag.
//   0x000A E_CORE_INTERNAL transport error with code 0, or write response
//                          error/aborted with code 0.
// On error all DDR traffic stops, core_error/core_error_code stay set (no DONE)
// until the next core_start or reset.  soft_reset/abort return to IDLE.
//
// Payload note (contract B.1 vs D.1): channel_alloc is *derived* from cfg_cin
// (align8(cfg_cin)) and the payload copies are only cross-checked, so both
// wirings work: w7.high16 -> cfg_cout carrying channel_alloc and/or w8 ->
// cfg_quant_cfg[15:0] carrying channel_alloc.  cfg_quant_cfg[15:0] == 0 means
// "not provided"; cfg_cout may be 0, C or channel_alloc (C == cfg_cin here).
//
// Storage: one 64-bit synchronous line buffer of MAX_K * MAX_IN_W chunks when
// the capacity is derived (align8(C)*W bytes = one input row), or the explicit
// MAX_IN_ROW_BYTES/8 real row when supplied; 1 write port, 1 read port, read
// latency 1 clk, no reset of the data array, plus output beat holding
// registers.  No multiplier, no DSP, no asynchronous multi-index array, no
// whole-image / whole-output-row buffer.  The real up0 fragment elaborates
// 640 chunks = 5,120 B (C=256, W=20), i.e. exactly one real input row.
// -----------------------------------------------------------------------------

module resize_nearest2x_rowbuf #(
    parameter integer DEPTH = 32,
    parameter integer AW    = 5
) (
    input  wire          clk,
    input  wire          wr_en,
    input  wire [AW-1:0] wr_addr,
    input  wire [63:0]   wr_data,
    input  wire          rd_en,
    input  wire [AW-1:0] rd_addr,
    output reg  [63:0]   rd_data
);

    reg [63:0] mem [0:DEPTH-1];

    // Single synchronous write port; the core drives one chunk per cycle.
    always @(posedge clk) begin
        if (wr_en) begin
            mem[wr_addr] <= wr_data;
        end
    end

    // Single synchronous read port, latency 1 clk.  The data array has no
    // reset: every address read by the core was written earlier in the same
    // input-row iteration.
    always @(posedge clk) begin
        if (rd_en) begin
            rd_data <= mem[rd_addr];
        end
    end

endmodule


module resize_nearest2x_core #(
    parameter integer MAX_CH = 16,   // channel_alloc (align8(C)) capacity
    parameter integer MAX_W  = 32,   // *input* width capacity,  legal 1..320
    parameter integer MAX_H  = 0,    // *input* height capacity: 0 = inherit
                                     // MAX_W (legacy heredity, same rule the
                                     // router uses when RESIZE_MAX_H=0); the
                                     // generated top passes the resolved value
    // v0.2 M3 optional explicit real row capacities from the graph.  0 keeps the
    // legacy derivation (align16(align8(MAX_CH)*MAX_W) and its 2x output form),
    // so every existing elaboration stays byte-identical.
    parameter integer MAX_IN_ROW_BYTES  = 0,
    parameter integer MAX_OUT_ROW_BYTES = 0
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

    output reg         wr_cmd_valid,
    input  wire        wr_cmd_ready,
    output reg  [31:0] wr_cmd_addr_byte,
    output reg  [15:0] wr_cmd_bytes,
    output reg  [7:0]  wr_cmd_tag,
    output reg         wr_data_valid,
    input  wire        wr_data_ready,
    output reg  [127:0] wr_data,
    output reg         wr_data_last,
    output reg  [7:0]  wr_data_tag,
    input  wire        wr_rsp_valid,
    output reg         wr_rsp_ready,
    input  wire        wr_rsp_error,
    input  wire        wr_rsp_aborted,
    input  wire [15:0] wr_rsp_error_code,
    input  wire [7:0]  wr_rsp_tag
);

    // v0.2 M3 capacity derivation.  MAX_CH is the real channel_alloc capacity
    // (the real model passes C=256 -> K=32); no K<=2 clamp.  The line buffer is
    // one input row of 64-bit chunks (align8(C)*W bytes), and the output beat
    // cap is the x2 output row; explicit real row capacities win when provided.
    localparam integer MAX_K_FULL = (MAX_CH + 7) / 8;
    localparam integer MAX_K      = (MAX_K_FULL < 1) ? 1 : MAX_K_FULL;
    // MAX_W/MAX_H are *input* limits.  The output is always 2x, so the legal
    // input range is 1..320 (output 2..640); clamp defensively so an
    // over-wide elaboration parameter can never widen the frozen capacity.
    localparam integer MAX_IN_W   = (MAX_W > 320) ? 320 : ((MAX_W < 1) ? 1 : MAX_W);
    // MAX_H=0 keeps the legacy heredity used by the router (H cap = resolved
    // MAX_W); an explicit positive value caps the input height directly.
    localparam integer MAX_IN_H   = (MAX_H <= 0) ? MAX_IN_W :
                                    ((MAX_H > 320) ? 320 : MAX_H);
    localparam integer MAX_CHUNKS = MAX_K * MAX_IN_W;
    localparam integer IN_CHUNKS_CAP = (MAX_IN_ROW_BYTES > 0) ?
                                       ((MAX_IN_ROW_BYTES + 7) / 8) : MAX_CHUNKS;
    localparam integer OUT_BEATS_CAP = (MAX_OUT_ROW_BYTES > 0) ?
                                       (MAX_OUT_ROW_BYTES / 16) : MAX_CHUNKS;
    localparam integer AW         = (IN_CHUNKS_CAP <= 1) ? 1 : $clog2(IN_CHUNKS_CAP);
    // rd_idx counts 64-bit chunks of one input pixel (0..K); it must be able to
    // hold K itself so the "all pairs emitted" comparison is exact for K=32.
    localparam integer RD_W       = (MAX_K < 2) ? 1 : $clog2(MAX_K + 1);

    localparam [15:0] E_NONE          = 16'h0000;
    localparam [15:0] E_CFG           = 16'h0001;
    localparam [15:0] E_UNSUPPORTED   = 16'h0002;
    localparam [15:0] E_ADDR_ALIGN    = 16'h0003;
    localparam [15:0] E_ADDR_RANGE    = 16'h0004;
    localparam [15:0] E_STRIDE        = 16'h0005;
    localparam [15:0] E_AXI_PROTOCOL  = 16'h0009;
    localparam [15:0] E_CORE_INTERNAL = 16'h000A;

    localparam [4:0] ST_IDLE      = 5'd0;
    localparam [4:0] ST_START     = 5'd1;
    localparam [4:0] ST_RD_CMD    = 5'd2;
    localparam [4:0] ST_RD_DATA   = 5'd3;
    localparam [4:0] ST_RD_STORE  = 5'd4;
    localparam [4:0] ST_WR_CMD    = 5'd5;
    localparam [4:0] ST_PIX_RD    = 5'd6;
    localparam [4:0] ST_PIX_WAIT  = 5'd7;
    localparam [4:0] ST_PIX_LATCH = 5'd8;
    localparam [4:0] ST_PIX_FORM  = 5'd9;
    localparam [4:0] ST_PIX_SEND  = 5'd10;
    localparam [4:0] ST_WR_RSP    = 5'd11;
    localparam [4:0] ST_ERROR     = 5'd12;

    localparam [7:0] TAG_INPUT  = 8'h01;
    localparam [7:0] TAG_OUTPUT = 8'h02;

    // ---- registered configuration / geometry ------------------------------
    reg [2:0]  core_start_pipe;
    reg [31:0] ca_s1;             // align8(cfg_cin)                (stage 1)
    reg [31:0] in_stride_s1;      // align16(alloc * W)             (stage 1)
    reg [31:0] out_stride_s1;     // align16(alloc * 2W)            (stage 1)
    reg [31:0] chunks_s1;         // K * W                          (stage 1)
    reg [31:0] in_bytes_s2;       // align16(H * in_stride)         (stage 2)
    reg [31:0] out_bytes_s2;      // align16(2H * out_stride)       (stage 2)
    reg        in_wrap_q;         // in_addr + in_bytes wraps 32 bit
    reg        out_wrap_q;        // out_addr + out_bytes wraps 32 bit

    wire [31:0] ca_next        = ((({16'b0, cfg_cin}) + 32'd7) >> 3) << 3;
    wire [31:0] row_bytes_next = ca_next * {16'b0, cfg_w};
    wire [31:0] in_stride_next = ((row_bytes_next + 32'd15) >> 4) << 4;
    wire [31:0] out_stride_next = (((row_bytes_next << 1) + 32'd15) >> 4) << 4;
    wire [31:0] chunks_next    = (ca_next >> 3) * {16'b0, cfg_w};
    wire [32:0] in_end_next    = {1'b0, cfg_in_addr_byte} + {1'b0, cfg_in_bytes};
    wire [32:0] out_end_next   = {1'b0, cfg_out_addr_byte} + {1'b0, cfg_out_bytes};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            core_start_pipe <= 3'b0;
            ca_s1 <= 32'b0; in_stride_s1 <= 32'b0; out_stride_s1 <= 32'b0;
            chunks_s1 <= 32'b0; in_bytes_s2 <= 32'b0; out_bytes_s2 <= 32'b0;
            in_wrap_q <= 1'b0; out_wrap_q <= 1'b0;
        end else begin
            core_start_pipe <= {core_start_pipe[1:0], core_start};
            ca_s1           <= ca_next;
            in_stride_s1    <= in_stride_next;
            out_stride_s1   <= out_stride_next;
            chunks_s1       <= chunks_next;
            in_bytes_s2     <= (((in_stride_s1 * {16'b0, cfg_h}) + 32'd15) >> 4) << 4;
            out_bytes_s2    <= (((out_stride_s1 * ({16'b0, cfg_h} << 1)) + 32'd15) >> 4) << 4;
            in_wrap_q       <= in_end_next[32];
            out_wrap_q      <= out_end_next[32];
        end
    end

    wire [31:0] in_beats_w  = in_stride_s1 >> 4;
    wire [31:0] out_beats_w = out_stride_s1 >> 4;

    // ---- run-time control state -------------------------------------------
    reg [4:0]  state;
    reg [15:0] y;              // input row index
    reg        dy;             // 0/1 output row within the pair
    reg [15:0] k_used;         // channel groups (1 or 2)
    reg [15:0] chunks_used;    // K * W line-buffer chunks
    reg [15:0] in_beats;       // beats per input row (protocol check)
    reg [15:0] h_used;
    reg [15:0] w_used;
    reg [31:0] in_row_addr;    // running input row base address
    reg [31:0] out_row_addr;   // running output row base address

    // ---- input row store path ---------------------------------------------
    reg         rb_wr_en;
    reg  [AW-1:0] rb_wr_addr;
    reg  [63:0] rb_wr_data;
    reg  [15:0] rd_beat_idx;
    reg  [15:0] rd_chunk_idx;
    reg         rd_phase;
    reg         rd_latched_last;
    reg  [63:0] rd_chunk_low;
    reg  [63:0] rd_chunk_high;

    // ---- line buffer read / output beat path ------------------------------
    reg         rb_rd_en;
    reg  [AW-1:0] rb_rd_addr;
    wire [63:0] rb_rd_data;
    reg  [15:0] opix;          // input pixel index feeding the current beats
    reg  [15:0] chunk_base;    // opix * k_used, advanced by k_used
    reg  [RD_W-1:0] rd_idx;    // 64-bit chunk index inside the current input pixel
    reg  [1:0]  osub;          // output beat index inside the current pixel
    reg  [63:0] chunk_lo;
    reg  [63:0] chunk_hi;

    wire [15:0] pix_chunk_addr = chunk_base + {{(16-RD_W){1'b0}}, rd_idx};
    wire [15:0] last_opix      = w_used - 16'd1;
    wire [15:0] last_y         = h_used - 16'd1;

    resize_nearest2x_rowbuf #(
        .DEPTH (IN_CHUNKS_CAP),
        .AW    (AW)
    ) u_linebuf (
        .clk     (clk),
        .wr_en   (rb_wr_en),
        .wr_addr (rb_wr_addr),
        .wr_data (rb_wr_data),
        .rd_en   (rb_rd_en),
        .rd_addr (rb_rd_addr),
        .rd_data (rb_rd_data)
    );

    task automatic fail_core;
        input [15:0] code;
        begin
            core_error_code <= code;
            core_error      <= 1'b1;
            core_busy       <= 1'b0;
            core_done       <= 1'b0;
            wr_data_valid   <= 1'b0;
            state           <= ST_ERROR;
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            y <= 16'b0;
            dy <= 1'b0;
            k_used <= 16'b0;
            chunks_used <= 16'b0;
            in_beats <= 16'b0;
            h_used <= 16'b0;
            w_used <= 16'b0;
            in_row_addr <= 32'b0;
            out_row_addr <= 32'b0;

            rb_wr_en <= 1'b0;
            rb_wr_addr <= {AW{1'b0}};
            rb_wr_data <= 64'b0;
            rd_beat_idx <= 16'b0;
            rd_chunk_idx <= 16'b0;
            rd_phase <= 1'b0;
            rd_latched_last <= 1'b0;
            rd_chunk_low <= 64'b0;
            rd_chunk_high <= 64'b0;

            rb_rd_en <= 1'b0;
            rb_rd_addr <= {AW{1'b0}};
            opix <= 16'b0;
            chunk_base <= 16'b0;
            rd_idx <= {RD_W{1'b0}};
            osub <= 2'b0;
            chunk_lo <= 64'b0;
            chunk_hi <= 64'b0;

            core_busy <= 1'b0;
            core_done <= 1'b0;
            core_error <= 1'b0;
            core_error_code <= E_NONE;

            rd_cmd_valid <= 1'b0;
            rd_cmd_addr_byte <= 32'b0;
            rd_cmd_bytes <= 16'b0;
            rd_cmd_tag <= TAG_INPUT;
            rd_data_ready <= 1'b0;

            wr_cmd_valid <= 1'b0;
            wr_cmd_addr_byte <= 32'b0;
            wr_cmd_bytes <= 16'b0;
            wr_cmd_tag <= TAG_OUTPUT;
            wr_data_valid <= 1'b0;
            wr_data <= 128'b0;
            wr_data_last <= 1'b0;
            wr_data_tag <= TAG_OUTPUT;
            wr_rsp_ready <= 1'b0;
        end else begin
            // Single-cycle strobes default to inactive; level signals
            // (wr_data_valid, wr_data, wr_data_last) are only advanced on a
            // real handshake so they stay stable under backpressure.
            rd_cmd_valid <= 1'b0;
            rd_data_ready <= 1'b0;
            rb_wr_en <= 1'b0;
            rb_rd_en <= 1'b0;
            wr_cmd_valid <= 1'b0;
            wr_rsp_ready <= 1'b0;

            if (core_soft_reset) begin
                state <= ST_IDLE;
                core_busy <= 1'b0;
                core_done <= 1'b0;
                core_error <= 1'b0;
                core_error_code <= E_NONE;
                wr_data_valid <= 1'b0;
            end else if (core_abort) begin
                state <= ST_IDLE;
                core_busy <= 1'b0;
                core_done <= 1'b0;
                wr_data_valid <= 1'b0;
            end else begin
                case (state)
                    ST_IDLE: begin
                        core_busy <= 1'b0;
                        core_done <= 1'b0;
                        core_error <= 1'b0;
                        core_error_code <= E_NONE;
                        if (core_start_pipe[2]) begin
                            state <= ST_START;
                        end
                    end

                    // Static configuration validation.  All geometry values
                    // are already registered (stage 1/2), so this is a
                    // register-to-register comparison chain, not a dynamic
                    // address/multiply path.
                    ST_START: begin
                        if ((cfg_stride != 2'd0) || (cfg_pad != 2'd0)) begin
                            fail_core(E_STRIDE);
                        end else if ((cfg_h == 16'b0) || (cfg_w == 16'b0) ||
                                     (cfg_cin == 16'b0)) begin
                            fail_core(E_CFG);
                        end else if (cfg_cin > MAX_CH) begin
                            // C above the elaborated channel capacity.  The old
                            // hard C<=16 / channel_alloc in {8,16} ABI clamp is
                            // gone: the real model needs C=256 (K=32), and any
                            // align8(C)=8*K group count is a legal pixel stream.
                            fail_core(E_CFG);
                        end else if ((ca_s1 == 32'd0) || (ca_s1 > MAX_CH)) begin
                            fail_core(E_CFG);
                        end else if ((ca_s1 != 32'd8) && ca_s1[3]) begin
                            // K>1 must be even: an odd group count has no
                            // integral 128-bit beat stream (K=1 is the single
                            // 64-bit-chunk case and stays supported).
                            fail_core(E_CFG);
                        end else if ((cfg_h > MAX_IN_H) || (cfg_w > MAX_IN_W)) begin
                            // v0.2 capacity: input H/W <= MAX_H/MAX_W (each
                            // capped at the 320/320 input range implied by the
                            // 640 output limit).  This is the over-capacity
                            // rejection (e.g. W=321 -> 642).
                            fail_core(E_UNSUPPORTED);
                        end else if (cfg_quant_cfg[15:0] != 16'b0 &&
                                     cfg_quant_cfg[15:0] != ca_s1[15:0]) begin
                            fail_core(E_CFG);
                        end else if ((cfg_cout != 16'b0) &&
                                     (cfg_cout != cfg_cin) &&
                                     (cfg_cout != ca_s1[15:0])) begin
                            fail_core(E_CFG);
                        end else if ((cfg_in_addr_byte[3:0] != 4'b0) ||
                                     (cfg_out_addr_byte[3:0] != 4'b0) ||
                                     (cfg_in_row_stride_bytes[3:0] != 4'b0) ||
                                     (cfg_out_row_stride_bytes[3:0] != 4'b0)) begin
                            fail_core(E_ADDR_ALIGN);
                        end else if ((cfg_in_addr_byte == 32'b0) ||
                                     (cfg_out_addr_byte == 32'b0)) begin
                            fail_core(E_ADDR_RANGE);
                        end else if (in_wrap_q || out_wrap_q) begin
                            fail_core(E_ADDR_RANGE);
                        end else if ((cfg_in_row_stride_bytes == 32'b0) ||
                                     (cfg_out_row_stride_bytes == 32'b0) ||
                                     (cfg_in_row_stride_bytes[15:0] == 16'b0) ||
                                     (cfg_out_row_stride_bytes[15:0] == 16'b0)) begin
                            fail_core(E_STRIDE);
                        end else if ((cfg_in_row_stride_bytes != in_stride_s1) ||
                                     (cfg_out_row_stride_bytes != out_stride_s1) ||
                                     (cfg_in_bytes != in_bytes_s2) ||
                                     (cfg_out_bytes != out_bytes_s2)) begin
                            fail_core(E_STRIDE);
                        end else if ((chunks_s1 > IN_CHUNKS_CAP) ||
                                     (in_beats_w > IN_CHUNKS_CAP) ||
                                     (out_beats_w > OUT_BEATS_CAP)) begin
                            // real row capacity: the line buffer holds one input
                            // row and the beat counter covers the x2 output row
                            fail_core(E_CFG);
                        end else begin
                            k_used        <= ca_s1[15:0] >> 3;
                            chunks_used   <= chunks_s1[15:0];
                            in_beats      <= in_beats_w[15:0];
                            h_used        <= cfg_h;
                            w_used        <= cfg_w;
                            y             <= 16'b0;
                            dy            <= 1'b0;
                            in_row_addr   <= cfg_in_addr_byte;
                            out_row_addr  <= cfg_out_addr_byte;
                            rd_beat_idx   <= 16'b0;
                            rd_chunk_idx  <= 16'b0;
                            rd_phase      <= 1'b0;
                            rd_latched_last <= 1'b0;
                            opix          <= 16'b0;
                            chunk_base    <= 16'b0;
                            rd_idx        <= {RD_W{1'b0}};
                            osub          <= 2'b0;
                            wr_data_valid <= 1'b0;
                            core_busy     <= 1'b1;
                            core_done     <= 1'b0;
                            core_error    <= 1'b0;
                            core_error_code <= E_NONE;
                            state         <= ST_RD_CMD;
                        end
                    end

                    // One command per input row, never per pixel.
                    ST_RD_CMD: begin
                        rd_cmd_valid <= 1'b1;
                        rd_cmd_addr_byte <= in_row_addr;
                        rd_cmd_bytes <= cfg_in_row_stride_bytes[15:0];
                        rd_cmd_tag <= TAG_INPUT;
                        if (rd_cmd_valid && rd_cmd_ready) begin
                            rd_cmd_valid <= 1'b0;
                            rd_beat_idx <= 16'b0;
                            rd_chunk_idx <= 16'b0;
                            rd_phase <= 1'b0;
                            rd_latched_last <= 1'b0;
                            rd_data_ready <= 1'b1;
                            state <= ST_RD_DATA;
                        end
                    end

                    ST_RD_DATA: begin
                        rd_data_ready <= 1'b1;
                        if (rd_data_valid && rd_data_ready) begin
                            if (rd_data_error) begin
                                fail_core((rd_data_error_code == 16'b0) ?
                                          E_CORE_INTERNAL : rd_data_error_code);
                            end else if (rd_data_tag != TAG_INPUT) begin
                                fail_core(E_AXI_PROTOCOL);
                            end else if (rd_beat_idx >= in_beats) begin
                                fail_core(E_AXI_PROTOCOL);
                            end else if (rd_data_last !=
                                         ((rd_beat_idx + 16'd1) >= in_beats)) begin
                                fail_core(E_AXI_PROTOCOL);
                            end else begin
                                rd_chunk_low  <= rd_data[63:0];
                                rd_chunk_high <= rd_data[127:64];
                                rd_latched_last <= rd_data_last;
                                rd_data_ready <= 1'b0;
                                state <= ST_RD_STORE;
                            end
                        end
                    end

                    // Two chunks per beat, one 64-bit line-buffer write per
                    // cycle (single write port).  Chunks at index >= W*K are
                    // the 16-byte row tail padding and are dropped.
                    ST_RD_STORE: begin
                        if (!rd_phase) begin
                            if (rd_chunk_idx < chunks_used) begin
                                rb_wr_en   <= 1'b1;
                                rb_wr_addr <= rd_chunk_idx[AW-1:0];
                                rb_wr_data <= rd_chunk_low;
                            end
                            rd_chunk_idx <= rd_chunk_idx + 16'd1;
                            rd_phase <= 1'b1;
                        end else begin
                            if (rd_chunk_idx < chunks_used) begin
                                rb_wr_en   <= 1'b1;
                                rb_wr_addr <= rd_chunk_idx[AW-1:0];
                                rb_wr_data <= rd_chunk_high;
                            end
                            rd_chunk_idx <= rd_chunk_idx + 16'd1;
                            rd_phase <= 1'b0;
                            if (rd_latched_last) begin
                                // Row fully buffered: emit output rows
                                // 2*y (dy=0) and 2*y+1 (dy=1).
                                opix <= 16'b0;
                                chunk_base <= 16'b0;
                                rd_idx <= {RD_W{1'b0}};
                                osub <= 2'b0;
                                dy <= 1'b0;
                                state <= ST_WR_CMD;
                            end else begin
                                rd_beat_idx <= rd_beat_idx + 16'd1;
                                rd_data_ready <= 1'b1;
                                state <= ST_RD_DATA;
                            end
                        end
                    end

                    ST_WR_CMD: begin
                        wr_cmd_valid <= 1'b1;
                        wr_cmd_addr_byte <= out_row_addr;
                        wr_cmd_bytes <= cfg_out_row_stride_bytes[15:0];
                        wr_cmd_tag <= TAG_OUTPUT;
                        if (wr_cmd_valid && wr_cmd_ready) begin
                            wr_cmd_valid <= 1'b0;
                            opix <= 16'b0;
                            chunk_base <= 16'b0;
                            rd_idx <= {RD_W{1'b0}};
                            osub <= 2'b0;
                            state <= ST_PIX_RD;
                        end
                    end

                    // Issue one line-buffer read (registered request, so the
                    // synchronous RAM samples it one cycle later; the data is
                    // latched in ST_PIX_LATCH).
                    ST_PIX_RD: begin
                        if (rd_idx < k_used) begin
                            rb_rd_en   <= 1'b1;
                            rb_rd_addr <= pix_chunk_addr[AW-1:0];
                            state <= ST_PIX_WAIT;
                        end else begin
                            state <= ST_PIX_FORM;
                        end
                    end

                    ST_PIX_WAIT: begin
                        state <= ST_PIX_LATCH;
                    end

                    // Latch one 64-bit chunk.  rd_idx[0]==0 stores the first
                    // chunk of a 128-bit beat, rd_idx[0]==1 the second one.  An
                    // even K>=2 always completes whole beats, so the beat is
                    // formed as soon as the odd chunk is latched (K=1 falls back
                    // to the ST_PIX_RD -> ST_PIX_FORM single-chunk path).
                    ST_PIX_LATCH: begin
                        if (!rd_idx[0]) begin
                            chunk_lo <= rb_rd_data;
                        end else begin
                            chunk_hi <= rb_rd_data;
                        end
                        rd_idx <= rd_idx + {{(RD_W-1){1'b0}}, 1'b1};
                        if (rd_idx[0]) begin
                            state <= ST_PIX_FORM;
                        end else begin
                            state <= ST_PIX_RD;
                        end
                    end

                    // K=1: the same 8-byte chunk feeds the two output pixels
                    // 2b and 2b+1 (one beat per input pixel).  K=2: one beat is
                    // the whole pixel and is emitted twice (output pixels 2b and
                    // 2b+1).  K>=4: the output pixels 2b and 2b+1 are the K/2
                    // beats of that input pixel each, so the second copy re-reads
                    // the line buffer from beat 0 (the beat order is
                    // 0..K/2-1 for pixel 2b, then 0..K/2-1 for pixel 2b+1).
                    ST_PIX_FORM: begin
                        wr_data <= (k_used == 16'd1) ? {chunk_lo, chunk_lo}
                                                     : {chunk_hi, chunk_lo};
                        wr_data_tag <= TAG_OUTPUT;
                        wr_data_valid <= 1'b1;
                        wr_data_last <= (k_used == 16'd1) ? (opix == last_opix) :
                                        ((k_used >= 16'd4) && (osub == 1'b1) &&
                                         (rd_idx == k_used) && (opix == last_opix));
                        // K>=4 keeps the copy index across the pair scan; the
                        // legacy single/double beat path restarts it each form.
                        osub <= (k_used >= 16'd4) ? osub : 2'b0;
                        state <= ST_PIX_SEND;
                    end

                    // wr_data/wr_data_last/wr_data_tag are held stable while
                    // wr_data_ready is low; only a real handshake advances.
                    // K=1 emits one beat; K>=2 emits each channel-group pair
                    // twice (once per duplicated output pixel).
                    ST_PIX_SEND: begin
                        if (wr_data_valid && wr_data_ready) begin
                            if (k_used <= 16'd2) begin
                                // legacy: K=1 single beat, K=2 the same beat held
                                // for the two output pixels 2b and 2b+1
                                if (osub == ((k_used == 16'd1) ? 2'd0 : 2'd1)) begin
                                    wr_data_valid <= 1'b0;
                                    if (opix == last_opix) begin
                                        state <= ST_WR_RSP;
                                    end else begin
                                        opix <= opix + 16'd1;
                                        chunk_base <= chunk_base + k_used;
                                        rd_idx <= {RD_W{1'b0}};
                                        osub <= 2'b0;
                                        state <= ST_PIX_RD;
                                    end
                                end else begin
                                    osub <= osub + 2'd1;
                                    wr_data_last <= (opix == last_opix);
                                end
                            end else begin
                                // K>=4: one beat per handshake; after the last
                                // pair of copy 0 restart the pair scan for copy 1
                                wr_data_valid <= 1'b0;
                                if (rd_idx == k_used) begin
                                    if (osub == 1'b0) begin
                                        osub <= 2'b1;
                                        rd_idx <= {RD_W{1'b0}};
                                        state <= ST_PIX_RD;
                                    end else if (opix == last_opix) begin
                                        state <= ST_WR_RSP;
                                    end else begin
                                        opix <= opix + 16'd1;
                                        chunk_base <= chunk_base + k_used;
                                        rd_idx <= {RD_W{1'b0}};
                                        osub <= 2'b0;
                                        state <= ST_PIX_RD;
                                    end
                                end else begin
                                    state <= ST_PIX_RD;
                                end
                            end
                        end
                    end

                    ST_WR_RSP: begin
                        wr_rsp_ready <= 1'b1;
                        if (wr_rsp_valid && wr_rsp_ready) begin
                            wr_rsp_ready <= 1'b0;
                            if (wr_rsp_error || wr_rsp_aborted) begin
                                fail_core((wr_rsp_error_code == 16'b0) ?
                                          E_CORE_INTERNAL : wr_rsp_error_code);
                            end else if (wr_rsp_tag != TAG_OUTPUT) begin
                                fail_core(E_AXI_PROTOCOL);
                            end else begin
                                out_row_addr <= out_row_addr +
                                                cfg_out_row_stride_bytes;
                                if (!dy) begin
                                    dy <= 1'b1;
                                    state <= ST_WR_CMD;
                                end else begin
                                    in_row_addr <= in_row_addr +
                                                   cfg_in_row_stride_bytes;
                                    if (y == last_y) begin
                                        core_busy <= 1'b0;
                                        core_done <= 1'b1;
                                        state <= ST_IDLE;
                                    end else begin
                                        y <= y + 16'd1;
                                        state <= ST_RD_CMD;
                                    end
                                end
                            end
                        end
                    end

                    ST_ERROR: begin
                        core_busy <= 1'b0;
                        core_done <= 1'b0;
                        wr_data_valid <= 1'b0;
                        // core_error stays asserted (no DONE) until the next
                        // core_start or reset; a new start re-validates.
                        if (core_start_pipe[2]) begin
                            core_error <= 1'b0;
                            core_error_code <= E_NONE;
                            state <= ST_START;
                        end
                    end

                    default: begin
                        fail_core(E_CORE_INTERNAL);
                    end
                endcase
            end
        end
    end

    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_cfg = |{cfg_weight_addr_byte, cfg_bias_addr_byte,
                        cfg_quant_addr_byte, cfg_x_zero_point, cfg_weight_bytes,
                        cfg_bias_bytes, cfg_quant_bytes, cfg_act_cfg,
                        cfg_conv_cfg};
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
