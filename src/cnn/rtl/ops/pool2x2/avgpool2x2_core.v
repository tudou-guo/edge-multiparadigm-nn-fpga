`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// avgpool2x2_core.v -- E1-P3 signed INT8 2x2 / stride-2 average-pool core
//
// Reusable operator core (not model specific).  It reuses the verified
// maxpool2x2_core stream/row-buffer machinery with one semantic difference:
// every output pixel is the exact rounded mean of the four valid pixels,
//   S = a + b + c + d,  q = ties_away(|S| / 4),  y = sat8(sign(S) * q),
// which is the P2 GlobalAveragePool rounding rule applied per 2x2 window.
//
//   * NHWC8_A16 rows are read one normalized shell row command at a time.
//   * Two row buffers hold input rows 2*oy and 2*oy+1.
//   * Every output pixel reads columns 2*ox and 2*ox+1 through the row-buffer
//     parallel channel banks.  The bank count is MAX_CH/8, not fixed at 8.
//   * The mean is never clamped to zero and padding never enters a window.
//   * Only even H/W is accepted, so a partial window is never dropped.
//   * The output row is packed through a 64-bit row buffer, then emitted as
//     128-bit AXI beats.  This preserves chunk continuity across source pixels
//     and writes zero pad chunks when row_stride has 16-byte alignment padding.
//   * MAX_W sizes the per-pixel row-buffer banks and the output chunk buffer;
//     MAX_H (new in v0.2) is a build-time height-capacity guard.  MAX_H = 0
//     keeps the legacy behaviour (no height check at all), so every old
//     configuration elaborates and behaves exactly as before.
//
// ABI 1.0 is frozen.  AveragePool uses only existing Conv-field semantics:
//   CONV_CFG = 0x00000404  (stride=2, pad=0, ACT_BYPASS=1)
//   QUANT_CFG = 0, ACT_CFG = 0.
// -----------------------------------------------------------------------------

module avgpool2x2_core #(
    parameter integer MAX_CH = 64,
    parameter integer MAX_W  = 256,
    // Build-time maximum feature-map height.  0 means "do not check" and is the
    // legacy default; any positive value rejects cfg_h > MAX_H at START.
    parameter integer MAX_H  = 0
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

    localparam integer MAX_K = (MAX_CH + 7) / 8;
    // Row-buffer depth must cover the 16-byte aligned output row, which can be a
    // chunk longer than (K*W)/2 when the output row width is odd.
    localparam integer MAX_OUT_CHUNKS = (MAX_K * MAX_W + 3) / 2 + 2;

    localparam [15:0] E_NONE          = 16'h0000;
    localparam [15:0] E_CFG           = 16'h0001;
    localparam [15:0] E_UNSUPPORTED   = 16'h0002;
    localparam [15:0] E_ADDR_ALIGN    = 16'h0003;
    localparam [15:0] E_ADDR_RANGE    = 16'h0004;
    localparam [15:0] E_STRIDE        = 16'h0005;
    localparam [15:0] E_AXI_PROTOCOL  = 16'h0009;
    localparam [15:0] E_CORE_INTERNAL = 16'h000A;

    localparam [4:0] ST_IDLE          = 5'd0;
    localparam [4:0] ST_RD_CMD        = 5'd1;
    localparam [4:0] ST_RD_DATA       = 5'd2;
    localparam [4:0] ST_RD_STORE      = 5'd3;
    localparam [4:0] ST_WR_CMD        = 5'd4;
    localparam [4:0] ST_CALC_RD0      = 5'd5;
    localparam [4:0] ST_CALC_W0       = 5'd6;
    localparam [4:0] ST_CALC_C0       = 5'd7;
    localparam [4:0] ST_POOL_LATCH    = 5'd8;
    localparam [4:0] ST_OUT_WRITE     = 5'd9;
    localparam [4:0] ST_OUT_PAD       = 5'd10;
    localparam [4:0] ST_STREAM_RD     = 5'd11;
    localparam [4:0] ST_STREAM_LATCH  = 5'd12;
    localparam [4:0] ST_STREAM_ACCEPT = 5'd13;
    localparam [4:0] ST_ROW_FLUSH     = 5'd14;
    localparam [4:0] ST_FLUSH_WAIT    = 5'd15;
    localparam [4:0] ST_WR_RSP        = 5'd16;
    localparam [4:0] ST_ERROR         = 5'd17;
    localparam [4:0] ST_DONE          = 5'd18;
    localparam [4:0] ST_CALC_W1       = 5'd19;
    localparam [4:0] ST_CALC_C1       = 5'd20;
    localparam [4:0] ST_STREAM_W      = 5'd21;

    localparam [7:0] TAG_INPUT  = 8'h01;
    localparam [7:0] TAG_OUTPUT = 8'h02;

    reg [4:0]  state;
    reg [31:0] oy;
    reg [31:0] ox;
    reg [15:0] k_used;
    reg [15:0] in_row_beats;
    reg [15:0] out_row_beats;
    reg [15:0] out_row_chunks;
    reg [15:0] logical_out_chunks;
    reg [15:0] pad_chunks;
    reg [15:0] hout;
    reg [15:0] wout;

    // Input row-buffer write path.
    reg         rb_wr_en;
    reg         rb_wr_sel;   // target row buffer captured with rb_wr_en
    reg  [15:0] rb_wr_pixel;
    reg  [3:0]  rb_wr_group;
    reg  [63:0] rb_wr_chunk;
    reg         rd_buf_sel;
    reg  [15:0] rd_beat_idx;
    reg  [15:0] rd_pixel;
    reg  [3:0]  rd_group;
    reg  [63:0] rd_chunk_low;
    reg  [63:0] rd_chunk_high;
    reg         rd_phase;
    reg         rd_latched_last;

    // Row-buffer read path.
    reg         rb_rd_en;
    reg  [15:0] rb_rd_pixel;
    wire [MAX_K*64-1:0] rb_a_chunks;
    wire [MAX_K*64-1:0] rb_b_chunks;
    reg  [MAX_K*64-1:0] row_a0;
    reg  [MAX_K*64-1:0] row_b0;
    reg  [MAX_K*64-1:0] row_a1;
    reg  [MAX_K*64-1:0] row_b1;
    reg  [MAX_K*64-1:0] pool_chunks;

    // Output row buffer and stream packer.
    reg         out_wr_en;
    reg  [15:0] out_wr_addr;
    reg  [63:0] out_wr_chunk;
    reg         out_rd_en;
    reg  [15:0] out_rd_addr;
    wire [63:0] out_rd_chunk;
    reg  [15:0] out_group;
    reg  [15:0] pad_idx;
    reg  [15:0] stream_idx;
    reg  [31:0] out_chunk_cursor;
    reg  [63:0] beat_lo;
    reg         beat_lo_valid;

    // Geometry pipeline: keep wide multiply/shift chains out of the core
    // control-register CE/D paths. Configuration is stable for an instruction;
    // the sequencer waits on core_busy, so the added launch latency is ABI-safe.
    reg [2:0]  core_start_pipe;
    reg [31:0] k_used_s1, alloc_bytes_s1, hout_s1, wout_s1;
    reg [31:0] in_stride_s2, out_stride_s2, logical_chunks_s2, out_chunks_s2;
    reg [31:0] in_bytes_s3, out_bytes_s3;
    wire [31:0] k_used_w       = k_used_s1;
    wire [31:0] alloc_bytes_w  = alloc_bytes_s1;
    wire [31:0] hout_w         = hout_s1;
    wire [31:0] wout_w         = wout_s1;
    wire [31:0] in_stride_w    = in_stride_s2;
    wire [31:0] out_stride_w   = out_stride_s2;
    wire [31:0] in_bytes_w     = in_bytes_s3;
    wire [31:0] out_bytes_w    = out_bytes_s3;
    wire [31:0] logical_chunks_w = logical_chunks_s2;
    wire [31:0] out_chunks_w    = out_chunks_s2;
    wire [31:0] k_used_next     = ({16'b0, cfg_cin} + 32'd7) >> 3;
    wire [31:0] alloc_bytes_next = k_used_next << 3;
    wire [31:0] hout_next       = {16'b0, cfg_h} >> 1;
    wire [31:0] wout_next       = {16'b0, cfg_w} >> 1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            core_start_pipe <= 3'b0;
            k_used_s1 <= 0; alloc_bytes_s1 <= 0; hout_s1 <= 0; wout_s1 <= 0;
            in_stride_s2 <= 0; out_stride_s2 <= 0; logical_chunks_s2 <= 0; out_chunks_s2 <= 0;
            in_bytes_s3 <= 0; out_bytes_s3 <= 0;
        end else begin
            core_start_pipe <= {core_start_pipe[1:0], core_start};
            k_used_s1 <= k_used_next;
            alloc_bytes_s1 <= alloc_bytes_next;
            hout_s1 <= hout_next;
            wout_s1 <= wout_next;
            in_stride_s2 <= (((alloc_bytes_s1 * {16'b0, cfg_w}) + 32'd15) >> 4) << 4;
            out_stride_s2 <= (((alloc_bytes_s1 * wout_s1) + 32'd15) >> 4) << 4;
            logical_chunks_s2 <= k_used_s1 * wout_s1;
            out_chunks_s2 <= ((((alloc_bytes_s1 * wout_s1) + 32'd15) >> 4) << 4) >> 3;
            in_bytes_s3 <= (((in_stride_s2 * {16'b0, cfg_h}) + 32'd15) >> 4) << 4;
            out_bytes_s3 <= (((out_stride_s2 * hout_s1) + 32'd15) >> 4) << 4;
        end
    end

    // W4: transaction-owned addresses. Advance only after the final input
    // beat has been stored, or after the output row's successful response.
    // All additions retain the original modulo-2^32 address behavior.
    reg [31:0] input_row_addr;
    reg [31:0] calc_out_addr;

    // Combinational next chunk cursor for the input row-store path.
    wire        rb_have_pixel = (rd_pixel < cfg_w);
    wire        rb_next_group_last = (rd_group == (k_used[3:0] - 4'd1));

    task automatic fail_core;
        input [15:0] code;
        begin
            core_error_code <= code;
            core_error      <= 1'b1;
            core_busy       <= 1'b0;
            core_done       <= 1'b0;
            state           <= ST_ERROR;
        end
    endtask

    task automatic advance_chunk_cursor;
        begin
            if (rb_next_group_last) begin
                rd_group <= 4'd0;
                rd_pixel <= rd_pixel + 16'd1;
            end else begin
                rd_group <= rd_group + 4'd1;
            end
        end
    endtask

    // Signed INT8 average lanes.  Invalid lanes beyond cfg_cin are forced to
    // zero so the aligned NHWC8 storage always carries the canonical padding
    // value (padding never participates in a window).
    wire [MAX_K*64-1:0] pool_chunks_w;
    genvar g, h;
    generate
        for (g = 0; g < MAX_K; g = g + 1) begin : GEN_GROUP
            for (h = 0; h < 8; h = h + 1) begin : GEN_LANE
                wire signed [7:0] lane_y;
                avgpool2x2_lane u_lane (
                    .a (row_a0[g*64 + h*8 +: 8]),
                    .b (row_a1[g*64 + h*8 +: 8]),
                    .c (row_b0[g*64 + h*8 +: 8]),
                    .d (row_b1[g*64 + h*8 +: 8]),
                    .y (lane_y)
                );
                assign pool_chunks_w[g*64 + h*8 +: 8] =
                    (((g * 8) + h) < cfg_cin) ? lane_y : 8'b0;
            end
        end
    endgenerate

    maxpool2x2_rowbuf #(
        .MAX_CH (MAX_CH),
        .MAX_W  (MAX_W),
        .MAX_K  (MAX_K)
    ) u_rowbuf_a (
        .clk      (clk),
        .rst_n    (rst_n),
        .wr_en    (rb_wr_en && (rb_wr_sel == 1'b0)),
        .wr_pixel (rb_wr_pixel),
        .wr_group (rb_wr_group),
        .wr_chunk (rb_wr_chunk),
        .rd_en    (rb_rd_en),
        .rd_pixel (rb_rd_pixel),
        .rd_chunks(rb_a_chunks)
    );

    maxpool2x2_rowbuf #(
        .MAX_CH (MAX_CH),
        .MAX_W  (MAX_W),
        .MAX_K  (MAX_K)
    ) u_rowbuf_b (
        .clk      (clk),
        .rst_n    (rst_n),
        .wr_en    (rb_wr_en && (rb_wr_sel == 1'b1)),
        .wr_pixel (rb_wr_pixel),
        .wr_group (rb_wr_group),
        .wr_chunk (rb_wr_chunk),
        .rd_en    (rb_rd_en),
        .rd_pixel (rb_rd_pixel),
        .rd_chunks(rb_b_chunks)
    );

    spram #(
        .DP   (MAX_OUT_CHUNKS),
        .DW   (64),
        .PIPE (1)
    ) u_out_rowbuf (
        .clk   (clk),
        .wdata (out_wr_chunk),
        .wen   (out_wr_en),
        .waddr (out_wr_addr),
        .ren   (out_rd_en),
        .raddr (out_rd_addr),
        .rdata (out_rd_chunk)
    );

    integer gi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            oy <= 32'b0;
            ox <= 32'b0;
            k_used <= 16'b0;
            in_row_beats <= 16'b0;
            out_row_beats <= 16'b0;
            out_row_chunks <= 16'b0;
            logical_out_chunks <= 16'b0;
            pad_chunks <= 16'b0;
            hout <= 16'b0;
            wout <= 16'b0;

            rb_wr_en <= 1'b0;
            rb_wr_sel <= 1'b0;
            rb_wr_pixel <= 16'b0;
            rb_wr_group <= 4'b0;
            rb_wr_chunk <= 64'b0;
            rd_buf_sel <= 1'b0;
            rd_beat_idx <= 16'b0;
            rd_pixel <= 16'b0;
            rd_group <= 4'b0;
            rd_chunk_low <= 64'b0;
            rd_chunk_high <= 64'b0;
            rd_phase <= 1'b0;
            rd_latched_last <= 1'b0;

            rb_rd_en <= 1'b0;
            rb_rd_pixel <= 16'b0;
            row_a0 <= {MAX_K*64{1'b0}};
            row_b0 <= {MAX_K*64{1'b0}};
            row_a1 <= {MAX_K*64{1'b0}};
            row_b1 <= {MAX_K*64{1'b0}};
            pool_chunks <= {MAX_K*64{1'b0}};

            out_wr_en <= 1'b0;
            out_wr_addr <= 16'b0;
            out_wr_chunk <= 64'b0;
            out_rd_en <= 1'b0;
            out_rd_addr <= 16'b0;
            out_group <= 16'b0;
            pad_idx <= 16'b0;
            stream_idx <= 16'b0;
            out_chunk_cursor <= 32'b0;
            input_row_addr <= 32'b0;
            calc_out_addr <= 32'b0;
            beat_lo <= 64'b0;
            beat_lo_valid <= 1'b0;

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
            rd_cmd_valid <= 1'b0;
            rd_data_ready <= 1'b0;
            rb_wr_en <= 1'b0;
            rb_rd_en <= 1'b0;
            wr_cmd_valid <= 1'b0;
            out_wr_en <= 1'b0;
            out_rd_en <= 1'b0;
            wr_rsp_ready <= 1'b0;

            if (core_soft_reset) begin
                state <= ST_IDLE;
                core_busy <= 1'b0;
                core_done <= 1'b0;
                core_error <= 1'b0;
                core_error_code <= E_NONE;
                rd_buf_sel <= 1'b0;
                beat_lo_valid <= 1'b0;
                wr_data_valid <= 1'b0;
                out_chunk_cursor <= 32'b0;
                input_row_addr <= 32'b0;
                calc_out_addr <= 32'b0;
            end else if (core_abort) begin
                state <= ST_IDLE;
                core_busy <= 1'b0;
                core_done <= 1'b0;
                wr_data_valid <= 1'b0;
                beat_lo_valid <= 1'b0;
                out_chunk_cursor <= 32'b0;
                input_row_addr <= 32'b0;
                calc_out_addr <= 32'b0;
            end else begin
                case (state)
                    ST_IDLE: begin
                        core_busy <= 1'b0;
                        core_done <= 1'b0;
                        core_error <= 1'b0;
                        core_error_code <= E_NONE;
                        if (core_start_pipe[2]) begin
                            if (cfg_conv_cfg != 32'h00000404) begin
                                fail_core(E_UNSUPPORTED);
                            end else if ((cfg_stride != 2'd1) || (cfg_pad != 2'd0)) begin
                                fail_core(E_UNSUPPORTED);
                            end else if ((cfg_quant_cfg != 32'b0) || (cfg_act_cfg != 32'b0)) begin
                                fail_core(E_CFG);
                            end else if ((cfg_h == 16'b0) || (cfg_w == 16'b0) || (cfg_cin == 16'b0)) begin
                                fail_core(E_CFG);
                            end else if (((cfg_h & 16'h0001) != 16'b0) || ((cfg_w & 16'h0001) != 16'b0)) begin
                                fail_core(E_CFG);
                            end else if ((cfg_cin > MAX_CH) || (cfg_w > MAX_W)) begin
                                fail_core(E_UNSUPPORTED);
                            end else if ((MAX_H > 0) && (cfg_h > MAX_H)) begin
                                // v0.2 spatial capacity: height is a separate
                                // build-time limit.  MAX_H = 0 skips this check.
                                fail_core(E_UNSUPPORTED);
                            end else if ((cfg_in_addr_byte[3:0] != 4'b0) ||
                                         (cfg_out_addr_byte[3:0] != 4'b0) ||
                                         (cfg_in_row_stride_bytes[3:0] != 4'b0) ||
                                         (cfg_out_row_stride_bytes[3:0] != 4'b0)) begin
                                fail_core(E_ADDR_ALIGN);
                            end else if ((cfg_in_addr_byte == 32'b0) || (cfg_out_addr_byte == 32'b0)) begin
                                fail_core(E_ADDR_RANGE);
                            end else if ((cfg_in_row_stride_bytes == 32'b0) ||
                                         (cfg_out_row_stride_bytes == 32'b0) ||
                                         (cfg_in_row_stride_bytes[15:0] == 16'b0) ||
                                         (cfg_out_row_stride_bytes[15:0] == 16'b0)) begin
                                fail_core(E_STRIDE);
                            end else if ((cfg_in_row_stride_bytes != in_stride_w) ||
                                         (cfg_out_row_stride_bytes != out_stride_w) ||
                                         (cfg_in_bytes != in_bytes_w) ||
                                         (cfg_out_bytes != out_bytes_w)) begin
                                fail_core(E_STRIDE);
                            end else if ((logical_chunks_w > 16'hffff) ||
                                         (out_chunks_w > MAX_OUT_CHUNKS)) begin
                                fail_core(E_UNSUPPORTED);
                            end else begin
                                k_used <= k_used_w[15:0];
                                in_row_beats <= cfg_in_row_stride_bytes[15:0] >> 4;
                                out_row_beats <= cfg_out_row_stride_bytes[15:0] >> 4;
                                out_row_chunks <= out_chunks_w[15:0];
                                logical_out_chunks <= logical_chunks_w[15:0];
                                pad_chunks <= out_chunks_w[15:0] - logical_chunks_w[15:0];
                                hout <= hout_w[15:0];
                                wout <= wout_w[15:0];
                                oy <= 32'b0;
                                ox <= 32'b0;
                                input_row_addr <= cfg_in_addr_byte;
                                calc_out_addr <= cfg_out_addr_byte;
                                out_chunk_cursor <= 32'b0;
                                rd_buf_sel <= 1'b0;
                                beat_lo_valid <= 1'b0;
                                wr_data_valid <= 1'b0;
                                core_busy <= 1'b1;
                                core_done <= 1'b0;
                                core_error <= 1'b0;
                                core_error_code <= E_NONE;
                                state <= ST_RD_CMD;
                            end
                        end
                    end

                    ST_RD_CMD: begin
                        rd_cmd_valid <= 1'b1;
                        rd_cmd_addr_byte <= input_row_addr;
                        rd_cmd_bytes <= cfg_in_row_stride_bytes[15:0];
                        rd_cmd_tag <= TAG_INPUT;
                        if (rd_cmd_valid && rd_cmd_ready) begin
                            rd_cmd_valid <= 1'b0;
                            rd_beat_idx  <= 16'b0;
                            rd_pixel     <= 16'b0;
                            rd_group     <= 4'b0;
                            rd_phase     <= 1'b0;
                            rd_latched_last <= 1'b0;
                            state        <= ST_RD_DATA;
                        end
                    end

                    ST_RD_DATA: begin
                        rd_data_ready <= 1'b1;
                        if (rd_data_valid && rd_data_ready) begin
                            if (rd_data_error) begin
                                fail_core((rd_data_error_code == 16'b0) ? E_CORE_INTERNAL : rd_data_error_code);
                            end else if (rd_data_tag != TAG_INPUT) begin
                                fail_core(E_AXI_PROTOCOL);
                            end else if (rd_beat_idx >= in_row_beats) begin
                                fail_core(E_AXI_PROTOCOL);
                            end else if (rd_data_last != ((rd_beat_idx + 16'd1) >= in_row_beats)) begin
                                fail_core(E_AXI_PROTOCOL);
                            end else begin
                                rd_chunk_low <= rd_data[63:0];
                                rd_chunk_high <= rd_data[127:64];
                                rd_latched_last <= rd_data_last;
                                rd_phase <= 1'b0;
                                rd_data_ready <= 1'b0;
                                state <= ST_RD_STORE;
                            end
                        end
                    end

                    ST_RD_STORE: begin
                        if (!rd_phase) begin
                            if (rb_have_pixel) begin
                                rb_wr_en    <= 1'b1;
                                rb_wr_sel   <= rd_buf_sel;
                                rb_wr_pixel <= rd_pixel;
                                rb_wr_group <= rd_group;
                                rb_wr_chunk <= rd_chunk_low;
                            end
                            advance_chunk_cursor();
                            rd_phase <= 1'b1;
                        end else begin
                            if (rb_have_pixel) begin
                                rb_wr_en    <= 1'b1;
                                rb_wr_sel   <= rd_buf_sel;
                                rb_wr_pixel <= rd_pixel;
                                rb_wr_group <= rd_group;
                                rb_wr_chunk <= rd_chunk_high;
                            end
                            advance_chunk_cursor();
                            if (rd_latched_last) begin
                                input_row_addr <= input_row_addr + cfg_in_row_stride_bytes;
                                rd_beat_idx <= 16'b0;
                                rd_phase <= 1'b0;
                                if (rd_buf_sel == 1'b0) begin
                                    rd_buf_sel <= 1'b1;
                                    rd_pixel <= 16'b0;
                                    rd_group <= 4'b0;
                                    state <= ST_RD_CMD;
                                end else begin
                                    state <= ST_WR_CMD;
                                end
                            end else begin
                                rd_beat_idx <= rd_beat_idx + 16'd1;
                                rd_phase <= 1'b0;
                                state <= ST_RD_DATA;
                            end
                        end
                    end

                    ST_WR_CMD: begin
                        wr_cmd_valid <= 1'b1;
                        wr_cmd_addr_byte <= calc_out_addr;
                        wr_cmd_bytes <= cfg_out_row_stride_bytes[15:0];
                        wr_cmd_tag <= TAG_OUTPUT;
                        if (wr_cmd_valid && wr_cmd_ready) begin
                            wr_cmd_valid <= 1'b0;
                            ox <= 32'b0;
                            out_group <= 16'b0;
                            pad_idx <= 16'b0;
                            stream_idx <= 16'b0;
                            out_chunk_cursor <= 32'b0;
                            beat_lo_valid <= 1'b0;
                            state <= ST_CALC_RD0;
                        end
                    end

                    ST_CALC_RD0: begin
                        // Issue the column 2*ox read.  rb_rd_en/rb_rd_pixel are
                        // registered FSM outputs, so the row buffer samples the
                        // new address one cycle later and the PIPE=1 SRAM data
                        // is valid two cycles later, i.e. in ST_CALC_C0.
                        rb_rd_en    <= 1'b1;
                        rb_rd_pixel <= (ox << 1);
                        state       <= ST_CALC_W0;
                    end

                    ST_CALC_W0: begin
                        state <= ST_CALC_C0;
                    end

                    ST_CALC_C0: begin
                        // Column 2*ox data is now valid; latch it and launch the
                        // column 2*ox+1 read.
                        row_a0 <= rb_a_chunks;
                        row_b0 <= rb_b_chunks;
                        rb_rd_en    <= 1'b1;
                        rb_rd_pixel <= (ox << 1) + 16'd1;
                        state       <= ST_CALC_W1;
                    end

                    ST_CALC_W1: begin
                        state <= ST_CALC_C1;
                    end

                    ST_CALC_C1: begin
                        // Column 2*ox+1 data is now valid.
                        row_a1 <= rb_a_chunks;
                        row_b1 <= rb_b_chunks;
                        state  <= ST_POOL_LATCH;
                    end

                    ST_POOL_LATCH: begin
                        pool_chunks <= pool_chunks_w;
                        out_group   <= 16'b0;
                        state       <= ST_OUT_WRITE;
                    end

                    ST_OUT_WRITE: begin
                        if (out_group < k_used) begin
                            out_wr_en   <= 1'b1;
                            out_wr_addr <= out_chunk_cursor[15:0];
                            out_chunk_cursor <= out_chunk_cursor + 32'd1;
                            out_wr_chunk <= pool_chunks[out_group*64 +: 64];
                            if ((out_group + 16'd1) >= k_used) begin
                                out_group <= 16'b0;
                                if ((ox + 32'd1) >= {16'b0, wout}) begin
                                    pad_idx <= 16'b0;
                                    if (pad_chunks != 16'b0)
                                        state <= ST_OUT_PAD;
                                    else
                                        state <= ST_STREAM_RD;
                                end else begin
                                    ox <= ox + 32'd1;
                                    state <= ST_CALC_RD0;
                                end
                            end else begin
                                out_group <= out_group + 16'd1;
                            end
                        end
                    end

                    ST_OUT_PAD: begin
                        out_wr_en    <= 1'b1;
                        out_wr_addr  <= out_chunk_cursor[15:0];
                        out_chunk_cursor <= out_chunk_cursor + 32'd1;
                        out_wr_chunk <= 64'b0;
                        if ((pad_idx + 16'd1) >= pad_chunks) begin
                            state <= ST_STREAM_RD;
                        end else begin
                            pad_idx <= pad_idx + 16'd1;
                        end
                    end

                    ST_STREAM_RD: begin
                        // Issue one 64-bit output row-buffer read.  out_rd_en is
                        // registered, so the PIPE=1 SRAM data is valid two
                        // cycles later (captured in ST_STREAM_LATCH).
                        out_rd_en   <= 1'b1;
                        out_rd_addr <= stream_idx;
                        state       <= ST_STREAM_W;
                    end

                    ST_STREAM_W: begin
                        state <= ST_STREAM_LATCH;
                    end

                    ST_STREAM_LATCH: begin
                        if (!beat_lo_valid) begin
                            beat_lo       <= out_rd_chunk;
                            beat_lo_valid <= 1'b1;
                            if ((stream_idx + 16'd1) >= out_row_chunks) begin
                                stream_idx <= stream_idx + 16'd1;
                                state      <= ST_ROW_FLUSH;
                            end else begin
                                stream_idx <= stream_idx + 16'd1;
                                state      <= ST_STREAM_RD;
                            end
                        end else begin
                            wr_data      <= {out_rd_chunk, beat_lo};
                            wr_data_last <= ((stream_idx + 16'd1) >= out_row_chunks);
                            wr_data_tag  <= TAG_OUTPUT;
                            wr_data_valid<= 1'b1;
                            beat_lo_valid<= 1'b0;
                            state        <= ST_STREAM_ACCEPT;
                        end
                    end

                    ST_STREAM_ACCEPT: begin
                        if (wr_data_valid && wr_data_ready) begin
                            wr_data_valid <= 1'b0;
                            if (wr_data_last) begin
                                state <= ST_WR_RSP;
                            end else begin
                                stream_idx <= stream_idx + 16'd1;
                                state      <= ST_STREAM_RD;
                            end
                        end
                    end

                    ST_ROW_FLUSH: begin
                        if (!wr_data_valid) begin
                            wr_data       <= {64'b0, beat_lo};
                            wr_data_last  <= 1'b1;
                            wr_data_tag   <= TAG_OUTPUT;
                            wr_data_valid <= 1'b1;
                            beat_lo_valid <= 1'b0;
                            state         <= ST_FLUSH_WAIT;
                        end
                    end

                    ST_FLUSH_WAIT: begin
                        if (wr_data_valid && wr_data_ready) begin
                            wr_data_valid <= 1'b0;
                            state <= ST_WR_RSP;
                        end
                    end

                    ST_WR_RSP: begin
                        wr_rsp_ready <= 1'b1;
                        if (wr_rsp_valid && wr_rsp_ready) begin
                            wr_rsp_ready <= 1'b0;
                            if (!wr_rsp_error && !wr_rsp_aborted)
                                calc_out_addr <= calc_out_addr + cfg_out_row_stride_bytes;
                            if (wr_rsp_error || wr_rsp_aborted) begin
                                fail_core((wr_rsp_error_code == 16'b0) ? E_CORE_INTERNAL : wr_rsp_error_code);
                            end else if ((oy + 32'd1) >= {16'b0, hout}) begin
                                core_busy <= 1'b0;
                                core_done <= 1'b1;
                                core_error <= 1'b0;
                                state <= ST_IDLE;
                            end else begin
                                oy <= oy + 32'd1;
                                rd_buf_sel <= 1'b0;
                                beat_lo_valid <= 1'b0;
                                wr_data_valid <= 1'b0;
                                state <= ST_RD_CMD;
                            end
                        end
                    end

                    ST_ERROR: begin
                        core_busy <= 1'b0;
                        core_done <= 1'b0;
                    end

                    ST_DONE: begin
                        core_busy <= 1'b0;
                        core_done <= 1'b0;
                        state     <= ST_IDLE;
                    end

                    default: begin
                        fail_core(E_CORE_INTERNAL);
                    end
                endcase
            end
        end
    end

    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_cfg = |{cfg_weight_addr_byte, cfg_bias_addr_byte, cfg_quant_addr_byte,
                        cfg_cout, cfg_x_zero_point, cfg_weight_bytes,
                        cfg_bias_bytes, cfg_quant_bytes, rd_data_tag, wr_rsp_tag};
    /* verilator lint_on UNUSEDSIGNAL */

endmodule

