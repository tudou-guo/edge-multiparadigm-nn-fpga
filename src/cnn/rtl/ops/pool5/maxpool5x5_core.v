`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// maxpool5x5_core.v  --  v0.2 second batch (capacity-expanded, 5-bank ring)
//
// Reusable signed-INT8 max-pool core: kernel 5x5, stride 1, pad 2, dilation 1,
// ceil_mode 0, output shape == input shape ("Pool5" of SPPF / G6).
//
// ABI (frozen; see docs/E1_DIRECT_OP_CONTRACT_20261023_101651.md B.2):
//   * NHWC8_A16 rows, signed INT8, channel_alloc = align8(C) = 8*K with
//     K = 1 or even K >= 2 (real pool5: C=128 -> K=16).  The padding value is
//     always the signed-domain minimum -128, never zero and never a Conv zs_in.
//   * CONV_CFG = 0x00000420 (stride bits[3:2] = 0, pad bits[5:4] = 2,
//     ACT_BYPASS bit10 = 1), ACT_CFG = 0, QUANT_CFG[15:0] = channel_alloc.
//   * Pure signed max: spatial padding is -128, never 0, and the result is never
//     clamped to 0.  All-negative input stays all-negative.
//   * 17-word instruction payload unchanged (word 8 carries channel_alloc).
//
// Capacity is elaboration-time only (v0.2 M3: real graph capacity, no K<=2 clamp):
//     maxpool5x5_core #(MAX_CH, MAX_W, MAX_H)
//   MAX_CH = channel_alloc (align8(C)) capacity -> MAX_K = MAX_CH/8 groups
//            (pixel word = K*64 bit).  The real model needs pool5 C=128
//            (K=16); any K=1 or even K>=2 group count is a legal pixel stream.
//            An odd K>1 has no integral 128-bit beat stream -> E_CFG at launch.
//   MAX_W  = 1..640 input width capacity  -> line buffer depth / ring depth
//   MAX_H  = 1..640 input height capacity -> legality check only
// The on-chip word storage is (MAX_W + 5*MAX_W) * MAX_K * 8 B: one real row of
// the elaborated capacity, never the image area and never the global MAX_C*MAX_W
// product of another engine.  Increasing H only extends the row loop and the
// address range.
//
// Architecture (row-serial streaming, no whole-image buffer, no 25x DDR reads):
//   linebuf : one pixel word (MAX_K*64 bit) per column, depth MAX_W, 1 write port
//             + 1 read port, synchronous read latency 1.  Every input row is
//             written here exactly once (one DDR row read per row).
//   ring    : FIVE row banks, each MAX_W pixel words, 1 write port + 1 read port,
//             synchronous read latency 1, plus one 16-bit row tag per bank.
//             Input row r is written directly into bank (r mod 5); the vertical
//             window of output row oy is rows [max(0,oy-2), min(H-1,oy+2)],
//             which is always at most 5 consecutive rows, so all five window
//             rows live in the five banks.  A bank whose tag is outside that
//             range contributes -128.  There is no separate cur bank and no
//             copy step: the row leaves the horizontal pass and lands in the ring.
//   hmax    : 5-deep pixel-word shift pipeline, two -128 preloads and two -128
//             flushes implement the -128 left/right column padding; four signed
//             per-byte pairwise compares per group form a registered tree.
//             One pass serves both channel groups in parallel (1 pixel/cycle).
//   vmax    : five ring banks are read at the same column (five separate RAMs,
//             one word each), the row-tag mask injects the -128 top/bottom
//             padding, and a registered 5-point per-byte tree produces one
//             output pixel per cycle.  A 2-stage valid/ready pipeline lets the
//             reads issue every cycle while the previous pixels are compared and
//             packed; output beats are held stable under wr_data_ready backpressure.
//   output  : one wr_cmd per output row, 128-bit beats (K=2: one pixel per beat;
//             K=1: two pixels per beat, odd-W tail padded with zero), one wr_rsp.
//   flush   : the last two output rows use the same vertical path with a clipped
//             window (no extra DDR read, no separate "flush" datapath).
//
// DDR service: one rd_cmd per input row (bytes = in_row_stride, tag 1) with
// in_row_stride/16 beats, one wr_cmd per output row (bytes = out_row_stride,
// tag 2) with out_row_stride/16 beats and one write response per row.
//
// Error codes are unchanged from the first round:
//   0x0001 E_CFG, 0x0002 E_UNSUPPORTED, 0x0003 E_ADDR_ALIGN,
//   0x0004 E_ADDR_RANGE, 0x0005 E_STRIDE, 0x0009 E_AXI_PROTOCOL,
//   0x000A E_CORE_INTERNAL.
// -----------------------------------------------------------------------------

// One write port, one read port, synchronous read latency 1, DW-bit data.
// The array is deliberately not reset (only control/valid/pointer state is reset).
module maxpool5x5_ram #(
    parameter integer DW = 64,
    parameter integer DP = 64,
    parameter integer AW = 6
) (
    input  wire          clk,
    input  wire          we,
    input  wire [AW-1:0] waddr,
    input  wire [DW-1:0] wdata,
    input  wire          ren,
    input  wire [AW-1:0] raddr,
    output reg  [DW-1:0] rdata
);

    reg [DW-1:0] mem [0:DP-1];

    always @(posedge clk) begin
        if (we)  mem[waddr] <= wdata;
        if (ren) rdata <= mem[raddr];
    end

    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_clk = clk;
    /* verilator lint_on UNUSEDSIGNAL */

endmodule

// -----------------------------------------------------------------------------
// Signed per-byte 5-point max of five 64-bit channel words.
// Four pairwise 8-bit signed compares per lane (8 lanes / 64-bit word).
// The caller registers the result, so this is the combinational compare tree.
// -----------------------------------------------------------------------------
module maxpool5x5_max5x64 (
    input  wire [63:0] w0,
    input  wire [63:0] w1,
    input  wire [63:0] w2,
    input  wire [63:0] w3,
    input  wire [63:0] w4,
    output wire [63:0] y
);

    genvar ln;
    generate
        for (ln = 0; ln < 8; ln = ln + 1) begin : GEN_LANE
            wire signed [7:0] a0 = w0[ln*8 +: 8];
            wire signed [7:0] a1 = w1[ln*8 +: 8];
            wire signed [7:0] a2 = w2[ln*8 +: 8];
            wire signed [7:0] a3 = w3[ln*8 +: 8];
            wire signed [7:0] a4 = w4[ln*8 +: 8];
            wire signed [7:0] m01   = (a0 > a1) ? a0 : a1;
            wire signed [7:0] m23   = (a2 > a3) ? a2 : a3;
            wire signed [7:0] m0123 = (m01 > m23) ? m01 : m23;
            wire signed [7:0] m     = (m0123 > a4) ? m0123 : a4;
            assign y[ln*8 +: 8] = m;
        end
    endgenerate

endmodule

module maxpool5x5_core #(
    parameter integer MAX_CH = 16,
    parameter integer MAX_W  = 640,
    parameter integer MAX_H  = 640
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
    output wire        rd_data_ready,
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
    output wire        wr_data_valid,
    input  wire        wr_data_ready,
    output wire [127:0] wr_data,
    output wire        wr_data_last,
    output wire [7:0]  wr_data_tag,
    input  wire        wr_rsp_valid,
    output reg         wr_rsp_ready,
    input  wire        wr_rsp_error,
    input  wire        wr_rsp_aborted,
    input  wire [15:0] wr_rsp_error_code,
    input  wire [7:0]  wr_rsp_tag
);

    localparam integer MAX_K    = (MAX_CH + 7) / 8;
    localparam integer PX_W     = MAX_K * 64;
    localparam integer RING_ROWS = 5;
    localparam integer AW_ROW   = (MAX_W <= 1) ? 1 : $clog2(MAX_W);
    // v0.2 M3 wide-kernel path: K>=2 pixel words are serialized/assembled as
    // integral 128-bit beats (two 64-bit channel groups per beat).  MAX_K>=3
    // always implies MAX_K>=4 because odd K>1 is rejected, so PX_W>=256 for the
    // beat-extend path below.
    localparam integer WIDE_K   = (MAX_K >= 3) ? 1 : 0;
    localparam integer BEAT_PAD = (PX_W > 128) ? (PX_W - 128) : 0;

    localparam [63:0] NEG128_W = 64'h8080_8080_8080_8080;
    localparam [PX_W-1:0] NEG128_PX = {MAX_K{NEG128_W}};

    localparam [15:0] E_NONE          = 16'h0000;
    localparam [15:0] E_CFG           = 16'h0001;
    localparam [15:0] E_UNSUPPORTED   = 16'h0002;
    localparam [15:0] E_ADDR_ALIGN    = 16'h0003;
    localparam [15:0] E_ADDR_RANGE    = 16'h0004;
    localparam [15:0] E_STRIDE        = 16'h0005;
    localparam [15:0] E_AXI_PROTOCOL  = 16'h0009;
    localparam [15:0] E_CORE_INTERNAL = 16'h000A;

    localparam [3:0] ST_IDLE     = 4'd0;
    localparam [3:0] ST_CHECK    = 4'd1;
    localparam [3:0] ST_RD_CMD   = 4'd2;
    localparam [3:0] ST_RD_DATA  = 4'd3;
    localparam [3:0] ST_HMAX     = 4'd4;
    localparam [3:0] ST_VROW_SET = 4'd5;
    localparam [3:0] ST_WR_CMD   = 4'd6;
    localparam [3:0] ST_VERT     = 4'd7;
    localparam [3:0] ST_WR_RSP   = 4'd8;
    localparam [3:0] ST_ERROR    = 4'd9;

    localparam [7:0] TAG_INPUT  = 8'h01;
    localparam [7:0] TAG_OUTPUT = 8'h02;

    reg [3:0]  state;
    reg [2:0]  core_start_pipe;   // core_start is delayed 3 cycles (2 regs)

    // Registered geometry (wide multiply/align chains stay out of the launch path;
    // the launch check fires 3 cycles after core_start).
    reg [31:0] g1_stride, g2_stride, g2_bytes;
    reg [15:0] g1_k, g1_wchunks, g2_h, g2_w, g2_k, g2_wchunks;
    reg [15:0] g1_h, g1_w;

    // Captured per-instruction geometry.
    reg [15:0] h_q, w_q, k_q, in_beats_q, out_beats_q;
    reg [31:0] in_addr_q, out_addr_q;
    reg [15:0] y_q, oy_q;
    reg [2:0]  y_bank;

    // Input row capture / line buffer write path.
    reg        rd_phase;
    reg [15:0] rdi_q;
    reg [63:0] rd_hi_q;
    reg        rd_last_q;

    // Horizontal 5-point max pass.
    reg [15:0] hc_q;
    reg [PX_W-1:0] hsh0, hsh1, hsh2, hsh3, hsh4;
    reg        hmx_v;
    reg [PX_W-1:0] hmx_q;
    reg [15:0] hmx_j;

    // Ring row tags (newest written row index per bank; 16'hFFFF = never written).
    reg [15:0] ring_tag [0:RING_ROWS-1];

    // Vertical 5-point max pass / output packing.
    reg [15:0] v_lo, v_hi;
    reg [15:0] v_ox;              // read issue column
    reg        v_sv;              // stage B valid
    reg [15:0] v_sx;              // stage B column
    reg        v_cv;              // stage C valid
    reg [15:0] v_cx;              // stage C column
    reg [PX_W-1:0] v_cdata;       // stage C max result (already lane-masked)
    reg [63:0] pack_lo;           // K=1 even-pixel store for the 128-bit beat
    reg [15:0] ob_q;              // output beats of the current row

    // v0.2 M3 wide-capacity (MAX_K>=3) input capture: pixel word assembly from
    // K/2 consecutive 128-bit beats; K=1 falls back to the two-pixels-per-beat
    // path shared with the legacy elaboration.
    reg [PX_W-1:0] px_asm;        // pixel word being assembled
    reg [15:0] in_pair_q;         // beat index inside the current pixel
    reg [15:0] in_pix_q;          // pixel index inside the current row

    // v0.2 M3 wide-capacity output serializer: one latched pixel word is drained
    // as K/2 consecutive 128-bit beats; the vertical pipeline stalls meanwhile.
    reg [PX_W-1:0] pdata_q;       // latched output pixel word
    reg [15:0] pcol_q;            // its column (LAST detection)
    reg [15:0] pb_q;              // beat index inside the latched pixel
    reg        pbeats_q;          // a latched pixel has beats left to send

    // ------------------------------------------------------------------
    // RAM ports.
    // ------------------------------------------------------------------
    wire [PX_W-1:0] lb_rdata;
    wire        lb_we;
    wire [AW_ROW-1:0] lb_waddr;
    wire [PX_W-1:0] lb_wdata;
    wire        lb_ren;
    wire [AW_ROW-1:0] lb_raddr;

    wire [PX_W-1:0] ring_rdata [0:RING_ROWS-1];
    wire [PX_W-1:0] ring_wdata;
    wire [RING_ROWS-1:0] ring_we;
    wire [AW_ROW-1:0] ring_waddr;
    wire        ring_ren;
    wire [AW_ROW-1:0] ring_raddr;

    maxpool5x5_ram #(.DW(PX_W), .DP(MAX_W), .AW(AW_ROW)) u_linebuf (
        .clk   (clk),
        .we    (lb_we),
        .waddr (lb_waddr),
        .wdata (lb_wdata),
        .ren   (lb_ren),
        .raddr (lb_raddr),
        .rdata (lb_rdata)
    );

    genvar gr_b, gr_g;
    generate
        for (gr_b = 0; gr_b < RING_ROWS; gr_b = gr_b + 1) begin : GEN_RING
            maxpool5x5_ram #(.DW(PX_W), .DP(MAX_W), .AW(AW_ROW)) u_ring (
                .clk   (clk),
                .we    (ring_we[gr_b]),
                .waddr (ring_waddr),
                .wdata (ring_wdata),
                .ren   (ring_ren),
                .raddr (ring_raddr),
                .rdata (ring_rdata[gr_b])
            );
        end
    endgenerate

    // ------------------------------------------------------------------
    // Horizontal 5-point max datapath.
    //
    // Feed sequence: two -128 preloads, pixels 0..W-1 (linebuf reads), two -128
    // flushes (total W+4 feeds at pass cycles 1..W+4).  After feed cycle c the
    // shift register holds seq[c-1..c-5], i.e. the window centred on pixel
    // c-5; the compare tree is registered and written to the ring bank one cycle
    // later, on pass cycles 7..W+6.
    // ------------------------------------------------------------------
    wire hmax_on = (state == ST_HMAX);
    wire [15:0] hmax_px    = hc_q - 16'd2;
    wire hmax_feed   = hmax_on && (hc_q <= (w_q + 16'd4));
    wire hmax_rd     = hmax_on && (hc_q >= 16'd2) && (hc_q <= (w_q + 16'd1));
    wire hmx_tree_en = hmax_on && (hc_q >= 16'd6) && (hc_q <= (w_q + 16'd5));
    wire [PX_W-1:0] hmax_word = ((hc_q >= 16'd3) && (hc_q <= (w_q + 16'd2))) ?
                                lb_rdata : NEG128_PX;
    wire [PX_W-1:0] hmx_comb;

    generate
        for (gr_g = 0; gr_g < MAX_K; gr_g = gr_g + 1) begin : GEN_HMAX_TREE
            maxpool5x5_max5x64 u_hmax (
                .w0 (hsh0[gr_g*64 +: 64]),
                .w1 (hsh1[gr_g*64 +: 64]),
                .w2 (hsh2[gr_g*64 +: 64]),
                .w3 (hsh3[gr_g*64 +: 64]),
                .w4 (hsh4[gr_g*64 +: 64]),
                .y  (hmx_comb[gr_g*64 +: 64])
            );
        end
    endgenerate

    assign lb_ren   = hmax_rd;
    assign lb_raddr = hmax_px[AW_ROW-1:0];

    // ------------------------------------------------------------------
    // Input row capture.  One line-buffer word is one pixel.
    //   K=1        : one 128-bit beat carries two pixels (single write port ->
    //                two store cycles per beat); chunks with pixel index >= W
    //                are the row-stride tail padding and are dropped.
    //   even K>=2  : K/2 consecutive 128-bit beats form one pixel word (beat h
    //                covers the two channel groups 2h, 2h+1); the word is
    //                written once, when the last beat of the pixel arrives.
    // ------------------------------------------------------------------
    wire rd_hs = rd_data_valid && rd_data_ready;
    wire [63:0] pix64_lo = rd_data[63:0];
    wire [63:0] pix64_hi = rd_hi_q;
    wire [15:0] lb_px_w  = (k_q == 16'd1) ?
                           (rd_phase ? ((rdi_q << 1) + 16'd1) : (rdi_q << 1)) : rdi_q;
    wire [15:0] in_beats_per_px = (k_q == 16'd1) ? 16'd1 : (k_q >> 1);
    wire in_last_pair = (in_pair_q == (in_beats_per_px - 16'd1));
    wire [PX_W-1:0] beat_ext = {{BEAT_PAD{1'b0}}, rd_data};
    wire [PX_W-1:0] beat_slice = beat_ext << {in_pair_q[8:0], 7'b0};
    wire [PX_W-1:0] px_next = px_asm | beat_slice;
    wire [PX_W-1:0] lb_pix_ext;

    generate
        if (MAX_K <= 2) begin : GEN_LB_NARROW
            // legacy capacity: identical to the second-batch datapath
            if (MAX_K == 1) begin : GEN_LB_W1
                assign lb_pix_ext = (rd_phase ? pix64_hi : pix64_lo);
            end else begin : GEN_LB_W2
                assign lb_pix_ext = {{(PX_W-64){1'b0}},
                                     (rd_phase ? pix64_hi : pix64_lo)};
            end
            assign lb_we    = (state == ST_RD_DATA) &&
                              (((k_q == 16'd2) && rd_hs) ||
                               ((k_q == 16'd1) && (rd_phase || rd_hs))) &&
                              (lb_px_w < w_q);
            assign lb_waddr = lb_px_w[AW_ROW-1:0];
            assign lb_wdata = (k_q == 16'd2) ? rd_data : lb_pix_ext;
        end else begin : GEN_LB_WIDE
            assign lb_we    = (state == ST_RD_DATA) &&
                              ((k_q == 16'd1) ? (rd_phase || rd_hs)
                                              : (rd_hs && in_last_pair)) &&
                              ((k_q == 16'd1) ? (lb_px_w < w_q)
                                              : (in_pix_q < w_q));
            assign lb_waddr = (k_q == 16'd1) ? lb_px_w[AW_ROW-1:0]
                                             : in_pix_q[AW_ROW-1:0];
            assign lb_wdata = (k_q == 16'd1) ? lb_pix_ext : px_next;
        end
    endgenerate

    // Wide-capacity pixel assembly (MAX_K>=3 elaboration only).
    generate
        if (MAX_K >= 3) begin : GEN_ASM_WIDE
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    px_asm   <= {PX_W{1'b0}};
                    in_pair_q <= 16'd0;
                    in_pix_q  <= 16'd0;
                end else if (state == ST_RD_CMD && rd_cmd_valid && rd_cmd_ready) begin
                    px_asm    <= {PX_W{1'b0}};
                    in_pair_q <= 16'd0;
                    in_pix_q  <= 16'd0;
                end else if (rd_hs && (k_q != 16'd1)) begin
                    if (in_last_pair) begin
                        px_asm    <= {PX_W{1'b0}};
                        in_pair_q <= 16'd0;
                        in_pix_q  <= in_pix_q + 16'd1;
                    end else begin
                        px_asm    <= px_next;
                        in_pair_q <= in_pair_q + 16'd1;
                    end
                end
            end
        end
    endgenerate

    // ------------------------------------------------------------------
    // Ring write (horizontal max result -> bank y mod 5) and read (vertical pass).
    // ------------------------------------------------------------------
    assign ring_waddr = hmx_j[AW_ROW-1:0];
    assign ring_wdata = hmx_q;
    generate
        for (gr_b = 0; gr_b < RING_ROWS; gr_b = gr_b + 1) begin : GEN_RING_WE
            assign ring_we[gr_b] = hmx_v && (y_bank == gr_b[2:0]);
        end
    endgenerate

    wire vert_on = (state == ST_VERT);
    wire vert_adv;   // vertical pipeline advance (declared early, assigned below)
    // Read only on advancing cycles: during a stall the RAM output register must
    // keep the word of the pixel that stage B is about to promote.
    assign ring_ren   = vert_on && vert_adv && (v_ox < w_q);
    assign ring_raddr = v_ox[AW_ROW-1:0];

    // Window row mask: a bank contributes only when its tagged row lies inside
    // [max(0,oy-2), min(H-1,oy+2)]; every other bank (and every never-written
    // bank) contributes -128, which is exactly the top/bottom spatial padding.
    wire [RING_ROWS-1:0] bank_hit;
    (* max_fanout = 64 *) reg [RING_ROWS-1:0] bank_hit_q;
    // Row bounds and tags are stable throughout ST_WR_CMD/ST_VERT.  Resolve
    // the comparisons while issuing the existing write command, keeping
    // v_lo/v_hi -> compare -> mask out of the vertical maximum tree cycle.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            bank_hit_q <= {RING_ROWS{1'b0}};
        else if (state == ST_WR_CMD)
            bank_hit_q <= bank_hit;
    end
    generate
        for (gr_b = 0; gr_b < RING_ROWS; gr_b = gr_b + 1) begin : GEN_BANK_HIT
            assign bank_hit[gr_b] = (ring_tag[gr_b] >= v_lo) &&
                                    (ring_tag[gr_b] <= v_hi);
        end
    endgenerate

    wire [PX_W-1:0] ring_rmask [0:RING_ROWS-1];
    generate
        for (gr_b = 0; gr_b < RING_ROWS; gr_b = gr_b + 1) begin : GEN_RING_MASK
            assign ring_rmask[gr_b] = bank_hit_q[gr_b] ? ring_rdata[gr_b] : NEG128_PX;
        end
    endgenerate

    wire [PX_W-1:0] vres_raw;
    generate
        for (gr_g = 0; gr_g < MAX_K; gr_g = gr_g + 1) begin : GEN_VMAX
            maxpool5x5_max5x64 u_vmax (
                .w0 (ring_rmask[0][gr_g*64 +: 64]),
                .w1 (ring_rmask[1][gr_g*64 +: 64]),
                .w2 (ring_rmask[2][gr_g*64 +: 64]),
                .w3 (ring_rmask[3][gr_g*64 +: 64]),
                .w4 (ring_rmask[4][gr_g*64 +: 64]),
                .y  (vres_raw[gr_g*64 +: 64])
            );
        end
    endgenerate

    // Output lanes beyond cfg_cin carry the canonical zero padding, exactly like
    // maxpool2x2_core, so the stored row is well formed for the next op.
    wire [PX_W-1:0] vres_masked;
    genvar vm_g, vm_l;
    generate
        for (vm_g = 0; vm_g < MAX_K; vm_g = vm_g + 1) begin : GEN_VMASK
            for (vm_l = 0; vm_l < 8; vm_l = vm_l + 1) begin : GEN_VMASK_LANE
                assign vres_masked[vm_g*64 + vm_l*8 +: 8] =
                    (((vm_g*8) + vm_l) < cfg_cin) ? vres_raw[vm_g*64 + vm_l*8 +: 8] : 8'h00;
            end
        end
    endgenerate

    // ------------------------------------------------------------------
    // Vertical pipeline (2 stages) and output beat formation.
    //   cycle t   : issue ring read for column v_ox (read latency 1)
    //   cycle t+1 : tree over ring_rdata, register into stage C
    //   cycle t+2 : stage C pixel is packed/sent as 128-bit output beat(s)
    // Legacy capacity (MAX_K<=2): one beat per pixel for K=2, two pixels per beat
    // for K=1.  Wide capacity (MAX_K>=3): the stage-C pixel word is latched and
    // drained as K/2 consecutive 128-bit beats (beat h = channel groups 2h,2h+1)
    // while the vertical pipeline is stalled; that keeps the ring RAM output
    // register valid and needs no extra pixel FIFO.  The K=1 two-pixels-per-beat
    // path is shared by both elaborations.
    // ------------------------------------------------------------------
    wire [127:0] v_cdata128;
    generate
        if (MAX_K == 1) begin : GEN_CD128_W1
            assign v_cdata128 = {64'h0, v_cdata};
        end else begin : GEN_CD128_W2
            assign v_cdata128 = v_cdata;
        end
    endgenerate

    // legacy K=1 beat formation (also used by a wide elaboration at runtime K=1)
    wire beat_valid_k1 = v_cv && (v_cx[0] || (v_cx == (w_q - 16'd1)));
    wire beat_last_k1  = (v_cx == (w_q - 16'd1));
    wire [127:0] beat_data_k1 = v_cx[0] ? {v_cdata128[63:0], pack_lo} :
                                          {64'h0, v_cdata128[63:0]};
    // legacy K=2 beat formation
    wire beat_valid_k2 = v_cv;
    wire [127:0] beat_data_k2 = v_cdata128;
    // capacity-selected beat handshake (assigned in the generate below)
    wire        beat_valid;
    wire        beat_last;
    wire [127:0] beat_data;
    // wide K>=2 serializer: beat pb_q of the latched pixel word
    wire [15:0] pb_base = {pb_q[8:0], 7'b0};
    wire [127:0] wide_beat = pdata_q[pb_base +: 128];
    wire wide_beat_last = (pcol_q == (w_q - 16'd1)) &&
                          (pb_q == ((k_q >> 1) - 16'd1));

    generate
        if (MAX_K <= 2) begin : GEN_BEAT_NARROW
            // legacy capacity: K=1 packs two pixels per beat, K=2 one pixel
            assign vert_adv = !(wr_data_valid && !wr_data_ready);
            assign beat_valid = (k_q == 16'd2) ? beat_valid_k2 : beat_valid_k1;
            assign beat_last  = (k_q == 16'd2) ? (v_cx == (w_q - 16'd1))
                                               : beat_last_k1;
            assign beat_data  = (k_q == 16'd2) ? beat_data_k2 : beat_data_k1;
        end else begin : GEN_BEAT_WIDE
            assign vert_adv = (k_q == 16'd1) ?
                              !(wr_data_valid && !wr_data_ready) : !pbeats_q;
            assign beat_valid = (k_q == 16'd1) ? beat_valid_k1 : pbeats_q;
            assign beat_last  = (k_q == 16'd1) ? beat_last_k1 : wide_beat_last;
            assign beat_data  = wide_beat;
        end
    endgenerate

    assign wr_data_valid = vert_on &&
                           ((k_q == 16'd1) ? beat_valid_k1 : beat_valid);
    assign wr_data       = (k_q == 16'd1) ? beat_data_k1 : beat_data;
    assign wr_data_last  = (k_q == 16'd1) ? beat_last_k1 : beat_last;
    assign wr_data_tag   = TAG_OUTPUT;

    // Input data ready: even K accepts one beat per cycle (one pixel word per
    // K/2 beats); K=1 needs two store cycles per beat (single line-buffer port).
    assign rd_data_ready = (state == ST_RD_DATA) &&
                           ((k_q != 16'd1) || !rd_phase);

    // ------------------------------------------------------------------
    // Error helper.
    // ------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // Geometry pipeline.  Stage 1: W*channel_alloc -> aligned row stride and
    // logical pixel count.  Stage 2: H*row_stride -> aligned byte count.
    // ------------------------------------------------------------------
    wire [31:0] cfg_alloc     = {16'd0, cfg_quant_cfg[15:0]};
    wire [15:0] cfg_k         = cfg_quant_cfg[15:0] >> 3;
    wire [31:0] cfg_wh        = {16'd0, cfg_w} * cfg_alloc;
    wire [31:0] cfg_stride_e  = (cfg_wh + 32'd15) & 32'hFFFFFFF0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            g1_stride <= 32'd0; g1_k <= 16'd0;
            g2_stride <= 32'd0; g2_bytes <= 32'd0;
            g2_h <= 16'd0; g2_w <= 16'd0; g2_k <= 16'd0;
        end else begin
            g1_stride   <= cfg_stride_e;
            g1_k        <= cfg_k;
            g2_stride   <= g1_stride;
            g2_bytes    <= ({16'd0, g1_h} * g1_stride + 32'd15) & 32'hFFFFFFF0;
            g2_h        <= g1_h;
            g2_w        <= g1_w;
            g2_k        <= g1_k;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            g1_h <= 16'd0;
            g1_w <= 16'd0;
        end else begin
            g1_h <= cfg_h;
            g1_w <= cfg_w;
        end
    end

    wire [31:0] cin_align8 = ({16'd0, cfg_cin} + 32'd7) & 32'hFFFFFFF8;
    wire [32:0] in_end_addr  = {1'b0, cfg_in_addr_byte}  + {1'b0, cfg_in_bytes};
    wire [32:0] out_end_addr = {1'b0, cfg_out_addr_byte} + {1'b0, cfg_out_bytes};

    // ------------------------------------------------------------------
    // Main control.
    // ------------------------------------------------------------------
    integer ti;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            core_start_pipe <= 3'b000;

            h_q <= 16'd0; w_q <= 16'd0; k_q <= 16'd0;
            in_beats_q <= 16'd0; out_beats_q <= 16'd0;
            in_addr_q <= 32'd0; out_addr_q <= 32'd0;
            y_q <= 16'd0; oy_q <= 16'd0; y_bank <= 3'd0;

            rd_phase <= 1'b0; rdi_q <= 16'd0; rd_hi_q <= 64'd0; rd_last_q <= 1'b0;

            hc_q <= 16'd0;
            hsh0 <= {PX_W{1'b0}}; hsh1 <= {PX_W{1'b0}}; hsh2 <= {PX_W{1'b0}};
            hsh3 <= {PX_W{1'b0}}; hsh4 <= {PX_W{1'b0}};
            hmx_v <= 1'b0; hmx_q <= {PX_W{1'b0}}; hmx_j <= 16'd0;
            for (ti = 0; ti < RING_ROWS; ti = ti + 1) ring_tag[ti] <= 16'hFFFF;

            v_lo <= 16'd0; v_hi <= 16'd0; v_ox <= 16'd0;
            v_sv <= 1'b0; v_sx <= 16'd0; v_cv <= 1'b0; v_cx <= 16'd0;
            v_cdata <= {PX_W{1'b0}}; pack_lo <= 64'd0; ob_q <= 16'd0;

            pdata_q <= {PX_W{1'b0}}; pcol_q <= 16'd0; pb_q <= 16'd0;
            pbeats_q <= 1'b0;

            core_busy <= 1'b0; core_done <= 1'b0;
            core_error <= 1'b0; core_error_code <= E_NONE;

            rd_cmd_valid <= 1'b0; rd_cmd_addr_byte <= 32'd0;
            rd_cmd_bytes <= 16'd0; rd_cmd_tag <= TAG_INPUT;

            wr_cmd_valid <= 1'b0; wr_cmd_addr_byte <= 32'd0;
            wr_cmd_bytes <= 16'd0; wr_cmd_tag <= TAG_OUTPUT;
            wr_rsp_ready <= 1'b0;
        end else begin
            core_start_pipe <= {core_start_pipe[1:0], core_start};

            rd_cmd_valid <= 1'b0;
            wr_cmd_valid <= 1'b0;
            wr_rsp_ready <= 1'b0;

            if (core_soft_reset) begin
                state <= ST_IDLE;
                core_busy <= 1'b0;
                core_done <= 1'b0;
                core_error <= 1'b0;
                core_error_code <= E_NONE;
                hmx_v <= 1'b0;
                rd_phase <= 1'b0;
                v_sv <= 1'b0;
                v_cv <= 1'b0;
            end else if (core_abort) begin
                state <= ST_IDLE;
                core_busy <= 1'b0;
                core_done <= 1'b0;
                hmx_v <= 1'b0;
                rd_phase <= 1'b0;
                v_sv <= 1'b0;
                v_cv <= 1'b0;
            end else begin
                // ---- horizontal pipeline datapath (only in ST_HMAX) ----
                if (hmax_feed) begin
                    hsh0 <= hmax_word;
                    hsh1 <= hsh0;
                    hsh2 <= hsh1;
                    hsh3 <= hsh2;
                    hsh4 <= hsh3;
                end
                if (hmx_tree_en) begin
                    hmx_q <= hmx_comb;
                    hmx_j <= hc_q - 16'd6;
                end
                hmx_v <= hmx_tree_en;
                if (hmx_v) ring_tag[y_bank] <= y_q;

                case (state)
                    ST_IDLE: begin
                        core_busy <= 1'b0;
                        core_done <= 1'b0;
                        if (core_start_pipe[2]) state <= ST_CHECK;
                    end

                    // Config validation.  Every DDR-affecting field is checked
                    // before the first rd_cmd is issued.
                    ST_CHECK: begin
                        if (cfg_conv_cfg != 32'h0000_0420) begin
                            fail_core(E_UNSUPPORTED);
                        end else if ((cfg_stride != 2'd0) || (cfg_pad != 2'd2)) begin
                            fail_core(E_UNSUPPORTED);
                        end else if (cfg_act_cfg != 32'b0) begin
                            fail_core(E_CFG);
                        end else if (cfg_quant_cfg[31:16] != 16'b0) begin
                            fail_core(E_CFG);
                        end else if ((cfg_alloc == 32'd0) || (cfg_alloc > MAX_CH)) begin
                            // channel_alloc capacity: any align8(C)=8*K with
                            // 1<=K<=MAX_K is legal (real pool5: C=128 -> K=16);
                            // the old {8,16} ABI clamp is gone.
                            fail_core(E_CFG);
                        end else if (cfg_alloc != cin_align8) begin
                            fail_core(E_CFG);
                        end else if ((cfg_alloc != 32'd8) && cfg_alloc[3]) begin
                            // K>1 must be even: an odd group count has no
                            // integral 128-bit beat stream (K=1 is the two-pixel
                            // single-beat case and stays supported).
                            fail_core(E_CFG);
                        end else if ((cfg_cin == 16'd0) || (cfg_cout != cfg_cin)) begin
                            fail_core(E_CFG);
                        end else if (cfg_cin > MAX_CH) begin
                            fail_core(E_UNSUPPORTED);
                        end else if (cfg_k > MAX_K) begin
                            fail_core(E_UNSUPPORTED);
                        end else if ((cfg_h == 16'd0) || (cfg_w == 16'd0)) begin
                            fail_core(E_CFG);
                        end else if (cfg_h > MAX_H) begin
                            fail_core(E_UNSUPPORTED);
                        end else if (cfg_w > MAX_W) begin
                            fail_core(E_UNSUPPORTED);
                        end else if ((cfg_in_addr_byte[3:0] != 4'd0) ||
                                     (cfg_out_addr_byte[3:0] != 4'd0) ||
                                     (cfg_in_row_stride_bytes[3:0] != 4'd0) ||
                                     (cfg_out_row_stride_bytes[3:0] != 4'd0)) begin
                            fail_core(E_ADDR_ALIGN);
                        end else if ((cfg_in_addr_byte == 32'd0) ||
                                     (cfg_out_addr_byte == 32'd0)) begin
                            fail_core(E_ADDR_RANGE);
                        end else if ((cfg_in_row_stride_bytes == 32'd0) ||
                                     (cfg_out_row_stride_bytes == 32'd0)) begin
                            fail_core(E_STRIDE);
                        end else if ((cfg_in_row_stride_bytes > 32'h0000_FFFF) ||
                                     (cfg_out_row_stride_bytes > 32'h0000_FFFF)) begin
                            fail_core(E_STRIDE);
                        end else if ((cfg_in_row_stride_bytes != g2_stride) ||
                                     (cfg_out_row_stride_bytes != g2_stride)) begin
                            fail_core(E_STRIDE);
                        end else if ((cfg_in_bytes != g2_bytes) ||
                                     (cfg_out_bytes != g2_bytes)) begin
                            fail_core(E_STRIDE);
                        end else if ((in_end_addr > 33'h1_0000_0000) ||
                                     (out_end_addr > 33'h1_0000_0000)) begin
                            fail_core(E_ADDR_RANGE);
                        end else begin
                            h_q <= g2_h;
                            w_q <= g2_w;
                            k_q <= g2_k;
                            in_beats_q <= cfg_in_row_stride_bytes[15:0] >> 4;
                            out_beats_q <= cfg_out_row_stride_bytes[15:0] >> 4;
                            in_addr_q <= cfg_in_addr_byte;
                            out_addr_q <= cfg_out_addr_byte;
                            y_q <= 16'd0;
                            oy_q <= 16'd0;
                            y_bank <= 3'd0;
                            for (ti = 0; ti < RING_ROWS; ti = ti + 1) ring_tag[ti] <= 16'hFFFF;
                            rd_phase <= 1'b0;
                            rdi_q <= 16'd0;
                            rd_hi_q <= 64'd0;
                            rd_last_q <= 1'b0;
                            hc_q <= 16'd1;
                            hmx_v <= 1'b0;
                            v_lo <= 16'd0; v_hi <= 16'd0; v_ox <= 16'd0;
                            v_sv <= 1'b0; v_cv <= 1'b0; pack_lo <= 64'd0; ob_q <= 16'd0;
                            pdata_q <= {PX_W{1'b0}}; pcol_q <= 16'd0;
                            pb_q <= 16'd0; pbeats_q <= 1'b0;
                            core_busy <= 1'b1;
                            core_done <= 1'b0;
                            core_error <= 1'b0;
                            core_error_code <= E_NONE;
                            state <= ST_RD_CMD;
                        end
                    end

                    ST_RD_CMD: begin
                        rd_cmd_valid <= 1'b1;
                        rd_cmd_addr_byte <= in_addr_q;
                        rd_cmd_bytes <= cfg_in_row_stride_bytes[15:0];
                        rd_cmd_tag <= TAG_INPUT;
                        if (rd_cmd_valid && rd_cmd_ready) begin
                            rd_cmd_valid <= 1'b0;
                            in_addr_q <= in_addr_q + cfg_in_row_stride_bytes;
                            rdi_q <= 16'd0;
                            rd_phase <= 1'b0;
                            state <= ST_RD_DATA;
                        end
                    end

                    ST_RD_DATA: begin
                        if (k_q != 16'd1) begin
                            // even K>=2: K/2 consecutive 128-bit beats form one
                            // pixel word; one DDR beat per cycle, the line-buffer
                            // store fires on the last beat of the pixel.
                            if (rd_hs) begin
                                if (rd_data_error) begin
                                    fail_core((rd_data_error_code == 16'h0) ?
                                              E_CORE_INTERNAL : rd_data_error_code);
                                end else if (rd_data_tag != TAG_INPUT) begin
                                    fail_core(E_AXI_PROTOCOL);
                                end else if (rdi_q >= in_beats_q) begin
                                    fail_core(E_AXI_PROTOCOL);
                                end else if (rd_data_last != ((rdi_q + 16'd1) >= in_beats_q)) begin
                                    fail_core(E_AXI_PROTOCOL);
                                end else begin
                                    rdi_q <= rdi_q + 16'd1;
                                    if (rd_data_last) begin
                                        hc_q <= 16'd1;
                                        state <= ST_HMAX;
                                    end
                                end
                            end
                        end else if (!rd_phase) begin
                            // K=1: low pixel of the beat.
                            if (rd_hs) begin
                                if (rd_data_error) begin
                                    fail_core((rd_data_error_code == 16'h0) ?
                                              E_CORE_INTERNAL : rd_data_error_code);
                                end else if (rd_data_tag != TAG_INPUT) begin
                                    fail_core(E_AXI_PROTOCOL);
                                end else if (rdi_q >= in_beats_q) begin
                                    fail_core(E_AXI_PROTOCOL);
                                end else if (rd_data_last != ((rdi_q + 16'd1) >= in_beats_q)) begin
                                    fail_core(E_AXI_PROTOCOL);
                                end else begin
                                    rd_hi_q <= rd_data[127:64];
                                    rd_last_q <= rd_data_last;
                                    rd_phase <= 1'b1;
                                end
                            end
                        end else begin
                            // K=1: high pixel of the beat (row-tail padding when
                            // the pixel index reaches W is dropped by lb_we).
                            rd_phase <= 1'b0;
                            rdi_q <= rdi_q + 16'd1;
                            if (rd_last_q) begin
                                hc_q <= 16'd1;
                                state <= ST_HMAX;
                            end
                        end
                    end

                    ST_HMAX: begin
                        if (hc_q == (w_q + 16'd6)) begin
                            hc_q <= 16'd1;
                            if (y_q >= 16'd2) begin
                                // normal output row oy = y-2 (window fully read)
                                state <= ST_VROW_SET;
                            end else if ((y_q + 16'd1) < h_q) begin
                                y_q <= y_q + 16'd1;
                                y_bank <= (y_bank == 3'd4) ? 3'd0 : (y_bank + 3'd1);
                                state <= ST_RD_CMD;
                            end else if (oy_q < h_q) begin
                                // all input rows read: flush the remaining rows
                                state <= ST_VROW_SET;
                            end else begin
                                core_busy <= 1'b0;
                                core_done <= 1'b1;
                                state <= ST_IDLE;
                            end
                        end else begin
                            hc_q <= hc_q + 16'd1;
                        end
                    end

                    ST_VROW_SET: begin
                        v_lo <= (oy_q >= 16'd2) ? (oy_q - 16'd2) : 16'd0;
                        v_hi <= ((oy_q + 16'd2) < h_q) ? (oy_q + 16'd2) : (h_q - 16'd1);
                        v_ox <= 16'd0;
                        v_sv <= 1'b0; v_sx <= 16'd0;
                        v_cv <= 1'b0; v_cx <= 16'd0;
                        pack_lo <= 64'd0;
                        ob_q <= 16'd0;
                        state <= ST_WR_CMD;
                    end

                    ST_WR_CMD: begin
                        wr_cmd_valid <= 1'b1;
                        wr_cmd_addr_byte <= out_addr_q;
                        wr_cmd_bytes <= cfg_out_row_stride_bytes[15:0];
                        wr_cmd_tag <= TAG_OUTPUT;
                        if (wr_cmd_valid && wr_cmd_ready) begin
                            wr_cmd_valid <= 1'b0;
                            out_addr_q <= out_addr_q + cfg_out_row_stride_bytes;
                            state <= ST_VERT;
                        end
                    end

                    ST_VERT: begin
                        if ((MAX_K <= 2) || (k_q == 16'd1)) begin
                            // Legacy unified path (also the K=1 fallback in a
                            // wide elaboration): one 128-bit beat per cycle
                            // carrying one K=2 pixel or two K=1 pixels.
                            if (vert_adv) begin
                                if (v_ox < w_q) v_ox <= v_ox + 16'd1;
                                v_sv <= (v_ox < w_q);
                                v_sx <= v_ox;
                                v_cv <= v_sv;
                                v_cx <= v_sx;
                                v_cdata <= vres_masked;
                                if ((k_q == 16'd1) && v_cv && !v_cx[0])
                                    pack_lo <= v_cdata[63:0];
                                if (beat_valid && wr_data_ready) begin
                                    ob_q <= ob_q + 16'd1;
                                    if (beat_last) begin
                                        if ((ob_q + 16'd1) != out_beats_q) begin
                                            fail_core(E_CORE_INTERNAL);
                                        end else begin
                                            state <= ST_WR_RSP;
                                        end
                                    end
                                end
                            end
                        end else if (~pbeats_q) begin
                            // Wide capacity: advance the 2-stage vertical
                            // pipeline and latch the stage-C pixel word; the
                            // pipeline freezes while its K/2 beats drain.
                            if (v_ox < w_q) v_ox <= v_ox + 16'd1;
                            v_sv <= (v_ox < w_q);
                            v_sx <= v_ox;
                            v_cv <= v_sv;
                            v_cx <= v_sx;
                            v_cdata <= vres_masked;
                            if (v_cv) begin
                                pdata_q <= v_cdata;
                                pcol_q  <= v_cx;
                                pb_q    <= 16'd0;
                                pbeats_q <= 1'b1;
                            end
                        end else begin
                            // Wide capacity serializer: one 128-bit beat per
                            // accepted handshake; beat pb_q = channel groups
                            // 2*pb_q, 2*pb_q+1 of the latched pixel word.
                            if (wr_data_ready) begin
                                ob_q <= ob_q + 16'd1;
                                if (pb_q == ((k_q >> 1) - 16'd1)) begin
                                    pb_q <= 16'd0;
                                    if (pcol_q == (w_q - 16'd1)) begin
                                        if ((ob_q + 16'd1) != out_beats_q) begin
                                            fail_core(E_CORE_INTERNAL);
                                        end else begin
                                            pbeats_q <= 1'b0;
                                            ob_q <= 16'd0;
                                            state <= ST_WR_RSP;
                                        end
                                    end else begin
                                        pbeats_q <= 1'b0;
                                    end
                                end else begin
                                    pb_q <= pb_q + 16'd1;
                                end
                            end
                        end
                    end

                    ST_WR_RSP: begin
                        wr_rsp_ready <= 1'b1;
                        if (wr_rsp_valid && wr_rsp_ready) begin
                            wr_rsp_ready <= 1'b0;
                            if (wr_rsp_error || wr_rsp_aborted) begin
                                fail_core((wr_rsp_error_code == 16'h0) ?
                                          E_CORE_INTERNAL : wr_rsp_error_code);
                            end else if ((y_q + 16'd1) < h_q) begin
                                oy_q <= oy_q + 16'd1;
                                y_q  <= y_q + 16'd1;
                                y_bank <= (y_bank == 3'd4) ? 3'd0 : (y_bank + 3'd1);
                                state <= ST_RD_CMD;
                            end else if ((oy_q + 16'd1) < h_q) begin
                                oy_q <= oy_q + 16'd1;
                                state <= ST_VROW_SET;
                            end else begin
                                core_busy <= 1'b0;
                                core_done <= 1'b1;
                                state <= ST_IDLE;
                            end
                        end
                    end

                    ST_ERROR: begin
                        core_busy <= 1'b0;
                        core_done <= 1'b0;
                        if (core_start_pipe[2]) begin
                            core_error <= 1'b0;
                            core_error_code <= E_NONE;
                            state <= ST_CHECK;
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
    wire unused_cfg = |{cfg_weight_addr_byte, cfg_bias_addr_byte, cfg_quant_addr_byte,
                        cfg_x_zero_point, cfg_weight_bytes, cfg_bias_bytes,
                        cfg_quant_bytes, wr_rsp_tag, cfg_k};
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
