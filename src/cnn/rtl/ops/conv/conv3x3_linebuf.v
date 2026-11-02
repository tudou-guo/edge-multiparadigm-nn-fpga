// ----------------------------------------------------------------------------
// conv3x3_linebuf.v (T-OP2e: copied from D:/yolov8/src/cnn reference style)
//
// Five row banks built from 30 single-port spram instances (5 banks x 6 slots).
// Each spram is the exact D:/yolov8/src/cnn/common/spram.v template: 1D array,
// single write port, single sync-read port, initial-zero, no reset on memory,
// no partial-word write, no 2D array. Block RAM inference friendly.
//
// Function and latency identical to previous version:
//   * write bank via sequential counter +1 mod5, single fallback mod
//   * only 3 distinct rows and 3 distinct cols by 3x3 construction
//   * 1-cycle read latency (spram PIPE=1), controls pipelined 1 stage
//
// M4 (2026-10-27): K1 eight-group reuse read mode, see
// m4/reports/K1_REUSE_DESIGN.md section 3.  k1_mode=0 keeps the 3x3 path
// byte-for-byte and cycle-for-cycle (same registers, same READ_CMD_PIPE
// stages, same spram array); k1_mode=1 reads four channel-group bases of ONE
// spatial pixel:
//   * read slot i (i=0..3) carries base b[i] = {rd_c, rd_c_b, rd_c_c, rd_c_d}
//     of pixel (rd_x[0], rd_y[0]) -> bank y0 % 5 and
//     beat = (x0 * cin_alloc + b[i]) >> 4 (the core drives all nine
//     rd_y/rd_x to the same centre pixel for a K1 token);
//   * physical tap g consumes input subgroup i = g/2, so taps 2i and 2i+1
//     return the same slot; tap 8 has no subgroup and returns the zero point
//     (the core masks its weight plane in K1 mode);
//   * K1_TN == 8 : bases are 8-byte aligned; the 128-bit beat half is chosen
//     by the full byte offset's bit3 (off[i][3] with off = x0*cin_alloc +
//     b[i], not b[i][3]) and rd_data_b is driven with zero (Tn=8
//     lane bank only consumes rd_data);
//   * K1_TN == 16: bases are 16-byte aligned; rd_data = beat[63:0] and
//     rd_data_b = beat[127:64];
//   * channel tail / out-of-range bytes are filled with x_zero_point exactly
//     like the 3x3 path (b[i] + byte >= cin); b[i] >= cin_alloc fills the
//     whole group;
//   * the mode is pipelined together with the command/data path, so the output
//     mux always sees the mode of the command that produced the data.
// ----------------------------------------------------------------------------

`timescale 1ns / 1ps

module conv3x3_linebuf #(
    parameter integer READ_CMD_PIPE = 0,
    parameter integer MAX_W         = 256,
    parameter integer MAX_CIN       = 64,
    parameter integer MAX_ROW_BYTES = ((((MAX_CIN + 7) / 8) * 8) * MAX_W + 15) / 16 * 16,
    parameter integer MAX_ROW_BEATS = MAX_ROW_BYTES / 16,
    // K1 reuse profile: Tn of the build (8 or 16).  Only 8/16 are legal for the
    // eight-group mode; other values never assert k1_mode and keep the 3x3 path.
    parameter integer K1_TN         = 8
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               wr_en,
    input  wire [15:0]        wr_row,
    input  wire [15:0]        wr_beat,
    input  wire [127:0]       wr_data,
    input  wire [15:0]        h,
    input  wire [15:0]        w,
    input  wire [15:0]        cin,
    input  wire [15:0]        cin_alloc,
    input  wire signed [7:0]  x_zero_point,
    input  wire signed [31:0] rd_y [0:8],
    input  wire signed [31:0] rd_x [0:8],
    input  wire [15:0]        rd_c,
    output reg  [63:0]        rd_data [0:8],
    input  wire [15:0]        rd_c_b,
    output reg  [63:0]        rd_data_b [0:8],
    // M4 K1 eight-group reuse (k1_mode=0: both ignored, 3x3 path unchanged).
    input  wire               k1_mode,
    input  wire [15:0]        rd_c_c,
    input  wire [15:0]        rd_c_d
);

    localparam integer AW = (MAX_ROW_BEATS <= 2) ? 1 : $clog2(MAX_ROW_BEATS);
    // 16-byte channel-group beat (K1_TN=16) versus 8-byte half-beat (K1_TN=8).
    localparam integer K1_WIDE = (K1_TN == 16) ? 1 : 0;

    reg [15:0] bank_row [0:4];
    reg        bank_valid [0:4];

    reg [2:0]  wr_bank_q;
    reg [15:0] wr_row_q;
    reg        wr_row_valid;
    wire [2:0] wr_bank_fallback = wr_row % 16'd5;
    reg  [2:0] wr_bank;
    integer bi;
    always @(*) begin
        if (wr_beat == 16'b0) begin
            if (wr_row == 16'b0)
                wr_bank = 3'd0;
            else if (wr_row_valid && (wr_row == wr_row_q + 16'b1))
                wr_bank = (wr_bank_q == 3'd4) ? 3'd0 : (wr_bank_q + 3'd1);
            else
                wr_bank = wr_bank_fallback;
        end else begin
            wr_bank = wr_bank_q;
        end
    end

    wire wr_beat_ok = (wr_beat < MAX_ROW_BEATS);
    wire [AW-1:0] wr_beat_aw = wr_beat[AW-1:0];

    always @(posedge clk) begin
        if (!rst_n) begin
            for (bi = 0; bi < 5; bi = bi + 1) begin
                bank_row[bi]   <= 16'b0;
                bank_valid[bi] <= 1'b0;
            end
            wr_bank_q    <= 3'b0;
            wr_row_q     <= 16'b0;
            wr_row_valid <= 1'b0;
        end else begin
            if (wr_en && wr_beat_ok && (wr_beat == 16'b0)) begin
                bank_row[wr_bank]   <= wr_row;
                bank_valid[wr_bank] <= 1'b1;
                wr_bank_q    <= wr_bank;
                wr_row_q     <= wr_row;
                wr_row_valid <= 1'b1;
            end
        end
    end

    wire signed [31:0] y0 = rd_y[0];
    wire signed [31:0] y1 = rd_y[3];
    wire signed [31:0] y2 = rd_y[6];
    wire signed [31:0] x0 = rd_x[0];
    wire signed [31:0] x1 = rd_x[1];
    wire signed [31:0] x2 = rd_x[2];

    wire y0_ok = (y0 >= 0) && (y0 < $signed({16'b0, h}));
    wire y1_ok = (y1 >= 0) && (y1 < $signed({16'b0, h}));
    wire y2_ok = (y2 >= 0) && (y2 < $signed({16'b0, h}));
    wire x0_ok = (x0 >= 0) && (x0 < $signed({16'b0, w}));
    wire x1_ok = (x1 >= 0) && (x1 < $signed({16'b0, w}));
    wire x2_ok = (x2 >= 0) && (x2 < $signed({16'b0, w}));

    wire [15:0] y0_u = y0[15:0];
    wire [15:0] y1_u = y1[15:0];
    wire [15:0] y2_u = y2[15:0];
    wire [2:0] bank0 = y0_ok ? (y0_u % 16'd5) : 3'd0;
    wire [2:0] bank1 = y1_ok ? (y1_u % 16'd5) : 3'd0;
    wire [2:0] bank2 = y2_ok ? (y2_u % 16'd5) : 3'd0;

    wire [15:0] cin_mul8 = cin_alloc >> 3;
    // 7-bit channel-block index: covers the affine C_alloc <= 512 envelope
    // (64 blocks; C_alloc=512 needs 7 bits).  The legacy 4-bit field truncated
    // at 16 blocks (C=128).
    wire [6:0]  cin_q = cin_mul8[6:0];
    wire [31:0] xb0 = ($signed(x0) * $signed({1'b0, cin_q})) << 3;
    wire [31:0] xb1 = ($signed(x1) * $signed({1'b0, cin_q})) << 3;
    wire [31:0] xb2 = ($signed(x2) * $signed({1'b0, cin_q})) << 3;

    wire [31:0] off0  = xb0 + {16'b0, rd_c};
    wire [31:0] off0b = xb0 + {16'b0, rd_c_b};
    wire [31:0] off1  = xb1 + {16'b0, rd_c};
    wire [31:0] off1b = xb1 + {16'b0, rd_c_b};
    wire [31:0] off2  = xb2 + {16'b0, rd_c};
    wire [31:0] off2b = xb2 + {16'b0, rd_c_b};

    wire [15:0] beat0  = off0  >> 4;
    wire [15:0] beat0b = off0b >> 4;
    wire [15:0] beat1  = off1  >> 4;
    wire [15:0] beat1b = off1b >> 4;
    wire [15:0] beat2  = off2  >> 4;
    wire [15:0] beat2b = off2b >> 4;

    wire [15:0] beats_safe0  = (beat0  < MAX_ROW_BEATS) ? beat0  : 16'b0;
    wire [15:0] beats_safe1  = (beat0b < MAX_ROW_BEATS) ? beat0b : 16'b0;
    wire [15:0] beats_safe2  = (beat1  < MAX_ROW_BEATS) ? beat1  : 16'b0;
    wire [15:0] beats_safe3  = (beat1b < MAX_ROW_BEATS) ? beat1b : 16'b0;
    wire [15:0] beats_safe4  = (beat2  < MAX_ROW_BEATS) ? beat2  : 16'b0;
    wire [15:0] beats_safe5  = (beat2b < MAX_ROW_BEATS) ? beat2b : 16'b0;

    wire [AW-1:0] raddr_aw [0:5];
    assign raddr_aw[0] = beats_safe0[AW-1:0];
    assign raddr_aw[1] = beats_safe1[AW-1:0];
    assign raddr_aw[2] = beats_safe2[AW-1:0];
    assign raddr_aw[3] = beats_safe3[AW-1:0];
    assign raddr_aw[4] = beats_safe4[AW-1:0];
    assign raddr_aw[5] = beats_safe5[AW-1:0];

    // ------------------------------------------------------------------
    // M4 K1: four channel-group bases of the same pixel.  The core drives all
    // nine rd_y/rd_x to the centre pixel for a K1 token, so tap 0's coordinate
    // is the token pixel; the bank is y0 % 5 for every tap and every base only
    // moves the channel window inside the row.
    // ------------------------------------------------------------------
    wire [31:0] k1_off0 = xb0 + {16'b0, rd_c};
    wire [31:0] k1_off1 = xb0 + {16'b0, rd_c_b};
    wire [31:0] k1_off2 = xb0 + {16'b0, rd_c_c};
    wire [31:0] k1_off3 = xb0 + {16'b0, rd_c_d};
    wire [15:0] k1_beat0 = k1_off0 >> 4;
    wire [15:0] k1_beat1 = k1_off1 >> 4;
    wire [15:0] k1_beat2 = k1_off2 >> 4;
    wire [15:0] k1_beat3 = k1_off3 >> 4;
    wire [15:0] k1_beat0_safe = (k1_beat0 < MAX_ROW_BEATS) ? k1_beat0 : 16'b0;
    wire [15:0] k1_beat1_safe = (k1_beat1 < MAX_ROW_BEATS) ? k1_beat1 : 16'b0;
    wire [15:0] k1_beat2_safe = (k1_beat2 < MAX_ROW_BEATS) ? k1_beat2 : 16'b0;
    wire [15:0] k1_beat3_safe = (k1_beat3 < MAX_ROW_BEATS) ? k1_beat3 : 16'b0;
    wire [AW-1:0] raddr_k1_0 = k1_beat0_safe[AW-1:0];
    wire [AW-1:0] raddr_k1_1 = k1_beat1_safe[AW-1:0];
    wire [AW-1:0] raddr_k1_2 = k1_beat2_safe[AW-1:0];
    wire [AW-1:0] raddr_k1_3 = k1_beat3_safe[AW-1:0];
    // Whole-group validity / half selection / byte-tail masks are sampled at
    // the same command boundary as the 3x3 controls (below).
    wire k1_c_ok0 = (rd_c   < cin_alloc);
    wire k1_c_ok1 = (rd_c_b < cin_alloc);
    wire k1_c_ok2 = (rd_c_c < cin_alloc);
    wire k1_c_ok3 = (rd_c_d < cin_alloc);
    wire k1_b_ok0 = (k1_beat0 < MAX_ROW_BEATS);
    wire k1_b_ok1 = (k1_beat1 < MAX_ROW_BEATS);
    wire k1_b_ok2 = (k1_beat2 < MAX_ROW_BEATS);
    wire k1_b_ok3 = (k1_beat3 < MAX_ROW_BEATS);
    wire k1_hi0 = k1_off0[3];
    wire k1_hi1 = k1_off1[3];
    wire k1_hi2 = k1_off2[3];
    wire k1_hi3 = k1_off3[3];

    // Read-address source: slots 0..3 follow k1_mode (the mode and the address
    // are sampled by the same register edge, so the registered address always
    // belongs to the command that selected it); slots 4/5 stay on the 3x3
    // addresses and are unused in K1 mode.
    wire [AW-1:0] raddr_in [0:5];
    assign raddr_in[0] = k1_mode ? raddr_k1_0 : raddr_aw[0];
    assign raddr_in[1] = k1_mode ? raddr_k1_1 : raddr_aw[1];
    assign raddr_in[2] = k1_mode ? raddr_k1_2 : raddr_aw[2];
    assign raddr_in[3] = k1_mode ? raddr_k1_3 : raddr_aw[3];
    assign raddr_in[4] = raddr_aw[4];
    assign raddr_in[5] = raddr_aw[5];

    reg [AW-1:0] read_address_q [0:5];
    wire [AW-1:0] ram_read_address [0:5];
    integer addr_i;
    always @(posedge clk)begin
      if(!rst_n)for(addr_i=0;addr_i<6;addr_i=addr_i+1)read_address_q[addr_i]<=0;
      else for(addr_i=0;addr_i<6;addr_i=addr_i+1)read_address_q[addr_i]<=raddr_in[addr_i];
    end
    genvar addr_g;
    generate for(addr_g=0;addr_g<6;addr_g=addr_g+1)begin : GEN_READ_ADDRESS_BOUNDARY
      assign ram_read_address[addr_g] = (READ_CMD_PIPE!=0) ? read_address_q[addr_g] : raddr_in[addr_g];
    end endgenerate
    wire [127:0] qb [0:4][0:5];
    genvar gb, gs;
    generate
        for (gb = 0; gb < 5; gb = gb + 1) begin : GEN_BANK
            for (gs = 0; gs < 6; gs = gs + 1) begin : GEN_SLOT
                wire wen_b;
                assign wen_b = wr_en && wr_beat_ok && (wr_bank == gb[2:0]);
                spram #(
                    .DP   (MAX_ROW_BEATS),
                    .DW   (128),
                    .PIPE (1)
                ) u_ram (
                    .clk   (clk),
                    .wdata (wr_data),
                    .wen   (wen_b),
                    .waddr (wr_beat_aw),
                    .ren   (1'b1),
                    .raddr (ram_read_address[gs]),
                    .rdata (qb[gb][gs])
                );
            end
        end
    endgenerate

    reg [2:0] bank0_d, bank1_d, bank2_d;
    reg y0_ok_d, y1_ok_d, y2_ok_d;
    reg x0_ok_d, x1_ok_d, x2_ok_d;
    reg c_ok_d, cb_ok_d;
    reg hi0_d, hi0b_d, hi1_d, hi1b_d, hi2_d, hi2b_d;
    reg b0_ok_d, b0b_ok_d, b1_ok_d, b1b_ok_d, b2_ok_d, b2b_ok_d;
    reg [15:0] y0_u_d, y1_u_d, y2_u_d;
    reg [15:0] rd_c_dly, rd_c_b_d, cin_d;
    reg signed [7:0] xzp_d;
    reg [2:0] cmd_bank0_d, cmd_bank1_d, cmd_bank2_d;
    reg cmd_y0_ok_d, cmd_y1_ok_d, cmd_y2_ok_d;
    reg cmd_x0_ok_d, cmd_x1_ok_d, cmd_x2_ok_d;
    reg cmd_c_ok_d, cmd_cb_ok_d;
    reg cmd_hi0_d, cmd_hi0b_d, cmd_hi1_d, cmd_hi1b_d, cmd_hi2_d, cmd_hi2b_d;
    reg cmd_b0_ok_d, cmd_b0b_ok_d, cmd_b1_ok_d, cmd_b1b_ok_d, cmd_b2_ok_d, cmd_b2b_ok_d;
    reg [15:0] cmd_y0_u_d, cmd_y1_u_d, cmd_y2_u_d;
    reg [15:0] cmd_rd_c_d, cmd_rd_c_b_d, cmd_cin_d;
    reg signed [7:0] cmd_xzp_d;
    always @(posedge clk) begin
        if (!rst_n) begin
            cmd_bank0_d <= 3'b0; cmd_bank1_d <= 3'b0; cmd_bank2_d <= 3'b0;
            cmd_y0_ok_d <= 1'b0; cmd_y1_ok_d <= 1'b0; cmd_y2_ok_d <= 1'b0;
            cmd_x0_ok_d <= 1'b0; cmd_x1_ok_d <= 1'b0; cmd_x2_ok_d <= 1'b0;
            cmd_c_ok_d <= 1'b0; cmd_cb_ok_d <= 1'b0;
            cmd_hi0_d <= 1'b0; cmd_hi0b_d <= 1'b0; cmd_hi1_d <= 1'b0;
            cmd_hi1b_d <= 1'b0; cmd_hi2_d <= 1'b0; cmd_hi2b_d <= 1'b0;
            cmd_b0_ok_d <= 1'b0; cmd_b0b_ok_d <= 1'b0; cmd_b1_ok_d <= 1'b0;
            cmd_b1b_ok_d <= 1'b0; cmd_b2_ok_d <= 1'b0; cmd_b2b_ok_d <= 1'b0;
            cmd_y0_u_d <= 16'b0; cmd_y1_u_d <= 16'b0; cmd_y2_u_d <= 16'b0;
            cmd_rd_c_d <= 16'b0; cmd_rd_c_b_d <= 16'b0; cmd_cin_d <= 16'b0;
            cmd_xzp_d <= 8'b0;
        end else begin
            cmd_bank0_d <= bank0; cmd_bank1_d <= bank1; cmd_bank2_d <= bank2;
            cmd_y0_ok_d <= y0_ok; cmd_y1_ok_d <= y1_ok; cmd_y2_ok_d <= y2_ok;
            cmd_x0_ok_d <= x0_ok; cmd_x1_ok_d <= x1_ok; cmd_x2_ok_d <= x2_ok;
            cmd_c_ok_d <= (rd_c < cin_alloc);
            cmd_cb_ok_d <= (rd_c_b < cin_alloc);
            cmd_hi0_d <= off0[3]; cmd_hi0b_d <= off0b[3]; cmd_hi1_d <= off1[3];
            cmd_hi1b_d <= off1b[3]; cmd_hi2_d <= off2[3]; cmd_hi2b_d <= off2b[3];
            cmd_b0_ok_d <= (beat0 < MAX_ROW_BEATS);
            cmd_b0b_ok_d <= (beat0b < MAX_ROW_BEATS);
            cmd_b1_ok_d <= (beat1 < MAX_ROW_BEATS);
            cmd_b1b_ok_d <= (beat1b < MAX_ROW_BEATS);
            cmd_b2_ok_d <= (beat2 < MAX_ROW_BEATS);
            cmd_b2b_ok_d <= (beat2b < MAX_ROW_BEATS);
            cmd_y0_u_d <= y0_u; cmd_y1_u_d <= y1_u; cmd_y2_u_d <= y2_u;
            cmd_rd_c_d <= rd_c; cmd_rd_c_b_d <= rd_c_b; cmd_cin_d <= cin;
            cmd_xzp_d <= x_zero_point;
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            bank0_d <= 3'b0; bank1_d <= 3'b0; bank2_d <= 3'b0;
            y0_ok_d <= 1'b0; y1_ok_d <= 1'b0; y2_ok_d <= 1'b0;
            x0_ok_d <= 1'b0; x1_ok_d <= 1'b0; x2_ok_d <= 1'b0;
            c_ok_d <= 1'b0; cb_ok_d <= 1'b0;
            hi0_d <= 1'b0; hi0b_d <= 1'b0; hi1_d <= 1'b0;
            hi1b_d <= 1'b0; hi2_d <= 1'b0; hi2b_d <= 1'b0;
            b0_ok_d <= 1'b0; b0b_ok_d <= 1'b0; b1_ok_d <= 1'b0;
            b1b_ok_d <= 1'b0; b2_ok_d <= 1'b0; b2b_ok_d <= 1'b0;
            y0_u_d <= 16'b0; y1_u_d <= 16'b0; y2_u_d <= 16'b0;
            rd_c_dly <= 16'b0; rd_c_b_d <= 16'b0; cin_d <= 16'b0;
            xzp_d <= 8'b0;
        end else begin
            bank0_d <= (READ_CMD_PIPE!=0) ? cmd_bank0_d : bank0; bank1_d <= (READ_CMD_PIPE!=0) ? cmd_bank1_d : bank1; bank2_d <= (READ_CMD_PIPE!=0) ? cmd_bank2_d : bank2;
            y0_ok_d <= (READ_CMD_PIPE!=0) ? cmd_y0_ok_d : y0_ok; y1_ok_d <= (READ_CMD_PIPE!=0) ? cmd_y1_ok_d : y1_ok; y2_ok_d <= (READ_CMD_PIPE!=0) ? cmd_y2_ok_d : y2_ok;
            x0_ok_d <= (READ_CMD_PIPE!=0) ? cmd_x0_ok_d : x0_ok; x1_ok_d <= (READ_CMD_PIPE!=0) ? cmd_x1_ok_d : x1_ok; x2_ok_d <= (READ_CMD_PIPE!=0) ? cmd_x2_ok_d : x2_ok;
            c_ok_d <= (READ_CMD_PIPE!=0) ? cmd_c_ok_d : (rd_c < cin_alloc);
            cb_ok_d <= (READ_CMD_PIPE!=0) ? cmd_cb_ok_d : (rd_c_b < cin_alloc);
            hi0_d <= (READ_CMD_PIPE!=0) ? cmd_hi0_d : off0[3]; hi0b_d <= (READ_CMD_PIPE!=0) ? cmd_hi0b_d : off0b[3]; hi1_d <= (READ_CMD_PIPE!=0) ? cmd_hi1_d : off1[3];
            hi1b_d <= (READ_CMD_PIPE!=0) ? cmd_hi1b_d : off1b[3]; hi2_d <= (READ_CMD_PIPE!=0) ? cmd_hi2_d : off2[3]; hi2b_d <= (READ_CMD_PIPE!=0) ? cmd_hi2b_d : off2b[3];
            b0_ok_d <= (READ_CMD_PIPE!=0) ? cmd_b0_ok_d : (beat0 < MAX_ROW_BEATS);
            b0b_ok_d <= (READ_CMD_PIPE!=0) ? cmd_b0b_ok_d : (beat0b < MAX_ROW_BEATS);
            b1_ok_d <= (READ_CMD_PIPE!=0) ? cmd_b1_ok_d : (beat1 < MAX_ROW_BEATS);
            b1b_ok_d <= (READ_CMD_PIPE!=0) ? cmd_b1b_ok_d : (beat1b < MAX_ROW_BEATS);
            b2_ok_d <= (READ_CMD_PIPE!=0) ? cmd_b2_ok_d : (beat2 < MAX_ROW_BEATS);
            b2b_ok_d <= (READ_CMD_PIPE!=0) ? cmd_b2b_ok_d : (beat2b < MAX_ROW_BEATS);
            y0_u_d <= (READ_CMD_PIPE!=0) ? cmd_y0_u_d : y0_u; y1_u_d <= (READ_CMD_PIPE!=0) ? cmd_y1_u_d : y1_u; y2_u_d <= (READ_CMD_PIPE!=0) ? cmd_y2_u_d : y2_u;
            rd_c_dly <= (READ_CMD_PIPE!=0) ? cmd_rd_c_d : rd_c; rd_c_b_d <= (READ_CMD_PIPE!=0) ? cmd_rd_c_b_d : rd_c_b; cin_d <= (READ_CMD_PIPE!=0) ? cmd_cin_d : cin;
            xzp_d <= (READ_CMD_PIPE!=0) ? cmd_xzp_d : x_zero_point;
        end
    end

    // ------------------------------------------------------------------
    // M4 K1 command pipeline: exactly the same two READ_CMD_PIPE stages as the
    // 3x3 controls above, so the K1 output mux is aligned with the spram data
    // and with the shared bank/y/x/cin/xzp registers.
    // ------------------------------------------------------------------
    reg        cmd_k1_mode_d;
    reg [3:0]  cmd_k1_c_ok_d;
    reg [3:0]  cmd_k1_b_ok_d;
    reg [3:0]  cmd_k1_hi_d;
    reg [15:0] cmd_k1_base0_d, cmd_k1_base1_d, cmd_k1_base2_d, cmd_k1_base3_d;
    always @(posedge clk) begin
        if (!rst_n) begin
            cmd_k1_mode_d <= 1'b0;
            cmd_k1_c_ok_d <= 4'b0;
            cmd_k1_b_ok_d <= 4'b0;
            cmd_k1_hi_d   <= 4'b0;
            cmd_k1_base0_d <= 16'b0; cmd_k1_base1_d <= 16'b0;
            cmd_k1_base2_d <= 16'b0; cmd_k1_base3_d <= 16'b0;
        end else begin
            cmd_k1_mode_d <= k1_mode;
            cmd_k1_c_ok_d <= {k1_c_ok3, k1_c_ok2, k1_c_ok1, k1_c_ok0};
            cmd_k1_b_ok_d <= {k1_b_ok3, k1_b_ok2, k1_b_ok1, k1_b_ok0};
            cmd_k1_hi_d   <= {k1_hi3, k1_hi2, k1_hi1, k1_hi0};
            cmd_k1_base0_d <= rd_c;   cmd_k1_base1_d <= rd_c_b;
            cmd_k1_base2_d <= rd_c_c; cmd_k1_base3_d <= rd_c_d;
        end
    end

    reg        k1_mode_d;
    reg [3:0]  k1_c_ok_d;
    reg [3:0]  k1_b_ok_d;
    reg [3:0]  k1_hi_d;
    reg [15:0] k1_base0_d, k1_base1_d, k1_base2_d, k1_base3_d;
    always @(posedge clk) begin
        if (!rst_n) begin
            k1_mode_d <= 1'b0;
            k1_c_ok_d <= 4'b0;
            k1_b_ok_d <= 4'b0;
            k1_hi_d   <= 4'b0;
            k1_base0_d <= 16'b0; k1_base1_d <= 16'b0;
            k1_base2_d <= 16'b0; k1_base3_d <= 16'b0;
        end else begin
            k1_mode_d <= (READ_CMD_PIPE!=0) ? cmd_k1_mode_d : k1_mode;
            k1_c_ok_d <= (READ_CMD_PIPE!=0) ? cmd_k1_c_ok_d :
                         {k1_c_ok3, k1_c_ok2, k1_c_ok1, k1_c_ok0};
            k1_b_ok_d <= (READ_CMD_PIPE!=0) ? cmd_k1_b_ok_d :
                         {k1_b_ok3, k1_b_ok2, k1_b_ok1, k1_b_ok0};
            k1_hi_d   <= (READ_CMD_PIPE!=0) ? cmd_k1_hi_d :
                         {k1_hi3, k1_hi2, k1_hi1, k1_hi0};
            k1_base0_d <= (READ_CMD_PIPE!=0) ? cmd_k1_base0_d : rd_c;
            k1_base1_d <= (READ_CMD_PIPE!=0) ? cmd_k1_base1_d : rd_c_b;
            k1_base2_d <= (READ_CMD_PIPE!=0) ? cmd_k1_base2_d : rd_c_c;
            k1_base3_d <= (READ_CMD_PIPE!=0) ? cmd_k1_base3_d : rd_c_d;
        end
    end

    wire bank_match0_d = bank_valid[bank0_d] && (bank_row[bank0_d] == y0_u_d);
    wire bank_match1_d = bank_valid[bank1_d] && (bank_row[bank1_d] == y1_u_d);
    wire bank_match2_d = bank_valid[bank2_d] && (bank_row[bank2_d] == y2_u_d);

    integer k, b, gy, gx, gi;
    reg [127:0] w128;
    reg [63:0]  w64;
    reg         wvalid;
    reg         whi;
    // K1 output-mux temporaries (same combinational block, no extra read port).
    reg [127:0] k1_w128;
    reg [15:0]  k1_base_sel;
    reg         k1_grp_ok;
    always @(*) begin
        if (k1_mode_d) begin
            // K1 eight-group mode: tap g <- read slot i = g/2 of the single
            // token pixel; taps 2i and 2i+1 share the slot.  Tap 8 has no
            // input subgroup (core masks its weight plane).
            for (k = 0; k < 9; k = k + 1) begin
                gi = k / 2;
                if (gi > 3) begin
                    rd_data[k]   = {8{xzp_d}};
                    rd_data_b[k] = (K1_WIDE != 0) ? {8{xzp_d}} : 64'b0;
                end else begin
                    case (gi)
                        0: k1_w128 = qb[bank0_d][0];
                        1: k1_w128 = qb[bank0_d][1];
                        2: k1_w128 = qb[bank0_d][2];
                        default: k1_w128 = qb[bank0_d][3];
                    endcase
                    case (gi)
                        0: k1_base_sel = k1_base0_d;
                        1: k1_base_sel = k1_base1_d;
                        2: k1_base_sel = k1_base2_d;
                        default: k1_base_sel = k1_base3_d;
                    endcase
                    // Same pixel for every tap: bank match and y/x validity of
                    // tap 0 (the core drives all nine coordinates identically).
                    k1_grp_ok = y0_ok_d && x0_ok_d && bank_match0_d &&
                                k1_c_ok_d[gi] && k1_b_ok_d[gi];
                    if (!k1_grp_ok) begin
                        rd_data[k]   = {8{xzp_d}};
                        rd_data_b[k] = (K1_WIDE != 0) ? {8{xzp_d}} : 64'b0;
                    end else begin
                        if (K1_WIDE != 0) begin
                            // 16-byte aligned group: both beat halves belong
                            // to the same 16-channel subgroup.
                            rd_data[k]   = k1_w128[63:0];
                            rd_data_b[k] = k1_w128[127:64];
                        end else begin
                            // 8-byte aligned group inside a 128-bit beat: pick
                            // the half selected by off[3]; rd_data_b stays 0.
                            rd_data[k]   = k1_hi_d[gi] ? k1_w128[127:64] :
                                                        k1_w128[63:0];
                            rd_data_b[k] = 64'b0;
                        end
                        for (b = 0; b < 8; b = b + 1) begin
                            if ((k1_base_sel + b) >= cin_d)
                                rd_data[k][b*8 +: 8] = xzp_d;
                        end
                        if (K1_WIDE != 0) begin
                            for (b = 0; b < 8; b = b + 1) begin
                                if ((k1_base_sel + 16'd8 + b) >= cin_d)
                                    rd_data_b[k][b*8 +: 8] = xzp_d;
                            end
                        end
                    end
                end
            end
        end else begin
            // 3x3 path: unchanged from the pre-M4 implementation.
            for (k = 0; k < 9; k = k + 1) begin
                gy = k / 3; gx = k % 3;
                case (gy)
                    0: w128 = qb[bank0_d][gx*2];
                    1: w128 = qb[bank1_d][gx*2];
                    default: w128 = qb[bank2_d][gx*2];
                endcase
                case (gx)
                    0: whi = hi0_d;
                    1: whi = hi1_d;
                    default: whi = hi2_d;
                endcase
                w64 = whi ? w128[127:64] : w128[63:0];
                case (gy)
                    0: wvalid = y0_ok_d && bank_match0_d;
                    1: wvalid = y1_ok_d && bank_match1_d;
                    default: wvalid = y2_ok_d && bank_match2_d;
                endcase
                case (gx)
                    0: wvalid = wvalid && x0_ok_d && b0_ok_d;
                    1: wvalid = wvalid && x1_ok_d && b1_ok_d;
                    default: wvalid = wvalid && x2_ok_d && b2_ok_d;
                endcase
                wvalid = wvalid && c_ok_d;
                if (!wvalid) begin
                    rd_data[k] = {8{xzp_d}};
                end else begin
                    rd_data[k] = w64;
                    for (b = 0; b < 8; b = b + 1) begin
                        if ((rd_c_dly + b) >= cin_d)
                            rd_data[k][b*8 +: 8] = xzp_d;
                    end
                end
                case (gy)
                    0: w128 = qb[bank0_d][gx*2+1];
                    1: w128 = qb[bank1_d][gx*2+1];
                    default: w128 = qb[bank2_d][gx*2+1];
                endcase
                case (gx)
                    0: whi = hi0b_d;
                    1: whi = hi1b_d;
                    default: whi = hi2b_d;
                endcase
                w64 = whi ? w128[127:64] : w128[63:0];
                case (gy)
                    0: wvalid = y0_ok_d && bank_match0_d;
                    1: wvalid = y1_ok_d && bank_match1_d;
                    default: wvalid = y2_ok_d && bank_match2_d;
                endcase
                case (gx)
                    0: wvalid = wvalid && x0_ok_d && b0b_ok_d;
                    1: wvalid = wvalid && x1_ok_d && b1b_ok_d;
                    default: wvalid = wvalid && x2_ok_d && b2b_ok_d;
                endcase
                wvalid = wvalid && cb_ok_d;
                if (!wvalid) begin
                    rd_data_b[k] = {8{xzp_d}};
                end else begin
                    rd_data_b[k] = w64;
                    for (b = 0; b < 8; b = b + 1) begin
                        if ((rd_c_b_d + b) >= cin_d)
                            rd_data_b[k][b*8 +: 8] = xzp_d;
                    end
                end
            end
        end
    end

endmodule
