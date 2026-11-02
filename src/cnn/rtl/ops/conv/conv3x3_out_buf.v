`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// conv3x3_out_buf.v (T3: BRAM ?)
//
// Two-page (ping/pang) row store. Each page is one tpram DP=1024 DW=128 PIPE=1
// using the reference template D:/yolov8/src/cnn/common/tpram.v verbatim.
// Full-word writes only, so synthesis infers Block RAM instead of registers.
//
// Write side: the ABI producer delivers 64-bit half beats (s_byte 0/8). A
// single hold slot (+bank/beat tag) assembles full 128-bit words:
//   mask 11 (s_zero_high low group) : full-word write through immediately AND
//       staged in hold, because a mask-10 high group to the same beat may
//       still follow (zero_high only means "no high group in this pixel group",
//       the next pixel can target the same beat's upper half).
//   mask 01 (low half)              : staged in hold, no RAM write yet.
//   mask 10 (high half)             : combined {high, hold_low} full write.
// Groups are sequential low-then-high per beat, so one hold slot suffices and
// the producer/consumer handshake timing is unchanged.
//
// Read side: 1-cycle latency. Both pages are read every cycle (en=1, like the
// reference); the page mux uses a 1-stage-delayed rd_bank.
// Reset: synchronous, control regs only. Memory is initial-zero, never cleared.
// -----------------------------------------------------------------------------
module conv3x3_out_buf #(
    parameter integer MAX_ROW_BEATS = 1024
) (
    input  wire         clk,
    input  wire         rst_n,

    input  wire         wr_bank,
    input  wire         wr_en,
    input  wire [15:0]  wr_beat,
    input  wire [1:0]   wr_mask,
    input  wire [127:0] wr_data,

    input  wire         rd_bank,
    input  wire [15:0]  rd_beat,
    output wire [127:0] rd_data
);

    localparam integer AW = (MAX_ROW_BEATS <= 2) ? 1 : $clog2(MAX_ROW_BEATS);

    wire [AW-1:0] wr_addr = wr_beat[AW-1:0];
    wire [AW-1:0] rd_addr = rd_beat[AW-1:0];

    // ---- 64-bit half-beat hold for full-word assembly ----
    reg        hold_valid;
    reg        hold_bank;
    reg [15:0] hold_beat;
    reg [63:0] hold_low;

    wire hold_match = hold_valid && (hold_bank == wr_bank) && (hold_beat == wr_beat);

    wire wr_full = wr_en && (wr_mask == 2'b11);
    wire wr_low  = wr_en && (wr_mask == 2'b01);
    wire wr_high = wr_en && (wr_mask == 2'b10);

    // Full-word write data: through path for mask 11, assembled word for the
    // high half (defensive zero low half on tag mismatch, unreachable by the
    // sequential low-then-high producer protocol).
    wire [127:0] wr_full_data = wr_full ? wr_data :
                                {wr_data[127:64], hold_match ? hold_low : 64'b0};
    wire wr_ping_wen = !wr_bank && (wr_full || wr_high);
    wire wr_pang_wen =  wr_bank && (wr_full || wr_high);

    // Every low write (mask 01, or mask 11 which is always the low half)
    // refreshes the hold; a matching high consumes it. The mask-11 through
    // write and the hold update are the same cycle, so a following high to
    // the same beat combines with fresh data.
    always @(posedge clk) begin
        if (!rst_n) begin
            hold_valid <= 1'b0;
            hold_bank  <= 1'b0;
            hold_beat  <= 16'b0;
            hold_low   <= 64'b0;
        end else if (wr_low || wr_full) begin
            hold_valid <= 1'b1;
            hold_bank  <= wr_bank;
            hold_beat  <= wr_beat;
            hold_low   <= wr_data[63:0];
        end else if (wr_high && hold_match) begin
            hold_valid <= 1'b0;
        end
    end

    // ---- ping/pang pages ----
    wire [127:0] ping_q;
    wire [127:0] pang_q;

    tpram #(
        .DP   (MAX_ROW_BEATS),
        .DW   (128),
        .PIPE (1)
    ) u_ping (
        .clk     (clk),
        .wdata_0 (wr_full_data),
        .wen_0   (wr_ping_wen),
        .addr_0  (wr_addr),
        .en_0    (1'b1),
        .rdata_0 (),
        .wdata_1 (128'b0),
        .wen_1   (1'b0),
        .addr_1  (rd_addr),
        .en_1    (1'b1),
        .rdata_1 (ping_q)
    );

    tpram #(
        .DP   (MAX_ROW_BEATS),
        .DW   (128),
        .PIPE (1)
    ) u_pang (
        .clk     (clk),
        .wdata_0 (wr_full_data),
        .wen_0   (wr_pang_wen),
        .addr_0  (wr_addr),
        .en_0    (1'b1),
        .rdata_0 (),
        .wdata_1 (128'b0),
        .wen_1   (1'b0),
        .addr_1  (rd_addr),
        .en_1    (1'b1),
        .rdata_1 (pang_q)
    );


`ifdef NPU_T3DBG
    // Optional bring-up probe; disabled by default for production simulation.
    always @(posedge clk) begin
        if (wr_en) $display("T3DBG t=%0t b=%b bt=%0d m=%b d=%h hm=%b hl=%h", $time, wr_bank, wr_beat, wr_mask, wr_data, hold_match, hold_low);
    end
`endif
    reg rd_bank_d;
    always @(posedge clk) begin
        if (!rst_n) begin
            rd_bank_d <= 1'b0;
        end else begin
            rd_bank_d <= rd_bank;
        end
    end

    assign rd_data = rd_bank_d ? pang_q : ping_q;

endmodule
