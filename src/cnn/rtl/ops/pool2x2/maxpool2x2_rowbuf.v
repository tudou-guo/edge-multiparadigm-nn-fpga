// -----------------------------------------------------------------------------
// maxpool2x2_rowbuf.v
//
// One activation row buffer for signed-INT8 maxpool2x2_s2.
//
// The logical NHWC8_A16 row is a sequence of 8-byte channel groups:
//   chunk_index = pixel * K + group,  K = Cin_alloc / 8.
// This module stores each group in its own one-read-port SRAM bank, so one
// read request returns all K groups of one pixel in parallel.  Writes are
// deliberately one 64-bit chunk per cycle; the core splits each 128-bit AXI
// beat into two writes.  That keeps the inferred memory single-write-port
// BRAM friendly and handles K=1 without a same-bank write conflict.
// -----------------------------------------------------------------------------

`timescale 1ns / 1ps

module maxpool2x2_rowbuf #(
    parameter integer MAX_CH = 64,
    parameter integer MAX_W  = 256,
    parameter integer MAX_K  = (MAX_CH + 7) / 8
) (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       wr_en,
    input  wire [15:0]                wr_pixel,
    input  wire [3:0]                 wr_group,
    input  wire [63:0]                wr_chunk,
    input  wire                       rd_en,
    input  wire [15:0]                rd_pixel,
    output wire [MAX_K*64-1:0]        rd_chunks
);

    localparam integer AW = (MAX_W <= 1) ? 1 : $clog2(MAX_W);

    wire [63:0] bank_rdata [0:MAX_K-1];
    wire [AW-1:0] rd_addr = rd_pixel[AW-1:0];

    genvar g;
    generate
        for (g = 0; g < MAX_K; g = g + 1) begin : GEN_GROUP_BANK
            wire we_g = wr_en && (wr_group == g[3:0]) && (wr_pixel < MAX_W);
            wire [AW-1:0] wr_addr = wr_pixel[AW-1:0];

            spram #(
                .DP   (MAX_W),
                .DW   (64),
                .PIPE (1)
            ) u_bank (
                .clk   (clk),
                .wdata (wr_chunk),
                .wen   (we_g),
                .waddr (wr_addr),
                .ren   (rd_en),
                .raddr (rd_addr),
                .rdata (bank_rdata[g])
            );

            assign rd_chunks[g*64 +: 64] = bank_rdata[g];
        end
    endgenerate

    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_rst = rst_n;
    /* verilator lint_on UNUSEDSIGNAL */

endmodule