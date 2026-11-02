// -----------------------------------------------------------------------------
// axi4_4kb_splitter.v
//
// Pure combinational burst-sizing helper for the ABI 1.0 AXI4-Full transport.
//
// Given the current (16-byte aligned) address and the number of bytes still to
// move, it reports how many 128-bit beats may be issued in the next burst
// without crossing a 4 KB boundary and without exceeding the maximum burst
// length. Both axi_burst_reader and axi_burst_writer instantiate this module;
// the 4 KB rule therefore lives in exactly one place.
//
// Contract (docs/op_abi.md 5.2):
//   - `addr` MUST be 16-byte aligned. With an aligned address `bytes_to_4k` is
//     always a multiple of 16, so the division below is exact.
//   - `remaining_bytes` MUST be a multiple of 16 for ABI 1.0 transfers.
//   - burst_beats is 1..MAX_BURST_BEATS for any remaining_bytes > 0.
// -----------------------------------------------------------------------------

`timescale 1ns / 1ps

module axi4_4kb_splitter #(
    parameter integer AXI_ADDR_WIDTH  = 32,
    parameter integer BEAT_BYTES      = 16,
    parameter integer MAX_BURST_BEATS = 16
) (
    input  wire [AXI_ADDR_WIDTH-1:0] addr,
    input  wire [31:0]               remaining_bytes,

    output wire [31:0]               bytes_to_4k,
    output wire [31:0]               beats_to_4k,
    output wire [31:0]               burst_beats,
    output wire [31:0]               burst_bytes
);

    // Bytes from `addr` up to (not including) the next 4 KB boundary.
    wire [31:0] bytes_to_4k_w = 32'd4096 - {20'b0, addr[11:0]};

    // Beats needed to cover all remaining bytes (rounded up).
    wire [31:0] beats_needed_w = (remaining_bytes + BEAT_BYTES - 1) / BEAT_BYTES;

    // Whole beats that still fit before the 4 KB boundary.
    wire [31:0] beats_to_4k_w = bytes_to_4k_w / BEAT_BYTES;

    // Take the smaller of "what is left" and "what fits before 4 KB", then
    // clamp to the AXI maximum burst length.
    wire [31:0] beats_capped_w =
        (beats_needed_w > beats_to_4k_w) ? beats_to_4k_w : beats_needed_w;

    assign bytes_to_4k = bytes_to_4k_w;
    assign beats_to_4k = beats_to_4k_w;
    assign burst_beats = (beats_capped_w > MAX_BURST_BEATS) ? MAX_BURST_BEATS
                                                            : beats_capped_w;
    assign burst_bytes = burst_beats * BEAT_BYTES;

endmodule
