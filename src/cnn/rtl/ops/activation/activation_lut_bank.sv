// -----------------------------------------------------------------------------
// activation_lut_bank.sv -- one 256x8 synchronous single-port-write / 1-cycle-read
// table bank for the ActivationLUT core.
//
// The core replicates this bank 8 times so one 8-byte lookup group can read 8
// independent table entries in the same cycle.  The table load path broadcasts
// the same table byte to all 8 banks (one byte per cycle per bank), so no
// second copy of the numeric table function is instantiated: the 2 KiB logical
// table (8 x 256 x 8) is exactly the 8 replicas.
//
// Read data is registered: rdata is valid one clock after re/raddr are sampled.
// Bank writes and table reads never overlap in the LUT engine (load happens
// before compute), so the read-during-write behaviour is not observable.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module activation_lut_bank (
    input  wire       clk,
    input  wire       we,
    input  wire [7:0] waddr,
    input  wire [7:0] wdata,
    input  wire       re,
    input  wire [7:0] raddr,
    output reg  [7:0] rdata
);

    // Small 256x8 table: distributed LUTRAM is the intended mapping (the
    // 2 KiB total cannot pay 8 BRAM36 read ports).  Synthesis decides finally.
    (* ram_style = "distributed" *) reg [7:0] mem [0:255];

    always @(posedge clk) begin
        if (we) begin
            mem[waddr] <= wdata;
        end
        if (re) begin
            rdata <= mem[raddr];
        end
    end

endmodule
