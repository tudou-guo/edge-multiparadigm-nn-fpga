`timescale 1ns/1ps
// cfc_weight_rom.sv -- 84x16 weight ROM + 17x32 bias ROM, initialized from the
// generated include export/gen/cfc_weight_init.svh (contract section 1.4/2).
// One synchronous read port pair (1 cycle latency), readable as a distributed
// memory by Vivado; the initial block is the only initialization path
// (no $readmemh, contract section 3.1).
module cfc_weight_rom (
  input  logic               clk,
  input  logic [6:0]         w_addr,   // 0..83
  output logic signed [15:0] w_data,
  input  logic [4:0]         b_addr,   // 0..16
  output logic signed [31:0] b_data
);
  logic signed [15:0] w_rom [0:83];
  logic signed [31:0] b_rom [0:16];

  `include "cfc_weight_init.svh"

  always_ff @(posedge clk) begin
    w_data <= w_rom[w_addr];
    if (b_addr <= 5'd16)
      b_data <= b_rom[b_addr];
    else
      b_data <= 32'sd0;
  end
endmodule
