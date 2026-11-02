`timescale 1ns/1ps
// cfc_activation_rom.sv -- tanh and sigmoid LUTs, 1025 x int16 each.
// Initialized from export/gen/cfc_lut_init.svh (contract section 1.3).
// Two independent synchronous read ports (1 cycle latency).
module cfc_activation_rom (
  input  logic               clk,
  input  logic [10:0]        tanh_addr,
  output logic signed [15:0] tanh_data,
  input  logic [10:0]        sig_addr,
  output logic signed [15:0] sig_data
);
  logic signed [15:0] tanh_rom    [0:1024];
  logic signed [15:0] sigmoid_rom [0:1024];

  `include "cfc_lut_init.svh"

  always_ff @(posedge clk) begin
    tanh_data <= tanh_rom[tanh_addr];
    sig_data  <= sigmoid_rom[sig_addr];
  end
endmodule
