`timescale 1ns/1ps
// Monitor-only helper bound by hierarchical refs from tb_block_replay.
module tb_block_mon;
  integer fed;
  integer consumed;
  initial begin fed = 0; consumed = 0; end
endmodule
