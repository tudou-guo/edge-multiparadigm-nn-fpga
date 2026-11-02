// Reference-style synchronous feature bank (SpikesTmpRam equivalent).
// Single 128-bit wide bank, 8192 deep (128 KiB). Two instances form the
// dual-bank ping-pong in spiking_sps_ref_top. Sync read (1-cycle latency),
// write-first? NO - read-during-write returns OLD data (matching reference
// TmpRam NO_CHANGE mode); controller guarantees no same-cycle RW hazard by
// draining in-flight results before bank swap (see ref ctrl).
// Maps to BRAM36 with 72b x 512 organization per RAMB36 pair; 128b width
// uses two parallel BRAM columns. No async reset of contents.
module spiking_feature_bram #(
  parameter int WIDTH = 128,
  parameter int DEPTH = 8192,
  parameter int AW = 13
)(
  input  logic s_clk,
  input  logic s_rst,
  // write port (controller / encoder side)
  input  logic i_we,
  input  logic [AW-1:0] i_waddr,
  input  logic [WIDTH-1:0] i_wdata,
  // read port (array / pool / adapter side, 1-cycle latency)
  input  logic [AW-1:0] i_raddr,
  output logic [WIDTH-1:0] o_rdata,
  output logic o_rvalid
);
  (* ram_style = "block" *) logic [WIDTH-1:0] mem [DEPTH];
  logic [AW-1:0] raddr_q;
  // True single-cycle read latency (Lmem=1): address registered, data
  // combinational from the registered address. Controller issues address
  // in cycle N and samples o_rdata at edge N+2 (one full cycle stable).
  always_ff @(posedge s_clk) begin
    if (i_we) mem[i_waddr] <= i_wdata;
    raddr_q <= i_raddr;
  end
  assign o_rdata = mem[raddr_q];
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) o_rvalid <= 1'b0;
    else o_rvalid <= 1'b1;
  end
endmodule
