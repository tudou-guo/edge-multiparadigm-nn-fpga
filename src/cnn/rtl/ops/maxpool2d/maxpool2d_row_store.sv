// Synchronous single-write/single-read RAM. Contents are intentionally not reset.
module maxpool2d_row_store #(
    parameter integer DEPTH = 64,
    parameter integer ADDR_WIDTH = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input wire clk,
    input wire write_enable,
    input wire [ADDR_WIDTH-1:0] write_addr,
    input wire [127:0] write_data,
    input wire read_enable,
    input wire [ADDR_WIDTH-1:0] read_addr,
    output reg [127:0] read_data
);
    reg [127:0] mem [0:DEPTH-1];
    always @(posedge clk) begin
        if (write_enable) mem[write_addr] <= write_data;
        if (read_enable) read_data <= mem[read_addr];
    end
endmodule
