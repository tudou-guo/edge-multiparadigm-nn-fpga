// Eight signed byte comparators, shared by the horizontal and vertical passes.
module maxpool2d_reduce8 (
    input wire [63:0] accum,
    input wire [63:0] sample,
    output wire [63:0] result
);
    genvar i;
    generate for (i = 0; i < 8; i = i + 1) begin: lane
        wire signed [7:0] a = accum[8*i +: 8];
        wire signed [7:0] b = sample[8*i +: 8];
        assign result[8*i +: 8] = (a > b) ? a : b;
    end endgenerate
endmodule
