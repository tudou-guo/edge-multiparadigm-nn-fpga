// -----------------------------------------------------------------------------
// avgpool2x2_lane.v -- E1-P3 signed INT8 2x2 average lane
//
//   S = a + b + c + d            (signed, |S| <= 512, exact in 11 bits)
//   q = |S| / 4 with ties away from zero  (remainder >= 2 increments q)
//   y = sat8(sign(S) * q)
//
// This is exactly the P2 GlobalAveragePool rounding rule applied to the four
// valid pixels of one window: no bias, no zero point, no padding.  The lane
// never clamps to zero (an average of negative activations stays negative).
// -----------------------------------------------------------------------------

`timescale 1ns / 1ps

module avgpool2x2_lane (
    input  wire signed [7:0] a,
    input  wire signed [7:0] b,
    input  wire signed [7:0] c,
    input  wire signed [7:0] d,
    output reg  signed [7:0] y
);

    wire signed [10:0] sum = a + b + c + d;
    wire        negative = sum[10];
    wire [10:0] magnitude = negative ? (~sum + 11'd1) : sum;
    wire [9:0]  quotient = magnitude[10:2];
    wire        round_up = magnitude[1];
    wire [9:0]  rounded = quotient + (round_up ? 10'd1 : 10'd0);
    wire signed [10:0] signed_value = negative ? -$signed({1'b0, rounded})
                                               : $signed({1'b0, rounded});

    always @(*) begin
        if (signed_value > 11'sd127)
            y = 8'sd127;
        else if (signed_value < -11'sd128)
            y = -8'sd128;
        else
            y = signed_value[7:0];
    end

endmodule
