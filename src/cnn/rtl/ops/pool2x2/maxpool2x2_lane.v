// -----------------------------------------------------------------------------
// maxpool2x2_lane.v
//
// One lane (one channel) of a signed INT8 2x2 / stride-2 max pool.
//
//   y = max_s8(a, b, c, d)
//   if (CLAMP_MIN_EN != 0)  y = (y < 0) ? 0 : y
//
// Every comparison is signed.  The reference YOLO max_pool_8ch.v compares
// [7:0] as unsigned, which is wrong for INT8 data, so it is deliberately NOT
// followed here.
//
// CLAMP_MIN_EN is an elaboration parameter (default 1): the frozen project_3
// chain folds ReLU into the pool as pool -> clamp(min = 0).
// -----------------------------------------------------------------------------

`timescale 1ns / 1ps

module maxpool2x2_lane #(
    parameter integer CLAMP_MIN_EN = 1
) (
    input  wire signed [7:0] a,
    input  wire signed [7:0] b,
    input  wire signed [7:0] c,
    input  wire signed [7:0] d,
    output reg  signed [7:0] y
);

    wire signed [7:0] m_ab = (a > b) ? a : b;
    wire signed [7:0] m_cd = (c > d) ? c : d;
    wire signed [7:0] m    = (m_ab > m_cd) ? m_ab : m_cd;

    always @(*) begin
        if (CLAMP_MIN_EN != 0)
            y = (m < 8'sd0) ? 8'sd0 : m;
        else
            y = m;
    end

endmodule