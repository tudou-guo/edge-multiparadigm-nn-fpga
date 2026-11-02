`timescale 1ns/1ps
module conv3x3_calc_dsp_wrap #(
    parameter integer Tn = 8,
    parameter integer Tm = 8,
    parameter integer WEIGHT_MODE = 0,
    parameter integer FIXED_LANES = 1,
    parameter integer IN_LANE_BASE = 0,
    parameter integer OUT_LANE_BASE = 0
) (
    input wire clk,
    input wire rst_n,
    input wire in_valid,
    input wire [3:0] in_lane_base,
    input wire [3:0] out_lane_base,
    input wire [63:0] act_data_0, input wire [63:0] act_data_1,
    input wire [63:0] act_data_2, input wire [63:0] act_data_3,
    input wire [63:0] act_data_4, input wire [63:0] act_data_5,
    input wire [63:0] act_data_6, input wire [63:0] act_data_7,
    input wire [63:0] act_data_8,
    input wire [63:0] act_data_b_0, input wire [63:0] act_data_b_1,
    input wire [63:0] act_data_b_2, input wire [63:0] act_data_b_3,
    input wire [63:0] act_data_b_4, input wire [63:0] act_data_b_5,
    input wire [63:0] act_data_b_6, input wire [63:0] act_data_b_7,
    input wire [63:0] act_data_b_8,
    input wire [18431:0] weight_flat,
    input wire [71:0] local_weight_flat,
    output wire [511:0] sum_packed,
    output wire [255:0] wsum_packed,
    output wire out_valid
);
    wire [511:0] weight [0:3][0:8];
    wire signed [7:0] local_weight [0:8];
    genvar gw, gt, gl;
    generate
        for (gw = 0; gw < 4; gw = gw + 1)
            for (gt = 0; gt < 9; gt = gt + 1)
                assign weight[gw][gt] = weight_flat[(gw*9+gt)*512 +: 512];
        for (gl = 0; gl < 9; gl = gl + 1)
            assign local_weight[gl] = local_weight_flat[gl*8 +: 8];
    endgenerate

    conv3x3_calc_dsp #(.Tn(Tn), .Tm(Tm), .WEIGHT_MODE(WEIGHT_MODE), .FIXED_LANES(FIXED_LANES), .IN_LANE_BASE(IN_LANE_BASE), .OUT_LANE_BASE(OUT_LANE_BASE)) u_calc (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid),
        .in_lane_base(in_lane_base), .out_lane_base(out_lane_base),
        .act_data_0(act_data_0), .act_data_1(act_data_1), .act_data_2(act_data_2),
        .act_data_3(act_data_3), .act_data_4(act_data_4), .act_data_5(act_data_5),
        .act_data_6(act_data_6), .act_data_7(act_data_7), .act_data_8(act_data_8),
        .act_data_b_0(act_data_b_0), .act_data_b_1(act_data_b_1), .act_data_b_2(act_data_b_2),
        .act_data_b_3(act_data_b_3), .act_data_b_4(act_data_b_4), .act_data_b_5(act_data_b_5),
        .act_data_b_6(act_data_b_6), .act_data_b_7(act_data_b_7), .act_data_b_8(act_data_b_8),
        .weight(weight), .local_weight(local_weight),
        .sum_packed(sum_packed), .wsum_packed(wsum_packed), .out_valid(out_valid)
    );
endmodule


