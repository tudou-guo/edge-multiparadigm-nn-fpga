`timescale 1ns / 1ps
// Parameterized 3x3 INT8 PE.
// Tn/Tm are elaboration-time lane counts: this module only instantiates
// 9*Tn*Tm multiply-accumulate lanes. The core issues 16x16 ABI groups as
// ordered micro-tiles and accumulates their partial sums.
module conv3x3_calc #(
    parameter integer Tn = 16,
    parameter integer Tm = 16,
    parameter integer WEIGHT_MODE = 0
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
    input wire [511:0] weight [0:3][0:8],
    input wire signed [7:0] local_weight [0:8],
    output wire [511:0] sum_packed,
    output wire [255:0] wsum_packed,
    output wire out_valid
);
    localparam integer TN = (Tn < 1) ? 1 : ((Tn > 16) ? 16 : Tn);
    localparam integer TM = (Tm < 1) ? 1 : ((Tm > 16) ? 16 : Tm);

    // Register the complete micro-tile at the PE boundary.  Weight faces are
    // synchronous RAMs, so this explicitly preserves the timing contract of
    // the original DSP pipeline instead of sampling a changing read face in a
    // combinational reduction tree.
    reg [7:0] act_ff [0:8][0:TN-1];
    reg signed [7:0] weight_ff [0:8][0:TN-1][0:TM-1];
    reg input_valid_ff;

    reg signed [31:0] sum_next [0:15];
    reg signed [15:0] wsum_next [0:15];
    reg signed [31:0] sum_pipe [0:8][0:15];
    reg signed [15:0] wsum_pipe [0:8][0:15];
    reg [8:0] valid_pipe;

    integer tap_i, in_i, out_i, pipe_i;
    integer lane_i, lane_o;
    reg signed [31:0] prod_acc;
    reg signed [15:0] weight_acc;

    function [7:0] act_at;
        input integer tap;
        input integer lane;
        begin
            case (tap)
                0: act_at = (lane < 8) ? act_data_0[lane*8 +: 8] : act_data_b_0[(lane-8)*8 +: 8];
                1: act_at = (lane < 8) ? act_data_1[lane*8 +: 8] : act_data_b_1[(lane-8)*8 +: 8];
                2: act_at = (lane < 8) ? act_data_2[lane*8 +: 8] : act_data_b_2[(lane-8)*8 +: 8];
                3: act_at = (lane < 8) ? act_data_3[lane*8 +: 8] : act_data_b_3[(lane-8)*8 +: 8];
                4: act_at = (lane < 8) ? act_data_4[lane*8 +: 8] : act_data_b_4[(lane-8)*8 +: 8];
                5: act_at = (lane < 8) ? act_data_5[lane*8 +: 8] : act_data_b_5[(lane-8)*8 +: 8];
                6: act_at = (lane < 8) ? act_data_6[lane*8 +: 8] : act_data_b_6[(lane-8)*8 +: 8];
                7: act_at = (lane < 8) ? act_data_7[lane*8 +: 8] : act_data_b_7[(lane-8)*8 +: 8];
                default: act_at = (lane < 8) ? act_data_8[lane*8 +: 8] : act_data_b_8[(lane-8)*8 +: 8];
            endcase
        end
    endfunction

    function signed [7:0] ddr_weight_at;
        input integer tap;
        input integer in_lane;
        input integer out_lane;
        begin
            if ((in_lane < 8) && (out_lane < 8))
                ddr_weight_at = $signed(weight[0][tap][((in_lane*8)+out_lane)*8 +: 8]);
            else if ((in_lane >= 8) && (out_lane < 8))
                ddr_weight_at = $signed(weight[1][tap][(((in_lane-8)*8)+out_lane)*8 +: 8]);
            else if ((in_lane < 8) && (out_lane >= 8))
                ddr_weight_at = $signed(weight[2][tap][((in_lane*8)+(out_lane-8))*8 +: 8]);
            else
                ddr_weight_at = $signed(weight[3][tap][(((in_lane-8)*8)+(out_lane-8))*8 +: 8]);
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            input_valid_ff <= 1'b0;
            valid_pipe <= 9'b0;
            for (pipe_i = 0; pipe_i < 9; pipe_i = pipe_i + 1)
                for (out_i = 0; out_i < 16; out_i = out_i + 1) begin
                    sum_pipe[pipe_i][out_i] <= 32'sd0;
                    wsum_pipe[pipe_i][out_i] <= 16'sd0;
                end
        end else begin
            input_valid_ff <= in_valid;
            for (tap_i = 0; tap_i < 9; tap_i = tap_i + 1)
                for (in_i = 0; in_i < TN; in_i = in_i + 1) begin
                    lane_i = in_lane_base + in_i;
                    act_ff[tap_i][in_i] <= (lane_i < 16) ? act_at(tap_i, lane_i) : 8'd0;
                    for (out_i = 0; out_i < TM; out_i = out_i + 1) begin
                        lane_o = out_lane_base + out_i;
                        if ((lane_i >= 16) || (lane_o >= 16))
                            weight_ff[tap_i][in_i][out_i] <= 8'sd0;
                        else if (WEIGHT_MODE == 0)
                            weight_ff[tap_i][in_i][out_i] <= ddr_weight_at(tap_i, lane_i, lane_o);
                        else if ((lane_i == 0) && (lane_o == 0))
                            weight_ff[tap_i][in_i][out_i] <= local_weight[tap_i];
                        else
                            weight_ff[tap_i][in_i][out_i] <= 8'sd0;
                    end
                end

            valid_pipe <= {valid_pipe[7:0], input_valid_ff};
            for (out_i = 0; out_i < 16; out_i = out_i + 1) begin
                sum_pipe[0][out_i] <= sum_next[out_i];
                wsum_pipe[0][out_i] <= wsum_next[out_i];
            end
            for (pipe_i = 1; pipe_i < 9; pipe_i = pipe_i + 1)
                for (out_i = 0; out_i < 16; out_i = out_i + 1) begin
                    sum_pipe[pipe_i][out_i] <= sum_pipe[pipe_i-1][out_i];
                    wsum_pipe[pipe_i][out_i] <= wsum_pipe[pipe_i-1][out_i];
                end
        end
    end

    always @* begin
        for (out_i = 0; out_i < 16; out_i = out_i + 1) begin
            sum_next[out_i] = 32'sd0;
            wsum_next[out_i] = 16'sd0;
        end
        for (out_i = 0; out_i < TM; out_i = out_i + 1) begin
            prod_acc = 32'sd0;
            weight_acc = 16'sd0;
            for (tap_i = 0; tap_i < 9; tap_i = tap_i + 1)
                for (in_i = 0; in_i < TN; in_i = in_i + 1) begin
                    prod_acc = prod_acc + $signed(weight_ff[tap_i][in_i][out_i]) * $signed({1'b0, act_ff[tap_i][in_i]});
                    weight_acc = weight_acc + $signed(weight_ff[tap_i][in_i][out_i]);
                end
            sum_next[out_i] = prod_acc;
            wsum_next[out_i] = weight_acc;
        end
    end

    assign out_valid = valid_pipe[8];
    genvar g;
    generate
        for (g = 0; g < 16; g = g + 1) begin : GEN_OUTPUT
            assign sum_packed[g*32 +: 32] = sum_pipe[8][g];
            assign wsum_packed[g*16 +: 16] = wsum_pipe[8][g];
        end
    endgenerate
endmodule