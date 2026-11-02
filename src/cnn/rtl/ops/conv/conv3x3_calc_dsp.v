`timescale 1ns/1ps
// DSP48-backed 3x3 calculator with the same external contract as
// conv3x3_calc.  This is a drop-in arithmetic candidate; conv3x3_core keeps
// the legacy implementation until the standalone equivalence gate passes.
//
// The current P3 data path presents activation bytes after the frozen
// x_zero_point/XOR transform, so ACTIVATION_SIGNED=0 preserves the existing
// weight(signed) x activation(unsigned byte) semantics.  The DSP primitive
// still has a signed activation mode for independent signed-INT8 use.
module conv3x3_calc_dsp #(
    parameter integer Tn = 16,
    parameter integer Tm = 16,
    parameter integer WEIGHT_MODE = 0,
    parameter integer FIXED_LANES = 0,
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
    input wire [511:0] weight [0:3][0:8],
    input wire signed [7:0] local_weight [0:8],
    output wire [511:0] sum_packed,
    output wire [255:0] wsum_packed,
    output wire out_valid
);
    localparam integer TN = (Tn < 1) ? 1 : ((Tn > 16) ? 16 : Tn);
    localparam integer TM = (Tm < 1) ? 1 : ((Tm > 16) ? 16 : Tm);
    localparam integer PAIRS = (TM + 1) / 2;

    reg [7:0] act_ff [0:8][0:TN-1];
    reg signed [7:0] weight_ff [0:8][0:TN-1][0:TM-1];
    reg input_valid_ff;

    wire signed [15:0] product_a [0:8][0:TN-1][0:PAIRS-1];
    wire signed [15:0] product_b [0:8][0:TN-1][0:PAIRS-1];
    wire mac_valid [0:8][0:TN-1][0:PAIRS-1];
    wire calc_mac_valid;
    wire signed [31:0] product_sum_comb [0:15];
    wire signed [15:0] weight_sum_comb [0:15];
    reg signed [15:0] weight_sum_delay [0:4][0:15];
    reg signed [31:0] sum_pipe [0:3][0:15];
    reg signed [15:0] wsum_pipe [0:3][0:15];
    reg [3:0] valid_pipe;

    integer tap_i, in_i, out_i, pipe_i;
    integer lane_i, lane_o;

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

    genvar gt, gi, gp;
    generate
        for (gt = 0; gt < 9; gt = gt + 1) begin : GEN_MAC_TAP
            for (gi = 0; gi < TN; gi = gi + 1) begin : GEN_MAC_IN
                for (gp = 0; gp < PAIRS; gp = gp + 1) begin : GEN_MAC_PAIR
                    signed_int8_dsp2_mac #(
                        .DSP_LATENCY(4),
                        .ACTIVATION_SIGNED(0)
                    ) u_mac (
                        .clk        (clk),
                        .rst_n      (rst_n),
                        .in_valid   (input_valid_ff),
                        .weight_a   (weight_ff[gt][gi][gp*2]),
                        .weight_b   ((gp*2+1 < TM) ? weight_ff[gt][gi][gp*2+1] : 8'sd0),
                        .activation (act_ff[gt][gi]),
                        .out_valid  (mac_valid[gt][gi][gp]),
                        .product_a  (product_a[gt][gi][gp]),
                        .product_b  (product_b[gt][gi][gp])
                    );
                end
            end
        end
    endgenerate

    genvar go_sum;
    generate
        for (go_sum = 0; go_sum < 16; go_sum = go_sum + 1) begin : GEN_SUM_LANE
            if (go_sum < TM) begin : GEN_SUM_ACTIVE
                reg signed [31:0] product_acc;
                reg signed [15:0] weight_acc;
                integer sum_tap_i, sum_in_i;
                always @* begin
                    product_acc = 32'sd0;
                    weight_acc = 16'sd0;
                    for (sum_tap_i = 0; sum_tap_i < 9; sum_tap_i = sum_tap_i + 1)
                        for (sum_in_i = 0; sum_in_i < TN; sum_in_i = sum_in_i + 1) begin
                            if ((go_sum % 2) == 0)
                                product_acc = product_acc + $signed(product_a[sum_tap_i][sum_in_i][go_sum/2]);
                            else
                                product_acc = product_acc + $signed(product_b[sum_tap_i][sum_in_i][go_sum/2]);
                            weight_acc = weight_acc + $signed(weight_ff[sum_tap_i][sum_in_i][go_sum]);
                        end
                end
                assign product_sum_comb[go_sum] = product_acc;
                assign weight_sum_comb[go_sum] = weight_acc;
            end else begin : GEN_SUM_INACTIVE
                assign product_sum_comb[go_sum] = 32'sd0;
                assign weight_sum_comb[go_sum] = 16'sd0;
            end
        end
    endgenerate
    always @(posedge clk) begin
        if (!rst_n) begin
            input_valid_ff <= 1'b0;
            valid_pipe <= 4'b0;
            for (pipe_i = 0; pipe_i < 5; pipe_i = pipe_i + 1)
                for (out_i = 0; out_i < 16; out_i = out_i + 1)
                    weight_sum_delay[pipe_i][out_i] <= 16'sd0;
            for (pipe_i = 0; pipe_i < 4; pipe_i = pipe_i + 1)
                for (out_i = 0; out_i < 16; out_i = out_i + 1) begin
                    sum_pipe[pipe_i][out_i] <= 32'sd0;
                    wsum_pipe[pipe_i][out_i] <= 16'sd0;
                end
            for (tap_i = 0; tap_i < 9; tap_i = tap_i + 1)
                for (in_i = 0; in_i < TN; in_i = in_i + 1) begin
                    act_ff[tap_i][in_i] <= 8'd0;
                    for (out_i = 0; out_i < TM; out_i = out_i + 1)
                        weight_ff[tap_i][in_i][out_i] <= 8'sd0;
                end
        end else begin
            input_valid_ff <= in_valid;
            valid_pipe <= {valid_pipe[2:0], calc_mac_valid};

            for (tap_i = 0; tap_i < 9; tap_i = tap_i + 1)
                for (in_i = 0; in_i < TN; in_i = in_i + 1) begin
                    lane_i = (FIXED_LANES != 0) ? (IN_LANE_BASE + in_i) : (in_lane_base + in_i);
                    act_ff[tap_i][in_i] <= (lane_i < 16) ? act_at(tap_i, lane_i) : 8'd0;
                    for (out_i = 0; out_i < TM; out_i = out_i + 1) begin
                        lane_o = (FIXED_LANES != 0) ? (OUT_LANE_BASE + out_i) : (out_lane_base + out_i);
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

            for (out_i = 0; out_i < 16; out_i = out_i + 1) begin
                weight_sum_delay[0][out_i] <= weight_sum_comb[out_i];
                sum_pipe[0][out_i] <= product_sum_comb[out_i];
                wsum_pipe[0][out_i] <= weight_sum_delay[4][out_i];
            end
            for (pipe_i = 1; pipe_i < 5; pipe_i = pipe_i + 1)
                for (out_i = 0; out_i < 16; out_i = out_i + 1)
                    weight_sum_delay[pipe_i][out_i] <= weight_sum_delay[pipe_i-1][out_i];
            for (pipe_i = 1; pipe_i < 4; pipe_i = pipe_i + 1)
                for (out_i = 0; out_i < 16; out_i = out_i + 1) begin
                    sum_pipe[pipe_i][out_i] <= sum_pipe[pipe_i-1][out_i];
                    wsum_pipe[pipe_i][out_i] <= wsum_pipe[pipe_i-1][out_i];
                end
        end
    end

    assign calc_mac_valid = mac_valid[0][0][0];
    assign out_valid = valid_pipe[3];
    genvar go;
    generate
        for (go = 0; go < 16; go = go + 1) begin : GEN_OUTPUT
            assign sum_packed[go*32 +: 32] = sum_pipe[3][go];
            assign wsum_packed[go*16 +: 16] = wsum_pipe[3][go];
        end
    endgenerate
endmodule


















