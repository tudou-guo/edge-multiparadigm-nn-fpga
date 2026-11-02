`timescale 1ns/1ps
// Reusable project_1-style dual-product DSP48 primitive.
//
// With ACTIVATION_SIGNED=1, activation is signed INT8: its magnitude is
// packed into the DSP48 B input and the common sign is restored after the
// DSP pipeline.  With ACTIVATION_SIGNED=0, activation is an unsigned byte;
// this is the mode used by the current project_3 x_zero_point/XOR path.
module signed_int8_dsp2_mac #(
    parameter integer DSP_LATENCY = 4,
    parameter integer ACTIVATION_SIGNED = 1
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    input  wire signed [7:0]    weight_a,
    input  wire signed [7:0]    weight_b,
    input  wire signed [7:0]    activation,
    output reg                  out_valid,
    output reg signed [15:0]    product_a,
    output reg signed [15:0]    product_b
);
    localparam integer L = (DSP_LATENCY < 1) ? 1 : DSP_LATENCY;

    wire [7:0] activation_mag = (ACTIVATION_SIGNED != 0) ?
                                  (activation[7] ? (~activation[7:0] + 8'd1) : activation[7:0]) :
                                  activation[7:0];
    wire [24:0] A = $signed({weight_a, 16'd0});
    wire [8:0]  B = {1'b0, activation_mag};
    wire [17:0] C = {1'b0, (weight_b[7] && (activation_mag != 8'd0)), 16'd0};
    wire [15:0] D = $signed(weight_b);
    wire [34:0] P;

    reg [L-1:0] valid_pipe;
    reg [L-1:0] neg_pipe;

    dsp_mul_pipe4 u_dsp (
        .CLK(clk), .A(A), .B(B), .C(C), .D(D), .P(P)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_pipe <= {L{1'b0}};
            neg_pipe   <= {L{1'b0}};
            out_valid  <= 1'b0;
            product_a  <= 16'sd0;
            product_b  <= 16'sd0;
        end else begin
            valid_pipe[0] <= in_valid;
            neg_pipe[0]   <= activation[7];
            if (L > 1) begin
                valid_pipe[L-1:1] <= valid_pipe[L-2:0];
                neg_pipe[L-1:1]   <= neg_pipe[L-2:0];
            end
            out_valid <= valid_pipe[L-1];
            if ((ACTIVATION_SIGNED != 0) && neg_pipe[L-1]) begin
                product_a <= -$signed(P[31:16]);
                product_b <= -$signed(P[15:0]);
            end else begin
                product_a <= $signed(P[31:16]);
                product_b <= $signed(P[15:0]);
            end
        end
    end
endmodule

