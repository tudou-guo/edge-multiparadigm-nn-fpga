`timescale 1ns / 1ps
module conv_mul_dsp (
    input         clk,
    input  [7:0]  a  ,
    input  [7:0]  b  ,
    input  [7:0]  c  ,
    output [15:0] p_a,
    output [15:0] p_b    
);

wire [24 : 0] A;
wire [8  : 0] B;
wire [17 : 0] C;
wire [15 : 0] D;
wire [34 : 0] P;

reg [15:0] temp_a;
reg [15:0] temp_b;

assign A = $signed({a, 16'd0});
assign D = $signed(b);
assign B = $signed({1'b0, c});
assign C = {1'b0, b[7], 16'd0};

//对于高版本的vivado要使用dsp macro这个ip核，dsp48 macro这个IP核在高版本的vivado会被弃用
//(A+D) * B
dsp_mul_pipe4 mul (
  .CLK(clk),  // input wire CLK
  .A(A),      // input wire [24 : 0] A
  .B(B),      // input wire [8  : 0] B
  .C(C),      // input wire [17 : 0] C
  .D(D),      // input wire [15 : 0] D
  .P(P)      // output wire [34 : 0] P
);

always @(posedge clk ) begin
    temp_a <= P[31:16];
    temp_b <= P[15: 0];
end

assign p_a = temp_a;
assign p_b = temp_b;
    
endmodule
