// Pipelined adder tree: direct port of upstream add_tree.
// Mechanism kept: zero-pad input count to power of 2, STAGES=clog2 pipeline
// registers, growing width per stage. Diff: sync active-high reset,
// signed accumulation, params renamed (N/W). Used for row+channel reduce.
module spiking_adder_tree #(
  parameter int N = 27,
  parameter int W = 8,
  parameter int STAGES = 5,
  parameter int NPOW = 32,
  parameter int OW = W + STAGES
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic [N*W-1:0] idata,
  output logic [OW-1:0] data_out
);
  logic [OW-1:0] node [STAGES+1][NPOW];
  genvar s, a;
  generate
    for (s = 0; s <= STAGES; s = s + 1) begin : g_stage
      localparam int CNT = NPOW >> s;
      localparam int SWD = W + s;
      if (s == 0) begin : g_in
        for (a = 0; a < CNT; a = a + 1) begin : g_leaf
          always_comb begin
            node[s][a] = 0;
            if (a < N) node[s][a][SWD-1:0] = idata[(a+1)*W-1 -: W];
          end
        end
      end else begin : g_add
        for (a = 0; a < CNT; a = a + 1) begin : g_n
          always_ff @(posedge s_clk or posedge s_rst) begin
            if (s_rst) node[s][a] <= 0;
            else node[s][a][SWD-1:0] <= $signed(node[s-1][2*a][SWD-2:0]) + $signed(node[s-1][2*a+1][SWD-2:0]);
          end
        end
      end
    end
  endgenerate
  assign data_out = node[STAGES][0];
endmodule
