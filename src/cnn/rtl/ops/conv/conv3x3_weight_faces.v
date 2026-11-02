// ----------------------------------------------------------------------------
// conv3x3_weight_faces.v (T-OP2e: copied from D:/yolov8/src/cnn reference style)
//
// Nine weight faces built from 144 single-port spram instances
// (9 taps x 4 words x 4 read ports). Each spram is the exact
// D:/yolov8/src/cnn/common/spram.v template: 1D array of 128-bit entries,
// single write port, single sync-read port, initial-zero, no reset on memory,
// full-word write only (no read-modify-write partial-word update).
// The tool replicates Block RAM per read port instead of building registers.
//
// Function and latency identical to previous version:
//   * full 512-bit word assembled from 4 x 128-bit slices {w3,w2,w1,w0}
//   * 1-cycle read latency (spram PIPE=1), rd_zero pipelined 1 stage
// ----------------------------------------------------------------------------

`timescale 1ns / 1ps

module conv3x3_weight_faces #(
    parameter integer MAX_WTILES = 64, parameter integer READ_PORTS = 4
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         wr_en,
    input  wire [3:0]   wr_tap,
    input  wire [15:0]  wr_index,
    input  wire [1:0]   wr_word,
    input  wire [127:0] wr_data,
    input  wire [15:0]  rd_index [0:3],
    input  wire         rd_zero  [0:3],
    output wire [511:0] weight   [0:3][0:8]
);

    localparam integer AW = (MAX_WTILES <= 2) ? 1 : $clog2(MAX_WTILES);

    wire wr_idx_ok = (wr_index < MAX_WTILES);
    wire [AW-1:0] wr_idx_aw = wr_index[AW-1:0];

    wire [AW-1:0] rd_aw [0:3];
    assign rd_aw[0] = rd_zero[0] ? {AW{1'b0}} : rd_index[0][AW-1:0];
    assign rd_aw[1] = rd_zero[1] ? {AW{1'b0}} : rd_index[1][AW-1:0];
    assign rd_aw[2] = rd_zero[2] ? {AW{1'b0}} : rd_index[2][AW-1:0];
    assign rd_aw[3] = rd_zero[3] ? {AW{1'b0}} : rd_index[3][AW-1:0];

    wire [127:0] q [0:3][0:8][0:3];
    genvar gt, gw, gp;
    generate
        for (gt = 0; gt < 9; gt = gt + 1) begin : GEN_TAP
            for (gw = 0; gw < 4; gw = gw + 1) begin : GEN_WORD
                for (gp = 0; gp < READ_PORTS; gp = gp + 1) begin : GEN_PORT
                    wire wen_t;
                    assign wen_t = wr_en && wr_idx_ok && (wr_tap == gt[3:0]) && (wr_word == gw[1:0]);
                    spram #(
                        .DP   (MAX_WTILES),
                        .DW   (128),
                        .PIPE (1)
                    ) u_ram (
                        .clk   (clk),
                        .wdata (wr_data),
                        .wen   (wen_t),
                        .waddr (wr_idx_aw),
                        .ren   (1'b1),
                        .raddr (rd_aw[gp]),
                        .rdata (q[gp][gt][gw])
                    );
                end
            end
        end
    endgenerate

    reg rz0_d, rz1_d, rz2_d, rz3_d;
    always @(posedge clk) begin
        if (!rst_n) begin
            rz0_d <= 1'b0; rz1_d <= 1'b0; rz2_d <= 1'b0; rz3_d <= 1'b0;
        end else begin
            rz0_d <= rd_zero[0]; rz1_d <= rd_zero[1];
            rz2_d <= rd_zero[2]; rz3_d <= rd_zero[3];
        end
    end

    genvar op, ot;
    generate
        for (op = 0; op < 4; op = op + 1) begin : GEN_OP
            for (ot = 0; ot < 9; ot = ot + 1) begin : GEN_OT
                wire [511:0] w_full;
                assign w_full = {q[op][ot][3], q[op][ot][2], q[op][ot][1], q[op][ot][0]};
                if(op < READ_PORTS) begin
                assign weight[op][ot] = ((op == 0) ? rz0_d : (op == 1) ? rz1_d : (op == 2) ? rz2_d : rz3_d) ? 512'b0 : w_full;
                end else begin
                    assign weight[op][ot] = 512'b0;
                end
            end
        end
    endgenerate

endmodule
