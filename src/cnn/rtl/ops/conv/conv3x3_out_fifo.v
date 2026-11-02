`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// conv3x3_out_fifo.v (T3: BRAM 化)
//
// 1024-deep x 129-bit ({last, data}) FWFT FIFO. Storage is one inferred Block
// RAM via the reference spram template (D:/yolov8/src/cnn/common/spram.v,
// PIPE=1). A first-word fall-through output stage (fwft reg + 1-deep skid +
// 1 outstanding prefetch) keeps the original combinational rd_valid/rd_data
// interface, so the drain side in out_ctrl is untouched.
// Reference behaviour: sync_fifo_129_512 instance in conv_data_out_buf_ctrl.v
// (valid = !empty, rd_en consumes).
// Reset/flush: synchronous, pointers + flags only (memory initial-zero).
// -----------------------------------------------------------------------------
module conv3x3_out_fifo #(
    parameter integer DEPTH = 1024
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         flush,

    input  wire         wr_en,
    input  wire [127:0] wr_data,
    input  wire         wr_last,
    output wire         full,

    input  wire         rd_en,
    output wire [127:0] rd_data,
    output wire         rd_last,
    output wire         rd_valid
);

    localparam integer AW = $clog2(DEPTH);

    reg [AW-1:0] wptr;
    reg [AW-1:0] rptr;
    reg [AW:0]   count;

    reg [128:0] fwft_data;
    reg         fwft_valid;
    reg [128:0] skid_data;
    reg         skid_valid;
    reg         inflight;

    wire [128:0] din = {wr_last, wr_data};
    wire wr_accept = wr_en && !full;
    wire consume   = rd_en && fwft_valid;

    wire [128:0] bram_q;
    // Exact BRAM occupancy: writes minus issued prefetches, always in range
    // because the accepted-minus-consumed total is capped at DEPTH by full.
    wire bram_occ = (wptr != rptr);
    // Issue one prefetch while the arriving word has a guaranteed home
    // (fwft or skid free); consumption only needs fwft_valid, so no deadlock.
    wire issue = bram_occ && !inflight && !(fwft_valid && skid_valid);

    spram #(
        .DP   (DEPTH),
        .DW   (129),
        .PIPE (1)
    ) u_mem (
        .clk   (clk),
        .wdata (din),
        .wen   (wr_accept),
        .waddr (wptr),
        .ren   (issue),
        .raddr (rptr),
        .rdata (bram_q)
    );

    // DEPTH need not be a power of two (e.g. 28 beats for a 224-byte row).
    // Wrap explicitly at DEPTH-1 so BRAM addresses never enter padded space.
    wire [AW-1:0] wptr_next = (wptr == DEPTH-1) ? {AW{1'b0}} :
                                 (wptr + {{(AW-1){1'b0}}, 1'b1});
    wire [AW-1:0] rptr_next = (rptr == DEPTH-1) ? {AW{1'b0}} :
                                 (rptr + {{(AW-1){1'b0}}, 1'b1});

    always @(posedge clk) begin
        if (!rst_n || flush) begin
            wptr       <= {AW{1'b0}};
            rptr       <= {AW{1'b0}};
            count      <= {(AW+1){1'b0}};
            fwft_data  <= 129'b0;
            fwft_valid <= 1'b0;
            skid_data  <= 129'b0;
            skid_valid <= 1'b0;
            inflight   <= 1'b0;
        end else begin
            if (wr_accept && !consume) begin
                count <= count + {{AW{1'b0}}, 1'b1};
            end else if (!wr_accept && consume) begin
                count <= count - {{AW{1'b0}}, 1'b1};
            end

            if (wr_accept) begin
                wptr <= wptr_next;
            end

            if (issue) begin
                rptr     <= rptr_next;
                inflight <= 1'b1;
            end

            if (inflight) begin
                inflight <= 1'b0;
                if (!fwft_valid || consume) begin
                    fwft_data  <= bram_q;
                    fwft_valid <= 1'b1;
                end else begin
                    // fwft blocked and not consumed: skid is free by the
                    // issue gate above.
                    skid_data  <= bram_q;
                    skid_valid <= 1'b1;
                end
            end else if (consume) begin
                if (skid_valid) begin
                    fwft_data  <= skid_data;
                    skid_valid <= 1'b0;
                    fwft_valid <= 1'b1;
                end else begin
                    fwft_valid <= 1'b0;
                end
            end else if (skid_valid && !fwft_valid) begin
                fwft_data  <= skid_data;
                skid_valid <= 1'b0;
                fwft_valid <= 1'b1;
            end
        end
    end

    assign full     = (count == DEPTH);
    assign rd_valid = fwft_valid;
    assign rd_data  = fwft_data[127:0];
    assign rd_last  = fwft_data[128];

endmodule