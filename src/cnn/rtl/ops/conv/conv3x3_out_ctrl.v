`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// conv3x3_out_ctrl.v
//
// Output-row handoff controller ported from the verified reference:
//   D:/yolov8/src/cnn/conv/conv_data_out_buf_top.v
//   D:/yolov8/src/cnn/conv/conv_data_out_buf_ctrl.v
//
// Reference behaviours retained here (names kept: w_ping / r_ping):
//   * independent w_ping / r_ping bank pointers -> row N+1 can be staged while
//     row N is being drained;
//   * a 129-bit beat FIFO carrying {last, data}
//     (reference: sync_fifo_129_512);
//   * a completed row advances w_ping immediately, without waiting for the AXI
//     write response;
//   * r_ping is released only once the whole row has crossed the FIFO boundary;
//   * exactly one AXI write command per output row.
//
// ABI-side producer framing: one 8-byte lane group per s_valid beat, selected by
// s_beat (16-byte beat index inside the row) and s_byte (0 or 8 byte offset
// inside that beat).  s_last_group marks the final group of the row.
// -----------------------------------------------------------------------------
module conv3x3_out_ctrl #(
    parameter integer MAX_ROW_BEATS = 1024
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         flush,

    input  wire [31:0]  cfg_out_addr_byte,
    input  wire [31:0]  cfg_out_row_stride_bytes,
    input  wire [15:0]  cfg_hout,

    input  wire         s_valid,
    output wire         s_ready,
    input  wire [15:0]  s_beat,
    input  wire [3:0]   s_byte,
    input  wire [63:0]  s_data,
    input  wire         s_last_group,
    input  wire         s_zero_high,

    output wire         wr_cmd_valid,
    input  wire         wr_cmd_ready,
    output wire [31:0]  wr_cmd_addr_byte,
    output wire [15:0]  wr_cmd_bytes,
    output wire [7:0]   wr_cmd_tag,
    output wire         wr_data_valid,
    input  wire         wr_data_ready,
    output wire [127:0] wr_data,
    output wire         wr_data_last,
    output wire [7:0]   wr_data_tag,
    input  wire         wr_rsp_valid,
    output wire         wr_rsp_ready,
    input  wire         wr_rsp_error,
    input  wire         wr_rsp_aborted,
    input  wire [15:0]  wr_rsp_error_code,
    input  wire [7:0]   wr_rsp_tag,

    output reg          out_done,
    output reg          out_error,
    output reg  [15:0]  out_error_code
);

    localparam [2:0] R_IDLE  = 3'd0;
    localparam [2:0] R_CMD   = 3'd1;
    localparam [2:0] R_FILL  = 3'd2;
    localparam [2:0] R_DATA  = 3'd3;
    localparam [2:0] R_RSP   = 3'd4;
    localparam [2:0] R_ERROR = 3'd5;
    localparam [2:0] R_DONE  = 3'd6;

    localparam [7:0]  TAG_OUTPUT     = 8'h05;
    localparam [15:0] E_AXI_RESP     = 16'h0008;
    localparam [15:0] E_AXI_PROTOCOL = 16'h0009;
    localparam [15:0] E_CORE_INTERNAL= 16'h000A;

    reg               w_ping;
    reg               r_ping;
    reg  [1:0]        row_present;
    reg  [2:0]        r_state;
    reg  [15:0]       tx_row;
    reg  [15:0]       fill_beat;
    reg  [31:0]       wr_addr_q;   // revision5 PF1: row write address registered in R_IDLE

    wire [15:0] row_beats = cfg_out_row_stride_bytes[15:0] >> 4;

    // Producer lane group -> 16-byte beat half.  s_zero_high additionally zeroes
    // the half of the beat that this row never writes, so the physical row
    // stride is fully defined without relying on stale bank contents.
    wire [1:0] wr_mask_w =
        s_zero_high ? 2'b11 : ((s_byte == 4'd8) ? 2'b10 : 2'b01);
    wire [127:0] wr_data_w =
        (s_byte == 4'd8) ? {s_data, 64'b0} : {64'b0, s_data};

    wire         fifo_full;
    wire [127:0] fifo_rd_word;
    wire         fifo_rd_last;
    wire         fifo_rd_valid;
    wire         fifo_wr_en;
    wire         fifo_rd_en;
    wire [127:0] bank_rd_word;

    // A group is accepted whenever its target page is free.  The page for the
    // next row (already toggled w_ping) stays busy until the reader has pulled
    // the previous row out of the bank, which is the reference back-pressure.
    assign s_ready = !row_present[w_ping];

    wire producer_fire      = s_valid && s_ready;
    wire producer_last_fire = producer_fire && s_last_group;

    // T3: bank read has 1-cycle latency (BRAM). Addresses issue a cycle
    // before their data is collected into the {last, data} FIFO: buf_issue
    // presents fill_beat to the bank, buf_valid_b marks the beat arriving
    // next cycle, buf_last_b tags its row-last flag.
    reg    buf_valid_b;
    reg    buf_last_b;
    wire   buf_pipe_free = !buf_valid_b || !fifo_full;
    wire   buf_issue     = (r_state == R_FILL) && (fill_beat < row_beats) && buf_pipe_free;
    wire   buf_collect   = buf_valid_b && !fifo_full;
    assign fifo_wr_en       = (r_state == R_FILL) && buf_collect;
    assign fifo_rd_en       = (r_state == R_DATA) && fifo_rd_valid && wr_data_ready;
    wire   reader_last_fire = fifo_wr_en && buf_last_b;

    assign wr_cmd_valid     = (r_state == R_CMD);
    // revision5 PF1: row address registered in R_IDLE (config is stable before
    // the first row arrives); the writer sees a register instead of the
    // cfg-add-multiply cone.  Same formula and same values, one cycle earlier.
    assign wr_cmd_addr_byte = wr_addr_q;
    assign wr_cmd_bytes     = cfg_out_row_stride_bytes[15:0];
    assign wr_cmd_tag       = TAG_OUTPUT;

    assign wr_data_valid = (r_state == R_DATA) && fifo_rd_valid;
    assign wr_data       = fifo_rd_word;
    assign wr_data_last  = fifo_rd_last;
    assign wr_data_tag   = TAG_OUTPUT;
    assign wr_rsp_ready  = (r_state == R_RSP);

    conv3x3_out_buf #(
        .MAX_ROW_BEATS (MAX_ROW_BEATS)
    ) u_out_buf (
        .clk        (clk),
        .rst_n      (rst_n),
        .wr_bank     (w_ping),
        .wr_en      (producer_fire),
        .wr_beat    (s_beat),
        .wr_mask    (wr_mask_w),
        .wr_data    (wr_data_w),
        .rd_bank    (r_ping),
        .rd_beat    (fill_beat),
        .rd_data    (bank_rd_word)
    );

    conv3x3_out_fifo #(
        .DEPTH (MAX_ROW_BEATS)
    ) u_out_fifo (
        .clk      (clk),
        .rst_n    (rst_n),
        .flush    (flush),
        .wr_en    (fifo_wr_en),
        .wr_data  (bank_rd_word),
        .wr_last  (reader_last_fire),
        .full     (fifo_full),
        .rd_en    (fifo_rd_en),
        .rd_data  (fifo_rd_word),
        .rd_last  (fifo_rd_last),
        .rd_valid (fifo_rd_valid)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_ping         <= 1'b0;
            r_ping         <= 1'b0;
            row_present    <= 2'b00;
            r_state        <= R_IDLE;
            tx_row         <= 16'b0;
            fill_beat      <= 16'b0;
            buf_valid_b    <= 1'b0;
            buf_last_b     <= 1'b0;
            wr_addr_q      <= 32'b0;
            out_done       <= 1'b0;
            out_error      <= 1'b0;
            out_error_code <= 16'b0;
        end else if (flush) begin
            w_ping         <= 1'b0;
            r_ping         <= 1'b0;
            row_present    <= 2'b00;
            r_state        <= R_IDLE;
            tx_row         <= 16'b0;
            fill_beat      <= 16'b0;
            buf_valid_b    <= 1'b0;
            buf_last_b     <= 1'b0;
            wr_addr_q      <= 32'b0;
            out_done       <= 1'b0;
            out_error      <= 1'b0;
            out_error_code <= 16'b0;
        end else begin
            // Write side page handoff: flips on a complete row, independent of
            // the AXI B response (reference w_ping toggle).
            if (producer_last_fire) begin
                w_ping <= ~w_ping;
            end

            // Read side page handoff: the page is released only after every
            // beat has crossed the 129-bit FIFO boundary.
            row_present[0] <= (row_present[0] &
                ~(reader_last_fire && !r_ping)) |
                (producer_last_fire && !w_ping);
            row_present[1] <= (row_present[1] &
                ~(reader_last_fire &&  r_ping)) |
                (producer_last_fire &&  w_ping);

            // T3: bank data arrives one cycle after its address issues.
            if (reader_last_fire) begin
                r_ping      <= ~r_ping;
                fill_beat   <= 16'b0;
                buf_valid_b <= 1'b0;
            end else begin
                if (buf_issue) begin
                    fill_beat   <= fill_beat + 16'd1;
                    buf_valid_b <= 1'b1;
                    buf_last_b  <= ((fill_beat + 16'd1) == row_beats);
                end else if (buf_valid_b && !fifo_full && (r_state == R_FILL)) begin
                    buf_valid_b <= 1'b0;
                end
            end

            case (r_state)
                R_IDLE: begin
                    if (row_present[r_ping]) begin
                        if ((row_beats == 16'b0) || (cfg_hout == 16'b0)) begin
                            out_error      <= 1'b1;
                            out_error_code <= E_CORE_INTERNAL;
                            r_state        <= R_ERROR;
                        end else begin
                            wr_addr_q <= cfg_out_addr_byte +
                                         ({16'b0, tx_row} *
                                          cfg_out_row_stride_bytes[15:0]);
                            r_state <= R_CMD;
                        end
                    end
                end

                R_CMD: begin
                    if (wr_cmd_valid && wr_cmd_ready) begin
                        fill_beat   <= 16'b0;
                        buf_valid_b <= 1'b0;
                        buf_last_b  <= 1'b0;
                        r_state     <= R_FILL;
                    end
                end

                R_FILL: begin
                    if (reader_last_fire) begin
                        r_state <= R_DATA;
                    end
                end

                R_DATA: begin
                    if (fifo_rd_en && fifo_rd_last) begin
                        r_state <= R_RSP;
                    end
                end

                R_RSP: begin
                    if (wr_rsp_valid && wr_rsp_ready) begin
                        if (wr_rsp_error || wr_rsp_aborted) begin
                            out_error      <= 1'b1;
                            out_error_code <= wr_rsp_error ?
                                ((wr_rsp_error_code == 16'b0) ?
                                 E_AXI_RESP : wr_rsp_error_code) :
                                E_CORE_INTERNAL;
                            r_state        <= R_ERROR;
                        end else if (wr_rsp_tag != TAG_OUTPUT) begin
                            out_error      <= 1'b1;
                            out_error_code <= E_AXI_PROTOCOL;
                            r_state        <= R_ERROR;
                        end else if ((tx_row + 16'd1) == cfg_hout) begin
                            out_done <= 1'b1;
                            r_state  <= R_DONE;
                        end else begin
                            tx_row  <= tx_row + 16'd1;
                            r_state <= R_IDLE;
                        end
                    end
                end

                R_DONE: begin
                    r_state <= R_DONE;
                end

                R_ERROR: begin
                    r_state <= R_ERROR;
                end

                default: r_state <= R_IDLE;
            endcase
        end
    end

endmodule
