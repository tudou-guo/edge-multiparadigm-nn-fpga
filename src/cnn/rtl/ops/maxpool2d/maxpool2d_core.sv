`timescale 1ns / 1ps
// Generic signed-INT8 MaxPool2D. See README.md for the descriptor and service
// contract. No array is reset; only the row tags and control state are reset.
module maxpool2d_core #(
    parameter integer MAX_IN_ROW_BYTES = 1024,
    parameter integer MAX_OUT_ROW_BYTES = 1024,
    parameter integer MAX_EH = 25,
    parameter integer CACHE_ROWS = 32,
    parameter integer MAX_HW = 640,
    parameter integer MAX_CHANNELS = 512,
    parameter integer MAX_KERNEL = 7,
    parameter integer MAX_STRIDE = 8,
    parameter integer MAX_DILATION = 4,
    parameter integer MAX_PADDING = 24
) (
    input wire clk, input wire rst_n,
    input wire pool_cfg_we, input wire [3:0] pool_cfg_addr,
    input wire [31:0] pool_cfg_data,
    input wire core_start, output reg core_busy, output reg core_done,
    output reg core_error, output reg [15:0] core_error_code,
    input wire core_soft_reset, input wire core_abort,
    output wire rd_cmd_valid, input wire rd_cmd_ready,
    output wire [31:0] rd_cmd_addr_byte, output wire [15:0] rd_cmd_bytes,
    output wire [7:0] rd_cmd_tag,
    input wire rd_data_valid, output wire rd_data_ready,
    input wire [127:0] rd_data, input wire rd_data_last,
    input wire rd_data_error, input wire [15:0] rd_data_error_code,
    input wire [7:0] rd_data_tag,
    output wire wr_cmd_valid, input wire wr_cmd_ready,
    output wire [31:0] wr_cmd_addr_byte, output wire [15:0] wr_cmd_bytes,
    output wire [7:0] wr_cmd_tag,
    output wire wr_data_valid, input wire wr_data_ready,
    output wire [127:0] wr_data, output wire wr_data_last,
    output wire [7:0] wr_data_tag,
    input wire wr_rsp_valid, output wire wr_rsp_ready,
    input wire wr_rsp_error, input wire wr_rsp_aborted,
    input wire [15:0] wr_rsp_error_code, input wire [7:0] wr_rsp_tag
);
    localparam integer IN_WORDS = (MAX_IN_ROW_BYTES + 15) / 16;
    localparam integer OUT_WORDS = (MAX_OUT_ROW_BYTES + 15) / 16;
    localparam integer RING_WORDS = CACHE_ROWS * OUT_WORDS;
    localparam integer IN_AW = IN_WORDS <= 1 ? 1 : $clog2(IN_WORDS);
    localparam integer OUT_AW = OUT_WORDS <= 1 ? 1 : $clog2(OUT_WORDS);
    localparam integer RING_AW = RING_WORDS <= 1 ? 1 : $clog2(RING_WORDS);
    localparam [15:0] E_CFG = 16'h0001, E_UNSUPPORTED = 16'h0002,
        E_ADDR_ALIGN = 16'h0003, E_ADDR_RANGE = 16'h0004,
        E_STRIDE = 16'h0005, E_AXI_PROTOCOL = 16'h0009,
        E_CORE_INTERNAL = 16'h000a;
    localparam [63:0] MIN8 = 64'h8080808080808080;
    localparam [5:0] IDLE=0, CHECK=1, DIV_H=2, DIV_W=3, SCHEDULE=4,
        RD_CMD=5, RD_DATA=6, H_INIT=7, H_REQ=8, H_USE=9, H_PACK=10,
        V_INIT=11, V_REQ=12, V_USE=13, V_PACK=14, WR_CMD=15,
        WR_READ=16, WR_WAIT=17, WR_SEND=18, WR_RSP=19;

    reg [5:0] state;
    reg [31:0] shadow [0:15];
    reg [31:0] active [0:15];
    reg [15:0] shadow_valid;
    reg [CACHE_ROWS-1:0] tag_valid;
    reg [15:0] row_tag [0:CACHE_ROWS-1];
    wire [15:0] hin=active[3][15:0], win=active[3][31:16];
    wire [15:0] channels=active[4][15:0], c_alloc=active[4][31:16];
    wire [15:0] hout=active[5][15:0], wout=active[5][31:16];
    wire [15:0] kh=active[6][15:0], kw=active[6][31:16];
    wire [15:0] sh=active[7][15:0], sw=active[7][31:16];
    wire [15:0] dh=active[8][15:0], dw=active[8][31:16];
    wire [15:0] pt=active[9][15:0], pl=active[9][31:16];
    wire [15:0] pb=active[10][15:0], pr=active[10][31:16];
    wire [31:0] eh = dh * (kh - 1) + 1;
    wire [31:0] ew = dw * (kw - 1) + 1;
    wire [31:0] expected_c_alloc = (channels + 32'd7) & 32'hfffffff8;
    wire [31:0] expected_in_row = ((win * expected_c_alloc) + 15) & 32'hfffffff0;
    wire [31:0] expected_out_row = ((wout * expected_c_alloc) + 15) & 32'hfffffff0;
    wire [32:0] input_end = {1'b0,active[1]} + {1'b0,active[13]};
    wire [32:0] output_end = {1'b0,active[2]} + {1'b0,active[14]};
    wire [31:0] groups = c_alloc >> 3;
    wire [31:0] total_groups = wout * groups;
    reg [31:0] div_rem, div_quot;
    reg [31:0] next_row, oy, ox, group_num, linear_group, tap, beat;
    reg [63:0] accumulator, result_hold, low_half;
    reg sample_pad, sample_half;
    reg [127:0] write_hold;
    reg pending_fault, wr_inflight;
    reg [15:0] pending_code;
    integer i;

    // Signed coordinates are checked before they become RAM addresses.
    wire signed [31:0] x_coord = $signed(ox * sw) - $signed({16'b0,pl}) + $signed(tap * dw);
    wire signed [31:0] y_coord = $signed(oy * sh) - $signed({16'b0,pt}) + $signed(tap * dh);
    wire signed [31:0] target_row = $signed(oy * sh) - $signed({16'b0,pt}) + $signed(eh) - 1;
    wire x_in = x_coord >= 0 && x_coord < $signed({16'b0,win});
    wire y_in = y_coord >= 0 && y_coord < $signed({16'b0,hin});
    wire [31:0] input_group = x_coord * groups + group_num;
    wire [31:0] y_slot = y_coord & (CACHE_ROWS-1);
    wire [31:0] row_slot = next_row & (CACHE_ROWS-1);
    wire [31:0] ring_read_addr = y_slot * OUT_WORDS + (linear_group >> 1);
    wire [31:0] ring_write_addr = row_slot * OUT_WORDS + (linear_group >> 1);
    wire [31:0] input_word_addr = input_group >> 1;
    wire [31:0] output_word_addr = linear_group >> 1;
    wire last_group = linear_group + 1 == total_groups;
    wire pack_write = linear_group[0] || last_group;
    wire [127:0] input_read, ring_read, output_read;
    wire [63:0] sample_word = sample_pad ? MIN8 :
        (state == H_USE ? (sample_half ? input_read[127:64] : input_read[63:0]) :
                           (sample_half ? ring_read[127:64] : ring_read[63:0]));
    wire [63:0] max_word;
    reg [63:0] masked_result;
    always @* begin
        masked_result = result_hold;
        for (integer lane = 0; lane < 8; lane = lane + 1)
            if ((group_num * 8 + lane) >= channels)
                masked_result[8*lane +: 8] = 8'd0;
    end
    maxpool2d_reduce8 reduce8(.accum(accumulator), .sample(sample_word), .result(max_word));
    maxpool2d_row_store #(.DEPTH(IN_WORDS),.ADDR_WIDTH(IN_AW)) input_ram (
        .clk(clk), .write_enable(state == RD_DATA && rd_data_valid && rd_data_ready && beat < (active[11] >> 4)),
        .write_addr(beat[IN_AW-1:0]), .write_data(rd_data),
        .read_enable(state == H_REQ && x_in), .read_addr(input_word_addr[IN_AW-1:0]), .read_data(input_read));
    maxpool2d_row_store #(.DEPTH(RING_WORDS),.ADDR_WIDTH(RING_AW)) ring_ram (
        .clk(clk), .write_enable(state == H_PACK && pack_write), .write_addr(ring_write_addr[RING_AW-1:0]),
        .write_data(linear_group[0] ? {result_hold,low_half} : {64'd0,result_hold}),
        .read_enable(state == V_REQ && y_in && tag_valid[y_slot] && row_tag[y_slot] == y_coord),
        .read_addr(ring_read_addr[RING_AW-1:0]), .read_data(ring_read));
    maxpool2d_row_store #(.DEPTH(OUT_WORDS),.ADDR_WIDTH(OUT_AW)) output_ram (
        .clk(clk), .write_enable(state == V_PACK && pack_write), .write_addr(output_word_addr[OUT_AW-1:0]),
        .write_data(linear_group[0] ? {masked_result,low_half} : {64'd0,masked_result}),
        .read_enable(state == WR_READ), .read_addr(beat[OUT_AW-1:0]), .read_data(output_read));

    wire request_fault = core_abort || pool_cfg_we || core_start;
    // Once valid is presented it must remain asserted until handshake.  A
    // fault in either command state therefore drains that announced command.
    assign rd_cmd_valid = state == RD_CMD;
    assign rd_cmd_addr_byte = active[1] + next_row * active[11];
    assign rd_cmd_bytes = active[11][15:0];
    assign rd_cmd_tag = 8'd1;
    assign rd_data_ready = state == RD_DATA;
    assign wr_cmd_valid = state == WR_CMD;
    assign wr_cmd_addr_byte = active[2] + oy * active[12];
    assign wr_cmd_bytes = active[12][15:0];
    assign wr_cmd_tag = 8'd2;
    assign wr_data_valid = state == WR_SEND;
    assign wr_data = write_hold;
    assign wr_data_last = beat + 1 == (active[12] >> 4);
    assign wr_data_tag = 8'd2;
    assign wr_rsp_ready = state == WR_RSP;

    task automatic fail(input [15:0] code);
        begin
            core_busy <= 0; core_done <= 0; core_error <= 1;
            core_error_code <= code; state <= IDLE;
            tag_valid <= 0; pending_fault <= 0; wr_inflight <= 0;
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; core_busy <= 0; core_done <= 0; core_error <= 0;
            core_error_code <= 0; shadow_valid <= 0; tag_valid <= 0;
            pending_fault <= 0; pending_code <= 0; wr_inflight <= 0;
            div_rem <= 0; div_quot <= 0; next_row <= 0; oy <= 0; ox <= 0;
            group_num <= 0; linear_group <= 0; tap <= 0; beat <= 0;
            accumulator <= MIN8; result_hold <= 0; low_half <= 0;
            sample_pad <= 0; sample_half <= 0; write_hold <= 0;
        end else if (core_soft_reset) begin
            state <= IDLE; core_busy <= 0; core_done <= 0; core_error <= 0;
            core_error_code <= 0; shadow_valid <= 0; tag_valid <= 0;
            pending_fault <= 0; pending_code <= 0; wr_inflight <= 0;
        end else begin
            if (state == IDLE && pool_cfg_we) begin
                shadow[pool_cfg_addr] <= pool_cfg_data;
                shadow_valid[pool_cfg_addr] <= 1'b1;
            end
            if (state != IDLE && request_fault && !pending_fault) begin
                pending_fault <= 1;
                pending_code <= core_abort ? E_CORE_INTERNAL : E_CFG;
            end
            case (state)
                IDLE: if (core_start) begin
                    shadow_valid <= 0;
                    if (shadow_valid != 16'hffff) begin
                        core_done <= 0; core_error <= 1; core_error_code <= E_CFG;
                    end else begin
                        for (i = 0; i < 16; i = i + 1) active[i] <= shadow[i];
                        core_busy <= 1; core_done <= 0; core_error <= 0; core_error_code <= 0;
                        tag_valid <= 0; pending_fault <= 0; wr_inflight <= 0;
                        state <= CHECK;
                    end
                end
                CHECK: begin
                    if (active[0] != 1 || active[15] != 0)
                        fail(E_UNSUPPORTED);
                    else if (hin == 0 || win == 0 || hout == 0 || wout == 0 ||
                             hin > MAX_HW || win > MAX_HW || hout > MAX_HW || wout > MAX_HW ||
                             channels == 0 || channels > MAX_CHANNELS || c_alloc != expected_c_alloc ||
                             kh == 0 || kw == 0 || kh > MAX_KERNEL || kw > MAX_KERNEL ||
                             sh == 0 || sw == 0 || sh > MAX_STRIDE || sw > MAX_STRIDE ||
                             dh == 0 || dw == 0 || dh > MAX_DILATION || dw > MAX_DILATION ||
                             pt > MAX_PADDING || pl > MAX_PADDING || pb > MAX_PADDING || pr > MAX_PADDING ||
                             eh > MAX_EH || eh > CACHE_ROWS || CACHE_ROWS < 1 ||
                             (CACHE_ROWS & (CACHE_ROWS-1)) != 0)
                        fail(E_CFG);
                    else if (active[1][3:0] != 0 || active[2][3:0] != 0)
                        fail(E_ADDR_ALIGN);
                    else if (active[11] != expected_in_row || active[12] != expected_out_row ||
                             active[11] > 65520 || active[12] > 65520 ||
                             active[11] > MAX_IN_ROW_BYTES || active[12] > MAX_OUT_ROW_BYTES)
                        fail(E_STRIDE);
                    else if (active[13] != hin * active[11] || active[14] != hout * active[12] ||
                             input_end > 33'h100000000 || output_end > 33'h100000000 ||
                             ({1'b0,active[1]} < output_end && {1'b0,active[2]} < input_end))
                        fail(E_ADDR_RANGE);
                    else if (hin + pt + pb < eh || win + pl + pr < ew)
                        fail(E_CFG);
                    else begin
                        div_rem <= hin + pt + pb - eh; div_quot <= 0; state <= DIV_H;
                    end
                end
                DIV_H: if (div_rem >= sh) begin
                    div_rem <= div_rem - sh; div_quot <= div_quot + 1;
                end else if (div_quot + 1 != hout) fail(E_CFG);
                else begin div_rem <= win + pl + pr - ew; div_quot <= 0; state <= DIV_W; end
                DIV_W: if (div_rem >= sw) begin
                    div_rem <= div_rem - sw; div_quot <= div_quot + 1;
                end else if (div_quot + 1 != wout) fail(E_CFG);
                else begin next_row <= 0; oy <= 0; state <= SCHEDULE; end
                SCHEDULE: begin
                    if (target_row >= 0 && next_row < hin && $signed(next_row) <= target_row)
                        state <= RD_CMD;
                    else state <= V_INIT;
                end
                RD_CMD: if (rd_cmd_valid && rd_cmd_ready) begin beat <= 0; state <= RD_DATA; end
                RD_DATA: if (rd_data_valid && rd_data_ready) begin
                    if (rd_data_error || rd_data_tag != 8'd1 ||
                        rd_data_last != (beat + 1 == (active[11] >> 4))) begin
                        pending_fault <= 1;
                        pending_code <= (rd_data_error && rd_data_error_code != 0) ? rd_data_error_code : E_AXI_PROTOCOL;
                    end
                    beat <= beat + 1;
                    if (rd_data_last) begin
                        if (pending_fault || request_fault || rd_data_error || rd_data_tag != 8'd1 ||
                            beat + 1 != (active[11] >> 4))
                            fail(pending_fault ? pending_code :
                                 (rd_data_error && rd_data_error_code != 0) ? rd_data_error_code :
                                 request_fault ? (core_abort ? E_CORE_INTERNAL : E_CFG) : E_AXI_PROTOCOL);
                        else state <= H_INIT;
                    end
                end
                H_INIT: begin ox <= 0; group_num <= 0; linear_group <= 0; tap <= 0; accumulator <= MIN8; state <= H_REQ; end
                H_REQ: begin
                    sample_pad <= !x_in; sample_half <= input_group[0]; state <= H_USE;
                end
                H_USE: begin
                    accumulator <= max_word;
                    if (tap + 1 == kw) begin result_hold <= max_word; state <= H_PACK; end
                    else begin tap <= tap + 1; state <= H_REQ; end
                end
                H_PACK: begin
                    if (!linear_group[0]) low_half <= result_hold;
                    if (last_group) begin
                        tag_valid[row_slot] <= 1; row_tag[row_slot] <= next_row;
                        next_row <= next_row + 1; state <= SCHEDULE;
                    end else begin
                        linear_group <= linear_group + 1;
                        if (group_num + 1 == groups) begin group_num <= 0; ox <= ox + 1; end
                        else group_num <= group_num + 1;
                        tap <= 0; accumulator <= MIN8; state <= H_REQ;
                    end
                end
                V_INIT: begin ox <= 0; group_num <= 0; linear_group <= 0; tap <= 0; accumulator <= MIN8; state <= V_REQ; end
                V_REQ: begin
                    if (y_in && (!tag_valid[y_slot] || row_tag[y_slot] != y_coord)) fail(E_CORE_INTERNAL);
                    else begin sample_pad <= !y_in; sample_half <= linear_group[0]; state <= V_USE; end
                end
                V_USE: begin
                    accumulator <= max_word;
                    if (tap + 1 == kh) begin result_hold <= max_word; state <= V_PACK; end
                    else begin tap <= tap + 1; state <= V_REQ; end
                end
                V_PACK: begin
                    if (!linear_group[0]) low_half <= masked_result;
                    if (last_group) state <= WR_CMD;
                    else begin
                        linear_group <= linear_group + 1;
                        if (group_num + 1 == groups) begin group_num <= 0; ox <= ox + 1; end
                        else group_num <= group_num + 1;
                        tap <= 0; accumulator <= MIN8; state <= V_REQ;
                    end
                end
                WR_CMD: if (wr_cmd_valid && wr_cmd_ready) begin
                    wr_inflight <= 1; beat <= 0; state <= WR_READ;
                end
                WR_READ: state <= WR_WAIT;
                WR_WAIT: begin write_hold <= output_read; state <= WR_SEND; end
                WR_SEND: if (wr_data_valid && wr_data_ready) begin
                    if (wr_data_last) state <= WR_RSP;
                    else begin beat <= beat + 1; state <= WR_READ; end
                end
                WR_RSP: if (wr_rsp_valid && wr_rsp_ready) begin
                    wr_inflight <= 0;
                    if (pending_fault || request_fault)
                        fail(pending_fault ? pending_code : core_abort ? E_CORE_INTERNAL : E_CFG);
                    else if (wr_rsp_error || wr_rsp_aborted || wr_rsp_tag != 8'd2)
                        fail((wr_rsp_error_code != 0) ? wr_rsp_error_code : E_AXI_PROTOCOL);
                    else if (oy + 1 == hout) begin
                        core_busy <= 0; core_done <= 1; core_error <= 0; state <= IDLE;
                    end else begin oy <= oy + 1; state <= SCHEDULE; end
                end
                default: fail(E_CORE_INTERNAL);
            endcase
            // An accepted read or write is drained before reporting a fault.
            if (state != IDLE && state != RD_CMD && state != RD_DATA &&
                state != WR_CMD && !wr_inflight &&
                (pending_fault || request_fault))
                fail(pending_fault ? pending_code : core_abort ? E_CORE_INTERNAL : E_CFG);
        end
    end
endmodule
