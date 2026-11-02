// -----------------------------------------------------------------------------
// npu_instr_seq.v
//
// Project_3 / E1 mixed-engine instruction sequencer.
//
// Opcode 1/2 retain the frozen 17-word Conv/Pool payload.  E1 adds opcode
// 0x0010/0x0011 with a 7-word elementwise payload.  The direct-operator
// extension adds 0x0019 Resize (9-word payload) and 0x001A MaxPool 5x5
// (17-word legacy-pool-layout payload).  S3 adds 0x001B ActivationLUT and
// 0x001C DepthwiseConv3x3, both with the 17-word Conv payload layout; their
// reserved words are rejected in S_PAY (E_SEQ_PAYLOAD) before launch and the
// 0x001C conv_cfg word must match the frozen formula exactly.  The selected
// opcode is latched by npu_engine_router at launch.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module npu_instr_seq #(
    parameter integer ADDR_W = 8,
    parameter [31:0] ROM_BASE_ADDR = 32'h1000_0000
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                core_start,
    input  wire [31:0]         runtime_base_addr,
    output reg                 core_busy,
    output reg                 core_done,
    output reg                 seq_error,
    output reg  [15:0]         seq_error_code,
    input  wire                core_soft_reset,
    input  wire                core_abort,
    input  wire                active_core_busy,
    input  wire                active_core_done,
    input  wire                active_core_error,
    input  wire [15:0]         active_core_error_code,
    output reg  [15:0]         record_index,
    output reg                 record_done,
    output reg  [ADDR_W-1:0]   rom_addr,
    input  wire [31:0]         rom_data,
    output reg  [7:0]          op_sel,
    output reg                 launch,
    output reg  [31:0]         cfg_in_addr,
    output reg  [31:0]         cfg_in1_addr,
    output reg  [31:0]         cfg_weight_addr,
    output reg  [31:0]         cfg_bias_addr,
    output reg  [31:0]         cfg_quant_addr,
    output reg  [31:0]         cfg_out_addr,
    output reg  [15:0]         cfg_h,
    output reg  [15:0]         cfg_w,
    output reg  [15:0]         cfg_cin,
    output reg  [15:0]         cfg_cout,
    output reg  [31:0]         cfg_conv_cfg,
    output reg  [31:0]         cfg_quant_cfg,
    output reg  [31:0]         cfg_act_cfg,
    output reg  [31:0]         cfg_in_row_stride_bytes,
    output reg  [31:0]         cfg_out_row_stride_bytes,
    output reg  [31:0]         cfg_in_bytes,
    output reg  [31:0]         cfg_in1_bytes,
    output reg  [31:0]         cfg_weight_bytes,
    output reg  [31:0]         cfg_bias_bytes,
    output reg  [31:0]         cfg_quant_bytes,
    output reg  [31:0]         cfg_out_bytes,
    output reg  [31:0]         cfg_elem_cfg,
    // affine-int8-v1 Add rescale parameters (only present in the 10-word
    // elementwise payload variant).
    output reg  [31:0]         cfg_rescale_a,
    output reg  [31:0]         cfg_rescale_b,
    output reg  [31:0]         cfg_rescale_cfg,
    // E1-P1 copy/transform configuration write port.  Address fields in
    // payload words 0..4 are relocated before they reach the copy core.
    output reg                 copy_cfg_we,
    output reg  [5:0]          copy_cfg_addr,
    output reg  [31:0]         copy_cfg_data,
    // Dedicated generic MaxPool descriptor bank; never aliases Copy/Conv.
    output reg                 pool_cfg_we,
    output reg  [3:0]          pool_cfg_addr,
    output reg  [31:0]         pool_cfg_data
);

    localparam [15:0] OP_CONV = 16'h0001;
    localparam [15:0] OP_POOL = 16'h0002;
    localparam [15:0] OP_RELU = 16'h0010;
    localparam [15:0] OP_ADD  = 16'h0011;
    localparam [15:0] OP_PAD  = 16'h0012;
    localparam [15:0] OP_SLICE = 16'h0013;
    localparam [15:0] OP_CONCAT = 16'h0014;
    localparam [15:0] OP_CLIP = 16'h0015;
    localparam [15:0] OP_GAP  = 16'h0016;
    localparam [15:0] OP_AVGPOOL = 16'h0017;
    localparam [15:0] OP_MUL     = 16'h0018;
    localparam [15:0] OP_RESIZE  = 16'h0019;
    localparam [15:0] OP_POOL5   = 16'h001A;
    localparam [15:0] OP_ACTLUT  = 16'h001B;
    localparam [15:0] OP_DEPTHWISE = 16'h001C;
    localparam [15:0] OP_POOL2D  = 16'h001D;
    localparam [15:0] PAY_POOL2D = 16'd16;
    localparam [15:0] PAY_CONV = 16'd17;
    localparam [15:0] PAY_RESIZE = 16'd9;
    localparam [15:0] PAY_EW   = 16'd7;
    // affine-int8-v1 Add: 7 legacy words + m0_a, m0_b, (shift,zp_a,zp_b,zp_out)
    localparam [15:0] PAY_EW_AFFINE = 16'd10;
    localparam [15:0] PAY_COPY = 16'd27;
    // affine-int8-v1 copy/transform v2: versioned 40-word record (word 23
    // bit16 marks the format).  Both lengths stay legal for 0x0012/13/14 so
    // the frozen 27-word payload continues to decode byte-identically.
    localparam [15:0] PAY_COPY_V2 = 16'd40;
    localparam [15:0] PAY_RED  = 16'd9;
    localparam [15:0] E_SEQ_STREAM  = 16'h00A1;
    localparam [15:0] E_SEQ_OPCODE  = 16'h00A2;
    localparam [15:0] E_SEQ_PAYLOAD = 16'h00A3;

    localparam [3:0] S_IDLE   = 4'd0;
    localparam [3:0] S_START  = 4'd1;
    localparam [3:0] S_LEN    = 4'd2;
    localparam [3:0] S_HDR    = 4'd3;
    localparam [3:0] S_PAY    = 4'd4;
    localparam [3:0] S_LAUNCH = 4'd5;
    localparam [3:0] S_WBUSY  = 4'd6;
    localparam [3:0] S_WDONE  = 4'd7;
    localparam [3:0] S_DONE   = 4'd8;
    localparam [3:0] S_ERROR  = 4'd9;

    reg [3:0]  state;
    reg        pay_bad;   // S_PAY reserved-word/config violation this beat
    reg [15:0] pay_left;
    reg [5:0]  pay_idx;
    reg [15:0] word_count;
    reg [15:0] stream_words_seen;
    reg [15:0] rec_index;
    wire [ADDR_W-1:0] rom_inc = rom_addr + {{(ADDR_W-1){1'b0}}, 1'b1};

    function [31:0] reloc_addr;
        input [31:0] rom_addr_value;
        reg [31:0] delta;
        begin
            if (rom_addr_value == 32'b0) begin
                reloc_addr = 32'b0;
            end else if (runtime_base_addr == 32'b0) begin
                reloc_addr = rom_addr_value;
            end else begin
                delta = rom_addr_value - ROM_BASE_ADDR;
                reloc_addr = runtime_base_addr + delta;
            end
        end
    endfunction

    function reloc_invalid;
        input [31:0] value;
        reg [32:0] sum;
        begin
            sum = {1'b0,runtime_base_addr} + {1'b0,(value - ROM_BASE_ADDR)};
            reloc_invalid = value != 0 && runtime_base_addr != 0 &&
                            (value < ROM_BASE_ADDR || sum[32]);
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; rom_addr <= {ADDR_W{1'b0}}; op_sel <= 8'd0;
            launch <= 1'b0; core_busy <= 1'b0; core_done <= 1'b0;
            seq_error <= 1'b0; seq_error_code <= 16'h0;
            pay_left <= 16'h0; pay_bad <= 1'b0; pay_idx <= 5'd0; word_count <= 16'h0;
            stream_words_seen <= 0;
            record_index <= 16'h0; record_done <= 1'b0; rec_index <= 16'h0;
            cfg_in_addr <= 0; cfg_in1_addr <= 0; cfg_weight_addr <= 0;
            cfg_bias_addr <= 0; cfg_quant_addr <= 0; cfg_out_addr <= 0;
            cfg_h <= 0; cfg_w <= 0; cfg_cin <= 0; cfg_cout <= 0;
            cfg_conv_cfg <= 0; cfg_quant_cfg <= 0; cfg_act_cfg <= 0;
            cfg_in_row_stride_bytes <= 0; cfg_out_row_stride_bytes <= 0;
            cfg_in_bytes <= 0; cfg_in1_bytes <= 0; cfg_weight_bytes <= 0;
            cfg_bias_bytes <= 0; cfg_quant_bytes <= 0; cfg_out_bytes <= 0;
            cfg_elem_cfg <= 0;
            cfg_rescale_a <= 0; cfg_rescale_b <= 0; cfg_rescale_cfg <= 0;
            copy_cfg_we <= 1'b0; copy_cfg_addr <= 6'd0; copy_cfg_data <= 32'd0;
            pool_cfg_we <= 1'b0; pool_cfg_addr <= 4'd0; pool_cfg_data <= 32'd0;
        end else if (core_soft_reset) begin
            state <= S_IDLE; core_busy <= 1'b0; seq_error <= 1'b0;
            seq_error_code <= 16'h0; rom_addr <= {ADDR_W{1'b0}}; op_sel <= 8'd0;
            record_done <= 1'b0; record_index <= 16'h0; rec_index <= 16'h0;
            copy_cfg_we <= 1'b0;
            pool_cfg_we <= 1'b0;
        end else if (core_abort) begin
            state <= S_IDLE; core_busy <= 1'b0;
            rom_addr <= {ADDR_W{1'b0}}; op_sel <= 8'd0;
            copy_cfg_we <= 1'b0;
            pool_cfg_we <= 1'b0;
        end else begin
            core_done <= 1'b0;
            record_done <= 1'b0;
            launch <= 1'b0;
            copy_cfg_we <= 1'b0;
            pool_cfg_we <= 1'b0;
            case (state)
                S_IDLE: begin
                    core_busy <= 1'b0;
                    rom_addr <= {ADDR_W{1'b0}};
                    if (core_start) begin
                        seq_error <= 1'b0; seq_error_code <= 16'h0;
                        record_done <= 1'b0; record_index <= 16'h0; rec_index <= 16'h0;
                        core_busy <= 1'b1; state <= S_START;
                    end
                end

                S_START: begin
                    if (rom_data == 32'hFFFFFFFF) begin
                        rom_addr <= rom_inc; state <= S_LEN;
                    end else begin
                        seq_error_code <= E_SEQ_STREAM; state <= S_ERROR;
                    end
                end

                S_LEN: begin
                    word_count <= rom_data[15:0]; stream_words_seen <= 0; rom_addr <= rom_inc; state <= S_HDR;
                end

                S_HDR: begin
                    if (rom_data == 32'hFFFFFFFF) begin
                        state <= S_DONE;
                    end else begin
                        stream_words_seen <= stream_words_seen + 1;
                        pay_idx <= 6'd0;
                        cfg_in1_addr <= 0; cfg_in1_bytes <= 0;
                        cfg_weight_addr <= 0; cfg_bias_addr <= 0; cfg_quant_addr <= 0;
                        cfg_h <= 0; cfg_w <= 0; cfg_cin <= 0; cfg_cout <= 0;
                        cfg_conv_cfg <= 0; cfg_quant_cfg <= 0; cfg_act_cfg <= 0;
                        cfg_in_row_stride_bytes <= 0; cfg_out_row_stride_bytes <= 0;
                        cfg_in_bytes <= 0; cfg_weight_bytes <= 0; cfg_bias_bytes <= 0;
                        cfg_quant_bytes <= 0; cfg_out_bytes <= 0; cfg_elem_cfg <= 0;
                        cfg_rescale_a <= 0; cfg_rescale_b <= 0; cfg_rescale_cfg <= 0;

                        if (rom_data[31:16] == OP_CONV) begin
                            if (rom_data[15:0] != PAY_CONV) begin
                                seq_error_code <= E_SEQ_PAYLOAD; state <= S_ERROR;
                            end else begin
                                op_sel <= 8'h01; pay_left <= rom_data[15:0];
                                rom_addr <= rom_inc; state <= S_PAY;
                            end
                        end else if (rom_data[31:16] == OP_POOL) begin
                            if (rom_data[15:0] != PAY_CONV) begin
                                seq_error_code <= E_SEQ_PAYLOAD; state <= S_ERROR;
                            end else begin
                                op_sel <= 8'h02; pay_left <= rom_data[15:0];
                                rom_addr <= rom_inc; state <= S_PAY;
                            end
                        end else if (rom_data[31:16] == OP_RELU) begin
                            if (rom_data[15:0] != PAY_EW) begin
                                seq_error_code <= E_SEQ_PAYLOAD; state <= S_ERROR;
                            end else begin
                                op_sel <= 8'h10; pay_left <= rom_data[15:0];
                                rom_addr <= rom_inc; state <= S_PAY;
                            end
                        end else if (rom_data[31:16] == OP_ADD) begin
                            if ((rom_data[15:0] != PAY_EW) &&
                                (rom_data[15:0] != PAY_EW_AFFINE)) begin
                                seq_error_code <= E_SEQ_PAYLOAD; state <= S_ERROR;
                            end else begin
                                op_sel <= 8'h11; pay_left <= rom_data[15:0];
                                rom_addr <= rom_inc; state <= S_PAY;
                            end
                        end else if (rom_data[31:16] == OP_PAD) begin
                            if ((rom_data[15:0] != PAY_COPY) &&
                                (rom_data[15:0] != PAY_COPY_V2)) begin
                                seq_error_code <= E_SEQ_PAYLOAD; state <= S_ERROR;
                            end else begin
                                op_sel <= 8'h12; pay_left <= rom_data[15:0];
                                rom_addr <= rom_inc; state <= S_PAY;
                            end
                        end else if (rom_data[31:16] == OP_SLICE) begin
                            if ((rom_data[15:0] != PAY_COPY) &&
                                (rom_data[15:0] != PAY_COPY_V2)) begin
                                seq_error_code <= E_SEQ_PAYLOAD; state <= S_ERROR;
                            end else begin
                                op_sel <= 8'h13; pay_left <= rom_data[15:0];
                                rom_addr <= rom_inc; state <= S_PAY;
                            end
                        end else if (rom_data[31:16] == OP_CONCAT) begin
                            if ((rom_data[15:0] != PAY_COPY) &&
                                (rom_data[15:0] != PAY_COPY_V2)) begin
                                seq_error_code <= E_SEQ_PAYLOAD; state <= S_ERROR;
                            end else begin
                                op_sel <= 8'h14; pay_left <= rom_data[15:0];
                                rom_addr <= rom_inc; state <= S_PAY;
                            end
                        end else if (rom_data[31:16] == OP_CLIP) begin
                            if (rom_data[15:0] != PAY_RED) begin
                                seq_error_code <= E_SEQ_PAYLOAD; state <= S_ERROR;
                            end else begin
                                op_sel <= 8'h15; pay_left <= rom_data[15:0];
                                rom_addr <= rom_inc; state <= S_PAY;
                            end
                        end else if (rom_data[31:16] == OP_GAP) begin
                            if (rom_data[15:0] != PAY_RED) begin
                                seq_error_code <= E_SEQ_PAYLOAD; state <= S_ERROR;
                            end else begin
                                op_sel <= 8'h16; pay_left <= rom_data[15:0];
                                rom_addr <= rom_inc; state <= S_PAY;
                            end
                        end else if (rom_data[31:16] == OP_AVGPOOL) begin
                            if (rom_data[15:0] != PAY_CONV) begin
                                seq_error_code <= E_SEQ_PAYLOAD; state <= S_ERROR;
                            end else begin
                                op_sel <= 8'h17; pay_left <= rom_data[15:0];
                                rom_addr <= rom_inc; state <= S_PAY;
                            end
                        end else if (rom_data[31:16] == OP_MUL) begin
                            if (rom_data[15:0] != PAY_EW) begin
                                seq_error_code <= E_SEQ_PAYLOAD; state <= S_ERROR;
                            end else begin
                                op_sel <= 8'h18; pay_left <= rom_data[15:0];
                                rom_addr <= rom_inc; state <= S_PAY;
                            end
                        end else if (rom_data[31:16] == OP_RESIZE) begin
                            if (rom_data[15:0] != PAY_RESIZE) begin
                                seq_error_code <= E_SEQ_PAYLOAD; state <= S_ERROR;
                            end else begin
                                op_sel <= 8'h19; pay_left <= rom_data[15:0];
                                rom_addr <= rom_inc; state <= S_PAY;
                            end
                        end else if (rom_data[31:16] == OP_POOL5) begin
                            if (rom_data[15:0] != PAY_CONV) begin
                                seq_error_code <= E_SEQ_PAYLOAD; state <= S_ERROR;
                            end else begin
                                op_sel <= 8'h1A; pay_left <= rom_data[15:0];
                                rom_addr <= rom_inc; state <= S_PAY;
                            end
                        end else if (rom_data[31:16] == OP_ACTLUT) begin
                            if (rom_data[15:0] != PAY_CONV) begin
                                seq_error_code <= E_SEQ_PAYLOAD; state <= S_ERROR;
                            end else begin
                                op_sel <= 8'h1B; pay_left <= rom_data[15:0];
                                rom_addr <= rom_inc; state <= S_PAY;
                            end
                        end else if (rom_data[31:16] == OP_DEPTHWISE) begin
                            if (rom_data[15:0] != PAY_CONV) begin
                                seq_error_code <= E_SEQ_PAYLOAD; state <= S_ERROR;
                            end else begin
                                op_sel <= 8'h1C; pay_left <= rom_data[15:0];
                                rom_addr <= rom_inc; state <= S_PAY;
                            end
                        end else if (rom_data[31:16] == OP_POOL2D) begin
                            // Require the entire record and its terminator to fit
                            // in the declared stream, before any config writes.
                            if (rom_data[15:0] != PAY_POOL2D ||
                                ({16'd0,stream_words_seen} + 32'd17) >= {16'd0,word_count}) begin
                                seq_error_code <= E_SEQ_PAYLOAD; state <= S_ERROR;
                            end else begin
                                op_sel <= 8'h1D; pay_left <= rom_data[15:0];
                                rom_addr <= rom_inc; state <= S_PAY;
                            end
                        end else begin
                            seq_error_code <= E_SEQ_OPCODE; state <= S_ERROR;
                        end
                    end
                end

                S_PAY: begin
                    stream_words_seen <= stream_words_seen + 1;
                    pay_bad = 1'b0;
                    if (op_sel == 8'h1D) begin
                        if ((pay_idx == 0 && rom_data != 32'd1) ||
                            (pay_idx == 15 && rom_data != 32'd0) ||
                            ((pay_idx == 1 || pay_idx == 2) && reloc_invalid(rom_data))) begin
                            pay_bad = 1'b1;
                        end else begin
                            pool_cfg_we <= 1'b1;
                            pool_cfg_addr <= pay_idx[3:0];
                            pool_cfg_data <= (pay_idx == 1 || pay_idx == 2) ? reloc_addr(rom_data) : rom_data;
                        end
                    end else if (op_sel == 8'h12 || op_sel == 8'h13 || op_sel == 8'h14) begin
                        copy_cfg_we <= 1'b1;
                        copy_cfg_addr <= pay_idx;
                        copy_cfg_data <= (pay_idx < 5'd5) ? reloc_addr(rom_data) : rom_data;
                    end else if (op_sel == 8'h15) begin
                        case (pay_idx)
                            5'd0: cfg_in_addr <= reloc_addr(rom_data);
                            5'd1: cfg_out_addr <= reloc_addr(rom_data);
                            5'd2: cfg_in_bytes <= rom_data;
                            5'd3: cfg_out_bytes <= rom_data;
                            5'd4: cfg_in_row_stride_bytes <= rom_data;
                            5'd5: cfg_out_row_stride_bytes <= rom_data;
                            5'd6: begin cfg_h <= rom_data[15:0]; cfg_w <= rom_data[31:16]; end
                            5'd7: begin cfg_cin <= rom_data[15:0]; cfg_cout <= rom_data[31:16]; end
                            5'd8: cfg_elem_cfg <= rom_data;
                        endcase
                    end else if (op_sel == 8'h16) begin
                        case (pay_idx)
                            5'd0: cfg_in_addr <= reloc_addr(rom_data);
                            5'd1: cfg_out_addr <= reloc_addr(rom_data);
                            5'd2: cfg_in_bytes <= rom_data;
                            5'd3: cfg_out_bytes <= rom_data;
                            5'd4: cfg_in_row_stride_bytes <= rom_data;
                            5'd5: begin cfg_h <= rom_data[15:0]; cfg_w <= rom_data[31:16]; end
                            5'd6: begin cfg_cin <= rom_data[15:0]; cfg_cout <= rom_data[31:16]; end
                            5'd7: cfg_elem_cfg <= rom_data;
                        endcase
                    end else if (op_sel == 8'h19) begin
                        case (pay_idx)
                            5'd0: cfg_in_addr <= reloc_addr(rom_data);
                            5'd1: cfg_out_addr <= reloc_addr(rom_data);
                            5'd2: cfg_in_bytes <= rom_data;
                            5'd3: cfg_out_bytes <= rom_data;
                            5'd4: cfg_in_row_stride_bytes <= rom_data;
                            5'd5: cfg_out_row_stride_bytes <= rom_data;
                            5'd6: begin cfg_h <= rom_data[15:0]; cfg_w <= rom_data[31:16]; end
                            5'd7: begin cfg_cin <= rom_data[15:0]; cfg_cout <= rom_data[31:16]; end
                            5'd8: cfg_quant_cfg <= rom_data;
                        endcase
                    end else if (op_sel == 8'h01 || op_sel == 8'h02 || op_sel == 8'h17 || op_sel == 8'h1A) begin
                        case (pay_idx)
                            5'd0: cfg_in_addr <= reloc_addr(rom_data);
                            5'd1: cfg_weight_addr <= reloc_addr(rom_data);
                            5'd2: cfg_bias_addr <= reloc_addr(rom_data);
                            5'd3: cfg_quant_addr <= reloc_addr(rom_data);
                            5'd4: cfg_out_addr <= reloc_addr(rom_data);
                            5'd5: begin cfg_h <= rom_data[15:0]; cfg_w <= rom_data[31:16]; end
                            5'd6: begin cfg_cin <= rom_data[15:0]; cfg_cout <= rom_data[31:16]; end
                            5'd7: cfg_conv_cfg <= rom_data;
                            5'd8: cfg_quant_cfg <= rom_data;
                            5'd9: cfg_act_cfg <= rom_data;
                            5'd10: cfg_in_row_stride_bytes <= rom_data;
                            5'd11: cfg_out_row_stride_bytes <= rom_data;
                            5'd12: cfg_in_bytes <= rom_data;
                            5'd13: cfg_weight_bytes <= rom_data;
                            5'd14: cfg_bias_bytes <= rom_data;
                            5'd15: cfg_quant_bytes <= rom_data;
                            5'd16: cfg_out_bytes <= rom_data;
                        endcase
                    end else if (op_sel == 8'h1B) begin
                        // ActivationLUT 17-word payload (SPEC 1.1):
                        // 0 in, 3 lut, 4 out (relocated); 5 H/W, 6 C/C_alloc;
                        // 10 in stride, 11 out stride, 12 in bytes,
                        // 15 lut_bytes, 16 out bytes; words 1,2,7,8,9,13,14 = 0.
                        case (pay_idx)
                            5'd0:  cfg_in_addr <= reloc_addr(rom_data);
                            5'd1:  if (rom_data != 32'b0) pay_bad = 1'b1;
                            5'd2:  if (rom_data != 32'b0) pay_bad = 1'b1;
                            5'd3:  cfg_quant_addr <= reloc_addr(rom_data);
                            5'd4:  cfg_out_addr <= reloc_addr(rom_data);
                            5'd5:  begin cfg_h <= rom_data[15:0]; cfg_w <= rom_data[31:16]; end
                            5'd6:  begin cfg_cin <= rom_data[15:0]; cfg_cout <= rom_data[31:16]; end
                            5'd7:  if (rom_data != 32'b0) pay_bad = 1'b1;
                            5'd8:  if (rom_data != 32'b0) pay_bad = 1'b1;
                            5'd9:  if (rom_data != 32'b0) pay_bad = 1'b1;
                            5'd10: cfg_in_row_stride_bytes <= rom_data;
                            5'd11: cfg_out_row_stride_bytes <= rom_data;
                            5'd12: cfg_in_bytes <= rom_data;
                            5'd13: if (rom_data != 32'b0) pay_bad = 1'b1;
                            5'd14: if (rom_data != 32'b0) pay_bad = 1'b1;
                            5'd15: cfg_quant_bytes <= rom_data;
                            5'd16: cfg_out_bytes <= rom_data;
                            default: pay_bad = 1'b1;
                        endcase
                    end else if (op_sel == 8'h1C) begin
                        // DepthwiseConv3x3 17-word Conv-layout payload (SPEC 1.2):
                        // words 0..4 relocated; 7 conv_cfg must be exactly
                        // 0x00000400 | ((stride-1)<<2) | (pad<<4), words 8/9 = 0.
                        case (pay_idx)
                            5'd0: cfg_in_addr <= reloc_addr(rom_data);
                            5'd1: cfg_weight_addr <= reloc_addr(rom_data);
                            5'd2: cfg_bias_addr <= reloc_addr(rom_data);
                            5'd3: cfg_quant_addr <= reloc_addr(rom_data);
                            5'd4: cfg_out_addr <= reloc_addr(rom_data);
                            5'd5: begin cfg_h <= rom_data[15:0]; cfg_w <= rom_data[31:16]; end
                            5'd6: begin cfg_cin <= rom_data[15:0]; cfg_cout <= rom_data[31:16]; end
                            5'd7: begin
                                // exact formula: bits other than {10,5,4,3,2} are 0,
                                // bit10 must be 1, stride-1 = [3:2] in {0,1},
                                // pad = [5:4] in {0,1}.
                                if (((rom_data & 32'hFFFFFBC3) != 32'b0) || (rom_data[10] != 1'b1) ||
                                    (rom_data[3:2] > 2'd1) || (rom_data[5:4] > 2'd1)) begin
                                    pay_bad = 1'b1;
                                end else begin
                                    cfg_conv_cfg <= rom_data;
                                end
                            end
                            5'd8:  if (rom_data != 32'b0) pay_bad = 1'b1;
                            5'd9:  if (rom_data != 32'b0) pay_bad = 1'b1;
                            5'd10: cfg_in_row_stride_bytes <= rom_data;
                            5'd11: cfg_out_row_stride_bytes <= rom_data;
                            5'd12: cfg_in_bytes <= rom_data;
                            5'd13: cfg_weight_bytes <= rom_data;
                            5'd14: cfg_bias_bytes <= rom_data;
                            5'd15: cfg_quant_bytes <= rom_data;
                            5'd16: cfg_out_bytes <= rom_data;
                            default: pay_bad = 1'b1;
                        endcase
                    end else if (op_sel == 8'h11) begin
                        // Add: legacy 7-word elementwise payload, or the affine
                        // 10-word variant whose extra words carry the two
                        // multipliers and the (shift, zp_a, zp_b, zp_out) pack.
                        case (pay_idx)
                            5'd0: cfg_in_addr <= reloc_addr(rom_data);
                            5'd1: cfg_in1_addr <= reloc_addr(rom_data);
                            5'd2: cfg_out_addr <= reloc_addr(rom_data);
                            5'd3: cfg_in_bytes <= rom_data;
                            5'd4: cfg_in1_bytes <= rom_data;
                            5'd5: cfg_out_bytes <= rom_data;
                            5'd6: cfg_elem_cfg <= rom_data;
                            5'd7: cfg_rescale_a <= rom_data;
                            5'd8: cfg_rescale_b <= rom_data;
                            5'd9: cfg_rescale_cfg <= rom_data;
                        endcase
                    end else begin
                        case (pay_idx)
                            5'd0: cfg_in_addr <= reloc_addr(rom_data);
                            5'd1: cfg_in1_addr <= reloc_addr(rom_data);
                            5'd2: cfg_out_addr <= reloc_addr(rom_data);
                            5'd3: cfg_in_bytes <= rom_data;
                            5'd4: cfg_in1_bytes <= rom_data;
                            5'd5: cfg_out_bytes <= rom_data;
                            5'd6: cfg_elem_cfg <= rom_data;
                        endcase
                    end
                    if (pay_bad) begin
                        seq_error_code <= E_SEQ_PAYLOAD;
                        state <= S_ERROR;
                    end else begin
                        rom_addr <= rom_inc;
                        if (pay_left <= 16'd1) begin
                            state <= S_LAUNCH;
                        end else begin
                            pay_left <= pay_left - 16'd1;
                            pay_idx <= pay_idx + 5'd1;
                        end
                    end
                end

                S_LAUNCH: begin launch <= 1'b1; state <= S_WBUSY; end
                S_WBUSY: begin
                    if (active_core_error) begin
                        seq_error_code <= active_core_error_code; state <= S_ERROR;
                    end else if (active_core_busy) begin
                        state <= S_WDONE;
                    end
                end
                S_WDONE: begin
                    if (active_core_error) begin
                        seq_error_code <= active_core_error_code; state <= S_ERROR;
                    end else if (active_core_done) begin
                        record_done <= 1'b1; record_index <= rec_index;
                        rec_index <= rec_index + 16'd1; state <= S_HDR;
                    end
                end
                S_DONE: begin core_busy <= 1'b0; core_done <= 1'b1; state <= S_IDLE; end
                S_ERROR: begin core_busy <= 1'b0; seq_error <= 1'b1; state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
