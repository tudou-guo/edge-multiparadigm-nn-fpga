`timescale 1ns/1ps
// Actual sequencer + router, with a bounded normalized-memory-service model.
module tb_npu_pool2d;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0, start = 0, soft_reset = 0;
    wire seq_busy, seq_done, seq_error;
    wire [15:0] seq_error_code, record_index;
    wire record_done;
    wire [7:0] rom_addr, op_sel;
    wire launch, pool_cfg_we;
    wire [3:0] pool_cfg_addr;
    wire [31:0] pool_cfg_data;
    wire active_busy, active_done, active_error;
    wire [15:0] active_error_code;
    wire [31:0] cfg_in_addr, cfg_in1_addr, cfg_weight_addr, cfg_bias_addr, cfg_quant_addr, cfg_out_addr;
    wire [15:0] cfg_h, cfg_w, cfg_cin, cfg_cout;
    wire [31:0] cfg_conv_cfg, cfg_quant_cfg, cfg_act_cfg, cfg_in_row, cfg_out_row;
    wire [31:0] cfg_in_bytes, cfg_in1_bytes, cfg_weight_bytes, cfg_bias_bytes, cfg_quant_bytes, cfg_out_bytes;
    wire [31:0] cfg_elem_cfg, cfg_rescale_a, cfg_rescale_b, cfg_rescale_cfg;
    reg [31:0] rom [0:255];
    wire [31:0] rom_data = rom[rom_addr];
    npu_instr_seq #(.ADDR_W(8), .ROM_BASE_ADDR(32'h10000000)) seq (
        .clk(clk), .rst_n(rst_n), .core_start(start), .runtime_base_addr(32'h20000000),
        .core_busy(seq_busy), .core_done(seq_done), .seq_error(seq_error), .seq_error_code(seq_error_code),
        .core_soft_reset(soft_reset), .core_abort(1'b0),
        .active_core_busy(active_busy), .active_core_done(active_done),
        .active_core_error(active_error), .active_core_error_code(active_error_code),
        .record_index(record_index), .record_done(record_done), .rom_addr(rom_addr), .rom_data(rom_data),
        .op_sel(op_sel), .launch(launch), .pool_cfg_we(pool_cfg_we),
        .pool_cfg_addr(pool_cfg_addr), .pool_cfg_data(pool_cfg_data),
        .cfg_in_addr(cfg_in_addr), .cfg_in1_addr(cfg_in1_addr), .cfg_weight_addr(cfg_weight_addr),
        .cfg_bias_addr(cfg_bias_addr), .cfg_quant_addr(cfg_quant_addr), .cfg_out_addr(cfg_out_addr),
        .cfg_h(cfg_h), .cfg_w(cfg_w), .cfg_cin(cfg_cin), .cfg_cout(cfg_cout),
        .cfg_conv_cfg(cfg_conv_cfg), .cfg_quant_cfg(cfg_quant_cfg), .cfg_act_cfg(cfg_act_cfg),
        .cfg_in_row_stride_bytes(cfg_in_row), .cfg_out_row_stride_bytes(cfg_out_row),
        .cfg_in_bytes(cfg_in_bytes), .cfg_in1_bytes(cfg_in1_bytes), .cfg_weight_bytes(cfg_weight_bytes),
        .cfg_bias_bytes(cfg_bias_bytes), .cfg_quant_bytes(cfg_quant_bytes), .cfg_out_bytes(cfg_out_bytes),
        .cfg_elem_cfg(cfg_elem_cfg), .cfg_rescale_a(cfg_rescale_a), .cfg_rescale_b(cfg_rescale_b),
        .cfg_rescale_cfg(cfg_rescale_cfg),
        .copy_cfg_we(), .copy_cfg_addr(), .copy_cfg_data()
    );

    wire rd_cmd_valid, rd_cmd_ready, rd_data_valid, rd_data_ready, rd_data_last;
    wire [31:0] rd_addr;
    wire [15:0] rd_bytes;
    wire [7:0] rd_tag;
    wire [127:0] rd_data;
    wire wr_cmd_valid, wr_cmd_ready, wr_data_valid, wr_data_ready, wr_data_last, wr_rsp_valid, wr_rsp_ready;
    wire [31:0] wr_addr;
    wire [15:0] wr_bytes;
    wire [7:0] wr_tag, wr_data_tag;
    wire [127:0] wr_data;
    npu_engine_router #(.MAX_CIN(8), .MAX_COUT(8), .MAX_CH(16), .MAX_W(8), .MAX_H(8),
                        .Tn(8), .Tm(8), .MAC_IMPL(2), .PHYSICAL_ENGINE_MASK(13'h1002),
                        .POOL2D_MAX_IN_ROW_BYTES(4096), .POOL2D_MAX_OUT_ROW_BYTES(4096),
                        .POOL2D_MAX_EH(25), .POOL2D_CACHE_ROWS(32)) router (
        .clk(clk), .rst_n(rst_n), .op_sel(op_sel), .launch(launch), .soft_reset(soft_reset), .abort(1'b0),
        .cfg_in_addr(cfg_in_addr), .cfg_in1_addr(cfg_in1_addr), .cfg_weight_addr(cfg_weight_addr),
        .cfg_bias_addr(cfg_bias_addr), .cfg_quant_addr(cfg_quant_addr), .cfg_out_addr(cfg_out_addr),
        .cfg_h(cfg_h), .cfg_w(cfg_w), .cfg_cin(cfg_cin), .cfg_cout(cfg_cout),
        .cfg_conv_cfg(cfg_conv_cfg), .cfg_quant_cfg(cfg_quant_cfg), .cfg_act_cfg(cfg_act_cfg),
        .cfg_in_row_stride_bytes(cfg_in_row), .cfg_out_row_stride_bytes(cfg_out_row),
        .cfg_in_bytes(cfg_in_bytes), .cfg_in1_bytes(cfg_in1_bytes), .cfg_elem_cfg(cfg_elem_cfg),
        .cfg_rescale_a(cfg_rescale_a), .cfg_rescale_b(cfg_rescale_b), .cfg_rescale_cfg(cfg_rescale_cfg),
        .cfg_weight_bytes(cfg_weight_bytes), .cfg_bias_bytes(cfg_bias_bytes),
        .cfg_quant_bytes(cfg_quant_bytes), .cfg_out_bytes(cfg_out_bytes),
        .copy_cfg_we(1'b0), .copy_cfg_addr(6'd0), .copy_cfg_data(32'd0),
        .pool_cfg_we(pool_cfg_we), .pool_cfg_addr(pool_cfg_addr), .pool_cfg_data(pool_cfg_data),
        .active_core_busy(active_busy), .active_core_done(active_done),
        .active_core_error(active_error), .active_core_error_code(active_error_code),
        .shell_rd_cmd_valid(rd_cmd_valid), .shell_rd_cmd_ready(rd_cmd_ready),
        .shell_rd_cmd_addr_byte(rd_addr), .shell_rd_cmd_bytes(rd_bytes), .shell_rd_cmd_tag(rd_tag),
        .shell_rd_data_valid(rd_data_valid), .shell_rd_data_ready(rd_data_ready), .shell_rd_data(rd_data),
        .shell_rd_data_last(rd_data_last), .shell_rd_data_error(1'b0), .shell_rd_data_error_code(16'd0),
        .shell_rd_data_tag(8'd1), .shell_wr_cmd_valid(wr_cmd_valid), .shell_wr_cmd_ready(wr_cmd_ready),
        .shell_wr_cmd_addr_byte(wr_addr), .shell_wr_cmd_bytes(wr_bytes), .shell_wr_cmd_tag(wr_tag),
        .shell_wr_data_valid(wr_data_valid), .shell_wr_data_ready(wr_data_ready), .shell_wr_data(wr_data),
        .shell_wr_data_last(wr_data_last), .shell_wr_data_tag(wr_data_tag),
        .shell_wr_rsp_valid(wr_rsp_valid), .shell_wr_rsp_ready(wr_rsp_ready),
        .shell_wr_rsp_error(1'b0), .shell_wr_rsp_aborted(1'b0),
        .shell_wr_rsp_error_code(16'd0), .shell_wr_rsp_tag(8'd2)
    );

    reg [127:0] memory [0:16383];
    reg [127:0] expected [0:16383];
    reg [31:0] rng = 20261029;
    reg read_active = 0, write_active = 0, write_complete = 0;
    integer read_ptr, read_remaining, write_ptr, write_remaining, response_delay;
    integer records = 0, write_beats = 0, expected_beats, fd, rc, i, negative_tests = 0;
    integer forbidden_launches = 0;
    assign rd_cmd_ready = !read_active && rng[0];
    assign rd_data_valid = read_active && rng[1];
    assign rd_data = memory[read_ptr];
    assign rd_data_last = read_remaining == 1;
    assign wr_cmd_ready = !write_active && rng[2];
    assign wr_data_ready = write_active && !write_complete && rng[3];
    assign wr_rsp_valid = write_active && write_complete && response_delay >= 3 && rng[4];

    always @(posedge clk) if (rst_n) begin
        rng <= {rng[30:0], rng[31] ^ rng[21] ^ rng[1] ^ rng[0]};
        if (record_done) begin
            if (record_index != records) $fatal(1, "FAIL record index got=%0d expected=%0d", record_index, records);
            records <= records + 1;
        end
        if (rd_cmd_valid && rd_cmd_ready) begin
            if (read_active || write_active || rd_tag != 1 || rd_addr < 32'h20000000 || rd_addr >= 32'h20040000)
                $fatal(1, "FAIL routed read command addr=%h tag=%h", rd_addr, rd_tag);
            read_active <= 1; read_ptr <= (rd_addr - 32'h20000000) >> 4; read_remaining <= rd_bytes >> 4;
        end
        if (rd_data_valid && rd_data_ready) begin
            if (read_remaining == 1) read_active <= 0;
            else begin read_ptr <= read_ptr + 1; read_remaining <= read_remaining - 1; end
        end
        if (wr_cmd_valid && wr_cmd_ready) begin
            if (read_active || write_active || wr_tag != 2 || wr_addr < 32'h20000000 || wr_addr >= 32'h20040000)
                $fatal(1, "FAIL routed write command addr=%h tag=%h", wr_addr, wr_tag);
            write_active <= 1; write_complete <= 0; response_delay <= 0;
            write_ptr <= (wr_addr - 32'h20000000) >> 4; write_remaining <= wr_bytes >> 4;
        end
        if (wr_data_valid && wr_data_ready) begin
            if (!write_active || wr_data_tag != 2 || wr_data_last != (write_remaining == 1))
                $fatal(1, "FAIL routed write framing");
            if (wr_data !== expected[write_ptr])
                $fatal(1, "FAIL routed output beat=%0d got=%h expected=%h", write_ptr, wr_data, expected[write_ptr]);
            write_beats <= write_beats + 1;
            if (write_remaining == 1) write_complete <= 1;
            else begin write_ptr <= write_ptr + 1; write_remaining <= write_remaining - 1; end
        end
        if (write_active && write_complete && !wr_rsp_valid) response_delay <= response_delay + 1;
        if (wr_rsp_valid && wr_rsp_ready) begin write_active <= 0; write_complete <= 0; end
    end

    task automatic reject_stream(input integer test_no);
        begin
            @(negedge clk); soft_reset = 1;
            @(negedge clk); soft_reset = 0; start = 1;
            @(negedge clk); start = 0;
            i = 0; forbidden_launches = 0;
            while (!seq_error && i < 1000) begin
                @(negedge clk); i = i + 1;
                if (launch || rd_cmd_valid || wr_cmd_valid) forbidden_launches = forbidden_launches + 1;
            end
            if (!seq_error || seq_error_code != 16'h00a3 || forbidden_launches != 0)
                $fatal(1, "FAIL malformed stream test=%0d error=%h launches=%0d", test_no, seq_error_code, forbidden_launches);
            negative_tests = negative_tests + 1;
            $display("ROUTER_REJECT_PASS test=%0d code=%h", test_no, seq_error_code);
        end
    endtask

    initial begin
        if ($value$plusargs("seed=%d", i)) rng = i;
        $readmemh("integration_instr.hex", rom, 0, 53);
        $readmemh("integration_input.hex", memory);
        $readmemh("integration_expected.hex", expected);
        fd = $fopen("integration_count.txt", "r");
        rc = $fscanf(fd, "%d", expected_beats);
        $fclose(fd);
        if (rc != 1) $fatal(1, "FAIL expected count");
        repeat (4) @(negedge clk);
        rst_n = 1; @(negedge clk); start = 1; @(negedge clk); start = 0;
        i = 0;
        while (!seq_done && !seq_error && i < 2000000) begin @(negedge clk); i = i + 1; end
        if (seq_error) $fatal(1, "FAIL sequencer error=%h records=%0d op=%h", seq_error_code, records, op_sel);
        if (!seq_done || records != 3 || write_beats != expected_beats || read_active || write_active)
            $fatal(1, "FAIL completion records=%0d beats=%0d expected=%0d busy=%b", records, write_beats, expected_beats, seq_busy);
        $display("ROUTER_RECORDS_PASS records=%0d output_beats=%0d cycles=%0d", records, write_beats, i);
        rom[2] = 32'h001d000f; reject_stream(1); rom[2] = 32'h001d0010;
        rom[3] = 2; reject_stream(2); rom[3] = 1;
        rom[18] = 1; reject_stream(3); rom[18] = 0;
        rom[1] = 10; reject_stream(4); rom[1] = 52;
        rom[4] = 32'hfffffff0; reject_stream(5);
        // Frozen legacy 0x0002 record between two generic records. The legacy
        // vector explicitly expects its historical clamp-to-zero behavior.
        @(negedge clk); soft_reset = 1;
        $readmemh("legacy_instr.hex", rom, 0, 54);
        $readmemh("legacy_input.hex", memory);
        $readmemh("legacy_expected.hex", expected);
        fd = $fopen("legacy_count.txt", "r"); rc = $fscanf(fd, "%d", expected_beats); $fclose(fd);
        if (rc != 1) $fatal(1, "FAIL legacy count");
        records = 0; write_beats = 0;
        @(negedge clk); soft_reset = 0; start = 1;
        @(negedge clk); start = 0; i = 0;
        while (!seq_done && !seq_error && i < 2000000) begin @(negedge clk); i = i + 1; end
        if (seq_error || !seq_done || records != 3 || write_beats != expected_beats)
            $fatal(1, "FAIL legacy sequence error=%h records=%0d beats=%0d/%0d", seq_error_code, records, write_beats, expected_beats);
        $display("LEGACY_MIX_TEST_PASS records=%0d output_beats=%0d", records, write_beats);
        $display("ROUTER_TEST_PASS records=%0d output_beats=%0d rejected=%0d", records, write_beats, negative_tests);
        @(negedge clk); soft_reset = 1;
        $readmemh("legacy_only_instr.hex", rom, 0, 20);
        records = 0; write_beats = 0;
        @(negedge clk); soft_reset = 0; start = 1;
        @(negedge clk); start = 0; i = 0;
        while (!seq_done && !seq_error && i < 2000000) begin @(negedge clk); i = i + 1; end
        if (seq_error || !seq_done || records != 1 || write_beats != 2)
            $fatal(1, "FAIL legacy-only sequence error=%h records=%0d beats=%0d", seq_error_code, records, write_beats);
        $display("LEGACY_ONLY_TEST_PASS records=%0d output_beats=%0d", records, write_beats);
        $finish;
    end
    initial begin #20000000; $fatal(1, "FAIL router watchdog"); end
endmodule
