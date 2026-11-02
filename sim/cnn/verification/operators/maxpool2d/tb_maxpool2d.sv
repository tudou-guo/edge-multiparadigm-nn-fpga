`timescale 1ns/1ps
module tb_maxpool2d;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0, pool_cfg_we = 0, core_start = 0, core_soft_reset = 0, core_abort = 0;
    reg [3:0] pool_cfg_addr = 0;
    reg [31:0] pool_cfg_data = 0;
    wire core_busy, core_done, core_error;
    wire [15:0] core_error_code;
    wire rd_cmd_valid, rd_cmd_ready, rd_data_valid, rd_data_ready, rd_data_last;
    wire [31:0] rd_cmd_addr_byte;
    wire [15:0] rd_cmd_bytes;
    wire [7:0] rd_cmd_tag, rd_data_tag;
    wire [127:0] rd_data;
    wire wr_cmd_valid, wr_cmd_ready, wr_data_valid, wr_data_ready, wr_data_last;
    wire [31:0] wr_cmd_addr_byte;
    wire [15:0] wr_cmd_bytes;
    wire [7:0] wr_cmd_tag, wr_data_tag;
    wire [127:0] wr_data;
    wire wr_rsp_valid, wr_rsp_ready;
    wire [7:0] wr_rsp_tag;
    reg inject_rd_error = 0, inject_rd_tag = 0, inject_rd_last = 0;
    reg inject_wr_error = 0, inject_wr_aborted = 0, inject_wr_tag = 0, force_wr_pause = 0;

    maxpool2d_core #(.MAX_IN_ROW_BYTES(8192), .MAX_OUT_ROW_BYTES(8192),
                     .MAX_EH(25), .CACHE_ROWS(32)) dut (
        .clk(clk), .rst_n(rst_n), .pool_cfg_we(pool_cfg_we), .pool_cfg_addr(pool_cfg_addr),
        .pool_cfg_data(pool_cfg_data), .core_start(core_start), .core_busy(core_busy),
        .core_done(core_done), .core_error(core_error), .core_error_code(core_error_code),
        .core_soft_reset(core_soft_reset), .core_abort(core_abort),
        .rd_cmd_valid(rd_cmd_valid), .rd_cmd_ready(rd_cmd_ready), .rd_cmd_addr_byte(rd_cmd_addr_byte),
        .rd_cmd_bytes(rd_cmd_bytes), .rd_cmd_tag(rd_cmd_tag), .rd_data_valid(rd_data_valid),
        .rd_data_ready(rd_data_ready), .rd_data(rd_data), .rd_data_last(rd_data_last),
        .rd_data_error(inject_rd_error), .rd_data_error_code(16'h0009), .rd_data_tag(rd_data_tag),
        .wr_cmd_valid(wr_cmd_valid), .wr_cmd_ready(wr_cmd_ready), .wr_cmd_addr_byte(wr_cmd_addr_byte),
        .wr_cmd_bytes(wr_cmd_bytes), .wr_cmd_tag(wr_cmd_tag), .wr_data_valid(wr_data_valid),
        .wr_data_ready(wr_data_ready), .wr_data(wr_data), .wr_data_last(wr_data_last), .wr_data_tag(wr_data_tag),
        .wr_rsp_valid(wr_rsp_valid), .wr_rsp_ready(wr_rsp_ready), .wr_rsp_error(inject_wr_error),
        .wr_rsp_aborted(inject_wr_aborted), .wr_rsp_error_code(16'h0009), .wr_rsp_tag(wr_rsp_tag)
    );

    reg [31:0] cfg [0:15];
    reg [127:0] input_words [0:16383];
    reg [127:0] expected_words [0:16383];
    reg [639:0] rows_seen;
    reg [31:0] rng = 32'd20261029;
    reg read_active = 0, write_active = 0, write_complete = 0, case_active = 0;
    integer read_ptr, read_remaining, write_ptr, write_remaining, response_delay;
    integer case_index, rd_commands, wr_commands, rd_beats, wr_beats, responses, cycles;
    integer load_cycles, horizontal_cycles, vertical_cycles, write_cycles, wait_cycles, check_cycles;
    integer file_handle, rc, count, i, row, observed_cycles, error_tests;
    reg [31:0] good_cfg [0:15];
    reg rd_stalled = 0, wr_stalled = 0, data_stalled = 0;
    reg [55:0] rd_cmd_old, wr_cmd_old;
    reg [136:0] wr_data_old;

    assign rd_cmd_ready = !read_active && rng[0];
    assign rd_data_valid = read_active && rng[1];
    assign rd_data = input_words[read_ptr];
    assign rd_data_last = inject_rd_last ? read_remaining == 2 : read_remaining == 1;
    assign rd_data_tag = inject_rd_tag ? 8'd3 : 8'd1;
    assign wr_cmd_ready = !write_active && rng[2];
    assign wr_data_ready = write_active && !write_complete && rng[3] && !force_wr_pause;
    assign wr_rsp_valid = write_active && write_complete && response_delay >= 3 && rng[4];
    assign wr_rsp_tag = inject_wr_tag ? 8'd3 : 8'd2;

    always @(posedge clk) begin
        if (rst_n) begin
            rng <= {rng[30:0], rng[31] ^ rng[21] ^ rng[1] ^ rng[0]};
            if (core_busy) begin
                cycles <= cycles + 1;
                if (dut.state == 5 || dut.state == 6) load_cycles <= load_cycles + 1;
                else if (dut.state >= 7 && dut.state <= 10) horizontal_cycles <= horizontal_cycles + 1;
                else if (dut.state >= 11 && dut.state <= 14) vertical_cycles <= vertical_cycles + 1;
                else if (dut.state >= 15 && dut.state <= 18) write_cycles <= write_cycles + 1;
                else if (dut.state == 19) wait_cycles <= wait_cycles + 1;
                else check_cycles <= check_cycles + 1;
            end
            if (rd_stalled && (!rd_cmd_valid || {rd_cmd_addr_byte,rd_cmd_bytes,rd_cmd_tag} !== rd_cmd_old))
                $fatal(1, "FAIL read command changed under backpressure case=%0d", case_index);
            if (wr_stalled && (!wr_cmd_valid || {wr_cmd_addr_byte,wr_cmd_bytes,wr_cmd_tag} !== wr_cmd_old))
                $fatal(1, "FAIL write command changed under backpressure case=%0d", case_index);
            if (data_stalled && (!wr_data_valid || {wr_data,wr_data_last,wr_data_tag} !== wr_data_old))
                $fatal(1, "FAIL write data changed under backpressure case=%0d", case_index);
            rd_stalled <= rd_cmd_valid && !rd_cmd_ready;
            wr_stalled <= wr_cmd_valid && !wr_cmd_ready;
            data_stalled <= wr_data_valid && !wr_data_ready;
            rd_cmd_old <= {rd_cmd_addr_byte,rd_cmd_bytes,rd_cmd_tag};
            wr_cmd_old <= {wr_cmd_addr_byte,wr_cmd_bytes,wr_cmd_tag};
            wr_data_old <= {wr_data,wr_data_last,wr_data_tag};
            if (rd_cmd_valid && rd_cmd_ready) begin
                if (read_active || write_active || rd_cmd_bytes != cfg[11] || rd_cmd_tag != 1 ||
                    rd_cmd_addr_byte < cfg[1] || ((rd_cmd_addr_byte - cfg[1]) % cfg[11]) != 0)
                    $fatal(1, "FAIL read command case=%0d addr=%h bytes=%0d", case_index, rd_cmd_addr_byte, rd_cmd_bytes);
                row = (rd_cmd_addr_byte - cfg[1]) / cfg[11];
                if (row >= cfg[3][15:0] || rows_seen[row])
                    $fatal(1, "FAIL duplicate or out-of-range input row %0d case=%0d", row, case_index);
                rows_seen[row] <= 1;
                read_ptr <= (rd_cmd_addr_byte - cfg[1]) >> 4;
                read_remaining <= rd_cmd_bytes >> 4;
                read_active <= 1;
                rd_commands <= rd_commands + 1;
            end
            if (rd_data_valid && rd_data_ready) begin
                rd_beats <= rd_beats + 1;
                if (rd_data_last) read_active <= 0;
                else begin read_ptr <= read_ptr + 1; read_remaining <= read_remaining - 1; end
            end
            if (wr_cmd_valid && wr_cmd_ready) begin
                if (read_active || write_active || wr_cmd_bytes != cfg[12] || wr_cmd_tag != 2 ||
                    wr_cmd_addr_byte != cfg[2] + wr_commands * cfg[12])
                    $fatal(1, "FAIL write command case=%0d addr=%h bytes=%0d", case_index, wr_cmd_addr_byte, wr_cmd_bytes);
                write_ptr <= (wr_cmd_addr_byte - cfg[2]) >> 4;
                write_remaining <= wr_cmd_bytes >> 4;
                write_active <= 1; write_complete <= 0; response_delay <= 0;
                wr_commands <= wr_commands + 1;
            end
            if (wr_data_valid && wr_data_ready) begin
                if (!write_active || write_complete || wr_data_tag != 2 ||
                    wr_data_last != (write_remaining == 1))
                    $fatal(1, "FAIL write beat framing case=%0d ptr=%0d", case_index, write_ptr);
                if (wr_data !== expected_words[write_ptr])
                    $fatal(1, "FAIL output case=%0d beat=%0d got=%032h expected=%032h", case_index,
                           write_ptr, wr_data, expected_words[write_ptr]);
                wr_beats <= wr_beats + 1;
                if (write_remaining == 1) write_complete <= 1;
                else begin write_ptr <= write_ptr + 1; write_remaining <= write_remaining - 1; end
            end
            if (write_active && write_complete && !wr_rsp_valid) response_delay <= response_delay + 1;
            if (wr_rsp_valid && wr_rsp_ready) begin
                write_active <= 0; write_complete <= 0; responses <= responses + 1;
            end
            if (case_active && core_done && (responses != cfg[5][15:0] || write_active || read_active))
                $fatal(1, "FAIL premature done case=%0d responses=%0d", case_index, responses);
        end
    end

    task automatic load_config;
        begin
            for (integer j = 0; j < 16; j = j + 1) begin
                @(negedge clk); pool_cfg_we = 1; pool_cfg_addr = j; pool_cfg_data = cfg[j];
            end
            @(negedge clk); pool_cfg_we = 0; core_start = 1;
            @(negedge clk); core_start = 0; case_active = 1;
        end
    endtask

    task automatic wait_done;
        begin
            observed_cycles = 0;
            while (!core_done && !core_error && observed_cycles < 2000000) begin
                @(negedge clk); observed_cycles = observed_cycles + 1;
            end
            if (core_error) $fatal(1, "FAIL core error case=%0d code=%h state=%0d", case_index, core_error_code, dut.state);
            if (!core_done) $fatal(1, "FAIL watchdog case=%0d state=%0d", case_index, dut.state);
        end
    endtask

    task automatic reset_counters;
        begin
            rows_seen = 0; rd_commands = 0; wr_commands = 0; rd_beats = 0;
            wr_beats = 0; responses = 0; cycles = 0; case_active = 0;
            load_cycles = 0; horizontal_cycles = 0; vertical_cycles = 0;
            write_cycles = 0; wait_cycles = 0; check_cycles = 0;
        end
    endtask

    task automatic restore_cfg;
        begin
            for (integer j = 0; j < 16; j = j + 1) cfg[j] = good_cfg[j];
        end
    endtask

    task automatic expect_error(input [15:0] code, input bit no_access);
        begin
            observed_cycles = 0;
            while ((!core_error || core_busy) && observed_cycles < 200000) begin
                @(negedge clk); observed_cycles = observed_cycles + 1;
            end
            if (!core_error || core_busy || core_done || core_error_code != code)
                $fatal(1, "FAIL expected error=%h got=%h busy=%b done=%b state=%0d",
                       code, core_error_code, core_busy, core_done, dut.state);
            if (read_active || write_active)
                $fatal(1, "FAIL error reported before transaction drained");
            if (no_access && (rd_commands != 0 || wr_commands != 0))
                $fatal(1, "FAIL invalid config issued a transaction");
            case_active = 0;
            error_tests = error_tests + 1;
            $display("ERROR_CASE_PASS test=%0d code=%h rd=%0d wr=%0d", error_tests, code, rd_commands, wr_commands);
        end
    endtask

    initial begin
        if ($value$plusargs("seed=%d", i)) rng = i;
        file_handle = $fopen("cases.txt", "r");
        if (!file_handle) $fatal(1, "FAIL cannot open cases.txt");
        rc = $fscanf(file_handle, "%d", count);
        if (rc != 1) $fatal(1, "FAIL cannot read case count");
        repeat (4) @(negedge clk);
        rst_n = 1;
        for (case_index = 0; case_index < count; case_index = case_index + 1) begin
            for (i = 0; i < 16; i = i + 1) begin
                rc = $fscanf(file_handle, "%h", cfg[i]);
                if (rc != 1) $fatal(1, "FAIL cannot read case=%0d word=%0d", case_index, i);
            end
            $readmemh($sformatf("in_%0d.hex", case_index), input_words, 0, (cfg[13] >> 4)-1);
            $readmemh($sformatf("expected_%0d.hex", case_index), expected_words, 0, (cfg[14] >> 4)-1);
            reset_counters();
            load_config();
            wait_done();
            case_active = 0;
            if (rd_commands > cfg[3][15:0] || rd_beats * 16 > cfg[13] ||
                wr_commands != cfg[5][15:0] || wr_beats * 16 != cfg[14] || responses != wr_commands)
                $fatal(1, "FAIL transaction counts case=%0d rd=%0d/%0d wr=%0d/%0d rsp=%0d",
                       case_index, rd_commands, rd_beats, wr_commands, wr_beats, responses);
            $display("CASE_PASS case=%0d cycles=%0d load=%0d horizontal=%0d vertical=%0d write=%0d wait_rsp=%0d check_schedule=%0d rd_commands=%0d rd_bytes=%0d wr_commands=%0d wr_bytes=%0d",
                     case_index, cycles, load_cycles, horizontal_cycles, vertical_cycles, write_cycles,
                     wait_cycles, check_cycles, rd_commands, rd_beats*16, wr_commands, wr_beats*16);
        end
        $fclose(file_handle);

        // Reload case 13 as a known long, multi-beat transaction for error and
        // drain tests; all negative checks reuse its valid descriptor.
        file_handle = $fopen("cases.txt", "r");
        rc = $fscanf(file_handle, "%d", i);
        for (integer t = 0; t <= 13; t = t + 1)
            for (integer j = 0; j < 16; j = j + 1) rc = $fscanf(file_handle, "%h", good_cfg[j]);
        $fclose(file_handle);
        case_index = 13; restore_cfg();
        $readmemh("in_13.hex", input_words, 0, (cfg[13] >> 4)-1);
        $readmemh("expected_13.hex", expected_words, 0, (cfg[14] >> 4)-1);
        error_tests = 0;

        // Each malformed record must fail before any service command.
        restore_cfg(); cfg[0] = 2; reset_counters(); load_config(); expect_error(16'h0002, 1);
        restore_cfg(); cfg[15] = 1; reset_counters(); load_config(); expect_error(16'h0002, 1);
        restore_cfg(); cfg[7][15:0] = 0; reset_counters(); load_config(); expect_error(16'h0001, 1);
        restore_cfg(); cfg[12] = 16; reset_counters(); load_config(); expect_error(16'h0005, 1);
        restore_cfg(); cfg[3] = {16'd32,16'd10}; cfg[4] = {16'd512,16'd512};
        cfg[5] = {16'd32,16'd10}; cfg[11] = 16384; cfg[12] = 16384;
        cfg[13] = 163840; cfg[14] = 163840;
        reset_counters(); load_config(); expect_error(16'h0005, 1); // legal row exceeds this instance
        restore_cfg(); cfg[1] = cfg[1] + 1; reset_counters(); load_config(); expect_error(16'h0003, 1);
        restore_cfg(); cfg[1] = 32'hfffffff0; reset_counters(); load_config(); expect_error(16'h0004, 1);
        restore_cfg(); cfg[2] = cfg[1]; reset_counters(); load_config(); expect_error(16'h0004, 1);
        restore_cfg(); cfg[5][15:0] = cfg[5][15:0] + 1; cfg[14] = cfg[5][15:0] * cfg[12];
        reset_counters(); load_config(); expect_error(16'h0001, 1);
        reset_counters(); @(negedge clk); core_start = 1; @(negedge clk); core_start = 0;
        expect_error(16'h0001, 1); // the previous start cleared all shadow-valid bits

        restore_cfg(); reset_counters(); inject_rd_error = 1; load_config(); expect_error(16'h0009, 0); inject_rd_error = 0;
        restore_cfg(); reset_counters(); inject_rd_tag = 1; load_config(); expect_error(16'h0009, 0); inject_rd_tag = 0;
        restore_cfg(); reset_counters(); inject_rd_last = 1; load_config(); expect_error(16'h0009, 0); inject_rd_last = 0;
        restore_cfg(); reset_counters(); inject_wr_error = 1; load_config(); expect_error(16'h0009, 0); inject_wr_error = 0;
        restore_cfg(); reset_counters(); inject_wr_aborted = 1; load_config(); expect_error(16'h0009, 0); inject_wr_aborted = 0;
        restore_cfg(); reset_counters(); inject_wr_tag = 1; load_config(); expect_error(16'h0009, 0); inject_wr_tag = 0;

        // Protocol fault while a read is in flight. The accepted row must drain.
        restore_cfg(); reset_counters(); load_config();
        while (!read_active) @(negedge clk);
        pool_cfg_we = 1; pool_cfg_addr = 0; pool_cfg_data = 32'hffffffff;
        @(negedge clk); pool_cfg_we = 0;
        expect_error(16'h0001, 0);

        restore_cfg(); reset_counters(); load_config();
        while (!read_active) @(negedge clk);
        core_start = 1; @(negedge clk); core_start = 0;
        expect_error(16'h0001, 0);

        restore_cfg(); reset_counters(); load_config();
        while (!read_active) @(negedge clk);
        core_abort = 1; @(negedge clk); core_abort = 0;
        expect_error(16'h000a, 0);

        // Abort with a write beat stalled: its data must remain stable and the
        // entire already-accepted write and response must be drained.
        restore_cfg(); reset_counters(); load_config();
        while (!write_active) @(negedge clk);
        force_wr_pause = 1;
        while (!wr_data_valid) @(negedge clk);
        core_abort = 1; @(negedge clk); core_abort = 0;
        repeat (3) @(negedge clk);
        force_wr_pause = 0;
        expect_error(16'h000a, 0);

        // A full clean run after faults proves recovery and row-tag isolation.
        restore_cfg(); reset_counters(); load_config(); wait_done(); case_active = 0;
        if (responses != cfg[5][15:0] || wr_beats * 16 != cfg[14])
            $fatal(1, "FAIL recovery transaction counts");
        @(negedge clk); core_soft_reset = 1;
        @(negedge clk); core_soft_reset = 0;
        if (core_done || core_error || core_busy) $fatal(1, "FAIL soft reset status");
        $display("TEST_PASS cases=%0d error_cases=%0d seed=20261029", count, error_tests);
        $finish;
    end
    initial begin
        #200000000;
        $fatal(1, "FAIL global watchdog case=%0d", case_index);
    end
endmodule
