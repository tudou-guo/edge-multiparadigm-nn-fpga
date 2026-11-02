// tb_p3_graph_shell.sv -- compile-manifest-driven P3 graph replay
`timescale 1ns / 1ps
`include "op_abi_defs.vh"
// v0.2 simulation-capacity contract.  The runner always supplies a resolvable
// tb_params.vh include directory: v0.2 compile directories carry
// generated/tb_params.vh, legacy compile directories receive a synthesized
// legacy default from the runner.  The ifndef fallbacks keep this TB
// numerically identical to the old 128KiB / 16KiB / 20M-cycle shell even if a
// consumer omits the include path or defines only a subset of the macros.
`include "tb_params.vh"

`ifndef TB_MEM_BYTES
`define TB_MEM_BYTES (8192*16)
`endif
`ifndef TB_MEM_WORDS
`define TB_MEM_WORDS (`TB_MEM_BYTES/16)
`endif
`ifndef TB_MAX_SNAPS
`define TB_MAX_SNAPS 128
`endif
`ifndef TB_SNAP_CHUNK
`define TB_SNAP_CHUNK 16384
`endif
`ifndef TB_WATCHDOG_CYCLES
`define TB_WATCHDOG_CYCLES 20000000
`endif

module tb_p3_graph_shell #(
    // Functional-replay MAC backend override (elaboration-time only).
    // Default 3 = the generated/OOC build (balanced PE + DSP48 macro models)
    // exactly as compiled.  Overriding to 2 elaborates the SAME balanced PE
    // schedule and sum tree with behavioral product leaves (PRODUCT_IMPL=0)
    // instead of the vendor DSP48 functional netlist: identical per-product
    // signed-int8 math, orders of magnitude faster in XSim.  Full-graph
    // functional replay only; OOC/IP always use the compiled MAC_IMPL=3 build.
    // Selected by the compile-time macro TB_MAC_IMPL_2 (xvlog -d; the value is
    // encoded in the macro name so the Vivado batch loader cannot mangle it) and
    // recorded in simulation_identity.json / summary.json as sim_define.
`ifdef TB_MAC_IMPL_2
    parameter integer DUT_MAC_IMPL = 2
`else
    parameter integer DUT_MAC_IMPL = 3
`endif
);
    localparam real CLK_PERIOD_NS = 5.0;
    // AXIL poll guard. It must never expire before the time-based watchdog:
    // each axil_read/write poll costs ~6-10 clocks, so MAX_WAIT is derived from
    // watchdog/5 with a 200000 floor (legacy 20M cycles -> 4M polls, which is
    // looser than the old fixed 200000 and cannot turn a passing old graph into
    // a failure). Large graphs therefore fail only on the explicit watchdog,
    // never on this guard.
    localparam integer MAX_WAIT =
        ((`TB_WATCHDOG_CYCLES/5) > 200000) ? (`TB_WATCHDOG_CYCLES/5) : 200000;
    localparam integer WATCHDOG_CYCLES = `TB_WATCHDOG_CYCLES;
    localparam integer MEM_WORDS = `TB_MEM_WORDS;
    localparam integer MEM_BYTES = `TB_MEM_BYTES;
    localparam integer MAX_SNAPS = `TB_MAX_SNAPS;
    localparam integer SNAP_CHUNK = `TB_SNAP_CHUNK;
    // Legacy heartbeat period 1000000 == 20000000/20.
    localparam integer HEARTBEAT_CYCLES =
        ((`TB_WATCHDOG_CYCLES/20) > 1000000) ? (`TB_WATCHDOG_CYCLES/20) : 1000000;
    // Modelled DDR window base; every memory address is computed in 64 bit
    // relative to this base, never by slicing addr[16:4].
    localparam logic [63:0] DDR_BASE = 64'h0000_0000_1000_0000;
    localparam logic [31:0] DDR_BASE32 = 32'h1000_0000;


    logic clk, rst_n;
    initial clk = 1'b0;
    always #(CLK_PERIOD_NS/2) clk = ~clk;

    logic [11:0] s_axil_awaddr; logic s_axil_awvalid; logic s_axil_awready;
    logic [31:0] s_axil_wdata; logic [3:0] s_axil_wstrb; logic s_axil_wvalid; logic s_axil_wready;
    logic [1:0] s_axil_bresp; logic s_axil_bvalid; logic s_axil_bready;
    logic [11:0] s_axil_araddr; logic s_axil_arvalid; logic s_axil_arready;
    logic [31:0] s_axil_rdata; logic [1:0] s_axil_rresp; logic s_axil_rvalid; logic s_axil_rready;

    logic [3:0] m_axi_awid; logic [31:0] m_axi_awaddr; logic [7:0] m_axi_awlen;
    logic [2:0] m_axi_awsize; logic [1:0] m_axi_awburst; logic m_axi_awlock;
    logic [3:0] m_axi_awcache; logic [2:0] m_axi_awprot; logic [3:0] m_axi_awqos;
    logic [3:0] m_axi_awregion; logic m_axi_awvalid; logic m_axi_awready;
    logic [127:0] m_axi_wdata; logic [15:0] m_axi_wstrb; logic m_axi_wlast;
    logic m_axi_wvalid; logic m_axi_wready; logic [3:0] m_axi_bid; logic [1:0] m_axi_bresp;
    logic m_axi_bvalid; logic m_axi_bready; logic [3:0] m_axi_arid; logic [31:0] m_axi_araddr;
    logic [7:0] m_axi_arlen; logic [2:0] m_axi_arsize; logic [1:0] m_axi_arburst;
    logic m_axi_arlock; logic [3:0] m_axi_arcache; logic [2:0] m_axi_arprot;
    logic [3:0] m_axi_arqos; logic [3:0] m_axi_arregion; logic m_axi_arvalid;
    logic m_axi_arready; logic [3:0] m_axi_rid; logic [127:0] m_axi_rdata;
    logic [1:0] m_axi_rresp; logic m_axi_rlast; logic m_axi_rvalid; logic m_axi_rready;

    npu_top #(.MAC_IMPL(DUT_MAC_IMPL)) dut (
        .clk(clk), .rst_n(rst_n),
        .s_axil_awaddr(s_axil_awaddr), .s_axil_awvalid(s_axil_awvalid), .s_axil_awready(s_axil_awready),
        .s_axil_wdata(s_axil_wdata), .s_axil_wstrb(s_axil_wstrb), .s_axil_wvalid(s_axil_wvalid), .s_axil_wready(s_axil_wready),
        .s_axil_bresp(s_axil_bresp), .s_axil_bvalid(s_axil_bvalid), .s_axil_bready(s_axil_bready),
        .s_axil_araddr(s_axil_araddr), .s_axil_arvalid(s_axil_arvalid), .s_axil_arready(s_axil_arready),
        .s_axil_rdata(s_axil_rdata), .s_axil_rresp(s_axil_rresp), .s_axil_rvalid(s_axil_rvalid), .s_axil_rready(s_axil_rready),
        .m_axi_awid(m_axi_awid), .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst), .m_axi_awlock(m_axi_awlock),
        .m_axi_awcache(m_axi_awcache), .m_axi_awprot(m_axi_awprot), .m_axi_awqos(m_axi_awqos),
        .m_axi_awregion(m_axi_awregion), .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb), .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready), .m_axi_bid(m_axi_bid),
        .m_axi_bresp(m_axi_bresp), .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid), .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst), .m_axi_arlock(m_axi_arlock),
        .m_axi_arcache(m_axi_arcache), .m_axi_arprot(m_axi_arprot), .m_axi_arqos(m_axi_arqos),
        .m_axi_arregion(m_axi_arregion), .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid), .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast), .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready)
    );

    logic [127:0] mem [0:MEM_WORDS-1];
    logic [7:0] init_chunk [0:SNAP_CHUNK-1];
    logic [7:0] snap_golden [0:SNAP_CHUNK-1];
    logic [7:0] snap_dut [0:SNAP_CHUNK-1];
    string snap_when [0:MAX_SNAPS-1];
    string snap_name [0:MAX_SNAPS-1];
    string snap_golden_path [0:MAX_SNAPS-1];
    integer snap_record [0:MAX_SNAPS-1];
    logic [31:0] snap_addr [0:MAX_SNAPS-1];
    integer snap_bytes [0:MAX_SNAPS-1];
    integer n_snap, snap_mismatch, snap_first, i, k, j, fd, nread, guard;
    integer diag_done, diag_j0, diag_before, diag_delta;
    string ddr_init, snap_desc, snap_root, result_dir, line, path;
    string a_when, a_name, a_golden;
    integer a_record, a_bytes;
    logic [31:0] a_addr;
    logic [31:0] rdata, status, error_code, error_info;
    logic [1:0] rresp;
    logic done_seen, error_seen, timeout_flag, start_seen, busy_seen;
    bit finished;    logic ar_busy; logic [7:0] ar_len_r, ar_index_r; logic [31:0] ar_addr_r;
    logic [31:0] ar_burst_count, r_beat_count;
    logic aw_busy; logic [7:0] aw_len_r, aw_index_r; logic [31:0] aw_addr_r; logic [3:0] aw_id_r;
    logic b_valid; logic [31:0] aw_burst_count, w_beat_count;
    logic [31:0] axi_rd_protocol_errors, axi_wr_protocol_errors;
    integer mwidx;

    // Observation only: sample accepted commands/events before NBA, then the
    // newly registered completion after NBA. No DUT force, delay or CSR added.
    // An interval is (start_edge, end_edge], total_cycles=end_edge-start_edge.
    // State/traffic counters may overlap; they are never summed into latency.
    `define PERF_SHELL dut.u_engine.u_shell
    `define PERF_ROUTER dut.u_engine.u_router
    `define PERF_CONV dut.u_engine.u_router.GEN_CONV.u_conv3x3_core
    `define PERF_AVG dut.u_engine.u_router.u_avgpool2x2_core
    // v0.2 batch 2: indices 0..14 are the frozen first-batch counters; 15..22 are
    // backend-agnostic ENGINE-CORE command/data counters observed on the npu_top
    // dbg_core_* ports (pre op_shell normalization). They are valid for every
    // engine (conv/pool/copy/resize/pool5/LUT) and are never summed with the
    // AXI-level beat counters (8..12).
    localparam integer PERF_N = 23;
    longint unsigned perf_edge = 0;
    longint unsigned perf_start [0:1];
    longint unsigned perf_count [0:1][0:PERF_N-1];
    longint unsigned perf_issue_edge [0:1023];
    integer perf_q_wr = 0, perf_q_rd = 0, perf_pending = 0;
    integer perf_scope, perf_metric, perf_fd = 0, perf_record = -1;
    integer perf_opcode = 0, perf_latency;
    bit perf_active [0:1];
    bit perf_tile_valid [0:1023];
    bit perf_current_tile_valid;
    integer perf_addr_wr = 0, perf_addr_rd = 0, perf_addr_pending = 0;
    integer perf_seed = 20261020;

    task automatic perf_begin(input integer s);
        integer m;
        begin
            if (perf_active[s]) $fatal(1, "PERF: overlapping interval scope=%0d", s);
            perf_start[s] = perf_edge;
            for (m=0; m<PERF_N; m=m+1) perf_count[s][m] = 0;
            perf_active[s] = 1;
        end
    endtask
    task automatic perf_end(input integer s);
        integer m;
        begin
            $fwrite(perf_fd, "%s,%0d,%0d,%0d,%0d,%0d",
                    s==0 ? "graph" : "layer", s==0 ? -1 : perf_record,
                    s==0 ? 0 : perf_opcode, perf_start[s], perf_edge,
                    perf_edge-perf_start[s]);
            for (m=0; m<PERF_N; m=m+1) $fwrite(perf_fd, ",%0d", perf_count[s][m]);
            $fwrite(perf_fd, "\n");
            $fflush(perf_fd);
            $display("P4_PERF: scope=%s record=%0d cycles=%0d issues=%0d valid_issues=%0d",
                     s==0 ? "graph" : "layer", s==0 ? -1 : perf_record,
                     perf_edge-perf_start[s], perf_count[s][2], perf_count[s][3]);
            perf_active[s] = 0;
        end
    endtask

    initial begin
        perf_active[0] = 0; perf_active[1] = 0;
    end
    always @(posedge clk) begin
        perf_edge = perf_edge + 1;
        if (rst_n) begin
            if (`PERF_SHELL.ctrl_cmd_pending[0] && `PERF_SHELL.status_ready &&
                (`PERF_SHELL.soft_reset_timer == 0)) perf_begin(0);
            if (`PERF_ROUTER.router_launch_q && !dut.u_engine.active_core_busy) begin
                perf_record = perf_record + 1;
                perf_opcode = `PERF_ROUTER.router_op_sel_q;
                perf_begin(1);
            end
            // v0.2 batch 2 fast functional profile: TB_LEAN_FUNC keeps the graph
            // and ENGINE-CORE counters (indices 8..12 and 15..22) but skips the
            // deep conv/avg internal sampling, which dominates TB wall clock on
            // very large graphs. Only owner-reported physical/synthesis claims
            // use the compiled build; this is a testbench observation switch.
`ifndef TB_LEAN_FUNC
            // Match scheduled tile metadata to the actual PE input by event
            // order. The address path has TWO register boundaries; a single
            // valid delay would mislabel the last issue of every pixel.
            if (`PERF_CONV.pe_addr_valid) begin
                if (perf_addr_pending >= 1024) $fatal(1, "PERF: address queue overflow");
                perf_tile_valid[perf_addr_wr] =
                    ((`PERF_CONV.calc_icb*8 + `PERF_CONV.calc_tn_offset) < `PERF_CONV.cfg_core_cin_q) &&
                    ((`PERF_CONV.calc_ocb*8 + `PERF_CONV.calc_tm_offset) < `PERF_CONV.cfg_core_cout_q);
                perf_addr_wr = (perf_addr_wr+1)%1024; perf_addr_pending = perf_addr_pending+1;
            end
            if (`PERF_CONV.pe_in_valid) begin
                if (perf_addr_pending == 0) $fatal(1, "PERF: issue without scheduled address");
                perf_current_tile_valid = perf_tile_valid[perf_addr_rd];
                perf_addr_rd = (perf_addr_rd+1)%1024; perf_addr_pending = perf_addr_pending-1;
                if (perf_pending >= 1024) $fatal(1, "PERF: issue queue overflow");
                perf_issue_edge[perf_q_wr] = perf_edge;
                perf_q_wr = (perf_q_wr+1)%1024; perf_pending = perf_pending+1;
            end
`endif
            for (perf_scope=0; perf_scope<2; perf_scope=perf_scope+1) begin
                if (perf_active[perf_scope] && (perf_edge > perf_start[perf_scope])) begin
`ifndef TB_LEAN_FUNC
                    if ((`PERF_ROUTER.is_conv && (`PERF_CONV.state >= 1) && (`PERF_CONV.state <= 8)) ||
                        (`PERF_ROUTER.is_avg && (`PERF_AVG.state >= 1) && (`PERF_AVG.state <= 3)))
                        perf_count[perf_scope][0] = perf_count[perf_scope][0]+1;
                    if (`PERF_ROUTER.is_conv && ((`PERF_CONV.state == 10) ||
                        (`PERF_CONV.state >= 20 && `PERF_CONV.state <= 23) || (`PERF_CONV.state == 28)))
                        perf_count[perf_scope][1] = perf_count[perf_scope][1]+1;
                    if (`PERF_CONV.pe_in_valid) begin
                        perf_count[perf_scope][2] = perf_count[perf_scope][2]+1;
                        if (perf_current_tile_valid) perf_count[perf_scope][3] = perf_count[perf_scope][3]+1;
                    end
                    if (`PERF_ROUTER.is_conv && (`PERF_CONV.state == 11) && `PERF_CONV.calc_issue_done)
                        perf_count[perf_scope][4] = perf_count[perf_scope][4]+1;
                    if ((`PERF_ROUTER.is_conv && ((`PERF_CONV.state == 12) || (`PERF_CONV.state == 17) ||
                        (`PERF_CONV.state == 18) || (`PERF_CONV.state == 19) ||
                        (`PERF_CONV.state >= 24 && `PERF_CONV.state <= 27) || (`PERF_CONV.state >= 29))) ||
                        (`PERF_ROUTER.is_avg && (`PERF_AVG.state >= 4) && (`PERF_AVG.state <= 16)))
                        perf_count[perf_scope][5] = perf_count[perf_scope][5]+1;
`endif
                    if (ar_busy && !m_axi_rvalid) perf_count[perf_scope][6] = perf_count[perf_scope][6]+1;
                    if ((m_axi_wvalid && !m_axi_wready) || (m_axi_awvalid && !m_axi_awready) ||
                        (m_axi_bvalid && !m_axi_bready)) perf_count[perf_scope][7] = perf_count[perf_scope][7]+1;
                    if (m_axi_rvalid && m_axi_rready) perf_count[perf_scope][8] = perf_count[perf_scope][8]+1;
                    if (m_axi_wvalid && m_axi_wready) perf_count[perf_scope][9] = perf_count[perf_scope][9]+1;
                    if (m_axi_bvalid && m_axi_bready) perf_count[perf_scope][10] = perf_count[perf_scope][10]+1;
                    if (m_axi_arvalid && m_axi_arready) perf_count[perf_scope][11] = perf_count[perf_scope][11]+1;
                    if (m_axi_awvalid && m_axi_awready) perf_count[perf_scope][12] = perf_count[perf_scope][12]+1;
`ifndef TB_LEAN_FUNC
                    if (`PERF_CONV.pe_out_valid) perf_count[perf_scope][13] = perf_count[perf_scope][13]+1;
`endif
                    if (m_axi_arvalid && !m_axi_arready) perf_count[perf_scope][14] = perf_count[perf_scope][14]+1;
                    // ---- generic engine-core counters (all engines) ----------
                    if (dut.dbg_core_busy)
                        perf_count[perf_scope][15] = perf_count[perf_scope][15]+1;
                    if (dut.dbg_core_rd_cmd_valid && dut.dbg_core_rd_cmd_ready) begin
                        perf_count[perf_scope][16] = perf_count[perf_scope][16]+1;
                        perf_count[perf_scope][17] = perf_count[perf_scope][17] + dut.dbg_core_rd_cmd_bytes;
                    end
                    if (dut.dbg_core_rd_data_valid && dut.dbg_core_rd_data_ready)
                        perf_count[perf_scope][18] = perf_count[perf_scope][18]+1;
                    if (dut.dbg_core_wr_cmd_valid && dut.dbg_core_wr_cmd_ready) begin
                        perf_count[perf_scope][19] = perf_count[perf_scope][19]+1;
                        perf_count[perf_scope][20] = perf_count[perf_scope][20] + dut.dbg_core_wr_cmd_bytes;
                    end
                    if (dut.dbg_core_wr_data_valid && dut.dbg_core_wr_data_ready)
                        perf_count[perf_scope][21] = perf_count[perf_scope][21]+1;
                    if (dut.dbg_core_wr_rsp_valid && dut.dbg_core_wr_rsp_ready)
                        perf_count[perf_scope][22] = perf_count[perf_scope][22]+1;
                end
            end
`ifndef TB_LEAN_FUNC
            #0.001;
            // Registered-output availability is post-NBA, 13 edges after the
            // accepted input. The core consumes that valid on the next edge;
            // pe_returns above counts consumption, not availability latency.
            if (`PERF_CONV.pe_out_valid) begin
                if (perf_pending == 0) $fatal(1, "PERF: return without issue");
                perf_latency = perf_edge - perf_issue_edge[perf_q_rd];
                if (perf_latency != `PERF_CONV.PE_TOKEN_LATENCY)
                    $fatal(1, "PERF: PE latency=%0d expected=%0d", perf_latency, `PERF_CONV.PE_TOKEN_LATENCY);
                perf_q_rd = (perf_q_rd+1)%1024; perf_pending = perf_pending-1;
            end
`endif
            if (perf_active[1] && dut.u_engine.active_core_done) perf_end(1);
            if (perf_active[0] && `PERF_SHELL.status_done) begin
                if (perf_active[1] || perf_pending != 0 || perf_addr_pending != 0) $fatal(1, "PERF: DONE before drain");
                perf_end(0);
                $fclose(perf_fd); perf_fd = 0;
            end
        end
    end
    `undef PERF_SHELL
    `undef PERF_ROUTER
    `undef PERF_CONV
    `undef PERF_AVG
    // 64-bit DDR window addressing helpers.  Reads outside the window return
    // zero and are counted as protocol errors at the accepting edge.
    logic [63:0] ar_word_base, ar_rd_idx;
    logic ar_bad;
    logic [63:0] aw_word_base;
    logic aw_bad;
    assign ar_rd_idx = ar_word_base + {56'b0, ar_index_r};
    assign m_axi_arready = !ar_busy;
    assign m_axi_rid = 4'h0;
    assign m_axi_rdata = (ar_bad || (ar_rd_idx >= {32'b0, MEM_WORDS})) ? 128'h0 : mem[ar_rd_idx];
    assign m_axi_rresp = 2'b00;
    assign m_axi_rlast = (ar_index_r == ar_len_r);
    assign m_axi_rvalid = ar_busy;
    assign m_axi_awready = !aw_busy && !b_valid;
    assign m_axi_wready = aw_busy && !b_valid;
    assign m_axi_bid = aw_id_r;
    assign m_axi_bresp = 2'b00;
    assign m_axi_bvalid = b_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ar_busy <= 0; ar_len_r <= 0; ar_index_r <= 0; ar_addr_r <= 0;
            ar_word_base <= 0; ar_bad <= 0;
            ar_burst_count <= 0; r_beat_count <= 0; axi_rd_protocol_errors <= 0;
        end else if (!ar_busy) begin
            if (m_axi_arvalid && m_axi_arready) begin
                ar_burst_count <= ar_burst_count + 1; ar_addr_r <= m_axi_araddr;
                ar_len_r <= m_axi_arlen; ar_index_r <= 0; ar_busy <= 1;
                ar_word_base <= ({32'b0, m_axi_araddr} - DDR_BASE) >> 4;
                ar_bad <= 1'b0;
                if ((m_axi_arid !== 4'h0) || (m_axi_arsize !== 3'd4) ||
                    (m_axi_arburst !== 2'b01) || (m_axi_arlen > 8'd15) ||
                    (m_axi_araddr[3:0] !== 4'b0) ||
                    ({32'b0, m_axi_araddr} < DDR_BASE) ||
                    ((({32'b0, m_axi_araddr} - DDR_BASE) +
                      (({56'b0, m_axi_arlen} + 64'd1) * 64'd16)) > {32'b0, MEM_BYTES})) begin
                    axi_rd_protocol_errors <= axi_rd_protocol_errors + 1;
                    ar_bad <= 1'b1;
                end
            end
        end else if (m_axi_rvalid && m_axi_rready) begin
            r_beat_count <= r_beat_count + 1;
            if (m_axi_rlast) ar_busy <= 0; else ar_index_r <= ar_index_r + 1;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aw_busy <= 0; aw_len_r <= 0; aw_index_r <= 0; aw_addr_r <= 0; aw_id_r <= 0;
            aw_word_base <= 0; aw_bad <= 0;
            b_valid <= 0; aw_burst_count <= 0; w_beat_count <= 0; axi_wr_protocol_errors <= 0;
        end else begin
            if (!aw_busy && !b_valid) begin
                if (m_axi_awvalid && m_axi_awready) begin
                    aw_burst_count <= aw_burst_count + 1; aw_addr_r <= m_axi_awaddr;
                    aw_len_r <= m_axi_awlen; aw_index_r <= 0; aw_id_r <= m_axi_awid; aw_busy <= 1;
                    aw_word_base <= ({32'b0, m_axi_awaddr} - DDR_BASE) >> 4;
                    aw_bad <= 1'b0;
                    if ((m_axi_awid !== 4'h1) || (m_axi_awsize !== 3'd4) ||
                        (m_axi_awburst !== 2'b01) || (m_axi_awlen > 8'd15) ||
                        (m_axi_awaddr[3:0] !== 4'b0) ||
                        ({32'b0, m_axi_awaddr} < DDR_BASE) ||
                        ((({32'b0, m_axi_awaddr} - DDR_BASE) +
                          (({56'b0, m_axi_awlen} + 64'd1) * 64'd16)) > {32'b0, MEM_BYTES})) begin
                        axi_wr_protocol_errors <= axi_wr_protocol_errors + 1;
                        aw_bad <= 1'b1;
                    end
                end
            end else if (aw_busy) begin
                if (m_axi_wvalid && m_axi_wready) begin
                    if (!aw_bad && ((aw_word_base + {56'b0, aw_index_r}) < {32'b0, MEM_WORDS}))
                        for (mwidx = 0; mwidx < 16; mwidx = mwidx + 1)
                            if (m_axi_wstrb[mwidx])
                                mem[aw_word_base + {56'b0, aw_index_r}][mwidx*8 +: 8] <= m_axi_wdata[mwidx*8 +: 8];
                    w_beat_count <= w_beat_count + 1;
                    if (m_axi_wstrb !== 16'hffff) axi_wr_protocol_errors <= axi_wr_protocol_errors + 1;
                    if (m_axi_wlast !== (aw_index_r == aw_len_r)) axi_wr_protocol_errors <= axi_wr_protocol_errors + 1;
                    if (aw_index_r == aw_len_r) begin aw_busy <= 0; b_valid <= 1; end
                    else aw_index_r <= aw_index_r + 1;
                end
            end
            if (b_valid && m_axi_bready) b_valid <= 0;
        end
    end

    // DDR model byte helper: 64-bit window arithmetic (no addr[16:4] slice).
    task automatic mem_write_byte(input logic [31:0] addr, input logic [7:0] val);
        logic [63:0] off;
        begin
            off = {32'b0, addr} - DDR_BASE;
            mem[off[63:4]][off[3:0]*8 +: 8] = val;
        end
    endtask

    // Stream the DDR init image at the window base in SNAP_CHUNK pieces. Writes
    // are 16-byte word granular (compiler images are 16-byte sized/aligned), so
    // a 540KiB image costs ~34k word stores instead of 540k task calls -- the
    // per-byte task form dominates xsim wall time on large spatial graphs.
    // The image length is whatever the compiler exported; it may extend into
    // the guard area beyond the legacy 128KiB (bounded by MEM_BYTES).
    task automatic load_ddr_init(input string init_path);
        integer fdi, nread2, total, kk, widx;
        logic [127:0] wdata;
        logic [63:0] woff;
        begin
            fdi = $fopen(init_path, "rb");
            if (fdi == 0) $fatal(1, "P2: cannot open DDR init %s", init_path);
            total = 0;
            nread2 = $fread(init_chunk, fdi, 0, SNAP_CHUNK);
            while (nread2 > 0) begin
                for (kk = 0; kk < nread2; kk = kk + 16) begin
                    if ((total + kk) >= MEM_BYTES) $fatal(1,
                        "P2: DDR init exceeds memory model: bytes=%0d limit=%0d",
                        total + kk + 1, MEM_BYTES);
                    wdata = 128'h0;
                    for (widx = 0; widx < 16; widx = widx + 1)
                        if ((kk + widx) < nread2) wdata[widx*8 +: 8] = init_chunk[kk + widx];
                    woff = (total + kk) >> 4;
                    mem[woff] = wdata;
                end
                total = total + nread2;
                if ((total % (SNAP_CHUNK*8)) == 0)
                    $display("P3_INFO: ddr_init load progress=%0d bytes", total);
                nread2 = $fread(init_chunk, fdi, 0, SNAP_CHUNK);
            end
            $fclose(fdi);
            if (total <= 0) $fatal(1, "P2: empty DDR init");
            $display("P3_INFO: ddr_init bytes=%0d", total);
        end
    endtask

    // Streamed snapshot compare/dump: golden is $fread in SNAP_CHUNK pieces,
    // compared byte by byte and written to snapshots/<name>.bin.  Any length is
    // allowed up to the modelled DDR window; per-snapshot PASS/FAIL reports the
    // mismatch count and the first mismatching byte offset.
    task automatic dump_snap(input integer idx);
        integer fd_in, fd_out, nread2, kk, b, chunk, off, n, first_off;
        logic [63:0] byte_off, woff;
        logic [127:0] snap_word;
        begin
            if ((snap_bytes[idx] < 0) || (snap_addr[idx][3:0] !== 4'b0) ||
                ({32'b0, snap_addr[idx]} < DDR_BASE) ||
                ((({32'b0, snap_addr[idx]} - DDR_BASE) + {32'b0, snap_bytes[idx]}) >
                 {32'b0, MEM_BYTES}))
                $fatal(1, "P2: snapshot %s range outside DDR model", snap_name[idx]);
            path = {snap_root, "/", snap_golden_path[idx]};
            fd_in = $fopen(path, "rb");
            if (fd_in == 0) $fatal(1, "P2: cannot open snapshot golden %s", path);
            path = {result_dir, "/snapshots/", snap_name[idx], ".bin"};
            fd_out = $fopen(path, "wb");
            if (fd_out == 0) $fatal(1, "P2: cannot create snapshot %s", path);
            n = 0; first_off = -1;
            $display("P3_SNAP_BEGIN name=%s bytes=%0d", snap_name[idx], snap_bytes[idx]);
            for (off = 0; off < snap_bytes[idx]; off = off + chunk) begin
                chunk = SNAP_CHUNK;
                if ((snap_bytes[idx] - off) < chunk) chunk = snap_bytes[idx] - off;
                nread2 = $fread(snap_golden, fd_in, 0, chunk);
                if (nread2 != chunk) $fatal(1, "P2: snapshot golden length mismatch %s", path);
                // 16-byte word path: one memory read + one buffered $fwrite of
                // 16 bytes per word keeps >16KiB snapshot compares affordable.
                for (kk = 0; kk + 16 <= chunk; kk = kk + 16) begin
                    woff = (({32'b0, snap_addr[idx]} - DDR_BASE) + off + kk) >> 4;
                    snap_word = mem[woff];
                    for (b = 0; b < 16; b = b + 1) begin
                        snap_dut[kk + b] = snap_word[b*8 +: 8];
                        if (snap_dut[kk + b] !== snap_golden[kk + b]) begin
                            n = n + 1;
                            if (first_off < 0) first_off = off + kk + b;
                        end
                    end
                    $fwrite(fd_out, "%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c",
                            snap_dut[kk], snap_dut[kk+1], snap_dut[kk+2], snap_dut[kk+3],
                            snap_dut[kk+4], snap_dut[kk+5], snap_dut[kk+6], snap_dut[kk+7],
                            snap_dut[kk+8], snap_dut[kk+9], snap_dut[kk+10], snap_dut[kk+11],
                            snap_dut[kk+12], snap_dut[kk+13], snap_dut[kk+14], snap_dut[kk+15]);
                end
                for (; kk < chunk; kk = kk + 1) begin
                    byte_off = ({32'b0, snap_addr[idx]} - DDR_BASE) + off + kk;
                    snap_dut[kk] = mem[byte_off[63:4]][byte_off[3:0]*8 +: 8];
                    $fwrite(fd_out, "%c", snap_dut[kk]);
                    if (snap_dut[kk] !== snap_golden[kk]) begin
                        n = n + 1;
                        if (first_off < 0) first_off = off + kk;
                    end
                end
            end
            $fclose(fd_in);
            $fclose(fd_out);
            if (n != 0) begin
                snap_mismatch = snap_mismatch + n;
                if (snap_first < 0) snap_first = idx;
                $display("P3_SNAP: FAIL name=%s bytes=%0d mismatch=%0d first_offset=%0d",
                         snap_name[idx], snap_bytes[idx], n, first_off);
            end else begin
                $display("P3_SNAP: PASS name=%s bytes=%0d", snap_name[idx], snap_bytes[idx]);
            end
        end
    endtask

    // Progress heartbeat: distinguishes "slow but advancing" from a stall.
    integer heartbeat_cycles;
    initial heartbeat_cycles = 0;
    always @(posedge clk) begin
        if (rst_n) begin
            heartbeat_cycles = heartbeat_cycles + 1;
            if (heartbeat_cycles % HEARTBEAT_CYCLES == 0)
                $display("P3_HEARTBEAT cycle=%0d time_ns=%0t rec_done=%0b rec_idx=%0d", heartbeat_cycles, $time, dut.u_engine.dbg_record_done, dut.u_engine.dbg_record_index);
        end
    end

`ifdef TB_CONV1_ROW0_DIAG
    // Board-mismatch diagnostic (verification-only, bounded): stop after record 0
    // (conv1) completes, compare its snapshot against the frozen golden, print an
    // explicit scope marker and finish.  NOT a graph-complete run.
    initial diag_done = 0;
    always @(posedge clk) begin
        if (rst_n && !diag_done && dut.u_engine.dbg_record_done &&
            (dut.u_engine.dbg_record_index == 0)) begin
            diag_done = 1;
            diag_j0 = -1;
            for (j = 0; j < n_snap; j = j + 1)
                if (snap_name[j] == "r00_conv1") diag_j0 = j;
            if (diag_j0 < 0) begin
                $display("M4_CONV1_DIAG_RESULT: FAIL scope=conv1_record0 reason=snapshot_not_found graph_complete=0");
                $finish(1);
            end
            diag_before = snap_mismatch;
            dump_snap(diag_j0);
            diag_delta = snap_mismatch - diag_before;
            if (diag_delta == 0)
                $display("M4_CONV1_DIAG_RESULT: PASS scope=conv1_record0 bytes=%0d mismatch=0 graph_complete=0", snap_bytes[diag_j0]);
            else
                $display("M4_CONV1_DIAG_RESULT: FAIL scope=conv1_record0 bytes=%0d mismatch=%0d graph_complete=0", snap_bytes[diag_j0], diag_delta);
            $display("M4_CONV1_DIAG_INFO: heartbeat_cycles=%0d snaps=%0d", heartbeat_cycles, n_snap);
            $finish(0);
        end
    end
`else
    always @(posedge clk) begin
        if (rst_n && dut.u_engine.dbg_record_done) begin
            for (j = 0; j < n_snap; j = j + 1)
                if (snap_when[j] == "record" && snap_record[j] == dut.u_engine.dbg_record_index)
                    dump_snap(j);
        end
    end
`endif

    task automatic axil_write(input logic [11:0] addr, input logic [31:0] data);
        logic aw_fired, w_fired; integer g;
        begin
            @(negedge clk);
            s_axil_awaddr = addr; s_axil_awvalid = 1; s_axil_wdata = data;
            s_axil_wstrb = 4'hf; s_axil_wvalid = 1; s_axil_bready = 1;
            aw_fired = 0; w_fired = 0; g = 0;
            while (!(aw_fired && w_fired) && g < MAX_WAIT) begin
                @(posedge clk);
                if (!aw_fired) aw_fired = s_axil_awvalid && s_axil_awready;
                if (!w_fired) w_fired = s_axil_wvalid && s_axil_wready;
                @(negedge clk);
                if (aw_fired) s_axil_awvalid = 0;
                if (w_fired) s_axil_wvalid = 0;
                g = g + 1;
            end
            if (!(aw_fired && w_fired)) $fatal(1, "P2: AXIL write timeout");
            g = 0;
            while (!s_axil_bvalid && g < MAX_WAIT) begin @(negedge clk); g = g + 1; end
            if (!s_axil_bvalid || s_axil_bresp != 2'b00) $fatal(1, "P2: AXIL write failed");
            @(posedge clk); @(negedge clk); s_axil_bready = 0;
        end
    endtask

    task automatic axil_read(input logic [11:0] addr, output logic [31:0] data,
                             output logic [1:0] resp);
        logic ar_fired; integer g;
        begin
            @(negedge clk);
            s_axil_araddr = addr; s_axil_arvalid = 1; s_axil_rready = 1;
            ar_fired = 0; g = 0;
            while (!ar_fired && g < MAX_WAIT) begin
                @(posedge clk); ar_fired = s_axil_arvalid && s_axil_arready;
                @(negedge clk); if (ar_fired) s_axil_arvalid = 0; g = g + 1;
            end
            if (!ar_fired) $fatal(1, "P2: AXIL read addr timeout");
            g = 0; while (!s_axil_rvalid && g < MAX_WAIT) begin @(negedge clk); g = g + 1; end
            if (!s_axil_rvalid) $fatal(1, "P2: AXIL read data timeout");
            data = s_axil_rdata; resp = s_axil_rresp;
            @(posedge clk); @(negedge clk); s_axil_rready = 0;
        end
    endtask
    initial begin
        for (i = 0; i < MEM_WORDS; i = i + 1) mem[i] = 128'b0;
        $display("P3_INFO: mem_model initialized words=%0d bytes=%0d", MEM_WORDS, MEM_BYTES);
        s_axil_awaddr = 0; s_axil_awvalid = 0; s_axil_wdata = 0; s_axil_wstrb = 0;
        s_axil_wvalid = 0; s_axil_bready = 0; s_axil_araddr = 0; s_axil_arvalid = 0;
        s_axil_rready = 0; rst_n = 0; finished = 0; done_seen = 0; error_seen = 0;
        timeout_flag = 0; start_seen = 0; busy_seen = 0; n_snap = 0;
        snap_mismatch = 0; snap_first = -1; error_code = 0; error_info = 0;

        if (!$value$plusargs("DDR_INIT=%s", ddr_init)) $fatal(1, "P2: missing +DDR_INIT");
        if (!$value$plusargs("SNAP_DESC=%s", snap_desc)) $fatal(1, "P2: missing +SNAP_DESC");
        if (!$value$plusargs("SNAP_ROOT=%s", snap_root)) $fatal(1, "P2: missing +SNAP_ROOT");
        if (!$value$plusargs("RESULT_DIR=%s", result_dir)) $fatal(1, "P2: missing +RESULT_DIR");
        if ($value$plusargs("PERF_SEED=%d", perf_seed)) begin end
        $display("P3_INFO: tb_params mem_bytes=%0d mem_words=%0d max_snaps=%0d snap_chunk=%0d watchdog_cycles=%0d",
                 MEM_BYTES, MEM_WORDS, MAX_SNAPS, SNAP_CHUNK, WATCHDOG_CYCLES);
        path = {result_dir, "/perf_raw.csv"};
        perf_fd = $fopen(path, "w");
        if (perf_fd == 0) $fatal(1, "PERF: cannot create raw counters");
        $fwrite(perf_fd, "scope,record_index,opcode,start_edge,end_edge,total_cycles,load_cycles,setup_cycles,pe_issues,valid_tile_issues,drain_cycles,output_cycles,read_wait_cycles,write_backpressure_cycles,rd_beats,wr_beats,wr_responses,ar_bursts,aw_bursts,pe_returns,ar_stall_cycles,core_busy_cycles,core_rd_cmd_count,core_rd_cmd_bytes,core_rd_data_beats,core_wr_cmd_count,core_wr_cmd_bytes,core_wr_data_beats,core_wr_rsp_count\n");
        $display("P4_PERF_CONFIG: schema=1 clock_ns=%0f seed=%0d stalls=none interval=(start,end]", CLK_PERIOD_NS, perf_seed);

        load_ddr_init(ddr_init);
`ifdef TB_DIAG_INPUT_XOR80
        // Counterfactual (diagnostic only): the deployment ddr_init stores the
        // graph input as raw u8 qu; the RTL activation reader converts storage to
        // signed with XOR 0x80.  Transform only the 3 real channel lanes of the
        // graph input (NHWC8: byte (pixel*8 + ch), ch<3) and leave every other
        // byte/instruction/config untouched.
        begin : diag_input_xor
            integer w, b;
            for (w = 0; w < (3276800/16); w = w + 1) begin
                for (b = 0; b < 16; b = b + 1) begin
                    if ((b % 8) < 3)
                        mem[w][b*8 +: 8] = mem[w][b*8 +: 8] ^ 8'h80;
                end
            end
            $display("M4_DIAG_INPUT_XOR80: transformed %0d input words", 3276800/16);
        end
`endif

        fd = $fopen(snap_desc, "r");
        if (fd == 0) $fatal(1, "P2: cannot open snapshot desc %s", snap_desc);
        while (!$feof(fd)) begin
            line = "";
            if ($fgets(line, fd) > 0) begin
                if ($sscanf(line, "snap %s %d %s %h %d %s", a_when, a_record, a_name,
                            a_addr, a_bytes, a_golden) == 6) begin
                    if (n_snap >= MAX_SNAPS) $fatal(1, "P2: snapshot desc exceeds MAX_SNAPS=%0d", MAX_SNAPS);
                    snap_when[n_snap] = a_when; snap_record[n_snap] = a_record;
                    snap_name[n_snap] = a_name; snap_addr[n_snap] = a_addr;
                    snap_bytes[n_snap] = a_bytes; snap_golden_path[n_snap] = a_golden;
                    n_snap = n_snap + 1;
                end
            end
        end
        $fclose(fd);
        if (n_snap == 0) $fatal(1, "P2: snapshot desc listed no snapshots");
        $display("P3_INFO: snapshots=%0d", n_snap);

        repeat (8) @(posedge clk);
        rst_n = 1;
        repeat (8) @(posedge clk);

        axil_read(12'h000, rdata, rresp);
        if (rdata !== 32'h4F504331) $fatal(1, "P2: OP_ID mismatch 0x%08x", rdata);
        axil_read(12'h004, rdata, rresp);
        if (rdata !== 32'h00010000) $fatal(1, "P2: ABI version mismatch 0x%08x", rdata);

        // The shell instruction watchdog defaults to 2^24 cycles (0x01000000,
        // OP_DEFAULT_TIMEOUT).  Large fusion graphs legitimately run for tens of
        // millions of cycles, so the host programs the documented timeout
        // register 0x01C with the TB graph budget before START.  This is runtime
        // ABI configuration, not an RTL change: the OOC/IP build keeps the
        // hardware default, and any graph shorter than 2^24 behaves as before.
        axil_write(12'h01C, `TB_WATCHDOG_CYCLES);
        $display("P3_INFO: instruction timeout configured 0x%08x (%0d cycles)",
                 `TB_WATCHDOG_CYCLES, `TB_WATCHDOG_CYCLES);

        axil_write(12'h008, 32'h00000001);
        start_seen = 1;
        guard = 0;
        while (!done_seen && !error_seen && guard < MAX_WAIT) begin
            axil_read(12'h00C, status, rresp);
            if (status[1]) busy_seen = 1;
            if (status[2]) done_seen = 1;
            if (status[3]) error_seen = 1;
            if (status[4]) timeout_flag = 1;
            guard = guard + 1;
        end
        axil_read(12'h010, error_code, rresp);
        axil_read(12'h014, error_info, rresp);
        if (!done_seen) begin
            $display("P3_RESULT: FAIL reason=no_done error=%0d timeout=%0d code=0x%08x",
                     error_seen, timeout_flag, error_code);
            $fatal(1, "P2: graph did not reach DONE");
        end
        $display("P3_INFO: DONE busy_seen=%0d status=0x%08x", busy_seen, status);

        for (j = 0; j < n_snap; j = j + 1)
            if (snap_when[j] == "final") dump_snap(j);

        path = {result_dir, "/tb_status.txt"};
        fd = $fopen(path, "w");
        if (fd == 0) $fatal(1, "P2: cannot create tb_status");
        $fwrite(fd, "snapshots=%0d\n", n_snap);
        $fwrite(fd, "snapshot_mismatch=%0d\n", snap_mismatch);
        $fwrite(fd, "done=%0d\n", done_seen);
        $fwrite(fd, "error=%0d\n", error_seen);
        $fwrite(fd, "timeout=%0d\n", timeout_flag);
        $fwrite(fd, "busy_seen=%0d\n", busy_seen);
        $fwrite(fd, "graph_cycles=%0d\n", perf_edge);
        $fwrite(fd, "start_seen=%0d\n", start_seen);
        $fwrite(fd, "status_word=0x%08x\n", status);
        $fwrite(fd, "error_code=0x%08x\n", error_code);
        $fwrite(fd, "error_info=0x%08x\n", error_info);
        $fwrite(fd, "axi_ar_bursts=%0d\n", ar_burst_count);
        $fwrite(fd, "axi_aw_bursts=%0d\n", aw_burst_count);
        $fwrite(fd, "axi_r_beats=%0d\n", r_beat_count);
        $fwrite(fd, "axi_w_beats=%0d\n", w_beat_count);
        $fwrite(fd, "axi_rd_protocol_errors=%0d\n", axi_rd_protocol_errors);
        $fwrite(fd, "axi_wr_protocol_errors=%0d\n", axi_wr_protocol_errors);
        $fclose(fd);
        finished = 1;

        if (snap_mismatch != 0 || !done_seen || error_seen || timeout_flag ||
            !busy_seen || !start_seen || axi_rd_protocol_errors != 0 ||
            axi_wr_protocol_errors != 0) begin
            $display("P3_RESULT: FAIL snap_mismatch=%0d done=%0d error=%0d timeout=%0d rd_err=%0d wr_err=%0d",
                     snap_mismatch, done_seen, error_seen, timeout_flag,
                     axi_rd_protocol_errors, axi_wr_protocol_errors);
            $fatal(1, "P2: graph replay failed");
        end
        $display("P3_RESULT: PASS snapshots=%0d mismatch=0 done=1 error=0 release_pass=false board_gate_pass=false", n_snap);
        $finish;
    end

    initial begin
        #(CLK_PERIOD_NS * WATCHDOG_CYCLES);
        if (!finished) begin
            $display("P3_RESULT: FAIL reason=watchdog");
            $fatal(1, "P2: watchdog expired");
        end
    end
endmodule
