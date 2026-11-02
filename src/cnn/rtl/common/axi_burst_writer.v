// -----------------------------------------------------------------------------
// axi_burst_writer.v
//
// ABI 1.0 AXI4-Full write engine.
//
// Accepts one normalized write command plus its 128-bit data beats from
// op_core, splits the transfer into legal INCR bursts (<= 16 beats, never
// crossing a 4 KB boundary) and reports one response when the command is done.
//
// T-OP2a scope:
//   - single outstanding burst at a time (well inside the ABI limit of 8);
//   - WSTRB is always all-ones and lengths are 16-byte multiples, exactly as
//     frozen in docs/op_abi.md 5.2 (ABI 1.0 uses no partial-byte writes);
//   - BRESP != OKAY, BID mismatch or a data_last mismatch aborts the command
//     and is reported through rsp_error / rsp_error_code;
//   - `abort` is honoured at burst boundaries. Because an accepted AW must be
//     matched by its W beats and its B response, the engine always completes
//     the burst it already started, then stops and reports rsp_aborted.
//
// The engine relies on the normalized-interface contract from docs/op_abi.md
// 2.3: the producer must hold wr_data_valid (and the beat payload) stable until
// wr_data_ready, which is what keeps m_axi_wvalid stable for the interconnect.
//
// TODO(T-OP2b): outstanding-burst pipelining (>1) is not implemented.
// -----------------------------------------------------------------------------

`timescale 1ns / 1ps
`include "op_abi_defs.vh"

module axi_burst_writer #(
    parameter integer AXI_ADDR_WIDTH  = 32,
    parameter integer AXI_DATA_WIDTH  = 128,
    parameter integer AXI_ID_WIDTH    = 4,
    parameter integer MAX_BURST_BEATS = 16,
    parameter [AXI_ID_WIDTH-1:0] M_AWID_VALUE = `OP_AXI_ID_WRITE
) (
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        abort,
    input  wire                        stat_clear,

    // Normalized write command (from op_core).
    input  wire                        cmd_valid,
    output wire                        cmd_ready,
    input  wire [31:0]                 cmd_addr_byte,
    input  wire [15:0]                 cmd_bytes,
    input  wire [7:0]                  cmd_tag,

    // Normalized write data (from op_core).
    input  wire                        data_valid,
    output wire                        data_ready,
    input  wire [AXI_DATA_WIDTH-1:0]   data,
    input  wire                        data_last,
    input  wire [7:0]                  data_tag,

    // Normalized write response (to op_core).
    output reg                         rsp_valid,
    input  wire                        rsp_ready,
    output wire                        rsp_error,
    output wire                        rsp_aborted,
    output wire [15:0]                 rsp_error_code,
    output reg  [7:0]                  rsp_tag,

    // AXI4-Full write master.
    output wire [AXI_ID_WIDTH-1:0]     m_awid,
    output wire [AXI_ADDR_WIDTH-1:0]   m_awaddr,
    output wire [7:0]                  m_awlen,
    output wire [2:0]                  m_awsize,
    output wire [1:0]                  m_awburst,
    output wire                        m_awlock,
    output wire [3:0]                  m_awcache,
    output wire [2:0]                  m_awprot,
    output wire [3:0]                  m_awqos,
    output wire [3:0]                  m_awregion,
    output wire                        m_awvalid,
    input  wire                        m_awready,
    output wire [AXI_DATA_WIDTH-1:0]   m_wdata,
    output wire [AXI_DATA_WIDTH/8-1:0] m_wstrb,
    output wire                        m_wlast,
    output wire                        m_wvalid,
    input  wire                        m_wready,
    input  wire [AXI_ID_WIDTH-1:0]     m_bid,
    input  wire [1:0]                  m_bresp,
    input  wire                        m_bvalid,
    output wire                        m_bready,

    // Debug counters consumed by op_shell (ABI rule 5.2.10).
    output reg  [31:0]                 stat_burst_count,
    output reg  [31:0]                 stat_last_addr,
    output reg  [1:0]                  stat_last_resp,
    output wire                        idle
);

    localparam integer BEAT_BYTES = AXI_DATA_WIDTH / 8;

    localparam [2:0] W_IDLE = 3'd0;
    localparam [2:0] W_AW   = 3'd1;
    localparam [2:0] W_DATA = 3'd2;
    localparam [2:0] W_B    = 3'd3;
    localparam [2:0] W_NEXT = 3'd4;
    localparam [2:0] W_RSP  = 3'd5;

    reg [2:0]  state;
    reg [31:0] cur_addr;
    reg [31:0] remaining;
    reg [31:0] beats_left;
    reg [31:0] burst_bytes_r;
    reg [7:0]  tag_r;
    reg        error_seen;
    reg        aborted_r;
    reg [15:0] error_code_r;

    assign idle = (state == W_IDLE);
    assign cmd_ready = (state == W_IDLE);

    // --- Burst sizing --------------------------------------------------------
    wire [31:0] split_burst_beats;
    wire [31:0] split_burst_bytes;
    wire [31:0] split_beats_to_4k;
    wire [31:0] split_bytes_to_4k;

    axi4_4kb_splitter #(
        .AXI_ADDR_WIDTH  (AXI_ADDR_WIDTH),
        .BEAT_BYTES      (BEAT_BYTES),
        .MAX_BURST_BEATS (MAX_BURST_BEATS)
    ) u_split (
        .addr            (cur_addr),
        .remaining_bytes (remaining),
        .bytes_to_4k     (split_bytes_to_4k),
        .beats_to_4k     (split_beats_to_4k),
        .burst_beats     (split_burst_beats),
        .burst_bytes     (split_burst_bytes)
    );

    // --- AXI4-Full write channels -------------------------------------------
    assign m_awid     = M_AWID_VALUE;
    assign m_awaddr   = cur_addr;
    assign m_awlen    = split_burst_beats[7:0] - 8'd1;
    assign m_awsize   = 3'd4;         // 2^4 = 16 bytes per beat
    assign m_awburst  = 2'b01;        // INCR
    assign m_awlock   = 1'b0;
    assign m_awcache  = 4'b0011;      // normal, non-cacheable bufferable
    assign m_awprot   = 3'b000;
    assign m_awqos    = 4'b0000;
    assign m_awregion = 4'b0000;
    assign m_awvalid  = (state == W_AW);

    assign m_wvalid   = (state == W_DATA) && data_valid;
    assign m_wdata    = data;
    assign m_wstrb    = {AXI_DATA_WIDTH/8{1'b1}};   // ABI 1.0: full beats only
    assign m_wlast    = (state == W_DATA) && (beats_left == 32'd1);
    assign m_bready   = (state == W_B);

    assign data_ready = (state == W_DATA) && m_wready;

    wire w_fire = (state == W_DATA) && m_wvalid && m_wready;

    wire [31:0] beat_bytes_used = (remaining > BEAT_BYTES) ? BEAT_BYTES : remaining;
    wire [31:0] remaining_after = remaining - beat_bytes_used;
    wire        burst_ends_now  = (beats_left == 32'd1);
    wire        beat_should_be_last = (remaining_after == 32'b0);
    wire        data_last_bad   = w_fire && (data_last !== beat_should_be_last);
    wire        beat_resp_bad   = (m_bresp != `OP_AXI_RESP_OKAY);
    wire        beat_id_bad     = (m_bid != M_AWID_VALUE);

    assign rsp_error      = error_seen;
    assign rsp_aborted    = aborted_r;
    assign rsp_error_code = error_seen ? error_code_r : `OP_E_NONE;

    // --- FSM -----------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= W_IDLE;
            cur_addr         <= 32'b0;
            remaining        <= 32'b0;
            beats_left       <= 32'b0;
            burst_bytes_r    <= 32'b0;
            tag_r            <= 8'b0;
            error_seen       <= 1'b0;
            aborted_r        <= 1'b0;
            error_code_r     <= `OP_E_NONE;
            rsp_valid        <= 1'b0;
            rsp_tag          <= 8'b0;
            stat_burst_count <= 32'b0;
            stat_last_addr   <= 32'b0;
            stat_last_resp   <= 2'b0;
        end else if (stat_clear) begin
            stat_burst_count <= 32'b0;
            stat_last_addr   <= 32'b0;
            stat_last_resp   <= 2'b0;
        end else begin
            case (state)
                W_IDLE: begin
                    rsp_valid <= 1'b0;
                    if (cmd_valid && cmd_ready) begin
                        cur_addr     <= cmd_addr_byte;
                        remaining    <= {16'b0, cmd_bytes};
                        tag_r        <= cmd_tag;
                        rsp_tag      <= cmd_tag;
                        error_seen   <= 1'b0;
                        aborted_r    <= 1'b0;
                        error_code_r <= `OP_E_NONE;
                        if (cmd_addr_byte == 32'b0) begin
                            error_seen   <= 1'b1;
                            error_code_r <= `OP_E_ADDR_RANGE;
                            state        <= W_RSP;
                        end else if ((cmd_addr_byte[3:0] != 4'b0) ||
                                     (cmd_bytes == 16'd0) ||
                                     (cmd_bytes[3:0] != 4'b0)) begin
                            error_seen   <= 1'b1;
                            error_code_r <= `OP_E_ADDR_ALIGN;
                            state        <= W_RSP;
                        end else if ({1'b0, cmd_addr_byte} + {17'b0, cmd_bytes} >
                                     33'h1_0000_0000) begin
                            error_seen   <= 1'b1;
                            error_code_r <= `OP_E_ADDR_RANGE;
                            state        <= W_RSP;
                        end else begin
                            state <= W_AW;
                        end
                    end
                end

                W_AW: begin
                    if (m_awready) begin
                        beats_left       <= split_burst_beats;
                        burst_bytes_r    <= split_burst_bytes;
                        stat_burst_count <= stat_burst_count + 32'd1;
                        stat_last_addr   <= cur_addr;
                        state            <= W_DATA;
                    end else if (abort) begin
                        // AW was never accepted, so dropping awvalid is legal.
                        aborted_r <= 1'b1;
                        state     <= W_RSP;
                    end
                end

                W_DATA: begin
                    if (w_fire) begin
                        beats_left <= beats_left - 32'd1;
                        remaining  <= remaining_after;
                        if (data_last_bad) begin
                            error_seen   <= 1'b1;
                            error_code_r <= `OP_E_AXI_PROTOCOL;
                        end
                        if (burst_ends_now) begin
                            // The burst was issued, so its B response must still
                            // be collected before the command can end.
                            if (abort && (remaining_after != 32'b0)) begin
                                aborted_r <= 1'b1;
                            end
                            state <= W_B;
                        end
                    end
                end

                W_B: begin
                    if (m_bvalid) begin
                        stat_last_resp <= m_bresp;
                        if (beat_resp_bad) begin
                            error_seen   <= 1'b1;
                            error_code_r <= `OP_E_AXI_RESP;
                            state        <= W_RSP;
                        end else if (beat_id_bad) begin
                            error_seen   <= 1'b1;
                            error_code_r <= `OP_E_AXI_PROTOCOL;
                            state        <= W_RSP;
                        end else if (aborted_r) begin
                            state <= W_RSP;
                        end else if (remaining != 32'b0) begin
                            state <= W_NEXT;
                        end else begin
                            state <= W_RSP;
                        end
                    end
                end

                W_NEXT: begin
                    if (abort) begin
                        aborted_r <= 1'b1;
                        state     <= W_RSP;
                    end else begin
                        cur_addr <= cur_addr + burst_bytes_r;
                        state    <= W_AW;
                    end
                end

                W_RSP: begin
                    if (!rsp_valid) begin
                        rsp_valid <= 1'b1;
                    end else if (rsp_ready) begin
                        rsp_valid <= 1'b0;
                        state     <= W_IDLE;
                    end
                end

                default: state <= W_IDLE;
            endcase
        end
    end

    // Abort note (docs/op_abi.md 3.4 and 9.4):
    //   - W_IDLE / W_AW : the request is dropped at the next clock edge, no AXI
    //     traffic has been issued.
    //   - W_DATA / W_B  : the accepted AW burst is always completed (W beats and
    //     B response) so the interconnect is never left mid-burst; the command
    //     then ends with rsp_error = 0 and rsp_aborted = 1.
    //   - rsp_aborted is NOT an error: the caller (op_shell) already owns the
    //     abort reason (timeout, protocol fault, soft reset) and MUST NOT turn
    //     rsp_aborted into a second sticky error.

    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_split = |split_bytes_to_4k | |split_beats_to_4k;
    wire unused_data_tag = |data_tag;
    /* verilator lint_on UNUSEDSIGNAL */

endmodule

