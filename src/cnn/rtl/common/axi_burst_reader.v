// -----------------------------------------------------------------------------
// axi_burst_reader.v
//
// ABI 1.0 AXI4-Full read engine.
//
// Accepts one normalized read command from op_core, splits it into legal INCR
// bursts (<= 16 beats, never crossing a 4 KB boundary) and streams the
// returned 128-bit beats back to the core in order.
//
// T-OP2a scope:
//   - single outstanding burst at a time (well inside the ABI limit of 8);
//   - 16-byte aligned addresses and 16-byte multiple lengths, as mandated by
//     docs/op_abi.md 5.2. Anything else is rejected with an error beat and
//     produces NO AXI traffic;
//   - RRESP != OKAY, RID mismatch or RLAST mismatch aborts the command at the
//     end of the current burst and reports the ABI error code on the data
//     beats;
//   - `abort` stops at the next burst boundary (see the note at the bottom).
//
// TODO(T-OP2b): outstanding-burst pipelining (>1) is not implemented; the
// current single-outstanding design already satisfies the ABI and is what G0
// verifies.
// -----------------------------------------------------------------------------

`timescale 1ns / 1ps
`include "op_abi_defs.vh"

module axi_burst_reader #(
    parameter integer AXI_ADDR_WIDTH  = 32,
    parameter integer AXI_DATA_WIDTH  = 128,
    parameter integer AXI_ID_WIDTH    = 4,
    parameter integer MAX_BURST_BEATS = 16,
    parameter [AXI_ID_WIDTH-1:0] M_ARID_VALUE = `OP_AXI_RID_READ
) (
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        abort,
    input  wire                        stat_clear,

    // Normalized read command (from op_core).
    input  wire                        cmd_valid,
    output wire                        cmd_ready,
    input  wire [31:0]                 cmd_addr_byte,
    input  wire [15:0]                 cmd_bytes,
    input  wire [7:0]                  cmd_tag,

    // Normalized read data (to op_core).
    output wire                        data_valid,
    input  wire                        data_ready,
    output wire [AXI_DATA_WIDTH-1:0]   data,
    output wire                        data_last,
    output wire                        data_error,
    output wire [15:0]                 data_error_code,
    output wire [7:0]                  data_tag,

    // AXI4-Full read master.
    output wire [AXI_ID_WIDTH-1:0]     m_arid,
    output wire [AXI_ADDR_WIDTH-1:0]   m_araddr,
    output wire [7:0]                  m_arlen,
    output wire [2:0]                  m_arsize,
    output wire [1:0]                  m_arburst,
    output wire                        m_arlock,
    output wire [3:0]                  m_arcache,
    output wire [2:0]                  m_arprot,
    output wire [3:0]                  m_arqos,
    output wire [3:0]                  m_arregion,
    output wire                        m_arvalid,
    input  wire                        m_arready,
    input  wire [AXI_ID_WIDTH-1:0]     m_rid,
    input  wire [AXI_DATA_WIDTH-1:0]   m_rdata,
    input  wire [1:0]                  m_rresp,
    input  wire                        m_rlast,
    input  wire                        m_rvalid,
    output wire                        m_rready,

    // Debug counters consumed by op_shell (ABI rule 5.2.10).
    output reg  [31:0]                 stat_burst_count,
    output reg  [31:0]                 stat_last_addr,
    output reg  [1:0]                  stat_last_resp,
    output wire                        idle
);

    localparam integer BEAT_BYTES = AXI_DATA_WIDTH / 8;

    localparam [2:0] R_IDLE = 3'd0;
    localparam [2:0] R_AR   = 3'd1;
    localparam [2:0] R_DATA = 3'd2;
    localparam [2:0] R_NEXT = 3'd3;
    localparam [2:0] R_FAIL = 3'd4;
    localparam [2:0] R_ABORT = 3'd5;

    reg [2:0]  state;
    reg [31:0] cur_addr;
    reg [31:0] remaining;
    reg [31:0] beats_left;
    reg [31:0] burst_bytes_r;
    reg [7:0]  tag_r;
    reg        error_seen;
    reg [15:0] error_code_r;
    reg        command_active;

    assign idle = (state == R_IDLE);

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

    // --- AXI4-Full read channel ---------------------------------------------
    assign m_arid     = M_ARID_VALUE;
    assign m_araddr   = cur_addr;
    assign m_arlen    = split_burst_beats[7:0] - 8'd1;
    assign m_arsize   = 3'd4;         // 2^4 = 16 bytes per beat
    assign m_arburst  = 2'b01;        // INCR
    assign m_arlock   = 1'b0;
    assign m_arcache  = 4'b0011;      // normal, non-cacheable bufferable
    assign m_arprot   = 3'b000;
    assign m_arqos    = 4'b0000;
    assign m_arregion = 4'b0000;
    assign m_arvalid  = (state == R_AR);
    assign m_rready   = (state == R_DATA) && data_ready;

    // --- Response / protocol checking ---------------------------------------
    wire beat_ok          = (m_rresp == `OP_AXI_RESP_OKAY);
    wire beat_id_ok       = (m_rid == M_ARID_VALUE);
    wire beat_is_bad      = (state == R_DATA) && m_rvalid && (!beat_ok || !beat_id_ok);
    wire burst_ends_now   = (beats_left == 32'd1);
    wire rlast_mismatch   = (state == R_DATA) && m_rvalid && (m_rlast != burst_ends_now);
    wire stop_after_burst = error_seen || beat_is_bad || abort;
    wire end_of_command   = (remaining <= BEAT_BYTES);
    wire [31:0] beat_bytes_used = (remaining > BEAT_BYTES) ? BEAT_BYTES : remaining;
    wire [31:0] beats_after     = beats_left - 32'd1;
    wire [31:0] remaining_after = remaining - beat_bytes_used;
    wire r_fire = (state == R_DATA) && m_rvalid && m_rready;

    // --- Normalized data output ---------------------------------------------
    assign cmd_ready = (state == R_IDLE) && !abort;

    assign data_valid      = ((state == R_DATA) && m_rvalid) ||
                             (state == R_FAIL) ||
                             ((state == R_ABORT) && command_active);
    assign data            = (state == R_FAIL) ? {AXI_DATA_WIDTH{1'b0}} : m_rdata;
    assign data_last       = (state == R_FAIL) ||
                             ((state == R_ABORT) && command_active) ||
                             ((state == R_DATA) &&
                              (end_of_command || (stop_after_burst && m_rlast)));
    assign data_error      = (state == R_FAIL) ||
                             ((state == R_DATA) && m_rvalid &&
                              (error_seen || beat_is_bad || rlast_mismatch));
    assign data_error_code = (state == R_FAIL || error_seen || beat_is_bad || rlast_mismatch)
                             ? error_code_r : `OP_E_NONE;
    assign data_tag        = tag_r;

    // --- FSM -----------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= R_IDLE;
            cur_addr         <= 32'b0;
            remaining        <= 32'b0;
            beats_left       <= 32'b0;
            burst_bytes_r    <= 32'b0;
            tag_r            <= 8'b0;
            error_seen       <= 1'b0;
            error_code_r     <= `OP_E_NONE;
            command_active   <= 1'b0;
            stat_burst_count <= 32'b0;
            stat_last_addr   <= 32'b0;
            stat_last_resp   <= 2'b0;
        end else if (stat_clear) begin
            stat_burst_count <= 32'b0;
            stat_last_addr   <= 32'b0;
            stat_last_resp   <= 2'b0;
        end else begin
            case (state)
                R_IDLE: begin
                    if (abort && command_active) begin
                        state <= R_ABORT;
                    end else if (cmd_valid && cmd_ready) begin
                        cur_addr     <= cmd_addr_byte;
                        remaining    <= {16'b0, cmd_bytes};
                        tag_r        <= cmd_tag;
                        error_seen   <= 1'b0;
                        error_code_r <= `OP_E_NONE;
                        command_active <= 1'b1;
                        if (cmd_addr_byte == 32'b0) begin
                            error_seen   <= 1'b1;
                            error_code_r <= `OP_E_ADDR_RANGE;
                            state        <= R_FAIL;
                        end else if ((cmd_addr_byte[3:0] != 4'b0) ||
                                     (cmd_bytes == 16'd0) ||
                                     (cmd_bytes[3:0] != 4'b0)) begin
                            // ABI 1.0 forbids unaligned or partial-beat transfers.
                            error_seen   <= 1'b1;
                            error_code_r <= `OP_E_ADDR_ALIGN;
                            state        <= R_FAIL;
                        end else if ({1'b0, cmd_addr_byte} + {17'b0, cmd_bytes} >
                                     33'h1_0000_0000) begin
                            error_seen   <= 1'b1;
                            error_code_r <= `OP_E_ADDR_RANGE;
                            state        <= R_FAIL;
                        end else begin
                            state <= R_AR;
                        end
                    end
                end

                R_AR: begin
                    if (m_arready) begin
                        beats_left       <= split_burst_beats;
                        burst_bytes_r    <= split_burst_bytes;
                        stat_burst_count <= stat_burst_count + 32'd1;
                        stat_last_addr   <= cur_addr;
                        state            <= R_DATA;
                    end else if (abort) begin
                        // AR was never accepted; if a command was accepted, give the
                        // core one terminal beat so it can leave its wait state.
                        state <= command_active ? R_ABORT : R_IDLE;
                    end
                end

                R_DATA: begin
                    stat_last_resp <= m_rresp;
                    if (r_fire) begin
                        beats_left <= beats_after;
                        remaining  <= remaining_after;
                        if (burst_ends_now) begin
                            if (stop_after_burst || (remaining_after == 32'b0)) begin
                                state          <= R_IDLE;
                                command_active <= 1'b0;
                            end else begin
                                cur_addr <= cur_addr + burst_bytes_r;
                                state    <= R_NEXT;
                            end
                        end
                    end
                    if (beat_is_bad) begin
                        error_seen   <= 1'b1;
                        error_code_r <= beat_id_ok ? `OP_E_AXI_RESP : `OP_E_AXI_PROTOCOL;
                    end else if (rlast_mismatch) begin
                        error_seen   <= 1'b1;
                        error_code_r <= `OP_E_AXI_PROTOCOL;
                    end
                end

                R_NEXT: begin
                    state <= abort ? R_ABORT : R_AR;
                end

                R_FAIL: begin
                    // Single synthetic error beat; no AXI traffic was issued.
                    if (data_ready) begin
                        command_active <= 1'b0;
                        state          <= R_IDLE;
                    end
                end

                R_ABORT: begin
                    // One synthetic terminal beat for a command that was accepted
                    // but cancelled before all of its beats could be delivered.
                    if (data_ready) begin
                        command_active <= 1'b0;
                        state          <= R_IDLE;
                    end
                end

                default: state <= R_IDLE;
            endcase
        end
    end

    // Abort note (docs/op_abi.md 3.4 and 9.4):
    //   - R_IDLE / R_AR : the request is dropped at the next clock edge.
    //   - R_DATA        : the accepted burst is completed and delivered to the
    //     core, then no further burst is started. The in-flight burst is never
    //     abandoned, because an accepted AR must always be matched by its R
    //     beats or the interconnect can deadlock. The paired write engine
    //     relies on this: it must be able to finish its own in-flight burst.
    //   - Soft reset does not zero the debug counters; only hard rst_n does.

    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_split = |split_bytes_to_4k | |split_beats_to_4k;
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
