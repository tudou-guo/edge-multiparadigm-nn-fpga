// Platform wrapper: our clock/reset, AXI-Lite CSR, DDR transport hooks,
// PL backbone (SPS + shared block array + head reduce), PS Head handoff.
// Upstream TOP.v/TOP_Transformer.v used 32-bit addrs as MODEL-WINDOW offsets
// and tied intr_clr/done inputs to 0; here they are lifecycle-driven.
// CSR map (offsets from AXI-Lite base, 32-bit):
//   0x00 ID (0xSPIK0001)   0x04 VERSION   0x08 CONTROL[start,soft_rst]
//   0x0C STATUS[busy,done,error]  0x10 ERROR  0x14 IN_BASE_LO/HI window ptr
//   0x1C OUT_BASE  0x20 W_BASE (block0)  0x24 W_STRIDE  0x28 DESC_PTR
//   0x2C TIMEOUT_CYC  0x30 ACK  0x34 SAMPLE_TAG
// Overflow rule: model-window offsets + CPU/PL base, range-checked in
// runtime; never truncate >4GiB host addresses into 32-bit PL fields.
module spikformer_platform_wrapper #(
  parameter [31:0] ID = 32'h5350494B,
  parameter [31:0] VERSION = 32'h20261029,
  parameter int T = 4,
  parameter int DEPTH = 1,
  parameter int CH = 384,
  parameter int HEADS = 12,
  parameter int HEAD_DIM = 32,
  parameter int HIDDEN = 1536,
  parameter int TOKENS = 64,
  parameter int SPIKE_BITS = 2,
  parameter int RES_BITS = 4,
  parameter int SUM_W = 6
)(
  input  logic s_clk,
  input  logic s_rst_n,
  // AXI-Lite slave (simplified: single-beat, ready-always)
  input  logic [7:0]  s_axil_awaddr,
  input  logic [31:0] s_axil_wdata,
  input  logic        s_axil_wvalid,
  output logic        s_axil_wready,
  input  logic [7:0]  s_axil_araddr,
  input  logic        s_axil_arvalid,
  output logic [31:0] s_axil_rdata,
  output logic        s_axil_rvalid,
  // DDR transport hooks (reuse dl_v1 common burst reader/writer)
  output logic o_ddr_req,
  input  logic i_ddr_gnt,
  output logic [63:0] o_ddr_addr,
  // PL<->PS head handoff
  output logic [31:0] o_head_sums [CH],
  output logic o_head_valid,
  output logic o_busy,
  output logic o_done,
  output logic o_error
);
  logic s_rst;
  assign s_rst = !s_rst_n;
  // --- CSR ---
  logic start_bit, soft_rst, ack_bit;
  logic [63:0] in_base, out_base, w_base;
  logic [31:0] w_stride, desc_ptr, timeout_cyc, tag;
  assign s_axil_wready = 1'b1;
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin
      start_bit <= 1'b0; soft_rst <= 1'b0; ack_bit <= 1'b0;
      in_base <= 0; out_base <= 0; w_base <= 0;
      w_stride <= 0; desc_ptr <= 0; timeout_cyc <= 0; tag <= 0;
    end else if (s_axil_wvalid) begin
      case (s_axil_awaddr)
        8'h08: {soft_rst, start_bit} <= {s_axil_wdata[1], s_axil_wdata[0]};
        8'h14: in_base[31:0]  <= s_axil_wdata;
        8'h18: in_base[63:32] <= s_axil_wdata;
        8'h1C: out_base <= {32'b0, s_axil_wdata};
        8'h20: w_base   <= {32'b0, s_axil_wdata};
        8'h24: w_stride  <= s_axil_wdata;
        8'h28: desc_ptr  <= s_axil_wdata;
        8'h2C: timeout_cyc <= s_axil_wdata;
        8'h30: ack_bit   <= s_axil_wdata[0];
        8'h34: tag       <= s_axil_wdata;
        default: ;
      endcase
    end
  end
  // --- lifecycle: edge-triggered start/ack; reject double-start while busy.
  // start_bit/ack_bit are CSR levels; only their rising edges act, so a
  // held start cannot retrigger and a held ack cannot swallow the next done.
  logic busy, done, error;
  logic prev_start, prev_ack;
  logic seq_start, seq_done;
  logic sps_done, blk_done, head_done;
  logic [$clog2(DEPTH)-1:0] blk_idx;
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst || soft_rst) begin busy <= 1'b0; done <= 1'b0; error <= 1'b0; prev_start <= 1'b0; prev_ack <= 1'b0; end
    else begin
      prev_start <= start_bit; prev_ack <= ack_bit;
      if (start_bit && !prev_start && !busy) begin busy <= 1'b1; done <= 1'b0; error <= 1'b0; end
      else if (start_bit && !prev_start && busy) error <= 1'b1; // busy: reject repeat start
      if (seq_done && busy) begin busy <= 1'b0; done <= 1'b1; end
      if (ack_bit && !prev_ack) begin error <= 1'b0; if (done) done <= 1'b0; end
    end
  end
  assign seq_start = start_bit && !prev_start && !busy;
  spikformer_block_sequencer #(.DEPTH(DEPTH), .CH(CH)) u_seq (
    .s_clk(s_clk), .s_rst(s_rst || soft_rst),
    .i_sample_start(seq_start),
    .o_sample_done(seq_done), .o_sample_error(),
    .o_block_idx(blk_idx), .o_block_first(), .o_block_last(),
    .o_block_weight_base(), .i_block_weight_stride(w_stride),
    .o_sps_start(), .i_sps_done(sps_done),
    .o_blk_start(), .i_blk_done(blk_done),
    .o_head_start(), .i_head_done(head_done),
    .o_lif_rst(), .o_bank_sel(), .o_busy()
  );
  // Sub-engine done sources land with the full datapath; shell ties safe
  // idle values so the block compiles and simulates lifecycle standalone.
  assign sps_done = 1'b0;
  assign blk_done = 1'b0;
  assign head_done = 1'b0;
  assign o_ddr_req = busy;
  assign o_ddr_addr = in_base;
  // --- AXI-Lite read ---
  always_comb begin
    case (s_axil_araddr)
      8'h00: s_axil_rdata = ID;
      8'h04: s_axil_rdata = VERSION;
      8'h08: s_axil_rdata = {30'b0, soft_rst, start_bit};
      8'h0C: s_axil_rdata = {29'b0, error, done, busy};
      8'h10: s_axil_rdata = {31'b0, error};
      8'h34: s_axil_rdata = tag;
      default: s_axil_rdata = 32'b0;
    endcase
  end
  assign s_axil_rvalid = s_axil_arvalid;
  assign o_busy = busy; assign o_done = done; assign o_error = error;
  assign o_head_valid = 1'b0;
  genvar hh;
  generate
    for (hh = 0; hh < CH; hh = hh + 1) begin : g_sums
      assign o_head_sums[hh] = 32'b0;
    end
  endgenerate
endmodule
