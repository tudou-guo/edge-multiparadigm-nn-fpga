// GENERATED target adapter - do not edit. Fixed parameter binding of the
// frozen 48h target configuration; listed in sources.f and the manifest.
//
// Input hold-margin cells (OOC hold closure, added 2026-10-30).
// The 5ns OOC gate models every data input with set_input_delay 0.000 because
// the upstream driver is outside the analysis boundary. The routed r2 run
// failed hold with WHS -0.080 ns / 32,852 endpoints. Read-only DCP analysis
// of that run classified every failing path: all start at an input port
// (i_w_data 29,414 / i_tab_data 3,054 / i_tok_id 384) with <=0.071 ns data
// delay; zero internal startpoints. A post-route phys_opt_design -hold_fix
// probe could repair only 50 of 4,821 candidate nets, so the same hold repair
// is instantiated explicitly here: an even number of LUT1 inverter stages per
// input bit (non-inverting). It is pure combinational delay - no state, no
// protocol change, no clock change, no timing exception; the 5.000 ns clock
// and the 0.000 ns input-delay models are unchanged. Stages are DONT_TOUCH so
// implementation cannot optimize the hold repair away. The stage count is a
// trade-off: a 16-stage synthesis probe overspent input setup (synth WNS
// -0.191 ns) and was rejected. The next routed cut (10 stages, documented in
// docs/SPIKFORMER_NLP_48H_HOLD_CLOSURE_RESULT.md) already met the gate but
// left only +0.039 ns setup and +0.001 ns hold: the binding hold item was the
// asynchronous s_rst removal path (not covered by the data cells) and the
// placed input fanout nets consumed most of the input setup budget. Final
// cut: 8 stages on ALL sampled inputs AND on s_rst (s_clk passes through),
// with Explore place/phys_opt directives in the generated OOC Tcl.
// Simulation: define SPK_HOLD_MARGIN_SIM to elaborate the equivalent
// behavioural chain (the LUT1 cells are unisim).
module spk_input_hold_margin #(
  parameter int W = 64,
  parameter int STAGES = 8
)(
  input  logic [W-1:0] i_data,
  output logic [W-1:0] o_data
);
  logic [W-1:0] dly [0:STAGES];
  assign dly[0] = i_data;
  genvar gs, gb;
  generate
    for (gs = 0; gs < STAGES; gs = gs + 1) begin : g_stage
      for (gb = 0; gb < W; gb = gb + 1) begin : g_bit
`ifdef SPK_HOLD_MARGIN_SIM
        assign dly[gs+1][gb] = ~dly[gs][gb];
`else
        (* DONT_TOUCH = "TRUE", KEEP = "TRUE" *)
        LUT1 #(.INIT(2'b01)) u_dly (.O(dly[gs+1][gb]), .I0(dly[gs][gb]));
`endif
      end
    end
  endgenerate
  assign o_data = dly[STAGES];
  generate
    if ((STAGES % 2) != 0) begin : g_stages_even
      initial $error("spk_input_hold_margin STAGES must be even (non-inverting chain)");
    end
  endgenerate
endmodule

module spikformer_nlp_target_top (
  input  logic s_clk,
  input  logic s_rst,
  input  logic i_start,
  output logic o_done,
  input  logic [63:0] i_tab_data,
  input  logic i_tab_valid,
  output logic o_tab_ready,
  input  logic [15:0] i_tok_id,
  input  logic i_tok_valid,
  output logic o_tok_ready,
  input  logic [63:0] i_w_data,
  input  logic i_w_valid,
  output logic o_w_ready,
  output logic [31:0] o_sums [128],
  output logic o_sums_valid
);
  // Hold-margin cells on every non-clock/reset input. The routed r2 hold
  // failures start only at i_w_data/i_tab_data/i_tok_id, but a 16-stage probe
  // still left one i_start hold endpoint failing at synthesis, and the valid
  // strobes share the same boundary; delaying all sampled inputs keeps the
  // whole input boundary in the same (pessimistic 0-delay) hold model.
  logic start_d;
  logic rst_d;
  logic [63:0] tab_data_d;
  logic tab_valid_d;
  logic [15:0] tok_id_d;
  logic tok_valid_d;
  logic [63:0] w_data_d;
  logic w_valid_d;
  spk_input_hold_margin #(.W(1))  u_hm_start     (.i_data(i_start),     .o_data(start_d));
  spk_input_hold_margin #(.W(1))  u_hm_rst       (.i_data(s_rst),       .o_data(rst_d));
  spk_input_hold_margin #(.W(64)) u_hm_tab_data  (.i_data(i_tab_data),  .o_data(tab_data_d));
  spk_input_hold_margin #(.W(1))  u_hm_tab_valid (.i_data(i_tab_valid), .o_data(tab_valid_d));
  spk_input_hold_margin #(.W(16)) u_hm_tok_id    (.i_data(i_tok_id),    .o_data(tok_id_d));
  spk_input_hold_margin #(.W(1))  u_hm_tok_valid (.i_data(i_tok_valid), .o_data(tok_valid_d));
  spk_input_hold_margin #(.W(64)) u_hm_w_data    (.i_data(i_w_data),    .o_data(w_data_d));
  spk_input_hold_margin #(.W(1))  u_hm_w_valid   (.i_data(i_w_valid),   .o_data(w_valid_d));
  spikformer_nlp_top #(
    .E(128), .HEADS(4), .HEAD_DIM(32), .HIDDEN(512), .T(4),
    .TOKENS(32), .DEPTH(2), .TL(8), .TC(8), .RW(4),
    .PSUM_SLICE(32), .SUM_W(6), .LANES(8), .VTAB(4000), .OW(32)
  ) u_top (
    .s_clk(s_clk), .s_rst(rst_d), .i_start(start_d), .o_done(o_done),
    .i_tab_data(tab_data_d), .i_tab_valid(tab_valid_d), .o_tab_ready(o_tab_ready),
    .i_tok_id(tok_id_d), .i_tok_valid(tok_valid_d), .o_tok_ready(o_tok_ready),
    .i_w_data(w_data_d), .i_w_valid(w_valid_d), .o_w_ready(o_w_ready),
    .o_sums(o_sums), .o_sums_valid(o_sums_valid)
  );
endmodule
