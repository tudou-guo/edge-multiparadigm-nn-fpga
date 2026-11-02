"""Codegen: header, target adapter, sources.f, constraints, project + OOC Tcl.

No Vivado launch here. All paths are resolved for the real host checkout
the checkout root so the generated Tcl so the generated Tcl can be executed directly by scripts/spikformer/
run_physical.py (which is the only entry allowed to start Vivado, and only
under the shared heavy lock).
"""
from __future__ import annotations
import hashlib, json, os, pathlib
ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[3]).resolve()

NORM = {'sum': 0, 'ps_div': 1, 'folded': 2}
TICK = chr(96)
HOST_ROOT = ROOT


def _rtl_sources():
    man = json.loads((HOST_ROOT / 'rtl/ops/spiking/source_manifest.json').read_text())
    return list(man['sources'])


HOLD_MARGIN_STAGES = 8


def _adapter(cfg):
    d = cfg['derived']
    E, H, HD = cfg['embed'], cfg['heads'], cfg['head_dim']
    MLP, T, D = cfg['mlp_hidden'], cfg['time_steps'], cfg['depth']
    tokens = d['tokens']
    rw, ps, sw = d['rw'], d['psum_slice'], d['sum_w']
    vtab = cfg['vocab_size'] if cfg.get('family', '').startswith('spikformer-nlp') else 0
    lines = [
        '// GENERATED target adapter - do not edit. Fixed parameter binding of the',
        '// frozen 48h target configuration; listed in sources.f and the manifest.',
        '//',
        '// Input hold-margin cells (OOC hold closure, added 2026-10-30).',
        '// The 5ns OOC gate models every data input with set_input_delay 0.000 because',
        '// the upstream driver is outside the analysis boundary. The routed r2 run',
        '// failed hold with WHS -0.080 ns / 32,852 endpoints. Read-only DCP analysis',
        '// of that run classified every failing path: all start at an input port',
        '// (i_w_data 29,414 / i_tab_data 3,054 / i_tok_id 384) with <=0.071 ns data',
        '// delay; zero internal startpoints. A post-route phys_opt_design -hold_fix',
        '// probe could repair only 50 of 4,821 candidate nets, so the same hold repair',
        '// is instantiated explicitly here: an even number of LUT1 inverter stages per',
        '// input bit (non-inverting). It is pure combinational delay - no state, no',
        '// protocol change, no clock change, no timing exception; the 5.000 ns clock',
        '// and the 0.000 ns input-delay models are unchanged. Stages are DONT_TOUCH so',
        '// implementation cannot optimize the hold repair away. The stage count is a',
        '// trade-off: a 16-stage synthesis probe overspent input setup (synth WNS',
        '// -0.191 ns) and was rejected. The next routed cut (10 stages, documented in',
        '// docs/SPIKFORMER_NLP_48H_HOLD_CLOSURE_RESULT.md) already met the gate but',
        '// left only +0.039 ns setup and +0.001 ns hold: the binding hold item was the',
        '// asynchronous s_rst removal path (not covered by the data cells) and the',
        '// placed input fanout nets consumed most of the input setup budget. Final',
        '// cut: 8 stages on ALL sampled inputs AND on s_rst (s_clk passes through),',
        '// with Explore place/phys_opt directives in the generated OOC Tcl.',
        '// Simulation: define SPK_HOLD_MARGIN_SIM to elaborate the equivalent',
        '// behavioural chain (the LUT1 cells are unisim).',
        'module spk_input_hold_margin #(',
        '  parameter int W = 64,',
        '  parameter int STAGES = %d' % HOLD_MARGIN_STAGES,
        ')(',
        '  input  logic [W-1:0] i_data,',
        '  output logic [W-1:0] o_data',
        ');',
        '  logic [W-1:0] dly [0:STAGES];',
        '  assign dly[0] = i_data;',
        '  genvar gs, gb;',
        '  generate',
        '    for (gs = 0; gs < STAGES; gs = gs + 1) begin : g_stage',
        '      for (gb = 0; gb < W; gb = gb + 1) begin : g_bit',
        '`ifdef SPK_HOLD_MARGIN_SIM',
        '        assign dly[gs+1][gb] = ~dly[gs][gb];',
        '`else',
        '        (* DONT_TOUCH = "TRUE", KEEP = "TRUE" *)',
        '        LUT1 #(.INIT(2\'b01)) u_dly (.O(dly[gs+1][gb]), .I0(dly[gs][gb]));',
        '`endif',
        '      end',
        '    end',
        '  endgenerate',
        '  assign o_data = dly[STAGES];',
        '  generate',
        '    if ((STAGES % 2) != 0) begin : g_stages_even',
        '      initial $error("spk_input_hold_margin STAGES must be even (non-inverting chain)");',
        '    end',
        '  endgenerate',
        'endmodule',
        '',
        'module spikformer_nlp_target_top (',
        '  input  logic s_clk,',
        '  input  logic s_rst,',
        '  input  logic i_start,',
        '  output logic o_done,',
        '  input  logic [63:0] i_tab_data,',
        '  input  logic i_tab_valid,',
        '  output logic o_tab_ready,',
        '  input  logic [15:0] i_tok_id,',
        '  input  logic i_tok_valid,',
        '  output logic o_tok_ready,',
        '  input  logic [63:0] i_w_data,',
        '  input  logic i_w_valid,',
        '  output logic o_w_ready,',
        '  output logic [31:0] o_sums [%d],' % E,
        '  output logic o_sums_valid',
        ');',
        '  // Hold-margin cells on every non-clock/reset input. The routed r2 hold',
        '  // failures start only at i_w_data/i_tab_data/i_tok_id, but a 16-stage probe',
        '  // still left one i_start hold endpoint failing at synthesis, and the valid',
        '  // strobes share the same boundary; delaying all sampled inputs keeps the',
        '  // whole input boundary in the same (pessimistic 0-delay) hold model.',
        '  logic start_d;',
        '  logic rst_d;',
        '  logic [63:0] tab_data_d;',
        '  logic tab_valid_d;',
        '  logic [15:0] tok_id_d;',
        '  logic tok_valid_d;',
        '  logic [63:0] w_data_d;',
        '  logic w_valid_d;',
        '  spk_input_hold_margin #(.W(1))  u_hm_start     (.i_data(i_start),     .o_data(start_d));',
        '  spk_input_hold_margin #(.W(1))  u_hm_rst       (.i_data(s_rst),       .o_data(rst_d));',
        '  spk_input_hold_margin #(.W(64)) u_hm_tab_data  (.i_data(i_tab_data),  .o_data(tab_data_d));',
        '  spk_input_hold_margin #(.W(1))  u_hm_tab_valid (.i_data(i_tab_valid), .o_data(tab_valid_d));',
        '  spk_input_hold_margin #(.W(16)) u_hm_tok_id    (.i_data(i_tok_id),    .o_data(tok_id_d));',
        '  spk_input_hold_margin #(.W(1))  u_hm_tok_valid (.i_data(i_tok_valid), .o_data(tok_valid_d));',
        '  spk_input_hold_margin #(.W(64)) u_hm_w_data    (.i_data(i_w_data),    .o_data(w_data_d));',
        '  spk_input_hold_margin #(.W(1))  u_hm_w_valid   (.i_data(i_w_valid),   .o_data(w_valid_d));',
        '  spikformer_nlp_top #(',
        '    .E(%d), .HEADS(%d), .HEAD_DIM(%d), .HIDDEN(%d), .T(%d),' % (E, H, HD, MLP, T),
        '    .TOKENS(%d), .DEPTH(%d), .TL(8), .TC(8), .RW(%d),' % (tokens, D, rw),
        '    .PSUM_SLICE(%d), .SUM_W(%d), .LANES(8), .VTAB(%d), .OW(32)' % (ps, sw, vtab),
        '  ) u_top (',
        '    .s_clk(s_clk), .s_rst(rst_d), .i_start(start_d), .o_done(o_done),',
        '    .i_tab_data(tab_data_d), .i_tab_valid(tab_valid_d), .o_tab_ready(o_tab_ready),',
        '    .i_tok_id(tok_id_d), .i_tok_valid(tok_valid_d), .o_tok_ready(o_tok_ready),',
        '    .i_w_data(w_data_d), .i_w_valid(w_valid_d), .o_w_ready(o_w_ready),',
        '    .o_sums(o_sums), .o_sums_valid(o_sums_valid)',
        '  );',
        'endmodule',
        '',
    ]
    return chr(10).join(lines)


def generate(cfg, sched, param_files, out):
    out = pathlib.Path(out)
    gen = out / 'generated'
    gen.mkdir(parents=True, exist_ok=True)
    d = cfg['derived']
    L = []
    L.append('// GENERATED - do not edit. Per-build include for this run.')
    L.append(TICK + 'ifndef SPIKFORMER_CONFIG_VH')
    L.append(TICK + 'define SPIKFORMER_CONFIG_VH')
    L.append(TICK + 'define SPIK_T %d' % cfg['time_steps'])
    L.append(TICK + 'define SPIK_DEPTH %d' % cfg['depth'])
    L.append(TICK + 'define SPIK_EMBED %d' % cfg['embed'])
    L.append(TICK + 'define SPIK_HEADS %d' % cfg['heads'])
    L.append(TICK + 'define SPIK_HEAD_DIM %d' % cfg['head_dim'])
    L.append(TICK + 'define SPIK_MLP_HIDDEN %d' % cfg['mlp_hidden'])
    L.append(TICK + 'define SPIK_TOKENS %d' % d['tokens'])
    L.append(TICK + 'define SPIK_RW %d' % d['rw'])
    L.append(TICK + 'define SPIK_RES_BITS %d' % d['rw'])
    L.append(TICK + 'define SPIK_RES_BITS_MIN %d' % d['res_bits_min'])
    L.append(TICK + 'define SPIK_PSUM_SLICE %d' % d['psum_slice'])
    L.append(TICK + 'define SPIK_LIF_ADD_BITS %d' % d['lif_add_bits'])
    L.append(TICK + 'define SPIK_AV_BITS %d' % d['av_bits'])
    L.append(TICK + 'define SPIK_SUM_W %d' % d['sum_w'])
    L.append(TICK + 'define SPIK_HEAD_NORM %d' % NORM[cfg['head']['reduction']])
    L.append('// thresholds are runtime parameters carried by the per-block descriptor')
    L.append('// beats (see contract.json hardware_abi), not compile-time constants.')
    sps = d.get('sps')
    if sps:
        L.append(TICK + 'define SPIK_SPS_H %d' % sps['h'])
        L.append(TICK + 'define SPIK_SPS_W %d' % sps['w'])
        L.append(TICK + 'define SPIK_SPS_C0 %d' % sps['c0'])
        L.append(TICK + 'define SPIK_SPS_C1 %d' % sps['c1'])
        L.append(TICK + 'define SPIK_SPS_C2 %d' % sps['c2'])
        L.append(TICK + 'define SPIK_SPS_TOKENS %d' % sps['tokens'])
        L.append(TICK + 'define SPIK_SPS_CONV1_BEATS %d' % sps['conv1_beats'])
        L.append(TICK + 'define SPIK_SPS_CONV2_BEATS %d' % sps['conv2_beats'])
    L.append(TICK + 'endif')
    (gen / 'spikformer_config.vh').write_text(chr(10).join(L) + chr(10))
    (gen / 'spikformer_nlp_target_top.sv').write_text(_adapter(cfg))

    sources = _rtl_sources()
    (gen / 'sources.f').write_text(chr(10).join(sources) + chr(10))
    adapter_rel = ('generated/spikformer_nlp_target_top.sv')
    all_srcs = [(HOST_ROOT / s).as_posix() for s in sources] + [(out / adapter_rel).resolve().as_posix()]
    xdc = [
        '# GENERATED constraints: one clock, NO timing exceptions, no downclock.',
        'create_clock -name s_clk -period 5.000 [get_ports s_clk]',
        'set_input_delay -clock s_clk 0.000 [all_inputs]',
        'set_output_delay -clock s_clk 0.000 [all_outputs]',
        '',
    ]
    (gen / 'spikformer.xdc').write_text(chr(10).join(xdc))

    common = [
        'set snn_root {%s}' % HOST_ROOT.as_posix(),
        'set script_dir [file normalize [file dirname [info script]]]',
        'set out_dir [lindex $argv 0]',
        'set stage [lindex $argv 1]',
        'if {$out_dir eq ""} { set out_dir [file normalize [file join $script_dir .. ooc_run]] }',
        'if {$stage eq ""} { set stage "ooc" }',
        'file mkdir $out_dir',
        'set srcs {}',
        'foreach s {%s} { lappend srcs $s }' % ' '.join(all_srcs),
        'if {[llength $srcs] == 0} { error "no sources" }',
        'read_verilog -sv $srcs',
        'read_xdc [file join $script_dir spikformer.xdc]',
        'synth_design -top spikformer_nlp_target_top -part xczu15eg-ffvb1156-2-i -mode out_of_context',
        'write_checkpoint -force [file join $out_dir post_synth.dcp]',
        'report_utilization -file [file join $out_dir util_synth.rpt]',
        'report_drc -file [file join $out_dir drc_synth.rpt]',
        'report_timing_summary -file [file join $out_dir timing_synth.rpt]',
        'set latch_cells [get_cells -hierarchical -quiet -filter {REF_NAME =~ *LATCH*}]',
        'set fh [open [file join $out_dir facts.txt] w]',
        'puts $fh "stage=$stage"',
        'puts $fh "latch_cells=[llength $latch_cells]"',
        'close $fh',
        'if {$stage eq "synth"} {',
        '  puts "SYNTH_DONE_OK"',
        '  return',
        '}',
        'opt_design',
        'place_design -directive Explore',
        'phys_opt_design -directive Explore',
        'route_design',
        'write_checkpoint -force [file join $out_dir post_route.dcp]',
        'report_timing_summary -delay_type min_max -max_paths 10 -file [file join $out_dir timing_route.rpt]',
        'report_timing -delay_type min_max -max_paths 10 -nworst 1 -file [file join $out_dir timing_route_paths.rpt]',
        'report_utilization -file [file join $out_dir util_route.rpt]',
        'report_route_status -file [file join $out_dir route_status.rpt]',
        'report_drc -file [file join $out_dir drc_route.rpt]',
        'report_pulse_width -file [file join $out_dir pulse_width.rpt]',
        'report_clocks -file [file join $out_dir clocks.rpt]',
        'report_exceptions -file [file join $out_dir exceptions.rpt]',
        'set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1 -nworst 1]]',
        'set wns_hold [get_property SLACK [get_timing_paths -delay_type min -max_paths 1 -nworst 1]]',
        'set unrouted [get_nets -hierarchical -quiet -filter {ROUTE_STATUS == "UNROUTED"}]',
        'set fh [open [file join $out_dir facts.txt] w]',
        'puts $fh "stage=$stage"',
        'puts $fh "wns_setup_ns=$wns"',
        'puts $fh "wns_hold_ns=$wns_hold"',
        'puts $fh "unrouted_nets=[llength $unrouted]"',
        'puts $fh "latch_cells=[llength $latch_cells]"',
        'close $fh',
        'puts "OOC_WNS_SETUP=$wns"',
        'puts "OOC_WNS_HOLD=$wns_hold"',
        'puts "OOC_UNROUTED=[llength $unrouted]"',
        'puts "OOC_DONE_OK"',
        '',
    ]
    (gen / 'spikformer_ooc.tcl').write_text(chr(10).join(common))
    proj = [
        '# GENERATED project assembly (IP packaging / interactive use); heavy start',
        '# only under the shared heavy lock via scripts/spikformer/run_physical.py.',
        'set snn_root {%s}' % HOST_ROOT,
        'set script_dir [file normalize [file dirname [info script]]]',
        'set proj_dir [lindex $argv 0]',
        'if {$proj_dir eq ""} { set proj_dir [file join $script_dir .. project] }',
        'create_project -force spikformer_nlp $proj_dir -part xczu15eg-ffvb1156-2-i',
        'set srcs {}',
        'foreach s {%s} { lappend srcs $s }' % ' '.join(all_srcs),
        'add_files -norecurse -fileset sources_1 $srcs',
        'set_property top spikformer_nlp_target_top [current_fileset]',
        'add_files -norecurse -fileset constrs_1 [file join $script_dir spikformer.xdc]',
        'update_compile_order -fileset sources_1',
        '',
    ]
    (gen / 'spikformer_project.tcl').write_text(chr(10).join(proj))
    key = hashlib.sha256(json.dumps({'cfg': cfg, 'sched': sched, 'tool': '2021.1',
                                     'part': 'xczu15eg-ffvb1156-2-i'},
                                    sort_keys=True).encode()).hexdigest()
    (gen / 'cache_key.txt').write_text(key + chr(10))
    return {'header': 'generated/spikformer_config.vh',
            'adapter': 'generated/spikformer_nlp_target_top.sv',
            'sources_f': 'generated/sources.f',
            'xdc': 'generated/spikformer.xdc',
            'ooc_tcl': 'generated/spikformer_ooc.tcl',
            'project_tcl': 'generated/spikformer_project.tcl',
            'cache_key': key}