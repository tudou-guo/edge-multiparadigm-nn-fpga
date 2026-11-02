set snn_root {D:/dl_snn}
set script_dir [file normalize [file dirname [info script]]]
set out_dir [lindex $argv 0]
set stage [lindex $argv 1]
if {$out_dir eq ""} { set out_dir [file normalize [file join $script_dir .. ooc_run]] }
if {$stage eq ""} { set stage "ooc" }
file mkdir $out_dir
set srcs {}
foreach s {D:/dl_snn/rtl/ops/spiking/attention/spiking_attention_pe.sv D:/dl_snn/rtl/ops/spiking/attention/spiking_linemac_pe.sv D:/dl_snn/rtl/ops/spiking/attention/spiking_popcount.sv D:/dl_snn/rtl/ops/spiking/control/spikformer_block_sequencer.sv D:/dl_snn/rtl/ops/spiking/control/spikformer_nlp_top.sv D:/dl_snn/rtl/ops/spiking/control/spiking_block_top.sv D:/dl_snn/rtl/ops/spiking/embed/spiking_embed_top.sv D:/dl_snn/rtl/ops/spiking/include/spikformer_cfg.vh D:/dl_snn/rtl/ops/spiking/lif/spiking_lif_cell.sv D:/dl_snn/rtl/ops/spiking/lif/spiking_lif_group.sv D:/dl_snn/rtl/ops/spiking/linear/spiking_systolic_array.sv D:/dl_snn/rtl/ops/spiking/linear/spiking_systolic_pe.sv D:/dl_snn/rtl/ops/spiking/mlp/spiking_mlp_controller.sv D:/dl_snn/rtl/ops/spiking/reduce/spiking_head_reduce.sv D:/dl_snn/rtl/ops/spiking/sps/spiking_adder_tree.sv D:/dl_snn/rtl/ops/spiking/sps/spiking_eyeriss_pe.sv D:/dl_snn/rtl/ops/spiking/sps/spiking_maxpool2x2.sv D:/dl_snn/rtl/ops/spiking/sps/spiking_sps_top.sv D:/dl_snn/rtl/ops/spiking/wrappers/spikformer_platform_wrapper.sv D:/dl_snn/build/spikformer/nlp_48h_20261029_1915/final/nlp_real_5stage_r4_holdfix/generated/spikformer_nlp_target_top.sv} { lappend srcs $s }
if {[llength $srcs] == 0} { error "no sources" }
read_verilog -sv $srcs
read_xdc [file join $script_dir spikformer.xdc]
synth_design -top spikformer_nlp_target_top -part xczu15eg-ffvb1156-2-i -mode out_of_context
write_checkpoint -force [file join $out_dir post_synth.dcp]
report_utilization -file [file join $out_dir util_synth.rpt]
report_drc -file [file join $out_dir drc_synth.rpt]
report_timing_summary -file [file join $out_dir timing_synth.rpt]
set latch_cells [get_cells -hierarchical -quiet -filter {REF_NAME =~ *LATCH*}]
set fh [open [file join $out_dir facts.txt] w]
puts $fh "stage=$stage"
puts $fh "latch_cells=[llength $latch_cells]"
close $fh
if {$stage eq "synth"} {
  puts "SYNTH_DONE_OK"
  return
}
opt_design
place_design -directive Explore
phys_opt_design -directive Explore
route_design
write_checkpoint -force [file join $out_dir post_route.dcp]
report_timing_summary -delay_type min_max -max_paths 10 -file [file join $out_dir timing_route.rpt]
report_timing -delay_type min_max -max_paths 10 -nworst 1 -file [file join $out_dir timing_route_paths.rpt]
report_utilization -file [file join $out_dir util_route.rpt]
report_route_status -file [file join $out_dir route_status.rpt]
report_drc -file [file join $out_dir drc_route.rpt]
report_pulse_width -file [file join $out_dir pulse_width.rpt]
report_clocks -file [file join $out_dir clocks.rpt]
report_exceptions -file [file join $out_dir exceptions.rpt]
set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1 -nworst 1]]
set wns_hold [get_property SLACK [get_timing_paths -delay_type min -max_paths 1 -nworst 1]]
set unrouted [get_nets -hierarchical -quiet -filter {ROUTE_STATUS == "UNROUTED"}]
set fh [open [file join $out_dir facts.txt] w]
puts $fh "stage=$stage"
puts $fh "wns_setup_ns=$wns"
puts $fh "wns_hold_ns=$wns_hold"
puts $fh "unrouted_nets=[llength $unrouted]"
puts $fh "latch_cells=[llength $latch_cells]"
close $fh
puts "OOC_WNS_SETUP=$wns"
puts "OOC_WNS_HOLD=$wns_hold"
puts "OOC_UNROUTED=[llength $unrouted]"
puts "OOC_DONE_OK"
