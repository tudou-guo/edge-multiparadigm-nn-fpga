# configure_e1p3_bd.tcl -- pin the E1-P3 candidate IP parameters on the BD cell.
# USAGE: vivado -mode batch -source configure_e1p3_bd.tcl \
#   -tclargs <project_root> <cell> <max_cin> <max_cout> <max_ch> <max_w> \
#            <num_layers> <instr_words> <instr_addr_w> <instr_init> <mac_impl> <tn> <tm> \
#            [<dw_pc> <dw_tap_parallel> <dw_rq_lanes>]
#
# The three optional DW arguments (defaults 8 1 1 = dw_c8_t1_q1) are applied
# only when the packaged IP exposes the matching CONFIG.* property.  Old IPs
# packaged before the parameterized DW core keep the original 13-argument call.
if {$argc < 13} { error "USAGE: configure_e1p3_bd.tcl <root> <cell> <max_cin> <max_cout> <max_ch> <max_w> <num_layers> <instr_words> <instr_addr_w> <instr_init> <mac_impl> <tn> <tm> [<dw_pc> <dw_tap_parallel> <dw_rq_lanes>]" }
set root [file normalize [lindex $argv 0]]
set cell [lindex $argv 1]
set max_cin [lindex $argv 2]
set max_cout [lindex $argv 3]
set max_ch [lindex $argv 4]
set max_w [lindex $argv 5]
set num_layers [lindex $argv 6]
set instr_words [lindex $argv 7]
set instr_addr_w [lindex $argv 8]
set instr_init [lindex $argv 9]
set mac_impl [lindex $argv 10]
set tn [lindex $argv 11]
set tm [lindex $argv 12]
set dw_pc 8
set dw_tap_parallel 1
set dw_rq_lanes 1
if {$argc >= 14} { set dw_pc [lindex $argv 13] }
if {$argc >= 15} { set dw_tap_parallel [lindex $argv 14] }
if {$argc >= 16} { set dw_rq_lanes [lindex $argv 15] }
foreach {name value} [list dw_pc $dw_pc dw_tap_parallel $dw_tap_parallel \
                          dw_rq_lanes $dw_rq_lanes] {
  if {![string is integer -strict $value] || $value < 1} {
    error "invalid DW parameter $name=$value"
  }
}

open_project [file join $root project_2.xpr]
set bd_file [file join $root project_2.srcs sources_1 bd design_1 design_1.bd]
open_bd_design $bd_file
set c [get_bd_cells $cell]
puts "E1P3_BD_BEFORE:"
foreach p {MAC_IMPL Tn Tm MAX_CIN MAX_COUT MAX_CH MAX_W NUM_LAYERS INSTR_WORDS INSTR_ADDR_W INSTR_INIT WEIGHT_MODE CLAMP_MIN_EN DW_PC DW_TAP_PARALLEL DW_RQ_LANES} {
  if {[llength [get_property -quiet CONFIG.$p $c]] > 0} {
    puts "  $p = [get_property CONFIG.$p $c]"
  } else {
    puts "  $p = <unset>"
  }
}
# Depthwise parameters only exist on IPs packaged from the parameterized core.
set dw_props {}
foreach {name value} [list DW_PC $dw_pc DW_TAP_PARALLEL $dw_tap_parallel \
                          DW_RQ_LANES $dw_rq_lanes] {
  if {[llength [get_property -quiet CONFIG.$name $c]] > 0} {
    lappend dw_props CONFIG.$name $value
  } else {
    puts "WARNING: IP has no CONFIG.$name property (pre-parameterized IP); skipped"
  }
}
set_property -dict [list \
  CONFIG.MAC_IMPL $mac_impl \
  CONFIG.Tn $tn \
  CONFIG.Tm $tm \
  CONFIG.MAX_CIN $max_cin \
  CONFIG.MAX_COUT $max_cout \
  CONFIG.MAX_CH $max_ch \
  CONFIG.MAX_W $max_w \
  CONFIG.NUM_LAYERS $num_layers \
  CONFIG.INSTR_WORDS $instr_words \
  CONFIG.INSTR_ADDR_W $instr_addr_w \
  CONFIG.INSTR_INIT $instr_init \
  CONFIG.WEIGHT_MODE {0} \
  CONFIG.CLAMP_MIN_EN {1} \
] $c
if {[llength $dw_props] > 0} {
  set_property -dict $dw_props $c
}
puts "E1P3_BD_AFTER:"
foreach p {MAC_IMPL Tn Tm MAX_CIN MAX_COUT MAX_CH MAX_W NUM_LAYERS INSTR_WORDS INSTR_ADDR_W INSTR_INIT DW_PC DW_TAP_PARALLEL DW_RQ_LANES} {
  puts "  $p = [get_property CONFIG.$p $c]"
}
validate_bd_design
save_bd_design
generate_target all [get_files $bd_file]
puts "E1P3_BD_CONFIG_PASS=true"
close_project
