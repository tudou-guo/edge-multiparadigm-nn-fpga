# configure_e1p2_bd.tcl -- pin the E1-P2 G2 first-fit IP parameters on the BD cell.
# USAGE: vivado -mode batch -source configure_e1hw1_bd.tcl -tclargs <project_root>
if {$argc < 1} { error "USAGE: configure_e1hw1_bd.tcl <project_root>" }
set root [file normalize [lindex $argv 0]]
open_project [file join $root project_2.xpr]
set bd_file [file join $root project_2.srcs sources_1 bd design_1 design_1.bd]
open_bd_design $bd_file
set c [get_bd_cells e1p2npu_0]
puts "E1P2_BD_BEFORE:"
foreach p {MAC_IMPL Tn Tm MAX_CIN MAX_COUT MAX_CH MAX_W NUM_LAYERS INSTR_WORDS INSTR_ADDR_W INSTR_INIT WEIGHT_MODE CLAMP_MIN_EN} {
  if {[llength [get_property -quiet CONFIG.$p $c]] > 0} {
    puts "  $p = [get_property CONFIG.$p $c]"
  } else {
    puts "  $p = <unset>"
  }
}
set_property -dict [list \
  CONFIG.MAC_IMPL {3} \
  CONFIG.Tn {8} \
  CONFIG.Tm {8} \
  CONFIG.MAX_CIN {4} \
  CONFIG.MAX_COUT {8} \
  CONFIG.MAX_CH {8} \
  CONFIG.MAX_W {8} \
  CONFIG.NUM_LAYERS {9} \
  CONFIG.INSTR_WORDS {256} \
  CONFIG.INSTR_ADDR_W {8} \
  CONFIG.INSTR_INIT {g2_base_instr.mem} \
  CONFIG.WEIGHT_MODE {0} \
  CONFIG.CLAMP_MIN_EN {1} \
] $c
puts "E1P2_BD_AFTER:"
foreach p {MAC_IMPL Tn Tm MAX_CIN MAX_COUT MAX_CH MAX_W NUM_LAYERS INSTR_WORDS INSTR_INIT} {
  puts "  $p = [get_property CONFIG.$p $c]"
}
validate_bd_design
save_bd_design
generate_target all [get_files $bd_file]
puts "E1P2_BD_CONFIG_PASS=true"
close_project
