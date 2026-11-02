# run_hw1_ooc_synth.tcl -- E1-HW1 W1 out-of-context synthesis of the P1 target npu_top.
# USAGE: vivado -mode batch -notrace -source run_hw1_ooc_synth.tcl \
#          -tclargs <generated_dir> <out_dir> <project_root>
#
# The design needs the dsp_mul_pipe4 dsp_macro IP (xilinx.com:ip:dsp_macro:1.0).
# A copy of the packaged XCI is re-customized for the target part inside a local
# project in <out_dir>/proj so the protected platform IP source stays untouched.
if {$argc < 3} { error "USAGE: run_hw1_ooc_synth.tcl <generated_dir> <out_dir> <project_root>" }
set gen  [file normalize [lindex $argv 0]]
set out  [file normalize [lindex $argv 1]]
set root [file normalize [lindex $argv 2]]
set part "xczu15eg-ffvb1156-2-i"

foreach f {sources.f npu_top.v p1_target_instr.mem} {
  if {![file exists [file join $gen $f]]} { error "missing $f in $gen" }
}
file mkdir $out
# $readmemh uses the process working directory: copy the mem next to the run and cd there.
file copy -force [file join $gen p1_target_instr.mem] [file join $out p1_target_instr.mem]
cd $out

# ---- local project (keeps IP output products inside this run) ----
create_project -force hw1_ooc [file join $out proj] -part $part

# ---- dsp_mul_pipe4 IP (copy of the packaged XCI, re-customized for the target part) ----
set ipdir [file join $out dsp_ip]
file mkdir $ipdir
set ipxci [file join $ipdir dsp_mul_pipe4.xci]
set ipxci_src [file join $root build/mnist_project2_board/reloc_mn8_ip_repo3/mn8_1.0/src/dsp_mul_pipe4/dsp_mul_pipe4.xci]
if {![file exists $ipxci_src]} { error "missing $ipxci_src" }
file copy -force $ipxci_src $ipxci
set ip [read_ip $ipxci]
puts "HW1_OOC_IP=[get_property NAME $ip] locked=[get_property IS_LOCKED $ip]"
if {[get_property IS_LOCKED $ip]} { upgrade_ip $ip; puts "HW1_OOC_IP_UPGRADED=true" }
set_property GENERATE_SYNTH_CHECKPOINT true $ip
generate_target synthesis $ip

# ---- source list (paths in sources.f are relative to the project root) ----
set srcs {}
set fh [open [file join $gen sources.f] r]
foreach raw [split [read $fh] "\n"] {
  set t [string trim $raw]
  if {$t eq "" || [string index $t 0] eq "#"} { continue }
  if {[string match "-i *" $t]} { continue }
  if {[string match "*_sim_netlist.v" $t]} { puts "HW1_OOC_SKIP_SIM_NETLIST=$t"; continue }
  set p [file normalize [file join $root $t]]
  if {![file exists $p]} { error "missing source $p" }
  lappend srcs $p
}
close $fh
lappend srcs [file normalize [file join $gen npu_top.v]]
puts "HW1_OOC_SOURCES=[llength $srcs]"
add_files -norecurse -fileset sources_1 $srcs
# Only user RTL is retyped; IP-managed files (e.g. dsp_mul_pipe4 stubs) must not be touched.
foreach f [get_files -of_objects [get_filesets sources_1]] {
  set fname [file normalize [get_property NAME $f]]
  if {[lsearch -exact $srcs $fname] >= 0 && [lsearch -exact {.v .sv} [file extension $fname]] >= 0} {
    set_property file_type SystemVerilog $f
  }
}
set_property top npu_top [current_fileset]
update_compile_order -fileset sources_1

# ---- 5ns constraint, no exceptions ----
set xdc [file join $out clk.xdc]
set fh [open $xdc w]
puts $fh "create_clock -name clk -period 5.000 \[get_ports clk\]"
close $fh
add_files -norecurse -fileset constrs_1 $xdc

synth_design -top npu_top -part $part -mode out_of_context -flatten_hierarchy rebuilt -directive default -generic MAC_IMPL=3

report_utilization -file [file join $out utilization.rpt]
report_utilization -hierarchical -hierarchical_depth 3 -file [file join $out utilization_hier.rpt]
report_timing_summary -delay_type max -max_paths 20 -report_unconstrained -file [file join $out timing_summary.rpt]
report_timing -delay_type max -max_paths 20 -file [file join $out timing_max.rpt]
report_timing -delay_type min -max_paths 20 -file [file join $out timing_min.rpt]
report_drc -file [file join $out drc.rpt]
report_design_analysis -complexity -file [file join $out design_analysis.rpt]

# ---- memory inference evidence ----
set ramfh [open [file join $out ram_cells.txt] w]
set cells [get_cells -hierarchical -quiet -filter {REF_NAME =~ RAM*}]
puts $ramfh "RAM_CELL_COUNT=[llength $cells]"
foreach c [lsort $cells] { puts $ramfh "[get_property REF_NAME $c] [get_property NAME $c]" }
close $ramfh
set bramfh [open [file join $out bram_cells.txt] w]
set bcells [get_cells -hierarchical -quiet -filter {REF_NAME =~ RAMB* || REF_NAME =~ URAM*}]
puts $bramfh "BLOCK_RAM_CELL_COUNT=[llength $bcells]"
foreach c [lsort $bcells] { puts $bramfh "[get_property REF_NAME $c] [get_property NAME $c]" }
close $bramfh
puts "HW1_OOC_RAM_CELLS=[llength $cells]"
puts "HW1_OOC_BLOCK_RAM_CELLS=[llength $bcells]"

set dspcells [get_cells -hierarchical -quiet -filter {REF_NAME =~ DSP48*}]
puts "HW1_OOC_DSP48_CELLS=[llength $dspcells]"
write_checkpoint -force [file join $out npu_top_ooc.dcp]
puts "HW1_OOC_SYNTH_DONE=true"

