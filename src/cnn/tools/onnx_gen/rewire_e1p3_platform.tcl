# rewire_e1p3_platform.tcl -- replace the frozen mn8 cell with the E1-P3 candidate IP
# in a local copy of project_2 and regenerate all BD products.
# USAGE: vivado -mode batch -source rewire_e1p3_platform.tcl \
#          -tclargs <project_root> <ip_repo> <ip_vlnv> <cell_name>
if {$argc < 4} { error "USAGE: rewire_e1p3_platform.tcl <project_root> <ip_repo> <vlnv> <cell_name>" }
set root   [file normalize [lindex $argv 0]]
set iprepo [file normalize [lindex $argv 1]]
set vlnv   [lindex $argv 2]
set cell   [lindex $argv 3]

open_project [file join $root project_2.xpr]
set_property ip_repo_paths [list $iprepo] [current_project]
update_ip_catalog -rebuild

set bd_file [file join $root project_2.srcs sources_1 bd design_1 design_1.bd]
open_bd_design $bd_file

set old [get_bd_cells -quiet mn8_0]
if {[llength $old] != 1} { error "expected exactly one mn8_0 cell, found [llength $old]" }
delete_bd_objs $old

set newcell [create_bd_cell -type ip -vlnv $vlnv $cell]
puts "E1P3_BD_CELL=[get_property VLNV $newcell]"

connect_bd_intf_net [get_bd_intf_pins ps8_0_axi_periph/M00_AXI] [get_bd_intf_pins $cell/s_axil]
connect_bd_intf_net [get_bd_intf_pins axi_smc/S00_AXI] [get_bd_intf_pins $cell/m_axi]
connect_bd_net [get_bd_pins rst_ps8_0_199M/peripheral_aresetn] [get_bd_pins $cell/rst_n]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins $cell/clk]

assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces $cell/m_axi] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_LOW] -force
assign_bd_address -offset 0xFF000000 -range 0x01000000 -target_address_space [get_bd_addr_spaces $cell/m_axi] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_LPS_OCM] -force
assign_bd_address -offset 0xE0000000 -range 0x10000000 -target_address_space [get_bd_addr_spaces $cell/m_axi] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_PCIE_LOW] -force
assign_bd_address -offset 0xC0000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces $cell/m_axi] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_QSPI] -force
assign_bd_address -offset 0x000480000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] [get_bd_addr_segs $cell/s_axil/reg0] -force

validate_bd_design
puts "E1P3_BD_VALIDATE_PASS=true"
save_bd_design
generate_target all [get_files $bd_file]

set wrapper [file join $root project_2.gen sources_1 bd design_1 hdl design_1_wrapper.v]
if {[llength [get_files -quiet $wrapper]] == 0} {
    add_files -norecurse $wrapper
} else {
    puts "E1P3_WRAPPER_ALREADY_PRESENT=$wrapper"
}
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1
puts "E1P3_WRAPPER=$wrapper"
close_project
puts "E1P3_REWIRE_DONE=true"
