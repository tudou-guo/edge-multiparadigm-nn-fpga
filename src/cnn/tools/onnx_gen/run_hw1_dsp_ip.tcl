if {$argc < 2} { error "USAGE: run_hw1_dsp_ip.tcl <xci> <out_dcp>" }
create_project -in_memory -part xczu15eg-ffvb1156-2-i
set ip [read_ip [file normalize [lindex $argv 0]]]
puts "IP_READ=[get_property NAME $ip]"
if {[get_property IS_LOCKED $ip]} {
  puts "IP_WAS_LOCKED=true"
  upgrade_ip $ip
}
set_property GENERATE_SYNTH_CHECKPOINT true $ip
generate_target synthesis $ip
synth_ip $ip
write_checkpoint -force [file normalize [lindex $argv 1]]
puts "DSP_IP_SYNTH_DONE=true"

