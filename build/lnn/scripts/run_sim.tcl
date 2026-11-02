# scripts/run_sim.tcl -- one-shot xsim compile + elaborate + run for all CfC
# self-checking testbenches (contract 3.5).
#
# Usage (from anywhere, after sourcing E:/Vitis/Vivado/2021.1/settings64.bat):
#     vivado -mode batch -notrace -source scripts/run_sim.tcl
#
# The script compiles rtl/*.sv + the three testbenches with
#     xvlog -sv -i ../export/gen -f filelist.f          (cwd = verification/)
# then elaborates and runs tb_cfc_cell, tb_cfc_seq and tb_cfc_protocol.
# Each testbench prints CFC_TB_PASS / CFC_TB_FAIL.  The script exits non-zero
# on any failure and prints the overall CFC_SIM_PASS / CFC_SIM_FAIL marker.

set script_dir [file dirname [file normalize [info script]]]
set root       [file normalize [file join $script_dir ".."]]
set sim_dir    [file join $root "verification"]
set gen_dir    [file join $root "export" "gen"]
set log_dir    [file join $sim_dir "logs"]

proc find_tool {name} {
    set t [auto_execok $name]
    if {$t ne ""} { return $t }
    if {[info exists ::env(XILINX_VIVADO)]} {
        foreach cand [list [file join $::env(XILINX_VIVADO) bin "$name.bat"] \
                           [file join $::env(XILINX_VIVADO) bin $name]] {
            if {[file exists $cand]} { return $cand }
        }
    }
    set d [file dirname [info nameofexecutable]]
    foreach cand [list [file join $d "$name.bat"] [file join $d $name]] {
        if {[file exists $cand]} { return $cand }
    }
    return ""
}

proc run_cmd {argv logfile} {
    set rc [catch {exec {*}$argv > $logfile 2>@1} err]
    return $rc
}

proc log_has {logfile needle} {
    if {![file exists $logfile]} { return 0 }
    set fh [open $logfile r]
    set data [read $fh]
    close $fh
    if {[string first $needle $data] >= 0} { return 1 }
    return 0
}

proc log_tail {logfile n} {
    if {![file exists $logfile]} { return }
    set fh [open $logfile r]
    set lines [split [read $fh] "\n"]
    close $fh
    set cnt [llength $lines]
    set start [expr {$cnt - $n}]
    if {$start < 0} { set start 0 }
    puts "----- $logfile (tail) -----"
    for {set i $start} {$i < $cnt} {incr i} {
        puts [lindex $lines $i]
    }
}

# ---------------------------------------------------------------- environment
foreach f [list cfc_weight_init.svh cfc_lut_init.svh cfc_rom_params.svh \
                x_input.hex y_expected.hex h_expected.hex] {
    if {![file exists [file join $gen_dir $f]]} {
        puts "CFC_SIM_FAIL missing export/gen/$f"
        exit 1
    }
}
set xvlog [find_tool xvlog]
set xelab [find_tool xelab]
set xsim  [find_tool xsim]
if {($xvlog eq "") || ($xelab eq "") || ($xsim eq "")} {
    puts "CFC_SIM_FAIL xsim tools not found (source settings64.bat first)"
    exit 1
}
file mkdir $log_dir
cd $sim_dir
foreach p [list xsim.dir .Xil] {
    if {[file exists $p]} { file delete -force $p }
}

puts "CFC_SIM: root=$root"
puts "CFC_SIM: xvlog=$xvlog"
puts "CFC_SIM: compile rtl/*.sv + tb_*.sv (include ../export/gen)"
set rc [run_cmd [list cmd /c $xvlog -sv -i ../export/gen -f filelist.f] [file join $log_dir compile.log]]
if {($rc != 0) || [log_has [file join $log_dir compile.log] "ERROR:"]} {
    log_tail [file join $log_dir compile.log] 40
    puts "CFC_SIM_FAIL compile"
    exit 1
}
puts "CFC_SIM: compile OK"

set tbs [list [list tb_cfc_cell cfc_cell_sim] \
               [list tb_cfc_seq  cfc_seq_sim] \
               [list tb_cfc_protocol cfc_proto_sim]]
set all_ok 1
foreach pair $tbs {
    set tb   [lindex $pair 0]
    set snap [lindex $pair 1]
    set elog [file join $log_dir "elab_$tb.log"]
    set slog [file join $log_dir "sim_$tb.log"]
    set rc [run_cmd [list cmd /c $xelab -debug typical $tb -s $snap] $elog]
    if {($rc != 0) || ![log_has $elog "Completed static elaboration"]} {
        log_tail $elog 40
        puts "CFC_SIM_FAIL elaborate $tb"
        set all_ok 0
        continue
    }
    set rc [run_cmd [list cmd /c $xsim $snap -runall] $slog]
    set pass_marker [log_has $slog "CFC_TB_PASS"]
    set fail_marker [log_has $slog "CFC_TB_FAIL"]
    if {($rc != 0) || !$pass_marker || $fail_marker} {
        log_tail $slog 60
        puts "CFC_SIM_FAIL simulate $tb"
        set all_ok 0
    } else {
        puts "CFC_SIM: $tb PASS  (log: [file join $log_dir "sim_$tb.log"])"
    }
}

# tidy up waveform dumps xsim writes next to the snapshots
foreach f [glob -nocomplain [file join $sim_dir *.wdb]] { file delete -force $f }
foreach f [glob -nocomplain [file join $sim_dir webtalk*]] { file delete -force $f }

if {$all_ok} {
    puts "CFC_SIM_PASS tb_cfc_cell tb_cfc_seq tb_cfc_protocol"
    exit 0
}
puts "CFC_SIM_FAIL one or more testbenches failed"
exit 1