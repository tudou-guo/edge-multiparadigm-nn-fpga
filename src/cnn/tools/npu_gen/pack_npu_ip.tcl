# pack_npu_ip.tcl -- generated NPU graph -> Vivado IP source package (no synth/sim)
# This packer is reusable for graphs produced by this toolchain's npu_top template.
# It is not a generic packager for arbitrary third-party RTL tops.
#
# OS shell / Vivado batch:
#   vivado -mode batch -source <this-script> -tclargs \
#     <generated_dir> <ip_repo> <ip_name> <version> <source_root> [part] [full|manifest_only]
# If omitted, part defaults to xczu15eg-ffvb1156-2-i (the toolchain target used here).
#
# Contract (S3, 2026-10-24):
#  * Modern compiles bind repo_root in source_root.txt. A supplied 5th argument
#    must match it; legacy compiles require an explicit 5th argument. It is the
#    source tree used to resolve sources.f "rtl/..." entries, include dirs and the
#    DSP XCI.  It no longer defaults to out_dir/../.. (a wrong default silently
#    packed a different candidate than the compile it was bound to).
#  * mode "manifest_only" (7th arg) resolves sources.f and writes
#    <out_dir>/pack_manifest.json (per-file SHA256, sources.f SHA, instr mem SHA)
#    and exits BEFORE any Vivado project/IP command.  It is pure Tcl so the
#    packer file-resolution logic can be validated with a plain tclsh.
#  * "full" (default) keeps the historical packaging behaviour and rewrites
#    pack_manifest.json with ip_dir/component.xml identity after ipx::save_core.
#
# RTL language handling is unchanged: .v files (except the generated top) are
# copied to a staging dir with a .sv extension so Vivado parses them as
# SystemVerilog; RTL content is never modified.
if {$argc < 5} {
  error "USAGE: pack_npu_ip.tcl <generated> <ip_repo> <name> <version> <source_root> [part] [full|manifest_only]"
}
set out_dir [file normalize [lindex $argv 0]]
set ip_repo [file normalize [lindex $argv 1]]
set net_name [lindex $argv 2]
set ip_ver [lindex $argv 3]
set requested_root [lindex $argv 4]
set part "xczu15eg-ffvb1156-2-i"
set mode "full"
if {$argc >= 6} {
  set sixth [string tolower [lindex $argv 5]]
  if {$sixth in {full manifest_only}} {
    set mode $sixth
  } else {
    set part [lindex $argv 5]
  }
}
if {$argc >= 7} { set mode [string tolower [lindex $argv 6]] }
if {$net_name eq "" || $ip_ver eq "" || $part eq ""} {
  error "IP name, version, and FPGA part must be non-empty"
}
if {![regexp {^[A-Za-z0-9_-]+$} $net_name] || ![regexp {^[0-9]+(\.[0-9]+)*$} $ip_ver]} {
  error "unsafe IP name/version"
}
set bound_root ""
set binding_root_file [file join $out_dir source_root.txt]
if {[file exists $binding_root_file]} {
  set fp [open $binding_root_file r]
  set bound_root [file normalize [string trim [read $fp]]]
  close $fp
}
if {$requested_root ne ""} {
  set repo_root [file normalize $requested_root]
} elseif {$bound_root ne ""} {
  set repo_root $bound_root
} else {
  error "compile has no bound source root; supply the source_root argument"
}
if {$bound_root ne "" && ![string equal -nocase $repo_root $bound_root]} {
  error "source root differs from compile binding: requested=$repo_root compiled=$bound_root"
}
if {$mode ni {full manifest_only}} {
  puts "ERROR: mode must be full or manifest_only, got '$mode'"
  exit 1
}
if {![file isdirectory $repo_root]} {
  puts "ERROR: repo_root is not a directory: $repo_root"
  exit 1
}
set ip_tag [format "%s_%s" $net_name $ip_ver]
set top_file [file join $out_dir "npu_top.v"]
set src_f    [file join $out_dir "sources.f"]
if {![file exists $top_file]} { puts "ERROR: missing $top_file"; exit 1 }
if {![file exists $src_f]} { puts "ERROR: missing $src_f"; exit 1 }
puts "REPO_ROOT: $repo_root"
puts "MODE: $mode"

proc sha256_hex {path} {
  # Windows hosts: certutil; MSYS/tclsh validation: sha256sum fallback.
  if {[catch {exec certutil -hashfile $path SHA256} out] == 0} {
    foreach line [split $out "\n"] {
      set compact [string map {" " "" "\t" "" "\r" ""} [string trim $line]]
      if {[regexp {^[0-9A-Fa-f]{64}$} $compact]} { return [string tolower $compact] }
    }
  }
  if {[catch {exec sha256sum $path} out2] == 0} {
    set compact [string tolower [lindex [split [string trim $out2]] 0]]
    if {[regexp {^[0-9a-f]{64}$} $compact]} { return $compact }
  }
  error "cannot compute SHA256 for $path (need certutil or sha256sum)"
}

proc posix_path {path} {
  return [string map [list "\\" "/"] $path]
}

proc json_string {value} {
  set out ""
  foreach char [split $value ""] {
    switch -- $char {
      "\\" { append out "\\\\" }
      "\"" { append out "\\\"" }
      "\n" { append out "\\n" }
      "\r" { append out "\\r" }
      "\t" { append out "\\t" }
      default { append out $char }
    }
  }
  return "\"$out\""
}

proc json_file_row {path sha role staged} {
  set text "    {\"path\": [json_string [posix_path $path]], \"sha256\": [json_string $sha], \"role\": [json_string $role]"
  if {$staged ne ""} { append text ", \"staged\": [json_string [posix_path $staged]]" }
  append text "}"
  return $text
}

proc json_string_row {value} {
  return "    [json_string $value]"
}

proc json_string_array {values} {
  set parts {}
  foreach value $values { lappend parts [json_string $value] }
  set text "\["
  append text [join $parts {, }]
  append text "\]"
  return $text
}

# The modern generator emits plain-data hash rows. Verify before Vivado
# touches the IP repository; never fall back to another source tree.
proc verify_source_binding {} {
  global out_dir repo_root bound_root bound_paths
  set bound_paths [dict create]
  set binding [file join $out_dir source_hashes.tsv]
  if {$bound_root eq ""} { return }
  if {![file exists $binding]} { error "missing source_hashes.tsv" }
  set fp [open $binding r]
  set count 0
  while {[gets $fp line] >= 0} {
    set columns [split [string trim $line] "\t"]
    if {[llength $columns] != 3} { close $fp; error "malformed source binding row" }
    lassign $columns kind expected relative
    if {$kind eq "source"} { set base $repo_root } elseif {$kind eq "generated"} {
      set base $out_dir
    } else { close $fp; error "unknown source binding kind $kind" }
    set path [file normalize [file join $base $relative]]
    if {![string match -nocase "${base}/*" $path]} { close $fp; error "binding path escaped root" }
    if {![file isfile $path] || [sha256_hex $path] ne $expected} {
      close $fp
      error "compiled source changed: $path"
    }
    if {[dict exists $bound_paths $path]} { close $fp; error "duplicate source binding row: $path" }
    dict set bound_paths $path $expected
    incr count
  }
  close $fp
  if {$count == 0} { error "empty source binding table" }
  puts "SOURCE_BINDING_VERIFIED: $count files"
}
verify_source_binding

# ---- resolve sources.f (relative-path based) against the candidate tree ----
set file_entries {}
set seen_paths {}
set skipped_sim {}
set resolution {}
set fp [open $src_f r]
while {[gets $fp line] >= 0} {
  set t [string trim $line]
  if {$t eq ""} { continue }
  if {[string index $t 0] eq "#"} { continue }
  if {[string match "-i *" $t] || [string match "+incdir*" $t]} { continue }
  if {[string match "*sim_netlist*" $t]} {
    lappend skipped_sim $t
    puts "SKIP_SIM: $t"
    continue
  }
  set role "rtl"
  set resolved ""
  if {[string match "*npu_top.v" $t]} {
    set resolved $top_file
    set role "top"
  } else {
    set relative [string map [list "\\" "/"] $t]
    set candidate [file normalize [file join $repo_root $t]]
    if {![string match -nocase "${repo_root}/*" $candidate]} {
      error "source escaped selected root: $t"
    }
    if {[file exists $candidate]} {
      set resolved $candidate
    } else {
      error "sources.f references missing source: $t (repo_root=$repo_root)"
    }
    if {[string match "rtl/ops/activation/*" $relative] || [string match "rtl/ops/depthwise/*" $relative]} {
      set role "new_core"
    }
  }
  if {[lsearch -exact $seen_paths $resolved] >= 0} {
    puts "DEDUPE: $resolved"
    continue
  }
  lappend seen_paths $resolved
  lappend file_entries [list $t $resolved $role]
  lappend resolution "  $role  $t -> $resolved"
}
close $fp
puts "SOURCES_RESOLVED: [llength $file_entries]"
foreach row $resolution { puts $row }

set dsp_xci [file join $repo_root "rtl" "ip" "dsp_mul_pipe4" "dsp_mul_pipe4.xci"]
if {[file exists $dsp_xci]} {
  if {[lsearch -exact $seen_paths [file normalize $dsp_xci]] < 0} {
    lappend seen_paths [file normalize $dsp_xci]
    lappend file_entries [list "rtl/ip/dsp_mul_pipe4/dsp_mul_pipe4.xci" [file normalize $dsp_xci] "xci"]
    puts "DSP_XCI: $dsp_xci"
  }
} else {
  error "required DSP XCI missing: $dsp_xci"
}
set vh [file join $repo_root "rtl" "common" "op_abi_defs.vh"]
if {[file exists $vh] && [lsearch -exact $seen_paths [file normalize $vh]] < 0} {
  lappend seen_paths [file normalize $vh]
  lappend file_entries [list "rtl/common/op_abi_defs.vh" [file normalize $vh] "include"]
}
set instr_files [glob -nocomplain [file join $out_dir "*_instr.mem"]]
if {[llength $instr_files] != 1} {
  error "expected exactly one generated *_instr.mem in $out_dir, found [llength $instr_files]"
}
set instr_mem [lindex $instr_files 0]
if {[lsearch -exact $seen_paths [file normalize $instr_mem]] < 0} {
  lappend seen_paths [file normalize $instr_mem]
  lappend file_entries [list [file tail $instr_mem] [file normalize $instr_mem] "instr_mem"]
}

# A truncated binding must not silently reduce the set of checked inputs.
if {$bound_root ne ""} {
  set must_be_bound [list $src_f [file join $repo_root tools npu_gen npu_top.v.tpl]]
  foreach entry $file_entries { lappend must_be_bound [lindex $entry 1] }
  foreach path $must_be_bound {
    if {![dict exists $bound_paths [file normalize $path]]} {
      error "pack input is absent from compiled source binding: $path"
    }
  }
}

# Duplicate .xci guard: exactly the same XCI must never be added twice.
set xci_paths {}
foreach entry $file_entries {
  lassign $entry rel path role
  if {[file extension $path] eq ".xci"} { lappend xci_paths $path }
}
set xci_duplicates [expr {[llength $xci_paths] - [llength [lsort -unique $xci_paths]]}]
if {$xci_duplicates != 0} {
  error "duplicate .xci entries in pack file list: $xci_paths"
}

# ---- staging plan (.v -> .sv copy, identically to the historical packer) ----
set stage_id [format ".pack_tmp_%s_%s_%s" $ip_tag [clock seconds] [pid]]
set sv_stage [file join $ip_repo $stage_id]
set staged_entries {}
set staged_names [dict create]
foreach entry $file_entries {
  lassign $entry rel path role
  set staged ""
  if {[file extension $path] eq ".v" && $path ne $top_file} {
    set dst [file join $sv_stage "[file rootname [file tail $path]].sv"]
    if {[dict exists $staged_names $dst]} {
      error "two sources.f entries map to the same staged name $dst"
    }
    dict set staged_names $dst 1
    set staged $dst
  }
  lappend staged_entries [list $path $role $staged]
}

# ---- pack_manifest.json (per-file SHA256, sources.f SHA, instr mem SHA) ----
set manifest_path [file join $out_dir "pack_manifest.json"]
set new_core_rows {}
set file_rows {}
set add_files_list {}
set n_rtl 0
set n_new_core 0
set n_xci 0
foreach entry $staged_entries {
  lassign $entry path role staged
  set digest [sha256_hex $path]
  lappend file_rows [json_file_row $path $digest $role $staged]
  if {$role eq "new_core"} { lappend new_core_rows [posix_path $path] }
  if {$role eq "rtl" || $role eq "new_core" || $role eq "top"} { incr n_rtl }
  if {$role eq "new_core"} { incr n_new_core }
  if {$role eq "xci"} { incr n_xci }
  if {$staged ne ""} {
    lappend add_files_list $staged
  } else {
    lappend add_files_list $path
  }
}
set created_local [clock format [clock seconds] -format "%Y-%m-%dT%H:%M:%S%z"]
set body ""
append body "{\n"
append body "  \"schema\": \"e1-npu-ip-pack-manifest-1\",\n"
append body "  \"created_local\": [json_string $created_local],\n"
append body "  \"mode\": [json_string $mode],\n"
append body "  \"out_dir\": [json_string [posix_path $out_dir]],\n"
append body "  \"ip_repo\": [json_string [posix_path $ip_repo]],\n"
append body "  \"net_name\": [json_string $net_name],\n"
append body "  \"ip_version\": [json_string $ip_ver],\n"
append body "  \"part\": [json_string $part],\n"
append body "  \"repo_root\": [json_string [posix_path $repo_root]],\n"
append body "  \"sources_f\": {\"path\": [json_string [posix_path $src_f]], \"sha256\": [json_string [sha256_hex $src_f]]},\n"
append body "  \"top\": {\"path\": [json_string [posix_path $top_file]], \"sha256\": [json_string [sha256_hex $top_file]]},\n"
append body "  \"instr_mem\": {\"path\": [json_string [posix_path $instr_mem]], \"name\": [json_string [file tail $instr_mem]], \"sha256\": [json_string [sha256_hex $instr_mem]]},\n"
append body "  \"counts\": {\"total\": [llength $staged_entries], \"rtl\": $n_rtl, \"new_core\": $n_new_core, \"xci\": $n_xci, \"xci_duplicates\": $xci_duplicates},\n"
set files_joined [join $file_rows ",\n"]
append body "  \"skipped_sim\": [json_string_array $skipped_sim],\n"
append body "  \"new_core_files\": [json_string_array $new_core_rows],\n"
append body "  \"files\": \[\n$files_joined\n  \]\n"
append body "}\n"
set manifest_body $body
set fp [open $manifest_path w]
puts $fp $body
close $fp
puts "PACK_MANIFEST: $manifest_path"
if {$mode eq "manifest_only"} {
  puts "PACK_MANIFEST_ONLY_DONE"
  exit 0
}

# ================= full packaging (Vivado commands below this line) =========
if {[file exists $sv_stage]} { error "staging directory already exists: $sv_stage" }
file mkdir $sv_stage
file mkdir $ip_repo
puts "FILE_COUNT: [llength $staged_entries]"
foreach entry $staged_entries {
  lassign $entry path role staged
  if {$staged ne ""} {
    file copy $path $staged
    puts "SV_COPY: $path -> $staged"
  }
}
create_project -force -in_memory -part $part
add_files -norecurse $add_files_list
set_property include_dirs [list [file join $repo_root "rtl" "common"]] [current_fileset]
set_property top npu_top [current_fileset]
set_property file_type SystemVerilog [get_files $top_file]
update_compile_order -fileset sources_1
set ip_dir [file join $ip_repo $ip_tag]
if {[file exists $ip_dir]} { error "IP output already exists; choose a fresh repository: $ip_dir" }
file mkdir $ip_dir
ipx::package_project -root_dir $ip_dir -vendor user.org -library npu -taxonomy {/UserIP} -import_files -force_update_compile_order -set_current true
set core [ipx::current_core]
set_property vendor user.org $core
set_property library npu $core
set_property name $net_name $core
set_property version $ip_ver $core
set_property display_name "NPU $net_name" $core
set_property description "Generated mixed NPU net top (reusable conv/pool/activation/depthwise engines); model instruction stream and compile-time capacity are generated from ops metadata." $core
ipx::infer_bus_interfaces xilinx.com:interface:aximm_rtl:1.0 $core
ipx::infer_bus_interface clk xilinx.com:signal:clock_rtl:1.0 $core
ipx::infer_bus_interface rst_n xilinx.com:signal:reset_rtl:1.0 $core
ipx::associate_bus_interfaces -busif s_axil -clock clk $core
ipx::associate_bus_interfaces -busif m_axi -clock clk $core
catch {ipx::remove_bus_interface dbg_core_soft_reset $core}
catch {ipx::remove_bus_interface dbg_core_abort $core}
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::save_core $core
ipx::unload_core $core
set component_xml [file join $ip_dir "component.xml"]
if {![file exists $component_xml]} { puts "ERROR: missing $component_xml"; exit 1 }
# rewrite the manifest with the packaged identity (same file, final state)
# Reuse the in-memory manifest body instead of re-reading the freshly written
# file: on this host a second open() of the new file occasionally fails
# (file lock / antivirus), which failed the whole IP pack.
set body $manifest_body
set tail "  \"ip\": {\"ip_dir\": [json_string [posix_path $ip_dir]], \"component_xml\": [json_string [posix_path $component_xml]], \"component_xml_sha256\": [json_string [sha256_hex $component_xml]]}\n"
set idx [string last "\n}" $body]
set body "[string trimright [string range $body 0 $idx-1]],\n$tail}\n"
verify_source_binding
set fp [open $manifest_path w]
puts -nonewline $fp $body
close $fp
file copy -force $manifest_path [file join $ip_repo "$ip_tag.pack_manifest.json"]
puts "IP_PACKED: $ip_dir"
puts "PACK_MANIFEST: $manifest_path"
puts "NEXT: Add this IP repository to the target BD; connect s_axil, m_axi, clk, and rst_n for that board."
