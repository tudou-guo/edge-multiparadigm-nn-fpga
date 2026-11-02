# GENERATED project assembly (IP packaging / interactive use); heavy start
# only under the shared heavy lock via scripts/spikformer/run_physical.py.
set snn_root {D:\DL_FINAL\SNN}
set script_dir [file normalize [file dirname [info script]]]
set proj_dir [lindex $argv 0]
if {$proj_dir eq ""} { set proj_dir [file join $script_dir .. project] }
create_project -force spikformer_nlp $proj_dir -part xczu15eg-ffvb1156-2-i
set srcs {}
foreach s {D:/DL_FINAL/SNN/rtl/ops/spiking/attention/spiking_attention_pe.sv D:/DL_FINAL/SNN/rtl/ops/spiking/attention/spiking_linemac_pe.sv D:/DL_FINAL/SNN/rtl/ops/spiking/attention/spiking_popcount.sv D:/DL_FINAL/SNN/rtl/ops/spiking/control/spikformer_block_sequencer.sv D:/DL_FINAL/SNN/rtl/ops/spiking/control/spikformer_nlp_top.sv D:/DL_FINAL/SNN/rtl/ops/spiking/control/spiking_block_top.sv D:/DL_FINAL/SNN/rtl/ops/spiking/embed/spiking_embed_top.sv D:/DL_FINAL/SNN/rtl/ops/spiking/include/spikformer_cfg.vh D:/DL_FINAL/SNN/rtl/ops/spiking/lif/spiking_lif_cell.sv D:/DL_FINAL/SNN/rtl/ops/spiking/lif/spiking_lif_group.sv D:/DL_FINAL/SNN/rtl/ops/spiking/linear/spiking_systolic_array.sv D:/DL_FINAL/SNN/rtl/ops/spiking/linear/spiking_systolic_pe.sv D:/DL_FINAL/SNN/rtl/ops/spiking/mlp/spiking_mlp_controller.sv D:/DL_FINAL/SNN/rtl/ops/spiking/reduce/spiking_head_reduce.sv D:/DL_FINAL/SNN/rtl/ops/spiking/sps/spiking_adder_tree.sv D:/DL_FINAL/SNN/rtl/ops/spiking/sps/spiking_eyeriss_pe.sv D:/DL_FINAL/SNN/rtl/ops/spiking/sps/spiking_maxpool2x2.sv D:/DL_FINAL/SNN/rtl/ops/spiking/sps/spiking_sps_top.sv D:/DL_FINAL/SNN/rtl/ops/spiking/wrappers/spikformer_platform_wrapper.sv D:/DL_FINAL/SNN/rtl/ops/spiking/sps/spiking_feature_bram.sv D:/DL_FINAL/SNN/rtl/ops/spiking/sps/spiking_rgb_encoder.sv D:/DL_FINAL/SNN/rtl/ops/spiking/sps/spiking_eyeriss_array.sv D:/DL_FINAL/SNN/rtl/ops/spiking/sps/spiking_sps_psum.sv D:/DL_FINAL/SNN/rtl/ops/spiking/sps/spiking_sps_ref_ctrl.sv D:/DL_FINAL/SNN/rtl/ops/spiking/sps/spiking_sps_token_adapter.sv D:/DL_FINAL/SNN/rtl/ops/spiking/sps/spiking_sps_ref_top.sv D:/DL_FINAL/SNN/rtl/ops/spiking/control/spikformer_vis_native_top.sv D:/DL_FINAL/SNN/rtl/ops/spiking/control/spikformer_vis_top.sv D:/DL_FINAL/SNN/build/_lead_final_check/generated/spikformer_nlp_target_top.sv} { lappend srcs $s }
add_files -norecurse -fileset sources_1 $srcs
set_property top spikformer_nlp_target_top [current_fileset]
add_files -norecurse -fileset constrs_1 [file join $script_dir spikformer.xdc]
update_compile_order -fileset sources_1
