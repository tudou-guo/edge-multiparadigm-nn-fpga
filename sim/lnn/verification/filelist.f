# CfC simulation compilation file list (contract 3.5 / script 3.2).
# Paths are relative to the sim/ directory.
# Generated ROM initialisation files (include-only, same list as the Vivado
# synthesis file set; they are included from rtl/*.sv via include_dirs):
#   ../export/gen/cfc_weight_init.svh   (w_rom[0..83], b_rom[0..16])
#   ../export/gen/cfc_lut_init.svh      (tanh_rom[0..1024], sigmoid_rom[0..1024])
#   ../export/gen/cfc_rom_params.svh    (CFC_DT_INT, CFC_N_STEPS_MAX, ...)
# Include search path:
#   xsim   : xvlog -sv -i ../export/gen -f filelist.f
#   Vivado : set_property include_dirs [list $root/export/gen] [get_filesets sources_1]
# The eight RTL files below are exactly the CfC synthesis source list; the three
# testbenches are simulation-only.
../rtl/cfc_mac.sv
../rtl/cfc_weight_rom.sv
../rtl/cfc_activation_rom.sv
../rtl/cfc_state_regs.sv
../rtl/cfc_cell_core.sv
../rtl/cfc_sequence_ctrl.sv
../rtl/cfc_axi_lite_regs.sv
../rtl/cfc_platform_wrapper.sv
tb_cfc_cell.sv
tb_cfc_seq.sv
tb_cfc_protocol.sv