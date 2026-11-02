import hashlib, json, pathlib
UP = pathlib.Path("D:/dl_v0.1/FPGA_SpikingTransformer-main")
FILES = [
 "hyper_para.v", "Transformer_part/TOP_Transformer.v",
 "Transformer_part/LIF_group.v",
 "Transformer_part/systolic_array/Systolic_pe.v",
 "Transformer_part/systolic_array/SystolicArray.v",
 "Transformer_part/AttnCalc/Attention_PE.v",
 "Transformer_part/AttnCalc/PipelineAdder.v",
 "Transformer_part/AttnCalc/AddTreeUnsigned.v",
 "Transformer_part/AttnMultiplySpikes/LineMac_PE.v",
 "Transformer_part/AttnMultiplySpikes/MM_Calculator.v",
 "Transformer_part/Head/Ram2Head.v",
 "Transformer_part/mlp/mlp_controller.v",
 "Transformer_part/patch_embed/PatchEmbed.v",
 "proj_lif/proj_lif.v", "eyeriss_part/simple_eyeriss_top.v",
 "eyeriss_part/simple_eyeriss_array.v", "PE_dsp_part/SpikingEncoder.v",
 "python_scripts/quant_dorefa.py",
 "python_scripts/SpikformerEncoderBlock_view.py",
 "python_scripts/model_view.py", "python_scripts/model.py",
 "c_src/src/main.c", "c_src/src/post_proc/post_proc.c",
 "IPs_init/init_ips.tcl",
]
rows = {}
for f in FILES:
    p = UP / f
    rows[f] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"
print(json.dumps(rows, indent=1))
