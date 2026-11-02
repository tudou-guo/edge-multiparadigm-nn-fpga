`ifndef NPU_BACKEND_CONTRACT_VH
`define NPU_BACKEND_CONTRACT_VH
// Sole numeric backend/latency contract. Python reads these constants too.
// Latency = cycles after the accepted input edge (same convention as core metadata).
`define NPU_BACKEND_CONTRACT_VERSION 1
`define NPU_MAC_LEGACY 0
`define NPU_MAC_DSP_ADAPTER 1
`define NPU_MAC_BALANCED 2
`define NPU_MAC_BALANCED_DSP 3
`define NPU_FIXED_CALC_LATENCY 9
`define NPU_TREE_BASE_LATENCY 2
`define NPU_DSP_PRODUCT_EXTRA_LATENCY 4
`define NPU_DSP_PRIMITIVE_LATENCY 4
`define NPU_CALC_LATENCY(TN,IMPL) (((IMPL)==`NPU_MAC_BALANCED || (IMPL)==`NPU_MAC_BALANCED_DSP) ? ($clog2(9*(TN)) + `NPU_TREE_BASE_LATENCY + (((IMPL)==`NPU_MAC_BALANCED_DSP) ? `NPU_DSP_PRODUCT_EXTRA_LATENCY : 0)) : `NPU_FIXED_CALC_LATENCY)
`endif
