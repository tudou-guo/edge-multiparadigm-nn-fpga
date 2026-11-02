// GENERATED - do not edit. Per-build include for this run.
`ifndef SPIKFORMER_CONFIG_VH
`define SPIKFORMER_CONFIG_VH
`define SPIK_T 4
`define SPIK_DEPTH 1
`define SPIK_EMBED 384
`define SPIK_HEADS 12
`define SPIK_HEAD_DIM 32
`define SPIK_MLP_HIDDEN 1536
`define SPIK_TOKENS 64
`define SPIK_RW 4
`define SPIK_RES_BITS 4
`define SPIK_RES_BITS_MIN 3
`define SPIK_PSUM_SLICE 32
`define SPIK_LIF_ADD_BITS 33
`define SPIK_AV_BITS 12
`define SPIK_SUM_W 6
`define SPIK_HEAD_NORM 0
// thresholds are runtime parameters carried by the per-block descriptor
// beats (see contract.json hardware_abi), not compile-time constants.
`define SPIK_SPS_H 32
`define SPIK_SPS_W 32
`define SPIK_SPS_C0 3
`define SPIK_SPS_C1 96
`define SPIK_SPS_C2 384
`define SPIK_SPS_TOKENS 64
`define SPIK_SPS_CONV1_BEATS 324
`define SPIK_SPS_CONV2_BEATS 41472
`endif
