// -----------------------------------------------------------------------------
// op_abi_defs.vh
//
// Single-source Verilog macro definitions for ABI 1.0.0.
//
// These values MUST match docs/op_abi.md. They exist so that op_shell and the
// transport modules do not each invent a second copy of the register/error
// encoding. The ABI generator (not yet implemented) will eventually emit this
// file; until then it is hand maintained and MUST be reviewed with the doc.
//
// Verilog-2001 macros only (no SystemVerilog) so the file can be included by
// both .v and .sv sources.
// -----------------------------------------------------------------------------

`ifndef OP_ABI_DEFS_VH
`define OP_ABI_DEFS_VH

// --- Identity ---------------------------------------------------------------
`define OP_ID                     32'h4F504331
`define OP_ABI_VERSION            32'h00010000
`define OP_LAYOUT_ID              32'h00010101
`define OP_CORE_FEATURES          32'h00000001
`define OP_DEFAULT_TIMEOUT        32'h01000000

// --- Physical parameters ----------------------------------------------------
`define OP_AXIL_ADDR_WIDTH        12
`define OP_AXI_ADDR_WIDTH         32
`define OP_AXI_DATA_WIDTH         128
`define OP_AXI_ID_WIDTH           4
`define OP_BEAT_BYTES             16
`define OP_MAX_BURST_BEATS        16
`define OP_PAGE_BYTES             4096
`define OP_AXI_RID_READ           4'h0
`define OP_AXI_ID_WRITE           4'h1

// --- Error codes (docs/op_abi.md 9.1) --------------------------------------
`define OP_E_NONE                 16'h0000
`define OP_E_CFG                  16'h0001
`define OP_E_UNSUPPORTED          16'h0002
`define OP_E_ADDR_ALIGN           16'h0003
`define OP_E_ADDR_RANGE           16'h0004
`define OP_E_STRIDE               16'h0005
`define OP_E_QUANT                16'h0006
`define OP_E_TIMEOUT              16'h0007
`define OP_E_AXI_RESP             16'h0008
`define OP_E_AXI_PROTOCOL         16'h0009
`define OP_E_CORE_INTERNAL        16'h000A
`define OP_E_ILLEGAL_START        16'h000B
`define OP_E_ILLEGAL_CTRL         16'h000C
`define OP_E_CORE_NOT_IMPLEMENTED 16'h00FF

// --- AXI response encodings -------------------------------------------------
`define OP_AXI_RESP_OKAY          2'b00

`endif // OP_ABI_DEFS_VH
