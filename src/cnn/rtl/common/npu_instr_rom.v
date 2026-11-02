// -----------------------------------------------------------------------------
// npu_instr_rom.v
//
// Reusable compile-time instruction ROM for the project_3 mixed NPU engine.
//
// It is NOT model-specific: the whole 32-bit word stream lives in an external
// hex file (one word per line, no data width qualifier) so the same primitives
// are reused for every generated network.  The npu_gen generator writes both
// the <net>_instr.mem payload and the top-level parameter that points at it.
//
// Framing (continuous variable-length word stream, reference-project style):
//   word[0]      = 32'hFFFFFFFF  start marker
//   word[1]      = word_count    number of words that follow
//   word[2..]    = records, each = {op_type[31:16], payload_words[15:0]} + payload
//
// Address space / word count is a compile-time parameter (INSTR_WORDS), never a
// runtime CSR.  capacity_is_runtime_csr stays false for the whole engine.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

`ifndef NPU_INSTR_ROM_INIT
`define NPU_INSTR_ROM_INIT "instr_rom.mem"
`endif

module npu_instr_rom #(
    parameter integer     INSTR_WORDS = 256,
    parameter integer     ADDR_W      = 8,
    parameter string      INIT_FILE   = `NPU_INSTR_ROM_INIT
) (
    input  wire [ADDR_W-1:0] addr,
    output reg  [31:0]       dout
);

    reg [31:0] mem [0:INSTR_WORDS-1];
    integer i;

    initial begin
        for (i = 0; i < INSTR_WORDS; i = i + 1) begin
            mem[i] = 32'hFFFFFFFF;
        end
        $readmemh(INIT_FILE, mem);
    end

    // Combinational (distributed-ROM) read: the sequencer resolves one word per
    // clock with no hidden latency, which keeps the control FSM simple.
    always @(*) begin
        dout = mem[addr];
    end

endmodule

