// T-step LIF group: sequential evaluation of the T timesteps of one input
// beat. Upstream LIF_group chains T proj_lif cells over T cycles; this port
// keeps exactly that recurrence but drives it from a simple counter so the
// membrane never leaks between beats.
//   pre  = 0 at t=0
//   mem  = (pre + delta_t) >>> 1        (arithmetic shift, floor)
//   spike= valid && mem >= 0 && mem >= THR
//   pre  = spike ? 0 : mem
// Latency T+1 cycles; one-cycle o_spikes_valid pulse after the last step.
//
// 48h NLP contract fix (2026-10-29): pre+delta is sign-extended to BITS+1 bits
// BEFORE the add and the arithmetic shift, so a 32-bit accumulator can never
// wrap before the >>>1. Spike/mem semantics and the T+1 latency are unchanged.
module spiking_lif_group #(
  parameter int T    = 4,
  parameter int BITS = 20
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic signed [BITS-1:0] i_lif_thrd,
  input  logic i_PsumValid,
  input  logic signed [T*BITS-1:0] i_PsumData,
  output logic [T-1:0] o_spikes_out,
  output logic o_spikes_valid
);
  logic [T*BITS-1:0] w_hold;
  logic [$clog2(T+1)-1:0] cnt;
  logic busy;
  logic signed [BITS-1:0] pre;
  logic [T-1:0] spk;
  logic signed [BITS-1:0] d;
  logic signed [BITS:0] sum_ext;      // BITS+1 bit sum, no 32-bit wrap
  logic signed [BITS-1:0] mem;
  logic hit;
  logic [T-1:0] spk_next;
  always_comb begin
    d = w_hold[cnt*BITS +: BITS];
    sum_ext = {pre[BITS-1], pre} + {d[BITS-1], d};
    mem = sum_ext[BITS:1];            // floor((pre+d)/2), arithmetic shift
    hit = !mem[BITS-1] && (mem >= i_lif_thrd);
    spk_next = spk;
    spk_next[cnt] = hit;
  end
  always_ff @(posedge s_clk or posedge s_rst) begin
    if (s_rst) begin
      w_hold <= 0; cnt <= 0; busy <= 1'b0; pre <= 0; spk <= 0;
      o_spikes_valid <= 1'b0; o_spikes_out <= 0;
    end else if (!busy) begin
      o_spikes_valid <= 1'b0;
      if (i_PsumValid) begin
        w_hold <= i_PsumData;
        cnt <= 0;
        busy <= 1'b1;
        pre <= 0;
        spk <= 0;
      end
    end else begin
      spk <= spk_next;
      pre <= hit ? '0 : mem;
      if (cnt == T-1) begin
        busy <= 1'b0;
        o_spikes_out <= spk_next;
        o_spikes_valid <= 1'b1;
      end else begin
        cnt <= cnt + 1'b1;
      end
    end
  end
endmodule
