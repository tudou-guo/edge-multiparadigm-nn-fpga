// cfc_ref_model.svh -- bit-exact integer reference for the xsim testbenches.
// Implements docs/CFC_CONTRACT.md V1.1 section 1.3 (48-bit exact accumulate,
// round-half-away-from-zero Q12 rounding, LUT index clamp, V1.1 gate:
//   gate_pre = ROUND( lin_a*dt + (lin_b << 12) ) ).
// The including module must declare:
//   logic signed [15:0] w_rom [0:83];
//   logic signed [31:0] b_rom [0:16];
//   logic signed [15:0] tanh_rom [0:1024];
//   logic signed [15:0] sigmoid_rom [0:1024];
// and include the generated init files before including this file.

function automatic logic signed [15:0] ref_round_q12_sat16 (input logic signed [47:0] v);
  logic [48:0] mag;
  logic [48:0] q;
  logic signed [15:0] r;
  begin
    if (v >= 48'sd0) begin
      q = ({1'b0, v} + 49'd2048) >> 12;
      if (q > 49'd32767) r = 16'sh7FFF;
      else               r = q[15:0];
    end else begin
      mag = -{{1{v[47]}}, v};          // 49-bit magnitude of a negative int48
      q   = (mag + 49'd2048) >> 12;
      if (q >= 49'd32768) r = 16'sh8000;
      else                r = -$signed(q[15:0]);
    end
    ref_round_q12_sat16 = r;
  end
endfunction

function automatic logic [10:0] ref_lut_index (input logic signed [15:0] v);
  logic signed [18:0] t;
  logic [11:0]        idx;
  begin
    t   = {{3{v[15]}}, v} + 19'sd32800;
    idx = t[17:6];
    if (idx > 12'd1024) idx = 12'd1024;
    ref_lut_index = idx[10:0];
  end
endfunction

// One exact integer timestep.  All four branches share the same old state.
task automatic ref_step (
  input  logic signed [15:0] x,
  input  logic signed [15:0] h0,
  input  logic signed [15:0] h1,
  input  logic signed [15:0] h2,
  input  logic signed [15:0] h3,
  input  logic signed [15:0] dtq,
  output logic signed [15:0] yv,
  output logic signed [15:0] hn0,
  output logic signed [15:0] hn1,
  output logic signed [15:0] hn2,
  output logic signed [15:0] hn3
);
  logic signed [15:0] z  [0:4];
  logic signed [15:0] l1 [0:3];
  logic signed [15:0] l2 [0:3];
  logic signed [15:0] la [0:3];
  logic signed [15:0] lb [0:3];
  logic signed [15:0] f1v [0:3];
  logic signed [15:0] f2v [0:3];
  logic signed [15:0] gv [0:3];
  logic signed [15:0] hv [0:3];
  logic signed [15:0] gp;
  logic signed [47:0] acc;
  logic signed [31:0] prd;
  int br, o, i;
  begin
    z[0] = x;
    z[1] = h0;
    z[2] = h1;
    z[3] = h2;
    z[4] = h3;

    // step 1: linear branches (bias Q24 + 5 exact int32 products -> int48)
    for (br = 0; br < 4; br = br + 1) begin
      for (o = 0; o < 4; o = o + 1) begin
        acc = {{16{b_rom[br*4+o][31]}}, b_rom[br*4+o]};
        for (i = 0; i < 5; i = i + 1) begin
          prd = w_rom[br*20 + o*5 + i] * z[i];
          acc = acc + prd;
        end
        case (br)
          0:       l1[o] = ref_round_q12_sat16(acc);
          1:       l2[o] = ref_round_q12_sat16(acc);
          2:       la[o] = ref_round_q12_sat16(acc);
          default: lb[o] = ref_round_q12_sat16(acc);
        endcase
      end
    end

    // steps 2..5: tanh LUTs, gate (V1.1), sigmoid LUT, state fusion
    for (o = 0; o < 4; o = o + 1) begin
      f1v[o] = tanh_rom[ref_lut_index(l1[o])];
      f2v[o] = tanh_rom[ref_lut_index(l2[o])];

      prd = la[o] * dtq;                                    // Q24
      acc = {{16{prd[31]}}, prd};                           // sign-extend before mixing
      acc = acc + {{20{lb[o][15]}}, lb[o], 12'd0};          // + (lin_b << 12), 48-bit
      gp  = ref_round_q12_sat16(acc);
      gv[o] = sigmoid_rom[ref_lut_index(gp)];

      prd = f1v[o] * (16'sd4096 - gv[o]);
      acc = prd;
      prd = f2v[o] * gv[o];
      acc = acc + prd;
      hv[o] = ref_round_q12_sat16(acc);
    end

    // step 6: readout
    acc = {{16{b_rom[16][31]}}, b_rom[16]};
    for (i = 0; i < 4; i = i + 1) begin
      prd = w_rom[80 + i] * hv[i];
      acc = acc + prd;
    end
    yv = ref_round_q12_sat16(acc);

    hn0 = hv[0];
    hn1 = hv[1];
    hn2 = hv[2];
    hn3 = hv[3];
  end
endtask