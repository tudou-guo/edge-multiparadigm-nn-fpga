`timescale 1ns / 1ps
`include "npu_backend_contract.vh"
// Parameterized 3x3 INT8 PE.
// Tn/Tm are elaboration-time lane counts: this module only instantiates
// 9*Tn*Tm multiply-accumulate lanes. The core issues 16x16 ABI groups as
// ordered micro-tiles and accumulates their partial sums.
module conv3x3_calc_balanced #(
    parameter integer Tn = 16,
    parameter integer Tm = 16,
    parameter integer WEIGHT_MODE = 0, parameter integer PRODUCT_IMPL = 0,
    // Internal static-gather mode (plan sections 4/5).  Defaults keep the
    // original runtime dynamic gather so nothing outside the optimized core
    // path can change.  STATIC_DDR_GATHER=1 is only legal for the MAC3/DDR
    // backend with an aligned 4/8/16 tile; the core sets it explicitly.
    parameter integer STATIC_DDR_GATHER = 0,
    parameter integer GATHER_IN_SPAN = 16,
    parameter integer GATHER_OUT_SPAN = 16
) (
    input wire clk,
    input wire rst_n,
    input wire in_valid,
    input wire [3:0] in_lane_base,
    input wire [3:0] out_lane_base,
    input wire [63:0] act_data_0, input wire [63:0] act_data_1,
    input wire [63:0] act_data_2, input wire [63:0] act_data_3,
    input wire [63:0] act_data_4, input wire [63:0] act_data_5,
    input wire [63:0] act_data_6, input wire [63:0] act_data_7,
    input wire [63:0] act_data_8,
    input wire [63:0] act_data_b_0, input wire [63:0] act_data_b_1,
    input wire [63:0] act_data_b_2, input wire [63:0] act_data_b_3,
    input wire [63:0] act_data_b_4, input wire [63:0] act_data_b_5,
    input wire [63:0] act_data_b_6, input wire [63:0] act_data_b_7,
    input wire [63:0] act_data_b_8,
    input wire [511:0] weight [0:3][0:8],
    input wire signed [7:0] local_weight [0:8],
    // M4 K1 eight-group reuse.  k1_mode is a run-time mode select; the
    // per-token block-valid mask is bit[2*i + a] = ABI block a of input
    // subgroup i carries real channels.  The widened 2*Tm output bank only
    // needs the high half for the 16/16 profile.
    input wire k1_mode,
    input wire [7:0] k1_grp_valid,
    output wire [511:0] sum_packed,
    output wire [255:0] wsum_packed,
    output wire [511:0] sum_packed_hi,
    output wire [255:0] wsum_packed_hi,
    output wire out_valid
);
    // Fully parameterized pairwise reduction.  The product stage is followed
    // by ceil(log2(9*TN)) registered adder levels; no serial 9*TN accumulator.
    function integer ceil_log2;
        input integer value;
        integer p, levels;
        begin
            p = 1; levels = 0;
            while (p < value) begin p = p << 1; levels = levels + 1; end
            ceil_log2 = levels;
        end
    endfunction
    localparam integer TN = (Tn < 1) ? 1 : ((Tn > 16) ? 16 : Tn);
    localparam integer TM = (Tm < 1) ? 1 : ((Tm > 16) ? 16 : Tm);
    localparam integer TERMS = 9 * TN;
    localparam integer TREE_LEVELS = ceil_log2(TERMS);
    localparam integer TREE_LEAVES = (1 << TREE_LEVELS);
    localparam integer EXTRA_PRODUCT_LATENCY = (PRODUCT_IMPL == 0) ? 0 : `NPU_DSP_PRODUCT_EXTRA_LATENCY;
    localparam integer LATENCY_CYCLES = TREE_LEVELS + `NPU_TREE_BASE_LATENCY + EXTRA_PRODUCT_LATENCY;
    localparam integer SUM_WIDTH = 16 + TREE_LEVELS;
    localparam integer WEIGHT_SUM_WIDTH = 8 + TREE_LEVELS;
    // M4 K1 eight-group reuse.  taps 0..7 become the physical groups
    // g = 2*i + j (i = input subgroup, j = output subgroup); tap 8 is unused.
    // TREE_LEVELS = 4 + log2(Tn) for Tn in {8,16}, so tree[TREE_LEVELS-4]
    // is the tap layer: node k is the sum of tap k's Tn product leaves.
    localparam integer K1_LEVEL = TREE_LEVELS - 4;
    localparam integer K1_LEVEL_OK = ((1 << K1_LEVEL) == TN) ? 1 : 0;
    localparam integer K1_LEGAL = ((TN == TM) && ((TN == 8) || (TN == 16)) &&
                                   (K1_LEVEL_OK != 0)) ? 1 : 0;
    // One spare adder bit per reduction level; the four-node sum still fits
    // the 32-bit output lane with room to spare.
    localparam integer K1_SUM_W = SUM_WIDTH + 2;
    localparam integer K1_WS_W = WEIGHT_SUM_WIDTH + 2;
    // Illegal parameter sets silently keep the K3 path; simulation asserts
    // instead of returning a wrong reduction.
    wire k1_en = k1_mode && (K1_LEGAL != 0);

    // ------------------------------------------------------------------
    // Static weight/activation gather (plan sections 4.2 and 5.2).
    //
    // The old path selected, per PE and per clock, one byte out of the whole
    // 4-face weight array and one activation lane out of the 16-lane bus with
    // a runtime variable part-select.  For the MAC3/DDR backend the set of
    // legal (in_lane_base,out_lane_base) pairs is a small aligned set, so the
    // mapping I/O/FACE/BYTE is elaboration-constant and only a <=4-candidate
    // fixed slice selection remains.  The mapping is:
    //     I = bn*TN + n,  O = bm*TM + m
    //     FACE = 2*(O/8) + (I/8),  BYTE = 8*(I%8) + (O%8)
    //     W = weight[FACE][t][8*BYTE +: 8], A = {act_b[t],act_a[t]}[8*I +: 8]
    // ------------------------------------------------------------------
    localparam integer MAP_PARAMS_OK =
        (Tn == TN) && (Tm == TM) &&
        ((TN == 4) || (TN == 8) || (TN == 16)) &&
        ((TM == 4) || (TM == 8) || (TM == 16)) &&
        ((GATHER_IN_SPAN == 8) || (GATHER_IN_SPAN == 16)) &&
        ((GATHER_OUT_SPAN == 8) || (GATHER_OUT_SPAN == 16)) &&
        (TN <= GATHER_IN_SPAN) && (TM <= GATHER_OUT_SPAN) &&
        ((GATHER_IN_SPAN % TN) == 0) && ((GATHER_OUT_SPAN % TM) == 0) &&
        (((GATHER_IN_SPAN / TN) * (GATHER_OUT_SPAN / TM)) <= 4);
    // A request that names an illegal span is never silently downgraded: the
    // optimized path simply does not engage (and the sizing assertions below
    // mark it unverified), while the core-level entry rejects illegal tiles.
    localparam integer USE_STATIC = (STATIC_DDR_GATHER != 0) && MAP_PARAMS_OK &&
                                    (PRODUCT_IMPL != 0) && (WEIGHT_MODE == 0);
    // Safety-only dimensions: keep the disabled branch free of zero-length arrays.
    localparam integer NB_I = MAP_PARAMS_OK ? (GATHER_IN_SPAN / TN) : 1;
    localparam integer NB_O = MAP_PARAMS_OK ? (GATHER_OUT_SPAN / TM) : 1;
    localparam integer NC = NB_I * NB_O;

    wire [127:0] act_bus [0:8];
    wire [NB_I-1:0] choose_i;
    wire [NB_O-1:0] choose_o;
    wire [7:0] picked_act [0:8][0:TN-1];
    wire signed [7:0] picked_weight [0:8][0:TN-1][0:TM-1];

    genvar gi, gj, gt, gn, gm, gq;
    generate
      if (USE_STATIC) begin : GEN_STATIC_GATHER
        assign act_bus[0] = {act_data_b_0, act_data_0};
        assign act_bus[1] = {act_data_b_1, act_data_1};
        assign act_bus[2] = {act_data_b_2, act_data_2};
        assign act_bus[3] = {act_data_b_3, act_data_3};
        assign act_bus[4] = {act_data_b_4, act_data_4};
        assign act_bus[5] = {act_data_b_5, act_data_5};
        assign act_bus[6] = {act_data_b_6, act_data_6};
        assign act_bus[7] = {act_data_b_7, act_data_7};
        assign act_bus[8] = {act_data_b_8, act_data_8};
        for (gi = 0; gi < NB_I; gi = gi + 1) begin : GI_DECODE
          localparam [3:0] BASE = gi * TN;
          if (NB_I == 1) assign choose_i[gi] = 1'b1;
          else assign choose_i[gi] = (in_lane_base == BASE);
        end
        for (gj = 0; gj < NB_O; gj = gj + 1) begin : GO_DECODE
          localparam [3:0] BASE = gj * TM;
          if (NB_O == 1) assign choose_o[gj] = 1'b1;
          else assign choose_o[gj] = (out_lane_base == BASE);
        end
        for (gt = 0; gt < 9; gt = gt + 1) begin : TAP
          for (gn = 0; gn < TN; gn = gn + 1) begin : IN_LANE
            wire [7:0] a_chain [0:NB_I];
            assign a_chain[0] = 8'd0;
            for (gi = 0; gi < NB_I; gi = gi + 1) begin : A_CHOICE
              localparam integer I = gi * TN + gn;
              assign a_chain[gi+1] = a_chain[gi] |
                                     (act_bus[gt][8*I +: 8] & {8{choose_i[gi]}});
            end
            assign picked_act[gt][gn] = a_chain[NB_I];
            for (gm = 0; gm < TM; gm = gm + 1) begin : OUT_LANE
              wire [7:0] masked [0:NC-1];
              wire [7:0] w_chain [0:NC];
              assign w_chain[0] = 8'd0;
              for (gi = 0; gi < NB_I; gi = gi + 1) begin : W_IB
                for (gj = 0; gj < NB_O; gj = gj + 1) begin : W_OB
                  localparam integer I = gi * TN + gn;
                  localparam integer O = gj * TM + gm;
                  localparam integer FACE = 2*(O/8) + (I/8);
                  localparam integer BYTE = 8*(I%8) + (O%8);
                  localparam integer Q = gi * NB_O + gj;
                  assign masked[Q] = weight[FACE][gt][8*BYTE +: 8] &
                                     {8{choose_i[gi] & choose_o[gj]}};
                end
              end
              for (gq = 0; gq < NC; gq = gq + 1) begin : W_OR
                assign w_chain[gq+1] = w_chain[gq] | masked[gq];
              end
              assign picked_weight[gt][gn][gm] = $signed(w_chain[NC]);
            end
          end
        end
        // Simulation-only configuration assertion: a valid token must carry a
        // legal aligned base on every multi-candidate axis.  A wrong base must
        // not be silently truncated onto a different candidate.
        // synthesis translate_off
        always @(posedge clk) begin
          if (in_valid) begin
            if ((NB_I > 1) && (^choose_i !== 1'b1))
              $error("CP_STATIC_GATHER: illegal in_lane_base %0d on a valid token",
                     in_lane_base);
            if ((NB_O > 1) && (^choose_o !== 1'b1))
              $error("CP_STATIC_GATHER: illegal out_lane_base %0d on a valid token",
                     out_lane_base);
          end
        end
        // synthesis translate_on
      end else begin : GEN_STATIC_DISABLED
        assign choose_i = {NB_I{1'b0}};
        assign choose_o = {NB_O{1'b0}};
        for (gt = 0; gt < 9; gt = gt + 1) begin : TAP0
          for (gn = 0; gn < TN; gn = gn + 1) begin : IN0
            assign picked_act[gt][gn] = 8'd0;
            for (gm = 0; gm < TM; gm = gm + 1) begin : OUT0
              assign picked_weight[gt][gn][gm] = 8'sd0;
            end
          end
        end
      end
    endgenerate

    reg [7:0] act_ff [0:8][0:TN-1];
    reg signed [7:0] weight_ff [0:8][0:TN-1][0:TM-1];
    reg input_valid_ff;
    reg [TREE_LEVELS+1+EXTRA_PRODUCT_LATENCY:0] valid_pipe;
    reg signed [SUM_WIDTH-1:0] sum_tree [0:TREE_LEVELS][0:TREE_LEAVES-1][0:15];
    reg signed [WEIGHT_SUM_WIDTH-1:0] weight_tree [0:TREE_LEVELS][0:TREE_LEAVES-1][0:15];
    reg signed [31:0] sum_out [0:15];
    reg signed [15:0] weight_out [0:15];
    // K1 eight-group reduction pipeline.  Five registers from the tap layer to
    // the output lane, identical in depth to the K3 path
    // tree[K1_LEVEL] -> ... -> sum_out, so lanes land on the out_valid edge.
    reg signed [K1_SUM_W-1:0] k1_add0 [0:3][0:15];
    reg signed [K1_SUM_W-1:0] k1_add1 [0:1][0:15];
    reg signed [K1_SUM_W-1:0] k1_dly1 [0:1][0:15];
    reg signed [K1_SUM_W-1:0] k1_dly2 [0:1][0:15];
    reg signed [K1_SUM_W-1:0] k1_dly3 [0:1][0:15];
    reg signed [K1_WS_W-1:0]  k1_wadd0 [0:3][0:15];
    reg signed [K1_WS_W-1:0]  k1_wadd1 [0:1][0:15];
    reg signed [K1_WS_W-1:0]  k1_wdly1 [0:1][0:15];
    reg signed [K1_WS_W-1:0]  k1_wdly2 [0:1][0:15];
    reg signed [K1_WS_W-1:0]  k1_wdly3 [0:1][0:15];

    wire signed [SUM_WIDTH-1:0] product_leaf [0:TREE_LEAVES-1][0:15];
    wire signed [15:0] dsp_product [0:TERMS-1][0:TM-1];
    reg signed [15:0] weight_delay [0:`NPU_DSP_PRODUCT_EXTRA_LATENCY-1][0:15];
    integer delay_i, delay_o;
    genvar dt, dp, dl, dout;
    generate
      if(PRODUCT_IMPL != 0)begin : GEN_DSP_PAIR_PRODUCTS
        for(dt=0;dt<TERMS;dt=dt+1)begin : TERM
          for(dp=0;dp<(TM+1)/2;dp=dp+1)begin : PAIR
            wire signed [7:0] second_weight;
            wire signed [15:0] pa,pb;
            if(2*dp+1<TM)assign second_weight=weight_ff[dt/TN][dt%TN][2*dp+1];
            else assign second_weight=8'sd0;
            signed_int8_dsp2_mac #(.DSP_LATENCY(`NPU_DSP_PRIMITIVE_LATENCY),.ACTIVATION_SIGNED(0)) u_product(
              .clk(clk),.rst_n(rst_n),.in_valid(input_valid_ff),
              .weight_a(weight_ff[dt/TN][dt%TN][2*dp]),.weight_b(second_weight),
              .activation($signed(act_ff[dt/TN][dt%TN])),
              .out_valid(),.product_a(pa),.product_b(pb));
            assign dsp_product[dt][2*dp]=pa;
            if(2*dp+1<TM)assign dsp_product[dt][2*dp+1]=pb;
          end
        end
      end
      for(dl=0;dl<TREE_LEAVES;dl=dl+1)begin : GEN_PRODUCT_LEAF
        for(dout=0;dout<16;dout=dout+1)begin : OUT
          if(PRODUCT_IMPL==0)assign product_leaf[dl][dout]=sum_tree[0][dl][dout];
          else if(dl<TERMS && dout<TM)
            assign product_leaf[dl][dout]={{(SUM_WIDTH-16){dsp_product[dl][dout][15]}},dsp_product[dl][dout]};
          else assign product_leaf[dl][dout]=0;
        end
      end
      if(PRODUCT_IMPL != 0)begin : GEN_WEIGHT_SUM_ALIGN
        always @(posedge clk)begin
          if(!rst_n)begin
            for(delay_i=0;delay_i<`NPU_DSP_PRODUCT_EXTRA_LATENCY;delay_i=delay_i+1)
              for(delay_o=0;delay_o<16;delay_o=delay_o+1)weight_delay[delay_i][delay_o]<=0;
          end else begin
            for(delay_o=0;delay_o<16;delay_o=delay_o+1)weight_delay[0][delay_o]<=weight_out[delay_o];
            for(delay_i=1;delay_i<`NPU_DSP_PRODUCT_EXTRA_LATENCY;delay_i=delay_i+1)
              for(delay_o=0;delay_o<16;delay_o=delay_o+1)weight_delay[delay_i][delay_o]<=weight_delay[delay_i-1][delay_o];
          end
        end
      end
    endgenerate

    // M4 K1 wsum phase.  The K1 reduction taps the weight tree at K1_LEVEL,
    // but in the DSP backend the product tree runs
    // NPU_DSP_PRODUCT_EXTRA_LATENCY cycles behind the weight tree (the K3
    // output compensates at weight_out with GEN_WEIGHT_SUM_ALIGN).  Delay the
    // tap-layer weight sums by the same amount, so the K1 sum and wsum lanes
    // always come from the same token and land on the out_valid edge in both
    // backend modes.  Behavioural backend: zero delay.
    wire signed [WEIGHT_SUM_WIDTH-1:0] k1_wtap [0:7][0:15];
    genvar gk1, gk1o;
    generate
        if (EXTRA_PRODUCT_LATENCY == 0) begin : GEN_K1_WTAP_DIRECT
            for (gk1 = 0; gk1 < 8; gk1 = gk1 + 1)
                for (gk1o = 0; gk1o < 16; gk1o = gk1o + 1)
                    assign k1_wtap[gk1][gk1o] = weight_tree[K1_LEVEL][gk1][gk1o];
        end else begin : GEN_K1_WTAP_DELAY
            reg signed [WEIGHT_SUM_WIDTH-1:0] k1_wtap_dly [0:`NPU_DSP_PRODUCT_EXTRA_LATENCY-1][0:7][0:15];
            integer wd_i, wd_k, wd_o;
            always @(posedge clk) begin
                if (!rst_n) begin
                    for (wd_i = 0; wd_i < `NPU_DSP_PRODUCT_EXTRA_LATENCY; wd_i = wd_i + 1)
                        for (wd_k = 0; wd_k < 8; wd_k = wd_k + 1)
                            for (wd_o = 0; wd_o < 16; wd_o = wd_o + 1)
                                k1_wtap_dly[wd_i][wd_k][wd_o] <= {WEIGHT_SUM_WIDTH{1'b0}};
                end else begin
                    for (wd_k = 0; wd_k < 8; wd_k = wd_k + 1)
                        for (wd_o = 0; wd_o < 16; wd_o = wd_o + 1)
                            k1_wtap_dly[0][wd_k][wd_o] <= weight_tree[K1_LEVEL][wd_k][wd_o];
                    for (wd_i = 1; wd_i < `NPU_DSP_PRODUCT_EXTRA_LATENCY; wd_i = wd_i + 1)
                        for (wd_k = 0; wd_k < 8; wd_k = wd_k + 1)
                            for (wd_o = 0; wd_o < 16; wd_o = wd_o + 1)
                                k1_wtap_dly[wd_i][wd_k][wd_o] <= k1_wtap_dly[wd_i-1][wd_k][wd_o];
                end
            end
            for (gk1 = 0; gk1 < 8; gk1 = gk1 + 1)
                for (gk1o = 0; gk1o < 16; gk1o = gk1o + 1)
                    assign k1_wtap[gk1][gk1o] =
                        k1_wtap_dly[`NPU_DSP_PRODUCT_EXTRA_LATENCY-1][gk1][gk1o];
        end
    endgenerate

    integer tap_i, in_i, out_i, lane_i, lane_o;
    integer term_i, level_i, node_i;

    function signed [SUM_WIDTH-1:0] product_s8_u8;
        input signed [7:0] w;
        input [7:0] a;
        reg signed [8:0] a_ext;
        begin
            a_ext = $signed({1'b0, a});
            product_s8_u8 = $signed(w) * a_ext;
        end
    endfunction
    function [7:0] act_at;
        input integer tap;
        input integer lane;
        begin
            case (tap)
                0: act_at = (lane < 8) ? act_data_0[lane*8 +: 8] : act_data_b_0[(lane-8)*8 +: 8];
                1: act_at = (lane < 8) ? act_data_1[lane*8 +: 8] : act_data_b_1[(lane-8)*8 +: 8];
                2: act_at = (lane < 8) ? act_data_2[lane*8 +: 8] : act_data_b_2[(lane-8)*8 +: 8];
                3: act_at = (lane < 8) ? act_data_3[lane*8 +: 8] : act_data_b_3[(lane-8)*8 +: 8];
                4: act_at = (lane < 8) ? act_data_4[lane*8 +: 8] : act_data_b_4[(lane-8)*8 +: 8];
                5: act_at = (lane < 8) ? act_data_5[lane*8 +: 8] : act_data_b_5[(lane-8)*8 +: 8];
                6: act_at = (lane < 8) ? act_data_6[lane*8 +: 8] : act_data_b_6[(lane-8)*8 +: 8];
                7: act_at = (lane < 8) ? act_data_7[lane*8 +: 8] : act_data_b_7[(lane-8)*8 +: 8];
                default: act_at = (lane < 8) ? act_data_8[lane*8 +: 8] : act_data_b_8[(lane-8)*8 +: 8];
            endcase
        end
    endfunction

    function signed [7:0] ddr_weight_at;
        input integer tap;
        input integer in_lane;
        input integer out_lane;
        begin
            if ((in_lane < 8) && (out_lane < 8))
                ddr_weight_at = $signed(weight[0][tap][((in_lane*8)+out_lane)*8 +: 8]);
            else if ((in_lane >= 8) && (out_lane < 8))
                ddr_weight_at = $signed(weight[1][tap][(((in_lane-8)*8)+out_lane)*8 +: 8]);
            else if ((in_lane < 8) && (out_lane >= 8))
                ddr_weight_at = $signed(weight[2][tap][((in_lane*8)+(out_lane-8))*8 +: 8]);
            else
                ddr_weight_at = $signed(weight[3][tap][(((in_lane-8)*8)+(out_lane-8))*8 +: 8]);
        end
    endfunction

    // K1 tail mask: weight lane (tap, in_lane) is live only while its ABI input
    // block is inside the real Cin.  The unused ninth tap never contributes, so
    // bit[8] is not decoded (the core masks that face as well).
    function k1_weight_lane_en;
        input [7:0] valid;
        input integer tap;
        input integer lane;
        begin
            if (tap >= 8)
                k1_weight_lane_en = 1'b0;
            else
                k1_weight_lane_en = valid[2*(tap/2) + (lane/8)];
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            input_valid_ff <= 1'b0;
            valid_pipe <= {(TREE_LEVELS+2+EXTRA_PRODUCT_LATENCY){1'b0}};
            for (out_i = 0; out_i < 16; out_i = out_i + 1) begin
                sum_out[out_i] <= 32'sd0;
                weight_out[out_i] <= 16'sd0;
                k1_add0[0][out_i] <= {K1_SUM_W{1'b0}};
                k1_add0[1][out_i] <= {K1_SUM_W{1'b0}};
                k1_add0[2][out_i] <= {K1_SUM_W{1'b0}};
                k1_add0[3][out_i] <= {K1_SUM_W{1'b0}};
                k1_add1[0][out_i] <= {K1_SUM_W{1'b0}};
                k1_add1[1][out_i] <= {K1_SUM_W{1'b0}};
                k1_dly1[0][out_i] <= {K1_SUM_W{1'b0}};
                k1_dly1[1][out_i] <= {K1_SUM_W{1'b0}};
                k1_dly2[0][out_i] <= {K1_SUM_W{1'b0}};
                k1_dly2[1][out_i] <= {K1_SUM_W{1'b0}};
                k1_dly3[0][out_i] <= {K1_SUM_W{1'b0}};
                k1_dly3[1][out_i] <= {K1_SUM_W{1'b0}};
                k1_wadd0[0][out_i] <= {K1_WS_W{1'b0}};
                k1_wadd0[1][out_i] <= {K1_WS_W{1'b0}};
                k1_wadd0[2][out_i] <= {K1_WS_W{1'b0}};
                k1_wadd0[3][out_i] <= {K1_WS_W{1'b0}};
                k1_wadd1[0][out_i] <= {K1_WS_W{1'b0}};
                k1_wadd1[1][out_i] <= {K1_WS_W{1'b0}};
                k1_wdly1[0][out_i] <= {K1_WS_W{1'b0}};
                k1_wdly1[1][out_i] <= {K1_WS_W{1'b0}};
                k1_wdly2[0][out_i] <= {K1_WS_W{1'b0}};
                k1_wdly2[1][out_i] <= {K1_WS_W{1'b0}};
                k1_wdly3[0][out_i] <= {K1_WS_W{1'b0}};
                k1_wdly3[1][out_i] <= {K1_WS_W{1'b0}};
            end
        end else begin
            input_valid_ff <= in_valid;
            for (tap_i = 0; tap_i < 9; tap_i = tap_i + 1)
                for (in_i = 0; in_i < TN; in_i = in_i + 1) begin
                    if (k1_en) begin
                        // K1: lane base is forced to 0.  Physical group
                        // g = 2*i + j carries input block i and output block j,
                        // so lane (tap, n, m) is group channel (i*Tn+n,
                        // j*Tm+m); a tail input block and the unused ninth tap
                        // are masked to an exact zero weight.
                        act_ff[tap_i][in_i] <= act_at(tap_i, in_i);
                        for (out_i = 0; out_i < TM; out_i = out_i + 1)
                            weight_ff[tap_i][in_i][out_i] <=
                                k1_weight_lane_en(k1_grp_valid, tap_i, in_i) ?
                                    ddr_weight_at(tap_i, in_i, out_i) : 8'sd0;
                    end else if (USE_STATIC) begin
                        // Constant-candidate path: no runtime lane arithmetic and
                        // no variable part-select survive synthesis.
                        act_ff[tap_i][in_i] <= picked_act[tap_i][in_i];
                        for (out_i = 0; out_i < TM; out_i = out_i + 1)
                            weight_ff[tap_i][in_i][out_i] <= picked_weight[tap_i][in_i][out_i];
                    end else begin
                        lane_i = in_lane_base + in_i;
                        act_ff[tap_i][in_i] <= (lane_i < 16) ? act_at(tap_i, lane_i) : 8'd0;
                        for (out_i = 0; out_i < TM; out_i = out_i + 1) begin
                            lane_o = out_lane_base + out_i;
                            if ((lane_i >= 16) || (lane_o >= 16))
                                weight_ff[tap_i][in_i][out_i] <= 8'sd0;
                            else if (WEIGHT_MODE == 0)
                                weight_ff[tap_i][in_i][out_i] <= ddr_weight_at(tap_i, lane_i, lane_o);
                            else if ((lane_i == 0) && (lane_o == 0))
                                weight_ff[tap_i][in_i][out_i] <= local_weight[tap_i];
                            else
                                weight_ff[tap_i][in_i][out_i] <= 8'sd0;
                        end
                    end
                end

            valid_pipe[0] <= input_valid_ff;
            for (level_i = 1; level_i <= TREE_LEVELS + 1 + EXTRA_PRODUCT_LATENCY; level_i = level_i + 1)
                valid_pipe[level_i] <= valid_pipe[level_i-1];

            // Stage 0: one registered signed-int8 x unsigned-int8 product
            // per term; pad inactive leaves and output lanes with exact zero.
            for (term_i = 0; term_i < TREE_LEAVES; term_i = term_i + 1)
                for (out_i = 0; out_i < 16; out_i = out_i + 1) begin
                    if ((term_i < TERMS) && (out_i < TM)) begin
                        if (PRODUCT_IMPL == 0) sum_tree[0][term_i][out_i] <= product_s8_u8(
                            weight_ff[term_i / TN][term_i % TN][out_i],
                            act_ff[term_i / TN][term_i % TN]);
                        weight_tree[0][term_i][out_i] <=
                            {{(WEIGHT_SUM_WIDTH-8){weight_ff[term_i / TN][term_i % TN][out_i][7]}},
                              weight_ff[term_i / TN][term_i % TN][out_i]};
                    end else begin
                        sum_tree[0][term_i][out_i] <= {SUM_WIDTH{1'b0}};
                        weight_tree[0][term_i][out_i] <= {WEIGHT_SUM_WIDTH{1'b0}};
                    end
                end

            // One balanced pairwise reduction level per clock.
            for (level_i = 1; level_i <= TREE_LEVELS; level_i = level_i + 1)
                for (node_i = 0; node_i < (TREE_LEAVES >> level_i); node_i = node_i + 1)
                    for (out_i = 0; out_i < 16; out_i = out_i + 1) begin
                        if(level_i==1)
                            sum_tree[level_i][node_i][out_i] <= product_leaf[2*node_i][out_i]+product_leaf[2*node_i+1][out_i];
                        else
                            sum_tree[level_i][node_i][out_i] <=
                                sum_tree[level_i-1][2*node_i][out_i]+sum_tree[level_i-1][2*node_i+1][out_i];
                        weight_tree[level_i][node_i][out_i] <=
                            weight_tree[level_i-1][2*node_i][out_i] +
                            weight_tree[level_i-1][2*node_i+1][out_i];
                    end

            // K1 tap layer -> two independent output subgroups.  tree level
            // K1_LEVEL holds one node per tap, and group g = 2*i + j, so
            //   j0 = node0 + node2 + node4 + node6
            //   j1 = node1 + node3 + node5 + node7
            // Two registered adder levels plus three delay levels are exactly
            // the five K3 registers below tree[K1_LEVEL] (four tree levels to
            // tree[TREE_LEVELS] plus sum_out), so K1 lanes land on out_valid.
            for (out_i = 0; out_i < TM; out_i = out_i + 1) begin
                k1_add0[0][out_i] <= sum_tree[K1_LEVEL][0][out_i] + sum_tree[K1_LEVEL][2][out_i];
                k1_add0[1][out_i] <= sum_tree[K1_LEVEL][4][out_i] + sum_tree[K1_LEVEL][6][out_i];
                k1_add0[2][out_i] <= sum_tree[K1_LEVEL][1][out_i] + sum_tree[K1_LEVEL][3][out_i];
                k1_add0[3][out_i] <= sum_tree[K1_LEVEL][5][out_i] + sum_tree[K1_LEVEL][7][out_i];
                k1_add1[0][out_i] <= k1_add0[0][out_i] + k1_add0[1][out_i];
                k1_add1[1][out_i] <= k1_add0[2][out_i] + k1_add0[3][out_i];
                k1_dly1[0][out_i] <= k1_add1[0][out_i];
                k1_dly1[1][out_i] <= k1_add1[1][out_i];
                k1_dly2[0][out_i] <= k1_dly1[0][out_i];
                k1_dly2[1][out_i] <= k1_dly1[1][out_i];
                k1_dly3[0][out_i] <= k1_dly2[0][out_i];
                k1_dly3[1][out_i] <= k1_dly2[1][out_i];
                k1_wadd0[0][out_i] <= k1_wtap[0][out_i] + k1_wtap[2][out_i];
                k1_wadd0[1][out_i] <= k1_wtap[4][out_i] + k1_wtap[6][out_i];
                k1_wadd0[2][out_i] <= k1_wtap[1][out_i] + k1_wtap[3][out_i];
                k1_wadd0[3][out_i] <= k1_wtap[5][out_i] + k1_wtap[7][out_i];
                k1_wadd1[0][out_i] <= k1_wadd0[0][out_i] + k1_wadd0[1][out_i];
                k1_wadd1[1][out_i] <= k1_wadd0[2][out_i] + k1_wadd0[3][out_i];
                k1_wdly1[0][out_i] <= k1_wadd1[0][out_i];
                k1_wdly1[1][out_i] <= k1_wadd1[1][out_i];
                k1_wdly2[0][out_i] <= k1_wdly1[0][out_i];
                k1_wdly2[1][out_i] <= k1_wdly1[1][out_i];
                k1_wdly3[0][out_i] <= k1_wdly2[0][out_i];
                k1_wdly3[1][out_i] <= k1_wdly2[1][out_i];
            end

            // Final output register keeps Tn=8 latency equal to the legacy
            // calculator while other Tn values report their derived latency.
            for (out_i = 0; out_i < 16; out_i = out_i + 1) begin
                sum_out[out_i] <= sum_tree[TREE_LEVELS][0][out_i];
                weight_out[out_i] <= weight_tree[TREE_LEVELS][0][out_i];
            end
        end
    end

    assign out_valid = valid_pipe[TREE_LEVELS+1+EXTRA_PRODUCT_LATENCY];
    genvar g;
    generate
        for (g = 0; g < 16; g = g + 1) begin : GEN_OUTPUT
            // K3 (k1_en=0) keeps the historic mapping unchanged.  K1 output
            // lane L = j*TM + m: L < 16 lives here, L >= 16 in the high bank.
            if (g < TM) begin : GEN_OUT_J0
                assign sum_packed[g*32 +: 32] = k1_en ?
                    {{(32-K1_SUM_W){k1_dly3[0][g][K1_SUM_W-1]}}, k1_dly3[0][g]} : sum_out[g];
                assign wsum_packed[g*16 +: 16] = k1_en ? k1_wdly3[0][g][15:0] :
                    ((PRODUCT_IMPL==0) ? weight_out[g] : weight_delay[`NPU_DSP_PRODUCT_EXTRA_LATENCY-1][g]);
            end else if ((g >= TM) && (g < 16) && (g < (2*TM))) begin : GEN_OUT_J1
                // TM=16 (B) never reaches this branch: lanes 16..31 are all
                // in sum_packed_hi, so sum_packed must not be indexed past
                // lane 15.  Only TM<16 (A) puts J1 into the low bank.
                assign sum_packed[g*32 +: 32] = k1_en ?
                    {{(32-K1_SUM_W){k1_dly3[1][g-TM][K1_SUM_W-1]}}, k1_dly3[1][g-TM]} : sum_out[g];
                assign wsum_packed[g*16 +: 16] = k1_en ? k1_wdly3[1][g-TM][15:0] :
                    ((PRODUCT_IMPL==0) ? weight_out[g] : weight_delay[`NPU_DSP_PRODUCT_EXTRA_LATENCY-1][g]);
            end else begin : GEN_OUT_UNUSED
                assign sum_packed[g*32 +: 32] = k1_en ? 32'sd0 : sum_out[g];
                assign wsum_packed[g*16 +: 16] = k1_en ? 16'sd0 :
                    ((PRODUCT_IMPL==0) ? weight_out[g] : weight_delay[`NPU_DSP_PRODUCT_EXTRA_LATENCY-1][g]);
            end
        end
        for (g = 0; g < 16; g = g + 1) begin : GEN_OUTPUT_HI
            if ((g + 16) < (2*TM)) begin : GEN_OUT_HI_J1
                assign sum_packed_hi[g*32 +: 32] = k1_en ?
                    {{(32-K1_SUM_W){k1_dly3[1][g+16-TM][K1_SUM_W-1]}}, k1_dly3[1][g+16-TM]} : 32'sd0;
                assign wsum_packed_hi[g*16 +: 16] = k1_en ? k1_wdly3[1][g+16-TM][15:0] : 16'sd0;
            end else begin : GEN_OUT_HI_ZERO
                assign sum_packed_hi[g*32 +: 32] = 32'sd0;
                assign wsum_packed_hi[g*16 +: 16] = 16'sd0;
            end
        end
    endgenerate

    // synthesis translate_off
    // K1 is only defined for the square 8/8 and 16/16 physical profiles (the
    // reduction reads the tap layer tree[TREE_LEVELS-4]) with DDR weights.  An
    // illegal mode request must fail loudly instead of yielding a silently
    // wrong reduction.
    always @(posedge clk) begin
        if (rst_n && in_valid && k1_mode) begin
            if ((K1_LEGAL == 0) || (WEIGHT_MODE != 0))
                $error("CP_K1_REUSE: illegal k1_mode (Tn=%0d Tm=%0d WEIGHT_MODE=%0d)",
                       Tn, Tm, WEIGHT_MODE);
        end
    end
    // synthesis translate_on
endmodule


