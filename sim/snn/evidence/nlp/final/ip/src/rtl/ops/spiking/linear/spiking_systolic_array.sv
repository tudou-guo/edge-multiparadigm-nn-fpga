// Spiking systolic array ROWSxCOLS of spiking_systolic_pe.
// Upstream: Transformer_part/systolic_array/SystolicArray.v +
//   SystolicController/SystolicController_Slave.v + Mtrx_slice_fifo.
// Diffs: ROWS/COLS/T/SPIKE_BITS/W/PSUM_SLICE are parameters; weight ping-pong
// per column (CalcPtr/LoadPtr broadcast); data skewed one clk per row;
// psum cascades top-to-bottom per column. FIFOs map to XPM in codegen.
module spiking_systolic_array #(
  parameter int ROWS       = 16,
  parameter int COLS       = 16,
  parameter int T          = 4,
  parameter int SPIKE_BITS = 2,
  parameter int W          = 8,
  parameter int PSUM_SLICE = 20,
  parameter int DATA_W     = T*SPIKE_BITS,
  parameter int PSUM_W     = T*PSUM_SLICE
)(
  input  logic s_clk,
  input  logic s_rst,
  input  logic weight_LoadPtr,
  input  logic weight_CalcPtr,
  input  logic [COLS-1:0] weight_valid_col,
  input  logic signed [W-1:0] weights_col [COLS],
  input  logic [ROWS-1:0] in_data_valid_row,
  input  logic [DATA_W-1:0] in_raw_data_row [ROWS],
  output logic [COLS-1:0] out_psum_valid_col,
  output logic signed [PSUM_W-1:0] out_psum_data_col [COLS]
);
  logic row_v [ROWS+1][COLS];
  logic [DATA_W-1:0] row_d [ROWS+1][COLS];
  logic col_pv [ROWS+1][COLS];
  logic signed [PSUM_W-1:0] col_ps [ROWS+1][COLS];
  genvar r, c;
  generate
    for (c = 0; c < COLS; c = c + 1) begin : g_top
      assign col_pv[0][c] = 1'b0;
      assign col_ps[0][c] = 0;
    end
    for (r = 0; r < ROWS; r = r + 1) begin : g_rows
      for (c = 0; c < COLS; c = c + 1) begin : g_cols
        logic rv_in; logic [DATA_W-1:0] rd_in;
        if (c == 0) begin : g_first
          assign rv_in = in_data_valid_row[r];
          assign rd_in = in_raw_data_row[r];
        end else begin : g_chain
          assign rv_in = row_v[r+1][c-1];
          assign rd_in = row_d[r+1][c-1];
        end
        // one-clk row skew keeps wavefront aligned (upstream data_gen equivalent)
        logic rv_q; logic [DATA_W-1:0] rd_q;
        always_ff @(posedge s_clk) begin
          rv_q <= rv_in;
          rd_q <= rd_in;
        end
        spiking_systolic_pe #(
          .T(T), .SPIKE_BITS(SPIKE_BITS), .W(W), .PSUM_SLICE(PSUM_SLICE)
        ) u_pe (
          .s_clk(s_clk), .s_rst(s_rst),
          .weight_LoadPtr(weight_LoadPtr), .weight_CalcPtr(weight_CalcPtr),
          .weight_valid(weight_valid_col[c]), .weights(weights_col[c]),
          .in_data_valid(rv_q), .in_raw_data(rd_q),
          .out_data_valid(row_v[r+1][c]), .out_raw_data(row_d[r+1][c]),
          .in_psum_data(col_ps[r][c]),
          .out_psum_valid(col_pv[r+1][c]), .out_psum_data(col_ps[r+1][c])
        );
      end
    end
    for (c = 0; c < COLS; c = c + 1) begin : g_out
      assign out_psum_valid_col[c] = col_pv[ROWS][c];
      assign out_psum_data_col[c]  = col_ps[ROWS][c];
    end
  endgenerate
endmodule
