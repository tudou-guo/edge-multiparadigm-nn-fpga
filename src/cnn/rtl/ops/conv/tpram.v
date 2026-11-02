module tpram #(
    parameter DP = 1024,
    parameter DW = 8   ,
    parameter PIPE = 2  
) (
    input                             clk    ,
    input        [DW - 1 : 0]         wdata_0,
    input                             wen_0  ,
    input        [$clog2(DP) - 1 : 0] addr_0 ,
    input                             en_0   ,
    output reg   [DW - 1 : 0]         rdata_0,
    input        [DW - 1 : 0]         wdata_1,
    input                             wen_1  ,
    input        [$clog2(DP) - 1 : 0] addr_1 ,
    input                             en_1   ,
    output reg   [DW - 1 : 0]         rdata_1
);

reg [DW - 1:0] mem [DP - 1 : 0];

integer i;
initial for (i=0; i<DP; i=i+1) mem[i] = 0;

always @(posedge clk ) begin
    if(wen_0 & en_0)begin
        mem[addr_0] <= wdata_0;
    end
end

generate
    if(PIPE == 2)begin
        reg [DW - 1:0] rdata_temp_0;
        always @(posedge clk ) begin
            if(en_0)begin
                rdata_temp_0 <= mem[addr_0];
            end
        end
        
        always @(posedge clk ) begin
            rdata_0 <= rdata_temp_0;
        end
    end
    else begin
        always @(posedge clk ) begin
            if(en_0)begin
                rdata_0 <= mem[addr_0];
            end
        end
    end
endgenerate

always @(posedge clk ) begin
    if(wen_1 & en_1)begin
        mem[addr_1] <= wdata_1;
    end
end

generate
    if(PIPE == 2)begin
        reg [DW - 1:0] rdata_temp_1;
        always @(posedge clk ) begin
            if(en_1)begin
                rdata_temp_1 <= mem[addr_1];
            end
        end
        
        always @(posedge clk ) begin
            rdata_1 <= rdata_temp_1;
        end
    end
    else begin
        always @(posedge clk ) begin
            if(en_1)begin
                rdata_1 <= mem[addr_1];
            end
        end
    end
endgenerate
    
endmodule