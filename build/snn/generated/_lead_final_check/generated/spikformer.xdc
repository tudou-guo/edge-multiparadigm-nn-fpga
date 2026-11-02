# GENERATED constraints: one clock, NO timing exceptions, no downclock.
create_clock -name s_clk -period 5.000 [get_ports s_clk]
set_input_delay -clock s_clk 0.000 [all_inputs]
set_output_delay -clock s_clk 0.000 [all_outputs]
