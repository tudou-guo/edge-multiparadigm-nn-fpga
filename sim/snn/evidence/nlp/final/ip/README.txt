Spikformer NLP IP source package (48h run).
same-source: True (mismatches=0)
part=xczu15eg-ffvb1156-2-i clock=5.000 ns (200 MHz) s_clk
timing_qualified=True (ooc status=pass)
bit=None hwh=None
package_sha256=894599b92c8de7d4edf65320db5f23a1413e7e9fabc81740b357e2134b292f22

SCOPE / ASSUMPTIONS (read before using this package)
- OOC 5.000 ns sign-off only (xczu15eg-ffvb1156-2-i, Vivado 2021.1, no timing exceptions).
- Input delay model is 0.000 ns and HD.CLK_SRC is unset: hold is pessimistic, setup is optimistic.
- Keep the 1,192 DONT_TOUCH LUT1 hold-margin cells (8 stages x 149 sampled input bits incl. s_rst).
- Routed WNS binding path is the hold-margin chain itself (from-port setup slack +0.135 ns under the
  0-delay model); a real upstream max input delay greater than 0.135 ns would violate it - re-constrain
  with real interface min/max and re-time/reduce the chain or register-slice at the integrated top
  before platform use. Not integration-ready timing sign-off.
- platform / bit / HWH / board = not_run.
