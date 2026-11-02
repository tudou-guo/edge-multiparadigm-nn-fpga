"""Generate golden vectors for the small integer chain replay.

Case: E32/H1/HD32/seq4/T2/MLP64/depth2, V64, full top ABI
(embed -> 2 blocks -> sum), two consecutive samples with different token ids
(no global reset), two blocks with different weights/thresholds.

Output: build/spikformer/nlp_48h_20261029_1915/replay/case_small/
Golden truth: tools/backends/spikformer/integer_ref.py (cross-checked; result
recorded in golden.json). Contract: contract.json (hardware_abi).
Toolbox TB: verification/sim/spikformer/tb_nlp_small.v
Driver: scripts/spikformer/run_nlp_sim.py --case small
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gen_nlp_common as G


def main():
    G.generate('small')


if __name__ == '__main__':
    main()
