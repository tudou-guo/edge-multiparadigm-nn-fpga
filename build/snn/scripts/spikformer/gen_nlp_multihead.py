"""Generate golden vectors for the multi-head short replay.

Case: block-level E64/H2/HD32/seq4/T2/MLP64/depth1 (single block), two runs with
head-swapped input spike patterns so any cross-head leakage / head swap changes
the output. Thresholds/wights are calibrated so both heads carry distinct
non-trivial spike patterns (checked in golden.json activity).

Output: build/spikformer/nlp_48h_20261029_1915/replay/case_multihead/
Toolbox TB: verification/sim/spikformer/tb_nlp_multihead.v
Driver: scripts/spikformer/run_nlp_sim.py --case multihead
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gen_nlp_common as G


def main():
    G.generate('multihead')


if __name__ == '__main__':
    main()
