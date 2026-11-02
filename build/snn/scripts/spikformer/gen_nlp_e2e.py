"""Generate golden vectors for the frozen-contract target end-to-end replay.

Two modes:

  fixture mode (default) - development fixture TARGET case E128/H4/HD32/seq32/
      MLP512/depth2/T4, V4000, full frozen top stream ABI, two consecutive samples
      (different sentences, PAD=0 to fixed length 32, no global reset):
        python scripts/spikformer/gen_nlp_e2e.py
      -> build/spikformer/nlp_48h_20261029_1915/replay/case_e2e/

  external-checkpoint mode (final gate: real_weight_replay) - loads a trained .npz
      strictly through tools/backends/spikformer/integer_ref.py, takes 1-3 FIXED
      samples (token ids from the final tokenizer/encoded dev+holdout set) and
      writes the same stream/exp_sums/exp_out vectors + head logits/class:
        python scripts/spikformer/gen_nlp_e2e.py --checkpoint <real.npz>
            --tokens-file <samples.json> [--vocab <vocab.json>] [--samples N]
      -> build/spikformer/nlp_48h_20261029_1915/replay/case_real/
      golden.json records checkpoint sha256 + vocab sha256 + tokens source.

tokens file formats: .json {"samples": [[...], ...], "names": [...]} or [[...],...];
.npy int array [n_samples, seq_len]; .hex/.txt one sample per line.
Toolbox TB: verification/sim/spikformer/tb_nlp_e2e.v
Driver:     scripts/spikformer/run_nlp_sim.py --case e2e   (fixture)
            scripts/spikformer/run_nlp_sim.py --case real  (checkpoint vectors)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gen_nlp_common as G


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None,
                    help="external trained .npz loaded strictly via integer_ref.load_checkpoint")
    ap.add_argument("--tokens-file", default=None,
                    help="fixed token-id samples (json/npy/hex); required with --checkpoint")
    ap.add_argument("--vocab", default=None,
                    help="vocab.json (recorded sha256 provenance)")
    ap.add_argument("--samples", type=int, default=3,
                    help="max fixed samples to use from the tokens file (1-3)")
    ap.add_argument("--out-kind", default="real", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.checkpoint:
        if not a.tokens_file:
            raise SystemExit("--checkpoint requires --tokens-file with the fixed samples")
        G.generate_real(a.checkpoint, a.tokens_file, a.vocab,
                        out_kind=a.out_kind, max_samples=a.samples)
    else:
        if a.tokens_file or a.vocab:
            raise SystemExit("--tokens-file/--vocab are only valid with --checkpoint")
        G.generate("e2e")


if __name__ == "__main__":
    main()
