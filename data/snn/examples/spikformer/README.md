# Spikformer examples

- model_native_b1.json: native single-block (T4/depth1/E384/heads12/hidden1536/32x32). First bring-up target.
- model_multiblock_b.json: non-default sample (T2/depth2/E192/heads6/hidden768/32x32). Covers T, depth, channel changes.

Compile (explicit fixture dev mode; never random-by-default):

E:/miniconda/python.exe -B -m tools.toolchain --backend spikformer compile --config examples/spikformer/config/model_native_b1.json --fixture-seed 1 --out-dir build/spikformer/native_b1

Real weights (blocked until checkpoint arrives):

E:/miniconda/python.exe -B -m tools.toolchain --backend spikformer compile --config <config> --weights <checkpoint> --out-dir build/spikformer/<run>
