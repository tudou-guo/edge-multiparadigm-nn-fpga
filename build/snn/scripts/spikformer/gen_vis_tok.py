import io, pathlib
import os
ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[2]).resolve()
# golden SPS tokens -> flat [token][channel][t] file for the TB
lines = [l.strip() for l in io.open(str(ROOT / 'build/spikformer/sps_replay/exp.hex')) if l.strip()]
NTOK, E, T, LANES = 4, 32, 2, 8
G2 = E // LANES
tok = [[[0]*T for _ in range(E)] for _ in range(NTOK)]
for tb, line in enumerate(lines):
    g, pos = tb % G2, tb // G2
    for a in range(LANES):
        bits = int(line[a], 16)
        for t in range(T):
            tok[pos][g*LANES+a][t] = (bits >> t) & 1
out = pathlib.Path(str(ROOT / 'build/spikformer/vis_replay'))
out.mkdir(parents=True, exist_ok=True)
with io.open(out/'exp_tok.hex', 'w') as f:
    for n in range(NTOK):
        for c in range(E):
            for t in range(T):
                f.write('%x\n' % tok[n][c][t])
print('wrote exp_tok.hex', NTOK*E*T)
print('token0 ch0..3:', tok[0][0], tok[0][1], tok[0][2], tok[0][3])
