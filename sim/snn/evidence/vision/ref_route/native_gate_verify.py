# -*- coding: utf-8 -*-
# Independent Python diff of the r4 dumped beats vs exp.hex (PASS evidence).
import io, collections, os
VD='D:/dl_snn/build/spikformer/vision_ref_20261031/vectors/sps_native_tc2_8'
SIM='D:/dl_snn/build/spikformer/vision_ref_20261031/sim'
exp=[int(l.strip(),16) for l in io.open(os.path.join(VD,'exp.hex')) if l.strip()]
got=[]
for l in io.open(os.path.join(SIM,'tb_sps_native_gotbeats_imgnative_20261031.hex')):
    l=l.strip()
    if not l: continue
    v=int(l,16); words=[]
    for k in range(8):
        w=0
        for t in range(4): w |= ((v>>(4*(k*4+t)))&1)<<t
        words.append(w)
    p=0
    for k,w in enumerate(words): p |= (w&0xF)<<(4*(7-k))
    got.append(p)
print('exp beats=%d got beats=%d' % (len(exp), len(got)))
exact=sum(1 for a,b in zip(got,exp) if a==b)
ham=sum(bin(a^b).count('1') for a,b in zip(got,exp))
print('exact beats=%d/%d' % (exact, len(exp)))
print('total hamming nibbles and bits=%d' % ham)
# also verify no nibble outside 0/1 in raw dump
badn=0
for l in io.open(os.path.join(SIM,'tb_sps_native_gotbeats_imgnative_20261031.hex')):
    l=l.strip()
    if not l: continue
    v=int(l,16)
    for nib in range(32):
        if ((v>>(4*nib))&0xF) not in (0,1): badn+=1
print('nibbles outside {0,1}=%d' % badn)
