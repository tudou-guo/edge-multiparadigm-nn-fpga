"""Text -> SNN hardware inference -> sentiment, on PYNQ.

Tokenisation matches the trained corpus contract: char level, keep the first
SEQ chars, right-pad with 0 (<pad>), unknown -> 1 (<unk>).
The PL runs the whole network; the PS applies the FP32 head (sums/(T*N) then
Linear), identical to tools/backends/spikformer/integer_ref.py.

Assets are JSON + a plain numeric .npz on purpose: no pickle, so any numpy
version can load them.
"""
import io
import json
import numpy as np
from pynq import Overlay
import snn_platform_run as S

SEQ = 32
PAD, UNK = 0, 1
T_TIMES_N = 128      # head_norm = ps_div, T*N = 4*32
ROT = 64             # sums read-window offset of this bitstream build
BAR_W = 24


def _bar(p, width=BAR_W):
    n = int(round(max(0.0, min(1.0, p)) * width))
    return '█' * n + '·' * (width - n)


def _level(c):
    return '高' if c >= 0.80 else ('中' if c >= 0.60 else '低')


class SnnNLP:
    def __init__(self, vocab_json='demo_vocab.json', head_npz='demo_head.npz',
                 samples_json='demo_samples.json', bit='snn_platform.bit',
                 vecdir='vec/case_real'):
        v = json.load(io.open(vocab_json, encoding='utf-8'))
        self.chars = [str(c) for c in v['chars']]
        self.c2i = {c: i for i, c in enumerate(self.chars)}
        z = np.load(head_npz, allow_pickle=False)
        self.head_w = np.asarray(z['head_w'], np.float32)
        self.head_b = np.asarray(z['head_b'], np.float32)
        s = json.load(io.open(samples_json, encoding='utf-8'))
        self.demo_texts = [d['text'] for d in s['samples']]
        self.demo_labels = [int(d['label']) for d in s['samples']]
        self.seq_len = int(v.get('seq_len', SEQ))
        self.bit = bit
        self.vecdir = S.Path(vecdir).resolve()

    # ---------------------------------------------------------------- core
    def tokenize(self, text):
        ids = [self.c2i.get(c, UNK) for c in text][:self.seq_len]
        return np.asarray(ids + [PAD] * (self.seq_len - len(ids)), dtype=np.uint16)

    def predict(self, text, timeout_s=180.0):
        Overlay(self.bit)                      # reload per call: the PL does not return to idle
        dev = S.SnnPlatform()
        ident, ver = dev.id_check()
        dev.load_sample(self.vecdir, 0)        # tables + weights (sample independent)
        dev.write_ddr(S.IN_OFF + S.TAB_BYTES, self.tokenize(text))
        dev.program()
        cycles = dev.start_and_wait(timeout_s=timeout_s)
        sums = np.asarray([dev.rd(S.CSR['SUMS'] + 4 * ((j + ROT) % 128))
                           for j in range(128)], dtype=np.float32)
        pooled = sums / np.float32(T_TIMES_N)
        logits = pooled @ self.head_w.T + self.head_b
        pr = np.exp(logits - logits.max())
        pr /= pr.sum()
        lab = int(np.argmax(logits))
        return {'label': 'positive' if lab else 'negative', 'cls': lab,
                'conf': float(pr[lab]), 'probs': pr, 'logits': logits,
                'cycles': int(cycles), 'id': '0x%08X' % ident}

    # ---------------------------------------------------------------- pretty
    def show(self, text, timeout_s=180.0):
        r = self.predict(text, timeout_s=timeout_s)
        pn, pp = float(r['probs'][0]), float(r['probs'][1])
        verdict = '正面 / POSITIVE' if r['cls'] == 1 else '负面 / NEGATIVE'
        print('┌' + '─' * 56)
        print('│ 输入：' + text)
        print('├' + '─' * 56)
        print('│ 负面 ' + _bar(pn) + ' %5.1f%%' % (pn * 100))
        print('│ 正面 ' + _bar(pp) + ' %5.1f%%' % (pp * 100))
        print('├' + '─' * 56)
        print('│ 结论：%s   置信度 %.1f%%（%s）' % (verdict, r['conf'] * 100, _level(r['conf'])))
        print('└' + '─' * 56 + '  PL cycles=%d' % r['cycles'])
        return r

    def show_many(self, texts=None, labels=None):
        texts = self.demo_texts if texts is None else texts
        labels = self.demo_labels if labels is None else labels
        print('%-3s %-30s %-8s %-8s %s' % ('#', '句子', '模型判断', '置信度', '人工标注'))
        print('-' * 76)
        hit = n = 0
        rows = []
        for i, t in enumerate(texts, 1):
            r = self.predict(t)
            exp = labels[i - 1] if labels and i - 1 < len(labels) else None
            tag = ''
            if exp is not None:
                n += 1
                ok = (r['cls'] == exp)
                hit += int(ok)
                tag = ('✓ ' if ok else '✗ ') + ('正面' if exp else '负面')
            short = (t[:14] + '..') if len(t) > 16 else t
            print('%-3d %-30s %-8s %-7.1f%% %s' % (i, short, '正面' if r['cls'] else '负面',
                                                    r['conf'] * 100, tag))
            rows.append(r)
        print('-' * 76)
        if n:
            print('这 %d 句里判对 %d 句（演示句是挑出来的，不代表整体准确率）' % (n, hit))
        return rows
