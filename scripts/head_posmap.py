"""Позиционная карта головы: топ-1/H/категории предсказаний по позициям окна.
Проверяет гипотезу obs 15: конец окна — вне обучающего распределения (там пунктуация),
середина — содержательные предсказания (слова).
"""
import os, sys, math, re

import torch
import torch.nn.functional as F
from torch.serialization import add_safe_globals

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'scripts'))

from core import WideBindConfig, WideBindStack
from generate import load_russian_tokenizer

add_safe_globals([WideBindConfig])
torch.set_num_threads(4)

LOG_PATH = os.path.join(os.environ.get('TEMP', BASE), 'opencode', 'posmap.log')
log = open(LOG_PATH, 'w', encoding='utf-8')

PUNCT = re.compile(r'^[\s\.,:;!?\-—–…"«»()\[\]{}]+$')
WORD = re.compile(r'[а-яёА-ЯЁ]')


def w(s=''):
    log.write(s + '\n')
    log.flush()


def load(ckpt_path):
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = state.get('cfg', WideBindConfig())
    m = WideBindStack(cfg)
    m.load_state_dict(state['model'], strict=False)
    if m.explicit_reasoning:
        m.reasoning_enabled_step = int(state.get('reasoning_enabled_step', 0))
    m.eval()
    return m, state


def position_map(m, tok, text, n_windows, L, temp_base=0.8):
    ids = tok.encode(text).ids
    log_temp = m.lm_head.log_temp.data.mean().item()
    t_eff = temp_base / math.exp(log_temp)
    stats = {}
    for start in range(0, min(len(ids) - L, n_windows * L), L):
        window = torch.tensor(ids[start:start + L], dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            h = m.embed_tokens(window)
            out, _, _, _ = m(h, None, adaptive=False)
            logits = m.lm_head(out[0])  # (L, V)
        pr = F.softmax((logits.double() / t_eff), dim=-1)
        top1 = pr.max(dim=-1)
        H = -(pr * torch.log2(pr.clamp_min(1e-12))).sum(dim=-1)
        for pos in range(L):
            tid = int(top1.indices[pos])
            tstr = tok.decode([tid]).strip()
            cat = 'punct' if PUNCT.match(tstr) else ('word' if WORD.search(tstr) else 'other')
            d = stats.setdefault(pos, {'top1': 0.0, 'H': 0.0, 'cnt': 0, 'punct': 0, 'word': 0})
            d['top1'] += top1.values[pos].item()
            d['H'] += H[pos].item()
            d['cnt'] += 1
            if cat == 'punct':
                d['punct'] += 1
            elif cat == 'word':
                d['word'] += 1
    return stats, t_eff, log_temp


def report(name, stats, t_eff, log_temp, L):
    w(f'--- {name} | log_temp={log_temp:.4f} t_eff={t_eff:.4f}')
    w('   pos | top-1 avg | H avg | punct% | word%')
    bins = [(0, 0, 15), (16, 15, 33), (32, 31, 65), (64, 63, 129), (128, 127, 193),
            (192, 191, 225), (224, 223, 241), (240, 239, 249), (250, 249, 255)]
    for label, lo, hi in bins:
        top1 = sum(stats[p]['top1'] / stats[p]['cnt'] for p in range(lo, min(hi, L) + 1)) / (hi - lo + 1)
        H = sum(stats[p]['H'] / stats[p]['cnt'] for p in range(lo, min(hi, L) + 1)) / (hi - lo + 1)
        pc = sum(stats[p]['punct'] for p in range(lo, min(hi, L) + 1)) / (hi - lo + 1) * 100
        wc = sum(stats[p]['word'] for p in range(lo, min(hi, L) + 1)) / (hi - lo + 1) * 100
        w(f'   {label:3d} | {top1:.4f} | {H:.3f} | {pc:5.1f}% | {wc:5.1f}%')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--windows', type=int, default=3)
    ap.add_argument('--corpus', type=str, default='')
    ap.add_argument('--temp', type=float, default=0.8)
    args = ap.parse_args()
    tok = load_russian_tokenizer()
    if args.corpus:
        with open(args.corpus, encoding='utf-8') as f:
            text = f.read()
    else:
        text = 'Москва — столица России, и в ней живут миллионы людей. ' * 300
    for ck in args.ckpts:
        m, _ = load(ck)
        L = m.cfg.seq_len
        stats, t_eff, log_temp = position_map(m, tok, text, args.windows, L, args.temp)
        report(ck, stats, t_eff, log_temp, L)
    log.close()
    print('log:', LOG_PATH)