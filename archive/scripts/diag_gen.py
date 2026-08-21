"""Диагностический стенд генерации: численные замеры с PASS/FAIL проверками.

Субкоманды:
  fresh   — eval на свежей загрузке (EMA=1.0, как был чекпоинт) — воспроизведение взрыва
  warm    — train-прогрев EMA (N окон) затем eval — контроль стабильности
  ab      — A/B головы: reasoning OFF / natural / FULL (top-1, H, KL, дивергенция)
  sample  — текстовая генерация (static и adaptive)

Все результаты пишутся в UTF-8 лог (TEMP\\opencode\\diag.log), в консоль — только итог.
"""
import os
import sys
import math
import argparse

import torch
import torch.nn.functional as F
from torch.serialization import add_safe_globals

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'scripts'))

from tokenizers import Tokenizer

from core import WideBindConfig, WideBindStack
from generate import load_russian_tokenizer

add_safe_globals([WideBindConfig])

LOG_PATH = os.path.join(os.environ.get('TEMP', BASE), 'opencode', 'diag.log')
CORPUS_PATH = os.path.join(os.environ.get('TEMP', BASE), 'opencode', 'k3_text.txt')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

device = 'cpu'
torch.set_num_threads(4)


class LOG:
    """UTF-8 лог с дублированием в консоль только итоговых строк."""
    _f = None

    @classmethod
    def open(cls):
        cls._f = open(LOG_PATH, 'w', encoding='utf-8')

    @classmethod
    def w(cls, s='', console=True):
        cls._f.write(s + '\n')
        cls._f.flush()
        if console:
            try:
                print(s, flush=True)
            except UnicodeEncodeError:
                print(s.encode('utf-8', 'replace').decode('utf-8', 'replace'), flush=True)

    @classmethod
    def close(cls):
        cls._f.close()


def load_model(ckpt_path):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get('cfg', WideBindConfig())
    m = WideBindStack(cfg).to(device)
    m.load_state_dict(state['model'], strict=False)
    if m.explicit_reasoning:
        m.reasoning_enabled_step = int(state.get('reasoning_enabled_step', 0))
    return m, state


def load_corpus():
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, encoding='utf-8') as f:
            return f.read()
    return 'Москва — столица России, и в ней живут миллионы людей. ' * 40


def corpus_windows(tok, text, n_windows, L):
    ids = tok.encode(text).ids
    out = []
    for i in range(n_windows):
        start = min(i * L, max(0, len(ids) - L))
        out.append(torch.tensor(ids[start:start + L], dtype=torch.long))
    return out


@torch.no_grad()
def warm_emas(model, tok, n_windows=12, text=None):
    """train-режим + no_grad: EMA нормировки сигналов прогоняется по реальному корпусу."""
    if text is None:
        text = load_corpus()
    L = model.cfg.seq_len
    model.train()
    for i, win in enumerate(corpus_windows(tok, text, n_windows, L)):
        h = model.embed_tokens(win.unsqueeze(0))
        model(h, None, adaptive=False, allow_write=False)
    model.eval()
    ema = model.layers[0].mirror._signal_norm_ema[0]
    return float(ema.mean())


class Observer:
    """Forward-хуки: per-layer hp/gate/out на каждом блоке."""

    def __init__(self, model):
        self.hp = {}
        self.gate = {}
        self.h_out = {}
        self.hooks = []
        for i, blk in enumerate(model.layers):
            self.hooks.append(blk.register_forward_hook(
                lambda mod, inp, out, idx=i: self._cap(idx, mod, out[0])))

    def _cap(self, i, mod, h):
        self.hp[i] = mod.mirror._cached_hp.abs().max().item()
        self.gate[i] = mod.mirror._cached_gate.mean().item()
        self.h_out[i] = h.abs().max().item()

    def detach(self):
        for h in self.hooks:
            h.remove()

    def report(self, tag):
        LOG.w(f'  [{tag}] per-layer h_out/hp absmax + gate mean:')
        for i in sorted(self.hp):
            LOG.w(f'    L{i:2d}  h={self.h_out[i]:10.4g}  hp={self.hp[i]:10.4g}  gate={self.gate[i]:.4f}')


def eval_window(model, tok, text, label, n=1):
    """Один eval-проход (как generate.py: state=None) с замером head и выхода."""
    obs = Observer(model)
    L = model.cfg.seq_len
    model.eval()
    with torch.no_grad():
        win = corpus_windows(tok, text, 1, L)[0]
        h = model.embed_tokens(win.unsqueeze(0))
        out, state, _, _ = model(h, None, adaptive=False)
        logits = model.lm_head(out[:, -1:, :])[0, 0]
        p = F.softmax(logits.double(), dim=-1)
        top1 = p.max().item()
        H = -(p * torch.log2(p.clamp_min(1e-12))).sum().item()
        LOG.w(f'[{label}] out absmax = {out.abs().max().item():.4g} | '
              f'top-1 = {top1:.4f} | H = {H:.3f} bit')
        obs.report(label)
    obs.detach()
    return top1, H


def run_fresh(model, tok, args):
    LOG.w('=== FRESH (EMA=1.0 на свежей загрузке) ===')
    text = load_corpus()[:args.corpus]
    top1, H = eval_window(model, tok, text, 'fresh')
    ok = out_ok = None
    LOG.w(f'RESULT fresh: top-1 = {top1:.4f} | H = {H:.3f} bit')
    return top1, H


def run_warm(model, tok, args):
    LOG.w(f'=== WARM (train-прогрев EMA, {args.windows} окон) ===')
    text = load_corpus()
    ema = warm_emas(model, tok, n_windows=args.windows, text=text)
    LOG.w(f'EMA[0] mean после прогрева = {ema:.1f}')
    top1, H = eval_window(model, tok, text, 'warm')
    LOG.w(f'RESULT warm: top-1 = {top1:.4f} | H = {H:.3f} bit')
    return top1, H, ema


def run_ab(model, tok, args):
    """A/B головы: reasoning OFF / natural / FULL — те же окна, seed фикс."""
    LOG.w('=== A/B HEAD: reasoning OFF / natural / FULL ===')
    text = load_corpus()
    L = model.cfg.seq_len
    wins = corpus_windows(tok, text, args.windows, L)
    obs = Observer(model)
    model.eval()
    # Перспектива семплера: AdaptiveSampler делит на base_temp/exp(log_temp)
    log_temp_ref = 0.0
    for name, par in model.lm_head.named_parameters():
        if 'log_temp' in name:
            log_temp_ref = par.data.mean().item()
    t_eff_sampler = args.temperature / math.exp(log_temp_ref)
    LOG.w(f'  lm_head.log_temp mean = {log_temp_ref:.4f} -> e^log_temp = {math.exp(log_temp_ref):.3f} | '
          f'sampler t_eff (base 0.8) = {t_eff_sampler:.4f}')
    results = {}
    for mode, override in (('OFF', 0.0), ('natural', None), ('FULL', 1.0)):
        LOG.w(f'  mode={mode}')
        model.reasoning_scale_override = override
        model.reset_reasoning()
        stats = {'top1': [], 'H': [], 'top1_eff': [], 'H_eff': []}
        saved_tokens = None
        dist0 = None
        with torch.no_grad():
            for w in wins:
                tokens = w
                for step in range(1):  # один шаг на окно — распределение на конце
                    ctx = tokens[-L:].unsqueeze(0)
                    h = model.embed_tokens(ctx)
                    out, state, _, _ = model(h, None, adaptive=False)
                    logits = model.lm_head(out[:, -1:, :])[0, 0]
                    p = F.softmax(logits.double(), dim=-1)
                    top1 = p.max().item()
                    H = -(p * torch.log2(p.clamp_min(1e-12))).sum().item()
                    p_eff = F.softmax((logits.double() / t_eff_sampler), dim=-1)
                    stats['top1'].append(top1)
                    stats['H'].append(H)
                    stats['top1_eff'].append(p_eff.max().item())
                    stats['H_eff'].append(-(p_eff * torch.log2(p_eff.clamp_min(1e-12))).sum().item())
                    if dist0 is None:
                        dist0 = p
        results[mode] = (sum(stats['top1']) / len(stats['top1']),
                         sum(stats['H']) / len(stats['H']),
                         sum(stats['top1_eff']) / len(stats['top1_eff']),
                         sum(stats['H_eff']) / len(stats['H_eff']),
                         dist0)
        LOG.w(f'    raw:      top-1 = {results[mode][0]:.4f} | H = {results[mode][1]:.3f} bit')
        LOG.w(f'    sampler:  top-1 = {results[mode][2]:.4f} | H = {results[mode][3]:.3f} bit')
    # KL: natural vs OFF на первом окне (одинаковый вход)
    pn = results['natural'][4].clamp_min(1e-12)
    po = results['OFF'][4].clamp_min(1e-12)
    kl_no = (pn * torch.log2(pn / po)).sum().item()
    kl_on = (po * torch.log2(po / pn)).sum().item()
    LOG.w(f'KL natural||OFF = {kl_no:.4f} bit | KL OFF||natural = {kl_on:.4f} bit')
    LOG.w(f'RESULT ab: top-1(raw) OFF {results["OFF"][0]:.4f} vs natural {results["natural"][0]:.4f} '
          f'vs FULL {results["FULL"][0]:.4f}')
    model.reasoning_scale_override = None
    obs.detach()
    return results


@torch.no_grad()
def run_sample(model, tok, args):
    """Короткие тексты: static (t=0.8 top_k=40) и adaptive — для примера, не метрика."""
    LOG.w('=== SAMPLE (иллюстрация, не метрика) ===')
    model.eval()
    model.reasoning_scale_override = {'natural': None, 'off': 0.0, 'full': 1.0}[args.reasoning]
    model.reset_reasoning()
    if args.seed:
        torch.manual_seed(args.seed)
    prompts = args.prompt.split('||') if args.prompt else ['Москва — столица России, и']
    L = model.cfg.seq_len
    log_temp_val = None
    for name, par in model.lm_head.named_parameters():
        if 'log_temp' in name:
            log_temp_val = par.data.mean().item()
    tb = model.lm_head.token_bias.data
    LOG.w(f'  lm_head.log_temp mean = {log_temp_val:.4f} | bias_alpha = {args.bias_alpha}')

    TAU_LADDER = [8, 32, 128, 512]

    def adaptive_topk(pos):
        """top_k = масштаб памяти (τ-лестница) или φ-масштаб (Фибоначчи),
        покрывающий текущую длину контекста внутри окна."""
        if args.topk_sched == 'tau':
            for tau in TAU_LADDER:
                if pos < tau:
                    return tau
            return TAU_LADDER[-1]
        if args.topk_sched == 'fib':
            a, b = 1, 1
            while a < pos + 1:
                a, b = b, a + b
            return min(max(a, 8), 512)
        return args.topk

    for p in prompts:
        ids = tok.encode(p).ids
        tokens = torch.tensor(ids, dtype=torch.long)
        sampler = None
        if not args.static:
            from generate import AdaptiveSampler
            sampler = AdaptiveSampler(base_temp=args.temperature, top_k=args.topk, base_top_p=0.90,
                                      rep_penalty=2.0, rep_window=5)
            sampler.log_temp_ref = log_temp_val or 0.0
        state = None
        for step in range(args.tokens):
            tk = adaptive_topk(step % L)
            if sampler is not None:
                sampler.top_k = tk
            ctx = tokens[-L:].unsqueeze(0)
            h = model.embed_tokens(ctx)
            out, state, _, _ = model(h, state, adaptive=False)
            logits = model.lm_head(out[:, -1:, :])[0, 0]
            if args.bias_alpha != 1.0:
                logits = (logits - tb) + args.bias_alpha * tb
            if step % 10 == 0:
                topk = torch.topk(logits, 5)
                toks = [tok.decode([int(i)]) for i in topk.indices.tolist()]
                p_eff = F.softmax(logits.double() / (args.temperature / math.exp(log_temp_val or 0.0)), dim=-1)
                probs = p_eff[topk.indices].tolist()
                LOG.w(f'    step {step}: ' + ' | '.join(f'{t!r}({pr:.3f})' for t, pr in zip(toks, probs)), console=False)
            if sampler is not None:
                nxt = torch.tensor([sampler.sample(logits)], dtype=torch.long)
            else:
                z = logits / args.temperature
                vals, _ = torch.topk(z, args.topk)
                z[z < vals[-1:]] = -float('inf')
                probs = F.softmax(z, dim=-1)
                nxt = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, nxt])
        LOG.w(f'  > {p}')
        LOG.w(f'  {tok.decode(tokens.tolist(), skip_special_tokens=True)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt', help='path to .pt')
    ap.add_argument('cmd', choices=['fresh', 'warm', 'ab', 'sample'])
    ap.add_argument('--windows', type=int, default=12)
    ap.add_argument('--corpus', type=int, default=20000)
    ap.add_argument('--tokens', type=int, default=80)
    ap.add_argument('--prompt', type=str, default='')
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--reasoning', choices=['natural', 'off', 'full'], default='natural')
    ap.add_argument('--static', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bias-alpha', type=float, default=1.0)
    ap.add_argument('--topk', type=int, default=40)
    ap.add_argument('--topk-sched', choices=['fixed', 'tau', 'fib'], default='fixed')
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

    LOG.open()
    model, state = load_model(args.ckpt)
    tok = load_russian_tokenizer()
    LOG.w(f'ckpt: {args.ckpt} | step={state.get("step", "?")} | '
          f'reasoning t={model.reasoning_enabled_step} scale={model.reasoning_scale:.4f} | '
          f'device={device}')
    if args.cmd == 'fresh':
        run_fresh(model, tok, args)
    elif args.cmd == 'warm':
        run_warm(model, tok, args)
    elif args.cmd == 'ab':
        run_ab(model, tok, args)
    elif args.cmd == 'sample':
        run_sample(model, tok, args)
    LOG.close()
    print(f'log: {LOG_PATH}')


if __name__ == '__main__':
    main()