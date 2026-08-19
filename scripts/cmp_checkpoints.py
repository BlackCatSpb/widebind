"""Точечное сравнение двух чекпоинтов на ИДЕНТИЧНОМ входе:
распределение головы (шаг 0), token_bias/readout/log_temp, контекст-зависимость.
"""
import os, sys, math

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

LOG_PATH = os.path.join(os.environ.get('TEMP', BASE), 'opencode', 'cmp.log')
log = open(LOG_PATH, 'w', encoding='utf-8')


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


def head_params(m):
    h = m.lm_head
    log_temp = h.log_temp.data
    tb = h.token_bias.data
    ro = h.readout.data
    return dict(
        log_temp_mean=float(log_temp.mean()),
        log_temp_max=float(log_temp.max()),
        log_temp_min=float(log_temp.min()),
        token_bias_max=float(tb.abs().max()),
        token_bias_std=float(tb.std()),
        readout_norm=float(ro.norm()),
        K=int(log_temp.numel()),
    )


def step0(m, tok, prompt):
    ids = tok.encode(prompt).ids
    tokens = torch.tensor(ids, dtype=torch.long)
    L = m.cfg.seq_len
    with torch.no_grad():
        ctx = tokens[-L:].unsqueeze(0)
        h = m.embed_tokens(ctx)
        out, state, _, _ = m(h, None, adaptive=False)
        logits = m.lm_head(out[:, -1:, :])[0, 0]
    return logits


def report(name, m, tok, prompt):
    w(f'--- {name} | reasoning t={m.reasoning_enabled_step} scale={m.reasoning_scale:.4f}')
    p = head_params(m)
    w('  head: log_temp mean=%.4f max=%.4f min=%.4f | tb_max=%.3f tb_std=%.3f | readout_norm=%.3f K=%d'
      % (p['log_temp_mean'], p['log_temp_max'], p['log_temp_min'],
         p['token_bias_max'], p['token_bias_std'], p['readout_norm'], p['K']))
    for ov, tag in ((0.0, 'reasoning OFF'), (None, 'natural'), (1.0, 'FULL')):
        m.reasoning_scale_override = ov
        m.reset_reasoning()
        logits = step0(m, tok, prompt)
        eff = args.temp / math.exp(p['log_temp_mean'])
        for tname, z in (('raw', logits.double()), ('sampler(t_eff=%.3f)' % eff, logits.double() / eff)):
            pr = F.softmax(z, dim=-1)
            topk = torch.topk(pr, 8)
            toks = [tok.decode([int(i)]) for i in topk.indices.tolist()]
            w('  [%s] %s top-1=%.4f H=%.3f bit: %s'
              % (tag, tname, pr.max().item(),
                 -(pr * torch.log2(pr.clamp_min(1e-12))).sum().item(),
                 ' | '.join('%r(%.3f)' % (t, v) for t, v in zip(toks, topk.values.tolist()))))
    m.reasoning_scale_override = None


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--prompt', default='В один холодный зимний вечер')
    ap.add_argument('--temp', type=float, default=0.8)
    args = ap.parse_args()
    tok = load_russian_tokenizer()
    for ck in args.ckpts:
        m, _ = load(ck)
        report(ck, m, tok, args.prompt)
    log.close()
    print('log:', LOG_PATH)