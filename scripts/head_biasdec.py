"""Декомпозиция head: logits = scores@codes.T + token_bias.
Вопрос: топ-1 определяется контекстом (scores) или bias'ом?
Мерим на реальном корпусе: доля позиций, где argmax(logits) == argmax(token_bias),
и предсказания по чистому контексту (logits - token_bias).
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

LOG_PATH = os.path.join(os.environ.get('TEMP', BASE), 'opencode', 'biasdec.log')
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


def analyze(m, tok, text, n_windows, L, temp_base=0.8, alphas=(1.0,)):
    ids = tok.encode(text).ids
    log_temp = m.lm_head.log_temp.data.mean().item()
    t_eff = temp_base / math.exp(log_temp)
    tb = m.lm_head.token_bias.data
    tb_argmax = int(tb.argmax())
    tb_top = int(tb.abs().argmax())
    top_bias_toks = [tok.decode([int(i)]) for i in torch.topk(tb, 10).indices.tolist()]
    w(f'  token_bias: max_idx={tok.decode([tb_argmax])!r} | top-10: ' +
      ' | '.join(repr(t) for t in top_bias_toks))
    for alpha in alphas:
        stats = dict(match=0, total=0, ctx_word=0, ctx_punct=0, full_word=0, full_punct=0,
                     ctx_top_toks={}, full_top_toks={}, bias_rank=[])
        ctx_all, ctx_std_agg = [], []
        for start in range(0, min(len(ids) - L, n_windows * L), L):
            window = torch.tensor(ids[start:start + L], dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                h = m.embed_tokens(window)
                out, _, _, _ = m(h, None, adaptive=False)
                logits = m.lm_head(out[0])  # (L, V)
            ctx = logits - tb.unsqueeze(0)  # чистый контекст
            ctx_std_agg.append(float(ctx.std()))
            ctx_all.append(ctx)
            for pos in range(0, L, 8):
                lt = (ctx + alpha * tb.unsqueeze(0))[pos]
                ct = ctx[pos]
                stats['total'] += 1
                if int(lt.argmax()) == tb_argmax:
                    stats['match'] += 1
                rank = int((lt.argsort(descending=True) == tb_argmax).nonzero()[0])
                stats['bias_rank'].append(rank)
                for tid, tstr, cat, store in (
                        (int(ct.argmax()), tok.decode([int(ct.argmax())]).strip(), None, 'ctx_top_toks'),
                        (int(lt.argmax()), tok.decode([int(lt.argmax())]).strip(), None, 'full_top_toks')):
                    if cat is None:
                        cat = 'punct' if PUNCT.match(tstr) else ('word' if WORD.search(tstr) else 'other')
                    if store == 'ctx_top_toks':
                        stats['ctx_word' if cat == 'word' else ('ctx_punct' if cat == 'punct' else 'ctx_other')] = \
                            stats.get('ctx_word' if cat == 'word' else ('ctx_punct' if cat == 'punct' else 'ctx_other'), 0) + 1
                    else:
                        stats['full_word' if cat == 'word' else ('full_punct' if cat == 'punct' else 'full_other')] = \
                            stats.get('full_word' if cat == 'word' else ('full_punct' if cat == 'punct' else 'full_other'), 0) + 1
                    d = stats.setdefault(store, {})
                    d[tstr] = d.get(tstr, 0) + 1
        t = stats['total']
        ctx_std_mean = sum(ctx_std_agg) / len(ctx_std_agg)
        w(f'  alphas надёжность: std(ctx)={ctx_std_mean:.4f} | std(tb)={tb.std().item():.4f} | '
          f'МАТЕМАТИЧЕСКОЕ alpha = std(ctx)/std(tb) = {ctx_std_mean / tb.std().item():.4f}')
        w(f'  alpha={alpha:.2f}: argmax(logits)==argmax(bias) {100.0 * stats["match"] / t:.1f}% | '
          f'bias-rank med={sorted(stats["bias_rank"])[t // 2]} p90={sorted(stats["bias_rank"])[int(t * 0.9)]} | '
          f'топ-1 слово {100.0 * stats.get("full_word", 0) / t:.1f}% | пункт {100.0 * stats.get("full_punct", 0) / t:.1f}%')
        top_full = sorted(stats['full_top_toks'].items(), key=lambda kv: -kv[1])[:8]
        w('    топ-8 (полные, alpha=%.2f): ' % alpha + ' | '.join('%r(%d)' % (k, v) for k, v in top_full))
    w('  топ-1 по КОНТЕКСТУ (logits-bias, эталон): '
      f'word {100.0 * stats.get("ctx_word", 0) / stats["total"]:.1f}% | '
      f'punct {100.0 * stats.get("ctx_punct", 0) / stats["total"]:.1f}%')
    top_ctx = sorted(stats['ctx_top_toks'].items(), key=lambda kv: -kv[1])[:8]
    w('  топ-8 токенов (контекст): ' + ' | '.join('%r(%d)' % (k, v) for k, v in top_ctx))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--windows', type=int, default=4)
    ap.add_argument('--corpus', type=str, default='')
    ap.add_argument('--alphas', type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 1.0])
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
        w(f'--- {ck} | t={m.reasoning_enabled_step} scale={m.reasoning_scale:.4f}')
        analyze(m, tok, text, args.windows, L, alphas=args.alphas)
    log.close()
    print('log:', LOG_PATH)