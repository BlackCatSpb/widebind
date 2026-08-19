"""
WideBind text generation via ONNX Runtime (CPU).

Runs the WideBindStack forward through an exported ONNX model
(see docs/TRAINING_JOURNAL.md ch. 71 for the export recipe), keeping
embedding / lm_head / sampling in torch — identical to scripts/generate.py.

Notes:
  - state feeds are carried between steps (mem/mu/conv/traj per layer); the
    pen (pred-error) channel is input-only in the exported graph -> zeros.
  - global_state is re-created from zeros every step — same as generate.py.
  - reasoning_scale is FROZEN in the ONNX graph (exported at checkpoint step;
    for step_12831.pt it is ~1.0 = "full"). This equals python
    --reasoning full; it CANNOT be turned off without re-export.
"""

import os, sys, time, math, torch, json, inspect
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch.nn.functional as F
import onnxruntime as ort

from core import WideBindConfig, WideBindStack
from core.migrate import migrate_state_dict
from scripts.generate import AdaptiveSampler, load_russian_tokenizer


def load_session(onnx_path, threads=0):
    opt = ort.SessionOptions()
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads > 0:
        opt.intra_op_num_threads = threads
    ort.set_default_logger_severity(3)
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'],
                                sess_options=opt)
    inames = [i.name for i in sess.get_inputs()]
    onames = [o.name for o in sess.get_outputs()]
    return sess, inames, onames


def state_feeds(shapes, n_layers):
    """Ordered state feeds: state_<layer>_<field> per ONNX input names.
    Fields 0..3 (mem, mu, conv, traj) are outputs of the graph and carried
    between steps; field 4 (pen) is an input-only channel -> zeros always.
    """
    st = {}
    for n, sh in shapes.items():
        if n.startswith('state_'):
            st[n] = torch.zeros(*sh)
    return st


def generate_onnx(onnx_path, ckpt, prompt, max_new_tokens=128, temperature=1.0,
                  top_k=50, sampler=None, rep_penalty=2.0, rep_window=5,
                  bias_alpha=0.0, threads=0, verbose=False):
    cfg = ckpt['cfg']
    model = WideBindStack(cfg).eval()
    sd = dict(ckpt['model'])
    sd, n_migrated = migrate_state_dict(sd, model)
    model.load_state_dict(sd, strict=False)
    if n_migrated:
        print(f'Checkpoint migrated for spiral coherence: {n_migrated} keys', flush=True)
    if model.explicit_reasoning:
        model.reasoning_enabled_step = int(ckpt.get('reasoning_enabled_step', 0))
    print(f'Reasoning: enabled, ramp t={model.reasoning_enabled_step} '
          f'scale={model.reasoning_scale:.4f} (FROZEN in ONNX graph)', flush=True)

    sess, inames, onames = load_session(onnx_path, threads)
    shapes = {i.name: list(i.shape) for i in sess.get_inputs()}
    print(f'ONNX session ready: {len(inames)} inputs, {len(onames)} outputs '
          f'(threads={threads or "default"})', flush=True)

    tok = load_russian_tokenizer()
    if tok is None:
        raise FileNotFoundError('russian_tokenizer/tokenizer.json not found')
    prompt_tokens = tok.encode(prompt).ids
    detokenize = lambda ids: tok.decode(ids, skip_special_tokens=True)
    tokens = torch.tensor(prompt_tokens, dtype=torch.long)

    L = cfg.seq_len
    B = 1
    n_layers = len(model.layers)
    st = state_feeds(shapes, n_layers)
    gs = torch.zeros(*shapes['global_state'])
    rb = torch.zeros(*shapes['reasoning_buffer'])
    rc = torch.zeros((), dtype=torch.long)
    tb = model.lm_head.token_bias.data
    try:
        h_emb_ok = 'h_emb' in inspect.signature(model.lm_head.forward).parameters
    except (TypeError, ValueError):
        h_emb_ok = False

    head = model.lm_head
    recent = set()
    n_state_out = 4 * n_layers  # mem, mu, conv, traj per layer (pen not an output)
    t0 = time.perf_counter()
    for step in range(max_new_tokens):
        ctx = tokens[-L:]
        if len(ctx) < L:
            ctx = torch.cat([torch.zeros(L - len(ctx), dtype=torch.long), ctx])
        ctx = ctx.unsqueeze(0)
        with torch.no_grad():
            h = model.embed_tokens(ctx)

        feeds = {'h': h.detach().numpy(),
                 'global_state': gs.numpy(),
                 'step': torch.tensor(step, dtype=torch.long).numpy(),
                 'reasoning_buffer': rb.numpy(),
                 'reasoning_count': rc.numpy()}
        for n, t in st.items():
            feeds[n] = t.numpy()

        res = sess.run(onames, feeds)
        out = torch.from_numpy(res[0])          # h (B, L, D)
        for li in range(n_layers):
            for fi in range(4):
                st[f'state_{li}_{fi}'] = torch.from_numpy(res[1 + li * 4 + fi]).clone()
        rb = torch.from_numpy(res[2 + n_state_out])
        rc = torch.from_numpy(res[3 + n_state_out])
        gs = torch.zeros(*shapes['global_state'])

        with torch.no_grad():
            if h_emb_ok:
                logits = head(out[:, -1:, :], h[:, -1:, :])[0, 0]
            else:
                logits = head(out[:, -1:, :])[0, 0]
            if bias_alpha != 1.0:
                logits = (logits - tb) + bias_alpha * tb
            if sampler is not None:
                nxt = torch.tensor([sampler.sample(logits)])
            else:
                logits = logits / temperature
                for rid in list(recent)[-rep_window:]:
                    logits[rid] -= rep_penalty
                if top_k > 0:
                    vals, _ = torch.topk(logits, top_k)
                    logits[logits < vals[-1:]] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)

        recent.add(int(nxt))
        tokens = torch.cat([tokens, nxt], dim=0)

    dt = time.perf_counter() - t0
    print(f'generated {max_new_tokens} tokens in {dt:.1f}s '
          f'({dt / max_new_tokens * 1000:.0f} ms/token)', flush=True)
    return detokenize(tokens.tolist())


if __name__ == '__main__':
    import argparse
    from torch.serialization import add_safe_globals
    add_safe_globals([WideBindConfig])

    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Path to .pt checkpoint')
    parser.add_argument('--onnx', type=str, required=True, help='Path to exported .onnx')
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--top-p', type=float, default=0.90)
    parser.add_argument('--rep-penalty', type=float, default=2.0)
    parser.add_argument('--rep-window', type=int, default=5)
    parser.add_argument('--rep-ngram', type=int, default=3)
    parser.add_argument('--alarm-window', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--static', action='store_true')
    parser.add_argument('--no-log-temp-norm', action='store_true')
    parser.add_argument('--adaptive-verbose', action='store_true')
    parser.add_argument('--bias-alpha', type=float, default=0.0)
    parser.add_argument('--threads', type=int, default=0)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if args.seed:
        torch.manual_seed(args.seed)

    sampler = None
    if not args.static:
        sampler = AdaptiveSampler(
            base_temp=args.temperature, top_k=args.top_k, base_top_p=args.top_p,
            rep_penalty=args.rep_penalty, rep_window=args.rep_window,
            rep_ngram=args.rep_ngram, alarm_window=args.alarm_window,
            norm_log_temp=not args.no_log_temp_norm, verbose=args.adaptive_verbose)
    print(f'Loaded checkpoint: step={ckpt.get("step", "?")}')

    if args.prompt:
        text = generate_onnx(args.onnx, ckpt, args.prompt, args.tokens,
                             args.temperature, args.top_k, sampler,
                             args.rep_penalty, args.rep_window,
                             args.bias_alpha, args.threads)
        print(f'Prompt: {args.prompt}')
        print(f'Generated: {text}')
    else:
        for p in ['Привет, как дела?', 'Москва — столица',
                  'В начале было Слово', 'Искусственный интеллект']:
            text = generate_onnx(args.onnx, ckpt, p, 100, 0.8, 0, sampler,
                                 args.rep_penalty, args.rep_window,
                                 args.bias_alpha, args.threads)
            print(f'> {p}')
            print(text)
            print()
