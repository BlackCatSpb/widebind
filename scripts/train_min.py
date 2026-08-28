"""Mini testbed for WideBind semantic-connectivity experiments.

Goal: try the "double signal" idea cheaply, locally, on CPU.
- Head is UNTOUCHED: training uses plain CE (model.compute_losses).
- Bridge gets an optional self-supervised aux loss (--bridge_conn): predict the
  next element from the bridge/stream signal. Deterministic tokenization => the
  next element is a free label, so the bridge learns "how things connect".
- MLP wake (reopen mod_scale_mlp + depth gradient boost) is toggled via
  --mlp_wake (uses the already-implemented core hooks/reinit).
No core changes: the aux head lives here, outside the model.
"""
import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from core import WideBindConfig, WideBindStack


def make_corpus(seed=0):
    """Structured deterministic corpus: subj verb obj .  (tests connectivity rules)."""
    random.seed(seed)
    subj = ["cat", "dog", "bird", "fish", "fox", "owl"]
    verb = ["sat", "ran", "flew", "swam", "ate", "saw"]
    obj = ["mat", "park", "sky", "sea", "bone", "star"]
    sents = [f"{s} {v} {o} ." for s in subj for v in verb for o in obj]
    random.shuffle(sents)
    # repeat to a few thousand tokens
    text = " ".join(sents)
    while len(text.split()) < 4000:
        random.shuffle(sents)
        text = text + " " + " ".join(sents)
    toks = text.split()
    vocab = sorted(set(toks))
    v2i = {w: i for i, w in enumerate(vocab)}
    ids = [v2i[w] for w in toks]
    return ids, vocab, v2i


def to_batches(ids, seq_len, batch, n_batches):
    # rolling windows; label = next token
    X, Y = [], []
    for b in range(n_batches):
        off = (b * batch * 1) % max(1, len(ids) - seq_len - 1)
        for _ in range(batch):
            if off + seq_len + 1 >= len(ids):
                off = 0
            x = ids[off:off + seq_len]
            y = ids[off + 1:off + seq_len + 1]
            X.append(x)
            Y.append(y)
            off += seq_len
    return torch.tensor(X), torch.tensor(Y)


def build_model(vocab_size, bridge_glu=False):
    cfg = WideBindConfig(
        D=256, n_layers=4, orth_weight=0.0,
        bind_K=32, mlp_groups=4, mlp_expand=2,
        vocab=vocab_size, seq_len=32,
        lr=1e-3, max_steps=2000, warmup_steps=100,
        log_interval=50, eval_interval=400,
        mlp_gate_b_init=0.25, mlp_depth_lr_exp=0.10,
        explicit_reasoning=False,
        intent_bridge=True,
        bridge_glu=bridge_glu,
    )
    model = WideBindStack(cfg)
    return model, cfg


def train(args):
    ids, vocab, v2i = make_corpus()
    V = len(vocab)
    model, cfg = build_model(V, bridge_glu=args.bridge_glu)
    device = "cpu"
    model.to(device)
    if args.mlp_wake:
        model.apply_mlp_depth_gradient_boost()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95))

    # external bridge aux head (separate, head untouched)
    if args.bridge_conn > 0:
        bridge_head = torch.nn.Linear(cfg.D, V).to(device)
        bopt = torch.optim.AdamW(bridge_head.parameters(), lr=cfg.lr)
    else:
        bridge_head = None

    n_batches = 200
    batch = 8
    seq_len = cfg.seq_len
    t0 = time.time()
    for step in range(args.steps):
        model.train()
        X, Y = to_batches(ids, seq_len, batch, n_batches)
        X, Y = X.to(device), Y.to(device)
        x = X[step % len(X)].unsqueeze(0)
        y = Y[step % len(Y)].unsqueeze(0)
        h = model.embed_tokens(x)
        state = None
        intent_state = getattr(model, "_last_intent_state", None)
        out, state, _, _ = model(h, state, step=step, intent_state=intent_state)
        ce_loss, _ = model.compute_losses(out, y, h_emb=h)
        loss = ce_loss
        if bridge_head is not None:
            sig = out[:, -1]  # bridge/stream signal proxy (grad flows into bridge)
            lc = F.cross_entropy(bridge_head(sig), y[:, -1])
            loss = loss + args.bridge_conn * lc
            lc_val = lc.item()
        else:
            lc_val = float("nan")
        opt.zero_grad(set_to_none=True)
        if bridge_head is not None:
            bopt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if bridge_head is not None:
            bopt.step()

        if step % args.log == 0:
            # gate spread (wake check)
            with torch.no_grad():
                if any(l.mirror.bridge_glu_net is not None for l in model.layers):
                    mlp_s = "bglu=on"
                else:
                    ms = torch.stack([torch.sigmoid(l.mirror.mod_scale_mlp).mean()
                                      for l in model.layers])
                    mlp_s = f"mod_mlp_mean={ms.mean().item():.3f} mod_mlp_std={ms.std().item():.3f}"
            lc_s = "" if bridge_head is None else f" Lconn={lc_val:.3f}"
            print(f"step={step:>5} ce={ce_loss.item():.3f} {mlp_s}{lc_s} t={time.time()-t0:.0f}s")
    # generation sample (greedy) with bridge stream carried
    print("\n--- generation (prompt: 'cat') ---")
    model.eval()
    prompt = v2i.get("cat", 0)
    gen = [prompt]
    st = None
    with torch.no_grad():
        for _ in range(12):
            x = torch.tensor([[gen[-1]]]).to(device)
            h = model.embed_tokens(x)
            o, st, _, _ = model(h, st, step=0, intent_state=getattr(model, "_last_intent_state", None))
            nxt = int(model.lm_head(o)[:, -1].argmax())
            gen.append(nxt)
            if nxt == v2i.get(".", 0):
                break
    inv = {i: w for w, i in v2i.items()}
    print(" ".join(inv.get(i, "?") for i in gen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bridge_conn", type=float, default=0.0,
                    help="weight of bridge self-supervised aux loss (0 = off)")
    ap.add_argument("--mlp_wake", action="store_true",
                    help="apply mlp depth gradient boost (reopen mod_scale_mlp)")
    ap.add_argument("--bridge_glu", action="store_true",
                    help="BridgeGLU: relocate SwiGLU gating into the mirror (semantic gate)")
    ap.add_argument("--log", type=int, default=100)
    args = ap.parse_args()
    print(f"[mini] bridge_conn={args.bridge_conn} mlp_wake={args.mlp_wake} bridge_glu={args.bridge_glu} steps={args.steps}")
    train(args)


if __name__ == "__main__":
    main()
