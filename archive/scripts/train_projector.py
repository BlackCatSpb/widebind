"""Обучение обучаемого прожектора (ProjectorNet) на учительских парах.

Путь A: WideBind не затрагивается, обучается отдельный модуль (CPU/GPU локально).

Лоссы:
  L = BCE_order (арбитр, sigmoid-pointer по каждому шагу)
    + λ·BCE_gate (роли: ядро/актант/модификатор/служебное, multi-label sigmoid)

Метрики: точность первого слова, доля полностью восстановленных порядков,
LCS-близость порядка.
"""

import argparse
import json
import math
import random
import sys
import time

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F

from core.projector_net import (MAX_LEN, POS_INDEX, ROLE_INDEX, ProjectorNet,
                                build_profile, profile_dim)
from core.stencil import Stencil


def load_data(path: str, limit: int):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def collate(rows, stencil, device, rng=None):
    import numpy as np
    L = MAX_LEN
    B_sz = len(rows)
    if rng is None:
        rng = np.random.RandomState(7)
    pdim = profile_dim(stencil is not None)
    profs = np.zeros((B_sz, L, pdim), dtype=np.float32)
    lens = np.zeros(B_sz, dtype=np.int64)
    orders = np.full((B_sz, L), L - 1, dtype=np.int64)
    roles = np.full((B_sz, L), ROLE_INDEX['service'], dtype=np.int64)
    for b, r in enumerate(rows):
        words = r['words'][:L]
        poss = r['poss'][:L]
        rls = r['roles'][:L]
        nw = len(words)
        lens[b] = nw
        perm = rng.permutation(nw)
        for j in range(nw):
            i = int(perm[j])
            profs[b, j] = build_profile(words[i], poss[i], stencil)
            roles[b, j] = ROLE_INDEX.get(rls[i], ROLE_INDEX['service'])
        inv = np.zeros(nw, dtype=np.int64)
        inv[perm] = np.arange(nw)
        orders[b, :nw] = inv
    P = torch.from_numpy(profs).to(device)
    lens_t = torch.from_numpy(lens).to(device)
    order_t = torch.from_numpy(orders).to(device)
    roles_t = torch.from_numpy(roles).to(device)
    mask = torch.arange(L, device=device).unsqueeze(0) >= lens_t.unsqueeze(1)
    return P, lens_t, order_t, roles_t, mask


def lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/supervision.jsonl')
    ap.add_argument('--stencil', default='data/stencil.json.gz')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lambda-gate', type=float, default=0.5)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')
    stencil = Stencil.load(args.stencil) if args.stencil else None
    rows = load_data(args.data, args.limit)
    rng = random.Random(7)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f'train={len(train_rows)} val={len(val_rows)}')

    import numpy as np
    collate_rng = np.random.RandomState(7)
    pdim = profile_dim(stencil is not None)
    model = ProjectorNet(pdim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = 0
    for ep in range(args.epochs):
        t0 = time.time()
        rng.shuffle(train_rows)
        tot_loss = 0.0
        tot_o = 0.0
        tot_g = 0.0
        nb = 0
        for i in range(0, len(train_rows), args.batch):
            batch = train_rows[i:i + args.batch]
            P, lens, order, roles, mask = collate(batch, stencil, device, collate_rng)
            h = model.encode(P, lens)
            gate_p = model.gate_probs(h)
            role_target = F.one_hot(roles, num_classes=model.n_roles).float()
            loss_gate = F.binary_cross_entropy(gate_p, role_target)
            logits, chosen = model.forward_order(h, order, lens, teacher_forcing=True)
            loss_order = model.order_loss(logits, order, lens)
            loss = loss_order + args.lambda_gate * loss_gate
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss.item())
            tot_o += float(loss_order.item())
            tot_g += float(loss_gate.item())
            nb += 1
            steps += 1
        # валидация
        model.eval()
        acc1 = 0.0
        accfull = 0.0
        lcs_norm = 0.0
        nv = 0
        nbt = 0
        with torch.no_grad():
            for i in range(0, len(val_rows), args.batch):
                batch = val_rows[i:i + args.batch]
                P, lens, order, roles, mask = collate(batch, stencil, device, collate_rng)
                h = model.encode(P, lens)
                logits, chosen = model.forward_order(h, order, lens, teacher_forcing=False)
                acc1 += model.acc_first(logits, order, lens)
                accfull += model.acc_full(chosen, order, lens)
                nbt += 1
                for b in range(P.shape[0]):
                    lb = int(lens[b])
                    if lb == 0:
                        continue
                    nv += 1
                    lcs_norm += lcs(chosen[b][:lb].tolist(), order[b][:lb].tolist()) / lb
        model.train()
        print(f'ep {ep + 1}/{args.epochs}: loss={tot_loss / max(nb, 1):.4f} '
              f'(order={tot_o / max(nb, 1):.4f}, gate={tot_g / max(nb, 1):.4f}) '
              f'acc1={acc1 / max(nbt, 1):.3f} '
              f'accfull={accfull / max(nbt, 1):.3f} '
              f'lcs={lcs_norm / max(nv, 1):.3f} [{time.time() - t0:.0f}s]', flush=True)


if __name__ == '__main__':
    main()