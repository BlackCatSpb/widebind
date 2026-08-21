"""ProjectorNet — обучаемый прожектор: гейт (sigmoid, multi-label) + арбитр (sigmoid-pointer).

Дизайн: docs/PROJECTOR_LEARNING.md, docs/WORD_ARITHMETIC.md.
Числовой профиль слова: факторный вектор (счёт букв), log N, коды первой/последней
буквы, salience-признаки трафарета, POS-onehot, длина.

Гейт: σ(MLP(h_i)) — независимая вероятность роли (ядро/актант/модификатор/служебное).
Арбитр: авторегрессивный sigmoid-pointer — на шаге t для каждого невыбранного
слова p_i = σ(score(h_i, ctx_t)), выбирается argmax; BCE против правильного порядка.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.word_num import ALPHABET, _BASE, _CODE, n_of

B = float(_BASE)

POS_LIST = ['NOUN', 'NPRO', 'VERB', 'INFN', 'ADJF', 'ADVB', 'NUMR', 'GRND',
            'PREP', 'CONJ', 'PRCL', 'PRTF', 'OTHER']
POS_INDEX = {p: i for i, p in enumerate(POS_LIST)}

ROLE_INDEX = {'predicate': 0, 'actant': 1, 'modifier': 2, 'service': 3}
N_ROLES = len(ROLE_INDEX)
MAX_LEN = 12


def build_profile(word: str, pos: str, stencil=None) -> List[float]:
    """Числовой профиль слова (без torch, для датасета)."""
    w = word.lower()
    n = n_of(w) if len(w) > 0 else 1
    counts = [0] * len(ALPHABET)
    for c in w:
        if c == 'ё':
            c = 'е'
        if c in _CODE:
            counts[_CODE[c] - 1] += 1
    total = max(1, sum(counts))
    prof = [c / total for c in counts]
    prof.append(math.log(n) / 30.0 if n > 1 else 0.0)
    prof.append((_CODE.get(w[0], 0) / B) if w else 0.0)
    prof.append((_CODE.get(w[-1], 0) / B) if w else 0.0)
    prof.append(min(1.0, len(w) / 20.0))
    prof.append(POS_INDEX.get(pos, POS_INDEX['OTHER']) / len(POS_LIST))
    if stencil is not None:
        try:
            wgt, k = stencil.salience(w)
            prof.append(wgt)
            prof.append(min(1.0, k / 5.0))
        except Exception:
            prof.append(0.5)
            prof.append(0.2)
    else:
        prof.append(0.5)
        prof.append(0.2)
    return prof


def profile_dim(stencil: bool = True) -> int:
    return len(ALPHABET) + 4 + 1 + (2 if stencil else 0)


def _mask_of(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(1, seq_len, seq_len, dtype=torch.bool, device=device)


class ProjectorNet(nn.Module):
    def __init__(self, p_dim: int, d_model: int = 128, nhead: int = 4,
                 n_layers: int = 2, n_roles: int = N_ROLES, max_len: int = MAX_LEN):
        super().__init__()
        self.max_len = max_len
        self.n_roles = n_roles
        self.d_model = d_model
        self.proj = nn.Linear(p_dim, d_model)
        self.pos = nn.Embedding(max_len + 2, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256,
                                           batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.gate_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(),
                                       nn.Linear(64, n_roles))
        self.ctx_proj = nn.Linear(d_model * 2, d_model)
        self.score_head = nn.Sequential(nn.Linear(d_model * 2, 64), nn.ReLU(),
                                        nn.Linear(64, 1))

    def encode(self, prof: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """prof: (B, L, Dp), lengths: (B,) — фактические длины (паддинг-маска)."""
        B_sz, L, _ = prof.shape
        h = self.proj(prof) + self.pos(torch.arange(L, device=prof.device).unsqueeze(0))
        mask = (torch.arange(L, device=prof.device).unsqueeze(0) >=
                lengths.unsqueeze(1))
        h = self.enc(h, src_key_padding_mask=mask)
        return h

    def gate_probs(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate_head(h))

    def _step_scores(self, h: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        n = h.shape[1]
        ctx_e = ctx.unsqueeze(1).expand(-1, n, -1)
        return self.score_head(torch.cat([h, ctx_e], dim=-1)).squeeze(-1)

    def forward_order(self, h: torch.Tensor, order: torch.Tensor,
                      lengths: torch.Tensor, teacher_forcing: bool = True,
                      temp: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Арбитр: BCE-предсказание следующего слова на каждом шаге.

        h: (B, L, D), order: (B, L) — целевая последовательность индексов,
        lengths: (B,). Возвращает (логисы каждого шага (B, L, L), выбранные индексы).
        Векторизовано: цикл только по шагам t (L ≤ 12).
        """
        B_sz, L, _ = h.shape
        device = h.device
        logits = torch.zeros(B_sz, L, L, device=device)
        chosen = torch.zeros(B_sz, L, dtype=torch.long, device=device)
        ctx = h.mean(dim=1)
        done = torch.zeros(B_sz, L, dtype=torch.bool, device=device)
        sel = torch.arange(L, device=device).unsqueeze(0).expand(B_sz, L)
        for t in range(L):
            live = lengths > t
            if not live.any():
                break
            l_t = self._step_scores(h, ctx)
            l_t = l_t.masked_fill(done, -1e9)
            l_t = l_t.masked_fill(sel >= lengths.unsqueeze(1), -1e9)
            logits[:, t] = l_t
            if teacher_forcing:
                nxt = order[:, t].clamp(0, L - 1)
            else:
                nxt = (l_t / max(temp, 1e-6)).softmax(dim=-1).argmax(dim=-1)
            chosen[:, t] = nxt
            nxt_d = nxt.detach()
            hn = h.gather(1, nxt_d.unsqueeze(-1).unsqueeze(-1).expand(B_sz, 1, h.shape[-1])).squeeze(1)
            ctx = self.ctx_proj(torch.cat([ctx, hn], dim=-1))
            done = done | torch.zeros(B_sz, L, dtype=torch.bool, device=device).scatter_(
                1, nxt_d.clamp(0, L - 1).unsqueeze(1), True)
        return logits, chosen

    def order_loss(self, logits: torch.Tensor, order: torch.Tensor,
                   lengths: torch.Tensor) -> torch.Tensor:
        """BCE по шагам: target_t — индекс следующего слова в правильном порядке."""
        B_sz, L, _ = logits.shape
        device = logits.device
        total = 0.0
        cnt = 0
        for t in range(L):
            live = lengths > t
            if not live.any():
                break
            target = order[:, t]
            m = torch.zeros_like(logits[:, t])
            for b in range(B_sz):
                if live[b]:
                    m[b, target[b]] = 1.0
            total = total + F.binary_cross_entropy_with_logits(
                logits[:, t], m, reduction='sum')
            cnt += int(live.sum().item())
        return total / max(cnt, 1)

    def acc_first(self, logits: torch.Tensor, order: torch.Tensor,
                  lengths: torch.Tensor) -> float:
        live = lengths > 0
        pred = logits[:, 0].argmax(dim=-1)
        if not live.any():
            return 1.0
        ok = (pred == order[:, 0]) & live
        return float(ok.sum().item() / live.sum().item())

    def acc_full(self, chosen: torch.Tensor, order: torch.Tensor,
                 lengths: torch.Tensor) -> float:
        ok = (chosen == order).sum(dim=-1)
        return float((ok == lengths).float().mean().item())