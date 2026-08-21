"""Projector gate head — учит границы слов прямо в процессе обучения WideBind.

Путь B: прожектор читает скрытые состояния h модели и предсказывает вероятность,
что текущий токен завершает слово (word-end). Супервизия — дёшево вычисляемые
метки границ слов: токен начинает новое слово, если его декодированная строка
начинается с пробела (соглашение BPE). Спецтокены (PAD/BOS/EOS) — всегда границы.

Головка опциональна (cfg.projector_gate=False по умолчанию), чтобы не менять
базовое LM-обучение. Включается единым флагом в том же тренировочном цикле.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_SPECIALS = (0, 1, 2)  # PAD, BOS, EOS — всегда границы слов


def word_end_labels(ids: torch.Tensor, tokenizer, specials=_SPECIALS) -> torch.Tensor:
    """Метки границ слов из токенов.

    ids: (B, L) long. Возвращает (B, L) long: 1 = токен завершает слово.
    Токен t — граница, если: спецтокен; последний в последовательности;
    следующий токен (t+1) начинает новое слово (декодируется с пробела).
    """
    B, L = ids.shape
    out = torch.zeros(B, L, dtype=torch.long)
    ids_l = ids.tolist()
    for b in range(B):
        starts = [False] * L
        for t in range(L):
            tid = ids_l[b][t]
            if tid in specials:
                starts[t] = True
                continue
            try:
                d = tokenizer.decode([tid])
            except Exception:
                d = ''
            starts[t] = d.startswith(' ')
        for t in range(L):
            if ids_l[b][t] in specials:
                out[b, t] = 1
            elif t == L - 1:
                out[b, t] = 1
            else:
                out[b, t] = 1 if starts[t + 1] else 0
    return out


class ProjectorGateHead(nn.Module):
    """sigmoid-гейт: вероятность word-end по скрытому состоянию h[t]."""

    def __init__(self, D: int):
        super().__init__()
        hid = max(16, D // 4)
        self.net = nn.Sequential(
            nn.Linear(D, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
        )
        # Инициализация в "неуверенную" зону (p≈0.5) — стабильный старт
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, L, D) -> (B, L) вероятностей
        return torch.sigmoid(self.net(h).squeeze(-1))


def projector_gate_loss(gate_head: ProjectorGateHead, h: torch.Tensor,
                        x: torch.Tensor, tokenizer, weight: float = 1.0) -> torch.Tensor:
    """BCE по меткам границ слов."""
    labels = word_end_labels(x.detach().cpu(), tokenizer).to(h.device)
    p = gate_head(h)
    return weight * F.binary_cross_entropy(p, labels.float())
