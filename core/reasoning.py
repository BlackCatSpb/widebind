"""
Explicit Reasoning Module for EVA.
Adds chain-of-thought reasoning with thinking tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ReasoningTokens:
    """Special tokens for chain-of-thought reasoning."""
    THINK = 65536  # <think>
    STEP = 65537   # <step>
    ANSWER = 65538 # <answer>
    END = 65539    # </think>


class ReasoningMemory(nn.Module):
    """
    Explicit reasoning memory for chain-of-thought.
    Stores intermediate reasoning steps in a dedicated buffer.
    """
    def __init__(self, D, max_steps=8):
        super().__init__()
        self.D = D
        self.max_steps = max_steps

        # Reasoning step encoder
        self.step_encoder = nn.Linear(D, D)

        # Reasoning attention (attend to previous steps)
        self.step_query = nn.Linear(D, D)
        self.step_key = nn.Linear(D, D)
        self.step_value = nn.Linear(D, D)

        # Output projection
        self.output_proj = nn.Linear(D, D)

    def forward(self, h, reasoning_buffer=None, record=True):
        """
        h: (B, L, D) current hidden state
        reasoning_buffer: list of previous reasoning step tensors
        record: False — step not written to the buffer (closed gate step):
        it still produces a contribution but does not pollute attention.
        """
        B, L, D = h.shape

        # Encode current step
        current_step = self.step_encoder(h[:, -1:, :])  # (B, 1, D)

        if reasoning_buffer is None or len(reasoning_buffer) == 0:
            new_buffer = [current_step.detach()] if record else []
            return current_step.squeeze(1), new_buffer

        # Attend to previous reasoning steps
        # Stack previous steps: (B, num_steps, D)
        prev_steps = torch.cat(reasoning_buffer, dim=1)

        q = self.step_query(current_step)  # (B, 1, D)
        k = self.step_key(prev_steps)      # (B, num_steps, D)
        v = self.step_value(prev_steps)    # (B, num_steps, D)

        # Attention: независимые сигмоид-гейты (не softmax, без sum-to-1).
        # Каждый шаг буфера взвешивается самостоятельно и может быть
        # полностью отброшен (attn=0); размазывание softmax по шагам
        # убирает селективность внимания к релевантному шагу.
        attn = torch.sigmoid(q @ k.transpose(-2, -1) / math.sqrt(D))
        context = attn @ v  # (B, 1, D)

        # Combine current step with context
        output = current_step + context
        output = self.output_proj(output)

        # Update buffer
        new_buffer = reasoning_buffer + ([current_step.detach()] if record else [])
        if len(new_buffer) > self.max_steps:
            new_buffer = new_buffer[-self.max_steps:]

        return output.squeeze(1), new_buffer


class ReasoningGate(nn.Module):
    """Per-step decision gates for adaptive reasoning depth.

    The reasoning loop runs up to `max_steps` iterations; each iteration i
    produces a reasoning vector r_i. A gate α_i = σ(Linear(h)) ∈ (0,1) decides
    how much of r_i is added to the hidden state. The loop stops early when
    the gate falls below `stop_threshold` (adaptive depth per token).

    Critical property: the gates are initialized so that the FIRST step is
    fully on (bias[0]=+4 → α≈0.98) and the REST are off (bias=-8 → α≈0.0003).
    This makes an adaptive model resume from a checkpoint trained with the
    old single-step reasoning EXACTLY as before: only step 0 contributes
    (α≈1), extra steps contribute ~0. The gates then learn to open up only
    where deeper reasoning helps the CE loss — no behavior jump on resume.
    """
    def __init__(self, D, max_steps=8, know_dim=8):
        super().__init__()
        self.max_steps = max_steps
        self.proj = nn.Linear(D, max_steps)
        nn.init.zeros_(self.proj.weight)
        self.know_proj = nn.Linear(know_dim, max_steps)
        nn.init.zeros_(self.know_proj.weight)
        nn.init.zeros_(self.know_proj.bias)
        # Гейт должен «знать, что формализуется»: вход кандидата r_i
        # (результат шага рассуждения) — связь кандидата с полем знаний
        # оценивается гейтом напрямую, а не пост-маской.
        self.r_proj = nn.Linear(D, max_steps)
        nn.init.zeros_(self.r_proj.weight)
        nn.init.zeros_(self.r_proj.bias)
        with torch.no_grad():
            self.proj.bias[0] = 10.0   # tanh ≈ 1.0 — первый шаг = старое поведение
            self.proj.bias[1:] = 0.0   # tanh = 0.0 — закрыт точно, вклад = 0 (сегментация)

    def forward(self, h, know=None, r=None):
        """h: (B, L, D) -> gates (B, L, max_steps) in (-1, 1).
        Сигмоид-гейт с отрицательными значениями (tanh, не softmax):
        независимые гейты на каждый шаг; закрыт = 0; отрицательный гейт
        вычитает вклад (шаг против знания). know: (B, know_dim) — сигнал
        собственных знаний модели. r: (B, 1, D) — кандидат шага рассуждения
        («то, что формализуется»): гейт решает по связи кандидата с полем
        знаний. Нулевые проекции сохраняют резюм со старых чекпоинтов."""
        logit = self.proj(h)
        if know is not None:
            logit = logit + self.know_proj(know).unsqueeze(1)
        if r is not None:
            logit = logit + self.r_proj(r)
        return torch.tanh(logit)

    def logits(self, h, know=None, r=None):
        """Raw logits (B, L, max_steps) for straight-through gradient."""
        logit = self.proj(h)
        if know is not None:
            logit = logit + self.know_proj(know).unsqueeze(1)
        if r is not None:
            logit = logit + self.r_proj(r)
        return logit


class ThinkingTokenHead(nn.Module):
    """
    Head that predicts thinking tokens for explicit reasoning.
    Extends the main head with reasoning token predictions.
    """
    def __init__(self, D, num_reasoning_tokens=4):
        super().__init__()
        self.reasoning_proj = nn.Linear(D, num_reasoning_tokens)

    def forward(self, h):
        """
        h: (B, L, D)
        Returns: (B, L, num_reasoning_tokens) logits for reasoning tokens
        """
        return self.reasoning_proj(h)
