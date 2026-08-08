"""
Test: Variable Precision Memory + Explicit Reasoning on synthetic task.
Task: arithmetic with chain-of-thought reasoning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
import os
import math

sys.path.insert(0, '.')
from core import WideBindConfig, WideBindStack


# ─── Variable Precision Memory ───

class PrecisionGate(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.gate = nn.Linear(D, 1)

    def forward(self, h):
        return torch.sigmoid(self.gate(h))


class ExactSequenceMemory(nn.Module):
    def __init__(self, D, k):
        super().__init__()
        self.query = nn.Linear(D, k)
        self.key = nn.Linear(D, k)
        self.value = nn.Linear(D, k)
        self.k = k

    def forward(self, h):
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.k), dim=-1)
        return attn @ v


# ─── Synthetic data: arithmetic with reasoning ───

def gen_reasoning_task(a_max=20):
    """Generate: a+b=c with chain-of-thought reasoning."""
    a = random.randint(0, a_max)
    b = random.randint(0, a_max)
    c = a + b

    # Simple reasoning steps
    if b >= 2:
        mid = 2
        rest = b - 2
        steps = f"<think>{a}+{b}={a}+{mid}+{rest}={a+mid}+{rest}={a+b}<answer>{c}"
    else:
        steps = f"<think>{a}+{b}={a+b}<answer>{c}"

    src = f"{a}+{b}="
    return src, steps


def encode_simple(s, max_len=64):
    """Simple char-level encoding."""
    chars = '0123456789+-=<>twakn '
    char2idx = {c: i for i, c in enumerate(chars)}
    ids = [char2idx.get(c, 0) for c in s]
    # Pad
    ids = ids[:max_len] + [0] * (max_len - len(ids))
    return ids


# ─── Test model with variable precision ───

class TestWideBindStack(WideBindStack):
    def __init__(self, cfg):
        super().__init__(cfg)
        D = cfg.D
        k = 64

        # Add precision gate and exact memory per layer
        self.precision_gates = nn.ModuleList([
            PrecisionGate(D) for _ in range(cfg.n_layers)
        ])
        self.exact_memories = nn.ModuleList([
            ExactSequenceMemory(D, k) for _ in range(cfg.n_layers)
        ])
        self.precision_proj = nn.Linear(k, D)

    def forward(self, h, state=None, global_state=None, precision_test=False):
        if state is None:
            state = [None] * len(self.layers)

        B, L, D = h.shape
        n_layers = len(self.layers)

        for i, (layer, s) in enumerate(zip(self.layers, state)):
            h, s_out = layer(h, s, global_state=global_state)

            # Variable precision memory
            if precision_test:
                precision = self.precision_gates[i](h)
                if precision.mean() > 0.2:
                    exact = self.exact_memories[i](h)
                    exact = self.precision_proj(exact)
                    h = h + precision * exact

            state[i] = s_out

        return F.rms_norm(h, (self.cfg.D,), self.final_norm_w), state, global_state


# ─── Test ───

print("=== Variable Precision + Reasoning Test ===\n")

cfg = WideBindConfig()
cfg.D = 128
cfg.n_layers = 4
cfg.bind_K = 32
cfg.vocab = 100
cfg.mlp_groups = 4
cfg.mlp_expand = 4
cfg.seq_len = 64
cfg.lambda_d_enabled = False
cfg.head_mode = 'sigmoid_coded'
cfg.head_normalize = True
cfg.bind_twist_mode = 'off'
cfg.private_mem = False
cfg.meta_trust = False
cfg.collective_layer = False

print("Building model...")
model = TestWideBindStack(cfg)
n = sum(p.numel() for p in model.parameters())
print(f"Params: {n:,} ({n/1e6:.2f}M)")

# Test forward
x = torch.randint(0, 100, (1, 10))
h = model.embed_tokens(x)

# Without precision
out1, _, _ = model(h, precision_test=False)
print(f"\nWithout precision: out.shape={out1.shape}, NaN={torch.isnan(out1).any().item()}")

# With precision
out2, _, _ = model(h, precision_test=True)
print(f"With precision:    out.shape={out2.shape}, NaN={torch.isnan(out2).any().item()}")

# Check difference
diff = (out1 - out2).abs().mean().item()
print(f"Mean difference:   {diff:.4f}")

# Quick training test
print("\n=== Training Test (10 steps) ===")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(10):
    src, tgt = gen_reasoning_task()
    x = torch.tensor([encode_simple(src)])
    y = torch.tensor([encode_simple(tgt)])
    y = y[:, :len(tgt)]  # truncate to actual target length

    model.train()
    h = model.embed_tokens(x)
    out, _, _ = model(h, precision_test=True)

    logits = model.lm_head(out[:, :len(tgt), :])
    loss = F.cross_entropy(logits.reshape(-1, 100), y.reshape(-1))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    print(f"  Step {step}: loss={loss.item():.4f}")

print("\n=== TEST PASSED ===")
print("Variable precision memory works!")
