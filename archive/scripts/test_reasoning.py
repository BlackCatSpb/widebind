import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
sys.path.insert(0, '.')
from core import WideBindConfig, WideBindStack

# Synthetic task: a + b = c (arithmetic with chain-of-thought)
# Format: "3+5=8" or with thinking: "3+5=<think>3+5=8<answer>8"

VOCAB_SIZE = 100
D = 128
L = 2
G = 4
k = 16
K = 16

chars = '0123456789+-=<>twakn'
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}

THINK = char2idx['<']
ANSWER = char2idx['>']

def gen_problem(a_max=20, with_thinking=False):
    a = random.randint(0, a_max)
    b = random.randint(0, a_max)
    c = a + b
    src = f"{a}+{b}="
    if with_thinking:
        # Simple reasoning: repeat + answer
        tgt = f"{a}+{b}={c}"
    else:
        tgt = f"{c}"
    return src, tgt

def encode(s):
    return [char2idx[c] for c in s]

def decode(indices):
    return ''.join(idx2char.get(i, '?') for i in indices)

# Build model
cfg = WideBindConfig()
cfg.D = D
cfg.n_layers = L
cfg.bind_K = K
cfg.vocab = VOCAB_SIZE
cfg.mlp_groups = G
cfg.mlp_expand = 4
cfg.seq_len = 32
cfg.batch_size = 1
cfg.lambda_d_enabled = False
cfg.head_mode = 'sigmoid_coded'
cfg.head_normalize = True
cfg.bind_twist_mode = 'trajectory_spiral'
cfg.bind_traj_dims = 1
cfg.hybrid_alpha_max = 0.7
cfg.hybrid_alpha_min = 0.3
cfg.w_pred_scale_init = 3.0
cfg.private_mem = True
cfg.meta_trust = False
cfg.collective_layer = True
cfg.collective_layer_idx = None
cfg.collective_read_out = False
cfg.uncert_theta = 0.5
cfg.uncert_kappa = 3.0
cfg.contra_thresh = -0.1
cfg.contra_gain = 6.0
cfg.maturity_thresh = 0.12
cfg.mirror_k = k
cfg.mirror_k_staircase = False
cfg.lr = 1e-3
cfg.grad_clip = 0.5

model = WideBindStack(cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f'Model: {n_params:,} params ({n_params/1e6:.2f}M)')

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

# Training loop
def run_epoch(n_steps=100, with_thinking=False):
    model.train()
    total_loss = 0
    correct = 0
    state = None
    for step in range(n_steps):
        src, tgt = gen_problem(with_thinking=with_thinking)
        x = torch.tensor([encode(src)])
        y = torch.tensor([encode(tgt)])

        model.train()
        h = model.embed_tokens(x)
        out, state, _, _ = model(h, state, step=step)
        def detach_state(st):
            if st is None:
                return None
            if isinstance(st, torch.Tensor):
                return st.detach()
            if isinstance(st, (tuple, list)):
                return type(st)(detach_state(x) for x in st)
            return st
        state = detach_state(state)

        loss, _ = model.compute_losses(out, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if step % 20 == 0:
            model.eval()
            with torch.no_grad():
                for _ in range(5):
                    src_test, tgt_test = gen_problem(a_max=20)
                    x_test = torch.tensor([encode(src_test)])
                    h_test = model.embed_tokens(x_test)
                    out_test, _, _, _, _ = model(h_test, None, step=0)
                    pred = out_test[0, -len(tgt_test):, :].argmax(dim=-1)
                    pred_str = decode(pred.tolist())
                    if tgt_test == pred_str:
                        correct += 1
            model.train()

    return total_loss / n_steps, correct / max(1, n_steps // 20)

print('\n=== Training WITHOUT reasoning ===')
for epoch in range(10):
    loss, acc = run_epoch(n_steps=50, with_thinking=False)
    print(f'Epoch {epoch}: loss={loss:.4f}, acc={acc:.2f}')

print('\n=== Training WITH reasoning (CoT) ===')
for epoch in range(10):
    loss, acc = run_epoch(n_steps=50, with_thinking=True)
    print(f'Epoch {epoch}: loss={loss:.4f}, acc={acc:.2f}')

# Debug: show some predictions
print('\n=== Sample predictions (WITH reasoning) ===')
model.eval()
with torch.no_grad():
    for _ in range(8):
        src, tgt = gen_problem(a_max=9, with_thinking=False)
        x = torch.tensor([encode(src)])
        h = model.embed_tokens(x)
        hidden, _, _, _, _ = model(h, None, step=0)
        logits = model.lm_head(hidden)
        pred = logits[0, -len(tgt):, :].argmax(dim=-1)
        pred_str = decode(pred.tolist())
        match = 'OK' if pred_str == tgt else 'WRONG'
        print(f'  {src} -> pred={pred_str!r} tgt={tgt!r} [{match}]')

print('\nDone!')
