"""Verify with 24-layer config (production)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from core.config import WideBindConfig
from core.stack import WideBindStack

cfg = WideBindConfig(
    D=2560, n_layers=24, vocab=65536, bind_K=32,
    bridge_dim=256, bridge_conn=0.1, memory_bank=True,
    seq_len=32, batch_size=1, use_amp=False,
    mem_l1_slots=3, mem_l2_slots=8, mem_l3_concepts=4,
    collective_S=4, collective_layer_idx=None, mlp_groups=32
)
model = WideBindStack(cfg).float()
model.train()

tokens = torch.randint(0, cfg.vocab, (1, 32))
h = model.embed_tokens(tokens)
targets = tokens[:, 1:]
h_in = h[:, :-1]

model.zero_grad()
h_out, state, gs, rb = model(h_in, step=10000, adaptive=True, tokens=tokens[:, :-1])
ce_loss, aux = model.compute_losses(h_out, targets)

total = ce_loss
for v in aux.values():
    if isinstance(v, torch.Tensor):
        total = total + v
total.backward()

print("=== P0.1: layer_bridge_gate.log_tau ===")
lbg_alive = 0
lbg_dead = 0
for name, p in model.named_parameters():
    if 'layer_bridge_gate' in name and 'log_tau' in name:
        g = p.grad.norm().item() if p.grad is not None else 0.0
        if g > 1e-10:
            lbg_alive += 1
        else:
            lbg_dead += 1
print(f"  ALIVE: {lbg_alive}, DEAD: {lbg_dead}")

print("\n=== P0.2: memory_bank tau_config ===")
print(f"  has tau_config: {hasattr(model.memory_bank, 'tau_config')}")
for lname in ['l1', 'l2', 'l3']:
    level = getattr(model.memory_bank, lname)
    v = level.log_tau.data.item()
    print(f"  {lname}.log_tau={v:.4f}")

print("\n=== P0.3: _tau_dev ===")
for name, p in model.named_parameters():
    if '_tau_dev' in name:
        g = p.grad.norm().item() if p.grad is not None else 0.0
        print(f"  {name}: val_range=[{p.data.min().item():.4f}, {p.data.max().item():.4f}], grad_norm={g:.6e}")

print("\n=== Tau ladder ===")
for l in range(0, cfg.n_layers, 4):
    print(f"  tau[{l}]={model.tau_config.tau_l[l].item():.2f}")
print(f"  tau[{cfg.n_layers-1}]={model.tau_config.tau_l[-1].item():.2f}")

print("\n=== Aux losses with gradients ===")
for k, v in aux.items():
    if isinstance(v, torch.Tensor):
        g = v.grad_fn
        print(f"  {k}: val={v.item():.6f}, has_grad_fn={g is not None}")
