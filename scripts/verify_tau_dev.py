"""Quick check: _tau_dev gradient."""
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

# Direct check
dev = model.tau_config._tau_dev
print(f"_tau_dev shape: {dev.shape}")
print(f"_tau_dev requires_grad: {dev.requires_grad}")
print(f"_tau_dev grad is None: {dev.grad is None}")
if dev.grad is not None:
    print(f"_tau_dev grad norm: {dev.grad.norm().item():.6e}")
    print(f"_tau_dev grad range: [{dev.grad.min().item():.6e}, {dev.grad.max().item():.6e}]")
else:
    print("  _tau_dev.grad is None — gradient not flowing!")

# Check if _tau_dev is in optimizer param groups
print("\n_param_groups check:")
found = False
for pg_idx, pg in enumerate(model._param_groups):
    for p in pg['params']:
        if p is dev:
            print(f"  Found in param_group {pg_idx}, lr={pg['lr']:.2e}")
            found = True
            break
if not found:
    print("  NOT in any param_group!")
