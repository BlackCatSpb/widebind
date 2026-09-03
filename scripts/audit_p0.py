"""P0 audit: gradient flow, tau_config pass-through, _tau_dev, LLRD."""
import torch
import sys
sys.path.insert(0, '.')
from core.config import WideBindConfig
from core.stack import WideBindStack

cfg = WideBindConfig(
    D=2560, n_layers=24, vocab=65536, bind_K=32,
    bridge_dim=256, bridge_conn=0.1, memory_bank=True,
    seq_len=128, batch_size=1, use_amp=False,
    mem_l1_slots=3, mem_l2_slots=8, mem_l3_concepts=4,
    collective_S=4, collective_layer_idx=None,
    mlp_groups=32,
)
model = WideBindStack(cfg)
model = model.float()

# === P0.1: layer_bridge_gate gradient flow ===
print("=== P0.1: layer_bridge_gate gradient check ===")
for name, p in model.named_parameters():
    if "layer_bridge_gate" in name and "log_tau" in name:
        print(f"  {name}: shape={p.shape}, requires_grad={p.requires_grad}")

tokens = torch.randint(0, cfg.vocab, (1, 128))
h = model.embed_tokens(tokens)
model.zero_grad()
out = model(h, step=10000, adaptive=True, tokens=tokens)
loss = out[0].sum() if isinstance(out, tuple) else out.sum()
loss.backward()

print("  Gradient norms after backward at step=10000:")
dead_params = []
for name, p in model.named_parameters():
    if "layer_bridge_gate" in name and "log_tau" in name:
        g = p.grad.norm().item() if p.grad is not None else 0.0
        print(f"  {name}: grad_norm={g:.6e}")
        if g < 1e-10:
            dead_params.append(name)
print(f"  DEAD params (grad < 1e-10): {len(dead_params)}")

# === P0.2: memory_bank tau_config ===
print("\n=== P0.2: memory_bank tau_config ===")
if model.memory_bank is not None:
    has_tc = hasattr(model.memory_bank, "tau_config")
    print(f"  Has tau_config: {has_tc}")
    if has_tc:
        print(f"  tau_config is None: {model.memory_bank.tau_config is None}")
    for lname in ["l1", "l2", "l3"]:
        level = getattr(model.memory_bank, lname, None)
        if level is not None and hasattr(level, "log_tau"):
            v = level.log_tau.data
            print(f"  {lname}.log_tau: ndim={v.ndim}, val={v.flatten()[:5].tolist()}")
            if hasattr(level, "_init_log_tau"):
                iv = level._init_log_tau.data
                print(f"  {lname}._init_log_tau: ndim={iv.ndim}, val={iv.flatten()[:5].tolist()}")
else:
    print("  memory_bank is None (disabled)")

# === P0.3: _tau_dev ===
print("\n=== P0.3: _tau_dev initial distribution ===")
dev = model.tau_config._tau_dev
print(f"  min={dev.min().item():.4f}, max={dev.max().item():.4f}, mean={dev.mean().item():.4f}")
print(f"  All zeros: {(dev == 0).all().item()}")
print(f"  Positive: {(dev > 0).sum().item()}/{dev.numel()}")
print(f"  Negative: {(dev < 0).sum().item()}/{dev.numel()}")

# === P0.4: LLRD duplication ===
print("\n=== P0.4: LLRD check ===")
print(f"  cfg.tau_llrd_gamma = {cfg.tau_llrd_gamma}")
print(f"  cfg.llrd = {getattr(cfg, 'llrd', 'NOT SET')}")
# Check param_groups for index-based LLRD
groups = model.param_groups(lr=cfg.lr)
print(f"  param_groups count: {len(groups)}")
for g in groups:
    lr_val = g['lr']
    n_params = len(g['params'])
    if n_params > 0:
        print(f"    lr={lr_val:.2e}, params={n_params}")

# === P0.5: _tau_intent_dev (dead?) ===
print("\n=== P0.5: _tau_intent_dev check ===")
has_intent_dev = False
for name, p in model.named_parameters():
    if "tau_intent_dev" in name:
        has_intent_dev = True
        g = p.grad.norm().item() if p.grad is not None else 0.0
        print(f"  {name}: shape={p.shape}, requires_grad={p.requires_grad}, grad_norm={g:.6e}")
if not has_intent_dev:
    print("  NOT FOUND in parameters (already removed)")
