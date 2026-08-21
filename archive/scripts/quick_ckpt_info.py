import torch
import sys
sys.path.insert(0, '.')

ckpt = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False)
cfg = ckpt['cfg']

print('=== Current best.pt ===')
print(f"step={ckpt.get('step')}")
print(f"val_loss={ckpt.get('best_val_loss', '?'):.4f}" if isinstance(ckpt.get('best_val_loss'), float) else f"val_loss={ckpt.get('best_val_loss')}")
print(f"D={cfg.D}, L={cfg.n_layers}, bind_K={cfg.bind_K}, vocab={cfg.vocab}")
print(f"head_mode={cfg.head_mode}, bind_twist_mode={cfg.bind_twist_mode}")
print(f"collective_layer={cfg.collective_layer}, private_mem={cfg.private_mem}")
print(f"surprisal_weight={cfg.surprisal_weight}, branch_balance={cfg.branch_balance_weight}")
print(f"gradient_checkpointing={cfg.gradient_checkpointing}")
