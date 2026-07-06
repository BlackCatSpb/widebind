import torch
s = torch.load('checkpoints/best.pt', map_location='cpu')
print(f'Step: {s["step"]}')
print(f'Best val_loss: {s["best_val_loss"]:.4f}')
print(f'CFG lr: {s["cfg"].lr}')
print(f'Model keys: {len(s["model"])}')
