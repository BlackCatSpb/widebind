import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, math

ck = torch.load('checkpoints/best 4.pt', map_location='cpu', weights_only=False)
print('=== BEST 4 CHECKPOINT ===')
print('keys:', list(ck.keys()))
if 'step' in ck: print('step:', ck['step'])
if 'val_loss' in ck: print('val_loss:', ck['val_loss'])
if 'best_val_loss' in ck: print('best_val_loss:', ck['best_val_loss'])
if 'cfg' in ck: print('cfg:', ck['cfg'])
if 'active_depth' in ck: print('active_depth:', ck['active_depth'])
sd = ck.get('model', ck.get('state_dict', {}))
if not sd:
    print('No model weights found')
else:
    params = sum(v.numel() for v in sd.values())
    print('params:', params)
    for k,v in sd.items():
        if 'maturation' in k.lower() or 'mat_gate' in k.lower() or k.endswith('.mat'):
            print(f'  {k}: shape={list(v.shape)} val={v.mean().item():.4f} [{v.min().item():.4f}, {v.max().item():.4f}]')
    for k,v in sd.items():
        if 'bridge' in k.lower():
            print(f'  {k}: shape={list(v.shape)} mean={v.mean().item():.6f}')
    for k,v in sd.items():
        if 'concept' in k.lower():
            print(f'  {k}: shape={list(v.shape)} mean={v.mean().item():.6f}')
    for k,v in sd.items():
        if 'head' in k.lower():
            print(f'  {k}: shape={list(v.shape)} mean={v.mean().item():.6f}')
    for k,v in sd.items():
        if 'mirror' in k.lower():
            print(f'  {k}: shape={list(v.shape)} mean={v.mean().item():.6f}')
    nan_count = sum(1 for v in sd.values() if torch.isnan(v).any())
    inf_count = sum(1 for v in sd.values() if torch.isinf(v).any())
    print(f'NaN tensors: {nan_count}, Inf tensors: {inf_count}')
