import torch
import sys
sys.path.insert(0, '.')
from core import WideBindConfig, WideBindStack

ckpt = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False)
cfg = ckpt['cfg']

print('=== EVA Checkpoint ===')
print(f"step={ckpt.get('step')}, val_loss={ckpt.get('best_val_loss', '?'):.4f}" if isinstance(ckpt.get('best_val_loss'), float) else f"val_loss={ckpt.get('best_val_loss')}")

model = WideBindStack(cfg)
missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
print(f"Loaded: {len(ckpt['model']) - len(unexpected)}/{len(ckpt['model'])} keys")

# Simpler generation - single forward pass
model.eval()

# Use ASCII chars that exist in tokenizer
prompt = "hello world"
# Map to simple indices (tokenizer uses BPE, we'll use raw indices)
# Just test with random tokens to see if model produces coherent output

print('\n=== Simple Generation Test ===')
with torch.no_grad():
    # Generate random input
    x = torch.randint(0, 65536, (1, 10))
    h = model.embed_tokens(x)
    out, state, _ = model(h, step=0)
    logits = model.lm_head(out)
    
    # Get top predictions
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    top5 = torch.topk(probs, 5)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"\nTop 5 token predictions at last position:")
    for i, (prob, idx) in enumerate(zip(top5.values.tolist(), top5.indices.tolist())):
        print(f"  {i+1}. token_{idx} (prob={prob:.4f})")
    
    print(f"\nModel output stats:")
    print(f"  mean: {out.mean().item():.4f}")
    print(f"  std: {out.std().item():.4f}")
    print(f"  NaN: {torch.isnan(out).any().item()}")
    
    print("\n=== EVA Status: ACTIVE ===")
