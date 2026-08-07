import torch
import sys
sys.path.insert(0, '.')
from core import WideBindConfig, WideBindStack

ckpt = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False)
cfg = ckpt['cfg']
model = WideBindStack(cfg)
model.load_state_dict(ckpt['model'], strict=False)
model.eval()

# Use English prompts to avoid encoding issues in PowerShell
prompts = [
    "Hello",
    "The capital of Russia is",
    "Artificial intelligence",
    "In the beginning was the Word",
]

results = []
with torch.no_grad():
    for prompt in prompts:
        token_id = hash(prompt) % 65536
        x = torch.tensor([[token_id]])
        h = model.embed_tokens(x)
        out, state, _ = model(h, step=0)
        logits = model.lm_head(out[:, -1:, :])
        probs = torch.softmax(logits[0, 0], dim=-1)
        top3 = torch.topk(probs, 3)
        
        results.append({
            'prompt': prompt,
            'top_tokens': top3.indices.tolist(),
            'top_probs': [f"{p:.6f}" for p in top3.values.tolist()],
            'output_mean': f"{out.mean().item():.4f}",
            'output_std': f"{out.std().item():.4f}",
        })

# Save results
with open('generation_results.txt', 'w', encoding='utf-8') as f:
    f.write("=== EVA Generation Results ===\n")
    f.write(f"Checkpoint: step={ckpt['step']}, val_loss={ckpt.get('best_val_loss', 0):.4f}\n")
    f.write(f"Params: {sum(p.numel() for p in model.parameters()):,}\n\n")
    
    for r in results:
        f.write(f"Prompt: {r['prompt']}\n")
        f.write(f"  Top tokens: {r['top_tokens']}\n")
        f.write(f"  Top probs: {r['top_probs']}\n")
        f.write(f"  Output stats: mean={r['output_mean']}, std={r['output_std']}\n\n")

print("Results saved to generation_results.txt")
print(f"Checkpoint: step={ckpt['step']}, val_loss={ckpt.get('best_val_loss', 0):.4f}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
print("\nGeneration stats:")
for r in results:
    print(f"  {r['prompt']}: top_token={r['top_tokens'][0]}, prob={r['top_probs'][0]}")

# Also test full generation with generate.py
print("\n=== Full generation test ===")
import subprocess
subprocess.run(['python', 'scripts/generate.py', 'checkpoints/best.pt', 
                '--prompt', 'Hello world', '--tokens', '20', '--temperature', '0.7'])
