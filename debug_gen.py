"""Debug generation: show raw token IDs."""
import sys, torch
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)
from torch.serialization import add_safe_globals
from config import WideBindConfig
from core import WideBindStack
from generate import load_russian_tokenizer
add_safe_globals([WideBindConfig])

ckpt = torch.load('checkpoints/step_5000.pt', map_location='cpu', weights_only=True)
cfg = ckpt.get('cfg', WideBindConfig())
model = WideBindStack(cfg)
model.load_state_dict(ckpt['model'])
model.eval()
device = 'cpu'
tok = load_russian_tokenizer()

# Encode prompt
encoded = tok.encode('Привет, как дела?')
tokens = torch.tensor(encoded.ids, dtype=torch.long, device=device)

# Generate 20 tokens, collect logits + ids
state = None
for i in range(20):
    ctx = tokens[-128:].unsqueeze(0)
    h = model.embed_tokens(ctx)
    out, state = model(h, state)
    logits = model.lm_head(out[:, -1, :])
    probs = torch.softmax(logits / 1.0, dim=-1)
    next_id = torch.multinomial(probs, 1).item()
    tokens = torch.cat([tokens, torch.tensor([next_id])], dim=0)
    
    top5 = torch.topk(logits, 5)
    top5_ids = top5.indices[0].tolist()
    top5_toks = [tok.id_to_token(t) for t in top5_ids]
    top5_probs = torch.softmax(logits, dim=-1)[0][top5.indices[0]].tolist()
    print(f'  step {i}: pred={next_id} top5={list(zip(top5_ids, top5_toks, [f"{p:.3f}" for p in top5_probs]))}')

generated = tokens.tolist()
text = tok.decode(generated, skip_special_tokens=True)
print(f'Full decode: {repr(text)}')
