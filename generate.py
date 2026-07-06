"""
WideBind text generation.
"""

import os, sys, math, torch, pickle, glob, json
import torch.nn.functional as F
import numpy as np

from config import WideBindConfig
from core import WideBindStack


@torch.no_grad()
def generate(model, prompt, max_new_tokens=128, temperature=1.0, top_k=50):
    """Generate tokens from prompt."""
    model.eval()
    device = next(model.parameters()).device
    L = model.cfg.seq_len
    
    # Handle prompt as tokens or string
    if isinstance(prompt, str):
        # Try loading tokenizer
        tokenizer_file = os.path.join(os.path.dirname(__file__), 'tokenizer', 'tokenizer.pkl')
        if os.path.exists(tokenizer_file):
            with open(tokenizer_file, 'rb') as f:
                tokenizer = pickle.load(f)
            prompt_tokens = tokenizer.encode(prompt)
            detokenize = lambda ids: tokenizer.decode(ids)
        else:
            # Fallback: character-level
            chars = sorted(list(set(prompt)))
            stoi = {ch: i for i, ch in enumerate(chars)}
            prompt_tokens = [stoi.get(c, 0) for c in prompt]
            detokenize = lambda ids: ''.join(chr(i) if i < 128 else '?' for i in ids)
    else:
        prompt_tokens = prompt
        detokenize = lambda ids: ' '.join(str(i) for i in ids)
    
    # Pad / truncate to full blocks
    tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    
    # Generate
    generated = tokens.tolist()
    state = None
    
    for _ in range(max_new_tokens):
        # Take last L tokens
        ctx = tokens[-L:].unsqueeze(0)  # (1, ctx_len)
        
        h = model.embed_tokens(ctx)
        out, state = model(h, state)
        
        logits = model.lm_head(out[:, -1, :])  # (1, vocab)
        logits = logits / temperature
        
        if top_k > 0:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, -1:]] = -float('inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        generated.append(next_token.item())
        tokens = torch.cat([tokens, next_token.squeeze(0)])
    
    return detokenize(generated)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Path to .pt checkpoint')
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    state = torch.load(args.checkpoint, map_location=device)
    cfg = state.get('cfg', WideBindConfig())
    model = WideBindStack(cfg).to(device)
    model.load_state_dict(state['model'])
    
    print(f'Loaded checkpoint: step={state.get("step", "?")}  params={model.param_count():,}')
    
    # Prompt
    if args.prompt:
        text = generate(model, args.prompt, args.tokens, args.temperature, args.top_k)
        print(f'Prompt: {args.prompt}')
        print(f'Generated: {text}')
    else:
        # Interactive mode
        print('Enter prompts (empty line to quit):')
        while True:
            prompt = input('> ')
            if not prompt:
                break
            text = generate(model, prompt, args.tokens, args.temperature, args.top_k)
            print(text)
