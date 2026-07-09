"""
WideBind text generation.
Uses HuggingFace tokenizer from the training data directory.
"""

import os, sys, math, torch, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch.nn.functional as F
from tokenizers import Tokenizer

from core import WideBindConfig, WideBindStack


def load_russian_tokenizer(path=None):
    """Load BPE tokenizer from russian_tokenizer/tokenizer.json."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), '..', '..', 'fcp')
    tok_file = os.path.join(path, 'russian_tokenizer', 'tokenizer.json')
    if not os.path.exists(tok_file):
        tok_file = os.path.join(os.path.dirname(__file__), '..', 'fcp', 'russian_tokenizer', 'tokenizer.json')
    if os.path.exists(tok_file):
        tok = Tokenizer.from_file(tok_file)
        tok.enable_padding(pad_id=0, pad_token='<|pad|>')
        tok.enable_truncation(max_length=512)
        return tok
    return None


@torch.no_grad()
def generate(model, prompt, max_new_tokens=128, temperature=1.0, top_k=50):
    """Generate tokens from prompt string."""
    model.eval()
    device = next(model.parameters()).device
    L = model.cfg.seq_len
    
    # Load tokenizer
    tok = load_russian_tokenizer()
    if tok is None:
        raise FileNotFoundError('russian_tokenizer/tokenizer.json not found')
    
    # Encode prompt
    encoded = tok.encode(prompt)
    prompt_tokens = encoded.ids
    detokenize = lambda ids: tok.decode(ids, skip_special_tokens=True)
    
    tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    
    # Generate
    state = None
    
    for _ in range(max_new_tokens):
        ctx = tokens[-L:].unsqueeze(0)
        
        h = model.embed_tokens(ctx)
        out, state = model(h, state)
        
        logits = model.lm_head(out[:, -1, :])
        logits = logits / temperature
        
        if top_k > 0:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, -1:]] = -float('inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        tokens = torch.cat([tokens, next_token.squeeze(0)], dim=0)
    
    return detokenize(tokens.tolist())


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
    from torch.serialization import add_safe_globals
    add_safe_globals([WideBindConfig])
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = state.get('cfg', WideBindConfig())
    model = WideBindStack(cfg).to(device)
    model.load_state_dict(state['model'])
    
    print(f'Loaded checkpoint: step={state.get("step", "?")}  params={model.param_count():,}')
    
    # Prompts
    if args.prompt:
        text = generate(model, args.prompt, args.tokens, args.temperature, args.top_k)
        print(f'Prompt: {args.prompt}')
        print(f'Generated: {text}')
    else:
        prompts = [
            'Привет, как дела?',
            'Москва — столица',
            'В начале было Слово',
            'Искусственный интеллект',
        ]
        for p in prompts:
            text = generate(model, p, 100, 0.8, 40)
            print(f'> {p}')
            print(text)
            print()
