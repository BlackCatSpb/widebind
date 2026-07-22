"""
WideBind text generation.
Uses HuggingFace tokenizer from the training data directory.
"""

import os, sys, math, torch, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch.nn.functional as F
from tokenizers import Tokenizer

from core import WideBindConfig, WideBindStack
from compression import FCF_CPR


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
def generate(model, prompt, max_new_tokens=128, temperature=1.0, top_k=50,
             show_mind=False, continuous_learn=False, context_mem=None):
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
    allow_write = continuous_learn or None
    
    mind_log = []
    
    recent = set()
    for step in range(max_new_tokens):
        ctx = tokens[-L:].unsqueeze(0)
        
        h = model.embed_tokens(ctx)
        out, state, _ = model(h, state, adaptive=False,
                              context_mem=context_mem, allow_write=allow_write)
        
        if show_mind and step % 10 == 0:
            info = model.layers[0].mirror.debug_mind()
            info['step'] = step
            mind_log.append(info)
            if step % 50 == 0:
                print(f'  step {step}: mem_norm={info.get("private_mem_norm",0):.4f} '
                      f'w_help={info.get("w_help",0):.4f} '
                      f'trust_diag={info.get("trust_diag_mean",0):.4f}')
        
        logits = model.lm_head(out[:, -1:, :])[0, 0]
        logits = logits / temperature
        
        # Repetition penalty: subtract fixed penalty (sign-safe)
        for rid in list(recent)[-5:]:
            logits[rid] -= 2.0
        
        if top_k > 0:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[-1:]] = -float('inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        recent.add(next_token.item())
        tokens = torch.cat([tokens, next_token], dim=0)
    
    if show_mind and mind_log:
        import json
        log_path = f'mind_log_{hash(prompt) & 0xFFFFFFFF:08x}.json'
        with open(log_path, 'w') as f:
            json.dump(mind_log, f, indent=2, default=float)
        print(f'  Mind log saved to {log_path}')
    
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
    parser.add_argument('--show-mind', action='store_true', help='Log meta-cognitive mirror stats')
    parser.add_argument('--continuous-learn', action='store_true', help='Allow memory writes during generation')
    parser.add_argument('--context-mem', type=str, default='', help='Path to .pt file with context memory tensor (G, k)')
    args = parser.parse_args()
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint (handle FCF_CPR compressed format)
    from torch.serialization import add_safe_globals
    add_safe_globals([WideBindConfig])
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_compressed' in state:
        cfg = state.get('cfg', WideBindConfig())
        cpr = FCF_CPR()
        state = cpr.load_compressed(args.checkpoint, cfg=cfg)
    cfg = state.get('cfg', WideBindConfig())
    model = WideBindStack(cfg).to(device)
    model.load_state_dict(state['model'], strict=False)
    
    print(f'Loaded checkpoint: step={state.get("step", "?")}  params={model.param_count():,}')
    
    # Prompts
    context_mem = None
    if args.context_mem:
        cm = torch.load(args.context_mem, map_location=device, weights_only=True)
        context_mem = cm.to(device)

    if args.prompt:
        text = generate(model, args.prompt, args.tokens, args.temperature, args.top_k,
                        show_mind=args.show_mind, continuous_learn=args.continuous_learn,
                        context_mem=context_mem)
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
            text = generate(model, p, 100, 0.8, 40,
                            show_mind=args.show_mind, continuous_learn=args.continuous_learn,
                            context_mem=context_mem)
            print(f'> {p}')
            print(text)
            print()
