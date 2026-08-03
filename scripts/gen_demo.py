"""
Generate text from compressed WideBind checkpoint with tokenizer.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from compression import FCF_CPR
from core import WideBindStack
from core.checkpoints import find_latest_ckpt
from tokenizers import Tokenizer


def load_russian_tokenizer(path=None, extended=True):
    names = (['tokenizer_v65536.json', 'tokenizer.json'] if extended
             else ['tokenizer.json'])
    if path is None:
        path = r'C:\Users\black\OneDrive\Desktop\fcp'
    for name in names:
        tok_file = os.path.join(path, 'russian_tokenizer', name)
        if not os.path.isfile(tok_file):
            tok_file = os.path.join(os.path.dirname(__file__), '..', 'wb', 'russian_tokenizer', name)
        if not os.path.isfile(tok_file):
            tok_file = os.path.join(os.path.dirname(__file__), '..', 'fcp', 'russian_tokenizer', name)
        if os.path.isfile(tok_file):
            return Tokenizer.from_file(tok_file)
    raise FileNotFoundError('russian_tokenizer/tokenizer*.json not found')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"})')

# Load
ckpt_path = find_latest_ckpt(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
if ckpt_path is None:
    print('No checkpoint found in checkpoints/; pass --ckpt or put one there.')
    sys.exit(1)
print(f'Loading latest checkpoint: {ckpt_path}')
cpr = FCF_CPR()
ckpt = cpr.load_compressed(ckpt_path)
cfg = ckpt['cfg']

model = WideBindStack(cfg).to(device)
model.load_state_dict(ckpt['model'], strict=False)
model.eval()
model.half()

# Tokenizer
tok = load_russian_tokenizer()
print(f'Vocab: {tok.get_vocab_size()}')

# Prompt
prompt_text = 'В начале было'
prompt_ids = tok.encode(prompt_text).ids
if max(prompt_ids, default=0) >= cfg.vocab:
    base_tok = load_russian_tokenizer(extended=False)
    if base_tok is not None and max(base_tok.encode(prompt_text).ids, default=0) < cfg.vocab:
        tok = base_tok
        prompt_ids = base_tok.encode(prompt_text).ids
print(f'Prompt: "{prompt_text}" -> {prompt_ids}')
print()

# Generate
prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
gen_len = 256

with torch.no_grad():
    h = model.embed_tokens(prompt)
    state = None
    out, state, _ = model(h, state, adaptive=False)

    tokens = prompt_ids[:]
    # First token comes from the prompt forward pass itself (not from
    # feeding the hidden state back as an embedding — that input is OOD).
    logits = model.lm_head(out).float()
    next_id = logits[:, -1].argmax(dim=-1).item()
    tokens.append(next_id)
    x = model.embed_tokens(torch.tensor([[next_id]], device=device))
    t0 = time.time()
    for i in range(1, gen_len):
        out, state, _ = model(x, state, adaptive=False)
        logits = model.lm_head(out).float()
        next_id = logits[:, -1].argmax(dim=-1).item()
        tokens.append(next_id)
        x = model.embed_tokens(torch.tensor([[next_id]], device=device))
        
        if i % 32 == 0:
            dt = time.time() - t0
            tok_s = (i + 1) / max(dt, 1e-10)
            print(f'  gen {i+1:3d}/{gen_len} tok/s={tok_s:.0f} last_id={next_id}')

t_total = time.time() - t0
print(f'\nGeneration: {gen_len} tok in {t_total:.1f}s ({gen_len/t_total:.0f} tok/s)')
print()

# Decode
text = tok.decode(tokens, skip_special_tokens=True)
print('=' * 60)
print(text)
print('=' * 60)
