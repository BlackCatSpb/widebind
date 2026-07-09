"""
Generate text from compressed WideBind checkpoint with tokenizer.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from compression import FCF_CPR
from core import WideBindStack
from tokenizers import Tokenizer


def load_russian_tokenizer(path=None):
    if path is None:
        path = r'C:\Users\black\OneDrive\Desktop\fcp'
    tok_file = os.path.join(path, 'russian_tokenizer', 'tokenizer.json')
    if not os.path.isfile(tok_file):
        tok_file = os.path.join(os.path.dirname(__file__), '..', 'fcp', 'russian_tokenizer', 'tokenizer.json')
    return Tokenizer.from_file(tok_file)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"})')

# Load
ckpt_path = r'C:\Users\black\OneDrive\Desktop\WideBind\checkpoints\step_15000_infer.pt'
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
print(f'Prompt: "{prompt_text}" -> {prompt_ids}')
print()

# Generate
prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
gen_len = 256

with torch.no_grad():
    h = model.embed_tokens(prompt)
    state = None
    out, state = model(h, state)

    tokens = prompt_ids[:]
    x = out[:, -1:, :]
    t0 = time.time()
    for i in range(gen_len):
        out, state = model(x, state)
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
