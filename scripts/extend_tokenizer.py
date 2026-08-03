"""
WideBind: extend Russian tokenizer with whole-word tokens.

Strategy (measured on corpus):
  baseline tok/char 0.2917
  +15536 spaced whole-words (frequent, currently split into >=2 BPE
  tokens) -> vocab 65536, tok/char 0.2751 (5.7% faster overall)
  on dialogue-heavy prose -> up to 27% fewer tokens

Old token ids and sparse codes are preserved (prefix-stable), so the
existing trained weights stay valid. Vocab stays a multiple of 16
(segment-load balance: V*S/K integer).

Usage:
  python scripts/extend_tokenizer.py
  -> writes wb/russian_tokenizer/tokenizer_v65536.json
"""
import sys, glob, random, re, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tokenizers import Tokenizer
import numpy as np
import torch

BASE = os.path.join(os.path.dirname(__file__), '..', 'wb', 'russian_tokenizer', 'tokenizer.json')
STREAMS = os.path.join(os.path.dirname(__file__), '..', 'wb', 'token_stream_*_clean.bin')
OUT = os.path.join(os.path.dirname(__file__), '..', 'wb', 'russian_tokenizer', 'tokenizer_v65536.json')
VOCAB_NEW = 65536
ADD = VOCAB_NEW - 50000  # 15536
SAMPLE_CHARS = 12_000_000


def load(truncate=10**9):
    tok = Tokenizer.from_file(BASE)
    tok.enable_padding(pad_id=0, pad_token='<|pad|>')
    tok.enable_truncation(max_length=truncate)
    return tok


def word_freq():
    tok = load()
    files = sorted(glob.glob(STREAMS))
    counter = {}
    n = 0
    total_avail = sum(os.path.getsize(f) for f in files)
    for idx, f in enumerate(files):
        arr = np.memmap(f, dtype=np.uint16, mode='r')
        step = 1000  # dense enough sample
        for st in range(0, arr.size - 500, step):
            text = tok.decode(arr[st:st + 400].tolist(), skip_special_tokens=True)
            for w in re.findall(r'[\u0410-\u0451]+', text.lower()):
                counter[w] = counter.get(w, 0) + 1
            n += len(text)
            if n >= SAMPLE_CHARS:
                break
        del arr
        pct = min(100.0, n / SAMPLE_CHARS * 100)
        print(f'  [{idx+1}/{len(files)}] {os.path.basename(f):40s} chars={n/1e6:.1f}M '
              f'({pct:.0f}%) words={len(counter)}', flush=True)
        if n >= SAMPLE_CHARS:
            break
    print(f'done: sampled {n/1e6:.1f}M chars, {len(counter)} unique words', flush=True)
    return counter


def candidates(counter, tok, max_add=ADD):
    cands = []
    for w, c in counter.items():
        if len(w) < 4 or c < 2:
            continue
        cur = len(tok.encode(' ' + w).ids)
        if cur >= 2:
            cands.append((w, c, cur))
    cands.sort(key=lambda x: -(x[2] - 1) * len(x[0]) * x[1])
    return cands[:max_add]


def main():
    tok = load()
    print('=== word frequency ===', flush=True)
    counter = word_freq()
    print('=== candidates ===', flush=True)
    cands = candidates(counter, tok)
    print(f'{len(cands)} candidates', flush=True)
    words = [' ' + w for w, c, cur in cands]
    print(f'adding {len(words)} whole-word tokens...', flush=True)
    for i in range(0, len(words), 1000):
        tok.add_tokens(words[i:i + 1000])
        print(f'  added {min(i+1000, len(words))}/{len(words)}', flush=True)
    n = tok.get_vocab_size()
    print(f'vocab: {n} (target {VOCAB_NEW})', flush=True)
    assert n == VOCAB_NEW, f'vocab {n} != {VOCAB_NEW}'

    # consistency checks
    from core.vsa_utils import sparse_block_codes
    old_vocab = Tokenizer.from_file(BASE).get_vocab()
    new_vocab = tok.get_vocab()
    mism = [t for t, i in old_vocab.items() if new_vocab.get(t) != i]
    print(f'old token->id mismatches: {len(mism)}')
    oc = sparse_block_codes(50000, K=32, S=6)
    nc = sparse_block_codes(n, K=32, S=6)
    print(f'codes prefix stable: {bool(torch.equal(oc, nc[:50000]))}')
    print(f'vocab % 16 == 0: {n % 16 == 0}')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok.save(OUT)
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main()
