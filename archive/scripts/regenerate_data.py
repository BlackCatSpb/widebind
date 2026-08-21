"""
WideBind: regenerate token streams from cp1251 source (fix for UTF-8 errors='replace' bug).

Old pipeline read Windows-1251 source files as UTF-8 with errors='replace', which
turned every Cyrillic byte into U+FFFD (tokens 125/127/175) — ~92% of the data was
garbage. This script reads sources as cp1251 and writes uint16 flat token streams.

Usage:
    python scripts/regenerate_data.py --genre FANTASY --out wb/token_stream_FANTASY_clean.bin
"""

import os, sys, re, time, argparse, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from tokenizers import Tokenizer

SRC_ROOT = r'C:\Users\black\OneDrive\Desktop\EVA-Ai\libinpoc_txt\main-russian'


def clean_text(text: str) -> str:
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'\s+', ' ', line)
        line = re.sub(r'\.{4,}', '...', line)
        line = re.sub(r'!{2,}', '!', line)
        line = re.sub(r'\?{2,}', '?', line)
        lines.append(line)
    return '\n'.join(lines)


def collect_files(src, genre):
    items = []
    genre_dir = os.path.join(src, genre)
    if not os.path.isdir(genre_dir):
        return items
    for fn in sorted(os.listdir(genre_dir)):
        if fn.endswith('.txt'):
            items.append(os.path.join(genre_dir, fn))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--genre', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--src', default=SRC_ROOT)
    ap.add_argument('--tokenizer', default=r'C:\Users\black\OneDrive\Desktop\FCP\russian_tokenizer\tokenizer.json')
    ap.add_argument('--encoding', default='cp1251')
    args = ap.parse_args()

    files = collect_files(args.src, args.genre)
    if not files:
        print(f'[ERROR] no .txt files for genre {args.genre} in {args.src}')
        sys.exit(1)
    print(f'Files: {len(files)}')

    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f'Tokenizer vocab: {tokenizer.get_vocab_size()}')

    # ─── Pass 1: count tokens ───
    counts = []
    t0 = time.perf_counter()
    for i, path in enumerate(files):
        with open(path, 'r', encoding=args.encoding, errors='replace') as f:
            raw = f.read()
        text = clean_text(raw)
        counts.append(len(tokenizer.encode(text).ids))
        if (i + 1) % 250 == 0:
            print(f'  count [{i+1}/{len(files)}] {sum(counts)//1e3:.0f}K tok')
    total = sum(counts)
    print(f'Counted: {total//1e6:.0f}M tokens in {time.perf_counter()-t0:.0f}s')

    # ─── Pass 2: write uint16 ───
    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    arr = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=(total,))
    pos = 0
    t0 = time.perf_counter()
    for i, (path, n) in enumerate(zip(files, counts)):
        if n == 0:
            continue
        with open(path, 'r', encoding=args.encoding, errors='replace') as f:
            raw = f.read()
        ids = tokenizer.encode(clean_text(raw)).ids
        assert len(ids) == n, f'count mismatch {path}: {len(ids)} vs {n}'
        arr[pos:pos + n] = ids
        pos += n
        if (i + 1) % 250 == 0:
            print(f'  write [{i+1}/{len(files)}] {pos//1e6:.0f}M tok')
    arr.flush()
    del arr
    size_gb = os.path.getsize(out_path) / 1e9
    print(f'Saved: {pos//1e6:.0f}M tokens, {size_gb:.2f} GB, {time.perf_counter()-t0:.0f}s')

    # ─── Health check ───
    data = np.memmap(out_path, dtype=np.uint16, mode='r')
    off = len(data) // 2
    sample = data[off:off + 2_000_000].tolist()
    import collections
    c = collections.Counter(sample)
    fffd = (c.get(175, 0) + c.get(127, 0) + c.get(125, 0)) / len(sample)
    txt = tokenizer.decode(sample[:2000], skip_special_tokens=True)
    cyr = sum(1 for ch in txt if 0x0400 <= ord(ch) <= 0x04FF)
    print(f'Health: len={len(data):,} distinct(top2M)={len(c)} U+FFFD={fffd:.1%} '
          f'cyrillicChars(in2k)={cyr}')
    print('Top tokens:', c.most_common(8))


if __name__ == '__main__':
    main()
