"""
Check tokenizer efficiency across the training streams.

Per genre (token_stream_${GENRE}_clean.bin):
  - file size (MB)
  - token count under base (50000) and extended (65536) tokenizers
  - relative change (should be negative for Russian prose: ~-5.7% corpus avg,
    dialog prose up to -27%).

Read-only. Exits non-zero if extended is EVER WORSE than base overall
(a regression meaning the extension hurt).

Usage:
    python scripts/check_tokenizer_efficiency.py [--data-dir wb] [--sample 2000000]
"""

import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tokenizers import Tokenizer

TOK_DIR = os.path.join(os.path.dirname(__file__), '..', 'wb', 'russian_tokenizer')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='wb')
    parser.add_argument('--sample', type=int, default=2_000_000,
                        help='max tokens decoded per stream (0 = whole stream)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='limit number of streams processed (quick check)')
    args = parser.parse_args()

    # Load tokenizers RAW (no truncation/padding — those would skew the count).
    base = Tokenizer.from_file(os.path.join(TOK_DIR, 'tokenizer.json'))
    ext = Tokenizer.from_file(os.path.join(TOK_DIR, 'tokenizer_v65536.json'))

    files = sorted(glob.glob(os.path.join(args.data_dir, 'token_stream_*_clean.bin'))
                   + glob.glob(os.path.join(args.data_dir, 'token_stream_*.bin')))
    files = sorted({f: None for f in files})  # de-dup, prefer _clean first match
    if not files:
        raise SystemExit(f'no token_stream_*.bin in {args.data_dir}')
    if args.max_files:
        files = files[:args.max_files]

    print(f"{'genre':<22}{'MB':>8}{'tok(base)':>13}{'tok(ext)':>13}{'delta':>9}")
    tot_b = tot_e = 0
    worst_name, worst_delta = '', float('inf')
    for fp in files:
        data = np.memmap(fp, dtype=np.uint16, mode='r')
        n_use = args.sample if args.sample else len(data)
        ids = data[:n_use].tolist()
        mb = os.path.getsize(fp) / 1024 / 1024
        name = os.path.basename(fp).replace('token_stream_', '') \
            .replace('_clean.bin', '').replace('.bin', '')
        text = ext.decode(ids, skip_special_tokens=True)
        n_b = len(base.encode(text).ids)
        n_e = len(ext.encode(text).ids)
        delta = (n_e - n_b) / max(n_b, 1)
        print(f"{name:<22}{mb:>8.2f}{n_b:>13,}{n_e:>13,}{delta:>+8.2%}")
        tot_b += n_b
        tot_e += n_e
        if delta < worst_delta:
            worst_delta, worst_name = delta, name

    tot_delta = (tot_e - tot_b) / max(tot_b, 1)
    print(f"\nTOTAL  base={tot_b:,}  ext={tot_e:,}  delta={tot_delta:+.2%}  "
          f"best genre: {worst_name} ({worst_delta:+.2%})")
    if tot_e > tot_b:
        print("REGRESSION: extended tokenizer produces MORE tokens than base!")
        sys.exit(1)
    print("OK: extended tokenizer is at worst as efficient as base.")
    sys.exit(0)


if __name__ == '__main__':
    main()
