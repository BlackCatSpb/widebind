"""Project — «прожектор»: сборка предложений из слов бинарника по трафарету.

Прожектор берёт фрагмент бинарника (token_stream_*.bin), декодирует его в
слова через BPE-токенизатор, отбирает уникальные словоформы и собирает из
них грамматически корректное русское предложение по трафарету порядка слов
(data/stencil.json.gz), снятому с исходного корпуса.

Usage:
  py -3.12 scripts/project.py
  py -3.12 scripts/project.py --bin wb/token_stream_SFICTION_clean.bin --tokens 3000 --words 28
  py -3.12 scripts/project.py --sample 5
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.sentence_builder import SentenceBuilder
from core.stencil import Stencil

REPO = os.path.join(os.path.dirname(__file__), '..')
WORD_RE = re.compile(r"[а-яёa-z]+(?:-[а-яёa-z]+)*", re.IGNORECASE)


def load_tokenizer() -> Tokenizer:
    for cand in (os.path.join(REPO, 'wb', 'russian_tokenizer', 'tokenizer.json'),
                 os.path.join(REPO, 'wb', 'russian_tokenizer', 'tokenizer_v65536.json')):
        if os.path.exists(cand):
            return Tokenizer.from_file(cand)
    raise FileNotFoundError('russian_tokenizer/tokenizer.json not found')


def decode_words(tok: Tokenizer, ids, max_words: int) -> list:
    text = tok.decode(ids, skip_special_tokens=True)
    words = [w for w in WORD_RE.findall(text.lower()) if len(w) > 2]
    return list(dict.fromkeys(words))[:max_words]


def project(bin_path: str, tokens: int, words: int, stencil: Stencil,
            builder: SentenceBuilder, seed: int = 0, rng=None):
    stream = np.memmap(bin_path, dtype=np.uint16, mode='r')
    n = len(stream)
    if seed == 0:
        start = 0
    else:
        start = int(rng.integers(0, max(1, n - tokens)))
    ids = stream[start:start + tokens].tolist()
    uniq = decode_words(load_tokenizer(), ids, words)
    res = builder.build(uniq)
    return uniq, res


def main():
    ap = argparse.ArgumentParser(description='Прожектор: сборка предложений из бинарника')
    ap.add_argument('--bin', default=os.path.join(REPO, 'wb', 'token_stream_DETECT_clean.bin'))
    ap.add_argument('--tokens', type=int, default=3000)
    ap.add_argument('--words', type=int, default=28)
    ap.add_argument('--stencil', default=os.path.join(REPO, 'data', 'stencil.json.gz'))
    ap.add_argument('--sample', type=int, default=0,
                    help='сколько случайных фрагментов прогнать (0 = только первый)')
    args = ap.parse_args()

    st = Stencil.load(args.stencil)
    builder = SentenceBuilder(stencil=st)
    rng = np.random.default_rng(42)

    for i in range(max(1, args.sample)):
        seed = 0 if i == 0 and args.sample == 0 else i + 1
        uniq, res = project(args.bin, args.tokens, args.words, st, builder,
                            seed=seed, rng=rng)
        print(f'--- фрагмент {i + 1} ({os.path.basename(args.bin)}) ---')
        print('слова:', ', '.join(uniq[:20]))
        print('ПРЕДЛОЖЕНИЕ:', res.text)
        print('ЯДРО:', ', '.join(res.core))
        print('РАЗБОР:')
        for line in res.explanation[:14]:
            print('  ', line)
        print()


if __name__ == '__main__':
    main()