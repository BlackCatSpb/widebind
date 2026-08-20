"""Пересборка жанровых бинарников с EOS-границами предложений.

Восстановленный pipeline (из FCP/EVA-Ai):
  main-russian/{GENRE}/*.txt  →  token_stream_{GENRE}_eos.bin (uint16)

Отличия от prepare_corpus.py:
  - разбивка текста на предложения, <|eos|> (id=2) вставляется после каждого;
  - чистка OCR-шапок («пер.», «OCR», «копирайт») — короткие строки в начале;
  - без чередующихся PAD (uint16, чисто), без SEQ_LEN-чанков;
  - предложения короче 3 слов не сохраняются (мусор).

Порядок: id = [eos] + предложение + [eos] ...
"""

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

EOS = 2
MIN_SENT_TOKENS = 3
_HEADER_LINES = 3

_SENT_SPLIT = re.compile(r'(?<=[.!?…])\s+')
_DIRTY = re.compile(
    r'^\s*(перев|пер\.|пер |ocr|скан|копирайт|©|подготовк|оформлен|'
    r'в книге|издание|серия|сайт|lib\.ru|либрусек|флибуста|spellcheck|'
    r'translated|translation|digitized|scanned|proofread|HarryFan|'
    r'ISBN|переводчик|вычитка)', re.IGNORECASE)
_TITLE_FRAME = re.compile(r'^[-=—_*~+]{4,}$')
_DASH_PAIR = re.compile(r'^[A-ZА-ЯЁ].*\.\s*[-–—]\s*[A-ZА-ЯЁ]')


def clean_header(text: str) -> str:
    """Удаляет шапки (OCR/пер./титульные строки) в начале файла."""
    lines = text.split('\n')
    cut = 0
    for i in range(min(_HEADER_LINES, len(lines))):
        s = lines[i].strip()
        if _TITLE_FRAME.match(s) or len(s) < 60 and (_DIRTY.search(s) or len(s) < 15):
            cut = i + 1
        else:
            break
    return '\n'.join(lines[cut:])


def split_sentences(text: str):
    """Предложения по .!?… (не режет по аббревиатурам), отдаёт список."""
    out = []
    for part in re.split(r'(?<=[.!?…])\s+', text):
        part = part.strip()
        if len(part) < 3:
            continue
        out.append(part)
    return out


def process_genre(genre, tokenizer, src, out_dir, encoding='cp1251'):
    out = os.path.join(out_dir, f'token_stream_{genre}_eos.bin')
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f'  {genre}: пропуск (уже собран: {out})')
        return 0, 0
    files = sorted(glob.glob(os.path.join(src, genre, '*.txt')))
    if not files:
        print(f'  [WARN] {genre}: no files')
        return 0, 0
    CHUNK = 100_000_000
    buf = np.empty(CHUNK, dtype=np.uint16)
    pos = 0
    n_total = 0
    n_sents = 0
    t0 = time.time()

    def flush():
        nonlocal pos, n_total
        if pos:
            n_total += pos
            buf[:pos].tofile(fh)
            pos = 0

    fh = open(out, 'wb')
    buf[pos] = EOS
    pos += 1
    pbar = tqdm(files, desc=genre, unit='файл', ncols=110,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                           '{rate_fmt} {postfix}')
    for idx, path in enumerate(pbar):
        try:
            with open(path, encoding=encoding, errors='replace') as f:
                raw = f.read()
        except Exception as e:
            print(f'  [WARN] {os.path.basename(path)}: {e}')
            continue
        text = clean_header(raw)
        for sent in split_sentences(text):
            ids = tokenizer.encode(sent).ids
            if len(ids) < MIN_SENT_TOKENS:
                continue
            if pos + len(ids) + 1 > CHUNK:
                flush()
            ids_arr = np.asarray(ids, dtype=np.uint16)
            buf[pos:pos + len(ids_arr)] = ids_arr
            pos += len(ids_arr)
            buf[pos] = EOS
            pos += 1
            n_sents += 1
        pbar.set_postfix(tok=f'{n_sents * 12 // 1e6:.0f}M', sents=f'{n_sents // 1000}K')
    flush()
    fh.close()
    pbar.close()
    print(f'  {genre}: {len(files)} файлов, {n_total // 1e6:.0f}M токенов, '
          f'{n_sents // 1000}K предложений, {time.time() - t0:.0f}s -> {out}', flush=True)
    return len(files), n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=r'C:\Users\black\OneDrive\Desktop\EVA-Ai\libinpoc_txt\main-russian')
    ap.add_argument('--out-dir', default='wb')
    ap.add_argument('--tokenizer', default='wb/russian_tokenizer/tokenizer.json')
    ap.add_argument('--encoding', default='cp1251')
    ap.add_argument('--genre', default='all')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.src):
        print(f'[ERROR] src not found: {args.src}')
        sys.exit(1)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f'Tokenizer: vocab={tokenizer.get_vocab_size()}')

    if args.genre == 'all':
        genres = sorted(d for d in os.listdir(args.src)
                        if os.path.isdir(os.path.join(args.src, d)))
    else:
        genres = [args.genre]

    if args.dry_run:
        for g in genres:
            n = len(glob.glob(os.path.join(args.src, g, '*.txt')))
            print(f'  {g}: {n} файлов')
        return

    os.makedirs(args.out_dir, exist_ok=True)
    total_f, total_t = 0, 0
    for g in genres:
        nf, nt = process_genre(g, tokenizer, args.src, args.out_dir, args.encoding)
        total_f += nf
        total_t += nt
    print(f'Итого: {total_f} файлов, {total_t // 1e6:.0f}M токенов')


if __name__ == '__main__':
    main()