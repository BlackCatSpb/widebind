"""Diagnostic tests for the v65536 whole-word tokenizer extension.

Cover:
  1. Prefix-stable ids: common words keep identical ids between base (50000)
     and extended (65536) tokenizer — weights learned on old ids stay valid.
  2. Efficiency: extended tokenizer must not produce MORE tokens than base on
     Russian prose (goal is strictly fewer / equal).
  3. Sparse-block codes prefix-stability at the real 50000 -> 65536 boundary.
  4. The extended tokenizer's own vocab size is exactly 65536.

These are read-only checks over wb/russian_tokenizer/*.json — they need the
tokenizer files present (they are committed). Skip gracefully if missing.
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from tokenizers import Tokenizer

from core.model import sparse_block_codes

TOK_DIR = os.path.join(os.path.dirname(__file__), '..', 'wb', 'russian_tokenizer')
BASE = os.path.join(TOK_DIR, 'tokenizer.json')
EXT = os.path.join(TOK_DIR, 'tokenizer_v65536.json')

# Частые русские слова/фразы: у целых слов в расширенном токенизаторе есть
# отдельные id >= 50000, но id базовых подстрок обязаны остаться прежними.
SAMPLE = ('Привет, как дела? Москва — столица России. В начале было Слово, '
          'и Слово было у Бога. Искусственный интеллект изменит мир. '
          'Кот спал на подоконнике, а собака смотрела в окно. '
          'Николай пришёл домой поздно вечером и сразу лёг спать.')


def _toks(path):
    return Tokenizer.from_file(path)


def test_tokenizers_exist():
    assert os.path.exists(BASE), f'base tokenizer missing: {BASE}'
    assert os.path.exists(EXT), f'extended tokenizer missing: {EXT}'


def test_extended_vocab_size():
    t = _toks(EXT)
    assert t.get_vocab_size() == 65536, f'vocab={t.get_vocab_size()} != 65536'


def test_base_vocab_size():
    t = _toks(BASE)
    assert t.get_vocab_size() == 50000, f'vocab={t.get_vocab_size()} != 50000'


def test_extended_keeps_base_ids_prefix_stable():
    """Расширение добавляет токены, НЕ меняя id существующих (префикс-стабильность словаря)."""
    b, e = _toks(BASE), _toks(EXT)
    b_vocab = b.get_vocab()   # token->id для базового
    e_vocab = e.get_vocab()   # token->id для расширенного
    assert len(b_vocab) == 50000
    changed = []
    for tok, tid in b_vocab.items():
        if e_vocab.get(tok, -1) != tid:
            changed.append((tok, tid, e_vocab.get(tok, -1)))
            if len(changed) > 10:
                break
    assert not changed, (
        f'base tokens changed ids after extension (first {len(changed)}): {changed}')
    # Убедиться, что целые слова реально добавлены в хвост (не сдвинули базу)
    new_ids = sorted(i for i in e_vocab.values() if i >= 50000)
    assert len(new_ids) == 65536 - 50000
    assert new_ids[0] == 50000 and new_ids[-1] == 65535, 'new ids not contiguous at tail'


def test_extended_no_more_tokens_than_base():
    """v65536 должен давать <= токенов, чем базовый (цель: меньше)."""
    b, e = _toks(BASE), _toks(EXT)
    n_b = len(b.encode(SAMPLE).ids)
    n_e = len(e.encode(SAMPLE).ids)
    assert n_e <= n_b, f'extended produced MORE tokens ({n_e}) than base ({n_b})'


def test_extended_uses_whole_words():
    """В русской прозе должны встречаться токены из расширенного диапазона."""
    e = _toks(EXT)
    for word in ['посмотрел', 'Вечером', 'подоконнике', 'Искусственный',
                 'интеллект', 'Николай', 'России']:
        ids = e.encode(word).ids
        if len(ids) == 1 and ids[0] >= 50000:
            return  # хотя бы одно целое слово — отдельный новый токен
    ids = e.encode(SAMPLE).ids
    assert any(i >= 50000 for i in ids), f'no extended tokens used for Russian prose: {ids}'


def test_sparse_codes_prefix_stable_50000_65536():
    """Реальная граница расширения: первые 50000 кодов идентичны."""
    small = sparse_block_codes(vocab=50000, K=32, S=6)
    big = sparse_block_codes(vocab=65536, K=32, S=6)
    assert big.shape == (65536, 32)
    assert torch.equal(big[:50000], small), 'sparse codes prefix changed at 50000->65536'


if __name__ == '__main__':
    tests = [fn for fn in dir() if fn.startswith('test_')]
    passed = failed = 0
    for name in tests:
        try:
            globals()[name]()
            print(f'  PASS  {name}')
            passed += 1
        except Exception as e:
            print(f'  FAIL  {name}: {e}')
            failed += 1
    print(f'\n{passed}/{passed + failed} passed')
    sys.exit(0 if failed == 0 else 1)
