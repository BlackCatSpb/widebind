"""Stencil — трафарет русского порядка слов, снятый с исходного корпуса.

Трафарет — это статистика, извлечённая из исходного текста (с пунктуацией
и явным порядком слов в пределах предложения). Он служит обучающим сигналом
для "прожектора" — механизма сборки предложений из набора слов.

Извлекаемые данные:
  - частеречные цепочки (последовательности POS внутри предложения)
  - словесные n-граммы (леммы) 2..4 с PMI по соседним парам
  - позиционные веса POS (тема → рема: первые и последние позиции)
  - падежные рамки топ-глаголов (какие предлоги/падежи следуют за глаголом)
  - частоты слов (Zipf) → λ_d-глубина семантического поля
  - распределение длины предложений
"""

from __future__ import annotations

import io
import json
import math
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import pymorphy3

LAMBDA = 1.46557

_WORD_RE = re.compile(r"[а-яёa-z]+(?:-[а-яёa-z]+)*", re.IGNORECASE)
_SENT_END = {'.', '!', '?', '…', '!?', '?!', '...'}

# POS, которые считаем «служебными» (соединители, периферия)
FUNCTION_POS = frozenset({'PREP', 'CONJ', 'PRCL', 'INTJ'})


class Stencil:
    """Накопительная статистика трафарета."""

    def __init__(self):
        self.word_freq: Counter = Counter()
        self.lemma_freq: Counter = Counter()
        self.pos_freq: Counter = Counter()
        self.pos_chains: Counter = Counter()
        self.pos_position: Dict[str, List[int]] = defaultdict(list)
        self.word_ngrams: Counter = Counter()          # n=2..4 по леммам
        self.adjacency: Dict[str, Counter] = defaultdict(Counter)
        self.verb_frames: Dict[str, Counter] = defaultdict(Counter)
        self.sentence_len: Counter = Counter()
        self.total_sentences: int = 0
        self.total_words: int = 0
        self.ngram_order: int = 4

    # ── построение ────────────────────────────────────────────────

    def build(self, corpus_path: str, max_lines: Optional[int] = None,
              morph: Optional[pymorphy3.MorphAnalyzer] = None) -> 'Stencil':
        if morph is None:
            morph = pymorphy3.MorphAnalyzer()
        n_lines = 0
        seen = set()
        with io.open(corpus_path, 'r', encoding='utf-8', errors='replace') as f:
            for raw in f:
                if max_lines is not None and n_lines >= max_lines:
                    break
                n_lines += 1
                sent = self._clean_sentence(raw)
                if not sent or len(sent) < 8:
                    continue
                if sent in seen:
                    continue
                seen.add(sent)
                words = self._tokenize(sent)
                if len(words) < 2:
                    continue
                self._ingest(words, morph)
        return self

    def _clean_sentence(self, raw: str) -> str:
        s = raw.strip().lower()
        if not s:
            return ''
        if not s[-1].isalnum() and s[-1] in _SENT_END:
            s = s[:-1]
        return s

    def _tokenize(self, sent: str) -> List[str]:
        return _WORD_RE.findall(sent)

    def _ingest(self, words: List[str], morph: pymorphy3.MorphAnalyzer):
        n = len(words)
        self.total_sentences += 1
        self.total_words += n
        self.sentence_len[n] += 1

        parsed = []
        for i, w in enumerate(words):
            p = morph.parse(w)[0]
            tag = p.tag
            pos = tag.POS or 'UNKN'
            lemma = p.normal_form
            case = getattr(tag, 'case', None)
            self.word_freq[w] += 1
            self.lemma_freq[lemma] += 1
            self.pos_freq[pos] += 1
            parsed.append((w, lemma, pos, case))
            pos_idx = 'first' if i == 0 else ('second' if i == 1
                       else ('last' if i == n - 1 else ('prelast' if i == n - 2 else 'mid')))
            self.pos_position[pos].append(1 if pos_idx == 'first' else
                                         2 if pos_idx == 'second' else
                                         3 if pos_idx == 'last' else
                                         4 if pos_idx == 'prelast' else 5)

        self.pos_chains[tuple(p[2] for p in parsed)] += 1

        for order in (2, 3, 4):
            for i in range(n - order + 1):
                key = tuple(parsed[j][1] for j in range(i, i + order))
                self.word_ngrams[(order, key)] += 1

        for i in range(n - 1):
            self.adjacency[parsed[i][1]][parsed[i + 1][1]] += 1

        for i, (w, lemma, pos, case) in enumerate(parsed):
            if pos not in ('VERB', 'INFN') or i == n - 1:
                continue
            nxt = parsed[i + 1]
            npos, ncase = nxt[2], nxt[3]
            if npos in ('NOUN', 'NPRO') and ncase is not None:
                self.verb_frames[lemma][ncase] += 1
            elif npos == 'PREP':
                after = parsed[i + 2] if i + 2 < n else None
                if after is not None and after[2] in ('NOUN', 'NPRO') and after[3] is not None:
                    self.verb_frames[lemma][(nxt[0], after[3])] += 1
                else:
                    self.verb_frames[lemma][nxt[0]] += 1

    # ── производные метрики ───────────────────────────────────────

    def zipf_rank(self, lemma: str) -> int:
        ordered = [l for l, _ in self.lemma_freq.most_common()]
        try:
            return ordered.index(lemma) + 1
        except ValueError:
            return len(ordered) + 1

    def lambda_depth(self, lemma: str) -> float:
        """λ_d-глубина в семантическом поле: частота ядра λ⁻ᵏ от максимума."""
        top = self.lemma_freq.most_common(1)
        if not top:
            return 3.0
        top_f = top[0][1]
        f = self.lemma_freq.get(lemma, 0)
        if f <= 0 or top_f <= 0:
            return 3.0
        k = max(0.0, -math_log(f / top_f) / math_log(LAMBDA))
        return min(k, 6.0)

    def pmi(self, a: str, b: str) -> float:
        """Pointwise mutual information для пары лемм (лог-биты)."""
        pa = self.lemma_freq.get(a, 0) / max(self.total_words, 1)
        pb = self.lemma_freq.get(b, 0) / max(self.total_words, 1)
        pab = self.adjacency.get(a, {}).get(b, 0) / max(self.total_words, 1)
        if pa <= 0 or pb <= 0 or pab <= 0:
            return 0.0
        return max(0.0, math_log(pab / (pa * pb)) / math_log(2))

    def top_pos_chains(self, k: int = 20) -> List[Tuple[tuple, int]]:
        return self.pos_chains.most_common(k)

    def top_ngrams(self, k: int = 20) -> List[Tuple[tuple, int]]:
        return self.word_ngrams.most_common(k)

    # ── сохранение / загрузка ─────────────────────────────────────

    def save(self, path: str, max_ngrams: int = 200000,
             max_adj: int = 50, max_frames: int = 60):
        data = {
            'total_sentences': self.total_sentences,
            'total_words': self.total_words,
            'sentence_len': dict(self.sentence_len),
            'pos_freq': dict(self.pos_freq),
            'pos_chains': {' '.join(k): v for k, v in self.pos_chains.most_common(20000)},
            'pos_position': {k: v for k, v in self.pos_position.items()},
            'lemma_freq': dict(self.lemma_freq),
            'word_ngrams': {f'{o} {" ".join(k)}': v for (o, k), v in self.word_ngrams.most_common(max_ngrams)},
            'adjacency': {k: dict(v.most_common(max_adj)) for k, v in self.adjacency.items()},
            'verb_frames': {k: {str(c): v for c, v in v.most_common(max_frames)}
                            for k, v in self.verb_frames.items()},
            'lambda': LAMBDA,
        }
        payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        if path.endswith('.gz'):
            import gzip
            with gzip.open(path, 'wb') as f:
                f.write(payload)
        else:
            with io.open(path, 'wb') as f:
                f.write(payload)

    @classmethod
    def load(cls, path: str) -> 'Stencil':
        s = cls()
        if path.endswith('.gz'):
            import gzip
            with gzip.open(path, 'rb') as f:
                data = json.loads(f.read().decode('utf-8'))
        else:
            with io.open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        s.total_sentences = data['total_sentences']
        s.total_words = data['total_words']
        s.sentence_len = Counter(data['sentence_len'])
        s.pos_freq = Counter(data['pos_freq'])
        s.pos_chains = Counter({tuple(k.split(' ')): v for k, v in data['pos_chains'].items()})
        s.pos_position = defaultdict(list, data['pos_position'])
        s.lemma_freq = Counter(data['lemma_freq'])
        s.word_ngrams = Counter(
            {(int(k.split(' ')[0]), tuple(k.split(' ')[1:])): v for k, v in data['word_ngrams'].items()})
        s.adjacency = defaultdict(Counter, {k: Counter(v) for k, v in data['adjacency'].items()})
        s.verb_frames = defaultdict(Counter, {k: Counter(v) for k, v in data['verb_frames'].items()})
        return s

    def summary(self) -> str:
        lens = self.sentence_len.most_common(8)
        return (
            f'Stencil: {self.total_sentences:,} предложений, {self.total_words:,} слов\n'
            f'  длина предложений: {lens}\n'
            f'  топ-POS: {self.pos_freq.most_common(6)}\n'
            f'  топ-POS-цепочки: {self.top_pos_chains(5)}\n'
            f'  топ-n-граммы: {self.top_ngrams(5)}\n'
        )


def math_log(x: float) -> float:
    return math.log(x)


if __name__ == '__main__':
    import time
    path = sys.argv[1] if len(sys.argv) > 1 else (
        r'C:\Users\black\OneDrive\Desktop\FCF\real_data\corpus_1m.txt')
    out = sys.argv[2] if len(sys.argv) > 2 else 'data/stencil.json.gz'
    max_lines = int(sys.argv[3]) if len(sys.argv) > 3 else None
    print(f'Building stencil from {path} (max_lines={max_lines})...')
    t0 = time.perf_counter()
    st = Stencil().build(path, max_lines=max_lines)
    print(f'Built in {time.perf_counter() - t0:.1f}s')
    print(st.summary())
    st.save(out)
    print(f'Saved to {out}')
    print('Done.')