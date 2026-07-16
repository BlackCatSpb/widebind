"""Temporal Zeckendorf hierarchy for VSA memory decay.

Port from FCF/eva/symbolic/fibonacci_utils.py.

Replaces exp(-distance/tau) with Fibonacci-based multi-scale decay.
Zeckendorf digit count grows as O(log_phi(distance)), giving
automatic multi-scale temporal hierarchy with no free tau parameter.
"""

import math


_FIB_CACHE: dict[int, int] = {0: 0, 1: 1, 2: 1}


def _fib(n: int) -> int:
    """Fibonacci number F_n (cached)."""
    if n not in _FIB_CACHE:
        _FIB_CACHE[n] = _fib(n - 1) + _fib(n - 2)
    return _FIB_CACHE[n]


_FIB_SEQ_CACHE: list[int] = []


def _fib_seq_up_to(limit: int) -> list[int]:
    """Fibonacci numbers 1, 2, 3, 5, 8, ... up to limit."""
    if not _FIB_SEQ_CACHE or _FIB_SEQ_CACHE[-1] < limit:
        seq = [1, 2]
        while seq[-1] < limit:
            seq.append(seq[-1] + seq[-2])
        _FIB_SEQ_CACHE[:] = seq
    return _FIB_SEQ_CACHE


def _zeckendorf(n: int) -> list[int]:
    """Zeckendorf decomposition: non-consecutive Fib sum, MSB first."""
    if n <= 0:
        return [0]
    fibs = _fib_seq_up_to(n)
    bits = []
    remaining = n
    prev = False
    for f in reversed(fibs):
        if remaining >= f and not prev:
            bits.append(f)
            remaining -= f
            prev = True
        else:
            prev = False
    return bits


class TemporalZeckendorf:
    """Zeckendorf-based temporal encoding for VSA memory decay.

    Each distance d is encoded by its Zeckendorf digit count.
    theta(d) = 1 / (1 + len(zeckendorf(d))) — natural multi-scale decay.

    Properties:
    - No free tau parameter
    - O(log_phi(d)) digit count → automatic log-scale decay
    - Discrete levels → natural temporal clustering
    """

    def __init__(self, max_distance: int = 1000000):
        self._max_depth = len(_zeckendorf(max_distance)) + 1

    def _largest_fib_idx(self, t: int) -> int:
        """Index of largest Fibonacci number <= t."""
        i = 2
        while _fib(i) <= t:
            i += 1
        return i - 1

    def trace(self, t: int) -> float:
        """Monotonic trace: fib_index / max_depth."""
        if t <= 0:
            return 0.0
        idx = self._largest_fib_idx(t)
        return idx / max(self._max_depth, 1)

    def theta(self, distance: int, fast_window: int = 5,
              slow_window: int = 10) -> tuple[float, float]:
        """Zeckendorf-based temporal decay for VSA memory.

        theta_base = 1 / (1 + len(zeckendorf(distance)))
        Short distances -> few digits -> high theta.
        Long distances -> many digits -> low theta.

        Returns (fast_theta, slow_theta), both in (0, 1].
        """
        if distance <= 0:
            return (1.0, 1.0)
        zlen = len(_zeckendorf(distance))
        theta_base = 1.0 / (1.0 + zlen)
        if distance <= fast_window:
            fast = theta_base
        else:
            fast = theta_base * max(0.0, 1.0 - (distance - fast_window) / max(fast_window, 1))
        if distance <= slow_window:
            slow = theta_base
        else:
            slow = theta_base * max(0.0, 1.0 - (distance - slow_window) / max(slow_window, 1))
        return (max(fast, 0.0), max(slow, 0.0))
