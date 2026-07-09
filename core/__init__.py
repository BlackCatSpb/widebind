"""Core model package — re-exports everything for backward-compatible imports."""

from .config import WideBindConfig
from .model import (
    WideBindStack, WideBindBlock, CognitiveMirror, GroupedMLP,
    ZeckendorfEmbedding, LmHead, AdaptiveController, MirrorLRScheduler,
    dct_basis, zeckendorf_codes, vsa_prefix_scan, compute_timescales,
    compute_spectrum,
)
