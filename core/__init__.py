"""Core model package — re-exports everything for backward-compatible imports."""

from .config import WideBindConfig
from .model import (
    WideBindStack, WideBindBlock, GroupedCognitiveMirror, GroupedMLP,
    ZeckendorfEmbedding, PartitionedEmbedding, LmHead, PartitionedHead,
    AdaptiveController, MirrorLRScheduler,
    dct_basis, zeckendorf_codes, sparse_block_codes,
    vsa_prefix_scan,
)
from .zeckendorf_readout import ZeckendorfReadout, fibonacci_bases, zeckendorf_code

from .live_inference import LiveInference, MirrorMonitor
from .curriculum import CurriculumTracker

# Backward compat
CognitiveMirror = GroupedCognitiveMirror
