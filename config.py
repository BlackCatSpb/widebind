"""Backward-compatibility shim: config → core.config."""
import warnings
warnings.warn("config is deprecated, use core instead", DeprecationWarning, stacklevel=2)
from core.config import WideBindConfig
