"""Backward-compatibility shim: wbconfig → core.config."""
import warnings
warnings.warn("wbconfig is deprecated, use core instead", DeprecationWarning, stacklevel=2)
from core.config import WideBindConfig
