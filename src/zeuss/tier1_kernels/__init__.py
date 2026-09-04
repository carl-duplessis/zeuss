"""Tier 1 - compute kernels (hardware / compute layer).

The hot operations of the substrate - complex phase interference and the
topological collapse step - are expressed here behind a stable interface. The
portable NumPy reference in ``reference.py`` always works and is the ground
truth for tests. GPU implementations (Triton / CUDA) drop in behind the same
functions when available; see ``triton_kernels.py``.
"""
from .reference import phase_interference, topological_collapse_step, HAS_TRITON

__all__ = ["phase_interference", "topological_collapse_step", "HAS_TRITON"]
