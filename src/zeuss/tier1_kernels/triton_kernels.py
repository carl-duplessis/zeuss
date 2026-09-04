"""Optional Triton/CUDA kernels (GPU compute layer).

This module is a guarded placeholder. On a machine with a CUDA GPU and Triton
installed, implement the same semantics as ``reference.py`` here and set
``AVAILABLE = True``. Everything else in Zeuss will transparently pick these up.

Why Triton: write GPU kernels in a Python-like syntax that compiles to PTX,
avoiding framework overhead for custom, non-transformer ops like complex
hypervector phase alignment and the topological-collapse reduction.
"""
from __future__ import annotations

AVAILABLE = False

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    # TODO(zeuss): port phase_interference and topological_collapse_step to
    # Triton kernels here, then flip AVAILABLE to True after validating them
    # against tier1_kernels.reference within tests/test_kernels_parity.py.
    AVAILABLE = False
except Exception:
    AVAILABLE = False
