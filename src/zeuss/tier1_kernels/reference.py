"""Portable NumPy reference kernels (correctness ground truth).

These are intentionally simple and vectorised. They define the *semantics* that
the optimised Triton/CUDA kernels must match bit-for-bit (within tolerance).
"""
from __future__ import annotations

import numpy as np

try:  # optional acceleration
    from . import triton_kernels as _tk  # noqa: F401

    HAS_TRITON = getattr(_tk, "AVAILABLE", False)
except Exception:
    HAS_TRITON = False


def phase_interference(fields: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted complex superposition of a stack of fields.

    ``fields`` has shape (N, D). Returns the (D,) interference field. This is
    the kernel a GPU would parallelise across the D dimension.
    """
    fields = np.asarray(fields, dtype=np.complex128)
    if weights is None:
        return fields.sum(axis=0)
    w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
    return (w * fields).sum(axis=0)


def topological_collapse_step(z: np.ndarray, basis: np.ndarray, beta: float) -> np.ndarray:
    """One collapse step: pull ``z`` toward the ``basis`` rows by resonance.

    ``basis`` has shape (K, D). Higher ``beta`` (lower temperature) collapses
    harder toward the single best-matching basis vector - the space losing
    dimensionality as entropy drops.
    """
    z = np.asarray(z, dtype=np.complex128)
    basis = np.asarray(basis, dtype=np.complex128)
    sims = (basis @ np.conj(z)).real / z.shape[0]
    sims = sims - sims.max()
    p = np.exp(beta * sims)
    p = p / p.sum()
    acc = (p.reshape(-1, 1) * basis).sum(axis=0)
    mag = np.abs(acc)
    mag = np.where(mag == 0.0, 1.0, mag)
    return acc / mag
