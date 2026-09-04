"""Frontier 3 - sub-symbolic algorithmic resonators (hyperdimensional waves).

Entities and rules are continuous high-frequency waveforms. Deterministic
deduction happens through constructive interference (aligned phases reinforce
into a crisp signal); probabilistic reasoning happens through destructive
interference / phase shifts. Deduction becomes a physical resonance phenomenon
rather than an algorithmic step.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .hypervectors import bind, normalize, similarity, unbind


def interfere(waves: Sequence[np.ndarray], weights: Sequence[float] | None = None) -> np.ndarray:
    """Superpose waveforms *without* renormalising per element.

    The returned complex field carries amplitude information: where phases
    agree the amplitude survives (constructive); where they disagree it cancels
    (destructive).
    """
    mats = np.stack([np.asarray(w, dtype=np.complex128) for w in waves], axis=0)
    if weights is not None:
        wv = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
        return np.sum(wv * mats, axis=0)
    return np.sum(mats, axis=0)


def coherence(field: np.ndarray) -> float:
    """Mean surviving amplitude in [0, 1]: 1.0 = fully constructive (crisp),
    0.0 = fully destructive (maximal uncertainty)."""
    field = np.asarray(field, dtype=np.complex128)
    return float(np.mean(np.abs(field)))


def resonate(query: np.ndarray, memory: np.ndarray, role: np.ndarray) -> tuple[np.ndarray, float]:
    """Probe a bound memory with a role and report resonance strength.

    Returns ``(recovered_filler, coherence)``. High coherence -> the deduction
    'rings true' (a determinate answer); low coherence -> the answer is
    uncertain and should stay probabilistic.
    """
    recovered = unbind(memory, role)
    field = interfere([query, recovered])
    return normalize(recovered), coherence(field) / 2.0


def phase_lock(a: np.ndarray, b: np.ndarray) -> float:
    """Kuramoto-style order parameter for two waveforms (their alignment)."""
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    return float(np.abs(np.mean(a * np.conj(b))))
