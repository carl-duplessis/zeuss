"""Frontier 3 - sub-symbolic algorithmic resonators (hyperdimensional waves).

Entities and rules are continuous high-frequency waveforms. Deterministic
deduction happens through constructive interference (aligned phases reinforce
into a crisp signal); probabilistic reasoning happens through destructive
interference / phase shifts. Deduction becomes a physical resonance phenomenon
rather than an algorithmic step.
"""
from __future__ import annotations

from typing import Sequence

from ..backend import CDTYPE, RDTYPE, xp
from .hypervectors import normalize, unbind


def interfere(waves: Sequence, weights: Sequence[float] | None = None) -> "xp.ndarray":
    """Superpose waveforms *without* renormalising per element.

    The returned complex field carries amplitude information: where phases
    agree the amplitude survives (constructive); where they disagree it cancels
    (destructive).
    """
    mats = xp.stack([xp.asarray(w, dtype=CDTYPE) for w in waves], axis=0)
    if weights is not None:
        wv = xp.asarray(weights, dtype=RDTYPE).reshape(-1, 1)
        return xp.sum(wv * mats, axis=0)
    return xp.sum(mats, axis=0)


def coherence(field) -> float:
    """Mean surviving amplitude in [0, 1]: 1.0 = fully constructive (crisp),
    0.0 = fully destructive (maximal uncertainty)."""
    field = xp.asarray(field, dtype=CDTYPE)
    return float(xp.mean(xp.abs(field)))


def resonate(query, memory, role) -> tuple["xp.ndarray", float]:
    """Probe a bound memory with a role and report resonance strength.

    Returns ``(recovered_filler, coherence)``. High coherence -> the deduction
    'rings true' (a determinate answer); low coherence -> the answer is
    uncertain and should stay probabilistic.
    """
    recovered = unbind(memory, role)
    field = interfere([query, recovered])
    return normalize(recovered), coherence(field) / 2.0


def phase_lock(a, b) -> float:
    """Kuramoto-style order parameter for two waveforms (their alignment)."""
    a = xp.asarray(a, dtype=CDTYPE)
    b = xp.asarray(b, dtype=CDTYPE)
    return float(xp.abs(xp.mean(a * xp.conj(b))))
