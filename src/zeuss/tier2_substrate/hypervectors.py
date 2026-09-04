"""Continuous-phase hypervectors (Fourier Holographic Reduced Representations).

Each hypervector is a complex vector of (near) unit-modulus phasors,
``z_j = exp(i * theta_j)``. This VSA (Vector Symbolic Architecture) gives us a
continuous algebra with clean, invertible operations:

  bind(a, b)     = a * b            (elementwise complex mult = phase addition)
  unbind(a, b)   = a * conj(b)      (inverse of bind)
  bundle([...])  = normalise(sum)   (superposition / "OR"-like blend)
  permute(a, k)  = roll(a, k)       (protect order; e.g. sequence position)
  similarity(a,b)= Re<a, conj(b)>/D (mean cos of phase difference, in [-1, 1])

These are the building blocks every higher tier composes. Deterministic
structure and probabilistic blur are the *same* representation seen at
different phase-coherence levels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..backend import CDTYPE, RDTYPE, xp

DEFAULT_DIM = 10000


def _as_complex(a) -> "xp.ndarray":
    return xp.asarray(a, dtype=CDTYPE)


def random_hypervector(dim: int = DEFAULT_DIM, rng: np.random.Generator | None = None) -> "xp.ndarray":
    """A random unit-modulus hypervector with phases uniform on (-pi, pi].

    Randomness is drawn from an explicit NumPy ``Generator`` (the determinism
    convention) and lifted onto the active backend so the same seed gives the
    same hypervector on NumPy or JAX.
    """
    rng = np.random.default_rng() if rng is None else rng
    theta = rng.uniform(-np.pi, np.pi, size=dim)
    return xp.asarray(np.exp(1j * theta), dtype=CDTYPE)


def normalize(z) -> "xp.ndarray":
    """Project each element back onto the unit circle (phase-only)."""
    z = _as_complex(z)
    mag = xp.abs(z)
    mag = xp.where(mag == 0.0, 1.0, mag)
    return z / mag


def bind(a, b) -> "xp.ndarray":
    """Bind two hypervectors (role<->filler). Invertible via :func:`unbind`."""
    return normalize(_as_complex(a) * _as_complex(b))


def unbind(a, b) -> "xp.ndarray":
    """Recover the partner of ``b`` from a bound pair ``bind(a, b)``."""
    return normalize(_as_complex(a) * xp.conj(_as_complex(b)))


def bundle(vectors: Sequence, weights: Sequence[float] | None = None) -> "xp.ndarray":
    """Superpose several hypervectors into one (all remain partly recoverable)."""
    mats = xp.stack([_as_complex(v) for v in vectors], axis=0)
    if weights is not None:
        w = xp.asarray(weights, dtype=RDTYPE).reshape(-1, 1)
        acc = xp.sum(w * mats, axis=0)
    else:
        acc = xp.sum(mats, axis=0)
    return normalize(acc)


def permute(a, shift: int = 1) -> "xp.ndarray":
    """Cyclic permutation - a quasi-orthogonal, invertible relabelling."""
    return xp.roll(_as_complex(a), shift)


def similarity(a, b) -> float:
    """Cosine-like similarity in [-1, 1]; 1.0 iff phases are identical."""
    a = _as_complex(a)
    b = _as_complex(b)
    return float(xp.real(xp.vdot(b, a)) / a.shape[0])


@dataclass
class Codebook:
    """A named set of atomic hypervectors ('the alphabet of the substrate')."""

    dim: int = DEFAULT_DIM
    seed: int | None = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._items: dict[str, "xp.ndarray"] = {}

    def symbol(self, name: str) -> "xp.ndarray":
        """Get (or lazily mint) the hypervector for ``name``."""
        if name not in self._items:
            self._items[name] = random_hypervector(self.dim, self._rng)
        return self._items[name]

    def add(self, name: str, vector) -> None:
        self._items[name] = normalize(vector)

    def names(self) -> list[str]:
        return list(self._items)

    def matrix(self) -> "xp.ndarray":
        return xp.stack([self._items[n] for n in self._items], axis=0)

    def cleanup(self, z) -> tuple[str, float]:
        """Nearest stored symbol to ``z`` (associative-memory read-out)."""
        best_name, best_sim = None, float("-inf")
        for name, vec in self._items.items():
            s = similarity(z, vec)
            if s > best_sim:
                best_name, best_sim = name, s
        return best_name, float(best_sim)

    def similarities(self, z) -> dict[str, float]:
        return {name: similarity(z, vec) for name, vec in self._items.items()}


def encode_record(codebook: Codebook, pairs: Iterable[tuple[str, str]]) -> "xp.ndarray":
    """Encode a set of (role, filler) pairs as one bound-and-bundled record.

    Example: ``[("colour", "red"), ("shape", "square")]`` becomes a single
    hypervector from which any filler is recoverable given its role.
    """
    bound = [bind(codebook.symbol(role), codebook.symbol(filler)) for role, filler in pairs]
    return bundle(bound)
