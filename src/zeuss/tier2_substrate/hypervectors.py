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

DEFAULT_DIM = 10000


def _as_complex(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.complex128)


def random_hypervector(dim: int = DEFAULT_DIM, rng: np.random.Generator | None = None) -> np.ndarray:
    """A random unit-modulus hypervector with phases uniform on (-pi, pi]."""
    rng = np.random.default_rng() if rng is None else rng
    theta = rng.uniform(-np.pi, np.pi, size=dim)
    return np.exp(1j * theta).astype(np.complex128)


def normalize(z: np.ndarray) -> np.ndarray:
    """Project each element back onto the unit circle (phase-only)."""
    z = _as_complex(z)
    mag = np.abs(z)
    mag = np.where(mag == 0.0, 1.0, mag)
    return z / mag


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bind two hypervectors (role<->filler). Invertible via :func:`unbind`."""
    return normalize(_as_complex(a) * _as_complex(b))


def unbind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Recover the partner of ``b`` from a bound pair ``bind(a, b)``."""
    return normalize(_as_complex(a) * np.conj(_as_complex(b)))


def bundle(vectors: Sequence[np.ndarray], weights: Sequence[float] | None = None) -> np.ndarray:
    """Superpose several hypervectors into one (all remain partly recoverable)."""
    mats = np.stack([_as_complex(v) for v in vectors], axis=0)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
        acc = np.sum(w * mats, axis=0)
    else:
        acc = np.sum(mats, axis=0)
    return normalize(acc)


def permute(a: np.ndarray, shift: int = 1) -> np.ndarray:
    """Cyclic permutation - a quasi-orthogonal, invertible relabelling."""
    return np.roll(_as_complex(a), shift)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine-like similarity in [-1, 1]; 1.0 iff phases are identical."""
    a = _as_complex(a)
    b = _as_complex(b)
    return float(np.real(np.vdot(b, a)) / a.shape[0])


@dataclass
class Codebook:
    """A named set of atomic hypervectors ('the alphabet of the substrate')."""

    dim: int = DEFAULT_DIM
    seed: int | None = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._items: dict[str, np.ndarray] = {}

    def symbol(self, name: str) -> np.ndarray:
        """Get (or lazily mint) the hypervector for ``name``."""
        if name not in self._items:
            self._items[name] = random_hypervector(self.dim, self._rng)
        return self._items[name]

    def add(self, name: str, vector: np.ndarray) -> None:
        self._items[name] = normalize(vector)

    def names(self) -> list[str]:
        return list(self._items)

    def matrix(self) -> np.ndarray:
        return np.stack([self._items[n] for n in self._items], axis=0)

    def cleanup(self, z: np.ndarray) -> tuple[str, float]:
        """Nearest stored symbol to ``z`` (associative-memory read-out)."""
        best_name, best_sim = None, -np.inf
        for name, vec in self._items.items():
            s = similarity(z, vec)
            if s > best_sim:
                best_name, best_sim = name, s
        return best_name, float(best_sim)

    def similarities(self, z: np.ndarray) -> dict[str, float]:
        return {name: similarity(z, vec) for name, vec in self._items.items()}


def encode_record(codebook: Codebook, pairs: Iterable[tuple[str, str]]) -> np.ndarray:
    """Encode a set of (role, filler) pairs as one bound-and-bundled record.

    Example: ``[("colour", "red"), ("shape", "square")]`` becomes a single
    hypervector from which any filler is recoverable given its role.
    """
    bound = [bind(codebook.symbol(role), codebook.symbol(filler)) for role, filler in pairs]
    return bundle(bound)
