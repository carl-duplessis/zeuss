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


def unbind_raw(a, b) -> "xp.ndarray":
    """Like :func:`unbind`, but skips the final unit-modulus projection -
    see `docs/ROADMAP.md`'s "Phase 2 addendum" entry for the full
    derivation. That projection is a no-op when both operands are already
    unit-modulus (atomic hypervectors, or anything built purely from
    `bind`ing atomics), so this only differs from `unbind` when ``a`` is a
    *raw, un-normalised* bundle (e.g. `Ontology.ground_raw()`/
    `IncrementalMemory.raw`) - reading such a bundle out this way preserves
    the per-dimension interference amplitude `normalize()` would otherwise
    discard, which is what lets `dim` actually rescue single-bundle
    capacity instead of only tightening an already-capped estimate (see
    `Ontology.step_raw`)."""
    return _as_complex(a) * xp.conj(_as_complex(b))


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

    def has(self, name: str) -> bool:
        """Whether ``name`` has already been minted/set, without the
        lazy-mint side effect ``symbol()`` has - lets a caller offer a
        fallback for a name that was never explicitly written (see
        `Ontology.entity_refined`) instead of silently minting and
        returning an unrelated random vector under that key."""
        return name in self._items

    def add(self, name: str, vector) -> None:
        self._items[name] = normalize(vector)

    def items(self) -> dict[str, "xp.ndarray"]:
        """A copy of every minted ``{name: vector}`` pair - a read accessor
        for persistence (see :func:`.tier4_synthesis.grammar_bias.
        save_grammar_bias`): ``symbol()`` draws lazily from one evolving RNG
        stream, so which vector a name gets depends on *when* it was first
        requested, not just ``seed`` - reconstructing a fresh ``Codebook``
        with the same seed does not reproduce the same per-name vectors
        unless names are requested in the exact original order. Persisting
        the actual vectors (via this and :meth:`load`) is the only way to
        restore a codebook's meaning across a process restart."""
        return dict(self._items)

    def load(self, items: dict[str, "xp.ndarray"]) -> None:
        """Bulk-restore previously-minted ``{name: vector}`` pairs (see
        :meth:`items`) without touching ``_rng`` - names encountered *after*
        loading still mint fresh vectors normally, only names present in
        ``items`` are restored verbatim."""
        self._items.update(items)

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


_SEQUENCE_ANCHOR_NAME = "__POSITION_BASIS__"


def encode_sequence(codebook: Codebook, items: Sequence[str]) -> "xp.ndarray":
    """Encode an *ordered* sequence of named codebook items into one
    hypervector, combining ``bind`` and ``permute`` (a good-first-task from
    ``docs/ROADMAP.md``).

    Each position ``i`` gets its own role vector, ``permute(anchor, i)`` - a
    single fixed anchor hypervector cyclically shifted by ``i``, which is
    exactly what this module's own top-of-file summary says ``permute`` is
    for ("protect order; e.g. sequence position"). The item at that position
    is then ``bind``-bound to its positional role - the same role-filler
    binding :func:`encode_record` already uses, just with positions as roles
    instead of named fields - and every (role, item) pair is bundled into
    one vector. Different positions get quasi-orthogonal roles (``permute``
    is a quasi-orthogonal transform), so position, not just membership, is
    recoverable: encoding the same items in a different order produces a
    dissimilar vector, confirmed directly
    (``test_encode_sequence_is_order_sensitive``).

    Pair with :func:`decode_sequence` to read the sequence back out. Like
    any bundle, more items packed into one sequence means more cross-talk
    for the decoder to resolve against - the same bundling-interference
    tradeoff :func:`~zeuss.tier2_substrate.collapse.train_codebook`'s tests
    already demonstrate elsewhere in this codebase, not a new failure mode.
    """
    anchor = codebook.symbol(_SEQUENCE_ANCHOR_NAME)
    bound = [bind(permute(anchor, i), codebook.symbol(name)) for i, name in enumerate(items)]
    return bundle(bound)


def decode_sequence(codebook: Codebook, seq, length: int) -> list[tuple[str, float]]:
    """Decode an :func:`encode_sequence` hypervector back into its
    per-position items via cleanup-memory readout.

    For each position ``i``, unbinds that position's role
    (``permute(anchor, i)``) from ``seq`` and looks up the nearest stored
    symbol via :meth:`Codebook.cleanup` - the classic VSA "cleanup memory"
    step, resolving a noisy recovered vector back onto the nearest exact
    atomic symbol. Returns a list of ``(name, similarity)`` pairs, one per
    position, in order. Decoding is reliable at the dimensions this project
    otherwise uses (``dim=8192``: exact recovery up to at least 24 bundled
    items, checked directly), and genuinely degrades - not silently, a real
    measured failure mode - at dimensions too low for the sequence length
    (e.g. ``dim=64`` for 12 items recovers only 9/12 positions correctly,
    see ``test_decode_sequence_degrades_at_low_dimension``), the same
    dimension-vs-bundle-size tradeoff this project already documents for
    :func:`~zeuss.tier2_substrate.collapse.train_codebook`.
    """
    anchor = codebook.symbol(_SEQUENCE_ANCHOR_NAME)
    out = []
    for i in range(length):
        role = permute(anchor, i)
        recovered = unbind(seq, role)
        out.append(codebook.cleanup(recovered))
    return out
