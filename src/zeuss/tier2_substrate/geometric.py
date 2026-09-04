"""Small-grade Clifford geometric algebra Cl(n,0), n <= 6 - GA-HDC (experimental).

**ADDITIVE relation-encoding layer, not a replacement.** `docs/ADAMAI_SPEC.md`
asks for "D > 10,000"-dimensional Clifford multivectors; taken literally that
needs ``2**10000`` blade components, which is computationally nonsensical. The
honest scoping used here: the existing ``D``-dimensional complex-phasor
hypervectors (`hypervectors.py`) remain the bulk high-dimensional memory,
unchanged. This module adds a small, separate Clifford algebra (``n <= 6``,
up to 64 blade coefficients) for encoding *relation structure* as rotors -
a rotation in a chosen plane, generalizing `hypervectors.bind`'s phase
multiplication. It is bridged back to the existing algebra by a genuine,
testable fact: ``Cl(2,0)``'s even subalgebra (scalar + pseudoscalar) is
isomorphic to the complex numbers `hypervectors.py` already uses (see
`tests/test_geometric.py`) - this is a proper generalisation, not an
unrelated bolt-on.

A multivector is a vector of ``2**n`` blade coefficients (scalar, ``n``
vectors, ``C(n,2)`` bivectors, ..., one pseudoscalar), indexed by bitmask
(bit ``i`` set means basis vector ``e_i`` is present in that blade). The
geometric product is computed via a precomputed sign/index table over blade
bitmasks (every ``e_i**2 = +1``, since the signature is ``(n, 0)``) -
verified against known ``Cl(2,0)`` identities (e.g. ``e12*e12 == -1``, the
same relation that makes the even subalgebra behave like ``i**2 == -1``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_GRADE = 6  # 2**6 = 64 blade components - keeps the multiplication table tiny.


def _blade_multiply(a: int, b: int) -> tuple[int, int]:
    """``e_a * e_b -> (bitmask, sign)`` in an orthonormal Cl(n,0) basis."""
    swaps = 0
    bb = b
    while bb:
        lowest = bb & (-bb)
        i = lowest.bit_length() - 1
        higher_mask = a & ~((1 << (i + 1)) - 1)
        swaps += bin(higher_mask).count("1")
        bb ^= lowest
    return a ^ b, (-1 if swaps % 2 else 1)


_TABLE_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _table(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``(result_index, sign)`` tables of shape ``(2**n, 2**n)``, cached per ``n``."""
    if n not in _TABLE_CACHE:
        if n > MAX_GRADE:
            raise ValueError(f"n={n} exceeds MAX_GRADE={MAX_GRADE} (2**n blade components)")
        dim = 1 << n
        result = np.zeros((dim, dim), dtype=np.int64)
        sign = np.zeros((dim, dim), dtype=np.float64)
        for a in range(dim):
            for b in range(dim):
                r, s = _blade_multiply(a, b)
                result[a, b] = r
                sign[a, b] = s
        _TABLE_CACHE[n] = (result, sign)
    return _TABLE_CACHE[n]


@dataclass
class Multivector:
    """An element of Cl(n,0): ``2**n`` real blade coefficients."""

    n: int
    coeffs: np.ndarray

    @staticmethod
    def scalar(n: int, value: float = 1.0) -> "Multivector":
        c = np.zeros(1 << n)
        c[0] = value
        return Multivector(n, c)

    @staticmethod
    def basis_vector(n: int, i: int) -> "Multivector":
        c = np.zeros(1 << n)
        c[1 << i] = 1.0
        return Multivector(n, c)

    @staticmethod
    def basis_bivector(n: int, i: int, j: int) -> "Multivector":
        if i == j:
            raise ValueError("a bivector needs two distinct basis vectors")
        c = np.zeros(1 << n)
        c[(1 << i) | (1 << j)] = 1.0
        return Multivector(n, c)

    def reverse(self) -> "Multivector":
        """Grade-involution reverse: flips the sign of each grade-``k`` blade
        by ``(-1)**(k*(k-1)//2)`` - used to build the sandwich product."""
        c = self.coeffs.copy()
        for blade in range(len(c)):
            k = bin(blade).count("1")
            if (k * (k - 1) // 2) % 2:
                c[blade] = -c[blade]
        return Multivector(self.n, c)

    def norm(self) -> float:
        return float(np.sqrt(np.sum(self.coeffs**2)))


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    if a.n != b.n:
        raise ValueError("multivectors must share the same n")
    result_idx, sign = _table(a.n)
    outer = np.outer(a.coeffs, b.coeffs) * sign
    out = np.zeros(1 << a.n)
    np.add.at(out, result_idx.ravel(), outer.ravel())
    return Multivector(a.n, out)


def rotor(bivector: Multivector, angle: float) -> Multivector:
    """``exp(bivector * angle / 2)`` for a *simple* bivector (a single
    basis-plane generator, e.g. :meth:`Multivector.basis_bivector`).

    A unit simple bivector ``B`` satisfies ``B*B == -1`` (the same relation
    that makes ``e12`` behave like the imaginary unit in Cl(2,0)'s even
    subalgebra), so the exponential has the closed form
    ``cos(theta) + sin(theta) * B`` - this closed form is only valid for a
    *simple* bivector; a sum of bivectors spanning more than one plane (e.g.
    ``e12 + e34``) is not handled by this function.
    """
    norm = bivector.norm()
    if norm == 0.0:
        return Multivector.scalar(bivector.n, 1.0)
    unit = bivector.coeffs / norm
    half = angle / 2.0
    coeffs = np.cos(half) * Multivector.scalar(bivector.n, 1.0).coeffs + np.sin(half) * unit
    return Multivector(bivector.n, coeffs)


def apply_rotor(r: Multivector, v: Multivector) -> Multivector:
    """The sandwich product ``r * v * reverse(r)`` - the GA analogue of
    ``hypervectors.bind`` for relation application: relation-as-rotation
    instead of relation-as-phase-multiply."""
    return geometric_product(geometric_product(r, v), r.reverse())
