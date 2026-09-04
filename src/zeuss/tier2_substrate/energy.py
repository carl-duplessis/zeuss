"""Frontier 2 - micro-energy landscape (thermodynamic invariance).

Logical axioms are encoded as physical energy minima (attractor basins) in a
continuous dynamical system. Probabilistic updates act as thermal fluctuations;
deterministic rules are absolute-zero energy wells. The system does not 'check'
a constraint - it settles into the logical ground state by minimising energy.
Fulfilling a rule and minimising thermodynamic entropy become one operation.

We implement a mean-field settle over the torus of phases. Energy is the
negative alignment with the (softmax-weighted) attractors; temperature injects
von-Mises-like phase noise. At T = 0 the walk descends deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .collapse import softmax
from .hypervectors import normalize, similarity


@dataclass
class Landscape:
    """A set of weighted attractor hypervectors (the axioms)."""

    attractors: list[np.ndarray] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def add(self, vector: np.ndarray, weight: float = 1.0) -> "Landscape":
        self.attractors.append(normalize(vector))
        self.weights.append(float(weight))
        return self

    def energy(self, z: np.ndarray, inverse_temperature: float = 8.0) -> float:
        """Free-energy-like scalar; lower means better satisfied."""
        sims = np.array([similarity(z, a) for a in self.attractors])
        w = np.asarray(self.weights)
        p = softmax(inverse_temperature * sims)
        # Weighted alignment minus an entropy term (log-sum-exp is the free energy).
        return float(-np.sum(w * p * sims))

    def target(self, z: np.ndarray, inverse_temperature: float) -> np.ndarray:
        sims = np.array([similarity(z, a) for a in self.attractors])
        p = softmax(inverse_temperature * sims) * np.asarray(self.weights)
        acc = np.zeros_like(self.attractors[0])
        for pk, a in zip(p, self.attractors):
            acc = acc + pk * a
        return normalize(acc)


def settle(
    landscape: Landscape,
    z0: np.ndarray,
    steps: int = 60,
    step_size: float = 0.3,
    temperature: float = 0.0,
    inverse_temperature: float = 8.0,
    rng: np.random.Generator | None = None,
):
    """Relax ``z0`` toward the logical ground state of ``landscape``.

    ``temperature`` > 0 explores (probabilistic reasoning); ``temperature`` = 0
    is a deterministic descent. Returns ``(z_final, energies)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    z = normalize(z0)
    energies = [landscape.energy(z, inverse_temperature)]
    for _ in range(steps):
        t = landscape.target(z, inverse_temperature)
        z = normalize((1.0 - step_size) * z + step_size * t)
        if temperature > 0.0:
            noise = rng.normal(0.0, temperature, size=z.shape[0])
            z = normalize(z * np.exp(1j * noise))
        energies.append(landscape.energy(z, inverse_temperature))
    return z, np.asarray(energies)
