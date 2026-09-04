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

import numpy as np

from ..backend import RDTYPE, xp
from .collapse import softmax
from .hypervectors import normalize, similarity


@dataclass
class Landscape:
    """A set of weighted attractor hypervectors (the axioms)."""

    attractors: list = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def add(self, vector, weight: float = 1.0) -> "Landscape":
        self.attractors.append(normalize(vector))
        self.weights.append(float(weight))
        return self

    def energy(self, z, inverse_temperature: float = 8.0) -> float:
        """Free-energy-like scalar; lower means better satisfied."""
        sims = xp.asarray([similarity(z, a) for a in self.attractors], dtype=RDTYPE)
        w = xp.asarray(self.weights, dtype=RDTYPE)
        p = softmax(inverse_temperature * sims)
        # Weighted alignment minus an entropy term (log-sum-exp is the free energy).
        return float(-xp.sum(w * p * sims))

    def target(self, z, inverse_temperature: float) -> "xp.ndarray":
        sims = xp.asarray([similarity(z, a) for a in self.attractors], dtype=RDTYPE)
        p = softmax(inverse_temperature * sims) * xp.asarray(self.weights, dtype=RDTYPE)
        acc = xp.zeros_like(self.attractors[0])
        for pk, a in zip(p, self.attractors):
            acc = acc + pk * a
        return normalize(acc)


def settle(
    landscape: Landscape,
    z0,
    steps: int = 60,
    step_size: float = 0.3,
    temperature: float = 0.0,
    inverse_temperature: float = 8.0,
    rng: np.random.Generator | None = None,
):
    """Relax ``z0`` toward the logical ground state of ``landscape``.

    ``temperature`` > 0 explores (probabilistic reasoning); ``temperature`` = 0
    is a deterministic descent. Thermal noise is drawn from an explicit NumPy
    ``Generator`` and lifted onto the active backend. Returns ``(z_final,
    energies)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    z = normalize(z0)
    energies = [landscape.energy(z, inverse_temperature)]
    for _ in range(steps):
        t = landscape.target(z, inverse_temperature)
        z = normalize((1.0 - step_size) * z + step_size * t)
        if temperature > 0.0:
            noise = rng.normal(0.0, temperature, size=z.shape[0])
            z = normalize(z * xp.exp(1j * xp.asarray(noise, dtype=RDTYPE)))
        energies.append(landscape.energy(z, inverse_temperature))
    return z, xp.asarray(energies, dtype=RDTYPE)


def settle_adaptive(
    landscape: Landscape,
    z0,
    max_steps: int = 200,
    step_size_range: tuple[float, float] = (0.05, 0.5),
    energy_tol: float = 1e-4,
    inverse_temperature: float = 8.0,
    rng: np.random.Generator | None = None,
):
    """Deterministic descent with a step size that adapts to local complexity.

    "Liquid time-step": the step size for the *next* move scales with how much
    energy the *previous* move actually removed. A small improvement (near a
    plateau or a sharp constraint boundary - a genuinely hard/ambiguous step)
    shrinks the step; a large improvement (an open, easy gradient) grows it,
    clipped to ``step_size_range``. Stops early once the energy improvement
    stays below ``energy_tol`` for three consecutive steps (converged), or at
    ``max_steps``. This is plain adaptive step-size control, not a neural ODE
    solver - the concrete, testable content behind "adapts its time-step to
    problem complexity" rather than a literal claim of continuous-time dynamics.

    Returns ``(z_final, energies, step_sizes)`` - ``step_sizes`` is the trace
    of step sizes actually used, direct evidence the adaptation happened.
    """
    rng = np.random.default_rng() if rng is None else rng
    lo, hi = step_size_range
    z = normalize(z0)
    step_size = 0.5 * (lo + hi)
    energies = [landscape.energy(z, inverse_temperature)]
    step_sizes: list[float] = []
    plateau_steps = 0
    for _ in range(max_steps):
        prev_energy = energies[-1]
        t = landscape.target(z, inverse_temperature)
        z = normalize((1.0 - step_size) * z + step_size * t)
        e = landscape.energy(z, inverse_temperature)
        energies.append(e)
        step_sizes.append(step_size)

        improvement = prev_energy - e  # positive = energy went down
        if abs(improvement) < energy_tol:
            plateau_steps += 1
        else:
            plateau_steps = 0
        if plateau_steps >= 3:
            break

        # tanh squashes the (unbounded) improvement into a bounded growth
        # factor in [1 - gain, 1 + gain]; a bigger drop grows the next step.
        gain = 0.5
        factor = 1.0 + gain * float(np.tanh(improvement * 10.0))
        step_size = float(np.clip(step_size * factor, lo, hi))

    return z, xp.asarray(energies, dtype=RDTYPE), step_sizes
