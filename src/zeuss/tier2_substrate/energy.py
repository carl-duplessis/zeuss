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

from ..backend import HAS_JAX, RDTYPE, xp
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
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
):
    """Descent with a step size that adapts to local complexity - optionally
    thermal, unlike before.

    "Liquid time-step": the step size for the *next* move scales with how much
    energy the *previous* move actually removed. A small improvement (near a
    plateau or a sharp constraint boundary - a genuinely hard/ambiguous step)
    shrinks the step; a large improvement (an open, easy gradient) grows it,
    clipped to ``step_size_range``. Stops early once the energy improvement
    stays below ``energy_tol`` for three consecutive steps (converged), or at
    ``max_steps``. This is plain adaptive step-size control, not a neural ODE
    solver - the concrete, testable content behind "adapts its time-step to
    problem complexity" rather than a literal claim of continuous-time dynamics.

    ``temperature`` (default ``0.0``, matching every prior caller's exact
    behavior - a true no-op) injects the same von-Mises-like phase noise
    :func:`settle` already does, after each adaptive step - added so the
    adaptive-step-size mechanism and probabilistic exploration aren't
    mutually exclusive; before this, only plain :func:`settle` could explore
    thermally, and only this function could adapt its own step size.
    ``plateau_steps``' convergence check uses the improvement *before* noise
    is applied (the noise itself perturbs energy every step, which would
    otherwise make "3 consecutive tiny-improvement steps" nearly impossible
    to ever trigger at ``temperature > 0`` - convergence should track whether
    the *descent* has stalled, not whether thermal noise happens to be
    quiet).

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
        step_sizes.append(step_size)

        improvement = prev_energy - e  # positive = energy went down, before any noise
        if abs(improvement) < energy_tol:
            plateau_steps += 1
        else:
            plateau_steps = 0

        if temperature > 0.0:
            noise = rng.normal(0.0, temperature, size=z.shape[0])
            z = normalize(z * xp.exp(1j * xp.asarray(noise, dtype=RDTYPE)))
            e = landscape.energy(z, inverse_temperature)
        energies.append(e)

        if plateau_steps >= 3:
            break

        # tanh squashes the (unbounded) improvement into a bounded growth
        # factor in [1 - gain, 1 + gain]; a bigger drop grows the next step.
        gain = 0.5
        factor = 1.0 + gain * float(np.tanh(improvement * 10.0))
        step_size = float(np.clip(step_size * factor, lo, hi))

    return z, xp.asarray(energies, dtype=RDTYPE), step_sizes


def settle_grad(
    landscape: Landscape,
    z0,
    steps: int = 60,
    learning_rate: float = 0.5,
    inverse_temperature: float = 8.0,
):
    """Energy descent via ``jax.grad`` on phase angles - the v0.2 roadmap item
    ("a JAX `grad`-based energy-descent variant of `energy.settle`").

    :func:`settle` and :func:`settle_adaptive` move ``z`` by relaxing it
    toward ``landscape.target(z)`` (a hand-derived mean-field fixed point).
    This instead takes a literal gradient step on the energy itself: state is
    reparameterised as real phase angles ``theta`` (``z = exp(i*theta)``),
    and ``theta`` is updated by ``jax.grad`` of the energy w.r.t. ``theta``.
    Differentiating on angles rather than the raw complex ``z`` is
    deliberate - ``theta`` is real-valued and unconstrained, so ordinary
    real-to-real ``jax.grad`` applies directly, and ``z = exp(i*theta)`` is
    *exactly* unit-modulus by construction, with no separate ``normalize()``
    projection needed after each step (unlike ``settle``/``settle_adaptive``).

    Requires the JAX backend (raises ``RuntimeError`` otherwise - there is no
    meaningful autodiff fallback on plain NumPy). :func:`Landscape.energy`
    and :func:`hypervectors.similarity` return plain Python ``float``s via an
    explicit cast, which is correct for their normal (non-traced) call sites
    everywhere else in this project but would abort a JAX trace the moment
    ``jax.grad`` reached it. So the energy here is a small, self-contained
    restatement of :meth:`Landscape.energy`'s exact formula directly in terms
    of ``theta`` - kept numerically identical to it (see
    ``test_settle_grad_matches_landscape_energy_formula``) rather than
    routing through those existing NumPy-facing functions.

    ``learning_rate`` is scaled by the hypervector dimension ``D`` internally.
    The raw gradient is tiny (measured: norm ~0.008 over D=8192 components,
    i.e. ~1e-4 per component) because the energy formula divides by ``D``
    twice - once in similarity's own normalisation, once in the
    softmax-weighted sum over a fixed-size attractor set - so an unscaled
    step needs a learning rate in the thousands to move at all. Scaling by
    ``D`` keeps ``learning_rate`` on a step-size-like scale comparable to
    ``settle``'s (confirmed empirically: ``learning_rate=0.5`` reaches
    essentially the same ground-state energy as ``settle``'s default
    ``step_size=0.3`` in the same 60 steps, on the noisy-probe test case
    below), independent of dimensionality.

    Returns ``(z_final, energies)`` - the same shape of result as
    :func:`settle`.
    """
    if not HAS_JAX:
        raise RuntimeError(
            "settle_grad requires the JAX backend - install the 'jax' extra "
            "(pip install 'zeuss[jax]') and leave ZEUSS_BACKEND unset or set it to 'jax'"
        )
    import jax

    attractors = xp.stack(landscape.attractors, axis=0)
    weights = xp.asarray(landscape.weights, dtype=RDTYPE)
    dim = attractors.shape[1]

    def energy_of_theta(theta):
        z = xp.exp(1j * theta)
        sims = xp.real(xp.conj(attractors) @ z) / z.shape[0]
        logits = inverse_temperature * sims
        p = xp.exp(logits - xp.max(logits))
        p = p / xp.sum(p)
        return -xp.sum(weights * p * sims)

    grad_fn = jax.grad(energy_of_theta)
    effective_lr = learning_rate * dim

    theta = xp.angle(normalize(z0))
    energies = [float(energy_of_theta(theta))]
    for _ in range(steps):
        theta = theta - effective_lr * grad_fn(theta)
        energies.append(float(energy_of_theta(theta)))

    return normalize(xp.exp(1j * theta)), xp.asarray(energies, dtype=RDTYPE)
