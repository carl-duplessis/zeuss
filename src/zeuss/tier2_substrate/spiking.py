"""Frontier: event-spiking asynchronous activation.

The concrete, testable content behind "neuromorphic threshold-gated
activation" is a sparse activation gate: sub-groups of a :class:`Landscape`
stay excluded from the (per-group O(k)) settle computation until an incoming
probe's similarity to that group crosses an activation threshold, and then
stay active for a short refractory period afterward (hysteresis, so a probe
drifting near the boundary doesn't thrash the gate open and shut). This is not
a spiking-neuron circuit simulation - it's a boolean mask over which attractor
groups participate in a settle step, which is the actual compute-saving idea
the spec is reaching for.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..backend import RDTYPE, xp
from .energy import Landscape, settle
from .hypervectors import bundle, normalize, similarity


@dataclass
class SpikingGate:
    """Threshold-gated activation with refractory hysteresis over named groups."""

    threshold: float = 0.15
    refractory_steps: int = 3
    _active_until: dict = field(default_factory=dict)
    _tick: int = 0

    def poll(self, probe, groups: dict) -> list[str]:
        """Advance one tick; return the names currently active.

        A group whose similarity to ``probe`` crosses ``threshold`` this tick
        stays active for ``refractory_steps`` further ticks even if the probe
        then drifts away, using ``hypervectors.similarity`` as the sole signal.
        """
        self._tick += 1
        active = []
        for name, seed_vec in groups.items():
            s = similarity(probe, seed_vec)
            if s >= self.threshold:
                self._active_until[name] = self._tick + self.refractory_steps
            if self._active_until.get(name, -1) >= self._tick:
                active.append(name)
        return active


def group_seeds(landscape: Landscape, group_of: list[str]) -> dict:
    """One representative hypervector per group: the bundle of its attractors."""
    buckets: dict[str, list] = defaultdict(list)
    for vec, name in zip(landscape.attractors, group_of):
        buckets[name].append(vec)
    return {name: bundle(vecs) for name, vecs in buckets.items()}


def gated_settle(
    gate: SpikingGate,
    landscape: Landscape,
    group_of: list[str],
    z0,
    steps: int = 60,
    poll_every: int = 5,
    step_size: float = 0.3,
    inverse_temperature: float = 8.0,
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
):
    """Like :func:`energy.settle`, but only currently-active groups pull the state.

    Every ``poll_every`` steps, ``gate.poll`` re-checks which groups are
    active against the current state and a filtered sub-``Landscape``
    containing only their attractors is built; the actual descent steps in
    between polls are delegated to the existing :func:`energy.settle` on that
    sub-landscape (reuse, not reimplementation). Energy is always reported
    against the *full* landscape, so it stays comparable to a plain
    ``settle`` run. Returns ``(z_final, energies)``.
    """
    if len(group_of) != len(landscape.attractors):
        raise ValueError("group_of must have one entry per landscape attractor")
    seeds = group_seeds(landscape, group_of)
    z = normalize(z0)
    energies = [landscape.energy(z, inverse_temperature)]
    remaining = steps
    while remaining > 0:
        chunk = min(poll_every, remaining)
        active = set(gate.poll(z, seeds))
        sub = Landscape()
        for vec, w, name in zip(landscape.attractors, landscape.weights, group_of):
            if name in active:
                sub.add(vec, w)
        if sub.attractors:
            z, _sub_energies = settle(
                sub, z, steps=chunk, step_size=step_size,
                temperature=temperature, inverse_temperature=inverse_temperature, rng=rng,
            )
        # If nothing is active, the state simply doesn't move this chunk.
        for _ in range(chunk):
            energies.append(landscape.energy(z, inverse_temperature))
        remaining -= chunk
    return z, xp.asarray(energies, dtype=RDTYPE)
