"""Threshold-gated activation: dormant groups stay dormant, active ones settle."""
import numpy as np

from zeuss.tier2_substrate.energy import Landscape, settle
from zeuss.tier2_substrate.hypervectors import normalize, random_hypervector, similarity
from zeuss.tier2_substrate.spiking import SpikingGate, gated_settle, group_seeds


def _two_group_landscape(dim, seed, closed_weight=20.0):
    rng = np.random.default_rng(seed)
    a = random_hypervector(dim, rng)  # "open" group's attractor
    b = random_hypervector(dim, rng)  # "closed" group's attractor
    land = Landscape().add(a, 1.0).add(b, closed_weight)
    group_of = ["open", "closed"]
    return rng, a, b, land, group_of


def test_gate_stays_dormant_below_threshold():
    _rng, a, b, _land, _group_of = _two_group_landscape(2048, seed=0)
    gate = SpikingGate(threshold=0.3, refractory_steps=3)
    active = gate.poll(a, {"open": a, "closed": b})
    assert active == ["open"]  # far below threshold for "closed" (quasi-orthogonal)


def test_gate_activates_and_holds_through_refractory_period():
    _rng, a, b, _land, _group_of = _two_group_landscape(2048, seed=0)
    gate = SpikingGate(threshold=0.3, refractory_steps=3)
    groups = {"closed": b}

    assert gate.poll(a, groups) == []  # probe far from b: dormant
    assert "closed" in gate.poll(b, groups)  # probe crosses threshold: activates

    # Probe drifts back to being far from b, but stays active through the
    # refractory window (hysteresis), then goes dormant again after it lapses.
    for _ in range(3):
        assert "closed" in gate.poll(a, groups)
    assert gate.poll(a, groups) == []


def test_gated_settle_only_moves_toward_active_attractors():
    rng, a, b, land, group_of = _two_group_landscape(4096, seed=0, closed_weight=20.0)
    z0 = normalize(a * np.exp(1j * rng.normal(0.0, 0.3, size=a.shape[0])))
    gate = SpikingGate(threshold=0.3, refractory_steps=3)

    z_gated, _ = gated_settle(gate, land, group_of, z0, steps=40, poll_every=5, inverse_temperature=1.0)
    z_plain, _ = settle(land, z0, steps=40, inverse_temperature=1.0)

    # Plain settle is pulled hard toward "closed" (it has 20x the weight);
    # gated settle never lets "closed" activate, since z0 never crosses the
    # threshold, so it should stay near its start instead.
    assert similarity(z_plain, b) > 0.9
    assert similarity(z_gated, b) < 0.1


def test_group_seeds_bundles_each_groups_attractors():
    _rng, a, b, land, group_of = _two_group_landscape(1024, seed=1)
    seeds = group_seeds(land, group_of)
    assert set(seeds) == {"open", "closed"}
    assert similarity(seeds["open"], a) > 0.99
    assert similarity(seeds["closed"], b) > 0.99
