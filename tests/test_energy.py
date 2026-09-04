import numpy as np

from zeuss.tier2_substrate.energy import Landscape, settle, settle_adaptive
from zeuss.tier2_substrate.hypervectors import (
    Codebook,
    encode_record,
    normalize,
    similarity,
    unbind,
)


def _noisy(z, sigma, seed):
    rng = np.random.default_rng(seed)
    return normalize(z * np.exp(1j * rng.normal(0.0, sigma, size=z.shape[0])))


def test_settle_reduces_energy_from_noisy_start():
    cb = Codebook(dim=8192, seed=2)
    rec = encode_record(cb, [("colour", "red")])
    probe = unbind(rec, cb.symbol("colour"))
    noisy = _noisy(probe, sigma=1.2, seed=99)  # far from any basin
    land = Landscape().add(cb.symbol("red")).add(cb.symbol("blue"))
    _, energies = settle(land, noisy, steps=60, temperature=0.0)
    # A deterministic descent must not end above where it started, and from a
    # noisy start it should make real progress toward the ground state.
    assert energies[-1] <= energies[0] + 1e-9
    assert energies[-1] < energies[0] - 1e-3


def test_settle_recovers_correct_basin():
    cb = Codebook(dim=8192, seed=5)
    rec = encode_record(cb, [("colour", "red")])
    probe = unbind(rec, cb.symbol("colour"))
    noisy = _noisy(probe, sigma=1.0, seed=3)
    land = Landscape().add(cb.symbol("red")).add(cb.symbol("blue"))
    z_final, _ = settle(land, noisy, steps=80, temperature=0.0)
    assert similarity(z_final, cb.symbol("red")) > similarity(z_final, cb.symbol("blue"))


def test_zero_temperature_is_deterministic():
    cb = Codebook(dim=4096, seed=2)
    rec = encode_record(cb, [("colour", "red")])
    probe = unbind(rec, cb.symbol("colour"))
    land = Landscape().add(cb.symbol("red")).add(cb.symbol("blue"))
    z1, _ = settle(land, probe, steps=20, temperature=0.0)
    z2, _ = settle(land, probe, steps=20, temperature=0.0)
    assert np.allclose(z1, z2)


def _two_attractor_landscape(dim, seed):
    rng = np.random.default_rng(seed)
    from zeuss.tier2_substrate.hypervectors import random_hypervector

    a = random_hypervector(dim, rng)
    b = random_hypervector(dim, rng)
    return rng, a, b, Landscape().add(a, 1.0).add(b, 1.0)


def test_settle_adaptive_terminates_early_via_convergence_detection():
    rng, a, _b, land = _two_attractor_landscape(4096, seed=0)
    z0 = normalize(a * np.exp(1j * rng.normal(0.0, 0.2, size=a.shape[0])))
    _, energies, step_sizes = settle_adaptive(land, z0, max_steps=200)
    # Convergence detection should fire well before exhausting max_steps.
    assert len(step_sizes) < 200
    assert energies[-1] < energies[0]


def test_settle_adaptive_step_size_grows_on_open_gradient_shrinks_near_plateau():
    from zeuss.tier2_substrate.hypervectors import bundle

    rng, a, b, land = _two_attractor_landscape(4096, seed=0)
    # Easy: start close to one attractor - an unambiguous, open gradient.
    z0_easy = normalize(a * np.exp(1j * rng.normal(0.0, 0.2, size=a.shape[0])))
    _, energies_easy, steps_easy = settle_adaptive(land, z0_easy, max_steps=200)
    # Ambiguous: start exactly equidistant between the two attractors, a
    # genuine saddle where the pull toward each attractor cancels out.
    z0_amb = bundle([a, b])
    _, energies_amb, steps_amb = settle_adaptive(land, z0_amb, max_steps=200)

    assert np.mean(steps_easy) > np.mean(steps_amb)
    assert max(steps_easy) > max(steps_amb)
    # The ambiguous start also settles to a visibly worse (higher) energy,
    # since it gets stuck at the saddle rather than resolving toward either
    # attractor.
    assert float(energies_amb[-1]) > float(energies_easy[-1])


def test_settle_adaptive_step_sizes_stay_within_range():
    _rng, a, _b, land = _two_attractor_landscape(2048, seed=1)
    lo, hi = 0.05, 0.5
    _, _, step_sizes = settle_adaptive(land, a, max_steps=200, step_size_range=(lo, hi))
    assert all(lo <= s <= hi for s in step_sizes)
