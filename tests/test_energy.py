import numpy as np
import pytest

from zeuss.backend import HAS_JAX
from zeuss.tier2_substrate.energy import Landscape, settle, settle_adaptive, settle_grad
from zeuss.tier2_substrate.hypervectors import (
    Codebook,
    encode_record,
    normalize,
    random_hypervector,
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


def test_settle_adaptive_temperature_is_a_true_noop_by_default():
    """`temperature=0.0` (the default - see energy.py v0.33) must reproduce
    the exact prior behavior bit-for-bit, mirroring this project's own
    established true-no-op convention for every new opt-in parameter.
    Checked directly, not inferred from the code's `if temperature > 0.0`
    guard shape."""
    rng, a, _b, land = _two_attractor_landscape(2048, seed=0)
    z0 = normalize(a * np.exp(1j * rng.normal(0.0, 0.2, size=a.shape[0])))
    z_a, e_a, s_a = settle_adaptive(land, z0, max_steps=50, rng=np.random.default_rng(1))
    z_b, e_b, s_b = settle_adaptive(land, z0, max_steps=50, temperature=0.0, rng=np.random.default_rng(1))
    assert np.allclose(z_a, z_b)
    assert np.allclose(e_a, e_b)
    assert s_a == s_b


def test_settle_adaptive_temperature_explores_thermally():
    """`temperature > 0.0` must actually perturb the trajectory (the same
    von-Mises-like phase noise `settle` already injects) - a real behavior
    change, not just an accepted-but-ignored parameter."""
    rng, a, _b, land = _two_attractor_landscape(2048, seed=0)
    z0 = normalize(a * np.exp(1j * rng.normal(0.0, 0.2, size=a.shape[0])))
    z_cold, _e_cold, _s_cold = settle_adaptive(land, z0, max_steps=50, rng=np.random.default_rng(1))
    z_hot, _e_hot, _s_hot = settle_adaptive(
        land, z0, max_steps=50, temperature=0.3, rng=np.random.default_rng(1)
    )
    assert not np.allclose(z_cold, z_hot)


def test_settle_grad_requires_jax_backend():
    if HAS_JAX:
        pytest.skip("this environment's active backend is already JAX")
    land = Landscape().add(random_hypervector(64, np.random.default_rng(0)))
    with pytest.raises(RuntimeError):
        settle_grad(land, random_hypervector(64, np.random.default_rng(1)))


def test_settle_grad_matches_landscape_energy_formula():
    """settle_grad's internal energy_of_theta is a self-contained restatement
    of Landscape.energy (see settle_grad's docstring for why it can't just
    call Landscape.energy/similarity directly under jax.grad) - pin down that
    restatement is numerically exact, not just "close enough", by comparing
    settle_grad's very first energy entry (steps=0, no descent yet) against
    Landscape.energy on the same (normalized) starting point."""
    pytest.importorskip("jax")
    if not HAS_JAX:
        pytest.skip("jax is importable but not the active backend")
    cb = Codebook(dim=1024, seed=2)
    rec = encode_record(cb, [("colour", "red")])
    probe = unbind(rec, cb.symbol("colour"))
    land = Landscape().add(cb.symbol("red")).add(cb.symbol("blue"))
    _z, energies = settle_grad(land, probe, steps=0)
    assert float(energies[0]) == pytest.approx(land.energy(normalize(probe), 8.0), abs=1e-9)


def test_settle_grad_reduces_energy_and_recovers_correct_basin():
    pytest.importorskip("jax")
    if not HAS_JAX:
        pytest.skip("jax is importable but not the active backend")
    cb = Codebook(dim=8192, seed=2)
    rec = encode_record(cb, [("colour", "red")])
    probe = unbind(rec, cb.symbol("colour"))
    noisy = _noisy(probe, sigma=1.2, seed=99)
    land = Landscape().add(cb.symbol("red")).add(cb.symbol("blue"))
    z_final, energies = settle_grad(land, noisy, steps=60)
    # Same shape of claim as test_settle_reduces_energy_from_noisy_start:
    # a real gradient step never increases energy, and real progress is made.
    assert float(energies[-1]) <= float(energies[0]) + 1e-9
    assert float(energies[-1]) < float(energies[0]) - 1e-3
    assert similarity(z_final, cb.symbol("red")) > similarity(z_final, cb.symbol("blue"))


def test_settle_grad_reaches_comparable_ground_state_to_settle():
    """Different update rule (a literal gradient step vs. settle()'s
    hand-derived mean-field fixed point), same claim: from the same noisy
    start, both should settle close to the same low-energy ground state -
    confirming settle_grad's dimension-scaled learning rate (see its
    docstring for why the raw gradient needs that scaling) is actually
    calibrated to a comparable step size, not just "eventually converges"."""
    pytest.importorskip("jax")
    if not HAS_JAX:
        pytest.skip("jax is importable but not the active backend")
    cb = Codebook(dim=8192, seed=2)
    rec = encode_record(cb, [("colour", "red")])
    probe = unbind(rec, cb.symbol("colour"))
    noisy = _noisy(probe, sigma=1.2, seed=99)
    land = Landscape().add(cb.symbol("red")).add(cb.symbol("blue"))
    _z_fp, energies_fp = settle(land, noisy, steps=60, temperature=0.0)
    _z_grad, energies_grad = settle_grad(land, noisy, steps=60)
    assert float(energies_grad[-1]) == pytest.approx(float(energies_fp[-1]), abs=5e-3)
