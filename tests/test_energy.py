import numpy as np

from zeuss.tier2_substrate.energy import Landscape, settle
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
