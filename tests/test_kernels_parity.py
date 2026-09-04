import numpy as np

from zeuss.tier1_kernels.reference import phase_interference, topological_collapse_step
from zeuss.tier2_substrate.hypervectors import random_hypervector


def test_phase_interference_matches_manual_sum():
    rng = np.random.default_rng(0)
    fields = np.stack([random_hypervector(1024, rng) for _ in range(4)])
    got = phase_interference(fields)
    assert np.allclose(got, fields.sum(axis=0))


def test_collapse_step_moves_toward_best_basis():
    rng = np.random.default_rng(0)
    basis = np.stack([random_hypervector(2048, rng) for _ in range(3)])
    z = basis[1].copy()
    out = topological_collapse_step(z, basis, beta=32.0)
    sims = (basis @ np.conj(out)).real / out.shape[0]
    assert int(np.argmax(sims)) == 1
