"""Tier-2 is routed through ``zeuss.backend.xp`` (v0.2, roadmap item 1).

These tests pin the contract that the substrate math runs on the *active*
backend array namespace - NumPy today, JAX when installed - rather than a
hard-coded ``numpy``. The heavy behavioural coverage lives in the other Tier-2
test modules; here we only assert routing and dtype, plus a backend-parity
check that activates once JAX is available.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from zeuss.backend import CDTYPE, HAS_JAX, backend_name, to_numpy, xp
from zeuss.tier2_substrate.energy import Landscape, settle
from zeuss.tier2_substrate.hypervectors import (
    Codebook,
    bind,
    bundle,
    encode_record,
    random_hypervector,
    unbind,
)
from zeuss.tier2_substrate.resonance import interfere

# The concrete array class of whatever backend is active (np.ndarray or jax.Array).
_ARRAY_TYPE = type(xp.asarray([0.0]))


def test_backend_name_is_known():
    assert backend_name() in ("numpy", "jax")


def test_substrate_ops_return_backend_arrays():
    rng = np.random.default_rng(0)
    a = random_hypervector(1024, rng)
    b = random_hypervector(1024, rng)
    for out in (a, bind(a, b), unbind(a, b), bundle([a, b]), interfere([a, b])):
        assert isinstance(out, _ARRAY_TYPE)
        assert out.dtype == CDTYPE


def test_settle_energies_are_backend_arrays():
    cb = Codebook(dim=1024, seed=2)
    rec = encode_record(cb, [("colour", "red")])
    probe = unbind(rec, cb.symbol("colour"))
    land = Landscape().add(cb.symbol("red")).add(cb.symbol("blue"))
    z, energies = settle(land, probe, steps=5, temperature=0.0)
    assert isinstance(z, _ARRAY_TYPE)
    assert isinstance(energies, _ARRAY_TYPE)


def test_random_hypervector_seed_is_backend_independent():
    # Randomness comes from an explicit NumPy Generator and is *lifted* onto the
    # backend, so the same seed yields the same phasors regardless of backend.
    v = random_hypervector(256, np.random.default_rng(7))
    expected_theta = np.random.default_rng(7).uniform(-np.pi, np.pi, size=256)
    assert np.allclose(to_numpy(v), np.exp(1j * expected_theta))


@pytest.mark.skipif(HAS_JAX, reason="active backend is already JAX")
def test_jax_backend_matches_numpy_parity():
    """When JAX is installed, forcing it must reproduce the NumPy results.

    Run in a subprocess so the module-level backend selection happens under
    ``ZEUSS_BACKEND=jax`` without disturbing this (NumPy) process.
    """
    pytest.importorskip("jax")
    env = dict(os.environ, ZEUSS_BACKEND="jax", PYTHONPATH="src")
    code = (
        "import numpy as np;"
        "from zeuss.backend import backend_name, to_numpy;"
        "from zeuss.tier2_substrate.hypervectors import random_hypervector, bind, unbind, similarity;"
        "assert backend_name() == 'jax';"
        "rng = np.random.default_rng(0);"
        "a = random_hypervector(1024, rng); b = random_hypervector(1024, rng);"
        "print(round(similarity(unbind(bind(a, b), a), b), 6))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert float(proc.stdout.strip()) > 0.99
