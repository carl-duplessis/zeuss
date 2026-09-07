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
from zeuss.tier2_substrate.collapse import collapse_batch
from zeuss.tier2_substrate.energy import Landscape, settle
from zeuss.tier2_substrate.hypervectors import (
    Codebook,
    bind,
    bundle,
    encode_record,
    random_hypervector,
    unbind,
)
from zeuss.tier2_substrate.resonance import interfere, resonate

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


def _run_under_jax(code: str) -> str:
    """Run ``code`` in a subprocess with ``ZEUSS_BACKEND=jax`` forced, so the
    module-level backend selection happens under JAX without disturbing this
    (NumPy) process, and return its stdout. Shared by the property-parity
    tests below - each checks one genuine property this project claims
    (hypervector algebra, energy dynamics, batched collapse), not just one
    hand-picked operation chain, on both backends with the *same* numbers,
    not just "doesn't crash on either."
    """
    env = dict(os.environ, ZEUSS_BACKEND="jax", PYTHONPATH="src")
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@pytest.mark.skipif(HAS_JAX, reason="active backend is already JAX")
def test_jax_backend_matches_numpy_parity_bind_unbind():
    """When JAX is installed, forcing it must reproduce the NumPy result for
    the algebra's core invertibility property: unbind(bind(a, b), a) ~ b."""
    pytest.importorskip("jax")
    stdout = _run_under_jax(
        "import numpy as np;"
        "from zeuss.backend import backend_name;"
        "from zeuss.tier2_substrate.hypervectors import random_hypervector, bind, unbind, similarity;"
        "assert backend_name() == 'jax';"
        "rng = np.random.default_rng(0);"
        "a = random_hypervector(1024, rng); b = random_hypervector(1024, rng);"
        "print(round(similarity(unbind(bind(a, b), a), b), 6))"
    )
    assert float(stdout) > 0.99


@pytest.mark.skipif(HAS_JAX, reason="active backend is already JAX")
def test_jax_backend_matches_numpy_settle_energy_trace():
    """settle()'s full energy trace - not just a spot-checked end value -
    must match numerically between backends. Codebook symbols are derived
    from the same NumPy-seeded generator on both sides
    (test_random_hypervector_seed_is_backend_independent), so the entire
    deterministic-descent trace should agree to float precision, not just
    share the same qualitative "energy decreases" shape."""
    pytest.importorskip("jax")
    cb = Codebook(dim=1024, seed=2)
    rec = encode_record(cb, [("colour", "red")])
    probe = unbind(rec, cb.symbol("colour"))
    land = Landscape().add(cb.symbol("red")).add(cb.symbol("blue"))
    _z, energies_numpy = settle(land, probe, steps=20, temperature=0.0)

    stdout = _run_under_jax(
        "from zeuss.backend import backend_name;"
        "from zeuss.tier2_substrate.energy import Landscape, settle;"
        "from zeuss.tier2_substrate.hypervectors import Codebook, encode_record, unbind;"
        "assert backend_name() == 'jax';"
        "cb = Codebook(dim=1024, seed=2);"
        "rec = encode_record(cb, [('colour', 'red')]);"
        "probe = unbind(rec, cb.symbol('colour'));"
        "land = Landscape().add(cb.symbol('red')).add(cb.symbol('blue'));"
        "_z, energies = settle(land, probe, steps=20, temperature=0.0);"
        "print(','.join(f'{float(e):.10f}' for e in energies))"
    )
    energies_jax = [float(x) for x in stdout.split(",")]
    assert np.allclose(np.asarray(energies_numpy), energies_jax, atol=1e-6)


@pytest.mark.skipif(HAS_JAX, reason="active backend is already JAX")
def test_jax_backend_matches_numpy_resonate():
    """resonate() (Frontier 3's interference readout) must report the same
    recovered filler and coherence on both backends - the fourth Tier-2
    module (hypervectors, energy, collapse, resonance), so this parity
    coverage isn't three-quarters of the substrate's claimed properties."""
    pytest.importorskip("jax")
    rng = np.random.default_rng(5)
    query = random_hypervector(1024, rng)
    role = random_hypervector(1024, rng)
    filler = random_hypervector(1024, rng)
    memory = bind(role, filler)
    _recovered_numpy, coherence_numpy = resonate(query, memory, role)

    stdout = _run_under_jax(
        "import numpy as np;"
        "from zeuss.backend import backend_name;"
        "from zeuss.tier2_substrate.hypervectors import random_hypervector, bind;"
        "from zeuss.tier2_substrate.resonance import resonate;"
        "assert backend_name() == 'jax';"
        "rng = np.random.default_rng(5);"
        "query = random_hypervector(1024, rng); role = random_hypervector(1024, rng);"
        "filler = random_hypervector(1024, rng);"
        "memory = bind(role, filler);"
        "_recovered, coherence = resonate(query, memory, role);"
        "print(round(coherence, 10))"
    )
    assert abs(float(stdout) - coherence_numpy) < 1e-6


@pytest.mark.skipif(HAS_JAX, reason="active backend is already JAX")
def test_jax_backend_matches_numpy_collapse_batch():
    """collapse_batch (the new batched-collapse roadmap item) must resolve
    the same probes to the same winners and entropies on both backends."""
    pytest.importorskip("jax")
    cb = Codebook(dim=1024, seed=1)
    for name in ("red", "blue", "green"):
        cb.symbol(name)
    rng = np.random.default_rng(4)
    probes = [random_hypervector(1024, rng) for _ in range(4)]
    _z, info_numpy = collapse_batch(cb, probes, inverse_temperature=8.0)

    stdout = _run_under_jax(
        "import numpy as np;"
        "from zeuss.backend import backend_name;"
        "from zeuss.tier2_substrate.collapse import collapse_batch;"
        "from zeuss.tier2_substrate.hypervectors import Codebook, random_hypervector;"
        "assert backend_name() == 'jax';"
        "cb = Codebook(dim=1024, seed=1);"
        "[cb.symbol(n) for n in ('red', 'blue', 'green')];"
        "rng = np.random.default_rng(4);"
        "probes = [random_hypervector(1024, rng) for _ in range(4)];"
        "_z, info = collapse_batch(cb, probes, inverse_temperature=8.0);"
        "print('|'.join(info['winner']));"
        "print(','.join(f'{float(e):.10f}' for e in info['entropy_bits']))"
    )
    winners_jax, entropy_line = stdout.splitlines()
    assert winners_jax.split("|") == list(info_numpy["winner"])
    entropy_jax = [float(x) for x in entropy_line.split(",")]
    assert np.allclose(np.asarray(info_numpy["entropy_bits"]), entropy_jax, atol=1e-6)
