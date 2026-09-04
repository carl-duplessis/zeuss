"""Compiling a fuzzy Theory into a Landscape: settling reproduces its truth."""
import numpy as np
import pytest

from zeuss.tier2_substrate.energy import settle
from zeuss.tier2_substrate.hypervectors import Codebook, random_hypervector
from zeuss.tier3_logic.compiler import Rule, Theory
from zeuss.tier3_logic.grounding import InconsistentTheoriesError, compile_theories, compile_theory, readout


def test_compile_theory_ground_state_matches_satisfying_valuation():
    codebook = Codebook(dim=4096, seed=11)
    variables = ["rain", "wet"]
    theory = Theory(rules=[Rule("rain", "wet", weight=1.0)])
    landscape = compile_theory(codebook, theory, variables)

    rng = np.random.default_rng(0)
    z0 = random_hypervector(codebook.dim, rng)
    z_final, energies = settle(landscape, z0, steps=80, rng=rng)

    valuation = readout(codebook, variables, z_final)
    assert theory.satisfied(valuation, tol=0.2)
    # Energy should have gone down over the descent.
    assert float(energies[-1]) <= float(energies[0])


def test_compile_theory_excludes_high_energy_corners():
    # A pure-implication Theory is always vacuously satisfied by the "all true"
    # corner (a=b=1 => truth=1), so there is no such thing as an unsatisfiable
    # Theory of this shape - that's an honest property of implication-only
    # logic, not a gap in compile_theory. What *is* testable: a strongly
    # weighted, violated rule should push a corner's Boltzmann weight below
    # WEIGHT_FLOOR, so compile_theory doesn't register it as an attractor at
    # all (it never fabricates a basin for a badly-violated configuration).
    codebook = Codebook(dim=2048, seed=12)
    variables = ["a", "b"]
    theory = Theory(rules=[Rule("a", "b", weight=20.0)])
    landscape = compile_theory(codebook, theory, variables)

    # Only the two corners consistent with "a -> b" (a=0,* and a=1,b=1) should
    # survive the weight floor; the violating corner (a=1, b=0) should not.
    assert len(landscape.attractors) == 3
    assert all(w == 1.0 for w in landscape.weights)  # each surviving corner is exactly satisfying


def test_compile_theories_merges_agreeing_theories():
    codebook = Codebook(dim=4096, seed=0)
    theories = {
        "alice": Theory(rules=[Rule("TRUE", "door_open", weight=10.0)]),
        "bob": Theory(rules=[Rule("TRUE", "door_open", weight=10.0)]),
    }
    shared_vars = {"alice": ["door_open"], "bob": ["door_open"]}
    landscape = compile_theories(codebook, theories, shared_vars)

    rng = np.random.default_rng(1)
    z0 = random_hypervector(codebook.dim, rng)
    z_final, _energies = settle(landscape, z0, steps=80, rng=rng)
    valuation = readout(codebook, ["door_open"], z_final)
    assert valuation["door_open"] > 0.5  # both agents agree it should be true


def test_compile_theories_raises_on_disagreement_instead_of_blending():
    codebook = Codebook(dim=4096, seed=0)
    theories = {
        "alice": Theory(rules=[Rule("TRUE", "door_open", weight=10.0)]),   # believes open
        "carol": Theory(rules=[Rule("door_open", "ZERO", weight=10.0)]),   # believes closed
    }
    shared_vars = {"alice": ["door_open"], "carol": ["door_open"]}

    with pytest.raises(InconsistentTheoriesError) as excinfo:
        compile_theories(codebook, theories, shared_vars)

    assert excinfo.value.violations == [("alice:door_open", "carol:door_open", 1.0)]
