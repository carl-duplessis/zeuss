"""Compiling a fuzzy Theory into a Landscape: settling reproduces its truth."""
import numpy as np
import pytest

from zeuss.tier2_substrate.energy import settle
from zeuss.tier2_substrate.hypervectors import Codebook, random_hypervector
from zeuss.tier3_logic.compiler import Rule, Theory
from zeuss.tier3_logic.grounding import (
    InconsistentTheoriesError,
    anneal_theory,
    compile_theories,
    compile_theory,
    readout,
    valuation_to_hypervector,
)


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


def test_readout_reaches_near_certainty_for_an_isolated_crisp_variable():
    """FALSE is now TRUE's exact phase-antipode (-true_pole), not an
    independently-drawn codebook symbol - so a single crisply-true variable
    (no other variables bundled in to dilute it) must read out at ~1.0, the
    achievable ceiling the [-2,2]->[0,1] remap assumes. Before this fix,
    false_sim was only ~0 (two merely-distinct random vectors), capping even
    this best case at ~0.75."""
    codebook = Codebook(dim=8192, seed=1)
    z = valuation_to_hypervector(codebook, ["a"], {"a": 1.0})
    out = readout(codebook, ["a"], z)
    assert out["a"] > 0.999


def test_compile_theory_inverse_temperature_narrows_the_live_corner_set():
    """Low inverse_temperature keeps a genuine, broad prior (every corner
    close to satisfying stays a live attractor); high inverse_temperature
    sharpens it toward only the theory's exact zero-energy corners - the
    core "temperature controls prior breadth" claim of the v0.5 roadmap
    item, checked on the compiled Landscape itself before any settling
    dynamics get involved."""
    codebook = Codebook(dim=4096, seed=3)
    theory = Theory(
        rules=[Rule("a", "b", weight=1.0), Rule("b", "a", weight=1.0), Rule("bias", "a", weight=0.5)]
    )
    fixed = {"bias": 1.0}

    low = compile_theory(codebook, theory, ["a", "b"], fixed=fixed, inverse_temperature=0.1)
    high = compile_theory(codebook, theory, ["a", "b"], fixed=fixed, inverse_temperature=50.0)

    assert len(low.attractors) == 4  # all four corners still above the weight floor
    assert len(high.attractors) == 1  # only the theory's unique zero-energy corner survives
    assert high.weights == [1.0]


def test_anneal_theory_crystallises_toward_unique_ground_state():
    """A theory with one unique lowest-energy corner (biconditional a<->b
    plus a rule biasing a toward true) must have its annealed entropy drop
    toward 0 and its final readout land on that corner, not just report a
    lower energy number."""
    codebook = Codebook(dim=4096, seed=3)
    theory = Theory(
        rules=[Rule("a", "b", weight=1.0), Rule("b", "a", weight=1.0), Rule("bias", "a", weight=0.5)]
    )
    fixed = {"bias": 1.0}
    rng = np.random.default_rng(7)

    _z_final, trace = anneal_theory(codebook, theory, ["a", "b"], fixed=fixed, rng=rng)

    assert trace[0]["entropy_bits"] > trace[-1]["entropy_bits"]
    assert trace[-1]["entropy_bits"] < 0.01
    assert trace[-1]["readout"]["a"] > 0.7
    assert trace[-1]["readout"]["b"] > 0.7
    assert theory.energy({**trace[-1]["readout"], **fixed}) < 0.15


def test_anneal_theory_settles_into_a_genuinely_satisfying_corner_even_when_degenerate():
    """A genuinely underdetermined theory (a->b alone: three of four corners
    equally satisfy it) still crystallises to ~0 entropy - a single settling
    trajectory spontaneously breaks the symmetry and commits to *one*
    corner, exactly as a ferromagnet's mean-field descent picks one
    degenerate ground state rather than hovering between them. What must
    hold is that the corner it commits to is a real, satisfying one, not an
    arbitrary point."""
    codebook = Codebook(dim=4096, seed=11)
    theory = Theory(rules=[Rule("a", "b", weight=1.0)])
    rng = np.random.default_rng(5)

    _z_final, trace = anneal_theory(codebook, theory, ["a", "b"], rng=rng)

    assert trace[-1]["entropy_bits"] < 0.01
    assert theory.satisfied(trace[-1]["readout"], tol=0.2)
