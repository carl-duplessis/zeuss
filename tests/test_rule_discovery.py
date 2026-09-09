"""Bridging tier4 boolean synthesis into tier3 Rule/Theory objects - rules
that are discovered from example data rather than hand-typed."""
import numpy as np
import pytest

from zeuss.tier2_substrate.hypervectors import Codebook
from zeuss.tier3_logic.compiler import Rule, Theory
from zeuss.tier3_logic.grounding import readout
from zeuss.tier3_logic.rule_discovery import (
    DiscoveredRule,
    compile_discovered_theory,
    discover_antecedent,
    discover_rule,
    expand_valuation,
)
from zeuss.tier4_synthesis.dsl import evaluate, Fuel


def _or_examples():
    examples = []
    for rain in (False, True):
        for sprinkler in (False, True):
            examples.append(({"rain": rain, "sprinkler": sprinkler}, rain or sprinkler))
    return examples


def _corner_set(landscape, codebook, variables):
    out = []
    for vector in landscape.attractors:
        values = readout(codebook, variables, vector)
        out.append(tuple(1 if values[name] > 0.5 else 0 for name in variables))
    return sorted(set(out))


def test_discover_antecedent_recovers_a_known_or_formula():
    """The core claim: given examples generated from a *known* hidden rule
    (`wet <- rain or sprinkler`), `discover_antecedent` finds a genuinely
    correct, verified antecedent - not just "some formula that happens to
    fit these four rows", checked directly against every row."""
    node, verified = discover_antecedent(["rain", "sprinkler"], _or_examples(), rng=np.random.default_rng(0))
    assert verified
    for inputs, expected in _or_examples():
        result = evaluate(node, dict(inputs), Fuel(50))
        assert result == expected


def test_discover_antecedent_recovers_a_known_and_formula():
    """Same claim, AND-shaped hidden rule - confirms this isn't tuned to
    just the OR case."""
    examples = []
    for a in (False, True):
        for b in (False, True):
            examples.append(({"a": a, "b": b}, a and b))
    node, verified = discover_antecedent(["a", "b"], examples, rng=np.random.default_rng(1))
    assert verified
    for inputs, expected in examples:
        assert evaluate(node, dict(inputs), Fuel(50)) == expected


def test_discover_rule_names_the_synthetic_antecedent_after_the_consequent():
    """The synthetic antecedent name is deterministic and consequent-scoped
    (`_discovered_<consequent>_antecedent`) - so two discovered rules for
    different consequents never collide, and the same discovery for the
    same consequent is always addressable by the same name."""
    dr = discover_rule(["rain", "sprinkler"], _or_examples(), consequent="wet", rng=np.random.default_rng(0))
    assert isinstance(dr, DiscoveredRule)
    assert dr.verified
    assert dr.rule.antecedent == "_discovered_wet_antecedent"
    assert dr.rule.consequent == "wet"
    assert dr.rule.antecedent in dr.derived


def test_expand_valuation_computes_the_derived_antecedent_correctly():
    """`expand_valuation` must reproduce the hidden rule's actual truth
    value at every corner, not just at the training rows - checked directly
    against the ground-truth OR formula, including a fractional
    (non-corner) valuation rounded before evaluation."""
    dr = discover_rule(["rain", "sprinkler"], _or_examples(), consequent="wet", rng=np.random.default_rng(0))
    for rain, sprinkler in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]:
        expanded = expand_valuation(dr.derived, {"rain": rain, "sprinkler": sprinkler})
        expected = 1.0 if (rain or sprinkler) else 0.0
        assert expanded[dr.rule.antecedent] == expected

    # A fractional valuation rounds to booleans before evaluation.
    expanded = expand_valuation(dr.derived, {"rain": 0.9, "sprinkler": 0.1})
    assert expanded[dr.rule.antecedent] == 1.0  # rounds to rain=True -> antecedent True


def test_expand_valuation_is_a_noop_with_no_derived_rules():
    assert expand_valuation({}, {"a": 1.0, "b": 0.0}) == {"a": 1.0, "b": 0.0}


def test_compile_discovered_theory_registers_only_logically_consistent_corners():
    """The actual deliverable: compiling a discovered rule into a Landscape
    at a temperature sharp enough to isolate satisfying corners must
    produce *only* corners consistent with the discovered antecedent =>
    consequent implication (antecedent True forces consequent True;
    antecedent False leaves it unconstrained) - the concrete, checkable
    version of "the discovered rule genuinely constrains the compiled
    energy landscape", not just "a landscape got built"."""
    dr = discover_rule(
        ["rain", "sprinkler"], _or_examples(), consequent="wet", weight=10.0, rng=np.random.default_rng(0)
    )
    codebook = Codebook(dim=4096, seed=7)
    landscape = compile_discovered_theory(codebook, [dr], ["rain", "sprinkler", "wet"], inverse_temperature=20.0)

    assert len(landscape.attractors) > 0
    for corner in _corner_set(landscape, codebook, ["rain", "sprinkler", "wet"]):
        rain, sprinkler, wet = corner
        antecedent_true = bool(rain) or bool(sprinkler)
        assert (not antecedent_true) or bool(wet), f"inconsistent corner: {corner}"
    # The one violating corner (rain or sprinkler True, wet False) must not survive.
    assert (1, 0, 0) not in _corner_set(landscape, codebook, ["rain", "sprinkler", "wet"])
    assert (0, 1, 0) not in _corner_set(landscape, codebook, ["rain", "sprinkler", "wet"])


def test_compile_discovered_theory_rejects_unverified_rules():
    """Compiling an unverified discovered rule would silently promote a
    guess to an axiom - the opposite of this project's honesty conventions
    (see `discover_antecedent`'s docstring). Checked directly with a
    hand-built unverified `DiscoveredRule` rather than trying to force a
    real discovery to fail (which would be seed-fragile)."""
    fake_unverified = DiscoveredRule(
        rule=Rule(antecedent="_discovered_x_antecedent", consequent="x", weight=1.0),
        derived={},
        verified=False,
    )
    codebook = Codebook(dim=256, seed=0)
    with pytest.raises(ValueError):
        compile_discovered_theory(codebook, [fake_unverified], ["x"])
