from zeuss.tier3_logic.compiler import (
    Rule,
    Theory,
    godel_tnorm,
    lukasiewicz_implies,
    lukasiewicz_tnorm,
    product_tnorm,
)


def test_tnorm_boundary_conditions():
    for tnorm in (lukasiewicz_tnorm, godel_tnorm, product_tnorm):
        assert tnorm(1.0, 1.0) == 1.0
        assert tnorm(1.0, 0.0) == 0.0
        assert tnorm(0.0, 0.0) == 0.0


def test_lukasiewicz_implication_true_when_consequent_holds():
    assert lukasiewicz_implies(1.0, 1.0) == 1.0
    assert lukasiewicz_implies(1.0, 0.0) == 0.0


def test_theory_energy_zero_when_satisfied():
    theory = Theory([Rule("rain", "wet", weight=1.0)])
    assert theory.satisfied({"rain": 1.0, "wet": 1.0})
    assert theory.energy({"rain": 1.0, "wet": 0.0}) > 0.9
