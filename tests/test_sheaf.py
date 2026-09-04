"""Sheaf cohomology over a small consistency graph: real rank-nullity, not a heuristic.

Honesty note (see sheaf.py's module docstring): for this homogeneous
scalar-stalk construction, a frustrated (sign-flipped) cycle is *full rank*
(H^1 == 0) and instead collapses H^0 to {0} (only the trivial section
survives) - the opposite of the naive "H^1 != 0 means contradiction"
intuition. Whether *specific* observed data disagrees is answered directly by
`local_section`/`is_consistent_with`, not by H^1.
"""
from zeuss.tier3_logic.compiler import Rule, Theory
from zeuss.tier3_logic.sheaf import SheafGraph, from_theories


def test_tree_has_one_free_parameter_and_no_redundancy():
    graph = SheafGraph().add_edge("a", "b", 1.0, 1.0)
    assert graph.h0_dimension() == 1  # a=b, one free real parameter
    assert graph.h1_dimension() == 0  # no cycle at all


def test_consistent_cycle_has_nontrivial_section_and_redundant_h1():
    graph = (
        SheafGraph()
        .add_edge("a", "b", 1.0, 1.0)
        .add_edge("b", "c", 1.0, 1.0)
        .add_edge("a", "c", 1.0, 1.0)
    )
    assert graph.h0_dimension() == 1  # a=b=c, still one free parameter
    assert graph.h1_dimension() == 1  # the third edge is redundant (implied by the other two)
    assert not graph.has_only_trivial_section()


def test_frustrated_cycle_has_only_the_trivial_section():
    graph = (
        SheafGraph()
        .add_edge("a", "b", 1.0, 1.0)
        .add_edge("b", "c", 1.0, 1.0)
        .add_edge("a", "c", 1.0, -1.0)  # sign flip: a = -c, not a = c
    )
    assert graph.h0_dimension() == 0  # only a=b=c=0 satisfies every edge
    assert graph.h1_dimension() == 0  # the sign flip makes every edge independent (full rank)
    assert graph.has_only_trivial_section()


def test_local_section_localizes_violated_edge():
    graph = SheafGraph().add_edge("a", "b", 1.0, 1.0).add_edge("b", "c", 1.0, 1.0)
    agreeing = {"a": 1.0, "b": 1.0, "c": 1.0}
    disagreeing = {"a": 1.0, "b": 1.0, "c": 0.5}
    assert graph.is_consistent_with(agreeing)
    assert not graph.is_consistent_with(disagreeing)
    residual = graph.local_section(disagreeing)
    assert abs(residual[0]) < 1e-9   # a=b edge: still satisfied
    assert abs(residual[1]) > 1e-9   # b=c edge: violated


def test_from_theories_detects_conflicting_agent_conclusions():
    # alice's rules force "shared" toward 1 (TRUE -> shared); bob's rules
    # force it toward 0 (shared -> ZERO, ZERO defaulting to 0.0). Each is
    # individually satisfiable in isolation.
    alice = Theory(rules=[Rule("TRUE", "shared", weight=10.0)])
    bob = Theory(rules=[Rule("shared", "ZERO", weight=10.0)])
    shared_vars = {"alice": ["shared"], "bob": ["shared"]}

    graph = from_theories({"alice": alice, "bob": bob}, shared_vars)

    assert alice.satisfied({"shared": graph.local_values["alice:shared"], "TRUE": 1.0})
    assert bob.satisfied({"shared": graph.local_values["bob:shared"]})
    assert graph.local_values["alice:shared"] != graph.local_values["bob:shared"]
    assert not graph.is_consistent_with()


def test_from_theories_agrees_when_conclusions_actually_match():
    alice = Theory(rules=[Rule("TRUE", "shared", weight=10.0)])
    bob = Theory(rules=[Rule("TRUE", "shared", weight=10.0)])  # same bias as alice
    shared_vars = {"alice": ["shared"], "bob": ["shared"]}

    graph = from_theories({"alice": alice, "bob": bob}, shared_vars)
    assert graph.is_consistent_with()
