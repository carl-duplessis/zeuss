"""Multi-hop deduction: chain walks a relation's transitive closure honestly."""
from zeuss.qa import COHERENCE_FLOOR, chain, entails, demo_ontology
from zeuss.tier3_logic.ontology import Ontology


def test_chain_walks_transitive_closure():
    # A dedicated, low-crosstalk chain (larger dim, few facts) so the
    # substrate resolves all three hops unambiguously.
    onto = Ontology(dim=8192, seed=7)
    onto.add("socrates", "is_a", "human")
    onto.add("human", "is_a", "mortal")
    onto.add("mortal", "is_a", "thing")
    memory = onto.ground()
    c = chain(onto, memory, "socrates", "is_a")
    names = [name for name, _ in c.hops]
    assert names == ["human", "mortal", "thing"]
    # Coherence compounds: each cumulative entry is <= the previous one.
    assert all(a >= b for a, b in zip(c.cumulative, c.cumulative[1:]))
    assert all(0.0 <= v <= 1.0 for v in c.cumulative)


def test_chain_on_demo_ontology_walks_the_full_closure():
    onto = demo_ontology()
    memory = onto.ground()
    c = chain(onto, memory, "socrates", "is_a")
    names = [name for name, _ in c.hops]
    assert names == ["human", "mortal", "thing"]


def test_chain_stops_on_cycle():
    onto = Ontology(dim=2048, seed=3)
    onto.add("a", "next", "b")
    onto.add("b", "next", "a")
    memory = onto.ground()
    c = chain(onto, memory, "a", "next", max_hops=10)
    # Must terminate well before max_hops once "a" would be revisited.
    assert len(c.hops) <= 3
    names = [name for name, _ in c.hops]
    assert len(names) == len(set(names))


def test_entails_true_and_false():
    onto = demo_ontology()
    memory = onto.ground()

    holds = entails(onto, memory, "socrates", "is_a", "mortal")
    assert holds.holds
    assert holds.hops == 2
    assert holds.coherence > 0.0

    not_holds = entails(onto, memory, "socrates", "is_a", "sun")
    assert not not_holds.holds
    assert not_holds.coherence == 0.0
    assert not_holds.hops == 0

    three_hop = entails(onto, memory, "socrates", "is_a", "thing")
    assert three_hop.holds
    assert three_hop.hops == 3


def test_entails_direct_fact_is_one_hop():
    onto = demo_ontology()
    memory = onto.ground()
    verdict = entails(onto, memory, "socrates", "is_a", "human")
    assert verdict.holds
    assert verdict.hops == 1
    assert verdict.coherence >= COHERENCE_FLOOR
