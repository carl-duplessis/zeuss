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


def test_chain_resonance_coherence_is_high_for_a_clean_transitive_chain():
    """`Chain.resonance_coherence` (see `qa.py`'s v0.34 docstring) composes
    the wave operator straight through, without the per-hop collapse-and-
    reinject `chain()` itself does, and checks how strongly that raw
    composition still resonates with the same final entity. On a
    dedicated, low-crosstalk ontology, a genuinely clean 3-hop chain must
    stay well above the noise floor - not a claim of near-1.0 certainty
    (composing three real-valued wave operations without any intermediate
    error-correction accumulates real dispersion), but a real, positive
    signal clearly distinguishable from the near-zero this project's own
    `test_stored_and_guessed_coherence_are_well_separated`-style crosstalk
    floor represents."""
    onto = Ontology(dim=8192, seed=7)
    onto.add("socrates", "is_a", "human")
    onto.add("human", "is_a", "mortal")
    onto.add("mortal", "is_a", "thing")
    memory = onto.ground()
    c = chain(onto, memory, "socrates", "is_a")
    assert len(c.hops) == 3
    assert c.resonance_coherence > 0.3


def test_chain_resonance_coherence_is_zero_with_no_hops():
    onto = demo_ontology()
    memory = onto.ground()
    c = chain(onto, memory, "dragon", "is_a")  # unknown subject -> no hops
    assert c.hops == []
    assert c.resonance_coherence == 0.0


def test_chain_resonance_coherence_carries_information_cumulative_does_not():
    """Not a cosmetic duplicate of the existing per-hop-multiplied
    `cumulative` product - a genuinely independent signal, checked directly
    rather than assumed: on the demo ontology's own 3-hop chain, the two
    numbers are not close to each other at all (the compounded per-hop
    product decays sharply over three hops; the whole-chain resonance
    check does not)."""
    onto = demo_ontology()
    memory = onto.ground()
    c = chain(onto, memory, "socrates", "is_a")
    assert len(c.hops) == 3
    assert abs(c.resonance_coherence - c.cumulative[-1]) > 0.3
