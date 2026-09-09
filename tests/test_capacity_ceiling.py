"""Characterizes a real capacity ceiling in Ontology.ground()'s single-
bundle design (see docs/ROADMAP.md v0.39) - not a bug this session
introduced, but a pre-existing substrate limit nothing had ever measured
because every prior test/demo stayed well under it. These tests lock the
measured numbers in so a future change to Ontology/hypervectors.bundle can
be checked against them, rather than letting the finding silently rot in a
docs file no test ever re-verifies."""
from zeuss.qa import ask
from zeuss.tier3_logic.ontology import Ontology


def _bundled_ontology(n_per_class: int, dim: int = 8192, seed: int = 1) -> Ontology:
    onto = Ontology(dim=dim, seed=seed)
    for i in range(n_per_class):
        onto.add(f"human{i}", "is_a", "human")
        onto.add(f"human{i}", "walks_on", "earth")
    for i in range(n_per_class):
        onto.add(f"star{i}", "is_a", "star")
        onto.add(f"star{i}", "orbits", "galaxy")
    return onto


def _correct_and_known_count(onto: Ontology, memory, n: int = 5) -> int:
    return sum(
        1 for i in range(n) if (a := ask(onto, memory, f"human{i}", "is_a")).answer == "human" and a.known
    )


def test_bundling_is_reliable_at_80_triples():
    """At the scale every other test in this project actually operates at
    (<=40 triples in the largest prior test ontology), recovery of a
    directly-stored, unambiguous fact is fully reliable."""
    onto = _bundled_ontology(n_per_class=20)
    memory = onto.ground()
    assert len(onto.triples) == 80
    assert _correct_and_known_count(onto, memory) == 5


def test_bundling_degrades_sharply_by_120_triples():
    """The measured ceiling: a 1.5x increase in bundled triples (80 -> 120)
    is enough to collapse recovery from perfect to mostly-broken - a sharp
    transition, not a gradual one. This is the substrate's own limit, not
    anything v0.34-v0.38 added; it just had never been measured before."""
    onto = _bundled_ontology(n_per_class=30)
    memory = onto.ground()
    assert len(onto.triples) == 120
    assert _correct_and_known_count(onto, memory) <= 2


def test_bundling_essentially_fails_at_400_triples():
    """Well past the ceiling: a genuinely stored, unambiguous fact reads as
    'unknown' (below COHERENCE_FLOOR) even though nothing about the query
    itself is ambiguous - the substrate has run out of capacity, not
    encountered real uncertainty."""
    onto = _bundled_ontology(n_per_class=100)
    memory = onto.ground()
    assert len(onto.triples) == 400
    assert _correct_and_known_count(onto, memory) == 0


def test_raising_dim_does_not_fix_the_ceiling():
    """The surprising half of the v0.39 finding: the obvious fix (more
    dimensions) does not clearly help. Checked directly at 8x the
    dimension on the same 400-triple case - coherence for a genuinely
    stored fact does not meaningfully improve."""
    small = _bundled_ontology(n_per_class=100, dim=8192)
    large = _bundled_ontology(n_per_class=100, dim=65536)
    coherence_small = ask(small, small.ground(), "human0", "is_a").coherence
    coherence_large = ask(large, large.ground(), "human0", "is_a").coherence
    assert coherence_small < 0.1
    assert coherence_large < 0.1  # not meaningfully rescued by 8x the dimension
