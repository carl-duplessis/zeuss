"""The fix for v0.39's old, unexplained "raising `dim` doesn't rescue the
single-bundle capacity ceiling" finding (see `docs/ROADMAP.md`'s "Phase 2
addendum" and `Ontology.ground_raw`'s docstring for the full mechanism):
`bundle()`/`unbind()`'s shared per-element unit-modulus projection caps the
*expected* recovered similarity as a function of bundle size alone,
independent of `dim`. Reading from the raw, un-projected sum instead
(`Ontology.ground_raw`/`step_raw`, `qa.ask_raw`) lets `dim` do the job
standard HRR capacity theory predicts.

This file checks the fix directly, at the *exact* dim (65536) v0.39 itself
tested and found gave zero improvement for the old pipeline - so a pass
here is a real, apples-to-apples regression against that historical
finding, not a new, unrelated scale.
"""
from zeuss.qa import RAW_COHERENCE_FLOOR, ask, ask_raw, demo_ontology
from zeuss.tier3_logic.ontology import Ontology


def test_ask_raw_matches_ask_on_small_known_facts():
    """At ordinary (non-capacity-stressed) scale, ask_raw should recover
    exactly what ask() recovers - the raw pipeline is a different read-out
    of the same underlying algebra, not a different answer."""
    onto = demo_ontology()
    memory = onto.ground()
    memory_raw = onto.ground_raw().raw
    for subject, relation in [
        ("socrates", "is_a"),
        ("plato", "is_a"),
        ("sky", "has_color"),
        ("sun", "is_a"),
    ]:
        ordinary = ask(onto, memory, subject, relation)
        raw = ask_raw(onto, memory_raw, subject, relation)
        assert raw.answer == ordinary.answer
        assert raw.known
        # Raw coherence concentrates near 1.0 for a correct candidate,
        # regardless of scale - not comparable in magnitude to the
        # ordinary pipeline's own (much smaller, N-dependent) coherence.
        assert raw.coherence > 0.9


def test_ask_raw_flags_unknown_queries_as_guesses():
    onto = demo_ontology()
    memory_raw = onto.ground_raw().raw
    for subject, relation in [
        ("dragon", "is_a"),
        ("socrates", "has_color"),
    ]:
        answer = ask_raw(onto, memory_raw, subject, relation)
        assert not answer.known
        assert answer.coherence < RAW_COHERENCE_FLOOR


def test_ask_raw_rescues_the_v039_ceiling_at_v039s_own_tested_dim():
    """v0.39 measured, on a 400-triple single bundle: raising dim 8192 ->
    65536 (8x) barely moved coherence (0.044 -> 0.039, if anything
    slightly worse) - dim did not rescue the ceiling. This recreates that
    exact scenario (400 triples, dim=65536) through the new raw pipeline
    instead, and checks it actually works: every sampled fact recovered
    correctly *and* reported `known`, with a comfortable margin over a
    genuinely-absent entity's score (measured margin at this exact dim/N
    point: 4.76x - see RAW_COHERENCE_FLOOR's own comment)."""
    dim = 65536
    n_triples = 400
    onto = Ontology(dim=dim, seed=0)
    for i in range(n_triples):
        onto.add(f"subj_{i}", "is_a", f"filler_{i % 5}")
    memory_raw = onto.ground_raw().raw

    for i in range(0, n_triples, 40):  # sample every 40th - 10 checks, keeps this fast
        subject = f"subj_{i}"
        expected = f"filler_{i % 5}"
        answer = ask_raw(onto, memory_raw, subject, "is_a")
        assert answer.answer == expected
        assert answer.known
        assert answer.coherence > 0.9

    for i in range(3):
        answer = ask_raw(onto, memory_raw, f"nonexistent_entity_{i}", "is_a")
        assert not answer.known
        assert answer.coherence < 0.4  # measured worst case at this dim/N: 0.198
