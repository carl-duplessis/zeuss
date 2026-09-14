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


def test_ask_raw_candidate_names_accepts_a_set_and_matches_unscoped():
    """ask_raw's candidate_names indexes the result positionally (`names[i]`
    for the winning candidate) - the exact bug this test pins: an earlier
    version broke with `TypeError: 'set' object is not subscriptable` the
    first time this was actually exercised with a set (the natural type for
    "this shard's own entities"), not just a list. Also checks the more
    important property: scoping to a candidate set that still contains the
    true answer must not change which answer wins or its coherence."""
    dim = 8192
    n_triples = 40
    onto = Ontology(dim=dim, seed=0)
    for i in range(n_triples):
        onto.add(f"subj_{i}", "is_a", f"filler_{i % 5}")
    memory_raw = onto.ground_raw().raw

    unscoped = ask_raw(onto, memory_raw, "subj_0", "is_a")
    scoped_names = {f"subj_{i}" for i in range(n_triples)} | {f"filler_{i}" for i in range(5)}
    scoped = ask_raw(onto, memory_raw, "subj_0", "is_a", candidate_names=scoped_names)

    assert scoped.answer == unscoped.answer == "filler_0"
    assert scoped.coherence == unscoped.coherence


def test_ask_raw_candidate_names_gives_quadratic_scaling_query_cost_relief():
    """The mechanism behind docs/ROADMAP.md's "Phase 2 addendum, continued"
    N=6000 stress test: an *unscoped* per-shard raw query loop scores every
    shard against the WHOLE ontology's entities, not that shard's own - so
    per-query cost is O(shards * total_entities) = O(N^2/shard_size), not
    O(N). This doesn't re-run that full stress test (slow, real-data-style
    validation lives in ROADMAP.md instead, this project's usual pattern) -
    it just pins the *correctness* half fast: scoping to a strict subset
    that excludes the true answer must correctly fail to find it (proving
    the scoping is real, not silently falling back to the full codebook)."""
    dim = 8192
    n_triples = 40
    onto = Ontology(dim=dim, seed=0)
    for i in range(n_triples):
        onto.add(f"subj_{i}", "is_a", f"filler_{i % 5}")
    memory_raw = onto.ground_raw().raw

    # Deliberately exclude "filler_0", the true answer for subj_0.
    wrong_scope = {f"filler_{i}" for i in range(1, 5)} | {"subj_0"}
    answer = ask_raw(onto, memory_raw, "subj_0", "is_a", candidate_names=wrong_scope)
    assert answer.answer != "filler_0"
    assert answer.answer in wrong_scope


def test_ground_raw_refined_differs_from_ground_raw_after_refinement():
    """The empty corner of the ground/ground_refined/ground_raw matrix,
    now filled: the raw pipeline's generalising counterpart. Must actually
    carry the refined signal, not silently duplicate ground_raw()."""
    from zeuss.tier2_substrate.hypervectors import similarity

    onto = demo_ontology()
    onto.refine_entity_vectors()
    plain = onto.ground_raw().raw
    refined = onto.ground_raw_refined().raw
    assert similarity(plain, refined) < 0.999


def test_ground_raw_refined_equals_ground_raw_without_refinement():
    """entity_refined() falls back to entity() for anything never refined,
    so on an un-refined ontology the two grounding paths must agree
    exactly - the same safe-fallback contract entity_refined() promises."""
    from zeuss.tier2_substrate.hypervectors import similarity

    onto = demo_ontology()
    plain = onto.ground_raw().raw
    refined = onto.ground_raw_refined().raw
    assert similarity(plain, refined) > 0.9999


def test_ground_sharded_raw_splits_without_losing_triples():
    onto = demo_ontology()
    shards = onto.ground_sharded_raw(shard_size=3)
    assert len(shards) == -(-len(onto.triples) // 3)
    assert all(s.shape == (onto.dim,) for s in shards)


def test_ask_raw_entity_vectors_is_the_third_leg_of_a_generalising_query():
    """refine_entity_vectors' docstring establishes that a generalising
    query needs memory, probe AND candidate scoring to agree. ask_raw had
    subject_vector but no entity_vectors, so the raw pipeline could only
    ever do two of the three. This checks the parameter actually reaches
    candidate scoring - overriding it changes the reported coherence."""
    onto = demo_ontology()
    onto.refine_entity_vectors()
    memory = onto.ground_raw_refined().raw
    probe_only = ask_raw(onto, memory, "socrates", "is_a",
                         subject_vector=onto.entity_refined("socrates"))
    all_three = ask_raw(onto, memory, "socrates", "is_a",
                        subject_vector=onto.entity_refined("socrates"),
                        entity_vectors=onto.entity_refined)
    assert probe_only.coherence != all_three.coherence


def test_ask_raw_entity_vectors_defaults_to_an_exact_no_op():
    onto = demo_ontology()
    onto.refine_entity_vectors()
    memory = onto.ground_raw().raw
    default = ask_raw(onto, memory, "socrates", "is_a")
    explicit = ask_raw(onto, memory, "socrates", "is_a", entity_vectors=onto.entity)
    assert default.answer == explicit.answer
    assert default.coherence == explicit.coherence


def test_raw_refined_coherence_floor_sits_between_the_measured_classes():
    """Pins the calibrated constant against the distributions it was
    derived from (see RAW_REFINED_COHERENCE_FLOOR's own comment for the
    full measurement on real UMLS). It must sit above the no-answer
    negatives' bulk and below the positives' bulk - and well above the
    ordinary raw path's floor, since the generalising path runs on an
    unnormalised, much larger scale."""
    from zeuss.qa import RAW_COHERENCE_FLOOR, RAW_REFINED_COHERENCE_FLOOR

    assert RAW_REFINED_COHERENCE_FLOOR > RAW_COHERENCE_FLOOR
    # Measured UMLS maxima/minima the threshold was chosen to separate.
    assert RAW_REFINED_COHERENCE_FLOOR > 0.5799   # max nonexistent-subject negative
    assert RAW_REFINED_COHERENCE_FLOOR < 2.6080   # p05 of positives


def test_chain_sharded_raw_walks_the_same_chain_as_the_normalised_path():
    """The raw pipeline's multi-hop counterpart. At a scale where both
    paths are reliable they must agree on the walk itself - the raw path
    exists to change shard economics and capacity, not to reach different
    conclusions."""
    from zeuss.qa import chain_sharded, chain_sharded_raw

    onto = demo_ontology()
    normalised = onto.ground_sharded(shard_size=80, skip_irregular=False)
    raw = onto.ground_sharded_raw(shard_size=80)
    a = chain_sharded(onto, normalised, "socrates", "is_a")
    b = chain_sharded_raw(onto, raw, "socrates", "is_a")
    assert [n for n, _ in a.hops] == [n for n, _ in b.hops] == ["human", "mortal", "thing"]


def test_chain_sharded_raw_cumulative_grows_rather_than_decays():
    """Pins the documented wrinkle rather than leaving it to surprise a
    caller: raw coherence is not bounded by 1, so Chain's inherited
    multiplicative convention *grows* across hops here instead of decaying.
    It is retained only so `cumulative` means the same thing on every path
    (the product of per-hop coherences); min_hop_coherence() is the signal
    to actually use."""
    from zeuss.qa import chain_sharded_raw

    onto = demo_ontology()
    raw = onto.ground_sharded_raw(shard_size=80)
    c = chain_sharded_raw(onto, raw, "socrates", "is_a")
    assert len(c.hops) == 3
    assert c.cumulative[-1] > c.cumulative[0]          # grows, unlike the normalised path
    assert c.min_hop_coherence() == min(x for _n, x in c.hops)


def test_chain_sharded_raw_rejects_mismatched_shard_entities():
    from zeuss.qa import chain_sharded_raw

    onto = demo_ontology()
    raw = onto.ground_sharded_raw(shard_size=80)
    try:
        chain_sharded_raw(onto, raw, "socrates", "is_a", shard_entities=[{"socrates"}] * (len(raw) + 1))
        assert False, "expected a ValueError"
    except ValueError:
        pass


def test_chain_sharded_raw_requires_memories():
    from zeuss.qa import chain_sharded_raw

    onto = demo_ontology()
    try:
        chain_sharded_raw(onto, [], "socrates", "is_a")
        assert False, "expected a ValueError"
    except ValueError:
        pass
