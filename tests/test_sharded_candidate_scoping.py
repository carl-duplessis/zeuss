"""Scoping ask_sharded's candidate comparison to each shard's own entities
instead of the whole ontology - found while investigating whether ask_raw
(the v0.39 dim-ceiling fix) obsoletes sharding. It doesn't obsolete it, but
isolating the comparison found ask_sharded was paying for an unforced
inefficiency: every shard's `ask` call scored against the *entire*
ontology's entity codebook regardless of that shard's own (much smaller)
size. Measured directly at 1600 triples/20 shards/~1605 entities: ~48s per
query unscoped, ~3s per query scoped to each shard's own ~85 entities -
identical dimensional_collapse/settle machinery, identical accuracy and
false-positive rate (see docs/ROADMAP.md's "Phase 2 addendum, continued").

This file checks the fix at a small, fast scale: correctness/robustness
equivalence between scoped and unscoped (the property that matters more
than the timing number itself, which isn't pinned here - fragile across
machines), plus the alignment guarantee `ground_shards_with_entities`
promises when a shard gets dropped for being structurally irregular.
"""
from zeuss.qa import ask_sharded
from zeuss.tier2_substrate.hypervectors import similarity
from zeuss.tier3_logic.ontology import Ontology


def _build(n_subjects=200, n_fillers=5, dim=8192, seed=0):
    onto = Ontology(dim=dim, seed=seed)
    for i in range(n_subjects):
        onto.add(f"subj_{i}", "is_a", f"filler_{i % n_fillers}")
    return onto


def test_ground_sharded_with_entities_matches_ground_sharded():
    onto = _build()
    memories_plain = onto.ground_sharded(shard_size=40, skip_irregular=False)
    memories_with_ent, entity_sets = onto.ground_sharded_with_entities(shard_size=40, skip_irregular=False)
    assert len(memories_with_ent) == len(memories_plain) == len(entity_sets)
    for plain, with_ent in zip(memories_plain, memories_with_ent):
        assert similarity(plain, with_ent) > 0.9999
    for i, names in enumerate(entity_sets):
        shard_triples = onto.triples[i * 40 : (i + 1) * 40]
        expected = {s for s, _, _ in shard_triples} | {o for _, _, o in shard_triples}
        assert names == expected


def test_ground_shards_with_entities_stays_aligned_when_a_shard_is_dropped():
    """The exact risk this method's docstring warns about: computing entity
    names separately from ground_shards' own filtering could desync if a
    shard gets dropped for being structurally irregular. This checks the
    combined method never lets that happen."""
    good_shard_1 = [("a", "knows", "b"), ("c", "knows", "d")]
    irregular_shard = [("socrates", "is_a", "human"), ("socrates", "is_a", "star")]  # contradiction
    good_shard_2 = [("e", "knows", "f")]
    onto = Ontology(dim=2048, seed=0)
    for shard in (good_shard_1, irregular_shard, good_shard_2):
        for t in shard:
            onto.add(*t)

    memories, entity_sets = onto.ground_shards_with_entities(
        [good_shard_1, irregular_shard, good_shard_2], skip_irregular=True
    )
    assert len(memories) == len(entity_sets) == 2  # the irregular shard was dropped from both
    assert entity_sets[0] == {"a", "b", "c", "d"}
    assert entity_sets[1] == {"e", "f"}


def test_ask_sharded_with_shard_entities_matches_unscoped_answers():
    onto = _build()
    memories, shard_entities = onto.ground_sharded_with_entities(shard_size=40, skip_irregular=False)

    for i in range(0, 200, 40):
        subject = f"subj_{i}"
        expected = f"filler_{i % 5}"
        unscoped = ask_sharded(onto, memories, subject, "is_a")
        scoped = ask_sharded(onto, memories, subject, "is_a", shard_entities=shard_entities)
        assert unscoped.answer == expected
        assert scoped.answer == expected
        assert unscoped.known and scoped.known


def test_ask_sharded_with_shard_entities_still_flags_unknown_queries():
    onto = _build()
    memories, shard_entities = onto.ground_sharded_with_entities(shard_size=40, skip_irregular=False)
    for subject in ["nonexistent_entity_1", "nonexistent_entity_2"]:
        scoped = ask_sharded(onto, memories, subject, "is_a", shard_entities=shard_entities)
        assert not scoped.known


def test_ask_sharded_rejects_mismatched_shard_entities_length():
    onto = _build()
    memories = onto.ground_sharded(shard_size=40, skip_irregular=False)
    try:
        ask_sharded(onto, memories, "subj_0", "is_a", shard_entities=[{"subj_0"}])
        assert False, "expected a ValueError"
    except ValueError:
        pass
