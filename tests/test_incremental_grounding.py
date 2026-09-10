"""Phase 1: incremental grounding - every `ground*` method on `Ontology`
rebuilds its whole bundle from scratch on every call, with no way to
cheaply add one new fact to an already-large knowledge base. That's a
real operational gap: it means Zeuss can only be used as a static,
batch-compiled snapshot, never as a living memory that absorbs new facts
over time.

`IncrementalMemory`/`IncrementalShardedMemory` close it: `bundle()`'s
final `normalize()` step is *per-element* phase projection, not a single
global magnitude, so a new fact can't be cheaply folded into an already-
normalised bundle - but the raw, pre-normalisation complex sum can be
added to trivially. These classes keep that raw sum as their actual
state and only normalise on read, making `add()` O(dim) regardless of
how many facts already exist, instead of O(n) triples re-processed."""
from zeuss.qa import ask, ask_sharded
from zeuss.tier2_substrate.hypervectors import bundle, similarity
from zeuss.tier3_logic.ontology import Ontology


def test_incremental_memory_matches_batch_bundle_exactly():
    """The critical correctness check, verified directly rather than
    assumed from the algebra: after the same sequence of adds,
    IncrementalMemory.vector must be numerically equivalent to what
    bundle() computes in one batch call on the identical vectors."""
    onto = Ontology(dim=512, seed=0)
    mem = onto.incremental_memory()
    for s, r, o in [("a", "knows", "b"), ("b", "knows", "c"), ("c", "knows", "a"), ("a", "likes", "pizza")]:
        onto.add_incremental(mem, s, r, o)

    batch = bundle([onto._triple_vector(*t) for t in onto.triples])
    assert similarity(mem.vector, batch) > 0.9999999


def test_incremental_memory_is_queryable_via_ask():
    """Real-API smoke test: an incrementally-built memory must work with
    qa.ask() exactly like a batch-ground()ed one - no special-casing
    needed downstream."""
    onto = Ontology(dim=2048, seed=3)
    mem = onto.incremental_memory()
    onto.add_incremental(mem, "socrates", "is_a", "human")
    onto.add_incremental(mem, "socrates", "walks_on", "earth")

    answer = ask(onto, mem.vector, "socrates", "is_a")
    assert answer.answer == "human"
    assert answer.known


def test_add_incremental_returns_the_memory_for_chaining():
    onto = Ontology(dim=64, seed=0)
    mem = onto.incremental_memory()
    result = onto.add_incremental(mem, "a", "r", "b")
    assert result is mem


def test_incremental_sharded_memory_starts_a_new_shard_at_the_boundary():
    onto = Ontology(dim=128, seed=0)
    smem = onto.incremental_sharded_memory(shard_size=2)
    for i in range(5):
        onto.add_incremental(smem, f"e{i}", "r", f"e{i+1}")
    assert len(smem.memories) == 3  # 2 + 2 + 1
    assert len(smem) == 5


def test_incremental_sharded_memory_matches_ground_sharded_batch_equivalent():
    """Same correctness check as the single-memory case, extended to the
    sharded structure: adding facts in the same order incrementally must
    produce shards numerically identical to Ontology.ground_sharded's
    batch result at the same shard_size - not just "close", exact."""
    facts = [(f"e{i}", "r", f"e{i+1}") for i in range(10)]

    onto_inc = Ontology(dim=512, seed=7)
    smem = onto_inc.incremental_sharded_memory(shard_size=3)
    for s, r, o in facts:
        onto_inc.add_incremental(smem, s, r, o)

    onto_batch = Ontology(dim=512, seed=7)
    for s, r, o in facts:
        onto_batch.add(s, r, o)
    batch_shards = onto_batch.ground_sharded(shard_size=3, skip_irregular=False)

    assert len(smem.memories) == len(batch_shards)
    for inc_vector, batch_vector in zip(smem.memories, batch_shards):
        assert similarity(inc_vector, batch_vector) > 0.9999999


def test_incremental_sharded_memory_is_queryable_via_ask_sharded():
    onto = Ontology(dim=8192, seed=1)
    smem = onto.incremental_sharded_memory(shard_size=80)
    for i in range(100):
        onto.add_incremental(smem, f"human{i}", "is_a", "human")
        onto.add_incremental(smem, f"human{i}", "walks_on", "earth")

    correct = sum(
        1
        for i in range(10)
        if (a := ask_sharded(onto, smem.memories, f"human{i}", "is_a")).answer == "human" and a.known
    )
    assert correct >= 8  # matches ground_sharded's own measured reliability at this scale


def test_incremental_memory_requires_no_rebuild_to_add_a_fact_after_grounding():
    """The actual operational point of this whole mechanism: query a
    memory, add a new fact, query again - the second answer must reflect
    the new fact without ever calling ground() again."""
    onto = Ontology(dim=2048, seed=2)
    mem = onto.incremental_memory()
    onto.add_incremental(mem, "socrates", "is_a", "human")

    before = ask(onto, mem.vector, "socrates", "walks_on")
    assert not before.known  # genuinely not told yet

    onto.add_incremental(mem, "socrates", "walks_on", "earth")
    after = ask(onto, mem.vector, "socrates", "walks_on")
    assert after.answer == "earth"
    assert after.known
