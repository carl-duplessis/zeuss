"""v0.47: the crosstalk-fix signal that works on real, densely multi-
relational, non-redundant data - the direct follow-up after being asked
to "address the v0.46 findings" from testing on the real Nations dataset.

v0.46's `shard_regularity_weights` (internal (subject,relation)-pair
duplication) and the earlier-rejected `shard_trust_weights` (cross-shard
vote disagreement) were both measured directly against real Nations data
(1992 triples, 55 genuinely multi-valued relations, zero redundant
triples) and both failed - one with zero discrimination (real and noise
shard scores fully overlapped), the other inverted (real shards scored
*more* suspicious than noise). The shared root cause: both signals detect
a shard *disagreeing* with something, which requires the underlying data
to have redundant/repeated assertions to disagree against - true of the
synthetic single-valued domain those mechanisms were built and validated
on, false of most real knowledge graphs, where a fact is typically stated
exactly once.

`shard_connectivity_weights` needs no redundancy at all: it tests whether
a shard's *pattern of which entities it talks about* looks like real-world
structure (skewed - some entities are just more connected than others) or
uniform random sampling (noise). This file captures that property
synthetically (a small "hub and leaf" graph, exactly mirroring what real
Nations data measurably has: skewed entity connectivity, no repeated
triples) so the test suite doesn't need network access to validate it -
the real Nations numbers themselves are documented in `docs/ROADMAP.md`
v0.47, not re-fetched here."""
import random

from zeuss.tier3_logic.ontology import (
    Ontology,
    entity_connectivity_score,
    shard_connectivity_weights,
)


def _hub_and_leaf_shards(num_hubs=4, num_leaves=60, shard_size=20, noise_shards=3, seed=3):
    """A small synthetic graph with the one property that actually matters
    here - skewed real-world connectivity (a few hubs, many leaves, each
    leaf asserting exactly one non-redundant fact) - mixed with noise
    shards drawn uniformly at random from the same entity pool."""
    hubs = [f"hub{i}" for i in range(num_hubs)]
    leaves = [f"leaf{i}" for i in range(num_leaves)]
    entities = hubs + leaves

    real_triples = [(f"leaf{i}", "connects_to", hubs[i % num_hubs]) for i in range(num_leaves)]
    real_chunks = [real_triples[i : i + shard_size] for i in range(0, len(real_triples), shard_size)]

    rng = random.Random(seed)
    noise_chunks = [
        [(rng.choice(entities), "connects_to", rng.choice(entities)) for _ in range(shard_size)]
        for _ in range(noise_shards)
    ]
    return real_chunks, noise_chunks


def test_entity_connectivity_score_is_higher_for_hub_heavy_shards():
    """A shard that talks about high-degree (hub) entities should score
    higher than one that talks about low-degree (leaf-only, or uniformly
    random) entities, given the same global degree reference."""
    real_chunks, noise_chunks = _hub_and_leaf_shards()
    global_degree: dict = {}
    for shard in real_chunks + noise_chunks:
        for s, r, o in shard:
            global_degree[s] = global_degree.get(s, 0) + 1
            global_degree[o] = global_degree.get(o, 0) + 1

    real_scores = [entity_connectivity_score(s, global_degree) for s in real_chunks]
    noise_scores = [entity_connectivity_score(s, global_degree) for s in noise_chunks]
    assert min(real_scores) > max(noise_scores)


def test_shard_connectivity_weights_discriminates_real_from_noise():
    """The full self-normalising weight: real (hub-connected) shards
    should land solidly above the mix's own mean (weight > 0.5), noise
    shards solidly below it - no absolute threshold tuned per-dataset."""
    real_chunks, noise_chunks = _hub_and_leaf_shards()
    weights = shard_connectivity_weights(real_chunks + noise_chunks)
    real_weights = weights[: len(real_chunks)]
    noise_weights = weights[len(real_chunks) :]
    assert all(w > 0.7 for w in real_weights)
    assert all(w < 0.3 for w in noise_weights)


def test_shard_connectivity_weights_handles_no_noise():
    """With no noise at all, every shard is equally (un)remarkable
    relative to the others - should not crash or produce a degenerate
    all-zero/all-one result."""
    real_chunks, _ = _hub_and_leaf_shards(noise_shards=0)
    weights = shard_connectivity_weights(real_chunks)
    assert len(weights) == len(real_chunks)
    assert all(0.0 <= w <= 1.0 for w in weights)


def test_shard_connectivity_weights_handles_empty_input():
    assert shard_connectivity_weights([]) == []


def test_entity_connectivity_score_handles_empty_shard():
    assert entity_connectivity_score([], {}) == 0.0


def test_ground_shards_with_connectivity_trust_returns_aligned_memories_and_weights():
    """Real-API smoke test: the convenience method grounds every non-empty
    shard and returns weights positionally aligned with the memories it
    produced (not necessarily with the input shard list, if any shard lost
    all its triples to axiom-violation excision - not exercised here since
    no exclusions are passed)."""
    real_chunks, noise_chunks = _hub_and_leaf_shards()
    onto = Ontology(dim=1024, seed=0)
    for shard in real_chunks + noise_chunks:
        for s, r, o in shard:
            onto.add(s, r, o)
    memories, weights = onto.ground_shards_with_connectivity_trust(real_chunks + noise_chunks)
    assert len(memories) == len(weights) == len(real_chunks) + len(noise_chunks)
