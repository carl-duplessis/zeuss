"""v0.46: fixing the crosstalk limitation v0.44/v0.45 disclosed - the
direct follow-up to being asked for three fixes for it, one abstract, then
told to implement the abstract one (continuous, gate-dissolving
trust-weighted bundling) and fall back to two more concrete ones combined
if it didn't work.

It didn't work, and this file locks in *why*, not just the fact: a first
attempt (`shard_trust_weights`, still in `ontology.py` as a documented
negative result) tried turning `classify_multi_valued_relations`'s hard
gate into a continuous per-shard weight derived from cross-shard vote
disagreement. Measured directly on the exact 50/50 real/garbage-shard
scenario that motivated this: every shard - real and noise alike - came
back with zero disagreement energy and therefore weight 1.0. The reason:
with a small shared entity pool and heavy contamination, most real-vs-
noise collisions on a given `(subject, relation)` pair land as an exact
1-vs-1 tie, which is deliberately treated as neutral (not evidence either
way) so genuine multi-valued plurality (`has_friend`-like) isn't punished
- but that protection makes a real+noise tie structurally indistinguishable
from two genuine friends, at the level of one pair's own vote count.

The fix that actually works (`internal_collision_energy`/
`shard_regularity_weights`) sidesteps the ambiguity instead of trying to
resolve it: it never looks at cross-shard votes at all, only whether a
single shard duplicates one of its own `(subject, relation)` pairs
*within itself* - a signal immune to both contamination volume (what
broke v0.43's classifier) and cross-shard tie ambiguity (what broke the
first attempt), because it never looks past one shard's own boundary."""
import random

from zeuss.qa import ask_sharded
from zeuss.tier3_logic.ontology import (
    Ontology,
    internal_collision_energy,
    shard_regularity_weights,
    shard_trust_weights,
)


def _bundled_ontology(n_per_class: int, dim: int = 8192, seed: int = 1) -> Ontology:
    onto = Ontology(dim=dim, seed=seed)
    for i in range(n_per_class):
        onto.add(f"human{i}", "is_a", "human")
        onto.add(f"human{i}", "walks_on", "earth")
    for i in range(n_per_class):
        onto.add(f"star{i}", "is_a", "star")
        onto.add(f"star{i}", "orbits", "galaxy")
    return onto


def _random_noise_chunks(onto: Ontology, k: int, shard_size: int = 80, seed: int = 99) -> list:
    rng = random.Random(seed)
    entities = onto.entity_names()
    relations = sorted({r for _, r, _ in onto.triples})
    return [
        [(rng.choice(entities), rng.choice(relations), rng.choice(entities)) for _ in range(shard_size)]
        for _ in range(k)
    ]


def test_internal_collision_energy_is_zero_for_real_structured_data():
    """Real Ontology data asserts each (subject, relation) pair once - no
    internal duplication, by construction, regardless of shard size."""
    onto = _bundled_ontology(n_per_class=40)
    shard = onto.triples[:80]
    assert internal_collision_energy(shard) == 0.0


def test_internal_collision_energy_is_positive_for_random_noise():
    """Random (entity, relation, entity) sampling with replacement from a
    finite pool collides with itself at a measurable rate - the
    birthday-paradox effect this signal is built to detect."""
    onto = _bundled_ontology(n_per_class=100)
    (noise_shard,) = _random_noise_chunks(onto, k=1)
    energy = internal_collision_energy(noise_shard)
    assert 0.0 < energy < 0.3  # measured range at this pool/shard size: ~0.04-0.10


def test_internal_collision_energy_handles_empty_shard():
    assert internal_collision_energy([]) == 0.0


def test_shard_trust_weights_first_attempt_fails_to_discriminate_at_50_50():
    """The documented negative result: at the exact 50/50 contamination
    scenario, the cross-shard-vote-based weighting (v0.46's first, pure
    continuous-relaxation attempt) assigns every shard - real and noise
    alike - the same weight, because most collisions land as neutral
    ties. This is not a strawman: it's the honest reason a second,
    different signal was needed."""
    onto = _bundled_ontology(n_per_class=100)
    real_chunks = [onto.triples[i : i + 80] for i in range(0, len(onto.triples), 80)]
    noise_chunks = _random_noise_chunks(onto, k=5)  # 50% contamination

    weights = shard_trust_weights(real_chunks + noise_chunks)
    assert weights == [1.0] * len(weights)  # zero discrimination, not just weak discrimination


def test_shard_regularity_weights_discriminates_real_from_noise_at_50_50():
    """The signal that actually works: real shards get weight 1.0 (zero
    internal collisions), noise shards get a measurably smaller weight -
    checked directly on the same scenario the first attempt failed on."""
    onto = _bundled_ontology(n_per_class=100)
    real_chunks = [onto.triples[i : i + 80] for i in range(0, len(onto.triples), 80)]
    noise_chunks = _random_noise_chunks(onto, k=5)

    weights = shard_regularity_weights(real_chunks + noise_chunks)
    real_weights = weights[: len(real_chunks)]
    noise_weights = weights[len(real_chunks) :]
    assert all(w == 1.0 for w in real_weights)
    assert all(w < 0.2 for w in noise_weights)  # measured range: 0.003-0.097 at inverse_temperature=60


def test_ground_shards_with_regularity_trust_fixes_the_v043_garbage_majority_false_positives():
    """The full, real-API pipeline on the exact scenario `test_shard_
    conflict_resolution.py`'s own honest-limit test measured as v0.43's
    disclosed failure point (50/50 real/garbage shards). Measured, not
    assumed: false positives restored from a plain-grounding baseline of
    2/10 down to 0/10 - matching the noise-free baseline exactly, not
    just "fewer than before"."""
    onto = _bundled_ontology(n_per_class=100)
    real_chunks = [onto.triples[i : i + 80] for i in range(0, len(onto.triples), 80)]
    noise_chunks = _random_noise_chunks(onto, k=5)

    baseline_memories = onto.ground_shards(real_chunks + noise_chunks, skip_irregular=False)
    baseline_fp = sum(1 for i in range(10) if ask_sharded(onto, baseline_memories, f"star{i}", "walks_on").known)

    memories, weights = onto.ground_shards_with_regularity_trust(real_chunks + noise_chunks)
    fixed_fp = sum(
        1 for i in range(10) if ask_sharded(onto, memories, f"star{i}", "walks_on", shard_weights=weights).known
    )

    assert baseline_fp == 2  # matches this project's own re-measured v0.43 boundary figure
    assert fixed_fp == 0


def test_ground_shards_with_regularity_trust_matches_noise_free_accuracy_baseline():
    """Not just "fewer false positives" - full parity with a noise-free
    ontology's own answer accuracy, at 50% contamination. The noise-free
    baseline itself is 9/10 (not 10/10 - one human entity is naturally a
    harder resonance case at this scale, unrelated to any noise), so this
    checks the weighted pipeline reaches that same ceiling, not a
    strawman 10/10."""
    onto = _bundled_ontology(n_per_class=100)
    real_chunks = [onto.triples[i : i + 80] for i in range(0, len(onto.triples), 80)]
    noise_chunks = _random_noise_chunks(onto, k=5)

    noise_free_memories = onto.ground_shards(real_chunks, skip_irregular=False)
    noise_free_correct = sum(
        1
        for i in range(10)
        if (a := ask_sharded(onto, noise_free_memories, f"human{i}", "is_a")).answer == "human" and a.known
    )

    memories, weights = onto.ground_shards_with_regularity_trust(real_chunks + noise_chunks)
    weighted_correct = sum(
        1
        for i in range(10)
        if (a := ask_sharded(onto, memories, f"human{i}", "is_a", shard_weights=weights)).answer == "human"
        and a.known
    )

    assert noise_free_correct == 9
    assert weighted_correct == noise_free_correct


def test_ground_shards_with_regularity_trust_holds_far_past_the_v043_boundary():
    """The new, substantially higher honest ceiling, measured directly:
    v0.43's consensus mechanism failed open already at 50% contamination.
    This mechanism was checked (not assumed) to still give zero false
    positives and noise-free-matching accuracy at 94% contamination (80
    noise shards vs 5 real ones) - a qualitatively different regime, not
    a marginal improvement. This does not claim there is no ceiling at
    all, only that it was not found in the range actually tested."""
    onto = _bundled_ontology(n_per_class=100)
    real_chunks = [onto.triples[i : i + 80] for i in range(0, len(onto.triples), 80)]
    noise_chunks = _random_noise_chunks(onto, k=80)  # 80/85 shards = 94% contamination

    memories, weights = onto.ground_shards_with_regularity_trust(real_chunks + noise_chunks)
    fp = sum(1 for i in range(10) if ask_sharded(onto, memories, f"star{i}", "walks_on", shard_weights=weights).known)
    correct = sum(
        1
        for i in range(10)
        if (a := ask_sharded(onto, memories, f"human{i}", "is_a", shard_weights=weights)).answer == "human"
        and a.known
    )
    assert fp == 0
    assert correct == 9
