"""v0.40: sharding raises Ontology's effective bundling capacity while
keeping each shard at the size v0.39 measured recovery to be reliable at
(~80 triples) - the user's proposed fix for v0.39's found ceiling, tested
before being wired in, same as every other mechanism this project adds.

v0.41 stress-tested that mechanism harder (the user's own follow-up ask)
and found its safety is specific to shards carrying real, structured data.

v0.42 (the user's follow-up to v0.41) answers "how do we tell the
difference": is_structurally_regular checks for duplicate (subject,
relation) bindings within a shard - not semantic truthfulness - and
Ontology.ground_shards uses it to filter untrustworthy candidate shards
before grounding. See the tests at the bottom of this file."""
import random

from zeuss.qa import ask, ask_sharded, chain_sharded, COHERENCE_FLOOR
from zeuss.tier2_substrate.hypervectors import bundle
from zeuss.tier3_logic.ontology import Ontology, is_structurally_regular


def _bundled_ontology(n_per_class: int, dim: int = 8192, seed: int = 1) -> Ontology:
    onto = Ontology(dim=dim, seed=seed)
    for i in range(n_per_class):
        onto.add(f"human{i}", "is_a", "human")
        onto.add(f"human{i}", "walks_on", "earth")
    for i in range(n_per_class):
        onto.add(f"star{i}", "is_a", "star")
        onto.add(f"star{i}", "orbits", "galaxy")
    return onto


def test_ground_sharded_splits_into_correctly_sized_chunks():
    onto = _bundled_ontology(n_per_class=100)  # 400 triples
    shards = onto.ground_sharded(shard_size=80)
    assert len(shards) == 5


def test_ground_sharded_requires_a_nonempty_ontology():
    onto = Ontology(dim=64, seed=0)
    try:
        onto.ground_sharded()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_sharding_recovers_accuracy_past_the_single_bundle_ceiling():
    """The core claim: at 400 triples (test_capacity_ceiling.py measured
    the single bundle at 0/5 correct-and-known here), sharding into 5
    shards of 80 - exactly the size ground()'s own docstring/v0.39 measured
    as reliable - recovers most of that accuracy back. Not claimed as a
    100% fix (measured directly: 9/10 on this sample, one genuine miss),
    but a large, real recovery from a baseline that was completely broken."""
    onto = _bundled_ontology(n_per_class=100)  # 400 triples - the failing case
    shards = onto.ground_sharded(shard_size=80)

    single_bundle_correct = sum(
        1 for i in range(10) if (a := ask(onto, onto.ground(), f"human{i}", "is_a")).answer == "human" and a.known
    )
    sharded_correct = sum(
        1 for i in range(10) if (a := ask_sharded(onto, shards, f"human{i}", "is_a")).answer == "human" and a.known
    )

    assert single_bundle_correct == 0  # confirms this is the same broken baseline v0.39 measured
    assert sharded_correct >= 8  # a real, large recovery - not claimed as perfect


def test_sharding_does_not_inflate_false_positives_on_genuine_guesses():
    """The real risk this mechanism could have introduced: taking the max
    coherence across N independent shards raises the chance some shard's
    pure noise clears COHERENCE_FLOOR by chance, silently breaking the
    guess-detection guarantee (order statistics of a max grow with N).
    Checked directly, not assumed: zero false positives across 20 genuine
    guesses (stars asked about a relation only humans have) even at 5
    shards where every shard carries genuine, structured facts.

    v0.41 found this guarantee is *specific to shards carrying real data*
    - see `test_sharding_is_fragile_to_unstructured_noise_shards` below for
    the sharply different result once shards carry random/unstructured
    triples instead of real-but-irrelevant ones.
    """
    onto = _bundled_ontology(n_per_class=100)
    shards = onto.ground_sharded(shard_size=80)
    false_positives = sum(
        1 for i in range(20) if ask_sharded(onto, shards, f"star{i}", "walks_on").known
    )
    assert false_positives == 0


def test_sharding_scales_to_800_triples_with_more_shards():
    """The recovery isn't a one-off at exactly 400 triples/5 shards - checked
    at 2x that (800 triples, 10 shards): still a large majority correct,
    still zero false positives on genuine guesses."""
    onto = _bundled_ontology(n_per_class=200)  # 800 triples
    shards = onto.ground_sharded(shard_size=80)
    assert len(shards) == 10

    correct = sum(
        1 for i in range(20) if (a := ask_sharded(onto, shards, f"human{i}", "is_a")).answer == "human" and a.known
    )
    false_positives = sum(1 for i in range(20) if ask_sharded(onto, shards, f"star{i}", "walks_on").known)

    assert correct >= 16  # measured 18/20 (90%) - asserting a safe margin below that
    assert false_positives == 0


def test_ask_sharded_requires_at_least_one_memory():
    onto = _bundled_ontology(n_per_class=5)
    try:
        ask_sharded(onto, [], "human0", "is_a")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_chain_sharded_walks_a_transitive_chain_correctly():
    """chain_sharded generalizes ask_sharded's per-query shard selection to
    every hop of a multi-hop chain - a fact needed partway through could
    live in a different shard than the fact before it."""
    onto = Ontology(dim=8192, seed=7)
    onto.add("socrates", "is_a", "human")
    onto.add("human", "is_a", "mortal")
    onto.add("mortal", "is_a", "thing")
    shards = onto.ground_sharded(shard_size=80)  # small ontology -> 1 shard, still exercises the API
    c = chain_sharded(onto, shards, "socrates", "is_a")
    assert [name for name, _ in c.hops] == ["human", "mortal", "thing"]


def test_chain_sharded_requires_at_least_one_memory():
    onto = _bundled_ontology(n_per_class=5)
    try:
        chain_sharded(onto, [], "human0", "is_a")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_chain_sharded_shard_weights_is_a_noop_by_default():
    """`shard_weights=None` (default) must reproduce the exact same chain
    as calling `chain_sharded` without the parameter at all - checked
    directly, not just assumed from the implementation."""
    onto = Ontology(dim=8192, seed=7)
    onto.add("socrates", "is_a", "human")
    onto.add("human", "is_a", "mortal")
    onto.add("mortal", "is_a", "thing")
    shards = onto.ground_sharded(shard_size=80)
    without_param = chain_sharded(onto, shards, "socrates", "is_a")
    with_none = chain_sharded(onto, shards, "socrates", "is_a", shard_weights=None)
    assert [n for n, _ in without_param.hops] == [n for n, _ in with_none.hops]


def test_chain_sharded_shard_weights_suppresses_a_shard_at_every_hop():
    """Phase 1: `shard_sharded` gained the same `shard_weights` mechanism
    `ask_sharded` has had since v0.46 - this checks it actually changes
    the outcome, not just that it accepts the parameter. Two shards each
    carry a complete, internally consistent but mutually exclusive two-hop
    chain for the same starting subject (`socrates is_a human is_a
    mortal` vs `socrates is_a martian is_a alien`) - both are equally
    "real" structured data, so without weighting the winner at each hop
    is decided by whichever shard's raw resonance happens to be strongest
    (not asserted here, since that's not the point being tested). Setting
    `shard_weights` to heavily suppress one shard must force the *entire*
    chain to come from the other one - at both hops, not just the first,
    directly verifying the "applied at every hop" claim in this
    parameter's own docstring rather than trusting it un-checked."""
    onto = Ontology(dim=8192, seed=3)
    onto.add("socrates", "is_a", "human")
    onto.add("human", "is_a", "mortal")
    shard_a = list(onto.triples)

    onto.add("socrates", "is_a", "martian")
    onto.add("martian", "is_a", "alien")
    shard_b = onto.triples[len(shard_a) :]

    shards = [shard_a, shard_b]
    memories = onto.ground_shards(shards, skip_irregular=False)

    suppress_a = chain_sharded(onto, memories, "socrates", "is_a", shard_weights=[0.0, 1.0])
    assert [n for n, _ in suppress_a.hops] == ["martian", "alien"]

    suppress_b = chain_sharded(onto, memories, "socrates", "is_a", shard_weights=[1.0, 0.0])
    assert [n for n, _ in suppress_b.hops] == ["human", "mortal"]


def test_chain_sharded_shard_weights_rejects_mismatched_length():
    onto = Ontology(dim=256, seed=0)
    onto.add("socrates", "is_a", "human")
    shards = onto.ground_sharded(shard_size=80)
    try:
        chain_sharded(onto, shards, "socrates", "is_a", shard_weights=[1.0, 1.0])
        assert False, "expected ValueError"
    except ValueError:
        pass


def _random_noise_shards(onto: Ontology, k: int, shard_size: int = 80, seed: int = 99) -> list:
    """K extra shards of random (entity, relation, entity) nonsense
    triples, reusing onto's existing vocabulary so entity_names() (and
    _entity_codebook cost) doesn't grow - isolating "shard carries garbage"
    as the only new variable versus the all-real-shards tests above."""
    rng = random.Random(seed)
    entities = onto.entity_names()
    relations = sorted({r for _, r, _ in onto.triples})
    shards = []
    for _ in range(k):
        fake_triples = [(rng.choice(entities), rng.choice(relations), rng.choice(entities)) for _ in range(shard_size)]
        shards.append(bundle([onto._triple_vector(*t) for t in fake_triples]))
    return shards


def test_sharding_is_fragile_to_unstructured_noise_shards():
    """v0.41 - the user's own stress-test follow-up to v0.40. The false-
    positive safety measured above holds when every shard carries real,
    structured facts (even ones irrelevant to the query at hand). It does
    NOT hold when shards instead carry unstructured random triples: at just
    5 real shards + 10 random-noise shards (15 total - far short of the 85
    shards that stayed safe in the all-real-data test), false positives on
    genuine guesses jump from 0/10 to several/10.

    Checked directly, not assumed, that this isn't an artifact of noise
    triples occasionally reconstructing a real answer by chance (a possible
    confound): the same collapse happens even when noise fillers are drawn
    only from entities that can never be a valid answer to the relations
    being tested (ruled out in the investigation that produced this test;
    see docs/ROADMAP.md v0.41).

    Practical reading: ask_sharded/chain_sharded's validated safe regime is
    "shards that are genuine partitions of real data" (v0.40), not
    "arbitrary many shards regardless of content" - mixing in unstructured
    noise breaks the guess-detection guarantee far sooner than growing the
    same amount of real data across more real shards does.
    """
    onto = _bundled_ontology(n_per_class=100)
    real = onto.ground_sharded(shard_size=80)  # 5 real shards
    noisy = real + _random_noise_shards(onto, k=10)  # +10 pure-noise shards

    false_positives_real_only = sum(1 for i in range(10) if ask_sharded(onto, real, f"star{i}", "walks_on").known)
    false_positives_with_noise = sum(1 for i in range(10) if ask_sharded(onto, noisy, f"star{i}", "walks_on").known)

    assert false_positives_real_only == 0  # the v0.40 baseline, reconfirmed
    assert false_positives_with_noise >= 3  # a real, measured collapse - not a strawman lower bound


def test_is_structurally_regular_distinguishes_real_from_random_noise():
    """The core v0.42 claim: this cheap, purely-structural check (no
    semantics, no ground truth about truthfulness) correctly separates
    every real shard from every random-noise shard on the exact data used
    to find the v0.41 fragility - real shards never repeat a (subject,
    relation) pair by construction; random sampling with replacement from
    a finite entity pool almost always does."""
    onto = _bundled_ontology(n_per_class=100)
    real_chunks = [onto.triples[i : i + 80] for i in range(0, len(onto.triples), 80)]
    assert all(is_structurally_regular(chunk) for chunk in real_chunks)

    noise_shards_as_triples = []
    rng = random.Random(99)
    entities = onto.entity_names()
    relations = sorted({r for _, r, _ in onto.triples})
    for _ in range(10):
        noise_shards_as_triples.append(
            [(rng.choice(entities), rng.choice(relations), rng.choice(entities)) for _ in range(80)]
        )
    assert not any(is_structurally_regular(chunk) for chunk in noise_shards_as_triples)


def test_is_structurally_regular_flags_a_genuine_data_contradiction():
    """Doubles as a contradiction detector for real data, not just a noise
    filter for untrusted data: the same (subject, relation) asserted with
    two different fillers (v0.37's socrates is_a human/star case) is
    exactly what this check is looking for."""
    assert not is_structurally_regular([("socrates", "is_a", "human"), ("socrates", "is_a", "star")])
    assert is_structurally_regular([("socrates", "is_a", "human"), ("plato", "is_a", "human")])


def test_filtering_irregular_shards_restores_safety():
    """The concrete fix: Ontology.ground_shards(skip_irregular=True) on the
    exact adversarial mix that broke ask_sharded in v0.41 (5 real shards +
    10 random-noise shards) restores the original v0.40 guarantee
    completely - not partially. Checked directly against the real API, not
    re-derived from the offline experiment that motivated this."""
    onto = _bundled_ontology(n_per_class=100)
    real_chunks = [onto.triples[i : i + 80] for i in range(0, len(onto.triples), 80)]

    rng = random.Random(99)
    entities = onto.entity_names()
    relations = sorted({r for _, r, _ in onto.triples})
    noise_chunks = [
        [(rng.choice(entities), rng.choice(relations), rng.choice(entities)) for _ in range(80)] for _ in range(10)
    ]

    mixed_chunks = real_chunks + noise_chunks
    unfiltered = onto.ground_shards(mixed_chunks, skip_irregular=False)
    filtered = onto.ground_shards(mixed_chunks, skip_irregular=True)

    assert len(unfiltered) == 15
    assert len(filtered) == 5  # every noise shard correctly dropped, every real shard kept

    false_pos_unfiltered = sum(1 for i in range(10) if ask_sharded(onto, unfiltered, f"star{i}", "walks_on").known)
    false_pos_filtered = sum(1 for i in range(10) if ask_sharded(onto, filtered, f"star{i}", "walks_on").known)
    correct_filtered = sum(
        1 for i in range(10) if (a := ask_sharded(onto, filtered, f"human{i}", "is_a")).answer == "human" and a.known
    )

    assert false_pos_unfiltered >= 3  # reconfirms the break is real before checking the fix
    assert false_pos_filtered == 0  # fully restored
    assert correct_filtered >= 8  # matches the original v0.40 all-real-shards accuracy


def test_ground_sharded_skips_irregular_shards_by_default():
    """ground_sharded's default (skip_irregular=True) applies the same
    filter to its own sequential chunking - relevant when the ontology's
    own triples contain a genuine contradiction that a chunk boundary
    happens to land inside."""
    onto = Ontology(dim=8192, seed=1)
    onto.add("socrates", "is_a", "human")
    onto.add("socrates", "is_a", "star")  # contradiction, lands in the same (small) shard
    onto.add("plato", "is_a", "human")
    shards = onto.ground_sharded(shard_size=80)
    assert shards == []  # the only shard is irregular, so nothing is grounded

    shards_unfiltered = onto.ground_sharded(shard_size=80, skip_irregular=False)
    assert len(shards_unfiltered) == 1
