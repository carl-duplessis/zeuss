"""v0.40: sharding raises Ontology's effective bundling capacity while
keeping each shard at the size v0.39 measured recovery to be reliable at
(~80 triples) - the user's proposed fix for v0.39's found ceiling, tested
before being wired in, same as every other mechanism this project adds."""
from zeuss.qa import ask, ask_sharded, chain_sharded, COHERENCE_FLOOR
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
    shards."""
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
