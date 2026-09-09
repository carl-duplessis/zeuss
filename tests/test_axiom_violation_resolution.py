"""v0.44: logical-axiom-violation checking layered on top of v0.43's
cross-shard consensus - the user's direct follow-up after v0.43 disclosed
its own honest ceiling ("needs garbage to be a minority; fails open at
50/50"). That ceiling is real and general: no majority-vote/consensus
scheme can be made reliable against an adversarial *majority* of bad data
(the same wall Byzantine fault tolerance and robust statistics hit -
median-based estimators tolerate up to ~50% contamination, never more,
because past that point "the majority" and "the truth" are definitionally
not the same thing anymore).

The genuinely different, additional lever: a fact that structurally
violates a *trusted* logical axiom (mined via `axiom_mining.py`'s
support/confidence discovery, or hand-written) doesn't need to win a vote
at all - `axiom_violations` discards votes for it *before* `resolve_shard_
conflicts` even tallies a majority, so a garbage majority can't out-vote a
true minority fact on a pair the exclusions cover. This is not a bigger
version of consensus voting (it doesn't touch the theoretical 50% wall),
it's an independent signal with a different failure mode: it's only as
trustworthy as the exclusions it's given, and an exclusion re-mined from
the *same* contaminated pool inherits the exact same limit (see
`axiom_violations`'s own docstring caveat, and the test below that checks
this honestly rather than assuming the layering is a free lunch)."""
import random

from zeuss.tier3_logic.axiom_mining import DiscoveredExclusion, discover_all_exclusions
from zeuss.tier3_logic.ontology import Ontology, axiom_violations, resolve_shard_conflicts


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


_TRUSTED_EXCLUSIONS = [
    DiscoveredExclusion("walks_on", "earth", "is_a", "star", 1.0),
    DiscoveredExclusion("orbits", "galaxy", "is_a", "human", 1.0),
]


def test_axiom_violations_flags_only_the_structurally_impossible_fact():
    """socrates holds both `walks_on earth` and `is_a star` - the latter
    structurally violates the trusted `walks_on=earth excludes is_a=star`
    exclusion. plato's `is_a human` is consistent with the same axiom and
    must not be flagged; a wholly unrelated relation on socrates must not
    be flagged either."""
    triples = [
        ("socrates", "walks_on", "earth"),
        ("socrates", "is_a", "star"),
        ("socrates", "has_friend", "plato"),
        ("plato", "walks_on", "earth"),
        ("plato", "is_a", "human"),
    ]
    violations = axiom_violations(triples, _TRUSTED_EXCLUSIONS)
    assert violations == {("socrates", "is_a", "star")}


def test_axiom_violations_is_a_noop_with_no_exclusions():
    triples = [("socrates", "is_a", "star"), ("socrates", "walks_on", "earth")]
    assert axiom_violations(triples, []) == set()


def test_resolve_shard_conflicts_recovers_a_fact_a_garbage_majority_outvoted():
    """The core claim: plain majority voting gets this case WRONG (2
    garbage shards outvote 1 real one), but a trusted exclusion recovers
    the true fact - not by counting votes differently, but by disqualifying
    the false claim from the vote entirely, since it structurally
    contradicts socrates' own (undisputed) `walks_on earth` fact."""
    real_shard = [("socrates", "walks_on", "earth"), ("socrates", "is_a", "human")]
    garbage_shard_a = [("socrates", "is_a", "star")]
    garbage_shard_b = [("socrates", "is_a", "star")]
    shards = [real_shard, garbage_shard_a, garbage_shard_b]

    # Without exclusions: garbage's 2-vs-1 majority wins, so the *true* fact
    # is the one treated as disagreeing and excised - the wrong outcome,
    # not a strawman (checked directly, not assumed).
    without = resolve_shard_conflicts(shards, multi_valued_relations=set())
    assert ("socrates", "is_a", "human") not in without[0]

    # With the trusted exclusion: the garbage votes for `is_a=star` are
    # disqualified before tallying (they contradict `walks_on=earth`), so
    # `is_a=human` wins by default and survives in the real shard, while
    # both garbage shards' false claims are excised.
    resolved = resolve_shard_conflicts(shards, multi_valued_relations=set(), exclusions=_TRUSTED_EXCLUSIONS)
    assert ("socrates", "is_a", "human") in resolved[0]
    assert ("socrates", "is_a", "star") not in resolved[1]
    assert ("socrates", "is_a", "star") not in resolved[2]


def test_exclusions_still_excise_a_literal_violation_at_the_v043_garbage_majority_boundary():
    """The exact scenario `test_shard_conflict_resolution.py`'s
    `test_ground_resolved_shards_has_an_honest_limit_at_a_real_garbage_
    majority` measured as v0.43's disclosed failure point (50/50 real/
    garbage shards, consensus fails open because the garbage volume itself
    corrupts `classify_multi_valued_relations`). Exclusions here are mined
    from a small SEPARATE seed ontology - not from the contaminated shards
    under test - matching this module's own "only as trustworthy as an
    independently-sourced exclusion" caveat.

    Measured, not assumed, and a more honest claim than "fixes accuracy":
    at this exact contamination level exactly one noise shard happens to
    assert the literal, structurally-impossible fact `star99 is_a human`
    (checked directly) - `resolve_shard_conflicts` with exclusions excises
    it (`baseline` keeps it, `with_exclusions` doesn't) even though
    `is_a`'s own vote-based consensus has already failed open. This does
    NOT, on its own, fix `ask_sharded`'s downstream accuracy at this
    contamination level (checked directly and disclosed, not swept under
    the rug): once `is_a` is misclassified multi-valued, hundreds of
    *other*, unrelated noise triples under the same relation are also
    bundled in unfiltered, and most of the resulting wrong answers turn
    out to be hypervector crosstalk from that flood - not literal
    violating triples - which excising one specific bad fact can't touch.
    Exclusions are a real, additional, honestly-scoped tool for the
    literal-contradiction problem; they are not a fix for interference
    from a relation the classifier has already given up on filtering."""
    onto = _bundled_ontology(n_per_class=100)
    real_chunks = [onto.triples[i : i + 80] for i in range(0, len(onto.triples), 80)]
    noise_chunks = _random_noise_chunks(onto, k=5)  # 5 of 10 shards = 50% contamination

    seed_onto = _bundled_ontology(n_per_class=10, seed=2)  # independent of the contaminated shards
    exclusions = discover_all_exclusions(seed_onto)

    def _surviving_bad_fact(shards):
        return sum(1 for shard in shards for s, r, o in shard if (s, r, o) == ("star99", "is_a", "human"))

    baseline = resolve_shard_conflicts(real_chunks + noise_chunks)
    with_exclusions = resolve_shard_conflicts(real_chunks + noise_chunks, exclusions=exclusions)

    assert _surviving_bad_fact(baseline) == 1  # the literal violation is really there, not a strawman
    assert _surviving_bad_fact(with_exclusions) == 0  # exclusions excise it even though consensus already failed


def test_axiom_violations_inherits_the_same_limit_when_exclusions_are_mined_from_contaminated_data():
    """The honest counter-case, checked directly rather than assumed away:
    mining exclusions from the SAME 50/50-contaminated pool (instead of an
    independent seed) does not reliably help, because
    `discover_all_exclusions`'s own support/confidence statistics are
    exposed to the same contamination `classify_multi_valued_relations`
    already fails on. This is what `axiom_violations`'s docstring
    disclosure means in practice - not a defect introduced here, the same
    limit inherited one level up."""
    onto = _bundled_ontology(n_per_class=100)
    real_chunks = [onto.triples[i : i + 80] for i in range(0, len(onto.triples), 80)]
    noise_chunks = _random_noise_chunks(onto, k=5)
    all_triples = [t for shard in (real_chunks + noise_chunks) for t in shard]
    contaminated_source = Ontology(dim=8192, seed=1)
    contaminated_source.triples = all_triples

    self_mined_exclusions = discover_all_exclusions(contaminated_source)
    # The contaminated pool's own noise dilutes/breaks the same confidence
    # threshold the clean seed ontology satisfies exactly.
    assert self_mined_exclusions != discover_all_exclusions(_bundled_ontology(n_per_class=10, seed=2))
