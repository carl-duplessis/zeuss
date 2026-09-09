"""v0.45: chain mined implications into a genuine transitive-closure
constraint network before checking exclusions - the direct follow-up to
v0.44's flat pairwise `axiom_violations` check, after being asked
specifically to "build a genuine constraint network from the mined
axioms (transitivity chains, exclusivity webs) and check global
consistency across that network."

Deliberately NOT built by reusing `sheaf.py`'s H0/H1 cohomology machinery:
that module's restriction edges assert equality (`restrict_u * x_u ==
restrict_v * x_v`) between two scalar stalks, which is the right tool for
"do two independently-derived conclusions agree" (see `grounding.
compile_theories`/`zeuss audit`) but the wrong shape for implication
(directional, not symmetric) and exclusion (`not both`, not `always
equal`). `_implied_closure`'s forward-chaining BFS over a directed
implication graph is the standard, correctly-scoped tool for this instead
- the same thing Horn-clause/Datalog-style rule engines do.

The concrete gap this closes: v0.44's `axiom_violations` only caught a
violation when a mined exclusion's antecedent was one of the subject's own
*directly-asserted* facts. A subject whose held facts only reach the
exclusion's antecedent by composing two or more separately-mined
implication rules was invisible to it - exactly the "different, stronger
signal" asked for here."""
from zeuss.tier3_logic.axiom_mining import DiscoveredExclusion, DiscoveredImplication, discover_implications
from zeuss.tier3_logic.ontology import Ontology, axiom_violations, resolve_shard_conflicts

_CHAIN_IMPLICATIONS = [
    DiscoveredImplication("born_on", "mars", "is_a", "martian", support=10, confidence=1.0),
    DiscoveredImplication("is_a", "martian", "breathes", "co2", support=10, confidence=1.0),
]
_CHAIN_EXCLUSIONS = [
    DiscoveredExclusion("breathes", "co2", "needs", "oxygen", confidence=1.0),
]


def test_flat_pairwise_check_misses_a_two_hop_violation():
    """The honest gap being closed: `curiosity` holds `born_on=mars` and
    `needs=oxygen` - genuinely contradictory once you chain `born_on=mars
    -> is_a=martian -> breathes=co2` and note `breathes=co2` excludes
    `needs=oxygen` - but neither mined exclusion's antecedent
    (`breathes=co2`) is `curiosity`'s own *direct* fact, so the flat v0.44
    check (no `implications` given) finds nothing."""
    triples = [("curiosity", "born_on", "mars"), ("curiosity", "needs", "oxygen")]
    assert axiom_violations(triples, _CHAIN_EXCLUSIONS) == set()


def test_chained_check_catches_the_same_two_hop_violation():
    """Passing `implications` closes exactly that gap: chaining
    `born_on=mars -> is_a=martian -> breathes=co2` puts `breathes=co2` in
    `curiosity`'s reachable closure, so the mined exclusion's antecedent
    is now satisfied and `needs=oxygen` is correctly flagged."""
    triples = [("curiosity", "born_on", "mars"), ("curiosity", "needs", "oxygen")]
    violations = axiom_violations(triples, _CHAIN_EXCLUSIONS, implications=_CHAIN_IMPLICATIONS)
    assert violations == {("curiosity", "needs", "oxygen")}


def test_chained_check_does_not_flag_a_subject_missing_a_link_in_the_chain():
    """A subject who only holds `born_on=mars` (never `needs=oxygen`) has
    nothing to contradict - closure includes the derived `is_a=martian`
    and `breathes=co2`, but no exclusion fires because `curiosity`
    (renamed here `perseverance`) never asserts the excluded fact."""
    triples = [("perseverance", "born_on", "mars")]
    assert axiom_violations(triples, _CHAIN_EXCLUSIONS, implications=_CHAIN_IMPLICATIONS) == set()


def test_chained_check_is_a_noop_when_implications_is_none():
    """Backward compatibility with v0.44: omitting `implications` (default
    `None`) must reproduce the exact flat-check result, not silently
    change behaviour for existing callers."""
    triples = [("socrates", "walks_on", "earth"), ("socrates", "is_a", "star")]
    exclusions = [DiscoveredExclusion("walks_on", "earth", "is_a", "star", confidence=1.0)]
    assert axiom_violations(triples, exclusions) == axiom_violations(triples, exclusions, implications=None)
    assert axiom_violations(triples, exclusions) == {("socrates", "is_a", "star")}


def test_resolve_shard_conflicts_recovers_a_two_hop_contradiction_across_shards():
    """The full pipeline: a real shard asserts the two-hop-contradicting
    facts about the same subject, and a competing garbage shard's
    `needs=oxygen` claim would otherwise tie/out-vote it. With the chained
    exclusion wired in via `implications`, the structurally impossible
    `needs=oxygen` fact is discarded regardless of the vote count."""
    shard_a = [("curiosity", "born_on", "mars"), ("curiosity", "needs", "oxygen")]
    resolved = resolve_shard_conflicts(
        [shard_a],
        multi_valued_relations=set(),
        exclusions=_CHAIN_EXCLUSIONS,
        implications=_CHAIN_IMPLICATIONS,
    )
    (kept,) = resolved
    assert ("curiosity", "needs", "oxygen") not in kept
    assert ("curiosity", "born_on", "mars") in kept


def test_discover_implications_can_supply_a_real_mined_chain():
    """Not just a hand-authored toy: `discover_implications` (v0.38's real
    mining API) genuinely mines both links of the chain from an ontology's
    own data at full confidence, which `axiom_violations` then chains
    through exactly as it would a hand-written rule."""
    onto = Ontology(dim=2048, seed=3)
    for i in range(10):
        onto.add(f"rover{i}", "born_on", "mars")
        onto.add(f"rover{i}", "is_a", "martian")
        onto.add(f"rover{i}", "breathes", "co2")

    link_1 = discover_implications(onto, "born_on", "is_a")
    link_2 = discover_implications(onto, "is_a", "breathes")
    assert link_1 == [DiscoveredImplication("born_on", "mars", "is_a", "martian", 10, 1.0)]
    assert link_2 == [DiscoveredImplication("is_a", "martian", "breathes", "co2", 10, 1.0)]

    triples = [("probe1", "born_on", "mars"), ("probe1", "needs", "oxygen")]
    violations = axiom_violations(triples, _CHAIN_EXCLUSIONS, implications=link_1 + link_2)
    assert violations == {("probe1", "needs", "oxygen")}
