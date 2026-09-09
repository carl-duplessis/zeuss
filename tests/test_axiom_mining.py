"""Automatic axiom discovery (v0.38): mine Rule-shaped patterns from an
Ontology's own facts, and use them as a real qa.py axiom_bias - the
automatic-discovery counterpart to v0.37's hand-written Rule."""
from zeuss.qa import ask
from zeuss.tier3_logic.axiom_mining import (
    DiscoveredExclusion,
    DiscoveredImplication,
    axiom_bias_from_exclusions,
    axiom_bias_from_ontology,
    discover_all_exclusions,
    discover_exclusions,
    discover_implications,
)
from zeuss.tier3_logic.ontology import Ontology


def _humans_and_stars_with_contradiction(seed: int) -> Ontology:
    """10 clean human entities, 10 clean star entities, plus one genuine
    data contradiction (socrates is_a BOTH human and star) - the scenario
    this module's mining was actually measured against, not a toy."""
    onto = Ontology(dim=4096, seed=seed)
    for i in range(10):
        onto.add(f"human{i}", "is_a", "human")
        onto.add(f"human{i}", "walks_on", "earth")
    for i in range(10):
        onto.add(f"star{i}", "is_a", "star")
        onto.add(f"star{i}", "orbits", "galaxy")
    onto.add("socrates", "is_a", "star")
    onto.add("socrates", "is_a", "human")  # the contradiction
    onto.add("socrates", "walks_on", "earth")
    return onto


def test_discover_implications_finds_the_real_pattern_with_full_confidence():
    """support/confidence mining must find walks_on=earth -> is_a=human at
    support=11 (10 clean humans + socrates, who also has the human fact)
    and confidence=1.0 - unaffected by socrates' contradiction, since he
    still holds the human fact too, not just the star one."""
    onto = _humans_and_stars_with_contradiction(seed=1)
    found = discover_implications(onto, "walks_on", "is_a")
    assert found == [DiscoveredImplication("walks_on", "earth", "is_a", "human", 11, 1.0)]


def test_discover_implications_respects_min_support():
    onto = _humans_and_stars_with_contradiction(seed=1)
    # No subject has any filler under a nonexistent relation - 0 support.
    assert discover_implications(onto, "nonexistent_relation", "is_a") == []


def test_discover_exclusions_derives_the_negative_counterpart():
    """Each implication toward one filler of is_a becomes a veto against
    every *other* observed filler of that same relation."""
    onto = _humans_and_stars_with_contradiction(seed=1)
    excl = discover_exclusions(onto, "walks_on", "is_a")
    assert excl == [DiscoveredExclusion("walks_on", "earth", "is_a", "star", 1.0)]


def test_axiom_bias_from_exclusions_corrects_most_of_a_systematically_wrong_baseline():
    """Measured directly, not assumed: at this ontology's scale (21
    entities bundled into one memory), plain resonance picks the wrong
    is_a answer for socrates on every one of 15 seeds tested. Mining
    exclusions from the *other* two relations (walks_on, orbits) and using
    them as a real axiom_bias recovers 12 of the 15 - a genuine, measured
    majority fix, not a 100% claim (this project doesn't overstate partial
    results - see docs/ROADMAP.md v0.38)."""
    wrong_before, wrong_after = [], []
    for seed in range(15):
        onto = _humans_and_stars_with_contradiction(seed)
        memory = onto.ground()
        excl = discover_exclusions(onto, "walks_on", "is_a") + discover_exclusions(onto, "orbits", "is_a")

        baseline = ask(onto, memory, "socrates", "is_a").answer
        if baseline != "human":
            wrong_before.append(seed)

        bias = axiom_bias_from_exclusions(onto, memory, "socrates", "is_a", excl)
        corrected = ask(onto, memory, "socrates", "is_a", axiom_bias=bias).answer
        if corrected != "human":
            wrong_after.append(seed)

    assert len(wrong_before) == 15  # the baseline is systematically wrong here, not a strawman
    assert len(wrong_after) == 3  # 12 of 15 recovered
    assert set(wrong_after).issubset(set(wrong_before))  # never makes a correct case wrong


def test_axiom_bias_from_exclusions_does_not_regress_clean_entities():
    """The mined bias must not damage the many already-correct queries it
    isn't needed for - checked across every seed's clean human/star
    entities, not just the one contradictory case."""
    for seed in range(15):
        onto = _humans_and_stars_with_contradiction(seed)
        memory = onto.ground()
        excl = discover_exclusions(onto, "walks_on", "is_a") + discover_exclusions(onto, "orbits", "is_a")

        for i in range(3):
            for name, relation, expected in [(f"human{i}", "is_a", "human"), (f"star{i}", "is_a", "star")]:
                baseline = ask(onto, memory, name, relation).answer
                if baseline != expected:
                    continue  # not this test's concern - see the systematic-fix test above
                bias = axiom_bias_from_exclusions(onto, memory, name, "is_a", excl)
                corrected = ask(onto, memory, name, relation, axiom_bias=bias).answer
                assert corrected == expected, f"seed={seed} {name}/{relation} regressed: {baseline} -> {corrected}"


def test_discover_all_exclusions_finds_exactly_the_two_real_patterns():
    """discover_all_exclusions tries every ordered pair of the ontology's 3
    relations (6 pairs) automatically - no relation names passed in at all -
    and must land on exactly the same two patterns discover_exclusions
    finds when hand-pointed at (walks_on, is_a) and (orbits, is_a). The
    other 4 pairs (e.g. is_a -> walks_on) must not produce spurious noise:
    checked by requiring the result set match exactly, not just "contains"."""
    onto = _humans_and_stars_with_contradiction(seed=1)
    found = set(discover_all_exclusions(onto))
    expected = {
        DiscoveredExclusion("walks_on", "earth", "is_a", "star", 1.0),
        DiscoveredExclusion("orbits", "galaxy", "is_a", "human", 1.0),
    }
    assert found == expected


def test_axiom_bias_from_ontology_matches_the_hand_picked_version_exactly():
    """axiom_bias_from_ontology (fully automatic - no relation pair named by
    the caller at all) must reproduce the exact same 12/15 systematic-fix
    result axiom_bias_from_exclusions gets when hand-pointed at (walks_on,
    is_a) and (orbits, is_a) - full automation should not change the
    outcome on a case where the hand-picked pairs already covered every
    relevant relation."""
    wrong_before, wrong_after = [], []
    for seed in range(15):
        onto = _humans_and_stars_with_contradiction(seed)
        memory = onto.ground()

        baseline = ask(onto, memory, "socrates", "is_a").answer
        if baseline != "human":
            wrong_before.append(seed)

        bias = axiom_bias_from_ontology(onto, memory, "socrates", "is_a")
        corrected = ask(onto, memory, "socrates", "is_a", axiom_bias=bias).answer
        if corrected != "human":
            wrong_after.append(seed)

    assert len(wrong_before) == 15
    assert wrong_after == [0, 9, 13]  # identical to the hand-picked-relations result
