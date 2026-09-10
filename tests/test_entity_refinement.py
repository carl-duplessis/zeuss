"""Phase 2: closing the generalisation gap Phase 0 established Zeuss has -
Codebook.symbol's independently-random entity vectors mean nothing lets
Zeuss infer a fact it was never told, only recall what it was told or
deduce what's logically entailed by it. Ontology.refine_entity_vectors is
a genuinely different mechanism from the usual fix (gradient-trained
embeddings): iterated neighbor-vector blending using the bind/bundle
algebra already in the substrate, no loss function, no training loop.

This file captures the synthetic property that actually matters (a
relationally-distinct group structure a withheld fact is only inferable
through) so the suite doesn't need network access - the real Nations/UMLS
numbers (Nations: tail MRR 0.389->0.500-0.544; UMLS: tail MRR
0.041->0.651, checked specifically because UMLS has none of Nations' own
documented inverse-relation-redundancy quirk) are documented as prose in
docs/ROADMAP.md instead, the same pattern already used for every other
real-data validation this project has done.

A follow-up `rounds`/`alpha` sweep (done to properly map the parameter
space, not just spot-check 2-3 points) found the method's *original*
default (`rounds=3, alpha=0.5`, what the Nations/UMLS numbers above
actually used) silently broke `ask()`'s honest `known` confidence flag -
every fact was still recalled correctly, but never reported as confident.
Never caught before because nothing had checked `known`, only exact-match
accuracy. The default shipped here (`rounds=8, alpha=0.7`) was chosen
specifically because it's inside the sweep's one region where inference
accuracy, known-fact recall, *and* honest confidence all hold together -
`test_refine_entity_vectors_default_preserves_honest_confidence` is the
regression test for that specific failure mode."""
import random

from zeuss.qa import ask
from zeuss.tier2_substrate.hypervectors import similarity
from zeuss.tier3_logic.ontology import Ontology


def _color_groups_domain(group_size=6, n_withheld_per_group=2, seed=0, dim=2048):
    """3 relationally-distinct groups (dense `knows` edges within a group,
    never across groups) sharing one `likes_color` fact per member,
    consistent within a group - except withheld for a few test entities
    per group, inferable (if at all) only through their `knows`-neighbors'
    colors, never asserted directly for the withheld entities themselves.
    ``known_facts`` are the complementary set - entities that DO directly
    hold their own `likes_color` fact - used to check refinement doesn't
    degrade recall of what the KB actually already knows."""
    colors = ["blue", "red", "green"]
    rng = random.Random(seed)
    groups = {c: [f"{c}_{i}" for i in range(group_size)] for c in colors}
    onto = Ontology(dim=dim, seed=1)
    withheld = []
    known_facts = []
    for color, members in groups.items():
        for a in members:
            for b in members:
                if a != b:
                    onto.add(a, "knows", b)
        to_withhold = set(rng.sample(members, n_withheld_per_group))
        for m in members:
            if m in to_withhold:
                withheld.append((m, color))
            else:
                onto.add(m, "likes_color", color)
                known_facts.append((m, color))
    return onto, withheld, known_facts, colors


def _top1_among_colors_correct(onto, memory, pairs, colors) -> int:
    """Count how many (entity, true_color) pairs are correctly top-1,
    scored only among the 3 real color candidates (not the whole entity
    set) - isolates whether refinement shifts similarity toward the true
    color at all, independent of _cleanup's much larger, mostly-irrelevant
    candidate pool (every other person-entity in the domain)."""
    correct = 0
    for entity, true_color in pairs:
        residue = onto.step(memory, onto.entity(entity), "likes_color")
        sims = {c: similarity(residue, onto.entity(c)) for c in colors}
        top = max(sims, key=sims.get)
        correct += top == true_color
    return correct


def test_refine_entity_vectors_returns_self_for_chaining():
    onto = Ontology(dim=64, seed=0)
    onto.add("a", "knows", "b")
    assert onto.refine_entity_vectors() is onto


def test_refine_entity_vectors_actually_changes_the_stored_vector():
    """The exact bug this method was caught by during development: an
    earlier version wrote refined vectors back under the wrong codebook
    key, so `entity()` kept returning the original untouched random
    vector - every configuration looked identical regardless of
    `rounds`/`alpha`. This checks the fix directly: the vector `entity()`
    returns must actually change after refinement, not silently stay the
    same."""
    onto = Ontology(dim=256, seed=0)
    onto.add("a", "knows", "b")
    onto.add("b", "knows", "a")
    before = onto.entity("a").copy()
    onto.refine_entity_vectors()
    after = onto.entity("a")
    assert similarity(before, after) < 0.999  # genuinely different, not a no-op


def test_refine_entity_vectors_is_deterministic():
    """Same domain, same (default) rounds/alpha, two independent Ontology
    instances - must produce bit-for-bit identical refined vectors, not
    something that depends on incidental dict/set iteration order."""
    onto1, _, _, _ = _color_groups_domain()
    onto1.refine_entity_vectors()
    onto2, _, _, _ = _color_groups_domain()
    onto2.refine_entity_vectors()
    for name in onto1.entity_names():
        assert similarity(onto1.entity(name), onto2.entity(name)) > 0.9999


def test_refine_entity_vectors_infers_a_withheld_fact_via_neighbor_structure():
    """The core capability claim, checked directly rather than assumed to
    hold from the (separately, real-data) measured numbers in docs/
    ROADMAP.md: refinement at its shipped default must measurably beat
    both the 1/3 chance rate and the unrefined baseline on the same
    domain - not just "do something", genuinely infer the withheld fact
    through relational structure alone."""
    onto_baseline, withheld_baseline, _, colors = _color_groups_domain()
    for name in onto_baseline.entity_names():  # match refine's own eager-minting access pattern,
        onto_baseline.entity(name)             # so this is a fair baseline, not an access-order artifact
    memory_baseline = onto_baseline.ground()
    baseline_correct = _top1_among_colors_correct(onto_baseline, memory_baseline, withheld_baseline, colors)

    onto_refined, withheld_refined, _, _ = _color_groups_domain()
    onto_refined.refine_entity_vectors()  # shipped default: rounds=8, alpha=0.7
    memory_refined = onto_refined.ground()
    refined_correct = _top1_among_colors_correct(onto_refined, memory_refined, withheld_refined, colors)

    assert refined_correct > baseline_correct
    assert refined_correct >= 4  # measured 5/6; asserting a safe margin below that, not the exact number


def test_refine_entity_vectors_default_preserves_honest_confidence():
    """Regression test for the specific failure a rounds/alpha sweep found
    in this method's *original* default (rounds=3, alpha=0.5): accuracy
    looked perfect, but ask()'s honest `known` confidence flag silently
    never fired, even for facts recalled correctly - invisible to any
    test that only checks top-1 correctness, which is exactly how it went
    undetected until the sweep. The shipped default (rounds=8, alpha=0.7)
    was chosen specifically to avoid this - checked directly here, not
    just assumed from the sweep's own numbers."""
    onto, _, known_facts, _ = _color_groups_domain()
    onto.refine_entity_vectors()
    memory = onto.ground()

    known_count = sum(1 for entity, _ in known_facts if ask(onto, memory, entity, "likes_color").known)
    assert known_count == len(known_facts)  # measured 12/12 at the shipped default
