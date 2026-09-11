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
numbers are documented as prose in docs/ROADMAP.md instead, the same
pattern already used for every other real-data validation this project
has done.

A follow-up `rounds`/`alpha` sweep found the method's *original* default
(`rounds=3, alpha=0.5`) silently broke `ask()`'s honest `known` confidence
flag on the synthetic domain; the default shipped here (`rounds=8,
alpha=0.7`) was chosen to avoid that. But real-data reverification found
that fix didn't transfer - UMLS's `known_rate` still dropped, identically
at both defaults. Tracing *why* (see `refine_entity_vectors`'s own
docstring for the full investigation) found the real cause: correlating
entity vectors with their neighbours - the method's entire point - is
directly in tension with `bind`/`unbind`'s need for near-independent
atomic vectors, and that tension degrades *every* query against a memory
touched by it, not just the queries that want the generalisation benefit.
The fix is architectural: refined vectors now live in a separate
`entity_refined()` namespace that `ground()`/`entity()`/ordinary
`ask()`/`ask_sharded()` never read - generalisation is opt-in, not a
blanket mutation. A first attempt at the opt-in seam (`subject_vector`
alone) was itself a caught mistake: it broke these very tests
(`refined_correct` fell to 2/6, *below* the 3/6 unrefined baseline).
Traced to the original design's real source of power - a fully
consistent refined universe, where the memory's own stored triples were
also built from refined vectors, not just the query probe - so the fix
needed three consistent pieces, not one: `ground_refined()`/`ground_
sharded_refined()` for the memory, `subject_vector` for the probe, and
`entity_vectors` for candidate scoring. These tests use all three
together for generalising queries, and check `entity()`/ordinary
`ask()`/`ask_sharded()` stay a provable no-op for everything else."""
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
    candidate pool (every other person-entity in the domain). Probes and
    scores with `entity_refined` consistently on both sides - probe-only
    refinement was measured to recover markedly less of the effect (see
    `refine_entity_vectors`'s docstring). Falls back to `entity`'s own
    vector when nothing was refined, so this works unchanged for the
    baseline case - caller should pass `memory` from `ground_refined()`
    for this consistency to actually hold at the memory level too."""
    correct = 0
    for entity, true_color in pairs:
        residue = onto.step(memory, onto.entity_refined(entity), "likes_color")
        sims = {c: similarity(residue, onto.entity_refined(c)) for c in colors}
        top = max(sims, key=sims.get)
        correct += top == true_color
    return correct


def test_refine_entity_vectors_returns_self_for_chaining():
    onto = Ontology(dim=64, seed=0)
    onto.add("a", "knows", "b")
    assert onto.refine_entity_vectors() is onto


def test_refine_entity_vectors_does_not_change_minting_order():
    """A second, subtler way this method used to break the "provable
    no-op" guarantee: `Codebook.symbol` mints lazily and depends on *when*
    a name is first requested, so even though refinement stopped writing
    to `ENT:` keys, its own internal access pattern (`entity_names()`'s
    *sorted* order) was still the first thing to touch any not-yet-minted
    entity when called before grounding (the normal, recommended order) -
    silently minting every entity in a different order than `ground()`
    would have, and so handing out different random vectors than an
    identical ontology that never called this method at all. Caught on
    real UMLS data (an "ordinary", override-free query's known_rate came
    back 0.950/0.975 instead of the true no-refinement baseline's
    0.917/1.000) - this is the fast synthetic regression check for the
    fix (pre-minting in `self.triples`' own insertion order first)."""
    onto_never_refined = Ontology(dim=256, seed=0)
    onto_never_refined.add("z", "knows", "a")
    onto_never_refined.add("a", "knows", "b")
    onto_never_refined.add("m", "knows", "z")
    onto_never_refined.ground()  # mints every ENT: entry in ground()'s own (triple-insertion) order
    baseline = {name: onto_never_refined.entity(name).copy() for name in onto_never_refined.entity_names()}

    onto_refined = Ontology(dim=256, seed=0)
    onto_refined.add("z", "knows", "a")
    onto_refined.add("a", "knows", "b")
    onto_refined.add("m", "knows", "z")
    onto_refined.refine_entity_vectors()  # called before any entity() access, same as the recommended order

    for name in onto_refined.entity_names():
        assert similarity(onto_refined.entity(name), baseline[name]) > 0.9999


def test_refine_entity_vectors_does_not_touch_entity():
    """The architectural fix this method was rebuilt around: refined
    vectors used to overwrite `entity()`'s own lookup key, which (traced
    down after a real-data known_rate regression) turned out to degrade
    bind/unbind fidelity for *every* query, not just the ones that wanted
    the generalisation benefit. Refined vectors now live in a separate
    `entity_refined()` namespace - `entity()` must come back bit-for-bit
    identical after refinement, not "different" (the opposite of what
    this test used to check, before the fix)."""
    onto = Ontology(dim=256, seed=0)
    onto.add("a", "knows", "b")
    onto.add("b", "knows", "a")
    before = onto.entity("a").copy()
    onto.refine_entity_vectors()
    after = onto.entity("a")
    assert similarity(before, after) > 0.9999  # exact no-op, by construction


def test_entity_refined_actually_differs_from_entity_after_refinement():
    """The corresponding positive check: `entity_refined()` - the new,
    opt-in accessor - must actually carry the refined signal, even though
    `entity()` itself no longer does."""
    onto = Ontology(dim=256, seed=0)
    onto.add("a", "knows", "b")
    onto.add("b", "knows", "a")
    onto.refine_entity_vectors()
    assert similarity(onto.entity("a"), onto.entity_refined("a")) < 0.999


def test_entity_refined_falls_back_to_entity_when_unrefined():
    """An entity never covered by refinement (or an ontology that never
    refined at all) should behave exactly like `entity()`, not raise or
    return an unrelated freshly-minted vector."""
    onto = Ontology(dim=256, seed=0)
    onto.add("a", "knows", "b")
    assert similarity(onto.entity("a"), onto.entity_refined("a")) > 0.9999


def test_refine_entity_vectors_is_deterministic():
    """Same domain, same (default) rounds/alpha, two independent Ontology
    instances - must produce bit-for-bit identical refined vectors, not
    something that depends on incidental dict/set iteration order."""
    onto1, _, _, _ = _color_groups_domain()
    onto1.refine_entity_vectors()
    onto2, _, _, _ = _color_groups_domain()
    onto2.refine_entity_vectors()
    for name in onto1.entity_names():
        assert similarity(onto1.entity_refined(name), onto2.entity_refined(name)) > 0.9999


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
    memory_baseline = onto_baseline.ground_refined()  # == ground() when nothing was refined
    baseline_correct = _top1_among_colors_correct(onto_baseline, memory_baseline, withheld_baseline, colors)

    onto_refined, withheld_refined, _, _ = _color_groups_domain()
    onto_refined.refine_entity_vectors()  # shipped default: rounds=8, alpha=0.7
    memory_refined = onto_refined.ground_refined()  # the refined-universe memory - see its own docstring
    refined_correct = _top1_among_colors_correct(onto_refined, memory_refined, withheld_refined, colors)

    assert refined_correct > baseline_correct
    assert refined_correct >= 4  # measured 5/6; asserting a safe margin below that, not the exact number


def test_refine_entity_vectors_does_not_change_ordinary_known_rate():
    """The architectural fix's central guarantee, checked on the actual
    `ask()` path rather than assumed from `entity()` alone being
    unchanged: refinement must not change which known facts are reported
    `known`, at all - not "mostly preserved", an exact match against a
    domain that never called `refine_entity_vectors`. This replaces an
    older version of this test that asserted a magic 12/12 constant - that
    number turned out to depend on refinement *also* silently boosting
    ordinary recall (a confound from the pre-fix design, not a genuine
    per-fact property), and is no longer 12/12 now that ordinary queries
    are correctly isolated from refinement's effect."""
    onto_baseline, _, known_facts, _ = _color_groups_domain()
    memory_baseline = onto_baseline.ground()
    baseline_known = {
        entity for entity, _ in known_facts if ask(onto_baseline, memory_baseline, entity, "likes_color").known
    }

    onto_refined, _, known_facts_refined, _ = _color_groups_domain()
    onto_refined.refine_entity_vectors()
    memory_refined = onto_refined.ground()
    refined_known = {
        entity
        for entity, _ in known_facts_refined
        if ask(onto_refined, memory_refined, entity, "likes_color").known
    }

    assert refined_known == baseline_known


def test_generalized_query_via_entity_refined_preserves_honest_confidence():
    """The confidence question that actually matters post-fix: for a
    *withheld* fact (only recoverable through refinement, queried the
    fully-consistent way - `ground_refined()` memory, refined probe,
    refined candidates), does `known` still fire for the ones refinement
    actually gets right?

    Measured answer, kept honest rather than forced to pass: **no, not on
    this domain** - 4/6 withheld facts come back correct (matching the
    ~5/6-among-just-colors number the accuracy test checks), but every
    one's raw coherence sits at 0.02-0.06, consistently below the 0.08
    floor, so `known` is `False` even where the answer is right.

    Real UMLS data, measured on this exact code path (`ground_refined()`
    + `subject_vector`/`entity_vectors` both `entity_refined`, not the
    pre-separation numbers this docstring used to cite - those were an
    *ordinary* query against a memory refinement had mutated directly,
    not a measurement of this opt-in path at all) shows a *milder* version
    of the same direction, not the same near-total collapse: `known_rate`
    0.833/0.933 (tail/head, n=30) against a 0.917/1.000 no-refinement
    baseline - a real but small confidence gap, alongside a large accuracy
    gain (tail MRR 0.041 -> 0.507, Hits@10 -> 0.733; head MRR -> 0.532,
    Hits@10 -> 0.933; see docs/ROADMAP.md's Phase 2 entry). The six-entity
    synthetic pilot's *total* collapse (0/6 known) doesn't generalise to
    UMLS's scale - both domains show correct-but-less-confident, but the
    size of the gap is domain-dependent, not a fixed property of this
    mechanism.

    *Why* the gap exists at all, not just its size: it is not a
    coherence-scale artifact of the refined vector space (`COHERENCE_
    FLOOR` being miscalibrated for it). Querying facts the KB already
    holds directly through this same generalising path shows coherence
    *rising*, not falling - synthetic known facts 0.090 -> 0.159 mean
    coherence, real UMLS training facts 0.110 -> 0.169, both 100% still
    `known=True`. The floor is fine; genuinely-inferred (withheld/test)
    queries are just carrying a structurally weaker signal than directly-
    grounded ones, on both domains - see docs/ROADMAP.md's Phase 2 entry.
    This regression test pins the synthetic domain's own
    measured reality, not a hoped-for one - a future change that makes it
    fail by *improving* known_rate should be treated as good news and
    given a new assertion, not silently
    tightened back to this one."""
    onto, withheld, _, _ = _color_groups_domain()
    onto.refine_entity_vectors()
    memory = onto.ground_refined()

    known_count = sum(
        1
        for entity, _ in withheld
        if ask(
            onto, memory, entity, "likes_color",
            subject_vector=onto.entity_refined(entity), entity_vectors=onto.entity_refined,
        ).known
    )
    assert known_count == 0  # measured: correct != confident for generalised queries on this domain
