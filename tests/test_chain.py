"""Multi-hop deduction: chain walks a relation's transitive closure honestly."""
from zeuss.qa import COHERENCE_FLOOR, chain, entails, demo_ontology
from zeuss.tier3_logic.ontology import Ontology


def test_chain_walks_transitive_closure():
    # A dedicated, low-crosstalk chain (larger dim, few facts) so the
    # substrate resolves all three hops unambiguously.
    onto = Ontology(dim=8192, seed=7)
    onto.add("socrates", "is_a", "human")
    onto.add("human", "is_a", "mortal")
    onto.add("mortal", "is_a", "thing")
    memory = onto.ground()
    c = chain(onto, memory, "socrates", "is_a")
    names = [name for name, _ in c.hops]
    assert names == ["human", "mortal", "thing"]
    # Coherence compounds: each cumulative entry is <= the previous one.
    assert all(a >= b for a, b in zip(c.cumulative, c.cumulative[1:]))
    assert all(0.0 <= v <= 1.0 for v in c.cumulative)


def test_chain_on_demo_ontology_walks_the_full_closure():
    onto = demo_ontology()
    memory = onto.ground()
    c = chain(onto, memory, "socrates", "is_a")
    names = [name for name, _ in c.hops]
    assert names == ["human", "mortal", "thing"]


def test_chain_stops_on_cycle():
    onto = Ontology(dim=2048, seed=3)
    onto.add("a", "next", "b")
    onto.add("b", "next", "a")
    memory = onto.ground()
    c = chain(onto, memory, "a", "next", max_hops=10)
    # Must terminate well before max_hops once "a" would be revisited.
    assert len(c.hops) <= 3
    names = [name for name, _ in c.hops]
    assert len(names) == len(set(names))


def test_entails_true_and_false():
    onto = demo_ontology()
    memory = onto.ground()

    holds = entails(onto, memory, "socrates", "is_a", "mortal")
    assert holds.holds
    assert holds.hops == 2
    assert holds.coherence > 0.0

    not_holds = entails(onto, memory, "socrates", "is_a", "sun")
    assert not not_holds.holds
    assert not_holds.coherence == 0.0
    assert not_holds.hops == 0

    three_hop = entails(onto, memory, "socrates", "is_a", "thing")
    assert three_hop.holds
    assert three_hop.hops == 3


def test_entails_direct_fact_is_one_hop():
    onto = demo_ontology()
    memory = onto.ground()
    verdict = entails(onto, memory, "socrates", "is_a", "human")
    assert verdict.holds
    assert verdict.hops == 1
    assert verdict.coherence >= COHERENCE_FLOOR


def test_chain_resonance_coherence_is_high_for_a_clean_transitive_chain():
    """`Chain.resonance_coherence` (see `qa.py`'s v0.34 docstring) composes
    the wave operator straight through, without the per-hop collapse-and-
    reinject `chain()` itself does, and checks how strongly that raw
    composition still resonates with the same final entity. On a
    dedicated, low-crosstalk ontology, a genuinely clean 3-hop chain must
    stay well above the noise floor - not a claim of near-1.0 certainty
    (composing three real-valued wave operations without any intermediate
    error-correction accumulates real dispersion), but a real, positive
    signal clearly distinguishable from the near-zero this project's own
    `test_stored_and_guessed_coherence_are_well_separated`-style crosstalk
    floor represents."""
    onto = Ontology(dim=8192, seed=7)
    onto.add("socrates", "is_a", "human")
    onto.add("human", "is_a", "mortal")
    onto.add("mortal", "is_a", "thing")
    memory = onto.ground()
    c = chain(onto, memory, "socrates", "is_a")
    assert len(c.hops) == 3
    assert c.resonance_coherence > 0.3


def test_chain_resonance_coherence_is_zero_with_no_hops():
    onto = demo_ontology()
    memory = onto.ground()
    c = chain(onto, memory, "dragon", "is_a")  # unknown subject -> no hops
    assert c.hops == []
    assert c.resonance_coherence == 0.0


def test_chain_resonance_coherence_carries_information_cumulative_does_not():
    """Not a cosmetic duplicate of the existing per-hop-multiplied
    `cumulative` product - a genuinely independent signal, checked directly
    rather than assumed: on the demo ontology's own 3-hop chain, the two
    numbers are not close to each other at all (the compounded per-hop
    product decays sharply over three hops; the whole-chain resonance
    check does not)."""
    onto = demo_ontology()
    memory = onto.ground()
    c = chain(onto, memory, "socrates", "is_a")
    assert len(c.hops) == 3
    assert abs(c.resonance_coherence - c.cumulative[-1]) > 0.3


def test_prior_coherence_defaults_to_an_exact_no_op():
    """Every pre-existing call site passes no prior - the default must be
    bit-for-bit what those calls always did, not merely close."""
    onto = demo_ontology()
    memory = onto.ground()
    without = chain(onto, memory, "socrates", "is_a")
    explicit_one = chain(onto, memory, "socrates", "is_a", prior_coherence=1.0)
    assert [n for n, _ in without.hops] == [n for n, _ in explicit_one.hops]
    assert without.cumulative == explicit_one.cumulative


def test_prior_coherence_compounds_into_cumulative():
    """The generalise-then-deduce composition (see docs/ROADMAP.md): a
    chain whose starting entity was itself *inferred* at some coherence
    should report that uncertainty compounded through every hop, rather
    than reporting the deduction as though it had started from certainty.
    Uses the identical multiplicative convention already used between
    hops, so the whole cumulative series just scales by the prior."""
    onto = demo_ontology()
    memory = onto.ground()
    prior = 0.5
    baseline = chain(onto, memory, "socrates", "is_a")
    seeded = chain(onto, memory, "socrates", "is_a", prior_coherence=prior)

    # Same trajectory - a prior is about confidence, not about which path is walked.
    assert [n for n, _ in seeded.hops] == [n for n, _ in baseline.hops]
    assert [c for _, c in seeded.hops] == [c for _, c in baseline.hops]
    # ...but every cumulative entry is scaled by the prior.
    for base_c, seeded_c in zip(baseline.cumulative, seeded.cumulative):
        assert abs(seeded_c - base_c * prior) < 1e-12


def test_prior_coherence_does_not_gate_the_per_hop_coherence_floor():
    """A deliberate design decision, pinned so it isn't 'tidied' later: the
    per-hop COHERENCE_FLOOR check asks 'does *this hop* resonate against
    the memory' - a question about hop evidence quality, not about whether
    the chain started somewhere correct. A tiny prior (here far below the
    floor) must therefore NOT truncate a chain whose hops are each
    individually clean; it only makes the reported cumulative confidence
    honestly small."""
    onto = demo_ontology()
    memory = onto.ground()
    tiny_prior = COHERENCE_FLOOR / 100.0
    baseline = chain(onto, memory, "socrates", "is_a")
    seeded = chain(onto, memory, "socrates", "is_a", prior_coherence=tiny_prior)

    assert len(baseline.hops) == 3
    assert [n for n, _ in seeded.hops] == [n for n, _ in baseline.hops]  # not truncated
    assert seeded.cumulative[-1] < COHERENCE_FLOOR  # but honestly reported as weak


def test_coherence_floor_defaults_to_an_exact_no_op():
    onto = demo_ontology()
    memory = onto.ground()
    default = chain(onto, memory, "socrates", "is_a")
    explicit = chain(onto, memory, "socrates", "is_a", coherence_floor=COHERENCE_FLOOR)
    assert [n for n, _ in default.hops] == [n for n, _ in explicit.hops]
    assert default.cumulative == explicit.cumulative


def test_lowering_coherence_floor_admits_hops_the_default_rejects():
    """The seam docs/ROADMAP.md's compounding-calibration experiment needs:
    with the default floor every recorded hop has already passed the
    calibrated single-hop gate, so chains are ~100% valid and contain no
    errors for a confidence signal to discriminate. Dropping the floor to
    0 lets the chain keep walking past that gate - the only way to get
    chains that contain genuine mistakes."""
    onto = demo_ontology()
    memory = onto.ground()
    default = chain(onto, memory, "dragon", "is_a")  # unknown subject: no hops clear the floor
    assert default.hops == []

    admitted = chain(onto, memory, "dragon", "is_a", coherence_floor=0.0)
    assert len(admitted.hops) > 0  # now it walks anyway, honestly low-coherence
    assert all(coh < COHERENCE_FLOOR for _, coh in admitted.hops[:1])


def test_min_hop_coherence_is_the_weakest_hop():
    onto = demo_ontology()
    memory = onto.ground()
    c = chain(onto, memory, "socrates", "is_a")
    assert len(c.hops) == 3
    assert c.min_hop_coherence() == min(coh for _, coh in c.hops)


def test_min_hop_coherence_is_one_for_an_empty_chain():
    """No hops means nothing weakened the chain - matches chain()'s own
    prior_coherence identity of 1.0 rather than returning 0.0, which would
    read as 'maximally unreliable' for a chain that simply never ran."""
    onto = demo_ontology()
    memory = onto.ground()
    c = chain(onto, memory, "dragon", "is_a")
    assert c.hops == []
    assert c.min_hop_coherence() == 1.0


def test_min_hop_coherence_does_not_decay_with_chain_length():
    """The property that makes it preferable to `cumulative` as a
    chain-level reliability signal (see docs/ROADMAP.md): cumulative
    shrinks with every additional hop regardless of hop quality, so it is
    not comparable across chains of different lengths. The weakest-hop
    signal only moves when a genuinely weaker hop appears."""
    onto = demo_ontology()
    memory = onto.ground()
    c = chain(onto, memory, "socrates", "is_a")
    assert len(c.hops) == 3
    # cumulative strictly decays hop over hop...
    assert c.cumulative[-1] < c.cumulative[0]
    # ...while the weakest-hop signal stays at the scale of a single hop.
    assert c.min_hop_coherence() >= min(c.cumulative)
    assert c.min_hop_coherence() == min(coh for _, coh in c.hops)
