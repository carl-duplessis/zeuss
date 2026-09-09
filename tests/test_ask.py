"""The ask read-out: stored facts resonate; unknown queries are flagged."""
from zeuss.qa import COHERENCE_FLOOR, ask, demo_ontology
from zeuss.tier3_logic.ontology import Ontology


def test_known_facts_are_answered_with_high_coherence():
    onto = demo_ontology()
    memory = onto.ground()
    for subject, relation, expected in [
        ("socrates", "is_a", "human"),
        ("plato", "is_a", "human"),
        ("sky", "has_color", "blue"),
        ("sun", "is_a", "star"),
    ]:
        answer = ask(onto, memory, subject, relation)
        assert answer.answer == expected
        assert answer.known
        assert answer.coherence > 0.2


def test_unknown_queries_are_flagged_as_guesses():
    onto = demo_ontology()
    memory = onto.ground()
    for subject, relation in [
        ("dragon", "is_a"),             # unknown subject
        ("socrates", "has_color"),      # wrong relation for this subject
        ("socrates", "is_mortal_via"),  # would need multi-hop chaining
    ]:
        answer = ask(onto, memory, subject, relation)
        assert not answer.known
        assert answer.coherence < COHERENCE_FLOOR


def test_stored_and_guessed_coherence_are_well_separated():
    onto = demo_ontology()
    memory = onto.ground()
    known = ask(onto, memory, "socrates", "is_a").coherence
    guess = ask(onto, memory, "dragon", "is_a").coherence
    # A stored fact should ring at least 5x louder than pure crosstalk.
    assert known > 5 * abs(guess)


def test_known_facts_shrink_the_live_comparison_set():
    """Frontier 1's dimensional_collapse, wired into qa.py's _cleanup (see
    its docstring, v0.34): a known fact should genuinely restrict the
    candidate-comparison set, not just report a low entropy number - the
    literal "space collapses to fewer dimensions as it becomes more
    certain" behaviour, checked as an observable count, not assumed."""
    onto = demo_ontology()
    memory = onto.ground()
    total = len(onto.entity_names())
    for subject, relation in [("socrates", "is_a"), ("sky", "has_color"), ("sun", "is_a")]:
        answer = ask(onto, memory, subject, relation)
        assert answer.known
        assert answer.k_live < total
        assert answer.eff_dim < total
        assert len(answer.ranked) == answer.k_live


def test_unknown_queries_do_not_falsely_shrink_the_comparison_set():
    """The flip side: a genuine guess has no real winner for entropy to
    collapse toward, so the live basis should stay at (or very near) the
    full entity set - dimensional_collapse must not be fooled into reporting
    false confidence just because a fixed high beta sharpens noise."""
    onto = demo_ontology()
    memory = onto.ground()
    total = len(onto.entity_names())
    for subject, relation in [("dragon", "is_a"), ("socrates", "has_color")]:
        answer = ask(onto, memory, subject, relation)
        assert not answer.known
        assert answer.k_live >= total - 1


def _chain_ontology(dim, seed):
    """A deliberately crosstalk-heavy ontology (small dim, 40 chained
    entities all bundled into one memory) - the demo KB's dim=8192 has
    almost no crosstalk for energy settling to correct, so this is the
    regime _cleanup's v0.36 Frontier-2 wiring was actually measured against."""
    onto = Ontology(dim=dim, seed=seed)
    names = [f"e{i}" for i in range(40)]
    rels = ["r0", "r1"]
    for i in range(len(names) - 1):
        onto.add(names[i], rels[i % 2], names[i + 1])
    return onto, names, rels


def test_energy_settling_corrects_a_crosstalk_error_without_regressing():
    """v0.36: _cleanup's winning candidate now comes from settling the
    residue against a Landscape built from the live basis (Frontier 2),
    not the raw one-shot residue - see _cleanup's docstring for why this is
    a genuine mechanism, not cosmetic. Measured directly on this exact
    ontology before wiring it in: settle-informed selection recovers 1 of 3
    cases plain phase_lock cleanup gets wrong here, and regresses none of
    the 36 it already got right. This test only asserts the regression-
    safety half directly (every previously-correct case stays correct) -
    the correction itself is a real but modest effect, not overstated."""
    onto, names, rels = _chain_ontology(dim=512, seed=3)
    memory = onto.ground()

    # The specific case settling corrects (verified against real ask(), not
    # just the offline experiment): plain one-shot phase_lock cleanup picks
    # 'e39' here; settling against the live-basis Landscape recovers 'e20'.
    assert ask(onto, memory, "e19", "r1").answer == "e20"

    regressions = []
    for i in range(len(names) - 1):
        subject, relation, expected = names[i], rels[i % 2], names[i + 1]
        answer = ask(onto, memory, subject, relation)
        # A case this mechanism doesn't fix is an acceptable, documented
        # limitation - but a case that was *already right* turning wrong
        # would be a real regression, which is what this guards against.
        if answer.answer != expected and (subject, relation) not in {("e4", "r0"), ("e18", "r0")}:
            regressions.append((subject, relation, expected, answer.answer))
    assert regressions == []


def test_energy_settling_does_not_inflate_guessed_coherence():
    """Regression guard for a real failure mode found while designing v0.36:
    a first attempt read coherence/confidence from the *settled* state, and
    settle()'s dynamics are a self-reinforcing attractor network by
    construction - any residue, including pure noise, drifts toward
    amplitude ~1.0 against whichever candidate it happens to lean toward
    first. Measured directly: that attempt collapsed demo_ontology's own
    'dragon is_a ?' guess to coherence +1.000 (known=True), silently
    breaking test_unknown_queries_are_flagged_as_guesses. Coherence/
    confidence must keep coming from the untouched original residue."""
    onto = demo_ontology()
    memory = onto.ground()
    answer = ask(onto, memory, "dragon", "is_a")
    assert not answer.known
    assert answer.coherence < COHERENCE_FLOOR
    assert answer.coherence < 0.1  # nowhere near settle()'s self-reinforced ~1.0
