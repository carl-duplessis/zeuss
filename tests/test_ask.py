"""The ask read-out: stored facts resonate; unknown queries are flagged."""
from zeuss.qa import COHERENCE_FLOOR, ask, demo_ontology


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
