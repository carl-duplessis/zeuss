"""Mine candidate logical axioms directly from an Ontology's own facts.

`qa.py`'s v0.37 `axiom_bias` hook lets a hand-written `Rule` veto a candidate
answer that contradicts an already-established fact - but the caller had to
write the `Rule` and wire up which `ask()` result fills in which variable, by
hand, per domain. This module is the automatic-discovery counterpart: plain
support/confidence association-rule mining (Agrawal et al.) over an
`Ontology`'s stored triples, restated in this project's own `Rule` vocabulary
so a mined pattern becomes an ordinary weighted `Rule`, not a bespoke
statistic - the same "reuse the existing formalism, don't invent a new one"
discipline `grounding.compile_theory`'s `exp(-energy)` weighting already
established.

Frontier 2 tie-in: `axiom_bias_from_exclusions` builds a `qa.py`-compatible
callable straight from what's discovered here, closing the loop from "the
ontology's own data" to "a Landscape-biasing energy term" with no hand-typed
Rule in between.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .compiler import Rule
from .ontology import Ontology


def _relation_index(onto: Ontology, relation: str) -> dict[str, set[str]]:
    """subject -> set of fillers stored under ``relation``.

    A subject with more than one filler is a genuine data contradiction
    (e.g. ``socrates is_a`` both ``human`` and ``star``) - it still counts
    toward *every* filler it holds when mining, so one contradictory
    subject dilutes but does not usually flip a pattern that is real
    elsewhere in the ontology (see `docs/ROADMAP.md` v0.38 for the measured
    case this was designed against: support/confidence stayed at 1.0 with
    one contradictory subject mixed into ten clean ones).
    """
    index: dict[str, set[str]] = {}
    for subject, rel, obj in onto.triples:
        if rel == relation:
            index.setdefault(subject, set()).add(obj)
    return index


@dataclass(frozen=True)
class DiscoveredImplication:
    """``antecedent_relation=antecedent_filler`` implies
    ``consequent_relation=consequent_filler``, mined from ``support`` many
    subjects at ``confidence`` (the fraction of them where it actually held).

    Frozen (and thus hashable) since these are plain value objects, never
    mutated after mining - useful for de-duplicating/set-comparing results.
    """

    antecedent_relation: str
    antecedent_filler: str
    consequent_relation: str
    consequent_filler: str
    support: int
    confidence: float


def discover_implications(
    onto: Ontology,
    antecedent_relation: str,
    consequent_relation: str,
    min_support: int = 2,
    min_confidence: float = 0.8,
) -> list[DiscoveredImplication]:
    """Mine ``antecedent_relation=f1 -> consequent_relation=f2`` patterns via
    plain support/confidence over ``onto``'s own stored triples.

    ``support`` = how many subjects have ``antecedent_relation=f1`` at all;
    ``confidence`` = what fraction of those *also* have
    ``consequent_relation=f2``. Both are the standard association-rule
    statistics, not bespoke ones - `min_support` guards against mining
    "patterns" from one or two coincidental subjects.
    """
    ante_index = _relation_index(onto, antecedent_relation)
    cons_index = _relation_index(onto, consequent_relation)

    ante_fillers = {f for fillers in ante_index.values() for f in fillers}
    cons_fillers = {f for fillers in cons_index.values() for f in fillers}

    found = []
    for f_ante in ante_fillers:
        subjects_with_ante = {s for s, fillers in ante_index.items() if f_ante in fillers}
        support = len(subjects_with_ante)
        if support < min_support:
            continue
        for f_cons in cons_fillers:
            hits = sum(1 for s in subjects_with_ante if f_cons in cons_index.get(s, set()))
            confidence = hits / support
            if confidence >= min_confidence:
                found.append(
                    DiscoveredImplication(antecedent_relation, f_ante, consequent_relation, f_cons, support, confidence)
                )
    return found


@dataclass(frozen=True)
class DiscoveredExclusion:
    """``antecedent_relation=antecedent_filler`` excludes
    ``consequent_relation=excluded_filler`` - the corollary of a
    :class:`DiscoveredImplication` toward some *other* filler of the same
    relation (see :func:`discover_exclusions`). Frozen/hashable for the
    same reason as :class:`DiscoveredImplication`."""

    antecedent_relation: str
    antecedent_filler: str
    consequent_relation: str
    excluded_filler: str
    confidence: float


def discover_exclusions(
    onto: Ontology,
    antecedent_relation: str,
    consequent_relation: str,
    min_support: int = 2,
    min_confidence: float = 0.8,
) -> list[DiscoveredExclusion]:
    """For every :class:`DiscoveredImplication` toward one specific filler of
    ``consequent_relation``, derive a veto against every *other* filler that
    relation is observed to take elsewhere in the ontology.

    This is the natural corollary once a strong implication toward exactly
    one filler is established: if subjects with the antecedent fact
    essentially always land on ``f_cons``, and other subjects (without that
    antecedent) land on a different filler ``other``, then the antecedent is
    evidence *against* ``other`` too - the concrete "is_a-like relations are
    normally single-valued" assumption, checked statistically rather than
    hard-coded as a schema constraint.
    """
    cons_index = _relation_index(onto, consequent_relation)
    all_cons_fillers = {f for fillers in cons_index.values() for f in fillers}

    exclusions = []
    for imp in discover_implications(onto, antecedent_relation, consequent_relation, min_support, min_confidence):
        for other in all_cons_fillers - {imp.consequent_filler}:
            exclusions.append(
                DiscoveredExclusion(imp.antecedent_relation, imp.antecedent_filler, imp.consequent_relation, other, imp.confidence)
            )
    return exclusions


def discover_all_exclusions(
    onto: Ontology,
    min_support: int = 2,
    min_confidence: float = 0.8,
) -> list[DiscoveredExclusion]:
    """The fully-automatic counterpart to :func:`discover_exclusions`: try
    every ordered pair of distinct relations actually used in ``onto`` as
    ``(antecedent_relation, consequent_relation)`` and keep whichever mined
    exclusions clear the thresholds, instead of the caller naming both
    relations by hand.

    Closes v0.38's own honestly-scoped gap ("nothing here decides *which*
    relation pairs to mine") - "point it at an ontology, get every axiom
    that holds out", with `min_support`/`min_confidence` as the only knobs
    left. `O(R^2)` relation pairs, each `O(F^2)` fillers - fine at the scale
    this project's own ontologies operate at; not meant for a KB with
    hundreds of relations.
    """
    relations = sorted({relation for _, relation, _ in onto.triples})
    exclusions = []
    for antecedent_relation in relations:
        for consequent_relation in relations:
            if antecedent_relation == consequent_relation:
                continue
            exclusions.extend(
                discover_exclusions(onto, antecedent_relation, consequent_relation, min_support, min_confidence)
            )
    return exclusions


def axiom_bias_from_ontology(
    onto: Ontology,
    memory,
    subject: str,
    target_relation: str,
    min_support: int = 2,
    min_confidence: float = 0.8,
) -> Callable[[str], float]:
    """Fully automatic ``axiom_bias`` (v0.37): mines every relation pair via
    :func:`discover_all_exclusions`, keeps only the exclusions relevant to
    ``target_relation``, and hands the result straight to
    :func:`axiom_bias_from_exclusions` - so ``ask``/``chain`` can take this
    with zero hand-wiring of which relations to check, closing the loop
    from "an Ontology" to "a biased query" completely.
    """
    exclusions = [
        excl
        for excl in discover_all_exclusions(onto, min_support, min_confidence)
        if excl.consequent_relation == target_relation
    ]
    return axiom_bias_from_exclusions(onto, memory, subject, target_relation, exclusions)


def axiom_bias_from_exclusions(
    onto: Ontology,
    memory,
    subject: str,
    target_relation: str,
    exclusions: list[DiscoveredExclusion],
) -> Callable[[str], float]:
    """Build a `qa.py`-compatible ``axiom_bias`` callable (v0.37) straight
    from mined :class:`DiscoveredExclusion`\\ s - the automatic-discovery
    counterpart to hand-writing one.

    For a candidate ``c``, sums a `Rule` penalty over every exclusion whose
    ``excluded_filler`` is ``c`` and ``consequent_relation`` is
    ``target_relation``, evaluating each exclusion's antecedent via a
    *separate* `ask()` call about ``subject``.

    Gated on ``known``, not ``confidence`` - measured directly, not assumed:
    a first version gated on `Answer.confidence` alone and silently failed
    once the ontology grew past a handful of entities (`confidence` is a
    *relative* softmax share against every other candidate, which dilutes
    toward 0 as the candidate count grows, even though the antecedent fact
    is just as objectively true; `known`'s absolute floor is the correct
    gate - see `docs/ROADMAP.md` v0.38).
    """
    from ..qa import ask  # local import: avoids a qa <-> tier3_logic import cycle at module load

    def bias(candidate: str) -> float:
        total = 0.0
        for excl in exclusions:
            if excl.consequent_relation != target_relation or excl.excluded_filler != candidate:
                continue
            antecedent_answer = ask(onto, memory, subject, excl.antecedent_relation)
            antecedent_truth = (
                excl.confidence
                if (antecedent_answer.known and antecedent_answer.answer == excl.antecedent_filler)
                else 0.0
            )
            rule = Rule(antecedent="_antecedent", consequent="_not_excluded", weight=excl.confidence)
            # candidate IS excl.excluded_filler here, so "not excluded" is false.
            total += rule.penalty({"_antecedent": antecedent_truth, "_not_excluded": 0.0})
        return total

    return bias
