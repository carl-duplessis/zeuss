"""Exclusion-set builders for the two validated, exactly-reusable constraint
shapes: a structural invariant over a directed hierarchy (no cycles), and a
declared category-disjointness constraint. Both return a plain ``set[str]``
of excluded candidate names for one query subject - pass the result to
``core.rerank`` via a closure::

    excluded = no_cycle_exclusions(children, subject)
    core.rerank(candidates, lambda c: 10.0 if c in excluded else 0.0)

A third validated shape - mining a cross-relation implication via plain
support/confidence association-rule statistics - is deliberately not
reimplemented here. It needs the actual triples of a real knowledge graph to
mine from, which is domain-specific enough that duplicating it here (instead
of wherever the triples already live) wouldn't save real work.
"""
from __future__ import annotations

from typing import Iterable, Mapping


def no_cycle_exclusions(
    children: Mapping[str, Iterable[str]],
    subject: str,
    max_depth: int = 50,
) -> set[str]:
    """Every descendant of ``subject`` in a directed hierarchy, given a
    ``parent -> children`` adjacency map - entities that cannot *also* be an
    ancestor of ``subject`` without creating a cycle.

    Validated with zero false vetoes against real held-out data on both
    WordNet's hypernym hierarchy and Freebase's administrative-containment
    hierarchy; the computation itself is sound for any relation that forms a
    genuine directed acyclic graph, independent of what the hierarchy means.

    ``children[x]`` should be every entity ``y`` whose edge points *to* ``x``
    (``y`` sits one step below ``x``). Plain BFS over that adjacency map, no
    assumptions about the rest of the graph.
    """
    seen: set[str] = set()
    frontier = {subject}
    for _ in range(max_depth):
        nxt: set[str] = set()
        for node in frontier:
            for child in children.get(node, ()):
                if child not in seen:
                    nxt.add(child)
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return seen


def disjoint_category_exclusions(
    category_of: Mapping[str, str],
    disjoint_groups: Iterable[Iterable[str]],
    subject: str,
    candidates: Iterable[str],
) -> set[str]:
    """Candidates whose declared category is mutually exclusive with
    ``subject``'s own category, given a fixed partition of categories into
    pairwise-disjoint groups (e.g. "person", "animal", "plant" each belong to
    one such group - chosen in advance from domain knowledge, never fit to
    the data being queried).

    Validated with 0/122 false vetoes using WordNet's own lexicographer-file
    classification as the category source; the mechanism itself is generic -
    swap in any category assignment your domain already declares (an
    ontology's class hierarchy, a schema's type system, a taxonomy).

    Unlike ``no_cycle_exclusions``, this is NOT exactly sound by
    construction - a declared classification inherits its source's own edge
    cases (metaphor, metonymy, ambiguous boundaries: measured 0%-46.2% false
    vetoes across different relations on the same category source during
    validation). Measure your own false-veto rate against held-out data
    before trusting this on a new domain, the same way this project's own
    validation did.
    """
    subject_category = category_of.get(subject)
    if subject_category is None:
        return set()

    group_of_category: dict[str, frozenset] = {}
    for group in disjoint_groups:
        frozen = frozenset(group)
        for category in frozen:
            group_of_category[category] = frozen

    subject_group = group_of_category.get(subject_category)
    if subject_group is None:
        return set()

    excluded = set()
    for candidate in candidates:
        candidate_category = category_of.get(candidate)
        if (
            candidate_category is not None
            and candidate_category in subject_group
            and candidate_category != subject_category
        ):
            excluded.add(candidate)
    return excluded
