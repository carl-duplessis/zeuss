"""Compile a fuzzy-logic ``Theory`` into a Tier-2 energy ``Landscape``.

A propositional variable becomes a pair of poles on the phase torus,
``VAR:x`` bound to a ``TRUE`` filler and to a ``FALSE`` filler; a continuous
truth value ``v in [0, 1]`` for that variable is the ``bundle`` (weighted
superposition) of the two poles with weights ``(v, 1 - v)`` - the same
interpolation-by-weighted-bundle idea already used everywhere else in Tier 2,
just applied to a truth value instead of a set of candidate objects. A joint
valuation over several variables bundles every variable's pole pair together
into one hypervector.

``compile_theory`` enumerates the theory's Boolean corners (``O(2**n)`` -
appropriate for ``Theory``'s expected small-``n`` scale, and stated as such),
scores each with ``Theory.energy``, and registers every low-energy corner as a
``Landscape`` attractor weighted by ``exp(-energy)`` (a Boltzmann weighting,
mirroring the softmax-of-negative-energy idea already used in
``Landscape.energy`` and ``collapse.softmax``, just applied once at compile
time). Settling from any start state then relaxes toward the theory's
satisfying region - the axiom *is* the basin, per :mod:`zeuss.tier2_substrate.energy`.

``compile_theories`` extends this to *several* named theories (e.g. one per
agent/subsystem) that may share variables: it runs the same sheaf consistency
audit :func:`zeuss.tier3_logic.sheaf.from_theories` uses (and ``zeuss audit``
demos) *before* compiling, so two theories that are each individually
satisfiable but disagree on a shared variable raise
:class:`InconsistentTheoriesError` with the specific violated edges, instead
of silently blending the contradiction into an unexplained fuzzy landscape.
"""
from __future__ import annotations

import itertools
import math

from ..tier2_substrate.energy import Landscape
from ..tier2_substrate.hypervectors import Codebook, bind, bundle, similarity
from .compiler import Theory
from .sheaf import TRUE_ANCHOR, SheafGraph, from_theories

# Below this Boltzmann weight a corner is treated as effectively zero-probability
# and not worth registering as a separate attractor.
WEIGHT_FLOOR = 1e-6


class InconsistentTheoriesError(ValueError):
    """Raised by :func:`compile_theories` when agents' theories disagree on a
    shared variable. ``violations`` lists the specific ``(agent_a, agent_b,
    residual)`` edges that failed to agree; ``graph`` is the full
    :class:`~zeuss.tier3_logic.sheaf.SheafGraph` for further inspection."""

    def __init__(self, graph: SheafGraph) -> None:
        residual = graph.local_section(graph.local_values)
        self.violations = [
            (u, v, float(r))
            for (u, v, _ru, _rv), r in zip(graph.edges, residual)
            if abs(r) > 1e-9
        ]
        self.graph = graph
        detail = "; ".join(f"{u} != {v} (residual={r:+.3f})" for u, v, r in self.violations)
        super().__init__(f"theories disagree on shared variables: {detail}")


def valuation_to_hypervector(codebook: Codebook, variables: list[str], valuation: dict[str, float]):
    """Represent a ``[0, 1]``-valued valuation as one joint hypervector.

    Each variable's two poles are bound under a shared ``VAR:<name>`` role wave
    so every variable lives in its own slot of the same vector, then all
    variables' pole-pairs are bundled together.
    """
    parts = []
    weights = []
    for name in variables:
        v = float(min(1.0, max(0.0, valuation.get(name, 0.0))))
        role = codebook.symbol(f"VAR:{name}")
        true_pole = bind(role, codebook.symbol(f"{name}:TRUE"))
        false_pole = bind(role, codebook.symbol(f"{name}:FALSE"))
        parts.append(true_pole)
        parts.append(false_pole)
        weights.append(v)
        weights.append(1.0 - v)
    return bundle(parts, weights=weights)


def _boolean_corners(variables: list[str]):
    for bits in itertools.product((0.0, 1.0), repeat=len(variables)):
        yield dict(zip(variables, bits))


def compile_theory(
    codebook: Codebook, theory: Theory, variables: list[str], fixed: dict[str, float] | None = None
) -> Landscape:
    """Sample the theory's low-energy corners and register them as attractors.

    Scoped to exhaustive enumeration over ``2 ** len(variables)`` Boolean
    corners - matches ``Theory``'s expected scale (a handful of named
    propositions). A continuous-relaxation variant (fewer samples + gradient
    refinement) is the natural upgrade once ``energy.settle_grad`` lands.

    ``fixed`` optionally pins extra named values into every corner's
    valuation before scoring (e.g. the ``TRUE_ANCHOR`` sentinel some rules
    reference as a fixed bias point rather than a free variable - see
    :func:`compile_theories`); it does not add those names to the grounded
    hypervector's own variable slots.
    """
    landscape = Landscape()
    for corner in _boolean_corners(variables):
        valuation = {**corner, **(fixed or {})}
        energy = theory.energy(valuation)
        weight = math.exp(-energy)
        if weight > WEIGHT_FLOOR:
            vector = valuation_to_hypervector(codebook, variables, valuation)
            landscape.add(vector, weight=weight)
    return landscape


def compile_theories(
    codebook: Codebook, theories: dict[str, Theory], shared_vars: dict[str, list[str]]
) -> Landscape:
    """Audit several agents' theories for agreement, then compile the union.

    Runs :func:`zeuss.tier3_logic.sheaf.from_theories` first - the same check
    ``zeuss audit`` demos - and raises :class:`InconsistentTheoriesError` if
    any two agents' independently-derived best local valuations disagree on a
    shared variable. On success, merges every agent's rules into one
    ``Theory`` over the union of every variable referenced (any rule using
    the ``TRUE_ANCHOR`` sentinel - see ``sheaf.py`` - keeps it pinned to 1.0
    rather than treated as a free variable) and compiles that via
    :func:`compile_theory`.
    """
    graph = from_theories(theories, shared_vars)
    if not graph.is_consistent_with():
        raise InconsistentTheoriesError(graph)

    all_rules = [rule for theory in theories.values() for rule in theory.rules]
    referenced = {
        name
        for theory in theories.values()
        for rule in theory.rules
        for name in (rule.antecedent, rule.consequent)
    }
    shared = {var for variables in shared_vars.values() for var in variables}
    free_variables = sorted((referenced | shared) - {TRUE_ANCHOR})

    combined = Theory(rules=all_rules)
    return compile_theory(codebook, combined, free_variables, fixed={TRUE_ANCHOR: 1.0})


def readout(codebook: Codebook, variables: list[str], z) -> dict[str, float]:
    """Invert :func:`valuation_to_hypervector`: recover each variable's truth value.

    For each variable, the similarity margin between its ``TRUE`` and
    ``FALSE`` poles lies in ``[-2, 2]`` (each pole similarity is itself in
    ``[-1, 1]``); remapped linearly onto ``[0, 1]``.
    """
    out = {}
    for name in variables:
        role = codebook.symbol(f"VAR:{name}")
        true_sim = similarity(z, bind(role, codebook.symbol(f"{name}:TRUE")))
        false_sim = similarity(z, bind(role, codebook.symbol(f"{name}:FALSE")))
        margin = true_sim - false_sim
        out[name] = min(1.0, max(0.0, (margin + 2.0) / 4.0))
    return out
