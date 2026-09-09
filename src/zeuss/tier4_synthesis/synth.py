"""Public entry point: synthesize and verify a program against I/O examples.

**Honesty statement.** This synthesizes and verifies small programs in a
closed, total DSL against example-based specs. "Verified" means the winning
candidate is re-checked against every given example (and, in tests, against
held-out examples too) - it does *not* mean formally proven correct for all
possible inputs, and this module makes no such claim. `docs/ADAMAI_SPEC.md`'s
Part 2 Tier 4 language ("zero-error", "absolute deterministic mathematical
proofs", "bug-free execution code") describes a materially larger and, for
general programs, unachievable goal; this is the honest, scoped, and
concretely testable version of "anneal candidate programs toward a verified
ground state" - a real mutation/crossover search with rising selection
pressure, scored by real execution against real examples, not a claim of
universal correctness.
"""
from __future__ import annotations

import numpy as np

from .dsl import Node, pretty
from .grammar_bias import GrammarBias
from .search import Example, synthesize as _synthesize


def synthesize(
    inputs: list[str],
    examples: list[Example],
    list_inputs: tuple[str, ...] = (),
    population_size: int = 200,
    max_generations: int = 60,
    max_depth: int = 4,
    allow_recursion: bool = False,
    allow_bool_template: bool = False,
    allow_fold_template: bool = True,
    allow_motif_bias: bool = False,
    allow_recursion_template: bool = True,
    allow_grammar_bias: bool = False,
    grammar_bias: "GrammarBias | None" = None,
    allow_shape_elitism: bool = False,
    allow_semantic_bias: bool = False,
    allow_numeric_bias: bool = False,
    resonant_bias: bool = True,
    fuel_budget: int = 60,
    rng: np.random.Generator | None = None,
):
    """Search for a program satisfying ``examples``.

    ``allow_recursion`` opts into ``Letrec``/``Recur`` generation (off by
    default - see :func:`.search.synthesize`'s docstring for why); when set,
    ``resonant_bias`` enables resonance-guided template seeding, which can
    discover genuinely recursive solutions that uniform random search does
    not reliably find (see :mod:`.resonance_bias`). ``fuel_budget`` may need
    raising well above the default for recursive targets: the ``double_recur``
    template shape's exponential call trees exhaust a small fuel budget long
    before reaching a correct answer, even when the candidate is actually
    correct (see :func:`.search.program_energy`'s docstring).

    ``allow_bool_template`` opts into a structural bias toward compound
    boolean formulas (``AND``/``OR`` of modular-arithmetic comparisons - see
    :data:`.search.BOOL_TEMPLATE_CATEGORIES`), off by default for the same
    diversity-dilution reason ``allow_recursion`` is - it's a separate opt-in
    from ``allow_recursion`` (a target can request either, both, or neither).

    ``allow_fold_template`` (default ``True``) is a kill switch for the fold
    template, useful only for controlled comparisons against
    ``allow_motif_bias`` on the same list-processing target - leave it at the
    default otherwise. ``allow_motif_bias`` opts into motif resonance (see
    :mod:`.motif_bias`): a fourth, independent mechanism that reinforces and
    reuses actual subtrees from the population's own history instead of a
    hand-built skeleton, off by default and not a claim of matching the other
    three templates' reliability (see :func:`.search.synthesize`'s docstring
    and ``docs/ROADMAP.md`` v0.28). ``allow_recursion_template`` (default
    ``True``) is the same kind of kill switch for :func:`.search.
    _recursive_template` specifically, decoupled from ``allow_recursion``
    itself - only useful for the same kind of controlled comparison.

    ``allow_grammar_bias``/``grammar_bias`` opt into production-level PCFG
    resonance (see :mod:`.grammar_bias`): a fifth, independent mechanism that
    biases individual production choices rather than whole subtrees, off by
    default (see :func:`.search.synthesize`'s docstring and
    ``docs/ROADMAP.md`` v0.29 for the honest measured comparison). Unlike
    every other bias, ``grammar_bias`` may be supplied already-populated
    (e.g. via :func:`.grammar_bias.load_grammar_bias`) to persist learned
    structure across calls - see that docstring for the ownership rule.

    ``allow_shape_elitism`` opts into a sixth, orthogonal mechanism: per-
    family elitism keyed by an automatically-derived root-shape signature
    instead of a template's own hole-choices, built to fix a diagnosed
    shared root cause behind both bias mechanisms' failures on some targets
    (see :func:`.search.synthesize`'s docstring and ``docs/ROADMAP.md``
    v0.30 for the honest measured comparison - measured to not help either).

    ``allow_semantic_bias`` opts into a seventh, categorically different
    mechanism (see :mod:`.semantic_bias`): instead of reinforcing/protecting
    whatever blind growth already produced (all six mechanisms above), it
    decomposes a compound boolean target's actual required output through
    AND/OR/NOT's real truth-table semantics at generation time, constructing
    the shape directly from what the examples demand. Scoped to boolean
    compound targets only - see :func:`.search.synthesize`'s docstring and
    ``docs/ROADMAP.md`` v0.31 for the honest measured result.

    ``allow_numeric_bias`` opts into an eighth mechanism (see
    :mod:`.numeric_bias`) - the recursion-domain counterpart to
    ``allow_semantic_bias``: mines a recurrence directly from a dense-enough
    ``(param, output)`` example table instead of drawing hole-fillers
    randomly. Builds a genuine ``_recursive_template``-shaped node, so every
    existing recursion mechanism (hole mutation, per-family elitism,
    ``ResonantBias`` reinforcement) already applies to it - see
    :func:`.search.synthesize`'s docstring and ``docs/ROADMAP.md`` v0.32 for
    the honest measured result.

    Returns ``(best_node, beta_trace, verified)`` - see :func:`.search.synthesize`.
    """
    return _synthesize(
        inputs,
        examples,
        list_inputs=list_inputs,
        population_size=population_size,
        max_generations=max_generations,
        max_depth=max_depth,
        allow_recursion=allow_recursion,
        allow_bool_template=allow_bool_template,
        allow_fold_template=allow_fold_template,
        allow_motif_bias=allow_motif_bias,
        allow_recursion_template=allow_recursion_template,
        allow_grammar_bias=allow_grammar_bias,
        grammar_bias=grammar_bias,
        allow_shape_elitism=allow_shape_elitism,
        allow_semantic_bias=allow_semantic_bias,
        allow_numeric_bias=allow_numeric_bias,
        resonant_bias=resonant_bias,
        fuel_budget=fuel_budget,
        rng=rng,
    )


def describe(node: Node) -> str:
    """A human-readable rendering of a synthesized program."""
    return pretty(node)
