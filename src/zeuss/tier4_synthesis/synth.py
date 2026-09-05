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
from .search import Example, synthesize as _synthesize


def synthesize(
    inputs: list[str],
    examples: list[Example],
    list_inputs: tuple[str, ...] = (),
    population_size: int = 200,
    max_generations: int = 60,
    max_depth: int = 4,
    allow_recursion: bool = False,
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
        resonant_bias=resonant_bias,
        fuel_budget=fuel_budget,
        rng=rng,
    )


def describe(node: Node) -> str:
    """A human-readable rendering of a synthesized program."""
    return pretty(node)
