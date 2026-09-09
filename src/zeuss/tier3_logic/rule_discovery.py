"""Bridge tier4's boolean synthesis into tier3 `Rule`/`Theory` objects - the
Frontier 2 gap `energy.py`/`grounding.py` themselves have no answer for at
all: nothing anywhere lets a `Rule`'s antecedent be *discovered* rather than
hand-typed by calling code (`Rule(antecedent, consequent, weight)` is always
built from literal strings supplied by a human).

But a rule's antecedent is structurally identical to what `tier4_synthesis.
semantic_bias`'s boolean backpropagation already discovers (v0.31 - 30/30
leap-year seeds): a compound boolean formula over named variables that
matches a given set of (valuation -> outcome) examples. This module wraps
that exact mechanism - `tier4_synthesis.search.synthesize` with
`allow_semantic_bias=True` - to turn example rows like `{rain: True,
sprinkler: False} -> wet: True` into a genuine discovered antecedent, the
concrete, testable version of "rules emerge from data" rather than being
declared.

**Why a new module instead of extending `Rule` itself:** `compiler.py`'s
`Rule.antecedent` is deliberately just a dict key (`valuation.get(self.
antecedent, 0.0)`) - the simplest possible contract, with no compound-
expression case to support, and every existing theory/test in this codebase
relies on that simplicity. Rather than complicate `Rule` itself, a
discovered antecedent is registered as its own *synthetic* named variable
(e.g. `_discovered_wet_antecedent`), computed from the real (base) variables
via `dsl.evaluate`, and :func:`expand_valuation` injects that computed value
into a valuation dict before it reaches `Theory.energy` - `Rule`/`Theory`
never need to know the antecedent they're scoring was discovered rather than
declared.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..tier2_substrate.energy import Landscape
from ..tier4_synthesis.dsl import Fuel, Node, evaluate
from ..tier4_synthesis.search import Example as SynthExample
from ..tier4_synthesis.search import synthesize
from .compiler import Rule, Theory
from .grounding import WEIGHT_FLOOR, valuation_to_hypervector


def discover_antecedent(
    base_variables: list[str],
    examples: list[tuple[dict[str, bool], bool]],
    rng: np.random.Generator | None = None,
    population_size: int = 300,
    max_generations: int = 150,
    max_depth: int = 4,
) -> tuple[Node | None, bool]:
    """Discover a compound boolean formula over ``base_variables`` matching
    every ``(valuation, outcome)`` row in ``examples``, via
    :func:`tier4_synthesis.search.synthesize`'s semantic backpropagation
    (``allow_semantic_bias=True``, ``allow_bool_template=False`` - the
    general mechanism itself must do the discovering, not the hand-built
    template). Returns ``(node, verified)`` - ``node`` is ``None`` only if
    the search's own generation-time mechanisms found nothing to return at
    all (practically never, per :func:`.semantic_bias.boolean_backprop_
    template`'s own contract); ``verified`` is ``False`` if the winning
    candidate doesn't actually match every given example (this project's
    standing "verified means matched the examples, never proven correct"
    honesty statement - see ``tier4_synthesis/synth.py``), the caller's cue
    to not trust the discovered rule.
    """
    rng = np.random.default_rng() if rng is None else rng
    synth_examples = [SynthExample(dict(inputs), bool(outcome)) for inputs, outcome in examples]
    best, _trace, verified = synthesize(
        base_variables,
        synth_examples,
        population_size=population_size,
        max_generations=max_generations,
        max_depth=max_depth,
        allow_bool_template=False,
        allow_semantic_bias=True,
        rng=rng,
    )
    return best, verified


@dataclass
class DiscoveredRule:
    """A `Rule` whose antecedent is a synthetic name standing in for a
    discovered compound formula (`derived`) over the real base variables -
    see :func:`expand_valuation` for how that formula actually gets scored."""

    rule: Rule
    derived: dict[str, Node]
    verified: bool


def discover_rule(
    base_variables: list[str],
    examples: list[tuple[dict[str, bool], bool]],
    consequent: str,
    weight: float = 1.0,
    rng: np.random.Generator | None = None,
    **synth_kwargs,
) -> DiscoveredRule:
    """:func:`discover_antecedent` wrapped into a ready-to-compile
    :class:`DiscoveredRule`. The synthetic antecedent name is derived
    deterministically from ``consequent`` (``f"_discovered_{consequent}_
    antecedent"``) so two calls discovering rules for the same consequent
    don't silently collide on an accidental name reuse from elsewhere."""
    node, verified = discover_antecedent(base_variables, examples, rng=rng, **synth_kwargs)
    antecedent_name = f"_discovered_{consequent}_antecedent"
    rule = Rule(antecedent=antecedent_name, consequent=consequent, weight=weight)
    derived = {antecedent_name: node} if node is not None else {}
    return DiscoveredRule(rule=rule, derived=derived, verified=verified)


def expand_valuation(derived: dict[str, Node], valuation: dict[str, float], fuel_budget: int = 60) -> dict[str, float]:
    """Compute every synthetic antecedent's value from ``valuation``'s real
    (base) variables and merge it in, returning a new dict - the step that
    lets :class:`DiscoveredRule`'s `Rule` (whose antecedent is just that
    synthetic name) be scored by the ordinary, unmodified `Theory.energy`/
    `Rule.truth`. ``valuation``'s own values are rounded to booleans before
    evaluating (exact and lossless for the crisp ``0.0``/``1.0`` corners
    :func:`.grounding._boolean_corners` enumerates; an approximation for a
    genuinely fractional valuation, since the discovered formula is a
    boolean DSL expression, not a fuzzy one). A crash or non-boolean result
    (should not happen for a `verified` discovered rule, but this project's
    own practice - see `tier4_synthesis/search.py`'s `program_energy` - is
    to never let a candidate's evaluation crash the caller) is treated as
    ``False``, not propagated."""
    if not derived:
        return dict(valuation)
    env = {name: bool(round(min(1.0, max(0.0, v)))) for name, v in valuation.items()}
    out = dict(valuation)
    for name, node in derived.items():
        try:
            result = evaluate(node, env, Fuel(fuel_budget))
            out[name] = 1.0 if (isinstance(result, bool) and result) else 0.0
        except Exception:
            out[name] = 0.0
    return out


def compile_discovered_theory(
    codebook,
    discovered: list[DiscoveredRule],
    variables: list[str],
    fixed: dict[str, float] | None = None,
    inverse_temperature: float = 1.0,
) -> Landscape:
    """The discovered-rule counterpart to :func:`.grounding.compile_theory`:
    same exhaustive-corner-enumeration/Boltzmann-weighting/``WEIGHT_FLOOR``
    algorithm, over ``variables`` (the *base* free variables only - the
    synthetic antecedent names are never added to the grounded hypervector's
    own slots, so a compiled :class:`~zeuss.tier2_substrate.energy.Landscape`
    here is directly comparable/interchangeable with an ordinary
    ``compile_theory`` result over the same base variables), except each
    corner's valuation is expanded (:func:`expand_valuation`) with every
    discovered rule's synthetic antecedent value before scoring. A separate
    function rather than a modification to ``compile_theory`` itself - which
    has no notion of "derived" names and is already relied on, unmodified,
    by every existing test in `test_grounding.py`.

    Raises ``ValueError`` if any `DiscoveredRule` in ``discovered`` didn't
    verify (see :func:`discover_rule`) - compiling an unverified discovered
    rule into a Landscape would silently promote a guess to an axiom, the
    opposite of this project's honesty conventions.
    """
    import itertools
    import math

    for d in discovered:
        if not d.verified:
            raise ValueError(
                f"DiscoveredRule for consequent {d.rule.consequent!r} did not verify - "
                "refusing to compile an unverified discovered rule as if it were an axiom"
            )

    rules = [d.rule for d in discovered]
    derived: dict[str, Node] = {}
    for d in discovered:
        derived.update(d.derived)
    theory = Theory(rules=rules)

    landscape = Landscape()
    for bits in itertools.product((0.0, 1.0), repeat=len(variables)):
        corner = dict(zip(variables, bits))
        valuation = expand_valuation(derived, {**corner, **(fixed or {})})
        corner_energy = theory.energy(valuation)
        weight = math.exp(-inverse_temperature * corner_energy)
        if weight > WEIGHT_FLOOR:
            vector = valuation_to_hypervector(codebook, variables, corner)
            landscape.add(vector, weight=weight)
    return landscape
