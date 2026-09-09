"""Numeric relationship-mining over the example table: the recursion-domain
counterpart to :mod:`.semantic_bias`'s boolean backpropagation (see
``docs/ROADMAP.md`` v0.31's own follow-up note - this is that follow-up).

Six-then-seven general mechanisms were tried against the leap-year/``2**n``
"deceptive local optimum" failures before this one (three hand-built
templates, motif resonance, grammar resonance, shape elitism, semantic
backpropagation - v0.20-v0.31). All but the last only reinforce, protect, or
reshuffle individuals that blind top-down random growth already produced -
powerless if the correct shape essentially never gets generated at all,
which is exactly what ``_recursive_template``'s own docstring already says
about recursion: "under 5% of random depth-4 trees even contain a ``Letrec``
with an ``If``-shaped body." Semantic backpropagation (v0.31) fixed this for
*boolean* compound targets by decomposing the target through AND/OR/NOT's
real semantics at generation time instead of hoping blind growth stumbles
onto the shape - this module applies the same "act at generation time,
using the data" philosophy to recursion, via a different, arithmetic-native
technique: numeric relationship-mining.

The idea: given a *contiguous* table of ``(param, expected_output)`` pairs
(the exact shape every recursion example set in this project's own test
suite already is - ``range(5)`` for ``2**n``, ``enumerate(fib)`` for
Fibonacci), a recursive relationship like ``f(n) = f(n-1) + f(n-1)`` can be
read directly off the table by testing candidate ``(step, op, delta,
combine_kind)`` combinations against the table's own values in place of
recursive calls - no actual program evaluation, recursion, or random search
needed at all when the table is dense enough to resolve every offset a
candidate needs. This is literally how a human would spot a recurrence from
a value table: compute ``y_n / y_(n-1)`` (or try ``+``, ``-``, ``*``, ``//``,
``%``) and see which relationship holds across every row it can check.

Deliberately reuses :func:`.search._build_template_node`'s exact
``choices`` schema (:data:`.search.TEMPLATE_CATEGORIES`) rather than
building its own tree - a mined result is *literally* a
``_recursive_template``-shaped node, so every piece of existing machinery
that already operates on that shape (``extract_template_choices``, hole
mutation, per-family elitism, ``ResonantBias`` reinforcement) applies to it
automatically, with zero further changes needed anywhere else in
``search.py``. ``build_template_node`` is passed in by the caller (like
:mod:`.grammar_bias`/:mod:`.semantic_bias` take their builders/vocab) to
avoid a circular import with ``search.py``.

Scope, stated honestly: only fires when the table is dense enough to
resolve a candidate's needed offsets *and* the uncoverable ("base case")
rows form a contiguous prefix starting at the table's smallest value - the
shape every recursion target in this codebase's own test suite already has.
A sparse or non-contiguous example set simply gets ``None`` back (falls
through to blind growth/`_recursive_template`, unaffected).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .dsl import Node

# Only arithmetic ops make sense for combining two integer table values into
# a third - `_BINOPS`'s comparison/boolean operators (`==`, `and`, ...) have
# no sensible role here, so this module keeps its own small, focused subset
# rather than reusing `_BINOPS` (search.py) wholesale.
_ARITH_OPS: dict[str, Callable[[int, int], "int | None"]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "//": lambda a, b: a // b if b != 0 else None,
    "%": lambda a, b: a % b if b != 0 else None,
}

# A candidate is only accepted if every row it can check matches exactly
# (this module mines an exact relationship, not a best-effort fit) *and* it
# can check at least this many rows - guards against a spurious "fit" on a
# table too small to distinguish a real relationship from luck (a 2-row
# table trivially "fits" almost any op).
_MIN_COVERED_ROWS = 3


def build_table(examples: list, var: str) -> dict[int, int] | None:
    """A ``{param_value: expected_output}`` lookup for ``var``, or ``None``
    if any example's input isn't an ``int`` (``bool`` excluded explicitly -
    it's an ``int`` subclass in Python but not a numeric recursion target)
    or any output isn't an ``int`` (this module doesn't apply to boolean or
    list-valued targets - see :mod:`.semantic_bias` for the boolean case).
    Duplicate ``var`` values silently keep the last-seen output, mirroring
    how a plain dict comprehension over the same examples would behave."""
    table: dict[int, int] = {}
    for example in examples:
        value = example.inputs.get(var)
        output = example.expected_output
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        if not isinstance(output, int) or isinstance(output, bool):
            return None
        table[value] = output
    return table


def _score_candidate(
    table: dict[int, int], step: int, combine_kind: str, op: str, delta: int
) -> tuple[int, list[int], list[int]]:
    """For one ``(step, combine_kind, op, delta)`` candidate: how many rows
    of ``table`` it can check (``covered``), how many of those it predicts
    correctly (the return value), and which rows it can't check at all
    (``uncoverable`` - the base-case candidates, since their value isn't
    determined by this recurrence at all)."""
    fn = _ARITH_OPS[op]
    covered: list[int] = []
    uncoverable: list[int] = []
    matched = 0
    for n, y in table.items():
        needed = (n - step, n - step - delta) if combine_kind == "double_recur" else (n - step,)
        if not all(k in table for k in needed):
            uncoverable.append(n)
            continue
        covered.append(n)
        if combine_kind == "double_recur":
            predicted = fn(table[needed[0]], table[needed[1]])
        else:
            predicted = fn(n, table[needed[0]])
        if predicted is not None and predicted == y:
            matched += 1
    return matched, covered, uncoverable


def _derive_base(table: dict[int, int], uncoverable: list[int]) -> dict | None:
    """The base-case hole-fillers (``base_kind``/``base_val``/``cmp``/
    ``base_const``) consistent with ``uncoverable`` - the rows a candidate
    recurrence can't check at all. Requires ``uncoverable`` to be exactly
    the smallest-``len(uncoverable)`` values in ``table`` (a contiguous
    prefix starting at the table's minimum): only then does ``param <=
    max(uncoverable)`` correctly generalize to inputs *outside* the table
    too (larger held-out ``n`` must stay on the recursive side) - see the
    module docstring's scope note. Returns ``None`` if the base rows aren't
    self-consistent (not all equal to their own ``n``, and not all equal to
    one shared constant) or don't form that prefix."""
    if not uncoverable:
        return None
    prefix = sorted(table)[: len(uncoverable)]
    if sorted(uncoverable) != prefix:
        return None
    base_vals = {n: table[n] for n in uncoverable}
    if all(v == n for n, v in base_vals.items()):
        base_kind_fields: dict = {"base_kind": "param"}
    else:
        values = set(base_vals.values())
        if len(values) != 1:
            return None
        base_kind_fields = {"base_kind": "const", "base_val": next(iter(values))}
    return {"cmp": "<=", "base_const": max(uncoverable), **base_kind_fields}


def mine_recursion_choices(
    table: dict[int, int],
    step_options: tuple[int, ...] = (1, 2),
    delta_options: tuple[int, ...] = (0, 1),
    min_covered: int = _MIN_COVERED_ROWS,
) -> dict | None:
    """Search every ``(combine_kind, step, delta, op)`` combination for an
    *exact* fit against ``table`` (every checkable row matches, no
    best-effort scoring - see the module docstring), then derive the base
    case for whichever fits (see :func:`_derive_base`). Among multiple exact
    fits (possible on a small table), prefers the one attested by the most
    covered rows - the strongest, least-coincidental signal. Returns a
    ``choices`` dict compatible with :data:`.search.TEMPLATE_CATEGORIES`, or
    ``None`` if nothing fits cleanly."""
    best: tuple[int, dict] | None = None
    for combine_kind in ("param_recur", "double_recur"):
        deltas = delta_options if combine_kind == "double_recur" else (0,)
        for step in step_options:
            for delta in deltas:
                for op in _ARITH_OPS:
                    matched, covered, uncoverable = _score_candidate(table, step, combine_kind, op, delta)
                    if len(covered) < min_covered or matched != len(covered) or not uncoverable:
                        continue
                    base = _derive_base(table, uncoverable)
                    if base is None:
                        continue
                    choices = {"combine_kind": combine_kind, "op": op, "step": step, **base}
                    if combine_kind == "double_recur":
                        choices["delta"] = delta
                    if best is None or len(covered) > best[0]:
                        best = (len(covered), choices)
    return best[1] if best else None


def numeric_recursion_template(
    inputs: list[str],
    list_inputs: tuple[str, ...],
    examples: list,
    rng: np.random.Generator,
    build_template_node: Callable[[str, str, str, dict], Node],
    step_options: tuple[int, ...] = (1, 2),
    delta_options: tuple[int, ...] = (0, 1),
) -> Node | None:
    """Top-level entry point, mirroring :func:`._recursive_template`'s own
    ``(inputs, list_inputs, rng, ...) -> Node | (None, None)`` signature
    convention closely enough to plug into the exact same
    ``random_program``/``_replace_at_scoped`` insertion point (just a plain
    ``Node | None`` return, like :func:`.semantic_bias.boolean_backprop_
    template` - there's no ``bias``-driven random draw here to report
    ``choices`` back for). Tries every scalar input as the recursion
    parameter (in case ``inputs`` has more than one) and returns the first
    one that mines cleanly; ``None`` if none do."""
    scalars = [n for n in inputs if n not in list_inputs]
    for var in scalars:
        table = build_table(examples, var)
        if table is None or len(table) < _MIN_COVERED_ROWS + 1:
            continue
        choices = mine_recursion_choices(table, step_options, delta_options)
        if choices is None:
            continue
        name = f"_rec{int(rng.integers(0, 10_000))}"
        param = f"_p{int(rng.integers(0, 10_000))}"
        return build_template_node(name, param, var, choices)
    return None
