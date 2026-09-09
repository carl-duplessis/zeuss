"""Semantic backpropagation for compound boolean targets: a materially
different alternative to every mechanism tried so far for the leap-year-style
"deceptive local optimum" failure (see ``docs/ROADMAP.md`` v0.28-v0.30).

Six prior mechanisms - three hand-built structural templates
(``_recursive_template``, ``_fold_template``, ``_bool_template`` in
``search.py``), motif resonance (v0.28), grammar resonance (v0.29), and
shape elitism (v0.30) - all operate *after* generation: they reinforce,
protect, or reshuffle individuals that blind top-down random growth already
produced. None of them change what blind growth actually produces, and this
project's own docstrings already say why that's fatal for a target like leap
year: blind growth reliably misses the rare 3-atom AND/OR nesting in the
first place, so there is nothing for reinforcement/elitism to ever find and
protect. Measured directly (v0.30): adding shape elitism on top of motif or
grammar resonance still produced 0/10 on leap year, identical to every prior
attempt.

This module instead acts *at* generation time: given the actual per-example
targets a node needs to satisfy, it decomposes that requirement through the
real truth-table semantics of whichever boolean connective gets chosen (AND/
OR/NOT), so a compound shape is *constructed* from what the examples demand
instead of stumbled onto by blind luck. Deliberately shape-agnostic - no
hand-declared vocabulary like ``BOOL_TEMPLATE_CATEGORIES["shape_kind"]`` - so
it generalizes to any future target expressible as nested AND/OR/NOT of
comparison atoms, not just leap year's specific and_or3 shape.

Scope, stated honestly up front: this only covers boolean compound targets.
The ``2**n``/recursion failure is a structurally analogous but separate
problem (numeric relationship-mining over the example table, e.g. reading
``y_n / y_(n-1) == 2`` directly off the given ``(n, 2**n)`` pairs) and is not
attempted here - a clear, well-understood follow-up, not silently dropped.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .dsl import BinOp, Fuel, Node, UnaryOp
from .dsl import evaluate as _evaluate

# `Example` (search.py) is only ever used here as a type-hint string (see
# `from __future__ import annotations`) - importing it for real would create
# search.py <-> semantic_bias.py circular import, since search.py imports
# this module's `boolean_backprop_template`.

# Kind weights for the boolean sub-grammar this module grows - deliberately
# fixed/uniform, never resonance-biased. Mirrors this project's own
# established caution (see BOOL_TEMPLATE_CATEGORIES's/TEMPLATE_CATEGORIES's
# comments on shape_kind/combine_kind/base_kind): a structural/family choice
# risks premature commitment to the wrong family from early noise, exactly
# the failure mode this whole mechanism exists to route around.
_KIND_WEIGHTS: dict[str, float] = {"and": 2.0, "or": 2.0, "not": 1.0, "atom": 3.0}


def safe_eval_bool(node: Node, example: "Example", fuel_budget: int = 60) -> bool | None:
    """Evaluate ``node`` against one example, the same exception-guarded
    contract :func:`.search.program_energy` already uses - any crash, type
    error, or fuel exhaustion becomes ``None`` (indistinguishable from "not a
    bool"), never an uncaught exception. ``None`` doubles as this module's
    "don't know" sentinel, the same value used for don't-care target rows."""
    try:
        result = _evaluate(node, dict(example.inputs), Fuel(fuel_budget))
    except Exception:
        return None
    return result if isinstance(result, bool) else None


def and_left_target(parent: list[bool | None]) -> list[bool | None]:
    """``AND(l, r) = parent``. Rows where ``parent`` is ``True`` are a hard
    requirement on ``l`` (both sides must be ``True``); everywhere else is
    don't-care for whichever child is grown first - ``AND`` can still reach
    ``False`` via either side, so committing ``l`` to ``False`` there would
    be an arbitrary, unjustified constraint this module doesn't have grounds
    for yet (see :func:`and_right_target`, computed only after ``l`` is
    actually known)."""
    return [True if t is True else None for t in parent]


def and_right_target(parent: list[bool | None], left_actual: list[bool | None]) -> list[bool | None]:
    """The other half of :func:`and_left_target`'s decomposition, computed
    *after* the left child is actually grown and evaluated (unlike a
    symmetric AND-both-sides split, this needs a concrete ``left_actual`` to
    resolve ambiguity). ``AND(True, r) = r``, so rows where ``left_actual`` is
    ``True`` inherit ``parent``'s requirement directly. Everywhere else - left
    is ``False``, ``None`` (a crash/non-bool), or ``parent`` was already
    don't-care - ``AND`` is already ``False``/broken regardless of ``r``, so
    there is no genuine signal left for the right child at that row."""
    out: list[bool | None] = []
    for t, l in zip(parent, left_actual):
        if t is None or l is not True:
            out.append(None)
        else:
            out.append(t)
    return out


def or_left_target(parent: list[bool | None]) -> list[bool | None]:
    """``OR(l, r) = parent``. Exact dual of :func:`and_left_target`: rows
    where ``parent`` is ``False`` are a hard requirement (``OR`` needs both
    sides ``False`` to reach ``False``); ``True`` rows are don't-care for
    whichever child is grown first, since either side alone can satisfy
    them."""
    return [False if t is False else None for t in parent]


def or_right_target(parent: list[bool | None], left_actual: list[bool | None]) -> list[bool | None]:
    """Dual of :func:`and_right_target`. ``OR(False, r) = r``, so rows where
    ``left_actual`` is ``False`` inherit ``parent``'s requirement; everywhere
    else ``OR`` is already ``True``/broken regardless of ``r``."""
    out: list[bool | None] = []
    for t, l in zip(parent, left_actual):
        if t is None or l is not False:
            out.append(None)
        else:
            out.append(t)
    return out


def not_target(parent: list[bool | None]) -> list[bool | None]:
    """``NOT(x) = parent`` inverts elementwise - unlike AND/OR, no
    don't-cares: a unary connective leaves nothing ambiguous."""
    return [None if t is None else (not t) for t in parent]


_CMP_OPS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "==": lambda a, k: a == k,
    "!=": lambda a, k: a != k,
    "<": lambda a, k: a < k,
    "<=": lambda a, k: a <= k,
    ">": lambda a, k: a > k,
    ">=": lambda a, k: a >= k,
}


def best_atom_for_target(
    var_pool: list[str],
    examples: list["Example"],
    targets: list[bool | None],
    atom_vocab: dict[str, list],
    build_atom: Callable[[str, dict], Node],
    rng: np.random.Generator,
) -> Node | None:
    """Search ``var_pool x atom_vocab["modulus"] x atom_vocab["cmp"] x
    atom_vocab["const"]`` (the same vocabulary :func:`._bool_template` in
    ``search.py`` already uses - passed in by the caller, like
    :class:`.grammar_bias.GrammarBias` takes ``choice_vocab``, to avoid a
    circular import with ``search.py`` and a second, driftable copy) for the
    atom that matches ``targets`` on the most non-``None`` rows. Vectorized
    with ``numpy`` over each candidate variable's example values - the atom
    shape (``(Var % modulus) CMP const``) is fixed and simple enough to score
    directly, without going through :func:`.dsl.evaluate` per candidate, so
    this stays cheap (a few hundred candidates) regardless of how many
    examples there are. Returns ``None`` only if ``var_pool`` is empty (there
    is genuinely nothing to build from - mirrors ``_bool_template``'s own
    "no scalar input" contract). If every row is don't-care, there is
    nothing to score *against*, but the caller (an AND/OR ancestor for whom
    this whole subtree is provably irrelevant to correctness) still needs a
    real node back - a uniformly random atom is picked instead of failing
    the entire candidate over a position nothing depends on."""
    if not var_pool:
        return None
    mask = np.array([t is not None for t in targets])
    if not mask.any():
        var = str(rng.choice(var_pool))
        atom = {
            "var": var,
            "modulus": atom_vocab["modulus"][int(rng.integers(0, len(atom_vocab["modulus"])))],
            "cmp": atom_vocab["cmp"][int(rng.integers(0, len(atom_vocab["cmp"])))],
            "const": atom_vocab["const"][int(rng.integers(0, len(atom_vocab["const"])))],
        }
        return build_atom(var, atom)
    target_arr = np.array([bool(t) if t is not None else False for t in targets])[mask]

    best_score = -1.0
    best_candidates: list[dict] = []
    for var in var_pool:
        try:
            values = np.array([example.inputs[var] for example in examples])[mask]
        except (KeyError, TypeError):
            continue
        for modulus in atom_vocab["modulus"]:
            residue = values % modulus
            for const in atom_vocab["const"]:
                for cmp in atom_vocab["cmp"]:
                    predicted = _CMP_OPS[cmp](residue, const)
                    score = float(np.count_nonzero(predicted == target_arr))
                    atom = {"var": var, "modulus": modulus, "cmp": cmp, "const": const}
                    if score > best_score:
                        best_score = score
                        best_candidates = [atom]
                    elif score == best_score:
                        best_candidates.append(atom)

    if not best_candidates:
        return None
    chosen = best_candidates[int(rng.integers(0, len(best_candidates)))]
    return build_atom(chosen["var"], chosen)


def grow_boolean_targeted(
    var_pool: list[str],
    examples: list["Example"],
    targets: list[bool | None],
    rng: np.random.Generator,
    depth: int,
    atom_vocab: dict[str, list],
    build_atom: Callable[[str, dict], Node],
    fuel_budget: int = 60,
) -> Node | None:
    """Grow one node of the boolean sub-grammar (``and``/``or``/``not``/atom)
    whose semantics are chosen to satisfy ``targets`` (parallel to
    ``examples``, ``None`` = don't-care), recursing via the decomposition
    functions above rather than growing blind and hoping. Bottoms out at an
    atom (see :func:`best_atom_for_target`) once ``depth`` is exhausted or
    every row is already don't-care - there is nothing left to decompose."""
    if not var_pool:
        return None
    non_none = any(t is not None for t in targets)
    if depth <= 0 or not non_none:
        return best_atom_for_target(var_pool, examples, targets, atom_vocab, build_atom, rng)

    kinds = list(_KIND_WEIGHTS)
    weights = np.array([_KIND_WEIGHTS[k] for k in kinds], dtype=float)
    kind = kinds[int(rng.choice(len(kinds), p=weights / weights.sum()))]

    if kind == "atom":
        return best_atom_for_target(var_pool, examples, targets, atom_vocab, build_atom, rng)
    if kind == "not":
        child = grow_boolean_targeted(
            var_pool, examples, not_target(targets), rng, depth - 1, atom_vocab, build_atom, fuel_budget
        )
        return UnaryOp("not", child) if child is not None else None
    if kind == "and":
        left = grow_boolean_targeted(
            var_pool, examples, and_left_target(targets), rng, depth - 1, atom_vocab, build_atom, fuel_budget
        )
        if left is None:
            return None
        left_actual = [safe_eval_bool(left, ex, fuel_budget) for ex in examples]
        right = grow_boolean_targeted(
            var_pool, examples, and_right_target(targets, left_actual), rng, depth - 1,
            atom_vocab, build_atom, fuel_budget,
        )
        return BinOp("and", left, right) if right is not None else None
    # or
    left = grow_boolean_targeted(
        var_pool, examples, or_left_target(targets), rng, depth - 1, atom_vocab, build_atom, fuel_budget
    )
    if left is None:
        return None
    left_actual = [safe_eval_bool(left, ex, fuel_budget) for ex in examples]
    right = grow_boolean_targeted(
        var_pool, examples, or_right_target(targets, left_actual), rng, depth - 1,
        atom_vocab, build_atom, fuel_budget,
    )
    return BinOp("or", left, right) if right is not None else None


def boolean_backprop_template(
    inputs: list[str],
    list_inputs: tuple[str, ...],
    examples: list["Example"],
    rng: np.random.Generator,
    max_depth: int,
    atom_vocab: dict[str, list],
    build_atom: Callable[[str, dict], Node],
    fuel_budget: int = 60,
) -> Node | None:
    """Top-level entry point, mirroring :func:`._bool_template`'s own
    ``(inputs, list_inputs, rng, ...) -> Node | None`` signature so it plugs
    into ``random_program``/``_replace_at_scoped`` exactly where the three
    existing templates already are. Returns ``None`` if there's no scalar
    input, or if ``examples`` isn't a genuinely boolean target (this
    mechanism doesn't apply to arithmetic/list targets at all) - mirrors
    ``_bool_template``'s own ``(None, None)`` "nothing to offer" contract."""
    scalars = [n for n in inputs if n not in list_inputs]
    if not scalars or not examples:
        return None
    if not all(isinstance(ex.expected_output, bool) for ex in examples):
        return None
    targets: list[bool | None] = [bool(ex.expected_output) for ex in examples]
    return grow_boolean_targeted(scalars, examples, targets, rng, max_depth, atom_vocab, build_atom, fuel_budget)
