"""Genetic-programming-style search over the DSL, scored by the existing substrate.

Random generation, mutation, and crossover target the *searchable* subset of
the DSL - ``Const``/``Var``/``ListLit``/``BinOp``/``UnaryOp``/``If``/``Let``/
``Fold``/``Length``/``Index``/``Map``/``Filter``/``Letrec``/``Recur``.
Selection pressure ("liquid time-step" spread across a generational search
rather than a single settle) reuses
:func:`zeuss.tier2_substrate.collapse.softmax` directly over negative
energies for fitness-proportionate parent selection - the exact same
Boltzmann-weighting idea :mod:`zeuss.tier3_logic.grounding` already uses at
compile time, applied once per generation instead of once at compile time.

**Recursion synthesis is attempted, with an honest caveat.** Generating a
well-scoped ``Letrec``/``Recur`` candidate needs to track which recursive
function names (and arities) are actually in scope at each tree position -
``_grow`` threads a ``recur_ctx`` for this. Mutation is the trickier case: a
naive context-*unaware* mutation (replace a random subtree with something
built from the top-level scope) would sometimes insert a ``Recur`` call or a
bare parameter reference that isn't valid at that position - safely, since
:func:`program_energy`'s exception guard turns that into a penalty rather
than a crash (fuel exhaustion is caught the same way), but *wastefully*, since
such candidates never contribute anything useful. ``mutate`` therefore uses a
scope-tracking traversal (``_replace_at_scoped``) that regenerates a
replacement using the *correct* (inputs, list_inputs, recur_ctx) valid at
that exact position, mirroring what ``_grow`` would have produced there.
Crossover stays scope-*unaware* (a donor subtree from one parent is spliced
into another without adjusting for scope) - safe for the same fuel/exception
reasons, simply lower-yield when it crosses a recursive scope boundary.
Whether this is enough to reliably *find* new recursive solutions (as
opposed to safely evaluate hand-built ones) is an empirical question
answered in ``tests/test_synthesis.py``, not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dsl import (
    BinOp,
    Const,
    Filter,
    Fold,
    Fuel,
    If,
    Index,
    Length,
    Let,
    Letrec,
    ListLit,
    Map,
    Node,
    Recur,
    UnaryOp,
    Var,
)
from ..tier2_substrate.collapse import softmax
from .dsl import children as _children
from .dsl import count_nodes, evaluate, rebuild

_BINOPS = ("+", "-", "*", "//", "%", "==", "!=", "<", "<=", ">", ">=", "and", "or")
_UNARYOPS = ("-", "not")
_LEAF_CONSTS = (0, 1, 2, 3, True, False)

# Node "kind" weights for random generation, keyed by whether a list input is
# available. Weighted (not uniform) so adding more list constructs doesn't
# dilute how often the most generally useful ones (binop, fold) get picked -
# with plain uniform weighting across all kinds, each new construct added to
# the grammar silently made every existing target harder to find.
_KIND_WEIGHTS_SCALAR = {"binop": 4, "unaryop": 1, "if": 2, "let": 1, "letrec": 1}
_KIND_WEIGHTS_WITH_LIST = {
    "binop": 4,
    "unaryop": 1,
    "if": 2,
    "let": 1,
    "letrec": 1,
    "fold": 3,
    "length": 1,
    "index": 1,
    "map": 2,
    "filter": 2,
}


def _choose_kind(rng: np.random.Generator, list_inputs: tuple[str, ...], recur_ctx: tuple) -> str:
    weights = dict(_KIND_WEIGHTS_WITH_LIST if list_inputs else _KIND_WEIGHTS_SCALAR)
    if recur_ctx:
        weights["recur"] = 2
    kinds = list(weights)
    probs = np.array([weights[k] for k in kinds], dtype=float)
    probs /= probs.sum()
    return kinds[int(rng.choice(len(kinds), p=probs))]


def _leaf(inputs: list[str], rng: np.random.Generator) -> Node:
    if inputs and rng.random() < 0.6:
        return Var(str(rng.choice(inputs)))
    return Const(_LEAF_CONSTS[int(rng.integers(0, len(_LEAF_CONSTS)))])


def _grow(
    inputs: list[str],
    list_inputs: tuple[str, ...],
    recur_ctx: tuple[tuple[str, int], ...],
    rng: np.random.Generator,
    depth: int,
) -> Node:
    # List-typed names are only ever useful directly as a Fold's list_expr -
    # using one as an ordinary scalar leaf is (almost) always a type error
    # that just adds dead weight to the search, so leaves are drawn from the
    # scalar-only subset.
    scalars = [n for n in inputs if n not in list_inputs]

    if depth <= 0 or rng.random() < 0.35:
        return _leaf(scalars, rng)

    kind = _choose_kind(rng, list_inputs, recur_ctx)

    if kind == "binop":
        op = _BINOPS[int(rng.integers(0, len(_BINOPS)))]
        return BinOp(
            op,
            _grow(inputs, list_inputs, recur_ctx, rng, depth - 1),
            _grow(inputs, list_inputs, recur_ctx, rng, depth - 1),
        )
    if kind == "unaryop":
        op = _UNARYOPS[int(rng.integers(0, len(_UNARYOPS)))]
        return UnaryOp(op, _grow(inputs, list_inputs, recur_ctx, rng, depth - 1))
    if kind == "if":
        return If(
            _grow(inputs, list_inputs, recur_ctx, rng, depth - 1),
            _grow(inputs, list_inputs, recur_ctx, rng, depth - 1),
            _grow(inputs, list_inputs, recur_ctx, rng, depth - 1),
        )
    if kind == "let":
        name = f"_let{int(rng.integers(0, 10_000))}"
        return Let(
            name,
            _grow(inputs, list_inputs, recur_ctx, rng, depth - 1),
            _grow(inputs + [name], list_inputs, recur_ctx, rng, depth - 1),
        )
    if kind == "letrec":
        name = f"_rec{int(rng.integers(0, 10_000))}"
        param = f"_p{int(rng.integers(0, 10_000))}"
        new_recur_ctx = recur_ctx + ((name, 1),)
        body = _grow(scalars + [param], (), new_recur_ctx, rng, depth - 1)
        in_expr = _grow(inputs, list_inputs, new_recur_ctx, rng, depth - 1)
        return Letrec(name, (param,), body, in_expr)
    if kind == "recur":
        name, arity = recur_ctx[int(rng.integers(0, len(recur_ctx)))]
        args = tuple(_grow(inputs, list_inputs, recur_ctx, rng, depth - 1) for _ in range(arity))
        return Recur(name, args)
    if kind == "length":
        return Length(Var(str(rng.choice(list_inputs))))
    if kind == "index":
        return Index(Var(str(rng.choice(list_inputs))), _grow(inputs, list_inputs, recur_ctx, rng, depth - 1))
    if kind == "map":
        list_name = str(rng.choice(list_inputs))
        body_inputs = scalars + ["_item"]
        return Map(Var(list_name), "_item", _grow(body_inputs, (), recur_ctx, rng, depth - 1))
    if kind == "filter":
        list_name = str(rng.choice(list_inputs))
        body_inputs = scalars + ["_item"]
        return Filter(Var(list_name), "_item", _grow(body_inputs, (), recur_ctx, rng, depth - 1))
    # fold - body sees only the fold-bound names plus existing scalars, not
    # the raw list itself (see note above).
    list_name = str(rng.choice(list_inputs))
    body_inputs = scalars + ["_acc", "_item"]
    return Fold(
        Var(list_name),
        _grow(inputs, list_inputs, recur_ctx, rng, depth - 1),
        "_acc",
        "_item",
        _grow(body_inputs, (), recur_ctx, rng, depth - 1),
    )


def random_program(
    inputs: list[str],
    rng: np.random.Generator,
    max_depth: int = 4,
    list_inputs: tuple[str, ...] = (),
) -> Node:
    """A randomly-grown candidate program over the full searchable DSL."""
    return _grow(list(inputs), tuple(list_inputs), (), rng, max_depth)


def _replace_at(node: Node, target_index: int, replacement: Node, counter: list[int]) -> Node:
    idx = counter[0]
    counter[0] += 1
    if idx == target_index:
        return replacement
    kids = _children(node)
    if not kids:
        return node
    new_kids = [_replace_at(child, target_index, replacement, counter) for child in kids]
    return rebuild(node, new_kids)


class _Found(Exception):
    def __init__(self, node: Node) -> None:
        self.node = node


def _find_at(node: Node, target_index: int, counter: list[int]) -> None:
    idx = counter[0]
    counter[0] += 1
    if idx == target_index:
        raise _Found(node)
    for child in _children(node):
        _find_at(child, target_index, counter)


def subtree_at(node: Node, target_index: int) -> Node:
    try:
        _find_at(node, target_index, [0])
    except _Found as found:
        return found.node
    raise IndexError(target_index)


def _replace_at_scoped(
    node: Node,
    target_index: int,
    rng: np.random.Generator,
    max_depth: int,
    counter: list[int],
    inputs: list[str],
    list_inputs: tuple[str, ...],
    recur_ctx: tuple[tuple[str, int], ...],
) -> Node:
    """Like :func:`_replace_at`, but regenerates the replacement using the
    (inputs, list_inputs, recur_ctx) actually valid at the target position -
    mirroring how :func:`_grow` would have built that subtree in the first
    place, so a mutation that introduces e.g. a fresh ``Recur`` call has a
    real chance of being well-scoped rather than an (safely, but uselessly)
    unbound-name penalty."""
    idx = counter[0]
    counter[0] += 1
    if idx == target_index:
        return _grow(inputs, list_inputs, recur_ctx, rng, max(1, max_depth - 1))

    scalars = [n for n in inputs if n not in list_inputs]

    if isinstance(node, Let):
        new_value = _replace_at_scoped(node.value, target_index, rng, max_depth, counter, inputs, list_inputs, recur_ctx)
        new_body = _replace_at_scoped(
            node.body, target_index, rng, max_depth, counter, inputs + [node.name], list_inputs, recur_ctx
        )
        return Let(node.name, new_value, new_body)
    if isinstance(node, Fold):
        new_list_expr = _replace_at_scoped(
            node.list_expr, target_index, rng, max_depth, counter, inputs, list_inputs, recur_ctx
        )
        new_init = _replace_at_scoped(node.init, target_index, rng, max_depth, counter, inputs, list_inputs, recur_ctx)
        body_inputs = scalars + [node.var_acc, node.var_item]
        new_body = _replace_at_scoped(node.body, target_index, rng, max_depth, counter, body_inputs, (), recur_ctx)
        return Fold(new_list_expr, new_init, node.var_acc, node.var_item, new_body)
    if isinstance(node, (Map, Filter)):
        new_list_expr = _replace_at_scoped(
            node.list_expr, target_index, rng, max_depth, counter, inputs, list_inputs, recur_ctx
        )
        body_inputs = scalars + [node.var_item]
        body_attr = node.body if isinstance(node, Map) else node.predicate
        new_body = _replace_at_scoped(body_attr, target_index, rng, max_depth, counter, body_inputs, (), recur_ctx)
        cls = Map if isinstance(node, Map) else Filter
        return cls(new_list_expr, node.var_item, new_body)
    if isinstance(node, Letrec):
        new_recur_ctx = recur_ctx + ((node.name, len(node.params)),)
        body_inputs = scalars + list(node.params)
        new_body = _replace_at_scoped(node.body, target_index, rng, max_depth, counter, body_inputs, (), new_recur_ctx)
        new_in_expr = _replace_at_scoped(
            node.in_expr, target_index, rng, max_depth, counter, inputs, list_inputs, new_recur_ctx
        )
        return Letrec(node.name, node.params, new_body, new_in_expr)
    if isinstance(node, Recur):
        new_args = [
            _replace_at_scoped(a, target_index, rng, max_depth, counter, inputs, list_inputs, recur_ctx)
            for a in node.args
        ]
        return Recur(node.name, tuple(new_args))
    if isinstance(node, ListLit):
        new_items = [
            _replace_at_scoped(item, target_index, rng, max_depth, counter, inputs, list_inputs, recur_ctx)
            for item in node.items
        ]
        return ListLit(tuple(new_items))

    # Generic same-scope-for-all-children nodes: BinOp, UnaryOp, If, Length, Index.
    kids = _children(node)
    if not kids:
        return node
    new_kids = [
        _replace_at_scoped(child, target_index, rng, max_depth, counter, inputs, list_inputs, recur_ctx)
        for child in kids
    ]
    return rebuild(node, new_kids)


def mutate(
    node: Node,
    rng: np.random.Generator,
    inputs: list[str],
    max_depth: int = 4,
    list_inputs: tuple[str, ...] = (),
) -> Node:
    """Replace a randomly chosen subtree with a freshly generated one, scoped
    correctly for that position (see module docstring)."""
    n = count_nodes(node)
    target = int(rng.integers(0, n))
    return _replace_at_scoped(node, target, rng, max_depth, [0], list(inputs), tuple(list_inputs), ())


def crossover(a: Node, b: Node, rng: np.random.Generator) -> Node:
    """Swap a randomly chosen subtree of ``a`` for one from ``b``."""
    na, nb = count_nodes(a), count_nodes(b)
    idx_a = int(rng.integers(0, na))
    idx_b = int(rng.integers(0, nb))
    donor = subtree_at(b, idx_b)
    return _replace_at(a, idx_a, donor, [0])


@dataclass
class Example:
    inputs: dict
    expected_output: object


def _mismatch(result, expected) -> float:
    # bool is a subclass of int in Python, so bool-ness must be checked
    # explicitly on *both* sides first - otherwise a stray True/False result
    # would silently "match" any truthy/falsy numeric target.
    if isinstance(expected, bool):
        if not isinstance(result, bool):
            return 10.0
        return 0.0 if result == expected else 1.0
    if isinstance(result, bool):
        return 10.0  # a bool leaked out where a non-bool was expected
    if isinstance(expected, (int, float)) and isinstance(result, (int, float)):
        return float(abs(result - expected))
    if isinstance(expected, list) and isinstance(result, list):
        if len(result) != len(expected):
            return 10.0 + abs(len(result) - len(expected))
        return float(sum(abs(a - b) for a, b in zip(result, expected)))
    return 10.0  # type mismatch


def program_energy(node: Node, examples: list[Example], fuel_budget: int = 500) -> float:
    """Sum of per-example mismatch penalties. A crash, type error, or fuel
    exhaustion on any example contributes a fixed high penalty - the search
    never sees an uncaught exception. ``_mismatch`` itself is inside the
    guard too: e.g. a nested list (from a badly-generated ``Map``/``Filter``
    whose body itself returns a list) can raise inside the comparison, not
    just inside ``evaluate`` - that must be penalized the same way, not
    allowed to crash the whole search."""
    total = 0.0
    for example in examples:
        try:
            result = evaluate(node, dict(example.inputs), Fuel(fuel_budget))
            total += _mismatch(result, example.expected_output)
        except Exception:
            total += 10.0
    return total


def synthesize(
    inputs: list[str],
    examples: list[Example],
    list_inputs: tuple[str, ...] = (),
    population_size: int = 200,
    max_generations: int = 60,
    max_depth: int = 4,
    beta_start: float = 0.5,
    beta_growth: float = 1.15,
    beta_max: float = 50.0,
    parsimony: float = 0.02,
    rng: np.random.Generator | None = None,
):
    """Search for a program satisfying ``examples`` via mutation/crossover,
    selected each generation by fitness-proportionate sampling (reusing
    :func:`collapse.softmax` over negative energies) with selection pressure
    (``beta``) rising across generations - exploration early, exploitation
    late, the same "liquid time-step" idea as
    :func:`zeuss.tier2_substrate.collapse.anneal_adaptive`, spread across a
    generational search instead of one annealing run.

    ``parsimony`` adds ``parsimony * count_nodes(candidate)`` on top of the
    raw mismatch energy *only* for the reproduction-selection probabilities
    (not for tracking the true best/``verified`` result) - without it, tree
    sizes grow unboundedly generation over generation (classic GP "bloat":
    empirically, mean population size roughly 6x'd over just 15 generations
    with no size pressure), which both wastes the search budget and makes
    every later generation slower to evaluate.

    Returns ``(best_node, beta_trace, verified)``. ``verified`` re-runs
    :func:`program_energy` on the winner one more time, post-hoc - never
    trust the search's own bookkeeping without re-checking.
    """
    rng = np.random.default_rng() if rng is None else rng
    population = [random_program(inputs, rng, max_depth, list_inputs) for _ in range(population_size)]

    best_node: Node = population[0]
    best_energy = float("inf")
    beta = beta_start
    beta_trace: list[float] = []

    for _generation in range(max_generations):
        raw_energies = np.array([program_energy(p, examples) for p in population])
        idx_best = int(np.argmin(raw_energies))
        if raw_energies[idx_best] < best_energy:
            best_energy = float(raw_energies[idx_best])
            best_node = population[idx_best]
        beta_trace.append(beta)
        if best_energy <= 1e-9:
            break

        sizes = np.array([count_nodes(p) for p in population], dtype=float)
        selection_energies = raw_energies + parsimony * sizes
        probs = softmax(-beta * selection_energies)
        next_population = [population[idx_best]]  # elitism (by raw energy, not parsimony-adjusted)
        while len(next_population) < population_size:
            i, j = rng.choice(len(population), size=2, p=np.asarray(probs))
            if rng.random() < 0.5:
                child = crossover(population[i], population[j], rng)
            else:
                child = mutate(population[i], rng, inputs, max_depth, list_inputs)
            next_population.append(child)
        population = next_population
        beta = min(beta * beta_growth, beta_max)

    verified = program_energy(best_node, examples) <= 1e-9
    return best_node, beta_trace, verified
