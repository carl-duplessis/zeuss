"""Genetic-programming-style search over the DSL, scored by the existing substrate.

Random generation, mutation, and crossover target the *searchable* subset of
the DSL - ``Const``/``Var``/``ListLit``/``BinOp``/``UnaryOp``/``If``/``Let``/
``Fold``/``Length``/``Index``/``Map``/``Filter``, plus ``Letrec``/``Recur``
when explicitly opted into (see below). Selection pressure ("liquid
time-step" spread across a generational search rather than a single settle)
reuses :func:`zeuss.tier2_substrate.collapse.softmax` directly over negative
energies for fitness-proportionate parent selection - the exact same
Boltzmann-weighting idea :mod:`zeuss.tier3_logic.grounding` already uses at
compile time, applied once per generation instead of once at compile time.

**Recursion synthesis is opt-in (``allow_recursion=False`` by default), and
attempted with an honest caveat.** Generating a well-scoped ``Letrec``/
``Recur`` candidate needs to track which recursive function names (and
arities) are actually in scope at each tree position - ``_grow`` threads a
``recur_ctx`` for this. Mutation is the trickier case: a naive context-
*unaware* mutation (replace a random subtree with something built from the
top-level scope) would sometimes insert a ``Recur`` call or a bare parameter
reference that isn't valid at that position - safely, since
:func:`program_energy`'s exception guard turns that into a penalty rather
than a crash, but *wastefully*, since such candidates never contribute
anything useful. ``mutate`` therefore uses a scope-tracking traversal
(``_replace_at_scoped``) that regenerates a replacement using the *correct*
(inputs, list_inputs, recur_ctx) valid at that exact position, mirroring what
``_grow`` would have produced there. Crossover stays scope-*unaware* (a donor
subtree from one parent is spliced into another without adjusting for scope)
- safe for the same reasons, simply lower-yield when it crosses a recursive
scope boundary. ``allow_recursion`` defaults to ``False`` because leaving
``letrec``/``recur`` unconditionally in the generation grammar was tried and
measured to make *every* search meaningfully slower per candidate - even for
targets that never touch recursion - since a recursive candidate costs more
to evaluate than a shallow one even when perfectly safe. When
``allow_recursion=True``, ``template_rate`` biases generation/mutation toward
:func:`_recursive_template` (a "decrement-and-combine" skeleton) instead of
hoping blind growth stumbles onto a working recursive shape (empirically,
under 5% of random depth-4 trees even contain a ``Letrec`` with an
``If``-shaped body). Two further robustness issues were found and fixed
while testing this against real candidates, not assumed away: a
``RecursionError`` (Python's own stack limit, not the fuel counter) could
otherwise escape as an uncaught crash (see ``dsl.py``), and an unbounded
value magnitude (a candidate that *grows* its argument instead of shrinking
it - e.g. squaring - produces numbers with millions of bits well within the
fuel budget's call-count limit, making arithmetic the actual bottleneck, not
recursion depth). Whether all of this is enough to reliably *find* new
recursive solutions (as opposed to safely generate and evaluate them) is an
empirical question answered in ``tests/test_synthesis.py``, not assumed -
and the honest answer, even after these fixes and template-biased seeding,
is still no for a target that genuinely requires recursion.
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
from ..tier2_substrate.hypervectors import Codebook
from .dsl import children as _children
from .dsl import count_nodes, evaluate, rebuild
from .grammar_bias import GrammarBias
from .motif_bias import MotifArchive, context_key
from .resonance_bias import ResonantBias

_BINOPS = ("+", "-", "*", "//", "%", "==", "!=", "<", "<=", ">", ">=", "and", "or")
_UNARYOPS = ("-", "not")
_LEAF_CONSTS = (0, 1, 2, 3, True, False)

# The fixed, target-independent option sets `GrammarBias` (grammar_bias.py)
# biases production choices over - supplied from here rather than
# redeclared there, avoiding both a circular import and a second, driftable
# copy of these tuples. Deliberately does not include a "kind" category
# (which node type to grow): its valid option pool varies per call (list
# ops only exist when list_inputs is non-empty, letrec/recur only when
# allow_recursion), which needs a ResonantBias variant that takes a
# caller-supplied candidate list rather than its own fixed
# categories[cat] - a well-scoped follow-up, not attempted in this version
# (see docs/ROADMAP.md v0.29).
GRAMMAR_CHOICE_VOCAB: dict[str, list] = {
    "binop_op": list(_BINOPS),
    "unaryop_op": list(_UNARYOPS),
    "leaf_kind": ["var", "const"],
    "leaf_const": list(_LEAF_CONSTS),
}

# Node "kind" weights for random generation, keyed by whether a list input is
# available. Weighted (not uniform) so adding more list constructs doesn't
# dilute how often the most generally useful ones (binop, fold) get picked -
# with plain uniform weighting across all kinds, each new construct added to
# the grammar silently made every existing target harder to find.
_KIND_WEIGHTS_SCALAR = {"binop": 4, "unaryop": 1, "if": 2, "let": 1}
_KIND_WEIGHTS_WITH_LIST = {
    "binop": 4,
    "unaryop": 1,
    "if": 2,
    "let": 1,
    "fold": 3,
    "length": 1,
    "index": 1,
    "map": 2,
    "filter": 2,
}


def _choose_kind(
    rng: np.random.Generator, list_inputs: tuple[str, ...], recur_ctx: tuple, allow_recursion: bool
) -> str:
    weights = dict(_KIND_WEIGHTS_WITH_LIST if list_inputs else _KIND_WEIGHTS_SCALAR)
    # letrec/recur are opt-in, not default-on: found empirically that leaving
    # them unconditionally in the pool made *every* search meaningfully more
    # expensive per-candidate (recursive candidates cost more to evaluate,
    # even when perfectly safe), regardless of whether the target needed
    # recursion at all - a real, measured regression, not a hypothetical one.
    if allow_recursion:
        weights["letrec"] = 1
        if recur_ctx:
            weights["recur"] = 2
    kinds = list(weights)
    probs = np.array([weights[k] for k in kinds], dtype=float)
    probs /= probs.sum()
    return kinds[int(rng.choice(len(kinds), p=probs))]


def _leaf(
    inputs: list[str],
    rng: np.random.Generator,
    parent_kind: str = "root",
    child_slot: int = 0,
    grammar_bias: "GrammarBias | None" = None,
) -> Node:
    if grammar_bias is not None:
        leaf_kind = grammar_bias.sample_or(
            "leaf_kind", parent_kind, child_slot, rng,
            lambda: "var" if inputs and rng.random() < 0.6 else "const",
        )
    else:
        leaf_kind = "var" if inputs and rng.random() < 0.6 else "const"
    if leaf_kind == "var" and inputs:
        return Var(str(rng.choice(inputs)))
    if grammar_bias is not None:
        value = grammar_bias.sample_or(
            "leaf_const", parent_kind, child_slot, rng,
            lambda: _LEAF_CONSTS[int(rng.integers(0, len(_LEAF_CONSTS)))],
        )
    else:
        value = _LEAF_CONSTS[int(rng.integers(0, len(_LEAF_CONSTS)))]
    return Const(value)


def _grow(
    inputs: list[str],
    list_inputs: tuple[str, ...],
    recur_ctx: tuple[tuple[str, int], ...],
    rng: np.random.Generator,
    depth: int,
    allow_recursion: bool = False,
    parent_kind: str = "root",
    child_slot: int = 0,
    template_rate: float = 0.15,
    allow_motif_bias: bool = False,
    motif_archive: "MotifArchive | None" = None,
    grammar_bias: "GrammarBias | None" = None,
) -> Node:
    # List-typed names are only ever useful directly as a Fold's list_expr -
    # using one as an ordinary scalar leaf is (almost) always a type error
    # that just adds dead weight to the search, so leaves are drawn from the
    # scalar-only subset.
    scalars = [n for n in inputs if n not in list_inputs]

    # Motif resonance (opt-in, see motif_bias.py): before growing fresh at
    # this exact structural position, check whether the archive already has
    # a reinforced subtree observed at this same (parent_kind, child_slot,
    # depth) context, and splice a copy of it in instead - the
    # generalization of _recursive_template/_fold_template/_bool_template's
    # "known-good skeleton" idea, except the skeleton is discovered from the
    # population's own history instead of hand-written. Guarded so
    # allow_motif_bias=False never spends an rng draw here, keeping every
    # existing caller's draw sequence bit-for-bit unchanged.
    if allow_motif_bias and motif_archive is not None:
        ctx = context_key(parent_kind, child_slot, depth)
        if motif_archive.has(ctx) and rng.random() < template_rate:
            motif = motif_archive.sample(ctx, rng)
            if motif is not None:
                return motif

    if depth <= 0 or rng.random() < 0.35:
        return _leaf(scalars, rng, parent_kind, child_slot, grammar_bias)

    kind = _choose_kind(rng, list_inputs, recur_ctx, allow_recursion)
    kwargs = {
        "allow_recursion": allow_recursion,
        "template_rate": template_rate,
        "allow_motif_bias": allow_motif_bias,
        "motif_archive": motif_archive,
        "grammar_bias": grammar_bias,
    }

    if kind == "binop":
        if grammar_bias is not None:
            op = grammar_bias.sample_or(
                "binop_op", parent_kind, child_slot, rng,
                lambda: _BINOPS[int(rng.integers(0, len(_BINOPS)))],
            )
        else:
            op = _BINOPS[int(rng.integers(0, len(_BINOPS)))]
        return BinOp(
            op,
            _grow(inputs, list_inputs, recur_ctx, rng, depth - 1, parent_kind="BinOp", child_slot=0, **kwargs),
            _grow(inputs, list_inputs, recur_ctx, rng, depth - 1, parent_kind="BinOp", child_slot=1, **kwargs),
        )
    if kind == "unaryop":
        if grammar_bias is not None:
            op = grammar_bias.sample_or(
                "unaryop_op", parent_kind, child_slot, rng,
                lambda: _UNARYOPS[int(rng.integers(0, len(_UNARYOPS)))],
            )
        else:
            op = _UNARYOPS[int(rng.integers(0, len(_UNARYOPS)))]
        return UnaryOp(
            op, _grow(inputs, list_inputs, recur_ctx, rng, depth - 1, parent_kind="UnaryOp", child_slot=0, **kwargs)
        )
    if kind == "if":
        return If(
            _grow(inputs, list_inputs, recur_ctx, rng, depth - 1, parent_kind="If", child_slot=0, **kwargs),
            _grow(inputs, list_inputs, recur_ctx, rng, depth - 1, parent_kind="If", child_slot=1, **kwargs),
            _grow(inputs, list_inputs, recur_ctx, rng, depth - 1, parent_kind="If", child_slot=2, **kwargs),
        )
    if kind == "let":
        name = f"_let{int(rng.integers(0, 10_000))}"
        return Let(
            name,
            _grow(inputs, list_inputs, recur_ctx, rng, depth - 1, parent_kind="Let", child_slot=0, **kwargs),
            _grow(inputs + [name], list_inputs, recur_ctx, rng, depth - 1, parent_kind="Let", child_slot=1, **kwargs),
        )
    if kind == "letrec":
        name = f"_rec{int(rng.integers(0, 10_000))}"
        param = f"_p{int(rng.integers(0, 10_000))}"
        new_recur_ctx = recur_ctx + ((name, 1),)
        body = _grow(
            scalars + [param], (), new_recur_ctx, rng, depth - 1, parent_kind="Letrec", child_slot=0, **kwargs
        )
        in_expr = _grow(
            inputs, list_inputs, new_recur_ctx, rng, depth - 1, parent_kind="Letrec", child_slot=1, **kwargs
        )
        return Letrec(name, (param,), body, in_expr)
    if kind == "recur":
        name, arity = recur_ctx[int(rng.integers(0, len(recur_ctx)))]
        args = tuple(
            _grow(inputs, list_inputs, recur_ctx, rng, depth - 1, parent_kind="Recur", child_slot=i, **kwargs)
            for i in range(arity)
        )
        return Recur(name, args)
    if kind == "length":
        return Length(Var(str(rng.choice(list_inputs))))
    if kind == "index":
        return Index(
            Var(str(rng.choice(list_inputs))),
            _grow(inputs, list_inputs, recur_ctx, rng, depth - 1, parent_kind="Index", child_slot=1, **kwargs),
        )
    if kind == "map":
        list_name = str(rng.choice(list_inputs))
        body_inputs = scalars + ["_item"]
        return Map(
            Var(list_name),
            "_item",
            _grow(body_inputs, (), recur_ctx, rng, depth - 1, parent_kind="Map", child_slot=1, **kwargs),
        )
    if kind == "filter":
        list_name = str(rng.choice(list_inputs))
        body_inputs = scalars + ["_item"]
        return Filter(
            Var(list_name),
            "_item",
            _grow(body_inputs, (), recur_ctx, rng, depth - 1, parent_kind="Filter", child_slot=1, **kwargs),
        )
    # fold - body sees only the fold-bound names plus existing scalars, not
    # the raw list itself (see note above).
    list_name = str(rng.choice(list_inputs))
    body_inputs = scalars + ["_acc", "_item"]
    return Fold(
        Var(list_name),
        _grow(inputs, list_inputs, recur_ctx, rng, depth - 1, parent_kind="Fold", child_slot=1, **kwargs),
        "_acc",
        "_item",
        _grow(body_inputs, (), recur_ctx, rng, depth - 1, parent_kind="Fold", child_slot=2, **kwargs),
    )


# The hole categories _recursive_template fills in, and their possible
# values - shared with ResonantBias so a bias object's categories always
# line up with what the template actually samples.
#
# "combine_kind" picks between two recursive shapes:
#   param_recur  - p OP f(p - step)                       (e.g. factorial: n * f(n-1))
#   double_recur - f(p - step) OP f(p - step - delta)      (e.g. 2**n with delta=0:
#                  f(n-1) + f(n-1); Fibonacci with step=1, delta=1: f(n-1) + f(n-2))
# The second shape was added after finding, empirically, that the actual
# winning 2**n solution discovered via mutation (f(n-1)+f(n-1)) does *not*
# match param_recur at all - mutation had to build it from scratch with zero
# direct template/reinforcement support. Giving the search direct access to
# generate *and* reinforce this shape, instead of hoping mutation stumbles
# into it, is the fix. "delta" (added later, for asymmetric recursion like
# Fibonacci) is a small offset from the shared "step" rather than a fully
# independent second step - keeps the combinatorial search burden down for
# what's usually a small asymmetry in practice, instead of reintroducing the
# fully-independent step2 that was tried and reverted (it roughly doubled
# the search space for comparatively little expressive gain).
TEMPLATE_CATEGORIES: dict[str, list] = {
    "cmp": ["==", "<=", "<"],
    "base_const": [0, 1, 2],
    "base_val": [0, 1, 2],
    "base_kind": ["const", "param"],
    "step": [1, 2],
    "delta": [0, 1],
    "op": list(_BINOPS),
    "combine_kind": ["param_recur", "double_recur"],
}


def _recursive_template(
    inputs: list[str],
    list_inputs: tuple[str, ...],
    rng: np.random.Generator,
    bias: "ResonantBias | None" = None,
    delta_p1: float = 0.15,
) -> tuple[Node, dict] | tuple[None, None]:
    """A structural bias toward the two most common recursive shapes:
    ``letrec f(p) = if p CMP k then base else COMBINE in f(seed)``, where
    COMBINE is either ``p OP f(p - step)`` (param_recur) or
    ``f(p - step) OP f(p - step - delta)`` (double_recur - symmetric when
    ``delta=0`` like ``2**n``, asymmetric divide-and-conquer like Fibonacci
    when ``delta != 0``), and ``base`` is either ``Const(base_val)``
    (``base_kind="const"``) or ``Var(param)`` itself (``base_kind="param"``).
    The latter is what makes a *true* zero-indexed Fibonacci
    (``F(0)=0, F(1)=1``, i.e. ``if p<=1 then p else f(p-1)+f(p-2)``)
    expressible - with only ``Const`` base cases, the base value is a fixed
    number, which cannot equal the varying parameter. GP still has to tune
    the comparison, constants, and combining operator, but starts from a
    working high-level *shape* instead of needing mutation to discover one
    from nothing: empirically, under 5% of random depth-4 trees even contain
    a ``Letrec`` with an ``If``-shaped body (the bare minimum for a base
    case). Returns ``(None, None)`` if there's no scalar input to recurse on.
    If ``bias`` is given, the fine-tuning hole-fillers (not
    ``combine_kind``/``delta``/``base_kind`` - see below) are drawn from it
    (resonance-biased toward historically successful values) instead of
    uniformly at random. Returns ``(node, choices)`` - ``choices`` is needed
    so the caller can later :meth:`ResonantBias.reinforce` based on how the
    individual scores.
    """
    scalars = [n for n in inputs if n not in list_inputs]
    if not scalars:
        return None, None
    name = f"_rec{int(rng.integers(0, 10_000))}"
    param = f"_p{int(rng.integers(0, 10_000))}"
    seed_var = str(rng.choice(scalars))
    # combine_kind (which recursive *shape*) and delta (symmetric vs
    # asymmetric double_recur) are deliberately drawn uniformly/from a fixed
    # schedule, never resonance-biased: reinforcing which structural family
    # to use risks premature commitment to the wrong one from early noise
    # (found empirically for combine_kind - double_recur candidates fail
    # *harder*, via fuel exhaustion on their exponential call trees, than
    # param_recur ones fail on theirs, making param_recur look artificially
    # better before either had a fair chance; the same risk applies to
    # delta=0 looking artificially better than delta=1 before an asymmetric
    # target has had a chance to prove delta=1 is actually needed).
    # base_kind is drawn the same way as combine_kind: uniformly, never
    # resonance-biased. It's a structural choice (does the base case depend
    # on the parameter at all?) of the same kind combine_kind/delta already
    # are, and the project's established caution is to keep structural/family
    # choices out of the learned bias so a locally-confident-but-wrong family
    # can't lock in from early noise - see the comment on combine_kind below.
    base_kind = str(rng.choice(TEMPLATE_CATEGORIES["base_kind"]))
    combine_kind = str(rng.choice(TEMPLATE_CATEGORIES["combine_kind"]))
    # delta's prior is a fixed *schedule*, not a learned bias (so no
    # premature-commitment risk): ``delta_p1`` is the probability of the
    # (rarer) asymmetric choice, meant to be annealed by the caller across
    # generations (see synthesize's docstring) - low early (symmetric
    # recursion is the more common case, and a flat 50/50 split was measured
    # to noticeably slow down delta=0 targets like 2**n, since an extra
    # unbiased binary split roughly halves the effective population
    # searching the right delta), rising later if the search hasn't
    # converged, so an asymmetric target like Fibonacci still gets a real
    # chance instead of the prior alone permanently disadvantaging it - the
    # same "liquid time-step" idea as collapse.anneal_adaptive, applied to a
    # discrete structural choice instead of a continuous beta.
    delta = (
        int(rng.choice(TEMPLATE_CATEGORIES["delta"], p=[1.0 - delta_p1, delta_p1]))
        if combine_kind == "double_recur"
        else 0
    )
    if bias is not None:
        cmp = bias.sample("cmp", rng)
        base_const = bias.sample("base_const", rng)
        base_val = bias.sample("base_val", rng)
        step = bias.sample("step", rng)
        op = bias.sample("op", rng)
    else:
        cmp = str(rng.choice(TEMPLATE_CATEGORIES["cmp"]))
        base_const = int(rng.choice(TEMPLATE_CATEGORIES["base_const"]))
        base_val = int(rng.choice(TEMPLATE_CATEGORIES["base_val"]))
        step = int(rng.choice(TEMPLATE_CATEGORIES["step"]))
        op = str(rng.choice(TEMPLATE_CATEGORIES["op"]))

    choices = {"cmp": cmp, "base_const": base_const, "step": step, "op": op, "combine_kind": combine_kind, "base_kind": base_kind}
    if base_kind != "param":
        choices["base_val"] = base_val
    if combine_kind == "double_recur":
        choices["delta"] = delta
    node = _build_template_node(name, param, seed_var, choices)
    return node, choices


def _build_template_body(name: str, param: str, choices: dict) -> Node:
    """Build just the ``if p CMP k then base else COMBINE`` body from a
    complete hole-filler ``choices`` dict - the part that's actually made of
    hole-fillers. Split from :func:`_build_template_node` so
    :func:`_mutate_template_hole` can rebuild *only* the body of an existing
    node, leaving its ``in_expr`` (and anything else outside the body)
    untouched - ``extract_template_choices`` only validates the body shape,
    never ``in_expr``, so a template-shaped individual that mutation or
    crossover has since altered elsewhere (e.g. spliced a different
    ``in_expr`` in) cannot be assumed to still have the canonical
    ``Recur(name, (Var(seed_var),))`` in_expr :func:`_recursive_template`
    always builds.
    """
    if choices.get("base_kind") == "param":
        then_node: Node = Var(param)
    else:
        then_node = Const(choices["base_val"])
    op = choices["op"]
    step = choices["step"]
    if choices["combine_kind"] == "double_recur":
        delta = choices.get("delta", 0)
        arg_a = BinOp("-", Var(param), Const(step))
        arg_b = arg_a if delta == 0 else BinOp("-", Var(param), Const(step + delta))
        combine = BinOp(op, Recur(name, (arg_a,)), Recur(name, (arg_b,)))
    else:
        combine = BinOp(op, Var(param), Recur(name, (BinOp("-", Var(param), Const(step)),)))
    return If(BinOp(choices["cmp"], Var(param), Const(choices["base_const"])), then_node, combine)


def _build_template_node(name: str, param: str, seed_var: str, choices: dict) -> Node:
    """Deterministically build the tree :func:`_recursive_template` describes
    from a complete ``choices`` dict (as returned by it, or by
    :func:`extract_template_choices`) plus the three names that aren't
    themselves hole-fillers.
    """
    body = _build_template_body(name, param, choices)
    return Letrec(name, (param,), body, Recur(name, (Var(seed_var),)))


# A structural bias toward the most common fold shapes, mirroring
# _recursive_template's role but for list processing rather than recursion -
# added after the v0.22 audit (docs/ROADMAP.md) found list-op targets were
# generated and mutated by the same blind uniform _grow/mutate used for
# arbitrary arithmetic, with no template, no per-shape elitism, and no bias
# toward historically-successful hole values, unlike recursion.
#
# "combine_op" picks the accumulator-combining operator - "+" for sum-like
# folds, "*" for product-like ones. "item_kind" picks how each element is
# transformed before combining: "identity" (sum), "square" (sum of squares -
# the target that originally motivated this), or "cmp_const" (element
# compared against a constant, e.g. counting elements > 0 - the comparison
# coerces to an int the same way _mismatch already treats bool/int
# elsewhere in this module). Both are drawn uniformly, never resonance-
# biased, for the same reason _recursive_template keeps combine_kind/
# base_kind out of the learned bias: they are structural/family choices, and
# reinforcing which family to use risks premature commitment to the wrong
# one from early noise. "cmp"/"const" (only meaningful for item_kind=
# "cmp_const") are the tunable holes - see _FOLD_TUNABLE_HOLES.
FOLD_TEMPLATE_CATEGORIES: dict[str, list] = {
    "combine_op": ["+", "*"],
    "item_kind": ["identity", "square", "cmp_const"],
    "cmp": ["==", "!=", "<", "<=", ">", ">="],
    "const": [-2, -1, 0, 1, 2],
}


def _fold_item_expr(item_var: str, choices: dict) -> Node:
    kind = choices["item_kind"]
    if kind == "identity":
        return Var(item_var)
    if kind == "square":
        return BinOp("*", Var(item_var), Var(item_var))
    return BinOp(choices["cmp"], Var(item_var), Const(choices["const"]))


def _fold_init_for(combine_op: str) -> Node:
    """The identity element for ``combine_op`` - ``0`` for ``+``, ``1`` for
    ``*`` - derived rather than sampled as its own hole, keeping the
    combinatorial search burden down the same way ``delta`` is a small
    offset from ``step`` rather than a fully independent value (see
    ``_recursive_template``'s docstring)."""
    return Const(0) if combine_op == "+" else Const(1)


def _build_fold_template_body(acc_var: str, item_var: str, choices: dict) -> Node:
    """Build just the ``acc COMBINE_OP item_expr`` body from a complete
    fold-template ``choices`` dict - split out the same way
    ``_build_template_body`` is, so :func:`_mutate_fold_template_hole` can
    rebuild only the body, leaving ``list_expr``/``init`` untouched."""
    return BinOp(choices["combine_op"], Var(acc_var), _fold_item_expr(item_var, choices))


def _build_fold_template_node(list_name: str, acc_var: str, item_var: str, choices: dict) -> Node:
    body = _build_fold_template_body(acc_var, item_var, choices)
    return Fold(Var(list_name), _fold_init_for(choices["combine_op"]), acc_var, item_var, body)


def _fold_template(
    list_inputs: tuple[str, ...],
    rng: np.random.Generator,
    bias: "ResonantBias | None" = None,
) -> tuple[Node, dict] | tuple[None, None]:
    """A structural bias toward ``fold(xs, init, acc, item, acc COMBINE_OP
    transform(item))`` - see :data:`FOLD_TEMPLATE_CATEGORIES` for the
    transform choices. Returns ``(None, None)`` if there's no list input to
    fold over. If ``bias`` is given, ``cmp``/``const`` are drawn from it
    (resonance-biased toward historically successful values) instead of
    uniformly at random - mirrors :func:`_recursive_template`'s bias/no-bias
    split exactly."""
    if not list_inputs:
        return None, None
    list_name = str(rng.choice(list_inputs))
    combine_op = str(rng.choice(FOLD_TEMPLATE_CATEGORIES["combine_op"]))
    item_kind = str(rng.choice(FOLD_TEMPLATE_CATEGORIES["item_kind"]))
    if bias is not None:
        cmp = bias.sample("cmp", rng)
        const = bias.sample("const", rng)
    else:
        cmp = str(rng.choice(FOLD_TEMPLATE_CATEGORIES["cmp"]))
        const = int(rng.choice(FOLD_TEMPLATE_CATEGORIES["const"]))
    choices = {"combine_op": combine_op, "item_kind": item_kind}
    if item_kind == "cmp_const":
        choices["cmp"] = cmp
        choices["const"] = const
    node = _build_fold_template_node(list_name, "_acc", "_item", choices)
    return node, choices


# The tunable holes for the fold template - mirrors _TUNABLE_HOLES exactly:
# only meaningful (and only ever populated in a choices dict) when
# item_kind == "cmp_const".
_FOLD_TUNABLE_HOLES = ("cmp", "const")


def _uniform_fold_hole(key: str, rng: np.random.Generator):
    options = FOLD_TEMPLATE_CATEGORIES[key]
    value = options[int(rng.integers(0, len(options)))]
    return int(value) if key == "const" else str(value)


def _mutate_fold_template_hole(node: Node, rng: np.random.Generator, bias: "ResonantBias | None") -> Node | None:
    """The fold-template counterpart to :func:`_mutate_template_hole`:
    redraw exactly one tunable hole-filler (``cmp`` or ``const``) of a
    fold-template-shaped node in place, rather than uniform random subtree
    replacement. Returns ``None`` if ``node`` doesn't structurally match the
    fold template shape or has no tunable hole (any ``item_kind`` other than
    ``cmp_const`` has none - a defensive fallback, not the common case, the
    same as ``_mutate_template_hole``'s)."""
    choices = extract_fold_template_choices(node)
    if choices is None or not isinstance(node, Fold):
        return None
    tunable = [k for k in _FOLD_TUNABLE_HOLES if k in choices]
    if not tunable:
        return None
    key = tunable[int(rng.integers(0, len(tunable)))]
    new_choices = dict(choices)
    new_choices[key] = bias.sample(key, rng) if bias is not None else _uniform_fold_hole(key, rng)

    new_body = _build_fold_template_body(node.var_acc, node.var_item, new_choices)
    return Fold(node.list_expr, node.init, node.var_acc, node.var_item, new_body)


def extract_fold_template_choices(node: Node) -> dict | None:
    """If ``node`` structurally matches the fold template shape (see
    :func:`_fold_template`) - regardless of whether it came from the
    template generator or was evolved into that shape by ordinary mutation/
    crossover - extract its hole-fillers for :meth:`ResonantBias.reinforce`.
    Returns ``None`` if it doesn't match. Only validates the body shape
    (``acc COMBINE_OP transform(item)``), never ``list_expr``/``init`` -
    mirrors :func:`extract_template_choices` only validating the recursive
    template's body, never its ``in_expr``, for the same reason: a
    template-shaped individual mutation/crossover has since altered
    elsewhere cannot be assumed to still have the canonical parts
    :func:`_fold_template` itself always builds."""
    if not isinstance(node, Fold):
        return None
    body = node.body
    if not (isinstance(body, BinOp) and isinstance(body.left, Var) and body.left.name == node.var_acc):
        return None
    if body.op not in FOLD_TEMPLATE_CATEGORIES["combine_op"]:
        return None
    item_expr = body.right
    if isinstance(item_expr, Var) and item_expr.name == node.var_item:
        choices = {"combine_op": body.op, "item_kind": "identity"}
    elif (
        isinstance(item_expr, BinOp)
        and item_expr.op == "*"
        and isinstance(item_expr.left, Var)
        and item_expr.left.name == node.var_item
        and isinstance(item_expr.right, Var)
        and item_expr.right.name == node.var_item
    ):
        choices = {"combine_op": body.op, "item_kind": "square"}
    elif (
        isinstance(item_expr, BinOp)
        and item_expr.op in FOLD_TEMPLATE_CATEGORIES["cmp"]
        and isinstance(item_expr.left, Var)
        and item_expr.left.name == node.var_item
        and isinstance(item_expr.right, Const)
    ):
        choices = {
            "combine_op": body.op,
            "item_kind": "cmp_const",
            "cmp": item_expr.op,
            "const": item_expr.right.value,
        }
    else:
        return None

    if any(choices[k] not in FOLD_TEMPLATE_CATEGORIES[k] for k in choices):
        return None
    return choices


# A structural template for compound boolean formulas - built after the
# Gregorian leap-year rule (year%4==0 and (year%100!=0 or year%400==0))
# turned out to be a genuine deceptive-fitness-landscape trap for blind
# growth: the search reliably converges to `not (year % 2)` ("is even"), a
# local optimum that satisfies every training example except the real
# century exceptions, with no smooth gradient from there to the true
# 4-clause structure. Confirmed by measurement, not assumed, that this is a
# structural problem rather than a budget one: 0/5 seeds verified at
# population=300/generations=80/depth=4; *adding more* century-exception
# training examples made it worse (0/8, some seeds not even reaching the
# "is even" energy); population=1000/generations=100/depth=6 still converged
# to exactly the same "is even" energy. Same character of problem
# _recursive_template and _fold_template were each built to solve.
#
# "shape_kind" picks one of four canonical 2-3-atom and/or compositions -
# drawn uniformly, never resonance-biased, for the same reason combine_kind/
# base_kind are excluded from bias in _recursive_template: it changes which
# sub-expressions exist at all, not a value within an already-fixed shape.
# Unlike combine_kind there's no asymmetric-failure-cost reason to anneal a
# schedule here (all four shapes are equal-cost BinOp compositions, no
# recursion, no exponential blowup), so a flat uniform draw is enough.
# "and_or3" is the leap-year target's literal shape
# (AND(atom, OR(atom, atom))); "or_and3" is not padding for symmetry - it is
# leap year's standard boolean-algebra dual (A&(B|C) == (A&B)|(A&C),
# specialized via year%400==0 implying year%100==0:
# OR(year%400==0, AND(year%4==0, year%100!=0))) - so the target is reachable
# through two independent structural doors, not one. "and2"/"or2" cover
# simpler compound targets (e.g. "divisible by both 3 and 5") and give the
# search bias an easy, fast-converging calibration case before the harder
# 3-atom shapes need it, the same role plain sum plays for the fold bias.
# "NOT" is deliberately not a shape: "!=" already gives negation where it
# matters, UnaryOp("not", ...) is already reachable by ordinary blind growth
# wrapping any subtree, and doubling the shape-matching surface for no
# capability gain matches this module's own tried-and-reverted pattern for
# unjustified complexity (see the abandoned independent step2 mentioned in
# _recursive_template's docstring).
#
# An atom is (Var(var) % Const(modulus)) CMP Const(const) - "modulus"/"cmp"/
# "const" are the tunable, resonance-biased holes. "modulus" is a curated
# list, not a derived range, for the same reason TEMPLATE_CATEGORIES["step"]
# is [1,2]: 2/3/5/7 cover parity/FizzBuzz/weekday-style small divisors,
# 4/100/400 are the exact three moduli leap year needs. "const" is
# non-negative only (unlike FOLD_TEMPLATE_CATEGORIES["const"], which can be
# negative): an atom compares against a modulus residue, which Python's `%`
# always returns non-negative for a positive modulus, so a negative const
# would be a dead, always-true-or-false atom - wasted search space, not
# useful expressiveness. The _LEAF_CONSTS constant-pool wall (no floats, no
# arbitrary literal - see agent.py's synthesize_door_predicate, which hit
# this same wall for a threshold like 0.35) is not a problem here: every
# value in this category list is built directly into a Const, the same way
# the other two templates build their own Consts directly - it never goes
# through _leaf().
BOOL_TEMPLATE_CATEGORIES: dict[str, list] = {
    "shape_kind": ["and2", "or2", "and_or3", "or_and3"],
    "modulus": [2, 3, 4, 5, 7, 100, 400],
    "cmp": ["==", "!=", "<", "<=", ">", ">="],
    "const": [0, 1, 2, 3, 4, 5, 6],
}


def _build_atom_node(var: str, atom: dict) -> Node:
    return BinOp(atom["cmp"], BinOp("%", Var(var), Const(atom["modulus"])), Const(atom["const"]))


def _build_bool_template_node(var: str, choices: dict) -> Node:
    """Build the whole tree from a complete choices dict - no separate
    "body vs. node" split the way _build_template_body/_build_fold_template_body
    have: those exist specifically to preserve a non-hole wrapper (in_expr,
    or init+list_expr) untouched during hole mutation. The bool template has
    no such wrapper - the entire returned node *is* the hole-filled region -
    so a single rebuild-everything builder is both correct and simpler."""
    atoms = choices["atoms"]
    shape = choices["shape_kind"]
    a0 = _build_atom_node(var, atoms[0])
    a1 = _build_atom_node(var, atoms[1])
    if shape == "and2":
        return BinOp("and", a0, a1)
    if shape == "or2":
        return BinOp("or", a0, a1)
    a2 = _build_atom_node(var, atoms[2])
    if shape == "and_or3":
        return BinOp("and", a0, BinOp("or", a1, a2))
    return BinOp("or", a0, BinOp("and", a1, a2))  # or_and3


def _bool_template(
    inputs: list[str],
    list_inputs: tuple[str, ...],
    rng: np.random.Generator,
    bias: "ResonantBias | None" = None,
) -> tuple[Node, dict] | tuple[None, None]:
    """A structural bias toward compound boolean formulas - see
    :data:`BOOL_TEMPLATE_CATEGORIES` for the shape/atom vocabulary. Returns
    ``(None, None)`` if there's no scalar input. All atoms share one
    variable, drawn once per template instance (like
    :func:`_recursive_template`'s ``seed_var``/:func:`_fold_template`'s
    ``list_name`` - a name, not a hole-filler) - the motivating target has
    exactly one scalar input, so independent per-atom variables would be
    untested complexity it doesn't need. If ``bias`` is given, each atom's
    ``modulus``/``cmp``/``const`` are drawn from it instead of uniformly at
    random - mirrors the other two templates' bias/no-bias split exactly."""
    scalars = [n for n in inputs if n not in list_inputs]
    if not scalars:
        return None, None
    var = str(rng.choice(scalars))
    shape = str(rng.choice(BOOL_TEMPLATE_CATEGORIES["shape_kind"]))
    n_atoms = 2 if shape in ("and2", "or2") else 3
    atoms = []
    for _ in range(n_atoms):
        if bias is not None:
            atoms.append(
                {"modulus": bias.sample("modulus", rng), "cmp": bias.sample("cmp", rng), "const": bias.sample("const", rng)}
            )
        else:
            atoms.append(
                {
                    "modulus": int(rng.choice(BOOL_TEMPLATE_CATEGORIES["modulus"])),
                    "cmp": str(rng.choice(BOOL_TEMPLATE_CATEGORIES["cmp"])),
                    "const": int(rng.choice(BOOL_TEMPLATE_CATEGORIES["const"])),
                }
            )
    choices = {"shape_kind": shape, "atoms": atoms}
    return _build_bool_template_node(var, choices), choices


_BOOL_TUNABLE_HOLES = ("modulus", "cmp", "const")


def _uniform_bool_hole(key: str, rng: np.random.Generator):
    options = BOOL_TEMPLATE_CATEGORIES[key]
    value = options[int(rng.integers(0, len(options)))]
    return int(value) if key in ("modulus", "const") else str(value)


def _mutate_bool_template_hole(node: Node, rng: np.random.Generator, bias: "ResonantBias | None") -> Node | None:
    """The bool-template counterpart to :func:`_mutate_fold_template_hole`:
    redraw exactly one tunable hole of exactly one atom in place. Returns
    ``None`` if ``node`` doesn't structurally match the bool template shape."""
    choices = extract_bool_template_choices(node)
    if choices is None:
        return None
    var = _bool_template_var(node)
    if var is None:
        return None
    atom_idx = int(rng.integers(0, len(choices["atoms"])))
    key = _BOOL_TUNABLE_HOLES[int(rng.integers(0, len(_BOOL_TUNABLE_HOLES)))]
    new_choices = {"shape_kind": choices["shape_kind"], "atoms": [dict(a) for a in choices["atoms"]]}
    new_choices["atoms"][atom_idx][key] = bias.sample(key, rng) if bias is not None else _uniform_bool_hole(key, rng)
    return _build_bool_template_node(var, new_choices)


def _match_atom(node: Node, var: str) -> dict | None:
    """If ``node`` is ``(Var(var) % Const(modulus)) CMP Const(const)``,
    return ``{"modulus", "cmp", "const"}``; else ``None``."""
    if not isinstance(node, BinOp) or node.op not in BOOL_TEMPLATE_CATEGORIES["cmp"]:
        return None
    left, right = node.left, node.right
    if not (
        isinstance(left, BinOp)
        and left.op == "%"
        and isinstance(left.left, Var)
        and left.left.name == var
        and isinstance(left.right, Const)
    ):
        return None
    if not isinstance(right, Const):
        return None
    return {"modulus": left.right.value, "cmp": node.op, "const": right.value}


def _bool_template_var(node: Node) -> str | None:
    """The shared variable name every atom in a bool-template-shaped
    ``node`` refers to, or ``None`` if ``node`` doesn't match at all -
    extracted once, up front, so :func:`extract_bool_template_choices` and
    :func:`_mutate_bool_template_hole` don't each re-derive it."""
    candidates: list[Node] = []
    if isinstance(node, BinOp) and node.op in ("and", "or"):
        candidates.append(node.left)
        if isinstance(node.right, BinOp) and node.right.op in ("and", "or"):
            candidates.append(node.right.left)
            candidates.append(node.right.right)
        else:
            candidates.append(node.right)
    for candidate in candidates:
        if (
            isinstance(candidate, BinOp)
            and isinstance(candidate.left, BinOp)
            and candidate.left.op == "%"
            and isinstance(candidate.left.left, Var)
        ):
            return candidate.left.left.name
    return None


def extract_bool_template_choices(node: Node) -> dict | None:
    """If ``node`` structurally matches one of the four bool template shapes
    (see :func:`_bool_template`) - regardless of whether it came from the
    template generator or was evolved into that shape by ordinary mutation/
    crossover - extract its hole-fillers for :meth:`ResonantBias.reinforce`.
    Returns ``None`` if it doesn't match any shape, or if it matches
    structurally but the shared variable differs between atoms (blind
    growth/crossover can produce that; it isn't a shape this template
    reinforces or refines)."""
    if not isinstance(node, BinOp) or node.op not in ("and", "or"):
        return None
    var = _bool_template_var(node)
    if var is None:
        return None
    a0 = _match_atom(node.left, var)
    if a0 is None:
        return None
    if isinstance(node.right, BinOp) and node.right.op in ("and", "or") and node.right.op != node.op:
        a1 = _match_atom(node.right.left, var)
        a2 = _match_atom(node.right.right, var)
        if a1 is None or a2 is None:
            return None
        shape = "and_or3" if node.op == "and" else "or_and3"
        atoms = [a0, a1, a2]
    else:
        a1 = _match_atom(node.right, var)
        if a1 is None:
            return None
        shape = "and2" if node.op == "and" else "or2"
        atoms = [a0, a1]

    if shape not in BOOL_TEMPLATE_CATEGORIES["shape_kind"]:
        return None
    for atom in atoms:
        if any(atom[k] not in BOOL_TEMPLATE_CATEGORIES[k] for k in atom):
            return None
    return {"shape_kind": shape, "atoms": atoms}


def root_shape(node: Node) -> tuple[str, object]:
    """An automatically-derived, hand-declaration-free family signature for
    `synthesize`'s "shape elitism" (see :data:`GRAMMAR_CHOICE_VOCAB`'s
    neighbourhood and `docs/ROADMAP.md` v0.30): unlike `combine_kind`/
    `base_kind`/`shape_kind`, which only exist for the three hand-built
    templates' own shapes, this applies to *any* individual regardless of
    how it was grown - just the root node's DSL type and, if it has one,
    its operator (``None`` otherwise, e.g. for `Letrec`/`Fold`/`If`)."""
    return (type(node).__name__, getattr(node, "op", None))


# Hole categories a template's *value* (not its structural family) is built
# from - the ones _mutate_template_hole is allowed to retarget one at a time.
# Mirrors _recursive_template's own bias/no-bias split: combine_kind, delta,
# and base_kind are structural and excluded here for the same reason they're
# never resonance-biased (see _recursive_template's docstring).
_TUNABLE_HOLES = ("cmp", "base_const", "base_val", "op", "step")

# Guaranteed hole-mutated offspring of the best template-shaped individual,
# produced every generation regardless of fitness-proportionate selection -
# see synthesize's template-elitism comment for why this is necessary, not
# just the protected elite slot itself.
_TEMPLATE_REFINEMENT_OFFSPRING = 8

# Share of the *run's own* max_generations budget an attempt can spend with
# zero improvement to its own best energy before synthesize() discards the
# whole population and starts a fresh one (see the "Random restart" comment
# in synthesize() for the full rationale) - a fraction of max_generations,
# not a fixed generation count, because a fixed count is a trap: a first cut
# at this used a flat 40, which seemed safely above what a "genuinely
# progressing" search should ever need - but measuring it against seed 1
# (previously reliable) found that assumption false. That seed's low wall-
# clock time in earlier sweeps was cheap-per-generation cost, not an early
# finish - it actually uses the *entire* 150-generation budget, including
# stretches well past 40 generations with no improvement, as a normal part
# of succeeding. A fixed 40 restarted it mid-convergence and lost the run
# entirely. Tying this to a fraction of whatever budget the caller actually
# gave means a long stagnation stretch is only ever judged "too long"
# relative to how much runway this call has to offer, not against a number
# tuned for one specific max_generations value.
_RESTART_STAGNATION_FRACTION = 0.6

# Probability that mutate() takes the surgical single-hole path on a
# template-shaped individual instead of falling through to full scoped
# regrowth. Must stay < 1: hole mutation only ever changes one field at a
# time, so a template stuck in a Hamming-distance-1 local optimum (no single
# hole change improves it, but the right combination is two or more away)
# has no way out if it's the *only* path mutate() ever takes on template
# shapes - measured directly by instrumenting a run on the 2**n target
# (seed 4): a family reached energy 1.0 at generation 0 and, with hole
# mutation as the sole path, sat at exactly 1.0 through generation 59 while
# its lineage grew to ~60% of the population. Full regrowth (drawing an
# entirely fresh (cmp, base_const, base_val, op, step) combination at once
# via _recursive_template) is what lets a stuck lineage jump past that kind
# of plateau instead of only ever hill-climbing from wherever it first got
# lucky.
#
# 0.5 is a genuine trade-off, not an arbitrary pick - confirmed by sweeping
# it (see docs/ROADMAP.md v0.20's seed-8 entry). Lowering it to 0.2 fixes a
# `2**n` seed (8) that fails at 0.5, but breaks
# test_resonant_bias_can_discover_fibonacci itself (seed 1 stops
# generalizing) - the same whack-a-mole shape as the delta_p1 schedule
# above. 0.5 is kept because it's the value that keeps every seed the
# committed tests actually assert passing; don't retune this without
# re-running that full seed sweep.
_TEMPLATE_HOLE_MUTATION_RATE = 0.5

# The fold-template counterpart to _TEMPLATE_HOLE_MUTATION_RATE. Same
# default: no evidence yet that fold's much smaller hole space (two
# tunable holes, cmp/const, only relevant for one of three item_kinds)
# needs a different rate, and re-using the value already validated
# against a full seed sweep is safer than guessing a new one.
_FOLD_TEMPLATE_HOLE_MUTATION_RATE = 0.5

# The bool-template counterpart. Same reasoning again: no evidence this
# template's hole space (three tunable holes per atom, 2-3 atoms) needs a
# different rate from the two already validated at 0.5.
_BOOL_TEMPLATE_HOLE_MUTATION_RATE = 0.5


def _mutate_template_hole(node: Node, rng: np.random.Generator, bias: "ResonantBias | None") -> Node | None:
    """Redraw exactly *one* hole-filler of a template-shaped node - the
    surgical counterpart to :func:`mutate`'s uniform random-subtree
    replacement, needed because a hole-filler like ``op`` or ``cmp`` is a
    plain string *attribute* of a ``BinOp`` node, not its own tree node:
    ``mutate`` can only ever replace the *entire* surrounding subtree (and
    get lucky on every field at once via fresh ``_grow`` growth), never tune
    one field of an otherwise-good template in place.

    Found necessary while diagnosing why Fibonacci discovery was unreliable
    on some seeds (``docs/ROADMAP.md`` v0.20): even after protecting the
    best template-shaped individual from being lost to selection (see
    ``synthesize``'s template elitism), its energy stayed stuck for the rest
    of the run - protecting good genetic material is useless if nothing can
    actually refine it. Never retargets ``combine_kind``/``delta``/
    ``base_kind`` (see :data:`_TUNABLE_HOLES`) - those are structural
    choices, and randomly flipping one mid-refinement would discard however
    much of the surrounding tree's fit already depends on the current
    family, the same reasoning ``_recursive_template`` already applies to
    keep those choices out of ``ResonantBias``.

    Returns ``None`` if ``node`` doesn't structurally match the template
    shape (mirroring :func:`extract_template_choices`) or has no tunable
    hole at all (only ``base_kind="param"`` templates with a
    ``param_recur`` combine and no ``base_val``/``delta`` fields could, in
    principle, hit this - in practice ``cmp``/``op``/``step`` are always
    present, so this is a defensive fallback, not the common case).
    """
    choices = extract_template_choices(node)
    if choices is None or not isinstance(node, Letrec):
        return None
    tunable = [k for k in _TUNABLE_HOLES if k in choices]
    if not tunable:
        return None
    key = tunable[int(rng.integers(0, len(tunable)))]
    new_choices = dict(choices)
    new_choices[key] = bias.sample(key, rng) if bias is not None else _uniform_hole(key, rng)

    new_body = _build_template_body(node.name, node.params[0], new_choices)
    return Letrec(node.name, node.params, new_body, node.in_expr)


def _uniform_hole(key: str, rng: np.random.Generator):
    options = TEMPLATE_CATEGORIES[key]
    value = options[int(rng.integers(0, len(options)))]
    return int(value) if key in ("base_const", "base_val", "step") else str(value)


def _is_decrement_of(expr: Node, param: str) -> bool:
    return (
        isinstance(expr, BinOp)
        and expr.op == "-"
        and isinstance(expr.left, Var)
        and expr.left.name == param
        and isinstance(expr.right, Const)
    )


def extract_template_choices(node: Node) -> dict | None:
    """If ``node`` structurally matches either recursive template shape (see
    :func:`_recursive_template`) - regardless of whether it came from the
    template generator or was evolved into that shape by ordinary mutation/
    crossover - extract its hole-fillers for :meth:`ResonantBias.reinforce`.
    Returns ``None`` if it doesn't match either shape. Matching on observed
    *structure* rather than provenance means the bias learns from whatever
    the population actually converges on, not just from candidates the
    template generator happened to produce.
    """
    if not isinstance(node, Letrec) or len(node.params) != 1:
        return None
    param = node.params[0]
    body = node.body
    if not isinstance(body, If):
        return None
    cond = body.cond
    if not (
        isinstance(cond, BinOp)
        and isinstance(cond.left, Var)
        and cond.left.name == param
        and isinstance(cond.right, Const)
    ):
        return None
    # base case: either Const(base_val) (base_kind="const") or the bare
    # parameter itself (base_kind="param" - what makes a true zero-indexed
    # Fibonacci expressible, see _recursive_template's docstring).
    if isinstance(body.then, Const):
        base_kind_fields: dict = {"base_kind": "const", "base_val": body.then.value}
    elif isinstance(body.then, Var) and body.then.name == param:
        base_kind_fields = {"base_kind": "param"}
    else:
        return None
    combine = body.orelse
    if not isinstance(combine, BinOp):
        return None

    base = {"cmp": cond.op, "base_const": cond.right.value, "op": combine.op, **base_kind_fields}

    # double_recur: f(p - step) OP f(p - step - delta) - symmetric (delta=0)
    # or asymmetric (delta>0), regardless of which side has the larger
    # decrement (mutation/crossover can swap the two Recur args' order).
    if (
        isinstance(combine.left, Recur)
        and isinstance(combine.right, Recur)
        and combine.left.name == node.name
        and combine.right.name == node.name
        and len(combine.left.args) == 1
        and len(combine.right.args) == 1
        and _is_decrement_of(combine.left.args[0], param)
        and _is_decrement_of(combine.right.args[0], param)
    ):
        step_a = combine.left.args[0].right.value
        step_b = combine.right.args[0].right.value
        choices = {**base, "step": min(step_a, step_b), "delta": abs(step_a - step_b), "combine_kind": "double_recur"}
    # param_recur: p OP f(p - step)
    elif (
        isinstance(combine.left, Var)
        and combine.left.name == param
        and isinstance(combine.right, Recur)
        and combine.right.name == node.name
        and len(combine.right.args) == 1
        and _is_decrement_of(combine.right.args[0], param)
    ):
        choices = {**base, "step": combine.right.args[0].right.value, "combine_kind": "param_recur"}
    else:
        return None

    if any(choices[k] not in TEMPLATE_CATEGORIES[k] for k in choices):
        return None  # a value outside the known category set - not reinforceable
    return choices


def random_program(
    inputs: list[str],
    rng: np.random.Generator,
    max_depth: int = 4,
    list_inputs: tuple[str, ...] = (),
    template_rate: float = 0.15,
    allow_recursion: bool = False,
    bias: ResonantBias | None = None,
    delta_p1: float = 0.15,
    fold_bias: "ResonantBias | None" = None,
    allow_bool_template: bool = False,
    bool_bias: "ResonantBias | None" = None,
    allow_fold_template: bool = True,
    allow_motif_bias: bool = False,
    motif_archive: "MotifArchive | None" = None,
    allow_recursion_template: bool = True,
    allow_grammar_bias: bool = False,
    grammar_bias: "GrammarBias | None" = None,
) -> Node:
    """A randomly-grown candidate program over the searchable DSL.

    ``allow_recursion`` defaults to ``False``: ``Letrec``/``Recur`` are
    opt-in, not part of the default grammar, because leaving them always
    available made *every* search meaningfully more expensive per candidate
    (see the module docstring) even for targets that never needed recursion.
    When ``allow_recursion`` is set, ``template_rate`` is the probability of
    returning a :func:`_recursive_template` skeleton instead of pure random
    growth - see its docstring for why that matters for actually finding
    recursive solutions. ``bias`` (a :class:`ResonantBias`), if given, steers
    the template's hole-fillers toward historically successful values instead
    of uniform random choice. ``delta_p1`` is the probability of the
    asymmetric ``double_recur`` variant (see :func:`_recursive_template`).

    ``fold_bias``, if given, is a second, independent :class:`ResonantBias`
    over :data:`FOLD_TEMPLATE_CATEGORIES` - reusing the same ``template_rate``
    to decide whether to return a :func:`_fold_template` skeleton instead of
    pure random growth when ``list_inputs`` is non-empty. Checked *after* the
    recursion template so the two never compete for the same draw when both
    are available (recursion synthesis and list-op synthesis are not
    currently combined in any committed target).

    ``allow_bool_template``, if set, is a *third* independent opt-in (checked
    last), for :func:`_bool_template` (compound boolean formulas over
    :data:`BOOL_TEMPLATE_CATEGORIES`). Unlike ``fold_bias``'s self-selecting
    ``if list_inputs`` gate, this needs its own flag: a bool template's
    activation condition ("has a scalar input") is true for almost every
    call, so leaving it unconditionally live would waste ``template_rate``
    draws on and/or skeletons for plain arithmetic targets - the same
    diversity-dilution concern that made ``allow_recursion`` itself opt-in,
    even though (unlike recursion) a bool candidate is cheap to evaluate.
    ``bool_bias`` mirrors ``bias``/``fold_bias`` for this third template.

    ``allow_fold_template`` (default ``True``) is a kill switch for
    :func:`_fold_template`, added so callers can run a controlled comparison
    against ``allow_motif_bias`` on the same list-processing targets without
    the hand-built fold skeleton also being available - defaulting to
    ``True`` keeps every existing caller's behavior unchanged.

    ``allow_motif_bias``/``motif_archive`` enable motif resonance (see
    :mod:`.motif_bias`) - a fourth, independent, opt-in mechanism that
    reinforces and reuses actual subtrees from the population's own history,
    addressed by structural context, instead of a hand-built skeleton.
    Forwarded straight through to :func:`_grow`, which is where the
    mechanism actually operates (at every grow site, not just this
    top-level entry point).

    ``allow_recursion_template`` (default ``True``) is a kill switch for
    :func:`_recursive_template` specifically, decoupled from ``allow_recursion``
    (which also controls whether ``letrec``/``recur`` are in ``_choose_kind``'s
    pool at all) - added, like ``allow_fold_template``, purely so a caller can
    run recursion synthesis (``allow_recursion=True``) with motif resonance
    instead of the hand-built template, for a controlled comparison. Default
    ``True`` keeps every existing caller's behavior unchanged.

    ``allow_grammar_bias``/``grammar_bias`` enable production-level PCFG
    resonance (see :mod:`.grammar_bias`) - a fifth, independent mechanism,
    materially different from motif resonance: instead of reusing whole
    subtrees, it biases individual production choices (which ``BinOp``
    operator, which ``UnaryOp`` operator, ``Var`` vs ``Const``, which
    constant) so a compound shape can assemble from marginal pushes on
    separate choice-points that never co-occurred in any single ancestor -
    see ``docs/ROADMAP.md`` v0.29 for the honest measured comparison against
    both blind growth and motif resonance. Unlike every other bias in this
    module, ``grammar_bias`` may be supplied already-populated by the caller
    (e.g. loaded via :func:`.grammar_bias.load_grammar_bias`) to persist
    learned structure *across* calls to :func:`synthesize` - see that
    function's docstring for the ownership rule this implies.
    """
    if allow_recursion and allow_recursion_template and template_rate > 0 and rng.random() < template_rate:
        template, _choices = _recursive_template(inputs, tuple(list_inputs), rng, bias, delta_p1)
        if template is not None:
            return template
    if list_inputs and allow_fold_template and template_rate > 0 and rng.random() < template_rate:
        fold_template, _fold_choices = _fold_template(tuple(list_inputs), rng, fold_bias)
        if fold_template is not None:
            return fold_template
    if allow_bool_template and template_rate > 0 and rng.random() < template_rate:
        bool_template, _bool_choices = _bool_template(inputs, tuple(list_inputs), rng, bool_bias)
        if bool_template is not None:
            return bool_template
    return _grow(
        list(inputs),
        tuple(list_inputs),
        (),
        rng,
        max_depth,
        allow_recursion,
        template_rate=template_rate,
        allow_motif_bias=allow_motif_bias,
        motif_archive=motif_archive,
        grammar_bias=grammar_bias if allow_grammar_bias else None,
    )


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
    template_rate: float = 0.15,
    allow_recursion: bool = False,
    bias: ResonantBias | None = None,
    delta_p1: float = 0.15,
    fold_bias: "ResonantBias | None" = None,
    allow_bool_template: bool = False,
    bool_bias: "ResonantBias | None" = None,
    allow_fold_template: bool = True,
    allow_motif_bias: bool = False,
    motif_archive: "MotifArchive | None" = None,
    allow_recursion_template: bool = True,
    allow_grammar_bias: bool = False,
    grammar_bias: "GrammarBias | None" = None,
) -> Node:
    """Like :func:`_replace_at`, but regenerates the replacement using the
    (inputs, list_inputs, recur_ctx) actually valid at the target position -
    mirroring how :func:`_grow` would have built that subtree in the first
    place, so a mutation that introduces e.g. a fresh ``Recur`` call has a
    real chance of being well-scoped rather than an (safely, but uselessly)
    unbound-name penalty. ``fold_bias``/``allow_bool_template``/``bool_bias``
    mirror ``bias`` for :func:`_fold_template`/:func:`_bool_template` - see
    :func:`random_program`'s docstring. ``allow_fold_template``/
    ``allow_motif_bias``/``motif_archive`` mirror :func:`random_program`'s
    own params of the same name.

    Note: the single :func:`_grow` call at the target index below passes
    ``parent_kind="root", child_slot=0`` rather than the true position
    immediately above the mutation point - an honest simplification, not an
    oversight: threading the exact surrounding context through this
    function's own many node-type branches would roughly double this
    function's size for a benefit that only matters for the *first* level of
    whatever gets regrown here, since ``_grow``'s own internal recursion
    already tracks true context precisely for everything *below* that first
    level, regardless of whether the call originated from here or from
    :func:`random_program`'s top-level (also ``root``/``0``) call.
    """
    idx = counter[0]
    counter[0] += 1
    if idx == target_index:
        if allow_recursion and allow_recursion_template and template_rate > 0 and rng.random() < template_rate:
            template, _choices = _recursive_template(inputs, list_inputs, rng, bias, delta_p1)
            if template is not None:
                return template
        if list_inputs and allow_fold_template and template_rate > 0 and rng.random() < template_rate:
            fold_node, _fold_choices = _fold_template(list_inputs, rng, fold_bias)
            if fold_node is not None:
                return fold_node
        if allow_bool_template and template_rate > 0 and rng.random() < template_rate:
            bool_node, _bool_choices = _bool_template(inputs, list_inputs, rng, bool_bias)
            if bool_node is not None:
                return bool_node
        return _grow(
            inputs,
            list_inputs,
            recur_ctx,
            rng,
            max(1, max_depth - 1),
            allow_recursion,
            template_rate=template_rate,
            allow_motif_bias=allow_motif_bias,
            motif_archive=motif_archive,
            grammar_bias=grammar_bias if allow_grammar_bias else None,
        )

    scalars = [n for n in inputs if n not in list_inputs]
    kwargs = {
        "template_rate": template_rate,
        "allow_recursion": allow_recursion,
        "bias": bias,
        "delta_p1": delta_p1,
        "fold_bias": fold_bias,
        "allow_bool_template": allow_bool_template,
        "bool_bias": bool_bias,
        "allow_fold_template": allow_fold_template,
        "allow_motif_bias": allow_motif_bias,
        "motif_archive": motif_archive,
        "allow_recursion_template": allow_recursion_template,
        "allow_grammar_bias": allow_grammar_bias,
        "grammar_bias": grammar_bias,
    }

    if isinstance(node, Let):
        new_value = _replace_at_scoped(
            node.value, target_index, rng, max_depth, counter, inputs, list_inputs, recur_ctx, **kwargs
        )
        new_body = _replace_at_scoped(
            node.body, target_index, rng, max_depth, counter, inputs + [node.name], list_inputs, recur_ctx, **kwargs
        )
        return Let(node.name, new_value, new_body)
    if isinstance(node, Fold):
        new_list_expr = _replace_at_scoped(
            node.list_expr, target_index, rng, max_depth, counter, inputs, list_inputs, recur_ctx, **kwargs
        )
        new_init = _replace_at_scoped(
            node.init, target_index, rng, max_depth, counter, inputs, list_inputs, recur_ctx, **kwargs
        )
        body_inputs = scalars + [node.var_acc, node.var_item]
        new_body = _replace_at_scoped(
            node.body, target_index, rng, max_depth, counter, body_inputs, (), recur_ctx, **kwargs
        )
        return Fold(new_list_expr, new_init, node.var_acc, node.var_item, new_body)
    if isinstance(node, (Map, Filter)):
        new_list_expr = _replace_at_scoped(
            node.list_expr, target_index, rng, max_depth, counter, inputs, list_inputs, recur_ctx, **kwargs
        )
        body_inputs = scalars + [node.var_item]
        body_attr = node.body if isinstance(node, Map) else node.predicate
        new_body = _replace_at_scoped(
            body_attr, target_index, rng, max_depth, counter, body_inputs, (), recur_ctx, **kwargs
        )
        cls = Map if isinstance(node, Map) else Filter
        return cls(new_list_expr, node.var_item, new_body)
    if isinstance(node, Letrec):
        new_recur_ctx = recur_ctx + ((node.name, len(node.params)),)
        body_inputs = scalars + list(node.params)
        new_body = _replace_at_scoped(
            node.body, target_index, rng, max_depth, counter, body_inputs, (), new_recur_ctx, **kwargs
        )
        new_in_expr = _replace_at_scoped(
            node.in_expr, target_index, rng, max_depth, counter, inputs, list_inputs, new_recur_ctx, **kwargs
        )
        return Letrec(node.name, node.params, new_body, new_in_expr)
    if isinstance(node, Recur):
        new_args = [
            _replace_at_scoped(a, target_index, rng, max_depth, counter, inputs, list_inputs, recur_ctx, **kwargs)
            for a in node.args
        ]
        return Recur(node.name, tuple(new_args))
    if isinstance(node, ListLit):
        new_items = [
            _replace_at_scoped(item, target_index, rng, max_depth, counter, inputs, list_inputs, recur_ctx, **kwargs)
            for item in node.items
        ]
        return ListLit(tuple(new_items))

    # Generic same-scope-for-all-children nodes: BinOp, UnaryOp, If, Length, Index.
    kids = _children(node)
    if not kids:
        return node
    new_kids = [
        _replace_at_scoped(child, target_index, rng, max_depth, counter, inputs, list_inputs, recur_ctx, **kwargs)
        for child in kids
    ]
    return rebuild(node, new_kids)


def mutate(
    node: Node,
    rng: np.random.Generator,
    inputs: list[str],
    max_depth: int = 4,
    list_inputs: tuple[str, ...] = (),
    template_rate: float = 0.15,
    allow_recursion: bool = False,
    bias: ResonantBias | None = None,
    delta_p1: float = 0.15,
    fold_bias: "ResonantBias | None" = None,
    allow_bool_template: bool = False,
    bool_bias: "ResonantBias | None" = None,
    allow_fold_template: bool = True,
    allow_motif_bias: bool = False,
    motif_archive: "MotifArchive | None" = None,
    allow_recursion_template: bool = True,
    allow_grammar_bias: bool = False,
    grammar_bias: "GrammarBias | None" = None,
) -> Node:
    """Replace a randomly chosen subtree with a freshly generated one, scoped
    correctly for that position (see module docstring).

    If ``node`` already structurally matches the recursive template shape
    (see :func:`extract_template_choices`), this calls
    :func:`_mutate_template_hole` to surgically retune one hole-filler in
    place *with probability* :data:`_TEMPLATE_HOLE_MUTATION_RATE`, rather than
    uniform-random subtree replacement: found necessary empirically
    (``docs/ROADMAP.md`` v0.20) - ordinary structural mutation on a good
    template essentially never improves it, because a hole-filler like
    ``op``/``cmp`` is a plain attribute, not its own tree node, so the only
    way structural mutation can "fix" it is by regenerating (and getting
    lucky on) the entire surrounding subtree at once. But this can't be the
    *only* path taken on a template-shaped node: hole mutation changes one
    field at a time, so a template stuck in a Hamming-distance-1 local
    optimum (no single hole change improves it, the right combination is two
    or more away) needs the full-regrowth path below to actually escape -
    confirmed by instrumenting a stuck run (see
    :data:`_TEMPLATE_HOLE_MUTATION_RATE`'s comment). Falling through at the
    complementary probability lets both mechanisms coexist instead of one
    eating the other.

    The same split applies, independently, to the fold template
    (:data:`_FOLD_TEMPLATE_HOLE_MUTATION_RATE`, :func:`_mutate_fold_template_hole`)
    when ``node`` is ``Fold``-shaped - not gated on ``allow_recursion``, since
    fold synthesis is orthogonal to recursion synthesis. The bool template
    (:data:`_BOOL_TEMPLATE_HOLE_MUTATION_RATE`, :func:`_mutate_bool_template_hole`)
    gets the same split too, but *is* gated on its own ``allow_bool_template``
    flag rather than an ``isinstance`` check: unlike ``Fold``, a bool-
    template-shaped node is just an ordinary ``BinOp`` (the same node type as
    any arithmetic/comparison expression), so there's no type to key off of -
    gating on the flag keeps ``allow_bool_template=False`` meaning "off"
    cleanly, the same way ``allow_recursion`` gates both its own entry points
    together.
    """
    if allow_recursion and allow_recursion_template and rng.random() < _TEMPLATE_HOLE_MUTATION_RATE:
        hole_mutated = _mutate_template_hole(node, rng, bias)
        if hole_mutated is not None:
            return hole_mutated
    if allow_fold_template and isinstance(node, Fold) and rng.random() < _FOLD_TEMPLATE_HOLE_MUTATION_RATE:
        fold_hole_mutated = _mutate_fold_template_hole(node, rng, fold_bias)
        if fold_hole_mutated is not None:
            return fold_hole_mutated
    if allow_bool_template and rng.random() < _BOOL_TEMPLATE_HOLE_MUTATION_RATE:
        bool_hole_mutated = _mutate_bool_template_hole(node, rng, bool_bias)
        if bool_hole_mutated is not None:
            return bool_hole_mutated
    n = count_nodes(node)
    target = int(rng.integers(0, n))
    return _replace_at_scoped(
        node,
        target,
        rng,
        max_depth,
        [0],
        list(inputs),
        tuple(list_inputs),
        (),
        template_rate=template_rate,
        allow_recursion=allow_recursion,
        bias=bias,
        delta_p1=delta_p1,
        fold_bias=fold_bias,
        allow_bool_template=allow_bool_template,
        bool_bias=bool_bias,
        allow_fold_template=allow_fold_template,
        allow_motif_bias=allow_motif_bias,
        motif_archive=motif_archive,
        allow_recursion_template=allow_recursion_template,
        allow_grammar_bias=allow_grammar_bias,
        grammar_bias=grammar_bias,
    )


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


def program_energy(node: Node, examples: list[Example], fuel_budget: int = 60) -> float:
    """Sum of per-example mismatch penalties. A crash, type error, or fuel
    exhaustion on any example contributes a fixed high penalty - the search
    never sees an uncaught exception. ``_mismatch`` itself is inside the
    guard too: e.g. a nested list (from a badly-generated ``Map``/``Filter``
    whose body itself returns a list) can raise inside the comparison, not
    just inside ``evaluate`` - that must be penalized the same way, not
    allowed to crash the whole search.

    ``fuel_budget`` defaults to a modest 150, not the DSL's own generous
    defaults: once recursion is in the search, individuals that recurse many
    times before hitting a base case (or failing to) are meaningfully more
    expensive to evaluate than shallow non-recursive ones, and a candidate
    population is mostly *not* going to be the answer - keeping the per-
    candidate ceiling low keeps the search's aggregate cost bounded without
    hurting real solutions, none of which need anywhere near 150 steps for
    the small examples this module's targets use.
    """
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
    allow_recursion: bool = False,
    template_rate: float = 0.15,
    allow_bool_template: bool = False,
    allow_fold_template: bool = True,
    allow_motif_bias: bool = False,
    allow_recursion_template: bool = True,
    allow_grammar_bias: bool = False,
    grammar_bias: "GrammarBias | None" = None,
    allow_shape_elitism: bool = False,
    resonant_bias: bool = True,
    fuel_budget: int = 60,
    delta_p1_start: float = 0.15,
    delta_p1_max: float = 0.5,
    delta_p1_stagnation_growth: float = 1.08,
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

    ``allow_recursion`` defaults to ``False``: ``Letrec``/``Recur`` are
    opt-in. Leaving them in the default grammar was tried and reverted after
    measuring a real cost - every search got meaningfully slower per
    candidate (recursive candidates, even safe ones, cost more to evaluate),
    regardless of whether the target needed recursion. Set it to ``True`` to
    attempt recursion synthesis; ``template_rate`` (only relevant when
    ``allow_recursion=True``) is the probability that a freshly-generated
    subtree is a :func:`_recursive_template` skeleton instead of pure random
    growth - see its docstring for why that matters for actually *finding*
    recursive solutions, not just safely evaluating hand-built ones.

    ``resonant_bias`` (only relevant when ``allow_recursion=True``) enables
    :class:`~zeuss.tier4_synthesis.resonance_bias.ResonantBias`: an
    Estimation-of-Distribution prior over the template's hole-fillers, read
    and written via the same hypervector resonance machinery the rest of
    this project uses instead of a bolted-on statistics table. Every
    generation, any population member that structurally matches the template
    shape (see :func:`extract_template_choices` - regardless of whether it
    came from the template generator or was evolved into that shape)
    reinforces the bias in proportion to ``exp(-energy)``, so later
    generations' template draws lean toward hole-fillers that have actually
    correlated with lower energy in *this run*, not a fixed prior.

    ``delta_p1_*`` control a second, independent liquid-time-step: the
    probability that a freshly-generated ``double_recur`` template is
    asymmetric (see :func:`_recursive_template`'s ``delta_p1`` parameter).
    Unlike the hole-filler bias above, this is deliberately *not* learned via
    :class:`ResonantBias` - a structural/family choice like this risks
    premature convergence to the wrong family from early noise (see that
    class's docstring). Instead it anneals on a fixed schedule keyed to
    *stagnation* (generations since ``best_energy`` last improved), the same
    "adapt the step to how hard progress currently is" idea as
    :func:`zeuss.tier2_substrate.collapse.anneal_adaptive`: while the search
    is still improving, ``delta_p1`` stays at ``delta_p1_start`` (low, since
    symmetric recursion is the common case and an unbiased 50/50 split was
    measured to slow down symmetric targets like ``2**n``); once a run stalls
    (no improvement for several generations - the sign a purely-symmetric
    search is exhausted, or a symmetric prior is actively fighting an
    asymmetric target like Fibonacci), it grows geometrically by
    ``delta_p1_stagnation_growth`` per stagnant generation, capped at
    ``delta_p1_max``, and resets to ``delta_p1_start`` the moment progress
    resumes. This was measured to fix a real, previously-hidden failure mode:
    a *static* skew (e.g. 85/15) was tried first and only partially helped -
    it let one bad seed for ``2**n`` eventually converge but at ~500x the
    unbiased cost (219s / 104 generations vs. 0.4s), while a large enough
    static skew to fix that seed broke Fibonacci discovery outright. The
    stagnation-triggered schedule instead only pays the asymmetric-search
    cost on runs that actually need it.

    ``allow_fold_template`` (default ``True``) is a kill switch for
    :func:`_fold_template`, existing solely so a caller can measure motif
    resonance against the same target on a level playing field (the fold
    template otherwise has no opt-out, unlike recursion's ``allow_recursion``
    and bool's ``allow_bool_template``) - default ``True`` means every
    existing caller's behavior is unchanged.

    ``allow_motif_bias`` enables motif resonance (see :mod:`.motif_bias`): a
    fourth, independent, opt-in mechanism that reinforces and reuses actual
    subtrees pulled from the population's own low-energy individuals,
    addressed by the structural context they were grown at, instead of a
    human hand-designing a whole skeleton the way the other three templates
    each needed. Every generation, :meth:`~.motif_bias.MotifArchive.
    reinforce_from_population` walks the scored population (mirroring the
    other three biases' own reinforcement loops) and :func:`_grow` may splice
    in an archived subtree instead of growing fresh, at *every* grow site,
    not just a template's fixed set of named holes. Default ``False``: a
    true no-op for every existing caller, and *not* a claim that it matches
    the reliability the three hand-built templates eventually reached after
    several dedicated versions each - see ``docs/ROADMAP.md`` v0.28 for the
    honest, measured comparison.

    ``allow_grammar_bias``/``grammar_bias`` enable production-level PCFG
    resonance (see :mod:`.grammar_bias`) - a fifth, independent mechanism,
    materially different from motif resonance: it biases individual
    production choices (which ``BinOp``/``UnaryOp`` operator, ``Var`` vs
    ``Const``, which constant), not whole subtrees, so a compound shape can
    assemble from marginal pushes on separate choice-points that never
    co-occurred in any single ancestor - see ``docs/ROADMAP.md`` v0.29 for
    the honest measured comparison against both blind growth and motif
    resonance. Unlike every other bias in this function, a caller may supply
    an already-populated ``grammar_bias`` (e.g. loaded via
    :func:`.grammar_bias.load_grammar_bias`) to persist learned structure
    *across* calls to this function - when supplied, it is reinforced in
    place and never replaced, including across this call's own internal
    stagnation-triggered restarts (a caller-supplied bias is caller-owned).
    Passing ``grammar_bias=None`` with ``allow_grammar_bias=True`` builds and
    discards a fresh one internally instead, exactly like the other four
    biases - a purely session-scoped use, with no persistence implied.
    Default ``False``: a true no-op for every existing caller.

    ``allow_shape_elitism`` enables a sixth mechanism, orthogonal to both
    biases above: per-family elitism (a protected slot plus guaranteed
    refinement offspring, mirroring the three hand-built templates' own
    ``best_template_node``/``best_fold_template_node``/``best_bool_
    template_node`` machinery exactly) keyed by an *automatically-derived*
    family signature (:func:`root_shape` - the root node's type and
    operator, no hand-declared vocabulary) instead of a template's own
    named hole-choices. Built to fix a diagnosed shared root cause behind
    both bias mechanisms' failures on some targets (see ``docs/ROADMAP.md``
    v0.29/v0.30): once blind growth converges to a wrong structure fast,
    ``exp(-energy)`` reinforcement amplifies *that* structure's own choices/
    subtrees more than a rarer, correct one's - the same premature-lock-in
    problem the three templates avoid by protecting every family's best
    individual from ever being starved out of reproduction, regardless of
    which family currently scores better. Default ``False``: a true no-op
    for every existing caller, measured independently and in combination
    with the two bias mechanisms rather than assumed to help (see
    ``docs/ROADMAP.md`` v0.30).

    Returns ``(best_node, beta_trace, verified)``. ``verified`` re-runs
    :func:`program_energy` on the winner one more time, post-hoc - never
    trust the search's own bookkeeping without re-checking.
    """
    rng = np.random.default_rng() if rng is None else rng
    bias: ResonantBias | None = None
    if allow_recursion and allow_recursion_template and resonant_bias:
        bias_codebook = Codebook(dim=512, seed=int(rng.integers(0, 2**31 - 1)))
        bias = ResonantBias(bias_codebook, TEMPLATE_CATEGORIES)
    fold_bias: ResonantBias | None = None
    if list_inputs and allow_fold_template and resonant_bias:
        fold_bias_codebook = Codebook(dim=512, seed=int(rng.integers(0, 2**31 - 1)))
        fold_bias = ResonantBias(fold_bias_codebook, FOLD_TEMPLATE_CATEGORIES)
    bool_bias: ResonantBias | None = None
    if allow_bool_template and resonant_bias:
        bool_bias_codebook = Codebook(dim=512, seed=int(rng.integers(0, 2**31 - 1)))
        bool_bias = ResonantBias(bool_bias_codebook, BOOL_TEMPLATE_CATEGORIES)
    motif_archive: MotifArchive | None = None
    if allow_motif_bias and resonant_bias:
        motif_codebook = Codebook(dim=512, seed=int(rng.integers(0, 2**31 - 1)))
        motif_archive = MotifArchive(ResonantBias(motif_codebook, {}))
    # Unlike the other four biases, grammar_bias may be caller-supplied (to
    # persist across calls) - grammar_bias_owned tracks whether *this call*
    # constructed it, since only a self-constructed one should ever be
    # reset on restart or discarded; a caller-supplied one is caller-owned
    # and must survive both.
    grammar_bias_owned = allow_grammar_bias and grammar_bias is None
    if grammar_bias_owned:
        gb_codebook = Codebook(dim=512, seed=int(rng.integers(0, 2**31 - 1)))
        grammar_bias = GrammarBias(ResonantBias(gb_codebook, {}), GRAMMAR_CHOICE_VOCAB)

    delta_p1 = delta_p1_start
    stagnation = 0
    restart_threshold = max(1, int(max_generations * _RESTART_STAGNATION_FRACTION))
    population = [
        random_program(
            inputs, rng, max_depth, list_inputs, template_rate, allow_recursion, bias, delta_p1, fold_bias,
            allow_bool_template, bool_bias, allow_fold_template, allow_motif_bias, motif_archive, allow_recursion_template,
            allow_grammar_bias, grammar_bias,
        )
        for _ in range(population_size)
    ]

    best_node: Node = population[0]
    best_energy = float("inf")
    attempt_best_energy = float("inf")
    best_template_node: dict[tuple[str, str], Node] = {}
    best_template_energy: dict[tuple[str, str], float] = {}
    best_fold_template_node: dict[tuple[str, str], Node] = {}
    best_fold_template_energy: dict[tuple[str, str], float] = {}
    best_bool_template_node: dict[tuple[str, ...], Node] = {}
    best_bool_template_energy: dict[tuple[str, ...], float] = {}
    best_shape_node: dict[tuple[str, object], Node] = {}
    best_shape_energy: dict[tuple[str, object], float] = {}
    beta = beta_start
    beta_trace: list[float] = []

    for _generation in range(max_generations):
        raw_energies = np.array([program_energy(p, examples, fuel_budget) for p in population])
        idx_best = int(np.argmin(raw_energies))
        # Occam tie-break on the *reported* winner only. `idx_best` itself is
        # left as plain argmin because it also drives the elitism slot below;
        # choosing a different tied individual there would perturb the search
        # dynamics rather than isolate this effect (measured: doing so just
        # shuffled which seeds fail, 19/64 -> 18/64, i.e. noise).
        min_energy = float(np.min(raw_energies))
        tied = np.flatnonzero(raw_energies <= min_energy + 1e-9)
        idx_report = int(min(tied, key=lambda i: count_nodes(population[i])))
        # Among programs tied at the best energy, prefer the smallest, and let
        # a later equal-energy-but-smaller candidate replace the recorded best
        # (strictly-better-energy alone freezes the first winner forever).
        # Targets the audit's most common failure shape: a program matching
        # every training example that still diverges on held-out input - the
        # coincidental solutions are large nested trees, the genuine ones are
        # small (docs/ROADMAP.md v0.22).
        if raw_energies[idx_report] < best_energy - 1e-9 or (
            abs(raw_energies[idx_report] - best_energy) <= 1e-9
            and count_nodes(population[idx_report]) < count_nodes(best_node)
        ):
            best_energy = float(raw_energies[idx_report])
            best_node = population[idx_report]
        if raw_energies[idx_best] < attempt_best_energy - 1e-9:
            attempt_best_energy = float(raw_energies[idx_best])
            stagnation = 0
        else:
            stagnation += 1
        delta_p1 = min(delta_p1_max, delta_p1_start * (delta_p1_stagnation_growth**stagnation))
        beta_trace.append(beta)
        if best_energy <= 1e-9:
            break

        # Random restart: five different in-population diversity mechanisms
        # were tried and confirmed (by instrumenting real runs, not just
        # reasoning about it) to be unable to rescue a population that has
        # genuinely converged on a fitness plateau below the target - the
        # documented seed-8 regression (docs/ROADMAP.md v0.20). Reweighting
        # selection (a lower/adaptive hole-mutation rate, fitness sharing
        # over structural niches) can't touch it once beta has collapsed
        # softmax to exact 0/1 for the losing side; unprotected fresh
        # individuals (random immigrants, biased or not) can't survive
        # selection long enough to matter; and even faithfully reproducing
        # the one historically-confirmed working combination (hole-mutation
        # forced to full regrowth *and* elitism dropped) scoped per-family
        # still left every family frozen at the same energy for the rest of
        # a 150-generation budget. All five of those try to fix the *current*
        # population from within. This instead treats "no improvement in this
        # attempt for a long time" as a signal that this population's
        # lineage is not worth rescuing at all, and throws the whole thing
        # away for a genuinely independent one - not another perturbation of
        # the same stuck gene pool, its structural-family records, and its
        # (by now probably corrupted-toward-the-plateau) ResonantBias prior.
        # This is intentionally indifferent to *why* a population is stuck:
        # unlike the five mechanisms above, it doesn't depend on the failure
        # being selection-side, population-side, or anything else in
        # particular, so it should generalize to failure modes this project
        # hasn't seen yet, not just seed 8's specific one. `stagnation` only
        # crosses this threshold after an attempt has made *zero* improvement
        # for _RESTART_STAGNATION_FRACTION of the *entire* budget - measured
        # directly against seed 1 (see that constant's comment): a
        # genuinely-progressing search can still have long stagnation
        # stretches as a normal part of succeeding, so this only fires once
        # most of the available runway is already spent going nowhere.
        # Consumes generations from the same overall `max_generations` budget
        # rather than adding to it - a stuck run previously burned its whole
        # remaining budget doing nothing after converging early; now it
        # spends that same budget on independent attempts instead of one
        # frozen one, which can only help. `best_energy`/`best_node` (the
        # function's eventual answer) are deliberately *not* reset here -
        # they track the best ever seen across every attempt, not just the
        # current one.
        if stagnation >= restart_threshold:
            if allow_recursion and allow_recursion_template and resonant_bias:
                bias_codebook = Codebook(dim=512, seed=int(rng.integers(0, 2**31 - 1)))
                bias = ResonantBias(bias_codebook, TEMPLATE_CATEGORIES)
            if list_inputs and allow_fold_template and resonant_bias:
                fold_bias_codebook = Codebook(dim=512, seed=int(rng.integers(0, 2**31 - 1)))
                fold_bias = ResonantBias(fold_bias_codebook, FOLD_TEMPLATE_CATEGORIES)
            if allow_bool_template and resonant_bias:
                bool_bias_codebook = Codebook(dim=512, seed=int(rng.integers(0, 2**31 - 1)))
                bool_bias = ResonantBias(bool_bias_codebook, BOOL_TEMPLATE_CATEGORIES)
            if allow_motif_bias and resonant_bias:
                motif_codebook = Codebook(dim=512, seed=int(rng.integers(0, 2**31 - 1)))
                motif_archive = MotifArchive(ResonantBias(motif_codebook, {}))
            if grammar_bias_owned:
                gb_codebook = Codebook(dim=512, seed=int(rng.integers(0, 2**31 - 1)))
                grammar_bias = GrammarBias(ResonantBias(gb_codebook, {}), GRAMMAR_CHOICE_VOCAB)
            delta_p1 = delta_p1_start
            population = [
                random_program(
                    inputs, rng, max_depth, list_inputs, template_rate, allow_recursion, bias, delta_p1, fold_bias,
                    allow_bool_template, bool_bias, allow_fold_template, allow_motif_bias, motif_archive, allow_recursion_template,
            allow_grammar_bias, grammar_bias,
                )
                for _ in range(population_size)
            ]
            best_template_node = {}
            best_template_energy = {}
            best_fold_template_node = {}
            best_fold_template_energy = {}
            best_bool_template_node = {}
            best_bool_template_energy = {}
            best_shape_node = {}
            best_shape_energy = {}
            attempt_best_energy = float("inf")
            stagnation = 0
            # beta is deliberately left untouched, not reset to beta_start:
            # selection pressure rising monotonically across the *whole* call
            # is an existing invariant elsewhere in this module
            # (test_selection_pressure_rises_across_generations) - resetting
            # it here would mean a restart generation's beta_trace entry
            # drops below the previous one, breaking that invariant for any
            # caller with a small enough budget for restarts to kick in
            # (confirmed by that exact test failing when this reset raw
            # beta_start). A fresh population still gets fresh
            # crossover/mutation diversity regardless of what beta currently
            # is; it doesn't need a fresh annealing schedule too.
            continue

        if bias is not None:
            for p, e in zip(population, raw_energies):
                choices = extract_template_choices(p)
                if choices is not None:
                    bias.reinforce(choices, weight=float(np.exp(-e)))
        if fold_bias is not None:
            for p, e in zip(population, raw_energies):
                fold_choices = extract_fold_template_choices(p)
                if fold_choices is not None:
                    fold_bias.reinforce(fold_choices, weight=float(np.exp(-e)))
        if bool_bias is not None:
            for p, e in zip(population, raw_energies):
                bool_choices = extract_bool_template_choices(p)
                if bool_choices is None:
                    continue
                weight = float(np.exp(-e))
                # shape_kind is not itself resonance-biased (see
                # BOOL_TEMPLATE_CATEGORIES's comment) but is still passed
                # through, same as combine_kind/base_kind/combine_op/
                # item_kind already are for the other two templates -
                # ResonantBias.reinforce only ever accumulates categories
                # .sample() is later called for, so this is a harmless,
                # already-established dead write, not new behavior.
                bool_bias.reinforce({"shape_kind": bool_choices["shape_kind"]}, weight=weight)
                # modulus/cmp/const are reinforced once *per atom*, not once
                # per individual, since 2-3 atoms share these category names
                # (they aren't indexed by atom position - positions are
                # interchangeable under AND/OR). This gives them 2-3x the
                # reinforcement weight shape_kind gets, per individual - a
                # real imbalance the other two templates don't have (each of
                # their tunable holes is a true singleton per individual).
                # Built this simple way first and validated by the leap-year
                # reliability sweep (docs/ROADMAP.md) rather than guessing a
                # correction was needed; the documented fallback if it ever
                # causes measured unreliability is reinforcing only one
                # randomly-chosen atom's triple per individual instead.
                for atom in bool_choices["atoms"]:
                    bool_bias.reinforce(atom, weight=weight)
        if motif_archive is not None:
            motif_archive.reinforce_from_population(population, raw_energies)
        if allow_grammar_bias and grammar_bias is not None:
            grammar_bias.reinforce_from_population(population, raw_energies)

        # Template elitism: track the best template-shaped individual *per
        # structural family* (combine_kind, base_kind), not one single best
        # overall. Found empirically while diagnosing why Fibonacci is
        # unreliable on some seeds (see docs/ROADMAP.md v0.20): a single
        # global "best template" slot reproduces exactly the premature-
        # family-lock-in failure mode this module's own docstrings already
        # warn ResonantBias away from (see _recursive_template) - once
        # param_recur (structurally incapable of ever matching a two-term
        # recurrence like Fibonacci, confirmed by exhaustively sweeping every
        # hole-filler value for a stuck param_recur candidate and finding
        # none reduce its energy) reaches a locally-competitive energy, it
        # permanently starves double_recur out of ever getting its own
        # protected slot, since a fresh double_recur draw usually starts
        # worse before refinement, and "best template" only ever updates on
        # strict improvement. Keying elitism by family gives every family a
        # fair, ongoing chance regardless of which one got lucky first.
        if allow_recursion:
            for p, e in zip(population, raw_energies):
                choices = extract_template_choices(p)
                if choices is None:
                    continue
                key = (choices["combine_kind"], choices["base_kind"])
                if e < best_template_energy.get(key, float("inf")) - 1e-9:
                    best_template_energy[key] = float(e)
                    best_template_node[key] = p

        # Fold-template elitism, mirroring the recursion elitism above
        # exactly (same "one family locking out a sibling family" failure
        # mode is possible here too - a fold that happens to hit a
        # zero-mismatch coincidence on "*"/"square" should not be able to
        # starve "+"/"identity" out of its own protected slot).
        if list_inputs:
            for p, e in zip(population, raw_energies):
                fold_choices = extract_fold_template_choices(p)
                if fold_choices is None:
                    continue
                fold_key = (fold_choices["combine_op"], fold_choices["item_kind"])
                if e < best_fold_template_energy.get(fold_key, float("inf")) - 1e-9:
                    best_fold_template_energy[fold_key] = float(e)
                    best_fold_template_node[fold_key] = p

        # Bool-template elitism, mirroring the recursion/fold elitism above -
        # same "one family locking out a sibling family" failure mode is
        # possible here too, e.g. and2 hitting a lucky coincidence starving
        # and_or3 out of its own protected slot before it gets a fair chance.
        # Keyed as a 1-tuple (only one structural axis, shape_kind, exists
        # here, unlike fold's two) purely so this dict has the same
        # dict[tuple[str, ...], Node] shape as the other two.
        if allow_bool_template:
            for p, e in zip(population, raw_energies):
                bool_choices = extract_bool_template_choices(p)
                if bool_choices is None:
                    continue
                bool_key = (bool_choices["shape_kind"],)
                if e < best_bool_template_energy.get(bool_key, float("inf")) - 1e-9:
                    best_bool_template_energy[bool_key] = float(e)
                    best_bool_template_node[bool_key] = p

        # Shape elitism: the same per-family protection as the three blocks
        # above, but keyed by an automatically-derived signature
        # (root_shape) instead of a template's own hand-declared hole-
        # choices - applies to *any* individual, not just template-shaped
        # ones, since the failure mode it targets (a numerically-dominant
        # structure starving a rarer, correct one out of reproduction
        # before either bias mechanism's reinforcement can act on it) isn't
        # specific to any one template (see docs/ROADMAP.md v0.30).
        if allow_shape_elitism:
            for p, e in zip(population, raw_energies):
                shape_key = root_shape(p)
                if e < best_shape_energy.get(shape_key, float("inf")) - 1e-9:
                    best_shape_energy[shape_key] = float(e)
                    best_shape_node[shape_key] = p

        sizes = np.array([count_nodes(p) for p in population], dtype=float)
        selection_energies = raw_energies + parsimony * sizes
        probs = softmax(-beta * selection_energies)
        next_population = [population[idx_best]]  # elitism (by raw energy, not parsimony-adjusted)
        for template_node in best_template_node.values():
            if template_node is population[idx_best] or len(next_population) >= population_size:
                continue
            next_population.append(template_node)
            # Elitism alone only *protects* a family's best template from
            # deletion - it does nothing to *refine* it, since fitness-
            # proportionate selection (below) gives a mediocre-energy
            # template near-zero probability of ever being chosen as a
            # parent once a lower-energy competitor (in or out of its
            # family) exists - a real, measured failure mode: a template
            # stuck at energy 8 never improved over 150 generations of
            # elitism-only protection (see docs/ROADMAP.md v0.20).
            # Guarantee it a handful of offspring every generation instead,
            # independent of selection pressure - the actual mechanism that
            # lets its hole-fillers get repeatedly refined toward the correct
            # combination. Routed through mutate() (not _mutate_template_hole
            # directly) so these offspring get the same
            # _TEMPLATE_HOLE_MUTATION_RATE chance of a full scoped regrowth
            # as ordinary mutation: calling _mutate_template_hole directly
            # here was tried first and measured to reproduce the exact
            # stuck-at-a-plateau failure this elitism was meant to fix - a
            # frozen anchor only ever hill-climbing one hole at a time from
            # itself, forever, can't escape a Hamming-distance-1 local
            # optimum (confirmed by instrumenting a run on 2**n, seed 4: a
            # family reached energy 1.0 at generation 0 and was still exactly
            # 1.0 at generation 59 with this loop as the only refinement
            # path). Mixing in occasional full redraws gives the guaranteed
            # offspring the same chance to jump past that plateau that
            # ordinary population members get.
            for _ in range(min(_TEMPLATE_REFINEMENT_OFFSPRING, population_size - len(next_population))):
                refined = mutate(
                    template_node, rng, inputs, max_depth, list_inputs, template_rate, allow_recursion, bias,
                    delta_p1, fold_bias, allow_bool_template, bool_bias,
                    allow_fold_template, allow_motif_bias, motif_archive, allow_recursion_template,
                    allow_grammar_bias, grammar_bias,
                )
                next_population.append(refined)
        for fold_template_node in best_fold_template_node.values():
            if fold_template_node is population[idx_best] or len(next_population) >= population_size:
                continue
            next_population.append(fold_template_node)
            # Same guaranteed-offspring reasoning as the recursion elitism
            # above: a protected slot alone doesn't refine anything, since
            # fitness-proportionate selection gives a mediocre fold template
            # near-zero probability of ever being chosen as a parent once a
            # lower-energy competitor exists.
            for _ in range(min(_TEMPLATE_REFINEMENT_OFFSPRING, population_size - len(next_population))):
                fold_refined = mutate(
                    fold_template_node, rng, inputs, max_depth, list_inputs, template_rate, allow_recursion, bias,
                    delta_p1, fold_bias, allow_bool_template, bool_bias,
                    allow_fold_template, allow_motif_bias, motif_archive, allow_recursion_template,
                    allow_grammar_bias, grammar_bias,
                )
                next_population.append(fold_refined)
        for bool_template_node in best_bool_template_node.values():
            if bool_template_node is population[idx_best] or len(next_population) >= population_size:
                continue
            next_population.append(bool_template_node)
            # Same guaranteed-offspring reasoning as the recursion/fold
            # elitism above - arguably more relevant here, since a 3-atom
            # shape needs all three atoms simultaneously correct, exactly
            # the kind of multi-field local optimum a frozen, refinement-free
            # anchor could never escape on its own.
            for _ in range(min(_TEMPLATE_REFINEMENT_OFFSPRING, population_size - len(next_population))):
                bool_refined = mutate(
                    bool_template_node, rng, inputs, max_depth, list_inputs, template_rate, allow_recursion, bias,
                    delta_p1, fold_bias, allow_bool_template, bool_bias,
                    allow_fold_template, allow_motif_bias, motif_archive, allow_recursion_template,
                    allow_grammar_bias, grammar_bias,
                )
                next_population.append(bool_refined)
        for shape_node in best_shape_node.values():
            if shape_node is population[idx_best] or len(next_population) >= population_size:
                continue
            next_population.append(shape_node)
            # Same guaranteed-offspring reasoning as the three template
            # elitism blocks above: a protected slot alone doesn't refine
            # anything, since fitness-proportionate selection gives a
            # mediocre-but-rare-shape individual near-zero probability of
            # ever being chosen as a parent once a numerically-dominant
            # competitor exists - the exact failure mode this mechanism was
            # built to fix (see docs/ROADMAP.md v0.30).
            for _ in range(min(_TEMPLATE_REFINEMENT_OFFSPRING, population_size - len(next_population))):
                shape_refined = mutate(
                    shape_node, rng, inputs, max_depth, list_inputs, template_rate, allow_recursion, bias,
                    delta_p1, fold_bias, allow_bool_template, bool_bias,
                    allow_fold_template, allow_motif_bias, motif_archive, allow_recursion_template,
                    allow_grammar_bias, grammar_bias,
                )
                next_population.append(shape_refined)
        while len(next_population) < population_size:
            i, j = rng.choice(len(population), size=2, p=np.asarray(probs))
            if rng.random() < 0.5:
                child = crossover(population[i], population[j], rng)
            else:
                child = mutate(
                    population[i], rng, inputs, max_depth, list_inputs, template_rate, allow_recursion, bias,
                    delta_p1, fold_bias, allow_bool_template, bool_bias,
                    allow_fold_template, allow_motif_bias, motif_archive, allow_recursion_template,
                    allow_grammar_bias, grammar_bias,
                )
            next_population.append(child)
        population = next_population
        beta = min(beta * beta_growth, beta_max)

    verified = program_energy(best_node, examples, fuel_budget) <= 1e-9
    return best_node, beta_trace, verified
