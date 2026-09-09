"""Production-level PCFG resonance: a second, materially different general
alternative to hand-built structural templates (`_recursive_template`,
`_fold_template`, `_bool_template` in `search.py`), complementing v0.28's
`motif_bias.py` rather than replacing it.

`motif_bias.py` reinforces and reuses *whole subtrees* pulled from the
population's own low-energy individuals - measured (see `docs/ROADMAP.md`
v0.28) to fail on exactly the targets whose difficulty is a compound
top-level shape that essentially never spontaneously assembles under blind
growth in the first place: subtree reuse can only propagate a shape some
individual has *already* fully built by luck, and if that never happens,
there is nothing genuine to ever discover.

This module instead biases every *production choice* `_grow`/`_leaf`
already make - which `BinOp` operator, which `UnaryOp` operator, `Var` vs
`Const`, which constant - independently, conditioned on *where* in the tree
the choice is made (`production_context`). This is the resonance-flavoured
analogue of Probabilistic Incremental Program Evolution (Salustowicz &
Schmidhuber 1997): a compound shape can assemble from marginal pushes on
separate choice-points that never co-occurred in any single ancestor, so it
does not need the whole shape to have already existed once - a materially
different mechanism from `motif_bias.py`'s, not a rehash of it.

Built on the same `ResonantBias` every other bias in this codebase already
uses, reinforced by the same `exp(-energy)` convention. The one addition
here beyond `motif_bias.py`'s pattern: `GrammarBias` can be constructed once,
saved, and reloaded (`save_grammar_bias`/`load_grammar_bias`), so a caller
that wants the bias to persist *across calls to `synthesize`* - not reset
every run, the way every other bias in this module is - can opt into that
explicitly, rather than via a hidden process-global singleton (which would
risk test-order-dependent behavior, a real hazard given how much this
suite's reliability rests on pinned-seed determinism).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..tier2_substrate.hypervectors import Codebook
from .dsl import BinOp, Const, Node, UnaryOp, Var
from .dsl import children as _children
from .resonance_bias import ResonantBias


def production_context(choice_name: str, parent_kind: str, child_slot: int) -> str:
    """A single string key for one production choice-point - deliberately
    *not* depth-bucketed, unlike `motif_bias.context_key`: keying by
    choice-point identity alone (e.g. "which `binop_op` under an `And`'s
    right slot") lets reinforcement accumulate across every depth that
    relative position occurs at, instead of fragmenting into near-empty
    per-depth buckets - and is what lets a compound shape assemble
    gradually from marginal, depth-independent pushes."""
    return f"{choice_name}:{parent_kind}:{child_slot}"


def collect_production_choices(node: Node, parent_kind: str = "root", child_slot: int = 0):
    """Yield ``(context, choice_name, value)`` for every production choice
    `_grow`/`_leaf` could have made while building ``node`` - a `BinOp`'s
    `op`, a `UnaryOp`'s `op`, a `Const`'s value (plus the `"const"`
    `leaf_kind`), and a bare `Var`'s `"var"` `leaf_kind`. Walks through every
    other node kind purely to reach nested choices, using the same
    `(parent_kind, child_slot)` convention `_grow` itself threads through
    its own recursion (added in v0.28 for `motif_bias.collect_motifs`) so a
    choice observed here is addressable at exactly the grow-site that could
    reuse it. Unbounded depth - unlike `collect_motifs`, this only
    categorizes one node at a time (no subtree copies), so walking a whole
    individual is cheap regardless of size."""
    if isinstance(node, BinOp):
        yield production_context("binop_op", parent_kind, child_slot), "binop_op", node.op
        yield from collect_production_choices(node.left, "BinOp", 0)
        yield from collect_production_choices(node.right, "BinOp", 1)
        return
    if isinstance(node, UnaryOp):
        yield production_context("unaryop_op", parent_kind, child_slot), "unaryop_op", node.op
        yield from collect_production_choices(node.operand, "UnaryOp", 0)
        return
    if isinstance(node, Const):
        yield production_context("leaf_kind", parent_kind, child_slot), "leaf_kind", "const"
        yield production_context("leaf_const", parent_kind, child_slot), "leaf_const", node.value
        return
    if isinstance(node, Var):
        # Deliberately no payload beyond "this slot picked a variable" - a
        # variable's *name* is problem-specific and won't recur across
        # targets, so reinforcing which one was picked would actively work
        # against the persistence goal (a name like "year" never transfers).
        yield production_context("leaf_kind", parent_kind, child_slot), "leaf_kind", "var"
        return
    kind = type(node).__name__
    for slot, child in enumerate(_children(node)):
        yield from collect_production_choices(child, kind, slot)


@dataclass
class GrammarBias:
    """Wraps one `ResonantBias` whose categories are production
    choice-points rather than a hand-declared template's named holes.
    ``choice_vocab`` (e.g. ``{"binop_op": list(_BINOPS), ...}``) is supplied
    by the caller (`search.py`, which already owns `_BINOPS`/`_UNARYOPS`/
    `_LEAF_CONSTS`) rather than redeclared here, avoiding both a circular
    import and a second, driftable copy of those tuples."""

    bias: ResonantBias
    choice_vocab: dict[str, list]

    def _register(self, ctx: str, choice_name: str) -> None:
        if ctx not in self.bias.categories:
            self.bias.categories[ctx] = list(self.choice_vocab[choice_name])

    def sample_or(self, choice_name: str, parent_kind: str, child_slot: int, rng: np.random.Generator, fallback: Callable[[], object]):
        """Resonance-sample the value for this choice-point if it has
        evidence; otherwise call ``fallback()`` - today's exact existing
        draw, the same "uniform/static until evidence exists" contract every
        other bias in this codebase already has."""
        ctx = production_context(choice_name, parent_kind, child_slot)
        if ctx in self.bias.categories and self.bias.has_evidence(ctx):
            return self.bias.sample(ctx, rng)
        return fallback()

    def reinforce_from_population(self, population: list[Node], energies: np.ndarray) -> None:
        """For every individual, walk its tree (see
        :func:`collect_production_choices`) and reinforce every production
        choice found, weighted by that individual's own ``exp(-energy)`` -
        the same reinforcement convention every existing bias already
        uses."""
        for p, e in zip(population, energies):
            weight = float(np.exp(-e))
            if weight <= 0:
                continue
            for ctx, choice_name, value in collect_production_choices(p):
                self._register(ctx, choice_name)
                self.bias.reinforce({ctx: value}, weight=weight)


def save_grammar_bias(gb: GrammarBias, path: str) -> None:
    """Persist a `GrammarBias` to ``path`` - the mechanism that lets a bias
    outlive one call to `synthesize`. Uses `pickle`: the payload is nested
    dicts of `str -> complex128 ndarray` (the codebook's minted vectors and
    the bias's accumulators) plus a plain `dict[str, list]` (categories) -
    all pickle-native, with no alignment bugs between parallel name/vector
    arrays the way a hand-rolled npz+json split would risk. This is a
    code-owned artifact this project's own code reads back, not an interop
    format - the usual "don't unpickle untrusted input" caveat is a
    documented note here, not a real constraint, since the file is one the
    caller wrote themselves."""
    payload = {
        "dim": gb.bias.codebook.dim,
        "beta": gb.bias.beta,
        "codebook_items": gb.bias.codebook.items(),
        "categories": gb.bias.categories,
        # `_accum` has no public accessor (only `has_evidence` was added to
        # ResonantBias, per the reviewed plan) - reached into directly here
        # rather than growing ResonantBias's public surface further for a
        # single same-package caller that already collaborates with it
        # tightly (unlike `Codebook`, which needed real new public API since
        # `motif_bias.py`/`search.py`/tests all need to read/write it too).
        "accum": gb.bias._accum,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_grammar_bias(path: str, choice_vocab: dict[str, list]) -> GrammarBias:
    """Load a `GrammarBias` previously written by :func:`save_grammar_bias`.
    Reconstructs a fresh `Codebook` (its `seed` doesn't matter for restored
    names - see `Codebook.load`'s docstring for why a seed alone can't
    reproduce them) and bulk-restores its minted vectors, then rebuilds the
    `ResonantBias` with the saved categories/accumulator state."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    codebook = Codebook(dim=payload["dim"])
    codebook.load(payload["codebook_items"])
    bias = ResonantBias(codebook, payload["categories"], beta=payload["beta"])
    bias._accum = payload["accum"]  # see save_grammar_bias's comment on this
    return GrammarBias(bias, choice_vocab)
