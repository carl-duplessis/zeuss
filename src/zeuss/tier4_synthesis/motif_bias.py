"""Motif resonance: a general alternative to hand-built structural templates
(``_recursive_template``, ``_fold_template``, ``_bool_template`` in
``search.py``). Each of those three needed a human to notice a new target's
failure shape and hand-design a whole skeleton with named holes before the
search could reliably find it - not scalable to every future problem shape,
and exactly the kind of special-cased symbolic path ``CLAUDE.md`` says this
project should avoid ("don't store rules as if/else tables when they can be
attractors... that's the whole point of Zeuss").

This module instead reinforces and reuses *actual subtrees* pulled from the
population's own low-energy individuals, addressed by *where in a tree* they
were found rather than by what a human anticipated they'd look like - the
resonance-flavoured analogue of Probabilistic Incremental Program Evolution
(Salustowicz & Schmidhuber 1997) / module-acquisition GP (Koza 1994; Angeline
& Pollack), built directly on :class:`~.resonance_bias.ResonantBias` rather
than importing either technique wholesale. ``ResonantBias`` is already fully
generic - parameterized by an arbitrary ``categories: dict[str, list]``,
silently skipping unknown reinforcement keys, re-reading ``categories[cat]``
fresh on every ``sample()`` call - so the missing piece was never a new EDA,
just a mechanism that discovers *what the categories and options should be*
from the population itself.

A "motif" is any subtree observed at a given structural context - the
``(parent_kind, child_slot, depth)`` position it was grown at (see
:func:`context_key`). Deliberately does not replace any of the three
existing templates: it is a fourth, independent, opt-in mechanism
(``allow_motif_bias`` in ``search.py``, default ``False``, a true no-op
otherwise), measured against the same targets that motivated each hand-built
template rather than assumed to match their reliability (see
``docs/ROADMAP.md`` v0.28).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dsl import Node
from .dsl import children as _children
from .dsl import pretty
from .resonance_bias import ResonantBias

# Depth is bucketed (not used raw) to keep the context space bounded - a
# motif observed at depth 11 and one at depth 12 are addressing the same
# "near the leaves" role, not meaningfully different positions.
_MAX_DEPTH_BUCKET = 4

# How many distinct subtrees a single context is allowed to accumulate
# before the lowest-weight one is evicted to make room for a new one - an
# unbounded archive over 150 generations x hundreds of individuals would
# otherwise grow without limit.
_MAX_MOTIFS_PER_CONTEXT = 20

# How many levels below a collected individual's root to pull motifs from.
# Bounded for the same reason fuel/depth are bounded everywhere else in this
# module: cheap per individual, and a motif near the root of one individual
# is still meaningful context for another individual's shallow position.
_MAX_COLLECTION_DEPTH = 3


def context_key(parent_kind: str, child_slot: int, depth: int) -> str:
    """A single string key for a structural growth site -
    :class:`ResonantBias`/``Codebook`` both key by string, so the
    ``(parent_kind, child_slot, depth_bucket)`` context is joined into one
    string rather than adding tuple-key support to either."""
    return f"{parent_kind}:{child_slot}:{min(depth, _MAX_DEPTH_BUCKET)}"


def collect_motifs(node: Node, parent_kind: str = "root", child_slot: int = 0, depth: int = 0):
    """Yield ``(context, subtree)`` for ``node`` and every descendant down to
    :data:`_MAX_COLLECTION_DEPTH`, using the same ``(parent_kind, child_slot,
    depth)`` convention :func:`.search._grow` threads through its own
    recursion, so a motif registered here is addressable at exactly the
    grow-site that would later consider reusing it. ``parent_kind`` is the
    DSL class name of the node it sits under (``"root"`` for the individual's
    own top-level node), keeping this in sync with ``dsl.py``'s node types
    automatically rather than via a separately-maintained name table."""
    yield context_key(parent_kind, child_slot, depth), node
    if depth >= _MAX_COLLECTION_DEPTH:
        return
    kind = type(node).__name__
    for slot, child in enumerate(_children(node)):
        yield from collect_motifs(child, kind, slot, depth + 1)


@dataclass
class MotifArchive:
    """Wraps one :class:`ResonantBias` whose ``categories`` dict is *not*
    fixed at construction: ``categories[context]`` starts absent and grows as
    distinct subtrees are observed, which ``ResonantBias`` already supports
    without modification. Tracks its own lightweight per-option weight totals
    (separate from ``ResonantBias``'s own internal accumulator, which is one
    running hypervector *per category*, not per option) purely to decide
    which motif to evict once a context's option list hits
    :data:`_MAX_MOTIFS_PER_CONTEXT`.
    """

    bias: ResonantBias
    max_motifs_per_context: int = _MAX_MOTIFS_PER_CONTEXT
    _nodes: dict[str, dict[str, Node]] = field(default_factory=dict)
    _weights: dict[str, dict[str, float]] = field(default_factory=dict)

    def has(self, ctx: str) -> bool:
        return bool(self.bias.categories.get(ctx))

    def sample(self, ctx: str, rng: np.random.Generator) -> Node | None:
        """A copy of an archived subtree at ``ctx``, chosen via the wrapped
        ``ResonantBias``'s resonance-weighted sampling - ``None`` if ``ctx``
        has no archived motifs yet (the caller's cue to grow fresh instead,
        mirroring every template's own ``(None, None)`` "nothing to offer"
        return)."""
        if not self.has(ctx):
            return None
        key = self.bias.sample(ctx, rng)
        return self._nodes[ctx][key]

    def register(self, ctx: str, subtree: Node, weight: float) -> None:
        """Record one observation of ``subtree`` at ``ctx``, weighted by
        ``weight`` (``exp(-energy)`` of the individual it came from - the
        same convention every existing template's reinforcement already
        uses). A genuinely new subtree for this context is added, evicting
        the current lowest-weight entry first if the context is already at
        capacity (and only if the new observation actually outweighs it -
        otherwise this is a no-op, not a wasted eviction)."""
        if weight <= 0:
            return
        key = pretty(subtree)
        nodes = self._nodes.setdefault(ctx, {})
        weights = self._weights.setdefault(ctx, {})
        options = self.bias.categories.setdefault(ctx, [])
        if key not in nodes:
            if len(options) >= self.max_motifs_per_context:
                worst = min(options, key=lambda k: weights.get(k, 0.0))
                if weight <= weights.get(worst, 0.0):
                    return
                options.remove(worst)
                nodes.pop(worst, None)
                weights.pop(worst, None)
            options.append(key)
            nodes[key] = subtree
        weights[key] = weights.get(key, 0.0) + weight
        self.bias.reinforce({ctx: key}, weight=weight)

    def reinforce_from_population(self, population: list[Node], energies: np.ndarray) -> None:
        """For every individual, walk its tree (see :func:`collect_motifs`)
        and register every subtree found, weighted by that individual's own
        ``exp(-energy)`` - exactly the reinforcement convention every
        existing template's bias already uses, just applied to whole
        subtrees keyed by structural position instead of named hole-fillers
        keyed by a hand-declared category."""
        for p, e in zip(population, energies):
            weight = float(np.exp(-e))
            if weight <= 0:
                continue
            for ctx, subtree in collect_motifs(p):
                self.register(ctx, subtree, weight)
