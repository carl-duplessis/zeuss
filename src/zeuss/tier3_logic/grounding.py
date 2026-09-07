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

``anneal_theory`` (v0.5's "probabilistic priors as temperature schedules")
carries ``compile_theory``'s ``inverse_temperature`` across a cooling
schedule: at low temperature the Boltzmann weighting over Boolean corners is
a genuine probabilistic *prior* spread across every near-satisfying
valuation, sharpening toward only the lowest-energy corner(s) as temperature
drops - the same entropy-driven crystallisation
:func:`zeuss.tier2_substrate.collapse.anneal` already does for atomic
symbols, applied here to whole logical valuations.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from ..tier2_substrate.collapse import entropy, softmax
from ..tier2_substrate.energy import Landscape, settle
from ..tier2_substrate.hypervectors import Codebook, bind, bundle, normalize, random_hypervector, similarity
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

    The ``FALSE`` pole is the exact phase-antipode of ``TRUE`` (``-true_pole``),
    not an independently drawn codebook symbol - genuinely opposite poles on
    the phase torus, matching this module's own framing, rather than two
    merely-distinct (quasi-orthogonal, ``similarity ~ 0``) random vectors.
    This matters for :func:`readout`'s calibration: with independent poles,
    even a single crisply-true variable's achievable similarity margin tops
    out around ``1`` (``true_sim=1``, ``false_sim~0``) instead of the ``2``
    its ``[-2, 2] -> [0, 1]`` remap assumes, so a "certain" readout could
    never exceed ``0.75``. Antipodal poles make ``false_sim = -true_sim``
    exactly, so the full range is genuinely achievable in the well-separated
    (few-variable) limit - confirmed directly
    (`test_readout_reaches_near_certainty_for_an_isolated_crisp_variable`).
    """
    parts = []
    weights = []
    for name in variables:
        v = float(min(1.0, max(0.0, valuation.get(name, 0.0))))
        role = codebook.symbol(f"VAR:{name}")
        true_pole = bind(role, codebook.symbol(f"{name}:TRUE"))
        false_pole = -true_pole
        parts.append(true_pole)
        parts.append(false_pole)
        weights.append(v)
        weights.append(1.0 - v)
    return bundle(parts, weights=weights)


def _boolean_corners(variables: list[str]):
    for bits in itertools.product((0.0, 1.0), repeat=len(variables)):
        yield dict(zip(variables, bits))


def compile_theory(
    codebook: Codebook,
    theory: Theory,
    variables: list[str],
    fixed: dict[str, float] | None = None,
    inverse_temperature: float = 1.0,
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

    ``inverse_temperature`` scales the Boltzmann weighting
    (``exp(-inverse_temperature * energy)``): the default ``1.0`` is the
    original fixed weighting; lower values flatten the prior across every
    near-satisfying corner (more genuine uncertainty about which valuation
    holds), higher values sharpen it toward only the lowest-energy corner(s)
    - see :func:`anneal_theory` for cooling this across a schedule.
    """
    landscape = Landscape()
    for corner in _boolean_corners(variables):
        valuation = {**corner, **(fixed or {})}
        corner_energy = theory.energy(valuation)
        weight = math.exp(-inverse_temperature * corner_energy)
        if weight > WEIGHT_FLOOR:
            vector = valuation_to_hypervector(codebook, variables, valuation)
            landscape.add(vector, weight=weight)
    return landscape


def compile_theories(
    codebook: Codebook,
    theories: dict[str, Theory],
    shared_vars: dict[str, list[str]],
    inverse_temperature: float = 1.0,
) -> Landscape:
    """Audit several agents' theories for agreement, then compile the union.

    Runs :func:`zeuss.tier3_logic.sheaf.from_theories` first - the same check
    ``zeuss audit`` demos - and raises :class:`InconsistentTheoriesError` if
    any two agents' independently-derived best local valuations disagree on a
    shared variable. On success, merges every agent's rules into one
    ``Theory`` over the union of every variable referenced (any rule using
    the ``TRUE_ANCHOR`` sentinel - see ``sheaf.py`` - keeps it pinned to 1.0
    rather than treated as a free variable) and compiles that via
    :func:`compile_theory`. ``inverse_temperature`` is forwarded unchanged.
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
    return compile_theory(
        codebook, combined, free_variables, fixed={TRUE_ANCHOR: 1.0}, inverse_temperature=inverse_temperature
    )


def readout(codebook: Codebook, variables: list[str], z) -> dict[str, float]:
    """Invert :func:`valuation_to_hypervector`: recover each variable's truth value.

    For each variable, the similarity margin between its ``TRUE`` and
    ``FALSE`` poles lies in ``[-2, 2]`` (each pole similarity is itself in
    ``[-1, 1]``); remapped linearly onto ``[0, 1]``. Since ``FALSE`` is now
    ``TRUE``'s exact phase-antipode (see :func:`valuation_to_hypervector`),
    ``false_sim`` is always exactly ``-true_sim`` by construction, so this is
    genuinely a `(true_sim + 1) / 2` rescaling, not merely an upper bound.
    """
    out = {}
    for name in variables:
        role = codebook.symbol(f"VAR:{name}")
        true_pole = bind(role, codebook.symbol(f"{name}:TRUE"))
        true_sim = similarity(z, true_pole)
        false_sim = similarity(z, -true_pole)
        margin = true_sim - false_sim
        out[name] = min(1.0, max(0.0, (margin + 2.0) / 4.0))
    return out


def anneal_theory(
    codebook: Codebook,
    theory: Theory,
    variables: list[str],
    z0=None,
    schedule: tuple[float, ...] = (0.25, 0.5, 1, 2, 4, 8, 16, 32),
    fixed: dict[str, float] | None = None,
    settle_steps: int = 30,
    settle_step_size: float = 0.3,
    rng: np.random.Generator | None = None,
):
    """Anneal a probabilistic prior over the theory's satisfying valuations
    toward a crisp decision - the v0.5 roadmap item ("probabilistic priors
    as temperature schedules"), Frontier 2's analogue of
    :func:`zeuss.tier2_substrate.collapse.anneal`.

    At each ``beta`` in ``schedule``, recompiles the theory's Boolean corners
    into a fresh Boltzmann-weighted :class:`Landscape` at that temperature
    (:func:`compile_theory`'s ``inverse_temperature``) and settles the
    running state ``z`` into it via
    :func:`zeuss.tier2_substrate.energy.settle` - the *same* ``beta`` drives
    both how many corners stay live as a prior and how sharply settling
    pulls toward them, one temperature knob for both, per this project's
    "determinism and probability are one representation" framing. ``z``
    carries forward between schedule steps, same as :func:`~zeuss.
    tier2_substrate.collapse.anneal`'s fixed probe.

    Returns ``(z_final, trace)``: ``trace`` is a list of per-step ``{beta,
    entropy_bits, energy, readout}`` dicts. ``entropy_bits`` is the settled
    state's occupancy entropy over the *compiled corner attractors*
    themselves (:func:`zeuss.tier2_substrate.collapse.softmax`/``entropy``,
    the same readout :func:`zeuss.tier2_substrate.collapse.collapse` uses for
    atomic symbols).

    Checked directly, not assumed: on a theory with a *unique* lowest-energy
    corner, entropy crystallises to ~0 and the readout converges to that
    corner (`test_anneal_theory_crystallises_toward_unique_ground_state`).
    Perhaps counter-intuitively, entropy also crystallises to ~0 on a
    genuinely *underdetermined* theory (several equally-satisfying corners) -
    a single settling trajectory is one continuous state, and it can only
    occupy one point at a time, so it spontaneously breaks the symmetry and
    commits to *one* of the tied corners rather than hovering between them
    (the same phenomenon a ferromagnet's mean-field descent shows, picking
    one degenerate ground state, not a superposition - see
    :mod:`zeuss.tier2_substrate.energy`'s own framing). What stays true and
    is verified is that the corner it commits to is a genuinely satisfying
    one, not an arbitrary point
    (`test_anneal_theory_settles_into_a_genuinely_satisfying_corner_even_when_degenerate`).
    The actual site of "spread across multiple corners" as a real prior is
    the *compiled* :class:`Landscape` at low beta, before any single
    trajectory has committed - see :func:`compile_theory`'s
    ``inverse_temperature``
    (`test_compile_theory_inverse_temperature_narrows_the_live_corner_set`).
    """
    rng = np.random.default_rng() if rng is None else rng
    z = normalize(z0) if z0 is not None else random_hypervector(codebook.dim, rng)
    trace = []
    for beta in schedule:
        landscape = compile_theory(codebook, theory, variables, fixed=fixed, inverse_temperature=beta)
        z, energies = settle(
            landscape, z, steps=settle_steps, step_size=settle_step_size, inverse_temperature=beta, rng=rng
        )
        sims = [similarity(z, a) for a in landscape.attractors]
        probs = softmax([beta * s for s in sims])
        trace.append(
            {
                "beta": float(beta),
                "entropy_bits": entropy(probs),
                "energy": float(energies[-1]),
                "readout": readout(codebook, variables, z),
            }
        )
    return z, trace
