"""Ask-a-question read-out over the substrate (answer + coherence).

Two ways to interrogate a grounded knowledge base:

* :func:`ask` - a single deductive hop. Recover the object of
  ``(subject, relation, ?)`` by unbinding the role waves back out of the one
  memory hypervector and reading which entity the residue resonates with.
* :func:`chain` / :func:`entails` - *multi-hop* deduction. Iterate the very same
  one-hop wave operator (:meth:`Ontology.step`), collapsing the continuous
  residue onto the nearest entity between hops and feeding it back as the next
  subject. Following a relation's transitive closure is therefore a fixed-point
  of one wave operator - deduction as dynamics, not a graph walk or an ``if``
  over the stored facts.

Every result carries a **coherence** in [0, 1] (the amplitude of the recalled
wave). A stored fact rings loud; crosstalk rings quiet, so unknown queries are
flagged as guesses. Across a chain, coherence *compounds* (a product over hops),
so a two-hop deduction is honestly reported as less certain than a one-hop one.

A third function, :func:`ask_raw`, reads from a *raw* (pre-normalisation)
memory instead (`Ontology.ground_raw`) - the fix for a long-open capacity
question (see `docs/ROADMAP.md`'s "Phase 2 addendum"). Its `coherence` is
on a different scale (concentrates near 1.0 for a correct answer,
regardless of bundle size or `dim`, occasionally slightly above 1.0) and is
compared against its own `RAW_COHERENCE_FLOOR`, not this module's `[0, 1]`
convention above.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable

from .tier2_substrate.collapse import dimensional_collapse, softmax
from .tier2_substrate.energy import Landscape, settle
from .tier2_substrate.hypervectors import Codebook
from .tier2_substrate.hypervectors import similarity as raw_similarity
from .tier2_substrate.resonance import coherence as resonant_coherence
from .tier2_substrate.resonance import interfere, phase_lock
from .tier3_logic.ontology import Ontology

# Below this recalled amplitude the residue is noise - the engine is guessing.
COHERENCE_FLOOR = 0.08

# Steps for _cleanup's energy-settling pass (see its docstring) - measured
# insensitive to the exact count in (15, 20, 25, 30) on the crosstalk probe
# that motivated this, so a modest fixed value rather than a tunable knob.
_CLEANUP_SETTLE_STEPS = 20


@dataclass
class Answer:
    """The engine's response to one single-hop question."""

    answer: str
    coherence: float          # raw amplitude of the recalled wave (rings-true-ness)
    confidence: float         # softmax share vs. the other candidates, in [0, 1]
    ranked: list[tuple[str, float]]
    known: bool               # False => below COHERENCE_FLOOR => a guess
    k_live: int = 0           # size of the live comparison set - see _cleanup's docstring
    eff_dim: float = 0.0      # participation_ratio of the pre-restriction occupancy

    def __str__(self) -> str:
        bar = "#" * int(round(self.confidence * 20))
        flag = "" if self.known else "   << low coherence: guessing / not in KB"
        head = (
            f"{self.answer:<10} coherence={self.coherence:+.3f} "
            f"confidence={self.confidence:5.1%} |{bar:<20}|{flag}  "
            f"(k_live={self.k_live}, eff_dim={self.eff_dim:.1f})"
        )
        runners = "  ".join(f"{name}:{score:+.2f}" for name, score in self.ranked[:3])
        return f"{head}\n              candidates: {runners}"


@dataclass
class Chain:
    """A multi-hop deduction: the trajectory the substrate settled through."""

    start: str
    relation: str
    hops: list[tuple[str, float]]   # (entity, per-hop coherence) in order
    # Compounded coherence after each hop. Measured on chains containing
    # real errors (see docs/ROADMAP.md): *within* a fixed chain length
    # this separates valid from invalid paths perfectly (AUC 1.0000), so
    # it does genuinely track chain quality. Its real defect is that it is
    # not comparable ACROSS lengths - it multiplies in one more factor per
    # hop, so a valid long chain scores below an invalid short one. Hence:
    # do not compare it against COHERENCE_FLOOR (calibrated for single-hop
    # coherence), and do not rank chains of different lengths by it.
    #   Prefer `min_hop_coherence()` for anything cross-chain: it is
    #   equally perfect within a depth and carries no length dependence,
    #   so it stays comparable between chains and against a fixed
    #   threshold.
    cumulative: list[float]
    resonance_coherence: float = 0.0  # see chain()'s docstring

    def reached(self) -> dict[str, float]:
        """Every entity reached, mapped to its compounded coherence."""
        return {name: c for (name, _), c in zip(self.hops, self.cumulative)}

    def min_hop_coherence(self) -> float:
        """The weakest single hop in this chain - the chain-level
        reliability signal to prefer over :attr:`cumulative`.

        Measured head-to-head on real UMLS chains deliberately run with
        the per-hop gate disabled so they contained genuine errors (see
        `docs/ROADMAP.md`). Pooled across depths this separated valid from
        invalid paths perfectly (AUC 1.0000) where `cumulative` managed
        0.9584 and the latest hop's coherence alone managed 0.7694 - but
        the honest reading of that gap needs the stratified result:
        *within* a fixed depth, `cumulative` is also 1.0000. Both
        aggregate over every hop so far, so both catch a bad hop wherever
        it occurred; the latest-hop signal does not (0.5802 at depth 3,
        where a path can be invalid from an earlier hop it cannot see).

        The reason to prefer this one is therefore narrower and more
        precise than "it discriminates better": it carries no length
        dependence. `cumulative` multiplies in one more factor per hop, so
        a valid long chain scores below an invalid short one and its
        values cannot be compared between chains of different lengths (or
        against a fixed threshold such as `COHERENCE_FLOOR`). The weakest
        hop stays on the scale of a single hop no matter how long the
        chain is.

        ``1.0`` for a chain with no hops (nothing weakened it), matching
        `chain`'s own ``prior_coherence`` identity."""
        return min((coh for _, coh in self.hops), default=1.0)

    def __str__(self) -> str:
        if not self.hops:
            return f"{self.start} --{self.relation}--> (nothing resonates)"
        parts = [self.start]
        for (name, coh), cum in zip(self.hops, self.cumulative):
            parts.append(f"--{self.relation}[{coh:+.2f}]--> {name}(cum={cum:.2f})")
        return " ".join(parts)


@dataclass
class Verdict:
    """Answer to a yes/no multi-hop question (does subject reach target?)."""

    holds: bool
    coherence: float           # compounded coherence along the path (0 if not reached)
    hops: int                  # path length in hops (0 if not reached)
    chain: Chain

    def __str__(self) -> str:
        if self.holds:
            return f"YES  (coherence={self.coherence:.3f} over {self.hops} hop(s))  via {self.chain}"
        return "NO   (target does not resonate through this relation)"


def _entity_codebook(
    onto: Ontology,
    entity_vectors: Callable[[str], "Any"] | None = None,
    candidate_names: "Any" = None,
) -> Codebook:
    """A :class:`Codebook` view scoped to just ``onto``'s entities.

    :func:`~zeuss.tier2_substrate.collapse.dimensional_collapse` needs a
    codebook over exactly the answer candidates, not `Ontology.codebook`'s
    full internal alphabet (roles/relations mixed in with entities). Built
    via :meth:`Codebook.load`, which restores vectors verbatim without
    touching an RNG stream - so this reuses ``onto``'s actual entity waves,
    it does not mint new ones.

    ``entity_vectors`` (optional): looks candidates up through this instead
    of `Ontology.entity` - see `_cleanup`'s docstring for why this needs to
    stay consistent with ``subject_vector``, not just override the probe.

    ``candidate_names`` (optional): restricts the codebook to exactly these
    names instead of every entity `onto` knows about - see `_cleanup`'s
    docstring for why this matters for `ask_sharded` specifically (each
    shard only ever contains a fraction of the ontology's entities, so
    scoring it against the *whole* ontology's codebook was pure wasted
    work, not a correctness requirement). ``None`` (default) is an exact
    no-op, identical to every prior call site.
    """
    lookup = onto.entity if entity_vectors is None else entity_vectors
    names = onto.entity_names() if candidate_names is None else candidate_names
    cb = Codebook(dim=onto.dim)
    cb.load({name: lookup(name) for name in names})
    return cb


def _cleanup(
    onto: Ontology,
    residue,
    beta: float,
    axiom_bias: Callable[[str], float] | None = None,
    entity_vectors: Callable[[str], Any] | None = None,
    candidate_names: Any = None,
):
    """Collapse a residue wave onto the nearest KB entity (associative read).

    Scores each candidate by :func:`~zeuss.tier2_substrate.resonance.
    phase_lock` (a Kuramoto-style order parameter, ``|mean(a * conj(b))|``)
    rather than plain cosine ``similarity`` - Frontier 3's own module
    (`resonance.py`) had zero production call sites anywhere in this
    project before this; `phase_lock` in particular was entirely dead code.
    `phase_lock` also tolerates a global phase offset between the recalled
    residue and a candidate's wave (e.g. accumulated drift over several
    binds) that plain ``Re(similarity)`` would silently discount, without
    losing the two functions' near-equivalence on well-formed queries
    (measured directly, not assumed: on every case this project's own demo
    ontology exercises, both functions pick the identical top candidate,
    with scores agreeing to 3+ decimal places) - so this is a genuine
    mechanism swap, not a cosmetic rename, with no behavioral regression on
    this codebase's own committed test suite (`test_ask.py`/`test_chain.py`).

    Before scoring, the candidate set itself is restricted to Frontier 1's
    own live basis: :func:`~zeuss.tier2_substrate.collapse.
    dimensional_collapse` (`participation_ratio` over a similarity-based
    occupancy - previously wired into nothing downstream, see `docs/
    ROADMAP.md` v0.34) is run over the full entity set, and only its
    ``live_names`` survive to be scored by `phase_lock`. Checked directly
    on the demo ontology before wiring this in, not assumed: known queries
    shrink the 11-entity comparison set down to 2 (the true answer plus one
    runner-up); genuine guesses (unknown subject, wrong relation) correctly
    stay near the full 11 - there is no real winner for entropy to collapse
    toward - and the true `phase_lock` top pick is always inside the live
    set in every case this project's own demo ontology exercises. So this
    "restricts once local entropy is low" automatically, not via an
    explicit threshold: `dimensional_collapse` already returns the full set
    at high entropy (see `test_dimensional_collapse_matches_collapse_at_
    high_entropy`).

    Finally, the *winning* candidate is chosen after a Frontier 2 energy
    settle, not from the raw one-shot residue: the live basis (weighted by
    its own occupancy) becomes a :class:`~zeuss.tier2_substrate.energy.
    Landscape`, :func:`~zeuss.tier2_substrate.energy.settle` relaxes the
    residue toward it, and `phase_lock` against *that* settled state - not
    the raw residue - decides ``top``. This is what makes the `collapse ->
    energy -> resonance` pipeline `docs/ARCHITECTURE.md` diagrams literally
    true for a real query, not three independently-tested but disconnected
    mechanisms (see `docs/ROADMAP.md` v0.36 for how this gap was found).
    Measured before wiring in, not assumed: on a deliberately crosstalk-
    heavy synthetic ontology (dim=512, 40 chained entities - the demo KB's
    dim=8192 has almost no crosstalk to correct), settle-informed selection
    recovers 1 of 3 cases plain one-shot `phase_lock` gets wrong, breaking
    0 of the 36 it already got right - insensitive to the exact step count
    (checked 15/20/25/30, identical result).

    Critically, ``coherence``/``confidence``/``ranked`` are still computed
    from the *original*, unsettled residue, never from the settled state.
    A first attempt used the settled state for these too, and that is
    actively wrong, not just unnecessary: `settle`'s dynamics are a
    self-reinforcing attractor network by construction (Frontier 2's whole
    point) - `landscape.target()` pulls `z` toward its own softmax-weighted
    read of `z`, which sharpens that same read, which pulls harder - so
    *any* residue, including pure crosstalk noise with no real answer,
    converges toward amplitude ~1.0 against whichever candidate it drifted
    toward first. Measured directly: this collapsed `demo_ontology`'s own
    `dragon is_a ?` guess to coherence +1.000 (`known=True`), silently
    destroying the "unknown queries are flagged as guesses" guarantee
    `test_unknown_queries_are_flagged_as_guesses` depends on. Reporting
    coherence from the untouched residue keeps that guarantee exactly as
    before while still letting the settled state's argmax correct which
    candidate is picked.

    ``axiom_bias`` (v0.37) is the hook that closes v0.36's own documented
    remaining gap: a `Theory`/`Rule` (Frontier 2's *propositional* logic
    formalism, `compiler.py`) still had no bridge to `Ontology`'s relational
    facts. Rather than force that bridge through a contrived shared data
    model, this exposes the minimal real seam - an optional ``candidate ->
    additional energy penalty`` callable, applied as `exp(-penalty)` on top
    of that candidate's occupancy-derived `Landscape` weight (the same
    Boltzmann convention `grounding.compile_theory` already uses for its own
    attractor weights). The caller builds the penalty from a `Rule` and
    *other* `ask()` calls about the same subject - letting one relation's
    already-established fact veto a logically-inconsistent candidate for a
    different, genuinely ambiguous one. Measured on a constructed case where
    this matters (a KB with a real data contradiction - `socrates is_a` both
    `human` and `star` - so raw resonance alone is close to a coin flip,
    seed-dependent): a `Rule("walks_on_earth", "not_star")`, its antecedent
    filled in from a separate `ask(subject, "walks_on")`, corrects 4 of 10
    seeds' wrong `star` pick to `human`, and 0 of the 6 already-correct
    seeds regress (`test_axiom_bias_resolves_a_genuine_data_contradiction`).
    Like the settled-state itself, ``axiom_bias`` only ever influences which
    candidate wins - never ``coherence``/``confidence``, for the identical
    reason given above. Default ``None`` is a true no-op (unchanged
    `Landscape` weights), so every prior test/behavior is untouched.

    ``entity_vectors`` (optional, default ``onto.entity``): looks up every
    *candidate* through this instead. This exists for exactly one reason,
    found the hard way (see `Ontology.refine_entity_vectors`'s docstring):
    overriding only the probe (``subject_vector``) while candidates are
    still scored against `Ontology.entity` measurably starves a refined-
    vector query of most of its own effect - refined probe + raw
    candidates against a raw memory recovered only 2/6 of a synthetic
    domain's withheld facts, refined probe + refined candidates recovered
    3/6, and refined probe + refined candidates + a memory *also* built
    from refined vectors (`Ontology.ground_refined`) recovered 5/6 -
    matching the original, pre-separation design exactly. A generalising
    query needs all three consistent (``subject_vector``, ``entity_
    vectors``, and the memory itself), not just the probe.

    ``candidate_names`` (optional, default every entity `onto` knows
    about): restricts candidate scoring/settling to just these names -
    see `_entity_codebook`'s docstring. This is what `ask_sharded` uses to
    score each shard against only its own entities instead of the whole
    ontology's - measured to be most of the reason `ask_sharded` was ever
    slow at large ontology sizes in the first place (a 20-shard/1600-
    triple/~1605-entity case: ~48s/query unscoped -> ~3s/query scoped,
    both still through this exact `dimensional_collapse`+`settle` pass -
    see `docs/ROADMAP.md`'s "Phase 2 addendum, continued" for the full
    isolation). ``None`` (default) is an exact no-op.
    """
    lookup = onto.entity if entity_vectors is None else entity_vectors
    entity_cb = _entity_codebook(onto, entity_vectors, candidate_names)
    _, dim_info = dimensional_collapse(entity_cb, residue, inverse_temperature=beta)
    candidates = dim_info["live_names"]
    live_probs = dict(zip(dim_info["names"], (float(p) for p in dim_info["probs"])))

    scores = [phase_lock(residue, lookup(c)) for c in candidates]
    probs = softmax([beta * s for s in scores])
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    ranked = [(candidates[i], float(scores[i])) for i in order]

    landscape = Landscape()
    for c in candidates:
        weight = live_probs[c]
        if axiom_bias is not None:
            weight *= math.exp(-axiom_bias(c))
        landscape.add(lookup(c), weight=weight)
    z_settled, _ = settle(
        landscape, residue, steps=_CLEANUP_SETTLE_STEPS, step_size=0.3, inverse_temperature=beta
    )
    settled_scores = [phase_lock(z_settled, lookup(c)) for c in candidates]
    top = max(range(len(candidates)), key=lambda i: settled_scores[i])

    return candidates[top], ranked, float(scores[top]), float(probs[top]), dim_info["k_live"], dim_info["eff_dim"]


def ask(
    onto: Ontology,
    memory,
    subject: str,
    relation: str,
    beta: float = 12.0,
    axiom_bias: Callable[[str], float] | None = None,
    subject_vector=None,
    entity_vectors: Callable[[str], Any] | None = None,
    candidate_names: Any = None,
) -> Answer:
    """Single hop: probe ``memory`` for ``(subject, relation, ?)``.

    ``axiom_bias`` - see `_cleanup`'s docstring - lets a `Rule`/`Theory`
    penalty (typically built from other `ask()` calls about ``subject``)
    veto a candidate that would be logically inconsistent with an
    already-established fact. ``None`` (default) is a true no-op.

    ``subject_vector`` (optional): probe with this hypervector instead of
    ``onto.entity(subject)`` - the seam `Ontology.entity_refined` uses to
    query with a neighbour-refined representation (see its docstring and
    `Ontology.refine_entity_vectors`'s) without every caller needing to
    touch the codebook directly. ``subject`` is still used for candidate
    bookkeeping (``ranked``, ``answer``); only the probe vector changes.
    ``None`` (default) is an exact no-op, identical to every prior call
    site of this function.

    ``entity_vectors`` (optional): forwarded to `_cleanup` - see its
    docstring for why a generalising query needs this set *alongside*
    ``subject_vector`` (to ``onto.entity_refined``), not either alone, and
    needs ``memory`` itself built from `Ontology.ground_refined`/`ground_
    sharded_refined` too for the effect to reach its measured strength.

    ``candidate_names`` (optional): forwarded to `_cleanup` - restricts
    candidate scoring to just these entities instead of every entity
    `onto` knows about. ``None`` (default) is an exact no-op; the seam
    `ask_sharded` uses to score each shard against only its own entities
    (see `_cleanup`'s docstring for the measured cost of not doing this).
    """
    subject_hv = onto.entity(subject) if subject_vector is None else subject_vector
    residue = onto.step(memory, subject_hv, relation)
    name, ranked, coherence, confidence, k_live, eff_dim = _cleanup(
        onto, residue, beta, axiom_bias, entity_vectors, candidate_names
    )
    return Answer(
        answer=name,
        coherence=coherence,
        confidence=confidence,
        ranked=ranked,
        k_live=k_live,
        eff_dim=eff_dim,
        known=coherence >= COHERENCE_FLOOR,
    )


# Raw-pipeline correct-candidate coherence concentrates near a fixed
# constant (~1.0) regardless of N/dim (see ask_raw's docstring) - this
# floor is calibrated against that scale, not COHERENCE_FLOOR's. Measured
# directly at dim/N ratios matching v0.39's own tested scale (all with
# ~N+5 candidates in the codebook, since `known` is a max over all of
# them - see ask_raw's docstring on why the margin narrows with pool
# size): N=400/dim=65536 (v0.39's own "large dim" test point, where the
# OLD pipeline showed zero improvement) -> known 0.94-1.06, worst-case
# guess 0.20, a 4.76x margin; N=800/dim=65536 -> margin narrows to 3.18x
# (same dim, bigger bundle); N=800/dim=131072 (dim doubled to match) ->
# margin restored to 5.10x - textbook capacity scaling (margin ~
# sqrt(dim/N)), the thing v0.39's OLD pipeline never showed at all.
# 0.5 sits with real margin below every "known" measurement above and
# above every "guess" measurement *at those ratios* - but at a much lower
# dim/N ratio (dim=8192/N=400, ~20x) worst-case guesses reached ~0.60,
# ABOVE this floor. This default is safe for dim/N ratios in roughly the
# 80x-165x range actually measured, not for arbitrarily small ones -
# pass a larger `coherence_floor` (or a bigger `dim`) outside that range,
# per ask_raw's own docstring.
RAW_COHERENCE_FLOOR = 0.5


def ask_raw(
    onto: Ontology,
    memory_raw,
    subject: str,
    relation: str,
    beta: float = 12.0,
    subject_vector=None,
    coherence_floor: float = RAW_COHERENCE_FLOOR,
    candidate_names: Any = None,
) -> Answer:
    """Like `ask`, but reads from a *raw* (pre-normalisation) memory via
    `Ontology.step_raw` instead of `step` - the fix for v0.39's old,
    unexplained "raising `dim` doesn't rescue the single-bundle ceiling"
    finding (see `Ontology.ground_raw`'s docstring and `docs/ROADMAP.md`'s
    "Phase 2 addendum" for the full mechanism and derivation).

    ``memory_raw`` must come from `Ontology.ground_raw()` (or any
    `IncrementalMemory`'s own `.raw`), not `ground()`/`.vector` - the whole
    point is reading the sum *before* the projection that caps capacity.

    Deliberately scoped narrower than `ask`: no `axiom_bias`, no
    `dimensional_collapse`/`settle` candidate-narrowing/energy-refinement
    pass, no `entity_vectors` override - this is the minimal, directly-
    tested capacity fix, not a drop-in replacement for `ask`'s full
    feature set (nothing stops those from being layered on later; not
    done here since neither was needed to verify the capacity claim
    itself). Every candidate in `onto`'s codebook is scored directly via
    :func:`~zeuss.tier2_substrate.hypervectors.similarity` against the raw
    residue - deliberately *not* `_cleanup`'s `phase_lock` (`abs(mean(a *
    conj(b)))`): `phase_lock` takes the magnitude of a complex mean, whose
    expectation for a genuinely wrong candidate is *positive* (Rayleigh-
    distributed, not zero) - fine at the ordinary pipeline's scale, but on
    a raw residue this biases every wrong candidate's score upward by an
    amount that itself grows with bundle size, undermining exactly the
    separation this function exists to provide (measured directly: with
    `phase_lock`, a genuinely-absent entity scored ~0.55 at dim=8192/
    N=400, uncomfortably close to a real fact's ~1.0). `similarity`'s real
    part *is* zero-mean for a wrong candidate regardless of bundle size
    (see the derivation in `docs/ROADMAP.md`'s "Phase 2 addendum") - it's
    the metric that was actually validated there and is the one this
    function's whole capacity claim rests on.

    Uses its own confidence floor, ``coherence_floor`` (default
    `RAW_COHERENCE_FLOOR`) - the raw pipeline's scale is fundamentally
    different from the ordinary path's (correct-candidate coherence
    concentrates near a fixed constant regardless of bundle size or
    `dim`, rather than shrinking as the bundle grows), so `COHERENCE_
    FLOOR` (calibrated for the *other* pipeline's scale) does not apply
    here. **Reliability still depends on `dim` being comfortably larger
    than the bundle's triple count, and the margin narrows as the
    candidate pool grows** (`known` picks the *max* similarity across
    every entity in `onto` - an extreme-value statistic over however many
    candidates that is, the same "more chances for noise to spike" effect
    `docs/ROADMAP.md`'s v0.40/v0.41 entries already found for shard
    selection, now showing up at the candidate level instead) - see
    `RAW_COHERENCE_FLOOR`'s own comment for the measured margin at a few
    `dim`/`N` ratios, and pass a caller-tuned ``coherence_floor`` for a
    bundle/candidate-pool size outside what's been measured, rather than
    trusting the default blindly. This is a real, disclosed requirement
    of the fix, not a claim that any `dim` works for any bundle size.

    ``candidate_names`` (optional): restricts candidate scoring to just
    these entities instead of every entity `onto` knows about - see
    `_entity_codebook`'s docstring (the same seam `ask_sharded` uses).
    ``None`` (default) is an exact no-op. **This one matters more here
    than it does for `ask`/`ask_sharded`**: a naive per-shard raw query
    loop that doesn't pass this scores every shard against the *whole*
    ontology's entities, not just that shard's own - since the number of
    shards *and* the unscoped candidate count both grow with total KB
    size, unscoped per-query cost is `O(shards * total_entities) =
    O(N^2/shard_size)`, not `O(N)` - measured directly: going from 3200
    to 6000 total entities (1.875x) with an unscoped candidate loop made
    per-query latency ~3.8x worse, consistent with the quadratic term
    dominating, not the `O(N)` memory story sharding is supposed to buy.
    Passing each shard's own entity set here restores `O(N)` query cost
    to match `O(N)` memory - see `docs/ROADMAP.md`'s "Phase 2 addendum,
    continued" for the measured before/after.
    """
    subject_hv = onto.entity(subject) if subject_vector is None else subject_vector
    residue = onto.step_raw(memory_raw, subject_hv, relation)
    names = onto.entity_names() if candidate_names is None else list(candidate_names)
    scores = [raw_similarity(residue, onto.entity(name)) for name in names]
    order = sorted(range(len(names)), key=lambda i: scores[i], reverse=True)
    ranked = [(names[i], float(scores[i])) for i in order]
    probs = softmax([beta * scores[i] for i in order])
    top_name, top_score = ranked[0]
    return Answer(
        answer=top_name,
        coherence=float(top_score),
        confidence=float(probs[0]),
        ranked=ranked,
        k_live=len(names),
        eff_dim=float(len(names)),
        known=float(top_score) >= coherence_floor,
    )


def ask_sharded(
    onto: Ontology,
    memories: list,
    subject: str,
    relation: str,
    beta: float = 12.0,
    axiom_bias: Callable[[str], float] | None = None,
    shard_weights: list[float] | None = None,
    subject_vector=None,
    entity_vectors: Callable[[str], Any] | None = None,
    shard_entities: list | None = None,
) -> Answer:
    """`ask`, but across several independent memory hypervectors (see
    `Ontology.ground_sharded`) instead of one - queries every shard and
    keeps whichever `Answer` rings loudest (highest `coherence`).

    ``shard_weights`` (v0.46, optional, positionally aligned with
    ``memories``): a continuous per-shard trust score - see
    `ontology.shard_trust_weights` - multiplied into `coherence` *before*
    both the cross-shard argmax and the `known` floor check. This is
    where continuous shard trust actually has to live: weighting a
    shard's own triples uniformly before bundling (see
    `Ontology.ground_trust_weighted_shards`) is a no-op for
    `ask_sharded`'s purposes, since :func:`~zeuss.tier2_substrate.
    hypervectors.bundle` renormalizes every element back onto the unit
    circle regardless of a shared scalar weight - a low-trust shard's
    *own* bundle looks identical whether weighted or not. What genuinely
    changes outcomes is scaling *this* function's cross-shard comparison:
    a noise shard that happens to resonate loudly with a probe by chance
    is dampened below `COHERENCE_FLOOR` (or below a genuinely trustworthy
    shard's honest signal) instead of winning the argmax outright.
    ``None`` (default) is an exact no-op, equivalent to every weight being
    ``1.0``.

    ``subject_vector``/``entity_vectors`` (optional): see `ask`'s
    docstring - forwarded unchanged to every shard's own `ask` call.

    This is the real fix for `docs/ROADMAP.md` v0.39's measured ceiling: a
    single `ground()` bundle stays reliable to ~80 triples and collapses
    sharply past that, and raising `dim` doesn't rescue it. Splitting the
    same data into several shards at that reliable size and picking the
    best-resonating one raises *effective* capacity roughly linearly with
    shard count while keeping each shard at the size recovery is actually
    proven to work at - measured, not assumed: 90% accuracy at 800 triples
    (10 shards of 80) versus 0% for one 800-triple bundle, with **zero**
    false positives on genuine unknown queries at every scale tested
    when every shard carries real, structured data (`test_sharding.py`).

    **v0.41 - safety is specific to real-data shards, not shard count in
    general.** The zero-false-positives result above does not hold once
    shards carry *unstructured* content: mixing in as few as 10 random-
    noise shards alongside 5 real ones (still far fewer than the 85 shards
    that stayed safe with all-real data) measurably breaks the guess-
    detection guarantee (`test_sharding_is_fragile_to_unstructured_noise_
    shards`). Checked directly, ruling out an obvious confound (noise
    triples coincidentally reconstructing a valid answer by chance): the
    same collapse happens even when noise fillers can never be a valid
    answer to the query's relation. Don't treat "add more shards" as safe
    in general - it's validated for shards that are genuine partitions of
    real data, not for arbitrary or adversarial content.

    ``shard_entities`` (optional, positionally aligned with ``memories``):
    each element restricts that shard's own `ask` call to just those
    candidate names (`Ontology.ground_sharded_with_entities`/`ground_
    shards_with_entities` build this aligned with their returned memories
    automatically). ``None`` (default) is an exact no-op - every shard
    scores against the whole ontology, the original behaviour. This is
    not just an optimisation: measured directly at 1600 triples/20 shards/
    ~1605 entities, scoring every shard against the whole ontology (the
    unscoped default) cost ~48s/query; scoping each shard to just its own
    ~85 entities - identical `dimensional_collapse`/`settle` machinery,
    identical accuracy and false-positive rate - cost ~3s/query, a ~16x
    difference that was pure wasted comparison work, not anything
    `ask_sharded` needed to stay correct. See `docs/ROADMAP.md`'s "Phase 2
    addendum, continued" for the full isolation (including what the
    remaining gap against `ask_raw` actually is once this is accounted
    for).
    """
    if not memories:
        raise ValueError("ask_sharded requires at least one memory - see Ontology.ground_sharded")
    if shard_weights is not None and len(shard_weights) != len(memories):
        raise ValueError("shard_weights must be the same length as memories")
    if shard_entities is not None and len(shard_entities) != len(memories):
        raise ValueError("shard_entities must be the same length as memories")
    best: Answer | None = None
    best_score = -float("inf")
    for i, memory in enumerate(memories):
        answer = ask(
            onto, memory, subject, relation, beta, axiom_bias, subject_vector=subject_vector,
            entity_vectors=entity_vectors,
            candidate_names=None if shard_entities is None else shard_entities[i],
        )
        if shard_weights is None:
            score = answer.coherence
        else:
            score = answer.coherence * shard_weights[i]
            answer = replace(answer, coherence=score, known=score >= COHERENCE_FLOOR)
        if score > best_score:
            best_score = score
            best = answer
    return best


def chain(
    onto: Ontology,
    memory,
    subject: str,
    relation: str,
    max_hops: int = 6,
    beta: float = 12.0,
    axiom_bias: Callable[[str], float] | None = None,
    prior_coherence: float = 1.0,
    coherence_floor: float = COHERENCE_FLOOR,
) -> Chain:
    """Iterate the one-hop operator to walk ``relation``'s transitive closure.

    ``axiom_bias`` (see `ask`/`_cleanup`) is applied at every hop.

    ``coherence_floor`` (optional, default `COHERENCE_FLOOR` = exact
    no-op): the per-hop cutoff below which the chain stops. Lowering it
    admits hops the calibrated single-hop gate would normally reject -
    which is how `docs/ROADMAP.md`'s compounding-calibration experiment
    gets chains that contain *errors* at all (with the default floor,
    every recorded hop has already passed that gate, so chains are
    ~100% valid and there is no variance any confidence signal could
    predict). Not recommended for ordinary use: the floor is exactly
    what keeps multi-hop deduction honest.

    Each hop: take a wave step, collapse the residue onto the nearest entity
    (the discretisation), and re-inject that clean entity as the next subject.
    Stops when the residue stops resonating (coherence < floor), on a cycle, or
    at ``max_hops``.

    ``Chain.resonance_coherence`` is a second, independent signal about the
    *whole* trajectory, not just the per-hop product already in
    ``cumulative``: it composes :meth:`Ontology.step` the same number of
    times ``len(hops)`` demanded, but *without* collapsing onto a clean
    entity between hops (the raw wave operator applied straight through),
    then reads how strongly that uncollapsed composition still resonates
    with the same final entity ``chain`` actually settled on
    (:func:`~zeuss.tier2_substrate.resonance.coherence` of
    :func:`~zeuss.tier2_substrate.resonance.interfere`-superposing the two).
    High resonance means the deduction holds together as one continuous
    multi-hop composition, not merely as a sequence of individually-clean
    single hops; low resonance is an honest signal that the per-hop
    collapse-and-reinject was doing real error-correction work a single
    uninterrupted wave composition could not have done alone - a distinct,
    genuinely new piece of information ``cumulative``'s per-hop product
    alone doesn't carry (checked directly: `test_chain_resonance_coherence_
    is_high_for_a_clean_transitive_chain`). ``0.0`` when no hops were taken.

    ``prior_coherence`` (optional, default ``1.0`` = exact no-op): seeds
    ``cumulative``'s running product, for a chain whose *starting entity*
    was itself inferred rather than known. The motivating case is the
    generalise-then-deduce composition (see `docs/ROADMAP.md`'s
    composability entry): `ask`/`ask_sharded` with ``entity_vectors=
    onto.entity_refined`` recovers an entity that was never directly
    stored, at its own coherence; feeding that entity straight into
    `chain` previously reported the resulting deduction as though it had
    started from certainty, silently dropping the uncertainty the
    generalising step honestly reported. Passing that step's own
    ``coherence`` here compounds it into every hop's ``cumulative``, the
    identical multiplicative convention already used *between* hops.

    Propagating that uncertainty is the semantically right thing to do -
    the generalising step's own coherence *is* calibrated, and silently
    discarding it was worse - but see `Chain.cumulative`'s own comment
    before interpreting the result: the compounded number it feeds is
    **not** a calibrated confidence, so a small ``cumulative`` is not
    evidence the deduction is wrong.

    Deliberately does **not** gate the per-hop `COHERENCE_FLOOR` check -
    that check asks "does *this hop* resonate against the memory", a
    question about hop evidence quality, not about whether the chain
    started somewhere right. Folding the prior into it would truncate
    chains whose hops are each individually clean purely because the
    starting point was uncertain, conflating two genuinely different
    signals. Only ``cumulative`` reflects the prior.
    """
    ent = onto.entity(subject)
    visited = {subject}
    hops: list[tuple[str, float]] = []
    cumulative: list[float] = []
    running = prior_coherence
    for _ in range(max_hops):
        residue = onto.step(memory, ent, relation)
        name, _ranked, coherence, _conf, _k_live, _eff_dim = _cleanup(onto, residue, beta, axiom_bias)
        if coherence < coherence_floor or name in visited:
            break
        running *= coherence
        hops.append((name, coherence))
        cumulative.append(running)
        visited.add(name)
        ent = onto.entity(name)  # collapse -> re-enter the continuum clean

    resonance_coh = 0.0
    if hops:
        raw = onto.entity(subject)
        for _ in range(len(hops)):
            raw = onto.step(memory, raw, relation)  # composed straight through, no collapse
        final_entity = onto.entity(hops[-1][0])
        resonance_coh = resonant_coherence(interfere([raw, final_entity])) / 2.0
    return Chain(
        start=subject, relation=relation, hops=hops, cumulative=cumulative, resonance_coherence=resonance_coh
    )


def chain_sharded(
    onto: Ontology,
    memories: list,
    subject: str,
    relation: str,
    max_hops: int = 6,
    beta: float = 12.0,
    axiom_bias: Callable[[str], float] | None = None,
    shard_weights: list[float] | None = None,
    prior_coherence: float = 1.0,
    coherence_floor: float = COHERENCE_FLOOR,
    shard_entities: list | None = None,
) -> Chain:
    """`chain`, but across several independent memory hypervectors (see
    `Ontology.ground_sharded`/`ask_sharded`) instead of one - at *every*
    hop, queries every shard and keeps whichever gives the highest
    coherence, not just once at the start. A fact needed partway through a
    chain can live in a different shard than the fact before it.

    ``prior_coherence`` (optional, default ``1.0`` = exact no-op): see
    `chain`'s docstring - seeds ``cumulative``'s running product for a
    chain starting from an entity that was itself inferred (the
    generalise-then-deduce composition), and deliberately does not gate
    the per-hop `COHERENCE_FLOOR` check.

    ``coherence_floor`` (optional, default `COHERENCE_FLOOR` = exact
    no-op): see `chain`'s docstring - lowering it admits hops the
    calibrated single-hop gate would reject, which is only useful for
    deliberately studying chain errors, not for ordinary queries.

    ``Chain.resonance_coherence`` (see `chain`'s docstring) is computed
    against whichever shard won the *final* hop - the memory whose
    evidence the chain actually trusted most for where it ended up. This
    is a reasonable, not load-bearing, choice: unlike `ask_sharded`
    (measured directly against real accuracy numbers), the multi-hop case
    hasn't itself been measured beyond "it runs and stays honest" - treat
    `resonance_coherence` here as a smaller-scope extension, not a result
    carrying the same evidence `ask_sharded`'s per-hop selection does.

    v0.41's caveat on `ask_sharded` applies here too, at every hop: this is
    safe when every shard carries real, structured data, not when shards
    may carry arbitrary/unstructured content.

    ``shard_weights`` (Phase 1, optional, positionally aligned with
    ``memories``): the same continuous per-shard trust weight
    `ask_sharded` accepts - see `ontology.shard_connectivity_weights`
    (real data) or `shard_regularity_weights` (the synthetic, single-
    valued domain it was validated on). Applied identically at *every*
    hop, not just the first: `coherence * weight` decides both which
    shard's answer wins that hop's argmax and whether the hop clears
    `COHERENCE_FLOOR`, so a noise shard's occasional lucky resonance is
    damped at each step of the chain, the same mechanism `ask_sharded`
    already uses for a single hop - not yet independently measured for
    the *multi-hop* case the way `ask_sharded`'s own weighting was (see
    `docs/ROADMAP.md`'s Phase 1 entry); treat this as bringing the same
    mechanism to `chain_sharded`, not as a separately-validated result.
    ``None`` (default) is an exact no-op, equivalent to every weight
    being ``1.0``.

    ``shard_entities`` (optional, positionally aligned with ``memories``):
    the same per-shard candidate scoping `ask_sharded` takes, applied at
    *every* hop - each shard's cleanup compares against only that shard's
    own entities instead of the whole ontology's. Build it with
    `Ontology.ground_sharded_with_entities`/`ground_shards_with_entities`,
    which keep it aligned with the memories they return even when a shard
    is dropped. ``None`` (default) is an exact no-op.

    A chain pays this comparison cost ``max_hops`` times over rather than
    once, so the saving compounds - but **how much it saves depends
    entirely on how much there is to scope away**, and that is a property
    of the graph, not of chaining. Measured on real UMLS (135 entities,
    131 shards, but ~75 entities present in *each* shard because the graph
    is small and extremely dense): only **1.3-1.9x**, because 135 -> 75 is
    barely a reduction. The same parameter on `ask_sharded` over a
    sparser, larger-vocabulary ontology (~1605 entities, ~85 per shard -
    a 19x reduction) was worth ~16x. Expect a real but modest win on dense
    small-vocabulary graphs and a large one on sparse large-vocabulary
    ones.

    Correspondingly, unscoped candidate comparison is *not* the dominant
    cost of chaining on a graph like UMLS - the shard count is (131 shards
    x a `dimensional_collapse`+`settle` pass, per hop). Reducing that is a
    separate, unsolved problem; this parameter does not address it.

    Verified to leave the walked chain unchanged (`test_chain_sharded_
    shard_entities_matches_unscoped`, and on real UMLS chains): scoping
    only removes candidates that shard could never have legitimately
    supplied.
    """
    if not memories:
        raise ValueError("chain_sharded requires at least one memory - see Ontology.ground_sharded")
    if shard_weights is not None and len(shard_weights) != len(memories):
        raise ValueError("shard_weights must be the same length as memories")
    if shard_entities is not None and len(shard_entities) != len(memories):
        raise ValueError("shard_entities must be the same length as memories")
    ent = onto.entity(subject)
    visited = {subject}
    hops: list[tuple[str, float]] = []
    cumulative: list[float] = []
    running = prior_coherence
    winning_memory = memories[0]
    for _ in range(max_hops):
        best = None
        for i, memory in enumerate(memories):
            residue = onto.step(memory, ent, relation)
            name, _ranked, coherence, _conf, _k_live, _eff_dim = _cleanup(
                onto, residue, beta, axiom_bias,
                candidate_names=None if shard_entities is None else shard_entities[i],
            )
            weight = 1.0 if shard_weights is None else shard_weights[i]
            score = coherence * weight
            if best is None or score > best[1]:
                best = (name, score, memory)
        name, score, memory = best
        if score < coherence_floor or name in visited:
            break
        running *= score
        hops.append((name, score))
        cumulative.append(running)
        visited.add(name)
        ent = onto.entity(name)  # collapse -> re-enter the continuum clean
        winning_memory = memory

    resonance_coh = 0.0
    if hops:
        raw = onto.entity(subject)
        for _ in range(len(hops)):
            raw = onto.step(winning_memory, raw, relation)
        final_entity = onto.entity(hops[-1][0])
        resonance_coh = resonant_coherence(interfere([raw, final_entity])) / 2.0
    return Chain(
        start=subject, relation=relation, hops=hops, cumulative=cumulative, resonance_coherence=resonance_coh
    )


def entails(onto: Ontology, memory, subject: str, relation: str, target: str) -> Verdict:
    """Yes/no: does ``subject`` reach ``target`` through ``relation`` (any hops)?"""
    c = chain(onto, memory, subject, relation)
    reached = c.reached()
    if target in reached:
        hops = next(i for i, (name, _) in enumerate(c.hops, start=1) if name == target)
        return Verdict(holds=True, coherence=reached[target], hops=hops, chain=c)
    return Verdict(holds=False, coherence=0.0, hops=0, chain=c)


def demo_ontology(dim: int = 8192, seed: int = 1) -> Ontology:
    """A tiny canned knowledge base, grounded into the substrate."""
    onto = Ontology(dim=dim, seed=seed)
    onto.add("socrates", "is_a", "human")
    onto.add("plato", "is_a", "human")
    onto.add("human", "is_a", "mortal")
    onto.add("mortal", "is_a", "thing")
    onto.add("sun", "is_a", "star")
    onto.add("sky", "has_color", "blue")
    onto.add("grass", "has_color", "green")
    return onto
