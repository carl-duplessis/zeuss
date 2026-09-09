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
"""
from __future__ import annotations

from dataclasses import dataclass

from .tier2_substrate.collapse import dimensional_collapse, softmax
from .tier2_substrate.energy import Landscape, settle
from .tier2_substrate.hypervectors import Codebook
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
    cumulative: list[float]         # compounded coherence after each hop
    resonance_coherence: float = 0.0  # see chain()'s docstring

    def reached(self) -> dict[str, float]:
        """Every entity reached, mapped to its compounded coherence."""
        return {name: c for (name, _), c in zip(self.hops, self.cumulative)}

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


def _entity_codebook(onto: Ontology) -> Codebook:
    """A :class:`Codebook` view scoped to just ``onto``'s entities.

    :func:`~zeuss.tier2_substrate.collapse.dimensional_collapse` needs a
    codebook over exactly the answer candidates, not `Ontology.codebook`'s
    full internal alphabet (roles/relations mixed in with entities). Built
    via :meth:`Codebook.load`, which restores vectors verbatim without
    touching an RNG stream - so this reuses ``onto``'s actual entity waves,
    it does not mint new ones.
    """
    cb = Codebook(dim=onto.dim)
    cb.load({name: onto.entity(name) for name in onto.entity_names()})
    return cb


def _cleanup(onto: Ontology, residue, beta: float):
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
    """
    entity_cb = _entity_codebook(onto)
    _, dim_info = dimensional_collapse(entity_cb, residue, inverse_temperature=beta)
    candidates = dim_info["live_names"]
    live_probs = dict(zip(dim_info["names"], (float(p) for p in dim_info["probs"])))

    scores = [phase_lock(residue, onto.entity(c)) for c in candidates]
    probs = softmax([beta * s for s in scores])
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    ranked = [(candidates[i], float(scores[i])) for i in order]

    landscape = Landscape()
    for c in candidates:
        landscape.add(onto.entity(c), weight=live_probs[c])
    z_settled, _ = settle(
        landscape, residue, steps=_CLEANUP_SETTLE_STEPS, step_size=0.3, inverse_temperature=beta
    )
    settled_scores = [phase_lock(z_settled, onto.entity(c)) for c in candidates]
    top = max(range(len(candidates)), key=lambda i: settled_scores[i])

    return candidates[top], ranked, float(scores[top]), float(probs[top]), dim_info["k_live"], dim_info["eff_dim"]


def ask(onto: Ontology, memory, subject: str, relation: str, beta: float = 12.0) -> Answer:
    """Single hop: probe ``memory`` for ``(subject, relation, ?)``."""
    residue = onto.step(memory, onto.entity(subject), relation)
    name, ranked, coherence, confidence, k_live, eff_dim = _cleanup(onto, residue, beta)
    return Answer(
        answer=name,
        coherence=coherence,
        confidence=confidence,
        ranked=ranked,
        k_live=k_live,
        eff_dim=eff_dim,
        known=coherence >= COHERENCE_FLOOR,
    )


def chain(
    onto: Ontology,
    memory,
    subject: str,
    relation: str,
    max_hops: int = 6,
    beta: float = 12.0,
) -> Chain:
    """Iterate the one-hop operator to walk ``relation``'s transitive closure.

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
    """
    ent = onto.entity(subject)
    visited = {subject}
    hops: list[tuple[str, float]] = []
    cumulative: list[float] = []
    running = 1.0
    for _ in range(max_hops):
        residue = onto.step(memory, ent, relation)
        name, _ranked, coherence, _conf, _k_live, _eff_dim = _cleanup(onto, residue, beta)
        if coherence < COHERENCE_FLOOR or name in visited:
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
