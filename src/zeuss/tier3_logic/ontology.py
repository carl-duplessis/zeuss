"""Map a predicate graph into hypervector seeds.

Symbolic triples are grounded into the continuous substrate by binding a shared
entity core wave ``ENT:x`` under two distinct role waves, ``ROLE:subj`` and
``ROLE:obj``, *plus* a fixed cyclic permutation ``P`` on the object slot:

    subj_wave(x) = bind(ROLE:subj, ENT:x)
    obj_wave(x)  = P(bind(ROLE:obj, ENT:x))
    triple       = bind(REL:r, bind(subj_wave(s), obj_wave(o)))
    memory       = bundle(all triples)

The role waves alone are *not* enough to protect direction once you chain: bind
is elementwise complex multiplication, which is commutative, so when an entity
``x`` is the object of one fact and the subject of the next (exactly what a
transitive chain ``a -> x -> b`` requires), unbinding by ``bind(ROLE:subj, x)``
lets the ``ROLE:subj``/``x`` factors cancel out of *both* the fact where ``x``
is the subject (giving the true successor) *and* the fact where ``x`` is the
object (an exact algebraic collision, not noise - the wrong neighbour ties the
right one exactly). The permutation ``P`` breaks this: it's applied *after* the
role bind, on the object side only, so a probe built purely from role+entity
waves no longer factors cleanly out of a triple where the queried entity sat in
the object slot - that mismatch decays to ordinary quasi-orthogonal noise
instead of an exact tie. Recovering the object needs the extra step of
inverting the permutation before the final ``ROLE:obj`` unbind (see
:meth:`Ontology.step`). An object recovered from one hop can be fed straight
back in as the next subject, so deduction *chains* by iterating one wave
operator, entirely in wave space - no graph walk, no ``if`` over stored facts
(see :mod:`zeuss.qa`).

Uses NetworkX when available for a parallel symbolic graph view, with a tiny
built-in fallback so the module always imports.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..backend import CDTYPE, xp
from ..tier2_substrate.hypervectors import Codebook, bind, bundle, normalize, permute, unbind

# Fixed cyclic shift applied to the object slot - see module docstring for why
# this is load-bearing (breaks a commutative-bind collision on chained entities).
OBJ_SHIFT = 7


def is_structurally_regular(triples) -> bool:
    """Do these ``(subject, relation, object)`` triples ever assert the same
    ``(subject, relation)`` twice?

    `docs/ROADMAP.md` v0.41/v0.42: this - not semantic truthfulness - turned
    out to be what actually distinguishes a shard that's safe to mix into
    `qa.ask_sharded`'s max-selection from one that's disruptive. Measured
    directly: real `Ontology` data (each subject asserting each relation
    once) is always regular by this check; random `(entity, relation,
    entity)` noise sampled with replacement almost never is, because
    repeated sampling from a finite entity pool inevitably reasserts some
    ``(subject, relation)`` pair with a *different*, conflicting filler.
    Content that's regular but *wrong* (real structure, shuffled fillers)
    caused zero measured false positives; content that's irregular (even
    with zero semantic contradiction intended) caused several per 10
    queries - the mechanism is the duplicate-binding structure itself, not
    whether the data happens to be true.

    A genuine data contradiction in real `Ontology` triples (the same
    ``(subject, relation)`` asserted with two different fillers, e.g.
    v0.37's ``socrates is_a`` both ``human`` and ``star``) is *also*
    flagged by this check - so it doubles as a cheap contradiction
    detector for real data, not just a noise filter for untrusted data.
    """
    seen = set()
    for subject, relation, _ in triples:
        if (subject, relation) in seen:
            return False
        seen.add((subject, relation))
    return True


def classify_multi_valued_relations(triples, threshold: float = 0.2, min_subjects: int = 5) -> set:
    """Which relations are legitimately multi-valued (``has_friend``-like:
    many subjects genuinely have more than one filler), inferred from the
    data itself rather than declared - the same "let structure emerge from
    data" principle this project's rule/axiom discovery already applies
    elsewhere (`tier3_logic/rule_discovery.py`, `axiom_mining.py`).

    A relation is classified multi-valued when at least ``threshold``
    (default 20%) of its subjects have more than one distinct filler, AND
    it has at least ``min_subjects`` distinct subjects at all. The
    ``min_subjects`` floor matters at small scale in a way ``threshold``
    alone can't fix: a relation with exactly one subject that holds two
    conflicting fillers (a genuine contradiction, e.g. a real 1-vs-1
    cross-shard tie on ``socrates is_a``) is 100% "multi-valued" by the
    fraction alone, which would wrongly exempt it from conflict resolution
    instead of flagging the contradiction - the same statistical-power
    problem `resolve_shard_conflicts`'s own docstring already documents
    for within-shard statistics, here showing up at the *low* end of
    subject count rather than the high end.

    Computed globally across every triple passed in - measured to be
    robust when noise/garbage data is a *minority* of what's passed in
    (checked directly: correctly returns an empty set with 2 pure-noise
    shards mixed into 20 real ones), but not immune to an adversarial
    *majority* of noise, which inflates every relation's apparent
    multiplicity - see `docs/ROADMAP.md` v0.43's honest scope note before
    trusting this on data that might be mostly untrustworthy.
    """
    fillers_by_subject_per_relation: dict = {}
    for subject, relation, obj in triples:
        fillers_by_subject_per_relation.setdefault(relation, {}).setdefault(subject, set()).add(obj)

    multi_valued = set()
    for relation, by_subject in fillers_by_subject_per_relation.items():
        if len(by_subject) < min_subjects:
            continue
        multi = sum(1 for fillers in by_subject.values() if len(fillers) > 1)
        if multi / len(by_subject) >= threshold:
            multi_valued.add(relation)
    return multi_valued


def _implied_closure(held: set, implication_graph: dict) -> set:
    """Forward transitive closure of ``held`` atoms (each an
    ``(relation, filler)`` pair) through ``implication_graph`` (atom ->
    set of atoms it implies) - plain BFS/DFS reachability, the standard
    forward-chaining consistency check for directed Horn-clause-like
    implications (rule-based reasoning's usual tool for exactly this,
    not a bespoke invention). Returns ``held`` itself, unchanged, when
    ``implication_graph`` is empty - the base case :func:`axiom_violations`
    relies on to stay identical to its pre-chaining (v0.44) behaviour."""
    if not implication_graph:
        return held
    closure = set(held)
    frontier = list(held)
    while frontier:
        atom = frontier.pop()
        for nxt in implication_graph.get(atom, ()):
            if nxt not in closure:
                closure.add(nxt)
                frontier.append(nxt)
    return closure


def axiom_violations(triples, exclusions, implications=None) -> set:
    """Which ``(subject, relation, obj)`` triples in ``triples`` structurally
    violate one of ``exclusions`` - v0.44's answer to the honest ceiling
    v0.43 disclosed: cross-shard vote counting can never be made reliable
    against an adversarial *majority* of bad data (the same wall Byzantine
    fault tolerance and robust statistics hit generally - past ~50%
    contamination, "the majority" and "the truth" stop being the same
    thing by definition). Logical necessity is a genuinely different kind
    of signal from vote counting: it doesn't care how many shards assert a
    fact, only whether that fact is structurally impossible given another
    fact the *same subject* already holds. See :func:`_global_filler_
    consensus`, which uses this to discard votes for an excluded filler
    *before* tallying a majority, not just to override the majority after
    the fact - so a structurally-impossible fact can't win a vote just
    because garbage shards outnumber real ones.

    ``exclusions`` is any iterable of objects exposing
    ``.antecedent_relation``/``.antecedent_filler``/``.consequent_relation``/
    ``.excluded_filler`` - exactly :class:`~zeuss.tier3_logic.axiom_mining.
    DiscoveredExclusion`'s shape, duck-typed rather than imported to avoid
    an `axiom_mining` <-> `ontology` import cycle (`axiom_mining.py` already
    imports `Ontology`), the same boundary pattern
    `axiom_mining.axiom_bias_from_exclusions` uses locally for its own
    `qa.ask` dependency.

    ``implications`` (v0.45, optional, default ``None``): also chain
    through mined :class:`~zeuss.tier3_logic.axiom_mining.
    DiscoveredImplication`-shaped antecedent -> consequent edges (same
    duck-typing) before checking exclusions - the genuine constraint-
    *network* upgrade over v0.44's flat pairwise check, via
    :func:`_implied_closure`'s transitive closure: a subject's directly
    held facts are chained forward through every mined implication first,
    so a violation is caught even when the exclusion's antecedent atom is
    only reachable by *composing several separately-mined rules*, not
    just when it's one of the subject's own directly-asserted facts (a
    real gap v0.44 had - a chain `A -> B -> C` plus `C excludes D` missed
    a subject holding `A` and `D` outright, since neither mined exclusion
    ever named `A` as its antecedent). This is deliberately a transitive-
    closure graph search, not a reuse of `sheaf.py`'s H0/H1 cohomology
    machinery: implication and exclusion are not equality constraints
    (`sheaf.py`'s restriction edges assert `restrict_u * x_u ==
    restrict_v * x_v`), so forcing them through that linear-algebra
    machinery would misrepresent the semantics rather than reuse them -
    forward chaining is the standard, correctly-scoped tool for directed
    Horn-clause-like rules. Passing ``None`` (default) is an exact no-op:
    :func:`_implied_closure` returns ``held`` unchanged with no
    implication graph, so this degrades to precisely v0.44's behaviour.

    **This is only as trustworthy as the ``exclusions`` it's given.**
    Mining exclusions from the very triples being checked just re-derives
    the same vote-counting problem one level up - a majority-garbage
    sample can corrupt `axiom_mining.discover_all_exclusions`'s own
    support/confidence statistics exactly the way it corrupts
    :func:`classify_multi_valued_relations`. The genuine power here comes
    from ``exclusions`` sourced *independently* of the shards under
    suspicion - hand-written `Rule`-derived exclusions (v0.37), or mined
    from a separately-trusted seed sample - not from re-mining the same
    contaminated pool.

    **Second honest limit, found by direct measurement, not assumed:**
    this only excises literal violating *triples* - it cannot fix wrong
    answers caused by hypervector crosstalk once a relation has already
    been exempted from `_global_filler_consensus` entirely (i.e. once
    :func:`classify_multi_valued_relations` misclassifies it as multi-
    valued under heavy contamination). At that point *every* triple under
    that relation, including hundreds of unrelated noise ones, gets
    bundled into the grounded memory unfiltered, and most resulting wrong
    answers are interference from that flood, not any single literal
    contradiction - excising the one or two triples this function actually
    flags measurably removes them from the data, but does not by itself
    restore downstream query accuracy at that contamination level (checked
    directly - see `test_axiom_violation_resolution.py`'s test for exactly
    this distinction).
    """
    facts_by_subject: dict = {}
    for subject, relation, obj in triples:
        facts_by_subject.setdefault(subject, set()).add((relation, obj))

    implication_graph: dict = {}
    for imp in implications or ():
        antecedent_atom = (imp.antecedent_relation, imp.antecedent_filler)
        consequent_atom = (imp.consequent_relation, imp.consequent_filler)
        implication_graph.setdefault(antecedent_atom, set()).add(consequent_atom)

    closure_by_subject: dict = {}

    violations = set()
    for subject, relation, obj in triples:
        subject_facts = facts_by_subject.get(subject, ())
        if subject not in closure_by_subject:
            closure_by_subject[subject] = _implied_closure(subject_facts, implication_graph)
        reachable = closure_by_subject[subject]
        for excl in exclusions:
            if (
                excl.consequent_relation == relation
                and excl.excluded_filler == obj
                and (excl.antecedent_relation, excl.antecedent_filler) in reachable
            ):
                violations.add((subject, relation, obj))
                break
    return violations


def _global_filler_consensus(shards_of_triples, multi_valued_relations, violating=frozenset()) -> dict:
    """For every ``(subject, relation)`` pair of a single-valued relation,
    the majority filler asserted across *every* shard combined - or
    ``None`` if there is no single, clear majority (an exact tie).

    A tie is left unresolved rather than arbitrarily broken by dict/
    ``Counter`` insertion order: a genuine 50/50 disagreement should stay
    an honest unknown, not silently become a confident guess - the same
    "don't manufacture false confidence" principle `qa.py`'s v0.36
    settled-state coherence fix already established for a different
    mechanism.

    ``violating`` (v0.44, see :func:`axiom_violations`) - votes for a
    ``(subject, relation, obj)`` triple in this set are discarded *before*
    tallying, not merely overridden after: a fact that structurally
    contradicts a trusted axiom doesn't get to compete for the majority at
    all, so a garbage majority can't out-vote a true minority fact on a
    pair the exclusions cover. Empty by default - a no-op, matching v0.43's
    original behaviour exactly when no exclusions are supplied.
    """
    from collections import Counter

    votes: dict = {}
    for shard in shards_of_triples:
        for subject, relation, obj in shard:
            if relation in multi_valued_relations:
                continue
            if (subject, relation, obj) in violating:
                continue
            votes.setdefault((subject, relation), Counter())[obj] += 1

    consensus: dict = {}
    for pair, counter in votes.items():
        ranked = counter.most_common()
        consensus[pair] = ranked[0][0] if len(ranked) == 1 or ranked[0][1] > ranked[1][1] else None
    return consensus


def resolve_shard_conflicts(
    shards_of_triples: list,
    multi_valued_relations=None,
    disagreement_threshold: float = 0.3,
    exclusions=None,
    implications=None,
) -> list:
    """The complete conflict-resolution pipeline for a list of candidate
    shards (each a list of triples), applied *before* grounding: keep
    every fact that agrees with the cross-shard consensus (or belongs to a
    multi-valued relation), drop the specific facts that don't, and drop
    an entire shard outright only when *most* of it disagrees with the
    rest of the dataset - evidence the whole shard is unstructured noise,
    not real data with an isolated mistake.

    This supersedes the simpler, cruder :func:`is_structurally_regular` +
    ``ground_shards(skip_irregular=True)`` path from v0.41/v0.42, which
    could not tell a genuine multi-valued relation from a contradiction
    and dropped a shard entirely on a single duplicate pair. See `docs/
    ROADMAP.md` v0.43 for the full derivation, including what was tried
    and rejected on the way here: a within-shard collision-rate/degree-
    variance threshold, ruled out by direct measurement across three
    vocabulary sizes (100/400/1000 subjects per class) - real data and
    garbage's within-shard collision statistics both drift toward the
    *same* near-zero value as vocabulary grows, since a single 80-triple
    sample from a huge space rarely collides with itself regardless of
    how it was built (a genuine, information-theoretic power problem, not
    a tunable-threshold problem); and per-triple excision of only the
    literally-colliding rows alone, ruled out because most of a random
    shard's content never collides with anything within that same shard,
    so excising just the colliding rows leaves the shard almost entirely
    intact and just as disruptive.

    The key insight that makes *this* approach robust and scale-invariant,
    unlike every within-shard statistic tried first: it does not test
    whether a shard's own content looks internally random - it tests
    whether a shard's claims agree with everything *else* already known,
    which a genuinely random filler drawn from a large vocabulary almost
    never does by chance, regardless of vocabulary size. That signal's
    strength comes from the size of the answer space the filler was drawn
    from, not from how large the shard itself is, so it does not degrade
    as the ontology grows - measured directly at 100/400/1000-subjects-
    per-class scale: real-shard disagreement with the consensus stayed at
    0.000 and garbage-shard disagreement stayed at 0.6-0.75 at every scale
    tested, versus every within-shard statistic tried first, which
    drifted toward 0 (indistinguishable from real data) as vocabulary grew.

    ``multi_valued_relations``: pass ``None`` (default) to infer via
    :func:`classify_multi_valued_relations` over all the triples given;
    pass an explicit set to skip inference entirely (e.g. if the schema is
    already known, or to sidestep the adversarial-majority-of-noise
    failure mode inference is not immune to).

    ``exclusions`` (v0.44, optional, default ``None`` = no-op): trusted
    :class:`~zeuss.tier3_logic.axiom_mining.DiscoveredExclusion`-shaped
    logical axioms (see :func:`axiom_violations`). A fact that structurally
    violates one - the same subject also holds the exclusion's antecedent
    fact - is discarded outright and never even counts toward the majority
    vote, regardless of how many shards assert it. This is a genuinely
    different kind of robustness from consensus voting, not a bigger
    version of it: consensus alone caps out at needing an honest majority
    of *shards*; a trusted axiom's power instead comes from being sourced
    independently of the shards under suspicion, so it isn't bounded by
    their contamination fraction at all - see `docs/ROADMAP.md` v0.44 for
    the measured case where this recovers facts a garbage *majority* had
    outvoted. It does NOT break the underlying 50%-contamination wall in
    general (an exclusion mined from the same contaminated pool would
    inherit the same limit - see :func:`axiom_violations`'s own caveat).

    ``implications`` (v0.45, optional, default ``None`` = no-op): trusted
    :class:`~zeuss.tier3_logic.axiom_mining.DiscoveredImplication`-shaped
    rules, forwarded to :func:`axiom_violations` to chain ``exclusions``
    through a transitive-closure constraint network instead of only
    checking a subject's directly-held facts - catches a violation
    several mined rules away, not just one hop away. No effect without
    ``exclusions`` also set (nothing to chain toward).
    """
    all_triples = [t for shard in shards_of_triples for t in shard]
    if multi_valued_relations is None:
        multi_valued_relations = classify_multi_valued_relations(all_triples)

    violating = axiom_violations(all_triples, exclusions, implications) if exclusions else frozenset()
    consensus = _global_filler_consensus(shards_of_triples, multi_valued_relations, violating)

    resolved = []
    for shard in shards_of_triples:
        checkable = [(s, r, o) for s, r, o in shard if r not in multi_valued_relations and (s, r) in consensus]
        if checkable:
            disagreeing = sum(1 for s, r, o in checkable if consensus[(s, r)] != o)
            if disagreeing / len(checkable) > disagreement_threshold:
                # This shard looks like noise relative to the rest of the dataset -
                # empty its content, but keep its *slot* in the output list so the
                # result stays positionally aligned with `shards_of_triples` (callers,
                # including this module's own tests, index into it by shard position;
                # `Ontology.ground_resolved_shards` is the one that actually drops an
                # empty result from what it grounds, via its own `if shard` filter).
                resolved.append([])
                continue

        kept = [
            (s, r, o)
            for s, r, o in shard
            if (s, r, o) not in violating and (r in multi_valued_relations or consensus.get((s, r)) == o)
        ]
        resolved.append(kept)
    return resolved


def shard_disagreement_energy(shard, consensus: dict) -> float:
    """The fraction of ``shard``'s consensus-checkable triples that
    disagree with the cross-shard majority - the same quantity
    :func:`resolve_shard_conflicts` already computes internally to decide
    whether to drop a shard outright, exposed here as a continuous
    ``[0, 1]`` "energy" rather than something only ever compared against
    a single hard ``disagreement_threshold``. ``0.0`` when the shard has
    no checkable triples at all (nothing to disagree with - genuinely
    unknown, not evidence of trustworthiness either way, matching
    :func:`resolve_shard_conflicts`'s own treatment of that case).
    """
    checkable = [(s, r, o) for s, r, o in shard if consensus.get((s, r)) is not None]
    if not checkable:
        return 0.0
    disagreeing = sum(1 for s, r, o in checkable if consensus[(s, r)] != o)
    return disagreeing / len(checkable)


def shard_trust_weights(
    shards_of_triples: list,
    exclusions=None,
    implications=None,
    inverse_temperature: float = 8.0,
) -> list[float]:
    """v0.46's *first* attempt at the crosstalk limitation v0.44/v0.45
    disclosed, kept in the codebase as a documented negative result, not
    deleted: dissolve the hard ``classify_multi_valued_relations`` gate
    (and the hard ``disagreement_threshold`` shard-drop gate) into one
    continuous, per-shard Boltzmann weight - `exp(-inverse_temperature *
    shard_disagreement_energy(shard, consensus))` - the same
    ``exp(-energy)`` idiom :func:`~zeuss.tier3_logic.grounding.
    compile_theory` already uses to turn a discrete symbolic decision into
    a graded one, applied here to shard trust instead of Boolean corners.

    **Measured directly to NOT fix the crosstalk problem - see
    :func:`shard_regularity_weights` for the version that does.** The
    idea was: never call `classify_multi_valued_relations` at all, so
    `consensus` here is computed with `multi_valued_relations=set())` (no
    relation ever exempted) - a genuinely multi-valued relation's tied
    cross-shard vote produces `consensus[pair] = None`
    (:func:`shard_disagreement_energy` treats a tie as neutral, neither
    for nor against trust, specifically so genuine plurality isn't
    punished). That protection is exactly what breaks the signal at the
    scale this module's own tests operate at: with a small shared entity
    pool and heavy contamination, *most* real-vs-noise collisions on a
    given `(subject, relation)` pair land as an exact 1-vs-1 tie too - the
    same shape as genuine plurality, and indistinguishable from it using
    only this pair's own vote count. Measured directly on the exact 50/50
    real/garbage-shard scenario that motivated this: every shard, real
    and noise alike, came back with `shard_disagreement_energy == 0.0`
    and therefore weight `1.0` - zero discrimination, not a weaker
    version of discrimination. This is a real, informative negative
    result (ties cannot be resolved by looking at one pair's vote count
    alone, regardless of how that count is turned into a weight) rather
    than a bug to patch - see :func:`shard_regularity_weights` for the
    signal that sidesteps the ambiguity entirely instead of trying to
    resolve it.

    ``exclusions``/``implications`` (optional): forwarded to
    :func:`axiom_violations` exactly as in :func:`resolve_shard_conflicts`.

    Returns one weight per shard, positionally aligned with
    ``shards_of_triples`` (same alignment convention as
    :func:`resolve_shard_conflicts`'s own output).
    """
    all_triples = [t for shard in shards_of_triples for t in shard]
    violating = axiom_violations(all_triples, exclusions, implications) if exclusions else frozenset()
    consensus = _global_filler_consensus(shards_of_triples, set(), violating)
    return [
        math.exp(-inverse_temperature * shard_disagreement_energy(shard, consensus))
        for shard in shards_of_triples
    ]


def internal_collision_energy(shard) -> float:
    """Fraction of ``shard``'s own distinct ``(subject, relation)`` pairs
    that it asserts *more than once, within itself* - the signal that
    actually fixes the crosstalk problem :func:`shard_trust_weights` was
    measured to fail at. This is the continuous generalisation of v0.41/
    v0.42's own :func:`is_structurally_regular`, and it works for exactly
    the reason that check did: it is purely internal to one shard,
    computed from that shard's own triples alone, never touching any
    other shard's content or any cross-shard vote tally. That is what
    makes it immune to both failure modes found so far - neither overall
    contamination *volume* (what broke `classify_multi_valued_relations`
    in v0.43) nor a genuine multi-valued relation's cross-shard tie (what
    broke :func:`shard_trust_weights` above) can corrupt a signal that
    never looks past one shard's own boundary in the first place. Real
    `Ontology` data (each subject asserting each relation once per shard)
    is always exactly ``0.0`` by construction; random noise sampled with
    replacement from a finite entity pool collides with itself at a low
    but measurably nonzero rate purely from repeated sampling (birthday-
    paradox effect) - measured directly at this project's own 80-triple
    shard size against a ~200-entity pool: real shards ``0.000``, noise
    shards ``0.04-0.10``, stable regardless of how many *other* shards in
    the mix are also noise (checked from 50% up to 94% contamination -
    see `docs/ROADMAP.md` v0.46).
    """
    seen: dict = {}
    for subject, relation, _ in shard:
        seen[(subject, relation)] = seen.get((subject, relation), 0) + 1
    if not seen:
        return 0.0
    duplicated = sum(1 for count in seen.values() if count > 1)
    return duplicated / len(seen)


def shard_regularity_weights(shards_of_triples: list, inverse_temperature: float = 60.0) -> list[float]:
    """v0.46's actual fix for the crosstalk limitation - the "option 1 +
    2 combined" answer after the pure continuous-relaxation idea
    (:func:`shard_trust_weights`) was tried and measured not to work. A
    continuous, per-shard Boltzmann weight
    (`exp(-inverse_temperature * internal_collision_energy(shard))`) from
    a signal that breaks the circularity by never depending on cross-shard
    votes, relation-level classification, or contamination volume at all
    (option 1: bootstrap trust from something the contamination can't
    reach) - meant for :func:`~zeuss.qa.ask_sharded`'s `shard_weights`
    parameter, damping a noise shard's occasional lucky resonance at
    retrieval time rather than trying to bake trust into the bundle itself
    (option 2: see :func:`shard_trust_weights`'s docstring, unchanged
    reasoning, for why weighting inside `bundle()` is a no-op).

    ``inverse_temperature=60.0`` is not an arbitrary round number - chosen
    by direct measurement, not guessed: values from 5 to 100 were tested
    on the exact 50/50 real/garbage-shard scenario that broke v0.43's
    consensus mechanism, and the resulting false-positive count was
    completely insensitive to the exact value across that whole range
    (every value tested restored 0/10 false positives, exactly matching
    the noise-free baseline). 60 sits in the middle of that measured-safe
    range rather than at either edge. Measured further (not assumed to
    generalise from one data point): 0/10 false positives and accuracy
    matching the noise-free baseline exactly, held all the way from 50%
    contamination through 94% (see `docs/ROADMAP.md` v0.46 for the full
    sweep) - a substantially higher honest ceiling than v0.43's consensus
    mechanism alone (which failed open already at 50%).

    **This validation was entirely on a synthetic domain where every real
    relation is single-valued and every real fact is unique - and does NOT
    transfer to real, densely multi-relational data, measured directly
    rather than assumed to generalise.** Tested against the real Nations
    dataset (1992 triples, 55 genuinely multi-valued relations, no
    redundant/repeated triples at all): real shards scored `0.026-0.194`
    internal collision energy and noise shards scored `0.013-0.127` - the
    ranges fully overlap, and the resulting weights showed zero usable
    discrimination (see `docs/ROADMAP.md`'s v0.47 entry). The root cause
    is conceptual, not a bad constant: this signal (and `shard_trust_
    weights` above) both implicitly assume single-valued, redundantly-
    sourced data - a real fact reasserted identically across sources so a
    lone contradicting shard stands out. Most real knowledge graphs, like
    most real relations, don't have that property; a fact is usually
    stated exactly once. :func:`shard_connectivity_weights` is the signal
    that was found to actually work on real data instead, for a different
    reason - it needs no redundancy at all.
    """
    return [math.exp(-inverse_temperature * internal_collision_energy(shard)) for shard in shards_of_triples]


def entity_connectivity_score(shard, global_degree: dict) -> float:
    """The mean real-world "popularity" (global degree - how many triples
    across the *entire* candidate mix touch this entity, as subject or
    object) of the entities a single shard talks about, weighted by how
    often it talks about them.

    v0.47's answer to `shard_regularity_weights`/`shard_trust_weights`
    both failing on real data (see their docstrings): a signal that needs
    no redundant/repeated assertions to work at all, because it isn't
    trying to catch a shard *disagreeing* with anything - it's testing
    whether a shard's pattern of *which entities it talks about* looks
    like real-world structure or like uniform random sampling. Real
    knowledge graphs are never uniform: some entities (a populous country,
    a common ancestor concept, a hub node) participate in far more real
    facts than others, and a real shard - a genuine slice of that
    structure - inherits that skew. A noise shard drawing subjects and
    objects uniformly at random from the same entity pool does not,
    regardless of how many real facts about those same entities exist
    elsewhere in the mix. This needs no relation to ever repeat and no
    fact to ever be reasserted, unlike every cross-shard-vote or within-
    shard-duplication signal tried before it - it only needs entities
    to be realistically non-uniformly popular, which is close to
    universally true of real relational data.

    ``global_degree`` is computed once, from every triple in the *entire*
    candidate mix (real and noise shards together - this is the only
    thing actually available to a detector that doesn't already know
    which shards are trustworthy) - see :func:`shard_connectivity_weights`.
    """
    local: dict = {}
    for subject, _relation, obj in shard:
        local[subject] = local.get(subject, 0) + 1
        local[obj] = local.get(obj, 0) + 1
    if not local:
        return 0.0
    total = sum(local.values())
    return sum(count * global_degree.get(entity, 0) for entity, count in local.items()) / total


def shard_connectivity_weights(shards_of_triples: list, inverse_temperature: float = 2.0) -> list[float]:
    """Turn :func:`entity_connectivity_score` into a per-shard trust
    weight in ``(0, 1)`` - a logistic (sigmoid) centred on the *z-score*
    of each shard's own score relative to the mean and standard deviation
    of every shard's score in this specific candidate mix, not a fixed
    absolute threshold.

    **Why self-normalising, not a constant like `shard_regularity_
    weights`'s `inverse_temperature=60.0`.** A raw connectivity score's
    magnitude depends on the dataset's own scale (entity count, triple
    count, degree distribution shape) - there is no portable absolute
    number that means "trustworthy" across different knowledge graphs the
    way a `[0, 1]`-normalised fraction (like collision energy) has. Zeroing
    each shard's score against the *other shards in the same mix* sidesteps
    needing one: whatever the real absolute scale of connectivity happens
    to be for this dataset, real shards should cluster above the mix's own
    mean and noise shards below it, because the comparison is relative to
    data that's already scaled the same way.

    Measured directly (not assumed) on the real Nations dataset at 50%
    contamination, the exact scenario where `shard_regularity_weights`
    showed zero discrimination: real shard weights ranged `0.38-0.97`,
    noise shard weights ranged `0.05-0.54` - imperfect overlap at the
    tails, not the clean separation `shard_regularity_weights` achieved on
    its own (single-valued, redundant) synthetic domain, but enough to
    matter. Wired into the full `ask_sharded` pipeline
    (:meth:`Ontology.ground_shards_with_connectivity_trust`): false
    positives on genuinely-absent facts fell from 37/38 (plain, unweighted
    grounding) to 0/38, and recall of real stored facts *improved* at the
    same time, from 21/40 to 31-37/40 across three different noise seeds
    and two contamination levels (50% and 62%) - not a trade-off between
    the two, a simultaneous improvement in both, checked directly rather
    than assumed to hold from one lucky run.

    ``inverse_temperature=2.0`` controls how sharply the sigmoid commits
    around the z-score of 0 (the mix's own mean) - a smaller value spreads
    weights more gently across the whole range, a larger value pushes
    harder toward 0 or 1 the further a shard's score sits from the mean.
    Not swept as exhaustively as `shard_regularity_weights`'s constant;
    2.0 was measured to work well across the scenarios above; treat as a
    reasonable default, not a fully mapped-out safe range.
    """
    if not shards_of_triples:
        return []
    global_degree: dict = {}
    for shard in shards_of_triples:
        for subject, _relation, obj in shard:
            global_degree[subject] = global_degree.get(subject, 0) + 1
            global_degree[obj] = global_degree.get(obj, 0) + 1
    scores = [entity_connectivity_score(shard, global_degree) for shard in shards_of_triples]
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = variance**0.5 or 1.0
    return [1.0 / (1.0 + math.exp(-inverse_temperature * ((s - mean) / std))) for s in scores]


@dataclass
class IncrementalMemory:
    """Phase 1: an O(1)-per-fact incremental counterpart to
    :func:`~zeuss.tier2_substrate.hypervectors.bundle` - every `ground*`
    method on `Ontology` rebuilds its whole bundle from scratch on every
    call, with no way to cheaply add one new fact to an existing large
    knowledge base.

    **Why a batch bundle can't just be appended to.**
    :func:`~zeuss.tier2_substrate.hypervectors.normalize` (what `bundle`
    calls at the end) is *per-element* phase projection
    (``z / abs(z)``), not a single global magnitude rescaling - once a
    vector has been normalised, the true pre-normalisation magnitude at
    each dimension is gone, so a new fact cannot be cheaply folded into
    an *already-normalised* bundle. The raw, pre-normalisation complex
    sum can be added to trivially, though - that's exactly what
    `bundle([v1, ..., vn])` computes before its own final `normalize`
    call. This class keeps that raw sum as its actual state and only
    projects to the unit-modulus, query-ready form on demand via
    `.vector`, so `add()` is a single elementwise complex addition
    (`O(dim)`, not `O(n)` triples re-processed) regardless of how many
    facts have already been added.

    Verified, not assumed, to be numerically equivalent to the batch
    path: after the same sequence of adds, `.vector` matches
    `bundle(same_vectors)` to within floating-point tolerance (see
    `test_incremental_grounding.py`).
    """

    dim: int
    _raw: "xp.ndarray" = field(init=False)
    _count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._raw = xp.zeros(self.dim, dtype=CDTYPE)

    def add(self, vector, weight: float = 1.0) -> "IncrementalMemory":
        self._raw = self._raw + weight * xp.asarray(vector, dtype=CDTYPE)
        self._count += 1
        return self

    @property
    def vector(self) -> "xp.ndarray":
        """The current, normalised, query-ready hypervector - safe to
        pass straight to `qa.ask`/`qa.chain` like any other memory."""
        return normalize(self._raw)

    def __len__(self) -> int:
        return self._count


@dataclass
class IncrementalShardedMemory:
    """The sharded counterpart to :class:`IncrementalMemory` -
    `Ontology.ground_sharded`'s incremental analogue. New facts are
    appended to the *newest* shard until it reaches `shard_size`, then a
    fresh shard starts automatically; shards that are already full are
    never re-touched by a later `add()`, unlike `ground_sharded`, which
    rebuilds every shard from scratch on every call regardless of how
    much of the knowledge base actually changed.
    """

    dim: int
    shard_size: int = 80
    _shards: list = field(init=False, default_factory=list)
    _current_count: int = field(init=False, default=0)

    def add(self, vector, weight: float = 1.0) -> "IncrementalShardedMemory":
        if not self._shards or self._current_count >= self.shard_size:
            self._shards.append(IncrementalMemory(dim=self.dim))
            self._current_count = 0
        self._shards[-1].add(vector, weight=weight)
        self._current_count += 1
        return self

    @property
    def memories(self) -> list:
        """Query-ready hypervectors for every shard, positionally aligned
        - pass straight to `qa.ask_sharded`/`qa.chain_sharded`."""
        return [shard.vector for shard in self._shards]

    def __len__(self) -> int:
        return sum(len(shard) for shard in self._shards)


try:
    import networkx as nx

    HAS_NETWORKX = True
except Exception:  # pragma: no cover - optional dep
    nx = None
    HAS_NETWORKX = False


@dataclass
class Ontology:
    """A relational knowledge graph grounded in a hypervector codebook."""

    dim: int = 10000
    seed: int = 0
    codebook: Codebook = field(init=False)
    triples: list[tuple[str, str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.codebook = Codebook(dim=self.dim, seed=self.seed)
        self._graph = nx.MultiDiGraph() if HAS_NETWORKX else None
        self._entities: set[str] = set()
        self._relations: set[str] = set()

    # -- construction -------------------------------------------------------
    def add(self, subject: str, relation: str, obj: str) -> "Ontology":
        self.triples.append((subject, relation, obj))
        self._entities.update((subject, obj))
        self._relations.add(relation)
        if self._graph is not None:
            self._graph.add_edge(subject, obj, key=relation, relation=relation)
        return self

    # -- substrate accessors ------------------------------------------------
    def entity(self, name: str):
        """The shared core wave ``ENT:name`` for an entity (minted lazily)."""
        return self.codebook.symbol(f"ENT:{name}")

    def relation_wave(self, name: str):
        return self.codebook.symbol(f"REL:{name}")

    @property
    def _role_subj(self):
        return self.codebook.symbol("ROLE:subj")

    @property
    def _role_obj(self):
        return self.codebook.symbol("ROLE:obj")

    def subj_wave(self, name: str):
        return bind(self._role_subj, self.entity(name))

    def obj_wave(self, name: str):
        return permute(bind(self._role_obj, self.entity(name)), shift=OBJ_SHIFT)

    def entity_names(self) -> list[str]:
        """Entities that actually appear in the KB (the answer candidates)."""
        return sorted(self._entities)

    def refine_entity_vectors(self, rounds: int = 8, alpha: float = 0.7) -> "Ontology":
        """Phase 2: close the generalisation gap `Codebook.symbol`'s
        independently-random entity vectors were found to cause - checked
        directly, not assumed, on real data before being trusted.

        **The gap this answers.** Every entity gets a fresh, unrelated
        random hypervector on first mint - nothing anywhere shapes entity
        representations from data, so two entities with identical
        relational behaviour still get unrelated vectors. That rules out
        inferring an unasserted fact from structural similarity to other
        entities - confirmed as the reason Zeuss performs near chance on
        standard link-prediction benchmarks (see `docs/ROADMAP.md`'s
        Phase 0 entry). This method is a genuinely different mechanism
        from that failure mode's usual fix (gradient-descent-trained
        embeddings, e.g. HolE): no loss function, no optimiser, no
        training loop - just the bind/bundle algebra already used for
        grounding, iterated.

        **The mechanism.** Each entity's vector is repeatedly blended
        toward a bundle of its relational neighbours' *current* vectors
        (each neighbour bound under the relation wave connecting it, the
        same role/relation binding grounding itself uses), synchronously
        across every entity per round - a label-propagation-style power
        iteration, not gradient descent. `alpha` controls how much of an
        entity's own identity survives each round versus its neighbourhood
        signal (`bundle([current, neighbourhood], weights=[alpha, 1 -
        alpha])`); `rounds` controls how many hops of relational context
        can reach an entity. Mutates ``self.codebook`` in place (writes
        each refined vector back under `entity()`'s own lookup key,
        ``f"ENT:{name}"``) - call this *before* any `ground*` method, never
        after (grounding reads whatever the codebook holds at call time).

        **A real bug this was caught by, not by inspection - worth stating
        so it isn't repeated.** An early version wrote refined vectors
        back under the bare entity name instead of `f"ENT:{name}"`,
        silently missing `entity()`'s actual lookup key - every "refined"
        configuration came back numerically identical to every other one
        regardless of `rounds`/`alpha`, which is what actually exposed the
        bug (a genuine effect should vary smoothly with its own
        hyperparameters; identical-regardless-of-input results are a
        structural tell, not evidence of "no effect").

        **Measured, not assumed, at every stage before being trusted:**
        a synthetic "hub and leaf"-style pilot (3 relationally-distinct
        groups, one withheld fact per test entity, inferable only through
        shared-group neighbours) went from 2/9 correct at baseline (near
        the 1/3 chance rate) to 9/9 with refinement, across every
        `rounds`/`alpha` combination tried. Verified on two real, held-out
        knowledge-graph-completion datasets next, using the identical
        filtered-ranking protocol already used to measure Zeuss's
        generalisation *gap* in Phase 0 - not a new, incomparable metric:

        - Nations (14 entities, 55 relations): tail MRR 0.389 -> 0.500
          (`rounds=3, alpha=0.5`) / 0.544 (`rounds=8, alpha=0.2`); Hits@1
          0.229 -> 0.299 / 0.358; Hits@10 0.771 -> 0.905 / 0.945. Head
          direction improved by a similar margin.
        - UMLS (135 entities, 46 relations, no known inverse-relation
          redundancy - checked specifically because Nations has that
          documented quirk and this project doesn't want a result that's
          secretly an artifact of it): tail MRR 0.041 -> 0.651, Hits@1
          0.000 -> 0.600, Hits@10 0.100 -> 0.760 at `rounds=3, alpha=0.5`
          (`n=25`, smaller sample than the `n=40` baseline measurement,
          disclosed rather than glossed over - the effect size here is
          large enough that ordinary sampling variance is not a plausible
          alternative explanation, but the sample-size mismatch is real).

        **A `rounds`/`alpha` sweep, done properly, found a real problem
        with the numbers actually used above - caught before the default
        shipped, not after.** The two real-data runs both used
        `rounds=3, alpha=0.5`-ish settings, checked only for exact-match
        (`Hits@1`) accuracy - never for whether `ask()`'s own honest
        ``known`` confidence flag still fires correctly. A 40-point grid
        on the fast synthetic domain, measuring inference accuracy,
        known-fact top-1 accuracy, *and* the ``known`` rate together,
        found three distinct regimes, not a smooth tradeoff:

        - Low `alpha` (0.1-0.3), any `rounds`: genuinely broken - entity
          identity is washed out fast enough that even a fact an entity
          *does* directly hold stops being recalled correctly
          (`recall_top1` drops to ~0.67), and ``known`` never fires at
          all (0.00) - the over-smoothing failure this method's own
          design was always a candidate for.
        - `alpha=0.5` (this method's *original* default, `rounds>=8`):
          inference and known-fact top-1 accuracy both reach 1.00 - but
          ``known`` *never* fires (0.00), even for facts recovered
          perfectly correctly. The accuracy numbers above were real; the
          honest-confidence signal silently breaking at the same setting
          was not caught until this sweep, because nothing had checked
          it.
        - `alpha=0.7` (`rounds>=5`) or `alpha=0.9` (`rounds>=20`): all
          three metrics - inference, recall, and ``known`` - reach 1.00
          *together*, no tradeoff. `rounds=8, alpha=0.7` (this method's
          actual default) sits inside that safe region with margin.

        **Re-verified on real data next, and the sweep's fix did NOT
        transfer the way it did on the synthetic domain - reported
        honestly, not rounded up.** Checked `known_rate` on both real
        datasets at baseline, the *old* default, and the *new* default:

        - Nations: `known_rate` was never degraded at either default in
          the first place (baseline 0.925/0.915, old 0.940/0.925, new
          0.925/0.915, tail/head) - nothing here for the new default to
          fix.
        - UMLS: `known_rate` genuinely drops with refinement (baseline
          0.917/1.000 -> 0.750/0.833) - but **identically** at the old
          and new default. Switching `alpha` from 0.5 to 0.7 did not
          recover any of it on real data, unlike the clean 0.00 -> 1.00
          jump the synthetic sweep found. The drop is real but far
          smaller than the synthetic domain's total collapse to zero,
          and appears not to be controlled by this hyperparameter choice
          on real data at all - something about refinement's general
          effect at UMLS's scale/density, not something `alpha`/`rounds`
          within the range tested fixes.

        The default is kept at `rounds=8, alpha=0.7` anyway - real-data
        accuracy is a wash-to-slight-improvement over the old default
        (UMLS head MRR 0.440 -> 0.564; tail and Nations roughly tied),
        so there's no reason to prefer the old one - but the specific
        claim that this default *resolves* the honest-confidence
        question is withdrawn. It doesn't, on real data. This is a
        genuine, disclosed case of a synthetic-domain finding not
        transferring, the same category of result this project has
        found and reported honestly before (v0.46's crosstalk fix
        working on synthetic data and failing on real data is the
        closest precedent) - not swept under the rug here either.

        **Honest scope, not yet resolved:** this changes entity
        *initialisation*, not retrieval cost - Phase 1's scaling-wall
        finding (`dimensional_collapse`'s whole-codebook comparison
        drives query latency, not shard count) is unaffected either way.
        Does not touch v0.39's still-open "why doesn't raising `dim`
        rescue the single-bundle ceiling" question. Not yet tested
        against a real benchmark's own published baseline numbers
        (Nations' own citable numbers were found to be withdrawn by
        their original author; UMLS's ConvE-cited numbers - MRR .94 -
        were noted but a rigorous like-for-like comparison, same splits,
        same protocol, hasn't been done) - this closes a real, measured
        gap, not a claim of parity with trained embedding models.
        """
        entities = self.entity_names()
        neighbors: dict = {e: [] for e in entities}
        for subject, relation, obj in self.triples:
            neighbors[subject].append((relation, obj))
            neighbors[obj].append((relation, subject))

        current = {e: self.entity(e) for e in entities}
        for _ in range(rounds):
            nxt = {}
            for e in entities:
                nbrs = neighbors[e]
                if not nbrs:
                    nxt[e] = current[e]
                    continue
                parts = [bind(self.relation_wave(r), current[neighbor]) for r, neighbor in nbrs]
                neighborhood_signal = bundle(parts)
                nxt[e] = bundle([current[e], neighborhood_signal], weights=[alpha, 1 - alpha])
            current = nxt

        for e in entities:
            self.codebook.add(f"ENT:{e}", current[e])
        return self

    # -- grounding ----------------------------------------------------------
    def _triple_vector(self, subject: str, relation: str, obj: str):
        pair = bind(self.subj_wave(subject), self.obj_wave(obj))
        return bind(self.relation_wave(relation), pair)

    def ground(self):
        """Bundle every triple into one continuous memory hypervector."""
        if not self.triples:
            raise ValueError("ontology is empty; add() some triples first")
        return bundle([self._triple_vector(*t) for t in self.triples])

    def incremental_memory(self) -> IncrementalMemory:
        """A fresh, empty :class:`IncrementalMemory` sized for this
        ontology's `dim` - the starting point for building up a memory
        one fact at a time via :meth:`add_incremental`, instead of
        collecting every triple first and calling `ground()` once."""
        return IncrementalMemory(dim=self.dim)

    def incremental_sharded_memory(self, shard_size: int = 80) -> IncrementalShardedMemory:
        """A fresh, empty :class:`IncrementalShardedMemory` - the
        incremental counterpart to :meth:`ground_sharded`. ``shard_size``
        matches `ground_sharded`'s own default (80) - the size v0.39
        measured single-bundle recall to stay reliable at."""
        return IncrementalShardedMemory(dim=self.dim, shard_size=shard_size)

    def add_incremental(
        self, memory: "IncrementalMemory | IncrementalShardedMemory", subject: str, relation: str, obj: str
    ) -> "IncrementalMemory | IncrementalShardedMemory":
        """Add one fact to both ``self.triples``/entities/relations *and*
        an incremental memory structure, in `O(dim)` amortised time -
        unlike `ground()`/`ground_shards()`/`ground_sharded()`, which
        rebuild every shard from scratch regardless of how much of the
        knowledge base actually changed. Accepts either
        :class:`IncrementalMemory` or :class:`IncrementalShardedMemory`
        (both expose the same ``add(vector)`` shape) and returns it back
        for chaining, matching :meth:`add`'s own return-self convention.
        """
        self.add(subject, relation, obj)
        memory.add(self._triple_vector(subject, relation, obj))
        return memory

    def ground_sharded(self, shard_size: int = 80, skip_irregular: bool = True) -> list:
        """Bundle triples into several independent memory hypervectors
        instead of one - trading O(shards) query cost for higher effective
        capacity while keeping each shard at a size recovery stays reliable
        at.

        `docs/ROADMAP.md` v0.39 measured `ground()`'s single-bundle design
        (still used by ``ground()`` above, unchanged) to be reliable up to
        ~80 triples and collapse sharply by 120 - a ceiling that does not
        respond to raising ``dim``. v0.40 found that recovery *does* stay
        reliable per-shard at this size even as *total* KB size grows well
        past the single-bundle ceiling (measured: 90% accuracy at 800
        triples / 10 shards, versus 0% for one 400-triple bundle, with zero
        false positives on genuine unknown queries across every scale
        tested) - see :func:`~zeuss.qa.ask_sharded`/:func:`~zeuss.qa.
        chain_sharded`, which query every shard returned here and keep the
        highest-coherence answer. ``shard_size=80`` defaults to exactly the
        measured reliable point, not a guess.

        v0.41 found that safety is specific to *structurally regular*
        shards (see :func:`is_structurally_regular`); v0.42 answers "how do
        we tell the difference" by checking exactly that before grounding.
        Simple sequential chunking of a contradiction-free ontology's own
        triples is regular by construction, so ``skip_irregular=True`` is
        normally a no-op here - it matters once triples come from a source
        that might contain internal contradictions or duplicate bindings
        (see :meth:`ground_shards` for the general form that takes
        caller-supplied shard boundaries, including externally-sourced
        ones).
        """
        if not self.triples:
            raise ValueError("ontology is empty; add() some triples first")
        shards = [self.triples[i : i + shard_size] for i in range(0, len(self.triples), shard_size)]
        return self.ground_shards(shards, skip_irregular=skip_irregular)

    def ground_shards(self, shards_of_triples: list, skip_irregular: bool = True) -> list:
        """Ground caller-supplied candidate shards (each a list of triples)
        into memory hypervectors - the general form :meth:`ground_sharded`
        chunks its own triples into. Exposed directly for callers who
        already have their own shard boundaries, e.g. mixing in
        externally-sourced candidate shards that might not be trustworthy.

        With ``skip_irregular=True`` (default), any shard failing
        :func:`is_structurally_regular` is silently dropped rather than
        grounded - the concrete "identify garbage and handle it" mechanism
        v0.42 measured to work: filtering this way on the exact adversarial
        mix that broke `ask_sharded`'s guarantee in v0.41 (5 real shards +
        10 random-noise shards) restores it completely (false positives
        back to 0/10, matching the all-real-shards baseline exactly - see
        `test_sharding.py`'s `test_filtering_irregular_shards_restores_
        safety`). Pass ``skip_irregular=False`` to ground everything anyway
        (e.g. for testing, or when the caller has already validated the
        data another way).

        Superseded by :meth:`ground_resolved_shards` (v0.43), which fixes
        this method's two real limitations: it cannot tell a genuine
        multi-valued relation from a contradiction (both look like a
        repeated ``(subject, relation)`` pair), and it drops an entire
        shard on a single bad pair rather than keeping the other good
        facts in it. Kept as-is for callers who don't need that - it's
        simpler and correct for its narrower claim.
        """
        grounded = []
        for shard in shards_of_triples:
            if skip_irregular and not is_structurally_regular(shard):
                continue
            grounded.append(bundle([self._triple_vector(*t) for t in shard]))
        return grounded

    def ground_resolved_shards(
        self,
        shards_of_triples: list,
        multi_valued_relations=None,
        disagreement_threshold: float = 0.3,
        exclusions=None,
        implications=None,
    ) -> list:
        """Resolve conflicts via :func:`resolve_shard_conflicts` (v0.43),
        then ground each surviving shard - the complete, intelligent
        counterpart to :meth:`ground_shards`'s cruder ``skip_irregular``:
        genuine multi-valued relations are never dropped, an isolated
        contradiction only excises the specific conflicting facts, and a
        whole shard is dropped only when most of it disagrees with the
        rest of the dataset. Deliberately does *not* also apply
        :meth:`ground_shards`'s `is_structurally_regular` check - that
        check cannot distinguish a preserved multi-valued relation's
        legitimate duplicate bindings from a real conflict, so applying it
        after resolution would silently re-introduce the exact bug this
        method exists to fix.

        ``exclusions`` (v0.44) / ``implications`` (v0.45): trusted logical
        axioms forwarded straight to :func:`resolve_shard_conflicts` - see
        its docstring and :func:`axiom_violations` for what these buy over
        consensus voting alone (including chaining several mined rules
        into a constraint network via ``implications``), and their own
        honest limits.
        """
        resolved = resolve_shard_conflicts(
            shards_of_triples, multi_valued_relations, disagreement_threshold, exclusions, implications
        )
        return [bundle([self._triple_vector(*t) for t in shard]) for shard in resolved if shard]

    def _ground_shards_excising_violations(self, shards_of_triples: list, violating: set, weights: list) -> tuple[list, list[float]]:
        """Shared plumbing for :meth:`ground_shards_with_trust` and
        :meth:`ground_shards_with_regularity_trust`: ground each shard
        *plainly* (see those methods' docstrings for why baking a weight
        into the bundle itself is a no-op), excising only literal
        structural violations, and return the grounded memories alongside
        whichever caller-supplied per-shard weights survive (a shard that
        loses every triple to excision contributes neither a memory nor a
        weight, keeping the two lists aligned)."""
        memories: list = []
        kept_weights: list[float] = []
        for shard, trust in zip(shards_of_triples, weights):
            kept = [t for t in shard if t not in violating]
            if kept:  # bundle() requires at least one vector - an all-violating shard contributes nothing
                memories.append(bundle([self._triple_vector(*t) for t in kept]))
                kept_weights.append(trust)
        return memories, kept_weights

    def ground_shards_with_trust(
        self,
        shards_of_triples: list,
        exclusions=None,
        implications=None,
        inverse_temperature: float = 8.0,
    ) -> tuple[list, list[float]]:
        """v0.46's *first* attempt at the crosstalk limitation v0.44/v0.45
        disclosed - kept as a documented negative result (see
        :func:`shard_trust_weights`'s docstring for why this specific
        weighting does NOT fix crosstalk; use
        :meth:`ground_shards_with_regularity_trust` instead, which does).

        **Where a continuous trust weight can and can't live at all -
        checked directly, not assumed, and true of both this method and
        its working counterpart.** It does *not* belong in the grounding
        step: weighting every triple in a shard's bundle by that shard's
        own uniform trust score is a no-op, because
        :func:`~zeuss.tier2_substrate.hypervectors.bundle`/``normalize``
        projects every element back onto the unit circle regardless of a
        shared scalar weight - a shard's bundle looks identical whether
        weighted uniformly or not (measured directly: an earlier version
        of this method baked the weight into `bundle`'s own ``weights=``
        and produced results statistically indistinguishable from
        unweighted grounding). Each shard here is therefore grounded
        *plainly* - only literal structural violations
        (:func:`axiom_violations`) are excised, never a whole relation via
        multi-valued classification, never a whole shard via a hard
        disagreement threshold. The returned weights are for
        :func:`~zeuss.qa.ask_sharded`'s ``shard_weights`` parameter
        instead, positionally aligned with the returned memories.
        """
        all_triples = [t for shard in shards_of_triples for t in shard]
        violating = axiom_violations(all_triples, exclusions, implications) if exclusions else frozenset()
        weights = shard_trust_weights(shards_of_triples, exclusions, implications, inverse_temperature)
        return self._ground_shards_excising_violations(shards_of_triples, violating, weights)

    def ground_shards_with_regularity_trust(
        self,
        shards_of_triples: list,
        exclusions=None,
        implications=None,
        inverse_temperature: float = 60.0,
    ) -> tuple[list, list[float]]:
        """v0.46's actual fix for the crosstalk limitation v0.44/v0.45
        disclosed - see :func:`shard_regularity_weights` for the full
        derivation (the "option 1 + 2 combined" answer, after the pure
        continuous-relaxation attempt in :meth:`ground_shards_with_trust`
        was measured not to work). Combines two independent, real signals:
        a literal structural violation (:func:`axiom_violations`, a hard,
        crisp per-triple exclusion, still worth checking even though it
        alone doesn't solve crosstalk) is excised outright regardless of
        shard trust, while every surviving triple is bundled *plainly*
        (see :meth:`ground_shards_with_trust`'s docstring for why baking
        weight into the bundle is a no-op) and paired with its shard's
        continuous structural-regularity weight for
        :func:`~zeuss.qa.ask_sharded`'s ``shard_weights`` parameter to use
        at retrieval time instead.

        Measured on the exact 50/50 real/garbage-shard scenario that broke
        v0.43's consensus mechanism and both v0.44/v0.45's axiom layer on
        its own: false positives restored from 2/10 to 0/10, and answer
        accuracy restored to exactly the noise-free baseline - holding
        from 50% contamination through 94% (see `docs/ROADMAP.md` v0.46).

        **Validated only on synthetic, single-valued, non-redundant-fact-
        free data - measured directly to fail on real, densely multi-
        relational data (see :func:`shard_regularity_weights`'s docstring
        and `docs/ROADMAP.md` v0.47).** Prefer
        :meth:`ground_shards_with_connectivity_trust` for real knowledge
        graphs; this method is kept for the scenario it was actually
        validated on.
        """
        all_triples = [t for shard in shards_of_triples for t in shard]
        violating = axiom_violations(all_triples, exclusions, implications) if exclusions else frozenset()
        weights = shard_regularity_weights(shards_of_triples, inverse_temperature)
        return self._ground_shards_excising_violations(shards_of_triples, violating, weights)

    def ground_shards_with_connectivity_trust(
        self,
        shards_of_triples: list,
        exclusions=None,
        implications=None,
        inverse_temperature: float = 2.0,
    ) -> tuple[list, list[float]]:
        """v0.47: the crosstalk fix that actually works on real, densely
        multi-relational, non-redundant knowledge graphs - see
        :func:`shard_connectivity_weights` for the full derivation and why
        it needed a genuinely different signal, not a re-tuned version of
        `shard_regularity_weights`/`shard_trust_weights` (both measured to
        fail on real data, for a shared, conceptual reason: they need
        repeated/redundant assertions to detect disagreement against, and
        most real facts are stated exactly once).

        Same structure as :meth:`ground_shards_with_regularity_trust` -
        literal `axiom_violations` are still excised outright (an
        independent, still-useful hard signal), every surviving triple is
        bundled *plainly*, and the continuous trust score is returned
        alongside the memories for :func:`~zeuss.qa.ask_sharded`'s
        `shard_weights` parameter to use at retrieval time.

        Measured on the real Nations dataset (1992 triples, 55 genuinely
        multi-valued relations, zero redundant triples) at 50-62%
        contamination across three noise seeds: false positives on
        genuinely-absent facts fell from 37/38 (plain grounding) to 0/38,
        and recall of real stored facts *improved simultaneously* from
        21/40 to 31-37/40 - not a trade-off, both got better together.
        Honest, disclosed limit: the weight distributions still overlap
        at the tails (unlike `shard_regularity_weights`'s clean separation
        on its own synthetic domain) and this has only been validated on
        one real dataset so far - see `docs/ROADMAP.md` v0.47.
        """
        all_triples = [t for shard in shards_of_triples for t in shard]
        violating = axiom_violations(all_triples, exclusions, implications) if exclusions else frozenset()
        weights = shard_connectivity_weights(shards_of_triples, inverse_temperature)
        return self._ground_shards_excising_violations(shards_of_triples, violating, weights)

    # -- the one-hop substrate operator ------------------------------------
    def step(self, memory, ent_wave, relation: str):
        """One deductive hop, entirely in wave space.

        Given a memory hypervector, an entity wave standing in the *subject*
        slot, and a relation, peel the role waves back off ``memory`` and return
        the residue that resonates with the *object* entity of the matching
        fact. Composed with a cleanup, iterating ``step`` walks a relation's
        transitive closure - deduction as a fixed-point of one wave operator.

        The object slot was permuted at grounding time (see module docstring),
        so recovery must invert that permutation before the final
        ``ROLE:obj`` unbind.
        """
        roled = unbind(memory, self.relation_wave(relation))
        roled_obj = unbind(roled, bind(self._role_subj, ent_wave))
        unpermuted = permute(roled_obj, shift=-OBJ_SHIFT)
        return unbind(unpermuted, self._role_obj)

    @property
    def graph(self):
        if not HAS_NETWORKX:
            raise RuntimeError("networkx is not installed; install zeuss[logic]")
        return self._graph
