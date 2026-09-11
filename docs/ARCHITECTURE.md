# Architecture

Zeuss realises the three frontiers of `VISION.md` as three cooperating tiers.
Data flows **down** (symbols → continuous fields) and truth flows **up**
(measurement → discrete decisions).

```
        Tier 3  Logic compiler / symbol grounding
                predicate graphs, axioms  ─┐  fuzzy t-norm energy terms
                                           │
        Tier 2  Substrate math            ▼
                hypervectors ─► collapse ─► energy ─► resonance
                (FHRR/VSA)     (entropy)   (basins)   (interference)
                                           ▲
        Tier 1  Compute kernels            │  phase interference,
                NumPy ref | Triton/CUDA ───┘  topological collapse
```

## Tier 2 — the substrate (the part that runs today)

### Hypervectors (`hypervectors.py`)
Fourier Holographic Reduced Representations: each symbol is a vector of
unit-modulus phasors `exp(iθ)`. The algebra:

- `bind` = elementwise complex multiply (role↔filler), invertible by `unbind`.
- `bundle` = normalised sum (superposition; members stay recoverable).
- `permute` = cyclic roll (quasi-orthogonal relabelling for order/sequence).
- `similarity` = mean cosine of phase difference, in `[-1, 1]`.

`Codebook` mints and cleans up symbols (an associative memory). `encode_record`
binds `(role, filler)` pairs and bundles them into one hypervector.

### Collapse (`collapse.py`) — Frontier 1
A state is described by its softmax occupancy over the codebook. `entropy`
measures ambiguity; raising the inverse temperature β sharpens the distribution
until it snaps to one symbol. `anneal` runs a cooling schedule and records
entropy falling toward a discrete decision. **The decision is the collapse.**
`anneal_adaptive` grows β at a rate set once from the probe's intrinsic
top-2 similarity margin (fixed throughout a schedule, since only β varies) —
an unambiguous probe grows β quickly; a near-tie between two close symbols
grows it slowly, spending more of the schedule resolving the ambiguity.

### Energy (`energy.py`) — Frontier 2
A `Landscape` is weighted attractor hypervectors (the axioms). `settle` relaxes
a state toward the softmax-weighted attractor mean on the phase torus.
`temperature = 0` is a deterministic descent to the ground state; `temperature
> 0` injects thermal phase noise (probabilistic exploration). Satisfying a rule
and minimising energy are one operation. `settle_adaptive` scales its step
size by how much the *previous* step actually reduced energy — shrinking near
plateaus/saddles, growing on open gradients — and stops early on convergence
instead of always running a fixed step count (the "liquid time-step" idea).

### Event-spiking activation gate (`spiking.py`)
`SpikingGate` gates which named attractor groups of a `Landscape` participate
in a settle step: a group activates when a probe's similarity to it crosses a
threshold, and stays active through a short refractory window afterward
(hysteresis, so a probe hovering near the boundary doesn't thrash the gate).
`gated_settle` re-polls every few steps and delegates the actual descent to
`settle` on a filtered sub-`Landscape` of just the active groups — dormant
groups are never scored at all. The concrete content behind "neuromorphic
threshold-gated activation": a sparse compute mask, not a spiking-circuit
simulation.

### Resonance (`resonance.py`) — Frontier 3
`interfere` superposes waveforms *without* per-element renormalisation, so
constructive interference preserves amplitude and destructive interference
cancels it. `coherence` (mean amplitude) is a crispness measure; high coherence
means the deduction "rings true". `phase_lock` is a Kuramoto order parameter.

## Tier 1 — compute kernels
`reference.py` defines the semantics of the two hot ops (`phase_interference`,
`topological_collapse_step`) in NumPy. `triton_kernels.py` is a guarded stub;
GPU implementations must match the reference within tolerance (parity test).

## Tier 3 — logic compiler
`ontology.py` grounds `(subject, relation, object)` triples into bound-and-
bundled hypervectors (NetworkX for the graph, with a fallback). Each entity is
bound under a `ROLE:subj` or `ROLE:obj` role wave depending on slot, and the
object slot is additionally passed through a fixed cyclic permutation — plain
role binding alone is commutative, so an entity that is one fact's object and
the next fact's subject (any transitive chain) would otherwise produce an
*exact* algebraic tie between its true successor and its predecessor; the
permutation breaks that collision. `qa.py`'s `ask`/`chain`/`entails` iterate
`Ontology.step` (one deductive hop) to answer single- and multi-hop queries,
each carrying a coherence that honestly separates known facts from guesses.

As of v0.34-v0.36, `qa.py`'s internal `_cleanup` is where the diagram above
stops being aspirational and becomes one real pipeline for every query:
`dimensional_collapse` (Frontier 1) restricts the candidate entities to the
live basis, that basis becomes a `Landscape` and `settle` (Frontier 2)
relaxes the residue toward it, and `phase_lock` (Frontier 3) reads out both
the settled state (to pick the winning candidate) and the original residue
(to report an honest coherence - settling is a self-reinforcing attractor
network by construction, so it cannot be trusted for "is this a guess";
see `_cleanup`'s docstring for the failure mode this avoids). `chain`
inherits this for every hop for free.

v0.37 adds `axiom_bias`, an optional `candidate -> energy penalty` callable
on `ask`/`chain`: it biases that same `Landscape`'s weights, letting a
`compiler.py` `Rule`/`Theory` - built from *other* `ask()` calls about the
same subject - veto a candidate that contradicts an already-established
fact. This is a hook plus one measured example (a data-contradiction case
where it fixes 4/10 wrong seeds, regresses 0/6 correct ones), not a general
Ontology<->Theory bridge - the caller still writes the `Rule` by hand.

v0.38 (`tier3_logic/axiom_mining.py`) removes that hand-writing: plain
support/confidence association-rule mining over `Ontology.triples` finds
`Rule`-shaped exclusion patterns automatically, and `axiom_bias_from_
ontology` turns them straight into a v0.37 `axiom_bias` - "point it at an
Ontology, get a biased query out" with no relation names or Rules
hand-typed anywhere. Measured on a KB where plain resonance is
systematically wrong (15/15 seeds): mining fixes 12/15, regresses 0/90
clean checks, with the 3 holdouts diagnosed (not mysterious) as cases where
raw crosstalk skew is unusually large relative to the mined bias's fixed
strength - a real, acknowledged limit of this specific mechanism, not a bug.

**v0.39 found a real ceiling in `Ontology.ground()` itself - not in any of
v0.34-v0.38's work, but in the pre-existing single-bundle design underneath
all of it.** Every triple gets bundled into one memory hypervector; measured
directly (robust across 5 seeds), recovering a directly-stored, unambiguous
fact via plain `ask()` stays perfectly reliable at 80 bundled triples and
collapses sharply by 120 - a threshold every prior test/demo in this project
happened to sit just under (the largest prior test ontology used 40).
Surprisingly, raising `dim` 8x (8192 -> 65536) did not meaningfully rescue
a 400-triple case - checked directly, not assumed - so this isn't the plain
"more dimensions fixes capacity" story `VISION.md`'s own framing would
predict. Why not is an open question, not yet investigated (see `docs/
ROADMAP.md` v0.39): candidates include the `OBJ_SHIFT` permutation scheme
or how `bundle`'s normalisation behaves with many summed terms. Locked into
`test_capacity_ceiling.py` so the numbers don't silently drift.

**v0.40 works around that ceiling rather than explaining it.**
`Ontology.ground_sharded(shard_size=80)` bundles triples into several
independent memory hypervectors instead of one giant bundle (`ground()`
itself is unchanged); `qa.ask_sharded`/`qa.chain_sharded` query every shard
and keep whichever resonates loudest. Measured, not assumed: 9/10 correct
at 400 triples (vs. 0/10 for one bundle), 18/20 at 800 triples/10 shards,
and - the real risk with a max-over-N-shards selection - zero false
positives on genuine unknown queries at either scale, so guess-detection
isn't quietly traded away for the accuracy gain.

**v0.41 found that guarantee is specific to shards carrying real data.**
Stress-testing sharding itself (mixing in shards of unstructured random
triples instead of more real ones) broke the zero-false-positive result
fast: just 10 random-noise shards alongside 5 real ones pushed false
positives from 0/10 to 4-6/10 - checked directly to rule out a coincidence-
based confound. `ground_sharded`/`ask_sharded`/`chain_sharded` are
validated for splitting a KB's own real data across shards, not for a
setting where shard content might be noisy or adversarial.

**v0.42/v0.43 answer "how do we tell the difference", the second version
properly.** v0.42's `is_structurally_regular` (duplicate `(subject,
relation)` detection) fixed the immediate problem but couldn't tell a
genuine multi-valued relation (`has_friend`-like) from a contradiction, and
dropped whole shards on one bad pair. v0.43's `resolve_shard_conflicts`
fixes both: `classify_multi_valued_relations` infers which relations are
legitimately multi-valued from the data itself, and every other fact is
checked against a *cross-shard consensus* (the majority filler asserted
for that pair across every shard) rather than the shard's own internal
statistics - the mechanism that turned out to be the scale-invariant one,
after two within-shard statistics (collision rate, then degree variance)
were measured and rejected for degrading as vocabulary grows. Genuine ties
are left unresolved, not silently guessed. Honest boundary: this needs
garbage to be a minority of the data - measured robust through ~30%
contamination, failing open at a 50/50 split (the same "need an honest
majority" limit consensus systems generally have). `sheaf.py`'s
philosophy (independent sources should agree) inspired this, but its own
machinery (scalar `Theory` variables, not triples) wasn't reused directly.

**v0.44 answers v0.43's own disclosed ceiling with a genuinely different
kind of signal, not a bigger version of the same one.** No majority-vote
scheme can be made reliable against an adversarial *majority* of bad data
- `axiom_violations` sidesteps that wall instead of trying to push past it:
a fact that's structurally impossible given another fact the same subject
holds gets discarded *before* `_global_filler_consensus` even tallies a
vote, so a garbage majority can't out-vote a true minority fact on a pair
a trusted `DiscoveredExclusion` axiom covers. This is only as trustworthy
as the exclusions it's given - mining them from the same contaminated pool
just re-derives the vote-counting problem one level up; the real power
comes from sourcing them independently (hand-written, or from a separately
trusted seed sample). A second, distinct limit was found only by measuring
the full pipeline at v0.43's own garbage-majority boundary: once a relation
is already exempted from consensus entirely (v0.43's fail-open case),
hundreds of unrelated noise triples flood in unfiltered too, and most wrong
answers are hypervector crosstalk from that flood, not the one literal
contradiction `axiom_violations` actually excises - a problem this
mechanism was never designed to solve. (Confirming v0.43's own test suite
for the first time, while building this, also surfaced two real bugs in
it - a missing subject-count floor on `classify_multi_valued_relations`,
and a shard-dropping branch that broke positional list alignment - both
fixed alongside two test assertions that had been written before ever
being run.)

**v0.45 builds the genuine constraint *network* v0.44 was asked for next:
chaining mined implications, not just checking exclusions pairwise.**
`_implied_closure` forward-chains a subject's held facts through a directed
graph of mined `DiscoveredImplication` edges before `axiom_violations`
checks exclusions against the result - catching a violation that's only
reachable by composing two or more separately-mined rules, invisible to
v0.44's flat one-hop check (verified: a mined two-hop chain plus an
exclusion catches a contradiction the flat check misses on the identical
triples). Deliberately not built on `sheaf.py`'s H0/H1 cohomology again,
for the same reason as before: its restriction edges assert equality
between scalar stalks, and neither implication (directional) nor exclusion
("not both") is an equality constraint - forward-chaining graph reachability
is the correctly-scoped tool for directed rules, not a relabelling of
cohomology.

**v0.46 answers v0.44's second disclosed limitation (crosstalk) - after
trying the more elegant fix first and measuring it fail.** The natural
"dissolve the gate" idea - turn `classify_multi_valued_relations` into a
continuous per-shard trust weight instead of a hard exemption
(`shard_trust_weights`) - was implemented and measured directly on the
scenario that motivated it: every shard, real and noise alike, scored
identically. Root cause: at heavy contamination with a small shared entity
pool, most real-vs-noise collisions land as an exact tie, which is
deliberately treated as neutral so genuine multi-valued plurality isn't
punished - making a real+noise tie and two genuine facts indistinguishable
from one pair's vote count alone. The fix that actually works
(`internal_collision_energy`/`shard_regularity_weights`) sidesteps the
ambiguity instead of resolving it: a signal purely internal to one shard
(does it duplicate one of its own `(subject, relation)` pairs?), immune to
both contamination volume and cross-shard tie ambiguity because it never
looks past one shard's own boundary. Also discovered by direct
measurement: the weight can't live inside `bundle()` at all (uniform
per-shard weighting is a provable no-op after `normalize`'s per-element
phase projection) - it has to scale `qa.ask_sharded`'s cross-shard
comparison instead (`shard_weights`), damping a noise shard's occasional
lucky resonance at retrieval time. Measured result: 0/10 false positives
(from 2/10) and accuracy matching the noise-free baseline exactly, holding
from 50% contamination through 94% - a qualitatively higher ceiling than
v0.43's consensus mechanism, which failed open already at 50%.

**v0.47: v0.46 was validated only on synthetic, single-valued, non-
redundant-fact-free data - and measured to fail on real data, for a
conceptual reason, not a bad constant.** Tested directly against the real
Nations dataset (1992 triples, 55 genuinely multi-valued relations, zero
duplicate triples): `shard_regularity_weights` showed real and noise
shards' collision-energy ranges fully overlapping (zero discrimination),
and the earlier-rejected `shard_trust_weights` was tested on real data too
rather than assumed to fail the same way - it *inverted*, scoring real
shards as more suspicious than noise. Root cause common to both: they
detect a shard *disagreeing* with something, which needs redundant,
independently-repeated assertions of the same fact to work at all - true
of the synthetic domain by construction, false of Nations and most real
knowledge graphs, where a fact is normally stated exactly once. The fix
that actually works, `shard_connectivity_weights`, needs no redundancy:
it asks whether a shard's *pattern of which entities it talks about* looks
like real-world structure (skewed - some entities are simply more
connected than others) or uniform random sampling, self-normalised via
each shard's z-score against the mix's own mean rather than a fixed
constant (a raw connectivity score has no portable absolute scale across
datasets the way a `[0,1]` fraction does). Measured across three noise
seeds and two contamination levels on real data: false positives fell
from 37/38 to 0/38, and recall of real facts *improved simultaneously*
from 21/40 to 31-37/40 - not a trade-off. Honest limit: the weight
distributions still overlap at the tails, unlike v0.46's clean separation
on its own synthetic domain, and this is validated on one real dataset so
far.

**Phase 1 closes with incremental grounding: `IncrementalMemory`/
`IncrementalShardedMemory`.** Every `ground*` method rebuilt its whole
bundle from scratch on every call - Zeuss could only be used as a static,
batch-compiled snapshot, never a living memory absorbing new facts over
time. The reason a batch bundle couldn't just be appended to: `bundle()`'s
final `normalize()` is *per-element* phase projection, not one global
magnitude, so a new fact can't be folded into an already-normalised
bundle - but the raw, pre-normalisation complex sum can be, trivially.
These classes keep that raw sum as their actual state and normalise only
on read, making `add()` `O(dim)` regardless of how many facts already
exist. Verified numerically equivalent to the batch path at both the
single-memory and sharded level (similarity > 0.9999999 against
`bundle()`/`ground_sharded()`'s own output), and real-API tested through
`ask_sharded` at the same scale `ground_sharded` itself was originally
validated at.

**Phase 2: `Ontology.refine_entity_vectors` closes the generalisation gap
Phase 0 identified - the first capability this substrate did not
previously have.** `Codebook.symbol` mints an independently random vector
per entity; nothing shaped representations from data, so two entities
with identical relational behaviour got unrelated vectors, ruling out
inferring an unasserted fact from structural similarity. Given three
routes (gradient-trained embeddings; hypervector-native iterative
neighbour blending; a separate classical layer), the harder, more in-
character, no-gradient option was chosen deliberately. Each entity's
vector is repeatedly blended toward a bundle of its relational neighbours'
current vectors, bound under the connecting relation wave (the same
binding grounding itself uses) - a label-propagation-style power
iteration, no loss function or training loop anywhere. A synthetic pilot
first looked like a clean failure (0/9 correct everywhere) until a tell -
every configuration was numerically identical regardless of its own
hyperparameters - exposed a codebook-key bug silently discarding every
refined vector; fixed, it reached 9/9 (from 2/9 baseline). Verified on two
real datasets specifically to rule out a single-dataset artifact: Nations
(tail MRR 0.389 -> 0.500-0.544) and, because Nations has a documented
inverse-relation-redundancy quirk its own original benchmarking author
withdrew it for, UMLS as well (no such quirk) - which confirmed it far
more dramatically (tail MRR 0.041 -> 0.651, a 16x improvement, Hits@1
0.000 -> 0.600). A follow-up 40-point `rounds`/`alpha` sweep then found a
real problem with the very default those real-data numbers used: accuracy
was perfect, but `ask()`'s own honest `known` confidence flag silently
never fired - invisible because neither real-data run had checked it, only
exact-match accuracy. The sweep found this wasn't a smooth tradeoff but
three distinct regimes (low `alpha` breaks even known-fact recall;
`alpha=0.5` gets accuracy right but `known` never fires; `alpha=0.7+`
with enough rounds gets all three - inference, recall, confidence -
simultaneously), and the shipped default was corrected to `rounds=8,
alpha=0.7` before it was relied on further, with a dedicated regression
test for the exact failure mode found. **Re-verified on real data next,
and the fix did not transfer the way it did on the synthetic domain -
reported plainly, not rounded up.** Nations never showed the degradation
at either default (`known_rate` 0.925-0.940 throughout). UMLS did show a
real drop with refinement (0.917/1.000 baseline -> 0.750/0.833) - but
*identically* at the old and new default; switching `alpha` recovered
none of it on real data. The default is kept at `rounds=8, alpha=0.7`
regardless (real-data accuracy is a wash-to-slight-improvement over the
old default), but the specific claim that it resolves the honest-
confidence question is withdrawn - it doesn't, on real data. The same
category of result as v0.46's crosstalk fix working on synthetic data and
failing on real data: a synthetic finding that didn't transfer, disclosed
rather than hidden. Also still not compared against a real trained
embedding model's own published numbers under an identical protocol -
this closes a real, directly-measured gap in Zeuss's own before/after
performance, not a claim of parity with trained models.

**Tracing the UMLS drop to its actual mechanism found the real cause, and
it changed the design.** Two plausible theories were tested and both were
wrong: graph-density-driven over-smoothing predicted the sparse synthetic
pilot (mean degree ~10) and dense UMLS (mean degree ~155) should diverge
sharply in how fast entity vectors converge toward each other - measured
round-by-round, their convergence trajectories were nearly identical
(0.085 vs 0.096 by round 8), and forcing the synthetic pilot through the
same sharded evaluation path UMLS uses still showed zero degradation.
`_cleanup`'s `dimensional_collapse` candidate-restriction step was the
second candidate - ruled out the same way: refinement *shrank* the live
candidate set on real UMLS queries (127.7 -> 85.7 of 135 entities), the
opposite of what a "bigger ambiguous set" theory needed, with settle-vs-
raw disagreement staying negligible either way. The actual mechanism:
`bind`/`unbind` recovery assumes atomic vectors are close to independent;
this method's entire purpose is to correlate related entities, which is
directly in tension with that assumption, and the aggregate crosstalk
this produces scales with vocabulary size (entities x relations sharing
the space) rather than per-pair correlation strength alone - exactly why
the tiny synthetic domain and UMLS's much larger one showed similar
per-pair convergence while only the larger vocabulary degraded retrieval.
Not a mistunable hyperparameter, which is why the sweep above couldn't
fix it on real data - a structural tension between what this method needs
to do (correlate related entities) and what the substrate's core algebra
needs to stay reliable (keep them independent). **The fix separates the
two uses instead of trading one off against the other**: refined vectors
now live in a separate `entity_refined()` namespace that `ground()`/
`ground_sharded()`/ordinary `ask()`/`ask_sharded()` never read, so the
memory and every ordinary query are provably unaffected by refinement -
`entity()` is untouched by construction, not just measured to hold on the
samples checked. Generalisation becomes opt-in per query: `ask`/
`ask_sharded` gained a `subject_vector` parameter (`None` is an exact
no-op), so a caller trades fidelity for reach only for the one query that
wants it, against a memory that's otherwise still built entirely from
pristine vectors.

`compiler.py` provides t-norms, residuated implications, weighted `Rule`s and
a `Theory` whose total penalty is a continuous energy over `[0,1]` valuations.
`grounding.py` closes the loop: `compile_theory` represents each propositional
variable as a `[0,1]`-weighted bundle of two poles (`TRUE`/`FALSE`), enumerates
the theory's Boolean corners, and registers every low-energy corner as a
`Landscape` attractor weighted by `exp(-energy)` — settling from any start
state relaxes toward the theory's satisfying valuations, and `readout` reads
the settled truth values back out. Scoped to `O(2**n)` corner enumeration,
matching `Theory`'s expected small-`n` scale. `compile_theories` wires the
sheaf audit (below) directly into this compile step: it runs
`sheaf.from_theories` over several named agents' theories first, raising
`InconsistentTheoriesError` (with the specific violated edges) if any two
disagree on a shared variable, instead of silently merging the contradiction
into an unexplained fuzzy `Landscape`; on success it compiles the union of
every agent's rules via `compile_theory`.

### Sheaf cohomology auditor (`sheaf.py`)
A graph-level (1-skeleton) cellular sheaf: named scalar-stalk vertices, edges
asserting two vertices' restricted images must agree, `H^0`/`H^1` via
rank-nullity on the coboundary matrix (`np.linalg.matrix_rank` - real linear
algebra, not a heuristic). **Honesty note:** for this homogeneous
construction, a frustrated (sign-flipped) cycle is *full rank* (`H^1 == 0`)
and instead collapses `H^0` to `{0}` (only the trivial section survives) -
the opposite of the naive "`H^1 != 0` means contradiction" intuition the
ADAMAI spec's prose suggests. `h0_dimension() == 0` is the right *structural*
question ("is this topology so over-constrained only the zero assignment
works"); whether *specific* observed data actually agrees is answered
directly by `local_section`/`is_consistent_with` (a residual check, not a
cohomological one). `from_theories` builds a graph from several agents'
`Theory` objects, each agent's own best (lowest-energy) local valuation found
in isolation, connected pairwise on shared variables - catching disagreement
between individually-satisfied theories that neither one's own
`Theory.satisfied()` could see.

## Drive loop (active inference)
`drive.py` scores candidate `Action`s by `EFE = pragmatic_weight *
pragmatic_value - epistemic_weight * epistemic_value`: `pragmatic_value`
reuses `Landscape.energy` on a one-step hypothetical blend toward the
action's effect (predicted goal progress, lower is better); `epistemic_value`
reuses `collapse.entropy`/`occupancy` (predicted uncertainty reduction,
higher is better). `select_action` picks the minimum-EFE action, restricting
to actions flagged `is_discovery` when `missing_params` is set - a literal,
testable version of "autonomously execute low-risk discovery actions when
parameters are missing," not a general active-inference generative-model
agent (no beliefs, no learned world model, no perception-action loop).

## Tier 4 — program synthesis (`tier4_synthesis/`)
A different "Tier 4" numbering than this repo's internal tier1-3 layout -
it follows `docs/ADAMAI_SPEC.md`'s own scheme, kept separate to avoid
confusion. `dsl.py` is a closed, *total* expression language: arithmetic
(`+ - * // %`), comparisons (`== != < <= > >=`), booleans, lists, `If`/`Let`,
structural bounded loops (`Fold`, `Map`, `Filter`, all bounded by the length
of the list they operate over) plus `Length`/`Index`, and fuel-limited
`Letrec`/`Recur` for bounded recursion - every construct that could loop
forever carries an explicit finite bound, so `evaluate` always terminates
without a wall-clock timeout. `encode.py` structurally encodes a program tree
into one hypervector via the same recursive bind/bundle pattern `ontology.py`
uses for triples. `search.py` runs a mutation/crossover genetic search:
candidates are scored by real execution against I/O examples
(`program_energy`), parents are chosen by fitness-proportionate sampling that
reuses `collapse.softmax` directly over negative energies, and selection
pressure rises across generations (the "liquid time-step" idea spread across
a generational search instead of one `settle` call). Random generation uses
weighted (not uniform) construct selection - with plain uniform weighting,
each new construct added to the grammar silently diluted how often the
already-useful ones (`binop`, `fold`) got picked, making previously-easy
targets harder to find purely from grammar growth. `synth.py` is the public
entry point.

**Honesty statement:** "verified" means the winning candidate is re-checked
against every given (and, in tests, held-out) example - not a formal proof of
correctness for all inputs. This is the scoped, concretely testable version
of `ADAMAI_SPEC.md` Part 2 Tier 4's "zero-error", "provably correct" program
synthesis language, which is not an achievable target for general programs.

**Recursion synthesis: opt-in, tested, and - with resonance-guided search
plus a second template shape - reliably yes, for the target class this
project actually needs.**
`Letrec`/`Recur` generation and mutation are scope-tracking (`search.py`
threads a `recur_ctx` of in-scope function names/arities through generation,
and mutation regenerates replacements using the scope actually valid at that
tree position, not the top-level scope) and safe. `allow_recursion=False` is
the default in `random_program`/`mutate`/`synthesize` - leaving `letrec`/
`recur` unconditionally in the generation grammar was tried and *measured* to
make every search meaningfully slower per candidate (a recursive candidate
costs more to evaluate than a shallow one even when perfectly safe),
regardless of whether the target needed recursion at all. Setting
`allow_recursion=True` re-enables them, plus `template_rate`-biased seeding
toward a "decrement-and-combine" skeleton (`_recursive_template`) instead of
hoping blind growth stumbles onto a working recursive shape (empirically,
well under 5% of random depth-4 trees even contain a `Letrec` with an
`If`-shaped body).

Two further robustness issues were found and fixed by testing against real
generated candidates, not assumed away by the existing safety net: (1) a
real `RecursionError` (Python's own interpreter stack limit, distinct from
the fuel counter) is now converted to the documented `FuelExhausted` inside
`evaluate` itself, regardless of tree shape or fuel budget; (2) an unbounded
*value magnitude* - a candidate whose recursive argument grows instead of
shrinking (e.g. squaring) reaches numbers with millions of bits well within
the fuel budget's call-count limit, making bignum arithmetic the actual
runaway cost, not recursion depth - is now bounded by `_MAX_MAGNITUDE` in
`dsl.py`, raising `ValueOverflow` instead of hanging. Genetic bloat (mean
tree size growing unboundedly generation over generation - confirmed
empirically, roughly 6x over 15 generations with no correction) is
controlled with parsimony pressure in `synthesize`'s selection step.

Despite all of the above, blind mutation/crossover alone still does **not**
reliably *discover* a correct solution for a target that genuinely requires
recursion (`2**n`, which has no shortcut in this arithmetic-only grammar)
within a practical budget - confirmed by running it repeatedly after every
fix, not assumed.

**Resonance-guided template evolution changes this - the full story, in
three steps, each driven by checking a real failure, not by guessing.**

1. `resonance_bias.py`'s `ResonantBias` is an Estimation-of-Distribution
   prior over the template's hole-fillers - implemented with the same
   hypervector primitives every other tier uses (`Codebook`, weighted
   superposition, similarity-based readout via `collapse.softmax`) instead
   of a bolted-on probability table. Every generation, any population member
   that structurally matches a template shape (`extract_template_choices` -
   matched by observed structure, not provenance, so it also learns from
   candidates mutation/crossover evolved into that shape) reinforces the
   bias in proportion to `exp(-energy)`. First result: 2 of 9 seeds found a
   genuinely correct, held-out-generalizing `2**n` at a budget that found it
   0 times without the bias.
2. Checking *why* the other 7 failed found that the actual winning solution
   (`f(n) = 1 if n<1 else f(n-1)+f(n-1)`) doesn't match the only template
   shape that existed (`p OP f(p-step)`) at all - it needs **two** recursive
   calls combined together, not one combined with the parameter. Added a
   second shape, `f(p-step) OP f(p-step)` (`combine_kind`), directly
   generatable and reinforceable instead of relying on mutation to build it
   from nothing.
3. That alone made things *worse* (0 of 6 seeds) until checking why revealed
   a second real issue: the two-recursive-call shape's call tree grows
   exponentially, so it exhausts a small fuel budget (and gets scored as a
   near-total failure) far more easily than the one-recursive-call shape
   does for equally bad hole-fillers - biasing the resonance memory toward
   the wrong shape before either got a fair trial. Fixed two ways: made
   `combine_kind` itself immune to resonance bias (always drawn uniformly,
   so both shapes always get an even chance; only the *fine-tuning within*
   a chosen shape is biased), and exposed `fuel_budget` as a tunable on
   `synthesize` (raising it to 200 gives the exponential shape's call tree
   room to actually finish evaluating instead of being cut off mid-attempt).

With all three changes - `population_size=800, max_generations=150,
fuel_budget=200, allow_recursion=True, resonant_bias=True` - **every one of
9 seeds tried found a genuinely correct, held-out-generalizing `2**n`**
(checked for n up to 9, none of which were training examples). This is a
measured result from a single fixed configuration re-run across all 9 seeds,
not a cherry-picked lucky one. `Letrec`/`Recur` remain fully
interpreter-supported and tested directly on hand-built programs (factorial,
etc.) independent of what the search can find.

**Asymmetric recursion (`delta`): a real tradeoff, resolved with a second
liquid-time-step, not a free lunch.** `double_recur` originally used a single
shared step on both recursive calls (symmetric doubling only) - Fibonacci
(`f(n-1)+f(n-2)`) needs two *different* steps and wasn't solvable by that
shape at all. The natural fix is a `delta` hole (`f(p-step)` and
`f(p-step-delta)`, `delta=0` recovering the symmetric case), but adding it
uniformly (50/50) measurably slowed `2**n` discovery on some seeds (seed 4:
0.4s -> 72.1s) - an unbiased extra binary split roughly halves the effective
population correctly matching `delta=0`, and (as `ResonantBias`'s docstring
already warns) `delta` must *not* be resonance-learned like the other holes,
since a structural/family choice can converge to the wrong family from early
noise. A *static* skewed prior (`p=[0.85, 0.15]`) was tried next and only
partially helped: it let seed 4 eventually converge, but at ~500x the
unbiased cost (219s / 104 generations vs. 0.4s), and skewing it further to
fix that seed broke Fibonacci discovery outright (a seed that found it in
3.3s started timing out) - a genuine two-sided tradeoff no single fixed ratio
resolves.

The actual fix, mirroring `collapse.anneal_adaptive`'s "adapt the step to how
hard progress currently is" idea: `delta_p1` (the probability of drawing the
asymmetric branch) anneals on *stagnation* - generations since `best_energy`
last improved - staying at `delta_p1_start` (0.15) while the search is still
making progress, growing geometrically (`delta_p1_stagnation_growth=1.08`
per stagnant generation, capped at `delta_p1_max=0.5`) once it stalls, and
resetting the moment progress resumes. This only makes symmetric targets pay
the asymmetric-search cost on the runs that actually get stuck, rather than
on every run. Measured result at the same `population_size=800,
max_generations=150, fuel_budget=200` configuration: `2**n` still verifies on
seeds 0, 4, and 7 (seed 4 remains the hard case at ~178s - slower than the
pre-`delta` 0.4s, a real and documented cost of supporting asymmetric shapes
at all, but no longer ~500x worse or seed-dependent-unsolvable). Fibonacci
(shifted to `F(1)=F(2)=1` so the base case fits the template's fixed
`Const(base_val)` - real Fibonacci's `F(0)=0` does not, a deliberate,
documented scope limit) verifies and genuinely generalizes to held-out n on
seeds 0 and 1. It is **not** reliable across seeds the way `2**n` is: seed 4
fails to verify within this budget, and seed 7 "verifies" (matches all six
training examples) with a degenerate, coincidental arithmetic expression
that does *not* generalize - caught only by checking held-out points, exactly
the failure mode this module's "verified is not proof" honesty statement
exists for. See `tests/test_synthesis.py`'s
`test_resonant_bias_can_discover_fibonacci` for the exact, checked claim.

**Zero-indexed Fibonacci (`base_kind`): one clean win, one honest non-win.**
Two follow-ups were identified after `delta` landed. The first: the
template's base case was always `Const(base_val)` - a fixed number, which
can never equal the varying parameter, so real Fibonacci (`F(0)=0`) wasn't
expressible at all, only the reindexed `F(1)=F(2)=1` workaround was. Added
`base_kind` (`"const"` keeps the old `Const(base_val)` behaviour, `"param"`
uses `Var(param)` itself as the base case) as a third structural hole, drawn
uniformly and never resonance-biased - the same reasoning as `combine_kind`
and `delta`. Verified via 2000 round-trip draws (0 mismatches) and a
hand-built `if p<=1 then p else f(p-1)+f(p-2)` that evaluates correctly and
extracts to exactly the expected hole-fillers
(`test_extract_template_choices_handles_param_base_case`). Real zero-indexed
Fibonacci is then genuinely discoverable and held-out-generalizing on
multiple seeds (`test_resonant_bias_can_discover_zero_indexed_fibonacci`),
and re-checking `2**n` on seeds 0/4/7 confirmed the extra structural coin
flip doesn't regress its reliability (all three still verify and generalize,
just with different per-seed runtimes than before - expected chaotic
sensitivity in a GP search, not a regression).

The second follow-up - closing the `delta_p1`-schedule gap so Fibonacci
(seeds 4 and 7) is as reliable as `2**n` - was tried and genuinely does
**not** have a clean fix, a result worth recording rather than papering over.
Adding one more training example (`n=7`, 8 examples instead of 7 for the
shifted-indexing target) *did* fix both: seed 4 went from not-verified to
verified-and-generalizing, and seed 7 stopped overfitting. But the same
change broke seed 1, which had generalized fine on the original 7 examples.
This is a genuine whack-a-mole, not a fixable-with-more-data problem the way
the `list_sum` target's overfitting was (see `test_synthesize_recovers_
list_sum_via_fold`'s comment) - no single fixed example count was found that
gets every seed tried to generalize. Left as an open, honestly-scoped
good-first-task rather than closed with a claim the evidence doesn't support.

## GA-HDC (experimental) - `geometric.py`
A small Clifford algebra Cl(n,0), `n <= 6` (up to 64 blade coefficients),
**additive** alongside the existing D-dimensional complex-phasor
hypervectors, not a replacement - a literal Clifford algebra at D>10,000
(as `docs/ADAMAI_SPEC.md` asks for) would need `2**10000` blade components,
computationally nonsensical. `geometric_product` uses a precomputed
sign/index table over blade bitmasks, verified against known Cl(2,0)
identities (`e12*e12 == -1`). `rotor`/`apply_rotor` give relation-as-rotation
(a sandwich product) for a *simple* bivector, generalizing `bind`'s phase
multiplication. The genuine bridge back to `hypervectors.py`: Cl(2,0)'s even
subalgebra (scalar + pseudoscalar) is isomorphic to the complex numbers
already used there (`test_geometric.py` checks this directly against
Python's built-in `complex` arithmetic) - a proper generalisation, not an
unrelated bolt-on. Status: exploratory research spike; no other module
depends on it.

## Backends
`backend.py` exposes `xp` (NumPy or JAX), `HAS_JAX`, and complex dtypes. Tier-2
substrate math is now routed through `xp` end-to-end, so the same code runs on
NumPy (default) or JAX (`ZEUSS_BACKEND=jax`). Randomness is still drawn from an
explicit NumPy `Generator` and *lifted* onto the backend, so a given seed yields
identical hypervectors on either backend. `test_backend.py` pins the routing and
carries a NumPy↔JAX parity check that activates once JAX is installed.

**`energy.settle_grad`: literal gradient descent, not just the mean-field
fixed point.** `settle`/`settle_adaptive` move `z` toward
`landscape.target(z)`, a hand-derived mean-field update. `settle_grad`
instead reparameterises the state as real phase angles `theta`
(`z = exp(i*theta)`) and takes a real `jax.grad` step on the energy itself -
ordinary real-to-real autodiff applies directly to `theta`, and
`z = exp(i*theta)` is exactly unit-modulus by construction, needing no
`normalize()` projection after each step. Two real issues were found and
fixed while building this, not assumed away: (1) `Landscape.energy` and
`similarity` both return plain Python `float`s via an explicit cast - correct
for their normal call sites everywhere else, but an unconditional trace abort
the moment `jax.grad` reaches them - so `settle_grad` carries a small,
self-contained restatement of `Landscape.energy`'s exact formula in terms of
`theta`, checked numerically identical to it
(`test_settle_grad_matches_landscape_energy_formula`), rather than routing
through those functions; (2) the raw gradient is tiny (measured: norm ~0.008
over `D=8192`, ~1e-4 per component) because the energy divides by `D` twice
(once in `similarity`'s own normalisation, once in the softmax-weighted sum),
so an unscaled learning rate would need to be in the thousands to move at
all - `settle_grad` scales the step by `D` internally so `learning_rate`
behaves on a `settle`-`step_size`-like scale regardless of dimension
(confirmed: `learning_rate=0.5` reaches essentially the same ground-state
energy as `settle`'s default in the same 60 steps, from the same noisy
start). Requires the JAX backend; raises `RuntimeError` (not a confusing
`AttributeError` on a missing `jax.grad`) otherwise.

Discovered while building this: the project's own `.venv` already has JAX
installed and picks it as the *default* active backend (`ZEUSS_BACKEND`
defaults to `"auto"`, which prefers JAX when importable) - so the full test
suite was re-run end-to-end with JAX genuinely active by default (not just
the forced-subprocess parity check), confirming v0.2's "same code runs on
either backend" claim holds for real, not just in the one test designed to
check it.
