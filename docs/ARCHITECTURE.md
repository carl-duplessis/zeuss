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

**Recursion synthesis: opt-in, tested, honest result: still no.**
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

Despite all of the above, blind mutation/crossover still does **not**
reliably *discover* a correct solution for a target that genuinely requires
recursion (`2**n`, which has no shortcut in this arithmetic-only grammar)
within a practical budget - confirmed by running it repeatedly after every
fix, not assumed. `Letrec`/`Recur` remain fully interpreter-supported and
tested directly on hand-built programs (factorial, etc.); the search can
generate and safely evaluate them, but should not be described as reliably
finding new recursive solutions from scratch.

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
carries a NumPy↔JAX parity check that activates once JAX is installed. Remaining
milestone: a `jax.grad`-based `energy.settle` variant (see `docs/ROADMAP.md`).
