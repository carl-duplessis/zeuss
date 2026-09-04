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
matching `Theory`'s expected small-`n` scale.

## Backends
`backend.py` exposes `xp` (NumPy or JAX), `HAS_JAX`, and complex dtypes. Tier-2
substrate math is now routed through `xp` end-to-end, so the same code runs on
NumPy (default) or JAX (`ZEUSS_BACKEND=jax`). Randomness is still drawn from an
explicit NumPy `Generator` and *lifted* onto the backend, so a given seed yields
identical hypervectors on either backend. `test_backend.py` pins the routing and
carries a NumPy↔JAX parity check that activates once JAX is installed. Remaining
milestone: a `jax.grad`-based `energy.settle` variant (see `docs/ROADMAP.md`).
