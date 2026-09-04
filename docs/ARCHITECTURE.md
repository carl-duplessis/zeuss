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

### Energy (`energy.py`) — Frontier 2
A `Landscape` is weighted attractor hypervectors (the axioms). `settle` relaxes
a state toward the softmax-weighted attractor mean on the phase torus.
`temperature = 0` is a deterministic descent to the ground state; `temperature
> 0` injects thermal phase noise (probabilistic exploration). Satisfying a rule
and minimising energy are one operation.

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
bundled hypervectors (NetworkX for the graph, with a fallback). `compiler.py`
provides t-norms, residuated implications, weighted `Rule`s and a `Theory`
whose total penalty is a continuous energy over `[0,1]` valuations — the bridge
from symbolic axioms to Tier-2 basins.

## Backends
`backend.py` exposes `xp` (NumPy or JAX), `HAS_JAX`, and complex dtypes. Tier-2
currently computes in NumPy for a guaranteed, tested path; porting to `xp`
end-to-end (for `grad`/`jit`) is the next milestone.
