# Roadmap

## v0.1.0 — foundation (done)
- [x] Three-tier package scaffold with guarded optional deps.
- [x] FHRR/VSA hypervector algebra (bind/unbind/bundle/permute/similarity).
- [x] Entropy-driven collapse with a cooling schedule.
- [x] Energy-landscape settle (deterministic + thermal).
- [x] Wave-resonance primitives.
- [x] Fuzzy-logic compiler (Łukasiewicz / Gödel / product) + Theory energy.
- [x] NumPy reference kernels + parity test.
- [x] 17 passing tests, runnable demo, CLI.

## v0.2 — JAX-native substrate
- [x] Route all Tier-2 ops through `zeuss.backend.xp`.
- [ ] `energy.settle_grad`: energy descent via `jax.grad` on phase angles.
- [ ] `jit`/`vmap` batched collapse over many probes.
- [ ] Property tests run on both backends.

## v0.3 — the collapse made structural (Frontier 1, deeper)
- [ ] Entropy-conditioned dimensionality: expand basis when entropy is high,
      project onto a low-dimensional orthogonal subspace as entropy → 0.
- [ ] Learnable codebook (train symbols so real structure self-organises).

## v0.4 — GPU kernels (Frontier 3, faster)
- [ ] Triton `phase_interference` and `topological_collapse_step`.
- [ ] Parity tests vs. NumPy reference; benchmark harness.
- [ ] Optional FFT-based binding for very high dimensions.

## v0.5 — the logic compiler closes the loop (Frontier 2)
- [x] Compile a `Theory` directly into a Tier-2 `Landscape` and show that
      settling reproduces the theory's satisfying valuations
      (`tier3_logic/grounding.py`).
- [ ] Probabilistic priors as temperature schedules.

## v0.6 — liquid time-step dynamics (adaptive annealing)
- [x] `energy.settle_adaptive`: step size scales with how much the previous
      step reduced energy (shrink near plateaus/saddles, grow on open
      gradients), with early stopping on convergence.
- [x] `collapse.anneal_adaptive`: β growth rate is set once from the probe's
      intrinsic top-2 similarity margin (an ambiguous near-tie grows slowly,
      an unambiguous probe grows fast), instead of a fixed static schedule.

## v0.7 — event-spiking asynchronous activation
- [x] `tier2_substrate/spiking.py`: a `SpikingGate` with refractory hysteresis
      that gates which `Landscape` attractor groups participate in `settle`,
      skipping compute on dormant regions.

## v0.8 — sheaf cohomology topological consistency auditor
- [ ] `tier3_logic/sheaf.py`: a graph-level (1-skeleton) cellular sheaf over
      shared-variable agreement constraints; `H⁰`/`H¹` via rank-nullity on the
      coboundary matrix, detecting global contradictions that are invisible to
      any single `Theory.satisfied()` check in isolation.

## v0.9 — active inference / Expected Free Energy drive loop
- [ ] `drive.py`: score candidate actions by `EFE = pragmatic_value +
      epistemic_value`, reusing `Landscape.energy` and `collapse.entropy`;
      restrict to discovery actions when parameters are missing.

## v0.10 — Bayesian AST program synthesis
- [ ] `tier4_synthesis/`: a total (guaranteed-terminating) expression DSL with
      loops (bounded folds), bounded recursion (fuel-limited), lists, and
      conditionals; a mutation/crossover population search scored by
      example-based energy, with fitness-proportionate selection reusing
      `collapse.softmax` and selection pressure rising across generations.
      Verified against held-out examples — not a claim of general program
      correctness.

## v1.0 — GA-HDC (experimental, optional)
- [ ] `tier2_substrate/geometric.py`: a small-grade Clifford algebra `Cl(n,0)`,
      `n <= 6`, as an additive relation-rotor layer alongside (not replacing)
      the existing complex-phasor hypervectors.

## Notes on hardware
- JAX GPU/TPU and Triton need Linux + a CUDA GPU. On this Windows machine, use
  WSL2 for GPU work, or run the CPU path (which is the default and fully
  supported). See https://docs.jax.dev/en/latest/installation.html
