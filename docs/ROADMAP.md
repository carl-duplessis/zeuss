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
- [ ] Compile a `Theory` directly into a Tier-2 `Landscape` and show that
      settling reproduces the theory's satisfying valuations.
- [ ] Probabilistic priors as temperature schedules.

## Notes on hardware
- JAX GPU/TPU and Triton need Linux + a CUDA GPU. On this Windows machine, use
  WSL2 for GPU work, or run the CPU path (which is the default and fully
  supported). See https://docs.jax.dev/en/latest/installation.html
