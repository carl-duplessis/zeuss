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
- [x] `tier3_logic/sheaf.py`: a graph-level (1-skeleton) cellular sheaf over
      shared-variable agreement constraints; `H⁰`/`H¹` via rank-nullity on the
      coboundary matrix. Corrected during implementation: for this homogeneous
      construction `H¹ != 0` does *not* mean contradiction (a frustrated cycle
      is full rank, `H¹ == 0`) - the structural question is `H⁰ == 0` (only
      the trivial section survives); whether specific agents' conclusions
      disagree is answered by `local_section`/`is_consistent_with` instead.
      `from_theories` catches disagreement invisible to any single
      `Theory.satisfied()` check in isolation. `zeuss audit` CLI demo added.

## v0.9 — active inference / Expected Free Energy drive loop
- [x] `drive.py`: score candidate actions by `EFE = pragmatic_value -
      epistemic_value`, reusing `Landscape.energy` and `collapse.entropy`;
      restrict to discovery actions when parameters are missing.
      `zeuss drive` CLI demo added. Scoped as a discrete-action EFE scorer,
      not a full active-inference generative-model agent.

## v0.10 — Bayesian AST program synthesis
- [x] `tier4_synthesis/`: a total (guaranteed-terminating) expression DSL with
      loops (bounded folds), bounded recursion (fuel-limited), lists, and
      conditionals; a mutation/crossover population search scored by
      example-based energy, with fitness-proportionate selection reusing
      `collapse.softmax` and selection pressure rising across generations.
      Verified against held-out examples — not a claim of general program
      correctness. `zeuss synth` CLI demo added. Note: the auto-search only
      targets the non-recursive DSL subset; `Letrec`/`Recur` is interpreter-
      supported and tested directly, not auto-synthesized (see `dsl.py`).

## v0.11 — richer synthesis DSL: list ops
- [x] `tier4_synthesis/dsl.py`: added `Length`, `Index`, `Map`, `Filter`
      (all structural/bounded, same totality guarantee as `Fold`) and more
      `BinOp`s (`% <= >= != >`). Random generation switched from uniform to
      weighted construct selection - uniform weighting meant every new
      construct silently diluted how often `fold`/`binop` got picked, making
      previously-easy targets harder to find purely from grammar growth.
      Found and fixed a real robustness bug while extending: `_mismatch`
      could itself raise (a nested list from a badly-generated `Map`/`Filter`
      body) *outside* `program_energy`'s exception guard, which could crash
      the whole search instead of just penalizing one candidate - moved the
      mismatch scoring inside the same `try`. `zeuss synth` CLI demo now
      shows both a `Fold` and a `Filter` target.

## v0.12 — recursion synthesis attempt (honest result: safe, not reliable)
- [x] `tier4_synthesis/search.py`: made `Letrec`/`Recur` generation and
      mutation scope-aware (`recur_ctx` threaded through `_grow`; mutation
      uses a scope-tracking traversal, `_replace_at_scoped`, instead of the
      scope-blind one used for crossover). Two real bugs found and fixed
      while testing this against actual runs:
      1. A large-enough fuel budget let Python's own interpreter stack
         overflow *before* the fuel counter did, raising `RecursionError`
         instead of the documented `FuelExhausted` - `evaluate()` now
         converts any `RecursionError` into `FuelExhausted` itself,
         regardless of tree shape or budget.
      2. Genetic bloat: with no size pressure, mean population tree size grew
         ~6x over 15 generations (confirmed by direct measurement), making
         full-budget runs take 100+ seconds. Fixed with parsimony pressure
         (a small size penalty added only to the reproduction-selection
         score, not to the true "verified" energy) in `synthesize`.
      Despite both fixes, blind mutation/crossover does **not** reliably find
      a correct solution for a target that genuinely requires recursion
      (`2**n`, no shortcut in this arithmetic grammar) within a practical
      budget - confirmed empirically, not assumed. Shipped as: recursion
      generation/mutation is scope-correct and safe (tested), but the search
      is not claimed to reliably *discover* new recursive definitions.

## v0.13 — dug deeper into recursion discovery: two more real bugs, still no
- [x] Follow-up attempt at reliably *finding* recursive solutions (not just
      safely generating them), per user request. Added `_recursive_template`
      (a "decrement-and-combine" skeleton) mixed into generation/mutation via
      a `template_rate`, since empirically under 5% of random depth-4 trees
      even contain a `Letrec` with an `If`-shaped body. Two more real bugs
      found and fixed while testing this against actual runs, not assumed:
      1. **Unbounded value magnitude.** A generated candidate recursed with
         an argument that *grew* (squaring) instead of shrinking toward its
         base case - Python's arbitrary-precision integers reached numbers
         with over 100 million bits well within the fuel budget's call-count
         limit, making bignum arithmetic the actual runaway cost (confirmed:
         25 squarings from 4 produces a 134-million-bit integer). Fixed with
         a `_MAX_MAGNITUDE` bound checked after every arithmetic op in
         `dsl.py`, raising `ValueOverflow` (caught the same way as any other
         evaluation error) instead of hanging on bignum arithmetic.
      2. **Pervasive performance regression.** `letrec`/`recur` were
         unconditionally in the random-generation kind-weight pool, so *any*
         search - including ones with no scalar/recursive target at all,
         like list-sum-via-fold - could spawn expensive recursive candidates
         and got measurably slower. Fixed by making recursion opt-in
         (`allow_recursion=False` by default in `random_program`/`mutate`/
         `synthesize`), restoring every non-recursive test to its original
         speed while keeping the recursion-attempt path available for
         callers who want it.
      Honest result after all of this: `2**n` is still not reliably found
      within a practical budget. Recursion synthesis in this codebase is a
      genuinely tested negative result, not a limitation nobody looked at -
      consistent with the broader GP literature, where recursive program
      synthesis via blind mutation is known to be substantially harder than
      iterative/functional constructs.

## v1.0 — GA-HDC (experimental, optional)
- [x] `tier2_substrate/geometric.py`: a small-grade Clifford algebra `Cl(n,0)`,
      `n <= 6`, as an additive relation-rotor layer alongside (not replacing)
      the existing complex-phasor hypervectors. Verified: geometric product
      associativity, known Cl(2,0) blade identities, rotor magnitude
      preservation, rotor angle composition, and the Cl(2,0) even-subalgebra
      <-> complex-number bridge. Exploratory; no other module depends on it.

## Notes on hardware
- JAX GPU/TPU and Triton need Linux + a CUDA GPU. On this Windows machine, use
  WSL2 for GPU work, or run the CPU path (which is the default and fully
  supported). See https://docs.jax.dev/en/latest/installation.html
