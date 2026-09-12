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
- [x] `energy.settle_grad`: energy descent via `jax.grad` on phase angles.
      State is reparameterised as real phase angles `theta` (`z=exp(i*theta)`)
      so ordinary real-to-real `jax.grad` applies directly and `z` stays
      exactly unit-modulus with no `normalize()` projection needed. Two real
      issues found and fixed, not assumed away: `Landscape.energy`/
      `similarity`'s explicit `float()` casts abort a JAX trace, so
      `settle_grad` carries its own numerically-identical restatement of the
      energy formula in terms of `theta`; and the raw gradient is ~1e-4 per
      component (the energy divides by `D` twice), so the step is scaled by
      `D` internally, calibrated so `learning_rate=0.5` reaches the same
      ground-state energy as `settle`'s default in the same 60 steps.
      Requires the JAX backend, raises `RuntimeError` otherwise. Discovered
      along the way: this project's own `.venv` already has JAX installed and
      picks it as the default backend (`ZEUSS_BACKEND=auto` prefers JAX when
      importable), so the full suite was re-verified with JAX genuinely
      active by default, not just via the forced-subprocess parity test.
- [x] `jit`/`vmap` batched collapse over many probes: `collapse.collapse_batch`
      (eager, either backend) resolves N probes in one vectorized pass
      instead of a Python loop over `collapse()`, and `collapse_batch_jit`
      compiles the same computation via `jax.jit`. Needed a separate
      `_collapse_batch_core` pure-array function (no dict, no `float()`
      casts, no name lookups) because those aren't traceable - the same
      lesson `settle_grad` already hit with `Landscape.energy`. `softmax`
      generalised to normalise per-row (`axis=-1`) rather than globally, a
      strict generalisation verified bit-identical on the existing 1-D call
      sites. Verified: `collapse_batch` matches per-probe `collapse()`
      exactly; `collapse_batch_jit` matches `collapse_batch`; both checked
      against the `.venv`'s real JAX install (90 passed, 5 skipped), not
      just import-guarded.
- [x] Property tests run on both backends: `test_backend.py`'s single old
      parity test (bind/unbind only) split into four, one per Tier-2 module -
      hypervector algebra, `settle`'s full energy trace (not just an end
      value), `collapse_batch`, and `resonate` - each run once under NumPy
      and once with `ZEUSS_BACKEND=jax` forced in a subprocess, asserting the
      *same numbers* rather than "doesn't crash on either backend."

## v0.3 — the collapse made structural (Frontier 1, deeper)
- [x] Entropy-conditioned dimensionality: `collapse.participation_ratio`
      (inverse Simpson index, `1/sum(p_i^2)`) turns the occupancy
      distribution into a continuous *count* of how many codebook symbols
      the state is actually spread across - exactly `1.0` for a one-hot
      distribution, exactly `K` for uniform over `K` symbols. `dimensional_
      collapse` uses it to literally truncate the live basis to the top
      `ceil(participation_ratio)` symbols by occupancy, dropping the rest
      entirely rather than down-weighting them - a real `K`-dim-to-1-dim
      basis contraction as entropy falls, matching `VISION.md`'s "expands
      into high-dimensional... collapses into low-dimensional orthogonal...
      Boolean lattice" framing, not just a fixed-`D` reweighting (which is
      all plain `collapse` ever does). At maximal entropy (uniform
      occupancy) every symbol stays live and this is numerically identical
      to `collapse`, checked directly. Found one real bug while verifying:
      floating-point noise pushed a true one-hot `participation_ratio` a
      hair above `1.0` (`1.0000000000000862`), which plain `ceil` rounded up
      to `2` instead of `1` - fixed with a small epsilon tolerance before
      `ceil`, not by rounding the ratio itself (which would blur genuinely
      fractional values elsewhere on the schedule).
- [x] Learnable codebook (train symbols so real structure self-organises):
      `collapse.train_codebook` trains symbol *phases* via `jax.grad` (the
      same `theta`/`exp(i*theta)` reparameterisation `settle_grad` uses, and
      for the same reason - `occupancy`/`collapse` go through dict lookups
      and `float()` casts that abort a JAX trace, so this carries its own
      self-contained, differentiable restatement of the exact bind + bundle
      + unbind + softmax-cross-entropy math). Trains against *bundled*
      records (the shape `encode_record` consumes: several `(role, filler)`
      pairs superposed into one hypervector), not an isolated bind/unbind
      pair - a lone pair is exactly invertible by phase subtraction
      regardless of dimension or training, so it would give training
      nothing to fix. Interference from the *other* pairs bundled into the
      same record is the real, dimension-dependent error training reduces.
      Verified, not assumed: in a dimensionality-constrained regime
      (`dim=12` for 24 symbols, 5-pair records), a random codebook recovers
      only ~49% of bundled role/filler pairs correctly (near chance);
      training for 150 steps reaches 100% recovery, with mean cross-entropy
      loss dropping from ~1.5 to ~0.16 - real structure self-organising to
      fix a real, measured failure mode, not just a loss number going down.
      Requires the JAX backend; raises `RuntimeError` otherwise (mirroring
      `settle_grad`/`collapse_batch_jit`).

## v0.4 — GPU kernels (Frontier 3, faster) — retired for now, not done

- [ ] Triton `phase_interference` and `topological_collapse_step`.
- [ ] Parity tests vs. NumPy reference; benchmark harness.
- [ ] Optional FFT-based binding for very high dimensions.

**Explicitly retired during Phase 1 of the substrate-assessment roadmap,
stated plainly rather than left as a silently-stale open item:** this
development environment has no CUDA GPU (see "Notes on hardware" below -
Triton needs Linux + a CUDA GPU, this is Windows), so there is no way to
actually implement *or parity-test* a kernel here - writing one blind,
untested against real hardware, would violate this project's own "GPU
implementations must match the reference within tolerance" rule
(`CLAUDE.md`) rather than honor it. `tier1_kernels/triton_kernels.py`
remains exactly what it was designed to be from the start - a guarded,
optional stub (`AVAILABLE = False`, import-guarded) that the rest of the
substrate already runs correctly without; nothing is broken or blocked
by this being unimplemented, and every one of this session's own
measurements (v0.39 through Phase 1) ran entirely on the NumPy path.
Not deleted from the roadmap - genuinely worth revisiting if GPU hardware
becomes available - just no longer an implicitly-open item nobody has
looked at since v0.4.

## v0.5 — the logic compiler closes the loop (Frontier 2)
- [x] Compile a `Theory` directly into a Tier-2 `Landscape` and show that
      settling reproduces the theory's satisfying valuations
      (`tier3_logic/grounding.py`).
- [x] Probabilistic priors as temperature schedules: `compile_theory`/
      `compile_theories` take an `inverse_temperature` (default `1.0`,
      backward-compatible) scaling the Boltzmann weighting over Boolean
      corners - low values keep a genuine, broad prior across every
      near-satisfying corner, high values narrow it to only the theory's
      exact zero-energy corner(s) (checked directly on the compiled
      `Landscape`, not assumed). `anneal_theory` carries this across a
      cooling schedule, Frontier 2's analogue of `collapse.anneal`: at each
      `beta` it recompiles the theory and settles the running state into it
      (same `beta` drives both the prior's breadth and how sharply settling
      pulls - one temperature knob for both). A real, checked-not-assumed
      finding along the way: entropy crystallises toward 0 even on a
      genuinely *underdetermined* theory (several equally-satisfying
      corners) - a single settling trajectory is one continuous state, so it
      spontaneously breaks the symmetry and commits to *one* tied corner
      (the same phenomenon a ferromagnet's mean-field descent shows,
      picking one degenerate ground state rather than hovering between
      them), not the "stays ambiguous" behaviour first assumed and then
      disproved empirically. What's still verified true: the corner it
      commits to is a genuinely satisfying one, not an arbitrary point.

      Found and fixed a real, pre-existing calibration bug while verifying
      readout confidence during this work (predates this roadmap item -
      `valuation_to_hypervector`/`readout` are original v0.5 code): `FALSE`
      was an independently-drawn codebook symbol rather than `TRUE`'s
      literal phase-antipode, so even a single crisply-true variable with no
      other variables bundled in topped out at `readout ~0.75`, not `1.0`,
      and a 10-variable crisp corner read out at only `~0.57` - barely above
      the `0.5` "unknown" baseline. Fixed by making `FALSE = -TRUE` (a
      genuine opposite pole, matching this module's own "poles on the phase
      torus" framing); the 1-variable case now reads out at exactly `1.0`
      and the 10-variable case improves to `~0.64` (the remainder is honest
      bundling interference - the same effect `train_codebook` already
      demonstrated - not a calibration artifact). No other module referenced
      the `:FALSE` symbol name directly, so this was a safe, self-contained
      fix.

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

## v0.14 — a different paradigm: resonance-guided template evolution
- [x] Per user request to "try the untried" rather than keep tuning classical
      GP knobs: `tier4_synthesis/resonance_bias.py` adds `ResonantBias`, an
      Estimation-of-Distribution prior over `_recursive_template`'s
      hole-fillers implemented with the same hypervector primitives every
      other tier uses (`Codebook`, weighted superposition, similarity
      readout via `collapse.softmax`) instead of a bolted-on statistics
      table - a genuinely different search paradigm (belief tracked and read
      via resonance) rather than another round of hyperparameter tuning on
      the same blind-mutation algorithm. `extract_template_choices` matches
      *observed structure*, not provenance, so the bias learns from any
      population member that evolved into the template shape, not just ones
      the template generator produced. Reinforcement weight is
      `exp(-energy)` (the same Boltzmann convention `grounding.py` uses at
      compile time). Found and fixed a real design flaw before shipping: an
      initial `beta=4.0` made a *single* reinforcement event lock in a choice
      96% of the time (100% after three) - far too greedy, killing
      exploration exactly when it matters; retuned to `beta=0.2`, giving a
      gradual, evidence-proportional shift instead of instant collapse.

      Honest result, run and re-run rather than assumed: with
      `allow_recursion=True, resonant_bias=True` (the new default once
      recursion is opted into), the search **found** a genuinely correct,
      held-out-generalizing recursive definition of `2**n`
      (`f(n) = 1 if n<1 else f(n-1)+f(n-1)`, correct for n up to 8, none of
      which were training examples) on 2 of 9 seeds tried at the same budget
      that found it 0 times across every seed tried without the bias, both
      earlier in this session and in this same comparison. This is a real,
      measured improvement over the v0.13 negative result - not a claim that
      recursion synthesis is now reliable (7 of 9 seeds still fail). `zeuss
      synth` CLI demo now includes this as Scenario 3.

## v0.15 — closing the gap: 9/9, by diagnosing the other 7 rather than tuning
- [x] Per user request to actually reach 9/9 rather than stop at "sometimes":
      diagnosed *why* the other 7 seeds failed instead of just scaling
      compute. Found the winning `2**n` solution (`f(n-1)+f(n-1)`) doesn't
      match `_recursive_template`'s only shape at all - it needs two
      recursive calls combined, not one combined with the parameter. Added a
      second shape, `f(p-step) OP f(p-step)` (`combine_kind`), directly
      generatable and reinforceable via `ResonantBias` instead of relying on
      mutation to build it from nothing (confirmed via `extract_template_choices`
      that the actual winning structure was previously unreinforceable).
      This alone made the success rate *worse* (0/6) until diagnosing that
      too: the two-call shape's call tree grows exponentially, so it hits
      fuel exhaustion (and gets penalized as a near-total failure) far more
      readily than the one-call shape for equally-bad hole-fillers,
      confirmed by computing the exact fuel cost by hand (`T(n) = 1 +
      2*T(n-1)`, `T(4) = 31` - already close to the old 60-fuel default) -
      biasing the resonance memory toward the wrong shape before either got
      a fair trial. Fixed two ways: made `combine_kind` itself exempt from
      resonance bias (always drawn uniformly - only the *fine-tuning within*
      a chosen shape is biased, so premature commitment to the wrong family
      can't happen), and exposed `fuel_budget` as a `synthesize` parameter,
      raised to 200. Also simplified `double_recur` to a single shared step
      on both calls (dropping an independent `step2`) - halves the
      combinatorial search burden for a shape that's symmetric in the
      target case, at the honest cost of not covering asymmetric recursion
      like Fibonacci (now the load-bearing example in
      `test_recursion_synthesis_is_safe_but_not_reliably_found`).

      Result: at one fixed configuration (`population_size=800,
      max_generations=150, fuel_budget=200, allow_recursion=True,
      resonant_bias=True`), **all 9 of 9 seeds tried found a genuinely
      correct, held-out-generalizing `2**n`** (checked for n up to 9). Not a
      cherry-picked seed - the same configuration re-run across every seed
      tried in v0.14's comparison. `zeuss synth` CLI Scenario 3 updated to
      this configuration.

## v0.16 — asymmetric recursion (Fibonacci): a second liquid time-step
- [x] Per user request ("should we cover asymmetric recursion?" -> "yes, go
      ahead"): reintroduced `double_recur`'s dropped second step as `delta`
      (`f(p-step)` and `f(p-step-delta)`, `delta=0` recovering v0.15's
      symmetric shape) so asymmetric shapes like Fibonacci are expressible.
      Confirmed by hand-calculating `T(n) = T(n-1) + T(n-2) + 1` before
      touching code (`T(8) = 41`, cheaper than `2**n`'s `T(4) = 31`) that a
      shifted-indexing Fibonacci (`F(1)=F(2)=1`, since the template's base
      case is always a fixed `Const(base_val)`, not `Var(param)` - real
      Fibonacci's `F(0)=0` doesn't fit and is out of scope here) has a safe
      fuel cost. `extract_template_choices` generalized to recognize both
      symmetric and asymmetric shapes regardless of argument order
      (0 mismatches over 500 round-trip draws).
- [x] Diagnosed a real, two-sided tradeoff rather than picking one number and
      moving on: adding `delta` at a uniform 50/50 slowed `2**n` on some
      seeds (seed 4: 0.4s -> 72.1s - an unbiased extra binary split roughly
      halves the effective population matching the still-correct `delta=0`
      case). A static skewed prior (`p=[0.85, 0.15]`) only partially fixed
      it: seed 4 eventually converged, but at ~500x the unbiased cost (219s /
      104 generations), and skewing further to fix that seed broke Fibonacci
      discovery outright (a 3.3s seed started timing out at 60s). `delta` is
      deliberately *not* routed through `ResonantBias` - a structural/family
      choice risks premature convergence to the wrong family from early
      noise, same reasoning as `combine_kind` in v0.15.
- [x] Fix: `delta_p1` (probability of drawing the asymmetric branch) anneals
      on *stagnation* (generations since `best_energy` last improved) rather
      than a fixed ratio - the same "adapt the step to how hard progress
      currently is" idea as `collapse.anneal_adaptive`, applied to a discrete
      structural choice instead of a continuous beta. Stays at
      `delta_p1_start=0.15` while progress continues, grows geometrically
      (`delta_p1_stagnation_growth=1.08` per stagnant generation, capped at
      `delta_p1_max=0.5`) once stuck, resets on the next improvement - so a
      run only pays the asymmetric-search cost when it actually needs to.
      Threaded through `random_program`/`mutate`/`_replace_at_scoped`/
      `synthesize`.
- [x] Result, measured at the same configuration as v0.15
      (`population_size=800, max_generations=150, fuel_budget=200,
      allow_recursion=True, resonant_bias=True`): `2**n` still verifies on
      seeds 0, 4, 7 (seed 4 now ~178s - slower than the pre-`delta` 0.4s, a
      real and documented cost of supporting asymmetric shapes at all, but no
      longer ~500x worse or unsolvable). Fibonacci verifies and genuinely
      generalizes to held-out n on seeds 0 and 1. **Not** reliable the way
      `2**n` is: seed 4 fails to verify within this budget, and seed 7
      "verifies" against all six training examples with a degenerate,
      coincidental expression that does *not* generalize to held-out n -
      caught only by checking held-out points, a real demonstration of why
      "verified" never means "proven correct" in this module. Shipped as:
      `test_resonant_bias_can_discover_fibonacci` (the checked positive
      claim, seed 1) plus an updated docstring on
      `test_recursion_synthesis_is_safe_but_not_reliably_found` (no longer
      claims the asymmetric shape is missing from the grammar - it's a
      budget/reliability boundary now, not a grammar one).

## v0.17 — zero-indexed Fibonacci, and an honest non-fix
- [x] Two good-first-tasks from v0.16's writeup, tackled together. First:
      `_recursive_template`'s base case was always `Const(base_val)` - a
      fixed number, so real Fibonacci (`F(0)=0`) wasn't expressible, only the
      reindexed `F(1)=F(2)=1` workaround was. Added `base_kind` (`"const"` /
      `"param"`, the latter using `Var(param)` itself as the base case) as a
      third structural hole, uniformly drawn and never resonance-biased
      (same reasoning as `combine_kind`/`delta`). Verified: 2000 round-trip
      draws with 0 mismatches, a hand-built `if p<=1 then p else
      f(p-1)+f(p-2)` evaluates correctly and extracts to the expected
      hole-fillers, and re-checking `2**n` on seeds 0/4/7 confirmed the extra
      structural coin flip doesn't regress its reliability (all three still
      verify and generalize). Result: real zero-indexed Fibonacci is
      genuinely discoverable and held-out-generalizing (`test_resonant_bias_
      can_discover_zero_indexed_fibonacci`).
- [x] Second: tried to close the `delta_p1`-schedule gap so Fibonacci is as
      reliable as `2**n` (seeds 4/7 from v0.16). Found this does **not** have
      a clean fix - recorded honestly rather than papered over. Adding one
      more training example (8 instead of 7, shifted-indexing target) *did*
      fix both seed 4 (not-verified -> verified-and-generalizing) and seed 7
      (stopped overfitting), but broke seed 1, which had generalized fine on
      the original 7 examples. A genuine whack-a-mole across seeds, not a
      fixable-with-more-data problem the way `list_sum`'s overfitting was.
      Left open as a good-first-task with this finding attached, rather than
      closed on a claim the evidence doesn't support.
- [x] `zeuss synth` CLI Scenario 4 switched to the real zero-indexed target
      (was the reindexed workaround); docs (`docs/ARCHITECTURE.md`,
      `CLAUDE.md`'s good-first-tasks) updated to match.

## v0.18 — sequence encoding (bind + permute)
- [x] `hypervectors.encode_sequence`/`decode_sequence`: a good-first-task
      from `CLAUDE.md`. Each position `i` gets its own role,
      `permute(anchor, i)` (a single fixed anchor symbol cyclically shifted
      by position - exactly what this module's own docstring already says
      `permute` is for), the item at that position is `bind`-bound to its
      positional role (the same role-filler pattern `encode_record` already
      uses, positions instead of named fields), and every pair is bundled
      into one vector. Decoding unbinds each position's role in turn and
      runs `Codebook.cleanup` - the classic VSA cleanup-memory step.
      Verified: exact roundtrip recovery up to 24 bundled items at
      `dim=8192` (this project's usual dimension), a repeated item at two
      different positions decodes correctly at both (position, not just
      membership, is what's encoded), reordering the same items produces a
      genuinely dissimilar vector (`similarity < 0.5`), and - matching this
      project's habit of demonstrating real failure modes, not just success
      - decoding measurably degrades at a dimension too low for the
      sequence length (`dim=64` for 12 items: 9/12 correct, not silently
      wrong or silently fine), the same dimension-vs-bundle-size tradeoff
      already documented for `collapse.train_codebook`.

## v0.19 — entropy-vs-beta / energy-vs-step visualisation
- [x] `demos/experiment_entropy_energy.py`: a good-first-task from
      `CLAUDE.md`, now removed from that list. A standalone script (guarded
      `matplotlib` import, like every other optional dependency in this
      project - not wired into `zeuss`'s core CLI, following `ask_demo.py`'s
      precedent of standalone-but-not-CLI-wired demos) that plots the same
      probe's Frontier 1 cooling schedule (entropy vs. beta, a fine
      `geomspace(0.1, 64)` schedule instead of the 7-point schedule the text
      demo prints) side by side with its Frontier 2 energy descent (energy
      vs. step, deterministic vs. thermal, the same comparison
      `demo_collapse.py` narrates as printed numbers) - one probe, one
      landscape, two visual angles on the same continuous-to-discrete
      crystallisation this project is about. Saves a PNG (gitignored, like
      every other generated artifact) rather than opening a GUI window.

## v0.20 — fixed the recursion-discovery reliability gap from CLAUDE.md
- [x] Attempted fix: per-family template elitism (track the best
      template-shaped individual keyed by `(combine_kind, base_kind)`
      instead of one global slot, so `param_recur` reaching a competitive
      energy first can no longer permanently starve `double_recur` out of
      its own protected slot) plus `_mutate_template_hole` (a surgical
      mutator that redraws exactly one hole-filler in place, since
      `mutate`'s old uniform subtree replacement could only ever "fix" a
      hole-filler by getting lucky on the whole surrounding subtree at
      once) with a guaranteed 8 hole-mutated offspring per family per
      generation, independent of selection pressure.
- [x] Regression found before landing: both the guaranteed-offspring loop
      and `mutate`'s own template shortcut routed template-shaped
      individuals through `_mutate_template_hole` exclusively, which only
      ever changes one hole-filler at a time. Once a lineage reached a
      Hamming-distance-1 local optimum (no single hole change improves it,
      the right combination is two or more away), nothing could push it
      past that plateau - confirmed by instrumenting a `2**n` run (seed 4):
      a family reached energy 1.0 at generation 0 and was still exactly 1.0
      at generation 59, with that lineage grown to ~60% of the population.
      All four recursion-synthesis tests failed against this version,
      including the ones this fix was meant to make more reliable.
- [x] Real fix: gated `mutate`'s per-node shortcut with a new
      `_TEMPLATE_HOLE_MUTATION_RATE = 0.5` constant, and routed the
      guaranteed-offspring loop through `mutate()` instead of calling
      `_mutate_template_hole` directly, so both paths keep a real chance of
      full scoped regrowth (a fresh `_recursive_template` draw) alongside
      single-hole refinement. Verified: the same instrumented `2**n` run
      now solves at generation 3 instead of stalling; all four
      recursion-synthesis tests pass, including
      `test_resonant_bias_discovers_recursion_reliably_across_seeds` and
      `test_resonant_bias_can_discover_fibonacci`; full suite green (101
      passed, 10 skipped) in ~94s, down from ~820s when the stuck searches
      were burning their full generation budget every run.
- [x] Also checked every seed named in the older writeups this session
      touched, not just the committed tests, since a fix this targeted is
      worth verifying honestly rather than declaring done at green CI.
      Fibonacci: v0.16/v0.17 documented seed 4 as not verifying and seed 7
      as "verifying" on a degenerate, non-generalizing expression - both now
      verify *and* genuinely generalize, at the same seeds 0/1 that already
      worked. `2**n`: v0.15's "9/9 dev seeds" claim (seeds 0-8) no longer
      holds - seed 8 now fails, confirmed via an isolated worktree checkout
      of the pre-this-session commit (`fce9ba7`) that it did verify there
      (energy 0.0), so this is a real regression, not pre-existing drift.
- [x] Dug into seed 8 rather than leaving it as an unexplained flake.
      Instrumenting it showed a different failure mode than the Hamming-1
      lock-in above: a non-template, crossover-mangled expression reaches
      energy 1.0 by generation 8 (better than any clean template family, all
      stuck at 2.0), so fitness-proportionate selection starves templates
      down to ~30-40/800 for the rest of the 150-generation budget - neither
      the hack nor the templates ever reach 0. Isolated the cause by
      re-running with per-family elitism disabled entirely (still failed,
      ruling elitism out) and then with `_TEMPLATE_HOLE_MUTATION_RATE`
      forced to 0.0 (pure full-regrowth, no elitism) - that combination
      *did* solve it (generation 50, verified). So the rate introduced by
      the real fix above is the cause: 0.5 gives templates enough surgical
      refinement to escape v0.16/v0.17's stuck Fibonacci seeds, but costs
      them enough full-regrowth diversity that this one `2**n` seed's
      competing non-template hack wins the population instead. Tried
      lowering the rate to 0.2 as a candidate middle ground: fixed seed 8,
      but broke `test_resonant_bias_can_discover_fibonacci` itself (seed 1
      still "verified" but stopped generalizing) and lost fib seed 0's
      result too - the identical whack-a-mole shape as the `delta_p1`
      finding in v0.16, just in a different knob. `0.5` was kept as
      committed: it's the value that keeps every actually-asserted test
      green, not an arbitrary pick that happens to leave one gap. Seed 8 is
      left as an honest, deliberately-not-chased limitation, the same way
      v0.17 left its own seed-generalization gap rather than tune around it.
- [x] `CLAUDE.md`'s good-first-tasks entry for this (open since v0.17)
      removed - no longer an open item.

## v0.21 — random restarts for the seed-8 plateau (five dead ends first)
- [x] Followed up on v0.20's seed-8 gap with a real attempt to fix it, not
      just re-document it. Tried five different in-population diversity
      mechanisms in turn, each one confirmed dead by instrumenting a real
      run against seed 8, not just by reasoning about it:
      1. A per-family stagnation-adaptive `_TEMPLATE_HOLE_MUTATION_RATE`
         (decaying toward a floor once a family's own energy stalls) -
         fixed nothing (seed 8 still failed to verify) and broke
         `test_resonant_bias_can_discover_fibonacci` (seed 1) plus two more
         seeds not in the committed set - the same whack-a-mole shape as
         v0.20's static-0.2 attempt, just smoother.
      2. Fitness sharing (Goldberg & Richardson) over structural niches -
         dividing each individual's selection weight by its niche's
         population before renormalizing. Confirmed mathematically inert at
         the beta this module reaches: once `beta` grows large enough,
         `exp(-beta*e)` for the losing niche underflows to exact `0.0` in
         float64, and dividing/renormalizing an exact zero is still zero -
         no information survives for sharing to redistribute. Zero effect on
         seed 8, and separately regressed
         `test_resonant_bias_can_discover_zero_indexed_fibonacci` (seed 0).
      3. Random immigrants - injecting fresh `random_program()` draws (bias
         and unbiased both tried) into the reproduction loop once stagnant.
         No effect at all: a lone fresh individual has no path to matter
         once an incumbent is established - unless it beats the incumbent
         outright on its first draw, it has near-zero selection probability
         at this module's high late-run `beta` and vanishes within a
         generation, never getting the sustained protected refinement an
         established lineage gets.
      4. A per-family hard switch (not smooth decay) of
         `_TEMPLATE_HOLE_MUTATION_RATE` to `0.0` once a family's own
         stagnation crossed a threshold, elitism left untouched - still no
         effect. Root cause: elitism's unconditional reinsertion of the same
         frozen best-known individual every generation means its "full
         regrowth" offspring are all one mutation away from that one fixed
         point, not independent fresh draws - a genuinely fresh
         `_recursive_template()` redraw only happens on the rare occasion a
         mutation's randomly-chosen target is the tree's root, and even then
         isn't scoped to land back in the same family.
      5. The same per-family switch *plus* dropping that family's elitism
         slot and guaranteed refinement entirely once stuck (the closest
         possible per-family mirror of the one combination v0.20 confirmed
         solved seed 8 at generation 50: global rate 0.0 + elitism
         disabled). Instrumented trace showed all four families lapsing in
         turn and, by generation 33, *every* family fully lapsed - full
         regrowth everywhere, zero elitism anywhere - for the remaining
         ~117 generations. Energy stayed frozen at exactly `1.000` the
         entire time regardless. This was the most informative failure: it
         means the fix doesn't decompose per-family, or the original
         generation-50 solve was itself a lucky RNG path rather than a
         reliable property of that configuration.
- [x] The common thread in all five: each tries to rescue the *current*
      population from within, which means each one's fix is coupled to
      guessing the specific mechanism causing that population to be stuck.
      Switched to a mechanism that's indifferent to *why* a population is
      stuck: random restarts. Track `stagnation` against the *current
      attempt's own* best energy (separate from the all-time
      `best_energy`/`best_node`, which are never reset and always hold the
      best found across every attempt); once an attempt spends
      `_RESTART_STAGNATION_FRACTION` (0.6) of the *entire* `max_generations`
      budget with zero improvement, discard the population, the per-family
      elitism records, and the `ResonantBias` prior (which would otherwise
      keep steering fresh draws back toward the stuck lineage's own
      hole-filler values), and start over from a fresh random population -
      spending the same overall budget on several independent attempts
      instead of one that's already a lost cause.
- [x] A fixed generation-count threshold was tried first (40) and found
      unsafe: measuring it against seed 1 (previously reliable) showed that
      seed's fast wall-clock time in earlier sweeps was cheap-per-generation
      cost, not an early finish - it genuinely uses the full 150-generation
      budget, including stretches well past 40 generations with zero
      improvement, as a normal part of succeeding. The fixed threshold
      restarted it mid-convergence and lost the run outright. Switched to a
      fraction of the caller's own `max_generations` instead of an absolute
      count, precisely because a genuinely-progressing search's "how long is
      too long to wait" scales with how much budget it was given, not with
      a number tuned for one specific `max_generations` value.
- [x] A second regression found before landing: resetting `beta` to
      `beta_start` on restart broke
      `test_selection_pressure_rises_across_generations`, which asserts
      `beta` rises monotonically across an entire call - a real, deliberate
      invariant elsewhere in this module, not an incidental one. Fixed by
      leaving `beta` untouched across a restart; a fresh population still
      gets fresh crossover/mutation diversity regardless of what `beta`
      currently is, and doesn't need a fresh annealing schedule too.
- [x] Full seed sweep after both fixes, same configuration as v0.20's
      (population=800, generations=150, fuel_budget=200):
      `2**n` seeds 0-7 all verify and generalize, including seed 5 - a
      silent, previously-undocumented gap (verified but non-generalizing)
      this surfaced and fixed as a side effect, not something v0.20 knew
      about. Seed 8 still doesn't verify - unchanged from v0.20's baseline,
      not worse - now a case of "this particular target's true solution is
      hard enough to need more than one 150-generation attempt to hit by
      chance" rather than "permanently unreachable from this population."
      Fibonacci seeds 0/1/4/7 and zero-indexed seed 0 all still verify and
      generalize. Full suite green: 101 passed, 10 skipped, and the
      committed reliability tests run in the same ~57-59s as before these
      changes - confirming restarts never trigger on any currently-healthy
      seed, exactly as designed.
- [x] Honest status: seed 8 is not solved, but the mechanism addressing it
      is now general rather than specific to seed 8's own failure signature
      (unlike all five rejected attempts, which each depended on guessing
      *why* a population was stuck). A future occurrence of "population
      converges on a plateau below the target" in this search - on any
      target, any seed, for any underlying reason - gets the same
      independent-attempts treatment automatically, without needing its own
      bespoke diagnosis first.

## v0.22 — list-op reliability audit, diagnosed and fixed
- [x] After v0.21 closed out the recursion-search gap, checked whether the
      *other* half of synthesis - list processing via `Fold`/`Map`/`Filter`/
      `Index` - has the same kind of hidden seed-dependent fragility
      recursion turned out to have, since every committed test for these
      (`test_synthesize_recovers_list_sum_via_fold`, `_length`,
      `_map_doubling`, `_filter_positives`, `_first_element_via_index`) only
      ever pins down a single seed, unlike the recursion tests which sweep
      several after v0.15-v0.21 found single-seed claims didn't hold up.
      Swept all five across 8 seeds each at their committed configuration,
      plus three new, harder targets not in the suite at all
      (`sum_of_squares_via_fold`, `count_positive_via_filter_length`, and a
      recursion-template sanity check, `triangular_number_recursion`), using
      the same verified-and-generalizes-on-held-out-input methodology the
      recursion investigation established.
- [x] Result: **19 of 64 runs failed** - a materially bigger gap than seed 8
      ever was. By target: `length` 8/8 and `triangular_number_recursion`
      8/8 (both fully reliable); `first_via_index` 7/8 (fails at seed 0 -
      the committed test uses seed 1, which happens to dodge the one bad
      seed); `sum_via_fold` 7/8 (fails at seed 3); `map_doubling` 6/8 (fails
      at seeds 2, 4); `filter_positives` 4/8 (seeds 2/3/6/7 "verify" but
      *silently don't generalize* - the worst failure shape, since it reads
      as green); `sum_of_squares_via_fold` 3/8; `count_positive_via_filter_length`
      2/8, including one seed whose "verified" solution *crashes* on a
      held-out input (`modulo by zero`) rather than just answering wrong.
- [x] ~~Initial hypothesis (**superseded** by the instrumented diagnosis
      below - kept because the reasoning trail is the point: this looked
      compelling and was wrong)~~: the pattern seemed to point at a
      structural cause -
      every target that goes through `_recursive_template` is
      perfectly reliable (8/8), because recursion has had four versions
      (v0.14 resonant-bias templates, v0.15-v0.17 per-family elitism and
      structural-choice-vs-learned-bias separation, v0.20 hole-mutation-rate
      tuning, v0.21 random restarts) of dedicated structural hardening. List
      processing has had none - `Fold`/`Map`/`Filter`/`Index` are generated
      and mutated by the same blind uniform `_grow`/`mutate` used for
      arbitrary arithmetic, with no equivalent template, no per-shape
      elitism, and no bias toward historically-successful hole values. In
      hindsight, poor and seed-dependent reliability here isn't surprising;
      it was simply never measured until now because every committed test
      happened to be written against a seed that worked.
- [x] ~~Planned fix under that hypothesis (**not built** - the diagnosis
      below showed it would have been solving the wrong problem)~~: a
      list-processing equivalent of `_recursive_template` /`ResonantBias` /
      per-family elitism, estimated as comparable in scope to the whole
      v0.14-v0.21 recursion effort. Worth recording that this estimate was
      never tested, because the actual fix turned out to be an example-set
      change and two budget bumps - a reminder to diagnose before scoping.
- [x] Diagnosed each failure shape by instrumentation (printing the actual
      discovered expression and the first held-out input it diverges on)
      rather than guessing - and **the diagnosis refuted this section's own
      initial hypothesis above**. The failures are not caused by list
      processing lacking recursion's structural hardening. They are two
      distinct problems, neither of which is in the search machinery:
      - **Class A - underdetermined training examples** (the
        verified-but-wrong failures). `filter_positives`' four committed
        examples contain no `0` anywhere, leaving `x > 0` and `x >= 0`
        indistinguishable on the training set; seeds 6 and 7 both returned
        `Filter(xs, item, item >= False)` (i.e. `>= 0`) and duly failed the
        held-out `[0, 0, 1]`. Seed 4's `count_positive` crash is the same
        root cause wearing a different hat: it found `item % item`, which is
        only a `ZeroDivisionError` when an item *is* zero. And in
        `[1, -2, 3, -4]` the positives are exactly the odd values, so seed 2
        learned `item % 2` instead of positivity. The search was working
        correctly throughout - it found programs consistent with every
        example it was given; the examples simply did not pin down the
        intended concept.
      - **Class B - under-provisioned budget** (the verified=False
        failures). `map_doubling` at 150/60 and `first_via_index` at 100/30
        simply never find anything on some seeds.
- [x] Fixed both classes, each with the remedy matched to its own class -
      and confirmed the remedies are *not* interchangeable, which is the
      main transferable lesson here. Adding examples fixes Class A
      (`filter_positives` 4/8 -> **8/8** at the unchanged search budget, no
      `search.py` change at all) but actively *hurts* Class B: applying the
      same "richer examples" treatment to `sum_via_fold`, whose failure is
      verified=False, took it from 7/8 to 4/8, since more constraints make
      an already-failing search harder rather than better-determined. Raising
      the budget fixes Class B (`map_doubling` 6/8 -> **8/8** at 400/150;
      `first_via_index` 7/8 -> **8/8** at 300/100) and does nothing for
      Class A. This is the same shape as v0.16's finding that adding a
      Fibonacci example fixed two seeds and broke a third - "add more data"
      and "spend more compute" are each right only for one failure mode.
- [x] Systemic fix, so this cannot silently recur: added
      `test_list_ops_are_reliable_across_seeds`, which sweeps `length`,
      `map_doubling`, `filter_positives` and `first_via_index` across 8
      seeds each and asserts *verified **and** generalises to held-out
      input* - never `verified` alone, since the audit's most common failure
      shape was a program that matched every training example and still
      diverged. This is the list-op counterpart of
      `test_resonant_bias_discovers_recursion_reliably_across_seeds`, and it
      exists because single-seed tests are precisely what hid this for so
      long (`first_via_index`'s committed seed 1 passes; seed 0 does not).
      32 synthesis runs, ~1.6s total. Full suite: 102 passed, 10 skipped,
      ~68s - unchanged from before.
- [x] Not fixed, recorded honestly rather than tuned around:
      - `sum_via_fold` stays 7/8 (seed 3 finds nothing). Both available
        remedies were measured and neither helps: richer examples take it to
        4/8, and a larger budget leaves it at 7/8 while merely moving which
        seed fails. Its committed single-seed test passes.
      - `sum_of_squares_via_fold` (a new target this audit introduced, never
        a committed test) is 3/8 and, unusually, gets *worse* with a larger
        budget (1/8 at 800/200). A parsimony-pressure explanation was the
        obvious candidate - the correct fold body `acc + item*item` is
        strictly larger than near-miss bodies, so `parsimony * count_nodes`
        penalises exactly the right answer - but this was **tested and
        refuted**: lowering parsimony makes it monotonically worse (3/8 at
        0.02, 1/8 at 0.005, 0/8 at 0.0), so parsimony is helping, not
        hurting. Left as a documented capability limit of the current
        list-op grammar rather than an unexplained flake; a genuine
        structural-template mechanism for fold shapes (the fix this section
        originally proposed for *all* of these) is plausibly still the right
        answer for this one target specifically, but is no longer justified
        by the other failures, which turned out to have simpler causes.

## v0.23 — Occam tie-break on the reported winner
- [x] Followed up v0.22's leftovers. Diagnosed `sum_via_fold` seed 3 (the
      one committed list-op target still at 7/8) rather than leaving it
      unexplained: it converges to energy 20.0 on an `Index`/`Map`
      expression that is not even a `Fold`, and burns all 100 generations
      there, while seed 0 finds the answer at generation 8. A larger budget
      does fix seed 3 - but then seed 4 "verifies" on a wrong program
      instead, and adding examples to fix *that* moves the failure to seed
      6, then to seeds 0 and 7. Chasing it further would mean adding
      examples until this particular held-out set passes, which is fitting
      the test, not fixing the search - the anti-pattern v0.16 and v0.20
      both recorded. `sum_via_fold` stays 7/8 by choice.
- [x] That diagnosis did surface a general signal worth acting on. The
      coincidental, verified-but-wrong `sum` programs are *large* (deeply
      nested double `Fold`s with dead `If` branches), while the genuine
      solution is a small `Fold(xs, 0, acc, item, acc + item)`. `synthesize`
      recorded the first individual to reach the best energy and never
      replaced it with an equally-good smaller one, even though `parsimony`
      already encodes "smaller generalises better" for *selection*. Added an
      Occam tie-break: among everything tied at the best energy, report the
      smallest, and let a later equal-energy-but-smaller candidate replace
      the recorded best.
- [x] First implementation was **confounded and thrown away**, worth
      recording since the failure is instructive: it also changed
      `idx_best`, which drives the elitism slot, so it perturbed the search
      dynamics rather than isolating the effect - the sweep came back
      19/64 -> 18/64 with the *failing seeds shuffled*, the signature of
      noise, not improvement. Re-implemented so `idx_best` stays plain
      `argmin` for elitism and only the reported winner uses the tie-break.
- [x] Clean result: 19/64 -> **18/64 with the failure set otherwise
      identical to baseline** - no shuffling, no regressions, one genuine
      fix (`filter_positives` seed 3, previously verified-but-wrong). The
      honest size of this effect is one run in sixty-four: it is a cheap,
      principled improvement that can only ever change which of several
      equally-good programs is returned, not a fix for the remaining gaps.
      Full suite 102 passed, 10 skipped, ~68s, unchanged; recursion seed
      sweep unchanged.
- [x] `sum_of_squares_via_fold` (3/8, and worse with more budget - see
      v0.22, where a parsimony explanation was tested and refuted) was left
      as the one target where a genuine structural-template mechanism for
      fold shapes looked like the right answer, now that the cheaper causes
      elsewhere had been ruled out. Built in v0.24, below - fixed 3/8 -> 8/8.

## v0.24 — a structural template for fold shapes
- [x] Diagnosed the last open item from v0.22/v0.23 before building
      anything, the same discipline v0.23 recorded as a lesson: instrumented
      `sum_via_fold` seed 3 first (not `sum_of_squares_via_fold` - the
      cheapest remaining unknown). It converges to energy 20.0 on an
      `Index`/`Map` expression that is not even a `Fold`, and burns its
      whole budget there. Neither remedy from v0.22 (richer examples,
      larger budget) fixes it without moving the failure to a different
      seed - confirmed again this session, not assumed from the old sweep -
      so it stays 7/8 by choice rather than tuned around.
- [x] That diagnosis is what motivated the actual fix, not
      `sum_of_squares_via_fold` directly: the coincidental verified-but-
      wrong `sum_via_fold` programs are large nested `Fold`s, the genuine
      solution is small. Added an Occam tie-break in `synthesize` - among
      programs tied at the best energy, report the smallest, and let a
      later equal-energy-but-smaller candidate replace the recorded best.
      A first implementation was confounded (it also changed the elitism
      index and just shuffled which seeds failed, 19/64 -> 18/64) and was
      thrown away in favour of a version that only changes the *reported*
      winner. Clean result: 19/64 -> 18/64 with the failure set otherwise
      identical to baseline, fixing `filter_positives` seed 3. Landed
      separately, before this entry.
- [x] With the example-set and budget causes from v0.22 and the Occam
      tie-break above all already accounted for, the one target neither
      touched was `sum_of_squares_via_fold`, and the reason is structural
      rather than a tuning gap: recursion synthesis has `_recursive_template`
      (a direct structural draw of `letrec f(p) = if ... then base else
      COMBINE`), so a target matching that shape doesn't need blind growth
      to stumble onto it. List processing had no equivalent - `Fold`/`Map`/
      `Filter`/`Index` are all generated by the same uniform `_grow`, so a
      fold body like `acc + item*item` has to be discovered one random
      subtree at a time, same as arbitrary arithmetic. Added
      `_fold_template`: a direct structural draw of `fold(xs, init, acc,
      item, acc COMBINE_OP transform(item))`, where `transform` is
      `identity` (sum), `square` (sum of squares), or `item CMP const`
      (e.g. counting elements past a threshold - the comparison coerces to
      an int the same way `_mismatch` already treats bool/int elsewhere in
      this module), and `init` is derived from `combine_op` (`0` for `+`,
      `1` for `*`) rather than sampled as its own hole - the same "don't
      grow the search space for comparatively little expressive gain"
      reasoning `delta` already applies in the recursion template.
      `combine_op`/`item_kind` are drawn uniformly, never resonance-biased,
      mirroring `_recursive_template`'s own combine_kind/base_kind split -
      structural/family choices risk premature commitment to the wrong one
      from early noise. `cmp`/`const` (only meaningful for the comparison
      transform) are the tunable holes, resonance-biased through a second,
      independent `ResonantBias` instance (the class was already generic
      over its category dict - no changes needed there).
- [x] Wired in exactly parallel to the existing recursion machinery, not a
      parallel-but-different design: a fold-template draw at the same two
      injection points `_recursive_template` uses (`random_program`,
      `_replace_at_scoped`'s target match) at the *same* `template_rate`;
      a surgical `_mutate_fold_template_hole` in `mutate`, gated on
      `isinstance(node, Fold)` rather than `allow_recursion` (fold synthesis
      is orthogonal to recursion synthesis); per-family
      (`combine_op`, `item_kind`) elitism plus guaranteed refinement
      offspring in `synthesize`, mirroring the recursion per-family block
      bug-for-bug (including the same "a lucky family shouldn't starve a
      sibling family out of its own protected slot" reasoning); and a fresh
      fold `ResonantBias` created and reset on restart alongside the
      recursion one.
- [x] Measured, not assumed: `sum_of_squares_via_fold` **3/8 -> 8/8**
      (committed budget, unchanged), each seed now converging in a handful
      of generations rather than needing (and mostly failing at) 100. Full
      list-op sweep improved from 18/64 to **7/64** - the remaining failures
      are `count_positive_via_filter_length` (a Class A underdetermined-
      example failure already diagnosed in v0.22, using this sweep script's
      original un-fixed example set) and one RNG-perturbation seed shuffle
      on `map_doubling`/`filter_positives`, not new fold-template failures.
      Recursion seed sweep unchanged at 13/14 (pow2/Fibonacci calls never
      touch `list_inputs`, so the fold path never engages - confirmed
      directly rather than assumed from the code path). Full suite 103
      passed, 10 skipped, ~70s (was 102/10/~68s) - one new committed test,
      negligible added cost. `python -m zeuss demo` still runs.
- [x] Added `test_synthesize_recovers_sum_of_squares_via_fold` (single-seed,
      matching every other list-op test's shape) and extended
      `test_list_ops_are_reliable_across_seeds` (v0.22) with the same
      8-seed sweep this entry measured, so this target is now held to the
      same standard as the rest rather than trusted on one seed.
- [x] Correction, found within the hour of writing the entry above -
      recorded rather than quietly folded in, since the whole point of this
      running log is to keep the reasoning trail honest: `sum_via_fold`
      seed 3 (left at 7/8 "by choice" three separate times across v0.22,
      v0.23, and this entry) was re-swept after landing the fold template
      and is now **8/8**. Not a new fix - a side effect never checked for.
      "Sum" is exactly `combine_op="+", item_kind="identity"`, one of
      `_fold_template`'s three direct draws, so seed 3's population gained
      the same structural escape route `sum_of_squares_via_fold` did. The
      actual lesson isn't the miss itself but *why* it was missed: the fold
      template was built and validated against the target that motivated
      it, and the sweep re-confirmed recursion's seeds and the aggregate
      list-op count, but never re-checked the *other* already-diagnosed
      failure this same mechanism structurally overlapped with. Diagnosing
      a failure and fixing it in the same session doesn't automatically
      surface every other place the fix also applies - that still takes a
      deliberate re-check, and this one waited for the next question rather
      than being caught immediately. `test_synthesize_recovers_list_sum_via_fold`'s
      docstring now records this directly.

## v0.25 — Vault Run: composing the tiers into one agent, not five demos

- [x] Every tier (VSA hypervector algebra, entropy-driven collapse,
      energy-landscape settling, the fuzzy-logic-to-energy compiler, the EFE
      action scorer) had been demoed and tested in isolation but never
      composed into one running loop - `python -m zeuss demo`/`cli.py` calls
      each primitive independently in a single script. Built `src/zeuss/
      agent.py`: a small 5-room navigation toy domain ("Vault Run") with two
      doors that start unprobed (a shortcut and a detour), composing
      `Ontology`+`qa.ask` (room-kind lookup), `compile_theory`/`settle`/
      `readout` (belief update from sensor evidence), and `drive.
      select_action` (called twice per tick - once against a goal
      `Landscape` built from `Ontology` room waves to pick a move, once
      against the compiled belief `Landscape` to pick which door to probe,
      `missing_params=True` genuinely exercised) into one stateful
      perceive->represent->infer->choose loop. No existing tier module was
      changed - everything is achieved by calling existing public functions
      correctly. `python -m zeuss agent` is the new CLI entry point.
- [x] Two facts verified by reading source before designing anything (not
      assumed from the architecture doc): `Ontology` builds its own internal
      `Codebook` (`field(init=False)`), so every other API call in the new
      module explicitly passes `world.onto.codebook` to share one vector
      space; and Lukasiewicz implication
      (`lukasiewicz_implies(a,b)=clamp(1-a+b)`) makes a single-direction rule
      `Rule("sensor_X","door_X_open",w)` only ever pull belief *up* - penalty
      is zero whenever `door_belief >= sensor`, so a low sensor reading never
      pulls a high belief back down. Fixed by registering both implication
      directions per sensed door, summing to a true biconditional
      (`w * |sensor - belief|`) - covered by an explicit low-sensor test,
      since a high-only test would pass even with the one-directional bug.
- [x] Three more real bugs found by running the thing, not by reasoning
      about the code - each fixed by measuring the actual failure, the same
      discipline the whole tier4 line of work this session established:
      1. A first `add_door` design inferred "known open from the start" from
         `ground_truth == 1.0`, so whenever a scenario happened to set the
         shortcut open, it was silently pre-revealed before any probe -
         confusing "structurally always-open" (the three fixed doors) with
         "this scenario's uncertain door happens to come out open." Fixed
         with an explicit `known: bool` flag instead of inferring it.
      2. `compile_theory`/`settle`/`readout` for a genuinely unprobed door
         (no sensor rule at all, every Boolean corner tied on that bit) was
         assumed to read out near 0.5. Measured instead: both uncertain
         doors read ~0.3 with *zero* probes ever taken. Root cause is
         already documented elsewhere in this codebase -
         `grounding.anneal_theory`'s own docstring states a single settle
         trajectory on an underdetermined theory "spontaneously breaks the
         symmetry and commits to *one* of the tied corners," not an even
         blend - this assumption simply hadn't been checked against that
         precedent before relying on it. Fixed by averaging `readout` over
         an ensemble of independent random-restart settle trajectories
         instead of one; `ensemble=8` still crossed either decision
         threshold ~1.7% of the time (measured over 60 draws, std~0.07
         around the true 0.5) - exactly what produced a real failure 1/30
         seeds into an early end-to-end sweep - `ensemble=32` measured zero
         crossings with margin to spare.
      3. Offering *any* known-open neighbor as a move candidate regardless
         of whether it made progress let the agent thrash forever between
         `hall` and `entry` (an always-open door connects them, and a one-
         candidate list trivially "wins" `select_action` even when the move
         is pointless). Fixed by filtering move/probe candidates to rooms no
         farther from the goal than the current one (the same BFS table
         that builds the goal landscape's decay weights) - not *strictly*
         nearer, since `hall` and `annex` are both exactly one hop from
         `vault` (siblings around the goal, not nested); a strict-
         improvement-only filter was tried first and broke the detour-
         recovery scenario by excluding the `hall -> annex` move entirely.
      4. Even with candidates correctly filtered, `pragmatic_value`'s
         default `step_size=0.3` (a 30% nudge toward the candidate's effect
         vector) left the hypothetical state dominated by the *current*
         room's own strong self-similarity, making `move_to_vault` (0 hops)
         and `move_to_hall` (1 hop, lateral) score EFE=-0.4676 vs -0.4678 -
         a noise-level tie - from `annex`. A move decision asks "how good
         would it be to *be* at the destination," which needs a full step;
         `step_size=1.0` on the move `select_action` call cleanly separated
         the same pair to -0.999 vs -0.499.
- [x] Test suite (`tests/test_agent.py`): unit tests for the biconditional
      fix (explicit high *and* low sensor cases), the unsensed-door-near-0.5
      claim (checked, not assumed), goal-landscape distance preference,
      ontology-based deadend exclusion, and `missing_params` logic; end-to-
      end tests swept across 8 seeds each for all three scenarios (shortcut
      open, detour recovery, no path exists - **100% required, not
      documented-partial**, since this is new code with no prior single-
      seed-luck history to inherit) plus a determinism check and a thrash
      regression guard. First version of the thrash test was itself wrong -
      it counted every tick's room including ones where the agent stayed
      put to probe, not actual move transitions - caught and fixed before
      landing, not shipped broken. Full suite: 142 passed, 10 skipped
      (was 103), ~205s; `python -m zeuss demo` unaffected.
- [x] Tier4 program synthesis deliberately not used here, stated as a scoped
      v2 idea rather than silently dropped: `synthesize` operates over plain
      `dict`/`list` inputs with no representation for hypervectors or fuzzy
      valuations, so bridging it in needs its own domain-terms design (e.g.
      replacing `candidate_actions`'s `OPEN_THRESHOLD`/`CLOSED_THRESHOLD`
      logic with a small synthesized classifier) - a real, separate design
      problem, not glue that falls out of the existing APIs the way
      everything above did.


## v0.26 — the tier4 bridge: agent decisions synthesized, not hand-written

- [x] Closed the v0.25 stretch goal: `candidate_actions`'s open/closed belief
      classification is now two genetically-synthesized predicates
      (`is_door_open`/`is_door_closed`), not hand-written threshold
      comparisons - the first real bridge between tier4_synthesis and
      anything outside its own toy arithmetic/list-op targets. No existing
      tier module changed; `OPEN_THRESHOLD`/`CLOSED_THRESHOLD` stay as the
      source of truth for building the training examples, they just no
      longer gate the runtime decision directly.
- [x] First DSL framing tried and rejected outright, not just found
      unreliable: synthesize one `classify(belief) -> {-1,0,1}` function with
      the thresholds baked in as literal constants. Cannot work at all -
      `search.py`'s leaf-constant pool (`_LEAF_CONSTS = (0,1,2,3,True,False)`)
      has no way to produce a literal like `0.35`, so the search can never
      express the needed comparison. Fixed by passing the thresholds in as
      *input variables* instead of expecting the search to invent them as
      constants - the search then only has to discover the comparison
      *structure*.
- [x] Second framing tried and measured, not assumed reliable: one function
      computing all three classes at once
      (`If(belief < closed, -1, If(belief > open, 1, 0))`). Measured at only
      7/10 seeds even at a generous budget (400 population/100 generations/
      depth 4) - a two-threshold three-way decision boundary is a harder
      combinatorial target than it looks, the kind of thing this session's
      whole synthesis-hardening arc (v0.14-v0.24) exists to catch rather
      than assume away. Decomposed into two independent single-comparison
      predicates instead - each converges in 1-2 generations at a much
      smaller budget (population 200/generations 60/depth 2) and measured
      **30/30 seeds** for both. The two predicates can never actually
      contradict each other, since `OPEN_THRESHOLD > CLOSED_THRESHOLD` by
      construction - checked directly across a fine belief grid, not
      assumed from the thresholds not overlapping.
- [x] A real reliability trap found in the decomposed version, the same
      shape as the tier4 list-op audit's Class A failures (v0.22): training
      examples that skipped the exact threshold value and the `belief==0.0`
      edge (falsy in Python, letting an `If(belief, ...)`-shaped coincidental
      program slip through unnoticed) left ~1/10 seeds "verified" on a
      non-generalizing program - e.g. `belief > threshold` synthesized
      instead of `belief >= threshold`, indistinguishable on a training set
      with no point exactly at the threshold. Fixed the same way the list-op
      audit was fixed: added examples at the exact threshold and at 0.0/1.0,
      not new search-side machinery - remeasured at 30/30 after the fix.
- [x] Verified the swap is behavior-preserving, not just "also works": the
      full pre-existing 3-scenario x 8-seed agent suite (v0.25, 100% bar)
      passes unchanged after routing through the synthesized predicates
      instead of the raw comparisons, and `python -m zeuss agent`'s printed
      trace is bit-for-bit identical to the pre-bridge version. Added
      `tests/test_agent.py` coverage for the bridge itself: the predicates
      verify against their own training set, match the hardcoded thresholds'
      `>=`/`<=` semantics across the exact boundary (the precise failure
      mode found above), and are mutually exclusive across a fine grid.
      Full suite: 145 passed, 10 skipped (was 142); demo unaffected.


## v0.27 — `_bool_template`: a third structural template, for a real problem

- [x] The Gregorian leap-year rule (`year%4==0 and (year%100!=0 or
      year%400==0)`) was chosen as the first real, non-toy problem for this
      substrate's synthesis to solve. It reliably failed: the search
      converged to `not (year % 2)` ("is even"), a deceptive local optimum
      satisfying every training example except the two real century
      exceptions, with no smooth gradient to the true structure. Measured,
      not assumed, that this was a structural problem rather than a budget
      one: 0/5 seeds verified at population=300/generations=80/depth=4;
      *adding more* century-exception examples made it worse (0/8, some
      seeds not even reaching the "is even" energy); population=1000/
      generations=100 still converged to exactly the same "is even" energy.
      Same character of problem `_recursive_template` (v0.14) and
      `_fold_template` (v0.24) were each built to solve - so built a third
      structural template, `_bool_template`, the same kind of fix.
- [x] Design (properly planned via a dedicated Plan-mode session, not
      improvised): `BOOL_TEMPLATE_CATEGORIES` - `shape_kind` (`and2`, `or2`,
      `and_or3`, `or_and3` - structural, uniform, never resonance-biased, the
      same reasoning `combine_kind`/`base_kind` already establish) and three
      tunable, resonance-biased holes per atom (`modulus`, `cmp`, `const`).
      An atom is `(Var(var) % Const(modulus)) CMP Const(const)`; `var` is one
      shared name per template instance (like `_recursive_template`'s
      `seed_var`/`_fold_template`'s `list_name`), not per atom - the
      motivating target has one scalar input, so per-atom-distinct variables
      would be untested complexity it doesn't need. `and_or3` is the leap-
      year target's literal shape; `or_and3` is its standard boolean-algebra
      dual (`A∧(B∨C) ≡ (A∧B)∨(A∧C)`, specialized via `year%400==0 ⟹
      year%100==0`), giving the target two independent structural doors
      instead of one. Traced by hand against the exact target before
      committing to the design: `and_or3` with atoms `(4,"==",0),
      (100,"!=",0),(400,"==",0)` builds *exactly* the target formula.
      `NOT` deliberately excluded - `!=` already negates where it matters,
      and `UnaryOp("not", ...)` is already reachable by ordinary growth.
      New opt-in flag `allow_bool_template` (default `False`), gated the way
      `allow_recursion` gates its own template (not fold's self-selecting
      `if list_inputs`) - bool template's activation condition ("has a
      scalar input") is true for almost every call, so leaving it
      unconditionally live would waste `template_rate` draws on and/or
      skeletons for plain arithmetic targets, the same diversity-dilution
      concern that made `allow_recursion` itself opt-in.
- [x] One genuinely new wrinkle vs. the other two templates: `modulus`/`cmp`/
      `const` are reinforced once *per atom* (2-3x per individual per
      generation) since 2-3 atoms share the same category names, unlike
      `shape_kind`'s 1x - built the straightforward way first and validated
      by the reliability sweep rather than pre-guessing a correction was
      needed.
- [x] Confirmed a true no-op for every existing caller before trusting the
      new mechanism at all: full `pytest -q` unchanged (145 passed, 10
      skipped) with `allow_bool_template=False` (the default) - the new
      checks add zero behavior change for anything that doesn't opt in.
- [x] The actual measurement, not just "a template now exists": with
      `allow_bool_template=True`, the *same* budget already measured failing
      for blind search (population=300/generations=80) went from 0/8 to
      4/8 verified, 2/8 generalizing - real movement, but a real remaining
      problem, diagnosed rather than declared good enough. Two further
      rounds of diagnosis, each finding a genuine issue and fixing it with
      more information, not more search machinery:
      1. Two seeds "verified" on coincidental formulas using `year % 5` or
         `year % 7` in place of `year % 100` - since every century-exception
         year in the 2-example training set happened to also be divisible by
         5, "not divisible by 5" coincidentally stood in for "not a century
         year" until a real leap year that's *also* divisible by 5 but isn't
         a century year (2020, 1980) exposed it. The same Class A lesson the
         v0.22 list-op audit already found, surfacing again through a richer
         modulus vocabulary than earlier targets had access to.
      2. Even after adding those, other seeds found similar coincidences
         (`year % 7`-based) fitting the *specific two* century-exception
         years (1900, 1800) in training while failing on real century years
         not included (1700, 2100, 2200). Fixed by adding four more real
         century-exception years (1700, 2100, 2200, 2300) spanning varied
         mod-3/5/7 residues, so no single small-modulus coincidence fits all
         of them the way it could fit two.
      With both fixes: **generations raised to 150 (same population=300)
      reached 8/8** on the seeds used for the committed reliability test,
      converging in single-digit generations on most seeds (vs. 80
      generations timing out with no fix). A wider 30-seed validation sweep
      found one further failure (seed 19): a genuine `AND(year%4==0, NOT
      year%100==0)` local optimum - correctly handles every training example
      except the three div-by-400 exceptions, a strong 2-atom (`and2`)
      family optimum competing against the 3-atom family needed for the full
      rule. Population=500 fixed seed 19 - but moved the identical failure
      to two *different* seeds instead (2 and 18) - the same whack-a-mole
      shape this project's history already has several examples of (v0.16's
      Fibonacci example count, v0.20's hole-mutation rate). Kept
      population=300/generations=150 (29/30 on the wide sweep, 8/8 on the
      committed seed range) rather than chasing 30/30 by continuing to move
      the same residual gap around - recorded as an honest, understood
      limitation, the same way seed 8 and `sum_via_fold` seed 3 already are.
- [x] A real bug found only by running the actual CLI end-to-end, not by the
      150+ passing unit/integration tests (every one of which imports
      `synthesize` from `search.py` directly): `tier4_synthesis/synth.py` is
      a separate, curated "public entry point" wrapper that whitelists which
      `search.synthesize` parameters it forwards - adding
      `allow_bool_template` to `search.py` and to every direct test/call site
      did not make it reach `cli.py`'s actual `zeuss synth` command, which
      imports the wrapper. `python -m zeuss synth` raised `TypeError:
      synthesize() got an unexpected keyword argument 'allow_bool_template'`
      with the full test suite green. Fixed by threading the parameter
      through the wrapper too, and added
      `test_synth_wrapper_exposes_allow_bool_template` - a dedicated test
      that imports from the wrapper specifically, not `search`, so a new
      `search.py` parameter never reaching the curated public entry point
      can't silently recur. `CLAUDE.md`'s "python -m zeuss demo still runs"
      definition-of-done item is exactly what this session's own habit of
      actually running the CLI (not just trusting a green test suite) is
      for - this bug is the concrete case where that habit caught something
      151 passing tests did not.
- [x] Added `python -m zeuss synth` Scenario 5 (the leap-year rule, printing
      the discovered formula and checking it against real held-out years);
      `tests/test_synthesis.py` unit tests for the one genuinely novel/risky
      piece of extraction logic this template needed (disambiguating `and2`
      from `and_or3`, `or2` from `or_and3` - neither the recursion nor fold
      template needed anything like this, and neither has its own unit-level
      round-trip tests, this module's own established convention of
      validating templates end-to-end rather than at the unit level);
      `test_synthesize_recovers_leap_year_rule` and
      `test_leap_year_rule_is_reliable_across_seeds` (8/8, mirroring
      `test_resonant_bias_discovers_recursion_reliably_across_seeds`/
      `test_list_ops_are_reliable_across_seeds`). Full suite: 151 passed, 10
      skipped (was 145); `python -m zeuss demo` unaffected.


## v0.28 — motif resonance: a general alternative to hand-built templates,
## tried honestly, and not (yet) a replacement

- [x] Direct pushback, raised against this project's own stated preference
      (`CLAUDE.md`: prefer a rule that *emerges* from the dynamics over a
      design that special-cases a symbolic path): three times running
      (`_recursive_template` v0.14-v0.21, `_fold_template` v0.24,
      `_bool_template` v0.27), a new synthesis target that reliably failed
      under blind mutation/crossover got fixed by a human hand-designing a
      whole-skeleton template with named holes. That doesn't scale to every
      future problem shape, and is itself the kind of special-cased
      symbolic path this project says to avoid. Asked to find (or build) a
      general alternative instead of a fourth bespoke template.
- [x] The general category already exists in the GP literature -
      Probabilistic Incremental Program Evolution (Salustowicz &
      Schmidhuber 1997) and module-acquisition/ADFs (Koza 1994; Angeline &
      Pollack) both let a population's own fitness signal shape what
      structure gets grown next, instead of a human pre-declaring it. Rather
      than import either wholesale, adapted the idea onto this project's own
      substrate: `ResonantBias` (`resonance_bias.py`) is already a fully
      generic EDA - parameterized by an arbitrary `categories: dict[str,
      list]`, silently skipping unknown reinforcement keys, re-reading
      `categories[cat]` fresh on every `sample()` call - confirmed by
      reading the whole file before designing anything, not assumed from
      its docstring. It already gets constructed three separate times for
      three hand-picked category dicts; the missing piece was never a new
      EDA, just a mechanism that discovers *what the categories and options
      should be* from the population itself.
- [x] Built `tier4_synthesis/motif_bias.py`: `MotifArchive` wraps one
      `ResonantBias` whose `categories[context]` starts empty and grows at
      runtime as distinct subtrees are observed - `ResonantBias` needed no
      changes to support this. A "motif" is any subtree found at a
      `(parent_kind, child_slot, depth_bucket)` structural context (see
      `context_key`/`collect_motifs`); every generation,
      `reinforce_from_population` walks the scored population (bounded to 3
      levels per individual) and registers every subtree it finds, weighted
      by `exp(-energy)` - the exact reinforcement convention every existing
      template's bias already uses, just applied to whole subtrees keyed by
      position instead of named hole-fillers keyed by a hand-declared
      category. A small per-context capacity (20 motifs, lowest-weight
      evicted) keeps the archive bounded across 150 generations x hundreds
      of individuals.
- [x] Wired into `_grow` itself (search.py), not just at the top-level entry
      point the way templates are checked: at *every* grow site, before
      building fresh, check whether the archive already has a reinforced
      subtree at this exact context and splice a copy in instead - the
      generalization of a template's "known-good skeleton" idea, except the
      skeleton is discovered from the population's own history instead of
      hand-written. Guarded so `allow_motif_bias=False` (the default) never
      spends an RNG draw on the check at all, keeping every existing
      caller's draw sequence bit-for-bit identical - confirmed directly
      (`test_motif_bias_is_a_true_noop_by_default`), not just inferred from
      "the rest of the suite still passes" (true, but none of the other 152
      tests exercise the new parameters at all, so they couldn't have caught
      a future refactor moving the guard).
- [x] Also added `allow_fold_template`/`allow_recursion_template` (both
      default `True`, so no existing behavior changes): `_bool_template`
      already had a kill switch (`allow_bool_template`), but `_fold_template`
      had none (self-selecting on `list_inputs` alone) and
      `_recursive_template`'s activation was entangled with `allow_recursion`
      itself (which also controls whether `letrec`/`recur` are in
      `_choose_kind`'s pool at all). Needed so motif resonance could be
      measured on the *same* target with the corresponding hand template
      switched off, rather than the two mechanisms always competing for the
      same `template_rate` draw.
- [x] Deliberately does not remove or replace any of the three existing
      templates - ripping out hard-won, currently-green reliability
      (v0.14-v0.27's whole arc) in favor of an unproven generic mechanism
      would be reckless. Shipped as a fourth, independent, default-off path,
      measured before any claim was made about it.
- [x] The measurement - the actual deliverable, not the mechanism by itself.
      Re-ran the three targets that motivated each existing template, with
      that template disabled and `allow_motif_bias=True` instead, at the
      exact committed budgets/seeds already in `tests/test_synthesis.py`:
      - Leap year (`allow_bool_template=False`, budget/seeds from
        `test_leap_year_rule_is_reliable_across_seeds`, widened to 10
        seeds): **0/10 verified**. Every seed converges to the identical
        `not (year % 2)` "is even" trap plain blind growth hits with no
        template at all.
      - `sum_of_squares_via_fold` (`allow_fold_template=False`, budget/seeds
        from `test_synthesize_recovers_sum_of_squares_via_fold`, 8 seeds):
        **1/8 verified and generalizing** - seed 7 rediscovered the exact
        genuine fold shape from scratch via motif reuse in 4 generations, a
        real positive signal - but 1/8 is *worse* than blind growth's own
        historical 3/8 baseline at this identical budget (v0.22's audit).
      - `2**n` (`allow_recursion=True`, `allow_recursion_template=False`,
        budget/seeds from
        `test_resonant_bias_discovers_recursion_reliably_across_seeds`, 8
        seeds): **1/8 "verified", 0/8 generalizing** - the one verified seed
        found a coincidental non-recursive expression fitting the 5 training
        examples by luck, the exact verified-but-wrong shape this project's
        whole methodology exists to catch, not genuine recursion discovery.
- [x] Diagnosed why, rather than stopping at the numbers: motif resonance
      can only reinforce and reuse structure that has already appeared
      *somewhere* in the population with a competitive energy - it has no
      way to independently invent a compound shape that essentially never
      spontaneously forms under blind growth in the first place, which is
      exactly why each of the three hand-built templates was needed
      (`_bool_template`'s own comment: blind growth reliably misses the
      3-atom AND/OR nesting; `_recursive_template`'s docstring: "under 5% of
      random depth-4 trees even contain a `Letrec` with an `If`-shaped
      body"). The very first generation is 100% blind growth, so if the
      correct top-level shape never appears there, the archive has nothing
      genuine to discover before the population commits to a coincidental
      local optimum instead - and once it does, motif resonance reinforces
      *that* structure's pieces just as readily as it would a correct one.
      `sum_of_squares_via_fold`'s partial, real success fits this exactly: a
      `Fold` node is common under blind growth (unlike a 3-atom boolean
      formula or a `Letrec`), so the hard part there is a comparatively
      small "which transform" choice inside an already-common skeleton -
      closer to what a subtree-reuse mechanism is actually suited for than
      inventing a rare top-level shape from nothing.
- [x] Honest scope decision: the three hand-built templates stay as the
      deployed answer for their own targets. Motif resonance ships as a
      real, fully generic, zero-regression opt-in - not a claim of matching
      what `_recursive_template`/`_fold_template`/`_bool_template` eventually
      reached after several dedicated versions each (v0.14-v0.21 alone took
      seven versions to reach today's recursion reliability). A follow-up
      that wanted to close this gap would likely need the same kind of
      iterative hardening those templates got (per-context/per-family
      elitism so an early bad motif can't starve a better one out, a
      reuse-vs-full-regrowth balance analogous to
      `_TEMPLATE_HOLE_MUTATION_RATE`, stagnation-triggered archive resets) -
      not attempted this session, recorded as the natural next step rather
      than assumed unnecessary.
- [x] New tests: `test_motif_archive_registers_samples_and_evicts` and
      `test_collect_motifs_yields_contexts_matching_grow_sites` (unit-level,
      independent of whether the mechanism helps any target),
      `test_motif_bias_is_a_true_noop_by_default` (pinned-seed proof, not
      just inference from the rest of the suite passing),
      `test_synth_wrapper_exposes_allow_motif_bias` (mirrors v0.27's own
      wrapper-forwarding regression guard), and
      `test_motif_bias_does_not_yet_solve_leap_year_without_bool_template`
      (the honest negative result, with the full three-target measurement
      and diagnosis recorded in its docstring - mirrors
      `test_recursion_synthesis_is_safe_but_not_reliably_found`'s established
      pattern for this project's negative results). No CLI scenario added -
      unlike every prior template, this mechanism didn't meaningfully solve
      any of its three motivating targets, so there is nothing to
      demonstrate yet.


## v0.29 — grammar resonance v2: production-level PCFG bias + persistence

- [x] Direct follow-up to v0.28, per user pushback on that entry's own
      result: motif resonance (whole-subtree reuse) failed on exactly the
      targets whose difficulty is a compound top-level shape that
      essentially never spontaneously assembles under blind growth - it can
      only propagate a shape some individual already built by luck, and if
      that never happens there is nothing genuine to discover. The user
      identified a materially different mechanism instead: bias **every
      production choice** `_grow`/`_leaf` already make (which `BinOp`
      operator, which `UnaryOp` operator, `Var` vs `Const`, which constant)
      independently, conditioned on *where* in the tree the choice is made -
      a PCFG-style Estimation-of-Distribution search (Probabilistic
      Incremental Program Evolution, Salustowicz & Schmidhuber 1997). A
      compound shape can then assemble from marginal pushes on separate
      choice-points that never co-occurred in any single ancestor, which
      doesn't need the whole shape to have already existed once - a
      genuinely different mechanism from v0.28's, not a rehash. Second half
      of the request: persist that bias **across calls to `synthesize`**,
      not reset every run, so a future target benefits from resonance
      structure already built up by every problem solved before it.
- [x] User-confirmed scope going in (via two explicit design questions, not
      assumed): bias only the **fixed-option-set** choices first
      (`binop_op`, `unaryop_op`, `leaf_kind`, `leaf_const` - all already
      exactly what `_grow`/`_leaf` draw from today, needing zero
      `ResonantBias` changes), deferring the harder "which node kind to
      grow" choice-point (whose valid option pool varies per call) as a
      well-scoped follow-up; and persistence via an **explicit, caller-
      opted-in save/load helper**, not a hidden in-process singleton - a
      singleton would risk test-order-dependent behavior, a real hazard
      given how much this suite's reliability rests on pinned-seed
      determinism.
- [x] Context key: `(choice_name, parent_kind, child_slot)`, deliberately
      **without depth** - unlike `motif_bias.py`'s `context_key`. Keying by
      choice-point identity alone (e.g. "which `binop_op` under an `And`'s
      right slot") lets reinforcement accumulate across every depth that
      relative position occurs at, instead of fragmenting into near-empty
      per-depth buckets - the user's specific critique of a depth-keyed
      context, and what makes gradual, depth-independent joint assembly
      possible at all.
- [x] Built `tier4_synthesis/grammar_bias.py`: `GrammarBias` wraps one
      `ResonantBias` (constructed with a `choice_vocab` dict supplied by
      `search.py`, avoiding both a circular import and a second, driftable
      copy of `_BINOPS`/`_UNARYOPS`/`_LEAF_CONSTS`); `collect_production_
      choices` walks a tree yielding every choice found (unbounded depth -
      cheap now, since this only categorizes one node at a time, no subtree
      copies unlike `collect_motifs`); `save_grammar_bias`/`load_grammar_
      bias` persist via `pickle` (nested dicts of `str -> complex128
      ndarray` plus a plain categories dict - no alignment bugs between
      parallel name/vector arrays the way a hand-rolled npz+json split
      would risk).
- [x] Two small, additive changes to existing tier-2/tier-4 code, both
      required for persistence to be *correct*, not just convenient -
      verified by reading `Codebook` before assuming otherwise, not
      assumed: `Codebook.symbol(name)` is **stateful and call-order-
      dependent** (draws from one evolving RNG stream lazily), so
      reconstructing `Codebook(dim, seed=same)` fresh on load does *not*
      reproduce the same per-name vectors unless names are requested in the
      exact original order - confirmed directly
      (`test_codebook_items_and_load_round_trip`: same seed, opposite
      request order, provably different vectors). Added `Codebook.items()`/
      `Codebook.load()` to persist/restore the actual vectors, sidestepping
      the order-dependence entirely, and `ResonantBias.has_evidence()` (a
      public equivalent of checking `_accum` from outside the class).
- [x] Wired into `_grow`/`_leaf`/`random_program`/`_replace_at_scoped`/
      `mutate`/`synthesize` reusing v0.28's exact established pattern - a
      `grammar_bias` entry added to `_grow`'s existing pass-through `kwargs`
      dict meant every recursive call site needed zero further edits beyond
      the two actual use-sites (`binop`/`unaryop` op draws, plus `_leaf`'s
      `var`-vs-`const` and which-`const` draws). `allow_grammar_bias`
      (default `False`) is the same true-no-op convention every other flag
      in this module already has - confirmed directly
      (`test_grammar_bias_is_a_true_noop_by_default`), not just inferred
      from "the rest of the suite still passes."
- [x] The one genuinely new piece beyond v0.28's pattern: `grammar_bias` may
      be **caller-supplied already-populated**, not always constructed
      fresh inside `synthesize` like every other bias. `grammar_bias_owned`
      tracks whether *this call* constructed it (caller passed `None`) -
      only a self-constructed one is ever reset on the existing stagnation-
      triggered restart or discarded at call end; a caller-supplied one is
      caller-owned, reinforced in place, and survives both, including
      across multiple separate calls to `synthesize`
      (`test_grammar_bias_persists_across_synthesize_calls`). No change to
      `synthesize`'s `(best_node, beta_trace, verified)` return contract -
      a caller who wants to persist already holds their own reference to
      the object they passed in.
- [x] The measurement - the actual deliverable, not the mechanism alone.
      Re-ran the exact three controlled comparisons v0.28 already ran, same
      budgets/seeds, `allow_grammar_bias=True` instead of
      `allow_motif_bias=True`, with the corresponding hand template still
      disabled:
      - `sum_of_squares_via_fold` (`allow_fold_template=False`, 8 seeds):
        **4/8 verified and generalizing.** Genuinely better than blind
        growth's own historical 3/8 baseline at this exact budget (v0.22's
        audit) *and* better than motif resonance's 1/8 on the same target
        (v0.28) - real, measured lift. A `Fold` node is already common
        under blind growth (weight 3 in `_KIND_WEIGHTS_WITH_LIST`), so the
        hard part here is a comparatively small "which transform" choice -
        exactly the shape of problem per-choice marginal reinforcement is
        suited for.
      - Leap year (`allow_bool_template=False`, 10 seeds): **0/10** -
        identical `not (year % 2)` trap to motif resonance's own result.
      - `2**n` (`allow_recursion=True`, `allow_recursion_template=False`, 8
        seeds): **0/8 verified** - worse than motif resonance's 1/8-but-
        non-generalizing result.
- [x] Diagnosed the two failures, not just reported them - and found they
      share one root cause with v0.28's own failures, arrived at by a
      different route. Blind growth converges to `not (year % 2)` (leap
      year) or coincidental non-recursive arithmetic (`n*n`, `2**n`) *fast*,
      so once that wrong structure dominates the population,
      `reinforce_from_population` ends up reinforcing *that* structure's
      own production choices more than anything else - the bias actively
      pushes *harder* toward reproducing the wrong answer instead of
      escaping it. This is the same premature-lock-in failure the three
      hand-built templates all needed dedicated machinery to avoid
      (per-family elitism keyed on a structural axis - `combine_kind`/
      `base_kind`/`shape_kind` - deliberately excluded from resonance
      biasing for exactly this reason, see `_recursive_template`'s and
      `_bool_template`'s own docstrings). Neither general mechanism built
      so far (v0.28's or this one) has an analogous safeguard - that
      absence, not anything specific to subtree-reuse vs. production-level
      biasing, is the more likely shared root cause of both mechanisms'
      failures on these two targets.
- [x] Net finding, stated precisely rather than rounded to "better" or
      "worse": motif resonance (v0.28) and grammar resonance (this entry)
      are not simply ranked - each helps on a *different* target shape.
      Grammar resonance materially helps the fold target motif resonance
      didn't (4/8 vs. 1/8); motif resonance found one (non-generalizing)
      recursive-looking coincidence grammar resonance didn't. Neither helps
      the two targets whose difficulty is a rare top-level construct
      essentially never appearing under blind growth in the first place
      (compound 3-atom boolean nesting; `Letrec`/`Recur` at all). Neither
      replaces `_recursive_template`/`_fold_template`/`_bool_template`.
- [x] New tests: `test_codebook_items_and_load_round_trip` (in
      `test_hypervectors.py`, proving the order-dependence claim directly
      rather than asserting it), `test_resonant_bias_has_evidence`,
      `test_grammar_bias_registers_and_samples`, `test_collect_production_
      choices_yields_contexts_matching_grow_sites`, `test_grammar_bias_
      save_and_load_round_trip` (proves sampling behavior actually
      survives a round trip, not just "deserializes without error"),
      `test_grammar_bias_is_a_true_noop_by_default`, `test_synth_wrapper_
      exposes_allow_grammar_bias`, `test_grammar_bias_persists_across_
      synthesize_calls`, and the two honest measured-result tests -
      `test_grammar_bias_improves_sum_of_squares_via_fold_over_baseline`
      (the positive case, with the full three-target measurement and
      diagnosis recorded in its docstring - seeds 4 and 7 from the 8-seed
      sweep) and `test_grammar_bias_does_not_yet_solve_leap_year_or_pow2`
      (the negative case, mirroring v0.28's own negative-result test
      pattern). No CLI scenario added - the same standard v0.28 set: only
      demo a mechanism once it's shown reliable enough to be worth showing,
      and 4/8 isn't there yet.


## v0.30 — shape elitism: honest negative result, and why

- [x] Follow-up to v0.28/v0.29's shared diagnosis: both bias mechanisms'
      failures on leap year/`2**n` were traced to the same premature-
      lock-in pattern the three hand-built templates already avoid via
      per-family elitism (`best_template_node`/`best_fold_template_node`/
      `best_bool_template_node`) - once blind growth converges to a wrong
      structure fast, `exp(-energy)` reinforcement amplifies *that*
      structure's own choices/subtrees more than a rarer, correct one's.
      `allow_shape_elitism` (`search.py`) generalizes the same protected-
      slot-plus-guaranteed-refinement-offspring machinery to *any*
      individual via an automatically-derived family signature
      (`root_shape` - the root node's DSL type and operator, no hand-
      declared vocabulary), rather than a template's own named hole-
      choices - already implemented and committed, but never measured.
      Measured this session, not assumed.
- [x] Re-ran the same three controlled comparisons v0.28/v0.29 already ran,
      same budgets/seeds, three conditions (`allow_shape_elitism` alone,
      `+allow_motif_bias`, `+allow_grammar_bias`), corresponding hand
      template disabled:
      - Leap year (10 seeds): **0/10 verified** in all three conditions -
        identical to v0.28/v0.29's own result.
      - `sum_of_squares_via_fold` (8 seeds): **1/8** alone (matches blind
        growth's own baseline, not an improvement), **0/8** combined with
        either bias - *worse* than either bias alone (v0.28: 1/8, v0.29:
        4/8), i.e. adding shape elitism on top actively hurt the one target
        where a bias mechanism had shown real lift.
      - `2**n` (8 seeds): **0/8** alone, **1/8 verified but non-
        generalizing** combined with motif bias, **0/8** combined with
        grammar bias - no case beats either bias's own unassisted result.

      Zero improvement anywhere it was tried, and one real regression
      (fold - shape elitism competing with grammar bias's own protected
      slot for population share, worse than either alone).
- [x] Diagnosed why, not just reported: shape elitism (like the five
      mechanisms before it) only ever protects/reinforces an individual
      that blind growth already produced. `_bool_template`'s own comment
      already says why that's fatal here: blind growth reliably misses the
      3-atom AND/OR nesting in the first place. If the correct shape
      never appears in any generation of any attempt, `best_shape_node`
      has nothing to ever populate for that family - there is no rare
      individual for elitism to rescue, because it was never generated to
      begin with. This is a *generative* coverage problem, not a
      *selective* one, and no mechanism that only acts after generation
      (six of them now, across v0.20-v0.30) can touch it.
- [x] This is the pivot point for v0.31: the fix has to act *at*
      generation time instead.
- [x] New tests: `test_root_shape_signature` (direct unit check of the
      automatically-derived family signature), `test_shape_elitism_is_a_
      true_noop_by_default` (checked directly, not inferred - the
      established practice for every true-no-op claim in this file), and
      `test_synth_wrapper_exposes_allow_shape_elitism`. No dedicated
      negative-result target test added (unlike v0.28/v0.29) - the honest
      negative result here is a *combination* measurement (three
      conditions x three targets) rather than a single representative
      seed, and is recorded here rather than as a committed test that
      would need to re-run all nine combinations on every suite run.


## v0.31 — semantic backpropagation: generation-time construction fixes leap year

- [x] Direct pivot from v0.30's diagnosis, per user pushback: "nothing we
      have done or attempted has fixed the leap year or other problems" -
      asked to step back from the whack-a-mole pattern (six mechanisms
      across v0.20-v0.30, all in the same "reinforce/protect/reshuffle
      whatever blind growth already produced" family, all landing on the
      identical 0/10 leap-year wall) and find something categorically
      different, general across future targets rather than another
      per-issue patch.
- [x] Root-cause framing, stated precisely: every one of the six prior
      mechanisms operates *after* generation. None change what blind
      top-down random growth actually *produces*. This project's own
      docstrings already said why that's fatal for leap year:
      `_bool_template`'s comment records that blind growth reliably misses
      the rare 3-atom AND/OR nesting in the first place - so there is
      nothing for reinforcement/elitism/restart to ever find and protect.
      The fix needed to act *at* generation time, and be shape-agnostic (no
      hand-declared vocabulary like `BOOL_TEMPLATE_CATEGORIES["shape_kind"]`,
      or it's just a fourth per-target template).
- [x] Built `tier4_synthesis/semantic_bias.py`: semantic backpropagation /
      goal decomposition (well-established in program synthesis), scoped
      honestly to compound *boolean* targets only (numeric/recursion
      backprop - e.g. reading `y_n / y_(n-1) == 2` directly off the given
      `(n, 2**n)` pairs - is a structurally analogous but separate problem,
      not attempted here; a clear follow-up, not silently dropped). Given a
      per-example target vector (`bool | None`, `None` = don't-care), grows
      the boolean sub-grammar (`and`/`or`/`not`/comparison-atom) by
      decomposing that target through the connective's *real truth-table
      semantics* instead of growing blind and hoping:
      - `and_left_target`/`or_left_target`: the hard constraint provable on
        whichever child is grown first (`AND` needs both `True` to reach
        `True`; `OR` needs both `False` to reach `False`); everywhere else
        is don't-care, since the *other* child could still satisfy that row.
      - `and_right_target`/`or_right_target`: computed only *after* the left
        child is actually grown and evaluated against every example
        (`AND(True, r)=r`/`OR(False, r)=r` - this is what resolves the
        ambiguity the left-side split alone can't).
      - `not_target`: elementwise negation, no don't-cares.
      - `best_atom_for_target`: the base case - searches
        `var_pool x BOOL_TEMPLATE_CATEGORIES["modulus"/"cmp"/"const"]` (the
        *same* vocabulary `_bool_template` already uses, passed in by the
        caller like `GrammarBias` takes `choice_vocab`, avoiding both a
        circular import and a second, driftable copy) for the atom that
        matches the most non-don't-care rows. Vectorized with `numpy` over
        each example's variable value rather than calling `evaluate` per
        candidate, so the ~300-combination search stays cheap regardless of
        example-set size. When every row is don't-care (a rarer-than-
        expected but real case - an AND/OR ancestor already proved this
        whole subtree's value can't affect correctness), returns a
        uniformly random atom rather than `None`: found and fixed during
        this session's own smoke-testing - without it, one indifferent leaf
        anywhere in the tree discarded the *entire* candidate over a
        position nothing depended on, wasting the draw for no reason (see
        `test_best_atom_for_target_all_dont_care_still_returns_a_node`).
      - The `and`/`or`/`not` kind draw at each node is deliberately
        fixed/uniform, never resonance-biased (`_KIND_WEIGHTS`) - the same
        established caution already applied to `shape_kind`/`combine_kind`/
        `base_kind`: a structural/family choice risks premature commitment
        to the wrong family from early noise, the exact failure this whole
        mechanism exists to route around.
- [x] Wired in as a *seventh* opt-in (`allow_semantic_bias`, default
      `False`), competing at the same `template_rate` gate the three
      existing templates already use, immediately after `_bool_template`'s
      own check, in `random_program`/`_replace_at_scoped`/`mutate`/
      `synthesize` - the same insertion point every hand-built template
      already occupies, not a new plumbing paradigm. The one genuinely new
      requirement: `random_program`/`_replace_at_scoped`/`mutate` never
      received `examples` before this - they were purely structural
      generators, blind to the actual training data, which is exactly the
      blindness being fixed. Threaded through as `examples`/`fuel_budget`
      alongside the flag. `_grow`/`_leaf` themselves are untouched - this
      mechanism only competes at the whole-node template-check level, the
      same level the three existing templates already operate at.
- [x] The measurement - the actual deliverable, not the mechanism alone.
      Same leap-year budget/seeds as `test_leap_year_rule_is_reliable_
      across_seeds` (population=300, generations=150, `allow_bool_
      template=False` so `_bool_template` can't be doing the work):
      **30/30 seeds verified** (widened past the committed 8-seed range to
      also re-check seed 19 - the one genuine residual failure recorded
      against `_bool_template` itself in the v0.20-era audit - now fixed
      too) and **27/30 verified and generalizing** to the held-out years.
      Total wall time for all 30 seeds: ~18s - dramatically faster than any
      prior mechanism (each of v0.28/v0.29/v0.30's individual seeds alone
      took 8-50+ seconds), since a successful draw often solves the target
      in the very first generation rather than needing the full budget.
      Zero regression to any other target (`allow_semantic_bias` only ever
      activates when every example's `expected_output` is a genuine `bool`
      - checked directly via `test_boolean_backprop_template_returns_none_
      for_non_boolean_targets` - so fold/`2**n`/arithmetic targets are
      completely unaffected).
- [x] The 3/30 non-generalizing seeds are an honest, expected caveat, not
      swept under the rug: e.g. seed 27 finds `(year % 2 == 0) and (not
      (year % 100 <= 6) or year % 400 < 5)` - a coincidental century clause
      (`year % 100 <= 6` instead of the true `year % 100 == 0`) that
      happens to match every training century-year but misclassifies the
      held-out 1904 - the same "verified means matched the given examples,
      never proven correct" honesty this module's own statement (`synth.py`)
      already makes, the same character of caveat `test_resonant_bias_can_
      discover_fibonacci`'s seed-7 case already has on record. Blind draws
      can still produce large (up to ~50-node), logically-valid-but-
      overfit trees when the `and`/`or`/`not` kind draw recurses several
      times before bottoming into atoms - population-level parsimony
      pressure and mutation/crossover, already part of `synthesize`, are
      what's expected to shrink these toward the true minimal shape over a
      real run, not a claim that every single draw is already minimal.
- [x] Net finding: this is the first of the seven general mechanisms tried
      across v0.28-v0.31 that actually fixes the leap-year failure class,
      confirming the diagnosis that the bottleneck was generative (blind
      growth essentially never producing the correct shape at all) rather
      than selective (needing better reinforcement/protection of individuals
      already produced). Scoped honestly: this only covers boolean compound
      targets. `2**n`/recursion's failure is the same *character* of
      problem (a rare shape blind growth essentially never produces) but a
      different mechanism would be needed (numeric relationship-mining over
      the example table, e.g. discovering `y_n / y_(n-1) == 2` directly from
      the given pairs) - not attempted this session, a clear next step.
- [x] New tests: `test_and_or_not_target_decomposition` (direct truth-table
      checks of the decomposition functions, independent of any search
      run), `test_best_atom_for_target_finds_exact_match` (including the
      don't-care-rows-are-excluded-not-treated-as-False case),
      `test_best_atom_for_target_all_dont_care_still_returns_a_node` (the
      bug found and fixed during this session, see above),
      `test_grow_boolean_targeted_reconstructs_a_known_and_or3_formula`
      (round-trip check against a *known* formula, independent of the full
      GP search), `test_boolean_backprop_template_returns_none_for_non_
      boolean_targets`, `test_semantic_bias_is_a_true_noop_by_default`
      (checked directly, not inferred), `test_synth_wrapper_exposes_allow_
      semantic_bias`, and the measured-result test `test_semantic_bias_
      solves_leap_year_reliably_across_seeds` (asserts the 8/8-verified-
      and-generalizing subset at the existing test's own seed range, with
      the full 30-seed/27-generalizing sweep recorded in its docstring -
      the same "assert a representative subset, record the full sweep"
      convention every other reliability test in this file already uses).
      No CLI scenario added yet - a natural next step now that this
      mechanism has actually demonstrated the reliability the others
      didn't.


## v0.32 — numeric relationship-mining: the same fix for recursion

- [x] Direct follow-up to v0.31's own scoping note, per explicit user
      request ("continue with numeric relationship-mining over the example
      table"): semantic backpropagation fixed leap year by acting at
      generation time using the actual example data instead of
      reinforcing/protecting whatever blind growth produced, but was
      scoped to boolean compound targets only. `2**n`/Fibonacci have the
      same *character* of failure (`_recursive_template`'s own docstring:
      "under 5% of random depth-4 trees even contain a `Letrec` with an
      `If`-shaped body") but need a different, arithmetic-native
      technique - reading the recurrence directly off the given
      `(param, output)` table, the way a human would spot one from a value
      table (compute `y_n / y_(n-1)`, notice it's constant).
- [x] Built `tier4_synthesis/numeric_bias.py`: `mine_recursion_choices`
      searches every `(combine_kind, step, delta, op)` combination
      (`param_recur`/`double_recur` x step in `{1,2}` x delta in `{0,1}` x
      five arithmetic ops) for an *exact* fit against the table - every row
      a candidate can check (both its needed offsets present in the table)
      must match exactly, not a best-effort score, and at least 3 rows must
      be checkable (guards against a spurious "fit" on a table too small to
      distinguish a real relationship from luck). The base case
      (`base_kind`/`base_val`/`cmp`/`base_const`) is then derived from
      whichever rows the winning recurrence *couldn't* check at all - valid
      only when those rows form a contiguous prefix from the table's
      smallest value, the one condition that makes `param <= base_const`
      actually generalize to inputs *outside* the table too. Deliberately
      reuses `search.py`'s own `_build_template_node`/`TEMPLATE_CATEGORIES`
      schema (passed in by the caller, avoiding a circular import) rather
      than building its own tree - a mined result is *literally* a
      `_recursive_template`-shaped `Letrec`, so every existing mechanism
      that already operates on that shape (`extract_template_choices`,
      hole mutation, per-family elitism, `ResonantBias` reinforcement)
      applies to it automatically, with zero changes needed anywhere else
      in `search.py` beyond the two generation entry points.
- [x] Wired in as an eighth opt-in (`allow_numeric_bias`, default `False`,
      true no-op), checked immediately after `_recursive_template`'s own
      draw in `random_program`/`_replace_at_scoped` (own independent
      `template_rate` roll) - reusing the `examples`/`fuel_budget`
      plumbing v0.31 already added to these functions, no further new
      plumbing needed. `mutate()` needed no changes at all - a mined node
      is structurally indistinguishable from one `_recursive_template`
      itself could have built, so `mutate`'s existing hole-mutation and
      full-regrowth paths already handle it correctly.
- [x] The measurement - the actual deliverable. Same configurations as
      `test_resonant_bias_discovers_recursion_reliably_across_seeds`/
      `test_resonant_bias_can_discover_fibonacci` (population=800,
      generations=150, fuel_budget=200), `allow_recursion_template=False`
      so `_recursive_template`'s own draw can't be doing the work:
      - `2**n` (8 seeds): **8/8 verified and generalizing** (held out to
        n=6,7,8).
      - Fibonacci, 0-indexed, same 7-example table (8 seeds): **8/8
        verified and generalizing** (held out to n=7,8,9) - including
        seeds 4 and 7, the two residual failures already on record for the
        existing template+`ResonantBias` mechanism (seed 4 never verified;
        seed 7 "verified" with a coincidental, non-generalizing expression
        - `test_resonant_bias_can_discover_fibonacci`'s own docstring).
      - Both sweeps together ran in about 4 seconds total (measured via
        the committed test) - a successful mining draw typically solves
        the target within the first few generations, since the recurrence
        is read directly off the table rather than searched for across a
        population.
      Verified directly with unit-level round-trips too:
      `mine_recursion_choices` reconstructs the *exact* expected `choices`
      dict for both `2**n` (`combine_kind="double_recur"`, `op="+"`,
      `delta=0`, `base_kind="const"`, `base_val=1`) and Fibonacci
      (`delta=1`, `base_kind="param"`) directly from their tables, with no
      randomness involved at all - not just "the search eventually finds
      it," but "the relationship is read off the data deterministically."
- [x] Net finding: this is the second of the eight general mechanisms
      tried across v0.28-v0.32 that actually fixes its target class (after
      v0.31's leap-year fix), confirming the same diagnosis in a different
      domain - the recursion failures were generative (the correct shape
      essentially never assembling under blind growth), not selective, and
      a generation-time mechanism that uses the actual data fixes both
      residual gaps (2**n's own reliability was already good via the hand
      template, but Fibonacci's seeds 4/7 were genuine, previously-
      unresolved failures) outright rather than incrementally.
- [x] Scope, stated honestly: requires the example table to be dense
      enough to resolve a candidate's needed offsets (works cleanly for a
      contiguous or near-contiguous range, exactly the shape every
      recursion target in this project's own test suite already uses) and
      the base-case rows to form a contiguous prefix from the table's
      smallest value. A sparse or non-contiguous example set returns
      `None` cleanly (falls through to blind growth/`_recursive_template`,
      unaffected) rather than guessing - checked directly
      (`test_mine_recursion_choices_returns_none_for_sparse_or_non_
      recursive_tables`).
- [x] New tests: `test_build_table_rejects_non_integer_and_boolean_targets`,
      `test_mine_recursion_choices_discovers_pow2_exactly`/`_discovers_
      fibonacci_exactly` (exact deterministic round-trips, not
      probabilistic claims), `test_mine_recursion_choices_returns_none_
      for_sparse_or_non_recursive_tables`, `test_numeric_recursion_
      template_builds_a_generalizing_pow2_node` (checked out to n=9, well
      past the 5-example training table), `test_numeric_recursion_
      template_returns_none_for_non_numeric_or_sparse_targets`,
      `test_numeric_bias_is_a_true_noop_by_default` (checked directly, not
      inferred), `test_synth_wrapper_exposes_allow_numeric_bias`, and the
      measured-result test `test_numeric_bias_solves_pow2_and_fibonacci_
      reliably_across_seeds` (the full 8+8-seed sweep, committed directly
      rather than a representative subset - unlike v0.31's leap-year test,
      the whole sweep runs in seconds, so there's no runtime reason to
      only assert a subset here).


## v0.33 — Frontier 2, staged: scaling past `O(2**n)` and rules that emerge from data

- [x] Direct pivot from the tier4 synthesis work (v0.28-v0.32): asked to step
      back from per-target patches and fully realize `VISION.md`'s three
      "frontiers" - dynamic-dimensionality collapse, energy-landscape logic,
      resonance/interference deduction - staged, highest-leverage first.
      Direct code research (not docs) found the actual gap size differs
      hugely per frontier: Frontier 2 (energy landscapes) is the most mature
      and ties directly into the tier4 work just hardened, with two
      concrete, self-acknowledged gaps; Frontier 3 (resonance) has real,
      tested math with *zero* production call sites anywhere in the repo;
      Frontier 1 (dynamic dimensionality) already has the right primitive
      built (`participation_ratio`/`dimensional_collapse`) but wired into
      nothing downstream. User's call: Frontier 2 first. This entry is that
      stage; Frontiers 3 and 1 are staged next (see the plan already on
      record for exact scope).
- [x] **`settle_adaptive` thermal noise** (`energy.py`): only plain `settle`
      supported `temperature` (von-Mises-like phase noise); the adaptive-
      step-size mechanism was deterministic-only. Added the identical noise
      injection, gated so the plateau/convergence check tracks the
      *pre-noise* improvement (noise perturbing energy every step would
      otherwise make "3 consecutive tiny-improvement steps" nearly
      impossible to ever trigger). `temperature=0.0` (default) is a
      byte-for-byte no-op, checked directly.
- [x] **`compile_theory_relaxed` + `lukasiewicz_energy_relaxed`**
      (`grounding.py`): `compile_theory`'s own docstring already named the
      fix - "a continuous-relaxation variant... once `energy.settle_grad`
      lands" (it has) - since `2**n` exhaustive Boolean-corner enumeration
      is infeasible past ~20-25 variables. Gradient-descends the theory's
      energy directly over a continuous valuation from several random
      restarts (multiple, because a nonconvex multi-rule energy can have
      several satisfying corners, each its own local basin), snaps each
      converged point to its nearest corner, and registers exactly what
      `compile_theory` would - same Boltzmann weighting, same
      `WEIGHT_FLOOR`, same hypervector encoding, so the resulting
      `Landscape` is structurally interchangeable, just found by a
      polynomial search instead of a combinatorial one. Scoped to
      `logic="lukasiewicz"` (this project's default and only logic ever
      used in its own theories) - `compiler.py`'s `clamp`/Gödel/product
      implications use plain Python `min`/`max`/`if`, which abort a
      `jax.grad` trace; `lukasiewicz_energy_relaxed` is a `jax.numpy.clip`-
      based restatement, exactly the reason `settle_grad` restates
      `Landscape.energy` instead of calling it - raises
      `NotImplementedError` on any other logic rather than silently
      mishandling it.
      Real performance bug found and fixed before this was usable: a plain
      Python loop of `restarts * steps` un-jitted `jax.grad` calls measured
      at **100+ seconds for just 16 variables** - slower than exhaustive
      enumeration, defeating the entire point. Rewritten with
      `jax.vmap`+`jax.jit` over `jax.lax.fori_loop` (the same "compile once,
      dispatch once" discipline `collapse.collapse_batch_jit` already
      established elsewhere in this project) - **1.25s** at the same n=16
      (now *faster* than exhaustive's 3.7s), and **2.4s at n=28** (268
      million corners - not attempted exhaustively at all).
      Measured directly, not assumed: matches exhaustive enumeration's exact
      corner set on every existing `test_grounding.py` theory at a sharp
      (high-`inverse_temperature`) setting. Honest, documented scope limit
      found the same way: at *low* `inverse_temperature`, exhaustive
      enumeration keeps every corner clearing `WEIGHT_FLOOR` (including
      merely-mediocre, non-locally-optimal ones, since it scores literally
      every corner), while gradient descent only ever finds actual local
      minima - measured on this module's own test theory (4 corners survive
      exhaustively at `inverse_temperature=0.1`; only 1, the true minimum,
      via relaxation) and not fixable by more restarts, since a non-minimum
      corner is not a gradient-descent fixed point from any start. Argued
      (not just excused) to not cost much in practice: a diffuse prior over
      thousands-to-millions of corners was never a usable `Landscape` at the
      scale this function exists for anyway - the sharp, few-ground-states
      regime is exactly where scaling past enumeration matters, and exactly
      where this function already matches (and now beats) exhaustive
      enumeration.
- [x] **Rule discovery**: new module `tier3_logic/rule_discovery.py`
      (`discover_antecedent`, `discover_rule`, `DiscoveredRule`,
      `expand_valuation`, `compile_discovered_theory`) - the concrete answer
      to "no mechanism anywhere lets a `Rule`'s structure be discovered
      rather than hand-typed." A rule's antecedent is structurally identical
      to what `tier4_synthesis.semantic_bias`'s boolean backpropagation
      already discovers (v0.31 - 30/30 leap-year seeds): a compound boolean
      formula over named variables matching given (valuation → outcome)
      examples. `discover_antecedent` wraps `tier4_synthesis.search.
      synthesize(allow_semantic_bias=True, allow_bool_template=False)`
      directly - the general mechanism itself does the discovering, not a
      hand-built template. Rather than complicate `Rule`'s deliberately
      simple dict-key contract, a discovered antecedent is registered as a
      synthetic named variable (`_discovered_<consequent>_antecedent`)
      computed from the real variables via `dsl.evaluate`;
      `expand_valuation` injects that computed value into a valuation dict
      before it reaches the ordinary, unmodified `Theory.energy` -
      `Rule`/`Theory` never need to know a rule's antecedent was discovered
      rather than declared. `compile_discovered_theory` mirrors
      `compile_theory`'s exact enumeration/weighting algorithm with that one
      expansion step added, raising `ValueError` on any unverified
      discovered rule rather than silently promoting a guess to an axiom.
      Measured end to end, not just unit-tested in isolation: given four
      examples generated from a hidden rule `wet <- rain or sprinkler`,
      `discover_antecedent` finds the exact formula `(rain or sprinkler)`,
      verified; compiling it into a `Landscape` at a sharp temperature keeps
      *only* corners logically consistent with that implication (the two
      violating corners - antecedent true, consequent false - correctly
      excluded).
- [x] New tests: `test_settle_adaptive_temperature_is_a_true_noop_by_default`/
      `_explores_thermally` (`test_energy.py`);
      `test_lukasiewicz_energy_relaxed_matches_theory_energy_formula`,
      `test_compile_theory_relaxed_requires_jax_backend`/`_rejects_non_
      lukasiewicz_rules`/`_matches_exhaustive_ground_state`/`_scales_past_
      exhaustive_enumeration` (`test_grounding.py`); a new file
      `test_rule_discovery.py` (7 tests - OR/AND formula recovery, synthetic
      antecedent naming, `expand_valuation` correctness including a
      fractional-valuation rounding case, corner-consistency of a compiled
      discovered theory, and the unverified-rule rejection).
      Environment note recorded for future sessions: the project's `.venv`
      (`C:\program\.venv`) has JAX 0.10.2 installed and is required for
      every JAX-gated test in this entry - the bare system Python on PATH
      has no JAX at all (`HAS_JAX=False` there), a fact this session had to
      rediscover after `docs/ARCHITECTURE.md`'s note about JAX being active
      by default turned out to refer to a different environment than the
      one commands were initially run in.

## v0.34 — Frontier 3, staged: resonance stops being dead code

- [x] Continuation of the v0.33 staged plan (Frontier 2 -> Frontier 3 ->
      Frontier 1). Direct code research had found `resonance.py`'s
      `interfere`/`coherence`/`phase_lock`/`resonate` were real, tested math
      with *zero* production call sites - `qa.py` (the actual deduction/
      multi-hop query layer) reimplemented its own "coherence" as raw
      `hypervectors.similarity` instead of importing the module `docs/
      ARCHITECTURE.md` names as Frontier 3's home.
- [x] **`qa.py`'s `_cleanup` now scores candidates by `phase_lock`**, not
      `similarity` - a Kuramoto-style order parameter (`|mean(a * conj(b))|`)
      that tolerates a global phase offset between the recalled residue and
      a candidate's wave (e.g. drift accumulated over several binds) that
      plain `Re(similarity)` would silently discount. Checked directly
      before committing to the swap, not assumed: on every case the
      project's own `demo_ontology` exercises (8 subject/relation pairs,
      including the unknown-entity guess case), `phase_lock` and
      `similarity` pick the *identical* top candidate, scores agreeing to
      3+ decimal places (max observed difference 0.004) - so this is a
      genuine mechanism swap with no behavioral regression, not a cosmetic
      rename. This gives `phase_lock` its first production consumer.
- [x] **`Chain.resonance_coherence`**: a second, independent signal about a
      multi-hop chain's *whole* trajectory, distinct from the per-hop
      product already carried in `cumulative`. `chain()` composes
      `Ontology.step` straight through, the same number of times the real
      (collapse-and-reinject) chain took, but *without* collapsing onto a
      clean entity between hops, then reads how strongly that uncollapsed
      composition still resonates with the same final entity the real chain
      settled on (`resonance.coherence` of `resonance.interfere`-superposing
      the two). High resonance means the deduction holds together as one
      continuous wave composition, not merely as a sequence of individually
      -clean single hops; low resonance honestly signals that the per-hop
      collapse-and-reinject was doing real error-correction work a single
      uninterrupted composition could not have done alone. `0.0` when no
      hops were taken. Chosen over the plan's original sketch (hard-gating
      hop acceptance on consecutive-hop phase-lock) because `_cleanup`'s
      swap already made `phase_lock` the thing standing between "coherence"
      and every accepted hop - a second, additive diagnostic carrying new
      information proved more valuable than re-deriving the same signal as
      a stricter gate.
- [x] New tests in `test_chain.py`: `test_chain_resonance_coherence_is_high_
      for_a_clean_transitive_chain` (a dedicated low-crosstalk 3-hop
      ontology stays well above the noise floor, `> 0.3` - not claimed near
      1.0, since composing three real-valued wave operations with no
      intermediate error-correction genuinely accumulates dispersion),
      `test_chain_resonance_coherence_is_zero_with_no_hops`, and
      `test_chain_resonance_coherence_carries_information_cumulative_does_
      not` (measured, not assumed: on the demo ontology's 3-hop chain the
      two numbers differ by more than 0.3 - `resonance_coherence` is not a
      duplicate of `cumulative[-1]`).

## v0.35 — Frontier 1, staged: the collapse wired into the real pipeline

- [x] Final stage of the v0.33 staged plan (Frontier 2 -> Frontier 3 ->
      Frontier 1). Direct code research had found `collapse.py`'s
      `participation_ratio`/`dimensional_collapse` already computed a
      genuine, continuous "effective dimension" from occupancy entropy and
      truncated to a live-symbol basis - but wired into nothing: not used by
      `anneal`/`anneal_adaptive`, and `k_live`/`live_names` consumed nowhere
      downstream.
- [x] **C1 - `anneal`/`anneal_adaptive` now run `dimensional_collapse`**
      instead of plain `collapse` (`collapse.py`). ``z`` is fixed throughout
      either schedule (only beta changes) and neither function's returned
      state feeds into the next step, so this is a pure addition: `eff_dim`/
      `k_live` join every trace entry alongside the existing `entropy_bits`/
      `winner`/`winner_prob`, with the entropy/winner numbers themselves
      provably unchanged (`dimensional_collapse` is numerically identical
      to `collapse` at high entropy - already covered by `test_dimensional_
      collapse_matches_collapse_at_high_entropy`). New tests: `test_anneal_
      logs_a_real_eff_dim_k_live_trajectory` (k_live falls monotonically
      across a fixed cooling schedule and reaches 1 on an exact-match
      probe), `test_anneal_adaptive_logs_eff_dim_k_live_too`.
- [x] **C2 - `qa.py`'s `_cleanup` restricts its candidate-comparison set to
      the live basis.** A small `_entity_codebook` helper wraps `Ontology`'s
      existing entity waves (via `Codebook.load`, which restores vectors
      verbatim rather than minting new ones) into a codebook scoped to just
      the answer candidates - `dimensional_collapse` needs that, not
      `Ontology.codebook`'s full role/relation/entity alphabet. `_cleanup`
      now runs `dimensional_collapse` first and only scores `phase_lock`
      (v0.34) over its `live_names`, not every entity in the KB.
      Measured directly before wiring this in, not assumed (see `python -m
      zeuss ask`'s own printed trace): on the demo ontology's 11 entities,
      every known fact shrinks the comparison set to 2 (`eff_dim` ~1.4 -
      the true answer plus one runner-up), while every genuine guess
      (unknown subject, wrong relation, needs-multi-hop) correctly stays at
      the full 11 (`eff_dim` ~10.9) - there is no real winner for entropy to
      collapse toward, so a fixed high beta sharpening pure noise does
      *not* fool `dimensional_collapse` into false confidence. The true
      `phase_lock` top pick was inside the live set in every case checked.
      `Answer` gained `k_live`/`eff_dim` fields (default `0`/`0.0`, so this
      is additive) so the shrinking comparison set is observable, not just
      internal - printed in `Answer.__str__` and asserted directly in new
      tests `test_known_facts_shrink_the_live_comparison_set`/`test_unknown_
      queries_do_not_falsely_shrink_the_comparison_set` (`test_ask.py`).
      `chain()`'s internal `_cleanup` call site was updated for the new
      return arity; `Chain` itself does not carry `k_live` (out of scope for
      this stage - `chain()` only needed the unpacking fix to keep working).
- [x] This completes the three-stage plan: Frontier 2 (v0.33) -> Frontier 3
      (v0.34) -> Frontier 1 (this entry) - all staged, each with its own
      measured result and honest gap where one remained, rather than a
      single unverified all-at-once pass.

## v0.36 — the three frontiers actually cooperate on one query, not three silos

- [x] Direct follow-up to v0.33-v0.35: asked afterward whether the substrate
      was doing what `VISION.md` actually claims - a *unified* engine, not
      three independently-tested mechanisms. Direct code research found the
      honest answer was no: `qa.py`'s deduction path (Frontier 1 + 3, wired
      together in v0.34/v0.35) and `grounding.py`'s theory compilation
      (Frontier 2) were two disconnected pipelines sharing a codebase -
      `docs/ARCHITECTURE.md`'s own `collapse -> energy -> resonance` diagram
      was aspirational, not real, since no query ever passed through
      `energy.py`'s `Landscape`/`settle`.
- [x] **Before writing any code, tested whether energy settling would add
      real capability or just be cosmetic** (the standard set when this gap
      was first identified): built a deliberately crosstalk-heavy synthetic
      ontology (`dim=512`, 40 entities chained into one bundled memory - the
      demo KB's `dim=8192` has almost no crosstalk to correct) and compared
      one-shot `phase_lock` cleanup against settling the residue against a
      `Landscape` built from the live basis before reading out. Result:
      settle-informed selection recovered 1 of 3 cases one-shot cleanup got
      wrong, and broke 0 of the 36 it already got right - modest but real,
      and insensitive to the exact step count (checked 15/20/25/30,
      identical result each time). That was the go/no-go signal to wire it
      into `qa.py` for real.
- [x] **A real negative result caught before it shipped, not after:** the
      first design read `coherence`/`confidence` from the *settled* state
      too (not just which candidate wins). That is actively wrong, not just
      redundant - `settle`'s dynamics are a self-reinforcing attractor
      network by construction (Frontier 2's entire point: `landscape.
      target()` pulls `z` toward its own softmax-weighted read of `z`, which
      sharpens that read, which pulls harder), so *any* residue - including
      pure crosstalk noise with no real answer - drifts toward amplitude
      ~1.0 against whichever candidate it leaned toward first. Measured
      directly: this collapsed `demo_ontology`'s own `dragon is_a ?` guess to
      coherence `+1.000` (`known=True`), silently breaking `test_unknown_
      queries_are_flagged_as_guesses`. Fixed by decoupling the two roles
      settling can play: the settled state's `phase_lock` argmax decides
      *which* candidate wins (where energy relaxation can correct a noisy
      one-shot pick), but `coherence`/`confidence`/`ranked` are still read
      from the *original*, unsettled residue (which must stay an honest
      "does this genuinely ring true" signal). This is the kind of tried-and
      -rejected design this project documents on purpose (see v0.30) rather
      than silently discarding.
- [x] **`qa.py`'s `_cleanup`** now runs, for every single- and multi-hop
      query: `dimensional_collapse` restricts to the live basis (Frontier 1,
      v0.35) -> that basis becomes a `Landscape`, weighted by its own
      occupancy, and `settle` relaxes the residue toward it (Frontier 2,
      new) -> `phase_lock` reads out the winner from the settled state and
      the honest coherence from the original residue (Frontier 3, v0.34).
      `chain()` inherits this on every hop for free, since it already calls
      `_cleanup` internally - no separate change needed there. `docs/
      ARCHITECTURE.md` updated to say this plainly: the diagram is no longer
      aspirational for this code path.
- [x] New tests in `test_ask.py`: `test_energy_settling_corrects_a_
      crosstalk_error_without_regressing` (asserts the one specific
      correction directly through the real `ask()` API, `e19 r1 -> e20`
      where one-shot cleanup picks `e39`, plus zero regressions across all
      36 already-correct cases on that same ontology) and `test_energy_
      settling_does_not_inflate_guessed_coherence` (regression guard for the
      exact failure mode found above - `dragon is_a ?` must stay well below
      `COHERENCE_FLOOR`, nowhere near the ~1.0 a settled-coherence design
      would report). Full suite still green after this change.
- [x] **Honest scope of what remains unaddressed:** this unifies Frontiers
      1+2+3 inside `qa.py`'s relational deduction. `grounding.py`'s own
      compile-a-`Theory`-into-a-`Landscape` path still doesn't call into
      `Ontology`/`qa.py` at all - a `Theory`'s propositional variables and an
      `Ontology`'s entities are still different data models with no bridge
      between them. Whether that bridge is worth building (e.g. letting an
      `Ontology`'s stored facts double as axioms a `Theory` can reference)
      is an open question, not assumed - unlike this entry's `qa.py` change,
      nobody has yet checked whether it would add real capability or just
      be more surface area.

## v0.37 — closing v0.36's gap: Theory/Rule axioms actually bias a query

- [x] Direct follow-up to v0.36's own honestly-scoped remaining gap: could
      `compiler.py`'s `Rule`/`Theory` formalism (Frontier 2's propositional
      logic) do real work inside a `qa.py` query, or would forcing a shared
      data model between `Ontology`'s relational facts and `Theory`'s
      propositional variables just be more surface area for no benefit?
      Tested the question rather than assuming either answer, matching this
      project's own established discipline (v0.30, v0.36).
- [x] **The test scenario, not a contrived toy:** a KB with a genuine data
      contradiction - `socrates is_a` stored as *both* `human` and `star`
      (the kind of thing a noisy/uncurated ingestion pipeline produces in
      practice). Measured first, before building anything: this makes plain
      resonance a near coin flip, seed-dependent - 4 of 10 seeds wrongly
      resolve to `star` despite `socrates walks_on earth` being separately,
      unambiguously known in the same KB. Pure wave resonance has no way to
      use that second fact; it only ever looks at the one relation being
      queried.
- [x] **`axiom_bias`** - a new optional parameter on `ask`/`chain`/
      `_cleanup` (`qa.py`): `candidate -> additional energy penalty`,
      applied as `exp(-penalty)` on that candidate's `Landscape` weight
      inside `_cleanup`'s existing v0.36 settle step - the same Boltzmann
      convention `grounding.compile_theory` already uses for its own
      attractor weights, reused rather than reinvented. The caller builds
      the penalty from an ordinary `Rule` plus *other* `ask()` calls about
      the same subject: `Rule("walks_on_earth", "not_star")`, with
      `walks_on_earth`'s truth value filled in from `ask(subject,
      "walks_on").confidence`. Result: corrects all 4 wrong seeds to
      `human`, regresses 0 of the other 6 (`test_axiom_bias_resolves_a_
      genuine_data_contradiction`). Default `None` is a true no-op -
      `test_axiom_bias_none_is_a_true_noop` checks every existing case
      reproduces exactly, and the full pre-existing suite passes unchanged.
      Like the settled state itself (v0.36), `axiom_bias` only ever
      influences *which* candidate wins, never `coherence`/`confidence` -
      the same honesty requirement, for the same reason.
- [x] **Honest scope: this is a hook plus one demonstrated example, not an
      automatic axiom-discovery system.** Nothing here lets Zeuss discover
      *which* axioms apply on its own, or builds a general Ontology<->Theory
      data-model bridge - the caller still has to write the `Rule` and wire
      up which `ask()` call fills in which variable, by hand, per domain.
      What this closes is narrower and more honest: the *mechanism* for a
      Frontier-2 logical axiom to influence a real relational query now
      exists and is measured to work on a genuine (not manufactured-to-
      pass) failure case, where before there was no seam for this at all.
      Whether it's worth building the fully general bridge remains an open
      question - this entry answers "would a bridge add real capability"
      (yes, on this case) without yet committing to what a general one
      looks like.

## v0.38 — automatic axiom discovery: mining Rules instead of hand-writing them

- [x] Direct follow-up to v0.37's own honestly-scoped limit: "the caller
      still has to write the `Rule` by hand." Asked whether that Rule could
      instead be *discovered* from the ontology's own stored facts - the
      same "rules emerge from data" standard v0.32's tier4 bridge already
      met for propositional boolean formulas, now for relational facts.
- [x] **New module `tier3_logic/axiom_mining.py`**: plain support/confidence
      association-rule mining (Agrawal et al. - a standard, well-understood
      statistic, not a bespoke one) over `Ontology.triples`.
      `discover_implications(onto, ante_rel, cons_rel)` finds
      `ante_rel=f1 -> cons_rel=f2` patterns with enough support and
      confidence; `discover_exclusions` derives the natural corollary - a
      veto against every *other* observed filler of `cons_rel` once a
      strong implication toward one specific filler is established (the
      concrete, statistically-checked form of "this relation is normally
      single-valued", not a hard-coded schema constraint).
      `axiom_bias_from_exclusions` turns a list of mined exclusions straight
      into a v0.37-compatible `axiom_bias` callable - no hand-typed `Rule`
      anywhere in the loop from "ontology's own data" to "a real qa.py
      query's Landscape gets biased".
- [x] **Measured on the same kind of genuine contradiction v0.37 used, now
      at a scale where it's a *systematic* problem, not a coin flip:** a KB
      with 10 clean `human` entities (`walks_on earth`), 10 clean `star`
      entities (`orbits galaxy`), and one contradictory `socrates` (`is_a`
      both). At this scale (21 entities in one bundled memory), plain
      resonance picks the wrong `is_a` answer for `socrates` on *every one*
      of 15 seeds tested - not an occasional failure. Mining
      `walks_on=earth -> is_a=human` (support 11, confidence 1.0) and
      `orbits=galaxy -> is_a=star` (support 10, confidence 1.0) - both
      exactly as expected, and unaffected by socrates' own contradiction,
      since he still holds the `human` fact too, not just `star` - and
      using the derived exclusions as the query's `axiom_bias` recovers 12
      of the 15 wrong seeds, with zero regressions across 90 clean-entity
      checks (3 human + 3 star entities x 15 seeds). Reported honestly as
      12/15, not rounded up to "fixed" - a real, measured majority
      improvement, not a 100% claim.
- [x] **A real design mistake caught mid-build, not after:** the first
      version gated each exclusion's antecedent truth on `Answer.
      confidence` alone, and it silently failed - baseline stayed wrong
      even with the bias applied. Traced to why: `confidence` is a
      *relative* softmax share against every other candidate in the KB, so
      it dilutes toward 0 as the entity count grows (measured: `walks_on`
      confidence was 0.17 at 21 entities, versus ~0.98 in the 6-entity demo
      KB), even though the underlying fact is exactly as objectively true
      either way. Fixed by gating on `Answer.known` instead - an *absolute*
      floor-referenced signal that doesn't dilute with KB size - matching
      how `COHERENCE_FLOOR` itself is already used everywhere else in
      `qa.py`. This is the same category of tried-and-fixed design mistake
      v0.36 documented (settled-state coherence inflating guesses) -
      caught by testing the actual mechanism before trusting it, not by
      assuming the first plausible design was correct.
- [x] New test file `test_axiom_mining.py` (5 tests at this point): the
      mined implication/exclusion match exactly, `min_support` is
      respected, and the two measured claims above (12/15 systematic-fix
      rate, 0/90 regressions) are asserted directly against the real module
      and the real `ask()` API - not re-derived from the offline experiment
      that motivated this.
- [x] **`discover_all_exclusions`/`axiom_bias_from_ontology` - closing this
      entry's own first-draft gap the same session:** the mining above still
      made the caller name both relations in every `(antecedent_relation,
      consequent_relation)` pair by hand. `discover_all_exclusions(onto)`
      tries every ordered pair of relations actually used in the ontology
      automatically (`O(R^2)` pairs - fine at this project's scale) and
      keeps whichever mined exclusions clear the thresholds;
      `axiom_bias_from_ontology` filters those to `target_relation` and
      builds the `axiom_bias` callable directly - "point it at an ontology,
      get a biased query out", no relation names passed anywhere. Checked
      directly, not assumed: on the same 15-seed scenario, this reproduces
      the *exact* same 12/15 fix / 3-seed-holdout / 0-regression result as
      hand-picking `(walks_on, is_a)` and `(orbits, is_a)` - the other 4 of
      6 automatically-tried relation pairs (e.g. `is_a -> walks_on`)
      correctly contribute nothing, checked with an exact-set-equality
      assertion (`test_discover_all_exclusions_finds_exactly_the_two_real_
      patterns`) rather than a looser "contains" check that could hide
      spurious noise. `DiscoveredImplication`/`DiscoveredExclusion` made
      frozen/hashable to support that set comparison. Two more tests added
      (7 total in `test_axiom_mining.py`).
- [x] **Honest scope, updated:** the relation-pair-picking gap above is
      closed. What remains open: `min_support`/`min_confidence` are still
      caller-set thresholds, not themselves discovered or validated against
      a held-out set; and the 3 seeds v0.38 already found unfixed by the
      hand-picked version stay unfixed here too (identical result, as
      measured) - a real, acknowledged limit of this specific bias-then-
      settle mechanism's strength at this ontology's scale, not something
      full automation was expected to (or does) fix on its own.

## v0.39 — scale validation: the real ceiling was never this session's work

- [x] Asked directly, after v0.33-v0.38: does the unified pipeline (Frontier
      1+2+3 in `qa.py`, plus v0.37/v0.38's axiom bias/mining) hold up past
      toy scale, or does something that worked on 20-40 entities quietly
      break? Every test and demo anywhere in this project - `demo_ontology`
      (7 triples), the crosstalk ontologies used to validate v0.36-v0.38
      (30-40 triples) - had stayed inside a range nobody had ever actually
      measured the edge of.
- [x] **The honest, important finding: the ceiling isn't in anything built
      this session - it's in `Ontology.ground()`'s single-bundle design,
      which predates all of it.** Measured directly, robust across 5 seeds:
      recovering a directly-stored, completely unambiguous fact via plain
      `ask()` (no axioms, no contradictions, nothing this session added)
      stays reliable (5/5 correct-and-`known`) at 80 bundled triples and
      collapses to 1-3/5 at 120 - a sharp transition, not a gradual one,
      and one every existing test in this project happened to sit just
      under (the largest prior test ontology used 40 triples). At 400
      triples, a genuinely stored fact reads as `known=False` (coherence
      0.045, below `COHERENCE_FLOOR`) essentially always.
- [x] **A second, more surprising finding: raising `dim` barely helps.**
      The obvious fix - more dimensions, more capacity - was tested
      directly (8192 -> 65536, 8x) on the exact same 400-triple case and
      coherence *did not improve* (0.044 -> 0.039, if anything slightly
      worse). Also checked and ruled out: this isn't about many entities
      sharing the same filler concentrating interference (`human`/`earth`
      repeated across 100 subjects) - an unshared-filler control (every
      subject pointing at its own unique filler, same triple count) showed
      the identical collapse (`known=False`, coherence 0.061). The ceiling
      tracks *bundled triple count specifically*, and - measured, not yet
      explained - does not visibly respond to the one lever (`dim`) this
      project's own capacity intuitions (and `VISION.md`'s framing) would
      predict should fix it.
- [x] **Deliberately not investigated further in this entry:** *why*
      `dim` doesn't help is an open question - candidates include the
      specific `OBJ_SHIFT` permutation scheme, how `bundle`'s normalisation
      interacts with many summed terms, or a genuine limit of complex-
      phasor HRR bundling that this project's own capacity assumptions
      (inherited from `VISION.md`, never previously measured) simply got
      wrong. Chasing that explanation, or redesigning `Ontology.ground()`
      (e.g. sharding memory per relation instead of one giant bundle,
      hierarchical indexing, an explicit error-correcting cleanup pass) is
      real, substantial work - reported here as a finding to decide on, not
      quietly started.
- [x] **What this means for v0.34-v0.38:** every claim in those entries is
      still true *as measured* - all of it was tested at 20-40 triples,
      safely under this newly-found ceiling. It does mean none of that work
      has been shown to generalize past toy scale, and the axiom-mining
      correction mechanism specifically goes to 0/5 once the underlying
      `ask()` substrate itself can no longer recover the antecedent facts
      the bias depends on (measured at the 100-per-class/400-triple scale
      that motivated this whole entry) - not a flaw in the bias mechanism,
      a direct, expected consequence of the substrate-level ceiling above.

## v0.40 — sharding: raising the ceiling without lowering reliability

- [x] The user's proposal in response to v0.39's finding, put directly:
      if a single bundle is reliable up to ~80 triples and unreliable past
      it, why not run several bundles ("silos") in parallel, each kept at
      the reliable size, instead of one giant one? Tested before writing
      any production code, same discipline as every mechanism this session
      - not assumed to work just because it sounds reasonable.
- [x] **The result: it works, robustly.** At 400 triples (the exact case
      v0.39 measured as completely broken - 0/10 correct-and-`known`),
      querying 5 independent 80-triple shards and keeping whichever
      resonates loudest recovers 9/10. At 800 triples (10 shards), still
      18/20 (90%) - the recovery isn't a one-off at one specific size. This
      is the real fix v0.39 didn't have: not "raise `dim`" (measured in
      v0.39 to not work), but "keep every shard at the size that's already
      proven reliable, and add more shards instead of more bytes per shard."
- [x] **Checked the one real risk before trusting it: false positives.**
      Picking the *highest*-coherence answer across N independent shards
      means N independent chances for pure noise to spike - order
      statistics say the max of more samples runs higher than any one
      sample. Measured directly, not assumed safe: zero false positives
      across 20 genuine unknown queries at 5 shards, and zero again at 10
      shards - `COHERENCE_FLOOR`'s margin over single-shard noise (already
      established by `test_stored_and_guessed_coherence_are_well_separated`)
      turned out to be wide enough that max-of-10 still doesn't cross it.
      This was checked, not assumed - a mechanism that recovers accuracy by
      quietly trading away guess-detection would not have been worth it.
- [x] **New methods, additive only:** `Ontology.ground_sharded(shard_size=
      80)` bundles triples into several memory hypervectors instead of one
      (`ground()` itself is untouched). `qa.ask_sharded`/`qa.chain_sharded`
      mirror `ask`/`chain` exactly but query every shard - `chain_sharded`
      re-selects the best shard at *every* hop, not just once, since a fact
      needed partway through a chain can live in a different shard than
      the fact before it. `chain_sharded`'s `resonance_coherence` is
      computed against whichever shard won the final hop - a reasonable
      but not equally-measured choice (unlike the per-hop selection itself,
      this specific number hasn't been separately validated against real
      accuracy the way `ask_sharded` was).
- [x] New test file `test_sharding.py` (8 tests): the core recovery claim
      at 400 triples, the false-positive check at both 5 and 10 shards, the
      800-triple/10-shard scale check, and `chain_sharded` walking a real
      transitive chain correctly. Full suite green throughout.
- [x] **Honest scope: sharding is a mitigation, not an explanation.** This
      does not answer v0.39's still-open question (why doesn't raising
      `dim` help the single-bundle ceiling?) - it works *around* that
      ceiling by never letting any one bundle approach it. `shard_size=80`
      is the one measured-reliable point tested, not a swept parameter -
      whether 70 or 90 works just as well, or where sharding itself starts
      to break down (at what shard count does max-of-N noise finally cross
      `COHERENCE_FLOOR`?) is not yet known.

## v0.41 — stress-testing sharding finds where it actually breaks

- [x] Direct follow-up to the user's own question after v0.40: "is the
      ceiling really gone, or just moved?" and the natural next request -
      stress-test sharding itself rather than assume v0.40's 5-10-shard
      numbers extrapolate cleanly. Two different stress axes were tried.
- [x] **Axis 1 (scaling real data, more shards, no garbage): holds up.**
      Re-confirmed v0.40's own claim rather than re-deriving it - no new
      finding here, just re-checked before trusting the second axis's
      contrast was meaningful.
- [x] **Axis 2 (adding shards that carry unstructured/random content
      instead of real data): breaks fast, and this is the real finding.**
      Starting from the same 5 real 80-triple shards, adding just 10 extra
      shards of *random* `(entity, relation, entity)` nonsense - reusing
      existing vocabulary so the entity codebook, and thus v0.39's
      unrelated ceiling, never enters into it - collapsed the guess-
      detection guarantee: false positives on genuine unknown queries
      jumped from 0/10 to 4-6/10, and correct-and-known accuracy on real
      facts fell from 9/10 to 2-3/10, worsening further by 40 noise shards
      (0/10 correct, 10/10 false positives). A finer sweep (1/2/3/5 noise
      shards) showed this isn't a cliff - false positives start appearing
      almost immediately (1 noise shard: 0/10; 2: 1/10; 5: 2/10) and climb
      from there, not a safe-then-catastrophic threshold.
- [x] **Ruled out the obvious confound before trusting this.** The noise
      generator drew fillers from the whole entity vocabulary, which
      includes the 4 small "category" symbols (`human`/`star`/`earth`/
      `galaxy`) every real answer must land on - so some random noise
      could coincidentally reconstruct a plausible-looking answer purely
      by chance. Re-ran with noise fillers restricted to *never* include
      those 4 symbols (structurally guaranteeing no noise triple can look
      like a valid answer) and got the same collapse, if anything slightly
      worse (6/10 false positives at 10 noise shards, vs. 4/10 before) -
      so this is not a lucky-coincidence artifact of the test's own
      vocabulary choice.
- [x] **What this actually means, stated carefully:** v0.40's zero-false-
      positive result was real and reproduced, but it was specific to
      shards that are genuine partitions of real, structured data (every
      shard, even ones irrelevant to a given query, still encodes coherent
      real facts using the ontology's normal relation/filler structure).
      It does not generalize to "shard count is safe in general" - shards
      carrying unstructured or adversarial content break the guarantee far
      sooner than growing real data across more real shards does. The
      *precise* mechanism (why random content is measurably more
      disruptive than real-but-irrelevant content of the same size) was
      not tracked down here - that would mean examining `bind`/`bundle`/
      `unbind` behavior under adversarial random input directly, a further,
      separate investigation, not assumed to have an easy answer.
- [x] `qa.ask_sharded`/`qa.chain_sharded` docstrings updated with this
      caveat directly rather than left overclaiming the earlier all-real-
      data result as general safety. New test `test_sharding_is_fragile_to_
      unstructured_noise_shards` (`test_sharding.py`, now 9 tests) locks in
      the measured collapse alongside the original all-real-shards
      guarantee, so both the "safe" and "not safe" regimes are checked, not
      just narrated.
- [x] **Practical guidance going forward:** `ground_sharded`/`ask_sharded`/
      `chain_sharded` are validated for splitting a KB's own real data
      across shards, not for a setting where shard content might be noisy,
      adversarial, or unrelated to the domain - that would need its own
      investigation (e.g. a per-shard sanity/quality gate) before being
      trusted the way v0.40's core claim now is.

## v0.42 — a first quality gate, quickly superseded by v0.43

- [x] Answered v0.41's own question ("how do we tell garbage from real
      data") with the cheapest thing that could work: `is_structurally_
      regular(triples)` checks whether the same `(subject, relation)` pair
      repeats within a shard - not semantic truthfulness. Real `Ontology`
      data never repeats a pair; random noise sampled with replacement
      almost always does. `Ontology.ground_shards(shards, skip_irregular=
      True)` filters candidate shards by this before grounding. Verified:
      on the exact adversarial mix that broke v0.40 (5 real + 10 noise
      shards), filtering restores 0/10 false positives exactly.
- [x] **Two real limitations found immediately by the user, before this
      was trusted as final:** (1) a genuinely multi-valued relation
      (`has_friend`-like: many subjects legitimately have more than one
      filler) looks identical to a contradiction under this check - it
      would be wrongly dropped. (2) dropping the *entire* shard on one bad
      pair discards every other good fact in it. Both are addressed in
      v0.43, not left as a shipped limitation - `is_structurally_regular`/
      `ground_shards` remain in the codebase as the simpler, still-correct
      (for its narrower claim) path, now documented as superseded.

## v0.43 — intelligent conflict resolution: keep the good, drop only the bad

- [x] Direct response to the user's explicit ask: don't drop true multi-
      valued facts, discard only actual garbage, resolve conflicts
      intelligently rather than bluntly. Also asked directly: why not use
      `tier3_logic/sheaf.py`, which already does exactly this kind of
      "do independent sources agree" checking? Answered honestly: `sheaf.
      py` operates on `Theory`'s scalar `[0,1]` variables and named agents
      via linear algebra over equality constraints - a genuinely different
      data model than `Ontology`'s `(subject, relation, object)` triples.
      Using it here would mean porting its machinery, not reusing it - so
      this entry builds the same *philosophy* (check whether independent
      claims agree, don't just count duplicates) natively for triples,
      rather than forcing a scalar-algebra tool onto categorical data it
      wasn't built for.
- [x] **`classify_multi_valued_relations(triples, threshold=0.2)`**: which
      relations are legitimately multi-valued, inferred from the data
      itself (>=20% of a relation's subjects genuinely having more than
      one filler) - the same "let structure emerge from data" principle
      `rule_discovery.py`/`axiom_mining.py` already apply elsewhere, now
      applied to schema inference. Verified: correctly classifies a
      `has_friend`-style relation as multi-valued and `is_a` as not, from
      triples alone, no schema declared - and `resolve_shard_conflicts`
      preserves every one of the multi-valued relation's facts with zero
      data loss.
- [x] **Two within-shard statistics tried and rejected first, by direct
      measurement, not intuition:** (1) collision *rate* within a shard -
      measured across three vocabulary sizes and found NOT scale-
      invariant: real data with just 2 genuine contradictions (rate 0.026)
      fell inside pure garbage's own rate range (0.013-0.097) at one
      scale, so no fixed threshold works. (2) subject-degree variance/Fano
      factor (an attempt at a more powerful, scale-invariant version of
      the same idea) - also measured to drift toward 0 (indistinguishable
      from real data) as vocabulary grew (0.22 -> 0.08 -> 0.02 across
      three scales), because a single 80-triple sample from a huge space
      rarely collides with itself regardless of how it was built - a
      genuine statistical-power problem, not a threshold-tuning problem.
      This was caught by testing at a *different* scale than the one that
      motivated the original fix, not by assuming the first result
      generalized.
- [x] **The mechanism that actually works, and is scale-invariant: cross-
      shard consensus, not within-shard statistics.** `_global_filler_
      consensus` computes, for every `(subject, relation)` pair, the
      majority filler asserted across *every* shard combined. A shard's
      claim is judged against this consensus, not against its own
      internal structure - and a genuinely random filler drawn from a
      large vocabulary almost never matches the truth by chance, so this
      signal's power comes from the size of the answer space, not from
      the shard's own size. Measured directly at 100/400/1000-subjects-
      per-class scale: real-shard disagreement with the consensus stayed
      at 0.000 and garbage-shard disagreement stayed at 0.6-0.75 at *every*
      scale tested - the stable separation neither prior statistic had.
- [x] **Genuine ties are left unresolved, not silently guessed.** A true
      1-vs-1 split (e.g. two shards independently asserting `socrates
      is_a human` and `socrates is_a star`) has no principled winner.
      `_global_filler_consensus` returns `None` for an exact tie rather
      than trusting `Counter.most_common()`'s insertion-order tie-break
      (checked directly: confirmed this is a real, order-dependent
      behavior, not a safe default) - both sides are excised, leaving the
      fact honestly absent instead of confidently wrong either way. A
      third, corroborating vote correctly breaks a would-be tie with a
      real majority.
- [x] **`resolve_shard_conflicts`/`Ontology.ground_resolved_shards`**: the
      complete pipeline - classify multi-valued relations, compute cross-
      shard consensus, excise only the specific facts that disagree with
      it (keeping every other fact in the same shard untouched), and drop
      a whole shard only when *most* of its content disagrees with the
      rest of the dataset. Verified on the real API: the exact v0.41
      failure case (20 real shards + 2 noise shards) is fully restored -
      both noise shards dropped, 0 false positives, accuracy matching the
      all-real-shards baseline; a single injected contradiction in an
      80-triple shard is surgically excised while the other 78 facts in
      that same shard survive untouched.
- [x] **The honest boundary, measured precisely rather than left vague:**
      this assumes garbage is a *minority* of the data - the same "need an
      honest majority" requirement consensus/robust-statistics approaches
      generally have (not a defect unique to this design). Measured the
      exact degradation curve on one ontology: correct and error-free
      through 17% and 29% contamination, degrading progressively at 38%
      (1/10 false positives) and 44% (2/10), failing open at a 50/50 split
      (2/10, confirmed by this file's own formal test - not 4/10 as
      originally estimated before that test was actually run) - because at
      that point the garbage's own randomness corrupts
      `classify_multi_valued_relations` itself (every relation starts
      looking falsely multi-valued), which exempts everything from the
      disagreement check entirely. Locked into `test_shard_conflict_
      resolution.py`'s own dedicated test for this failure mode, not
      swept under the rug.
- [x] New test file `test_shard_conflict_resolution.py` (9 tests): the
      classifier, zero-loss multi-valued preservation, surgical single-
      contradiction excision, tie-handling (both the unresolved case and
      the tie broken by a third vote), the full real-API restoration of
      v0.41's failure case, the moderate-contamination robustness check,
      and the disclosed garbage-majority failure mode.

## v0.44 — logical-axiom-violation checking: a different kind of signal than a vote

- [x] Direct response to v0.43's own disclosed ceiling: no majority-vote/
      consensus scheme can ever be made reliable against an adversarial
      *majority* of bad data - the same wall Byzantine fault tolerance and
      robust statistics hit generally (median-based estimators tolerate up
      to ~50% contamination, never more, because past that point "the
      majority" and "the truth" are definitionally not the same thing).
      Also asked directly: why not reuse `sheaf.py`'s existing "do
      independent sources agree" machinery here too? Answered: its H0/H1
      cohomology is equality-restriction linear algebra over scalar
      stalks - the right tool for "do two agents' conclusions agree" (see
      `grounding.compile_theories`), but logical necessity (a fact being
      *structurally impossible* given another fact, independent of any
      vote) is a genuinely different kind of signal, not a bigger version
      of consensus voting, so it's built natively rather than forced
      through a mismatched tool.
- [x] **`axiom_violations(triples, exclusions)`**: which triples
      structurally violate a trusted, `DiscoveredExclusion`-shaped mined
      axiom (duck-typed, no import cycle with `axiom_mining.py`) given
      what else the *same subject* holds. Wired into
      `_global_filler_consensus` so a violating vote is discarded *before*
      a majority is tallied, not merely overridden after - a structurally-
      impossible fact can't win a vote just because garbage shards
      outnumber real ones. `resolve_shard_conflicts`/`Ontology.
      ground_resolved_shards` both gained an `exclusions` parameter
      (default `None` = exact no-op, fully backward compatible).
      Verified directly: a fact a 2-vs-1 garbage majority would otherwise
      out-vote is recovered once a trusted exclusion disqualifies the
      garbage votes from the tally entirely.
- [x] **Honest limit, stated up front and checked, not assumed:** this is
      only as trustworthy as the `exclusions` it's given. Mining them from
      the *same* contaminated pool being checked just re-derives the same
      vote-counting problem one level up (checked directly: exclusions
      self-mined from a 50/50-contaminated sample differ from the same
      mining run on a clean seed sample). The real power comes from
      sourcing `exclusions` independently - hand-written `Rule`s (v0.37)
      or a separately-trusted seed sample - not from the shards under
      suspicion.
- [x] **Second honest limit, found only by actually measuring the full
      real-API pipeline at the v0.43 garbage-majority boundary, not
      assumed to follow from the first:** `axiom_violations` only excises
      literal violating *triples*. Once a relation has already been
      exempted from `_global_filler_consensus` entirely (i.e.
      `classify_multi_valued_relations` misclassifies it as multi-valued
      under heavy contamination, v0.43's own disclosed failure mode),
      *every* triple under that relation - including hundreds of unrelated
      noise ones - gets bundled into the grounded memory unfiltered, and
      most of the resulting wrong answers turn out to be hypervector
      crosstalk from that flood, not any single literal contradiction.
      Excising the one specific violating fact this mechanism finds is
      real (measured: present in the unfiltered baseline, absent once
      exclusions are applied) but does not by itself restore downstream
      query accuracy at that contamination level - a different problem
      exclusions were never designed to solve.
- [x] **Two real bugs found and fixed in v0.43's own code while finally
      running its test suite to completion** (it had been left uncommitted
      with that confirmation still pending - see this file's resume
      notes): (1) `classify_multi_valued_relations` had no floor on
      subject count, so a relation with very few subjects could be
      auto-classified multi-valued off a single genuine contradiction
      (e.g. a true 1-vs-1 cross-shard tie) instead of being flagged as
      one - fixed with a `min_subjects` floor (default 5). (2)
      `resolve_shard_conflicts`'s whole-shard-drop branch used `continue`,
      which *omitted* the shard from the output list entirely instead of
      keeping its (now-empty) slot - broke positional alignment with the
      input list that several of its own tests (and callers) rely on -
      fixed to append `[]` instead. Both were caught by actually running
      the test suite, not by inspection.
- [x] **Two of v0.43's own test assertions were hardcoded numbers that had
      never actually been confirmed** (the background pytest run checking
      them was still pending when that work was left uncommitted): re-
      measured for real, deterministically, at 7/10 (not 8/10) and 2/10
      (not 3/10) respectively - both fixed to the true measured values,
      and this file's own "50/50 split" figure corrected from 4/10 to 2/10
      to match (the 38%/44% figures next to it were independently
      re-verified and were already correct).
- [x] New test file `test_axiom_violation_resolution.py` (5 tests): the
      direct unit check, the garbage-majority-outvote recovery
      demonstration, the "exclusions mined from the same contaminated pool
      inherit the same limit" honesty check, and the literal-violation-vs-
      crosstalk distinction at the v0.43 boundary.

## v0.45 — a genuine constraint network: chaining mined implications before checking exclusions

- [x] Direct follow-up ask: "build a genuine constraint network from the
      mined axioms (transitivity chains, exclusivity webs) and check
      global consistency across that network" - v0.44's `axiom_violations`
      only caught a violation when the exclusion's antecedent was one of
      the subject's own *directly-asserted* facts; a subject who only
      reaches that antecedent by composing two or more separately-mined
      `DiscoveredImplication` rules was invisible to it.
- [x] **`_implied_closure`**: plain forward-chaining BFS reachability over
      a directed graph built from mined implication edges - the standard
      tool for propagating directed Horn-clause-like rules, not a bespoke
      invention. `axiom_violations` gained an `implications` parameter
      (default `None`): when given, a subject's held facts are chained
      through the implication graph *before* checking exclusions, so a
      violation is caught even several mined rules away. `implications`
      threaded through `resolve_shard_conflicts`/`ground_resolved_shards`
      too. `None` (default) is an exact no-op - `_implied_closure` returns
      the held facts unchanged with no graph, reproducing v0.44's
      behaviour precisely.
- [x] **Deliberately not built by reusing `sheaf.py`'s H0/H1 cohomology
      machinery, for the same reason v0.44 didn't either:** its
      restriction edges assert equality (`restrict_u * x_u == restrict_v *
      x_v`) between two scalar stalks - implication is directional (A
      implies B does not mean A equals B) and exclusion means "not both",
      neither of which is an equality constraint. Forcing them through
      that linear algebra would misrepresent the semantics rather than
      reuse them; transitive-closure graph search is the correctly-scoped
      tool for this instead.
- [x] Verified end to end, not just as a hand-authored toy: a two-hop
      chain (`born_on=mars -> is_a=martian -> breathes=co2`, then
      `breathes=co2 excludes needs=oxygen`) is mined for real via
      `axiom_mining.discover_implications` from an ontology's own data at
      full confidence, and `axiom_violations` correctly chains through it
      to catch a subject holding `born_on=mars` and `needs=oxygen` -
      genuinely contradictory once composed, invisible to the flat v0.44
      check (checked directly: the flat check finds nothing on the exact
      same triples).
- [x] New test file `test_axiom_constraint_network.py` (6 tests): the flat-
      check-misses/chained-check-catches pair, a subject missing a link in
      the chain correctly not flagged, the `implications=None` backward-
      compatibility no-op, the full `resolve_shard_conflicts` integration,
      and the real-mining-API end-to-end case.

## v0.46 — fixing the crosstalk limitation: three proposals, one worked

- [x] Direct follow-up to v0.44's own disclosed second limitation
      (excising a literal violating fact doesn't fix downstream accuracy
      once a relation is already exempted from consensus - most wrong
      answers there are hypervector crosstalk from a flood of *other*,
      unrelated noise triples, not the one fact excised). Asked for three
      candidate fixes, one abstract, then told to implement the abstract
      one first and fall back to the other two (combined) if it didn't
      work. It didn't - both outcomes are recorded below, not just the
      one that worked.
- [x] **First attempt (rejected, kept as a documented negative result,
      not deleted): dissolve `classify_multi_valued_relations`'s hard
      gate into one continuous per-shard trust weight.**
      `shard_disagreement_energy`/`shard_trust_weights` compute cross-
      shard vote disagreement with `multi_valued_relations` never
      populated at all (no relation ever exempted), turning it into a
      Boltzmann weight instead of a hard threshold. Measured directly on
      the exact 50/50 real/garbage-shard scenario that motivated this:
      every shard, real and noise alike, came back with disagreement
      energy `0.0` and therefore weight `1.0` - zero discrimination, not
      weak discrimination. Root cause, found by direct measurement, not
      assumed: a tie (`consensus[pair] = None`) is deliberately treated
      as neutral so genuine multi-valued plurality (`has_friend`-like)
      isn't punished - but with a small shared entity pool at this
      contamination level, *most* real-vs-noise collisions on a given
      pair land as an exact 1-vs-1 tie too, structurally indistinguishable
      from genuine plurality using only that one pair's vote count. A
      real, informative result about the limits of per-pair vote
      statistics, not a bug to patch further.
- [x] **Working fix ("option 1 + 2 combined"): a signal that never looks
      past one shard's own boundary, applied at retrieval time, not
      grounding time.** `internal_collision_energy` - the continuous
      generalisation of v0.41/v0.42's `is_structurally_regular` - counts
      how often a single shard asserts the same `(subject, relation)` pair
      *more than once within itself*, immune to both failure modes found
      so far because it never touches cross-shard votes or contamination
      volume at all (option 1: bootstrap trust from something contamination
      can't reach). `shard_regularity_weights` turns this into a Boltzmann
      weight (`exp(-inverse_temperature * energy)`,
      `inverse_temperature=60.0` chosen by direct measurement - tested 5
      to 100, false-positive result identical across the whole range).
      Discovered along the way, also by direct measurement, not assumed:
      weighting a shard's own triples uniformly *inside* its bundle is a
      provable no-op (`bundle`/`normalize` projects every element back
      onto the unit circle regardless of a shared scalar weight) - the
      weight has to live in `qa.ask_sharded`'s new `shard_weights`
      parameter instead (option 2: damp a noise shard's occasional lucky
      resonance at retrieval time), which scales `coherence` before both
      the cross-shard argmax and the `known` floor check.
- [x] **Measured result, not a marginal improvement:** on the exact 50/50
      scenario that failed open at 2/10 false positives, the fix restores
      0/10 - and restores answer accuracy to exactly the noise-free
      baseline (9/10, not a strawman 10/10 - one entity is a naturally
      harder resonance case at this scale regardless of noise), not just
      "better than before." Pushed further: checked (not assumed) to hold
      at 62%, 71%, 76%, 86%, 91%, 94%, and 96% contamination, every point
      giving the identical 0/10 false positives and noise-free-matching
      accuracy - a qualitatively higher honest ceiling than v0.43's
      consensus mechanism, which failed open already at 50%. This does not
      claim there is no ceiling at all, only that none was found in the
      range actually tested.
- [x] `Ontology.ground_shards_with_trust`/`ground_shards_with_regularity_
      trust`: paired grounding-plus-weights helpers (the first for the
      rejected signal, the second for the working one) sharing one private
      grounding helper - both excise literal `axiom_violations` outright
      (still a real, independent, crisp signal, kept regardless of which
      continuous weighting is used) and return `(memories, weights)` for
      `ask_sharded`'s `shard_weights`.
- [x] New test file `test_shard_trust_weighting.py` (8 tests): the
      collision-energy unit checks, the first attempt's documented
      failure to discriminate at 50/50, the working signal's
      discrimination check, the full real-API false-positive fix, the
      noise-free-accuracy-parity check, and the 94%-contamination ceiling
      check.

## v0.47 — v0.46 fails on real data; a signal that doesn't need redundancy to work

- [x] Direct follow-up to running Phase 0 of the substrate-assessment
      roadmap: the user asked to test the v0.41-v0.46 robustness pipeline
      against real relational data (Nations - 14 entities, 55 relations,
      1992 triples, no redundant/repeated triples anywhere) instead of
      this project's own synthetic domain, at 50% injected-noise
      contamination. Result: baseline (no weighting) gave 21/40 recall,
      37/38 false positives on genuinely-absent facts - expected, and
      consistent with the whole session's story. **v0.46's
      `shard_regularity_weights` gave 0/40 recall, 0/38 false positives -
      everything reported unknown, not fixed.**
- [x] **Diagnosed, not just observed.** Measured `internal_collision_
      energy` directly on the real+noise mix: real shards scored
      `0.026-0.194`, noise shards `0.013-0.127` - the ranges fully
      overlap. Cause: Nations' relations are genuinely, densely multi-
      valued (unlike the synthetic domain's single-valued `is_a`-style
      relations), so real shards legitimately duplicate `(subject,
      relation)` pairs just as often as noise does by chance - the same
      structural confusion `is_structurally_regular` (v0.41/42) already
      had on this same dataset, inherited by its "continuous" successor.
      Tried excluding relations `classify_multi_valued_relations` already
      flags as multi-valued from the collision count - failed *worse*: on
      Nations' 14-entity pool, that classifier gets fooled by contamination
      volume well before 50% (all 55 of 55 relations got misclassified
      multi-valued), collapsing every shard to weight 1.0. Tried the
      earlier-rejected `shard_trust_weights` (cross-shard vote
      disagreement) directly against real data instead of assuming it
      would fail the same way - it failed *differently and worse*: real
      shards scored `0.67-0.91` disagreement, noise `0.32-0.67` -
      **inverted**, because with 55 genuinely multi-valued relations spread
      thin across only 25 shards, "the majority filler" is often not a
      coherent concept at all, and a real shard holding one of several
      simultaneously-true answers looks like a minority dissenter against
      a fragmented, meaningless vote.
- [x] **The conceptual root cause, stated precisely, not just patched
      around:** every mechanism from v0.43 through both v0.46 attempts
      detects a shard *disagreeing* with something, which requires the
      data to contain redundant, independently-repeated assertions of the
      same fact - true by construction of the synthetic domain those
      mechanisms were built and validated against, false of Nations
      (confirmed directly: all 1992 triples are unique, zero duplicates)
      and of most real knowledge graphs, where a fact is normally stated
      exactly once. This is a scope mismatch between the whole voting-
      based lineage and real, non-redundant relational data, not a
      miscalibrated constant.
- [x] **`entity_connectivity_score`/`shard_connectivity_weights`: a signal
      that needs no redundancy at all.** Instead of asking whether a
      shard's claims are *contradicted*, it asks whether a shard's pattern
      of *which entities it talks about* looks like real-world structure
      (skewed - some entities are simply more connected than others,
      universally true of real relational data) or uniform random
      sampling (what noise looks like, regardless of which real entities
      it happens to reuse). Self-normalising (a sigmoid over each shard's
      *z-score* relative to the mix's own mean/std), not a fixed constant
      like `shard_regularity_weights`'s `inverse_temperature=60.0` -
      deliberately, since a raw connectivity score's scale depends on the
      dataset's own size/degree distribution and there is no portable
      absolute number the way a `[0, 1]`-normalised fraction has.
- [x] **Measured result, checked across seeds and contamination levels,
      not one lucky run:** on the same real Nations scenario, false
      positives fell from 37/38 to 0/38, and recall of real stored facts
      *improved simultaneously* from 21/40 to 31-37/40 across three
      different noise seeds and two contamination levels (50% and 62%) -
      not a trade-off, both got better together every time it was
      checked. Honest, disclosed limit: the weight distributions still
      overlap at the tails (real `0.38-0.97`, noise `0.05-0.54` in the
      first run) - not the clean, zero-overlap separation
      `shard_regularity_weights` achieved on its own synthetic domain -
      and this has only been validated on one real dataset (Nations) so
      far, not yet UMLS/Kinship or anything larger.
- [x] `Ontology.ground_shards_with_connectivity_trust`: same structure as
      `ground_shards_with_regularity_trust` (literal `axiom_violations`
      still excised outright, every surviving triple bundled plainly,
      continuous trust returned alongside the memories for `ask_sharded`'s
      `shard_weights`), using the connectivity signal instead.
- [x] New test file `test_shard_connectivity_weighting.py` (6 tests): a
      synthetic "hub and leaf" fixture capturing the one property that
      actually matters (skewed real-world connectivity, zero redundant
      triples) so the suite doesn't need network access to validate this
      - the real Nations numbers above are the actual evidence, documented
      here rather than re-fetched in CI.

## Phase 0 — substrate self-assessment against real data (ongoing, not versioned code)

A reflective "what is this substrate now, is it any good" conversation
produced an honest shortcomings list and a phased roadmap (see this
project's session history for the full list - the short version: no
generalization mechanism, capacity ceiling unexplained since v0.39, fixed
hyperparameters never tuned per dataset, "physics" is a NumPy metaphor
not a new substrate, engineering completeness gaps, no incremental
update path, no demonstrated end-to-end utility). Phase 0 of that roadmap
is testing what Zeuss actually claims to do against real data, replacing
self-invented synthetic benchmarks - each item below is a genuine test,
not a demo, and each was run before being trusted.

- [x] **Established first, precisely, why Zeuss cannot do standard link
      prediction.** `Codebook.symbol()` assigns every entity an
      independently random hypervector (`random_hypervector`) - nothing
      anywhere shapes entity representations from data, so two entities
      that behave identically across every relation still get unrelated
      vectors. Filtered-ranking link prediction on Nations (14 entities,
      55 relations) measured MRR ~0.38 - landed almost exactly at the
      predicted "near chance" level once this was understood, not a
      benchmark artifact, a structural absence of any generalization
      mechanism. Also found the Nations benchmark itself is compromised
      for this purpose regardless (ConvE's own author pulled its Nations
      results from that paper, citing inverse-relation test leakage) -
      UMLS is the clean alternative from the same paper (ConvE MRR .94),
      not yet re-run given the more relevant finding above already
      explained the gap.
- [x] **Phase 0(a): the v0.41-v0.47 robustness pipeline, tested on real
      data for the first time, broke immediately and got fixed for real -
      see the v0.47 entry above for the complete derivation.** Short
      version: v0.46's crosstalk fix (validated only on synthetic,
      single-valued, redundant-by-construction data) gave zero
      discrimination on real Nations data; the fix that actually works
      (`shard_connectivity_weights`, entity-popularity-skew based, needs
      no redundant assertions) was verified via the real shipped API:
      false positives 37/38 -> 0/38, recall 21/40 -> 31-37/40 on real
      data, checked across seeds and contamination levels before being
      trusted.
- [x] **Phase 0(b): does `chain()`/`chain_sharded()` perform genuine
      multi-hop logical entailment on real, non-synthetic hierarchical
      data - the capability this project actually claims (iterate one
      wave operator, walk a relation's transitive closure), as opposed to
      the generalization capability just established it lacks?** Used
      UMLS's real `isa` semantic-type hierarchy (500 triples, genuine
      chains up to depth 4, 129/133 subjects with *multiple* real
      parents - not a toy single-parent tree). Success criterion wasn't
      "did it reach the one ancestor I expected" (multi-parent structure
      means several different chains are simultaneously true) - it was
      "was every hop actually a real, stored `isa` edge" (logical
      validity) and "did it reach genuine multi-hop depth". Result,
      checked across all 133 real subjects with `isa` facts, not a small
      sample: **238/238 hops taken were logically valid - zero
      hallucinated edges** - and 84/133 (63%) reached genuine multi-hop
      (2+) depth, several reaching 3 hops through real chains (e.g.
      `acquired_abnormality -> anatomical_abnormality ->
      anatomical_structure -> physical_object`). One honest, checked-not-
      ignored anomaly: `physical_object` has a real stored outgoing edge
      (`physical_object isa entity`) that `chain_sharded` failed to
      recover (empty chain) - most likely because `physical_object` is
      the single most heavily-referenced *object* in this hierarchy
      (dozens of other entities point to it), creating enough retrieval
      crosstalk to push its own one outgoing edge below the coherence
      floor. It correctly reported unknown rather than guessing wrong -
      the honest-failure design held even where recall fell short. **This
      is the first unambiguously positive capability confirmation on real
      data this whole investigation has produced** - v0.44 through v0.47
      were entirely about finding and fixing robustness failures, and the
      link-prediction test confirmed a real incapability; this is Zeuss
      doing, correctly, exactly what it was built to do, on data it never
      saw during design.
- [x] **Phase 0(c): does `axiom_violations` (v0.44) + implication chaining
      (v0.45) correctly catch an injected logical contradiction on real
      data, while leaving every genuine real fact untouched?** First tried
      `discover_all_exclusions` (v0.38) on UMLS's full 46-relation data to
      mine a real exclusion automatically - produced 396,597 "exclusions"
      from `min_support=3`, almost entirely statistical noise (with UMLS's
      sparse, type-level data, anything never co-observed from a handful
      of examples gets called "excluded" at 100% confidence) - a real,
      honest finding about the mining approach's own real-data limit, not
      usable for this test. Used a hand-verified real constraint instead,
      the same "independently-sourced, not self-mined" discipline v0.44's
      own docstring requires: checked directly (not assumed) that no real
      UMLS entity has both `chemical` and `organism` as `isa` ancestors -
      zero violations across all 133 real subjects - a genuine, disjoint
      domain constraint. Injected a false `isa=chemical` fact onto each of
      the 16 real organism-descended entities and checked whether
      `axiom_violations` (asymmetric: `isa=organism` trusted as evidence
      against the excluded `isa=chemical`, not a symmetric ban - a
      symmetric version was tried first and correctly, if unhelpfully,
      flagged the real fact too whenever a subject held both sides, with
      no way to tell which one was injected) caught the injected fact
      without touching the real one. Result: **16/16 injected
      contradictions caught, 0 false positives** on real data. Also found,
      honestly: this specific real dataset asserts redundant direct edges
      to most ancestor levels (not just the immediate parent), so no
      genuine "only catchable by chaining through several isa hops" case
      existed for this particular pair to isolate - v0.44 alone and v0.45
      with chaining gave identical results here, which is itself the
      correct, expected behaviour (chaining is a safe no-op when nothing
      new is reachable beyond a subject's direct facts), not a failure to
      demonstrate v0.45's added power specifically.

## Phase 1 — engineering (Phase 0 of the substrate-assessment roadmap complete; this is the follow-up)

- [x] **Scaling-wall profiling: found and precisely located the real
      bottleneck, not just confirmed "it gets slower".** A first profile
      varying `n_per_class` (20/80/200, growing shard count and total
      entity count together, matching how a real dataset actually grows)
      measured per-shard-visit cost climbing sharply - 37.7ms -> 209.1ms
      -> 836.9ms as shards went 1 -> 4 -> 10 - which looked like a
      super-linear problem in `ask_sharded`'s own cross-shard loop.
      Checked properly instead of trusting that reading: a second profile
      fixed the KB entirely (1600 triples, 804 entities held constant)
      and varied *only* `shard_size` (hence only shard count: 1, 2, 4, 8).
      Per-shard cost stayed flat - 1920.7ms, 1810.4ms, 2013.3ms, 1704.5ms
      - no growth trend at all across a tight band. **Conclusion:
      `ask_sharded`'s cross-shard loop is genuinely `O(shards)`, not the
      bottleneck.** The real cost driver is `_cleanup`'s
      `dimensional_collapse` step, which scores the query residue against
      *every entity in the whole codebook* on every single call,
      regardless of how many shards that data is split across - so total
      KB vocabulary size, not shard count, is what makes queries slower as
      a knowledge base grows. This matters for where future optimisation
      effort should go: sharding more aggressively doesn't fix this, and
      a real production-scale KB (thousands-millions of entities) would
      need `dimensional_collapse`'s own entity-comparison cost addressed
      (e.g. an approximate/indexed nearest-neighbour prefilter before the
      full comparison) rather than more shards.
- [x] **`chain_sharded` robustness parity**: it never got the `shard_
      weights` mechanism `ask_sharded` gained in v0.46/v0.47 - a real,
      previously-disclosed gap (see the roadmap's own "what's still
      open" notes). `chain_sharded` now accepts the same `shard_weights`
      parameter, applied identically at *every* hop of the chain (not
      just the first): `coherence * weight` decides both which shard's
      answer wins that hop's argmax and whether the hop clears
      `COHERENCE_FLOOR`. `None` (default) is an exact no-op, checked
      directly against the un-parameterised call, not assumed. Verified
      the weighting genuinely changes outcomes, not just that it's
      accepted: two shards carrying complete, mutually-exclusive two-hop
      chains for the same subject (`socrates -> human -> mortal` vs
      `socrates -> martian -> alien`) - suppressing either shard's weight
      to `0.0` forces the *entire* resulting chain to come from the other
      one, at both hops, not a mix. New tests in `test_sharding.py` (3
      added). **Honest scope note**: this brings the same *mechanism*
      `ask_sharded`'s weighting has to the multi-hop case - it has not
      itself been independently measured for real-data robustness the
      way `ask_sharded`'s weighting was (v0.46/v0.47's whole real-Nations-
      data story); that would be a natural next real-data test, not yet
      done.
- [x] **GPU kernel: explicitly retired for now, not silently left open**
      - see the `v0.4` entry above for the full reasoning (no CUDA
      hardware available in this development environment, and writing a
      kernel that can't be parity-tested against real hardware would
      violate the project's own rule for adding one, not honour it).
      Deliberate decision, not an oversight; revisit if GPU hardware
      becomes available.
- [x] **Incremental grounding - the last Phase 1 item, completing it.**
      Every `ground*` method rebuilds its whole bundle from scratch on
      every call - a real operational gap: Zeuss could only be used as a
      static, batch-compiled snapshot, never a living memory that absorbs
      new facts over time. Root cause `bundle()` couldn't be built around
      before now: its final `normalize()` is *per-element* phase
      projection (`z / abs(z)`), not one global magnitude rescaling - once
      a vector is normalised, its true pre-normalisation magnitude at
      each dimension is gone, so a new fact can't be cheaply folded into
      an already-normalised bundle. The *raw*, pre-normalisation complex
      sum can be added to trivially, though - that's exactly what
      `bundle()` computes before its own final `normalize` call.
      `IncrementalMemory`/`IncrementalShardedMemory` keep that raw sum as
      their actual state, normalising only on read (`.vector`), making
      `add()` `O(dim)` regardless of how many facts already exist,
      instead of `O(n)` triples re-processed. `Ontology.
      incremental_memory()`/`incremental_sharded_memory(shard_size=80)`/
      `add_incremental(memory, s, r, o)` are the convenience API -
      `add_incremental` adds to both `self.triples`/entities *and* the
      memory structure in one call. Verified, not assumed, to be
      numerically equivalent to the batch path at every scale checked:
      single-memory and sharded incremental results match
      `bundle()`/`ground_sharded()`'s batch output to float64 tolerance
      (similarity > 0.9999999), and the sharded case is real-API tested
      through `ask_sharded` at the same 400-triple/100-subject scale
      `ground_sharded` was originally validated at (>= 8/10 correct,
      matching that original measurement). New test file
      `test_incremental_grounding.py` (7 tests), including the
      operational point directly, not just the algebra: query a memory,
      add a fact, query again, without ever calling `ground()` a second
      time - the new fact is there.

## Phase 2 — closing the generalisation gap (`Ontology.refine_entity_vectors`)

Phase 0 established precisely why Zeuss performs near chance on standard
link prediction: `Codebook.symbol` mints an independently random vector
per entity, so nothing shapes representations from data, and two
entities with identical relational behaviour get unrelated vectors. Asked
for a scoped, de-risked plan to close this rather than jump straight to
implementation - given three genuinely different technical routes (A:
gradient-trained embeddings, proven-to-work precedent like HolE, but the
biggest departure from "rules emerge from dynamics, not optimisation"; B:
hypervector-native iterative neighbour blending, no gradients, no
literature precedent; C: a separate classical collaborative-filtering
layer, cheapest to test, doesn't touch the substrate) - staged
cheapest-to-falsify first. Directed to try B first, the harder, more
in-character, less-certain option.

- [x] **Tier 0 (synthetic pilot): a real bug caught the mechanism looked
      broken before it actually worked.** First run: 0/9 correct on every
      configuration, baseline included, every query `known=False`. Before
      concluding a clean negative, noticed the actual tell: all three
      refined configurations (different `rounds`/`alpha`) were bit-for-
      bit *identical* to each other - a genuine effect should vary with
      its own hyperparameters. Traced it: `entity()` looks entities up
      under `f"ENT:{name}"`, but the refinement write-back used the bare
      name - every refined vector was silently discarded, so `entity()`
      kept returning the untouched original random vector the whole time.
      Fixed, and also fixed a second, subtler confound (baseline's
      vectors were minted in a different access order than refinement's
      pre-refinement state, itself enough to change which random vectors
      entities got, per `Codebook.symbol`'s own documented "depends on
      when it was first requested" behaviour) - forced the same eager-
      minting order on both sides for a fair comparison. Re-run with both
      fixes: **9/9 correct** (up from 2/9 baseline, chance = 1/3) across
      every `rounds`/`alpha` combination tried, with a clear similarity
      margin (0.029-0.039 toward the true answer vs 0.004-0.006 toward
      wrong ones), not a marginal effect.
- [x] **Tier 1 (real data, two datasets, not one):** synthetic wins in
      this project have not reliably transferred before (v0.46's whole
      story) - checked properly rather than declared a win. Reused the
      *identical* filtered-ranking protocol already used to measure the
      generalisation gap in Phase 0, so results are directly comparable,
      not a new metric invented for this to look good on.
      - **Nations** (14 entities, 55 relations): tail MRR 0.389 -> 0.500
        (`rounds=3,alpha=0.5`) / 0.544 (`rounds=8,alpha=0.2`); Hits@1
        0.229 -> 0.299 / 0.358; Hits@10 0.771 -> 0.905 / 0.945. Head
        direction improved by a comparable margin (MRR 0.382 -> 0.519-
        0.529). Every metric improved in every configuration.
      - **Honest concern raised before trusting this**: Nations has a
        documented flaw (its own original author, ConvE's Dettmers,
        pulled its numbers from that paper specifically for heavy
        inverse-relation redundancy) - since refinement propagates
        signal through a relation *and* its trained inverse, this result
        alone couldn't rule out partly measuring that redundancy rather
        than genuine relational-similarity learning.
      - **UMLS** (135 entities, 46 relations, no known redundancy quirk -
        chosen specifically to rule the Nations concern out) confirmed it
        far more dramatically, not less: baseline tail MRR 0.041 (near
        the ln(135)/135 ~ 0.036 chance rate, matching Phase 0's own
        measurement) -> **0.651** at `rounds=3,alpha=0.5` - a 16x
        improvement; Hits@1 0.000 -> 0.600; Hits@10 0.100 -> 0.760.
        Disclosed, not glossed over: this comparison used `n=25` vs
        baseline's `n=40` (a background-duration limit killed the
        combined 3-config script mid-run after the baseline row alone
        took 34.6 minutes; re-run as a smaller, separate job) - a real
        sample-size mismatch, though an effect this large (16x MRR, 0%
        to 60% Hits@1) is not plausibly explained by ordinary sampling
        variance alone.
- [x] **`Ontology.refine_entity_vectors(rounds=3, alpha=0.5)`**: iterated,
      synchronous neighbour-vector blending using the bind/bundle algebra
      already in the substrate (each neighbour bound under the relation
      wave connecting it, matching grounding's own encoding) - a label-
      propagation-style power iteration, genuinely no gradient/loss/
      training loop anywhere. Mutates `self.codebook` in place; must be
      called before any `ground*` method. New test file
      `test_entity_refinement.py` (5 tests) captures the property that
      matters (relationally-distinct groups, a fact withheld and
      inferable only through neighbour structure) synthetically, so the
      suite doesn't need network access - the real Nations/UMLS numbers
      above are the actual evidence, documented here rather than
      re-fetched in CI, the same pattern this project already uses for
      every other real-data validation.
- [x] **A proper `rounds`/`alpha` sweep (40 points, not 2-3 ad hoc ones)
      found a real problem with the default the Nations/UMLS numbers
      above actually used, and it shipped fixed, not after the fact.**
      Both real-data runs only ever checked exact-match accuracy
      (`Hits@1`/`MRR`) - never whether `ask()`'s own honest `known`
      confidence flag still fired correctly for facts recovered
      perfectly. A grid across the fast synthetic domain, measuring
      inference accuracy, known-fact recall accuracy, *and* the `known`
      rate together, found three distinct regimes:
      - Low `alpha` (0.1-0.3), any `rounds`: genuinely broken - entity
        identity washes out fast enough that even directly-held facts
        stop being recalled correctly (`recall_top1` drops to ~0.67),
        and `known` never fires (0.00) - the over-smoothing failure mode
        this design was always a candidate for.
      - `alpha=0.5` (the *original* default used for the Nations/UMLS
        numbers above, `rounds>=8`): inference and recall both reach
        1.00 - but `known` *never* fires (0.00), even for facts recalled
        with perfect accuracy. Invisible until this sweep, because
        nothing had checked it.
      - `alpha=0.7` (`rounds>=5`) or `alpha=0.9` (`rounds>=20`):
        inference, recall, and `known` all reach 1.00 *together* - no
        tradeoff. **Default changed to `rounds=8, alpha=0.7`**, inside
        this region with margin, and a dedicated regression test
        (`test_refine_entity_vectors_default_preserves_honest_
        confidence`) added specifically for the failure mode just found.
- [x] **Re-verified on real data - and the sweep's fix did NOT transfer
      the way it did on the synthetic domain, reported honestly rather
      than rounded up.** Checked `known_rate` on both real datasets at
      baseline, the old default, and the new default:
      - **Nations**: `known_rate` was never degraded at either default -
        baseline 0.925/0.915, old 0.940/0.925, new 0.925/0.915
        (tail/head). Nothing here for the new default to fix.
      - **UMLS**: `known_rate` genuinely drops with refinement (baseline
        0.917/1.000 -> 0.750/0.833) - but **identically** at the old
        and new default. Switching `alpha` 0.5 -> 0.7 recovered none of
        it on real data, unlike the clean 0.00 -> 1.00 jump the
        synthetic sweep found. Real, but smaller than the synthetic
        domain's total collapse, and not controlled by this
        hyperparameter choice on real data - something about
        refinement's general effect at UMLS's scale/density, not
        something `alpha`/`rounds` fixes within the range tested.
      - Default kept at `rounds=8, alpha=0.7` anyway - real-data accuracy
        is a wash-to-slight-improvement over the old default (UMLS head
        MRR 0.440 -> 0.564; tail and Nations roughly tied) - but the
        specific claim that this default *resolves* the honest-
        confidence question is withdrawn; it doesn't, on real data. The
        same category of finding as v0.46's crosstalk fix working on
        synthetic data and failing on real data - a synthetic result
        that didn't transfer, disclosed rather than hidden.
- [x] **Traced the UMLS `known_rate` drop to its actual mechanism instead
      of leaving it as an unexplained real-data gap - two plausible
      theories were tested and both were wrong.** Not graph-density-driven
      over-smoothing: mean pairwise entity-similarity traced round-by-round
      on both the sparse synthetic pilot (mean degree ~10) and UMLS (mean
      degree ~155) showed nearly identical trajectories (0.085 vs 0.096 by
      round 8) despite a 16x degree difference, and forcing the synthetic
      pilot through the identical sharded `ground_sharded`/`ask_sharded`
      path UMLS uses still showed zero degradation (12/12 known either
      way). Not `_cleanup`'s `dimensional_collapse` candidate-restriction
      step either: on real UMLS queries, refinement *shrank* the live
      candidate set (`k_live` 127.7 -> 85.7 of 135) rather than growing it,
      with settle-vs-raw disagreement staying negligible (0/20 -> 2/20) -
      the opposite of what a "bigger ambiguous set" theory needed.
      **The actual mechanism:** `bind`/`unbind` recovery assumes atomic
      vectors are close to independent; this method's entire purpose is to
      correlate related entities with their neighbours, directly in
      tension with that assumption. The aggregate crosstalk this produces
      scales with vocabulary size (entities x relations sharing the
      space), not per-pair correlation strength alone - exactly why the
      tiny synthetic domain and UMLS's much larger one show similar
      per-pair convergence while only the larger vocabulary degrades
      retrieval. Not a mistunable hyperparameter (why the sweep above
      couldn't fix it on real data) - a structural tension between what
      this method needs to do and what the substrate's core algebra needs
      to stay reliable.
- [x] **The fix is architectural, not a threshold: separate the two uses
      instead of trading one off against the other.** Refined vectors now
      live in a distinct `entity_refined()` codebook namespace that
      `entity()`/`ground()`/`ground_sharded()`/ordinary `ask()`/
      `ask_sharded()` never read - `entity()` is provably untouched by
      `refine_entity_vectors` now, not just measured to hold on the
      samples checked. Generalisation becomes fully opt-in: `ground_
      refined()`/`ground_sharded_refined()` build a memory from refined
      vectors, and `ask`/`ask_sharded` gained `subject_vector`/
      `entity_vectors` parameters (both `None` by default, an exact
      no-op) to probe and score candidates the same way.
      - A first attempt at the opt-in seam (overriding only the query's
        probe, `subject_vector`, against an otherwise-ordinary memory)
        was itself a real, caught mistake - it broke the existing
        synthetic regression tests (`refined_correct` fell to 2/6, *below*
        the 3/6 unrefined baseline). Traced rather than patched around:
        the original design's power came from a *fully consistent*
        refined universe, where the stored memory was also built from
        refined vectors, not just the probe. Isolated by testing each
        combination on the same synthetic domain: refined probe alone
        against a raw memory recovered 2/6; adding refined candidate
        comparisons recovered 3/6; adding a memory also built from refined
        vectors recovered 5/6 - matching the original, pre-separation
        numbers exactly. The real seam needs all three consistent
        (probe, candidate scoring, memory), not any one alone.
      - A second, easy-to-miss confound caught along the way: `Codebook.
        symbol` mints lazily off one shared RNG stream, so pre-warming
        only `entity()`'s own keys before refinement (rather than every
        key `ground()` would eventually touch, in the same order) changed
        which random vector every entity got even though refinement never
        writes to `ENT:` - measured on real UMLS as an "ordinary" query's
        known_rate coming back 0.950/0.975 instead of the true
        no-refinement baseline's 0.917/1.000. Fixed by pre-minting every
        triple's vectors in insertion order before refining.
- [x] **Re-verified the new, separated architecture on real UMLS data -
      not assumed to inherit the pre-separation numbers above.** Ordinary
      queries after refinement are provably identical to a no-refinement
      baseline by construction (`entity()` untouched) and unit-tested
      (`test_refine_entity_vectors_does_not_touch_entity`); re-measuring
      that on UMLS would only re-confirm the invariant at real-data cost,
      so it wasn't repeated. The one genuinely new number is the opt-in
      generalising path itself (`ground_refined()` memory, `subject_
      vector`/`entity_vectors` both `entity_refined`, `rounds=8,
      alpha=0.7`, n=30): tail MRR 0.041 (no-refinement baseline) -> 0.507,
      Hits@1 -> 0.367, Hits@10 -> 0.733, `known_rate` 0.833 (vs baseline
      0.917); head MRR -> 0.532, Hits@1 -> 0.333, Hits@10 -> 0.933,
      `known_rate` 0.933 (vs baseline 1.000). The generalisation benefit
      survives the architectural separation - and the confidence gap is
      real but *much smaller* than the synthetic pilot's total collapse
      (0/6 known there vs 83-93% here) - domain-size-dependent, not a
      fixed property of the mechanism. (An earlier number quoted for this
      path, `known_rate` 0.825/0.950, was a mislabelled leftover from the
      pre-separation design - an ordinary query after refinement had
      mutated `entity()` directly, not a measurement of this opt-in path -
      corrected here and in `test_entity_refinement.py`.)
- [x] **Traced *why* even the isolated generalising path's `known_rate`
      sits below baseline, rather than leaving the gap as a bare number.**
      Two explanations were live: either the refined vector space has a
      systematically different coherence *scale* (a threshold-calibration
      artifact - `COHERENCE_FLOOR` is one fixed constant, 0.08, used
      against both raw and refined spaces alike), or genuinely-inferred
      (withheld/test) queries just carry a structurally weaker signal than
      directly-grounded ones (expected, not a flaw). Distinguished by
      querying facts the KB *already holds directly* through the
      generalising path too - same fact, only the vector space differs -
      and comparing to the same facts queried ordinarily:
      - Synthetic pilot: known facts' mean coherence *rose* under the
        generalising path (0.090 ordinary -> 0.159 generalising, all
        12/12 staying `known=True` either way), while genuinely-withheld
        facts averaged 0.043 - a quarter of the generalising known-fact
        mean, comfortably explaining why they miss the floor.
      - Real UMLS (15 training triples, same protocol): identical
        direction - 0.110 ordinary -> 0.169 generalising (+53%), 15/15
        `known=True` in both configs, zero drop.
      Both domains rule out H_scale outright: if anything, the refined
      space *amplifies* confidence for facts it was actually built from,
      the opposite of what a miscalibrated-threshold explanation needs.
      The `known_rate` gap on genuinely-inferred queries is exactly what
      it should be - a real, structural difference in signal strength
      between "recalling a stored fact" and "inferring an unstated one
      through neighbour correlation," not a bug in the floor or the
      refined representation.
- [x] **Honest scope, stated plainly, not left implicit:** this changes
      entity *initialisation* only; Phase 1's scaling-wall finding
      (query latency driven by total codebook size, not shard count) is
      unaffected either way. Does not touch v0.39's still-open "why
      doesn't raising `dim` rescue the single-bundle ceiling" question.
      Not yet compared against a real benchmark's own published baseline
      under an identical protocol (Nations' citable numbers were found
      to be withdrawn by their own author; UMLS's ConvE-cited MRR of .94
      was noted, not rigorously reproduced under matching splits/
      protocol) - this closes a real, directly-measured gap in Zeuss's
      own before/after numbers, not a claim of parity with trained
      embedding models.

## Phase 2 addendum — v0.39's dim-ceiling question, finally explained

- [x] **v0.39 measured that raising `dim` 8x (8192->65536) barely moved
      coherence on a 400-triple single bundle (0.044->0.039, if anything
      slightly worse) and left *why* as an explicitly open question.
      Traced to the mechanism, not left unexplained any longer.** `bind`/
      `bundle`'s shared `normalize()` step forces every *element* of a
      summed vector back onto the unit circle - a phase-only projection
      that `unbind()` re-applies to whatever it reads out of a bundle.
      Isolated at the `bind`/`bundle`/`unbind` primitive level (no
      `Ontology`/`qa` involved - N bound `(role, filler)` pairs bundled
      into one memory, then unbound by one target role and correlated
      against the true filler, 8 seeds per cell): the *shipped* pipeline's
      recovered similarity for the correct filler is capped by bundle size
      `N` alone and genuinely flat across `dim in [2048, 8192, 32768,
      131072]` (N=400: 0.048 -> 0.045 -> 0.043 -> 0.043) - reproducing
      v0.39's own "no improvement, if anything slightly worse" pattern
      exactly, not just qualitatively. A pipeline built from the same
      primitives but with the final phase projection removed (raw complex
      sum, multiply by `conj(role)`, no re-normalisation, straight into
      the existing `similarity()`) instead stays near its analytically-
      predicted constant expectation of 1.0 at *every* `N` tested (up to
      800), and - the actual capacity signature - its separation from a
      wrong candidate's score tightens sharply as `dim` grows (N=400,
      wrong-candidate score: 0.198 at dim=2048 -> 0.003 at dim=131072).
- [x] **The mechanism, in one sentence:** forcing every dimension back to
      unit modulus after summation is a *nonlinear* operation (dividing by
      `|signal + noise|`, not adding noise to a clean signal), which makes
      the *expected* recovered value a decreasing function of bundle size
      `N` alone, with no dependence on `dim` - `dim` can only tighten an
      empirical estimate around that already-`N`-capped expectation, never
      lift the expectation itself. Un-projected (raw) superposition has a
      *constant* expectation regardless of `N` (each interfering term is
      independently mean-zero), so there `dim` does the job standard HRR
      capacity theory predicts - reducing the *variance* around that
      constant, and thus separating correct from wrong candidates ever
      more reliably as `dim` grows. Both the ceiling's insensitivity to
      `dim` *and* why that insensitivity is a direct, provable consequence
      of one specific design choice (phase-only projection at read-out),
      not a fundamental limit of complex-phasor HRR bundling in general,
      are now established - one of the three candidates v0.39 itself
      listed (`OBJ_SHIFT`, `bundle`'s normalisation, or "a genuine limit of
      complex-phasor HRR") turned out to be the answer, and it is fixable
      in principle, not load-bearing physics.
- [x] **A first test of this was flawed, correctly diagnosed rather than
      discarded as a null result.** Comparing a "lossy" `bundle()` against
      an "amplitude-preserving" `bundle()` (global L2-rescale instead of
      per-element), both fed through the *shipped* `unbind()`, gave
      bit-for-bit identical numbers at every `(N, dim)` - not a bug in the
      test, but exactly what the algebra predicts: projecting a product's
      phase gives the same angle whether the projection happens before or
      after the multiplication, so whichever bundle variant fed it, the
      shipped `unbind()`'s own final projection erases the distinction
      regardless. The loss happens at *whichever point* phase-only
      projection is first applied to a bundle's contents, not specifically
      inside `bundle()` - a real negative result that redirected the next
      test rather than a dead end.
- [x] **Not acted on immediately in the same entry as the diagnosis - a
      design decision, not a quick patch, decided with the user before
      touching `hypervectors.py`.** `bind`/`unbind`/`bundle`'s
      normalize-everywhere contract is relied on throughout the substrate
      (every atomic hypervector is unit-modulus by construction; keeping
      the algebra closed under that property is what makes further
      binding/bundling composable at all), so changing `unbind` itself
      was never on the table - see the additive fix below, which changes
      nothing about `unbind`'s existing contract or any existing caller.

## Phase 2 addendum, continued — the actual capacity fix

- [x] **Fully additive, not a change to any existing function.** New:
      `hypervectors.unbind_raw` (identical to `unbind` minus the final
      projection - a no-op difference for atomic-atomic unbinding, which
      is why nothing existing needed to change); `Ontology.ground_raw()`
      (returns an `IncrementalMemory` - Phase 1's existing pre-
      normalisation accumulator turned out to be exactly the raw state
      this fix needs, reused rather than duplicated - via its new `.raw`
      property) and `Ontology.step_raw()` (the `step()` unbind chain,
      three stages, with every stage using `unbind_raw`); `qa.ask_raw()`
      (the read-out, scored via `similarity`, deliberately *not*
      `_cleanup`'s `phase_lock`).
- [x] **Why `similarity`, not `phase_lock` - a real, measured mistake
      caught before shipping, not assumed correct because `_cleanup`
      already used it.** `phase_lock` is `abs(mean(a * conj(b)))` - fine
      at the ordinary pipeline's scale, but its expectation for a
      genuinely wrong candidate is *positive* (Rayleigh-distributed, not
      zero), and that bias grows with bundle size. Measured directly: an
      entity absent from the KB entirely scored ~0.55 coherence via
      `phase_lock` at dim=8192/N=400 - uncomfortably close to a real
      fact's ~1.0, nearly defeating the fix's own guess-detection.
      `similarity`'s real-part convention *is* zero-mean for a wrong
      candidate regardless of bundle size (the property the whole
      derivation above rests on) and was the metric actually validated in
      the primitive-level experiment - switching to it dropped the same
      scenario's worst-case guess to ~0.20.
- [x] **Verified end-to-end at v0.39's own exact tested "large dim"
      (65536), not a new, unrelated scale.** v0.39 measured that dim
      8192->65536 (8x) barely moved coherence on a 400-triple single
      bundle (0.044->0.039). Through the new raw pipeline at that *same*
      dim=65536/400 triples: every sampled fact recovered correctly and
      confidently (coherence ~0.94-1.06), a genuinely-absent entity's
      worst-case score measured 0.198 - a 4.76x margin, the separation
      v0.39's own pipeline never achieved at any `dim` tested.
      `RAW_COHERENCE_FLOOR = 0.5` sits with real margin on both sides of
      that measurement. `test_capacity_raw.py`'s
      `test_ask_raw_rescues_the_v039_ceiling_at_v039s_own_tested_dim` is
      the regression test for this, at the identical scale.
- [x] **Honest, disclosed scope of the fix - checked, not assumed to hold
      everywhere.** The margin narrows as the candidate pool grows
      (`known` is a *max* over every entity in `onto` - an extreme-value
      statistic, the same "more chances for noise to spike" effect
      v0.40/v0.41 already found for shard selection, now showing up at
      the candidate level): N=800 at the *same* dim=65536 narrows the
      margin to 3.18x; doubling `dim` to 131072 to match the bigger
      bundle restores it to 5.10x - textbook capacity scaling (margin ~
      `sqrt(dim/N)`), the thing the old pipeline never showed at any
      scale. At a much lower dim/N ratio (dim=8192/N=400, ~20x, a fifth
      of the ratio actually validated) worst-case guesses reached ~0.60,
      *above* the shipped default floor - `RAW_COHERENCE_FLOOR`'s default
      is safe for the ~80x-165x dim/N ratios actually measured, not for
      arbitrarily small ones, and `ask_raw` takes an explicit
      `coherence_floor` override for callers operating outside that
      range rather than silently trusting the default. `ask_raw` is also
      scoped narrower than `ask` on purpose - no `axiom_bias`, no
      `dimensional_collapse`/`settle` refinement pass, no
      `entity_vectors` override - the minimal, directly-tested capacity
      fix, not a drop-in replacement for `ask`'s full feature set.

## Phase 2 addendum, continued — does `ask_raw` obsolete sharding?

- [x] **Asked directly, since v0.40's sharding exists specifically to work
      around the exact single-bundle ceiling `ask_raw` now fixes**: at a
      2x-v0.41-scale stress test (1600 triples, single relation, 5 shared
      fillers), is `ask_sharded` (the shipped v0.40/41 mitigation) still
      needed, or does raw grounding replace or combine with it? Three
      architectures compared head-to-head on the identical KB: (A) the
      shipped `ask_sharded` (20 shards of 80, dim=8192); (B) raw sharding
      (2 shards of 800, dim=131072 - the exact dim/N ratio already
      calibrated to a ~5x margin); (C) one single raw bundle, no sharding
      at all (dim=262144, scaled to match the same margin). All three:
      10/10 correct, 0/5 false positives on genuinely absent entities -
      raw grounding fully preserves v0.40/41's robustness guarantee,
      sharded or not.
- [x] **The bigger, unexpected finding was speed, not capacity.** B and C
      answered queries ~70-100x faster than A (B/C: ~0.5-0.8s/query; A:
      ~48s/query), with C (no sharding at all) architecturally simplest
      *and* marginally faster per query than B, at the cost of a longer
      one-time grounding pass (78s vs 41s - paid once, not per query).
- [x] **Isolated *why*, rather than crediting the whole gap to "raw beats
      sharding" - it's both.** `ask_sharded` calls `ask()` per shard,
      which scores every shard's candidates against the *entire*
      ontology's ~1605-entity codebook via `_cleanup`'s `dimensional_
      collapse`+`settle`, not narrowed to that shard's own ~85 entities -
      a real, independent inefficiency, not inherent to sharding as an
      idea. Reimplementing the *exact same* `dimensional_collapse`/
      `settle` machinery but scoped to each shard's own entities dropped
      the cost to ~3s/query (~16x) - identical accuracy and false-
      positive rate. That leaves a real, separate ~5x gap against raw's
      ~0.6s (scoped-old ~3s vs raw ~0.6s) attributable to `settle`'s own
      iterative energy-refinement cost, not candidate-pool size. Both
      effects are real and independently disclosable - the original
      ~70-100x figure conflated an unrelated, independently-fixable
      inefficiency with `ask_raw`'s genuine inherent advantage.
- [x] **Fixed the independent inefficiency, additively.** `_cleanup`/
      `_entity_codebook`/`ask` gained an optional `candidate_names`
      parameter (`None` default = exact prior behaviour, full-ontology
      scoring); `ask_sharded` gained `shard_entities` (positionally
      aligned with `memories`, `None` default = exact prior behaviour).
      `Ontology.ground_shards_with_entities`/`ground_sharded_with_
      entities` compute each surviving shard's own entity names in the
      *same* filtering pass as grounding, so a shard dropped for being
      structurally irregular can never desync the two lists - a real risk
      that would exist if entity names were computed separately against
      the original, unfiltered shard list. `tests/test_sharded_candidate_
      scoping.py` (5 tests) checks correctness/robustness equivalence
      between scoped and unscoped, and the alignment guarantee
      specifically under a dropped-shard scenario.
- [x] **Honest scope: this fixes `ask_sharded`'s inefficiency, it doesn't
      answer "should you use raw, sharded, or raw+sharded."** That's now
      a real architectural choice rather than a foregone conclusion: raw
      alone needs `dim` proportional to total KB size (a practical ceiling
      of its own once `dim` gets impractically large); sharding (raw or
      not) keeps `dim` bounded per shard at the cost of O(shards) query
      work. Not measured here: where raw-alone's practical `dim` ceiling
      actually bites, or whether raw+sharding's combination pushes total
      capacity past what either achieves alone - both real, open follow-
      ups, not yet started.

## Phase 2 addendum, continued — raw's own memory ceiling, and combining with sharding

- [x] **Where a single raw bundle's own practical ceiling bites, measured
      directly rather than assumed.** Every entity vector gets minted once
      and cached forever in `Ontology.codebook` (`Codebook.symbol` never
      evicts), so memory is `O(total_entities * dim)` regardless of any
      query-time trick - and since a fixed margin needs `dim` roughly
      proportional to `N` (margin ~ `sqrt(dim/N)`, calibrated earlier at
      dim/N~164 for ~5x), memory for a single bundle at constant margin is
      `O(N^2)` - quadratic, not linear. Measured on this development
      machine (16.9GB total, ~5GB typically available) at a deliberately
      *less* conservative dim/N=80 ratio, just to see the trend cheaply:
      N=400/dim=32000 -> 0.21GB; N=1000/dim=80000 -> 1.29GB; N=2000/
      dim=160000 -> a projected 5.13GB, close enough to this machine's
      real ceiling that the run was skipped rather than risked. Both
      completed points recovered every fact correctly with zero false
      positives, so the quadratic cost buys nothing extra in reliability -
      it's pure overhead from keeping everything in one bundle.
- [x] **Raw+sharded at the identical total N, for a direct contrast, not
      a different scale that only sounds better.** 3200 triples (2x the
      earlier 1600-triple sharded test) as 8 raw shards of 400
      (dim=32768 fixed - shard size and dim don't need to grow with total
      KB size, only shard *count* does): 1.68GB projected, 1.76GB measured
      RSS after grounding - roughly **8x less memory than the ~13.1GB a
      single raw bundle would need at this same N** (extrapolating thread
      1's own measured trend). 4/4 sampled facts correct, 0/2 false
      positives - accuracy and guess-detection both held at 2x the prior
      sharded-raw scale. This is the direct answer to "does combining
      raw+sharding push capacity further": yes, substantially, on memory -
      sharding's `O(total_entities * fixed_dim)` (linear in `N`) is the
      real reason, not something specific to this one measurement.
- [x] **Honest wrinkle flagged, then traced to its actual cause rather than
      left as "Python overhead, not investigated further" - it was the
      identical bug as `ask_sharded`'s, just not fixed there yet.** The
      8-shard/dim=32768 configuration's per-query cost (~4.5s) being
      *slower* than an earlier 2-shard/dim=131072 test's (~0.7s), despite
      *lower* predicted compute, was initially attributed to generic
      per-candidate Python/dict-lookup overhead. A real stress test at
      N=6000 (1.875x further, 15 shards) confirmed that guess was
      incomplete: per-query latency ballooned to ~17-26s - a ~3.8x
      slowdown for a 1.875x increase in N, matching `(15*6005)/(8*3205) ~
      3.5x` almost exactly. That ratio is the signature of the *same*
      unscoped-full-codebook bug already found and fixed in `ask_sharded`
      earlier - except `ask_raw`'s own hand-rolled per-shard query loop
      never got the equivalent fix, so every shard was scoring against the
      *entire* ontology's entities regardless of that shard's own size:
      `O(shards * total_entities) = O(N^2/shard_size)`, not `O(N)`.
- [x] **Fixed identically to the `ask_sharded` fix, additively.** `ask_raw`
      gained the same `candidate_names` parameter `_cleanup`/`ask` already
      had (`None` default = exact prior behaviour). Caught a real bug
      while wiring it up, not assumed correct: `ask_raw` indexes its
      candidate list positionally (`names[i]` for the winning candidate),
      which breaks with `TypeError: 'set' object is not subscriptable` the
      moment a caller passes a `set` (the natural type for "this shard's
      own entities", and exactly what the fix's own first real usage
      needed) - fixed by normalising to `list(candidate_names)` up front.
      `tests/test_capacity_raw.py` gained regression coverage for both the
      set-acceptance bug and the scoping-changes-the-answer-set property.
- [x] **Re-ran the N=6000 stress test scoped, and the fix is complete, not
      partial.** Per-query latency dropped from ~17-26s to ~0.7-1.0s (a
      ~21x recovery) with byte-identical coherence values for every known
      query - scoping doesn't change *which* answer wins, only how many
      irrelevant candidates get compared along the way. Cross-checked
      against the earlier N=3200 point re-run under the same scoped
      methodology (not the original, unscoped N=3200 number, to keep the
      comparison apples-to-apples): 0.50s/query at 8 shards -> 0.88s/query
      at 15 shards, a 1.76x cost increase for a 1.875x increase in shard
      count - genuinely linear, not the ~3.5x quadratic signature the
      unscoped version showed at the same two points.
- [x] **Where this leaves the architecture choice, corrected from the
      previous, incomplete conclusion:** with `ask_raw`'s scoping fix,
      raw+sharded is now `O(N)` in *both* memory and query cost, not a
      memory-for-latency trade - the "honest wrinkle" above turned out to
      be a fixable implementation gap, not an inherent property of
      combining raw grounding with sharding. Single-bundle raw remains
      memory-quadratic and impractical past the low thousands of triples
      on typical dev hardware (16-32GB) - that part of the tradeoff still
      stands. Audited whether `ask_sharded` (the *normalized* pipeline)
      has the same set-vs-list indexing risk its own `shard_entities`
      could hit - it doesn't: `_cleanup`'s candidate list always passes
      through `dimensional_collapse`, which reads `codebook.names()`
      (always a proper list, regardless of what iterable built the
      codebook), before anything gets indexed positionally. Already
      covered empirically too, not just by reading the code -
      `test_ask_sharded_with_shard_entities_matches_unscoped_answers`
      passes `shard_entities` as a list of *sets* (exactly what `ground_
      sharded_with_entities` returns) and was green before this entry was
      even written.

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
