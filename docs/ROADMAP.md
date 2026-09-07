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

## v0.4 — GPU kernels (Frontier 3, faster)
- [ ] Triton `phase_interference` and `topological_collapse_step`.
- [ ] Parity tests vs. NumPy reference; benchmark harness.
- [ ] Optional FFT-based binding for very high dimensions.

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
- [x] `CLAUDE.md`'s good-first-tasks entry for this (open since v0.17)
      removed - no longer an open item.

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
