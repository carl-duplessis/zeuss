# CLAUDE.md — working agreement for Zeuss

This file is loaded automatically by Claude Code at the start of every session.
Read `VISION.md` for the north star and `docs/ARCHITECTURE.md` for how the code
is organised.

## What this project is

Zeuss is a research substrate for a **unified continuous↔discrete computing
paradigm**. Determinism and probability are one representation at different
phase-coherence levels — not two subsystems glued together. Keep that framing
in mind: prefer designs where a "rule" *emerges* from the dynamics (collapse,
energy minimisation, resonance) over designs that special-case a symbolic path.

## Architecture (three tiers)

- `src/zeuss/tier1_kernels/` — hot compute ops. NumPy reference is the ground
  truth; Triton/CUDA kernels must match it (see `tests/test_kernels_parity.py`).
- `src/zeuss/tier2_substrate/` — the heart: `hypervectors` (FHRR/VSA algebra),
  `collapse` (entropy → discrete), `energy` (attractor basins), `resonance`
  (wave interference).
- `src/zeuss/tier3_logic/` — `ontology` (predicate graph → hypervectors) and
  `compiler` (axioms → fuzzy t-norm energy terms).

## Conventions

- **Backend:** write substrate math against `zeuss.backend.xp` so it runs on
  NumPy (default) or JAX. Never hard-import `jax` at module top level — guard it.
  Same for `networkx` and `triton` (all optional).
- **Complex phasors:** hypervectors are unit-modulus `complex128`. Use
  `normalize()` after any operation that can push elements off the unit circle.
- **Determinism:** thread an explicit `numpy.random.Generator` (`rng`) through
  anything stochastic. Default seeds make tests reproducible; don't call the
  global RNG.
- **Purity:** keep tier-2 ops as pure functions (arrays in, arrays out) so the
  JAX autodiff port stays mechanical.
- **Style:** ruff, line length 100. Type hints on public functions. Docstrings
  should say *which frontier / idea* a piece implements.

## Definition of done for a change

1. `pytest -q` is green (17 tests today; add tests for new behaviour).
2. `python -m zeuss demo` still runs and tells a coherent story.
3. New optional deps stay optional (guarded imports + `pyproject` extras).
4. If you add a GPU kernel, add a parity test against the NumPy reference.

## Good first tasks (see docs/ROADMAP.md for the full list)

- Add a JAX `grad`-based energy-descent variant of `energy.settle`
  (`energy.settle_grad`) — the `xp` routing it needs is already done (v0.2).
- Implement a sequence encoder (bind + permute) and a cleanup-memory decoder.
- Add a `matplotlib` experiment that plots entropy vs. β and energy vs. step.
- Improve asymmetric-recursion (Fibonacci-class) discovery reliability
  (`tier4_synthesis`, see `docs/ROADMAP.md` v0.17): still open after v0.17 -
  adding one more training example fixed seeds 4/7 but broke seed 1, a
  genuine whack-a-mole, not a fixable-with-more-data problem. Needs a real
  fix (e.g. tune the `delta_p1` schedule itself), not another example-count
  tweak — that direction was tried and shown not to generalize across seeds.

## Don't

- Don't add heavy dependencies to the core; the Tier-2 substrate must keep
  running on NumPy alone.
- Don't store "rules" as if/else tables when they can be attractors or
  interference patterns — that's the whole point of Zeuss.
