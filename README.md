# Zeuss

> ## Project status: closed, with its central question answered
>
> Zeuss was built to test one architectural claim: that unifying continuous
> (evidence-based) and discrete (logical) computation in a **single** substrate
> beats running them as two systems in sequence. **That claim was tested and
> refuted.** On the project's own canonical case — a deliberate data
> contradiction where raw retrieval is near a coin flip (48%) and a logical
> constraint demonstrably fires (31/60 seeds) — injecting the constraint *inside*
> the energy landscape and applying the *same* constraint as a post-hoc re-rank
> both reached 60/60 and **disagreed on 0 of 60 seeds**. Sweeping the constraint
> from soft to hard found no regime where the unified form wins, and one where it
> is markedly **worse** (37/60 vs 60/60 at low weight), because settling dilutes
> weak logical evidence instead of amplifying it.
>
> **What survives is real:** the logic layer itself is valuable (48% → 100%), the
> substrate has 318 passing tests and two-dataset validation, and several findings
> generalise beyond this repo — most notably *why* hyperdimensional bundling
> capacity does not respond to dimension (see below).
>
> **What does not:** the architectural premise that the logic had to live inside
> the substrate. A conventional retriever plus a rule engine reaches the same
> answer, more cheaply.
>
> Full derivations, measurements, failed predictions and retired claims are in
> [`docs/ROADMAP.md`](docs/ROADMAP.md). The headline results are summarised in
> [Findings](#findings) below.

**A unified continuous↔discrete computational substrate.**

Zeuss is a research platform for the idea in [`VISION.md`](VISION.md): a computing
paradigm where determinism and probability are the *same* representation seen at
different phase-coherence levels, rather than two systems bolted together in
software. There are no rules stored as text and no weights stored as static
matrices — information lives as continuous trajectories in a dynamic space that
**crystallises into deterministic truth when measured.**

## The three frontiers, as three tiers

| Tier | Package | What it does |
|------|---------|--------------|
| **1 — Compute kernels** | `zeuss.tier1_kernels` | Phase-interference & topological-collapse ops. Portable NumPy reference now; Triton/CUDA drop-ins later. |
| **2 — Substrate math** | `zeuss.tier2_substrate` | Complex-phase hypervector algebra (FHRR/VSA), entropy-driven collapse, energy-landscape attractors, wave resonance. **Runs on CPU today.** |
| **3 — Logic compiler** | `zeuss.tier3_logic` | Grounds predicate graphs & logical axioms into continuous fuzzy-logic energy terms (Łukasiewicz / Gödel / product t-norms). |

Each maps to a frontier from the vision:

- **Frontier 1 — topological dynamic collapse** → `tier2_substrate/collapse.py`: entropy drops, the space loses dimensionality and snaps onto a discrete symbol. *The logic is the collapse of the space itself.*
- **Frontier 2 — micro-energy landscapes** → `tier2_substrate/energy.py`: axioms are attractor basins; the system settles into the logical ground state by minimising energy. Deterministic rules are zero-temperature wells; probability is thermal noise.
- **Frontier 3 — algorithmic resonators** → `tier2_substrate/resonance.py`: deduction is constructive interference; uncertainty is destructive interference.

## Quick start

This repo already has a `.venv` at the project root with the `dev` and `jax`
extras installed, so JAX is available and `ZEUSS_BACKEND=auto` picks it as the
active backend by default (not just NumPy). Use it instead of system Python,
or `pytest` will silently skip every JAX-gated test:

```bash
# from the project root (this folder)
.venv\Scripts\activate                # Windows; source .venv/bin/activate on Linux/macOS
python -m zeuss info                  # backend / capability report - confirm JAX is active
python -m zeuss demo                  # run the collapse demo
pytest -q                             # full suite, JAX-backed tests included
```

Setting up a fresh environment instead:

```bash
python -m pip install -e ".[dev,jax]"  # editable install + test deps + JAX
pytest -q                              # CPU-only (NumPy) if JAX isn't installed
```

No install needed to try it:

```bash
PYTHONPATH="src:." python demos/demo_collapse.py
```

Expected demo output shows entropy falling from ~2.3 bits to 0 as a noisy probe
collapses onto the symbol `red`, and a deterministic energy descent reaching a
crisper ground state than the thermal one.

## Backends

The substrate is written against one array namespace (`zeuss.backend.xp`):

- **NumPy** — default, always available, CPU. This is what tests and the demo use.
- **JAX** — optional (`pip install jax`), for `grad`/`vmap`/`jit` and XLA on GPU/TPU. Set `ZEUSS_BACKEND=jax`. On Windows, use WSL2 or the CPU wheel.
- **Triton/CUDA** — optional Tier-1 GPU kernels (Linux + NVIDIA GPU). See `src/zeuss/tier1_kernels/triton_kernels.py`.

## Coding from your phone

This repo is set up to run under **Claude Code Remote Control** so you can start a
session on this machine and steer it from the Claude app on your phone. See
[`docs/REMOTE_CONTROL.md`](docs/REMOTE_CONTROL.md).

## Layout

```
src/zeuss/
  backend.py            # numpy/jax array-namespace switch
  tier1_kernels/        # compute kernels (reference + triton stub)
  tier2_substrate/      # hypervectors, collapse, energy, resonance
  tier3_logic/          # ontology (networkx), compiler (t-norms)
  cli.py                # `python -m zeuss ...`
  tier4_synthesis/       # DSL, encoding, and GP search for program synthesis
demos/demo_collapse.py, ask_demo.py
tests/                  # pytest suite
docs/                   # ARCHITECTURE, ROADMAP, REMOTE_CONTROL
```

## Findings

Results that hold up, each measured rather than argued. Full derivations, the
methodology, and the predictions that turned out wrong are in
[`docs/ROADMAP.md`](docs/ROADMAP.md).

**Why a fixed confidence threshold fails on a normalised bundle.** The most
generalisable result here — and the claim itself was corrected after this repo
was published, by a follow-up experiment across four VSA flavours.
`bundle()`/`unbind()` project every element back onto the unit circle. That
projection is nonlinear on a noisy sum and **caps the absolute recovered
similarity at a value set by bundle size**: measured flat at ~0.15 (N=32) across
a 64× sweep of `dim`, where reading the raw un-projected sum gives ~1.0.

What it does **not** do is cap capacity. Crosstalk still falls as `dim` grows, so
the margin between the correct item and its best competitor improves either way
(FHRR quantised: 2.3 → 19.6 over the same sweep). The same pattern holds for MAP
and BSC; for HRR it is provably a non-issue, since unit-norm rescaling is a global
scalar and cosine similarity is scale-invariant.

So the real failure was a **scale mismatch, not a capacity law**: `COHERENCE_FLOOR`
is a fixed absolute threshold applied to a quantity whose scale is capped, so
raising `dim` could never lift it over the bar. An earlier version of this section
claimed "capacity ignores dimension" — that was wrong, and the measurements
correcting it live in a separate follow-up study.

**Literature check, done rather than assumed:** neither half of this is new.
Plate's original 1995 capacity analysis (IEEE TNN, Appendix A) already gives the
raw, un-normalised bundle signal as `E=1` regardless of bundle size `k` — exactly
this project's raw-pipeline number — and normalising that bundle to unit length
(dividing by its norm, `~√k` for `k` quasi-orthogonal terms) is a two-line
corollary that produces the `~1/√k`, dim-independent ceiling measured here. The
discriminability side is the general superposition-capacity law unifying
HRR/FHRR/MAP/BSC (Frady, Kleyko & Sommer 2018, arXiv:1707.01429). So this is a
validated re-derivation of ~30-year-old theory, not a new result — what's
project-specific is stating scale and capacity as two separately-measured
quantities, and validating the constant by fitting it blind and checking it
against this project's own previously-unexplained failure point.

**The scale law.** Memory is `entities × dim × 16` bytes, and `dim` tracks *shard*
size, so `memory × shard_count` is invariant — you can trade memory for latency
but the product is fixed by the graph. Crucially, that memory buys the *abstention
margin*, not accuracy: recovery saturates at `dim/shard ≈ 20`, and everything
above only widens the known/unknown margin (as `√ratio`). Random quasi-orthogonal
codes cost ~655× more memory per entity than a trained model that learns a
compressed one.

**Calibrated abstention, scoped honestly.** The confidence signal is an
**answerability** detector (AUC 0.9907 UMLS, 0.9699 Nations against
deliberately-constructed unanswerable queries) — not a correctness detector. It
does not know whether its own answer is right: selective-prediction curves put it
no better than random abstention, and a margin-based rescue attempt failed on both
datasets.

**The raw pipeline.** Reading from pre-normalisation memory is both faster and
more accurate than the normalised path (UMLS MRR 0.8371 vs 0.7449 at ~45× the
speed; 33.8× faster multi-hop chains), and the advantage transfers to a second,
structurally different dataset.

**Benchmarked against a published baseline.** UMLS filtered link prediction, same
splits and pooled protocol as ConvE: MRR 0.8371 against ConvE's 0.94 — behind a
purpose-built trained model, far above the substrate's own no-generalisation
baseline of ~0.041.

**The logic layer, re-tested on real data.** "The logic layer earns its keep"
rested on one hand-built case (a data contradiction). Hunted for a real dataset
with a genuinely informative constraint — Kinship and Countries both ruled out
by measurement (either uninformative or, worse, confidently wrong 19/24 times).
WN18RR's `_hypernym` relation is the first that held up: 97.8% of synsets have
exactly one direct hypernym, so a structural no-cycles constraint (a known
descendant can never also be an ancestor) is sound rather than merely
correlational — 0/300 false vetoes against real held-out test triples. Run for
real on 198 held-out cases: BASELINE 10.6% → JOINT 21.2% → POST_FILTER 32.8%.
The logic layer's benefit is no longer resting on a toy case — but with a
realistic multi-candidate field (not the socrates case's 2), POST_FILTER
*strictly* beats JOINT (65 vs 42 correct), not just ties it: removing several
attractors reshapes which basin `settle`'s limited iteration converges to, a
cost a static re-rank never pays.

Does it generalise past one relation? Reused the same `_hypernym`-derived
constraint — mined once — as a cross-relation veto for three other WN18RR
relations: still 0 false vetoes on every one. Real substrate results replicate
the shape on the two with a usable sample (`_has_part` n=22: 0.0%→9.1%→31.8%;
`_synset_domain_topic_of` n=10: 10.0%→70.0%→80.0%) — baseline stays low, both
mechanisms clear it substantially, POST_FILTER is never worse than JOINT. A
bonus pattern the single-relation result couldn't show: disagreement rate
tracks candidate-field size cleanly across all three (6.1 candidates → 10%
disagree; 11.2 → 48.5%; 40.3 → 63.6%), exactly matching the settle-dynamics
explanation. See `docs/ROADMAP.md`, "Post-closure — does the logic layer earn
its keep on real, non-hand-built data?" and its "generalise past one
relation?" follow-up.

Does it generalise past one *kind* of constraint? Everything above derives
its constraint from the graph's own topology (no cycles). Tested a
genuinely different kind — WordNet's own lexicographer classification
(`noun.person`, `noun.animal`, `noun.plant`, …) as a declared-disjointness
constraint, the same role YAGO's schema plays when it declares `Person` and
`Location` disjoint — on `_instance_hypernym`, the one relation the earlier
constraint couldn't reach at all (instances are leaves, so "no cycles"
never fires there). 0/122 false vetoes on real test data, firing on 78.7%
of queries — the complementary case. Real result: BASELINE 33.3% → JOINT
41.7% → POST_FILTER 44.0%, same shape again. One honest asymmetry: unlike
the no-cycles constraint (exactly sound by construction), this one inherits
WordNet's real edge cases — false-veto rate across all 11 relations ranges
from 0% up to 46.2% (`_member_of_domain_region`, where a term's cultural
domain routinely crosses these categories) — checked per-relation before
trusting any of them. See `docs/ROADMAP.md`'s "a genuinely different KIND
of constraint" entry.

## Status

**Closed**, with one post-closure follow-up. The architectural thesis was
tested and refuted, and the project is archived at that answer rather than
abandoned mid-question — that verdict stands. A later, narrower question
("does the one surviving piece, the logic layer, hold up on real data?") was
asked and answered positively above; it doesn't reopen the architectural
question, it just replaces "demonstrated once, on a hand-built case" with
"demonstrated on a real dataset, at real scale." The substrate math is real,
tested (318 passing / 14 skipped), and runnable on CPU. The GPU kernel tier
was explicitly retired — this machine has no CUDA GPU, so a kernel could be
neither implemented nor parity-tested honestly.
