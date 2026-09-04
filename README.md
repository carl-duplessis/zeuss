# Zeuss

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

```bash
# from the project root (this folder)
python -m pip install -e ".[dev]"     # editable install + test deps
python -m zeuss info                  # backend / capability report
python -m zeuss demo                  # run the collapse demo
pytest -q                             # 17 tests, CPU-only
```

No install needed to try it:

```bash
PYTHONPATH="src:." python examples/demo_collapse.py
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
examples/demo_collapse.py
tests/                  # pytest suite
docs/                   # ARCHITECTURE, ROADMAP, REMOTE_CONTROL
```

## Status

`v0.1.0` — the substrate math is real, tested, and runnable on CPU. The GPU
kernel tier and the JAX-native autodiff path are scaffolded with clear TODOs;
see [`docs/ROADMAP.md`](docs/ROADMAP.md).
