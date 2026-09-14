# ADAMAI / Zeuss Engine — Architectural Specification (aspirational)

Document date: 2026-09-04. Status: design blueprint / pitch doc, not an
implementation record.

> **Note on scope.** This document is an aspirational full-system spec — it
> describes a much larger system than what currently exists in this repo. See
> `docs/ARCHITECTURE.md` for what is actually implemented (tier1 kernels,
> tier2 substrate: hypervectors/collapse/energy/resonance, tier3: ontology +
> a fuzzy-logic compiler) and `docs/ROADMAP.md` for near-term work. Treat the
> "4-Tier Engine" and "Bleeding-Edge Enhancements" below as candidate
> directions, not shipped components. Compare against `VISION.md` for the
> original framing this doc extends.

> **STATUS (project closed).** Beyond the scope note above, five specific claims
> in this document were later *measured to be false or unsupportable*. Recorded
> here so the spec is not read as validated:
>
> | Claim | Finding |
> |---|---|
> | "Hallucination Risk: Zero" | The confidence signal detects **answerability**, not correctness (AUC 0.99/0.97 for "can the KB answer this", but no better than random at judging its own answer). |
> | "Absolute deterministic mathematical proofs" | `tier4_synthesis/synth.py` verifies against supplied examples, not for all inputs — its own docstring says so. |
> | "Is this a world-first system? Yes." | Prior art exists for each ingredient; the closest relative is Frady & Sommer's vector-symbolic finite-state machines in attractor networks (2024). The defensible gap is narrower: the fuzzy-logic-over-KG literature is uniformly *gradient-trained*. |
> | "Compute Overhead: Low" | True only at small scale: ~655× more memory per entity than a trained model; web scale is 6–52 TB, out of reach architecturally. |
> | "Clifford multivectors, D > 10,000" | Taken literally that is 2^10000 blades. `tier2_substrate/geometric.py` implements the honest `Cl(n≤6)` rotor layer instead. |
>
> Most components described below *do* exist in scoped form. A grounded audit of
> what is implemented versus what is aspirational, plus a phased build plan, was
> produced separately; the architectural premise it rested on has since been
> refuted (see `README.md`). Full measurements: `docs/ROADMAP.md`.

## Executive Summary

This document formalizes the complete technical discussion, system breakdown,
and architectural analysis for the ADAMAI / Zeuss Engine.

The system represents a deterministic, non-transformer neuro-symbolic
cognitive architecture. It unifies non-Euclidean vector symbolic memory,
continuous multi-valued logic, topological consistency auditing, thermodynamic
energy-state annealing, and Bayesian program synthesis.

Unlike modern Large Language Models (LLMs) that rely on autoregressive
next-token prediction, this engine treats computation as a continuous physical
relaxation process into a zero-error ground state (T → 0). The LLM layer is
strictly relegated to an optional, external natural language interface.

## Part 1: LLM Interface & Dependency Analysis

### 1.1 Is the System Reliant on an LLM?

No. The core system is 100% independent of Large Language Models.

The cognitive substrate, knowledge representations, deductive reasoning loops,
energy-state minimization, and Abstract Syntax Tree (AST) execution operate
natively without an LLM.

| LLM Required? — YES (Optional): Translation & Parsing | LLM Required? — NO (100% Native): Reasoning, Memory, & Action |
| --- | --- |
| Translating English to symbols | Deductive & Fuzzy Logic Processing |
| Formatting output for humans | Hyperdimensional Vector Memory |
| Parsing open-ended user intent | AST Code & Task Execution |
| | Deterministic Proof Verification |
| | Thermodynamic Phase Collapse (T → 0) |

### 1.2 Role of the LLM (If Attached)

If an LLM interface is integrated, its sole purpose is acting as an untrusted,
natural language adapter:

- **Ingress (Parsing):** Translates chaotic human language into structured
  symbolic predicates and initial boundary conditions.
- **Egress (Formatting):** Translates deterministic state outputs and
  execution telemetry into human-readable text.

### 1.3 System Behavior Under LLM Removal

If the LLM is completely disconnected:

- The core engine remains fully operational.
- Input is received via direct JSON payloads, gRPC/API contracts, binary
  sensor streams, or formal logic predicates.
- Output is emitted as deterministic code execution, structured state
  changes, or verified graph updates.

## Part 2: The 4-Tier Engine Architecture

```
                  [ EXTERNAL ENVIRONMENT / SENSORS ]
                                  │
                                  ▼
                    [ Active Inference Drive ]
                Calculates Expected Free Energy:
               Pragmatic Value + Epistemic Value
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    ZEUSS NEURO-SYMBOLIC CORE                     │
├──────────────────────────────────────────────────────────────────┤
│  Tier 1: Continuous Łukasiewicz / Gödel t-Norm Logic             │
│  Tier 2: Clifford Geometric Algebra (GA-HDC / FHRR) Substrate    │
│  Tier 3: Topological Knowledge Graph & Sheaf Cohomology Auditor  │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
               [ Thermodynamic Annealing (T → 0) ]
           Relaxes Continuous Trajectories into Ground States
                                  │
                                  ▼
               [ Bayesian AST Program Synthesizer ]
           Executes Verified Code & Deterministic Actions
```

> This tier numbering is a different scheme from the repo's actual
> `tier1_kernels` / `tier2_substrate` / `tier3_logic` layout — don't conflate
> the two when reading code.

### Tier 1: Continuous Logic Gate Substrate

- **Mechanism:** Continuous t-Norm Logic (Łukasiewicz / Gödel logic
  operators).
- **Function:** Operates continuously without discrete step-discontinuities.
  It allows truth values to exist across a continuous range [0, 1], bridging
  fuzzy boundary conditions smoothly into crisp Boolean proofs.

### Tier 2: Hyperdimensional Memory Substrate (GA-HDC / FHRR)

- **Mechanism:** Fourier Holographic Reduced Representations (FHRR) upgraded
  with Clifford Geometric Algebra (GA-HDC).
- **Function:** Stores facts, concepts, and relationships as complex-phase
  multivectors in high-dimensional space (D > 10,000). Relations are encoded
  as rotational bivectors and rotors, making hierarchical sub-graph nesting
  and memory transformations mathematically lossless in single-step
  operations (O(1) without backpropagation).

### Tier 3: Topological Knowledge Graph & Sheaf Auditor

- **Mechanism:** Deterministic Graph Database integrated with a Cellular
  Sheaf Cohomology Auditor.
- **Function:** Tracks global state consistency. While local logic rules may
  be valid individually, the Sheaf Auditor calculates restriction maps across
  active data streams. If the first cohomology group is non-zero
  (H¹ ≠ 0), the engine mathematically detects a global deadlock or
  contradiction before code execution occurs.

### Tier 4: Thermodynamic Bayesian Engine & AST Synthesizer

- **Mechanism:** Continuous Energy-Based Model (EBM) paired with Bayesian
  Abstract Syntax Tree (AST) Synthesis.
- **Function:** System actions are generated as probabilistic candidate ASTs.
  The system anneals the energy landscape (T → 0). As thermal noise
  approaches zero, candidate programs collapse into the exact zero-energy
  ground state representing verified, bug-free execution code.

## Part 3: Bleeding-Edge Theoretical Enhancements

To achieve maximum performance, non-contradiction, and dynamic autonomy, five
advanced computational mechanics are integrated into the pipeline:

### 3.1 Active Inference & Expected Free Energy

- **Concept:** Derived from Karl Friston's biological physics framework. The
  engine does not passively wait for prompts. It continually minimizes
  Expected Free Energy (EFE).
- **Execution:** Before executing an action, it balances Pragmatic Value
  (achieving goals) with Epistemic Value (reducing uncertainty). If input
  parameters are missing, it autonomously executes low-risk discovery
  actions to gather environment state data.

### 3.2 Sheaf Cohomology Topological Consistency

- **Concept:** Algebraic topology applied to concurrent system logic.
- **Execution:** Evaluates local-to-global consistency. It prevents rule
  collisions across parallel swarm agents or microservices by guaranteeing
  global section convergence.

### 3.3 Clifford Geometric Algebra (GA-HDC)

- **Concept:** Multi-vector spaces incorporating scalar, vector, bivector,
  and trivector components.
- **Execution:** Replaces standard flat vector spaces. Memory
  transformations, temporal shifts, and structural hierarchies become simple
  multivector rotations.

### 3.4 Liquid Time-Step Dynamics

- **Concept:** Neural differential equations with time-continuous dynamics
  (dt).
- **Execution:** During Tier 4 phase collapse, the annealing rate (ΔT/Δt)
  dynamically adapts: slowing down during high-complexity constraint
  evaluations and accelerating through simple state transitions.

### 3.5 Event-Spiking Asynchronous Activation

- **Concept:** Neuromorphic event-driven gating.
- **Execution:** Vector spaces and sub-graphs remain at zero power draw
  until an incoming event exceeds a symbolic activation threshold,
  minimizing CPU/GPU cycles.

## Part 4: Industry Novelty & Comparative Analysis

### 4.1 Comparative Matrix

| Architectural Dimension | Traditional LLM Frameworks | ADAMAI / Zeuss Engine |
| --- | --- | --- |
| Core Brain | Autoregressive Transformer | Thermodynamic Bayesian Core + VSA |
| Execution Method | Probabilistic next-token prediction | Phase collapse (T → 0) to AST ground states |
| Memory Mechanics | Implicitly baked into weights / RAG | Continuous Clifford Hypervectors (O(1)) |
| Hallucination Risk | High (stochastic nature) | Zero (enforced by Łukasiewicz & Sheaf Audit) |
| Determinism | Non-deterministic | Absolute deterministic mathematical proofs |
| Compute Overhead | Astronomical (Massive GPU clusters) | Low (Runs portably on local compute) |

### 4.2 Is This a World-First System?

Yes. While individual sub-components (such as Energy-Based Models, Vector
Symbolic Architectures, or Bayesian Synthesis) exist in academic research and
physics laboratories, uniting all four paradigms into a single, cohesive
operational architecture is entirely novel and represents a world-first
software implementation.

## Part 5: Implementation Roadmap

The theoretical design phase is complete. Transitioning to software delivery
requires three sequential phases:

```
┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
│  Phase 1: Formalization  │───>│  Phase 2: Prototyping    │───>│   Phase 3: Integration   │
│ Tensor contracts, Sheaf  │    │ Benchmark FHRR/GA-HDC,   │    │ API contracts, runtime   │
│ maps & E(S) energy math. │    │ T->0 annealing speed.    │    │ daemons, portable app.   │
└──────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘
```

1. **Mathematical Formalization:** Explicitly defining the continuous t-norm
   mappings into Clifford algebra space, formalizing Sheaf restriction
   matrices, and deriving the exact system energy function E(S).
2. **Substrate Benchmarking:** Prototyping isolated modules to verify
   hypervector binding operations and measuring phase collapse convergence
   rates.
3. **Runtime Integration:** Packaging the engine daemons into a portable
   runtime application capable of running across local Linux/WSL2, Docker,
   and edge environments.
