"""Zeuss - a continuous<->discrete computational substrate.

Zeuss is a research platform for building a unified computing paradigm where
determinism and probability are structurally intertwined rather than glued
together in software. Three frontiers from VISION.md are realised as three tiers:

  Tier 1 (``zeuss.tier1_kernels``)  - low-level compute kernels
      Phase-interference and topological-collapse ops. A portable NumPy
      reference lives here; Triton/CUDA implementations plug in behind the
      same interface when a GPU is available.

  Tier 2 (``zeuss.tier2_substrate``) - the substrate math
      Complex-phase hypervector algebra (FHRR/VSA), entropy-driven
      dimensional collapse, energy-landscape attractor dynamics, and wave
      resonance. This is the heart and runs on CPU today.

  Tier 3 (``zeuss.tier3_logic``)     - logic compiler / symbol grounding
      Compiles predicate graphs and logical axioms into the continuous
      energy terms (fuzzy t-norms) that govern the substrate.

The design goal: information exists as continuous trajectories in a dynamic
space that crystallises into deterministic truth when measured.
"""

from .backend import xp, HAS_JAX, to_numpy

__all__ = ["xp", "HAS_JAX", "to_numpy", "__version__"]
__version__ = "0.1.0"
