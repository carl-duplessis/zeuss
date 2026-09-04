"""Tier 2 - the substrate math (the heart of Zeuss).

Continuous-phase hypervector algebra plus the three dynamical primitives that
turn continuous state into discrete truth:

  * hypervectors - FHRR/VSA algebra over unit-modulus complex vectors
  * collapse     - entropy-driven continuous -> discrete crystallisation
  * energy       - axioms as attractor basins; settle to the logical ground state
  * resonance    - deduction as constructive/destructive wave interference
"""
from . import hypervectors, collapse, energy, resonance

__all__ = ["hypervectors", "collapse", "energy", "resonance"]
