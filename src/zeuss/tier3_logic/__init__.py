"""Tier 3 - logic compiler / symbol grounding (user / API layer).

Compiles user-defined logical rules, ontology graphs and probabilistic priors
into the geometric energy terms that govern the substrate:

  * ontology - map a predicate graph into hypervector seeds
  * compiler - turn axioms into continuous fuzzy (Lukasiewicz / Godel / product)
               truth degrees and energy penalties
"""
from . import ontology, compiler

__all__ = ["ontology", "compiler"]
