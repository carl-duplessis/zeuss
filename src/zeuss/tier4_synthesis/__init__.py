"""Tier 4 - Bayesian AST program synthesis.

Note on numbering: this "Tier 4" follows `docs/ADAMAI_SPEC.md`'s own scheme
("Tier 4: Thermodynamic Bayesian Engine & AST Synthesizer"), a different
numbering from this repo's internal `tier1_kernels`/`tier2_substrate`/
`tier3_logic` packages - don't conflate the two.
"""
from . import dsl, encode, search, synth  # noqa: F401
