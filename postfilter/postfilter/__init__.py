"""Re-rank ranked candidates by a sound (or reasonably sound) exclusion signal.

Extracted from a real-data investigation, not designed in the abstract: a
knowledge-graph completion project (zeuss) found that applying a logical
constraint as a post-hoc re-rank of an existing candidate list consistently
matched or beat applying the same constraint inside an iterative settling
process, across four independent axes tested on real benchmarks (WN18RR,
FB15k-237) - which relation, which dataset, how the constraint was derived,
and what logical shape it took. See that project's docs/ROADMAP.md
("Post-closure" entries) for the full experiments this was extracted from.

This package has no dependency on zeuss, embeddings, or any particular
scoring model - it only re-ranks (name, score) pairs you already have.
"""
from .core import rerank, top1
from .constraints import disjoint_category_exclusions, no_cycle_exclusions

__all__ = [
    "rerank",
    "top1",
    "no_cycle_exclusions",
    "disjoint_category_exclusions",
]
