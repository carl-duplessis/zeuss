"""Map a predicate graph into hypervector seeds.

Symbolic predicate trees are grounded into the continuous substrate by binding
role and filler hypervectors. Uses NetworkX when available for graph
representation, with a tiny built-in fallback so the module always imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..tier2_substrate.hypervectors import Codebook, bind, bundle

try:
    import networkx as nx

    HAS_NETWORKX = True
except Exception:  # pragma: no cover - optional dep
    nx = None
    HAS_NETWORKX = False


@dataclass
class Ontology:
    """A relational knowledge graph grounded in a hypervector codebook.

    Each triple ``(subject, relation, object)`` is grounded as
    ``bind(REL:relation, bind(SUBJ:subject, OBJ:object))`` and bundled into a
    single memory hypervector that the Tier-2 resonators can probe.
    """

    dim: int = 10000
    seed: int = 0
    codebook: Codebook = field(init=False)
    triples: list[tuple[str, str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.codebook = Codebook(dim=self.dim, seed=self.seed)
        self._graph = nx.MultiDiGraph() if HAS_NETWORKX else None

    def add(self, subject: str, relation: str, obj: str) -> "Ontology":
        self.triples.append((subject, relation, obj))
        if self._graph is not None:
            self._graph.add_edge(subject, obj, key=relation, relation=relation)
        return self

    def _triple_vector(self, subject: str, relation: str, obj: str):
        cb = self.codebook
        inner = bind(cb.symbol(f"SUBJ:{subject}"), cb.symbol(f"OBJ:{obj}"))
        return bind(cb.symbol(f"REL:{relation}"), inner)

    def ground(self):
        """Bundle every triple into one continuous memory hypervector."""
        if not self.triples:
            raise ValueError("ontology is empty; add() some triples first")
        return bundle([self._triple_vector(*t) for t in self.triples])

    @property
    def graph(self):
        if not HAS_NETWORKX:
            raise RuntimeError("networkx is not installed; install zeuss[logic]")
        return self._graph
