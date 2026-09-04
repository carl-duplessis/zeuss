"""Map a predicate graph into hypervector seeds.

Symbolic triples are grounded into the continuous substrate by binding a shared
entity core wave ``ENT:x`` under two distinct role waves, ``ROLE:subj`` and
``ROLE:obj``, *plus* a fixed cyclic permutation ``P`` on the object slot:

    subj_wave(x) = bind(ROLE:subj, ENT:x)
    obj_wave(x)  = P(bind(ROLE:obj, ENT:x))
    triple       = bind(REL:r, bind(subj_wave(s), obj_wave(o)))
    memory       = bundle(all triples)

The role waves alone are *not* enough to protect direction once you chain: bind
is elementwise complex multiplication, which is commutative, so when an entity
``x`` is the object of one fact and the subject of the next (exactly what a
transitive chain ``a -> x -> b`` requires), unbinding by ``bind(ROLE:subj, x)``
lets the ``ROLE:subj``/``x`` factors cancel out of *both* the fact where ``x``
is the subject (giving the true successor) *and* the fact where ``x`` is the
object (an exact algebraic collision, not noise - the wrong neighbour ties the
right one exactly). The permutation ``P`` breaks this: it's applied *after* the
role bind, on the object side only, so a probe built purely from role+entity
waves no longer factors cleanly out of a triple where the queried entity sat in
the object slot - that mismatch decays to ordinary quasi-orthogonal noise
instead of an exact tie. Recovering the object needs the extra step of
inverting the permutation before the final ``ROLE:obj`` unbind (see
:meth:`Ontology.step`). An object recovered from one hop can be fed straight
back in as the next subject, so deduction *chains* by iterating one wave
operator, entirely in wave space - no graph walk, no ``if`` over stored facts
(see :mod:`zeuss.qa`).

Uses NetworkX when available for a parallel symbolic graph view, with a tiny
built-in fallback so the module always imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..tier2_substrate.hypervectors import Codebook, bind, bundle, permute, unbind

# Fixed cyclic shift applied to the object slot - see module docstring for why
# this is load-bearing (breaks a commutative-bind collision on chained entities).
OBJ_SHIFT = 7

try:
    import networkx as nx

    HAS_NETWORKX = True
except Exception:  # pragma: no cover - optional dep
    nx = None
    HAS_NETWORKX = False


@dataclass
class Ontology:
    """A relational knowledge graph grounded in a hypervector codebook."""

    dim: int = 10000
    seed: int = 0
    codebook: Codebook = field(init=False)
    triples: list[tuple[str, str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.codebook = Codebook(dim=self.dim, seed=self.seed)
        self._graph = nx.MultiDiGraph() if HAS_NETWORKX else None
        self._entities: set[str] = set()
        self._relations: set[str] = set()

    # -- construction -------------------------------------------------------
    def add(self, subject: str, relation: str, obj: str) -> "Ontology":
        self.triples.append((subject, relation, obj))
        self._entities.update((subject, obj))
        self._relations.add(relation)
        if self._graph is not None:
            self._graph.add_edge(subject, obj, key=relation, relation=relation)
        return self

    # -- substrate accessors ------------------------------------------------
    def entity(self, name: str):
        """The shared core wave ``ENT:name`` for an entity (minted lazily)."""
        return self.codebook.symbol(f"ENT:{name}")

    def relation_wave(self, name: str):
        return self.codebook.symbol(f"REL:{name}")

    @property
    def _role_subj(self):
        return self.codebook.symbol("ROLE:subj")

    @property
    def _role_obj(self):
        return self.codebook.symbol("ROLE:obj")

    def subj_wave(self, name: str):
        return bind(self._role_subj, self.entity(name))

    def obj_wave(self, name: str):
        return permute(bind(self._role_obj, self.entity(name)), shift=OBJ_SHIFT)

    def entity_names(self) -> list[str]:
        """Entities that actually appear in the KB (the answer candidates)."""
        return sorted(self._entities)

    # -- grounding ----------------------------------------------------------
    def _triple_vector(self, subject: str, relation: str, obj: str):
        pair = bind(self.subj_wave(subject), self.obj_wave(obj))
        return bind(self.relation_wave(relation), pair)

    def ground(self):
        """Bundle every triple into one continuous memory hypervector."""
        if not self.triples:
            raise ValueError("ontology is empty; add() some triples first")
        return bundle([self._triple_vector(*t) for t in self.triples])

    # -- the one-hop substrate operator ------------------------------------
    def step(self, memory, ent_wave, relation: str):
        """One deductive hop, entirely in wave space.

        Given a memory hypervector, an entity wave standing in the *subject*
        slot, and a relation, peel the role waves back off ``memory`` and return
        the residue that resonates with the *object* entity of the matching
        fact. Composed with a cleanup, iterating ``step`` walks a relation's
        transitive closure - deduction as a fixed-point of one wave operator.

        The object slot was permuted at grounding time (see module docstring),
        so recovery must invert that permutation before the final
        ``ROLE:obj`` unbind.
        """
        roled = unbind(memory, self.relation_wave(relation))
        roled_obj = unbind(roled, bind(self._role_subj, ent_wave))
        unpermuted = permute(roled_obj, shift=-OBJ_SHIFT)
        return unbind(unpermuted, self._role_obj)

    @property
    def graph(self):
        if not HAS_NETWORKX:
            raise RuntimeError("networkx is not installed; install zeuss[logic]")
        return self._graph
