"""Cellular sheaf cohomology over a small consistency graph.

A **graph-level (1-skeleton) sheaf**: vertices are named scalar stalks
(``R``), edges assert that two vertices' *restricted images* must agree,
where a restriction is a scalar (a 1x1 restriction map) - usually ``1.0``
(plain equality) but pluggable for sign/scale differences. Since there are no
2-cells, the cochain complex is ``0 -> C^0 --delta--> C^1 -> 0``, so
``H^0 = ker(delta)`` and ``H^1 = coker(delta) = C^1 / im(delta)``, computed via
``np.linalg.matrix_rank`` (rank-nullity) on the coboundary matrix - real,
falsifiable linear algebra, not a heuristic.

**Honest scoping note - what H^0 and H^1 actually detect here.** For this
*homogeneous* construction (every edge equation is "restrict_u * x_u =
restrict_v * x_v", never "= some fixed nonzero constant"), a frustrated cycle
(restriction maps compose to -1 around the loop) is *full rank* - it
contributes 0 to H^1, not a positive dimension. What it does instead is
*shrink H^0*: a frustrated cycle forces every vertex on it to exactly 0,
i.e. the only solution consistent with every edge is the trivial one.
``H^1`` measures something different: the graph's *cyclic redundancy* (how
many independent cycles compose to the identity, i.e. add no new constraint
beyond what a spanning tree already implies) - useful as a structural
diagnostic, but not a contradiction detector by itself.

So this module gives you two distinct tools, not one:

* ``h0_dimension() == 0`` - a **structural** question, ignoring any specific
  data: is this restriction topology so over-constrained that *only* the
  trivial (all-zero) assignment can satisfy every edge?
* ``local_section(valuation)`` - a **data** question: does *this specific*
  set of local values satisfy every edge? Its nonzero entries are the
  literally violated edges. This is the right tool for "do two agents'
  independently-derived conclusions actually agree" - see :func:`from_theories`.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .compiler import Theory

# Sentinel always available (fixed at 1.0) when searching an agent's best
# local valuation, so a Theory can reference it to encode a directional bias
# (e.g. ``Rule("TRUE", "x", weight=10.0)`` forces "x" toward 1; a rule with
# "x" as the antecedent and the (default-0) name "ZERO" as consequent forces
# "x" toward 0). Without some fixed reference point, plain implications
# between only-searched variables are always satisfiable by tying every
# variable to the same value, so no rule set could ever force a *specific*
# preferred value - see module docstring's honesty note in from_theories.
TRUE_ANCHOR = "TRUE"


@dataclass
class SheafGraph:
    """A cellular sheaf over a small consistency graph."""

    vertices: list[str] = field(default_factory=list)
    edges: list[tuple[str, str, float, float]] = field(default_factory=list)
    local_values: dict[str, float] = field(default_factory=dict)

    def add_vertex(self, name: str) -> "SheafGraph":
        if name not in self.vertices:
            self.vertices.append(name)
        return self

    def add_edge(self, u: str, v: str, restrict_u: float = 1.0, restrict_v: float = 1.0) -> "SheafGraph":
        self.add_vertex(u)
        self.add_vertex(v)
        self.edges.append((u, v, restrict_u, restrict_v))
        return self

    def coboundary_matrix(self) -> np.ndarray:
        """delta: |E| x |V|. Row for edge (u, v): +restrict_u at column u,
        -restrict_v at column v, 0 elsewhere - a weighted, signed incidence
        matrix."""
        index = {name: i for i, name in enumerate(self.vertices)}
        d = np.zeros((len(self.edges), len(self.vertices)))
        for row, (u, v, ru, rv) in enumerate(self.edges):
            d[row, index[u]] += ru
            d[row, index[v]] -= rv
        return d

    def h0_dimension(self) -> int:
        """dim ker(delta): degrees of freedom left in a globally consistent
        (but not necessarily specific/observed) assignment."""
        d = self.coboundary_matrix()
        if d.shape[1] == 0:
            return 0
        return d.shape[1] - int(np.linalg.matrix_rank(d))

    def h1_dimension(self) -> int:
        """dim coker(delta) = |E| - rank(delta): cyclic redundancy of the
        restriction structure - see module docstring's honesty note."""
        d = self.coboundary_matrix()
        if d.shape[0] == 0:
            return 0
        return d.shape[0] - int(np.linalg.matrix_rank(d))

    def has_only_trivial_section(self) -> bool:
        """Structural check: is the topology alone (ignoring any specific
        data) so over-constrained that only the all-zero assignment can
        satisfy every edge? Use :meth:`is_consistent_with` to check whether
        *specific* observed data agrees instead."""
        return self.h0_dimension() == 0

    def local_section(self, valuation: dict[str, float]) -> np.ndarray:
        """``delta @ x`` for a candidate assignment; nonzero entries are the
        specific violated edges."""
        d = self.coboundary_matrix()
        x = np.array([valuation.get(v, 0.0) for v in self.vertices])
        return d @ x

    def is_consistent_with(self, valuation: dict[str, float] | None = None, tol: float = 1e-9) -> bool:
        """Whether specific local data (default: ``self.local_values``)
        satisfies every restriction edge exactly."""
        v = valuation if valuation is not None else self.local_values
        return bool(np.all(np.abs(self.local_section(v)) <= tol))


def _best_local_valuation(theory: Theory, variables: list[str]) -> dict[str, float]:
    """The theory's lowest-energy Boolean corner over ``variables`` (plus the
    fixed ``TRUE_ANCHOR`` sentinel), found by exhaustive enumeration - the
    same ``O(2**n)`` pattern as :func:`zeuss.tier3_logic.grounding.compile_theory`,
    scoped to a single winning corner instead of a whole energy landscape.
    """
    best_corner: dict[str, float] = {}
    best_energy = float("inf")
    for bits in itertools.product((0.0, 1.0), repeat=len(variables)):
        corner = dict(zip(variables, bits))
        corner[TRUE_ANCHOR] = 1.0
        energy = theory.energy(corner)
        if energy < best_energy:
            best_energy, best_corner = energy, corner
    return best_corner


def from_theories(theories: dict[str, Theory], shared_vars: dict[str, list[str]]) -> SheafGraph:
    """Audit whether several agents' ``Theory`` objects agree on shared variables.

    Each agent's own best (lowest-energy) local valuation is found in
    isolation - agents never see each other's rules. Vertices are named
    ``"<agent>:<variable>"``; an equality edge (restriction 1.0) connects the
    same variable across every pair of agents that both list it in
    ``shared_vars``. The returned graph's ``local_values`` holds each agent's
    independently-derived values, so ``graph.is_consistent_with()`` directly
    answers "do these specific conclusions actually agree" - catching a
    disagreement that's invisible to any single agent's own
    ``Theory.satisfied()``, since each agent only ever checks its own rules
    in isolation.
    """
    graph = SheafGraph()
    local_valuations: dict[str, dict[str, float]] = {}
    for agent, theory in theories.items():
        variables = shared_vars.get(agent, [])
        local_valuations[agent] = _best_local_valuation(theory, variables)
        for var in variables:
            graph.add_vertex(f"{agent}:{var}")

    agents = list(theories)
    for i, agent_a in enumerate(agents):
        for agent_b in agents[i + 1:]:
            shared = set(shared_vars.get(agent_a, [])) & set(shared_vars.get(agent_b, []))
            for var in shared:
                graph.add_edge(f"{agent_a}:{var}", f"{agent_b}:{var}", 1.0, 1.0)

    for agent, variables in shared_vars.items():
        local = local_valuations.get(agent, {})
        for var in variables:
            graph.local_values[f"{agent}:{var}"] = local.get(var, 0.0)

    return graph
