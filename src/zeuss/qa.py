"""Ask-a-question read-out over the substrate (answer + coherence).

Two ways to interrogate a grounded knowledge base:

* :func:`ask` - a single deductive hop. Recover the object of
  ``(subject, relation, ?)`` by unbinding the role waves back out of the one
  memory hypervector and reading which entity the residue resonates with.
* :func:`chain` / :func:`entails` - *multi-hop* deduction. Iterate the very same
  one-hop wave operator (:meth:`Ontology.step`), collapsing the continuous
  residue onto the nearest entity between hops and feeding it back as the next
  subject. Following a relation's transitive closure is therefore a fixed-point
  of one wave operator - deduction as dynamics, not a graph walk or an ``if``
  over the stored facts.

Every result carries a **coherence** in [0, 1] (the amplitude of the recalled
wave). A stored fact rings loud; crosstalk rings quiet, so unknown queries are
flagged as guesses. Across a chain, coherence *compounds* (a product over hops),
so a two-hop deduction is honestly reported as less certain than a one-hop one.
"""
from __future__ import annotations

from dataclasses import dataclass

from .tier2_substrate.collapse import softmax
from .tier2_substrate.hypervectors import similarity
from .tier3_logic.ontology import Ontology

# Below this recalled amplitude the residue is noise - the engine is guessing.
COHERENCE_FLOOR = 0.08


@dataclass
class Answer:
    """The engine's response to one single-hop question."""

    answer: str
    coherence: float          # raw amplitude of the recalled wave (rings-true-ness)
    confidence: float         # softmax share vs. the other candidates, in [0, 1]
    ranked: list[tuple[str, float]]
    known: bool               # False => below COHERENCE_FLOOR => a guess

    def __str__(self) -> str:
        bar = "#" * int(round(self.confidence * 20))
        flag = "" if self.known else "   << low coherence: guessing / not in KB"
        head = (
            f"{self.answer:<10} coherence={self.coherence:+.3f} "
            f"confidence={self.confidence:5.1%} |{bar:<20}|{flag}"
        )
        runners = "  ".join(f"{name}:{score:+.2f}" for name, score in self.ranked[:3])
        return f"{head}\n              candidates: {runners}"


@dataclass
class Chain:
    """A multi-hop deduction: the trajectory the substrate settled through."""

    start: str
    relation: str
    hops: list[tuple[str, float]]   # (entity, per-hop coherence) in order
    cumulative: list[float]         # compounded coherence after each hop

    def reached(self) -> dict[str, float]:
        """Every entity reached, mapped to its compounded coherence."""
        return {name: c for (name, _), c in zip(self.hops, self.cumulative)}

    def __str__(self) -> str:
        if not self.hops:
            return f"{self.start} --{self.relation}--> (nothing resonates)"
        parts = [self.start]
        for (name, coh), cum in zip(self.hops, self.cumulative):
            parts.append(f"--{self.relation}[{coh:+.2f}]--> {name}(cum={cum:.2f})")
        return " ".join(parts)


@dataclass
class Verdict:
    """Answer to a yes/no multi-hop question (does subject reach target?)."""

    holds: bool
    coherence: float           # compounded coherence along the path (0 if not reached)
    hops: int                  # path length in hops (0 if not reached)
    chain: Chain

    def __str__(self) -> str:
        if self.holds:
            return f"YES  (coherence={self.coherence:.3f} over {self.hops} hop(s))  via {self.chain}"
        return "NO   (target does not resonate through this relation)"


def _cleanup(onto: Ontology, residue, beta: float):
    """Collapse a residue wave onto the nearest KB entity (associative read)."""
    candidates = onto.entity_names()
    scores = [similarity(residue, onto.entity(c)) for c in candidates]
    probs = softmax([beta * s for s in scores])
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    ranked = [(candidates[i], float(scores[i])) for i in order]
    top = order[0]
    return candidates[top], ranked, float(scores[top]), float(probs[top])


def ask(onto: Ontology, memory, subject: str, relation: str, beta: float = 12.0) -> Answer:
    """Single hop: probe ``memory`` for ``(subject, relation, ?)``."""
    residue = onto.step(memory, onto.entity(subject), relation)
    name, ranked, coherence, confidence = _cleanup(onto, residue, beta)
    return Answer(
        answer=name,
        coherence=coherence,
        confidence=confidence,
        ranked=ranked,
        known=coherence >= COHERENCE_FLOOR,
    )


def chain(
    onto: Ontology,
    memory,
    subject: str,
    relation: str,
    max_hops: int = 6,
    beta: float = 12.0,
) -> Chain:
    """Iterate the one-hop operator to walk ``relation``'s transitive closure.

    Each hop: take a wave step, collapse the residue onto the nearest entity
    (the discretisation), and re-inject that clean entity as the next subject.
    Stops when the residue stops resonating (coherence < floor), on a cycle, or
    at ``max_hops``.
    """
    ent = onto.entity(subject)
    visited = {subject}
    hops: list[tuple[str, float]] = []
    cumulative: list[float] = []
    running = 1.0
    for _ in range(max_hops):
        residue = onto.step(memory, ent, relation)
        name, _ranked, coherence, _conf = _cleanup(onto, residue, beta)
        if coherence < COHERENCE_FLOOR or name in visited:
            break
        running *= coherence
        hops.append((name, coherence))
        cumulative.append(running)
        visited.add(name)
        ent = onto.entity(name)  # collapse -> re-enter the continuum clean
    return Chain(start=subject, relation=relation, hops=hops, cumulative=cumulative)


def entails(onto: Ontology, memory, subject: str, relation: str, target: str) -> Verdict:
    """Yes/no: does ``subject`` reach ``target`` through ``relation`` (any hops)?"""
    c = chain(onto, memory, subject, relation)
    reached = c.reached()
    if target in reached:
        hops = next(i for i, (name, _) in enumerate(c.hops, start=1) if name == target)
        return Verdict(holds=True, coherence=reached[target], hops=hops, chain=c)
    return Verdict(holds=False, coherence=0.0, hops=0, chain=c)


def demo_ontology(dim: int = 8192, seed: int = 1) -> Ontology:
    """A tiny canned knowledge base, grounded into the substrate."""
    onto = Ontology(dim=dim, seed=seed)
    onto.add("socrates", "is_a", "human")
    onto.add("plato", "is_a", "human")
    onto.add("human", "is_a", "mortal")
    onto.add("mortal", "is_a", "thing")
    onto.add("sun", "is_a", "star")
    onto.add("sky", "has_color", "blue")
    onto.add("grass", "has_color", "green")
    return onto
