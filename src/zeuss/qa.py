"""Ask-a-question read-out over the substrate (answer + coherence).

This is the smallest honest way to *use* Zeuss as a reasoning engine. Facts are
grounded into one continuous memory hypervector (Tier 3 -> Tier 2); a question
``(subject, relation, ?)`` is answered by unbinding the role waves back out of
that memory and reading off which object wave the residue resonates with. There
is no lookup table and no ``if`` over the stored facts - the answer *emerges*
from the wave algebra.

Every answer carries a **coherence** in [0, 1]: the amplitude of the recalled
wave. A stored fact rings loud; an unknown query is crosstalk that rings quiet,
so the engine reports ``known == False`` instead of inventing an answer. That
honest "I don't know" is the point - multi-hop chaining that would raise those
quiet answers is a later roadmap milestone.
"""
from __future__ import annotations

from dataclasses import dataclass

from .tier2_substrate.collapse import softmax
from .tier2_substrate.hypervectors import similarity, unbind
from .tier3_logic.ontology import Ontology

# Below this recalled amplitude the residue is noise - the engine is guessing.
COHERENCE_FLOOR = 0.08


@dataclass
class Answer:
    """The engine's response to one question."""

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


def demo_ontology(dim: int = 8192, seed: int = 1) -> Ontology:
    """A tiny canned knowledge base, grounded into the substrate."""
    onto = Ontology(dim=dim, seed=seed)
    onto.add("socrates", "is_a", "human")
    onto.add("plato", "is_a", "human")
    onto.add("human", "is_a", "mortal")
    onto.add("sun", "is_a", "star")
    onto.add("sky", "has_color", "blue")
    onto.add("grass", "has_color", "green")
    return onto


def ask(onto: Ontology, memory, subject: str, relation: str, beta: float = 12.0) -> Answer:
    """Probe ``memory`` for ``(subject, relation, ?)`` and rank candidate objects.

    The object is recovered purely by unbinding the two role waves out of the
    superposed memory; the residue is then scored against every candidate object
    by resonance (cosine of phase difference) and the winner is read off.
    """
    cb = onto.codebook
    rel_v = cb.symbol(f"REL:{relation}")
    subj_v = cb.symbol(f"SUBJ:{subject}")
    # Peel the two role waves back off the memory -> residue that resonates with
    # the stored object (plus crosstalk from the other facts).
    residue = unbind(unbind(memory, rel_v), subj_v)

    candidates = sorted({o for (_, _, o) in onto.triples})
    scores = [similarity(residue, cb.symbol(f"OBJ:{o}")) for o in candidates]
    probs = softmax([beta * s for s in scores])

    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    ranked = [(candidates[i], float(scores[i])) for i in order]
    top = order[0]
    coherence = float(scores[top])
    return Answer(
        answer=candidates[top],
        coherence=coherence,
        confidence=float(probs[top]),
        ranked=ranked,
        known=coherence >= COHERENCE_FLOOR,
    )
