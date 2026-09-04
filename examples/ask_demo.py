"""Ask the substrate a question and get an answer *plus a coherence score*.

The reasoning lives in :mod:`zeuss.qa`; this script is just the narrative. We
ground a handful of facts into one continuous memory hypervector and then ask
questions - some stored, some deliberately not. Watch the coherence: stored
facts ring loud (high coherence, confident), while unknown or multi-hop queries
ring quiet and are flagged as guesses instead of being answered confidently.

Runs on NumPy or JAX (``ZEUSS_BACKEND=jax``).
"""
from __future__ import annotations

from zeuss.qa import ask, demo_ontology
from zeuss.tier2_substrate.energy import Landscape
from zeuss.tier2_substrate.spiking import SpikingGate, group_seeds


def main() -> None:
    onto = demo_ontology()
    memory = onto.ground()

    print("Knowledge base (grounded into one continuous memory hypervector):")
    for s, r, o in onto.triples:
        print(f"  {s} --{r}--> {o}")

    questions = [
        ("socrates", "is_a", "known fact"),
        ("plato", "is_a", "known fact"),
        ("sky", "has_color", "known fact"),
        ("sun", "is_a", "known fact"),
        ("socrates", "has_color", "wrong relation for this subject"),
        ("dragon", "is_a", "unknown subject -> should be flagged"),
        ("socrates", "is_mortal_via", "needs multi-hop chaining (not stored)"),
    ]

    print("\nAsking questions (answer + coherence; low coherence = honest 'I'm guessing'):\n")
    for subject, relation, note in questions:
        answer = ask(onto, memory, subject, relation)
        print(f"Q: {subject} {relation} ?   ({note})")
        print(f"A: {answer}\n")

    print("== Event-spiking: which relation-groups activate for a query? ==")
    by_relation: dict[str, set] = {}
    for s, r, o in onto.triples:
        by_relation.setdefault(r, set()).update((s, o))
    land = Landscape()
    group_of = []
    for rel, entities in by_relation.items():
        for name in entities:
            land.add(onto.entity(name), 1.0)
            group_of.append(rel)
    gate = SpikingGate(threshold=0.1, refractory_steps=0)
    seeds = group_seeds(land, group_of)
    residue = onto.step(memory, onto.entity("socrates"), "is_a")
    active = gate.poll(residue, seeds)
    active_attractors = sum(1 for g in group_of if g in active)
    print(f"relation-groups: {list(seeds)}")
    print(f"query 'socrates is_a ?' activates: {active}")
    print(f"{active_attractors}/{len(land.attractors)} attractors stayed in play"
          f" - the rest never had to be scored.")


if __name__ == "__main__":
    main()
