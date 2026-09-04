"""Zeuss command-line entry point.

    python -m zeuss info                 # environment + backend report
    python -m zeuss demo                 # continuous->discrete collapse demo
    python -m zeuss ask                  # ask-a-question demo (answer + coherence)
    python -m zeuss ask socrates is_a    # ask one question against the demo KB
    python -m zeuss audit                # sheaf cohomology consistency audit
    python -m zeuss drive                # active inference / EFE action selection
    python -m zeuss synth                # genetic-programming AST synthesis demo
"""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .backend import backend_name, HAS_JAX


def _info() -> int:
    from .tier1_kernels import HAS_TRITON
    from .tier3_logic.ontology import HAS_NETWORKX

    print(f"Zeuss {__version__}")
    print(f"  array backend : {backend_name()} (jax={'yes' if HAS_JAX else 'no'})")
    print(f"  triton kernels: {'yes' if HAS_TRITON else 'no (using NumPy reference)'}")
    print(f"  networkx      : {'yes' if HAS_NETWORKX else 'no (using fallback)'}")
    return 0


def _demo() -> int:
    from examples.demo_collapse import main as demo_main  # type: ignore

    demo_main()
    return 0


def _ask(subject: str | None, relation: str | None) -> int:
    from .qa import ask, demo_ontology

    if subject and relation:
        onto = demo_ontology()
        answer = ask(onto, onto.ground(), subject, relation)
        print(f"Q: {subject} {relation} ?")
        print(f"A: {answer}")
        return 0

    # No specific question -> run the full narrated demo.
    try:
        from examples.ask_demo import main as ask_main  # type: ignore

        ask_main()
    except Exception:
        onto = demo_ontology()
        memory = onto.ground()
        for subject_, relation_ in [("socrates", "is_a"), ("dragon", "is_a")]:
            print(f"Q: {subject_} {relation_} ?")
            print(f"A: {ask(onto, memory, subject_, relation_)}")
    return 0


def _audit() -> int:
    from .tier3_logic.compiler import Rule, Theory
    from .tier3_logic.sheaf import SheafGraph, from_theories

    print("== Sheaf cohomology consistency audit ==\n")

    print("Scenario 1: three sensors reporting on a shared variable 'door_open'")
    theories = {
        "sensorA": Theory(rules=[Rule("TRUE", "door_open", weight=10.0)]),  # believes open
        "sensorB": Theory(rules=[Rule("TRUE", "door_open", weight=10.0)]),  # believes open
        "sensorC": Theory(rules=[Rule("door_open", "ZERO", weight=10.0)]),  # believes closed
    }
    shared_vars = {name: ["door_open"] for name in theories}
    graph = from_theories(theories, shared_vars)
    print(f"local readings: {graph.local_values}")
    print(f"globally consistent (do all readings actually agree)? {graph.is_consistent_with()}")
    residual = graph.local_section(graph.local_values)
    for (u, v, _ru, _rv), r in zip(graph.edges, residual):
        flag = "VIOLATED" if abs(r) > 1e-9 else "ok"
        print(f"  edge {u} == {v}: residual={r:+.2f}  [{flag}]")

    print("\nScenario 2: structural diagnostics on toy consistency graphs")
    consistent = (
        SheafGraph().add_edge("a", "b", 1.0, 1.0).add_edge("b", "c", 1.0, 1.0).add_edge("a", "c", 1.0, 1.0)
    )
    frustrated = (
        SheafGraph().add_edge("a", "b", 1.0, 1.0).add_edge("b", "c", 1.0, 1.0).add_edge("a", "c", 1.0, -1.0)
    )
    print(
        f"consistent triangle: H0={consistent.h0_dimension()} H1={consistent.h1_dimension()}"
        f"  (nontrivial section exists: {not consistent.has_only_trivial_section()})"
    )
    print(
        f"frustrated triangle: H0={frustrated.h0_dimension()} H1={frustrated.h1_dimension()}"
        f"  (nontrivial section exists: {not frustrated.has_only_trivial_section()})"
    )
    return 0


def _drive() -> int:
    from .drive import Action, expected_free_energy, select_action
    from .tier2_substrate.energy import Landscape
    from .tier2_substrate.hypervectors import Codebook

    print("== Active inference: Expected Free Energy action selection ==\n")
    cb = Codebook(dim=4096, seed=3)
    goal = cb.symbol("goal")
    other = cb.symbol("other")
    land = Landscape().add(goal, 1.0)
    state = cb.symbol("state")

    advance = Action("advance", goal, is_discovery=False)
    probe = Action("probe_environment", other, is_discovery=True)
    actions = [advance, probe]

    print("Scenario: parameters known - pick the best goal-directed action")
    for a in actions:
        efe = expected_free_energy(land, cb, state, a, entropy_beta=2.0)
        print(f"  {a.name:<18} EFE={efe:+.3f}  discovery={a.is_discovery}")
    picked = select_action(land, cb, state, actions, missing_params=False, entropy_beta=2.0)
    print(f"  -> picked: {picked.name}\n")

    print("Scenario: parameters missing - restrict to low-risk discovery actions")
    picked_missing = select_action(land, cb, state, actions, missing_params=True, entropy_beta=2.0)
    print(f"  -> picked: {picked_missing.name}")
    return 0


def _synth() -> int:
    import numpy as np

    from .tier4_synthesis.dsl import Fuel, evaluate
    from .tier4_synthesis.search import Example
    from .tier4_synthesis.synth import describe, synthesize

    print("== Bayesian AST program synthesis (tier4_synthesis) ==\n")
    print("Target: f(xs) = sum(xs), from 4 I/O examples\n")
    examples = [
        Example({"xs": [1, 2, 3]}, 6),
        Example({"xs": [4, 5]}, 9),
        Example({"xs": [10]}, 10),
        Example({"xs": []}, 0),
    ]
    rng = np.random.default_rng(1)
    best, beta_trace, verified = synthesize(
        ["xs"], examples, list_inputs=("xs",), population_size=300, max_generations=100, max_depth=3, rng=rng
    )
    print(f"population=300, ran {len(beta_trace)} generation(s), selection pressure beta: "
          f"{beta_trace[0]:.2f} -> {beta_trace[-1]:.2f}")
    print(f"winning program: {describe(best)}")
    print(f"verified against training examples: {verified}")
    held_out = [[1, 1, 1, 1], [100, -50], [7]]
    print("held-out check (not used during search):")
    for xs in held_out:
        result = evaluate(best, {"xs": xs}, Fuel(500))
        print(f"  sum({xs}) -> {result}  (expected {sum(xs)})  {'OK' if result == sum(xs) else 'MISMATCH'}")
    print(
        "\n(This is example-based verification within a closed, total DSL - "
        "not a formal proof of correctness for all possible inputs.)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zeuss", description="Zeuss substrate CLI")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("info", help="print environment / backend report")
    sub.add_parser("demo", help="run the collapse demo")
    ask = sub.add_parser("ask", help="ask a question (answer + coherence)")
    ask.add_argument("subject", nargs="?", help="e.g. socrates")
    ask.add_argument("relation", nargs="?", help="e.g. is_a")
    sub.add_parser("audit", help="sheaf cohomology consistency audit")
    sub.add_parser("drive", help="active inference / EFE action selection")
    sub.add_parser("synth", help="genetic-programming AST synthesis demo")
    args = parser.parse_args(argv)

    if args.cmd == "info":
        return _info()
    if args.cmd == "ask":
        return _ask(args.subject, args.relation)
    if args.cmd == "audit":
        return _audit()
    if args.cmd == "drive":
        return _drive()
    if args.cmd == "synth":
        return _synth()
    if args.cmd == "demo":
        try:
            return _demo()
        except Exception:
            # Fall back to a self-contained demo if run outside the repo root.
            from .tier2_substrate.hypervectors import Codebook, encode_record, unbind
            from .tier2_substrate.collapse import anneal

            cb = Codebook(dim=4096, seed=1)
            rec = encode_record(cb, [("colour", "red"), ("shape", "square")])
            probe = unbind(rec, cb.symbol("colour"))
            for row in anneal(cb, probe):
                print(row)
            return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
