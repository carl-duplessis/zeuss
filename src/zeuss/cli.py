"""Zeuss command-line entry point.

    python -m zeuss info                 # environment + backend report
    python -m zeuss demo                 # continuous->discrete collapse demo
    python -m zeuss ask                  # ask-a-question demo (answer + coherence)
    python -m zeuss ask socrates is_a    # ask one question against the demo KB
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zeuss", description="Zeuss substrate CLI")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("info", help="print environment / backend report")
    sub.add_parser("demo", help="run the collapse demo")
    ask = sub.add_parser("ask", help="ask a question (answer + coherence)")
    ask.add_argument("subject", nargs="?", help="e.g. socrates")
    ask.add_argument("relation", nargs="?", help="e.g. is_a")
    args = parser.parse_args(argv)

    if args.cmd == "info":
        return _info()
    if args.cmd == "ask":
        return _ask(args.subject, args.relation)
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
