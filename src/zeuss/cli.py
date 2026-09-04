"""Zeuss command-line entry point.

    python -m zeuss info      # environment + backend report
    python -m zeuss demo      # run the continuous->discrete collapse demo
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zeuss", description="Zeuss substrate CLI")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("info", help="print environment / backend report")
    sub.add_parser("demo", help="run the collapse demo")
    args = parser.parse_args(argv)

    if args.cmd == "info":
        return _info()
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
