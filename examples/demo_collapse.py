"""End-to-end demo: continuous ambiguity crystallising into a discrete symbol.

We encode a structured record with role-filler binding, probe it for one role
(which returns a *noisy* superposition - the probabilistic, high-entropy state),
then cool the system and watch entropy fall as the state collapses onto the one
correct discrete symbol. Then we settle the same probe in an energy landscape of
'axiom' attractors to show the thermodynamic route to the same answer.

Runs on CPU with NumPy alone.
"""
from __future__ import annotations

import numpy as np

from zeuss.tier2_substrate.collapse import anneal, collapse
from zeuss.tier2_substrate.energy import Landscape, settle
from zeuss.tier2_substrate.hypervectors import (
    Codebook,
    encode_record,
    similarity,
    unbind,
)


def main() -> None:
    rng = np.random.default_rng(7)
    cb = Codebook(dim=8192, seed=1)

    # Ground truth: a red square.
    record = encode_record(cb, [("colour", "red"), ("shape", "square")])

    # Probe for the colour role -> a noisy estimate of 'red'.
    probe = unbind(record, cb.symbol("colour"))

    print("== Frontier 1: entropy-driven collapse ==")
    print(f"raw similarity(probe, red)  = {similarity(probe, cb.symbol('red')):+.3f}")
    print(f"raw similarity(probe, blue) = {similarity(probe, cb.symbol('blue')):+.3f}")
    print("cooling schedule (entropy should fall, winner should lock to 'red'):")
    for row in anneal(cb, probe, schedule=(0.5, 1, 2, 4, 8, 16, 32)):
        print(
            f"  beta={row['beta']:>5}  entropy={row['entropy_bits']:.3f} bits"
            f"  winner={row['winner']!r} p={row['winner_prob']:.3f}"
        )

    z_star, info = collapse(cb, probe, inverse_temperature=32)
    print(f"collapsed decision: {info['winner']!r}  (p={info['winner_prob']:.3f})\n")

    print("== Frontier 2: settle to the logical ground state ==")
    land = Landscape().add(cb.symbol("red"), 1.0).add(cb.symbol("blue"), 1.0)
    _, e_cold = settle(land, probe, steps=40, temperature=0.0)
    _, e_hot = settle(land, probe, steps=40, temperature=0.4, rng=rng)
    print(f"deterministic (T=0):   energy {e_cold[0]:+.3f} -> {e_cold[-1]:+.3f}")
    print(f"probabilistic (T=0.4): energy {e_hot[0]:+.3f} -> {e_hot[-1]:+.3f}")
    print("(cold descent reaches a lower, crisper ground state.)")


if __name__ == "__main__":
    main()
