"""Matplotlib experiment: entropy vs. beta (Frontier 1) and energy vs. step
(Frontier 2) - a good-first-task from ``CLAUDE.md``, plotting the two
cooling/descent trajectories the text-only demo (``demo_collapse.py``) only
narrates as printed numbers.

Requires the optional ``viz`` extra (``pip install 'zeuss[viz]'``, or plain
``pip install matplotlib``); guarded like every other optional dependency in
this project, not a hard core import.
"""
from __future__ import annotations

import numpy as np

from zeuss.tier2_substrate.collapse import anneal
from zeuss.tier2_substrate.energy import Landscape, settle
from zeuss.tier2_substrate.hypervectors import Codebook, encode_record, unbind


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("This experiment needs matplotlib - install with `pip install 'zeuss[viz]'`.")
        return 1

    rng = np.random.default_rng(7)
    cb = Codebook(dim=8192, seed=1)
    record = encode_record(cb, [("colour", "red"), ("shape", "square")])
    probe = unbind(record, cb.symbol("colour"))  # a noisy, high-entropy estimate of 'red'

    # Frontier 1: entropy vs. beta across a fine cooling schedule.
    schedule = np.geomspace(0.1, 64, num=40)
    trace = anneal(cb, probe, schedule=schedule)
    betas = [row["beta"] for row in trace]
    entropies = [row["entropy_bits"] for row in trace]

    # Frontier 2: energy vs. step, deterministic vs. thermal descent (same
    # probe, same landscape, so the two plots tell one consistent story).
    land = Landscape().add(cb.symbol("red"), 1.0).add(cb.symbol("blue"), 1.0)
    _, energies_cold = settle(land, probe, steps=40, temperature=0.0)
    _, energies_hot = settle(land, probe, steps=40, temperature=0.4, rng=rng)

    fig, (ax_entropy, ax_energy) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_entropy.plot(betas, entropies, marker="o", markersize=3)
    ax_entropy.set_xscale("log")
    ax_entropy.set_xlabel("inverse temperature (beta)")
    ax_entropy.set_ylabel("entropy (bits)")
    ax_entropy.set_title("Frontier 1: entropy collapses as beta grows")
    ax_entropy.axhline(0, color="grey", linewidth=0.5)

    ax_energy.plot(energies_cold, label="deterministic (T=0)")
    ax_energy.plot(energies_hot, label="thermal (T=0.4)")
    ax_energy.set_xlabel("settle step")
    ax_energy.set_ylabel("energy")
    ax_energy.set_title("Frontier 2: energy descent, cold vs. thermal")
    ax_energy.legend()

    fig.suptitle("Zeuss: entropy vs. beta, energy vs. step (same probe)")
    fig.tight_layout()

    out_path = "demos/entropy_vs_beta_energy_vs_step.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")
    print(f"entropy: {entropies[0]:.3f} bits at beta={betas[0]:.2f} -> "
          f"{entropies[-1]:.3f} bits at beta={betas[-1]:.2f}")
    print(f"energy (cold): {energies_cold[0]:+.3f} -> {energies_cold[-1]:+.3f}")
    print(f"energy (hot):  {energies_hot[0]:+.3f} -> {energies_hot[-1]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
