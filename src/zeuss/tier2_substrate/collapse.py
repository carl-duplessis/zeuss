"""Frontier 1 - entropy-driven dimensional collapse.

The logic is not a rule being evaluated; the logic *is* the structural collapse
of the space. We model a state ``z`` by how it distributes across a codebook of
atomic symbols. High entropy = ambiguous, continuous, high-dimensional. As
evidence accumulates the distribution sharpens (entropy -> 0) and the state
crystallises onto a single orthogonal symbol - a Boolean/discrete outcome.

Temperature is the knob: ``inverse_temperature`` (beta) high -> crisp/discrete,
low -> smeared/probabilistic. This is one continuum, not two modes.
"""
from __future__ import annotations

import numpy as np

from ..backend import RDTYPE, xp
from .hypervectors import Codebook, bundle, normalize


def softmax(x) -> "xp.ndarray":
    x = xp.asarray(x, dtype=RDTYPE)
    x = x - xp.max(x)
    e = xp.exp(x)
    return e / xp.sum(e)


def entropy(probs, base: float = 2.0) -> float:
    """Shannon entropy of a probability vector (bits by default)."""
    p = xp.asarray(probs, dtype=RDTYPE)
    # Mask zeros with xp.where (rather than boolean indexing) so the reduction
    # stays a fixed shape and is safe under jit/vmap: 0 * log(1) == 0.
    safe = xp.where(p > 0, p, 1.0)
    return float(-xp.sum(p * (xp.log(safe) / np.log(base))))


def occupancy(codebook: Codebook, z, inverse_temperature: float = 1.0) -> "xp.ndarray":
    """Probability the state occupies each symbol basin (a softmax over sims)."""
    sims = xp.asarray(list(codebook.similarities(z).values()), dtype=RDTYPE)
    return softmax(inverse_temperature * sims)


def collapse(codebook: Codebook, z, inverse_temperature: float = 8.0):
    """Collapse ``z`` toward the codebook under an inverse temperature.

    Returns ``(z_collapsed, info)`` where ``info`` carries the occupancy
    distribution, its entropy, and the winning symbol. As
    ``inverse_temperature`` grows, ``z_collapsed`` approaches a single symbol
    (the deterministic limit); as it shrinks, it stays a soft superposition.
    """
    names = codebook.names()
    probs = occupancy(codebook, z, inverse_temperature)
    mats = codebook.matrix()
    z_soft = bundle(list(mats), weights=list(probs))
    k = int(xp.argmax(probs))
    info = {
        "names": names,
        "probs": probs,
        "entropy_bits": entropy(probs),
        "winner": names[k],
        "winner_prob": float(probs[k]),
    }
    return normalize(z_soft), info


def anneal(codebook: Codebook, z, schedule=(0.5, 1, 2, 4, 8, 16, 32)):
    """Run a cooling schedule and record entropy dropping toward a decision.

    Returns a list of per-step info dicts - a trace of continuous ambiguity
    crystallising into a discrete symbol.
    """
    trace = []
    for beta in schedule:
        _, info = collapse(codebook, z, inverse_temperature=beta)
        trace.append({"beta": float(beta), **{k: info[k] for k in ("entropy_bits", "winner", "winner_prob")}})
    return trace


def anneal_adaptive(
    codebook: Codebook,
    z,
    beta_start: float = 0.5,
    beta_max: float = 64.0,
    entropy_tol: float = 0.05,
    growth_range: tuple[float, float] = (1.05, 3.0),
    margin_scale: float = 0.5,
    max_steps: int = 40,
):
    """Cooling schedule whose beta growth rate adapts to the probe's ambiguity.

    "Liquid time-step" applied to Frontier 1. ``z`` is fixed throughout a
    schedule - only beta changes - so the probe's intrinsic hardness (the raw
    similarity gap between its best and second-best codebook symbol) is a
    single beta-independent number, computed once up front: a small gap (two
    close, competing symbols - a genuinely hard decision) grows beta slowly,
    spending more of the schedule resolving the ambiguity; a wide gap
    (unambiguous probe) grows beta quickly, reaching the discrete decision in
    fewer steps than a fixed static schedule needs. Stops once entropy drops
    below ``entropy_tol`` or ``beta_max``/``max_steps`` is hit.
    """
    sims = sorted(codebook.similarities(z).values(), reverse=True)
    margin = (sims[0] - sims[1]) if len(sims) > 1 else 1.0
    lo, hi = growth_range
    growth = lo + (hi - lo) * float(np.clip(margin / margin_scale, 0.0, 1.0))

    beta = beta_start
    trace = []
    for _ in range(max_steps):
        _, info = collapse(codebook, z, inverse_temperature=beta)
        trace.append({"beta": float(beta), **{k: info[k] for k in ("entropy_bits", "winner", "winner_prob")}})
        if info["entropy_bits"] <= entropy_tol or beta >= beta_max:
            break
        beta = min(beta * growth, beta_max)
    return trace
