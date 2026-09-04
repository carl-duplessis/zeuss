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
