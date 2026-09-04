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

from .hypervectors import Codebook, bundle, normalize


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def entropy(probs: np.ndarray, base: float = 2.0) -> float:
    """Shannon entropy of a probability vector (bits by default)."""
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * (np.log(p) / np.log(base))))


def occupancy(codebook: Codebook, z: np.ndarray, inverse_temperature: float = 1.0) -> np.ndarray:
    """Probability the state occupies each symbol basin (a softmax over sims)."""
    sims = np.array(list(codebook.similarities(z).values()), dtype=np.float64)
    return softmax(inverse_temperature * sims)


def collapse(codebook: Codebook, z: np.ndarray, inverse_temperature: float = 8.0):
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
    k = int(np.argmax(probs))
    info = {
        "names": names,
        "probs": probs,
        "entropy_bits": entropy(probs),
        "winner": names[k],
        "winner_prob": float(probs[k]),
    }
    return normalize(z_soft), info


def anneal(codebook: Codebook, z: np.ndarray, schedule=(0.5, 1, 2, 4, 8, 16, 32)):
    """Run a cooling schedule and record entropy dropping toward a decision.

    Returns a list of per-step info dicts - a trace of continuous ambiguity
    crystallising into a discrete symbol.
    """
    trace = []
    for beta in schedule:
        _, info = collapse(codebook, z, inverse_temperature=beta)
        trace.append({"beta": float(beta), **{k: info[k] for k in ("entropy_bits", "winner", "winner_prob")}})
    return trace
