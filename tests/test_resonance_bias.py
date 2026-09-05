"""ResonantBias: an EoD prior over template hole-fillers, read via resonance."""
import numpy as np

from zeuss.tier2_substrate.hypervectors import Codebook
from zeuss.tier4_synthesis.resonance_bias import ResonantBias


def test_uniform_before_any_evidence():
    rng = np.random.default_rng(0)
    cb = Codebook(dim=1024, seed=0)
    bias = ResonantBias(cb, {"op": ["+", "-", "*"]})
    counts = {}
    for _ in range(3000):
        v = bias.sample("op", rng)
        counts[v] = counts.get(v, 0) + 1
    for v in ("+", "-", "*"):
        assert 800 < counts.get(v, 0) < 1200  # roughly uniform (~1000 each)


def test_reinforcement_gradually_biases_sampling_without_instant_lock_in():
    rng = np.random.default_rng(0)
    cb = Codebook(dim=1024, seed=0)
    bias = ResonantBias(cb, {"op": ["+", "-", "*"]})

    def frac_star():
        counts = {"star": 0, "total": 2000}
        for _ in range(counts["total"]):
            if bias.sample("op", rng) == "*":
                counts["star"] += 1
        return counts["star"] / counts["total"]

    # A single reinforcement should nudge the distribution, not collapse it -
    # exploration matters early in a search, so this must not be near-1.0
    # after just one observation (an earlier design with beta=4.0 hit ~96%
    # after exactly one reinforcement, which would kill exploration).
    bias.reinforce({"op": "*"}, weight=1.0)
    frac_after_one = frac_star()
    assert frac_after_one < 0.7

    # Sustained reinforcement should still clearly win out eventually.
    for _ in range(50):
        bias.reinforce({"op": "*"}, weight=1.0)
    frac_after_many = frac_star()
    assert frac_after_many > frac_after_one
    assert frac_after_many > 0.8


def test_reinforce_ignores_unknown_categories_and_nonpositive_weight():
    cb = Codebook(dim=512, seed=0)
    bias = ResonantBias(cb, {"op": ["+", "-"]})
    bias.reinforce({"unknown_category": "x", "op": "+"}, weight=1.0)
    bias.reinforce({"op": "-"}, weight=0.0)  # zero weight: no-op
    bias.reinforce({"op": "-"}, weight=-1.0)  # negative weight: no-op
    assert "unknown_category" not in bias._accum
    # Only the weight=1.0 "+" reinforcement should have taken effect.
    rng = np.random.default_rng(0)
    counts = {"+": 0, "-": 0}
    for _ in range(1000):
        counts[bias.sample("op", rng)] += 1
    assert counts["+"] > counts["-"]
