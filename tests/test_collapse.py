import numpy as np

from zeuss.tier2_substrate.collapse import anneal, collapse, entropy, softmax
from zeuss.tier2_substrate.hypervectors import Codebook, encode_record, unbind


def test_entropy_bounds():
    assert abs(entropy(np.array([0.5, 0.5])) - 1.0) < 1e-9
    assert entropy(np.array([1.0, 0.0])) == 0.0


def test_softmax_normalises():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(p.sum() - 1.0) < 1e-9


def test_cooling_reduces_entropy_and_locks_winner():
    cb = Codebook(dim=8192, seed=1)
    # populate several distractor symbols
    for name in ("red", "blue", "green", "square", "circle"):
        cb.symbol(name)
    rec = encode_record(cb, [("colour", "red")])
    probe = unbind(rec, cb.symbol("colour"))
    trace = anneal(cb, probe, schedule=(0.5, 2, 8, 32))
    entropies = [row["entropy_bits"] for row in trace]
    assert entropies[0] > entropies[-1]
    _, info = collapse(cb, probe, inverse_temperature=32)
    assert info["winner"] == "red"
