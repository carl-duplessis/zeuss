import numpy as np

from zeuss.tier2_substrate.collapse import anneal, anneal_adaptive, collapse, entropy, softmax
from zeuss.tier2_substrate.hypervectors import Codebook, bundle, encode_record, unbind


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


def _codebook_with_distractors(seed):
    cb = Codebook(dim=8192, seed=seed)
    for name in ("red", "blue", "green", "square", "circle"):
        cb.symbol(name)
    return cb


def test_anneal_adaptive_converges_faster_on_an_unambiguous_probe():
    cb = _codebook_with_distractors(seed=1)
    probe = cb.symbol("red")  # exact match - no ambiguity at all
    trace_fixed = anneal(cb, probe, schedule=(0.5, 1, 2, 4, 8, 16, 32))
    trace_adapt = anneal_adaptive(cb, probe)
    assert trace_adapt[-1]["winner"] == "red"
    assert trace_adapt[-1]["entropy_bits"] < 0.05
    assert len(trace_adapt) < len(trace_fixed)


def test_anneal_adaptive_grows_beta_slower_for_an_ambiguous_probe():
    cb = _codebook_with_distractors(seed=1)
    probe_clear = cb.symbol("red")
    probe_ambiguous = bundle([cb.symbol("red"), cb.symbol("blue")])  # a genuine 50/50 tie

    trace_clear = anneal_adaptive(cb, probe_clear, max_steps=10)
    trace_ambiguous = anneal_adaptive(cb, probe_ambiguous, max_steps=10)

    # At the same step index, the unambiguous probe's beta should have grown
    # much further than the ambiguous (near-tie) probe's beta.
    idx = min(len(trace_clear), len(trace_ambiguous)) - 1
    assert trace_clear[idx]["beta"] > trace_ambiguous[idx]["beta"]
