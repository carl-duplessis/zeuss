import numpy as np
import pytest

from zeuss.backend import HAS_JAX
from zeuss.tier2_substrate.collapse import (
    anneal,
    anneal_adaptive,
    collapse,
    collapse_batch,
    collapse_batch_jit,
    dimensional_collapse,
    entropy,
    participation_ratio,
    softmax,
)
from zeuss.tier2_substrate.hypervectors import Codebook, bundle, encode_record, random_hypervector, unbind


def test_entropy_bounds():
    assert abs(entropy(np.array([0.5, 0.5])) - 1.0) < 1e-9
    assert entropy(np.array([1.0, 0.0])) == 0.0


def test_softmax_normalises():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(p.sum() - 1.0) < 1e-9


def test_softmax_batches_independently_per_row():
    """softmax's axis=-1 generalisation (see its docstring) must normalise
    each row of a batch on its own, not the whole 2-D array together -
    otherwise a batch of N independent distributions would silently become
    one mis-scaled distribution."""
    rows = np.array([[1.0, 2.0, 3.0], [10.0, 0.0, 0.0]])
    batched = softmax(rows)
    assert np.allclose(batched.sum(axis=-1), [1.0, 1.0])
    assert np.allclose(batched[0], softmax(rows[0]))
    assert np.allclose(batched[1], softmax(rows[1]))


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


def test_collapse_batch_matches_per_probe_collapse():
    """collapse_batch resolves N probes in one vectorized pass instead of a
    Python loop calling collapse() once per probe - the batched-collapse
    roadmap item (docs/ROADMAP.md v0.2). Runs on whichever backend is active
    (no JAX required), since the batching itself is plain xp vectorization,
    not a JAX-specific feature - only collapse_batch_jit needs JAX."""
    cb = _codebook_with_distractors(seed=2)
    rng = np.random.default_rng(3)
    probes = [random_hypervector(8192, rng) for _ in range(6)]

    ref_winners, ref_entropy, ref_winner_prob = [], [], []
    for p in probes:
        _, info = collapse(cb, p, inverse_temperature=8.0)
        ref_winners.append(info["winner"])
        ref_entropy.append(info["entropy_bits"])
        ref_winner_prob.append(info["winner_prob"])

    _z_batch, info_batch = collapse_batch(cb, probes, inverse_temperature=8.0)
    assert info_batch["winner"] == ref_winners
    assert np.allclose(np.asarray(info_batch["entropy_bits"]), ref_entropy, atol=1e-6)
    assert np.allclose(np.asarray(info_batch["winner_prob"]), ref_winner_prob, atol=1e-6)


def test_collapse_batch_jit_requires_jax_backend():
    if HAS_JAX:
        pytest.skip("this environment's active backend is already JAX")
    cb = _codebook_with_distractors(seed=0)
    rng = np.random.default_rng(0)
    probes = [random_hypervector(64, rng) for _ in range(3)]
    with pytest.raises(RuntimeError):
        collapse_batch_jit(cb, probes)


def test_collapse_batch_jit_matches_collapse_batch():
    """The jit-compiled path (see collapse_batch_jit's docstring for why it
    needs a separate, pure-array core function) must produce the exact same
    result as the eager batched path - jax.jit is a compilation strategy, not
    a different computation."""
    pytest.importorskip("jax")
    if not HAS_JAX:
        pytest.skip("jax is importable but not the active backend")
    cb = _codebook_with_distractors(seed=2)
    rng = np.random.default_rng(3)
    probes = [random_hypervector(8192, rng) for _ in range(6)]

    z_eager, info_eager = collapse_batch(cb, probes, inverse_temperature=8.0)
    z_jit, info_jit = collapse_batch_jit(cb, probes, inverse_temperature=8.0)

    assert info_jit["winner"] == info_eager["winner"]
    assert np.allclose(np.asarray(info_jit["entropy_bits"]), np.asarray(info_eager["entropy_bits"]), atol=1e-6)
    assert np.allclose(np.asarray(z_jit), np.asarray(z_eager), atol=1e-6)


def test_participation_ratio_bounds():
    """1.0 for a one-hot distribution, K for the uniform distribution over K -
    the two limits dimensional_collapse's basis truncation must hit exactly."""
    k = 5
    assert abs(participation_ratio(np.array([1.0, 0.0, 0.0, 0.0, 0.0])) - 1.0) < 1e-9
    assert abs(participation_ratio(np.full(k, 1.0 / k)) - k) < 1e-9


def test_dimensional_collapse_matches_collapse_at_high_entropy():
    """At low beta (near-uniform occupancy over all 5 codebook symbols),
    participation_ratio ~ 5, so every symbol stays in the live basis and
    dimensional_collapse must reduce to plain collapse() exactly - the
    "expand basis when entropy is high" half of the roadmap item."""
    cb = _codebook_with_distractors(seed=1)
    rng = np.random.default_rng(9)
    probe = random_hypervector(8192, rng)

    z_dim, info_dim = dimensional_collapse(cb, probe, inverse_temperature=0.1)
    z_plain, info_plain = collapse(cb, probe, inverse_temperature=0.1)

    assert info_dim["k_live"] == len(cb.names())
    assert np.allclose(np.asarray(z_dim), np.asarray(z_plain), atol=1e-6)
    assert abs(info_dim["entropy_bits"] - info_plain["entropy_bits"]) < 1e-9


def test_dimensional_collapse_shrinks_to_one_symbol_at_low_entropy():
    """At high beta on an unambiguous probe (exact symbol match), occupancy
    collapses onto one symbol, participation_ratio -> 1, and the returned
    state must be (numerically) just that one symbol - a genuine basis
    contraction to a single point in the discrete Boolean lattice, not merely
    a down-weighted D-dim blend - the "project onto low-D orthogonal subspace
    as entropy -> 0" half of the roadmap item."""
    cb = _codebook_with_distractors(seed=1)
    probe = cb.symbol("red")

    z_dim, info_dim = dimensional_collapse(cb, probe, inverse_temperature=32.0)

    assert info_dim["k_live"] == 1
    assert info_dim["live_names"] == ["red"]
    assert info_dim["winner"] == "red"
    from zeuss.tier2_substrate.hypervectors import similarity

    assert similarity(z_dim, cb.symbol("red")) > 0.999


def test_dimensional_collapse_k_live_shrinks_along_cooling_schedule():
    """As beta rises across an anneal-style schedule on an ambiguous probe,
    the live basis size must shrink (not grow) - the space's dimensionality
    contracting as entropy drops, matching entropy's own monotonic drop
    (test_cooling_reduces_entropy_and_locks_winner) but measured directly in
    basis size rather than bits."""
    cb = _codebook_with_distractors(seed=1)
    rec = encode_record(cb, [("colour", "red")])
    probe = unbind(rec, cb.symbol("colour"))

    k_live_trace = []
    for beta in (0.1, 1, 4, 16, 64):
        _, info = dimensional_collapse(cb, probe, inverse_temperature=beta)
        k_live_trace.append(info["k_live"])

    assert k_live_trace[0] >= k_live_trace[-1]
    assert k_live_trace[-1] == 1
