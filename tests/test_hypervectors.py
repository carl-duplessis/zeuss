import numpy as np

from zeuss.tier2_substrate.hypervectors import (
    Codebook,
    bind,
    bundle,
    decode_sequence,
    encode_record,
    encode_sequence,
    permute,
    random_hypervector,
    similarity,
    unbind,
)


def test_self_similarity_is_one():
    v = random_hypervector(2048, np.random.default_rng(0))
    assert abs(similarity(v, v) - 1.0) < 1e-9


def test_random_vectors_are_quasi_orthogonal():
    rng = np.random.default_rng(0)
    a = random_hypervector(8192, rng)
    b = random_hypervector(8192, rng)
    assert abs(similarity(a, b)) < 0.05


def test_bind_unbind_recovers_partner():
    rng = np.random.default_rng(1)
    a = random_hypervector(8192, rng)
    b = random_hypervector(8192, rng)
    recovered = unbind(bind(a, b), a)
    assert similarity(recovered, b) > 0.99


def test_bundle_keeps_members_recoverable():
    rng = np.random.default_rng(2)
    parts = [random_hypervector(8192, rng) for _ in range(3)]
    blend = bundle(parts)
    for p in parts:
        assert similarity(blend, p) > 0.2


def test_permute_is_invertible_and_dissimilar():
    v = random_hypervector(8192, np.random.default_rng(3))
    p = permute(v, 5)
    assert abs(similarity(p, v)) < 0.05
    assert similarity(permute(p, -5), v) > 0.99


def test_encode_record_roundtrip():
    cb = Codebook(dim=8192, seed=4)
    rec = encode_record(cb, [("colour", "red"), ("shape", "square")])
    colour = unbind(rec, cb.symbol("colour"))
    name, sim = cb.cleanup(colour)
    assert name == "red"
    assert sim > 0.2


def _codebook_with_vocab(seed, n=30):
    cb = Codebook(dim=8192, seed=seed)
    vocab = [f"w{i}" for i in range(n)]
    for w in vocab:
        cb.symbol(w)
    return cb, vocab


def test_encode_sequence_roundtrip_recovers_every_position():
    cb, vocab = _codebook_with_vocab(seed=0)
    rng = np.random.default_rng(1)
    seq = [rng.choice(vocab) for _ in range(6)]

    enc = encode_sequence(cb, seq)
    decoded = decode_sequence(cb, enc, len(seq))

    assert [name for name, _sim in decoded] == seq
    assert all(sim > 0.2 for _name, sim in decoded)


def test_encode_sequence_handles_a_repeated_item_at_different_positions():
    """The same item appearing at two positions must decode correctly at
    both - each position has its own quasi-orthogonal role, so a repeat
    isn't a degenerate case the way it would be for a plain unordered bundle."""
    cb, _vocab = _codebook_with_vocab(seed=0)
    seq = ["w3", "w7", "w3", "w9"]

    enc = encode_sequence(cb, seq)
    decoded = decode_sequence(cb, enc, len(seq))

    assert [name for name, _sim in decoded] == seq


def test_encode_sequence_is_order_sensitive():
    """Same items, different order, must produce a genuinely different
    vector - position is encoded, not just membership (unlike a plain bundle,
    which is order-blind by construction)."""
    cb, _vocab = _codebook_with_vocab(seed=0)
    enc_forward = encode_sequence(cb, ["w1", "w2", "w3"])
    enc_reversed = encode_sequence(cb, ["w3", "w2", "w1"])
    assert similarity(enc_forward, enc_reversed) < 0.5


def test_decode_sequence_degrades_at_low_dimension():
    """Decoding is reliable at this project's usual dimension, but genuinely
    degrades - not silently - once dimensionality is too low relative to how
    many items are bundled into one sequence, the same honest
    dimension-vs-bundle-size tradeoff documented for train_codebook."""
    n = 12
    rng = np.random.default_rng(2)
    vocab = [f"w{i}" for i in range(n + 5)]

    cb_high = Codebook(dim=8192, seed=0)
    for w in vocab:
        cb_high.symbol(w)
    seq = [rng.choice(vocab) for _ in range(n)]
    enc_high = encode_sequence(cb_high, seq)
    decoded_high = decode_sequence(cb_high, enc_high, n)
    correct_high = sum(1 for (name, _sim), orig in zip(decoded_high, seq) if name == orig)

    cb_low = Codebook(dim=64, seed=0)
    for w in vocab:
        cb_low.symbol(w)
    enc_low = encode_sequence(cb_low, seq)
    decoded_low = decode_sequence(cb_low, enc_low, n)
    correct_low = sum(1 for (name, _sim), orig in zip(decoded_low, seq) if name == orig)

    assert correct_high == n  # reliable at this project's usual dimension
    assert correct_low < n  # genuinely, measurably worse at dim=64


def test_codebook_items_and_load_round_trip():
    """`Codebook.symbol` mints lazily from one evolving RNG stream, so a
    name's vector depends on *when* it was first requested, not just
    `seed` - reconstructing `Codebook(dim, seed=same)` fresh does NOT
    reproduce the same per-name vectors unless names are requested in the
    exact original order. `items()`/`load()` (added for
    `tier4_synthesis/grammar_bias.py`'s persistence - see docs/ROADMAP.md
    v0.29) exist to restore the actual vectors directly, sidestepping that
    order-dependence entirely."""
    original = Codebook(dim=128, seed=0)
    v_a = original.symbol("a")  # 1st draw from the seed-0 RNG stream
    v_b = original.symbol("b")  # 2nd draw

    fresh = Codebook(dim=128, seed=0)
    # Confirms the order-dependence claim above, not just asserts it: "b"
    # is the *1st* draw here (nothing requested "a" first on this instance),
    # but the *2nd* draw on `original` above - same seed, same eventual
    # name, different vector, because minting is keyed by call position,
    # not by hashing the name itself.
    assert not np.array_equal(fresh.symbol("b"), v_b)

    restored = Codebook(dim=128, seed=99)  # a *different* seed entirely
    restored.load(original.items())
    assert np.array_equal(restored.symbol("a"), v_a)
    assert np.array_equal(restored.symbol("b"), v_b)
    # A genuinely new name (never in the loaded items) still mints fresh,
    # rather than raising or returning something stale.
    v_c = restored.symbol("c")
    assert v_c.shape == (128,)
    assert "c" not in original.items()
