import numpy as np

from zeuss.tier2_substrate.hypervectors import (
    Codebook,
    bind,
    bundle,
    encode_record,
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
