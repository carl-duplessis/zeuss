from postfilter.core import rerank, top1


def test_rerank_no_penalty_keeps_order():
    candidates = [("a", 0.9), ("b", 0.5), ("c", 0.7)]
    assert rerank(candidates, lambda c: 0.0) == [("a", 0.9), ("c", 0.7), ("b", 0.5)]


def test_rerank_penalty_flips_order():
    candidates = [("a", 0.9), ("b", 0.5)]

    def penalty(c):
        return 10.0 if c == "a" else 0.0

    assert top1(candidates, penalty) == "b"


def test_rerank_does_not_mutate_input():
    candidates = [("a", 0.9), ("b", 0.5)]
    original = list(candidates)
    rerank(candidates, lambda c: 0.0)
    assert candidates == original


def test_hard_veto_is_not_literal_exclusion_when_everything_else_is_zero():
    # Documented edge case: exp(-10) times a nonzero score can still beat 0.0.
    # Measured 5/198 real cases during this package's own validation.
    candidates = [("vetoed", 0.01), ("other", 0.0)]

    def penalty(c):
        return 10.0 if c == "vetoed" else 0.0

    assert top1(candidates, penalty) == "vetoed"
