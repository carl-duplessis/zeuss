"""The generic re-ranking primitive: score * exp(-penalty), nothing more."""
from __future__ import annotations

import math
from typing import Callable, Sequence, Tuple

Candidate = Tuple[str, float]


def rerank(candidates: Sequence[Candidate], penalty: Callable[[str], float]) -> list[Candidate]:
    """Re-rank ``(name, score)`` pairs by ``score * exp(-penalty(name))``.

    ``penalty`` should return ``0.0`` for a candidate with no constraint
    violation, and a positive value otherwise. A "hard" veto is not truly
    infinite: ``penalty=10.0`` (``exp(-10) ~ 4.5e-5``) is effectively
    excluded except in the edge case where every other candidate scores
    exactly ``0.0``, in which case the vetoed candidate can still surface -
    measured directly during validation (5/198 real cases on one benchmark).
    Filter vetoed candidates out of ``candidates`` entirely before calling
    this if that edge case matters for your application.

    Returns a new list sorted by penalised score, descending. Does not
    mutate ``candidates``.
    """
    return sorted(candidates, key=lambda item: item[1] * math.exp(-penalty(item[0])), reverse=True)


def top1(candidates: Sequence[Candidate], penalty: Callable[[str], float]) -> str:
    """The single best candidate name after re-ranking."""
    return rerank(candidates, penalty)[0][0]
