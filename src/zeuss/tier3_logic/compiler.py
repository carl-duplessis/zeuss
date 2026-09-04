"""Compile symbolic axioms into continuous fuzzy-logic energy terms.

Truth values live in [0, 1]. Connectives are t-norms (AND) and t-conorms (OR);
implication uses the residuum of the chosen t-norm. A rule's *dissatisfaction*
(1 - truth) becomes an energy penalty the substrate minimises - this is how a
symbolic axiom turns into a basin in the Tier-2 energy landscape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def clamp(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


# --- t-norms (fuzzy AND) ---------------------------------------------------
def lukasiewicz_tnorm(a: float, b: float) -> float:
    return clamp(a + b - 1.0)


def godel_tnorm(a: float, b: float) -> float:
    return min(a, b)


def product_tnorm(a: float, b: float) -> float:
    return a * b


# --- residuated implications (a -> b) --------------------------------------
def lukasiewicz_implies(a: float, b: float) -> float:
    return clamp(1.0 - a + b)


def godel_implies(a: float, b: float) -> float:
    return 1.0 if a <= b else b


def product_implies(a: float, b: float) -> float:
    return 1.0 if a <= b else b / a


TNORMS: dict[str, Callable[[float, float], float]] = {
    "lukasiewicz": lukasiewicz_tnorm,
    "godel": godel_tnorm,
    "product": product_tnorm,
}
IMPLIES: dict[str, Callable[[float, float], float]] = {
    "lukasiewicz": lukasiewicz_implies,
    "godel": godel_implies,
    "product": product_implies,
}


@dataclass
class Rule:
    """A weighted implication ``antecedent -> consequent`` in some logic."""

    antecedent: str
    consequent: str
    weight: float = 1.0
    logic: str = "lukasiewicz"

    def truth(self, valuation: dict[str, float]) -> float:
        a = clamp(valuation.get(self.antecedent, 0.0))
        b = clamp(valuation.get(self.consequent, 0.0))
        return IMPLIES[self.logic](a, b)

    def penalty(self, valuation: dict[str, float]) -> float:
        """Energy contribution: weight * (1 - truth). Zero when satisfied."""
        return self.weight * (1.0 - self.truth(valuation))


@dataclass
class Theory:
    """A set of rules whose total penalty is an energy over valuations."""

    rules: list[Rule]

    def energy(self, valuation: dict[str, float]) -> float:
        return float(sum(r.penalty(valuation) for r in self.rules))

    def satisfied(self, valuation: dict[str, float], tol: float = 1e-9) -> bool:
        return self.energy(valuation) <= tol
