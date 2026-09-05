"""Resonance-guided search bias: an Estimation-of-Distribution prior over
:func:`.search._recursive_template`'s hole-fillers, implemented with the same
hypervector primitives every other tier of this project uses (``Codebook``,
weighted superposition, similarity-based readout), instead of a bolted-on
statistics table.

Each hole category (comparison operator, base constant/value, combine
operator, decrement step) gets a running hypervector memory: every time a
population member is observed to *structurally match* the template shape
(regardless of whether it came from the template generator or was evolved
into that shape by mutation/crossover - see
:func:`.search.extract_template_choices`), its specific hole-fillers are
added to that category's memory, weighted by how good the individual turned
out. Sampling reads the memory back via a softmax over similarity to each
candidate filler's symbol - the same "occupancy" idea
:mod:`zeuss.tier2_substrate.collapse` uses for reading a codebook, just
applied to the search's own evolving belief about what tends to work,
discovered *during* the run rather than hand-tuned in advance.

This is deliberately small in scope: a prior over five independent
categorical choices, not a general program-structure model. It is an
Estimation-of-Distribution Algorithm (a known GP technique), reimplemented on
top of VSA resonance instead of explicit probability tables, to see whether
that reframing helps - not assumed to.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..tier2_substrate.collapse import softmax
from ..tier2_substrate.hypervectors import Codebook, similarity


@dataclass
class ResonantBias:
    codebook: Codebook
    categories: dict[str, list]
    beta: float = 0.2
    _accum: dict = field(default_factory=dict)

    def sample(self, category: str, rng: np.random.Generator):
        """Pick a filler for ``category``, biased by accumulated resonance -
        uniform at random until any evidence exists."""
        options = self.categories[category]
        accum = self._accum.get(category)
        if accum is None:
            return options[int(rng.integers(0, len(options)))]
        sims = np.array([similarity(accum, self.codebook.symbol(f"{category}:{v}")) for v in options])
        probs = softmax(self.beta * sims)
        return options[int(rng.choice(len(options), p=np.asarray(probs)))]

    def reinforce(self, choices: dict[str, object], weight: float) -> None:
        """Add one observation - a set of hole-fillers that co-occurred in an
        individual scored at ``weight`` (higher is better) - to each
        category's memory.

        Deliberately *not* renormalized after every addition: FHRR
        ``normalize`` projects each component back to unit modulus, so
        renormalizing on every reinforcement would make each new observation
        dominate the accumulated history instead of properly weighting into
        a running sum. The raw (unnormalized) accumulator is compared via
        ``similarity`` at read time in :meth:`sample`, which doesn't require
        unit modulus - only the phase (direction), which a true running sum
        preserves correctly.
        """
        if weight <= 0:
            return
        for category, value in choices.items():
            if category not in self.categories:
                continue
            sym = self.codebook.symbol(f"{category}:{value}")
            contribution = weight * sym
            existing = self._accum.get(category)
            self._accum[category] = contribution if existing is None else existing + contribution
