"""Active-inference-flavoured action selection: minimise Expected Free Energy.

``EFE(a) = pragmatic_weight * pragmatic_value(a) - epistemic_weight * epistemic_value(a)``

``pragmatic_value`` is the predicted post-action energy under a goal
``Landscape`` (lower is better - closer to the goal); ``epistemic_value`` is
the predicted entropy *reduction* from acting (higher is better - more
uncertainty resolved). Both reuse existing substrate primitives
(``Landscape.energy``, ``collapse.entropy``/``occupancy``) on a one-step
hypothetical blend of the current state toward the action's effect - the same
blend :func:`zeuss.tier2_substrate.energy.settle` already uses internally.

This is a scoped, testable EFE-style scorer over a *discrete action set* -
not a full Friston active-inference generative model (no beliefs, no
perception-action loop over a learned world model). It is honestly a
multi-objective scoring function, not a claim of biological-agent autonomy.
"""
from __future__ import annotations

from dataclasses import dataclass

from .tier2_substrate.collapse import entropy, occupancy
from .tier2_substrate.energy import Landscape
from .tier2_substrate.hypervectors import Codebook, normalize


@dataclass
class Action:
    """A candidate action: the hypervector it would move the state toward."""

    name: str
    effect: "object"  # xp.ndarray
    is_discovery: bool = False  # a low-risk, information-gathering action


def hypothetical_state(state, action: Action, step_size: float = 0.3):
    """The state after taking one step of ``action`` (the same mean-field
    blend :func:`energy.settle` uses toward its target)."""
    return normalize((1.0 - step_size) * state + step_size * action.effect)


def pragmatic_value(
    landscape: Landscape, state, action: Action, step_size: float = 0.3, inverse_temperature: float = 8.0
) -> float:
    """Predicted post-action energy under the goal landscape (lower is better)."""
    return landscape.energy(hypothetical_state(state, action, step_size), inverse_temperature)


def epistemic_value(
    codebook: Codebook, state, action: Action, step_size: float = 0.3, inverse_temperature: float = 1.0
) -> float:
    """Predicted entropy *reduction* from acting (higher is better)."""
    current = entropy(occupancy(codebook, state, inverse_temperature))
    future = entropy(occupancy(codebook, hypothetical_state(state, action, step_size), inverse_temperature))
    return current - future


def expected_free_energy(
    landscape: Landscape,
    codebook: Codebook,
    state,
    action: Action,
    pragmatic_weight: float = 1.0,
    epistemic_weight: float = 1.0,
    step_size: float = 0.3,
    energy_beta: float = 8.0,
    entropy_beta: float = 1.0,
) -> float:
    """Lower is better: goal progress rewarded, uncertainty reduction rewarded."""
    pv = pragmatic_value(landscape, state, action, step_size, energy_beta)
    ev = epistemic_value(codebook, state, action, step_size, entropy_beta)
    return pragmatic_weight * pv - epistemic_weight * ev


def select_action(
    landscape: Landscape,
    codebook: Codebook,
    state,
    actions: list[Action],
    missing_params: bool = False,
    **efe_kwargs,
) -> Action:
    """Pick ``argmin EFE(a)`` over ``actions``.

    If ``missing_params`` is set and any action is flagged ``is_discovery``,
    the candidate set is restricted to discovery actions first - a literal
    implementation of "autonomously execute low-risk discovery actions to
    gather environment state data when input parameters are missing", not a
    vague autonomy claim.
    """
    candidates = actions
    if missing_params:
        discovery = [a for a in actions if a.is_discovery]
        if discovery:
            candidates = discovery
    scored = [(expected_free_energy(landscape, codebook, state, a, **efe_kwargs), a) for a in candidates]
    return min(scored, key=lambda pair: pair[0])[1]
