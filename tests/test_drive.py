"""Expected Free Energy action scoring: goal progress and uncertainty reduction."""
import numpy as np

from zeuss.drive import Action, epistemic_value, select_action
from zeuss.tier2_substrate.energy import Landscape
from zeuss.tier2_substrate.hypervectors import Codebook, bundle, random_hypervector


def test_select_action_prefers_lower_energy_goal_progress():
    cb = Codebook(dim=4096, seed=0)
    goal = cb.symbol("goal")
    away = cb.symbol("away")
    land = Landscape().add(goal, 1.0)
    state = cb.symbol("state")

    toward = Action("toward_goal", goal)
    away_action = Action("away", away)

    picked = select_action(land, cb, state, [toward, away_action], epistemic_weight=0.0)
    assert picked.name == "toward_goal"


def test_select_action_prefers_uncertainty_reducing_action_when_goals_tie():
    cb = Codebook(dim=4096, seed=1)
    red = cb.symbol("red")
    blue = cb.symbol("blue")
    for name in ("green", "square", "circle"):
        cb.symbol(name)

    # An ambiguous state, tied between two known symbols.
    state = bundle([red, blue])
    # A landscape with a negligible attractor, so the pragmatic term stays
    # ~flat and the epistemic term decides.
    land = Landscape().add(random_hypervector(cb.dim, np.random.default_rng(9)), weight=0.001)

    resolves_ambiguity = Action("resolve", red)  # pushes toward a known symbol
    unrelated = Action("noop", random_hypervector(cb.dim, np.random.default_rng(5)))

    ev_resolve = epistemic_value(cb, state, resolves_ambiguity, inverse_temperature=2.0)
    ev_noop = epistemic_value(cb, state, unrelated, inverse_temperature=2.0)
    assert ev_resolve > ev_noop

    picked = select_action(
        land, cb, state, [resolves_ambiguity, unrelated], pragmatic_weight=0.0, entropy_beta=2.0
    )
    assert picked.name == "resolve"


def test_missing_params_restricts_to_discovery_actions():
    cb = Codebook(dim=4096, seed=2)
    goal = cb.symbol("goal")
    other = cb.symbol("other")
    land = Landscape().add(goal, 1.0)
    state = cb.symbol("state")

    # This action has lower (better) raw EFE...
    good_non_discovery = Action("good", goal, is_discovery=False)
    # ...but this one is flagged as a low-risk discovery action.
    mediocre_discovery = Action("probe", other, is_discovery=True)

    picked_normal = select_action(
        land, cb, state, [good_non_discovery, mediocre_discovery], missing_params=False, entropy_beta=2.0
    )
    assert picked_normal.name == "good"

    picked_missing = select_action(
        land, cb, state, [good_non_discovery, mediocre_discovery], missing_params=True, entropy_beta=2.0
    )
    assert picked_missing.name == "probe"
