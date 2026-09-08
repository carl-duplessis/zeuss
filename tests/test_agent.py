"""Tests for src/zeuss/agent.py - the self-contained perceive->represent->
infer->choose loop composing tiers that were previously only ever demoed in
isolation. See the module docstring there and docs/ROADMAP.md for the full
design rationale, including two real bugs found and fixed during development
(not hypothetical edge cases): Lukasiewicz implication only pulling belief in
one direction, and a naive move-candidate filter causing infinite thrash
between two always-open rooms.
"""
import numpy as np
import pytest

from zeuss.agent import (
    CLOSED_THRESHOLD,
    OPEN_THRESHOLD,
    Agent,
    Door,
    believe_doors,
    candidate_actions,
    demo_world,
    distance_weighted_goal_landscape,
    door_beliefs,
)
from zeuss.tier2_substrate.energy import settle
from zeuss.tier2_substrate.hypervectors import Codebook, random_hypervector
from zeuss.tier3_logic.grounding import compile_theory, readout

# --- unit tests --------------------------------------------------------


def test_believe_doors_biconditional_pulls_toward_sensor_in_both_directions():
    """A single-direction Rule("sensor_X", "door_X_open", w) only ever pulls
    belief *up* (lukasiewicz_implies(a,b)=clamp(1-a+b) gives zero penalty
    whenever b>=a) - a high-sensor-only test would not catch a regression to
    that single-direction bug. Checks both a high and a low sensor reading."""
    cb = Codebook(dim=2048, seed=0)
    for sensor_reading, expect_high in [(0.95, True), (0.05, False)]:
        door = Door(name="door_x", rooms=("a", "b"), ground_truth=1.0 if expect_high else 0.0, sensed=sensor_reading)
        theory = believe_doors([door])
        landscape = compile_theory(cb, theory, ["door_x"], fixed={"sensor_door_x": door.sensed})
        rng = np.random.default_rng(0)
        z, _ = settle(landscape, random_hypervector(cb.dim, rng), steps=40, rng=rng)
        belief = readout(cb, ["door_x"], z)["door_x"]
        if expect_high:
            assert belief > OPEN_THRESHOLD, f"high sensor reading should pull belief up, got {belief}"
        else:
            assert belief < CLOSED_THRESHOLD, f"low sensor reading should pull belief down, got {belief}"


def test_unsensed_door_reads_out_near_fifty_fifty():
    """A door with no sensor rule at all should read out near 0.5, not near
    0 or 1 - checked directly (ensemble-averaged over independent settle
    restarts; a single restart symmetry-breaks toward one random tied
    corner, see door_beliefs's docstring), not assumed from the theory."""
    cb = Codebook(dim=2048, seed=1)
    door = Door(name="door_x", rooms=("a", "b"), ground_truth=1.0, sensed=None)
    rng = np.random.default_rng(2)
    beliefs = door_beliefs(cb, [door], ensemble=32, rng=rng)
    assert 0.3 < beliefs["door_x"] < 0.7


def test_distance_weighted_goal_landscape_prefers_nearer_room():
    """A room closer to the goal should have lower energy (more preferred)
    when the state is exactly at that room's own wave - checked across
    several seeds/dims, not assumed reliable at a single one."""
    for seed in range(5):
        world = demo_world(seed=seed)
        landscape = distance_weighted_goal_landscape(world.onto, world.adjacency, "vault")
        e_hall = landscape.energy(world.onto.entity("hall"))  # 1 hop from vault
        e_entry = landscape.energy(world.onto.entity("entry"))  # 2 hops from vault
        assert e_hall < e_entry, f"seed={seed}: hall should score better than entry ({e_hall} vs {e_entry})"


def test_candidate_actions_excludes_deadend_rooms_via_ontology():
    world = demo_world()
    beliefs = {"door_eh_open": 1.0, "door_eg_open": 1.0, "door_hv_open": 1.0, "door_ha_open": 1.0, "door_av_open": 1.0}
    actions, meta, _missing = candidate_actions(world, "entry", beliefs, "vault")
    targets = {meta[a.name][1] for a in actions}
    assert "garden" not in targets, "garden is a deadend and must never be offered as a candidate"


def test_candidate_actions_missing_params_true_iff_no_confident_open_neighbor():
    world = demo_world()
    all_open = {"door_eh_open": 1.0, "door_eg_open": 1.0, "door_hv_open": 1.0, "door_ha_open": 1.0, "door_av_open": 1.0}
    _actions, _meta, missing_when_open = candidate_actions(world, "entry", all_open, "vault")
    assert missing_when_open is False

    all_ambiguous = {k: 0.5 for k in all_open}
    _actions2, _meta2, missing_when_ambiguous = candidate_actions(world, "entry", all_ambiguous, "vault")
    assert missing_when_ambiguous is True


def test_room_kind_uses_qa_ask_and_reports_high_coherence_for_known_facts():
    world = demo_world()
    agent = Agent(world=world, current_room="entry", goal="vault", rng=np.random.default_rng(0))
    assert agent.room_kind("vault") == "goal_room"
    assert agent.room_kind("garden") == "deadend"
    assert agent.room_kind("entry") == "corridor"


# --- end-to-end, swept across seeds and all 3 scenarios -----------------
# 100% pass required across the sweep: this is new code with no prior
# single-seed-luck history to inherit (unlike tier4's recursion targets,
# which were already seed-fragile before anyone measured them), so this
# should be achievable without heroics at this toy scale. It wasn't achieved
# on the first attempt - see docs/ROADMAP.md for the two real bugs found and
# fixed by diagnosing actual failures against this exact sweep, not by
# special-casing a failing seed away.

SCENARIOS = {
    "S1_shortcut_open": {"door_hv_open": 1.0, "door_ha_open": 0.0},
    "S2_detour_recovery": {"door_hv_open": 0.0, "door_ha_open": 1.0},
    "S3_no_path": {"door_hv_open": 0.0, "door_ha_open": 0.0},
}
SEEDS = range(8)


@pytest.mark.parametrize("seed", SEEDS)
def test_agent_reaches_goal_via_shortcut_when_open(seed):
    world = demo_world(seed_doors=SCENARIOS["S1_shortcut_open"])
    agent = Agent(world=world, current_room="entry", goal="vault", rng=np.random.default_rng(seed))
    agent.run(max_ticks=20)
    assert agent.current_room == "vault"


@pytest.mark.parametrize("seed", SEEDS)
def test_agent_recovers_via_detour_when_shortcut_closed(seed):
    world = demo_world(seed_doors=SCENARIOS["S2_detour_recovery"])
    agent = Agent(world=world, current_room="entry", goal="vault", rng=np.random.default_rng(seed))
    history = agent.run(max_ticks=20)
    probed_names = [r.action for r in history if r.kind == "probe"]
    assert "probe_door_hv_open" in probed_names, "the shortcut must actually be probed, not skipped"
    assert agent.current_room == "vault"
    # The shortcut must have genuinely been found closed at some point during
    # the run (its belief drops below CLOSED_THRESHOLD after being probed),
    # not just probed without the reading ever being acted on.
    final_hv_belief = agent.world.door_by_name("door_hv_open").sensed
    assert final_hv_belief is not None and final_hv_belief < CLOSED_THRESHOLD


@pytest.mark.parametrize("seed", SEEDS)
def test_agent_reports_stuck_when_no_path_exists(seed):
    world = demo_world(seed_doors=SCENARIOS["S3_no_path"])
    agent = Agent(world=world, current_room="entry", goal="vault", rng=np.random.default_rng(seed))
    history = agent.run(max_ticks=20)
    assert history[-1].kind == "stuck"
    assert agent.current_room != "vault"
    assert len(history) < 20, "must terminate cleanly on its own, not merely stop at the tick budget"


def test_agent_run_is_deterministic_given_a_seed():
    def run():
        world = demo_world(seed_doors=SCENARIOS["S2_detour_recovery"])
        agent = Agent(world=world, current_room="entry", goal="vault", rng=np.random.default_rng(7))
        history = agent.run(max_ticks=20)
        return [(r.action, r.room, r.kind) for r in history]

    assert run() == run()


@pytest.mark.parametrize("seed", SEEDS)
def test_agent_does_not_thrash_between_known_open_rooms(seed):
    """Regression guard for the bug found during design: offering any known-
    open neighbor as a move candidate (regardless of whether it makes
    progress) let the agent bounce between hall and entry, or between hall
    and annex, forever. Counts actual room *transitions* (kind == "move"),
    not every tick's location - a room legitimately appears in consecutive
    ticks while being probed without the agent going anywhere, which is not
    thrashing (an earlier version of this test wrongly flagged exactly that
    as a failure). No room should be moved *into* more than twice."""
    for doors in SCENARIOS.values():
        world = demo_world(seed_doors=doors)
        agent = Agent(world=world, current_room="entry", goal="vault", rng=np.random.default_rng(seed))
        history = agent.run(max_ticks=20)
        moved_into = [r.room for r in history if r.kind == "move"]
        for room in set(moved_into):
            count = moved_into.count(room)
            assert count <= 2, f"seed={seed} doors={doors}: moved into {room!r} {count} times: {moved_into}"
