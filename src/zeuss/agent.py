"""Vault Run: a self-contained perceive->represent->infer->choose agent.

Composes tiers that have so far only ever been demoed in isolation
(``python -m zeuss demo`` / ``cli.py`` calls each primitive independently in one
script) into one stateful decision loop, on a small 5-room navigation toy domain
with uncertain doors to probe:

    perceive  - sense_door: a noisy reading of a probed door's true state.
    represent - Ontology (room kinds, shared Codebook) + Theory (door beliefs).
    infer     - compile_theory -> settle -> readout recovers each door's belief
                in [0, 1] from the sensor evidence gathered so far; qa.ask reads
                a room's kind (corridor / deadend / goal_room) off the same KB.
    choose    - drive.select_action, called twice per tick: once against a
                goal Landscape (built from Ontology room waves) to pick a move,
                once against the belief Landscape compile_theory produced to
                pick which ambiguous door to probe next (missing_params=True) -
                the literal closing of "nothing feeds a compiled Theory back in
                as a goal Landscape for select_action" (see docs/ROADMAP.md).

Two real correctness traps, verified by reading the actual math before relying
on it (see ``believe_doors`` and ``_bfs_distance``'s docstrings for the details):
Lukasiewicz implication only pulls belief in one direction unless both rule
directions are registered, and offering every known-open neighbor as a move
candidate (regardless of whether it makes progress) causes the agent to thrash
forever between two rooms connected by an always-open door.

Tier4 program synthesis is deliberately not used here: it operates over plain
dict/list inputs with no representation for hypervectors or fuzzy valuations,
so bridging it into this loop needs its own domain-terms design, not glue that
falls out of the existing APIs the way everything below does.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .drive import Action, select_action
from .qa import ask
from .tier2_substrate.energy import Landscape, settle
from .tier2_substrate.hypervectors import Codebook, bind, random_hypervector
from .tier3_logic.compiler import Rule, Theory
from .tier3_logic.grounding import compile_theory, readout
from .tier3_logic.ontology import Ontology

# Above this readout, a door is believed open enough to move through.
OPEN_THRESHOLD = 0.65
# Below this readout, a door is believed closed enough to stop considering it.
CLOSED_THRESHOLD = 0.35


@dataclass
class Door:
    """One Theory variable's worth of ground truth plus what's been sensed
    about it so far. ``sensed`` is ``None`` until :func:`sense_door` (via
    :meth:`Agent.tick`) records a reading - a door nobody has probed
    contributes no evidence to :func:`believe_doors`, not a false one."""

    name: str
    rooms: tuple[str, str]
    ground_truth: float
    sensed: float | None = None


@dataclass
class World:
    """The map: rooms (each with an ontology ``is_a`` kind), doors between
    them, and the one shared :class:`~zeuss.tier3_logic.ontology.Ontology`
    (and therefore the one shared :class:`Codebook`) every other subsystem in
    this module reads from - ``Ontology`` builds its own ``Codebook``
    internally (``field(init=False)``, no way to hand it an external one), so
    every call site below passes ``world.onto.codebook`` explicitly rather
    than minting a second, disconnected vector space."""

    dim: int = 4096
    seed: int = 0
    onto: Ontology = field(init=False)
    doors: list[Door] = field(default_factory=list)
    adjacency: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.onto = Ontology(dim=self.dim, seed=self.seed)
        self._doors_by_name: dict[str, Door] = {}
        self._memory = None

    def add_room(self, name: str, kind: str) -> None:
        self.onto.add(name, "is_a", kind)
        self.adjacency.setdefault(name, [])

    def add_door(self, name: str, room_a: str, room_b: str, ground_truth: float, known: bool = False) -> None:
        door = Door(name=name, rooms=(room_a, room_b), ground_truth=ground_truth)
        # known=True doors (eh/eg/av in demo_world's canned map) are
        # permanently "sensed" at construction, not left to be discovered -
        # an explicit flag, not inferred from ground_truth==1.0: the whole
        # point of a scenario is that the shortcut/detour's ground truth
        # varies while they still start unprobed regardless of which way
        # that scenario's truth happens to come out (a ground_truth==1.0
        # inference would have silently pre-revealed an "open" shortcut
        # before any probe - found and fixed during the first diagnostic
        # run, not assumed safe).
        if known:
            door.sensed = ground_truth
        self.doors.append(door)
        self._doors_by_name[name] = door
        self.adjacency.setdefault(room_a, []).append(room_b)
        self.adjacency.setdefault(room_b, []).append(room_a)

    def neighbors(self, room: str) -> list[str]:
        return list(self.adjacency.get(room, []))

    def door_between(self, a: str, b: str) -> Door:
        for door in self.doors:
            if set(door.rooms) == {a, b}:
                return door
        raise KeyError(f"no door between {a!r} and {b!r}")

    def door_by_name(self, name: str) -> Door:
        return self._doors_by_name[name]

    def memory(self):
        """``onto.ground()``, cached - safe because every room/door is added
        once at construction (only ``Door.sensed`` changes during a run, and
        that never touches the ontology's triples)."""
        if self._memory is None:
            self._memory = self.onto.ground()
        return self._memory


def demo_world(seed_doors: dict[str, float] | None = None, dim: int = 4096, seed: int = 0) -> World:
    """The canned 5-room "Vault Run" map. ``seed_doors`` overrides ground
    truth for the two doors that start uncertain (``door_hv_open`` - the
    shortcut, ``door_ha_open`` - the detour), selecting one of the three
    documented scenarios (see module docstring / docs/ROADMAP.md)."""
    world = World(dim=dim, seed=seed)
    world.add_room("entry", "corridor")
    world.add_room("hall", "corridor")
    world.add_room("garden", "deadend")
    world.add_room("annex", "corridor")
    world.add_room("vault", "goal_room")

    overrides = seed_doors or {}
    world.add_door("door_eh_open", "entry", "hall", ground_truth=1.0, known=True)
    world.add_door("door_eg_open", "entry", "garden", ground_truth=1.0, known=True)
    world.add_door("door_hv_open", "hall", "vault", ground_truth=overrides.get("door_hv_open", 1.0))
    world.add_door("door_ha_open", "hall", "annex", ground_truth=overrides.get("door_ha_open", 0.0))
    world.add_door("door_av_open", "annex", "vault", ground_truth=1.0, known=True)
    return world


def sense_door(door: Door, rng: np.random.Generator, noise: float = 0.05) -> float:
    """A noisy reading of ``door.ground_truth``, clipped to ``[0, 1]``."""
    reading = door.ground_truth + rng.normal(0.0, noise)
    return float(min(1.0, max(0.0, reading)))


def believe_doors(doors: list[Door]) -> Theory:
    """One rule pair per *sensed* door - both Lukasiewicz implication
    directions, not one.

    ``lukasiewicz_implies(a, b) = clamp(1 - a + b)``, so a single rule
    ``Rule("sensor_X", "door_X_open", w)`` gives penalty
    ``w * clamp(sensor - door_belief)`` - exactly zero whenever
    ``door_belief >= sensor``. A *low* sensor reading therefore never pulls a
    *high* door-belief back down; the rule only ever pulls belief up, never
    down (verified directly against ``compiler.py``'s actual implication
    function, not assumed from its name). Registering the reverse rule too
    (``Rule("door_X_open", "sensor_X", w)``, penalty
    ``w * clamp(door_belief - sensor)``) makes the two penalties sum to
    ``w * |sensor - door_belief|`` - a true biconditional, pulling belief
    toward the sensor reading from either side.

    A door with ``sensed is None`` contributes no rule at all, not a
    "sensed 0.5" rule - see :func:`door_beliefs`'s docstring for why that
    matters for an unprobed door's readout.
    """
    rules: list[Rule] = []
    for door in doors:
        if door.sensed is None:
            continue
        sensor_var = f"sensor_{door.name}"
        rules.append(Rule(sensor_var, door.name, weight=6.0))
        rules.append(Rule(door.name, sensor_var, weight=6.0))
    return Theory(rules=rules)


def door_beliefs(
    codebook: Codebook,
    doors: list[Door],
    settle_steps: int = 40,
    ensemble: int = 32,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """``compile_theory -> settle -> readout``, averaged over ``ensemble``
    independent random restarts - the Frontier-2 belief-update pipeline,
    re-solved fresh each tick rather than carrying ``z`` forward across ticks
    the way ``grounding.anneal_theory`` does across its own schedule (simpler
    and more testable at this toy scale - a deliberate v1 simplification).

    Sensor readings are known exactly, not enumerated - they're pinned in via
    ``fixed``, while the door-open variables (the actual unknowns) are what
    ``compile_theory`` enumerates over.

    The ensemble is load-bearing, not a robustness nicety: a *single* settle
    trajectory on a genuinely underdetermined theory does not converge to an
    even blend of its tied corners - ``grounding.anneal_theory``'s own
    docstring already documents this ("a single settling trajectory is one
    continuous state... it spontaneously breaks the symmetry and commits to
    *one* of the tied corners rather than hovering between them"). A door
    with no sensor rule at all leaves every corner tied on that one bit, so a
    single random-``z0`` settle reads out near 0 *or* 1 essentially at
    random, not near 0.5 - measured directly during development (both
    ``door_hv_open`` and ``door_ha_open`` read ~0.3 with zero probes ever
    taken, before this ensemble average was added) rather than assumed safe
    from the single-call version this function started as. Averaging several
    independent restarts washes an unprobed variable's random corner-choice
    back out toward 0.5, while a confidently-sensed variable (every corner's
    surviving weight agrees on its bit) stays stable regardless of how many
    restarts are averaged - checked directly in
    ``tests/test_agent.py::test_unsensed_door_reads_out_near_fifty_fifty``.

    ``ensemble=32`` is itself a measured choice, not a guess: an ensemble
    that's too small just trades one symmetry-breaking coin flip for another
    at one remove - measuring the actual sampling spread across 60 rng draws
    found ``ensemble=8`` (the first value tried) crosses either threshold
    ~1.7% of the time (std ~0.07 around the true 0.5), which is exactly what
    produced a real failure 1/30 seeds into an early end-to-end sweep of the
    detour-recovery scenario (an unprobed door read 0.347, one thousandth
    below :data:`CLOSED_THRESHOLD`, and the agent wrongly gave up on it
    without ever probing). ``ensemble=16`` already measured zero crossings
    in the same 60 draws (std ~0.05); ``32`` was picked for further margin at
    negligible added cost (``settle`` is cheap at this toy scale).
    """
    rng = np.random.default_rng() if rng is None else rng
    variables = [door.name for door in doors]
    fixed = {f"sensor_{door.name}": door.sensed for door in doors if door.sensed is not None}
    theory = believe_doors(doors)
    landscape = compile_theory(codebook, theory, variables, fixed=fixed)
    accum = {name: 0.0 for name in variables}
    for _ in range(ensemble):
        z0 = random_hypervector(codebook.dim, rng)
        z_final, _energies = settle(landscape, z0, steps=settle_steps, rng=rng)
        readings = readout(codebook, variables, z_final)
        for name in variables:
            accum[name] += readings[name]
    return {name: accum[name] / ensemble for name in variables}


def _bfs_distance(adjacency: dict[str, list[str]], goal: str) -> dict[str, int]:
    """Hop-count from every room to ``goal`` over the *physical* map layout -
    deliberately independent of door beliefs (a closed door still physically
    connects two rooms, it's just not yet known to be passable), since this
    is the agent's spatial prior about which rooms are worth visiting at all,
    computed before any sensor evidence is gathered."""
    dist = {goal: 0}
    queue = deque([goal])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, []):
            if neighbor not in dist:
                dist[neighbor] = dist[current] + 1
                queue.append(neighbor)
    return dist


def distance_weighted_goal_landscape(onto: Ontology, adjacency: dict[str, list[str]], goal: str, decay: float = 0.5) -> Landscape:
    """BFS the physical adjacency graph from ``goal`` outward; register every
    reached room as an attractor at weight ``decay ** hops``. Closes the gap:
    nothing in this codebase previously turned graph/ontology structure into
    a goal ``Landscape`` for ``drive.select_action`` - this is the graph-
    structure counterpart to ``compile_theory``'s ``Theory -> Landscape``
    compile."""
    dist = _bfs_distance(adjacency, goal)
    landscape = Landscape()
    for room, hops in dist.items():
        landscape.add(onto.entity(room), weight=decay**hops)
    return landscape


def candidate_actions(
    world: World, current_room: str, beliefs: dict[str, float], goal: str
) -> tuple[list[Action], dict[str, tuple[str, str]], bool]:
    """Turn ``qa.ask`` room-kind classification + ``door_beliefs`` readout
    into an ``Action`` list. Returns ``(actions, meta, missing_params)``;
    ``meta`` maps ``action.name -> ("move", room)`` or
    ``("probe", door_name)``. ``missing_params`` is True iff no move
    candidate exists (mirrors ``drive.py``'s own stated meaning: "input
    parameters are missing").

    A neighbor is only ever offered - as a move *or* a probe - if it is *no
    farther* from ``goal`` than the current room (the same BFS table
    :func:`distance_weighted_goal_landscape` uses) - lateral (equal-distance)
    moves are allowed, strictly-worse ones are not. Without excluding
    strictly-worse moves, any room connected by an always-open door (e.g.
    ``entry``, from ``hall``) is a legal move candidate forever, and a one-
    candidate list trivially wins `select_action` even though the move makes
    no progress - found and fixed during design, before it could reproduce
    it as a failing test: the agent thrashed between two known-open rooms
    indefinitely, never reaching the ambiguous doors that would actually
    resolve anything. Lateral moves have to stay allowed, not tightened to
    strict improvement only, for a different reason found the same way:
    ``hall`` and ``annex`` are *both* exactly one hop from ``vault`` in this
    map (siblings around the goal, not nested) - a strict-improvement-only
    filter excluded the ``hall -> annex`` move entirely, silently breaking
    the detour-recovery scenario (S2) even though ``door_ha_open`` read
    correctly as ambiguous; the agent simply had nowhere to apply that
    belief. Rooms of ``is_a`` kind ``"deadend"`` are excluded too,
    redundantly with the distance check in this domain (a deadend is always
    strictly farther from the goal) but kept as an explicit, self-
    documenting condition rather than relying only on the numeric side
    effect.
    """
    dist = _bfs_distance(world.adjacency, goal)
    current_dist = dist.get(current_room, float("inf"))
    actions: list[Action] = []
    meta: dict[str, tuple[str, str]] = {}
    for neighbor in world.neighbors(current_room):
        kind = ask(world.onto, world.memory(), neighbor, "is_a").answer
        if kind == "deadend":
            continue
        neighbor_dist = dist.get(neighbor, float("inf"))
        if neighbor_dist > current_dist:
            continue
        door = world.door_between(current_room, neighbor)
        belief = beliefs.get(door.name, 0.5)
        if belief >= OPEN_THRESHOLD:
            name = f"move_to_{neighbor}"
            actions.append(Action(name, world.onto.entity(neighbor), is_discovery=False))
            meta[name] = ("move", neighbor)
        elif belief > CLOSED_THRESHOLD:
            # Ambiguous - worth probing, since it would be a forward
            # improvement if it turns out to be open. Effect is the door
            # variable's own TRUE pole (the same wave grounding.readout
            # reads similarity against), reusing valuation_to_hypervector's
            # construction convention rather than an unrelated marker.
            true_pole = bind(
                world.onto.codebook.symbol(f"VAR:{door.name}"), world.onto.codebook.symbol(f"{door.name}:TRUE")
            )
            name = f"probe_{door.name}"
            actions.append(Action(name, true_pole, is_discovery=True))
            meta[name] = ("probe", door.name)
        # belief <= CLOSED_THRESHOLD: confirmed closed, no candidate at all.
    missing_params = not any(kind == "move" for kind, _ in meta.values())
    return actions, meta, missing_params


@dataclass
class TickResult:
    tick: int
    room: str
    action: str
    kind: str  # "move" | "probe" | "stuck"
    beliefs: dict[str, float]


@dataclass
class Agent:
    world: World
    current_room: str
    goal: str
    rng: np.random.Generator
    history: list[TickResult] = field(default_factory=list)

    def room_kind(self, room: str) -> str:
        return ask(self.world.onto, self.world.memory(), room, "is_a").answer

    def tick(self) -> TickResult:
        """One perceive -> represent -> infer -> choose -> act step."""
        beliefs = door_beliefs(self.world.onto.codebook, self.world.doors, rng=self.rng)
        actions, meta, missing_params = candidate_actions(self.world, self.current_room, beliefs, self.goal)
        move_actions = [a for a in actions if meta[a.name][0] == "move"]
        probe_actions = [a for a in actions if meta[a.name][0] == "probe"]

        if move_actions:
            landscape = distance_weighted_goal_landscape(self.world.onto, self.world.adjacency, self.goal)
            state = self.world.onto.entity(self.current_room)
            # step_size=1.0 (a full step, not hypothetical_state's own 0.3
            # default nudge): a move decision asks "how good would it be to
            # *be* at the destination," not "how good is a small nudge
            # toward it" - found necessary by measurement, not assumed. At
            # the default 0.3, the hypothetical state stays mostly similar
            # to the *current* room's own wave (itself a registered
            # attractor), which swamps the much smaller destination-
            # dependent signal: moving from `annex` (1 hop from goal),
            # `move_to_vault` (0 hops) and `move_to_hall` (1 hop, lateral)
            # scored EFE=-0.4676 vs -0.4678 - a noise-level tie that let the
            # agent thrash back and forth forever instead of ever taking the
            # one-hop-away, always-open move to the actual goal. At
            # step_size=1.0 the same pair scores -0.999 vs -0.499 - a clear,
            # correct preference.
            chosen = select_action(
                landscape, self.world.onto.codebook, state, move_actions, missing_params=False, step_size=1.0
            )
            _kind, target_room = meta[chosen.name]
            self.current_room = target_room
            result = TickResult(len(self.history), self.current_room, chosen.name, "move", beliefs)
        elif probe_actions:
            variables = [door.name for door in self.world.doors]
            fixed = {
                f"sensor_{door.name}": door.sensed for door in self.world.doors if door.sensed is not None
            }
            belief_landscape = compile_theory(
                self.world.onto.codebook, believe_doors(self.world.doors), variables, fixed=fixed
            )
            state = self.world.onto.entity(self.current_room)
            chosen = select_action(
                belief_landscape, self.world.onto.codebook, state, probe_actions, missing_params=True
            )
            _kind, door_name = meta[chosen.name]
            door = self.world.door_by_name(door_name)
            door.sensed = sense_door(door, self.rng)
            result = TickResult(len(self.history), self.current_room, chosen.name, "probe", beliefs)
        else:
            result = TickResult(len(self.history), self.current_room, "none", "stuck", beliefs)

        self.history.append(result)
        return result

    def run(self, max_ticks: int = 20) -> list[TickResult]:
        for _ in range(max_ticks):
            if self.current_room == self.goal:
                break
            result = self.tick()
            if result.kind == "stuck":
                break
        return self.history
