"""
Environment for a Quantum-Belief multi-agent reinforcement learning
testbed.

Each agent's uncertainty about a hidden target is represented using real
quantum formalism (QuTiP amplitudes, interference, entanglement, von
Neumann entropy) instead of classical probability, with a controlled
ablation testing whether that representation -- plus an explicit
metacognitive override controller -- changes learned RL behavior. See
README.md for the full project scope.

World: 5x5 rooms, each a 3x3 grid of searchable spots ("furniture", 225
total). One hidden room is the target (where the chest actually is).

Clues are deterministic math expressions (e.g. "9 - 6" -> 3), NOT LLM
generated -- fully offline, instant, reproducible. Exactly one REAL row
clue and one REAL column clue exist (so the puzzle is always solvable),
plus 2-4 DECOY clues that look identical but report a WRONG value for
whichever axis they're tagged with. There is no in-game signal that marks
a clue as real or fake when it's read.

--- Why belief is represented this way ---

Belief is tracked with `quantum_mind.QuantumBelief`, a QuTiP-backed
complex amplitude state per agent, per axis -- not a plain classical
list of candidate values -- because that gives three real properties a
plain probability list can't:

  1. INTERFERENCE -- decoys reporting the same wrong value can partially
     cancel instead of just adding up (see quantum_mind.py).
  2. NON-COMMUTING MEASUREMENT -- reading a row clue disturbs the col
     belief and vice versa, so exploration ORDER matters, unlike an
     order-independent candidate list would.
  3. ENTROPY-THRESHOLD COLLAPSE -- once an agent's joint belief entropy
     drops far enough (see OR_ENTROPY_THRESHOLD), an "objective
     reduction" event fires on its own -- the environment resolves the
     agent's best guess automatically, whether or not the agent chose to
     act. This sits ALONGSIDE the agent-initiated DECLARE action, not
     instead of it; either route can end the episode.

There's also an ENTANGLEMENT mechanic (`entangled_pair.py`): the two
agents share one Bell-pair-like resource encoding "is the target in the
top half or bottom half of the map?", accessible via a new SENSE action.
Whichever agent senses first collapses it for both -- the second agent to
sense just reads off the now-fixed correlated answer, not a fresh random
draw.

This is classical hardware running QuTiP's linear algebra to reproduce
the mathematical STRUCTURE (interference, non-commutativity,
entanglement, threshold-collapse) that a plain classical
candidate-list belief representation does not have -- see
quantum_mind.py for the details of each mechanic.

Clues are also physical, carryable objects (PICKUP/DROP): SEARCHing a
spot reveals AND reads a clue immediately (so reading never depends on
remembering to carry it), but an agent can additionally PICK UP a clue it
has found to physically remove it from that spot and DROP it somewhere
else. Both agents explore the SAME shared World object, so relocating a
clue can hide it from -- or later be rediscovered by -- the other agent.
"""

import random

import numpy as np

from quantum_mind import QuantumBelief, OR_ENTROPY_THRESHOLD
from entangled_pair import EntangledHalfBelief

GRID_SIZE = 5
SUB_SIZE = 3
MAX_STEPS = 500
MIN_DECOYS = 2
MAX_DECOYS = 4

# action ids
UP, DOWN, LEFT, RIGHT, SEARCH, PICKUP, DROP, DECLARE, SENSE = range(9)
ACTION_NAMES = ["up", "down", "left", "right", "search", "pickup", "drop", "declare", "sense"]
NUM_ACTIONS = 9

# obs vector (base): [agent_r, agent_c,
#              top_row_guess, top_col_guess,          <- quantum belief, not raw candidate list
#              row_marginal_entropy, col_marginal_entropy,
#              sensed_half (-1 unsensed / 0 top / 1 bottom),
#              room_fully_searched, carrying_flag,
#              visited_mask(25), current_room_spot_checked(9)]
OBS_DIM_BASE = 2 + 2 + 2 + 1 + 1 + 1 + GRID_SIZE * GRID_SIZE + SUB_SIZE * SUB_SIZE
# obs vector (metacognitive variant): base + [entropy_trend, confidence]
OBS_DIM_META = OBS_DIM_BASE + 2
# default OBS_DIM kept for backward-compatible imports (`from env import OBS_DIM`)
# -- equals the metacognitive size, since that's the "Model B" / default config.
OBS_DIM = OBS_DIM_META


def _make_expr(value):
    a = value + random.randint(1, 9)
    b = a - value
    return f"{a} - {b}"


class Clue:
    __slots__ = ("id", "kind", "real", "value", "expr", "room", "spot")

    def __init__(self, id, kind, real, value, room, spot):
        self.id = id
        self.kind = kind      # "row" or "col"
        self.real = real      # True = trustworthy, False = decoy
        self.value = value    # the number this clue reports (true value if real, wrong if decoy)
        self.expr = _make_expr(value)
        self.room = room      # CURRENT live position -- mutates on pickup/drop
        self.spot = spot


class World:
    """Ground truth: target room + all clues (mutable live positions) +
    the entangled top/bottom-half resource, all shared by both agents."""

    def __init__(self):
        self.target_room = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        tr, tc = self.target_room

        # "top half" = rows 0-1 of a 5-row map; used only to seed the
        # entangled SENSE resource, not otherwise exposed to agents
        target_is_top_half = tr <= 1
        self.entangled_pair = EntangledHalfBelief(target_is_top_half)

        all_spots = [
            (r, c, i, j)
            for r in range(GRID_SIZE) for c in range(GRID_SIZE)
            for i in range(SUB_SIZE) for j in range(SUB_SIZE)
            if (r, c) != self.target_room
        ]
        random.shuffle(all_spots)

        self.clues = {}           # clue_id -> Clue
        self.clue_positions = {}  # (room, spot) -> clue_id  (live; mutates on pickup/drop)

        def place(kind, real, value):
            r, c, i, j = all_spots.pop()
            cid = len(self.clues)
            clue = Clue(cid, kind, real, value, (r, c), (i, j))
            self.clues[cid] = clue
            self.clue_positions[((r, c), (i, j))] = cid
            return clue

        place("row", True, tr)
        place("col", True, tc)

        n_decoys = random.randint(MIN_DECOYS, MAX_DECOYS)
        for _ in range(n_decoys):
            kind = random.choice(["row", "col"])
            true_val = tr if kind == "row" else tc
            fake_val = random.choice([v for v in range(GRID_SIZE) if v != true_val])
            place(kind, False, fake_val)

    def clue_at(self, room, spot):
        cid = self.clue_positions.get((room, spot))
        return self.clues[cid] if cid is not None else None

    def remove_clue_at(self, room, spot):
        return self.clue_positions.pop((room, spot), None)

    def place_clue(self, clue_id, room, spot):
        clue = self.clues[clue_id]
        clue.room, clue.spot = room, spot
        self.clue_positions[(room, spot)] = clue_id


class QuantumBeliefEnv:
    """Single-agent view onto a (possibly shared) World."""

    def __init__(self, world=None, start_room=None, agent_id="A",
                 enable_collapse=True, metacognition=True):
        self.agent_id = agent_id  # "A" or "B" -- which half of the entangled pair this env measures
        # Ablation flags for the Model A / Model B comparison:
        # enable_collapse toggles whether the entropy-
        # threshold "objective reduction" event can fire at all;
        # metacognition toggles whether entropy_trend/confidence are
        # exposed in the observation. Both default True (= "Model B" /
        # the full quantum-belief simulation described earlier).
        # Model A ("conventional") = both False.
        self.enable_collapse = enable_collapse
        self.metacognition = metacognition
        self.world = world if world is not None else World()
        # start_room=(r, c) pins the agent to a fixed spawn every reset
        # (old default behavior). start_room=None (the new default) instead
        # drops the agent in a fresh uniformly-random room EVERY reset --
        # see _random_start_room. This matters for more than variety: a
        # fixed corner start means a DQN only ever learns from one spatial
        # relationship to the target, and it silently confounds
        # compare_models.py's Model A vs Model B comparison (any measured
        # difference could partly be "who started closer this seed" rather
        # than the mechanism under test). Random start removes both.
        self.start_room = start_room
        self.reset(world=self.world)

    def _random_start_room(self):
        """Uniformly random room, excluding the current world's target
        room -- an agent should never be handed a free win by literally
        spawning on the chest."""
        target = self.world.target_room
        while True:
            room = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if room != target:
                return room

    def reset(self, world=None, start_room=None):
        """start_room here OVERRIDES self.start_room for just this one
        reset, without changing the env's configured default -- lets a
        caller (e.g. compare_models.py) hand two otherwise-independent
        envs the SAME externally-drawn random room for a given episode,
        so "random start" doesn't silently turn into "each env randomizes
        on its own and the two models end up compared under different
        starting conditions." Falls back to self.start_room (fixed room,
        or None for this env's own random draw) when not given."""
        self.world = world if world is not None else World()
        if start_room is not None:
            self.agent_room = start_room
        elif self.start_room is not None:
            self.agent_room = self.start_room
        else:
            self.agent_room = self._random_start_room()
        self.visited = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.visited[self.agent_room[0]][self.agent_room[1]] = True
        self.spots_checked = {}     # room -> 3x3 bool grid (this agent's own search memory)
        self.qbelief = QuantumBelief()
        self.tried_combos = set()   # (row, col) pairs already declared-and-failed or auto-collapsed-and-failed
        self.inventory = None       # clue_id currently carried, or None
        self.sensed = False         # has this agent used SENSE yet this episode
        self.sensed_half = None     # 0 (top) / 1 (bottom) once sensed
        self.or_triggered = False   # has the entropy-threshold collapse already fired
        self.log = []               # list of (step, text) for UI
        self.steps = 0
        self.opened = False
        self.failed_attempts = 0
        self._noop_streak = 0
        self.qbelief.record()  # seed entropy history at t=0
        return self._obs()

    def _room_checked_grid(self, room):
        if room not in self.spots_checked:
            self.spots_checked[room] = [[False] * SUB_SIZE for _ in range(SUB_SIZE)]
        return self.spots_checked[room]

    def _log(self, text):
        self.log.append((self.steps, text))
        if len(self.log) > 60:
            self.log.pop(0)

    def fully_searched(self, room):
        grid = self.spots_checked.get(room)
        if grid is None:
            return False
        return all(grid[i][j] for i in range(SUB_SIZE) for j in range(SUB_SIZE))

    def untried_combos(self):
        """(row, col) pairs from the agent's current quantum-ranked
        candidates, excluding ones already tried and failed. Ranked
        candidates come from `QuantumBelief.ranked_candidates`, which is
        weighted by interference, not just "seen at least once"."""
        rows = self.qbelief.candidates("row") or [self.qbelief.top_value("row")]
        cols = self.qbelief.candidates("col") or [self.qbelief.top_value("col")]
        # A single agent's own SENSE result is, by construction, exactly
        # 50/50 and uninformative on its own (no-signaling theorem -- see
        # entangled_pair.resolved_top_half's docstring). The entangled
        # hint only becomes usable ONCE BOTH agents have sensed and their
        # two results can be compared, which is what resolved_top_half()
        # does. Only then do we narrow the search.
        resolved = self.world.entangled_pair.resolved_top_half()
        if resolved is not None:
            if resolved:
                rows = [r for r in rows if r <= 1] or rows
            else:
                rows = [r for r in rows if r >= 2] or rows
        out = []
        for r in rows:
            for c in cols:
                if (r, c) not in self.tried_combos:
                    out.append((r, c))
        return out

    def belief_grid(self):
        """Born-rule joint probability heatmap straight from the quantum
        belief state."""
        return self.qbelief.joint_grid()

    def _obs(self):
        r, c = self.agent_room
        obs = [r / (GRID_SIZE - 1), c / (GRID_SIZE - 1)]
        obs.append(self.qbelief.top_value("row") / (GRID_SIZE - 1))
        obs.append(self.qbelief.top_value("col") / (GRID_SIZE - 1))

        def shannon(p):
            p = np.clip(p, 1e-12, 1.0)
            return float(-(p * np.log2(p)).sum())

        max_ent = np.log2(GRID_SIZE)
        obs.append(shannon(self.qbelief.probabilities("row")) / max_ent)
        obs.append(shannon(self.qbelief.probabilities("col")) / max_ent)
        obs.append(-1.0 if self.sensed_half is None else float(self.sensed_half))

        obs.append(1.0 if self.fully_searched(self.agent_room) else 0.0)
        obs.append(1.0 if self.inventory is not None else 0.0)
        for rr in range(GRID_SIZE):
            for cc in range(GRID_SIZE):
                obs.append(1.0 if self.visited[rr][cc] else 0.0)
        grid = self._room_checked_grid(self.agent_room)
        for i in range(SUB_SIZE):
            for j in range(SUB_SIZE):
                obs.append(1.0 if grid[i][j] else 0.0)

        if self.metacognition:
            # second-order signal appended LAST so the base features keep
            # identical offsets whether or not this flag is on
            max_ent = np.log2(GRID_SIZE * GRID_SIZE)
            trend_norm = float(np.clip(self.qbelief.entropy_trend() / max_ent, -1.0, 1.0))
            obs.append(trend_norm)
            obs.append(self.qbelief.confidence())
        return obs

    def _check_objective_reduction(self, reward):
        """Entropy-threshold collapse: fires AT MOST ONCE per episode,
        independent of the agent's action or location -- unlike DECLARE,
        this isn't something the agent chooses to do. Returns the
        (possibly modified) reward. No-ops entirely when
        self.enable_collapse is False (the "Model A / conventional"
        ablation)."""
        if not self.enable_collapse:
            return reward
        if self.opened or self.or_triggered:
            return reward
        ent = self.qbelief.entropy()
        if ent >= OR_ENTROPY_THRESHOLD:
            return reward
        self.or_triggered = True
        gr, gc = self.qbelief.best_guess()
        self._log(f"objective reduction: entropy={ent:.2f} < threshold, auto-collapse check at ({gr},{gc})")
        if (gr, gc) == self.world.target_room:
            self.opened = True
            reward += 10.0
            self._log("collapse resolved onto the true target -- opened the chest on its own!")
        else:
            self.tried_combos.add((gr, gc))
            reward -= 0.5
            self._log("collapse resolved onto the wrong room -- ruled out, belief continues evolving")
        return reward

    def step(self, action):
        if self.opened:
            raise RuntimeError("step() called after the episode already succeeded; call reset()")

        self.steps += 1
        reward = -0.01
        r, c = self.agent_room

        if action in (UP, DOWN, LEFT, RIGHT):
            dr, dc = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}[action]
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                self.agent_room = (nr, nc)
                if not self.visited[nr][nc]:
                    reward = 0.05
                self.visited[nr][nc] = True
                self._noop_streak = 0
            else:
                self._noop_streak += 1
                reward = max(-2.0, -0.1 * self._noop_streak)

        elif action == SEARCH:
            grid = self._room_checked_grid(self.agent_room)
            unchecked = [(i, j) for i in range(SUB_SIZE) for j in range(SUB_SIZE) if not grid[i][j]]
            if not unchecked:
                self._noop_streak += 1
                reward = max(-1.0, -0.05 * self._noop_streak)
            else:
                self._noop_streak = 0
                i, j = unchecked[0]   # fixed scan order: one spot per SEARCH
                grid[i][j] = True
                clue = self.world.clue_at(self.agent_room, (i, j))
                if clue is not None:
                    # reading updates the quantum belief -- see
                    # quantum_mind.QuantumBelief.read for the interference
                    # / non-commuting-disturbance mechanics
                    self.qbelief.read(clue.kind, clue.value, clue.real, clue.id)
                    self._log(f"found a clue: {clue.expr} = {clue.value} ({clue.kind})")
                    reward = 1.0
                else:
                    reward = -0.02

        elif action == PICKUP:
            if self.inventory is not None:
                self._noop_streak += 1
                reward = max(-1.0, -0.05 * self._noop_streak)
            else:
                grid = self._room_checked_grid(self.agent_room)
                target = None
                for i in range(SUB_SIZE):
                    for j in range(SUB_SIZE):
                        if grid[i][j]:
                            clue = self.world.clue_at(self.agent_room, (i, j))
                            if clue is not None:
                                target = (clue, (i, j))
                                break
                    if target:
                        break
                if target is None:
                    self._noop_streak += 1
                    reward = max(-1.0, -0.05 * self._noop_streak)
                else:
                    clue, spot = target
                    self.world.remove_clue_at(self.agent_room, spot)
                    self.inventory = clue.id
                    self._log(f"picked up a clue ({clue.kind}) to carry it")
                    self._noop_streak = 0
                    reward = -0.01

        elif action == DROP:
            if self.inventory is None:
                self._noop_streak += 1
                reward = max(-1.0, -0.05 * self._noop_streak)
            else:
                grid_occupied = {
                    (i, j) for i in range(SUB_SIZE) for j in range(SUB_SIZE)
                    if self.world.clue_at(self.agent_room, (i, j)) is not None
                }
                dest = next(
                    ((i, j) for i in range(SUB_SIZE) for j in range(SUB_SIZE)
                     if (i, j) not in grid_occupied),
                    None,
                )
                if dest is None:
                    self._noop_streak += 1
                    reward = max(-1.0, -0.05 * self._noop_streak)
                else:
                    self.world.place_clue(self.inventory, self.agent_room, dest)
                    self._log("dropped the carried clue here")
                    self.inventory = None
                    self._noop_streak = 0
                    reward = -0.01

        elif action == DECLARE:
            if self.agent_room == self.world.target_room:
                self.opened = True
                reward = 10.0
                self._log("opened the chest!")
            else:
                self.failed_attempts += 1
                self.tried_combos.add((r, c))
                reward = -1.0
                self._log("declared here -- wrong room")

        elif action == SENSE:
            if self.sensed:
                self._noop_streak += 1
                reward = max(-1.0, -0.05 * self._noop_streak)
            else:
                self._noop_streak = 0
                outcome = self.world.entangled_pair.measure(self.agent_id)
                self.sensed = True
                self.sensed_half = outcome
                raw_name = "top half" if outcome == 0 else "bottom half"
                self._log(f"sensed the entangled pair: local read = {raw_name} "
                          f"(uninformative alone -- needs the other agent's read to mean anything)")
                reward = 0.02
                if self.world.entangled_pair.both_measured():
                    resolved = self.world.entangled_pair.resolved_top_half()
                    self._log(f"both agents have now sensed -- compared reads reveal target is "
                              f"in the {'top' if resolved else 'bottom'} half")
                    reward = 0.15

        reward = self._check_objective_reduction(reward)
        self.qbelief.record()  # log entropy every step, not just on reads

        return self._obs(), reward, self.done, {}

    @property
    def done(self):
        return self.opened or self.steps >= MAX_STEPS
