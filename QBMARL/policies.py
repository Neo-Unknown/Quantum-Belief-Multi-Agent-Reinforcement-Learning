"""
Scripted policy shared by visualize.py and any quick tests. Not a DQN --
this is the "always works, demonstrates every mechanic" baseline.

Strategy:
  1. If it's holding an untested (row, col) candidate combo, or one becomes
     available, walk to that room and DECLARE. Since every distinct
     reading is kept (env.py's aggregated belief), and the real clue on
     each axis is always discoverable, the correct combo is guaranteed to
     eventually be among the untried ones.
  2. Otherwise, keep exploring: search the current room fully (using
     env.fully_searched, NOT the coarser `visited` flag -- see the note
     below, this distinction matters) before moving to the next most
     promising unsearched room.
  3. Opportunistically demonstrates PICKUP/DROP: occasionally carries a
     found clue for a few steps then drops it, relocating it in the
     shared world, without that being required for solving.

IMPORTANT bug this avoids: picking the next room to explore by checking
`env.visited` (has this agent ever set foot here) rather than
`env.fully_searched` (has this agent actually checked every spot here) is
a trap -- an agent that only passes THROUGH a room (e.g. while carrying
something) marks it visited without searching it, and then permanently
skips it later even once it knows that room needs a second look. 

This policy always targets by search-completeness, not mere presence.
"""

import random

from env import (
    GRID_SIZE, SUB_SIZE,
    UP, DOWN, LEFT, RIGHT, SEARCH, PICKUP, DROP, DECLARE, SENSE,
)


def move_toward(r, c, gr, gc):
    dr = (gr > r) - (gr < r)
    dc = (gc > c) - (gc < c)
    if dr != 0:
        return UP if dr < 0 else DOWN
    if dc != 0:
        return LEFT if dc < 0 else RIGHT
    return None


def nearest_unsearched(env):
    """Nearest not-fully-searched room by Manhattan distance from the
    agent's current position.

    Breaking ties (more than one unsearched room at the same minimum
    distance) purely by scan order -- whichever (rr, cc) the double loop
    hits first, always top-to-bottom, left-to-right -- would be silent
    and wouldn't vary per agent or per call: two agents with identical
    distances to two candidate rooms would both walk toward the SAME one
    every single time, and a solo agent facing a repeated tie (common on
    a mostly-empty grid) would too, purely as an artifact of loop order
    rather than anything about the room itself.

    Ties are instead broken by the agent's OWN quantum belief: among the
    equally-close candidates, prefer whichever room env.qbelief currently
    rates as more probable to hold the target (env.belief_grid(), the
    Born-rule joint probability -- see quantum_mind.py). Since two agents
    accumulate different evidence from different clue reads, this alone
    already varies per agent. A tiny random jitter is added on top so an
    exact belief tie (e.g. right at episode start, before any clue has
    been read and every room is still uniform) doesn't fall back to scan
    order either.
    """
    r, c = env.agent_room
    grid = env.belief_grid()

    candidates = []
    best_dist = None
    for rr in range(GRID_SIZE):
        for cc in range(GRID_SIZE):
            if not env.fully_searched((rr, cc)):
                d = abs(rr - r) + abs(cc - c)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    candidates = [(rr, cc)]
                elif d == best_dist:
                    candidates.append((rr, cc))

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    def score(room):
        rr, cc = room
        # belief dominates; jitter only decides an exact belief tie
        return grid[rr][cc] + random.uniform(0.0, 1e-9)

    return max(candidates, key=score)


def nearest_combo(env, combos):
    r, c = env.agent_room
    best, best_dist = None, None
    for (gr, gc) in combos:
        d = abs(gr - r) + abs(gc - c)
        if best_dist is None or d < best_dist:
            best, best_dist = (gr, gc), d
    return best


def heuristic_action(env, carry_state):
    r, c = env.agent_room

    # priority 0: sense the entangled pair immediately -- it's free
    # (aside from one action's worth of time), usable only once, and
    # narrows every later candidate combo to half the map (see
    # env.untried_combos), so there's no reason not to do it first
    if not env.sensed:
        return SENSE

    # carrying something: demonstrate the relocate mechanic by holding it
    # a few steps then dropping it somewhere new, rather than forever
    if env.inventory is not None:
        carry_state["steps_carried"] = carry_state.get("steps_carried", 0) + 1
        if carry_state["steps_carried"] >= 4:
            carry_state["steps_carried"] = 0
            return DROP

    # priority 1: an untested candidate combo exists -- go test it. this is
    # what actually solves the puzzle (see module docstring)
    combos = env.untried_combos()
    if combos:
        target = nearest_combo(env, combos)
        if (r, c) == target:
            return DECLARE
        return move_toward(r, c, *target)

    # priority 2: keep exploring for more clues
    if not env.fully_searched(env.agent_room):
        return SEARCH

    # opportunistically pick up a freshly-found, still-present clue here
    if env.inventory is None and random.random() < 0.3:
        grid = env._room_checked_grid(env.agent_room)
        for i in range(SUB_SIZE):
            for j in range(SUB_SIZE):
                if grid[i][j] and env.world.clue_at(env.agent_room, (i, j)) is not None:
                    return PICKUP

    target = nearest_unsearched(env)
    if target is None:
        # every room fully searched and no untried combo -- nothing left to
        # do (shouldn't normally happen since real clues are guaranteed
        # discoverable); just hold position
        return SEARCH
    return move_toward(r, c, *target)


def make_dqn_policy(weights_path):
    from dqn import DQNAgent
    from env import OBS_DIM, NUM_ACTIONS
    agent = DQNAgent(OBS_DIM, NUM_ACTIONS)
    agent.load(weights_path)

    def policy(env, carry_state):
        return agent.act(env._obs(), greedy=True)
    return policy


def make_metacognitive_heuristic():
    """The scripted heuristic wrapped with the explicit metacognitive
    control layer (see metacognitive_controller.py). Returns a
    policy(env, carry_state) -> action callable, same interface as
    heuristic_action, so it's a drop-in replacement anywhere the bare
    heuristic is used (visualize.py's --policy flag, test harnesses,
    etc). The returned MetacognitiveController is attached as
    `.controller` on the callable so callers can inspect
    `.controller.overrides` for an audit log of every place this
    diverged from the bare heuristic and why."""
    from metacognitive_controller import MetacognitiveController
    controller = MetacognitiveController(heuristic_action, name="heuristic+metacognition")

    def policy(env, carry_state):
        return controller.act(env, carry_state)
    policy.controller = controller
    return policy


def make_metacognitive_dqn_policy(weights_path):
    """Same wrapping, but around a trained DQN's greedy policy instead of
    the scripted heuristic -- the controller can override a DQN action
    just as it overrides the heuristic's, since it only inspects
    env.qbelief and the action returned, not which policy produced it."""
    from metacognitive_controller import MetacognitiveController
    base = make_dqn_policy(weights_path)
    controller = MetacognitiveController(base, name="dqn+metacognition")

    def policy(env, carry_state):
        return controller.act(env, carry_state)
    policy.controller = controller
    return policy
