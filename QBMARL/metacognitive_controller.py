"""
Metacognitive CONTROL layer -- distinct from the metacognitive
MONITORING signal in quantum_mind.py (entropy_trend / confidence).

Monitoring alone doesn't make an agent "reason about" its own
uncertainty. If confidence/entropy_trend are just two more numbers in an
observation vector, a DQN treats them exactly like agent_row or
visited_mask: inputs to pattern-match against, with no guarantee they
influence anything -- and in fact the scripted heuristic policy in
policies.py NEVER referenced them at all; it always walked straight to
the nearest untried (row, col) combo the instant one existed, regardless
of whether that combo was a near-certainty or a coin flip among five
equally-weighted guesses.

This module is the missing piece: a CONTROLLER that reads the belief
state's own confidence/trend and DELIBERATELY branches its decision
because of that signal -- overriding the base policy (heuristic or DQN)
rather than just being available for it to ignore.

--- Design note: why this OWNS movement while deferring, not just the
    final DECLARE step ---

Intercepting only the `DECLARE` action itself, and deferring to the base
policy for everything else (including movement toward the candidate
room), doesn't work: the scripted heuristic RE-DERIVES "walk toward the
best untried combo" on EVERY call as long as any untried combo exists --
it's the heuristic's top priority (see policies.py). So the moment this
controller redirected the agent toward unsearched territory instead, the
very next call to the base policy would immediately steer it right back
toward the combo room. That's a one-step tug-of-war, not a decision, and
testing on identical worlds showed exactly that: agents spending all 500
steps oscillating without ever finishing exploration, far below the bare
heuristic's solve rate.

This is avoided by having the controller decide UP FRONT, before ever consulting
the base policy, whether it's in a "should defer and explore" state. If
so, it computes its own exploration action directly (search current
room, or walk to the nearest unsearched one) and never asks the base
policy for movement at all while deferring -- there's nothing left for
the base policy to overrule. The base policy is only consulted once the
controller has decided NOT to defer (confidence sufficient, budget
forcing urgency, or no evidence left to gather), at which point it will
correctly walk to and declare the best candidate.

--- Two explicit regimes ---

  1. LOW CONFIDENCE, UNTRIED CANDIDATES EXIST, EVIDENCE STILL GATHERABLE.
     Defer committing to a candidate combo and keep exploring instead,
     for as long as remaining step budget reasonably allows (see
     `_urgency_adjusted_threshold` -- required confidence relaxes toward
     0 as the episode's step budget runs out, so the controller always
     eventually commits rather than holding out for confidence it may
     never reach).

  2. RISING UNCERTAINTY RIGHT AFTER A DISTURBANCE. If entropy_trend is
     sharply positive (belief just got shaken up by a cross-axis
     disturbance -- see quantum_mind.py) and confidence hasn't recovered
     yet, commitment is suppressed for one step in favor of continued
     search, rather than acting on a belief state mid-disturbance.

Both regimes are explicit `if` branches keyed on `env.qbelief.confidence()`
/ `entropy_trend()` -- a real, inspectable, testable control dependency,
not a hope that a neural net's weights happen to have picked up on an
extra input dimension. Every override is logged with the reason (see
`MetacognitiveController.overrides`), so it's directly auditable how
often and why the controller diverged from the base policy.
"""

from env import SEARCH, DECLARE, SENSE, MAX_STEPS, GRID_SIZE

# below this confidence, prefer to keep gathering evidence over guessing,
# provided there's still evidence left to gather
DEFER_CONFIDENCE = 0.55
# entropy_trend above this (bits/step, post-disturbance spike) is treated
# as "belief still unsettled, don't act on it yet"
UNSETTLED_TREND = 0.05
# stop deferring this many steps before the episode's hard limit, no
# matter what confidence says -- guarantees enough travel time left to
# actually reach and declare the best candidate before running out of
# steps entirely
SAFETY_MARGIN_STEPS = 40


class MetacognitiveController:
    """Wraps any base_policy callable(env, carry_state) -> action."""

    def __init__(self, base_policy, name="controller"):
        self.base_policy = base_policy
        self.name = name
        self.overrides = []  # (episode_step, reason) log, for auditing/tests

    def reset_log(self):
        self.overrides = []

    def _urgency_adjusted_threshold(self, env):
        """DEFER_CONFIDENCE with the full episode ahead, relaxing
        LINEARLY to 0 by (MAX_STEPS - SAFETY_MARGIN_STEPS) -- reasoning
        about whether to keep deliberating has to account for the cost of
        deliberating (including travel time back to the chosen room), or
        it isn't actually better than not reasoning at all."""
        usable_horizon = max(1, MAX_STEPS - SAFETY_MARGIN_STEPS)
        remaining_frac = max(0.0, 1.0 - env.steps / usable_horizon)
        return DEFER_CONFIDENCE * remaining_frac

    def _any_unsearched(self, env):
        return any(
            not env.fully_searched((rr, cc))
            for rr in range(GRID_SIZE) for cc in range(GRID_SIZE)
        )

    def _explore_action(self, env):
        """The controller's OWN exploration action while deferring --
        deliberately bypasses the base policy so it can't immediately
        overrule this with 'go declare the combo instead' (see module
        docstring)."""
        if not env.fully_searched(env.agent_room):
            return SEARCH
        from policies import nearest_unsearched, move_toward
        target = nearest_unsearched(env)
        if target is not None:
            r, c = env.agent_room
            action = move_toward(r, c, *target)
            if action is not None:
                return action
        return SEARCH  # nothing better to do; harmless no-op-ish fallback

    def act(self, env, carry_state):
        # hard safety valve: past the safety margin, never defer --
        # whatever's about to happen, happens without this controller
        # eating the last steps on its own deliberation
        if env.steps >= MAX_STEPS - SAFETY_MARGIN_STEPS:
            return self.base_policy(env, carry_state)

        # SENSE, carrying/PICKUP/DROP demo behavior, and plain exploration
        # (no untried combo yet at all) aren't confidence-gated decisions
        # -- let the base policy run those exactly as it would alone.
        if not env.sensed or env.inventory is not None:
            return self.base_policy(env, carry_state)

        combos = env.untried_combos()
        if not combos:
            return self.base_policy(env, carry_state)

        confidence = env.qbelief.confidence()
        trend = env.qbelief.entropy_trend()
        threshold = self._urgency_adjusted_threshold(env)

        # Regime 2: belief just got disturbed and hasn't settled
        if trend > UNSETTLED_TREND and confidence < DEFER_CONFIDENCE:
            self.overrides.append((
                env.steps,
                f"deferred: entropy_trend={trend:.3f} (unsettled), "
                f"confidence={confidence:.2f} -- exploring instead"
            ))
            return self._explore_action(env)

        # Regime 1: confidence hasn't reached the (urgency-adjusted)
        # threshold yet, and there's still evidence left to gather
        if confidence < threshold and self._any_unsearched(env):
            self.overrides.append((
                env.steps,
                f"deferred: confidence={confidence:.2f} < threshold={threshold:.2f} "
                f"-- unsearched rooms remain, exploring instead"
            ))
            return self._explore_action(env)

        # confident enough, or out of evidence to gather, or urgency has
        # taken over -- let the base policy commit
        return self.base_policy(env, carry_state)
