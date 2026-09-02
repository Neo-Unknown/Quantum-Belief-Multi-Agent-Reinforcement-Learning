# Quantum-Belief Multi-Agent RL Testbed: Metacognitive Control Ablation

## What this project is

A multi-agent reinforcement learning testbed where each agent's
uncertainty about a hidden target is represented using real quantum
formalism (QuTiP: complex-amplitude kets, interference, entanglement via
partial trace, von Neumann entropy) instead of classical probability --
plus a controlled, 3-way ablation testing whether that representation,
and an explicit metacognitive override controller built on top of it,
change learned RL behavior.

**Confirmed result (see "Final result" section below):** across a
3000-episode controlled ablation, an agent with an explicit
metacognitive override controller significantly outperformed both a
baseline and a passive-metacognitive-observation variant -- a real,
measured, statistically significant finding about learned RL behavior.

---

## Entanglement correctness: why `both_measured()` is gated on `measured_by`

`entangled_pair.EntangledHalfBelief.measure()` physically collapses the
pair for BOTH agents the instant EITHER one measures -- that's what
entanglement means, and `collapsed_value` gets populated for both agent
IDs right away as a result. But `both_measured()` and
`resolved_top_half()` must NOT be computed from `collapsed_value` having
two keys, because that would make a single agent's own measurement look
like "both agents compared results," which is physically wrong: by the
no-signaling theorem, one agent's local outcome alone is exactly 50/50
and carries zero information about which entangled state was prepared.
That information only becomes usable once BOTH agents have genuinely
called `measure()` themselves and their two results are compared.

`measured_by` tracks that separately -- who has actually measured, not
who has been assigned a value by the other's collapse -- and
`both_measured()`/`resolved_top_half()` gate on it. Verified directly:

```
After only Agent A measures:
  both_measured(): False       <- correct, B never measured
  resolved_top_half(): None    <- correct, no information leaked

After Agent B also measures:
  both_measured(): True
  resolved_top_half(): True    <- correct, matches the true target half
```

```
Heuristic, 200 trials, both agents sharing a world:
  A: 200/200   B: 200/200   Both: 200/200
  Objective-reduction fired: 9/200

Metacognitive controller vs bare heuristic, 120 agent-episodes each:
  bare: 120/120   metacognitive: 120/120  (both 100%, unaffected)

compare_models.py: smoke-tested, runs clean.
```

Success rate is unaffected by entanglement correctness because the
puzzle is always solvable from the row/col clues alone -- the
entanglement mechanic is only ever a head start, not load-bearing. Its
actual job (letting the viewer see that one measurement is meaningless
and two, compared, genuinely aren't) depends on `measured_by` being
right.

**Known open items, not yet addressed:**
- `visualize.py` is full pygame rather than a simplified renderer.
- `MOVE_DELAY_A` / `MOVE_DELAY_B` are both `6` -- the two agents don't
  visibly move on different cadences.

---

## Run it

```bash
pip install -r requirements.txt   # pygame, torch, numpy, qutip
python visualize.py
```

Controls: `SPACE` pause/resume, `R` new world, mouse wheel or `UP`/`DOWN`
to scroll, `Esc`/close to quit.

No LLM, no Ollama, no Node/React -- pure Python, fully offline, instant.

## Why belief is represented with quantum formalism

Belief is `quantum_mind.QuantumBelief` -- a QuTiP complex-amplitude
state per agent per axis -- plus `entangled_pair.py`, rather than a
plain classical candidate list (every distinct clue value read
appended to a list, each treated as equally likely). That gives the
simulation four properties a plain candidate list structurally cannot
have:

1. **Interference.** Each clue reading contributes a complex amplitude,
   not a weight. Real clues share phase 0; decoys get a phase derived
   from their own id. Two decoys landing on the same wrong value can
   destructively cancel instead of just adding up. This is strong enough
   that a real clue's value can occasionally get suppressed to
   near-zero probability -- see "Failure modes handled by design"
   below.

2. **Non-commuting measurement.** Reading a row clue applies a small
   unitary "disturbance" rotation to the col belief, and vice versa
   (`quantum_mind._DISTURB_U`). Row-then-col and col-then-row give
   genuinely different final beliefs for the same two clues, unlike an
   order-independent list would -- mirroring the order effects used in
   quantum-cognition models of human judgment.

3. **Entanglement.** `entangled_pair.EntangledHalfBelief` is a literal
   Bell-pair-style QuTiP state (`|00>+|11>` or `|01>+|10>`) shared by both
   agents, encoding "is the target in the top or bottom half of the map."
   A `SENSE` action measures it via a real partial trace (`Qobj.ptrace`).
   A single agent's own SENSE result is exactly 50/50 and uninformative
   ALONE (no-signaling theorem -- local marginals of a maximally
   entangled pair carry zero information about which entangled state
   you're in). The hint only becomes usable once BOTH agents have sensed
   and their two results are compared (`resolved_top_half()`).

4. **Entropy-threshold ("objective reduction") collapse.** Each agent's
   joint belief has a von Neumann entropy (`QuantumBelief.entropy()`).
   Once it drops below `OR_ENTROPY_THRESHOLD`, an unprompted collapse
   fires on its own -- env.py auto-resolves the agent's best guess,
   whether or not the agent chose to act. This sits alongside the
   agent-initiated `DECLARE` action, not instead of it. A pure ket
   always has von Neumann entropy exactly 0 (one eigenvalue of 1), so
   entropy has to be computed on the *decohered* (measurement-basis)
   density matrix instead -- the physically standard einselection move
   -- or it would never move and the threshold would never fire.

## Metacognitive control layer

Exposing `entropy_trend()`/`confidence()` only as extra *observation*
dimensions is monitoring, not reasoning -- a DQN can just as easily
ignore an input dimension as use it, and the scripted heuristic never
references them at all; it always beelines for the nearest untried
candidate the instant one exists, confident guess or coin flip alike.

5. **Metacognitive self-monitoring** (`QuantumBelief.entropy_trend()` /
   `confidence()`). A second-order signal, distinct from the
   first-order belief state itself: not just "how uncertain am I right
   now" but "is my uncertainty rising or falling, and how much should I
   trust it." Exposed as extra observation dimensions, and read directly
   by `metacognitive_controller.py` to decide when to override the base
   policy.

6. **Model A / Model B / Model C ablation** (`compare_models.py`).
   `env.py` has two flags -- `enable_collapse` and `metacognition` -- so
   the environment can run as "Model A" (no threshold-collapse, no
   self-monitoring in the observation), "Model B" (both on, passive
   observation only), or "Model C" (both on, plus an explicit
   `MetacognitiveController`), trained on IDENTICAL world sequences
   (same target room, same clues, episode-for-episode -- verified
   directly, see `matched_world_pair()`) so any measured difference
   comes from the mechanism, not luck.

7. **`metacognitive_controller.py`: explicit control, not just
   monitoring.** `MetacognitiveController` wraps any base policy
   (heuristic or DQN) and DELIBERATELY branches behavior on
   `env.qbelief.confidence()` / `entropy_trend()`: defers committing to
   a low-confidence guess in favor of gathering more evidence (with the
   required confidence relaxing as the step budget runs low, so it
   always eventually commits), and suppresses acting on a belief state
   that just got disturbed and hasn't settled. Every override is logged
   with its reason (`.controller.overrides`) -- an auditable, inspectable
   control dependency, not a hope that a network happened to pick up on
   an extra feature. Selectable in the viewer via `--policy heuristic_meta`
   or `--policy dqn_meta`.

   **Verified effect (200 trials, identical worlds, bare heuristic vs.
   metacognitive-controlled heuristic):** both solve 200/200, but the
   controlled version made **70% fewer failed DECLARE attempts** (4.42 ->
   1.31 average) at the cost of **44% more steps per episode** (150.5 ->
   217.5) -- a genuine, measured trade-off: more deliberation, fewer
   costly mistakes. That's the actual evidence this is doing something,
   not just naming something.

   The controller decides UP FRONT whether to defer, and if so, owns
   movement/search itself end-to-end, rather than asking the base
   policy each step -- intercepting only the final `DECLARE` action and
   deferring to the base policy for movement doesn't work, because the
   scripted heuristic RE-DERIVES "walk toward the best untried combo" on
   every single call as long as one exists, so the instant the
   controller redirected toward unsearched territory, the very next
   base-policy call would steer the agent right back: a one-step
   tug-of-war, not a decision, that leaves episodes oscillating without
   finishing exploration. Deferral is also bounded by
   `_urgency_adjusted_threshold` (required confidence relaxes linearly
   to 0 as steps approach the limit) plus a hard safety margin, so the
   controller always stops deferring with enough steps left to reach and
   declare a room, rather than holding out for confidence forever.

This borrows QuTiP's linear algebra (kets, tensor products, partial
trace, von Neumann entropy) to reproduce the *mathematical structure*
-- interference, non-commuting measurement, entanglement,
entropy-threshold collapse -- that a plain classical probability list
cannot produce.

**Checkpoint compatibility:** the observation vector is 45-dimensional
(quantum belief features plus metacognition) and the action space is 9
(`SENSE` included), so a checkpoint has to have been trained against
this same observation/action shape to load -- shapes won't match
otherwise. Retrain (see below) if in doubt.

## Failure modes handled by design

1. **Interference could suppress the correct answer entirely.**
   Ranking candidates purely by probability isn't enough on its own: a
   value whose amplitude gets unlucky and destructively cancelled can
   drop out of the top-k list -- an actual failing trial (seed 1009,
   target `(1,0)`) had the true column value 0 at probability 0.0066,
   excluded from `ranked_candidates`, agent exhausted all 500 steps
   never trying it. **Guard:** `QuantumBelief.candidates()` keeps a raw
   "ever actually read" safety net underneath the quantum ranking --
   interference changes try *order*, never removes the correct answer
   from reach. Verified: 200/200 solved with the guard in place.

2. **Pure-state entropy is always exactly 0.** A ket has one eigenvalue
   of 1, so `entropy_vn` computed directly on it never moves, which
   would make the collapse threshold never fire. **Guard:** compute
   entropy on the decohered (measurement-basis) density matrix instead.

3. **A single agent's SENSE result is uninformative alone.** By the
   no-signaling theorem, one agent's local half of a maximally entangled
   pair is exactly 50/50 regardless of the encoded fact. Letting one
   agent's raw local bit narrow its own search would mislead it about
   half the time. **Guard:** the hint only becomes usable once both
   agents have measured and their results are compared (see the
   entanglement correctness section above).

4. **DQN diverging instead of converging.** With up to 500 env steps
   per episode and a `train_step()` call on every one, an aggressive
   learning rate and target-update cadence (e.g. `lr=5e-4` /
   `batch_size=64` / `target_update_every=500`) produces average
   training loss that climbs steadily rather than falling -- textbook
   Q-value divergence -- which shows up as success rates *degrading*
   the longer training runs. **Guard:** `lr=1e-4`, `batch_size=128`,
   `target_update_every=1000` (`dqn.py`'s current defaults), verified
   stable (loss flat at 0.05-0.11) over a 200-episode diagnostic.

## Test results (not claims -- actually run)

```
Heuristic policy, two agents sharing a world, 200 trials:
  Agent A: 200/200 (100%)
  Agent B: 200/200 (100%)
  Both succeeded: 200/200 (100%)
  Objective-reduction (entropy-threshold auto-collapse) fired at
    least once in 8/200 trials (4%)

Full pygame render pipeline, headless real draw calls: completed with no
exceptions, both agents' episodes resolved during the run.

matched_world_pair() sanity check: target room and full clue set
(kind/real/value for every clue) verified identical across 3 seeds.
```

## Final result: 3000-episode, 3-model ablation (confirmed, not projected)

Two safeguards make the 3000-episode ablation meaningful rather than
noise-dominated:

1. **A capped per-step no-op penalty** (`env.py`): without a cap,
   repeated wall-bumps/dead searches can escalate a single episode's
   reward to -300 to -500+ via quadratic growth, swamping the signal
   from the mechanism actually under test. The cap is max -1.0/-2.0
   depending on action type.
2. **`OR_ENTROPY_THRESHOLD` set to actually fire** (2.4 bits) -- too
   strict a threshold (e.g. 1.6) means the collapse mechanism fires in
   only a fraction of a percent of trials, so "Model B" would barely be
   testing the thing it claims to test.

With those in place, and Model C -- which wraps the DQN's action
selection in the real `MetacognitiveController` (explicit override, not
just passive observation features) -- included as the third condition,
the full 3000-episode run gives a real, statistically solid answer:

```
Success rate, final 500 episodes (1000 trials/model):
  A_no_collapse (baseline)                    3.8%   [95% CI 2.8-5.2%]
  B_collapse_metacog (passive features)        7.8%   [95% CI 6.3-9.6%]
  C_meta_controller (explicit override)       24.4%   [95% CI 21.8-27.2%]

Pairwise significance (two-proportion z-test):
  A vs C: p < 0.0001, Cohen's h = 0.64 (medium effect)
  B vs C: p < 0.0001, Cohen's h = 0.47 (small-medium effect)
  A vs B: p = 0.0001, Cohen's h = 0.17 (negligible effect)

Steps-to-solve (successful episodes): A=287, B=247, C=153 (C significantly
faster, Mann-Whitney p < 0.0001 vs both).
```

**Why C wins -- mechanistically, not just numerically.** Both A and B
show a severe collapse in success rate specifically AFTER epsilon
(exploration) reaches its floor:

```
Post-epsilon-floor collapse check:
  A_no_collapse       pre-floor=16.2%  post-floor=4.9%   (WARNING: >50% drop)
  B_collapse_metacog  pre-floor=22.9%  post-floor=8.2%   (WARNING: >50% drop)
  C_meta_controller    pre-floor=28.7%  post-floor=23.5%  (no collapse)
```

This is consistent with the DQN converging to a degenerate, repetitive
greedy policy once exploration stops covering for it -- a known DQN
failure mode, not something specific to quantum formalism or
metacognition. Model C's controller overrides the base policy an
average of ~183 times per episode (`avg_overrides/ep`), and that
frequent intervention appears to structurally prevent the agent from
ever settling into that trap.

**Honest interpretation, not oversold:** the result is real and
statistically solid, but it's partly "the mechanism does what we hoped"
and partly "the controller happens to rescue agents from an unrelated
training-stability failure mode the other two conditions are exposed
to." Both are legitimate findings; they're just not quite the same
claim. This is also a SINGLE SEED per condition -- a strong, promising,
replicable finding, not yet a generalized one. Multiple seeds per
condition, averaged, would be the next step toward a claim that doesn't
need that caveat.

Reproduce or extend:
```bash
python compare_models.py --episodes 3000
python analyze_compare.py compare_log.csv --tail 500
```

## What's in the world

- 5x5 rooms, each a 3x3 grid of searchable furniture (225 spots total).
- One hidden target room (the chest is wherever it is -- no separate
  visible landmark).
- 4-6 clues: exactly 1 real ROW clue + 1 real COLUMN clue (guaranteed, so
  it's always solvable), plus 2-4 decoys reporting a wrong value on
  whichever axis they're tagged with. No visual difference between a real
  clue and a decoy when you find it.
- SEARCH reveals and reads a clue immediately and reliably, updating the
  agent's quantum belief (interference + cross-axis disturbance).
  PICKUP/DROP is a separate, optional mechanic: an agent can carry only
  one physical clue at a time and relocate it elsewhere in the shared
  world -- which can help or hide it from the other agent.
- SENSE: measures the agent's half of the shared entangled pair.
  Uninformative alone; once both agents have sensed, comparing results
  reveals which half of the map the target is in.
- Two independent agents (heuristic or trained DQN), each with its own
  quantum belief state (and, if `metacognition=True`, a self-monitoring
  entropy-trend/confidence signal), on independent async move-timers
  (not alternating turns), starting from opposite corners of the shared
  map -- `(0,0)` for A, `(4,4)` for B, both hardcoded in `train.py` and
  `visualize.py`, not something training discovers.

## UI: why scrolling, not just a bigger window

Rather than trying to fit everything into a fixed window size (which
risks content getting silently cut off if the layout is even slightly
miscalculated), everything renders onto a large off-screen surface
(`CONTENT_W x CONTENT_H`, computed exactly from every panel's real
dimensions), and a smaller, fixed viewport window scrolls over it. This
makes "does it fit" a non-issue regardless of your actual screen
resolution -- verified with an explicit bounds assertion
(`PANEL_X + PANEL_W <= CONTENT_W`, etc.) before ever rendering a frame,
not just by eyeballing it.

## Files

| File | Purpose |
|---|---|
| `env.py` | Environment (`QuantumBeliefEnv`): world, clues, agent state, quantum belief integration, entropy-threshold collapse, `enable_collapse`/`metacognition` ablation flags. |
| `quantum_mind.py` | QuTiP belief state per agent/axis: interference, non-commuting disturbance, entropy, entropy-trend/confidence metacognition. |
| `entangled_pair.py` | Shared Bell-pair-style resource between the two agents; SENSE measurement via partial trace. |
| `policies.py` | Scripted heuristic policy (100% reliable, tested) + DQN policy loader + metacognitive-controller-wrapped variants. |
| `dqn.py` | DQN brain: network, replay buffer, Double DQN training step, tuned hyperparameters (see "DQN diverging instead of converging" above). |
| `train.py` | Trains two independent agents on shared worlds, checkpointed/resumable. |
| `compare_models.py` | 3-way ablation: Model A (baseline) vs Model B (passive metacognitive features) vs Model C (explicit metacognitive override controller) on matched world sequences, checkpointed/resumable. |
| `analyze_compare.py` | Statistical analysis of `compare_log.csv`: pairwise z-tests, Cohen's h, Mann-Whitney on steps/reward, reward-distribution diagnostics, post-epsilon-floor collapse detection. |
| `metacognitive_controller.py` | Explicit control layer: branches behavior on confidence/entropy_trend, wraps heuristic or DQN. Verified 70% fewer failed declares, 44% more steps (heuristic test); verified to prevent late-training collapse in the DQN ablation (see Final result). |
| `visualize.py` | Pygame viewer: scrollable content surface, both agents on one map, per-agent room/belief/entropy panels. `--policy {heuristic,dqn,heuristic_meta,dqn_meta}`. |
