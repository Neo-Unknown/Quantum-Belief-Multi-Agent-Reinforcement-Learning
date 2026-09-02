<div align="center">

# 🧠⚛️ Q-BMARL
### Quantum-Belief Multi-Agent RL Testbed & Metacognitive Control Ablation

*Two independent DQN agents hunt for a hidden target using belief states built from real quantum formalism — plus a controlled experiment testing whether an explicit metacognitive override controller actually changes learned behavior.*

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![QuTiP](https://img.shields.io/badge/quantum-QuTiP-9146FF)]()
[![PyTorch](https://img.shields.io/badge/RL-PyTorch%20DQN-EE4C2C)]()
[![Pygame](https://img.shields.io/badge/viz-pygame-00b894)]()
[![Status](https://img.shields.io/badge/ablation-3000%20episodes%20✓-success)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

</div>

---

## ⚡ TL;DR

> Across a **3000-episode, 3-model ablation**, an agent whose action selection is wrapped in an explicit **metacognitive override controller** significantly outperformed both a baseline and a passive-observation-only variant.

| Model | Success rate (last 500 ep) | 95% CI |
|---|---|---|
| **A** — no collapse, no metacognition | 3.8% | 2.8–5.2% |
| **B** — collapse + passive metacog. features | 7.8% | 6.3–9.6% |
| **C** — collapse + **explicit** metacog. controller | 🏆 **24.4%** | 21.8–27.2% |

`A vs C: p < 0.0001` &nbsp;•&nbsp; `B vs C: p < 0.0001` &nbsp;•&nbsp; `Cohen's h = 0.64 (medium)` — see [Final result](#-final-result-3000-episode-3-model-ablation) for the full breakdown, including *why* it wins, not just that it does.

---

## 📖 Table of contents

- [What this is](#-what-this-is)
- [Quick start](#-quick-start)
- [The world](#-the-world)
- [Why quantum formalism, not classical probability](#-why-belief-is-represented-with-quantum-formalism)
- [Metacognitive control layer](#-metacognitive-control-layer)
- [The ablation: Models A / B / C](#-the-ablation-models-a--b--c)
- [Final result](#-final-result-3000-episode-3-model-ablation)
- [Failure modes handled by design](#-failure-modes-handled-by-design)
- [Entanglement correctness](#-entanglement-correctness)
- [File guide](#-file-guide)
- [Known open items](#-known-open-items)

---

## 🔍 What this is

A multi-agent RL testbed where each agent's uncertainty about a hidden
target is represented with **real quantum formalism** — via [QuTiP](https://qutip.org/):
complex-amplitude kets, interference, entanglement through partial trace,
and von Neumann entropy — instead of a plain classical candidate list.

On top of that substrate sits a controlled ablation asking a sharper
question: does *representing* uncertainty this way, and *acting on it
explicitly* via a metacognitive controller, actually change learned RL
behavior? Both are tested, measured, and reported honestly below — including
where the result is more nuanced than the headline number suggests.

No LLM. No Ollama. No Node/React. Pure Python, fully offline, instant.

---

## 🚀 Quick start

```bash
pip install -r requirements.txt   # pygame, torch, numpy, qutip

python visualize.py               # watch two agents solve it live
```

| Key | Action |
|---|---|
| `SPACE` | pause / resume |
| `R` | new world |
| `↑` / `↓` / mouse wheel | scroll |
| `Esc` | quit |

<details>
<summary><b>Reproduce the full ablation from scratch</b></summary>

```bash
python train.py --episodes 3000                       # single-agent DQN training loop
python compare_models.py --episodes 3000               # 3-way A/B/C ablation, checkpointed
python analyze_compare.py compare_log.csv --tail 500    # stats: z-test, Cohen's h, Mann-Whitney
```

`compare_models.py` and `train.py` are both resumable (`--resume`) and
checkpoint every 50 episodes, so a killed run picks back up rather than
starting over.

</details>

<details>
<summary><b>Watch a specific policy</b></summary>

```bash
python visualize.py --policy heuristic        # scripted baseline, 100% reliable
python visualize.py --policy dqn              # trained DQN, greedy
python visualize.py --policy heuristic_meta   # heuristic + metacognitive controller
python visualize.py --policy dqn_meta         # DQN + metacognitive controller
```

</details>

---

## 🗺️ The world

```
        ┌─────────────────────────────┐
        │  5 × 5 rooms                │
        │  each a 3×3 searchable grid │  ← 225 total searchable spots
        │  1 hidden target room       │
        └─────────────────────────────┘
```

- **Clues:** exactly 1 real ROW clue + 1 real COLUMN clue (guaranteed
  solvable), plus 2–4 decoys reporting a wrong value on their axis. No
  visual tell between a real clue and a decoy.
- **`SEARCH`** reveals/reads a clue instantly, updating quantum belief
  (interference + cross-axis disturbance).
- **`PICKUP` / `DROP`** — optional: carry one clue at a time, relocate it
  in the shared world.
- **`SENSE`** — measure the agent's half of a shared entangled pair.
  Uninformative *alone*; becomes informative only once **both** agents
  have sensed and compared results.
- Two independent agents (heuristic or DQN), own belief state each, async
  move-timers, opposite starting corners — `(0,0)` and `(4,4)`.

---

## ⚛️ Why belief is represented with quantum formalism

A plain classical `row_candidates` list (every clue value ever seen,
equally weighted) can't produce any of the following. `quantum_mind.py` +
`entangled_pair.py` can, because belief is a genuine linear-algebra state:

| # | Property | What it means here |
|---|---|---|
| 1 | **Interference** | Each reading is a complex amplitude, not a weight. Real clues share phase 0; decoys get a phase derived from their own id. Two decoys on the same wrong value can *destructively cancel* instead of stacking. |
| 2 | **Non-commuting measurement** | Reading a row clue applies a small unitary "disturbance" to the col state, and vice versa. Row-then-col ≠ col-then-row — mirrors order effects in quantum-cognition models of human judgment. |
| 3 | **Entanglement** | A literal Bell-pair (`\|00⟩+\|11⟩` or `\|01⟩+\|10⟩`) shared by both agents, encoding "target in top or bottom half." Measured via a real partial trace (`Qobj.ptrace`). |
| 4 | **Entropy-threshold collapse** | Von Neumann entropy of the joint belief triggers an unprompted "objective reduction" once it drops far enough — resolution independent of the agent choosing to act. |

> A pure ket always has entropy exactly 0 — so entropy is computed on the
> **decohered** (measurement-basis) density matrix, the physically
> standard einselection move, or the collapse threshold would never fire.

---

## 🧭 Metacognitive control layer

Exposing `entropy_trend()` / `confidence()` as extra *observation*
dimensions is monitoring, not reasoning — a DQN can ignore an input
dimension as easily as use it, and the scripted heuristic never
references them at all.

**`metacognitive_controller.py`** is the missing piece: a controller that
reads the belief state's confidence/trend and **deliberately branches**
its decision — overriding the base policy (heuristic or DQN) rather than
leaving it to chance.

```
┌─────────────────────────────────────────────────────────────┐
│  MetacognitiveController.act()                              │
│                                                             │
│  low confidence + candidates untried + evidence gatherable  │
│      → DEFER, explore directly (bypasses base policy)       │
│                                                             │
│  entropy just spiked (post-disturbance, unsettled)          │
│      → DEFER, don't act on a belief mid-disturbance         │
│                                                             │
│  otherwise (confident / no evidence left / urgency)         │
│      → hand off to base policy, let it commit               │
└─────────────────────────────────────────────────────────────┘
```

Every override is logged with its reason (`.controller.overrides`) — an
auditable control dependency, not a hope the network noticed an extra
feature.

**Measured effect** (200 trials, identical worlds, bare heuristic vs.
controlled heuristic): both solve 200/200, but the controlled version
made **70% fewer failed `DECLARE`s** (4.42 → 1.31 avg) at the cost of
**44% more steps/episode** (150.5 → 217.5) — a real, quantified
deliberation-vs-mistakes trade-off.

---

## 🧪 The ablation: Models A / B / C

`env.py` exposes two flags — `enable_collapse`, `metacognition` — so the
same environment runs as three conditions, trained on **identical world
sequences** (same target, same clues, episode-for-episode — verified via
`matched_world_pair()`), so any measured difference comes from the
mechanism, not luck:

| Model | `enable_collapse` | `metacognition` | Explicit controller? |
|---|:---:|:---:|:---:|
| **A** — baseline | ❌ | ❌ | ❌ |
| **B** — passive observation | ✅ | ✅ (obs. features only) | ❌ |
| **C** — explicit control | ✅ | ✅ | ✅ `MetacognitiveController` |

B vs. C isolates whether an *explicit* override controller does more than
passively exposing the same signal as an observation feature. A vs.
(B or C) isolates whether collapse+metacognition change behavior at all.

---

## 🏆 Final result: 3000-episode, 3-model ablation

Two safeguards make this ablation meaningful rather than noise-dominated:

1. **Capped per-step no-op penalty** — uncapped, repeated wall-bumps can
   escalate a single episode to −300/−500+ via quadratic growth, swamping
   the actual signal. Capped at −1.0/−2.0 depending on action.
2. **`OR_ENTROPY_THRESHOLD = 2.4 bits`** — loose enough that the collapse
   mechanism actually fires on a meaningful fraction of episodes, so
   Model B is testing something real.

```
Success rate, final 500 episodes (1000 trials/model):
  A_no_collapse (baseline)                    3.8%   [95% CI 2.8–5.2%]
  B_collapse_metacog (passive features)        7.8%   [95% CI 6.3–9.6%]
  C_meta_controller (explicit override)       24.4%   [95% CI 21.8–27.2%]

Pairwise significance (two-proportion z-test):
  A vs C: p < 0.0001, Cohen's h = 0.64 (medium effect)
  B vs C: p < 0.0001, Cohen's h = 0.47 (small–medium effect)
  A vs B: p = 0.0001, Cohen's h = 0.17 (negligible effect)

Steps-to-solve (successful episodes): A=287, B=247, C=153
  (C significantly faster, Mann-Whitney p < 0.0001 vs both)
```

### Why C wins — mechanistically, not just numerically

A and B both **collapse in success rate right after epsilon (exploration)
hits its floor** — C doesn't:

```
Post-epsilon-floor collapse check:
  A_no_collapse        pre-floor=16.2%  post-floor=4.9%    ⚠️  >50% drop
  B_collapse_metacog    pre-floor=22.9%  post-floor=8.2%    ⚠️  >50% drop
  C_meta_controller      pre-floor=28.7%  post-floor=23.5%   ✅ no collapse
```

Consistent with the DQN converging to a degenerate, repetitive greedy
policy once exploration stops covering for it — a known DQN failure mode,
not something specific to quantum formalism. Model C's controller
overrides the base policy **~183 times/episode on average**, and that
frequent intervention appears to structurally prevent the agent from ever
settling into the trap.

> **Honest read:** the result is real and statistically solid, but it's
> partly "the mechanism does what we hoped" and partly "the controller
> rescues agents from an unrelated training-stability failure mode." Both
> are legitimate findings — just not quite the same claim. This is also a
> **single seed per condition**: a strong, replicable finding, not yet a
> generalized one. Multiple seeds, averaged, is the natural next step.

---

## 🛡️ Failure modes handled by design

| Failure mode | Why it's dangerous | Guard |
|---|---|---|
| Interference suppresses the *correct* answer | A value's amplitude can get destructively cancelled out of the top-k ranking entirely (seen live: seed 1009, true value at p=0.0066, excluded — agent exhausted all 500 steps) | `QuantumBelief.candidates()` keeps a raw "ever actually read" safety net beneath the quantum ranking — interference changes *try order*, never removes the answer from reach |
| Pure-state entropy always = 0 | A ket has one eigenvalue of 1 — `entropy_vn` on it directly never moves, so the collapse trigger never fires | Compute entropy on the **decohered** density matrix instead |
| A single agent's `SENSE` looks informative | By the no-signaling theorem, one agent's local half of an entangled pair is exactly 50/50 regardless of the encoded fact | Hint only usable once **both** agents have measured and results are compared |
| DQN diverges instead of converging | 500 `train_step()` calls/episode + an aggressive LR/target cadence (e.g. `lr=5e-4`, `target_update_every=500`) → climbing loss, degrading success over time | `lr=1e-4`, `batch_size=128`, `target_update_every=1000` — verified stable (loss flat 0.05–0.11) over a 200-episode diagnostic |

---

## 🔗 Entanglement correctness

`EntangledHalfBelief.measure()` physically collapses the pair for **both**
agents the instant *either* measures — that's what entanglement means.
But `both_measured()` / `resolved_top_half()` must gate on `measured_by`
(who has actually called `measure()`), never on `collapsed_value` having
two keys — otherwise a single agent's own measurement would incorrectly
look like "both agents compared results," which is physically wrong (a
lone local outcome is exactly 50/50 and carries zero information).

```text
After only Agent A measures:
  both_measured():      False   ✓ correct — B never measured
  resolved_top_half():  None    ✓ correct — no information leaked

After Agent B also measures:
  both_measured():      True
  resolved_top_half():  True    ✓ correct — matches the true target half
```

---

## 📁 File guide

| File | Purpose |
|---|---|
| `env.py` | `QuantumBeliefEnv` — world, clues, agent state, quantum belief integration, entropy-threshold collapse, `enable_collapse` / `metacognition` flags |
| `quantum_mind.py` | QuTiP belief state per agent/axis: interference, non-commuting disturbance, entropy, entropy-trend/confidence |
| `entangled_pair.py` | Shared Bell-pair-style resource between agents; `SENSE` via partial trace |
| `policies.py` | Scripted heuristic (100% reliable, tested) + DQN policy loader + metacognitive-wrapped variants |
| `dqn.py` | DQN brain: network, replay buffer, Double DQN training step, tuned hyperparameters |
| `metacognitive_controller.py` | Explicit control layer — branches on confidence/entropy_trend, wraps heuristic or DQN, logs every override |
| `train.py` | Trains two independent agents on shared worlds, checkpointed/resumable |
| `compare_models.py` | 3-way A/B/C ablation on matched world sequences, checkpointed/resumable |
| `analyze_compare.py` | Stats on `compare_log.csv`: pairwise z-tests, Cohen's h, Mann-Whitney, reward diagnostics, post-epsilon-floor collapse detection |
| `visualize.py` | Pygame viewer — scrollable surface, both agents on one map, per-agent room/belief/entropy panels |

> **Checkpoint compatibility:** observation vector is 45-dim, action space
> is 9 (`SENSE` included) — a checkpoint must have been trained against
> this exact shape to load. Retrain if in doubt.

---

## 🚧 Known open items

- `visualize.py` is full pygame rather than a simplified renderer.
- `MOVE_DELAY_A` / `MOVE_DELAY_B` are both `6` — the two agents don't yet
  visibly move on different cadences.

---

<div align="center">

*Built with QuTiP, PyTorch, and an unreasonable amount of care about not overselling a single-seed result.*

</div>
