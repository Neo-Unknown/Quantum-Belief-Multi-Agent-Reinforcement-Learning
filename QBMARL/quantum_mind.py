"""
QuTiP-based quantum belief representation for one agent's row/col
uncertainty about the target room.

Belief is a genuine linear-algebra state, not a plain classical
`row_candidates` / `col_candidates` list (a "have I seen this value"
set), giving the simulation three properties classical
probability-by-itself cannot produce:

1. INTERFERENCE. Each clue reading contributes a complex amplitude, not
   just a weight, to its reported value's basis state. A real clue's
   phase is fixed (real clues are evidence of the same fact, so multiple
   would reinforce). A decoy's phase is derived from its own id and
   effectively "random". Two decoys that happen to report the same wrong
   value can partially CANCEL each other (destructive interference)
   instead of simply stacking the way classical evidence-counting would.
   This is the actual mathematical signature of "quantum" belief, as
   opposed to Bayesian bookkeeping dressed up in quantum language.

2. NON-COMMUTING MEASUREMENT. Reading a row clue applies a small unitary
   "disturbance" rotation to the col state, and vice versa. That makes
   row-then-col and col-then-row give DIFFERENT final beliefs for the
   same two clues -- mirroring the order-effects that quantum-cognition
   models use to explain real human judgment data (classical probability
   is always order-independent; this deliberately isn't).

3. ENTROPY-DRIVEN COLLAPSE THRESHOLD. `entropy()` reports the von Neumann
   entropy of the joint row (x) col state. `env.py` uses this to trigger
   an unprompted "objective reduction" event once uncertainty drops far
   enough, so that resolution happens once a threshold is crossed, not
   only because an agent decided to act (contrast with the old
   DECLARE-on-demand mechanic, which is still available separately as
   the agent-initiated route).

4. METACOGNITIVE SELF-MONITORING. `entropy_trend()` and `confidence()`
   track how the agent's OWN uncertainty is changing over time -- a
   second-order signal, distinct from the first-order belief state
   itself, that `metacognitive_controller.py` reads and branches on
   directly (see that module for how it's used to override a base
   policy's action).

This is a *representation* built from QuTiP's linear algebra machinery
(kets, tensor products, partial trace, von Neumann entropy) to give the
belief state real interference, non-commuting measurement, entanglement,
and entropy-threshold behavior that a plain classical probability list
cannot produce. There is no physical superposition here -- only
classical bits simulating what a quantum amplitude vector would do.
"""

import cmath
import hashlib

import numpy as np
import qutip as qt

DIM = 5  # GRID_SIZE -- one basis state per row/col value

READ_WEIGHT = 1.4       # how strongly one clue reading nudges an amplitude
DISTURB_ANGLE = 0.18    # strength of the non-commuting cross-axis kick
# Set low enough that collapse actually fires a meaningful fraction of
# episodes (rather than demanding near-total certainty before triggering,
# which would leave the collapse mechanism almost never exercised) but
# not so low that it fires on nearly every episode immediately. If
# measurement shows it's rarely firing, raise this; if it's firing too
# eagerly, lower it -- watch the printed collapse_fired rate.
OR_ENTROPY_THRESHOLD = 2.4  # joint entropy (bits) below which collapse fires


def _uniform_ket(dim=DIM):
    amps = np.ones(dim, dtype=complex) / np.sqrt(dim)
    return qt.Qobj(amps.reshape(dim, 1))


def _phase_for_clue(clue_id, real):
    """Real clues share phase 0 (they'd reinforce, if there were more than
    one). Decoys get a phase scattered around the circle, deterministic
    per clue id so runs stay reproducible, but uncorrelated across
    different decoys."""
    if real:
        return 0.0
    h = int(hashlib.sha256(str(clue_id).encode()).hexdigest(), 16)
    return (h % 360) * (np.pi / 180.0)


def _disturb_operator(dim=DIM, angle=DISTURB_ANGLE):
    """A small, fixed unitary 'kick': the back-action of measuring one
    axis on the other. Built from a fixed-seed Hermitian generator so
    exp(-i*angle*H) is norm-preserving and reproducible across runs."""
    rng = np.random.RandomState(1234)
    H = rng.normal(size=(dim, dim))
    H = (H + H.T) / 2.0
    return (-1j * angle * qt.Qobj(H)).expm()


_DISTURB_U = _disturb_operator()


class QuantumBelief:
    """One agent's quantum-flavored belief over (row, col)."""

    def __init__(self):
        self.row_state = _uniform_ket()
        self.col_state = _uniform_ket()
        self.reads = []  # (kind, value, real) log, for the UI/debugging
        # Raw "was this value ever actually read" record, independent of
        # amplitude. Interference is genuine here -- a value CAN get
        # destructively cancelled down to near-zero probability (that's
        # the whole point of modeling it with amplitudes instead of
        # counts). But that must not make the puzzle unsolvable: the
        # real clue's value has to remain reachable even if its
        # amplitude gets unlucky, so this raw set is a completeness
        # safety net underneath the quantum-weighted ranking (see
        # `candidates()` below) -- interference is allowed to change
        # TRY ORDER, never to make the correct answer permanently
        # invisible.
        self._seen = {"row": [], "col": []}
        # Metacognitive layer: a record of entropy OVER TIME, not just its
        # current value. First-order belief (the amplitude state) answers
        # "what do I currently think"; this answers "how is what I think
        # changing" -- a distinct, second-order signal. See
        # entropy_trend()/confidence() below.
        self._entropy_history = []

    def read(self, kind, value, real, clue_id):
        phase = _phase_for_clue(clue_id, real)
        contribution = READ_WEIGHT * cmath.exp(1j * phase)

        if kind == "row":
            amps = self.row_state.full().flatten()
            amps[value] += contribution
            self.row_state = qt.Qobj(amps.reshape(DIM, 1)).unit()
            self.col_state = (_DISTURB_U * self.col_state).unit()
        else:
            amps = self.col_state.full().flatten()
            amps[value] += contribution
            self.col_state = qt.Qobj(amps.reshape(DIM, 1)).unit()
            self.row_state = (_DISTURB_U * self.row_state).unit()

        if value not in self._seen[kind]:
            self._seen[kind].append(value)
        self.reads.append((kind, value, real))

    def probabilities(self, kind):
        state = self.row_state if kind == "row" else self.col_state
        amps = state.full().flatten()
        p = np.abs(amps) ** 2
        p = p / p.sum()
        return p

    def joint_grid(self):
        """Born-rule joint probability over (row, col), treating row/col
        as an (unentangled, for one agent alone) product state."""
        pr = self.probabilities("row")
        pc = self.probabilities("col")
        return np.outer(pr, pc)

    def entropy(self):
        """Von Neumann entropy (bits) of the DECOHERED joint state.

        A pure ket always has von Neumann entropy exactly 0 (it has one
        eigenvalue of 1) -- computing entropy_vn directly on
        ket2dm(row_state (x) col_state) would therefore be stuck at 0
        forever, regardless of how spread-out the amplitudes are, which
        is useless as an uncertainty signal.

        What we actually want is the uncertainty of the OUTCOME an agent
        would get if it measured right now -- i.e. treat the belief as
        already decohered into the room-value basis (the physically
        standard move: einselection/pointer-state decoherence turns a
        pure superposition into an effectively classical mixture over
        measurement outcomes once you ask "which outcome"). That mixture
        IS a genuine density matrix (diagonal, in this basis) and its von
        Neumann entropy equals the Shannon entropy of the Born-rule
        probabilities -- which is the quantity that should shrink as
        evidence concentrates belief onto one room, and is what we watch
        for the entropy-threshold collapse trigger.
        """
        grid = self.joint_grid().flatten()
        grid = grid / grid.sum()
        rho_diag = qt.Qobj(np.diag(grid))
        return float(qt.entropy_vn(rho_diag, base=2))

    def best_guess(self):
        grid = self.joint_grid()
        idx = int(np.argmax(grid))
        return divmod(idx, DIM)

    def ranked_candidates(self, kind, top_k=3):
        """Values ranked by probability, above the uniform baseline only.
        Can legitimately exclude a value interference has suppressed; use
        `candidates()` instead when you need a COMPLETE, guaranteed-
        solvable list ordered by preference."""
        p = self.probabilities(kind)
        baseline = 1.0 / DIM
        order = np.argsort(-p)
        return [int(v) for v in order if p[v] > baseline][:top_k]

    def candidates(self, kind, top_k=3):
        """Complete candidate list for this axis: quantum-ranked values
        first (best guesses to try first), followed by any other
        actually-read value not already included (the completeness
        safety net -- guarantees the real clue's value is never
        permanently unreachable just because interference suppressed its
        amplitude)."""
        ranked = self.ranked_candidates(kind, top_k=top_k)
        rest = [v for v in self._seen[kind] if v not in ranked]
        return ranked + rest

    def top_value(self, kind):
        p = self.probabilities(kind)
        return int(np.argmax(p))

    # ---------------- metacognitive layer ----------------
    # Everything above is FIRST-ORDER belief: "what room do I currently
    # think it is." Everything below is SECOND-ORDER: a signal tracking
    # how that first-order uncertainty is behaving over time, exposed so
    # a controller (or a DQN's observation vector) can condition on it
    # directly rather than only reacting to the raw belief state.

    def record(self):
        """Call once per environment step to log the current entropy.
        Must be called even on steps with no new clue reading, or the
        trend below can't distinguish 'genuinely stable' from 'never
        sampled'."""
        self._entropy_history.append(self.entropy())
        if len(self._entropy_history) > 50:
            self._entropy_history.pop(0)

    def entropy_trend(self, window=6):
        """Slope of entropy over the last `window` recordings (bits per
        step). Negative = uncertainty shrinking (converging toward an
        answer); positive = uncertainty growing (e.g. right after a
        disturbance from reading the other axis); ~0 = plateaued."""
        hist = self._entropy_history[-window:]
        if len(hist) < 2:
            return 0.0
        xs = np.arange(len(hist), dtype=float)
        slope, _ = np.polyfit(xs, np.array(hist), 1)
        return float(slope)

    def confidence(self):
        """A single scalar combining CURRENT uncertainty level with its
        RECENT TREND: low entropy that's still actively dropping counts
        as more confident than the same low entropy sitting flat, because
        the former means evidence is still actively resolving toward an
        answer. This is the self-monitoring signal exposed to the agent
        (see env.py's `metacognition` flag) and to the heuristic policy's
        declare-timing decision."""
        max_ent = np.log2(DIM * DIM)
        level_term = 1.0 - min(self.entropy() / max_ent, 1.0)
        trend = self.entropy_trend()
        falling_term = float(np.clip(-trend, 0.0, 1.0))
        conf = 0.7 * level_term + 0.3 * falling_term
        return float(np.clip(conf, 0.0, 1.0))
