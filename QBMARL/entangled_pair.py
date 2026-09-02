"""
A genuinely entangled 2-qubit resource shared between Agent A and Agent B,
representing ONE coarse fact: "is the target room in the top half of the
map (rows 0-1) or the bottom half (rows 2-4)?"

This is deliberately separate from -- and stronger a claim than -- the
plain shared `World` object both agents already read from. Sharing a
mutable Python object is ordinary classical coupling: both agents just
see the same data. Genuine entanglement means something stricter:
neither agent's qubit has a definite value UNTIL one of them measures,
and that measurement instantaneously fixes what the OTHER agent's qubit
must read, without anything being computed or transmitted between them
at that moment.

We build this literally with QuTiP: a Bell state |00>+|11> (perfectly
correlated) if the target is in the top half, or |01>+|10> (perfectly
anti-correlated) if it's in the bottom half -- so which Bell state we
started in encodes the fact, invisibly, until measured. `measure()` uses
a real partial trace (`Qobj.ptrace`) to get each agent's local marginal
probabilities before sampling an outcome, exactly as a physical
measurement on one half of a Bell pair would be computed.

Each agent can measure this at most once per episode via the SENSE
action (see env.py) -- whichever agent senses first "uses up" the
entanglement for both.
"""

import numpy as np
import qutip as qt


class EntangledHalfBelief:
    def __init__(self, target_is_top_half: bool):
        b00 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
        b11 = qt.tensor(qt.basis(2, 1), qt.basis(2, 1))
        b01 = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
        b10 = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))

        self.same_parity = target_is_top_half
        self.state = (b00 + b11).unit() if target_is_top_half else (b01 + b10).unit()
        self.collapsed = False
        self.collapsed_value = {}  # agent_id -> 0 (top) / 1 (bottom)
        # WHICH agents have actually performed their own measurement --
        # separate from collapsed_value, which physically gets fixed for
        # BOTH sides the instant EITHER agent measures (that's what
        # entanglement means). both_measured() must gate on this set, not
        # on collapsed_value having two keys, or a single agent measuring
        # alone would incorrectly appear as "both compared results" --
        # exactly the no-signaling bug this class exists to avoid. Caught
        # by testing: a single-agent run without a second agent ever
        # calling measure('B') still had both_measured() return True and
        # resolved_top_half() return a real answer, which is physically
        # wrong -- no measurement by B ever happened.
        self.measured_by = set()

    def measure(self, agent_id):
        """Projective measurement of ONE agent's qubit via partial trace.
        Returns 0 (top half) or 1 (bottom half). Physically collapses the
        pair for BOTH agents immediately (that's what entanglement means
        -- the other agent's value becomes definite even before they've
        looked at it). But `measured_by` only grows when an agent
        actually calls this itself, and `both_measured()`/
        `resolved_top_half()` gate on THAT, not on collapsed_value having
        two entries -- otherwise a single agent measuring alone would
        incorrectly unlock information that requires comparing two
        independently-obtained results (see class docstring)."""
        self.measured_by.add(agent_id)

        if self.collapsed:
            return self.collapsed_value[agent_id]

        idx = 0 if agent_id == "A" else 1
        rho_local = qt.ket2dm(self.state).ptrace(idx)
        p0 = float(np.clip(rho_local[0, 0].real, 0.0, 1.0))
        p1 = float(np.clip(rho_local[1, 1].real, 0.0, 1.0))
        probs = np.array([p0, p1])
        probs = probs / probs.sum()
        outcome = int(np.random.choice([0, 1], p=probs))

        other_outcome = outcome if self.same_parity else 1 - outcome
        other_id = "B" if agent_id == "A" else "A"

        self.collapsed = True
        self.collapsed_value = {agent_id: outcome, other_id: other_outcome}

        # collapse the stored state too, for consistency if inspected again
        a_out = outcome if idx == 0 else other_outcome
        b_out = other_outcome if idx == 0 else outcome
        self.state = qt.tensor(qt.basis(2, a_out), qt.basis(2, b_out))

        return outcome

    def both_measured(self):
        return "A" in self.measured_by and "B" in self.measured_by

    def resolved_top_half(self):
        """The actual encoded fact, extracted the physically honest way:
        by comparing BOTH agents' individually-random local outcomes,
        not by peeking at self.same_parity directly.

        This matters: a single agent's own measurement result is exactly
        50/50 and carries zero information about the fact on its own
        (the no-signaling theorem -- local marginals of a maximally
        entangled pair never depend on which entangled state you're in).
        The correlation only becomes usable information once you compare
        two measurement outcomes against each other, which is why this
        method requires both() to have measured, and is the reason
        env.py doesn't let either agent use its own SENSE result alone
        to narrow the search -- only the compared result.
        """
        if not self.both_measured():
            return None
        return self.collapsed_value["A"] == self.collapsed_value["B"]
