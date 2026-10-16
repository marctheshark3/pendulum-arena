"""Choose one fixed feedback profile using a bounded native shadow rollout.

Only the observed launch state enters prediction. Shadow plants never receive
an episode seed or a future physical state, and cannot modify the live plant.
The first predicted complete, in-rail hold selects the remaining fixed policy.
"""
from __future__ import annotations

import copy

import numpy as np

from .canonical_benchmark import CanonicalEnv, score_tape


class NativeCartOffsetSelector:
    def __init__(self, policy, protocol, offsets):
        self.policy, self.protocol = policy, protocol
        self.offsets = tuple(float(value) for value in offsets)
        self.launch_tick = policy.park_ticks
        if (not self.offsets or self.offsets[0] != 0.
                or len(set(self.offsets)) != len(self.offsets)
                or not np.isfinite(self.offsets).all()
                or policy.align_launch_branches
                or self.launch_tick + len(policy.actions) != protocol.episode_ticks):
            raise ValueError("finite distinct profiles and a full remaining fixed route required")
        self.selected_offset = None
        self.selected_prediction = None
        self.forecasts = []

    def _select(self, state):
        # No reference to the physical environment is stored or accepted.
        # Each profile owns a separate controller clock and reset shadow data.
        env = CanonicalEnv(self.protocol)
        for offset in self.offsets:
            shadow = copy.copy(self.policy)
            shadow.launch_cart_offset = state[0] - shadow.states[0, 0]
            if offset != 0.:
                shadow.launch_cart_offset += offset
            shadow.launch_branch_offset = np.zeros(self.protocol.n_links)
            states, actions, raw_actions = [env.reset(state)], [], []
            invalid = None
            try:
                for tick in range(self.launch_tick, self.protocol.episode_ticks):
                    raw = shadow(states[-1].copy(), tick)
                    predicted, delivered = env.step(raw)
                    states.append(predicted)
                    actions.append(delivered)
                    raw_actions.append(raw)
                    if abs(predicted[0]) > self.protocol.rail_limit_m:
                        break
            except (ValueError, FloatingPointError) as error:
                invalid = f"{type(error).__name__}: {error}"
            score = score_tape(np.asarray(states), np.asarray(actions, dtype=np.float32),
                               self.protocol, invalid_reason=invalid)
            # The official scorer's 'complete' flag refers to all 1,500 commands;
            # this local forecast covers only commands not yet physically issued.
            passed = (len(actions) == self.protocol.episode_ticks - self.launch_tick
                      and invalid is None
                      and all(score[name] for name in
                              ("finite", "force_ok", "float32_delivery_ok", "rail_ok"))
                      and score["longest_upright_ticks"] >= self.protocol.hold_ticks)
            self.forecasts.append({"cart_error_offset_m": offset,
                                   "predicted_commands": len(actions),
                                   "local_score": score, "accepted": bool(passed)})
            if passed:
                self.selected_offset = offset
                self.selected_prediction = {
                    "states": np.asarray(states, dtype=np.float64),
                    "actions": np.asarray(actions, dtype=np.float32),
                    "raw_actions": np.asarray(raw_actions, dtype=np.float64),
                }
                self.policy.launch_cart_offset = state[0] - self.policy.states[0, 0]
                if offset != 0.:
                    self.policy.launch_cart_offset += offset
                self.policy.launch_branch_offset = np.zeros(self.protocol.n_links)
                return
        raise ValueError("no registered feedback profile passes the bounded native forecast")

    def __call__(self, state, tick):
        state = np.asarray(state, dtype=np.float64)
        if (state.shape != (2 * (self.protocol.n_links + 1),)
                or not np.isfinite(state).all() or type(tick) is not int
                or tick != self.policy.next_tick or not 0 <= tick < self.protocol.episode_ticks):
            raise ValueError("one finite observed state per consecutive physical command required")
        if tick == self.launch_tick:
            self._select(state.copy())
        return self.policy(state, tick)
