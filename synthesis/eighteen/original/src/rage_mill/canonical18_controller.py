"""Immutable binary64 finite-route feedback for the eighteen-link plant.

Each instance owns its command clock and launch coordinates. Arrays have
immutable byte backing so shallow forecast copies cannot alter shared tables.
Phase transitions change feedback coordinates only; physical states are read.
"""
from __future__ import annotations

import numpy as np


def _immutable(value):
    array = np.asarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


class Canonical18FiniteTrajectoryPolicy:
    def __init__(self, *, states, actions, gains, offsets, balance_gain,
                 park_gain, parking_target, park_ticks=0):
        arrays = {name: np.asarray(value, dtype=np.float64) for name, value in {
            "states": states, "actions": actions, "gains": gains, "offsets": offsets,
            "balance_gain": balance_gain, "park_gain": park_gain,
            "parking_target": parking_target,
        }.items()}
        count = len(arrays["actions"]) if arrays["actions"].ndim else 0
        shapes = {"states": (count + 1, 38), "actions": (count,),
                  "gains": (count, 38), "offsets": (count,),
                  "balance_gain": (38,), "park_gain": (38,), "parking_target": (38,)}
        if (count == 0 or type(park_ticks) is not int or park_ticks < 0
                or park_ticks + count > 1500
                or any(value.shape != shapes[name] or not np.isfinite(value).all()
                       for name, value in arrays.items())):
            raise ValueError("finite 38-state tables and canonical episode phase timing required")
        for name, value in arrays.items():
            setattr(self, name, _immutable(value))
        self.park_ticks = park_ticks
        self.align_launch_branches = False
        self.dot_scope = "float64"
        self.launch_cart_offset = None
        self.launch_branch_offset = None
        self.next_tick = 0

    def __call__(self, state, tick):
        state = np.asarray(state, dtype=np.float64)
        if (state.shape != (38,) or not np.isfinite(state).all() or type(tick) is not int
                or tick != self.next_tick or not 0 <= tick < 1500):
            raise ValueError("one finite physical state for each consecutive episode tick required")
        self.next_tick += 1
        if tick < self.park_ticks:
            return -float(np.dot(self.park_gain, state - self.parking_target))
        route_tick = tick - self.park_ticks
        if route_tick >= len(self.actions):
            return 0.
        if self.launch_cart_offset is None:
            self.launch_cart_offset = state[0] - self.states[0, 0]
            self.launch_branch_offset = np.zeros(18)
        centered = state.copy()
        centered[0] -= self.launch_cart_offset
        centered[1:19] -= self.launch_branch_offset
        return (self.actions[route_tick] - self.offsets[route_tick]
                - float(np.dot(self.gains[route_tick], centered - self.states[route_tick])))
