"""Offline canonical16 route transfer and native-defect-aware feedback.

Route transfer creates optimizer initializers, never accepted physical tapes.
The square-root feedback arithmetic follows our independently authored
canonical15 implementation; execution and scoring use canonical_benchmark.
"""
from __future__ import annotations

import numpy as np

DONOR14_SHA256 = "6bc6d9ea44c5fb7eba0b5182aeaaa7cd391721d52301fefc157e5ca99111a34d"


def lift_material(states, target_links=16):
    """Interpolate absolute angle/rate at uniform material-cell centers.

    Cart position/rate remain exact. Relative angles are reconstructed by
    adjacent differences. Angles remain continuous throughout interpolation;
    wrapping would introduce artificial internal joints at the pi seam.
    """
    states = np.asarray(states, dtype=np.float64)
    if (states.ndim != 2 or states.shape[1] < 4 or states.shape[1] % 2
            or not np.isfinite(states).all() or type(target_links) is not int
            or not 1 <= target_links <= 20):
        raise ValueError("finite serial relative-coordinate states required")
    source_links = states.shape[1]//2-1
    old = (np.arange(source_links)+.5)/source_links
    new = (np.arange(target_links)+.5)/target_links
    out = np.zeros((len(states), 2*(target_links+1)))
    for source_start, target_start in ((0, 0), (source_links+1, target_links+1)):
        out[:, target_start] = states[:, source_start]
        absolute = np.cumsum(states[:, source_start+1:source_start+source_links+1], axis=1)
        interpolated = np.array([np.interp(new, old, row) for row in absolute])
        out[:, target_start+1:target_start+target_links+1] = np.diff(
            np.c_[np.zeros(len(states)), interpolated], axis=1)
    return out


def route_initializer(states, actions, *, ticks=700, route_ticks=450):
    """Retiming bracket for an offline guess, followed by literal zero tail."""
    states, actions = np.asarray(states), np.asarray(actions)
    if (states.ndim != 2 or states.shape[1] != 34 or actions.ndim != 1
            or len(states) != len(actions)+1 or not np.isfinite(states).all()
            or not np.isfinite(actions).all() or not 300 <= route_ticks <= 700
            or not route_ticks+250 <= ticks <= 1000):
        raise ValueError("canonical16 route and at least 250 tail commands required")
    clock = np.linspace(0., len(actions), route_ticks+1)
    old = np.arange(len(states))
    out_s = np.zeros((ticks+1, 34))
    out_s[:route_ticks+1] = np.array([np.interp(clock, old, states[:, j]) for j in range(34)]).T
    out_u = np.zeros(ticks)
    # Retiming changes the guess's velocity; force remains only a warm guess.
    out_s[:route_ticks+1, 17:] *= len(actions)/route_ticks
    out_u[:route_ticks] = np.interp(clock[:-1], np.arange(len(actions)), actions)
    out_s[0] = 0.; out_s[0, 1] = np.pi
    # A small terminal source residue is tapered inside the hold initializer.
    fade = min(25, ticks-route_ticks)
    for tick in range(1, fade):
        out_s[route_ticks+tick] = out_s[route_ticks]*(1-tick/fade)
    return out_s, out_u


def trajectory_feedback(states, actions, first, terminal_root, Q, *, control_cost=1000., defects=None):
    """Square-root affine Riccati recursion with actual native defect column."""
    from scipy.linalg import qr

    states, actions = np.asarray(states), np.asarray(actions)
    count, dimension = len(actions), states.shape[1]
    a, b = first.map(count)(states[:-1].T, actions.reshape(1, -1))
    a, b = np.asarray(a), np.asarray(b)
    qroot = np.linalg.cholesky(Q).T
    root = np.asarray(terminal_root).copy()
    residual = np.zeros(dimension)
    gains = np.empty((count, dimension))
    offsets = np.empty(count)
    defects = np.zeros((count, dimension)) if defects is None else np.asarray(defects)
    if defects.shape != (count, dimension) or not np.isfinite(defects).all():
        raise ValueError("one finite transition defect per reference command required")
    for tick in reversed(range(count)):
        aa, bb = a[:, tick*dimension:(tick+1)*dimension], b[:, tick:tick+1]
        stack = np.block([[np.sqrt(control_cost)*np.ones((1, 1)), np.zeros((1, dimension)), np.zeros((1, 1))],
                          [root@bb, root@aa, (root@defects[tick]+residual)[:, None]],
                          [np.zeros((dimension, 1)), qroot, np.zeros((dimension, 1))]])
        upper = qr(stack, mode="economic")[1]
        gains[tick] = upper[0, 1:dimension+1]/upper[0, 0]
        offsets[tick] = upper[0, dimension+1]/upper[0, 0]
        root = upper[1:dimension+1, 1:dimension+1]
        residual = upper[1:dimension+1, dimension+1]
    return gains, offsets
