"""Owned one-step native affine-value selection over unchanged eighteen gains.

This diagnostic wrapper owns its ordinary policy clock, native forecast clone
and candidate ledger. It does not implement shallow-copy profile selection.
The underlying historical tracker is reused without changing its arithmetic.
"""
from __future__ import annotations

import numpy as np

from .canonical_benchmark import CanonicalEnv, CanonicalProtocol
from .canonical16_affine_feedback import AffineFeedback
from .canonical16_predictive_tracking import NativeValueTracker


def _immutable(array):
    array = np.asarray(array, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


class Canonical18NativeValueSuffix:
    def __init__(self, ordinary_policy, states, actions, feedback, *, active_route_start,
                 cart_offset, before_forecast=None):
        if active_route_start not in (0, 250) or ordinary_policy.next_tick != 850:
            raise ValueError('registered activation and an independently owned launch clock required')
        if np.asarray(states).shape != (701, 38) or np.asarray(actions).shape != (700,):
            raise ValueError('full own-count700 reference and affine value required')
        self.ordinary_policy = ordinary_policy
        self.active_route_start = active_route_start
        self.forecast = CanonicalEnv(CanonicalProtocol(n_links=18))
        immutable_feedback = AffineFeedback(*(_immutable(value) for value in (
            feedback.gains, feedback.offsets, feedback.roots, feedback.residuals)))
        self.tracker = NativeValueTracker(_immutable(states), _immutable(actions), immutable_feedback,
                                         self.forecast, control_cost=1000., cart_offset=cart_offset,
                                         before_forecast=before_forecast)
        self.last_decision = None
        self.last_original_raw = None
        self.last_baseline_delivery_agreement = None

    def __call__(self, state, tick):
        original_raw = self.ordinary_policy(state, tick)
        self.last_original_raw = float(original_raw)
        self.last_decision = None
        self.last_baseline_delivery_agreement = None
        route_tick = tick-850
        if route_tick < self.active_route_start:
            return original_raw
        decision = self.tracker.select(state, route_tick)
        baseline = self.tracker.ledger[decision.baseline_index]['command']
        original_delivery = np.clip(np.float32(original_raw), -1., 1.)
        self.last_baseline_delivery_agreement = bool(
            np.float32(baseline).tobytes() == np.float32(original_delivery).tobytes())
        self.last_decision = decision
        return decision.action
