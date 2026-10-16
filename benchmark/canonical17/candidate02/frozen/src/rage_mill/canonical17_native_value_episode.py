"""Full hanging episode using an independently owned native-value suffix.

Parking uses the literal ordinary policy. At launch the observed cart position
fixes feedback coordinates; only an owned forecast clone is reset thereafter.
This fixed-profile policy has no shallow-copy launch selector.
"""
from __future__ import annotations

import time

import numpy as np

from .canonical16_affine_feedback import AffineFeedback
from .canonical17_controller import Canonical17FiniteTrajectoryPolicy
from .canonical17_native_value import Canonical17NativeValueSuffix, _immutable


class Canonical17NativeValueEpisodePolicy:
    @staticmethod
    def _fresh_telemetry():
        return dict(native_candidate_commands=0, active_commands=0, max_candidates_per_active_command=0,
                    duplicate_proposals=0, native_forecast_errors=0, baseline_delivery_agreements=0,
                    baseline_delivery_disagreements=0, selected_to_baseline_ratio_count=0,
                    selected_to_baseline_ratio_sum=0., selected_to_baseline_ratio_min=None,
                    selected_to_baseline_ratio_max=None, selected_to_baseline_ratio_mean=None,
                    native_select_CPU_seconds=0., native_select_wall_seconds=0., steps=[])

    def __init__(self, ordinary_policy, states, actions, feedback, *, active_route_start,
                 cart_error_offset_m=0.):
        if (ordinary_policy.next_tick != 0 or ordinary_policy.park_ticks != 850
                or len(ordinary_policy.actions) != 650 or type(active_route_start) is not int
                or active_route_start not in (0, 250) or isinstance(cart_error_offset_m, bool)
                or not np.isfinite(cart_error_offset_m)):
            raise ValueError('fresh850/650 ordinary policy and fixed native-value parameters required')
        self.ordinary_policy = ordinary_policy
        self.full_states, self.full_actions = _immutable(states), _immutable(actions)
        self.feedback = AffineFeedback(*(_immutable(value) for value in (
            feedback.gains, feedback.offsets, feedback.roots, feedback.residuals)))
        if (self.full_states.shape != (701, 36) or self.full_actions.shape != (700,)
                or self.feedback.gains.shape != (700, 36) or self.feedback.offsets.shape != (700,)
                or self.feedback.roots.shape != (701, 36, 36) or self.feedback.residuals.shape != (701, 36)
                or not all(np.isfinite(value).all() for value in (self.full_states, self.full_actions,
                    self.feedback.gains, self.feedback.offsets, self.feedback.roots, self.feedback.residuals))
                or np.any(np.abs(self.full_actions) > 1.)
                or not np.array_equal(self.full_actions, self.full_actions.astype(np.float32).astype(np.float64))
                or self.full_states[:651].tobytes() != ordinary_policy.states.tobytes()
                or self.full_actions[:650].tobytes() != ordinary_policy.actions.tobytes()
                or self.feedback.gains[:650].tobytes() != ordinary_policy.gains.tobytes()
                or self.feedback.offsets[:650].tobytes() != ordinary_policy.offsets.tobytes()):
            raise ValueError('finite full700 values and literal ordinary650 prefix required')
        self.active_route_start = active_route_start
        self.cart_error_offset_m = float(cart_error_offset_m)
        self.suffix = None
        self.last_decision = None
        self._pending_prediction = None
        self.telemetry = self._fresh_telemetry()
        # These reporting references never enter command selection. Each copied
        # policy owns its own dictionary and its own list of descendant reports.
        self.copy_telemetry_reports = []

    @property
    def park_ticks(self):
        return self.ordinary_policy.park_ticks

    @property
    def states(self):
        return self.ordinary_policy.states

    @property
    def actions(self):
        return self.ordinary_policy.actions

    @property
    def align_launch_branches(self):
        return self.ordinary_policy.align_launch_branches

    @property
    def dot_scope(self):
        return self.ordinary_policy.dot_scope

    @property
    def next_tick(self):
        return self.ordinary_policy.next_tick

    @next_tick.setter
    def next_tick(self, value):
        if type(value) is not int or not 0 <= value <= 1500:
            raise ValueError('valid independent ordinary command clock required')
        self.ordinary_policy.next_tick = value

    @property
    def launch_cart_offset(self):
        return self.ordinary_policy.launch_cart_offset

    @launch_cart_offset.setter
    def launch_cart_offset(self, value):
        if value is not None and not np.isfinite(value):
            raise ValueError('finite observed launch coordinate required')
        self.ordinary_policy.launch_cart_offset = value

    @property
    def launch_branch_offset(self):
        return self.ordinary_policy.launch_branch_offset

    @launch_branch_offset.setter
    def launch_branch_offset(self, value):
        if value is None:
            self.ordinary_policy.launch_branch_offset = None
            return
        value = _immutable(value)
        if value.shape != (17,) or not np.isfinite(value).all():
            raise ValueError('finite seventeen launch branch coordinates required')
        self.ordinary_policy.launch_branch_offset = value

    def __copy__(self):
        # The unchanged generic selector copies only at its launch seam. Reject
        # a later copy instead of implicitly sharing a mutable running tracker.
        if self.suffix is not None or self.next_tick > 850:
            raise ValueError('native-value policy copies are restricted to the prelaunch seam')
        ordinary = self.ordinary_policy
        fresh = Canonical17FiniteTrajectoryPolicy(
            **{name: getattr(ordinary, name) for name in ('states', 'actions', 'gains', 'offsets',
                'balance_gain', 'park_gain', 'parking_target')}, park_ticks=850)
        fresh.next_tick = ordinary.next_tick
        fresh.launch_cart_offset = ordinary.launch_cart_offset
        fresh.launch_branch_offset = (None if ordinary.launch_branch_offset is None
                                      else _immutable(ordinary.launch_branch_offset))
        clone = object.__new__(type(self))
        clone.ordinary_policy = fresh
        clone.full_states, clone.full_actions, clone.feedback = self.full_states, self.full_actions, self.feedback
        clone.active_route_start, clone.cart_error_offset_m = self.active_route_start, self.cart_error_offset_m
        clone.suffix = clone.last_decision = None
        clone._pending_prediction = None
        clone.telemetry = self._fresh_telemetry()
        clone.copy_telemetry_reports = []
        self.copy_telemetry_reports.append(clone.telemetry)
        return clone

    def __call__(self, state, tick):
        state = np.asarray(state, dtype=np.float64)
        if (state.shape != (36,) or not np.isfinite(state).all() or type(tick) is not int
                or tick != self.ordinary_policy.next_tick or not 0 <= tick < 1500):
            raise ValueError('one finite physical observed state per consecutive episode command required')
        previous_parity = None
        if self._pending_prediction is not None:
            previous_parity = bool(state.dtype == self._pending_prediction.dtype
                                   and state.shape == self._pending_prediction.shape
                                   and state.tobytes() == self._pending_prediction.tobytes())
            if not previous_parity:
                raise ValueError('previous selected native prediction differs from observed physical state')
        self.last_decision = None
        if tick < 850:
            return self.ordinary_policy(state, tick)
        if self.suffix is None:
            if self.ordinary_policy.launch_cart_offset is None:
                self.ordinary_policy.launch_cart_offset = state[0]-self.ordinary_policy.states[0, 0]
                if self.cart_error_offset_m != 0.:
                    self.ordinary_policy.launch_cart_offset += self.cart_error_offset_m
            if self.ordinary_policy.launch_branch_offset is None:
                self.ordinary_policy.launch_branch_offset = np.zeros(17)
            self.suffix = Canonical17NativeValueSuffix(
                self.ordinary_policy, self.full_states, self.full_actions, self.feedback,
                active_route_start=self.active_route_start,
                cart_offset=self.ordinary_policy.launch_cart_offset)
        before = len(self.suffix.tracker.ledger)
        started_cpu, started_wall = time.process_time(), time.monotonic()
        try:
            command = self.suffix(state, tick)
        finally:
            entries = self.suffix.tracker.ledger[before:]
            self.telemetry['native_candidate_commands'] = len(self.suffix.tracker.ledger)
            self.telemetry['native_forecast_errors'] += sum(entry['error'] is not None for entry in entries)
            self.telemetry['native_select_CPU_seconds'] += time.process_time()-started_cpu
            self.telemetry['native_select_wall_seconds'] += time.monotonic()-started_wall
        self.last_decision = self.suffix.last_decision
        if self.last_decision is not None:
            decision = self.last_decision
            report = self.telemetry
            report['active_commands'] += 1
            report['max_candidates_per_active_command'] = max(report['max_candidates_per_active_command'], decision.unique_forecasts)
            report['duplicate_proposals'] += decision.duplicate_proposals
            agreement = self.suffix.last_baseline_delivery_agreement
            report['baseline_delivery_agreements'] += int(agreement)
            report['baseline_delivery_disagreements'] += int(not agreement)
            ledger = self.suffix.tracker.ledger
            denominator = ledger[decision.baseline_index]['cost']
            report['steps'].append(dict(
                episode_tick=tick, route_tick=tick-850, selected_index=decision.selected_index,
                baseline_index=decision.baseline_index, unique_forecasts=decision.unique_forecasts,
                duplicate_proposals=decision.duplicate_proposals,
                selected_cost=ledger[decision.selected_index]['cost'], baseline_cost=denominator,
                baseline_delivery_agreement=agreement, previous_selected_state_byte_parity=previous_parity))
            self._pending_prediction = _immutable(decision.predicted_state)
            if denominator is not None and denominator > 0.:
                ratio = ledger[decision.selected_index]['cost']/denominator
                report['selected_to_baseline_ratio_count'] += 1
                report['selected_to_baseline_ratio_sum'] += ratio
                report['selected_to_baseline_ratio_min'] = ratio if report['selected_to_baseline_ratio_min'] is None else min(report['selected_to_baseline_ratio_min'], ratio)
                report['selected_to_baseline_ratio_max'] = ratio if report['selected_to_baseline_ratio_max'] is None else max(report['selected_to_baseline_ratio_max'], ratio)
                report['selected_to_baseline_ratio_mean'] = report['selected_to_baseline_ratio_sum']/report['selected_to_baseline_ratio_count']
        return command
