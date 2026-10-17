"""Frozen own-count native-value tables behind the unchanged launch selector.

All route/value arrays are loaded bytes. No Casadi, synthesis or gain recursion
is used online. The single zero-offset outer forecast preserves complete
650-command shadow/live admission and tape parity.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from . import (canonical16_affine_feedback, canonical16_predictive_tracking,
               canonical18_controller, canonical18_native_value,
               canonical18_native_value_episode, canonical_profile_policy)
from .canonical_evaluation import canonical_json, pinned_file, read_pinned, require, validate_identity

ORDINARY_SHAPES = {
    "states": (651, 38), "actions": (650,), "gains": (650, 38), "offsets": (650,),
    "balance_gain": (38,), "park_gain": (38,), "parking_target": (38,),
    "native_defects": (650, 38), "terminal_root": (38, 38), "park_ticks": (),
}
VALUE_SHAPES = {
    "value_states": (701, 38), "value_actions": (700,), "value_gains": (700, 38),
    "value_offsets": (700,), "value_roots": (701, 38, 38),
    "value_residuals": (701, 38), "value_Q": (38, 38),
    "value_native_defects": (700, 38),
}
POLICY_MODULES = (canonical18_controller, canonical18_native_value,
                  canonical18_native_value_episode, canonical16_affine_feedback,
                  canonical16_predictive_tracking, canonical_profile_policy)


def policy_parameters(*, controller_sha256, design_metadata):
    require(isinstance(design_metadata, dict), "explicit native-value provenance required")
    return {
        "factory": "rage_mill.canonical18_native_value_policy.controller_factory",
        "controller_sha256": controller_sha256, "park_ticks": 850, "route_ticks": 650,
        "full_value_horizon_ticks": 700, "state_coordinates": "native_relative_q19_v19",
        "align_launch_branches": False, "dot_scope": "float64", "post_route": "zero",
        "selection": "first_complete_in_rail_native_forecast_with_250_postcommand_upright_ticks",
        "selection_tick": 850, "forecast_horizon_ticks": 650, "cart_error_offsets_m": [0.],
        "maximum_forecast_commands": 650,
        "forecast_state": "current_observed_native_q_v_only", "forecast_reset_draws": 0,
        "physical_state_modifications": 0, "native_value_active_route_start": 0,
        "native_value_cart_error_offset_m": 0., "native_value_control_cost": 1000.,
        "maximum_native_candidates_per_active_command": 16,
        "maximum_outer_shadow_native_candidates": 10400,
        "maximum_live_native_candidates": 10400,
        "maximum_total_inner_native_candidates": 20800,
        "maximum_native_commands_per_episode": 22950,
        "runtime_gain_synthesis": False, "design_metadata": design_metadata,
        "timing_scope": "50Hz simulated cadence; no20ms wall or hardware claim",
    }


def _tables(data):
    shapes = {**ORDINARY_SHAPES, **VALUE_SHAPES}
    with np.load(io.BytesIO(data), allow_pickle=False) as saved:
        require(set(saved.files) == set(shapes), "exact full native-value table fields required")
        tables = {name: saved[name].copy() for name in saved.files}
    for name, shape in shapes.items():
        value = tables[name]
        require(value.shape == shape, f"native-value table shape differs: {name}")
        if name == "park_ticks":
            require(value.dtype == np.dtype("int64") and value.item() == 850,
                    "literal850 parking scalar required")
        else:
            require(value.dtype == np.dtype("float64") and np.isfinite(value).all(),
                    f"finite binary64 native-value table required: {name}")
    for short, full in (("states", "value_states"), ("actions", "value_actions"),
                        ("gains", "value_gains"), ("offsets", "value_offsets"),
                        ("native_defects", "value_native_defects")):
        require(tables[short].tobytes() == tables[full][:len(tables[short])].tobytes(),
                f"ordinary650 table is not the literal full700 prefix: {short}")
    require(not np.any(tables["terminal_root"]) and not np.any(tables["value_roots"][-1])
            and not np.any(tables["value_residuals"][-1]), "zero-terminal complete affine value required")
    require(np.all(np.abs(tables["value_actions"]) <= 1.)
            and tables["value_actions"].tobytes()
            == tables["value_actions"].astype(np.float32).astype(np.float64).tobytes(),
            "full700 reference controls must be exact deliveredfloat32 values")
    count = 18; cumulative = np.tril(np.ones((count, count))); gram = cumulative.T @ cumulative
    q = np.eye(38); q[0, 0] = q[19, 19] = .1
    q[1:19, 1:19] += 100. * gram
    q[20:, 20:] = gram + .01 * np.eye(18)
    require(tables["value_Q"].tobytes() == q.tobytes(), "literal own18 state cost required")
    return tables


class NativeValueZeroOffsetSelector(canonical_profile_policy.NativeCartOffsetSelector):
    """Add reporting references after the generic zero-profile decision only."""
    def __init__(self, policy, protocol):
        super().__init__(policy, protocol, (0.,))

    def _select(self, state):
        before = len(self.policy.copy_telemetry_reports)
        try:
            return super()._select(state)
        finally:
            reports = self.policy.copy_telemetry_reports[before:]
            for forecast, report in zip(self.forecasts, reports):
                forecast["native_value_shadow_telemetry"] = report
                forecast["native_value_live_telemetry"] = self.policy.telemetry
                forecast["maximum_total_inner_native_candidates"] = 20800
                forecast["maximum_total_native_commands_including_physical"] = 22950


def controller_factory(identity):
    protocol = validate_identity(identity)
    require(protocol.n_links == 18, "own-count eighteen native-value identity required")
    expected = policy_parameters(controller_sha256=identity["controller"]["sha256"],
                                 design_metadata=identity["policy_parameters"].get("design_metadata"))
    require(canonical_json(identity["policy_parameters"]) == canonical_json(expected),
            "literal zero-offset activation0/R1000/cap16 native-value parameters required")
    sources = {row["path"]: row for row in identity["sources"]}
    for module in POLICY_MODULES:
        path = Path(module.__file__).resolve()
        require(str(path) in sources and sources[str(path)] == pinned_file(path),
                "all native-value arithmetic and selector source bytes must be frozen")
    own = Path(__file__).resolve()
    require(str(own) in sources and sources[str(own)] == pinned_file(own),
            "native-value factory source must be frozen")
    tables = _tables(read_pinned(identity["controller"]))
    ordinary = canonical18_controller.Canonical18FiniteTrajectoryPolicy(
        **{name: tables[name] for name in ("states", "actions", "gains", "offsets",
                                          "balance_gain", "park_gain", "parking_target")}, park_ticks=850)
    feedback = canonical16_affine_feedback.AffineFeedback(
        tables["value_gains"], tables["value_offsets"], tables["value_roots"], tables["value_residuals"])
    policy = canonical18_native_value_episode.Canonical18NativeValueEpisodePolicy(
        ordinary, tables["value_states"], tables["value_actions"], feedback,
        active_route_start=0, cart_error_offset_m=0.)
    return NativeValueZeroOffsetSelector(policy, protocol)
