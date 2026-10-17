"""Source-bound eighteen-link finite policy and causal cart-offset selector.

The identity fixes candidate tables, phase timing, profile order and provenance.
This module performs no synthesis or random resets and grants no qualification.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from . import canonical18_controller, canonical_profile_policy
from .canonical_evaluation import pinned_file, read_pinned, require, validate_identity


def policy_parameters(*, controller_sha256, park_ticks, route_ticks,
                      cart_error_offsets_m=(0.,), design_metadata=None):
    """Concrete prospective candidate parameters; include these in its identity."""
    offsets = [float(value) for value in cart_error_offsets_m]
    require(type(park_ticks) is int and 0 <= park_ticks < 1500
            and type(route_ticks) is int and park_ticks + route_ticks == 1500,
            "eighteen profile must cover every remaining command")
    require(offsets and offsets[0] == 0. and len(set(offsets)) == len(offsets)
            and np.isfinite(offsets).all(), "finite distinct profiles with zero first required")
    require(design_metadata is None or isinstance(design_metadata, dict),
            "explicit design provenance dictionary required")
    return {
        "factory": "rage_mill.canonical18_profile_policy.controller_factory",
        "controller_sha256": controller_sha256,
        "park_ticks": park_ticks, "route_ticks": route_ticks,
        "state_coordinates": "native_relative_q19_v19",
        "align_launch_branches": False, "dot_scope": "float64", "post_route": "zero",
        "selection": "first_complete_in_rail_native_forecast_with_250_postcommand_upright_ticks",
        "selection_tick": park_ticks, "forecast_horizon_ticks": route_ticks,
        "cart_error_offsets_m": offsets,
        "maximum_forecast_commands": len(offsets) * route_ticks,
        "forecast_state": "current_observed_native_q_v_only",
        "forecast_reset_draws": 0, "physical_state_modifications": 0,
        "design_metadata": {} if design_metadata is None else design_metadata,
    }


def _tables(data, *, park_ticks, route_ticks):
    shapes = {
        "states": (route_ticks + 1, 38), "actions": (route_ticks,),
        "gains": (route_ticks, 38), "offsets": (route_ticks,),
        "balance_gain": (38,), "park_gain": (38,), "parking_target": (38,),
        "native_defects": (route_ticks, 38), "terminal_root": (38, 38), "park_ticks": (),
    }
    with np.load(io.BytesIO(data), allow_pickle=False) as saved:
        require(set(saved.files) == set(shapes), "exact eighteen table fields required")
        tables = {name: saved[name].copy() for name in saved.files}
    for name, shape in shapes.items():
        value = tables[name]
        require(value.shape == shape, f"eighteen table shape differs: {name}")
        if name == "park_ticks":
            require(value.dtype == np.dtype("int64") and value.item() == park_ticks,
                    "declared integer eighteen parking timing required")
        else:
            require(value.dtype == np.dtype("float64") and np.isfinite(value).all(),
                    f"finite binary64 eighteen table required: {name}")
    require(not np.any(tables["terminal_root"]), "eighteen finite zero terminal root required")
    require(np.all(np.abs(tables["actions"]) <= 1.), "bounded nominal normalized controls required")
    return tables


def controller_factory(identity):
    protocol = validate_identity(identity)
    require(protocol.n_links == 18, "eighteen-link identity required")
    parameters = identity["policy_parameters"]
    expected = policy_parameters(
        controller_sha256=identity["controller"]["sha256"],
        park_ticks=parameters.get("park_ticks"), route_ticks=parameters.get("route_ticks"),
        cart_error_offsets_m=parameters.get("cart_error_offsets_m", ()),
        design_metadata=parameters.get("design_metadata"),
    )
    require(parameters == expected, "complete prospective eighteen policy parameters required")
    sources = {row["path"]: row for row in identity["sources"]}
    for module in (canonical18_controller, canonical_profile_policy):
        path = Path(module.__file__).resolve()
        require(str(path) in sources and pinned_file(path) == sources[str(path)],
                "eighteen policy and native selector sources must be pinned")
    own_path = Path(__file__).resolve()
    require(str(own_path) in sources and pinned_file(own_path) == sources[str(own_path)],
            "eighteen factory source must be pinned")
    tables = _tables(read_pinned(identity["controller"]),
                     park_ticks=parameters["park_ticks"], route_ticks=parameters["route_ticks"])
    base = canonical18_controller.Canonical18FiniteTrajectoryPolicy(
        **{name: tables[name] for name in ("states", "actions", "gains", "offsets",
                                          "balance_gain", "park_gain", "parking_target")},
        park_ticks=parameters["park_ticks"],
    )
    return canonical_profile_policy.NativeCartOffsetSelector(
        base, protocol, parameters["cart_error_offsets_m"])
