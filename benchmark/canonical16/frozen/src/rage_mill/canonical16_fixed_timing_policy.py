"""Literal registered 17-second parking candidate with unchanged gain prefixes."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np

from . import canonical16_controller, canonical16_finite_controller, canonical16_fixed_policy
from .canonical_evaluation import read_pinned, require, validate_identity

CONTROLLER_SHA256 = "de5afbb5db053a5094e01bd913bf0f7d2f4b17abd1a2679d3821f341e60ffcdb"
BASE_FACTORY_SHA256 = "9078d177f5e8bc9e6b13119aa4c229f65973be407266cf8249880a564f62197b"
POLICY_PARAMETERS = {
    "factory": "rage_mill.canonical16_fixed_timing_policy.controller_factory",
    "table": "registered_shortest_all11_timing_17s_exact_prefix",
    "park_ticks": 850, "route_ticks": 650,
    "design_origin_controller_sha256": canonical16_fixed_policy.CONTROLLER_SHA256,
    "original_feedback_horizon_ticks": 700,
    "retained_feedback_prefix_ticks": 650,
    "terminal_root_provenance": "original700_step_zero_terminal_design_not_recomputed_at650",
    "align_launch_branches": False, "dot_scope": "float64", "post_route": "zero",
    "state_coordinates": "native_relative_q17_v17",
}


def controller_factory(identity):
    protocol = validate_identity(identity)
    require(protocol.n_links == 16 and identity["policy_parameters"] == POLICY_PARAMETERS,
            "literal seventeen-second fixed candidate parameters required")
    row = identity["controller"]
    require(row["sha256"] == CONTROLLER_SHA256, "exact selected17 table bytes required")
    sources = {source["path"]: source for source in identity["sources"]}
    modules = (canonical16_controller, canonical16_finite_controller, canonical16_fixed_policy)
    expected = {**canonical16_fixed_policy.CORE_SHA256,
                "canonical16_fixed_policy.py": BASE_FACTORY_SHA256}
    for module in modules:
        path = Path(module.__file__).resolve()
        require(str(path) in sources and sources[str(path)]["sha256"] == expected[path.name]
                and hashlib.sha256(path.read_bytes()).hexdigest() == expected[path.name],
                "unchanged arithmetic and base-source dependency required")
    require(str(Path(__file__).resolve()) in sources, "literal17 factory source pin required")
    with np.load(io.BytesIO(read_pinned(row)), allow_pickle=False) as saved:
        require(set(saved.files) == set(canonical16_fixed_policy.SHAPES), "exact fixed table fields required")
        tables = {name: saved[name].copy() for name in saved.files}
    shapes = {**canonical16_fixed_policy.SHAPES, "states": (651, 34), "actions": (650,),
              "gains": (650, 34), "offsets": (650,), "native_defects": (650, 34)}
    for name, shape in shapes.items():
        value = tables[name]
        require(value.shape == shape, f"selected17 table shape differs: {name}")
        if name == "park_ticks":
            require(value.dtype == np.dtype("int64") and value.item() == 850,
                    "literal850-tick parking required")
        else:
            require(value.dtype == np.dtype("float64") and np.isfinite(value).all(),
                    f"finite binary64 selected17 table required: {name}")
    require(not np.any(tables["terminal_root"])
            and np.all(np.abs(tables["actions"]) <= 1.), "unchanged terminal root and bounded actions required")
    return canonical16_finite_controller.Canonical16FiniteTrajectoryPolicy(
        **{name: tables[name] for name in ("states", "actions", "gains", "offsets",
                                          "balance_gain", "park_gain", "parking_target")},
        park_ticks=850, align_launch_branches=False, dot_scope="float64")
