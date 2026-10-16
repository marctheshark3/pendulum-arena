"""Factory for the unchanged, saved probe07 binary64 policy.

This loads fixed tables; it performs no planning, gain design, plant calls,
reset draws or admission decisions. A fresh instance owns each episode clock.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np

from . import canonical16_controller, canonical16_finite_controller
from .canonical_evaluation import read_pinned, require, validate_identity

CONTROLLER_SHA256 = "d71eec45605803395859c780a965497dcd2fbf78fb8857f908012a4bb5522507"
POLICY_PARAMETERS = {
    "factory": "rage_mill.canonical16_fixed_policy.controller_factory",
    "table": "exact_probe07_controller_npz",
    "park_ticks": 800,
    "align_launch_branches": False,
    "dot_scope": "float64",
    "post_route": "zero",
    "state_coordinates": "native_relative_q17_v17",
}
CORE_SHA256 = {
    "canonical16_controller.py": "4920ade40cdc34a8da3e36e39dd0dec5aa341aad3964672cacdc2bd3481b6312",
    "canonical16_finite_controller.py": "2dbeed94e48a539165462bc364628b947b0ab22adceb32cffba89d7a28cb8081",
}
SHAPES = {
    "states": (701, 34), "actions": (700,), "gains": (700, 34),
    "offsets": (700,), "balance_gain": (34,), "park_gain": (34,),
    "parking_target": (34,), "native_defects": (700, 34),
    "terminal_root": (34, 34), "park_ticks": (),
}


def _tables(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as saved:
        require(set(saved.files) == set(SHAPES), "exact fixed table fields required")
        tables = {name: saved[name].copy() for name in saved.files}
    for name, shape in SHAPES.items():
        value = tables[name]
        require(value.shape == shape, f"fixed table shape differs: {name}")
        if name == "park_ticks":
            require(value.dtype == np.dtype("int64") and value.item() == 800,
                    "literal 800-tick parking table required")
        else:
            require(value.dtype == np.dtype("float64") and np.isfinite(value).all(),
                    f"finite binary64 fixed table required: {name}")
    require(not np.any(tables["terminal_root"]), "saved zero terminal root required")
    require(np.all(np.abs(tables["actions"]) <= 1.), "bounded normalized nominal controls required")
    return tables


def controller_factory(identity):
    """Create the literal probe07 policy from a source-bound evaluation identity."""
    protocol = validate_identity(identity)
    require(protocol.n_links == 16 and identity["policy_parameters"] == POLICY_PARAMETERS,
            "literal sixteen-link fixed policy parameters required")
    row = identity["controller"]
    require(row["sha256"] == CONTROLLER_SHA256, "exact probe07 controller bytes required")
    sources = {source["path"]: source for source in identity["sources"]}
    for module in (canonical16_controller, canonical16_finite_controller):
        path = Path(module.__file__).resolve()
        require(str(path) in sources and sources[str(path)]["sha256"] == CORE_SHA256[path.name]
                and hashlib.sha256(path.read_bytes()).hexdigest() == CORE_SHA256[path.name],
                "unchanged numerical policy source required")
    require(str(Path(__file__).resolve()) in sources, "fixed factory source pin required")
    tables = _tables(read_pinned(row))
    return canonical16_finite_controller.Canonical16FiniteTrajectoryPolicy(
        **{key: tables[key] for key in ("states", "actions", "gains", "offsets",
                                       "balance_gain", "park_gain", "parking_target")},
        park_ticks=800, align_launch_branches=False, dot_scope="float64")
