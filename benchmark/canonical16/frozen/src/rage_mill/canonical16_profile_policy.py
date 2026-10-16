"""Registered finite profile selection around the unchanged 17-second table."""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import canonical16_fixed_timing_policy, canonical_profile_policy
from .canonical_evaluation import require, validate_identity

BASE_FACTORY_SHA256 = "23a3115abcb0d8ca929a6c966650a4c77913227444a2dfaf555f966247f65377"
OFFSETS = (0., 1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2)
POLICY_PARAMETERS = {
    **canonical16_fixed_timing_policy.POLICY_PARAMETERS,
    "factory": "rage_mill.canonical16_profile_policy.controller_factory",
    "base_factory": "rage_mill.canonical16_fixed_timing_policy.controller_factory",
    "selection": "first_complete_in_rail_native_forecast_with_250_postcommand_upright_ticks",
    "selection_tick": 850,
    "forecast_horizon_ticks": 650,
    "cart_error_offsets_m": list(OFFSETS),
    "maximum_forecast_commands": 7150,
    "forecast_state": "current_observed_native_q_v_only",
    "forecast_reset_draws": 0,
    "physical_state_modifications": 0,
}


def controller_factory(identity):
    protocol = validate_identity(identity)
    require(protocol.n_links == 16 and identity["policy_parameters"] == POLICY_PARAMETERS,
            "literal bounded sixteen-link profile parameters required")
    sources = {row["path"]: row for row in identity["sources"]}
    for module in (canonical16_fixed_timing_policy, canonical_profile_policy):
        path = Path(module.__file__).resolve()
        require(str(path) in sources, "profile selector and base factory must be frozen")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == sources[str(path)]["sha256"],
                "profile source bytes differ")
    require(sources[str(Path(canonical16_fixed_timing_policy.__file__).resolve())]["sha256"]
            == BASE_FACTORY_SHA256, "unchanged fixed17 factory source required")
    require(str(Path(__file__).resolve()) in sources, "literal profile factory source pin required")
    # Reuse the unchanged base loader's exact table/shape/core-source checks.
    # This local loading view does not replace the recorded campaign identity.
    loading_identity = {**identity,
                        "policy_parameters": canonical16_fixed_timing_policy.POLICY_PARAMETERS}
    base = canonical16_fixed_timing_policy.controller_factory(loading_identity)
    return canonical_profile_policy.NativeCartOffsetSelector(base, protocol, OFFSETS)
