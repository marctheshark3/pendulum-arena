#!/usr/bin/env python3
"""Independently check the actual frozen native-value candidate without physics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--expected-inputs-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.run and sha(args.inputs) == args.expected_inputs_sha256 and not args.out.exists(),
            "prospective metadata-only audit required")
    cpu, wall = time.process_time(), time.perf_counter()
    inputs = json.loads(args.inputs.read_text())
    require(all(sha(path) == digest for path,digest in inputs["source_hashes"].items()),
            "candidate input pin changed")
    import numpy as np
    from rage_mill import canonical18_native_value, canonical_profile_policy
    from rage_mill.canonical18_native_value_policy import controller_factory
    from rage_mill.canonical_evaluation import pinned_file, read_pinned, validate_identity
    from rage_mill.canonical_runtime_manifest import validate_runtime_manifest

    def forbidden(*_args, **_kwargs):
        raise ValueError("metadata construction attempted a native model or reset RNG")

    np.random.default_rng = forbidden
    canonical18_native_value.CanonicalEnv = forbidden
    canonical_profile_policy.CanonicalEnv = forbidden
    candidate = Path(inputs["candidate"])
    prepared = json.loads((candidate / "preparation.json").read_text())
    identity = json.loads(read_pinned(prepared["identity"]))
    require(prepared["identity"]["sha256"] == inputs["expected_identity_sha256"]
            and prepared["runtime_closure"]["sha256"] == inputs["expected_runtime_sha256"]
            and validate_identity(identity).n_links == 18,
            "actual candidate identity/count differs")
    runtime = json.loads(read_pinned(prepared["runtime_closure"]))
    pins = validate_runtime_manifest(runtime)
    require(len(pins) == prepared["runtime_files"] == inputs["expected_runtime_files"]
            and not any("site-packages/casadi/" in path or "tmp_casadi_compiler" in path for path in pins),
            "fresh own-checkout NumPy/MuJoCo closure differs")
    require(prepared["metadata_only"] is True and prepared["qualification_authority"] is False
            and all(type(prepared[key]) is int and prepared[key] == 0 for key in
                    ("model_constructions", "native_commands", "controller_calls", "reset_draws"))
            and prepared["runtime_closure"] in identity["sources"],
            "metadata scope or source-bound runtime differs")
    for row in (identity["controller"], identity["configuration"], *identity["sources"]):
        read_pinned(row)
    design = identity["policy_parameters"]["design_metadata"]
    with np.load(identity["controller"]["path"], allow_pickle=False) as saved:
        frozen = {key:saved[key].copy() for key in saved.files}
    def load(row):
        with np.load(row["path"], allow_pickle=False) as saved:
            return {key:saved[key].copy() for key in saved.files}
    ordinary = load(design["ordinary_source"])
    reference = load(design["full_reference_source"])
    feedback = load(design["full_affine_value_source"])
    native = load(design["verified_native_defects_source"])
    expected = {**ordinary, **{"value_"+key:reference[key] for key in ("states","actions","native_defects")},
                **{"value_"+key:feedback[key] for key in ("gains","offsets","roots","residuals","Q")}}
    require(set(frozen) == set(expected) and all(frozen[key].dtype == expected[key].dtype
            and frozen[key].shape == expected[key].shape and frozen[key].tobytes() == expected[key].tobytes()
            for key in expected), "frozen combined archive differs from literal full700 design bytes")
    require(native["native_defects"].tobytes() == frozen["value_native_defects"].tobytes(),
            "full700 independently verified native bias differs")
    parameters = identity["policy_parameters"]
    required = dict(park_ticks=850,route_ticks=650,full_value_horizon_ticks=700,selection_tick=850,
                    native_value_active_route_start=0,native_value_cart_error_offset_m=0.,
                    native_value_control_cost=1000.,maximum_forecast_commands=650,
                    maximum_total_inner_native_candidates=20800,maximum_native_commands_per_episode=22950)
    require(all(type(parameters[key]) is type(value) and parameters[key] == value
                for key,value in required.items()) and parameters["cart_error_offsets_m"] == [0.]
            and parameters["runtime_gain_synthesis"] is False,
            "fixed zero-only native-value policy or total native bounds changed")
    selector = controller_factory(identity)
    require(selector.policy.next_tick == 0 and selector.selected_offset is None
            and selector.selected_prediction is None and selector.forecasts == []
            and selector.policy.suffix is None and selector.policy.feedback.roots.shape == (701,38,38)
            and not selector.policy.feedback.roots.flags.writeable,
            "actual source-bound factory failed fresh immutable metadata construction")
    # Independent saved diagnostic checks: no reconstruction/inference/native map.
    diagnostic = json.loads(read_pinned(design["actual_diagnostic"]))
    evidence = design["actual_diagnostic_evidence"]
    require(diagnostic["schema"] == "rage_mill.canonical18_native_value_hanging_diagnostic.v1"
            and diagnostic["n_links"] == 18 and diagnostic["all_success"] is True
            and len(diagnostic["cases"]) == 1, "actual own18 diagnostic scope differs")
    tape = load(evidence["recording"])
    ledger = load(evidence["candidate_ledger"])
    case = diagnostic["cases"][0]
    require(case["seed"] == 3181040 and case["exact"] is True
            and tape["states"].shape == (1501,38) and tape["states"].dtype == np.dtype("float64")
            and tape["actions"].shape == (1500,) and tape["raw_actions"].shape == (1500,)
            and np.isfinite(tape["states"]).all() and np.isfinite(tape["raw_actions"]).all(),
            "complete original-path actual own18 diagnostic required")
    exact = np.zeros(38); exact[1] = np.pi
    require(tape["states"][0].tobytes() == exact.tobytes()
            and np.max(np.abs(tape["states"][:,0])) <= 3.
            and np.clip(tape["raw_actions"].astype(np.float32),-1.,1.).astype(np.float64).tobytes()
                == tape["actions"].tobytes(), "exact start, rail or float32 diagnostic delivery differs")
    angles = (np.cumsum(tape["states"][1:,1:19],axis=1)+np.pi) % (2*np.pi)-np.pi
    upright = np.all(np.abs(angles) < .15,axis=1)
    run=longest=0
    for value in upright:
        run=run+1 if value else 0
        longest=max(longest,run)
    require(longest == case["longest_upright_ticks"] == 402 and case["success"] is True,
            "independent strict post-command hold differs")
    counts=tape["unique_counts"]
    selected=tape["selected_indices"]
    baseline=tape["baseline_indices"]
    require(counts.shape == selected.shape == baseline.shape == (650,)
            and np.all((1 <= counts) & (counts <= 16))
            and int(counts.sum()) == len(ledger["commands"]) == case["native_candidate_commands"]
            and ledger["states"].shape == (len(ledger["commands"]),38)
            and ledger["commands"].dtype == np.dtype("float32")
            and np.all(ledger["errors"] == "") and np.isfinite(ledger["costs"]).all(),
            "bounded saved actual candidate ledger differs")
    require(tape["selected_episode_ticks"].tobytes() == np.arange(850,1500,dtype=np.int64).tobytes()
            and ledger["ticks"].tobytes() == np.repeat(np.arange(650,dtype=np.int64),counts).tobytes()
            and tape["selected_predicted_states"].tobytes() == tape["states"][851:].tobytes()
            and tape["selected_predicted_actions"].tobytes() == tape["actions"][850:].astype(np.float32).tobytes()
            and ledger["states"][selected].tobytes() == tape["selected_predicted_states"].tobytes()
            and ledger["commands"][selected].tobytes() == tape["selected_predicted_actions"].tobytes(),
            "all650 actual selected native state/action pairs differ")
    start=0
    for tick,count in enumerate(counts):
        stop=start+int(count)
        require(start <= int(selected[tick]) < stop and start <= int(baseline[tick]) < stop
                and int(selected[tick]) == start+int(np.argmin(ledger["costs"][start:stop])),
                "saved selection is not first minimum from its own candidate group")
        commands=ledger["commands"][start:stop]
        require(len({value.tobytes() for value in commands}) == len(commands),
                "duplicate delivered-float32 candidate exceeded unique-call accounting")
        start=stop
    completion=json.loads(read_pinned(evidence["completion"]))
    launch=json.loads(read_pinned(evidence["launch"]))
    require(completion["naturally_closed_zero"] is True and completion["recipe_unchanged"] is True
            and completion["recipe_sha256"] == evidence["launch"]["sha256"]
            and completion["owned_group_cleanup"]
            and all(completion["source_hashes_unchanged"].values())
            and all(sha(path)==digest for path,digest in launch["source_hashes"].items()),
            "actual diagnostic owner/source closure differs")
    coverage=json.loads(read_pinned(evidence["runtime_coverage"]))
    original_runtime=json.loads(read_pinned(evidence["diagnostic_runtime"]))
    original_pins=validate_runtime_manifest(original_runtime)
    require(coverage["uncovered_or_changed"] == {} and coverage["all_runtime_pins_unchanged"] is True
            and all(original_pins.get(path)==digest and sha(path)==digest
                    for path,digest in coverage["actual_loaded_file_sha256"].items()),
            "actual diagnostic runtime coverage differs")
    # Identity paths are bound to this live checkout and its own new runtime.
    checkout=Path(inputs["checkout"]).resolve(strict=True)
    for source in identity["sources"]:
        path=Path(source["path"])
        if "/src/rage_mill/" in str(path) or "/scripts/" in str(path):
            require(path.is_relative_to(checkout), "candidate imports an original frozen checkout source")
    require(prepared["python_launcher"] == str(checkout/".venv/bin/python")
            and identity["runtime"]["executable"] == str((checkout/".venv/bin/python").resolve())
            and all(row["path"].startswith("/") for row in runtime["files"]),
            "original-path new18 runtime binding differs")
    # Rehash historical qualifications and permanent claims; execute neither.
    initial=json.loads(Path(inputs["initial_inventory"]).read_text())
    historical_claims=initial["historical_claim_files"]
    require(all(Path(row["path"]).stat().st_size==row["bytes"] and sha(row["path"])==row["sha256"]
                for row in historical_claims), "historical permanent claim bytes changed")
    ledger_root=Path(initial["ledger"])
    require(not any(ledger_root.glob("**/count-18")), "eighteen reserved campaign activated before DEV")
    old=json.loads(Path(inputs["historical17_identity"]).read_text())
    require(sha(inputs["historical17_identity"]) == "61993d118c341001ef7973fce77a200f85dd86588b78919bc03d629e09c95268",
            "historical17 original identity changed")
    old_rows=[old["controller"],old["configuration"],*old["sources"]]
    require(all(sha(row["path"])==row["sha256"] for row in old_rows),
            "historical17 numerical/source/factory bindings changed")
    old_runtime_row=next(row for row in old["sources"] if Path(row["path"]).name=="runtime-closure.json")
    old_runtime_pins=validate_runtime_manifest(json.loads(Path(old_runtime_row["path"]).read_text()))
    # Metadata-only shallow copies have independent clocks and immutable tables.
    import copy
    clone=copy.copy(selector.policy)
    clone.next_tick=850
    require(selector.policy.next_tick==0 and clone.next_tick==850
            and clone.ordinary_policy is not selector.policy.ordinary_policy
            and clone.telemetry is not selector.policy.telemetry
            and not clone.full_states.flags.writeable and not clone.full_actions.flags.writeable
            and not clone.feedback.roots.flags.writeable and not clone.feedback.residuals.flags.writeable,
            "shadow/live metadata copies share mutable clock or tables")
    require(all(sha(path) == digest for path,digest in inputs["source_hashes"].items()),
            "candidate metadata audit changed its frozen input")
    args.out.mkdir()
    result = dict(schema="rage_mill.canonical18_actual_candidate_metadata_independent_audit.v1",
        passed=True,identity=prepared["identity"],runtime_closure=prepared["runtime_closure"],runtime_files=len(pins),
        actual_frozen_full700_combined_archive_byte_exact=True,verified_native_bias700_byte_exact=True,
        complete_own_NumPy_MuJoCo_runtime=True,online_planning_library=False,
        actual_factory_fresh_clock_immutable_arrays_no_native_model_or_RNG=True,
        maximum_total_inner_native_candidates=20800,maximum_native_commands_per_actual_and_shadow_episode=22950,
        actual_diagnostic_complete1500_strict_hold_ticks=longest,
        actual_diagnostic_selected650_native_pairs_byte_exact=True,
        actual_diagnostic_candidate_first_minimum_unique_F32_checks=True,
        actual_diagnostic_runtime_coverage_pass=True,
        original_path_new_checkout_runtime_binding=True,
        historical17_source_numerical_identity_and_runtime_files_unchanged=len(old_runtime_pins),
        historical_claim_files_unchanged=len(historical_claims),
        count18_permanent_namespace_still_absent=True,
        independent_shadow_live_clocks_and_immutable_tables=True,
        source_hashes_unchanged=True,native_calls=0,controller_calls=0,inference_calls=0,reset_RNG_draws=0,
        new_physical_episodes=0,reserved_episodes=0,controller_qualified=False,
        CPU_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall)
    with (args.out / "receipt.json").open("x") as stream:
        json.dump(result,stream,indent=2);stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
