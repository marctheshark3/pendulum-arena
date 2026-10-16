"""Declare a native-value DEV owner with outer/live/inner costs separately capped."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rage_mill.canonical17_evidence import audit_owner
from rage_mill.canonical17_native_value_policy import policy_parameters
from rage_mill.canonical_evaluation import (
    canonical_json, pinned_file, read_pinned, require, validate_identity, write_json_exclusive,
)
from rage_mill.canonical_runtime_manifest import validate_runtime_manifest


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--mode", choices=("exact-calibration", "noisy20", "noisy100"), required=True)
    parser.add_argument("--cpu-seconds", type=int, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--calibration-owner", type=Path)
    parser.add_argument("--noisy20-owner", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    prepared = json.loads((args.candidate / "preparation.json").read_text())
    identity_row, runtime = prepared["identity"], prepared["runtime_closure"]
    identity = json.loads(read_pinned(identity_row))
    require(validate_identity(identity).n_links == 17, "seventeen native-value candidate required")
    expected = policy_parameters(controller_sha256=identity["controller"]["sha256"],
                                 design_metadata=identity["policy_parameters"]["design_metadata"])
    require(canonical_json(identity["policy_parameters"]) == canonical_json(expected),
            "literal native-value cost bounds and parameters required")
    require(args.cpu_seconds > 0 and args.wall_seconds > 0, "finite positive owner caps required")
    pins = validate_runtime_manifest(json.loads(read_pinned(runtime)))
    seen = set()

    def include(row):
        data = read_pinned(row)
        require(row["path"] not in pins or pins[row["path"]] == row["sha256"], "conflicting input pin")
        pins[row["path"]] = row["sha256"]
        if row["path"] in seen:
            return
        seen.add(row["path"])
        if Path(row["path"]).suffix == ".json":
            walk(json.loads(data))

    def walk(value):
        if isinstance(value, dict):
            if set(value) >= {"path", "sha256", "bytes"}:
                include(value)
            else:
                for path, digest in value.get("source_hashes", {}).items():
                    row = pinned_file(path)
                    require(row["sha256"] == digest, "predecessor input changed")
                    include(row)
                for item in value.values():
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for row in [identity_row, identity["controller"], identity["configuration"], *identity["sources"],
                prepared["supervisor"], pinned_file(__file__)]:
        include(row)
    declaration_row = identity["policy_parameters"]["design_metadata"]["declaration"]
    declaration = json.loads(read_pinned(declaration_row))
    out = args.out.resolve()
    worker = root / "scripts/evaluate_canonical17_profile_candidate.py"
    argv = [prepared["python_launcher"], str(worker), "--identity", identity_row["path"],
        "--expected-identity-sha256", identity_row["sha256"], "--runtime-closure", runtime["path"],
        "--declaration", declaration_row["path"], "--expected-declaration-sha256", declaration_row["sha256"],
        "--mode", args.mode, "--out", str(out / "result")]
    predecessors = []
    if args.mode != "exact-calibration":
        require(args.calibration_owner is not None, "naturally closed calibration owner required")
        checked = audit_owner(args.calibration_owner, identity=identity, runtime_closure=runtime,
                              seed_start=declaration["exact_calibration_seed"], episodes=1, exact=True)
        calibration = checked["source_calibration"]
        argv.extend(["--source-calibration", calibration["path"],
                     "--expected-source-calibration-sha256", calibration["sha256"]])
        predecessors.extend([calibration, checked["component"], checked["profile_decisions"], *checked["owner"].values()])
    if args.mode == "noisy100":
        require(args.noisy20_owner is not None, "naturally closed complete noisy20 owner required")
        checked = audit_owner(args.noisy20_owner, identity=identity, runtime_closure=runtime,
                              seed_start=declaration["development20"]["seed_start"], episodes=20, exact=False)
        cohort, component = checked["cohort"], checked["component"]
        argv.extend(["--noisy20-receipt", cohort["path"], "--expected-noisy20-receipt-sha256", cohort["sha256"],
                     "--noisy20-component-receipt", component["path"],
                     "--expected-noisy20-component-receipt-sha256", component["sha256"]])
        predecessors.extend([cohort, component, checked["profile_decisions"], *checked["owner"].values()])
    for row in predecessors:
        include(row)
    if args.mode == "exact-calibration":
        seed, count = declaration["exact_calibration_seed"], 1
    else:
        declared = declaration["development20" if args.mode == "noisy20" else "development100"]
        seed, count = declared["seed_start"], declared["episodes"]
    proof_commands = 3000 if args.mode == "exact-calibration" else 0
    out.mkdir(parents=True, exist_ok=False)
    recipe = {
        "schema": "rage_mill.next_stage_supervisor_recipe.v1", "argv": argv, "cwd": str(out),
        "environment": {"PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1",
                        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "CPU_cap_s": args.cpu_seconds, "wall_cap_s": args.wall_seconds, "address_cap_bytes": 8 * 1024**3,
        "output_path": str(out / "result"), "log": str(out / "worker.log"),
        "receipt": str(out / "completion.json"), "source_hashes": dict(sorted(pins.items())),
        "runtime_closure": runtime, "purpose": f"Own-count native-value {args.mode}; all failures retained",
        "seed_start": seed, "physical_episodes": count,
        "formal_development_episodes": count if args.mode != "exact-calibration" else 0,
        "maximum_saved_proof_transitions": proof_commands,
        "maximum_shadow_forecast_command_attempts": 650 * count,
        "maximum_outer_shadow_inner_native_candidates": 10400 * count,
        "maximum_live_inner_native_candidates": 10400 * count,
        "maximum_total_inner_native_candidates": 20800 * count,
        "maximum_actual_physical_commands": 1500 * count,
        "maximum_total_native_commands_including_proof": 22950 * count + proof_commands,
        "native_value_parameters": expected, "reserved_episodes": 0, "controller_qualified": False,
    }
    row = write_json_exclusive(out / "launch.json", recipe)
    print(json.dumps({"launch": row, "supervisor": prepared["supervisor"]}))


if __name__ == "__main__":
    main()
