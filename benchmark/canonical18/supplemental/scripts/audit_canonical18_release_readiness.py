"""Independently review saved own18 prerequisites and an original-path freeze.

This metadata-only worker reads actual tapes, command journals, complete saved
forecasts, telemetry and all native replay arrays. It executes no model,
controller, reset RNG or reserved activation. Its own runtime closure never
changes the candidate identity. A second phase also checks the source bundle,
clean-clone inventory and every identity-to-snapshot mapping.
"""
from __future__ import annotations

import argparse
import importlib
import io
import json
from pathlib import Path
import re
import runpy
import sys
import time

import numpy as np

from rage_mill.canonical_evaluation import (
    pinned_file, read_pinned, require, reserved_gates, validate_identity,
    value_sha256, write_json_exclusive,
)
from rage_mill.canonical_release import audit_source_snapshot
from rage_mill.canonical_replay import audit_native_replay_proof
from rage_mill.canonical_runtime_closure import runtime_binding_paths
from rage_mill.canonical_runtime_evidence import audit_formal_runtime_owners
from rage_mill.canonical_runtime_manifest import create_runtime_manifest, validate_runtime_manifest

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = Path("/home/marctheshark/.t3/artifacts/rage-mill-canonical-reserved-claims")


def read(row):
    return json.loads(read_pinned(row))


def closed_owner(directory):
    directory = Path(directory).resolve(strict=True)
    rows = {name: pinned_file(directory / file) for name, file in
            (("launch", "launch.json"), ("completion", "completion.json"), ("log", "worker.log"))}
    launch, completion = read(rows["launch"]), read(rows["completion"])
    require(completion.get("schema") == "rage_mill.next_stage_supervised_completion.v1"
        and completion.get("recipe_path") == rows["launch"]["path"]
        and completion.get("recipe_sha256") == rows["launch"]["sha256"]
        and type(completion.get("child_returncode")) is int and completion["child_returncode"] == 0
        and completion.get("naturally_closed_zero") is True and completion.get("failure") is None
        and completion.get("interrupted") is False and completion.get("wall_timeout") is False
        and completion.get("recipe_unchanged") is True
        and completion.get("owned_group_cleanup") == ["group_already_exited"]
        and completion.get("log_path") == rows["log"]["path"]
        and completion.get("log_sha256") == rows["log"]["sha256"],
        "original naturally closed/reaped immutable owner required")
    pins, unchanged = launch.get("source_hashes"), completion.get("source_hashes_unchanged")
    require(isinstance(pins, dict) and pins and isinstance(unchanged, dict)
        and set(pins) == set(unchanged) and all(value is True for value in unchanged.values())
        and all(pinned_file(path)["sha256"] == digest for path, digest in pins.items()),
        "all original prospective source/runtime inputs must still match")
    supervisors = [digest for path, digest in pins.items() if Path(path).name == "supervise-stage-v2.py"]
    require(len(supervisors) == 1 and completion.get("supervisor_source_sha256") == supervisors[0],
            "original completion must bind its prospectively pinned bounded supervisor")
    return directory, launch, rows


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def source_and_history(inputs, identity):
    source_review = read(inputs["source_review"])
    require(source_review.get("schema") == "rage_mill.canonical_reserved_source_review.v1"
        and source_review.get("passed") is True and source_review.get("activation_authorized") is False
        and source_review.get("identity") == inputs["identity"]
        and source_review.get("identity_sha256") == value_sha256(identity),
        "independent exact-candidate source-only review required")
    names = {Path(row["path"]).name for row in source_review["sources"]}
    require({"canonical_reserved.py", "canonical_evaluation.py", "canonical_release.py",
        "canonical_replay.py", "canonical_benchmark.py", "canonical_runtime_evidence.py",
        "canonical_runtime_manifest.py", "canonical_runtime_closure.py",
        "canonical18_native_value_policy.py", "canonical18_native_value_episode.py",
        "canonical18_native_value.py", "canonical18_controller.py", "canonical_profile_policy.py",
        "canonical16_affine_feedback.py", "canonical16_predictive_tracking.py",
        "evaluate_canonical18_profile_candidate.py", "audit_canonical18_saved_policy.py"} <= names,
        "reviewed generic/own-count causal runner source closure incomplete")
    require(all(row in identity["sources"] and pinned_file(row["path"]) == row
                for row in source_review["sources"]), "reviewed sources differ from candidate identity")
    _, _, tests = closed_owner(inputs["source_tests_owner"])
    require(re.search(r"\b131 passed\b", read_pinned(tests["log"]).decode()),
            "current full own18 131-test source owner required")
    inventory = read(inputs["initial_inventory"])
    require(inventory.get("ledger") == str(CLAIMS) and inventory.get("count18_namespace_exists") is False,
            "original before-development own-count ledger inventory required")
    historical = inventory["historical_claim_files"]
    require(historical and all(pinned_file(row["path"]) == row for row in historical),
            "historical sixteen/seventeen permanent claims changed")
    ledger = CLAIMS / ("published-" + identity["source_commit"]) / "count-18"
    require(not ledger.exists() and not ledger.is_symlink(),
            "count18 reserved namespace must remain untouched before activation")
    return dict(source_review=inputs["source_review"], current_source_tests=tests,
        source_tests_passed=131, historical_claim_files_verified=len(historical), count18_namespace_absent=True,
        superseded75_owner="Retained in costs/history; not used as a prerequisite because three unused sibling pins changed before candidate freeze.")


def native_owner(directory, *, original, identity_row, cohort_row, count):
    directory, launch, rows = closed_owner(directory)
    require(Path(launch["argv"][1]).name == "audit_canonical18_development_native.py"
        and launch.get("original_identity") == identity_row
        and launch.get("original_actual_owner") == str(Path(original).resolve(strict=True))
        and launch.get("original_saved_episodes") == count
        and launch.get("maximum_native_commands") == count*3000
        and launch.get("physical_episodes") == launch.get("policy_calls")
            == launch.get("reset_RNG_draws") == launch.get("recorded_reset_reconstructions")
            == launch.get("new_reserved_claims") == 0,
        "original own-count saved-native owner scope differs")
    output = directory / "output"
    require(launch.get("output_path") == str(output), "saved-native output escaped original owner")
    result_row = pinned_file(output / "receipt.json"); result = read(result_row)
    require(result.get("schema") == "rage_mill.canonical18_development_native_cohort_proof.v1"
        and result.get("passed") is True and result.get("identity") == identity_row
        and result.get("cohort") == cohort_row and result.get("episodes_proved") == count
        and result.get("all_continuous_and_cold_states_byte_exact") is True
        and result.get("maximum_dynamics_defect") == result.get("maximum_continuous_replay_error")
            == result.get("maximum_recorded_initial_replay_gap") == 0.
        and result.get("total_native_maps") == result.get("maximum_native_maps") == 3000*count
        and result.get("continuous_native_maps") == result.get("cold_native_maps") == 1500*count,
        "complete own-count saved-native cohort proof required")
    actual_cohort = read(cohort_row)
    actual_rows = actual_cohort["episode_receipts"]
    require(result.get("seed_start") == actual_cohort["seed_start"]
        and all(type(result.get(key)) is int and result[key] == 0 for key in
            ("physical_episodes", "new_reserved_claims", "reserved_episodes", "policy_calls",
             "reset_RNG_draws", "recorded_reset_reconstructions")),
        "saved-native proof must preserve the original cohort and zero-new-attempt scope")
    proofs = result.get("episode_proofs")
    require(isinstance(proofs, list) and len(proofs) == count, "one saved native proof per actual episode required")
    for index, proof in enumerate(proofs):
        episode = actual_rows[index]
        require(proof["episode"] == episode and proof["seed"] == result["seed_start"]+index
            and proof["native_maps"] == 3000, "saved proof original episode association differs")
        summary = audit_native_replay_proof(proof["native_proof"], episode_receipt=episode)
        native = read(proof["native_proof"])
        with np.load(io.BytesIO(read_pinned(read(episode)["recording"])), allow_pickle=False) as actual:
            with np.load(io.BytesIO(read_pinned(native["proof_arrays"])), allow_pickle=False) as replay:
                require(same(replay["continuous_states"], actual["states"])
                    and same(replay["cold_next_states"], actual["states"][1:]),
                    "every cold/continuous saved native state byte must match original actual tape")
        require(summary == proof["summary"] and summary["continuous_replay_identical"] is True
            and summary["maximum_dynamics_defect"] == summary["maximum_continuous_replay_error"] == 0.
            and native["continuous_transitions"] == native["one_step_transitions"] == 1500,
            "complete zero-defect saved-native summary differs")
    runtime = result["runtime_closure"]
    require(runtime == launch["runtime_closure"], "saved-native runtime manifest lineage differs")
    runtime_pins = validate_runtime_manifest(read(runtime))
    coverage = read(result["runtime_coverage"])
    actual = coverage.get("actual_loaded_file_sha256")
    require(coverage.get("runtime_closure") == runtime and coverage.get("original_identity") == identity_row
        and coverage.get("runtime_complete") is True and coverage.get("uncovered_or_changed") == {}
        and isinstance(actual, dict) and actual
        and all(runtime_pins.get(path) == digest and pinned_file(path)["sha256"] == digest
                for path, digest in actual.items()), "saved-native actual runtime coverage incomplete")
    return dict(receipt=result_row, original_owner=rows, episodes=count,
                continuous_native_maps=1500*count, cold_native_maps=1500*count,
                every_native_state_byte_exact=True, runtime_covered=True)


def audit_prerequisites(inputs, identity):
    helper = runpy.run_path(str(ROOT / "scripts/audit_canonical18_development_telemetry.py"),
                            run_name="saved_readiness_helpers_only")
    evaluator_row = next(row for row in identity["sources"]
        if Path(row["path"]).name == "evaluate_canonical18_profile_candidate.py")
    evaluator = runpy.run_path(evaluator_row["path"], run_name="saved_readiness_evaluator_metadata_only")
    modes = evaluator["development_modes"](read(identity["policy_parameters"]["design_metadata"]["declaration"]))
    require(modes == {"exact-calibration": (3187000, 1, True),
        "noisy20": (3188000, 20, False), "noisy100": (3188100, 100, False)},
        "candidate01 fresh prospective exact/DEV20/DEV100 allocation differs")
    prepared = read(inputs["candidate_preparation"])
    runtime = prepared["runtime_closure"]
    require(prepared["identity"] == inputs["identity"] and runtime in identity["sources"],
            "candidate preparation identity/runtime lineage differs")
    formal = {}
    for stage, directory in inputs["formal_owners"].items():
        launch = pinned_file(Path(directory) / "launch.json")
        result = helper["audit_owner"](directory, launch["sha256"], identity=identity,
            runtime=runtime, modes=modes, parity=evaluator["prediction_parity"])
        require(result["mode"] == stage, "formal stage owner association differs")
        formal[stage] = result
    require(set(formal) == set(modes), "exact calibration plus both fresh complete DEV cohorts required")
    natives = {}
    for stage, count in (("noisy20", 20), ("noisy100", 100)):
        natives[stage] = native_owner(inputs["native_owners"][stage],
            original=inputs["formal_owners"][stage], identity_row=inputs["identity"],
            cohort_row=formal[stage]["cohort"], count=count)
    return dict(formal=formal, development_native_proofs=natives,
        actual_episodes=121, fresh_development_episodes=120,
        development_continuous_native_maps=180000, development_cold_native_maps=180000,
        calibration_saved_native_maps=3000, every_actual_episode_strict_complete_pass=True,
        every_full_shadow_live_suffix_byte_exact=True, own_count_source_native_replay_complete=True)


def audit_freeze_metadata(inputs, identity, prerequisites):
    freeze = read(inputs["freeze"])
    require(freeze.get("schema") == "rage_mill.canonical_release_freeze.v1"
        and freeze.get("identity") == identity and freeze.get("identity_sha256") == value_sha256(identity)
        and freeze.get("activation_authorized") is False
        and freeze.get("reserved_gates") == reserved_gates(18)
        and freeze.get("requires_prerequisite_release") is False,
        "own-count original-path freeze and unchanged published reserved mapping required")
    formal = prerequisites["formal"]
    require(freeze["source_calibration"] == read(formal["exact-calibration"]["component"])["source_calibration"],
            "freeze exact calibration differs from the audited actual owner")
    for stage in ("noisy20", "noisy100"):
        require(freeze["development"][stage]["receipt"] == formal[stage]["cohort"]
            and freeze["development"][stage]["seed_start"] == formal[stage]["seed_start"],
            "freeze development cohort differs from audited actual owner")
    audit_formal_runtime_owners(freeze)
    audit_source_snapshot(freeze["source_snapshot"], identity)
    clone = Path(freeze["source_snapshot"]["fresh_clone"])
    mapping = read(pinned_file(clone / "CANONICAL18_IDENTITY_SOURCE_MAPPING.json"))
    require(mapping.get("identity") == inputs["identity"] and mapping.get("execution_paths_relocated") is False,
            "clean clone must remain source inventory, preserving original absolute execution identity")
    expected = [identity["controller"], identity["configuration"], *identity["sources"]]
    actual = mapping.get("mapping")
    require(isinstance(actual, list) and len(actual) == len(expected)
        and [item["original"] for item in actual] == expected,
        "complete ordered identity-to-snapshot mapping required")
    for item in actual:
        relative = Path(item["snapshot_relative_path"])
        require(not relative.is_absolute() and ".." not in relative.parts
            and pinned_file(clone / relative)["sha256"] == item["original"]["sha256"],
            "clean-clone mapped bytes differ from original execution source")
    reviews = freeze.get("independent_reviews")
    require(isinstance(reviews, list) and inputs["source_review"] in reviews
        and inputs["prerequisite_review"] in reviews,
        "actual freeze must preserve both independent source and full prerequisite reviews")
    prior = read(inputs["prerequisite_review"])
    require(prior.get("passed") is True and prior.get("phase") == "prerequisites"
        and prior.get("identity") == inputs["identity"], "positive exact-candidate prerequisite review required")
    return dict(freeze=inputs["freeze"], source_commit=freeze["source_snapshot"]["git_commit"],
        source_bundle=freeze["source_snapshot"]["bundle"], clean_clone_inventory_files=len(freeze["source_snapshot"]["inventory"]),
        all_identity_bytes_mapped=True, clean_clone_inventory_only=True,
        original_execution_paths_preserved=True, runtime_owners_reaudited=True)


def run(args):
    cpu, wall = time.process_time(), time.monotonic()
    row = pinned_file(args.inputs)
    require(row["sha256"] == args.expected_inputs_sha256, "prospective metadata-audit input SHA required")
    inputs = read(row); pins = inputs["source_hashes"]
    require(all(pinned_file(path)["sha256"] == digest for path, digest in pins.items()),
            "all prospective metadata-audit source/runtime/evidence bytes must match")
    import mujoco
    import rage_mill.canonical_benchmark as benchmark
    import rage_mill.canonical_evaluation as evaluation
    def forbidden(*a, **kw):
        raise AssertionError("independent readiness review forbids model, native, policy, episode and reset-RNG execution")
    benchmark.CanonicalEnv = benchmark.hanging_state = benchmark.record_episode = forbidden
    evaluation.hanging_state = evaluation.record_development_cohort = evaluation._record_episode = forbidden
    np.random.default_rng = mujoco.MjModel = mujoco.MjData = mujoco.mj_step = mujoco.mj_forward = forbidden
    identity = read(inputs["identity"])
    require(validate_identity(identity).n_links == 18, "own-count18 candidate identity required")
    value_module = importlib.import_module("rage_mill.canonical18_native_value_policy")
    value_module._tables(read_pinned(identity["controller"]))  # Table validation only; no policy/model.
    value_module.controller_factory = forbidden
    history = source_and_history(inputs, identity)
    prerequisites = audit_prerequisites(inputs, identity)
    freeze = audit_freeze_metadata(inputs, identity, prerequisites) if inputs["phase"] == "freeze" else None
    runtime_pins = validate_runtime_manifest(read(inputs["runtime_closure"]))
    loaded = {path: pinned_file(path)["sha256"] for path in runtime_binding_paths(
        package_names=(), include_stdlib=False, include_loaded=True)}
    missing = {path: digest for path, digest in loaded.items() if runtime_pins.get(path) != digest}
    require(not missing and all(pinned_file(path)["sha256"] == digest for path, digest in pins.items()),
            "review actual-loaded runtime coverage or unchanged input closure failed: " + json.dumps(missing))
    ledger = CLAIMS / ("published-" + identity["source_commit"]) / "count-18"
    require(not ledger.exists() and not ledger.is_symlink(), "count18 namespace changed during metadata review")
    args.out.mkdir(parents=True, exist_ok=False)
    coverage = write_json_exclusive(args.out / "runtime-coverage.json", dict(
        runtime_closure=inputs["runtime_closure"], actual_loaded_file_sha256=loaded,
        uncovered_or_changed=missing, runtime_complete=True, native_commands=0, reset_RNG_draws=0))
    receipt = write_json_exclusive(args.out / "receipt.json", dict(
        schema="rage_mill.canonical18_independent_readiness_review.v1", phase=inputs["phase"], passed=True,
        identity=inputs["identity"], identity_sha256=value_sha256(identity), activation_authorized=False,
        source_and_historical_claim_review=history, prerequisites=prerequisites, freeze_review=freeze,
        runtime_coverage=coverage, source_hashes_unchanged=True, controller_qualified=False,
        native_commands=0, controller_calls=0, reset_RNG_draws=0, new_physical_episodes=0,
        new_reserved_labels=0, CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall))
    print(json.dumps(dict(receipt=receipt, phase=inputs["phase"], passed=True)), flush=True)


def prepare(args):
    cpu, wall = time.process_time(), time.monotonic()
    identity_row = pinned_file(args.candidate / "identity.json")
    require(identity_row["sha256"] == args.expected_identity_sha256, "explicit fixed-candidate identity SHA required")
    require(validate_identity(read(identity_row)).n_links == 18, "own-count eighteen candidate required")
    import mujoco
    importlib.import_module("rage_mill.canonical18_native_value_policy")
    # Preload only helper definitions and native dependencies; never call their
    # validators, construct a model, evaluate a graph or draw reset noise here.
    runpy.run_path(str(ROOT / "scripts/audit_canonical18_development_telemetry.py"), run_name="readiness_metadata_only")
    runpy.run_path(str(ROOT / "scripts/evaluate_canonical18_profile_candidate.py"), run_name="readiness_metadata_only")
    inputs = dict(schema="rage_mill.canonical18_readiness_inputs.v1", phase=args.phase,
        identity=identity_row, candidate_preparation=pinned_file(args.candidate / "preparation.json"),
        source_review=pinned_file(args.source_review), initial_inventory=pinned_file(args.initial_inventory),
        source_tests_owner=str(args.source_tests_owner.resolve(strict=True)),
        formal_owners={name: str(path.resolve(strict=True)) for name, path in
            (("exact-calibration", args.calibration_owner), ("noisy20", args.noisy20_owner), ("noisy100", args.noisy100_owner))},
        native_owners={name: str(path.resolve(strict=True)) for name, path in
            (("noisy20", args.native20_owner), ("noisy100", args.native100_owner))})
    if args.phase == "freeze":
        require(args.freeze is not None and args.prerequisite_review is not None,
                "actual original-path freeze and passing prerequisite review required")
        inputs.update(freeze=pinned_file(args.freeze), prerequisite_review=pinned_file(args.prerequisite_review))
    files = [Path(__file__).resolve(), args.supervisor.resolve(strict=True), Path(sys.prefix) / "pyvenv.cfg",
             ROOT / "scripts/audit_canonical18_development_telemetry.py"]
    directories = [args.candidate, args.calibration_owner, args.noisy20_owner, args.noisy100_owner,
                   args.native20_owner, args.native100_owner, args.source_tests_owner]
    files += [path.resolve() for directory in directories for path in directory.rglob("*") if path.is_file()]
    files += [Path(value["path"]) for value in inputs.values() if isinstance(value, dict) and "path" in value]
    files += [Path(row["path"]) for row in read(inputs["initial_inventory"])["historical_claim_files"]]
    if args.phase == "freeze":
        frozen = read(inputs["freeze"])
        files += [Path(frozen["source_snapshot"]["bundle"]["path"])]
        files += [path.resolve() for path in Path(frozen["source_snapshot"]["fresh_clone"]).rglob("*")
                  if path.is_file() and ".git" not in path.parts]
    files += [Path(row["path"]) for row in read(identity_row)["sources"]]
    manifest = create_runtime_manifest(package_names=("numpy", "mujoco"), extra_paths=files)
    pins = validate_runtime_manifest(manifest)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    runtime = write_json_exclusive(out / "runtime-closure.json", manifest)
    pins[runtime["path"]] = runtime["sha256"]
    inputs.update(runtime_closure=runtime, source_hashes=dict(sorted(pins.items())))
    input_row = write_json_exclusive(out / "inputs.json", inputs)
    pins[input_row["path"]] = input_row["sha256"]
    source = Path(__file__).resolve()
    launch = write_json_exclusive(out / "launch.json", dict(
        schema="rage_mill.canonical18_readiness_owner.v1",
        argv=[str(Path(sys.executable).absolute()), str(source), "--run", "--inputs", input_row["path"],
              "--expected-inputs-sha256", input_row["sha256"], "--out", str(out / "output")],
        cwd=str(out), environment=dict(PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
            CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        CPU_cap_s=120, wall_cap_s=180, address_cap_bytes=8*1024**3,
        output_path=str(out / "output"), log=str(out / "worker.log"), receipt=str(out / "completion.json"),
        source_hashes=dict(sorted(pins.items())), runtime_closure=runtime,
        phase=args.phase, maximum_native_commands=0, controller_calls=0, reset_RNG_draws=0,
        new_physical_episodes=0, new_reserved_claims=0,
        purpose="Independent own18 saved-only " + args.phase + " review; no activation"))
    print(json.dumps(dict(preparation=write_json_exclusive(out / "preparation.json", dict(
        metadata_only=True, launch=launch, inputs=input_row, runtime_closure=runtime,
        native_commands=0, model_constructions=0, reset_RNG_draws=0,
        CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall)))), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument("--prepare", action="store_true"); parser.add_argument("--run", action="store_true")
    for name in ("candidate", "calibration-owner", "noisy20-owner", "noisy100-owner", "native20-owner", "native100-owner",
                 "source-tests-owner", "source-review", "initial-inventory", "supervisor", "freeze", "prerequisite-review", "inputs", "out"):
        parser.add_argument("--" + name, type=Path, required=name == "out")
    parser.add_argument("--phase", choices=("prerequisites", "freeze"))
    parser.add_argument("--expected-identity-sha256"); parser.add_argument("--expected-inputs-sha256")
    args = parser.parse_args()
    require(args.prepare != args.run, "one explicit metadata preparation or bounded read-only review required")
    if args.prepare:
        require(all(getattr(args, name) is not None for name in ("candidate", "calibration_owner", "noisy20_owner",
            "noisy100_owner", "native20_owner", "native100_owner", "source_tests_owner", "source_review",
            "initial_inventory", "supervisor", "phase", "expected_identity_sha256")), "complete declared review inputs required")
        prepare(args)
    else:
        require(args.inputs is not None and args.expected_inputs_sha256, "prospective pinned review inputs required")
        run(args)


if __name__ == "__main__":
    main()
