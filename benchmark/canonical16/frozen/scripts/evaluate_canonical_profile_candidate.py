"""Source-bound native-profile calibration and fresh gated DEV recording.

The only count16 identifiers are declared below. This runner has no published
reserved-label activation route. Preparing its source consumes no labels.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path

import numpy as np

from rage_mill.canonical_evaluation import (
    audit_episode, pinned_file, read_pinned, record_development_cohort, require,
    validate_identity, write_json_exclusive,
)
from rage_mill.canonical_release import (
    audit_cohort, audit_exact_source_calibration, source_calibration_from_proof,
)
from rage_mill.canonical_replay import prove_recorded_native_transitions
from rage_mill.canonical_runtime_evidence import audit_runtime_execution, record_runtime_execution

MODES = {
    "exact-calibration": (3167109, 1, True),
    "noisy20": (3168000, 20, False),
    "noisy100": (3168100, 100, False),
}


def prediction_parity(predicted, actual):
    """Saved arrays only: keep each byte/dtype/shape result independently."""
    details = {}
    for key in ("states", "actions", "raw_actions"):
        a, b = predicted[key], actual[key]
        details[key] = {"dtype": a.dtype == b.dtype, "shape": a.shape == b.shape,
                        "bytes": a.tobytes() == b.tobytes()}
    complete = all(all(value.values()) for value in details.values())
    final = (predicted["states"].shape == actual["states"].shape
             and predicted["states"].dtype == actual["states"].dtype
             and predicted["states"][-1].tobytes() == actual["states"][-1].tobytes())
    return details, bool(complete and final), bool(final)


def source_bound_factory(identity):
    """Resolve only the literal source-pinned top-level factory in the identity."""
    entry = identity["policy_parameters"].get("factory")
    require(isinstance(entry, str) and entry.startswith("rage_mill.")
            and all(part.isidentifier() for part in entry.split(".")), "literal package factory required")
    module_name, attribute = entry.rsplit(".", 1)
    # Find the source before importing it; a caller string cannot authorize an
    # unpinned module outside the reviewed frozen source inventory.
    relative = Path(*module_name.split(".")).with_suffix(".py")
    candidates = [row for row in identity["sources"]
                  if Path(row["path"]).parts[-len(relative.parts):] == relative.parts]
    require(len(candidates) == 1, "one frozen factory source required")
    source = candidates[0]; read_pinned(source)
    module = importlib.import_module(module_name)
    require(pinned_file(module.__file__) == source, "imported factory source differs from identity")
    factory = getattr(module, attribute)
    require(inspect.isfunction(factory) and factory.__module__ == module_name
            and factory.__name__ == attribute and factory.__closure__ is None
            and str(Path(factory.__code__.co_filename).resolve()) == source["path"],
            "literal source-defined factory required")
    return factory


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--expected-identity-sha256", required=True)
    parser.add_argument("--runtime-closure", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=tuple(MODES))
    parser.add_argument("--source-calibration", type=Path)
    parser.add_argument("--expected-source-calibration-sha256")
    parser.add_argument("--noisy20-receipt", type=Path)
    parser.add_argument("--expected-noisy20-receipt-sha256")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    data = args.identity.read_bytes()
    require(not args.identity.is_symlink()
            and hashlib.sha256(data).hexdigest() == args.expected_identity_sha256,
            "prospective fixed identity SHA required")
    identity = json.loads(data)
    require(validate_identity(identity).n_links == 16, "declared canonical16 modes required")
    matches = [row for row in identity["sources"]
               if row["path"] == str(args.runtime_closure.resolve(strict=True))]
    require(len(matches) == 1, "prospective runtime closure must belong to fixed identity")
    runtime_closure = matches[0]
    if args.mode in ("noisy20", "noisy100"):
        require(args.source_calibration is not None and args.expected_source_calibration_sha256,
                "actual source calibration required before formal DEV")
        calibration = pinned_file(args.source_calibration)
        require(calibration["sha256"] == args.expected_source_calibration_sha256,
                "prospectively pinned actual source calibration required")
        audit_exact_source_calibration(calibration, identity)
    else:
        require(args.source_calibration is None and args.expected_source_calibration_sha256 is None,
                "calibration and diagnostics do not supply a formal predecessor")
    if args.mode == "noisy100":
        require(args.noisy20_receipt is not None and args.expected_noisy20_receipt_sha256,
                "actual complete noisy20 evidence required before noisy100")
        first = pinned_file(args.noisy20_receipt)
        require(first["sha256"] == args.expected_noisy20_receipt_sha256,
                "prospectively pinned noisy20 cohort required")
        audit_cohort(first, identity=identity, count=20, exact=False,
                     seed_start=3168000, stage="development")
    else:
        require(args.noisy20_receipt is None and args.expected_noisy20_receipt_sha256 is None,
                "only noisy100 supplies a predecessor cohort")
    factory = source_bound_factory(identity)
    seed, count, exact = MODES[args.mode]
    args.out.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(args.out / "purpose.json", {
        "schema": "rage_mill.canonical_fixed_candidate_scope.v1", "mode": args.mode,
        "seed_start": seed, "episodes": count, "exact": exact,
        "formal_development_episodes": count if args.mode in ("noisy20", "noisy100") else 0,
        "reserved_episodes": 0, "qualification_authority": False,
    })
    created = []
    def create_policy():
        entry = {"policy": None, "factory_error": None}
        created.append(entry)
        try:
            policy = factory(identity)
            require(hasattr(policy, "selected_offset") and policy.selected_offset is None
                    and isinstance(policy.forecasts, list) and not policy.forecasts
                    and policy.selected_prediction is None,
                    "fresh profile selector and empty decision history required")
            entry["policy"] = policy
            return entry["policy"]
        except Exception as error:
            entry["factory_error"] = f"{type(error).__name__}: {error}"
            raise
    cohort = record_development_cohort(create_policy, identity,
                                      seed_start=seed, episodes=count, exact=exact,
                                      out=args.out / "episodes")
    document = json.loads(read_pinned(cohort))
    audits = [audit_episode(row, identity=identity) for row in document["episode_receipts"]]
    scores = [audit["score"] for audit in audits]
    require(len(created) == count, "one fresh profile policy attempt per actual episode required")
    decisions = []
    for index, entry in enumerate(created):
        policy = entry["policy"]
        forecasts = [] if policy is None else policy.forecasts
        predicted = None if policy is None else policy.selected_prediction
        prediction_row, parity, identical, final_identical = None, None, False, False
        if predicted is not None:
            path = args.out / f"selected-prediction-{seed + index}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **predicted)
            prediction_row = pinned_file(path)
            suffix = {key: audits[index][key][850:]
                      for key in ("states", "actions", "raw_actions")}
            parity, identical, final_identical = prediction_parity(predicted, suffix)
        decisions.append({
            "seed": seed + index, "episode_receipt": document["episode_receipts"][index],
            "selected_offset_m": None if policy is None else policy.selected_offset,
            "forecasts": forecasts, "factory_error": entry["factory_error"],
            "selected_prediction": prediction_row, "selected_prediction_parity": parity,
            "all_shadow_live_parity": identical, "final_state1500_byte_identical": final_identical,
            "completed_forecast_commands": sum(row["predicted_commands"] for row in forecasts),
            "maximum_forecast_command_attempts": 7150,
            "additional_physical_episodes": 0, "additional_reset_draws": 0,
        })
    decision_row = write_json_exclusive(args.out / "profile-decisions.json", {
        "schema": "rage_mill.canonical_profile_decisions.v1", "cohort": cohort,
        "decisions": decisions, "selection_state": "observed_launch_state_only",
        "additional_physical_episodes": 0, "additional_reset_draws": 0,
    })
    all_shadow_live_parity = all(row["all_shadow_live_parity"] for row in decisions)
    outputs = [cohort]
    calibration = None
    if exact:
        episode = document["episode_receipts"][0]
        proof = prove_recorded_native_transitions(episode, args.out / "native-proof")
        outputs = [episode, proof]
        # Failed candidates retain their tapes/proof; no feasibility document
        # is fabricated when the actual task or native proof fails.
        if scores[0] and scores[0]["success"] and all_shadow_live_parity:
            calibration = write_json_exclusive(args.out / "source-calibration.json",
                                               source_calibration_from_proof(episode, proof))
    write_json_exclusive(args.out / "receipt.json", {
        "schema": "rage_mill.canonical_fixed_candidate_component.v1", "mode": args.mode,
        "cohort": cohort, "profile_decisions": decision_row, "source_calibration": calibration, "scores": scores,
        "all_physical_pass": all(score and score["success"] for score in scores),
        "all_shadow_live_parity": all_shadow_live_parity,
        "scientific_pass": all(score and score["success"] for score in scores) and all_shadow_live_parity,
        "reserved_episodes": 0, "controller_qualified": False,
    })
    report = record_runtime_execution(args.out / "runtime-receipt.json", identity=identity,
                                      runtime_closure=runtime_closure, outputs=outputs)
    audit_runtime_execution(report, identity=identity, runtime_closure=runtime_closure, outputs=outputs)
    print(json.dumps({"mode": args.mode, "scores": scores}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
