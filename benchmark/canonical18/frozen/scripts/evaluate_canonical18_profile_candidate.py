"""Source-bound canonical18 calibration and prospectively declared DEV gates.

This worker has no reserved activation path. A pinned declaration selects fresh
disjoint nonreserved labels before calibration; unchanged identity is required
for the subsequent complete development20 and development100 gates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path

import numpy as np

from rage_mill.canonical18_evidence import audit_component
from rage_mill.canonical_evaluation import (
    audit_episode, is_reserved, pinned_file, read_pinned, record_development_cohort,
    require, validate_identity, write_json_exclusive,
)
from rage_mill.canonical_release import (
    audit_cohort, audit_exact_source_calibration, source_calibration_from_proof,
)
from rage_mill.canonical_replay import prove_recorded_native_transitions
from rage_mill.canonical_runtime_evidence import audit_runtime_execution, record_runtime_execution

MODE_REQUIREMENTS = {"exact-calibration": (1, True), "noisy20": (20, False),
                     "noisy100": (100, False)}


def development_modes(declaration):
    """Validate only prospective labels; never construct a plant or draw RNG."""
    require(isinstance(declaration, dict)
            and declaration.get("schema") == "rage_mill.canonical18_development_declaration.v1"
            and type(declaration.get("n_links")) is int and declaration["n_links"] == 18,
            "eighteen prospective development declaration required")
    modes = {
        "exact-calibration": {"seed_start": declaration.get("exact_calibration_seed"),
                              "episodes": 1, "exact": True},
        "noisy20": declaration.get("development20"),
        "noisy100": declaration.get("development100"),
    }
    labels = set()
    result = {}
    for name, (count, exact) in MODE_REQUIREMENTS.items():
        row = modes[name]
        require(isinstance(row, dict) and set(row) == {"seed_start", "episodes", "exact"}
                and type(row["seed_start"]) is int and row["seed_start"] >= 0
                and type(row["episodes"]) is int and row["episodes"] == count
                and row["exact"] is exact, "bounded declared canonical18 mode required")
        cohort = set(range(row["seed_start"], row["seed_start"] + count))
        require(not labels & cohort and not any(is_reserved(seed) for seed in cohort),
                "fresh modes must be disjoint and exclude every count's reserves")
        labels.update(cohort)
        result[name] = (row["seed_start"], count, exact)
    return result


def prediction_parity(predicted, actual):
    details = {}
    for key in ("states", "actions", "raw_actions"):
        a, b = predicted[key], actual[key]
        details[key] = {"dtype": a.dtype == b.dtype, "shape": a.shape == b.shape,
                        "bytes": a.tobytes() == b.tobytes()}
    final = (len(predicted["states"]) > 0
             and predicted["states"].shape == actual["states"].shape
             and predicted["states"].dtype == actual["states"].dtype
             and predicted["states"][-1].tobytes() == actual["states"][-1].tobytes())
    return details, bool(final and all(all(flags.values()) for flags in details.values())), bool(final)


def source_bound_factory(identity):
    entry = identity["policy_parameters"].get("factory")
    require(isinstance(entry, str) and entry.startswith("rage_mill.")
            and all(part.isidentifier() for part in entry.split(".")), "literal package factory required")
    module_name, attribute = entry.rsplit(".", 1)
    relative = Path(*module_name.split(".")).with_suffix(".py")
    candidates = [row for row in identity["sources"]
                  if Path(row["path"]).parts[-len(relative.parts):] == relative.parts]
    require(len(candidates) == 1, "one frozen factory source required")
    source = candidates[0]
    read_pinned(source)
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
    parser.add_argument("--declaration", required=True, type=Path)
    parser.add_argument("--expected-declaration-sha256", required=True)
    parser.add_argument("--runtime-closure", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=tuple(MODE_REQUIREMENTS))
    parser.add_argument("--source-calibration", type=Path)
    parser.add_argument("--expected-source-calibration-sha256")
    parser.add_argument("--noisy20-receipt", type=Path)
    parser.add_argument("--expected-noisy20-receipt-sha256")
    parser.add_argument("--noisy20-component-receipt", type=Path)
    parser.add_argument("--expected-noisy20-component-receipt-sha256")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    identity_row = pinned_file(args.identity)
    require(identity_row["sha256"] == args.expected_identity_sha256,
            "prospective fixed identity SHA required")
    identity = json.loads(read_pinned(identity_row))
    protocol = validate_identity(identity)
    require(protocol.n_links == 18, "eighteen candidate identity required")
    declaration_row = pinned_file(args.declaration)
    require(declaration_row["sha256"] == args.expected_declaration_sha256
            and declaration_row in identity["sources"],
            "prospective declaration must belong to fixed identity")
    modes = development_modes(json.loads(read_pinned(declaration_row)))
    runtime_closure = pinned_file(args.runtime_closure)
    require(runtime_closure in identity["sources"],
            "prospective runtime closure must belong to fixed identity")
    own_source = pinned_file(__file__)
    require(own_source in identity["sources"], "development worker source must be frozen")
    if args.mode in ("noisy20", "noisy100"):
        require(args.source_calibration is not None and args.expected_source_calibration_sha256,
                "actual source calibration required before formal DEV")
        calibration_row = pinned_file(args.source_calibration)
        require(calibration_row["sha256"] == args.expected_source_calibration_sha256,
                "prospectively pinned actual source calibration required")
        calibration = audit_exact_source_calibration(calibration_row, identity)
        audit_episode(calibration["episode_receipt"], identity=identity,
                      expected_seed=modes["exact-calibration"][0], expected_exact=True)
    else:
        require(args.source_calibration is None and args.expected_source_calibration_sha256 is None,
                "exact calibration does not supply a formal predecessor")
    if args.mode == "noisy100":
        require(args.noisy20_receipt is not None and args.expected_noisy20_receipt_sha256,
                "actual complete noisy20 evidence required before noisy100")
        first = pinned_file(args.noisy20_receipt)
        require(first["sha256"] == args.expected_noisy20_receipt_sha256,
                "prospectively pinned noisy20 cohort required")
        audit_cohort(first, identity=identity, count=20, exact=False,
                     seed_start=modes["noisy20"][0], stage="development")
        require(args.noisy20_component_receipt is not None
                and args.expected_noisy20_component_receipt_sha256,
                "passing noisy20 physical and profile component required before noisy100")
        first_component = pinned_file(args.noisy20_component_receipt)
        require(first_component["sha256"] == args.expected_noisy20_component_receipt_sha256,
                "prospectively pinned complete noisy20 component required")
        checked = audit_component(first_component, identity=identity,
                                  seed_start=modes["noisy20"][0], episodes=20, exact=False)
        require(checked["cohort"] == first, "noisy20 cohort/component lineage differs")
    else:
        require(args.noisy20_receipt is None and args.expected_noisy20_receipt_sha256 is None,
                "only noisy100 supplies a predecessor cohort")
        require(args.noisy20_component_receipt is None
                and args.expected_noisy20_component_receipt_sha256 is None,
                "only noisy100 supplies a predecessor profile component")
    factory = source_bound_factory(identity)
    seed, count, exact = modes[args.mode]
    launch_tick = identity["policy_parameters"]["selection_tick"]
    maximum_forecast_commands = identity["policy_parameters"]["maximum_forecast_commands"]
    require(type(launch_tick) is int and 0 <= launch_tick < protocol.episode_ticks
            and type(maximum_forecast_commands) is int and maximum_forecast_commands > 0,
            "prospective finite selection horizon and budget required")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(args.out / "purpose.json", {
        "schema": "rage_mill.canonical_fixed_candidate_scope.v1", "mode": args.mode,
        "seed_start": seed, "episodes": count, "exact": exact, "n_links": 18,
        "prospective_declaration": declaration_row,
        "formal_development_episodes": 0 if exact else count,
        "reserved_episodes": 0, "qualification_authority": False,
    })
    created = []

    def create_policy():
        entry = {"policy": None, "factory_error": None}
        created.append(entry)
        try:
            policy = factory(identity)
            require(policy.launch_tick == launch_tick and policy.selected_offset is None
                    and isinstance(policy.forecasts, list) and not policy.forecasts
                    and policy.selected_prediction is None,
                    "fresh profile selector and empty decision history required")
            entry["policy"] = policy
            return policy
        except Exception as error:
            entry["factory_error"] = f"{type(error).__name__}: {error}"
            raise

    cohort = record_development_cohort(create_policy, identity, seed_start=seed,
                                      episodes=count, exact=exact, out=args.out / "episodes")
    document = json.loads(read_pinned(cohort))
    audits = [audit_episode(row, identity=identity) for row in document["episode_receipts"]]
    scores = [audit["score"] for audit in audits]
    require(len(created) == count, "one fresh policy attempt per actual episode required")
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
            suffix = {key: audits[index][key][launch_tick:]
                      for key in ("states", "actions", "raw_actions")}
            parity, identical, final_identical = prediction_parity(predicted, suffix)
        attempts = sum(row["predicted_commands"] for row in forecasts)
        require(attempts <= maximum_forecast_commands, "registered forecast budget exceeded")
        decisions.append({
            "seed": seed + index, "episode_receipt": document["episode_receipts"][index],
            "selected_offset_m": None if policy is None else policy.selected_offset,
            "forecasts": forecasts, "factory_error": entry["factory_error"],
            "selected_prediction": prediction_row, "selected_prediction_parity": parity,
            "all_shadow_live_parity": identical, "final_state1500_byte_identical": final_identical,
            "completed_forecast_commands": attempts,
            "maximum_forecast_command_attempts": maximum_forecast_commands,
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
        if scores[0] and scores[0]["success"] and all_shadow_live_parity:
            calibration = write_json_exclusive(args.out / "source-calibration.json",
                                               source_calibration_from_proof(episode, proof))
    write_json_exclusive(args.out / "receipt.json", {
        "schema": "rage_mill.canonical_fixed_candidate_component.v1", "mode": args.mode,
        "cohort": cohort, "profile_decisions": decision_row, "source_calibration": calibration,
        "scores": scores, "all_physical_pass": all(score and score["success"] for score in scores),
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
