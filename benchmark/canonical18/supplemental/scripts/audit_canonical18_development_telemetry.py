"""Audit completed native-value DEV tapes and cost telemetry without execution.

No plant, controller, factory or reset generator is called. Candidate counts are
recomputed from the saved per-step rows. Full candidate ledgers are not retained
by the DEV worker, so this is a telemetry consistency audit, not a new numerical
evaluation of the alternatives.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import runpy
import time

import numpy as np

from rage_mill.canonical_benchmark import score_tape
from rage_mill.canonical_evaluation import (
    audit_journal, canonical_json, pinned_file, read_pinned, require,
    validate_identity, value_sha256, write_json_exclusive,
)
from rage_mill.canonical_replay import audit_native_replay_proof
from rage_mill.canonical_runtime_evidence import audit_runtime_execution
from rage_mill.canonical_runtime_manifest import validate_runtime_manifest


def read(row):
    return json.loads(read_pinned(row))


def literal_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def telemetry_counts(report):
    """Recount independent histories and validate cached counters, allowing F32 differences."""
    require(isinstance(report, dict) and isinstance(report.get("steps"), list)
            and len(report["steps"]) == 650, "complete650 telemetry rows required")
    calls = maximum = duplicates = agreements = ratio_count = 0
    ratio_sum, ratio_min, ratio_max = 0., None, None
    for tick, row in enumerate(report["steps"]):
        count, repeated = row.get("unique_forecasts"), row.get("duplicate_proposals")
        require(type(count) is int and 1 <= count <= 16 and type(repeated) is int and repeated >= 0
                and type(row.get("episode_tick")) is int and row["episode_tick"] == tick + 850
                and type(row.get("route_tick")) is int and row["route_tick"] == tick
                and row.get("previous_selected_state_byte_parity") is (None if tick == 0 else True),
                "native telemetry clock/count/previous-state parity differs")
        for key in ("selected_index", "baseline_index"):
            require(type(row.get(key)) is int and calls <= row[key] < calls + count,
                    "candidate index escaped fresh per-command block")
        require(row["baseline_index"] == calls, "baseline must be first in its native candidate block")
        selected, baseline = row.get("selected_cost"), row.get("baseline_cost")
        require(type(selected) in (int, float) and type(baseline) in (int, float)
                and math.isfinite(selected) and math.isfinite(baseline)
                and 0 <= selected <= baseline, "retained baseline objective differs")
        agreement = row.get("baseline_delivery_agreement")
        require(type(agreement) is bool, "baseline delivery reporting must be explicit")
        calls += count; maximum = max(maximum, count); duplicates += repeated; agreements += int(agreement)
        if baseline > 0.:
            ratio = selected / baseline
            ratio_count += 1; ratio_sum += ratio
            ratio_min = ratio if ratio_min is None else min(ratio_min, ratio)
            ratio_max = ratio if ratio_max is None else max(ratio_max, ratio)
    expected = dict(native_candidate_commands=calls, active_commands=650,
        max_candidates_per_active_command=maximum, duplicate_proposals=duplicates,
        native_forecast_errors=0, baseline_delivery_agreements=agreements,
        baseline_delivery_disagreements=650 - agreements,
        selected_to_baseline_ratio_count=ratio_count, selected_to_baseline_ratio_sum=ratio_sum,
        selected_to_baseline_ratio_min=ratio_min, selected_to_baseline_ratio_max=ratio_max,
        selected_to_baseline_ratio_mean=None if not ratio_count else ratio_sum / ratio_count)
    require(calls <= 10400 and canonical_json({key: report.get(key) for key in expected})
            == canonical_json(expected), "cached native telemetry counters differ from ordered650 rows")
    for key in ("native_select_CPU_seconds", "native_select_wall_seconds"):
        value = report.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0.,
                "finite nonnegative native selection cost required")
        expected[key] = value
    return expected


def saved_episode(row, identity, *, seed, exact, directory):
    """Read original development arrays/journal without reconstructing reset RNG."""
    document, protocol = read(row), validate_identity(identity)
    require(row["path"] == str(directory / "receipt.json")
            and document.get("schema") == "rage_mill.canonical_episode_evidence.v1"
            and document.get("identity") == identity and document.get("identity_sha256") == value_sha256(identity)
            and document.get("stage") == "development" and document.get("seed") == seed
            and document.get("exact") is exact and document.get("initialized") is True
            and document.get("source_pins_unchanged") is True and document.get("invalid_reason") is None
            and type(document.get("reset_count_after_initialization")) is int
            and document["reset_count_after_initialization"] == 0,
            "original complete development episode required")
    claim = read(document["claim"])
    require(claim.get("schema") == "rage_mill.canonical_development_episode_claim.v1"
            and claim.get("stage") == "development" and claim.get("seed") == seed
            and claim.get("exact") is exact and claim.get("identity_sha256") == value_sha256(identity),
            "development episode claim differs")
    require(all(Path(document[name]["path"]).parent == directory for name in ("claim", "recording", "journal")),
            "development episode arrays/claim/journal escaped actual owner")
    with np.load(io.BytesIO(read_pinned(document["recording"])), allow_pickle=False) as saved:
        require(set(saved.files) == {"states", "actions", "raw_actions", "times", "failure_state"},
                "complete original tape fields required")
        arrays = {key: saved[key].copy() for key in saved.files}
    for key, shape, dtype in (("states", (1501, 38), "float64"), ("actions", (1500,), "float32"),
            ("raw_actions", (1500,), "float64"), ("times", (1501,), "float64"),
            ("failure_state", (0, 38), "float64")):
        require(arrays[key].shape == shape and arrays[key].dtype == np.dtype(dtype)
                and np.isfinite(arrays[key]).all(), "complete finite original tape required: " + key)
    if exact:
        initial = np.zeros(38); initial[1] = np.pi
        require(literal_equal(arrays["states"][0], initial), "literal exact initial state differs")
    require(np.max(np.abs(arrays["times"] - np.arange(1501) * .02)) <= 1e-9
            and literal_equal(np.clip(arrays["raw_actions"].astype(np.float32), -1., 1.), arrays["actions"]),
            "original native clock or raw/F32 delivery differs")
    score = score_tape(arrays["states"], arrays["actions"], protocol)
    require(score == document.get("score") and score["success"], "actual complete strict score differs")
    audit_journal(read_pinned(document["journal"]), **{key: arrays[key] for key in
                  ("states", "actions", "raw_actions", "times")}, invalid_reason=None)
    return document, arrays, score


def audit_owner(directory, launch_sha256, *, identity, runtime, modes, parity):
    directory = Path(directory).resolve(strict=True)
    launch_row, completion_row = (pinned_file(directory / name) for name in ("launch.json", "completion.json"))
    require(launch_row["sha256"] == launch_sha256, "prospective original owner recipe SHA differs")
    launch, completion = read(launch_row), read(completion_row)
    require(completion.get("recipe_path") == launch_row["path"]
            and completion.get("recipe_sha256") == launch_row["sha256"]
            and completion.get("naturally_closed_zero") is True
            and type(completion.get("child_returncode")) is int and completion["child_returncode"] == 0
            and completion.get("failure") is None and completion.get("interrupted") is False
            and completion.get("wall_timeout") is False and completion.get("recipe_unchanged") is True
            and completion.get("owned_group_cleanup"), "completed naturally closed owner required")
    pins, unchanged = launch.get("source_hashes"), completion.get("source_hashes_unchanged")
    runtime_pins = validate_runtime_manifest(read(runtime), require_complete=True)
    require(isinstance(pins, dict) and isinstance(unchanged, dict) and set(pins) == set(unchanged)
            and all(flag is True for flag in unchanged.values())
            and all(pins.get(path) == digest for path, digest in runtime_pins.items())
            and all(pins.get(row["path"]) == row["sha256"] for row in
                    [identity["controller"], identity["configuration"], *identity["sources"]])
            and all(pinned_file(path)["sha256"] == digest for path, digest in pins.items()),
            "original owner full source/runtime pin closure differs")
    output = directory / "result"
    require(launch.get("runtime_closure") == runtime and launch.get("output_path") == str(output),
            "owner/runtime/output lineage differs")
    workers = [row for row in identity["sources"]
               if Path(row["path"]).name == "evaluate_canonical18_profile_candidate.py"]
    argv = launch.get("argv")
    require(len(workers) == 1 and isinstance(argv, list) and len(argv) >= 2
            and argv[1] == workers[0]["path"] and pins.get(argv[1]) == workers[0]["sha256"],
            "original owner must execute the unique source-bound evaluator")
    log_row = pinned_file(completion["log_path"])
    require(log_row["path"] == str(directory / "worker.log") and log_row["sha256"] == completion["log_sha256"],
            "original owner log differs")
    component_row = pinned_file(output / "receipt.json"); component = read(component_row)
    mode = component.get("mode"); require(mode in modes, "declared development mode required")
    seed, count, exact = modes[mode]
    expected_caps = dict(maximum_saved_proof_transitions=3000 if exact else 0,
        maximum_shadow_forecast_command_attempts=650 * count,
        maximum_outer_shadow_inner_native_candidates=10400 * count,
        maximum_live_inner_native_candidates=10400 * count,
        maximum_total_inner_native_candidates=20800 * count,
        maximum_actual_physical_commands=1500 * count,
        maximum_total_native_commands_including_proof=22950 * count + (3000 if exact else 0))
    require(all(type(launch.get(key)) is int and launch[key] == value for key, value in expected_caps.items()),
            "prospective nested owner caps differ")
    require(component.get("schema") == "rage_mill.canonical_fixed_candidate_component.v1"
            and all(component.get(key) is True for key in
                    ("scientific_pass", "all_physical_pass", "all_shadow_live_parity")),
            "passing physical and shadow/live component required")
    cohort_row, decisions_row = component["cohort"], component["profile_decisions"]
    require(cohort_row["path"] == str(output / "episodes/receipt.json")
            and decisions_row["path"] == str(output / "profile-decisions.json"), "actual result lineage differs")
    cohort, decisions = read(cohort_row), read(decisions_row)
    require(cohort.get("schema") == "rage_mill.canonical_cohort_evidence.v1"
            and cohort.get("identity") == identity and cohort.get("identity_sha256") == value_sha256(identity)
            and cohort.get("stage") == "development" and cohort.get("seed_start") == seed
            and cohort.get("episodes") == count and cohort.get("exact") is exact,
            "development cohort distribution/identity differs")
    claim = read(cohort["claim"])
    require(claim.get("schema") == "rage_mill.canonical_development_cohort_claim.v1"
            and claim.get("stage") == "development" and claim.get("seed_start") == seed
            and claim.get("episodes") == count and claim.get("exact") is exact
            and claim.get("identity_sha256") == value_sha256(identity), "development cohort claim differs")
    rows, entries = cohort.get("episode_receipts"), decisions.get("decisions")
    require(isinstance(rows, list) and len(rows) == count and len({r["path"] for r in rows}) == count
            and isinstance(entries, list) and len(entries) == count
            and decisions.get("cohort") == cohort_row
            and decisions.get("selection_state") == "observed_launch_state_only"
            and decisions.get("additional_physical_episodes") == decisions.get("additional_reset_draws") == 0,
            "distinct fresh episode/decision histories required")
    results, scores = [], []
    for index, (episode_row, decision) in enumerate(zip(rows, entries, strict=True)):
        label = seed + index
        document, actual, score = saved_episode(episode_row, identity, seed=label, exact=exact,
                                               directory=output / f"episodes/episode-{label}")
        scores.append(score)
        require(decision.get("seed") == label and decision.get("episode_receipt") == episode_row
                and decision.get("factory_error") is None and decision.get("selected_offset_m") == 0.
                and decision.get("additional_physical_episodes") == decision.get("additional_reset_draws") == 0,
                "fresh zero-only causal factory decision required")
        require(decision["selected_prediction"]["path"] == str(output / f"selected-prediction-{label}.npz"),
                "selected prediction escaped original owner")
        with np.load(io.BytesIO(read_pinned(decision["selected_prediction"])), allow_pickle=False) as saved:
            require(set(saved.files) == {"states", "actions", "raw_actions"}, "full selected prediction required")
            prediction = {key: saved[key].copy() for key in saved.files}
        flags, identical, final = parity(prediction, {key: actual[key][850:] for key in prediction})
        require(identical and final and flags == decision.get("selected_prediction_parity")
                and decision.get("all_shadow_live_parity") is True
                and decision.get("final_state1500_byte_identical") is True,
                "full650 raw/F32/state suffix or final1500 bytes differ")
        forecasts = decision.get("forecasts")
        require(isinstance(forecasts, list) and len(forecasts) == 1, "one zero-offset outer forecast required")
        forecast = forecasts[0]
        local = score_tape(prediction["states"], prediction["actions"], validate_identity(identity))
        require(forecast.get("cart_error_offset_m") == 0. and forecast.get("accepted") is True
                and forecast.get("predicted_commands") == 650 and forecast.get("local_score") == local
                and forecast.get("maximum_total_inner_native_candidates") == 20800
                and forecast.get("maximum_total_native_commands_including_physical") == 22950
                and local["longest_upright_ticks"] >= 250
                and decision.get("completed_forecast_commands") == decision.get("maximum_forecast_command_attempts") == 650,
                "registered complete zero forecast differs")
        shadow = telemetry_counts(forecast.get("native_value_shadow_telemetry"))
        live = telemetry_counts(forecast.get("native_value_live_telemetry"))
        require(forecast["native_value_shadow_telemetry"]["steps"]
                == forecast["native_value_live_telemetry"]["steps"],
                "independent shadow/live decision histories differ on identical saved suffix")
        inner = shadow["native_candidate_commands"] + live["native_candidate_commands"]
        require(inner <= 20800 and inner + 650 + 1500 <= 22950,
                "nested native-map cap exceeded")
        results.append(dict(seed=label, episode=episode_row, recording=document["recording"], score=score,
            selected_prediction=decision["selected_prediction"], full_suffix_and_final1500_byte_identical=True,
            shadow=shadow, live=live, outer_shadow_native_maps=650, actual_physical_commands=1500,
            total_inner_native_maps=inner, total_native_maps_before_saved_proof=inner + 2150))
    require(scores == component.get("scores"), "cached cohort scores differ from saved arrays")
    proof_maps, runtime_outputs = 0, [cohort_row]
    if exact:
        source = read(component["source_calibration"])
        require(source.get("schema") == "rage_mill.canonical_exact_source_calibration.v1"
                and source.get("identity_sha256") == value_sha256(identity)
                and source.get("episode_receipt") == rows[0]
                and source.get("trajectory_integrity") is True and source.get("is_feasible") is True,
                "exact source calibration lineage differs")
        native = audit_native_replay_proof(source["native_replay_proof"], episode_receipt=rows[0])
        require(native["continuous_replay_identical"] is True
                and native["maximum_dynamics_defect"] <= 1e-8 and native["initial_state_gap"] <= 1e-8
                and all(source.get(key) == native[key] for key in ("maximum_dynamics_defect", "initial_state_gap")),
                "saved exact cold/continuous native proof differs")
        proof = read(source["native_replay_proof"])
        proof_maps = proof["continuous_transitions"] + proof["one_step_transitions"]
        require(proof_maps == 3000, "complete3000-map saved source proof required")
        runtime_outputs = [rows[0], source["native_replay_proof"]]
    else:
        require(component.get("source_calibration") is None, "noisy owner cannot replace source calibration")
    runtime_row = pinned_file(output / "runtime-receipt.json")
    audit_runtime_execution(runtime_row, identity=identity, runtime_closure=runtime, outputs=runtime_outputs)
    native_maps = sum(row["total_native_maps_before_saved_proof"] for row in results) + proof_maps
    select_cpu = sum(row[k]["native_select_CPU_seconds"] for row in results for k in ("shadow", "live"))
    select_wall = sum(row[k]["native_select_wall_seconds"] for row in results for k in ("shadow", "live"))
    require(select_cpu <= completion["aggregate_child_CPU_seconds"] + 1e-6
            and select_wall <= completion["wall_seconds"] + 1e-6, "reported selection cost escaped owner cost")
    return dict(mode=mode, seed_start=seed, episodes=count, exact=exact, launch=launch_row,
        completion=completion_row, component=component_row, cohort=cohort_row, decisions=decisions_row,
        runtime_execution=runtime_row, all_pins_unchanged=True, runtime_complete=True, results=results,
        saved_native_proof_maps=proof_maps, total_native_maps=native_maps,
        maximum_native_maps=22950 * count + proof_maps,
        aggregate_child_CPU_seconds=completion["aggregate_child_CPU_seconds"],
        owner_wall_seconds=completion["wall_seconds"], native_selection_CPU_seconds=select_cpu,
        native_selection_wall_seconds=select_wall,
        baseline_delivery_agreements=sum(row[k]["baseline_delivery_agreements"] for row in results for k in ("shadow", "live")),
        baseline_delivery_disagreements=sum(row[k]["baseline_delivery_disagreements"] for row in results for k in ("shadow", "live")))


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--expected-identity-sha256", required=True)
    parser.add_argument("--runtime-closure", type=Path, required=True)
    parser.add_argument("--owner", nargs=2, action="append", metavar=("DIRECTORY", "LAUNCH_SHA256"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); cpu, wall = time.process_time(), time.monotonic()
    identity_row, runtime = pinned_file(args.identity), pinned_file(args.runtime_closure)
    require(identity_row["sha256"] == args.expected_identity_sha256, "prospective identity hash required")
    identity = read(identity_row)
    require(validate_identity(identity).n_links == 18 and runtime in identity["sources"]
            and identity["policy_parameters"].get("factory")
                == "rage_mill.canonical18_native_value_policy.controller_factory",
            "own-count frozen native-value identity/runtime required")
    evaluator_sources = [row for row in identity["sources"]
                         if Path(row["path"]).name == "evaluate_canonical18_profile_candidate.py"]
    require(len(evaluator_sources) == 1, "one frozen parity helper source required")
    read_pinned(evaluator_sources[0])
    evaluator = runpy.run_path(evaluator_sources[0]["path"], run_name="saved_metadata_parity_only")
    modes = evaluator["development_modes"](read(identity["policy_parameters"]["design_metadata"]["declaration"]))
    args.out.mkdir(parents=True, exist_ok=False)
    summaries = []
    try:
        for owner, digest in args.owner:
            result = audit_owner(owner, digest, identity=identity, runtime=runtime,
                                 modes=modes, parity=evaluator["prediction_parity"])
            row = write_json_exclusive(args.out / (result["mode"] + ".json"), result)
            summaries.append(row)
        write_json_exclusive(args.out / "receipt.json", {
            "schema": "rage_mill.canonical18_saved_development_telemetry_audit.v1", "passed": True,
            "identity": identity_row, "runtime_closure": runtime, "producer_source": pinned_file(__file__),
            "audited_owners": summaries, "native_commands": 0, "controller_calls": 0,
            "factory_constructions": 0, "model_constructions": 0, "reset_RNG_draws": 0,
            "recorded_reset_reconstructions": 0, "new_physical_episodes": 0, "new_reserved_labels": 0,
            "baseline_F32_agreement_scope": "reported counts only; disagreement is not a qualification failure",
            "factory_freshness_evidence": "source-bound one-factory-attempt-per-episode enforcement; distinct saved claims/tapes/decisions and each650-row history starts with route clock0 and candidate block0",
            "limitations": ["Direct process object identities are not retained in saved evidence.",
                "All candidate native states are not retained by DEV; counts are independently summed from ordered per-step telemetry, not regenerated through physics.",
                "Noisy reset RNG is not reconstructed by this audit; original source-bound evidence and journals remain authoritative for reset generation."],
            "CPU_seconds": time.process_time() - cpu, "wall_seconds": time.monotonic() - wall})
    except BaseException as error:
        write_json_exclusive(args.out / "failed-attempt.json", dict(error=f"{type(error).__name__}: {error}",
            audited_owners=summaries, identity=identity_row, native_commands=0, controller_calls=0,
            reset_RNG_draws=0, CPU_seconds=time.process_time() - cpu, wall_seconds=time.monotonic() - wall))
        raise


if __name__ == "__main__":
    main()
