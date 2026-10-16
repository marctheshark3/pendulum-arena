"""Prove frozen native-value policy commands on existing reserved tapes only.

This worker never imports a reset generator or activates reserved ownership.
Recorded states are the causal policy inputs. Native calls are explicitly
charged forecast maps plus bounded cold/continuous saved-input proofs.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time

import numpy as np

from rage_mill.canonical17_native_value_policy import controller_factory
from rage_mill.canonical_benchmark import score_tape
from rage_mill.canonical_evaluation import (
    audit_journal, pinned_file, read_pinned, require, reserved_gates,
    validate_identity, value_sha256, write_json_exclusive,
)
from rage_mill.canonical_replay import (
    audit_native_replay_proof, prove_recorded_native_transitions,
)
from rage_mill.canonical_runtime_evidence import audit_runtime_execution, record_runtime_execution


def literal_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def read_actual_episode(row, identity, *, seed, exact, freeze):
    """Check stored lineage/journal/score without reconstructing a noisy reset."""
    document = json.loads(read_pinned(row))
    protocol = validate_identity(identity)
    require(document.get("schema") == "rage_mill.canonical_episode_evidence.v1"
            and document.get("identity") == identity
            and document.get("identity_sha256") == value_sha256(identity)
            and document.get("stage") == "reserved" and document.get("seed") == seed
            and document.get("exact") is exact and document.get("freeze") == freeze
            and document.get("source_pins_unchanged") is True
            and type(document.get("reset_count_after_initialization")) is int
            and document["reset_count_after_initialization"] == 0
            and document.get("initialized") is True and document.get("invalid_reason") is None,
            "complete unchanged original reserved episode required")
    claim = json.loads(read_pinned(document["claim"]))
    require(claim.get("schema") == "rage_mill.canonical_reserved_episode_claim.v1"
            and claim.get("stage") == "reserved" and claim.get("seed") == seed
            and claim.get("exact") is exact and claim.get("freeze") == freeze
            and claim.get("identity_sha256") == value_sha256(identity),
            "original reserved episode ownership differs")
    read_pinned(claim["ledger_label_claim"])
    read_pinned(freeze)
    with np.load(io.BytesIO(read_pinned(document["recording"])), allow_pickle=False) as saved:
        require(set(saved.files) == {"states", "actions", "raw_actions", "times", "failure_state"},
                "exact original tape fields required")
        arrays = {key: saved[key].copy() for key in saved.files}
    for key, shape, dtype in (("states", (1501, 36), "float64"),
                              ("actions", (1500,), "float32"),
                              ("raw_actions", (1500,), "float64"),
                              ("times", (1501,), "float64"),
                              ("failure_state", (0, 36), "float64")):
        require(arrays[key].shape == shape and arrays[key].dtype == np.dtype(dtype)
                and np.isfinite(arrays[key]).all(), "complete finite original tape required: " + key)
    require(np.max(np.abs(arrays["times"] - np.arange(1501) * .02)) <= 1e-9,
            "original native clock differs")
    delivered = np.clip(arrays["raw_actions"].astype(np.float32), -1., 1.)
    require(literal_equal(delivered, arrays["actions"]), "recorded raw/F32 alignment differs")
    score = score_tape(arrays["states"], arrays["actions"], protocol)
    require(score == document.get("score") and score["success"] is True,
            "original actual tape does not pass its cached complete score")
    audit_journal(read_pinned(document["journal"]), **{key: arrays[key] for key in
                  ("states", "actions", "raw_actions", "times")}, invalid_reason=None)
    return document, arrays


def telemetry_check(report):
    require(isinstance(report, dict), "owned native-value telemetry required")
    steps = report.get("steps")
    require(isinstance(steps, list) and len(steps) == 650
            and report.get("active_commands") == 650
            and report.get("native_forecast_errors") == 0
            and type(report.get("baseline_delivery_agreements")) is int
            and type(report.get("baseline_delivery_disagreements")) is int
            and report["baseline_delivery_agreements"] >= 0
            and report["baseline_delivery_disagreements"] >= 0
            and report["baseline_delivery_agreements"] + report["baseline_delivery_disagreements"] == 650,
            "all650 native decisions require complete baseline delivery reporting")
    calls, maximum, agreements = 0, 0, 0
    for route_tick, step in enumerate(steps):
        count = step.get("unique_forecasts")
        require(type(count) is int and 1 <= count <= 16
                and step.get("episode_tick") == route_tick + 850
                and step.get("route_tick") == route_tick
                and type(step.get("baseline_delivery_agreement")) is bool
                and step.get("previous_selected_state_byte_parity") is (None if route_tick == 0 else True),
                "ordered native candidate limit/parity differs")
        for key in ("selected_index", "baseline_index"):
            require(type(step.get(key)) is int and calls <= step[key] < calls + count,
                    "selected/baseline index escaped its own native candidate block")
        selected, baseline = step.get("selected_cost"), step.get("baseline_cost")
        require(type(selected) in (float, int) and type(baseline) in (float, int)
                and np.isfinite(selected) and np.isfinite(baseline)
                and 0 <= selected <= baseline, "finite retained-baseline value required")
        calls += count
        maximum = max(maximum, count)
        agreements += int(step["baseline_delivery_agreement"])
    require(report.get("native_candidate_commands") == calls <= 10400
            and report.get("max_candidates_per_active_command") == maximum
            and report["baseline_delivery_agreements"] == agreements
            and report["baseline_delivery_disagreements"] == 650 - agreements,
            "full native candidate ledger count differs")
    return calls


def prove_policy_episode(episode_row, identity, *, seed, exact, freeze, out):
    out.mkdir(parents=True, exist_ok=False)
    cpu, wall = time.process_time(), time.monotonic()
    policy, completed, failure = None, 0, None
    counters = dict(policy_calls=0, shadow_inner_native_candidates=0,
                    recorded_input_inner_native_candidates=0, outer_shadow_native_steps=0,
                    saved_native_proof_maps=0)
    try:
        document, arrays = read_actual_episode(episode_row, identity, seed=seed, exact=exact, freeze=freeze)
        policy = controller_factory(identity)
        raw = np.empty(1500, dtype=np.float64)
        delivered = np.empty(1500, dtype=np.float32)
        selected_states, selected_actions = [], []
        with (out / "command-proof.jsonl").open("x") as journal:
            for tick in range(1500):
                counters["policy_calls"] += 1
                value = np.float64(policy(arrays["states"][tick].copy(), tick))
                raw[tick] = value
                delivered[tick] = np.float32(np.clip(np.float32(value), -1., 1.))
                require(value.tobytes() == arrays["raw_actions"][tick].tobytes()
                        and delivered[tick].tobytes() == arrays["actions"][tick].tobytes(),
                        f"saved causal policy command differs at command{tick}")
                decision = policy.policy.last_decision
                if decision is not None:
                    prediction = decision.predicted_state
                    require(literal_equal(prediction, arrays["states"][tick+1])
                            and np.float32(decision.action).tobytes() == delivered[tick].tobytes(),
                            f"selected native next-state/action differs at command{tick}")
                    selected_states.append(prediction.copy())
                    selected_actions.append(np.float32(decision.action))
                journal.write(json.dumps(dict(tick=tick, raw_f64_hex=value.tobytes().hex(),
                    delivered_f32_hex=delivered[tick].tobytes().hex(),
                    selected_next_state_byte_parity=None if decision is None else True)) + "\n")
                completed += 1
            journal.flush()
        require(literal_equal(raw, arrays["raw_actions"])
                and literal_equal(delivered, arrays["actions"]), "all1500 command bytes required")
        require(policy.selected_offset == 0. and len(policy.forecasts) == 1
                and policy.forecasts[0].get("accepted") is True
                and policy.forecasts[0].get("predicted_commands") == 650,
                "literal single zero-offset complete forecast required")
        prediction = policy.selected_prediction
        require(isinstance(prediction, dict) and set(prediction) == {"states", "actions", "raw_actions"},
                "full selected prediction required")
        for key in prediction:
            require(literal_equal(prediction[key], arrays[key][850:]),
                    "selected full suffix/final state differs: " + key)
        live, shadow = policy.policy.telemetry, policy.forecasts[0]["native_value_shadow_telemetry"]
        counters["recorded_input_inner_native_candidates"] = telemetry_check(live)
        counters["shadow_inner_native_candidates"] = telemetry_check(shadow)
        counters["outer_shadow_native_steps"] = 650
        require(len(policy.policy.suffix.tracker.ledger) == counters["recorded_input_inner_native_candidates"],
                "owned recorded-input candidate ledger differs")
        with (out / "recomputed-commands.npz").open("xb") as stream:
            np.savez_compressed(stream, raw_actions=raw, actions=delivered,
                selected_next_states=np.asarray(selected_states),
                selected_next_actions=np.asarray(selected_actions, dtype=np.float32))
        with (out / "selected-prediction.npz").open("xb") as stream:
            np.savez_compressed(stream, **prediction)
        telemetry = write_json_exclusive(out / "native-value-telemetry.json",
                                         dict(shadow=shadow, recorded_input=live))
        native = prove_recorded_native_transitions(episode_row, out / "native-proof",
                                                   maximum_native_transitions=3000)
        native_document = json.loads(read_pinned(native))
        counters["saved_native_proof_maps"] = (native_document["continuous_transitions"]
                                               + native_document["one_step_transitions"])
        native_summary = audit_native_replay_proof(native, episode_receipt=episode_row)
        require(native_summary["continuous_replay_identical"] is True
                and native_summary["maximum_dynamics_defect"] <= 1e-8
                and counters["saved_native_proof_maps"] == 3000,
                "recorded cold/continuous native maps differ")
        # For noisy episodes the generic proof's initial_state_gap measures the
        # recorded reset's distance from exact hanging. It is not a replay gap.
        if exact:
            require(native_summary["initial_state_gap"] <= 1e-8, "exact recorded hanging state differs")
        native_maps = sum(counters[key] for key in counters if key != "policy_calls")
        require(native_maps <= 24450, "declared complete per-episode map ceiling exceeded")
        read_actual_episode(episode_row, identity, seed=seed, exact=exact, freeze=freeze)
        return write_json_exclusive(out / "receipt.json", {
            "schema": "rage_mill.canonical17_saved_policy_episode_proof.v1",
            "episode": episode_row, "recording": document["recording"], "seed": seed,
            "rawF64_commands_byte_proved": 1500, "deliveredF32_commands_byte_proved": 1500,
            "selected_native_next_state_delivery_pairs_byte_proved": 650,
            "selected_full651state650action_raw_suffix_byte_identical": True,
            "final_state1500_byte_identical": True, "selected_offset_m": policy.selected_offset,
            "recomputed_commands": pinned_file(out / "recomputed-commands.npz"),
            "command_journal": pinned_file(out / "command-proof.jsonl"),
            "selected_prediction": pinned_file(out / "selected-prediction.npz"),
            "native_value_telemetry": telemetry, "native_transition_proof": native,
            "native_transition_summary": native_summary, "cost_counts": counters,
            "total_native_maps": native_maps, "maximum_native_maps": 24450,
            "physical_episodes": 0, "new_reserved_labels": 0, "reset_RNG_draws": 0,
            "recorded_reset_reconstructions": 0, "passed": True,
            "CPU_seconds": time.process_time()-cpu, "wall_seconds": time.monotonic()-wall})
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        if policy is not None:
            counters["recorded_input_inner_native_candidates"] = policy.policy.telemetry["native_candidate_commands"]
            counters["shadow_inner_native_candidates"] = sum(
                report["native_candidate_commands"] for report in policy.policy.copy_telemetry_reports)
            counters["outer_shadow_native_steps"] = sum(
                forecast["predicted_commands"] for forecast in policy.forecasts)
        write_json_exclusive(out / "attempt.json", {
            "episode": episode_row, "seed": seed, "completed_policy_commands": completed,
            "failure": failure, "cost_counts_at_last_completed_check": counters,
            "live_native_value_telemetry": None if policy is None else policy.policy.telemetry,
            "copy_native_value_telemetry": [] if policy is None else policy.policy.copy_telemetry_reports,
            "physical_episodes": 0, "new_reserved_labels": 0, "reset_RNG_draws": 0,
            "CPU_seconds": time.process_time()-cpu, "wall_seconds": time.monotonic()-wall})


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    for name in ("identity", "cohort", "runtime-closure", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-identity-sha256", required=True)
    parser.add_argument("--expected-cohort-sha256", required=True)
    args = parser.parse_args()
    identity_row, cohort_row = pinned_file(args.identity), pinned_file(args.cohort)
    require(identity_row["sha256"] == args.expected_identity_sha256
            and cohort_row["sha256"] == args.expected_cohort_sha256, "prospective input hashes required")
    identity, cohort = map(lambda row: json.loads(read_pinned(row)), (identity_row, cohort_row))
    protocol = validate_identity(identity)
    own = pinned_file(__file__)
    runtime_closure = pinned_file(args.runtime_closure)
    require(protocol.n_links == 17 and own in identity["sources"]
            and runtime_closure in identity["sources"]
            and identity["policy_parameters"].get("factory")
                == "rage_mill.canonical17_native_value_policy.controller_factory",
            "own-count frozen proof worker/factory/runtime required")
    require(type(cohort.get("seed_start")) is int and type(cohort.get("episodes")) is int
            and type(cohort.get("exact")) is bool, "typed original reserved gate fields required")
    matches = [(name, details) for name, details in reserved_gates(17).items()
               if all(cohort.get(key) == details[key] for key in ("seed_start", "episodes", "exact"))]
    require(len(matches) == 1 and cohort.get("schema") == "rage_mill.canonical_cohort_evidence.v1"
            and cohort.get("stage") == "reserved" and cohort.get("identity") == identity
            and cohort.get("identity_sha256") == value_sha256(identity),
            "one original complete reserved cohort of the frozen identity required")
    gate, details = matches[0]
    rows = cohort.get("episode_receipts")
    require(isinstance(rows, list) and len(rows) == details["episodes"]
            and len({row["path"] for row in rows}) == len(rows), "one distinct saved tape per label required")
    claim = json.loads(read_pinned(cohort["claim"]))
    require(claim.get("schema") == "rage_mill.canonical_reserved_cohort_claim.v1"
            and claim.get("identity_sha256") == value_sha256(identity)
            and claim.get("freeze") == cohort.get("freeze")
            and all(claim.get(key) == details[key] for key in ("seed_start", "episodes", "exact")),
            "original reserved cohort ownership differs")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(args.out / "purpose.json", {
        "schema": "rage_mill.canonical17_saved_policy_proof_purpose.v1",
        "identity": identity_row, "cohort": cohort_row, "runtime_closure": runtime_closure,
        "producer_source": own, "maximum_policy_calls": 1500*len(rows),
        "maximum_native_maps": 24450*len(rows), "physical_episodes": 0,
        "new_reserved_labels": 0, "reset_RNG_draws": 0,
        "causal_inputs": "only existing recorded observed states, one per original command",
        "timing": "simulated50Hz; native planning CPU/wall reported; no hardware real-time claim"})
    outputs = []
    with (args.out / "episode-progress.jsonl").open("x") as progress:
        for index, row in enumerate(rows):
            seed = details["seed_start"]+index
            try:
                result = prove_policy_episode(row, identity, seed=seed, exact=details["exact"],
                    freeze=cohort["freeze"], out=args.out/f"episode-{seed}")
                outputs.append(result)
                summary = json.loads(read_pinned(result))
                message = dict(seed=seed, passed=True, receipt=result,
                               total_native_maps=summary["total_native_maps"],
                               CPU_seconds=summary["CPU_seconds"], wall_seconds=summary["wall_seconds"])
            except BaseException as error:
                progress.write(json.dumps(dict(seed=seed, passed=False,
                    failure=f"{type(error).__name__}: {error}"))+"\n")
                progress.flush()
                raise
            progress.write(json.dumps(message)+"\n"); progress.flush()
            print(json.dumps(message), flush=True)
    require(pinned_file(__file__) == own and pinned_file(args.cohort) == cohort_row,
            "proof source or original cohort changed during saved-input verification")
    document = {
        "schema": "rage_mill.canonical17_saved_policy_cohort_proof.v1", "gate": gate,
        "identity": identity_row, "cohort": cohort_row, "episode_proofs": outputs,
        "episodes_proved": len(outputs), "rawF64_commands_byte_proved": 1500*len(outputs),
        "deliveredF32_commands_byte_proved": 1500*len(outputs),
        "all_fullsuffix_and_finalstate1500_byte_identical": True,
        "total_native_maps": sum(json.loads(read_pinned(row))["total_native_maps"] for row in outputs),
        "maximum_native_maps": 24450*len(outputs), "physical_episodes": 0,
        "new_reserved_labels": 0, "reset_RNG_draws": 0, "recorded_reset_reconstructions": 0,
        "passed": True, "controller_qualified": False}
    result = write_json_exclusive(args.out / "receipt.json", document)
    report = record_runtime_execution(args.out / "runtime-receipt.json", identity=identity,
                                     runtime_closure=runtime_closure, outputs=[result])
    audit_runtime_execution(report, identity=identity, runtime_closure=runtime_closure, outputs=[result])
    print(json.dumps(dict(receipt=result, runtime_receipt=report)), flush=True)


if __name__ == "__main__":
    main()
