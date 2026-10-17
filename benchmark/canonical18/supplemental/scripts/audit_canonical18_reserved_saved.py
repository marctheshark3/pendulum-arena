"""Independently audit closed eighteen reserved tapes and saved policy proofs.

Only pinned files and saved arrays are read. No authority activation, plant,
controller, factory, reset generator, or native transition is invoked. The
original reserved worker retains no candidate ledger: candidate counts here
describe the separately owned causal saved-input proof, not an original live
counter or a regenerated alternative-objective proof.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import time

import numpy as np

from rage_mill.canonical_benchmark import score_tape
from rage_mill.canonical_evaluation import (
    audit_journal, canonical_json, pinned_file, read_pinned, require,
    reserved_gates, validate_identity, value_sha256, write_json_exclusive,
)
from rage_mill.canonical_replay import audit_native_replay_proof
from rage_mill.canonical_reserved import _audit_storage
from rage_mill.canonical_runtime_evidence import audit_runtime_execution
from rage_mill.canonical_runtime_manifest import validate_runtime_manifest


def read(row):
    return json.loads(read_pinned(row))


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def arrays(row, fields):
    with np.load(io.BytesIO(read_pinned(row)), allow_pickle=False) as saved:
        require(set(saved.files) == set(fields), "exact saved array fields required")
        result = {key: saved[key].copy() for key in fields}
    for key, (shape, dtype) in fields.items():
        value = result[key]
        require(value.shape == shape and value.dtype == np.dtype(dtype)
                and np.isfinite(value).all(), "finite literal saved array required: " + key)
    return result


def independent_score(states, actions):
    """Count strict post-command hold directly, independently of score_tape."""
    theta = np.cumsum(states[1:, 1:19], axis=1)
    angles = np.where((theta >= -np.pi) & (theta < np.pi), theta,
                      (theta + np.pi) % (2 * np.pi) - np.pi)
    streak = longest = 0
    for upright in np.all(np.abs(angles) < .15, axis=1):
        streak = streak + 1 if upright else 0
        longest = max(longest, streak)
    finite = bool(np.isfinite(states).all() and np.isfinite(actions).all())
    force = bool(np.all(np.abs(actions) <= 1.))
    f32 = bool(np.array_equal(actions, actions.astype(np.float32).astype(actions.dtype)))
    rail = bool(np.all(np.abs(states[:, 0]) <= 3.))
    complete = len(actions) == 1500
    return dict(success=bool(finite and force and f32 and rail and complete and longest >= 250),
        complete=complete, finite=finite, force_ok=force, float32_delivery_ok=f32,
        rail_ok=rail, ticks=len(actions), longest_upright_ticks=longest,
        longest_upright_seconds=longest * .02,
        maximum_cart_m=float(np.max(np.abs(states[:, 0]))), invalid_reason=None)


def telemetry(report):
    require(isinstance(report, dict), "saved telemetry object required")
    steps = report.get("steps")
    require(isinstance(steps, list) and len(steps) == 650, "650 saved decision rows required")
    calls = maximum = duplicates = agreements = ratios = 0
    ratio_sum, ratio_min, ratio_max = 0., None, None
    for tick, step in enumerate(steps):
        count, repeated = step.get("unique_forecasts"), step.get("duplicate_proposals")
        require(type(count) is int and 1 <= count <= 16
                and type(repeated) is int and repeated >= 0
                and type(step.get("route_tick")) is int and step["route_tick"] == tick
                and type(step.get("episode_tick")) is int and step["episode_tick"] == tick + 850
                and step.get("previous_selected_state_byte_parity") is (None if tick == 0 else True)
                and type(step.get("baseline_delivery_agreement")) is bool,
                "ordered saved decision clock/count/parity required")
        require(type(step.get("baseline_index")) is int and step["baseline_index"] == calls
                and type(step.get("selected_index")) is int
                and calls <= step["selected_index"] < calls + count,
                "baseline first and selected candidate within its native block required")
        selected, baseline = step.get("selected_cost"), step.get("baseline_cost")
        require(type(selected) in (int, float) and type(baseline) in (int, float)
                and math.isfinite(selected) and math.isfinite(baseline)
                and 0 <= selected <= baseline, "finite retained baseline cost required")
        calls += count; maximum = max(maximum, count); duplicates += repeated
        agreements += int(step["baseline_delivery_agreement"])
        if baseline > 0.:
            ratio = selected / baseline
            ratios += 1; ratio_sum += ratio
            ratio_min = ratio if ratio_min is None else min(ratio_min, ratio)
            ratio_max = ratio if ratio_max is None else max(ratio_max, ratio)
    counts = dict(native_candidate_commands=calls, active_commands=650,
        max_candidates_per_active_command=maximum, duplicate_proposals=duplicates,
        native_forecast_errors=0, baseline_delivery_agreements=agreements,
        baseline_delivery_disagreements=650 - agreements,
        selected_to_baseline_ratio_count=ratios, selected_to_baseline_ratio_sum=ratio_sum,
        selected_to_baseline_ratio_min=ratio_min, selected_to_baseline_ratio_max=ratio_max,
        selected_to_baseline_ratio_mean=None if not ratios else ratio_sum / ratios)
    require(calls <= 10400 and canonical_json(counts)
            == canonical_json({key: report.get(key) for key in counts}),
            "cached telemetry counters differ from saved650 rows")
    for key in ("native_select_CPU_seconds", "native_select_wall_seconds"):
        value = report.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0.,
                "finite nonnegative native selection cost required")
        counts[key] = value
    return counts


def caps(mode, count):
    proof = mode == "proof"
    return dict(maximum_outer_shadow_native_maps=650 * count,
        maximum_outer_shadow_inner_native_candidates=10400 * count,
        maximum_live_inner_native_candidates=0 if proof else 10400 * count,
        maximum_recorded_input_inner_native_candidates=10400 * count if proof else 0,
        maximum_total_inner_native_candidates=20800 * count,
        maximum_actual_physical_commands=0 if proof else 1500 * count,
        maximum_saved_native_proof_maps=3000 * count if proof else 0,
        maximum_native_commands=(24450 if proof else 22950) * count)


def owner(directory, expected_sha, *, identity, runtime, worker_name, gate, authority_row):
    directory = Path(directory).resolve(strict=True)
    launch_row, completion_row = (pinned_file(directory / name) for name in ("launch.json", "completion.json"))
    launch, closed = read(launch_row), read(completion_row)
    require(launch_row["sha256"] == expected_sha
            and closed.get("schema") == "rage_mill.next_stage_supervised_completion.v1"
            and closed.get("recipe_path") == launch_row["path"]
            and closed.get("recipe_sha256") == launch_row["sha256"]
            and type(closed.get("child_returncode")) is int and closed["child_returncode"] == 0
            and closed.get("naturally_closed_zero") is True
            and closed.get("interrupted") is False and closed.get("wall_timeout") is False
            and closed.get("failure") is None and closed.get("recipe_unchanged") is True
            and closed.get("owned_group_cleanup") == ["group_already_exited"]
            and closed.get("supervisor_source_sha256") == launch.get("supervisor", {}).get("sha256"),
            "original naturally closed owner required")
    pins, stable = launch.get("source_hashes"), closed.get("source_hashes_unchanged")
    require(isinstance(pins, dict) and isinstance(stable, dict) and set(pins) == set(stable)
            and all(flag is True for flag in stable.values())
            and all(pinned_file(path)["sha256"] == digest for path, digest in pins.items()),
            "original full prospective owner pin closure differs")
    runtime_files = validate_runtime_manifest(read(runtime), require_complete=True)
    require(all(pins.get(path) == digest for path, digest in runtime_files.items())
            and all(pins.get(row["path"]) == row["sha256"] for row in
                    [identity["controller"], identity["configuration"], *identity["sources"]]),
            "owner omitted frozen controller/source/runtime input")
    worker = [row for row in identity["sources"] if Path(row["path"]).name == worker_name]
    require(len(worker) == 1 and isinstance(launch.get("argv"), list)
            and len(launch["argv"]) >= 2 and launch["argv"][1] == worker[0]["path"]
            and str(Path(launch["argv"][0]).resolve(strict=True)) == identity["runtime"]["executable"]
            and launch.get("authority") == authority_row and launch.get("gate") == gate
            and launch.get("runtime_closure") == runtime,
            "source-bound owner worker/launcher/authority/runtime differs")
    log = pinned_file(directory / "worker.log")
    require(closed.get("log_path") == log["path"] and closed.get("log_sha256") == log["sha256"],
            "original owner log differs")
    for key in ("CPU_cap_s", "wall_cap_s", "address_cap_bytes"):
        require(type(launch.get(key)) is int and launch[key] > 0, "positive finite owner caps required")
    require(closed.get("CPU_cap_s_per_process") == launch["CPU_cap_s"]
            and closed.get("wall_cap_s") == launch["wall_cap_s"]
            and closed.get("address_cap_bytes") == launch["address_cap_bytes"],
            "completion resource caps differ from prospective recipe")
    return launch, dict(launch=launch_row, completion=completion_row, log=log,
        source_pins=len(pins), naturally_closed_zero=True,
        CPU_seconds=closed["aggregate_child_CPU_seconds"], wall_seconds=closed["wall_seconds"],
        peak_rss_kib=closed["peak_rss_kib"])


def episode(row, *, identity, seed, exact, freeze, directory):
    document = read(row)
    require(row["path"] == str(directory / "receipt.json")
            and document.get("schema") == "rage_mill.canonical_episode_evidence.v1"
            and document.get("identity") == identity and document.get("identity_sha256") == value_sha256(identity)
            and document.get("stage") == "reserved" and type(document.get("seed")) is int
            and document["seed"] == seed and document.get("exact") is exact
            and document.get("freeze") == freeze and document.get("source_pins_unchanged") is True
            and document.get("initialized") is True and document.get("invalid_reason") is None
            and type(document.get("reset_count_after_initialization")) is int
            and document["reset_count_after_initialization"] == 0,
            "original complete unchanged reserved episode required")
    for key, name in (("claim", "claim.json"), ("recording", "tape.npz"), ("journal", "journal.jsonl")):
        require(document[key]["path"] == str(directory / name), "original episode file moved")
    claim = read(document["claim"])
    require(claim.get("schema") == "rage_mill.canonical_reserved_episode_claim.v1"
            and claim.get("stage") == "reserved" and claim.get("seed") == seed
            and claim.get("exact") is exact and claim.get("freeze") == freeze
            and claim.get("identity_sha256") == value_sha256(identity), "original episode claim differs")
    read_pinned(claim["ledger_label_claim"])
    saved = arrays(document["recording"], dict(states=((1501, 38), "float64"),
        actions=((1500,), "float32"), raw_actions=((1500,), "float64"),
        times=((1501,), "float64"), failure_state=((0, 38), "float64")))
    require(same(np.clip(saved["raw_actions"].astype(np.float32), -1., 1.), saved["actions"])
            and np.max(np.abs(saved["times"] - np.arange(1501) * .02)) <= 1e-9,
            "immediate F32 delivery or original native clock differs")
    if exact:
        initial = np.zeros(38); initial[1] = np.pi
        require(same(initial, saved["states"][0]), "exact literal hanging reset differs")
    score = independent_score(saved["states"], saved["actions"])
    require(score == document.get("score")
            and score == score_tape(saved["states"], saved["actions"], validate_identity(identity))
            and score["success"] is True, "actual strict full1500 score differs")
    audit_journal(read_pinned(document["journal"]), **{key: saved[key] for key in
        ("states", "actions", "raw_actions", "times")}, invalid_reason=None)
    return document, saved, score


def proof_episode(row, *, original_row, document, saved, exact, directory):
    proof = read(row)
    require(row["path"] == str(directory / "receipt.json")
            and proof.get("schema") == "rage_mill.canonical18_saved_policy_episode_proof.v1"
            and proof.get("episode") == original_row and proof.get("recording") == document["recording"]
            and proof.get("seed") == document["seed"] and proof.get("passed") is True
            and proof.get("rawF64_commands_byte_proved") == 1500
            and proof.get("deliveredF32_commands_byte_proved") == 1500
            and proof.get("selected_native_next_state_delivery_pairs_byte_proved") == 650
            and proof.get("selected_full651state650action_raw_suffix_byte_identical") is True
            and proof.get("final_state1500_byte_identical") is True
            and type(proof.get("selected_offset_m")) in (int, float) and proof["selected_offset_m"] == 0.,
            "complete original saved policy proof required")
    for key, name in (("recomputed_commands", "recomputed-commands.npz"),
            ("command_journal", "command-proof.jsonl"), ("selected_prediction", "selected-prediction.npz"),
            ("native_value_telemetry", "native-value-telemetry.json")):
        require(proof[key]["path"] == str(directory / name), "saved proof file moved")
    require(proof["native_transition_proof"]["path"] == str(directory / "native-proof/receipt.json"),
            "native proof escaped its original episode")
    native_document = read(proof["native_transition_proof"])
    require(native_document.get("proof_arrays", {}).get("path") == str(directory / "native-proof/native-replay.npz")
            and native_document.get("claim", {}).get("path") == str(directory / "native-proof/claim.json"),
            "native proof arrays/claim escaped their original owner")
    commands = arrays(proof["recomputed_commands"], dict(raw_actions=((1500,), "float64"),
        actions=((1500,), "float32"), selected_next_states=((650, 38), "float64"),
        selected_next_actions=((650,), "float32")))
    require(same(commands["raw_actions"], saved["raw_actions"])
            and same(commands["actions"], saved["actions"])
            and same(commands["selected_next_states"], saved["states"][851:])
            and same(commands["selected_next_actions"], saved["actions"][850:]),
            "all1500 causal raw/F32 or650 selected native pairs differ")
    prediction = arrays(proof["selected_prediction"], dict(states=((651, 38), "float64"),
        actions=((650,), "float32"), raw_actions=((650,), "float64")))
    require(all(same(prediction[key], saved[key][850:]) for key in prediction),
            "full651state650action/raw suffix or final1500 differs")
    journal = [json.loads(line) for line in read_pinned(proof["command_journal"]).splitlines()]
    require(len(journal) == 1500, "1500 saved command proof rows required")
    for tick, line in enumerate(journal):
        require(line == dict(tick=tick, raw_f64_hex=saved["raw_actions"][tick].tobytes().hex(),
            delivered_f32_hex=saved["actions"][tick].tobytes().hex(),
            selected_next_state_byte_parity=None if tick < 850 else True),
            "saved command byte journal differs")
    reports = read(proof["native_value_telemetry"])
    require(set(reports) == {"shadow", "recorded_input"}, "two independent saved histories required")
    shadow, live = telemetry(reports["shadow"]), telemetry(reports["recorded_input"])
    require(reports["shadow"]["steps"] == reports["recorded_input"]["steps"],
            "independent saved shadow/live decision histories differ")
    native = audit_native_replay_proof(proof["native_transition_proof"], episode_receipt=original_row)
    require(native == proof.get("native_transition_summary")
            and native["continuous_replay_identical"] is True
            and native["maximum_continuous_replay_error"] == 0.
            and native["maximum_dynamics_defect"] <= 1e-8
            and (not exact or native["initial_state_gap"] <= 1e-8),
            "saved cold/continuous proof summaries differ")
    counts = dict(policy_calls=1500, shadow_inner_native_candidates=shadow["native_candidate_commands"],
        recorded_input_inner_native_candidates=live["native_candidate_commands"],
        outer_shadow_native_steps=650, saved_native_proof_maps=3000)
    total = sum(value for key, value in counts.items() if key != "policy_calls")
    require(proof.get("cost_counts") == counts and proof.get("total_native_maps") == total <= 24450
            and proof.get("maximum_native_maps") == 24450
            and all(type(proof.get(key)) is int and proof[key] == 0 for key in
                ("physical_episodes", "new_reserved_labels", "reset_RNG_draws", "recorded_reset_reconstructions")),
            "complete saved proof counts/caps or zero-new-attempt scope differ")
    attempt = read(pinned_file(directory / "attempt.json"))
    require(attempt.get("episode") == original_row and attempt.get("seed") == document["seed"]
            and attempt.get("completed_policy_commands") == 1500 and attempt.get("failure") is None
            and attempt.get("cost_counts_at_last_completed_check") == counts
            and attempt.get("live_native_value_telemetry") == reports["recorded_input"]
            and attempt.get("copy_native_value_telemetry") == [reports["shadow"]]
            and all(type(attempt.get(key)) is int and attempt[key] == 0 for key in
                ("physical_episodes", "new_reserved_labels", "reset_RNG_draws")),
            "complete saved proof attempt differs")
    return dict(seed=document["seed"], proof=row, native_transition_summary=native,
        shadow=shadow, recorded_input=live, cost_counts=counts, total_native_maps=total,
        rawF64_commands_byte_proved=1500, deliveredF32_commands_byte_proved=1500,
        selected_native_pairs_byte_proved=650, full_suffix_and_final1500_byte_identical=True)


def audit(args):
    identity_row, authority_row = pinned_file(args.identity), pinned_file(args.authority)
    require(identity_row["sha256"] == args.expected_identity_sha256
            and authority_row["sha256"] == args.expected_authority_sha256, "prospective identity/authority hashes required")
    identity, authority = read(identity_row), read(authority_row)
    require(validate_identity(identity).n_links == 18
            and authority.get("schema") == "rage_mill.canonical_reserved_execution_authority.v1"
            and authority.get("identity_sha256") == value_sha256(identity)
            and authority.get("namespace", {}).get("n_links") == 18,
            "original own-count identity/authority required")
    runtime, freeze = authority["runtime_closure"], authority["freeze"]
    require(read(freeze).get("identity") == identity, "original freeze identity differs")
    details = reserved_gates(18)[args.gate]
    launch, actual_owner = owner(args.actual_owner, args.expected_actual_sha256, identity=identity,
        runtime=runtime, worker_name="run_canonical_reserved_gate.py", gate=args.gate, authority_row=authority_row)
    output = Path(authority["output_root"]) / args.gate
    require(launch.get("mode") == "reserved" and launch.get("output_path") == str(output)
            and launch.get("gate_details") == details and launch.get("new_reserved_labels") == details["episodes"]
            and launch.get("physical_episodes") == details["episodes"]
            and launch.get("identity") == identity_row
            and launch.get("identity_sha256") == value_sha256(identity)
            and all(type(launch.get(key)) is int and launch[key] == value
                    for key, value in caps("reserved", details["episodes"]).items())
            and launch["argv"] == [launch["argv"][0], launch["argv"][1], "--authority", authority_row["path"],
                "--expected-authority-sha256", authority_row["sha256"], "--gate", args.gate],
            "original actual owner gate/output/caps differ")
    cohort_row = pinned_file(output / "receipt.json"); cohort = read(cohort_row)
    require(cohort.get("schema") == "rage_mill.canonical_cohort_evidence.v1"
            and cohort.get("stage") == "reserved" and cohort.get("identity") == identity
            and cohort.get("identity_sha256") == value_sha256(identity) and cohort.get("freeze") == freeze
            and all(cohort.get(key) == value for key, value in details.items()), "original complete cohort differs")
    rows = cohort.get("episode_receipts")
    require(isinstance(rows, list) and len(rows) == details["episodes"]
            and len({row["path"] for row in rows}) == len(rows), "distinct original episode rows required")
    _audit_storage(authority_row, authority, validate_identity(identity), args.gate, cohort_row)
    proofs, proof_owner, proof_cohort_row, proof_cohort = None, None, None, None
    if args.proof_owner is not None:
        require(args.expected_proof_sha256 is not None, "prospective saved proof owner hash required")
        proof_launch, proof_owner = owner(args.proof_owner, args.expected_proof_sha256, identity=identity,
            runtime=runtime, worker_name="audit_canonical18_saved_policy.py", gate=args.gate, authority_row=authority_row)
        proof_output = Path(proof_launch["output_path"])
        require(proof_output == Path(args.proof_owner).resolve() / "result"
                and proof_launch.get("mode") == "proof"
                and proof_launch.get("gate_details") == details
                and proof_launch.get("physical_episodes") == proof_launch.get("new_reserved_labels") == 0
                and proof_launch.get("original_saved_episodes") == len(rows)
                and proof_launch.get("identity") == identity_row
                and proof_launch.get("identity_sha256") == value_sha256(identity)
                and all(type(proof_launch.get(key)) is int and proof_launch[key] == value
                    for key, value in caps("proof", len(rows)).items())
                and all(type(proof_launch.get(key)) is int and proof_launch[key] == 0 for key in
                    ("maximum_reset_RNG_draws", "maximum_recorded_reset_reconstructions",
                     "maximum_physical_resets", "new_reserved_claims"))
                and proof_launch["argv"] == [proof_launch["argv"][0], proof_launch["argv"][1],
                    "--identity", identity_row["path"], "--expected-identity-sha256", identity_row["sha256"],
                    "--cohort", cohort_row["path"], "--expected-cohort-sha256", cohort_row["sha256"],
                    "--runtime-closure", runtime["path"], "--out", str(proof_output)],
                "saved proof owner output/scope/caps differ")
        proof_cohort_row = pinned_file(proof_output / "receipt.json"); proof_cohort = read(proof_cohort_row)
        require(proof_cohort.get("schema") == "rage_mill.canonical18_saved_policy_cohort_proof.v1"
                and proof_cohort.get("identity") == identity_row and proof_cohort.get("cohort") == cohort_row
                and proof_cohort.get("gate") == args.gate and proof_cohort.get("passed") is True,
                "original saved proof cohort lineage differs")
        proofs = proof_cohort.get("episode_proofs")
        require(isinstance(proofs, list) and len(proofs) == len(rows)
                and len({row["path"] for row in proofs}) == len(proofs), "one saved proof per original episode required")
        audit_runtime_execution(pinned_file(proof_output / "runtime-receipt.json"), identity=identity,
            runtime_closure=runtime, outputs=[proof_cohort_row])
    results, saved_proofs = [], []
    for index, row in enumerate(rows):
        seed = details["seed_start"] + index
        document, saved, score = episode(row, identity=identity, seed=seed, exact=details["exact"],
            freeze=freeze, directory=output / f"episode-{seed}")
        results.append(dict(seed=seed, episode=row, recording=document["recording"], score=score))
        if proofs is not None:
            saved_proofs.append(proof_episode(proofs[index], original_row=row, document=document,
                saved=saved, exact=details["exact"], directory=proof_output / f"episode-{seed}"))
    total = sum(item["total_native_maps"] for item in saved_proofs)
    if proof_cohort is not None:
        require(proof_cohort.get("episodes_proved") == len(rows)
                and proof_cohort.get("rawF64_commands_byte_proved") == 1500 * len(rows)
                and proof_cohort.get("deliveredF32_commands_byte_proved") == 1500 * len(rows)
                and proof_cohort.get("all_fullsuffix_and_finalstate1500_byte_identical") is True
                and proof_cohort.get("total_native_maps") == total <= 24450 * len(rows)
                and proof_cohort.get("maximum_native_maps") == 24450 * len(rows)
                and all(type(proof_cohort.get(key)) is int and proof_cohort[key] == 0 for key in
                    ("physical_episodes", "new_reserved_labels", "reset_RNG_draws", "recorded_reset_reconstructions")),
                "cohort saved proof totals/caps differ")
    return dict(schema="rage_mill.canonical18_independent_reserved_saved_audit.v1", passed=True,
        identity=identity_row, authority=authority_row, gate=args.gate, cohort=cohort_row,
        actual_owner=actual_owner, episodes=results, all_actual1500_strict_pass=True,
        saved_proof_owner=proof_owner, saved_proof_cohort=proof_cohort_row, saved_episode_proofs=saved_proofs,
        saved_proof_total_native_maps=total, saved_proof_maximum_native_maps=24450 * len(saved_proofs),
        baseline_delivery_agreements=sum(p[key]["baseline_delivery_agreements"] for p in saved_proofs
            for key in ("shadow", "recorded_input")),
        baseline_delivery_disagreements=sum(p[key]["baseline_delivery_disagreements"] for p in saved_proofs
            for key in ("shadow", "recorded_input")), baseline_agreement_is_qualification_gate=False,
        limitations="No retained original live candidate ledger or full saved-proof alternative ledger. Counts and retained-baseline objectives audit saved telemetry; no alternative dynamics/objectives regenerated. No noisy reset RNG reconstruction.",
        additional_native_calls=0, additional_controller_calls=0, additional_RNG_draws=0,
        new_reserved_labels=0, physical_episodes=0, controller_qualified=False)


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    for name in ("identity", "authority", "actual-owner", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-identity-sha256", "expected-authority-sha256", "expected-actual-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gate", choices=tuple(reserved_gates(18)), required=True)
    parser.add_argument("--proof-owner", type=Path)
    parser.add_argument("--expected-proof-sha256")
    args = parser.parse_args()
    wall, cpu = time.monotonic(), time.process_time()
    # Catch accidental execution in this additive saved-only auditor, including
    # any future change in imported helpers. Imports themselves construct none.
    import rage_mill.canonical_benchmark as benchmark
    import rage_mill.canonical_replay as replay
    def forbidden(*a, **kw):
        raise AssertionError("saved-only independent auditor forbids model/native/reset/RNG execution")
    benchmark.CanonicalEnv = replay.CanonicalEnv = forbidden
    benchmark.hanging_state = np.random.default_rng = forbidden
    import mujoco
    mujoco.MjModel = mujoco.MjData = forbidden
    mujoco.mj_step = mujoco.mj_forward = forbidden
    result = audit(args)
    result.update(producer_source=pinned_file(__file__), CPU_seconds=time.process_time() - cpu,
                  wall_seconds=time.monotonic() - wall)
    print(json.dumps(write_json_exclusive(args.out, result)), flush=True)


if __name__ == "__main__":
    main()
