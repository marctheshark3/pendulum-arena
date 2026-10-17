#!/usr/bin/env python3
"""Independently audit one saved eighteen-link actual diagnostic and its ledger.

Every candidate objective is recomputed from the saved own-count reference,
affine value and candidate next state. No model, policy, factory, inference,
reset generator or native transition is invoked. Candidate next states are
retained evidence; this audit does not independently regenerate alternatives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy(order="C") for key in saved.files}


def finite_array(arrays, key, shape, dtype):
    value = arrays[key]
    require(value.shape == shape and value.dtype == np.dtype(dtype)
            and np.isfinite(value).all(), "finite own-count array differs: " + key)
    return value


def independent_score(states, actions):
    cumulative = np.cumsum(states[1:, 1:19], axis=1)
    angles = np.where((cumulative >= -np.pi) & (cumulative < np.pi), cumulative,
                      (cumulative + np.pi) % (2 * np.pi) - np.pi)
    longest = streak = 0
    for upright in np.all(np.abs(angles) < .15, axis=1):
        streak = streak + 1 if upright else 0
        longest = max(longest, streak)
    finite = bool(np.isfinite(states).all() and np.isfinite(actions).all())
    force = bool(np.all(np.abs(actions) <= 1.))
    delivery = bool(same(actions, actions.astype(np.float32).astype(actions.dtype)))
    rail = bool(np.all(np.abs(states[:, 0]) <= 3.))
    complete = len(actions) == 1500 and states.shape == (1501, 38)
    return dict(success=bool(finite and force and delivery and rail and complete and longest >= 250),
                complete=complete, finite=finite, force_ok=force, float32_delivery_ok=delivery,
                rail_ok=rail, ticks=len(actions), longest_upright_ticks=longest,
                longest_upright_seconds=longest * .02,
                maximum_cart_m=float(np.max(np.abs(states[:, 0]))), invalid_reason=None)


def audit_runtime_coverage(coverage, *, runtime_path, runtime_pins, worker_path):
    actual = coverage.get("actual_loaded_file_sha256")
    require(coverage.get("runtime_closure") == runtime_path and isinstance(actual, dict) and actual
            and all(Path(path).is_absolute() and str(Path(path).resolve(strict=True)) == path
                    and sha(path) == digest for path, digest in actual.items()),
            "actual diagnostic loaded runtime rows or original manifest path differ")
    missing = {path: digest for path, digest in actual.items() if runtime_pins.get(path) != digest}
    require(missing == coverage.get("uncovered_or_changed") == {}
            and actual.get(worker_path) == runtime_pins.get(worker_path)
            and worker_path in actual
            and coverage.get("all_runtime_pins_unchanged") is True
            and all(sha(path) == digest for path, digest in runtime_pins.items())
            and all(type(coverage.get(key)) is int and coverage[key] == 0 for key in ("native_commands", "reset_draws")),
            "actual diagnostic runtime coverage or unchanged complete pins failed")
    return dict(actual_loaded_files=len(actual), uncovered_or_changed={}, all_runtime_pins_unchanged=True)


def audit_arrays(*, controller, reference, feedback, tape, ledger, case):
    """Recompute saved-only arithmetic and first-minimum/selected-pair lineage."""
    from rage_mill.canonical_benchmark import CanonicalProtocol, score_tape

    shapes = dict(states=(1501, 38), actions=(1500,), raw_actions=(1500,),
                  selected_predicted_states=(650, 38), selected_predicted_actions=(650,),
                  selected_episode_ticks=(650,), selected_indices=(650,), baseline_indices=(650,),
                  unique_counts=(650,), baseline_delivery_agreement=(650,), ordinary_raw_actions=(1500,))
    require(set(tape) == set(shapes), "complete actual diagnostic fields required")
    integer = {"selected_episode_ticks", "selected_indices", "baseline_indices", "unique_counts"}
    for key, shape in shapes.items():
        dtype = ("int64" if key in integer else "bool" if key == "baseline_delivery_agreement"
                 else "float32" if key == "selected_predicted_actions" else "float64")
        finite_array(tape, key, shape, dtype)
    states, actions, raw = (tape[key] for key in ("states", "actions", "raw_actions"))
    initial = np.zeros(38); initial[1] = np.pi
    require(same(initial, states[0]) and same(actions, np.clip(raw.astype(np.float32), -1., 1.).astype(np.float64)),
            "literal exact hanging or immediate raw/F32 delivery differs")
    score = independent_score(states, actions)
    require(score == score_tape(states, actions, CanonicalProtocol(n_links=18)) and score["success"]
            and all(case.get(key) == value for key, value in score.items()),
            "complete strict250 post-command physical score differs")

    controller_shapes = dict(states=(651, 38), actions=(650,), gains=(650, 38), offsets=(650,),
                             native_defects=(650, 38), balance_gain=(38,), park_gain=(38,),
                             parking_target=(38,), terminal_root=(38, 38), park_ticks=())
    reference_shapes = dict(states=(701, 38), actions=(700,), gains=(700, 38), offsets=(700,),
                            native_defects=(700, 38))
    feedback_shapes = dict(gains=(700, 38), offsets=(700,), roots=(701, 38, 38),
                           residuals=(701, 38), Q=(38, 38))
    for tables, expected in ((controller, controller_shapes), (reference, reference_shapes), (feedback, feedback_shapes)):
        require(set(tables) == set(expected), "complete own-count reference/controller/value fields required")
        for key, shape in expected.items():
            finite_array(tables, key, shape, "int64" if key == "park_ticks" else "float64")
    require(controller["park_ticks"].item() == 850 and not np.any(controller["terminal_root"])
            and not np.any(feedback["roots"][-1]) and not np.any(feedback["residuals"][-1]),
            "literal850 parking and full700 zero-terminal value required")
    for key in ("states", "actions", "gains", "offsets", "native_defects"):
        require(same(controller[key], reference[key][:len(controller[key])]),
                "ordinary650 must be the literal full700 prefix: " + key)
    for key in ("gains", "offsets"):
        require(same(feedback[key], reference[key]), "full affine feedback/reference bytes differ: " + key)
    require(np.all(np.abs(reference["actions"]) <= 1.)
            and same(reference["actions"], reference["actions"].astype(np.float32).astype(np.float64)),
            "full700 reference commands must be delivered float32 values")
    cumulative = np.tril(np.ones((18, 18))); gram = cumulative.T @ cumulative
    q = np.eye(38); q[0, 0] = q[19, 19] = .1
    q[1:19, 1:19] += 100. * gram; q[20:, 20:] = gram + .01 * np.eye(18)
    require(same(feedback["Q"], q), "literal own18 physical-state cost differs")

    # Recalculate the scalar ordinary formula directly; never instantiate or
    # call the controller class. Its NumPy dot/subtraction order is preserved.
    ordinary = np.empty(1500, dtype=np.float64)
    cart_offset = states[850, 0] - reference["states"][0, 0]
    for tick in range(1500):
        if tick < 850:
            ordinary[tick] = -float(np.dot(controller["park_gain"], states[tick] - controller["parking_target"]))
        else:
            route_tick = tick - 850
            centered = states[tick].copy(); centered[0] -= cart_offset
            ordinary[tick] = (controller["actions"][route_tick] - controller["offsets"][route_tick]
                              - float(np.dot(controller["gains"][route_tick], centered - controller["states"][route_tick])))
    require(same(ordinary, tape["ordinary_raw_actions"]) and same(ordinary[:850], raw[:850]),
            "saved ordinary formula or first850 parked raw commands differ")
    ticks = np.arange(850, 1500, dtype=np.int64)
    require(same(tape["selected_episode_ticks"], ticks)
            and same(tape["selected_predicted_states"], states[851:])
            and same(tape["selected_predicted_actions"], actions[850:].astype(np.float32)),
            "all650 selected next-state/delivery pairs including final1500 must be byte-exact")

    require(set(ledger) == {"ticks", "commands", "costs", "states", "value_residuals", "errors"},
            "complete diagnostic candidate ledger fields required")
    count = len(ledger["ticks"])
    require(650 <= count <= 10400, "finite650-by16 native candidate cap required")
    for key, shape, dtype in (("ticks", (count,), "int64"), ("commands", (count,), "float32"),
                              ("costs", (count,), "float64"), ("states", (count, 38), "float64"),
                              ("value_residuals", (count, 38), "float64")):
        finite_array(ledger, key, shape, dtype)
    require(ledger["errors"].shape == (count,) and ledger["errors"].dtype.kind == "U"
            and np.all(ledger["errors"] == "") and np.all(ledger["costs"] >= 0.),
            "all actual diagnostic candidate objectives must be finite and error-free")
    report = case.get("native_value_telemetry")
    require(isinstance(report, dict) and isinstance(report.get("steps"), list)
            and len(report["steps"]) == 650, "full650 actual telemetry required")
    consumed = agreements = maximum = duplicates = ratio_count = 0
    ratio_sum = 0.; ratio_min = ratio_max = None
    for route_tick, step in enumerate(report["steps"]):
        group = np.flatnonzero(ledger["ticks"] == route_tick)
        require(1 <= len(group) <= 16 and np.array_equal(group, np.arange(consumed, consumed + len(group))),
                "candidate clock, contiguous block or sixteen cap differs")
        commands = ledger["commands"][group]
        require(np.all(np.abs(commands) <= 1.) and len({value.tobytes() for value in commands}) == len(group),
                "delivered candidate float32 commands must be bounded and distinct")
        baseline = int(tape["baseline_indices"][route_tick]); selected = int(tape["selected_indices"][route_tick])
        require(baseline == group[0] and selected == min(group, key=lambda row: ledger["costs"][row]),
                "baseline first or selected first minimum saved objective differs")
        require(tape["unique_counts"][route_tick] == len(group), "saved candidate block count differs")
        error = states[850 + route_tick] - reference["states"][route_tick]
        error[0] -= cart_offset
        expected_baseline = np.float32(np.clip(np.float32(float(reference["actions"][route_tick])
                            - float(feedback["gains"][route_tick] @ error)
                            - float(feedback["offsets"][route_tick])), -1., 1.))
        require(expected_baseline.tobytes() == ledger["commands"][baseline].tobytes(),
                "first candidate differs from independently calculated affine baseline")
        ordinary_delivery = np.float32(np.clip(np.float32(ordinary[850 + route_tick]), -1., 1.))
        agreement = ordinary_delivery.tobytes() == ledger["commands"][baseline].tobytes()
        require(bool(tape["baseline_delivery_agreement"][route_tick]) is agreement,
                "saved ordinary/baseline delivery agreement flag differs")
        agreements += int(agreement)
        require(same(ledger["states"][selected], states[851 + route_tick])
                and ledger["commands"][selected].tobytes() == np.float32(actions[850 + route_tick]).tobytes()
                and np.float64(float(ledger["commands"][selected])).tobytes() == raw[850 + route_tick].tobytes(),
                "selected candidate state/raw/F32 delivery differs from actual tape")
        for row in group:
            next_error = ledger["states"][row] - reference["states"][route_tick + 1]
            next_error[0] -= cart_offset
            residual = np.asarray([math.fsum(float(a) * float(b) for a, b in zip(weights, next_error, strict=True))
                                   + float(affine) for weights, affine in zip(feedback["roots"][route_tick + 1],
                                                feedback["residuals"][route_tick + 1], strict=True)])
            cost = math.fsum([1000. * (float(ledger["commands"][row]) - float(reference["actions"][route_tick])) ** 2]
                             + [float(value) * float(value) for value in residual])
            require(same(residual, ledger["value_residuals"][row])
                    and np.float64(cost).tobytes() == ledger["costs"][row].tobytes(),
                    "independent saved candidate affine residual/objective bytes differ")
        selected_cost, baseline_cost = float(ledger["costs"][selected]), float(ledger["costs"][baseline])
        require(type(step.get("duplicate_proposals")) is int and step["duplicate_proposals"] >= 0
                and step.get("episode_tick") == route_tick + 850 and step.get("route_tick") == route_tick
                and step.get("selected_index") == selected and step.get("baseline_index") == baseline
                and step.get("unique_forecasts") == len(group)
                and step.get("selected_cost") == selected_cost and step.get("baseline_cost") == baseline_cost
                and step.get("baseline_delivery_agreement") is agreement
                and step.get("previous_selected_state_byte_parity") is (None if route_tick == 0 else True)
                and selected_cost <= baseline_cost, "actual per-command telemetry differs from saved candidate block")
        consumed += len(group); maximum = max(maximum, len(group)); duplicates += step["duplicate_proposals"]
        if baseline_cost > 0.:
            ratio = selected_cost / baseline_cost
            ratio_count += 1; ratio_sum += ratio
            ratio_min = ratio if ratio_min is None else min(ratio_min, ratio)
            ratio_max = ratio if ratio_max is None else max(ratio_max, ratio)
    expected = dict(native_candidate_commands=count, active_commands=650, max_candidates_per_active_command=maximum,
                    duplicate_proposals=duplicates, native_forecast_errors=0,
                    baseline_delivery_agreements=agreements, baseline_delivery_disagreements=650 - agreements,
                    selected_to_baseline_ratio_count=ratio_count, selected_to_baseline_ratio_sum=ratio_sum,
                    selected_to_baseline_ratio_min=ratio_min, selected_to_baseline_ratio_max=ratio_max,
                    selected_to_baseline_ratio_mean=None if not ratio_count else ratio_sum / ratio_count)
    require(consumed == count and all(report.get(key) == value for key, value in expected.items())
            and case.get("native_candidate_commands") == count and case.get("selected_native_states") == 650
            and case.get("maximum_candidates_per_active_command") == maximum
            and case.get("all_selected_native_states_byte_exact") is True
            and case.get("baseline_delivery_disagreements") == 650 - agreements,
            "actual candidate/selected-pair count or cached telemetry differs")
    for key in ("native_select_CPU_seconds", "native_select_wall_seconds"):
        value = report.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0., "finite native selection cost required")
    return dict(score=score, candidate_objectives_independently_recomputed=count,
                maximum_value_residual_error=0., maximum_cost_error=0.,
                every_selected_candidate_is_first_minimum_saved_cost=True,
                all650_selected_state_and_float32_delivery_pairs_byte_exact=True,
                final_selected_transition_byte_exact=True, literal_ordinary_first850_formula_byte_exact=True,
                ordinary_all1500_raw_formula_byte_exact=True,
                baseline_float32_delivery_disagreements=650 - agreements,
                native_candidate_commands=count, native_value_counts=expected,
                original_total_native_commands=count + 1500, original_native_ceiling=11900)


def verify(inputs):
    """Audit original-path closure and direct saved inputs, without numerical execution."""
    from rage_mill.canonical_runtime_manifest import validate_runtime_manifest

    pins = inputs["source_hashes"]
    require(isinstance(pins, dict) and pins and all(sha(path) == digest for path, digest in pins.items()),
            "prospectively pinned saved inputs changed")
    owner = Path(inputs["episode_owner"])
    require(owner.is_absolute() and owner.resolve(strict=True) == owner, "original absolute diagnostic owner required")
    launch_path, completion_path = owner / "launch.json", owner / "completion.json"
    launch, completion = (json.loads(path.read_text()) for path in (launch_path, completion_path))
    source_pins, unchanged = launch.get("source_hashes"), completion.get("source_hashes_unchanged")
    require(completion.get("naturally_closed_zero") is True and completion.get("child_returncode") == 0
            and completion.get("recipe_path") == str(launch_path) and completion.get("recipe_sha256") == sha(launch_path)
            and completion.get("recipe_unchanged") is True and completion.get("failure") is None
            and completion.get("interrupted") is False and completion.get("wall_timeout") is False
            and completion.get("owned_group_cleanup") and isinstance(source_pins, dict) and source_pins
            and isinstance(unchanged, dict) and set(unchanged) == set(source_pins)
            and all(value is True for value in unchanged.values())
            and all(sha(path) == digest for path, digest in source_pins.items()),
            "actual diagnostic owner did not naturally close with unchanged complete pins")
    seed = inputs.get("seed", 3181040)
    require(type(seed) is int and 3181040 <= seed <= 3181199
            and launch.get("schema") == "rage_mill.canonical18_diagnostic_owner.v1"
            and launch.get("n_links") == 18 and launch.get("seed") == seed and launch.get("exact") is True
            and launch.get("CPU_cap_s") == 90 and launch.get("wall_cap_s") == 150
            and launch.get("physical_episodes") == 1 and launch.get("formal_development_episodes") == 0
            and launch.get("reserved_episodes") == 0
            and launch.get("maximum_actual_commands") == 1500
            and launch.get("maximum_native_candidates") == 10400
            and launch.get("maximum_total_native_commands") == 11900,
            "own-count prospective exact diagnostic/caps differ")
    output = owner / "output"
    require(launch.get("output_path") == str(output), "diagnostic output escaped its original owner")
    original = json.loads((output / "receipt.json").read_text())
    require(original.get("schema") == "rage_mill.canonical18_native_value_hanging_diagnostic.v1"
            and original.get("n_links") == 18 and original.get("all_success") is True
            and original.get("active_route_start") == 0 and original.get("cart_error_offset_m") == 0.
            and len(original.get("cases", [])) == 1, "one successful own-count activation0/offset0 diagnostic required")
    case = original["cases"][0]
    require(case.get("seed") == seed and case.get("exact") is True
            and case.get("recording_path") == str(output / f"episode-{seed}.npz")
            and case.get("candidate_ledger_path") == str(output / f"episode-{seed}-candidate-ledger.npz")
            and sha(case["recording_path"]) == case.get("recording_sha256")
            and sha(case["candidate_ledger_path"]) == case.get("candidate_ledger_sha256"),
            "original exact diagnostic tape/ledger lineage differs")
    for key in ("controller", "reference", "feedback"):
        path = str(Path(inputs[key]).resolve(strict=True))
        require(path == inputs[key] and original.get(key + "_sha256") == sha(path)
                and source_pins.get(path) == sha(path) and pins.get(path) == sha(path),
                "original diagnostic source table pin differs: " + key)
    argv = launch.get("argv")
    require(isinstance(argv, list) and len(argv) >= 2
            and Path(argv[0]).is_absolute() and Path(argv[0]).is_file()
            and Path(argv[1]).name == "probe_canonical18_native_value_episode.py"
            and source_pins.get(argv[1]) == sha(argv[1]),
            "original own-count source-bound diagnostic worker required")
    runtime = launch["runtime_closure"]
    expected_argv = [argv[0], argv[1], "--out", str(output), "--active-route-start", "0",
                     "--cart-error-offset", "0.", "--seed", str(seed), "--episodes", "1", "--exact",
                     "--runtime-closure", runtime["path"]]
    for key in ("controller", "reference", "feedback"):
        expected_argv.extend(["--" + key, inputs[key]])
    require(argv == expected_argv, "original exact diagnostic worker arguments differ")
    for path in (launch_path, completion_path, output / "receipt.json", Path(case["recording_path"]), Path(case["candidate_ledger_path"])):
        require(pins.get(str(path)) == sha(path), "prospective saved audit evidence pin missing")
    require(sha(runtime["path"]) == runtime["sha256"], "original diagnostic runtime manifest changed")
    runtime_pins = validate_runtime_manifest(json.loads(Path(runtime["path"]).read_text()), require_complete=True)
    require(all(source_pins.get(path) == digest for path, digest in runtime_pins.items()),
            "original diagnostic owner did not bind complete prepared runtime")
    coverage_row = original.get("runtime_coverage")
    coverage_path = output / "runtime-coverage.json"
    require(isinstance(coverage_row, dict) and coverage_row.get("path") == str(coverage_path)
            and coverage_row.get("sha256") == sha(coverage_path)
            and pins.get(str(coverage_path)) == sha(coverage_path),
            "original diagnostic actual runtime coverage must be prospectively pinned")
    coverage = audit_runtime_coverage(json.loads(coverage_path.read_text()), runtime_path=runtime["path"],
                                     runtime_pins=runtime_pins, worker_path=argv[1])
    log = owner / "worker.log"
    require(completion.get("log_path") == str(log) and completion.get("log_sha256") == sha(log),
            "original diagnostic owner log changed")
    result = audit_arrays(controller=load(inputs["controller"]), reference=load(inputs["reference"]),
                          feedback=load(inputs["feedback"]), tape=load(case["recording_path"]),
                          ledger=load(case["candidate_ledger_path"]), case=case)
    report = case["native_value_telemetry"]
    require(report["native_select_CPU_seconds"] <= completion["aggregate_child_CPU_seconds"] + 1e-6
            and report["native_select_wall_seconds"] <= completion["wall_seconds"] + 1e-6,
            "reported native selection cost escaped the actual owner cost")
    require(all(sha(path) == digest for path, digest in pins.items()) and result["original_total_native_commands"] <= 11900,
            "saved audit inputs changed or original native cap exceeded")
    return dict(schema="rage_mill.canonical18_native_value_episode_saved_independent_audit.v1",
                passed=True, n_links=18, seed=seed, exact=True, **result,
                original_owner_CPU_seconds=completion["aggregate_child_CPU_seconds"],
                original_owner_wall_seconds=completion["wall_seconds"],
                original_owner_peak_rss_kib=completion["peak_rss_kib"],
                original_runtime_pins_unchanged=True, original_owner_naturally_closed_zero=True,
                runtime_coverage=coverage, runtime_coverage_path=str(coverage_path),
                source_hashes_unchanged={path: True for path in pins},
                native_calls=0, controller_calls=0, inference_calls=0, reset_draws=0,
                new_physical_episodes=0, development_episodes=0, reserved_episodes=0, controller_qualified=False,
                scope="Saved actual exact diagnostic only; fresh fixed-policy qualification and native replay remain required",
                limitations="Alternative next states are saved ledger evidence, not regenerated native transitions. Full700 Riccati recursion is not rederived.")


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--expected-inputs-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.run and sha(args.inputs) == args.expected_inputs_sha256 and not args.out.exists(),
            "prospectively pinned saved-only audit required")
    cpu, wall = time.process_time(), time.monotonic()
    inputs = json.loads(args.inputs.read_text())
    inputs["source_hashes"][str(args.inputs.resolve())] = args.expected_inputs_sha256
    result = verify(inputs)
    result.update(CPU_seconds=time.process_time() - cpu, wall_seconds=time.monotonic() - wall,
                  producer_source_sha256=sha(__file__))
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "receipt.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({key: result[key] for key in ("passed", "score", "native_candidate_commands", "native_calls")}))


if __name__ == "__main__":
    main()
