"""Replay every original own18 DEV transition without policy or reset RNG.

The unchanged candidate identity remains authoritative. This independently
owned auditor freezes its own additional source/runtime closure; it never
extends or rewrites the candidate identity. Continuous and cold native replay
start exclusively from saved original states/actions, with no fresh episodes.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

from rage_mill.canonical_benchmark import score_tape
from rage_mill.canonical_evaluation import (
    audit_journal, is_reserved, pinned_file, read_pinned, require,
    validate_identity, value_sha256, write_json_exclusive,
)
from rage_mill.canonical_replay import audit_native_replay_proof, prove_recorded_native_transitions
from rage_mill.canonical_runtime_closure import runtime_binding_paths
from rage_mill.canonical_runtime_evidence import audit_runtime_execution
from rage_mill.canonical_runtime_manifest import create_runtime_manifest, validate_runtime_manifest

LIMITS = {"noisy20": (20, 120, 180), "noisy100": (100, 600, 900)}


def read(row):
    return json.loads(read_pinned(row))


def same(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def saved_episode(row, identity, *, seed, directory):
    """Validate the actual original tape/journal, never reconstruct noisy RNG."""
    episode = read(row)
    protocol = validate_identity(identity)
    require(protocol.n_links == 18 and row["path"] == str(directory / "receipt.json")
        and episode.get("schema") == "rage_mill.canonical_episode_evidence.v1"
        and episode.get("identity") == identity
        and episode.get("identity_sha256") == value_sha256(identity)
        and episode.get("stage") == "development" and type(episode.get("seed")) is int
        and episode["seed"] == seed and episode.get("exact") is False
        and episode.get("initialized") is True and episode.get("invalid_reason") is None
        and episode.get("source_pins_unchanged") is True
        and type(episode.get("reset_count_after_initialization")) is int
        and episode["reset_count_after_initialization"] == 0,
        "complete unchanged original own18 noisy development episode required")
    for key, name in (("claim", "claim.json"), ("recording", "tape.npz"), ("journal", "journal.jsonl")):
        require(episode[key]["path"] == str(directory / name), "original episode evidence escaped owner")
    claim = read(episode["claim"])
    require(claim.get("schema") == "rage_mill.canonical_development_episode_claim.v1"
        and claim.get("stage") == "development" and claim.get("seed") == seed
        and claim.get("exact") is False and claim.get("identity_sha256") == value_sha256(identity),
        "original development episode claim differs")
    with np.load(io.BytesIO(read_pinned(episode["recording"])), allow_pickle=False) as saved:
        fields = {key: saved[key].copy() for key in saved.files}
    shapes = dict(states=(1501, 38), actions=(1500,), raw_actions=(1500,),
                  times=(1501,), failure_state=(0, 38))
    require(set(fields) == set(shapes), "complete original native tape fields required")
    for key, shape in shapes.items():
        value = fields[key]
        require(value.shape == shape and value.dtype == np.dtype("float32" if key == "actions" else "float64")
            and np.isfinite(value).all(), "complete finite own38 saved field required: " + key)
    require(same(np.clip(fields["raw_actions"].astype(np.float32), -1., 1.), fields["actions"])
        and np.max(np.abs(fields["times"] - np.arange(1501) * .02)) <= 1e-9,
        "original immediate delivered F32 commands or 30-second clock differ")
    score = score_tape(fields["states"], fields["actions"], protocol)
    require(score == episode.get("score") and score["success"] is True,
            "original complete strict250-post-command score required")
    audit_journal(read_pinned(episode["journal"]), **{key: fields[key] for key in
        ("states", "actions", "raw_actions", "times")}, invalid_reason=None)
    return episode, fields


def original_owner(directory, identity_row, mode):
    """Recheck closed actual owner/runtime, without policy or reset generators."""
    directory = directory.resolve(strict=True)
    require(directory.is_absolute() and directory == Path(str(directory)), "original absolute owner required")
    identity = read(identity_row)
    require(validate_identity(identity).n_links == 18, "own-count eighteen identity required")
    launch_row, completion_row = (pinned_file(directory / name) for name in ("launch.json", "completion.json"))
    launch, completion = read(launch_row), read(completion_row)
    require(completion.get("schema") == "rage_mill.next_stage_supervised_completion.v1"
        and completion.get("recipe_path") == launch_row["path"]
        and completion.get("recipe_sha256") == launch_row["sha256"]
        and type(completion.get("child_returncode")) is int and completion["child_returncode"] == 0
        and completion.get("naturally_closed_zero") is True and completion.get("failure") is None
        and completion.get("interrupted") is False and completion.get("wall_timeout") is False
        and completion.get("recipe_unchanged") is True and completion.get("owned_group_cleanup"),
        "naturally closed and reaped original actual development owner required")
    pins, unchanged = launch.get("source_hashes"), completion.get("source_hashes_unchanged")
    require(isinstance(pins, dict) and pins and isinstance(unchanged, dict)
        and set(pins) == set(unchanged) and all(flag is True for flag in unchanged.values())
        and all(pinned_file(path)["sha256"] == digest for path, digest in pins.items()),
        "all original owner source/runtime pins must remain unchanged")
    log = pinned_file(directory / "worker.log")
    require(completion.get("log_path") == log["path"] and completion.get("log_sha256") == log["sha256"],
            "original owner log differs")
    output = directory / "result"
    require(launch.get("output_path") == str(output)
        and str(Path(launch["argv"][0]).resolve(strict=True)) == identity["runtime"]["executable"],
        "original owner output or Python launcher differs")
    worker = [row for row in identity["sources"]
              if Path(row["path"]).name == "evaluate_canonical18_profile_candidate.py"]
    require(len(worker) == 1 and launch["argv"][1] == worker[0]["path"],
            "unique frozen original development evaluator required")
    component_row = pinned_file(output / "receipt.json")
    component = read(component_row)
    require(component.get("schema") == "rage_mill.canonical_fixed_candidate_component.v1"
        and component.get("mode") == mode and all(component.get(key) is True for key in
            ("scientific_pass", "all_physical_pass", "all_shadow_live_parity")),
        "complete passing original development component required")
    declaration = read(identity["policy_parameters"]["design_metadata"]["declaration"])
    entry = declaration["development20" if mode == "noisy20" else "development100"]
    count = LIMITS[mode][0]
    seed = entry["seed_start"]
    require(declaration.get("n_links") == 18 and type(seed) is int and entry["episodes"] == count
        and entry["exact"] is False and not any(is_reserved(value) for value in range(seed, seed+count)),
        "fixed own-count nonreserved declared development cohort required")
    cohort_row = component["cohort"]
    cohort = read(cohort_row)
    require(cohort_row["path"] == str(output / "episodes/receipt.json")
        and cohort.get("schema") == "rage_mill.canonical_cohort_evidence.v1"
        and cohort.get("identity") == identity and cohort.get("identity_sha256") == value_sha256(identity)
        and cohort.get("stage") == "development" and cohort.get("seed_start") == seed
        and cohort.get("episodes") == count and cohort.get("exact") is False,
        "complete original fixed-policy cohort required")
    claim = read(cohort["claim"])
    require(claim.get("schema") == "rage_mill.canonical_development_cohort_claim.v1"
        and claim.get("identity_sha256") == value_sha256(identity)
        and claim.get("seed_start") == seed and claim.get("episodes") == count and claim.get("exact") is False,
        "original cohort claim differs")
    rows = cohort.get("episode_receipts")
    require(isinstance(rows, list) and len(rows) == count and len({row["path"] for row in rows}) == count,
            "one distinct original tape per declared development label required")
    runtime = launch["runtime_closure"]
    require(runtime in identity["sources"], "original candidate runtime manifest must belong to identity")
    runtime_pins = validate_runtime_manifest(read(runtime))
    require(all(pins.get(path) == digest for path, digest in runtime_pins.items())
        and all(pins.get(row["path"]) == row["sha256"] for row in
                [identity["controller"], identity["configuration"], *identity["sources"]]),
        "original owner must prospectively freeze all candidate/runtime files")
    runtime_row = pinned_file(output / "runtime-receipt.json")
    audit_runtime_execution(runtime_row, identity=identity, runtime_closure=runtime, outputs=[cohort_row])
    scores = []
    for index, row in enumerate(rows):
        episode, _ = saved_episode(row, identity, seed=seed+index,
                                  directory=output / f"episodes/episode-{seed+index}")
        scores.append(episode["score"])
    require(scores == component.get("scores"), "original component scores differ from saved actual tapes")
    return dict(identity=identity, seed_start=seed, episodes=count, cohort=cohort_row, rows=rows,
                owner=directory, output=output, original_rows=[launch_row, completion_row, log,
                component_row, cohort_row, cohort["claim"], runtime, runtime_row])


def prepare(args):
    start, cpu = time.monotonic(), time.process_time()
    identity_row = pinned_file(args.identity)
    require(identity_row["sha256"] == args.expected_identity_sha256, "fixed candidate identity SHA required")
    original = original_owner(args.owner, identity_row, args.mode)
    import mujoco  # Preload native dependencies; no model is constructed here.
    require(mujoco.__version__ == "3.3.5", "actual native MuJoCo3.3.5 required")
    source, supervisor = Path(__file__).resolve(), args.supervisor.resolve(strict=True)
    files = [source, supervisor, Path(sys.prefix) / "pyvenv.cfg", Path(identity_row["path"])]
    files += [Path(row["path"]) for row in original["original_rows"]]
    files += [path.resolve() for path in original["output"].rglob("*") if path.is_file()]
    files += [Path(row["path"]) for row in original["identity"]["sources"]]
    manifest = create_runtime_manifest(package_names=("numpy", "mujoco"), extra_paths=files)
    pins = validate_runtime_manifest(manifest)
    out = args.out.resolve()
    require(not out.exists(), "fresh create-only saved-native proof owner required")
    out.mkdir(parents=True)
    runtime = write_json_exclusive(out / "runtime-closure.json", manifest)
    pins[runtime["path"]] = runtime["sha256"]
    inputs = write_json_exclusive(out / "inputs.json", dict(
        schema="rage_mill.canonical18_development_native_inputs.v1", identity=identity_row,
        original_owner=str(original["owner"]), mode=args.mode, runtime_closure=runtime,
        source_hashes=dict(sorted(pins.items())), physical_episodes=0, reset_RNG_draws=0,
        policy_calls=0, maximum_native_maps=3000*original["episodes"]))
    pins[inputs["path"]] = inputs["sha256"]
    count, cpu_cap, wall_cap = LIMITS[args.mode]
    recipe = dict(schema="rage_mill.canonical18_development_native_owner.v1",
        argv=[str(Path(sys.executable).absolute()), str(source), "--run", "--inputs", inputs["path"],
              "--expected-inputs-sha256", inputs["sha256"], "--out", str(out / "output")],
        cwd=str(out), environment=dict(PYTHONPATH=str(source.parents[1] / "src"),
            PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        CPU_cap_s=cpu_cap, wall_cap_s=wall_cap, address_cap_bytes=8*1024**3,
        output_path=str(out / "output"), log=str(out / "worker.log"), receipt=str(out / "completion.json"),
        source_hashes=dict(sorted(pins.items())), runtime_closure=runtime,
        original_identity=identity_row, original_actual_owner=str(original["owner"]),
        mode=args.mode, original_saved_episodes=count, maximum_native_commands=3000*count,
        maximum_continuous_native_maps=1500*count, maximum_cold_native_maps=1500*count,
        physical_episodes=0, policy_calls=0, reset_RNG_draws=0,
        recorded_reset_reconstructions=0, new_reserved_claims=0, reserved_episodes=0,
        purpose="All original own18 " + args.mode + " saved native cold/continuous replays; no policy or reset RNG",
        stop_criteria="one bounded owner; no new episode, failed-label replacement or retry")
    launch = write_json_exclusive(out / "launch.json", recipe)
    return write_json_exclusive(out / "preparation.json", dict(metadata_only=True,
        saved_array_validation_only=True, launch=launch, inputs=inputs, runtime_closure=runtime,
        model_constructions=0, native_commands=0, policy_calls=0, reset_RNG_draws=0,
        source_files=len(pins), CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-start))


def run(args):
    start, cpu = time.monotonic(), time.process_time()
    inputs_row = pinned_file(args.inputs)
    require(inputs_row["sha256"] == args.expected_inputs_sha256, "prospective saved-native inputs SHA required")
    inputs = read(inputs_row)
    pins = inputs["source_hashes"]
    require(pins and all(pinned_file(path)["sha256"] == digest for path, digest in pins.items()),
            "all prospective saved-native source/runtime inputs must match")
    # Make reset-distribution or episode execution an explicit process failure.
    import rage_mill.canonical_benchmark as benchmark
    import rage_mill.canonical_evaluation as evaluation
    def forbidden(*unused_args, **unused_kwargs):
        raise AssertionError("saved development native proof forbids policy/episode/reset-RNG execution")
    benchmark.hanging_state = evaluation.hanging_state = np.random.default_rng = forbidden
    benchmark.record_episode = evaluation.record_development_cohort = evaluation._record_episode = forbidden
    original = original_owner(Path(inputs["original_owner"]), inputs["identity"], inputs["mode"])
    runtime_pins = validate_runtime_manifest(read(inputs["runtime_closure"]))
    require(inputs["maximum_native_maps"] == 3000*original["episodes"] and not args.out.exists(),
            "exact declared saved-native scope and fresh output required")
    args.out.mkdir(parents=True)
    results, native_calls = [], 0
    with (args.out / "progress.jsonl").open("x") as progress:
        for index, row in enumerate(original["rows"]):
            seed = original["seed_start"]+index
            _, actual = saved_episode(row, original["identity"], seed=seed,
                directory=original["output"] / f"episodes/episode-{seed}")
            proof = prove_recorded_native_transitions(row, args.out / f"episode-{seed}",
                                                      maximum_native_transitions=3000)
            document = read(proof)
            summary = audit_native_replay_proof(proof, episode_receipt=row)
            with np.load(io.BytesIO(read_pinned(document["proof_arrays"])), allow_pickle=False) as saved:
                require(same(saved["continuous_states"], actual["states"])
                    and same(saved["cold_next_states"], actual["states"][1:]),
                    "continuous and cold replay must reproduce every saved state byte for byte")
            require(document["continuous_transitions"] == document["one_step_transitions"] == 1500
                and summary["continuous_replay_identical"] is True
                and summary["maximum_continuous_replay_error"] == summary["maximum_dynamics_defect"] == 0.,
                "complete zero-defect cold/continuous original native replays required")
            native_calls += document["continuous_transitions"] + document["one_step_transitions"]
            result = dict(seed=seed, episode=row, native_proof=proof, summary=summary,
                continuous1501_states_byte_exact=True, cold1500_states_byte_exact=True,
                recorded_initial_replay_gap=0., native_maps=3000)
            results.append(result)
            progress.write(json.dumps(result, allow_nan=False)+"\n"); progress.flush()
            print(json.dumps(dict(seed=seed, passed=True, native_maps=3000)), flush=True)
    actual_loaded = {path: pinned_file(path)["sha256"] for path in runtime_binding_paths(
        package_names=(), include_stdlib=False, include_loaded=True)}
    missing = {path: digest for path, digest in actual_loaded.items() if runtime_pins.get(path) != digest}
    coverage = write_json_exclusive(args.out / "runtime-coverage.json", dict(
        schema="rage_mill.canonical18_development_native_runtime.v1",
        runtime_closure=inputs["runtime_closure"], original_identity=inputs["identity"],
        actual_loaded_file_sha256=actual_loaded, uncovered_or_changed=missing,
        runtime_complete=not missing, additional_native_calls=0, additional_RNG_draws=0))
    require(not missing, "actual proof runtime was not prospectively frozen: " + json.dumps(missing))
    require(native_calls == inputs["maximum_native_maps"]
        and all(pinned_file(path)["sha256"] == digest for path, digest in pins.items()),
        "all native counts and original prospective source/runtime pins must remain unchanged")
    result = write_json_exclusive(args.out / "receipt.json", dict(
        schema="rage_mill.canonical18_development_native_cohort_proof.v1", passed=True,
        identity=inputs["identity"], original_owner=str(original["owner"]), cohort=original["cohort"],
        mode=inputs["mode"], episodes_proved=len(results), seed_start=original["seed_start"],
        episode_proofs=results, runtime_closure=inputs["runtime_closure"], runtime_coverage=coverage,
        all_continuous_and_cold_states_byte_exact=True, maximum_dynamics_defect=0.,
        maximum_continuous_replay_error=0., maximum_recorded_initial_replay_gap=0.,
        total_native_maps=native_calls, continuous_native_maps=1500*len(results),
        cold_native_maps=1500*len(results), maximum_native_maps=inputs["maximum_native_maps"],
        physical_episodes=0, new_reserved_claims=0, reserved_episodes=0, policy_calls=0,
        reset_RNG_draws=0, recorded_reset_reconstructions=0, controller_qualified=False,
        noisy_initial_gap_scope="Generic initial_state_gap is distance from exact hanging, preserved as data; replay begins byte-exactly from the saved noisy state.",
        source_hashes_unchanged=True, CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-start))
    print(json.dumps(dict(receipt=result, total_native_maps=native_calls)), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    for name in ("identity", "owner", "supervisor", "inputs", "out"):
        parser.add_argument("--"+name, type=Path, required=name == "out")
    for name in ("expected-identity-sha256", "expected-inputs-sha256"):
        parser.add_argument("--"+name)
    parser.add_argument("--mode", choices=tuple(LIMITS))
    args = parser.parse_args()
    require(args.prepare != args.run, "one explicit metadata preparation or bounded saved-native run required")
    if args.prepare:
        require(all(getattr(args, name) is not None for name in
            ("identity", "expected_identity_sha256", "owner", "supervisor", "mode")),
            "actual fixed identity, naturally closed owner, mode and supervisor required")
        print(json.dumps(dict(preparation=prepare(args), claims_created=0, native_commands=0)))
    else:
        require(args.inputs is not None and args.expected_inputs_sha256,
                "pinned prospective inputs required")
        run(args)


if __name__ == "__main__":
    main()
