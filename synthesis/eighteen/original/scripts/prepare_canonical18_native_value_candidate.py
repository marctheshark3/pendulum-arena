"""Prepare stored700-value native tracking without online planning libraries."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np

from rage_mill.canonical_benchmark import CanonicalProtocol, canonical_xml, score_tape
from rage_mill.canonical18_native_value_policy import _tables, controller_factory, policy_parameters
from rage_mill.canonical_evaluation import (
    canonical_json, make_identity, pinned_file, read_pinned, require, write_json_exclusive,
)
from rage_mill.canonical_runtime_manifest import create_runtime_manifest, validate_runtime_manifest

MODULES = (
    "canonical_benchmark", "canonical_evaluation", "canonical_replay", "canonical_release",
    "canonical_reserved", "canonical_video", "canonical_runtime_evidence",
    "canonical_runtime_manifest", "canonical_runtime_closure", "canonical18_controller",
    "canonical18_evidence", "canonical_profile_policy", "canonical16_affine_feedback",
    "canonical16_predictive_tracking", "canonical18_native_value",
    "canonical18_native_value_episode", "canonical18_native_value_policy",
)
SCRIPTS = (
    "evaluate_canonical18_profile_candidate.py", "run_canonical_reserved_gate.py",
    "verify_canonical_release.py", "verify_canonical_reserved_release.py",
    "prove_canonical_episode_native.py", "render_canonical_episode_video.py",
    "audit_canonical18_saved_policy.py",
    "prepare_canonical18_native_value_candidate.py", "prepare_canonical18_native_value_stage.py",
)


def load(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def audit_actual_diagnostic(path, *, controller, reference, feedback, seed=3181040):
    """Bind a closed diagnostic owner and saved bytes; execute no dynamics or policy."""
    path = path.resolve(strict=True)
    diagnostic = json.loads(path.read_text())
    require(diagnostic.get("schema") == "rage_mill.canonical18_native_value_hanging_diagnostic.v1"
            and type(diagnostic.get("n_links")) is int and diagnostic["n_links"] == 18
            and diagnostic.get("all_success") is True
            and type(diagnostic.get("active_route_start")) is int and diagnostic["active_route_start"] == 0
            and type(diagnostic.get("cart_error_offset_m")) is float and diagnostic["cart_error_offset_m"] == 0.
            and diagnostic.get("controller_sha256") == pinned_file(controller)["sha256"]
            and diagnostic.get("reference_sha256") == pinned_file(reference)["sha256"]
            and diagnostic.get("feedback_sha256") == pinned_file(feedback)["sha256"],
            "passing source-matched native-value hanging diagnostic required before candidate prep")
    cases = diagnostic.get("cases", [])
    require(type(seed) is int and seed >= 0
            and len(cases) == 1 and cases[0].get("seed") == seed and cases[0].get("exact") is True,
            "one actual prospectively declared exact diagnostic required")
    case = cases[0]
    rows = {name: pinned_file(case[name + "_path"]) for name in ("recording", "candidate_ledger")}
    require(path.name == "receipt.json" and path.parent.name == "output"
            and rows["recording"]["path"] == str(path.parent / f"episode-{seed}.npz")
            and rows["candidate_ledger"]["path"] == str(path.parent / f"episode-{seed}-candidate-ledger.npz")
            and all(row["sha256"] == case[name + "_sha256"] for name, row in rows.items()),
            "actual diagnostic saved tape or ledger changed")
    owner = path.parent.parent
    rows.update({"diagnostic": pinned_file(path), "launch": pinned_file(owner / "launch.json"),
                 "completion": pinned_file(owner / "completion.json")})
    launch, completion = (json.loads(read_pinned(rows[name])) for name in ("launch", "completion"))
    require(launch.get("output_path") == str(path.parent)
            and completion.get("recipe_path") == rows["launch"]["path"]
            and completion.get("recipe_sha256") == rows["launch"]["sha256"]
            and completion.get("naturally_closed_zero") is True
            and type(completion.get("child_returncode")) is int and completion["child_returncode"] == 0
            and completion.get("failure") is None and completion.get("interrupted") is False
            and completion.get("wall_timeout") is False and completion.get("recipe_unchanged") is True
            and bool(completion.get("owned_group_cleanup")), "diagnostic owner lacks natural-zero closure")
    pins, unchanged = launch.get("source_hashes"), completion.get("source_hashes_unchanged")
    require(isinstance(pins, dict) and pins and isinstance(unchanged, dict)
            and set(pins) == set(unchanged) and all(flag is True for flag in unchanged.values())
            and all(pinned_file(source)["sha256"] == digest for source, digest in pins.items())
            and all(pins.get(str(source.resolve())) == pinned_file(source)["sha256"]
                    for source in (controller, reference, feedback)),
            "diagnostic source/runtime closure changed")
    rows["log"] = pinned_file(completion["log_path"])
    require(rows["log"]["path"] == str(owner / "worker.log")
            and rows["log"]["sha256"] == completion["log_sha256"], "diagnostic owner log changed")
    rows["diagnostic_runtime"] = launch["runtime_closure"]
    runtime_pins = validate_runtime_manifest(json.loads(read_pinned(rows["diagnostic_runtime"])), require_complete=True)
    require(pins.get(rows["diagnostic_runtime"]["path"]) == rows["diagnostic_runtime"]["sha256"]
            and all(pins.get(source) == digest for source, digest in runtime_pins.items()),
            "diagnostic owner did not pin its complete prospective runtime")
    coverage_row = diagnostic.get("runtime_coverage")
    rows["runtime_coverage"] = pinned_file(path.parent / "runtime-coverage.json")
    require(isinstance(coverage_row, dict) and coverage_row.get("path") == rows["runtime_coverage"]["path"]
            and coverage_row.get("sha256") == rows["runtime_coverage"]["sha256"],
            "original diagnostic actual runtime coverage file changed")
    coverage = json.loads(read_pinned(rows["runtime_coverage"]))
    actual = coverage.get("actual_loaded_file_sha256")
    require(coverage.get("runtime_closure") == rows["diagnostic_runtime"]["path"]
            and isinstance(actual, dict) and actual
            and all(pinned_file(source)["sha256"] == digest for source, digest in actual.items())
            and coverage.get("uncovered_or_changed") == {}
            and all(runtime_pins.get(source) == digest for source, digest in actual.items())
            and coverage.get("all_runtime_pins_unchanged") is True
            and all(type(coverage.get(key)) is int and coverage[key] == 0 for key in ("native_commands", "reset_draws")),
            "actual diagnostic runtime coverage did not pass unchanged original manifest")
    argv = launch.get("argv")
    require(isinstance(argv, list) and len(argv) >= 2
            and Path(argv[1]).name == "probe_canonical18_native_value_episode.py"
            and actual.get(argv[1]) == runtime_pins.get(argv[1]) and argv[1] in actual
            and "--runtime-closure" in argv
            and argv[argv.index("--runtime-closure") + 1] == rows["diagnostic_runtime"]["path"],
            "original own-count diagnostic worker/actual runtime manifest binding differs")
    tape, ledger = load(rows["recording"]["path"]), load(rows["candidate_ledger"]["path"])
    shapes = {"states": (1501, 38), "actions": (1500,), "raw_actions": (1500,),
              "selected_predicted_states": (650, 38), "selected_predicted_actions": (650,),
              "selected_episode_ticks": (650,), "selected_indices": (650,), "baseline_indices": (650,),
              "unique_counts": (650,), "baseline_delivery_agreement": (650,),
              "ordinary_raw_actions": (1500,)}
    for key, shape in shapes.items():
        dtype = ("float32" if key == "selected_predicted_actions" else "bool"
                 if key == "baseline_delivery_agreement" else "int64"
                 if key in {"selected_episode_ticks", "selected_indices", "baseline_indices", "unique_counts"}
                 else "float64")
        require(tape[key].shape == shape and tape[key].dtype == np.dtype(dtype),
                "actual diagnostic tape shape/dtype mismatch: " + key)
    initial = np.zeros(38); initial[1] = np.pi
    require(tape["states"][0].tobytes() == initial.tobytes(), "actual diagnostic exact initial state changed")
    score = score_tape(tape["states"], tape["actions"], CanonicalProtocol(n_links=18))
    require(score["success"] and canonical_json({key: case.get(key) for key in score}) == canonical_json(score),
            "actual diagnostic score differs from saved physical tape")
    selected, baseline, counts = (tape[key] for key in ("selected_indices", "baseline_indices", "unique_counts"))
    count = len(ledger["commands"])
    require(tape["selected_episode_ticks"].tobytes() == np.arange(850, 1500, dtype=np.int64).tobytes()
            and tape["selected_predicted_states"].tobytes() == tape["states"][851:].tobytes()
            and tape["selected_predicted_actions"].tobytes() == tape["actions"][850:].astype(np.float32).tobytes()
            and np.isfinite(tape["raw_actions"]).all()
            and np.clip(tape["raw_actions"].astype(np.float32), -1., 1.).astype(np.float64).tobytes()
                == tape["actions"].tobytes()
            and np.isfinite(tape["ordinary_raw_actions"]).all()
            and np.all((1 <= counts) & (counts <= 16)) and int(counts.sum()) == count
            and np.all((0 <= selected) & (selected < count))
            and np.all((0 <= baseline) & (baseline < count)),
            "actual diagnostic full650 selected state/action/candidate checks failed")
    require(ledger["states"].shape == (count, 38) and ledger["states"].dtype == np.dtype("float64")
            and ledger["commands"].dtype == np.dtype("float32")
            and ledger["ticks"].dtype == np.dtype("int64")
            and np.all(ledger["errors"] == "")
            and np.isfinite(ledger["states"]).all() and np.isfinite(ledger["costs"]).all()
            and np.isfinite(ledger["commands"]).all()
            and ledger["states"][selected].tobytes() == tape["selected_predicted_states"].tobytes()
            and ledger["commands"][selected].tobytes() == tape["selected_predicted_actions"].tobytes()
            and ledger["ticks"].tobytes() == np.repeat(np.arange(650, dtype=np.int64), counts).tobytes()
            and np.array_equal(ledger["ticks"][selected], np.arange(650))
            and np.array_equal(ledger["ticks"][baseline], np.arange(650))
            and case.get("selected_native_states") == 650
            and case.get("all_selected_native_states_byte_exact") is True
            and case.get("native_candidate_commands") == count
            and case.get("baseline_delivery_disagreements")
                == int(np.count_nonzero(~tape["baseline_delivery_agreement"]))
            and case.get("maximum_candidates_per_active_command") == int(counts.max()),
            "actual diagnostic candidate ledger does not bind saved selected650 states/actions")
    ordinary_delivery = np.clip(tape["ordinary_raw_actions"][850:].astype(np.float32), -1., 1.)
    agreements = np.asarray([left.tobytes() == right.tobytes()
                            for left, right in zip(ledger["commands"][baseline], ordinary_delivery, strict=True)],
                           dtype=bool)
    require(agreements.tobytes() == tape["baseline_delivery_agreement"].tobytes(),
            "saved ordinary/baseline delivery agreement flags differ")
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    for name in ("controller", "reference", "feedback", "native-defects", "actual-diagnostic", "declaration",
                 "design-metadata", "supervisor", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--actual-diagnostic-seed", type=int, default=3181040)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for name in MODULES:
        importlib.import_module("rage_mill." + name)
    evaluator = runpy.run_path(str(root / "scripts" / SCRIPTS[0]), run_name="runtime_metadata_only")
    for name in SCRIPTS[1:7]:
        runpy.run_path(str(root / "scripts" / name), run_name="runtime_metadata_only")
    evaluator["development_modes"](json.loads(args.declaration.read_text()))
    ordinary, reference, value = map(load, (args.controller, args.reference, args.feedback))
    actual_rows = audit_actual_diagnostic(args.actual_diagnostic, controller=args.controller,
                                         reference=args.reference, feedback=args.feedback,
                                         seed=args.actual_diagnostic_seed)
    defects = load(args.native_defects)
    require(set(defects) == {"native_defects"}
            and defects["native_defects"].dtype == np.dtype("float64")
            and defects["native_defects"].shape == (700, 38)
            and defects["native_defects"].tobytes() == reference["native_defects"].tobytes(),
            "full700 verified native defects must equal stored reference bytes")
    combined = {**ordinary,
        **{"value_" + key: reference[key] for key in ("states", "actions", "native_defects")},
        **{"value_" + key: value[key] for key in ("gains", "offsets", "roots", "residuals", "Q")}}
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    controller = out / "controller.npz"
    with controller.open("xb") as stream:
        np.savez_compressed(stream, **combined)
    _tables(controller.read_bytes())
    design = json.loads(args.design_metadata.read_text())
    parameters = policy_parameters(controller_sha256=pinned_file(controller)["sha256"], design_metadata={
        "declaration": pinned_file(args.declaration), "design_evidence": pinned_file(args.design_metadata),
        "actual_diagnostic": pinned_file(args.actual_diagnostic), "design": design,
        "actual_diagnostic_evidence": actual_rows,
        "ordinary_source": pinned_file(args.controller), "full_reference_source": pinned_file(args.reference),
        "full_affine_value_source": pinned_file(args.feedback),
        "verified_native_defects_source": pinned_file(args.native_defects),
    })
    protocol = CanonicalProtocol(n_links=18)
    configuration = write_json_exclusive(out / "configuration.json", {
        "schema": "rage_mill.canonical18_native_value_configuration.v1", "protocol": asdict(protocol),
        "policy_parameters": parameters, "controller": pinned_file(controller),
        "qualification_authority": False,
        "scientific_pass_requires": "complete actual success and full650 outer shadow/live byte parity",
    })
    xml = out / "model.xml"
    with xml.open("x") as stream:
        stream.write(canonical_xml(protocol))
    sources = ([root / f"src/rage_mill/{name}.py" for name in MODULES]
               + [root / "scripts" / name for name in SCRIPTS]
               + [args.declaration.resolve(), args.design_metadata.resolve(),
                  args.controller.resolve(), args.reference.resolve(), args.feedback.resolve(),
                  args.native_defects.resolve(), *[Path(row["path"]) for row in actual_rows.values()], xml])
    manifest = create_runtime_manifest(package_names=("numpy", "mujoco"), model_directories=(),
        extra_paths=[*sources, args.supervisor.resolve(), controller, Path(configuration["path"])])
    runtime = write_json_exclusive(out / "runtime-closure.json", manifest)
    pins = validate_runtime_manifest(manifest)
    require(not any("site-packages/casadi/" in path or "tmp_casadi_compiler" in path for path in pins),
            "native-value online execution must not load a planning library")
    identity = make_identity(protocol, controller=controller, configuration=configuration["path"],
                             source_files=[*sources, Path(runtime["path"])], policy_parameters=parameters)
    controller_factory(identity)  # Immutable tables/independent clocks only; no native model.
    identity_row = write_json_exclusive(out / "identity.json", identity)
    receipt = write_json_exclusive(out / "preparation.json", {
        "schema": "rage_mill.canonical18_candidate_preparation.v1", "metadata_only": True,
        "identity": identity_row, "configuration": configuration, "runtime_closure": runtime,
        "runtime_files": len(pins), "supervisor": pinned_file(args.supervisor),
        "python_launcher": str(Path(sys.executable).absolute()), "model_constructions": 0,
        "native_commands": 0, "controller_calls": 0, "reset_draws": 0,
        "qualification_authority": False,
    })
    print(json.dumps({"preparation": receipt, "identity": identity_row, "runtime_closure": runtime}))


if __name__ == "__main__":
    main()
