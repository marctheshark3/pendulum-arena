"""Runtime-file ownership for formal canonical calibration and development.

This module only hashes files and audits saved receipts. It creates no plant,
draws no RNG, imports no policy, and grants no reserved activation authority.
The runtime manifest JSON belongs in the source inventory; its binary library
rows remain external to the clean Git source bundle.
"""
from __future__ import annotations

import json
from pathlib import Path

from .canonical_evaluation import (
    pinned_file, read_pinned, require, runtime_metadata, validate_identity,
    value_sha256, write_json_exclusive,
)
from .canonical_runtime_closure import runtime_binding_paths
from .canonical_runtime_manifest import validate_runtime_manifest


def audit_runtime_closure(row, identity):
    """Rehash every runtime file and require its manifest in frozen sources."""
    require(row in identity["sources"], "runtime manifest JSON omitted from frozen source inventory")
    manifest = json.loads(read_pinned(row))
    return validate_runtime_manifest(manifest, require_complete=True)


def _allowed_files(identity, runtime_closure):
    allowed = audit_runtime_closure(runtime_closure, identity)
    for row in [identity["controller"], identity["configuration"], *identity["sources"]]:
        read_pinned(row)
        require(row["path"] not in allowed or allowed[row["path"]] == row["sha256"],
                "source and runtime manifest disagree")
        allowed[row["path"]] = row["sha256"]
    return allowed


def record_runtime_execution(out, *, identity, runtime_closure, outputs):
    """Save actual postloaded coverage for an already bounded owned worker.

    The caller validates this report before exiting zero. Report creation alone
    does not grant a pass: all actual loaded-file membership is checked again by
    the read-only auditor, together with the prospective launch and completion.
    """
    validate_identity(identity)
    require(runtime_metadata() == identity["runtime"], "runtime differs from frozen identity")
    require(isinstance(outputs, list) and outputs, "actual owned output pins required")
    for row in outputs:
        read_pinned(row)
    allowed = _allowed_files(identity, runtime_closure)
    actual = {path: pinned_file(path)["sha256"] for path in runtime_binding_paths(
        package_names=(), include_stdlib=False, include_loaded=True)}
    missing = {path: digest for path, digest in actual.items() if allowed.get(path) != digest}
    # Check all rows after enumeration too, rather than trusting unchanged flags.
    _allowed_files(identity, runtime_closure)
    return write_json_exclusive(out, {
        "schema": "rage_mill.canonical_runtime_execution.v1", "runtime_closure": runtime_closure,
        "identity_sha256": value_sha256(identity), "runtime": runtime_metadata(),
        "producer_source": pinned_file(__file__), "outputs": outputs,
        "actual_loaded_file_sha256": actual, "uncovered_or_changed": missing,
        "all_runtime_pins_unchanged": True, "runtime_pins_complete": not missing,
        "additional_native_calls": 0, "additional_controller_calls": 0, "additional_RNG_draws": 0})


def audit_runtime_execution(row, *, identity, runtime_closure, outputs):
    report = json.loads(read_pinned(row))
    require(report.get("schema") == "rage_mill.canonical_runtime_execution.v1"
            and report.get("runtime_closure") == runtime_closure
            and report.get("identity_sha256") == value_sha256(identity)
            and report.get("runtime") == identity["runtime"]
            and report.get("outputs") == outputs, "runtime report input/output lineage differs")
    producer = report.get("producer_source")
    require(producer == pinned_file(__file__) and producer in identity["sources"],
            "runtime report producer differs or was not frozen")
    for output in outputs:
        read_pinned(output)
    actual = report.get("actual_loaded_file_sha256")
    require(isinstance(actual, dict) and actual, "actual loaded-file inventory required")
    allowed = _allowed_files(identity, runtime_closure)
    for path, digest in actual.items():
        require(isinstance(path, str) and Path(path).is_absolute()
                and str(Path(path).resolve(strict=True)) == path
                and allowed.get(path) == digest and pinned_file(path)["sha256"] == digest,
                "actual loaded file was not prospectively bound by runtime/source identity")
    require(report.get("all_runtime_pins_unchanged") is True
            and report.get("runtime_pins_complete") is True
            and report.get("uncovered_or_changed") == {}, "runtime coverage failed")
    require(all(type(report.get(key)) is int and report[key] == 0 for key in
                ("additional_native_calls", "additional_controller_calls", "additional_RNG_draws")),
            "runtime metadata audit must not execute physics, policy or RNG")
    return report


def audit_formal_runtime_owners(frozen):
    """Recheck prospectively pinned bounded owners for the three prerequisites."""
    identity, runtime_closure = frozen["identity"], frozen.get("runtime_closure")
    runtime_files = audit_runtime_closure(runtime_closure, identity)
    source = json.loads(read_pinned(frozen["source_calibration"]))
    expected = {
        "calibration": [source["episode_receipt"], source["native_replay_proof"]],
        "noisy20": [frozen["development"]["noisy20"]["receipt"]],
        "noisy100": [frozen["development"]["noisy100"]["receipt"]],
    }
    owners = frozen.get("runtime_owners")
    require(isinstance(owners, dict) and set(owners) == set(expected),
            "calibration and both development runtime owners required")
    for stage, outputs in expected.items():
        owner = owners[stage]
        require(isinstance(owner, dict) and set(owner) == {"launch", "completion", "runtime_receipt"},
                "pinned launch, completion and runtime receipt required")
        launch = json.loads(read_pinned(owner["launch"]))
        complete = json.loads(read_pinned(owner["completion"]))
        require(launch.get("schema") == "rage_mill.next_stage_supervisor_recipe.v1"
                and launch.get("runtime_closure") == runtime_closure,
                "formal owner lacks its prospective runtime manifest")
        pins = launch.get("source_hashes")
        require(isinstance(pins, dict) and pins.get(runtime_closure["path"]) == runtime_closure["sha256"]
                and all(pins.get(path) == digest for path, digest in runtime_files.items()),
                "formal owner omitted prospective runtime file pins")
        require(all(pins.get(row["path"]) == row["sha256"] for row in
                    [identity["controller"], identity["configuration"], *identity["sources"]]),
                "formal owner omitted complete frozen controller/config/source pins")
        require(isinstance(launch.get("argv"), list) and len(launch["argv"]) >= 2
                and launch["argv"][1] in {row["path"] for row in identity["sources"]}
                and pins.get(launch["argv"][1]) == pinned_file(launch["argv"][1])["sha256"]
                and pins.get(str(Path(__file__).resolve())) == pinned_file(__file__)["sha256"],
                "formal worker/runtime producer omitted from prospective source freeze")
        require(all(pinned_file(path)["sha256"] == digest for path, digest in pins.items()),
                "formal owner's declared source/runtime pins changed")
        require(complete.get("schema") == "rage_mill.next_stage_supervised_completion.v1"
                and complete.get("recipe_path") == owner["launch"]["path"]
                and complete.get("recipe_sha256") == owner["launch"]["sha256"]
                and type(complete.get("child_returncode")) is int and complete["child_returncode"] == 0
                and complete.get("naturally_closed_zero") is True
                and complete.get("failure") is None and complete.get("interrupted") is False
                and complete.get("wall_timeout") is False and complete.get("recipe_unchanged") is True,
                "formal runtime owner failed or lacks natural-zero closure")
        unchanged = complete.get("source_hashes_unchanged")
        require(isinstance(unchanged, dict) and set(unchanged) == set(pins)
                and all(value is True for value in unchanged.values()), "formal full pin closure failed")
        for key in ("CPU_cap_s", "wall_cap_s"):
            value = launch.get(key)
            require(type(value) in (int, float) and 0 < value < float("inf"),
                    "bounded formal owner required")
        require(owner["runtime_receipt"]["path"] == str(Path(launch["output_path"]) / "runtime-receipt.json"),
                "runtime receipt moved outside its owned output")
        output_root = Path(launch["output_path"])
        require(output_root.is_absolute() and str(output_root.resolve(strict=True)) == str(output_root)
                and all(Path(output["path"]).is_relative_to(output_root) for output in outputs),
                "formal calibration/development outputs are outside their runtime owner")
        audit_runtime_execution(owner["runtime_receipt"], identity=identity,
                                runtime_closure=runtime_closure, outputs=outputs)
    return runtime_files
