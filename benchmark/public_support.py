"""New public adapters. Nothing in this module was part of the frozen campaign."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "frozen"
SOURCE_COMMIT = "886267f09ce67d7215c166c92e14a0adaca8e34f"
FREEZE_SHA256 = "874cac1d6ba42af1c9483e20e92a90cbbb78555bdb94412c4b7308d6055172a5"
AUDIT_SHA256 = "f13945b629125fc165581c367e5c046da6fe1ed50fc6c9f5a3f7b1abfc03e218"
COHORTS = {
    "calibration": ("planned-candidate02-calibration02", [("calibration", 3156000, 1, True)]),
    "development": ("development120-01", [("dev20", 3156200, 20, False), ("dev100", 3156300, 100, False)]),
    "reserved": ("reserved141-01", [("noisy20", 215000, 20, False), ("noisy100", 215100, 100, False),
                                      ("exact20", 215200, 20, True), ("video", 215500, 1, False)]),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_hash(path, expected):
    require(sha256(path) == expected, f"SHA256 mismatch: {path}")


def safe_join(base, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "recording path must be relative")
    base = Path(base).resolve()
    result = (base / relative).resolve()
    require(result.is_relative_to(base), "recording path escapes its cohort")
    return result


def verify_source(freeze=None):
    manifest = read_json(HERE / "frozen-manifest.json")
    require(manifest["original_commit"] == SOURCE_COMMIT, "unexpected original source commit")
    for name, expected in manifest["files"].items():
        require_hash(safe_join(FROZEN, name), expected)
    for name, entry in manifest["new_adapter_files"].items():
        require_hash(safe_join(HERE, name), entry["sha256"])
    actual = {name: sha256(FROZEN / name) for name in manifest["source_identity"]}
    require(actual == manifest["source_identity"], "curated source identity differs")
    if freeze is not None:
        require(actual == freeze["identities"]["source"], "curated source differs from the original freeze")
    return actual


def load_proof(proof_root):
    """Resolve relocation externally; never rewrite original receipts or their paths."""
    root = Path(proof_root).resolve()
    freeze_path = root / "frozen-release02/freeze.json"
    audit_path = root / "independent-final-audit01/receipt.json"
    require_hash(freeze_path, FREEZE_SHA256)
    require_hash(audit_path, AUDIT_SHA256)
    freeze, audit = read_json(freeze_path), read_json(audit_path)
    require(freeze["source_commit"] == SOURCE_COMMIT, "wrong source commit in freeze")
    require(audit["freeze_sha256"] == FREEZE_SHA256 and audit["passed"], "original audit does not bind this freeze")
    require(audit["identities"] == freeze["identities"], "original audit identities differ")
    verify_source(freeze)
    candidate = root / "frozen-release02/candidate"
    identity = {name: sha256(candidate / name) for name in freeze["identities"]["candidate"]}
    require(identity == freeze["identities"]["candidate"] == read_json(candidate / "identity.json"),
            "candidate differs from the original frozen identity")
    require_hash(root / "frozen-release02/requirements-evaluation.txt", freeze["requirements_sha256"])

    original_cases = {(row["gate"], row["seed"], row["exact"]): row for row in audit["cases"]}
    require(len(original_cases) == len(audit["cases"]) == 262, "original audit has unexpected case count")
    jobs, forecast_count, evidence = [], 0, {}
    for stage, (directory, declarations) in COHORTS.items():
        receipt_path = root / directory / "receipt.json"
        require_hash(receipt_path, audit["evidence"][stage]["sha256"])
        if stage != "reserved":
            require_hash(receipt_path, freeze["evidence"][stage]["sha256"])
        receipt = read_json(receipt_path)
        require(receipt["stage"] == stage and receipt["passed"] and receipt["complete"]
                and receipt["identity_unchanged"] and receipt["identities"] == freeze["identities"],
                f"cohort identity/admission mismatch: {stage}")
        expected = {(name, seed, exact) for name, start, count, exact in declarations
                    for seed in range(start, start + count)}
        actual = [(row["gate"], row["seed"], row["exact"]) for row in receipt["cases"]]
        require(len(actual) == len(expected) and set(actual) == expected, f"missing, duplicate or replaced cases: {stage}")
        for row in receipt["cases"]:
            recording = safe_join(receipt_path.parent, row["recording"])
            require_hash(recording, row["recording_sha256"])
            require(read_json(recording.parent / "receipt.json") == row, "case and aggregate receipt differ")
            key = row["gate"], row["seed"], row["exact"]
            require(original_cases[key]["source_recording_sha256"] == row["recording_sha256"],
                    "recording differs from the original final audit")
            selected = row["selected_profile"]
            require(type(selected) is int and [item["profile"] for item in row["forecasts"]] == list(range(selected + 1)),
                    "missing or reordered internal planning forecasts")
            for forecast in row["forecasts"]:
                require_hash(safe_join(recording.parent, forecast["recording"]), forecast["recording_sha256"])
                forecast_count += 1
            jobs.append((str(receipt_path), row))
        evidence[stage] = sha256(receipt_path)
    require(len(jobs) == 262, "public proof pack must contain all 262 actual episodes")

    video_path = root / "heldout-video01/receipt.json"
    require_hash(video_path, audit["video_proof_sha256"])
    video = read_json(video_path)
    require(video["passed"] and video["rendered_frames"] == 1500 and video["fps"] == 50
            and video["simulated_seconds"] == 30.0 and video["extra_held_frames"] == 0
            and not video["initial_frame_included"]
            and video["reserved_receipt_sha256"] == evidence["reserved"]
            and video["video_sha256"] == audit["video_sha256"], "held-out video receipt mismatch")
    video_case, = [row for _, row in jobs if row["gate"] == "video"]
    require(video_case["recording_sha256"] == video["source_recording_sha256"], "video source recording differs")
    return dict(root=root, freeze=freeze, audit=audit, candidate=candidate, jobs=jobs,
                forecast_count=forecast_count, evidence=evidence, video=video)


def install_frozen_path():
    path = str(FROZEN / "src")
    if path not in sys.path:
        sys.path.insert(0, path)


def load_frozen_script(name):
    install_frozen_path()
    module_name = f"_pendulum_arena_frozen_{name}"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, FROZEN / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def check_runtime(freeze, allow_mismatch=False):
    install_frozen_path()
    from rage_mill.canonical_campaign import runtime_identity
    # This unmodified helper always requires MuJoCo 3.3.5 and one BLAS/OMP thread.
    current = runtime_identity()
    expected = freeze["identities"]["runtime"]
    differences = [key for key in sorted(expected) if current.get(key) != expected[key]]
    if differences and not allow_mismatch:
        raise ValueError("runtime differs from the frozen Linux/aarch64 environment: " + ", ".join(differences)
                         + ". Use --allow-runtime-mismatch only for an explicitly exploratory recheck.")
    return current, differences


def write_new_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
