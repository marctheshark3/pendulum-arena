"""NEW public inventory/loading adapter; outside the original candidate identity.

Default inspection reads immutable files/arrays only. Explicit load_controller
creates a NEW local loading view; it never changes the original identity and
does not establish portable execution or qualification.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ORIGINAL_IDENTITY_SHA256 = "61993d118c341001ef7973fce77a200f85dd86588b78919bc03d629e09c95268"
FROZEN_MANIFEST_SHA256 = "e75eb97379ebbb7be95482e3cc71a74496a556bcf035a343dc7512fe12903a7e"
SOURCE_COMMIT = "b84c6413974d5c0831cdf261ce15e58ba65cd3f5"
MAX_FILES, MAX_BYTES = 96, 64 * 1024**2
SHAPES = {
    "states": (651, 36), "actions": (650,), "gains": (650, 36), "offsets": (650,),
    "balance_gain": (36,), "park_gain": (36,), "parking_target": (36,),
    "native_defects": (650, 36), "terminal_root": (36, 36), "park_ticks": (),
    "value_states": (701, 36), "value_actions": (700,), "value_gains": (700, 36),
    "value_offsets": (700,), "value_roots": (701, 36, 36),
    "value_residuals": (701, 36), "value_Q": (36, 36),
    "value_native_defects": (700, 36),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            digest.update(part)
    return digest.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError("nonfinite JSON refused: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def safe_file(relative):
    require(isinstance(relative, str) and relative and "\\" not in relative,
            "literal relative POSIX archive path required")
    path = PurePosixPath(relative)
    require(not path.is_absolute() and path.as_posix() == relative and ".." not in path.parts,
            "noncanonical or traversing archive path refused")
    current = HERE
    for component in path.parts:
        current = current / component
        require(not current.is_symlink(), "symlink archive entry refused")
    require(current.is_file() and current.resolve().is_relative_to(HERE),
            "regular contained archive file required")
    return current


def verify_row(relative, row):
    require(isinstance(row, dict) and type(row.get("bytes")) is int
            and 0 <= row["bytes"] <= MAX_BYTES
            and isinstance(row.get("sha256"), str)
            and re.fullmatch("[0-9a-f]{64}", row["sha256"]), "complete finite file pin required")
    path = safe_file(relative)
    require(path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"],
            "public archive bytes differ: " + relative)
    return path


def verify_export(*, expected_files_sha256):
    """Verify the externally hash-selected exact public inventory; no imports/physics."""
    require(isinstance(expected_files_sha256, str)
            and re.fullmatch("[0-9a-f]{64}", expected_files_sha256),
            "explicit independently published files.json SHA256 required")
    files_path = safe_file("files.json")
    require(sha256(files_path) == expected_files_sha256, "public file manifest hash differs")
    public = read_json(files_path)
    require(public.get("schema") == "pendulum_arena.canonical17_public_source_files.v1"
            and public.get("preparation_only") is True
            and public.get("qualification_authority") is False,
            "preparatory own17 public inventory required")
    rows = public.get("files")
    require(isinstance(rows, list) and 1 <= len(rows) <= MAX_FILES,
            "bounded public file list required")
    by_name, total_bytes = {}, 0
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"path", "sha256", "bytes"}
                and row["path"] not in by_name and row["path"] != "files.json",
                "one distinct canonical file row required")
        verify_row(row["path"], row)
        total_bytes += row["bytes"]
        require(total_bytes <= MAX_BYTES, "public inventory exceeds byte budget")
        by_name[row["path"]] = row
    actual = set()
    for path in HERE.rglob("*"):
        require(not path.is_symlink(), "public directory contains a symlink")
        if path.is_file():
            actual.add(path.relative_to(HERE).as_posix())
    require(actual == set(by_name) | {"files.json"}, "unlisted/missing public archive file")
    manifest_path = safe_file("frozen-manifest.json")
    require(sha256(manifest_path) == FROZEN_MANIFEST_SHA256,
            "original-path frozen source inventory differs")
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == "pendulum_arena.canonical17_candidate_source_inventory.v1"
            and manifest.get("preparation_only") is True
            and manifest.get("qualification_authority") is False
            and manifest.get("original_execution_paths_relocated") is False
            and manifest.get("native_runtime_binaries_included") is False
            and manifest.get("private_Git_history_included") is False,
            "wrong original candidate source inventory")
    for name, row in manifest["files"].items():
        require(by_name.get(name) == {"path": name, **row}, "frozen/public inventory mismatch")
        verify_row(name, row)
    mapping = {}
    for item in manifest["source_mapping"]:
        original, name = item["original"], item["relative_path"]
        require(isinstance(original.get("path"), str) and Path(original["path"]).is_absolute()
                and original["path"] not in mapping, "distinct original absolute source path required")
        verify_row(name, original)
        mapping[original["path"]] = item
    require(sha256(safe_file("metadata/identity.json")) == ORIGINAL_IDENTITY_SHA256
            and manifest["original_identity"]["sha256"] == ORIGINAL_IDENTITY_SHA256,
            "original candidate02 identity differs")
    identity = read_json(safe_file("metadata/identity.json"))
    require(identity.get("schema") == "rage_mill.canonical_evaluation_identity.v1"
            and identity.get("source_commit") == SOURCE_COMMIT
            and identity.get("protocol", {}).get("n_links") == 17
            and len(identity.get("sources", [])) == manifest["original_identity_source_rows"] == 41,
            "own17 original identity/source inventory required")
    for original in [identity["controller"], identity["configuration"], *identity["sources"]]:
        require(mapping.get(original["path"], {}).get("original") == original,
                "original identity pin omitted/changed in public mapping")
    require(identity["native_xml_sha256"] == sha256(safe_file("payload/model.xml")),
            "own17 XML differs from original identity")
    configuration = read_json(safe_file("payload/configuration.json"))
    require(configuration.get("protocol") == identity["protocol"]
            and configuration.get("policy_parameters") == identity["policy_parameters"]
            and configuration.get("controller") == identity["controller"],
            "original configuration/controller identity differs")
    return dict(identity=identity, source_mapping=mapping, manifest=manifest,
                files_sha256=expected_files_sha256, files=len(rows), bytes=total_bytes)


def inspect_tables(*, expected_files_sha256):
    """Strict own36/full700 stored-array check. Imports NumPy only, no factory."""
    verified = verify_export(expected_files_sha256=expected_files_sha256)
    import numpy as np
    data = safe_file("payload/controller.npz").read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        require(len(names) == len(set(names)) == len(SHAPES)
                and set(names) == {name + ".npy" for name in SHAPES}
                and sum(entry.file_size for entry in entries) <= 16 * 1024**2
                and all(entry.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        and not entry.flag_bits & 1 for entry in entries),
                "strict bounded unencrypted own17 NPZ fields required")
    with np.load(io.BytesIO(data), allow_pickle=False) as saved:
        require(set(saved.files) == set(SHAPES), "exact own17 archive fields required")
        tables = {name: saved[name].copy() for name in saved.files}
    for name, shape in SHAPES.items():
        value = tables[name]
        require(value.shape == shape, "own17 array shape differs: " + name)
        if name == "park_ticks":
            require(value.dtype == np.dtype("int64") and value.item() == 850,
                    "literal850 native parking scalar required")
        else:
            require(value.dtype == np.dtype("float64") and np.isfinite(value).all(),
                    "finite binary64 stored table required: " + name)
    for short, full in (("states", "value_states"), ("actions", "value_actions"),
                        ("gains", "value_gains"), ("offsets", "value_offsets"),
                        ("native_defects", "value_native_defects")):
        require(tables[short].tobytes() == tables[full][:len(tables[short])].tobytes(),
                "literal ordinary650/full700 prefix differs: " + short)
    require(not np.any(tables["terminal_root"]) and not np.any(tables["value_roots"][-1])
            and not np.any(tables["value_residuals"][-1]), "zero terminal at700 required")
    require(np.all(np.abs(tables["value_actions"]) <= 1.)
            and tables["value_actions"].tobytes()
                == tables["value_actions"].astype(np.float32).astype(np.float64).tobytes(),
            "all700 actions must be delivered float32 values")
    cumulative = np.tril(np.ones((17, 17))); gram = cumulative.T @ cumulative
    q = np.eye(36); q[0, 0] = q[18, 18] = .1
    q[1:18, 1:18] += 100. * gram
    q[19:, 19:] = gram + .01 * np.eye(17)
    require(q.tobytes() == tables["value_Q"].tobytes(), "own17 state cost differs")
    return dict(original_identity_sha256=ORIGINAL_IDENTITY_SHA256,
                files_sha256=verified["files_sha256"], n_links=17, state_dimension=36,
                full_value_commands=700, ordinary_route_commands=650, park_ticks=850,
                field_shapes={name: list(value.shape) for name, value in tables.items()},
                literal_prefixes=True, zero_terminal_at700=True, float32_actions=True,
                factory_calls=0, controller_calls=0, native_commands=0, reset_RNG_draws=0,
                qualification=False, scope="stored-array inspection only")


def controller_loading_view(verified):
    """NEW in-memory path view; never overwrite/resign original JSON or runtime."""
    view = copy.deepcopy(verified["identity"])
    for row in [view["controller"], view["configuration"], *view["sources"]]:
        item = verified["source_mapping"][row["path"]]
        row["path"] = str(safe_file(item["relative_path"]))
    return view


def load_controller(*, expected_files_sha256, allow_runtime_mismatch=False):
    """Explicit exploratory loading API; never invoked by the default audit CLI.

    Construction issues no command/model/reset. Calling the returned controller
    later creates native forecast models at launch and incurs all declared costs.
    No original qualification or numerical portability follows from loading.
    """
    require(type(allow_runtime_mismatch) is bool, "explicit boolean runtime policy required")
    verified = verify_export(expected_files_sha256=expected_files_sha256)
    inspect_tables(expected_files_sha256=expected_files_sha256)
    require(not any(name == "rage_mill" or name.startswith("rage_mill.") for name in sys.modules),
            "fresh interpreter required; existing rage_mill imports refused")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HERE / "frozen/src"))
    evaluation = importlib.import_module("rage_mill.canonical_evaluation")
    current, expected = evaluation.runtime_metadata(), verified["identity"]["runtime"]
    differences = {key: {"expected": value, "actual": current.get(key)}
                   for key, value in expected.items() if current.get(key) != value}
    require(current["mujoco"] == "3.3.5"
            and all(value == "1" for value in current["threads"].values()),
            "MuJoCo3.3.5 and one OMP/BLAS thread required")
    require(not differences or allow_runtime_mismatch,
            "metadata runtime mismatch; explicit exploratory override required")
    module = importlib.import_module("rage_mill.canonical17_native_value_policy")
    policy = module.controller_factory(controller_loading_view(verified))
    return policy, dict(original_identity_sha256=ORIGINAL_IDENTITY_SHA256,
                        loading_view_is_new=True, original_evidence_modified=False,
                        runtime_differences=differences, runtime_binary_hashes_verified=False,
                        factory_constructions=1, controller_calls=0, native_models=0,
                        native_steps=0, reset_RNG_draws=0, qualification=False,
                        portability_established=False)
