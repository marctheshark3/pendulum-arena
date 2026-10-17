"""Declare one fresh exact hanging diagnostic with complete prospective pins."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys
import time

from rage_mill.canonical_evaluation import pinned_file, require, write_json_exclusive
from rage_mill.canonical_runtime_manifest import create_runtime_manifest, validate_runtime_manifest


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    for name in ("controller", "reference", "feedback", "declaration", "supervisor", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3181040)
    args = parser.parse_args()
    cpu, wall = time.process_time(), time.monotonic()
    root = Path(__file__).resolve().parents[1]
    declaration = json.loads(args.declaration.read_text())
    require(declaration.get("n_links") == 18 and declaration.get("synthesis_band") == [3181000, 3181999],
            "inventoried eighteen diagnostic declaration required")
    require(3181040 <= args.seed <= 3181199, "allocated native-value diagnostic band required")
    worker = root / "scripts/probe_canonical18_native_value_episode.py"
    runpy.run_path(str(worker), run_name="diagnostic_metadata_only")
    import mujoco
    require(mujoco.__version__ == "3.3.5", "actual MuJoCo3.3.5 required")
    inputs = [worker, Path(__file__).resolve(), args.declaration.resolve(), args.supervisor.resolve(),
              Path(sys.executable).resolve(), Path(sys.prefix) / "pyvenv.cfg"]
    for name in ("controller", "reference", "feedback"):
        path = getattr(args, name).resolve(strict=True)
        inputs.append(path)
        owner = path.parent.parent
        for basename in ("launch.json", "completion.json", "runtime-closure.json"):
            if (owner / basename).is_file():
                inputs.append(owner / basename)
        if (path.parent / "receipt.json").is_file():
            inputs.append(path.parent / "receipt.json")
    for module in tuple(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if value and Path(value).is_file() and Path(value).resolve().is_relative_to(root / "src"):
            inputs.append(Path(value).resolve())
    manifest = create_runtime_manifest(package_names=("numpy", "mujoco"), extra_paths=inputs)
    pins = validate_runtime_manifest(manifest)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    runtime = write_json_exclusive(out / "runtime-closure.json", manifest)
    pins[runtime["path"]] = runtime["sha256"]
    argv = [str(Path(sys.executable).absolute()), str(worker), "--out", str(out / "output"),
            "--active-route-start", "0", "--cart-error-offset", "0.", "--seed", str(args.seed),
            "--episodes", "1", "--exact", "--runtime-closure", runtime["path"]]
    for name in ("controller", "reference", "feedback"):
        argv.extend(["--" + name, str(getattr(args, name).resolve())])
    launch = write_json_exclusive(out / "launch.json", {
        "schema": "rage_mill.canonical18_diagnostic_owner.v1", "argv": argv, "cwd": str(out),
        "environment": {"PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1",
                        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "CPU_cap_s": 90, "wall_cap_s": 150, "address_cap_bytes": 8 * 1024**3,
        "output_path": str(out / "output"), "log": str(out / "worker.log"),
        "receipt": str(out / "completion.json"), "source_hashes": dict(sorted(pins.items())),
        "runtime_closure": runtime, "purpose": "One fresh own-count exact native-value hanging diagnostic",
        "n_links": 18, "seed": args.seed, "exact": True, "physical_episodes": 1,
        "maximum_actual_commands": 1500, "maximum_native_candidates": 10400,
        "maximum_total_native_commands": 11900, "formal_development_episodes": 0,
        "reserved_episodes": 0, "controller_qualified": False,
        "stop_criteria": "one actual episode; retain every failed prefix; no tuning or retry inside owner",
    })
    preparation = write_json_exclusive(out / "preparation.json", {
        "schema": "rage_mill.canonical18_diagnostic_preparation.v1", "metadata_only": True,
        "launch": launch, "runtime_closure": runtime, "supervisor": pinned_file(args.supervisor),
        "native_commands": 0, "model_constructions": 0, "reset_draws": 0,
        "CPU_seconds": time.process_time() - cpu, "wall_seconds": time.monotonic() - wall,
    })
    print(json.dumps({"launch": launch, "preparation": preparation}))


if __name__ == "__main__":
    main()
