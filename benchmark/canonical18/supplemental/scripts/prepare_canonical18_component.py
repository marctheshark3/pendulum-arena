"""Metadata-only preparation for one prospective own-eighteen design owner.

Imports bind complete source/package/native runtime inputs without constructing
models, executing numerical functions, drawing reset RNG or issuing commands.
The immutable recipe is executed only by the separately pinned supervisor.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import runpy
import sys
import time

from rage_mill.canonical_evaluation import pinned_file, read_pinned, require, write_json_exclusive
from rage_mill.canonical_runtime_manifest import create_runtime_manifest, validate_runtime_manifest

STAGES = {
    "model": {"CPU": 300, "wall": 480, "native_maps": 50, "physical_episodes": 0},
    "orbit": {"CPU": 120, "wall": 180, "native_maps": 96, "physical_episodes": 0},
    "launch": {"CPU": 120, "wall": 180, "native_maps": 2550, "physical_episodes": 0},
    "bridge": {"CPU": 300, "wall": 480, "native_maps": 700, "physical_episodes": 0},
}


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument("stage", choices=STAGES)
    for name in ("declaration", "supervisor", "out"):
        parser.add_argument("--"+name, required=True, type=Path)
    for name in ("model", "orbit", "launch", "donor"):
        parser.add_argument("--"+name, type=Path)
    args = parser.parse_args()
    cpu, wall = time.process_time(), time.monotonic()
    root = Path(__file__).resolve().parents[1]
    source = root/"scripts/synthesize_canonical18.py"
    declaration = json.loads(args.declaration.read_text())
    require(declaration.get("n_links") == 18 and declaration.get("synthesis_band") == [3181000, 3181999],
            "inventoried prospective eighteen synthesis allocation required")
    required = {"model": ("donor",), "orbit": ("model",), "launch": ("model",),
                "bridge": ("model", "orbit", "launch", "donor")}[args.stage]
    require(all(getattr(args, name) is not None for name in required), "stage inputs missing")
    # All imports occur before prospective runtime capture. This is metadata only.
    runpy.run_path(str(source), run_name="canonical18_component_metadata_only")
    for name in ("mujoco", "numpy", "scipy", "casadi"):
        importlib.import_module(name)
    import mujoco
    require(mujoco.__version__ == "3.3.5", "native MuJoCo3.3.5 runtime required")
    inputs = [args.declaration.resolve(strict=True), args.supervisor.resolve(strict=True), source,
              Path(__file__).resolve(), Path(sys.executable).resolve(strict=True),
              Path(sys.prefix)/"pyvenv.cfg"]
    model_directories = []
    for name in required:
        path = getattr(args, name).resolve(strict=True)
        if path.is_dir():
            inputs.extend(p.resolve() for p in path.rglob("*") if p.is_file())
            model_directories.append(path)
            # Close the predecessor's ownership lifecycle as well as its outputs.
            for basename in ("launch.json", "completion.json", "runtime-closure.json"):
                parent = path.parent/basename
                if parent.is_file():
                    inputs.append(parent.resolve())
        else:
            inputs.append(path)
    # Pin only actually imported repository sources, so independent additive
    # online-policy development cannot invalidate a running synthesis owner.
    for module in tuple(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if value and Path(value).is_file() and Path(value).resolve().is_relative_to(root/"src"):
            inputs.append(Path(value).resolve())
    manifest = create_runtime_manifest(package_names=("numpy", "scipy", "casadi", "mujoco"),
                                       model_directories=model_directories, extra_paths=inputs)
    pins = validate_runtime_manifest(manifest)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    runtime = write_json_exclusive(out/"runtime-closure.json", manifest)
    pins[runtime["path"]] = runtime["sha256"]
    argv = [str(Path(sys.executable).absolute()), str(source), args.stage, "--out", str(out/"output")]
    for name in required:
        argv.extend(["--"+name, str(getattr(args, name).resolve(strict=True))])
    limits = STAGES[args.stage]
    recipe = {
        "schema": "rage_mill.canonical18_component_owner.v1", "argv": argv, "cwd": str(out),
        "environment": {"PYTHONPATH": str(root/"src"), "PYTHONDONTWRITEBYTECODE": "1",
                        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "CPU_cap_s": limits["CPU"], "wall_cap_s": limits["wall"], "address_cap_bytes": 8*1024**3,
        "output_path": str(out/"output"), "log": str(out/"worker.log"),
        "receipt": str(out/"completion.json"), "source_hashes": dict(sorted(pins.items())),
        "runtime_closure": runtime, "purpose": "Own eighteen "+args.stage+" component; every failure retained",
        "n_links": 18, "declared_synthesis_band": [3181000, 3181999],
        "diagnostic_labels": [3181020,3181021,3181022] if args.stage == "launch" else [],
        "maximum_native_commands": limits["native_maps"], "actual_full_episodes": 0,
        "formal_development_episodes": 0, "reserved_episodes": 0, "controller_qualified": False,
        "optimizer_bounds": {"maximum_iterations": 150, "maximum_solver_CPU_seconds": 180}
                             if args.stage == "bridge" else None,
        "stop_criteria": "one bounded owner; retain every native defect and negative result; no retries or replacement",
    }
    launch = write_json_exclusive(out/"launch.json", recipe)
    preparation = write_json_exclusive(out/"preparation.json", {
        "schema": "rage_mill.canonical18_component_preparation.v1", "metadata_only": True,
        "launch": launch, "runtime_closure": runtime, "supervisor": pinned_file(args.supervisor),
        "input_files": len(pins), "model_constructions": 0, "native_commands": 0,
        "reset_draws": 0, "controller_calls": 0, "CPU_seconds": time.process_time()-cpu,
        "wall_seconds": time.monotonic()-wall,
    })
    print(json.dumps({"launch": launch, "preparation": preparation,
                      "supervisor": pinned_file(args.supervisor)}, allow_nan=False))


if __name__ == "__main__":
    main()
