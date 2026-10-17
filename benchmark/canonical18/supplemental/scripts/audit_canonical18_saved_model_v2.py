#!/usr/bin/env python3
"""Saved18 graph audit with prospective serialized-loader dependency coverage.

The first audit's failed runtime coverage stays immutable. Metadata preparation
loads the three saved function definitions and compressed archive headers to
bind lazily loaded source/native dependencies. It evaluates no graph or physics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def put(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(model, components, declaration, supervisor, out):
    """Bind fresh own-checkout source/runtime files without executing a model."""
    start, cpu = time.perf_counter(), time.process_time()
    require(not out.exists(), "fresh create-only audit preparation required")
    model, components, declaration, supervisor = (
        path.resolve(strict=True) for path in (model, components, declaration, supervisor))
    declared = json.loads(declaration.read_text())
    require(declared.get("n_links") == 18 and declared.get("plant_parity_band") == [3180000, 3180099],
            "inventoried prospective own-count declaration required")
    receipt = json.loads((model / "receipt.json").read_text())
    completion = json.loads((model.parent / "completion.json").read_text())
    launch = json.loads((model.parent / "launch.json").read_text())
    require(receipt.get("passed") is True and receipt.get("n_links") == 18
            and completion.get("naturally_closed_zero") is True
            and completion.get("child_returncode") == 0
            and completion.get("owned_group_cleanup")
            and all(completion.get("source_hashes_unchanged", {}).values())
            and completion.get("recipe_sha256") == sha(model.parent / "launch.json")
            and launch.get("output_path") == str(model)
            and all(Path(path).is_file() and sha(path) == digest
                    for path, digest in launch["source_hashes"].items()),
            "positive naturally closed own-count model and unchanged original owner required")
    # Import and serialized-loader setup binds dependencies before the freeze.
    # No graph or physics function is evaluated and no reset state is drawn.
    import casadi
    import mujoco
    import numpy
    import zipfile  # Bind compressed tape loader dependencies before runtime capture.
    import bz2
    import lzma
    from rage_mill import canonical_benchmark, canonical_dynamics
    from rage_mill.canonical_runtime_manifest import create_runtime_manifest, validate_runtime_manifest
    require(mujoco.__version__ == "3.3.5", "actual pinned native runtime required")
    retained_functions = [casadi.Function.load(str(model / (name + ".casadi")))
                          for name in ("continuous", "transition", "first")]
    with numpy.load(components, allow_pickle=False) as archive:
        component_fields = tuple(archive.files)
    source = Path(__file__).resolve()
    inputs_files = [source, components, declaration, supervisor,
                    Path(canonical_benchmark.__file__).resolve(), Path(canonical_dynamics.__file__).resolve(),
                    Path(sys.prefix) / "pyvenv.cfg"]
    inputs_files += [path.resolve() for path in model.parent.rglob("*") if path.is_file()]
    # Preserve the original model directory's retained generated C/libraries;
    # saved CasADi functions bind those original absolute runtime paths.
    manifest = create_runtime_manifest(package_names=("numpy", "scipy", "casadi", "mujoco"),
        model_directories=(model.parent,), extra_paths=inputs_files)
    pins = validate_runtime_manifest(manifest)
    out.mkdir(parents=True)
    put(out / "runtime-closure.json", manifest)
    runtime = dict(path=str(out / "runtime-closure.json"), sha256=sha(out / "runtime-closure.json"))
    pins[runtime["path"]] = runtime["sha256"]
    inputs = dict(schema="rage_mill.canonical18_saved_model_audit_inputs.v1",
        source_hashes=dict(sorted(pins.items())), model_directory=str(model),
        independent_component_arrays=str(components), declaration=str(declaration), runtime_closure=runtime,
        epsilon=1e-5, tolerances=dict(continuous=1e-8, transition=1e-10, jacobian=1e-10,
                                     native_derivative=2e-6),
        question="Does the actual saved eighteen JIT graph match a separately built unjitted graph and native maps/derivatives?",
        limits=dict(CPU_seconds=60, wall_seconds=120, native_held_map_calls=56),
        controller_qualified=False, reset_draws=0, physical_episodes=0, reserved_claims=0)
    put(out / "inputs.json", inputs)
    inputs_row = dict(path=str(out / "inputs.json"), sha256=sha(out / "inputs.json"))
    pins[inputs_row["path"]] = inputs_row["sha256"]
    recipe = dict(schema="rage_mill.canonical18_saved_model_audit_owner.v1",
        argv=[str(Path(sys.executable).absolute()), str(source), "--run", "--inputs", inputs_row["path"],
              "--expected-inputs-sha256", inputs_row["sha256"], "--out", str(out / "output")],
        cwd=str(out), environment=dict(PYTHONPATH=str(source.parents[1] / "src"),
            PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        CPU_cap_s=60, wall_cap_s=120, address_cap_bytes=8*1024**3,
        output_path=str(out / "output"), log=str(out / "worker.log"), receipt=str(out / "completion.json"),
        source_hashes=dict(sorted(pins.items())), runtime_closure=runtime,
        maximum_native_commands=56, actual_physical_episodes=0, reset_RNG_draws=0,
        formal_development_episodes=0, reserved_episodes=0, new_reserved_claims=0,
        purpose=inputs["question"], stop_criteria="one bounded audit; retain all failures and owner costs")
    put(out / "launch.json", recipe)
    result = dict(metadata_only=True, launch=dict(path=str(out / "launch.json"), sha256=sha(out / "launch.json")),
        inputs=inputs_row, runtime_closure=runtime, source_files=len(pins),
        native_commands=0, graph_evaluations=0, serialized_graph_definitions_loaded=len(retained_functions),
        compressed_component_fields=list(component_fields), reset_draws=0,
        CPU_seconds=time.process_time()-cpu, wall_seconds=time.perf_counter()-start)
    put(out / "preparation.json", result)
    return result


def verify(inputs, output):
    start, cpu = time.perf_counter(), time.process_time()
    pins = inputs["source_hashes"]
    require(all(Path(path).is_file() and sha(path) == digest for path, digest in pins.items()),
            "all actual source/model/runtime inputs must match before imports")
    import casadi as ca
    import mujoco
    import numpy as np
    from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol
    from rage_mill.canonical_dynamics import planning_functions
    from rage_mill.canonical_runtime_closure import runtime_binding_paths
    from rage_mill.canonical_runtime_manifest import validate_runtime_manifest

    runtime = inputs["runtime_closure"]
    require(sha(runtime["path"]) == runtime["sha256"], "prospective runtime manifest changed")
    runtime_pins = validate_runtime_manifest(json.loads(Path(runtime["path"]).read_text()))

    require(mujoco.__version__ == "3.3.5", "actual pinned native runtime required")
    model = Path(inputs["model_directory"])
    original = json.loads((model / "receipt.json").read_text())
    require(original["passed"] and original["n_links"] == 18, "actual positive own-count model required")
    env = CanonicalEnv(CanonicalProtocol(n_links=18))
    require(env.model.nq == env.model.nv == 19 and env.xml_sha256 == original["xml_sha256"],
            "actual independently verified count18 XML differs")
    frozen = [ca.Function.load(str(model / (name + ".casadi"))) for name in ("continuous", "transition", "first")]
    independent = planning_functions(env.model, jit=False)
    with np.load(inputs["independent_component_arrays"], allow_pickle=False) as tape:
        components, commands = tape["component_states"].copy(), tape["component_actions"].copy()
    with np.load(model / "lifted-initializer.npz", allow_pickle=False) as tape:
        states = np.r_[components, tape["states"][[0, 150, 300, 449]]]
        actions = np.r_[commands, tape["actions"][[0, 150, 300, 449]]]
    native_data = mujoco.MjData(env.model)
    calls, rows, arrays = 0, [], {}

    def native_map(state, action):
        nonlocal calls
        mujoco.mj_resetData(env.model, native_data)
        native_data.qpos[:], native_data.qvel[:] = state[:19], state[19:]
        native_data.ctrl[0] = 80*float(action)
        mujoco.mj_forward(env.model, native_data)
        for _ in range(4):
            mujoco.mj_step(env.model, native_data)
        calls += 1
        require(not any(w.number for w in native_data.warning)
                and np.isfinite(np.r_[native_data.qpos, native_data.qvel, native_data.qacc]).all()
                and abs(native_data.time-.02) < 1e-10, "invalid independent native model probe")
        return np.r_[native_data.qpos, native_data.qvel]

    for index, (state, action) in enumerate(zip(states, actions)):
        require(abs(action) <= 1. and action == float(np.float32(action)),
                "all base component commands must be delivered normalized float32")
        actual = native_map(state, action)
        calculated = [np.asarray(function(state, action)).ravel() for function in frozen[:2]]
        own = [np.asarray(function(state, action)).ravel() for function in independent[:2]]
        a, b = (np.asarray(value) for value in frozen[2](state, action))
        own_a, own_b = (np.asarray(value) for value in independent[2](state, action))
        epsilon = inputs["epsilon"]
        directions = np.stack([np.sin(np.arange(38)+.3), np.cos(np.arange(38)*.47+.2)])
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        direction_errors = []
        for direction in directions:
            numeric = (native_map(state+epsilon*direction, action)-native_map(state-epsilon*direction, action))/(2*epsilon)
            direction_errors.append(float(np.max(np.abs(a@direction-numeric))))
        numeric_b = (native_map(state, action+epsilon)-native_map(state, action-epsilon))/(2*epsilon)
        entry = dict(component=index, saved_vs_independent_continuous=float(np.max(np.abs(calculated[0]-own[0]))),
            saved_vs_independent_transition=float(np.max(np.abs(calculated[1]-own[1]))),
            saved_vs_independent_A=float(np.max(np.abs(a-own_a))),
            saved_vs_independent_B=float(np.max(np.abs(b-own_b))),
            maximum_native_transition_error=float(np.max(np.abs(calculated[1]-actual))),
            native_state_direction_errors=direction_errors,
            maximum_native_B_error=float(np.max(np.abs(b.ravel()-numeric_b))))
        limits = inputs["tolerances"]
        require(entry["saved_vs_independent_continuous"] <= limits["continuous"]
            and entry["saved_vs_independent_transition"] <= limits["transition"]
            and entry["saved_vs_independent_A"] <= limits["jacobian"]
            and entry["saved_vs_independent_B"] <= limits["jacobian"]
            and entry["maximum_native_transition_error"] <= limits["transition"]
            and max(direction_errors) <= limits["native_derivative"]
            and entry["maximum_native_B_error"] <= limits["native_derivative"],
            "actual saved planning model fails independent component " + str(index))
        rows.append(entry)
        arrays[f"A_{index}"], arrays[f"B_{index}"], arrays[f"native_transition_{index}"] = a, b, actual
    loaded = {path: sha(path) for path in runtime_binding_paths(
        package_names=(), include_stdlib=False, include_loaded=True)}
    missing = {path: digest for path, digest in loaded.items() if runtime_pins.get(path) != digest}
    require(not missing, "actual loaded source/native runtime was not prospectively frozen: " + json.dumps(missing, sort_keys=True))
    unchanged = {path: Path(path).is_file() and sha(path) == digest for path, digest in pins.items()}
    require(all(unchanged.values()) and not output.exists(), "input changed or output already exists")
    output.mkdir()
    with (output / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, states=states, actions=actions, **arrays)
    result = dict(schema="rage_mill.canonical18_saved_model_independent_audit.v1", passed=True,
        model_directory=str(model), n_links=18, state_dimension=38, xml_sha256=env.xml_sha256,
        independent_graph_jit=False, actual_saved_graph_jit=True,
        components=rows, native_held_map_calls=calls, epsilon=epsilon,
        action_derivative_scope="smooth held input before quantizer; all base inputs float32 delivered values",
        source_hashes_unchanged=unchanged, runtime_closure=runtime,
        actual_loaded_file_sha256=loaded, actual_loaded_runtime_complete=True,
        runtime=dict(python=sys.version, numpy=np.__version__, mujoco=mujoco.__version__, casadi=ca.__version__),
        arrays=dict(path=str(output / "arrays.npz"), sha256=sha(output / "arrays.npz")),
        controller_calls=0, development_episodes=0, reserved_episodes=0, claims=0,
        controller_qualified=False, CPU_seconds=time.process_time()-cpu, wall_seconds=time.perf_counter()-start)
    put(output / "receipt.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--expected-inputs-sha256")
    parser.add_argument("--prepare", action="store_true")
    for name in ("model-directory", "component-arrays", "declaration", "supervisor"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.prepare != args.run, "exactly one explicit metadata preparation or bounded run required")
    if args.prepare:
        require(all(getattr(args, name) is not None for name in
                    ("model_directory", "component_arrays", "declaration", "supervisor")),
                "own-count model/component/declaration/supervisor inputs required")
        print(json.dumps(prepare(args.model_directory, args.component_arrays, args.declaration,
                                 args.supervisor, args.out.resolve())))
        return
    require(args.run and not args.out.exists() and sha(args.inputs) == args.expected_inputs_sha256,
            "explicit bounded run with unchanged inputs and fresh output required")
    inputs = json.loads(args.inputs.read_text())
    inputs["source_hashes"][str(args.inputs.resolve())] = args.expected_inputs_sha256
    result = verify(inputs, args.out)
    print(json.dumps(dict(passed=result["passed"], CPU_seconds=result["CPU_seconds"], wall_seconds=result["wall_seconds"])))


if __name__ == "__main__":
    main()
