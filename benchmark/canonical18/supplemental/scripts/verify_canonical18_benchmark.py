#!/usr/bin/env python3
"""Independently audit eighteen's published plant and finite native components.

Preparation imports the actual runtime and freezes inputs without physics. The
bounded run executes only public XML generation/reset/scoring methods, 100
diagnostic commands and four deterministic derivative probes. No policy,
development episode or held-out label is initialized.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = "b84c6413974d5c0831cdf261ce15e58ba65cd3f5"
FROZEN14 = "3277b4173a9333819bf35acb848e384f52a227ec"
DIAGNOSTIC_SEED = 3180000
ATTRIBUTES = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
    "dof_armature", "dof_damping", "dof_frictionloss", "jnt_axis", "jnt_pos", "jnt_range",
    "jnt_limited", "jnt_type", "jnt_stiffness", "actuator_gear", "actuator_ctrlrange",
    "actuator_trnid", "actuator_ctrllimited", "jnt_solref", "jnt_solimp", "body_parentid")
GEOM_ATTRIBUTES = ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype",
                   "geom_conaffinity", "geom_friction")
METHODS = ("reset", "_progress_value", "_angles", "_absolute_angular_velocity", "_is_upright",
           "_update_upright_tracking", "_termination_reason", "_success")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def row(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def put(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def same_bytes(first, second):
    return first.dtype == second.dtype and first.shape == second.shape and first.tobytes() == second.tobytes()


def runtime_files():
    paths = {Path(sys.executable), Path(sys.executable).resolve(), Path(sys.prefix) / "pyvenv.cfg"}
    for module in tuple(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if value and Path(value).is_file():
            paths.add(Path(value).resolve())
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith("/") and Path(parts[5]).is_file():
            paths.add(Path(parts[5]).resolve())
    # CasADi may load its own solver library lazily. Freeze every native library
    # in these actual package directories before graph construction as well.
    for name in ("mujoco", "numpy", "casadi"):
        root = Path(sys.modules[name].__file__).resolve().parent
        paths.update(path.resolve() for path in root.rglob("*.so*") if path.is_file())
    return {str(path): sha(path) for path in sorted(paths) if path.is_file()}


def modules(upstream):
    sys.dont_write_bytecode = True
    import mujoco
    import numpy as np
    import numpy.random
    import yaml
    import casadi
    from rage_mill import canonical_benchmark as own
    from rage_mill import canonical_dynamics as dynamics
    require(mujoco.__version__ == "3.3.5", "native MuJoCo3.3.5 required")
    sys.path.insert(0, str(Path(upstream) / "src"))
    xml = importlib.import_module("gcartpole.mjxml")
    morphology = importlib.import_module("gcartpole.morphology")
    angles = importlib.import_module("gcartpole.angles")
    return mujoco, np, yaml, casadi, own, dynamics, xml, morphology, angles


def source_methods(path, np, mj, wrap_angle):
    source = path.read_text()
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "NLinkCartPoleEnv")
    nodes = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in METHODS]
    require({node.name for node in nodes} == set(METHODS), "all exact source methods required")
    namespace = dict(np=np, mujoco=mj, wrap_angle=wrap_angle)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    hashes = {node.name: hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
              for node in nodes}
    return namespace, hashes


def source_context(namespace, config, model, data, np):
    context = SimpleNamespace(n=18, env_cfg=config, progress=1., model=model, data=data,
        action_route=None, last_action_norm=np.zeros(1, dtype=np.float32),
        last_policy_action_norm=np.zeros(1, dtype=np.float32), rail_limit=3., dt=.02,
        _potential_energy=lambda: 0., _handoff_distance=lambda: None,
        _get_obs=lambda: None, _info=lambda: {})
    for name in METHODS:
        setattr(context, name, MethodType(namespace[name], context))
    return context


def scoring_controls(own, namespace, config, model, mj, np):
    """Compare actual published tracking/termination methods on synthetic tapes."""
    p = own.CanonicalProtocol(n_links=18)
    results = []
    for case in ("250_anywhere", "249_only", "strict_boundary", "cumulative_angles",
                 "wrapped_upright", "high_speed_no_gate", "late_rail_escape", "incomplete",
                 "initial_does_not_count", "separated_holds"):
        states = np.zeros((1501, 38))
        states[:, 1] = np.pi
        states[100:350, 1] = 0.
        if case == "249_only":
            states[349, 1] = np.pi
        elif case == "strict_boundary":
            states[200, 1] = .15
        elif case == "cumulative_angles":
            states[100:350, 1:19] = .02
        elif case == "wrapped_upright":
            states[100:350, 1] = 2*np.pi
        elif case == "high_speed_no_gate":
            states[100:350, 19:] = 100.
        elif case == "late_rail_escape":
            states[-1, 0] = np.nextafter(3., np.inf)
        elif case == "incomplete":
            states = states[:350]
        elif case == "initial_does_not_count":
            states[:, 1] = np.pi
            states[:250, 1] = 0.
        elif case == "separated_holds":
            states[:, 1] = np.pi
            states[100:225, 1] = 0.
            states[300:425, 1] = 0.
        data = mj.MjData(model)
        context = source_context(namespace, config, model, data, np)
        # Reset initializes bookkeeping from a hanging state. The scored tape's
        # initial sample is deliberately excluded, as required by this task.
        context.reset(seed=DIAGNOSTIC_SEED)
        failed = False
        for tick, state in enumerate(states[1:], 1):
            data.qpos[:], data.qvel[:] = state[:19], state[19:]
            context.step_count = tick
            context._update_upright_tracking()
            failed |= context._termination_reason() is not None
        expected = len(states) == 1501 and not failed and context._success()
        measured = own.score_tape(states, np.zeros(len(states)-1), p)
        require(measured["success"] == bool(expected)
            and measured["longest_upright_ticks"] == context.max_upright_streak_steps,
            "source/scorer boundary differs: " + case)
        results.append(dict(case=case, success=measured["success"],
            longest_upright_ticks=measured["longest_upright_ticks"], source_matches=True))
    return results


def prepare(snapshot, output, declaration):
    require(not output.exists(), "fresh create-only input preparation required")
    declaration = declaration.resolve(strict=True)
    declared = json.loads(declaration.read_text())
    require(declared.get("n_links") == 18
        and declared.get("plant_parity_band") == [3180000, 3180099]
        and declared.get("upstream") == PUBLIC,
        "inventoried prospective count18 parity declaration required")
    snapshot = snapshot.resolve()
    manifest = json.loads((snapshot / "manifest.json").read_text())
    pins = {str(snapshot / item["commit"] / item["path"]): item["sha256"] for item in manifest["files"]}
    pins[str(snapshot / "manifest.json")] = sha(snapshot / "manifest.json")
    pins[str(declaration)] = sha(declaration)
    for name in ("scripts/verify_canonical18_benchmark.py", "src/rage_mill/canonical_benchmark.py",
                 "src/rage_mill/canonical_dynamics.py", "tests/test_canonical18_benchmark.py"):
        pins[str(ROOT / name)] = sha(ROOT / name)
    require(all(sha(path) == value for path, value in pins.items()), "pinned source snapshot differs")
    modules(snapshot / PUBLIC)
    pins.update(runtime_files())
    output.mkdir(parents=True)
    inputs = dict(schema="rage_mill.canonical18_plant_parity_inputs.v1", source_hashes=pins,
        upstream_snapshot=str(snapshot), declaration=row(declaration), diagnostic_seed=DIAGNOSTIC_SEED, action_seed=DIAGNOSTIC_SEED+1,
        diagnostic_band=[3180000, 3180099], reserved_labels_initialized=0,
        numerical_plan=dict(transition_commands=100, deterministic_component_states=4,
            finite_difference_epsilons=[1e-5, 1e-6], full_state_and_action_columns=True),
        tolerances=dict(continuous=1e-8, held_map=1e-10, state_jacobian=2e-6, input_jacobian=2e-6),
        limits=dict(CPU_seconds=120, wall_seconds=180, address_bytes=8*1024**3),
        question="Does count18 independently match published compiled mechanics, reset draw order, strict scoring, native held-map and derivatives?",
        stop_criteria="one bounded run; retain every failure; no policies or cohort episodes")
    put(output / "inputs.json", inputs)
    return row(output / "inputs.json")


def verify(inputs, output):
    started, cpu = time.perf_counter(), time.process_time()
    pins = inputs["source_hashes"]
    require(pins and all(Path(path).is_file() and sha(path) == digest for path, digest in pins.items()),
            "all source/runtime/input pins must match before imports")
    snapshot = Path(inputs["upstream_snapshot"])
    mj, np, yaml, ca, own, dynamics, xml_module, morphology, angles = modules(snapshot / PUBLIC)
    require(all(pins.get(path) == digest for path, digest in runtime_files().items()),
            "actual imported runtime is not frozen")
    require(inputs["diagnostic_seed"] == DIAGNOSTIC_SEED and inputs["action_seed"] == DIAGNOSTIC_SEED+1,
            "only prospectively declared diagnostic labels permitted")
    cfg = yaml.safe_load((snapshot / PUBLIC / "configs/swingup18_uniform.yaml").read_text())
    config = cfg["env"]
    require(config["n_links"] == 18, "official count18 configuration required")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    comparisons = [item for item in manifest["files"] if item["commit"] == FROZEN14]
    require(len(comparisons) == 11 and all(item["matches_public_pin"] for item in comparisons),
            "all eleven original frozen14 source comparisons required")
    for item in comparisons:
        require((snapshot / PUBLIC / item["path"]).read_bytes() ==
                (snapshot / FROZEN14 / item["path"]).read_bytes(), "frozen14/public source differs")
    morph = morphology.build_morphology(config, cfg["morphology"], 1.)
    source_xml = xml_module.generate_nlink_cartpole_xml(morph, **{key: config[key] for key in (
        "cart_mass", "rail_limit", "force_limit", "timestep", "cart_damping",
        "cart_frictionloss", "joint_armature", "link_radius")})
    original = mj.MjModel.from_xml_string(source_xml)
    p = own.CanonicalProtocol(n_links=18)
    env = own.CanonicalEnv(p)
    static = {name: bool(np.array_equal(getattr(original, name), getattr(env.model, name))) for name in ATTRIBUTES}
    names = ["cart_geom", *[f"link_{index}_geom" for index in range(1, 19)]]
    original_ids = [mj.mj_name2id(original, mj.mjtObj.mjOBJ_GEOM, name) for name in names]
    own_ids = [mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_GEOM, name) for name in names]
    geometry = {name: bool(np.array_equal(getattr(original, name)[original_ids],
                                         getattr(env.model, name)[own_ids])) for name in GEOM_ATTRIBUTES}
    require(min(original_ids) >= 0 and min(own_ids) >= 0 and all(geometry.values())
        and np.all(env.model.geom_type[own_ids[1:]] == mj.mjtGeom.mjGEOM_CAPSULE)
        and np.all(env.model.geom_size[own_ids[1:], 0] == .025),
        "compiled count18 capsule geometry differs")
    require(all(static.values()) and original.nq == original.nv == env.model.nq == env.model.nv == 19
        and original.nu == env.model.nu == 1 and np.array_equal(original.opt.gravity, env.model.opt.gravity)
        and original.opt.timestep == env.model.opt.timestep == .005
        and original.opt.integrator == env.model.opt.integrator == mj.mjtIntegrator.mjINT_RK4
        and original.opt.iterations == env.model.opt.iterations == 20, "compiled18 arrays/options differ")
    require(np.array_equal(env.model.body_mass[2:], np.full(18, .0555555556))
        and np.array_equal(env.model.body_pos[3:, 2], np.full(17, .166666667))
        and np.array_equal(env.model.dof_damping[1:], np.full(18, .000833333333)),
        "native .9g physical serialization differs")
    namespace, method_hashes = source_methods(snapshot / PUBLIC / "src/gcartpole/env.py", np, mj, angles.wrap_angle)
    data = mj.MjData(original)
    context = source_context(namespace, config, original, data, np)
    context.reset(seed=DIAGNOSTIC_SEED)
    noisy = np.r_[data.qpos, data.qvel]
    require(same_bytes(noisy, own.hanging_state(p, DIAGNOSTIC_SEED)), "published noisy draw order differs")
    exact_config = dict(config, init_cart_noise=0., init_cart_vel_noise=0., init_angle_noise=0., init_vel_noise=0.)
    exact_context = source_context(namespace, exact_config, original, mj.MjData(original), np)
    exact_context.reset(seed=DIAGNOSTIC_SEED)
    exact = np.r_[exact_context.data.qpos, exact_context.data.qvel]
    require(same_bytes(exact, own.hanging_state(p, DIAGNOSTIC_SEED, exact=True)), "published exact hanging differs")
    env.reset(noisy)
    actions = np.random.default_rng(DIAGNOSTIC_SEED+1).uniform(-.3, .3, 100).astype(np.float32)
    ours, theirs = [env.state], [noisy.copy()]
    for action in actions:
        state, delivered = env.step(action)
        require(delivered == float(action), "immediate float32 command differs")
        data.ctrl[0] = 80*float(action)
        for _ in range(4):
            before, warnings = data.time, tuple(w.number for w in data.warning)
            mj.mj_step(original, data)
            require(tuple(w.number for w in data.warning) == warnings and abs(data.time-before-.005) < 1e-10
                and np.isfinite(np.r_[data.qpos, data.qvel, data.qacc]).all(), "invalid published native integration")
        ours.append(state)
        theirs.append(np.r_[data.qpos, data.qvel])
    require(same_bytes(np.asarray(ours), np.asarray(theirs)) and env.ticks == 100,
            "100 held-input transitions must be byte exact")
    scorer = scoring_controls(own, namespace, config, original, mj, np)
    f, transition, first = dynamics.planning_functions(env.model, jit=False)
    component_states = np.zeros((4, 38))
    component_states[0, 1] = np.pi
    absolute = .06*np.sin(np.linspace(0., np.pi, 18))
    component_states[2, 0], component_states[2, 19] = .2, .15
    component_states[2, 1:19] = np.diff(np.r_[0., absolute])
    component_states[2, 20:] = .2*np.cos(np.arange(18))
    component_states[3] = .37*np.sin(np.arange(38)*.71)
    component_states[3, 19:] *= 3.
    component_actions = np.asarray([.16, -.125, .08, -.23], dtype=np.float32).astype(np.float64)
    native_data = mj.MjData(env.model)
    native_calls = 0

    def native_map(state, action):
        nonlocal native_calls
        mj.mj_resetData(env.model, native_data)
        native_data.qpos[:], native_data.qvel[:] = state[:19], state[19:]
        native_data.ctrl[0] = 80*float(action)
        mj.mj_forward(env.model, native_data)
        for _ in range(4):
            mj.mj_step(env.model, native_data)
        native_calls += 1
        require(not any(w.number for w in native_data.warning)
                and np.isfinite(np.r_[native_data.qpos, native_data.qvel, native_data.qacc]).all()
                and abs(native_data.time-.02) < 1e-10, "invalid native derivative probe")
        return np.r_[native_data.qpos, native_data.qvel]

    errors, saved = [], {}
    for index, (state, action) in enumerate(zip(component_states, component_actions)):
        native = native_map(state, action)
        mj.mj_resetData(env.model, native_data)
        native_data.qpos[:], native_data.qvel[:] = state[:19], state[19:]
        native_data.ctrl[0] = 80*float(action)
        mj.mj_forward(env.model, native_data)
        continuous_error = float(np.max(np.abs(np.asarray(f(state, action)).ravel()-np.r_[native_data.qvel, native_data.qacc])))
        map_error = float(np.max(np.abs(np.asarray(transition(state, action)).ravel()-native)))
        a, b = (np.asarray(value) for value in first(state, action))
        derivative_errors = []
        for epsilon in inputs["numerical_plan"]["finite_difference_epsilons"]:
            numeric_a = np.empty((38, 38))
            for column in range(38):
                delta = np.zeros(38)
                delta[column] = epsilon
                numeric_a[:, column] = (native_map(state+delta, action)-native_map(state-delta, action))/(2*epsilon)
            numeric_b = (native_map(state, action+epsilon)-native_map(state, action-epsilon))/(2*epsilon)
            derivative_errors.append(dict(epsilon=epsilon,
                A_maximum_absolute_error=float(np.max(np.abs(a-numeric_a))),
                B_maximum_absolute_error=float(np.max(np.abs(b.ravel()-numeric_b)))))
            saved[f"probe_{index}_A_native_{epsilon}"] = numeric_a
            saved[f"probe_{index}_B_native_{epsilon}"] = numeric_b
        limits = inputs["tolerances"]
        require(continuous_error <= limits["continuous"] and map_error <= limits["held_map"]
            and all(item["A_maximum_absolute_error"] <= limits["state_jacobian"]
                    and item["B_maximum_absolute_error"] <= limits["input_jacobian"] for item in derivative_errors),
            "native mechanics/derivative probe failed: " + str(index))
        saved[f"probe_{index}_A_analytic"], saved[f"probe_{index}_B_analytic"] = a, b
        errors.append(dict(component=index, continuous_maximum_absolute_error=continuous_error,
                           held_map_maximum_absolute_error=map_error, native_derivatives=derivative_errors))
    require(all(pins.get(path) == digest for path, digest in runtime_files().items()), "runtime coverage changed")
    require(not output.exists(), "fresh create-only diagnostic output required")
    output.mkdir(parents=True)
    with (output / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, independent_states=np.asarray(ours), upstream_states=np.asarray(theirs),
            actions=actions, noisy_initial=noisy, exact_initial=exact, component_states=component_states,
            component_actions=component_actions, **saved,
            **{"own_"+name: getattr(env.model, name) for name in ATTRIBUTES},
            **{"upstream_"+name: getattr(original, name) for name in ATTRIBUTES})
    after = {path: Path(path).is_file() and sha(path) == digest for path, digest in pins.items()}
    require(all(after.values()), "frozen input bytes changed")
    result = dict(schema="rage_mill.canonical18_plant_parity.v1", passed=True,
        source_commit=PUBLIC, frozen14_source_commit=FROZEN14, frozen14_identical_files=len(comparisons),
        protocol=asdict(p), state_dimension=38, qpos_dimension=19, qvel_dimension=19,
        static_arrays_identical=static, compiled_geometry_identical=geometry, compiled_options_identical=True,
        compiled_uniform_constants=dict(link_length_m=.166666667, link_mass_kg=.0555555556,
            hinge_damping=.000833333333, capsule_radius_m=.025,
            nominal_total_length_m=3., nominal_total_mass_kg=1.,
            serialized_total_length_m=18*.166666667, serialized_total_mass_kg=18*.0555555556),
        upstream_xml_sha256=hashlib.sha256(source_xml.encode()).hexdigest(), independent_xml_sha256=env.xml_sha256,
        noisy_reset_rng_order_byte_exact=True, noise0_exact_hanging_byte_exact=True,
        upstream_method_source_hashes=method_hashes, reset_scope="exact source reset/progress methods; physical bookkeeping and scoring methods retained; reward/observation stubbed",
        diagnostic_seed=DIAGNOSTIC_SEED, action_seed=DIAGNOSTIC_SEED+1,
        transition_count=100, maximum_state_error=0., transition_byte_parity=True,
        scoring_controls=scorer, component_dynamics=errors, native_derivative_held_map_calls=native_calls,
        derivative_action_contract="Derivative of smooth held normalized input before float32 quantizer; base controls are delivered float32. Perturbed inputs remain continuous solely for derivative comparison.",
        arrays=row(output / "arrays.npz"), all_source_pins_unchanged=all(after.values()),
        source_hashes_unchanged=after, runtime=dict(python=sys.version, numpy=np.__version__, mujoco=mj.__version__, casadi=ca.__version__),
        controller_calls=0, controller_qualified=False, development_episodes=0, reserved_episodes=0, claims=0,
        scope="count18 plant/reset/scoring and native component mechanics only; no episode control qualification",
        CPU_seconds=time.process_time()-cpu, wall_seconds=time.perf_counter()-started)
    put(output / "receipt.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--upstream-snapshot", type=Path)
    parser.add_argument("--declaration", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--expected-inputs-sha256")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.prepare != args.run, "exactly one explicit prepare or bounded run required")
    if args.prepare:
        require(args.upstream_snapshot is not None and args.declaration is not None and args.inputs is None,
                "snapshot and prospective declaration preparation required")
        print(json.dumps(prepare(args.upstream_snapshot, args.out, args.declaration)))
    else:
        require(args.inputs is not None and sha(args.inputs) == args.expected_inputs_sha256,
                "frozen inputs SHA256 differs")
        inputs = json.loads(args.inputs.read_text())
        inputs["source_hashes"][str(args.inputs.resolve())] = args.expected_inputs_sha256
        result = verify(inputs, args.out)
        print(json.dumps(dict(passed=result["passed"], receipt=row(args.out / "receipt.json"),
            CPU_seconds=result["CPU_seconds"], wall_seconds=result["wall_seconds"])))


if __name__ == "__main__":
    main()
