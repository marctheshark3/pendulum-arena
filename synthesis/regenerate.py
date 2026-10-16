"""Public synthesis adapters. Historical source and inputs live in original/ and data/."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def verify():
    manifest = json.loads((ROOT / 'original-manifest.json').read_text())
    for row in manifest['files']:
        path = ROOT / row['path']
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError(f'exported input/source mismatch: {row["path"]}')
    export_path = ROOT/'export-manifest.json'
    if export_path.exists():
        export = json.loads(export_path.read_text())
        for row in export['files']:
            path = ROOT/row['path']
            if path.is_symlink() or not path.is_file() or digest(path) != row['sha256']:
                raise ValueError(f'public adapter/documentation mismatch: {row["path"]}')
    return dict(verified_files=len(manifest['files']), reserved_episodes=0)


def load_count(n):
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
        if os.environ.get(variable) != '1':
            raise ValueError(f'set {variable}=1 before starting Python')
    if any(name == 'rage_mill' or name.startswith('rage_mill.') for name in sys.modules):
        raise ValueError('run each command in a fresh interpreter')
    sys.path.insert(0, str(ROOT / f'original/{n}/src'))
    import mujoco
    if mujoco.__version__ != '3.3.5':
        raise ValueError('MuJoCo 3.3.5 is required')


def fresh(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def arrays(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def comparisons(actual, expected):
    import numpy as np
    rows = {}
    for name, reference in expected.items():
        value = actual.get(name)
        same = value is not None and value.shape == reference.shape and value.dtype == reference.dtype
        rows[name] = dict(shape=list(reference.shape), dtype=str(reference.dtype),
            array_bytes_equal=bool(same and value.tobytes() == reference.tobytes()),
            max_absolute_difference=(float(np.max(np.abs(value-reference))) if same and value.size else None))
    return dict(all_array_bytes_equal=all(v['array_bytes_equal'] for v in rows.values()), fields=rows)


def runtime():
    import importlib.metadata
    return dict(python=sys.version, machine=platform.machine(),
        packages={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'mujoco', 'casadi', 'mpmath')},
        runtime_binary_hashes_verified=False)


def fetch(args):
    out = fresh(args.out)
    manifest = json.loads((ROOT/'upstream-inputs.json').read_text())
    for row in manifest['files']:
        target = out / Path(row['path']).name
        with urllib.request.urlopen(row['url'], timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'upstream input hash mismatch: {row["path"]}')
        target.write_bytes(data)
    write_json(out/'receipt.json', manifest)
    return dict(downloads=len(manifest['files']), out=str(out))


def model(args):
    import casadi as ca
    import numpy as np
    from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol
    from rage_mill.canonical_dynamics import planning_functions
    out = fresh(args.out)
    env = CanonicalEnv(CanonicalProtocol(n_links=args.count))
    previous = Path.cwd()
    os.chdir(out)
    try:
        continuous, transition, first = planning_functions(env.model, jit=True)
    finally:
        os.chdir(previous)
    checkpoint = arrays(ROOT/f'data/{args.count}'/('physical-reference.npz' if args.count == 15 else 'early-reference.npz'))
    s, u = checkpoint['states'], checkpoint['actions']
    checks = []
    for tick in (0, 10, 100, 200, 300, 400, len(u)-1):
        env.reset(s[tick]); following, delivered = env.step(u[tick])
        predicted = np.asarray(transition(s[tick], delivered)).ravel()
        error = float(np.max(np.abs(following-predicted)))
        if not np.isfinite(error) or error > 1e-8:
            raise ValueError('rebuilt planning map fails native comparison')
        checks.append(dict(tick=tick, native_transition_max_error=error))
    for name, function in (('continuous', continuous), ('transition', transition), ('first', first)):
        function.save(str(out/f'{name}.casadi'))
    aa, bb = map(np.asarray, first.map(len(u))(s[:-1].T, u.reshape(1, -1)))
    original = arrays(ROOT/f'data/{args.count}/derivatives.npz')
    comparison = comparisons({'A': aa, 'B': bb}, {k: original[k] for k in ('A', 'B')})
    receipt = dict(passed=True, n_links=args.count, xml_sha256=env.xml_sha256,
        runtime=runtime()['packages'], checks=checks, derivative_checkpoint_comparison=comparison,
        scope='New planning component; sampled native checks, not controller qualification', reserved_episodes=0)
    write_json(out/'receipt.json', receipt)
    return receipt


def design(args):
    import numpy as np
    from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol
    out = fresh(args.out)
    env = CanonicalEnv(CanonicalProtocol(n_links=args.count))
    if args.count == 15:
        from rage_mill.canonical_control import balance_design
        result = balance_design(env.model)
        diagnostics = result.pop('diagnostics')
    else:
        from rage_mill.canonical_capture_v2 import upright_design
        result, diagnostics = upright_design(env)
        result['gain'] = result['K'].copy()
    np.savez_compressed(out/'design.npz', **result)
    expected = arrays(ROOT/f'data/{args.count}/design.npz')
    comparison = comparisons(result, {k: v for k, v in expected.items() if k in result})
    receipt = dict(n_links=args.count, comparison=comparison, diagnostics=diagnostics,
                   original_design_sha256=digest(ROOT/f'data/{args.count}/design.npz'), reserved_episodes=0)
    write_json(out/'receipt.json', receipt)
    return receipt


def parking_gain(n, derivative=None):
    import numpy as np
    from scipy.linalg import solve_discrete_are
    d = n+1
    if n == 15:
        a, b = derivative['hanging_A'], derivative['hanging_B']
    else:
        original = arrays(ROOT/'data/16/parking-design.npz')
        a, b = original['A'], original['B']
    total = np.tril(np.ones((n, n)))
    q = np.eye(2*d)*1e-8
    q[0, 0], q[d, d] = 10., 5.
    q[1:d, 1:d] += 10*(total.T@total)
    q[d+1:, d+1:] += total.T@total
    decay = np.exp(1.2*.02) if n == 15 else 1.
    aa, bb = decay*a, decay*b
    p = solve_discrete_are(aa, bb, q, np.array([[1000.]]))
    gain = ((bb.T@p@aa)/(1000+(bb.T@p@bb)[0, 0])).ravel()
    return gain, dict(A=a, B=b, Q=q, P=p, gain=gain)


def feedback(args):
    import numpy as np
    from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol
    n = args.count
    out = fresh(args.out)
    data = ROOT/f'data/{n}'
    ref_path = Path(args.reference).resolve() if args.reference else data/('physical-reference.npz' if n == 15 else 'early-reference.npz')
    ref, des = arrays(ref_path), arrays(data/'design.npz')
    states, actions = ref['states'], ref['actions']
    derivative = arrays(data/'derivatives.npz')
    if args.model:
        import casadi as ca
        first = ca.Function.load(str(Path(args.model).resolve()/'first.casadi'))
        derivative_mode = 'locally rebuilt original symbolic derivative provider'
    else:
        meta = json.loads((data/'derivatives.json').read_text())
        if digest(ref_path) != meta['reference_sha256']:
            raise ValueError('derivative checkpoint belongs to a different reference; supply --model for a new reference')
        class First:
            def map(self, count):
                if count != len(actions):
                    raise ValueError('derivative length mismatch')
                return lambda x, u: (derivative['A'], derivative['B'])
        first = First()
        derivative_mode = 'public numeric checkpoint evaluated from the original derivative provider'
    env = CanonicalEnv(CanonicalProtocol(n_links=n))
    defects = np.empty_like(states[:-1])
    for tick, action in enumerate(actions):
        env.reset(states[tick]); following, _ = env.step(action)
        defects[tick] = following-states[tick+1]
    terminal = des['terminal_root'] if n == 15 else np.zeros((34, 34))
    if n == 15:
        from rage_mill.canonical_control import trajectory_feedback as feedback15
        gains, offsets, roots, residuals = feedback15(states, actions, first, terminal, des['Q'],
            control_cost=1000., defects=defects, return_values=True)
    else:
        from rage_mill.canonical16_trajectory import trajectory_feedback
        gains, offsets = trajectory_feedback(states, actions, first, terminal, des['Q'],
                                             control_cost=1000., defects=defects)
    park, park_arrays = parking_gain(n, derivative)
    np.savez_compressed(out/'parking-design.npz', **park_arrays)
    if n == 15:
        payload = dict(states=states, actions=actions, gains=gains, offsets=offsets,
            roots=roots, residuals=residuals, balance_gain=des['gain'], park_gain=park,
            terminal_root=terminal, Q=des['Q'])
        parameters = json.loads((data/'policy.json').read_text())
    else:
        target = np.zeros(34); target[0], target[1] = -.05, np.pi
        full = dict(states=states, actions=actions, gains=gains, offsets=offsets,
            balance_gain=des['gain'], park_gain=park, parking_target=target,
            park_ticks=np.asarray(800, dtype=np.int64), native_defects=defects, terminal_root=terminal)
        np.savez_compressed(out/'probe07-controller.npz', **full)
        full_comparison = comparisons(full, arrays(data/'probe07-controller.npz'))
        payload = {k: v.copy() for k, v in full.items()}
        payload['states'] = states[:651].copy()
        for name in ('actions', 'gains', 'offsets', 'native_defects'):
            payload[name] = payload[name][:650].copy()
        payload['park_ticks'] = np.asarray(850, dtype=np.int64)
        parameters = dict(cart_error_offsets_m=[0., 1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2])
    np.savez_compressed(out/'controller.npz', **payload)
    write_json(out/'policy.json', parameters)
    comparison = comparisons(payload, arrays(data/'released-controller.npz'))
    receipt = dict(n_links=n, reference_sha256=digest(ref_path), derivative_mode=derivative_mode,
        released_payload_comparison=comparison, maximum_native_reference_defect=float(np.max(np.abs(defects))),
        runtime=runtime(), reserved_episodes=0,
        scope='New reconstruction of selected feedback arrays; no historical freeze is changed')
    if n == 16:
        receipt['original_probe07_comparison'] = full_comparison
    write_json(out/'receipt.json', receipt)
    return receipt


def diagnostic(args):
    import numpy as np
    from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol, hanging_state, record_episode
    n = args.count
    out = fresh(args.out)
    candidate = Path(args.controller).resolve()
    payload = arrays(candidate/'controller.npz')
    parameters = json.loads((candidate/'policy.json').read_text())
    if n == 15:
        from rage_mill.canonical_control import CanonicalPlanningPolicy
        policy = CanonicalPlanningPolicy(payload, **parameters)
    else:
        from rage_mill.canonical16_finite_controller import Canonical16FiniteTrajectoryPolicy
        from rage_mill.canonical_profile_policy import NativeCartOffsetSelector
        base = Canonical16FiniteTrajectoryPolicy(**{name: payload[name] for name in
            ('states', 'actions', 'gains', 'offsets', 'balance_gain', 'park_gain', 'parking_target')},
            park_ticks=int(payload['park_ticks']), align_launch_branches=False, dot_scope='float64')
        policy = NativeCartOffsetSelector(base, CanonicalProtocol(n_links=16), parameters['cart_error_offsets_m'])
    protocol = CanonicalProtocol(n_links=n)
    initial = hanging_state(protocol, 0, exact=True)
    states, actions, raw, score = record_episode(policy, initial, protocol)
    np.savez_compressed(out/'episode.npz', states=states, actions=actions, raw_actions=raw)
    env = CanonicalEnv(protocol)
    maximum_defect = 0.
    for tick, action in enumerate(actions):
        env.reset(states[tick]); following, _ = env.step(action)
        maximum_defect = max(maximum_defect, float(np.max(np.abs(following-states[tick+1]))))
    receipt = dict(n_links=n, exact_hanging=True, score=score,
        native_cold_transition_maximum_defect=maximum_defect,
        controller_sha256=digest(candidate/'controller.npz'), recording_sha256=digest(out/'episode.npz'),
        runtime=runtime(), reserved_episodes=0,
        scope='New public diagnostic. No seed/cohort claim, no reserved runner, no new qualification.')
    write_json(out/'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify')
    p = sub.add_parser('fetch-inputs'); p.add_argument('--out', type=Path, required=True)
    for command in ('model', 'design', 'feedback', 'diagnostic'):
        p = sub.add_parser(command)
        p.add_argument('--count', type=int, choices=(15, 16), required=True)
        p.add_argument('--out', type=Path, required=True)
        if command == 'feedback':
            p.add_argument('--model', type=Path)
            p.add_argument('--reference', type=Path)
        if command == 'diagnostic':
            p.add_argument('--controller', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    checked = verify()
    if args.command == 'verify':
        result = checked
    elif args.command == 'fetch-inputs':
        result = fetch(args)
    else:
        load_count(args.count)
        result = globals()[args.command](args)
    print(json.dumps(dict(command=args.command, result=result, adapter_wall_seconds=time.monotonic()-started), allow_nan=False))


if __name__ == '__main__':
    main()
