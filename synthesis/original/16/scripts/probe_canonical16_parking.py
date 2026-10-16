"""Measure hanging launch settling, independently of swing-up qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import scipy
from scipy.linalg import solve_discrete_are

from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol, hanging_state


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() or mujoco.__version__ != '3.3.5':
        raise ValueError('fresh output and native MuJoCo3.3.5 required')
    args.out.mkdir(parents=True)
    wall, cpu = time.monotonic(), time.process_time()
    root = Path(__file__).resolve().parents[1]
    sources = {str(path): sha(path) for path in (
        Path(__file__), root/'src/rage_mill/canonical_benchmark.py', root/'configs/canonical16.json')}
    protocol = CanonicalProtocol(n_links=16)
    env = CanonicalEnv(protocol)
    initial = hanging_state(protocol, 3167200, exact=True)
    dimension = len(initial)

    def following(state, action):
        env.reset(state)
        return env.step(action)[0]

    # Native held-input derivative at the hanging equilibrium. This is an
    # offline derivative component; each subsequent trajectory resets once.
    epsilon = 1e-5
    a = np.column_stack([(following(initial+epsilon*np.eye(dimension)[j], 0.)
                          -following(initial-epsilon*np.eye(dimension)[j], 0.))/(2*epsilon)
                         for j in range(dimension)])
    delivered_epsilon = float(np.float32(epsilon))
    b = ((following(initial, delivered_epsilon)-following(initial, -delivered_epsilon))
         /(2*delivered_epsilon)).reshape(-1, 1)
    q = np.eye(dimension)*1e-8
    q[0, 0], q[17, 17] = 10., 5.
    p = solve_discrete_are(a, b, q, np.array([[1000.]]))
    gain = ((b.T@p@a)/(1000.+(b.T@p@b)[0, 0])).ravel()
    np.savez_compressed(args.out/'design.npz', A=a, B=b, Q=q, P=p, gain=gain)
    target = initial.copy(); target[0] = -.05
    rows = []
    for seed in range(3167200, 3167210):
        state = env.reset(hanging_state(protocol, seed))
        states, actions, raw, times = [state], [], [], [0.]
        metrics = []
        invalid = None
        for tick in range(1000):
            command = -float(gain@(state-target))
            try:
                state, delivered = env.step(command)
            except ValueError as error:
                invalid = str(error)
                break
            states.append(state); actions.append(delivered); raw.append(command); times.append(float(env.data.time))
            if tick+1 in (400, 800, 1000):
                error = state-target
                metrics.append(dict(ticks=tick+1, seconds=(tick+1)*.02,
                    error_norm=float(np.linalg.norm(error)), cart_error=float(error[0]),
                    maximum_absolute_angle_error=float(np.max(np.abs(np.cumsum(error[1:17])))),
                    maximum_absolute_rate=float(np.max(np.abs(np.cumsum(state[18:]))))))
            if abs(state[0]) > 3.:
                invalid = 'rail escape'
                break
        path = args.out/f'parking-{seed}.npz'
        np.savez_compressed(path, states=np.asarray(states), actions=np.asarray(actions, dtype=np.float32),
                            raw_actions=np.asarray(raw), times=np.asarray(times))
        row = dict(seed=seed, ticks=len(actions), valid=invalid is None, invalid_reason=invalid,
            initial_error_norm=float(np.linalg.norm(states[0]-target)), measurements=metrics,
            maximum_cart_m=float(np.max(np.abs(np.asarray(states)[:, 0]))),
            recording=dict(path=str(path), sha256=sha(path)))
        rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    unchanged = {path: sha(path) == value for path, value in sources.items()}
    result = dict(schema='rage_mill.canonical16_hanging_settling_component.v1',
        purpose='Measure launch deviations after8/16/20s, not hanging swing-up or qualification',
        method='native centered held-input hanging linearization; cart Q10/rate5, other diagonal1e-8, R1000, target-.05m',
        cases=rows, source_hashes=sources, source_hashes_unchanged=unchanged,
        runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, mujoco=mujoco.__version__),
        gain_norm=float(np.linalg.norm(gain)), design=dict(path=str(args.out/'design.npz'), sha256=sha(args.out/'design.npz')),
        CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall,
        full_swing_up_episodes=0, development_gate_episodes=0, reserved_episodes=0, controller_qualified=False)
    with (args.out/'receipt.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    if not all(unchanged.values()):
        raise ValueError('source changed during component')
    print(json.dumps(dict(receipt=str(args.out/'receipt.json'), cases=len(rows),
        all_valid=all(row['valid'] for row in rows)), allow_nan=False))


if __name__ == '__main__':
    main()
