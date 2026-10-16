"""Build and check planning derivatives without consuming a controller trial."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import casadi as ca
import mujoco
import numpy as np

from rage_mill.canonical_benchmark import CanonicalEnv
from rage_mill.canonical_dynamics import planning_functions


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--tape", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    env = CanonicalEnv()
    continuous, transition, first = planning_functions(env.model, jit=True)
    build_seconds = time.monotonic()-started
    print(json.dumps(dict(build_seconds=build_seconds)), flush=True)
    with np.load(args.tape, allow_pickle=False) as saved:
        states, actions = saved["states"].copy(), saved["actions"].copy()
    rows = []
    for tick in (0, 10, 100, 200, 250, 300, 350, 400, 440):
        state, action = states[tick], actions[tick]
        env.reset(state)
        env.data.ctrl[0] = 80*action
        mujoco.mj_forward(env.model, env.data)
        expected = np.r_[env.data.qvel, env.data.qacc]
        predicted = np.asarray(continuous(state, action)).ravel()
        actual, _ = env.step(action)
        discrete = np.asarray(transition(state, action)).ravel()
        continuous_error = float(np.max(np.abs(expected-predicted)))
        transition_error = float(np.max(np.abs(actual-discrete)))
        assert continuous_error < 1e-9 and transition_error < 1e-11
        a, b = (np.asarray(v) for v in first(state, action))
        rng = np.random.default_rng(42+tick)
        direction = rng.normal(size=32)
        direction /= np.linalg.norm(direction)
        epsilon = 1e-5
        finite = (np.asarray(transition(state+epsilon*direction, action)).ravel()
                  - np.asarray(transition(state-epsilon*direction, action)).ravel())/(2*epsilon)
        directional_error = float(np.max(np.abs(a@direction-finite)))
        assert directional_error < 1e-5
        rows.append(dict(tick=tick, continuous_error=continuous_error,
                         native_transition_error=transition_error, directional_error=directional_error))
    before = time.monotonic()
    for _ in range(100):
        first(states[300], actions[300])
    first_seconds = (time.monotonic()-before)/100
    for name, function in (("continuous", continuous), ("transition", transition), ("first", first)):
        function.save(str(args.out/f"{name}.casadi"))
    sources = {}
    for name in ("canonical_benchmark.py", "canonical_dynamics.py"):
        path = Path(__file__).resolve().parents[1]/"src/rage_mill"/name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = dict(schema="rage_mill.canonical_planning_dynamics.v1", passed=True,
                   checks=rows, build_seconds=build_seconds, first_seconds=first_seconds,
                   wall_seconds=time.monotonic()-started, source_sha256=sources,
                   runtime=dict(mujoco=mujoco.__version__, numpy=np.__version__, casadi=ca.__version__),
                   native_commands=len(rows), controller_qualified=False, reserved_episodes=0)
    (args.out/"receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
