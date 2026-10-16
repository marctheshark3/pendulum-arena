"""Record complete native development episodes for a canonical15 candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import casadi as ca
import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize_scalar, least_squares

from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol, hanging_state, record_episode
from rage_mill.canonical_control import TrajectoryPolicy, trajectory_feedback, trajectory_feedback_quad


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=3156000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--park-seconds", type=float, default=0.)
    parser.add_argument("--cart-target", type=float, default=.05)
    parser.add_argument("--tracking-cost", type=float, default=1000.)
    parser.add_argument("--native-bias", action="store_true")
    parser.add_argument("--native-value", action="store_true")
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--optimizer", choices=("bounded", "quadratic"), default="bounded")
    parser.add_argument("--parking-decay", type=float, default=0.)
    parser.add_argument("--balance-after", type=int, default=1500)
    parser.add_argument("--riccati-precision", choices=("double", "quad"), default="double")
    parser.add_argument("--joint-lookahead", type=int, default=1)
    parser.add_argument("--fixed-prefix-ticks", type=int, default=0)
    args = parser.parse_args()
    # This entrypoint cannot draw a reserved seed. Those require a separate
    # source/controller freeze and prospective once-only evaluation.
    if not 3156000 <= args.seed <= args.seed+args.episodes-1 < 3156100:
        raise ValueError("development-only seed range required")
    if not 0 <= args.park_seconds < 30:
        raise ValueError("parking must be inside the thirty-second episode")
    if not 1 <= args.lookahead <= 12:
        raise ValueError("bounded native lookahead must be in 1..12")
    if mujoco.__version__ != "3.3.5":
        raise ValueError("canonical development runtime requires MuJoCo 3.3.5")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    p = CanonicalProtocol()
    with np.load(args.reference, allow_pickle=False) as saved:
        states, actions = saved["states"].copy(), saved["actions"].copy()
    with np.load(args.design, allow_pickle=False) as saved:
        design = {k: saved[k].copy() for k in saved.files}
    first = ca.Function.load(str(args.model/"first.casadi"))
    env = CanonicalEnv(p)
    defects = np.zeros((len(actions), 32))
    if args.native_bias:
        for tick, action in enumerate(actions):
            env.reset(states[tick])
            following, _ = env.step(action)
            defects[tick] = following-states[tick+1]
    feedback = trajectory_feedback_quad if args.riccati_precision == "quad" else trajectory_feedback
    extra = {} if args.riccati_precision == "quad" else {"return_values": True}
    gains, offsets, roots, residuals = feedback(
        states, actions, first, design["terminal_root"], design["Q"],
        control_cost=args.tracking_cost, defects=defects, **extra)
    offsets[:args.fixed_prefix_ticks] = 0.
    initial = hanging_state(p, 0, exact=True)
    a, b = map(np.asarray, first(initial, 0))
    q = np.eye(32)*1e-8
    q[0, 0], q[16, 16] = 10., 5.
    cumulative = np.tril(np.ones((15, 15)))
    q[1:16, 1:16] += 10*cumulative.T@cumulative
    q[17:, 17:] += cumulative.T@cumulative
    decay = np.exp(args.parking_decay*.02)
    park_a, park_b = decay*a, decay*b
    park_p = solve_discrete_are(park_a, park_b, q, np.array([[1000.]]))
    park_gain = ((park_b.T@park_p@park_a)/(1000+(park_b.T@park_p@park_b)[0, 0])).ravel()
    park_ticks = round(args.park_seconds/.02)
    np.savez_compressed(args.out/"controller.npz", states=states, actions=actions, gains=gains,
                        balance_gain=design["gain"], park_gain=park_gain, park_ticks=park_ticks,
                        offsets=offsets, native_defects=defects, roots=roots, residuals=residuals)
    rows = []
    for seed in range(args.seed, args.seed+args.episodes):
        policy = TrajectoryPolicy(states, actions, gains, design["gain"], offsets)
        launch_offset = None
        predictor = CanonicalEnv(p)
        prediction_calls = 0
        stage_root = np.linalg.cholesky(design["Q"]).T

        def controller(state, tick):
            nonlocal launch_offset, prediction_calls
            if tick < park_ticks:
                error = state-initial
                error[0] -= args.cart_target
                return -float(park_gain@error)
            if launch_offset is None:
                launch_offset = state[0]-states[0, 0]
            centered = state.copy()
            centered[0] -= launch_offset
            route_tick = tick-park_ticks
            if route_tick >= args.balance_after:
                return -float(design["gain"]@centered)
            proposed = policy(centered, route_tick)
            if not args.native_value or route_tick < max(180, args.fixed_prefix_ticks):
                return proposed
            action_target = actions[route_tick] if route_tick < len(actions) else 0.

            if args.joint_lookahead > 1 and route_tick >= 390:
                horizon = args.joint_lookahead

                def joint_residual(controls):
                    nonlocal prediction_calls
                    predictor.reset(state)
                    residual_vector = []
                    for ahead, control in enumerate(controls):
                        future_tick = route_tick+ahead
                        following, delivered = predictor.step(control)
                        prediction_calls += 1
                        following[0] -= launch_offset
                        if future_tick < len(actions):
                            target, target_u = states[future_tick+1], actions[future_tick]
                            root, linear = roots[future_tick+1], residuals[future_tick+1]
                        else:
                            target, target_u = np.zeros(32), 0.
                            root, linear = design["terminal_root"], np.zeros(32)
                        residual_vector.append(np.sqrt(args.tracking_cost)*(delivered-target_u))
                        if ahead+1 == horizon:
                            residual_vector.extend(root@(following-target)+linear)
                        else:
                            residual_vector.extend(stage_root@(following-target))
                    return np.array(residual_vector)

                def joint_jacobian(controls):
                    columns = []
                    for coordinate in range(horizon):
                        left, right = controls.copy(), controls.copy()
                        left[coordinate] -= 1e-4
                        right[coordinate] += 1e-4
                        columns.append((joint_residual(right)-joint_residual(left))/2e-4)
                    return np.array(columns).T

                guess = np.array([actions[route_tick+i] if route_tick+i < len(actions) else 0.
                                  for i in range(horizon)])
                guess[0] = np.clip(proposed, -1., 1.)
                solution = least_squares(joint_residual, np.clip(guess, -1., 1.),
                                         jac=joint_jacobian, bounds=(-1., 1.),
                                         max_nfev=20, xtol=1e-10, ftol=1e-8, gtol=1e-6)
                alternatives = (guess, solution.x)
                return float(min(alternatives, key=lambda v: np.linalg.norm(joint_residual(v)))[0])

            def cost(action):
                nonlocal prediction_calls
                # Preserve the actual world-frame cart coordinate in native
                # forecasts. Translation is an error-coordinate operation;
                # changing the forecast plant state can change roundoff in
                # MuJoCo's world-frame COM calculations on this weakly
                # controllable chain.
                predictor.reset(state)
                total, future_action = 0., action
                for ahead in range(args.lookahead):
                    future_tick = route_tick+ahead
                    following, delivered = predictor.step(future_action)
                    following[0] -= launch_offset
                    prediction_calls += 1
                    if future_tick < len(actions):
                        root, linear = roots[future_tick+1], residuals[future_tick+1]
                        next_target, target_action = states[future_tick+1], actions[future_tick]
                    else:
                        root, linear = design["terminal_root"], np.zeros(32)
                        next_target, target_action = np.zeros(32), 0.
                    total += args.tracking_cost*(delivered-target_action)**2
                    if ahead+1 == args.lookahead:
                        error = root@(following-next_target)+linear
                    else:
                        error = stage_root@(following-next_target)
                        future_action = float(np.clip(policy(following, future_tick+1), -1., 1.))
                    total += float(error@error)
                return total

            incumbent = float(np.clip(proposed, -1., 1.))
            # The delivered-command native map is scalar and quantized. Keep
            # the incumbent and the reference command as measured alternatives.
            if args.optimizer == "bounded":
                optimized = minimize_scalar(cost, method="bounded", bounds=(-1., 1.),
                                            options={"xatol": 2e-8, "maxiter": 40})
                optimized_action = float(optimized.x)
            else:
                optimized_action = incumbent
                for epsilon in (.01, .003, .001):
                    center = float(np.clip(optimized_action, -1.+epsilon, 1.-epsilon))
                    left, middle, right = cost(center-epsilon), cost(center), cost(center+epsilon)
                    curvature = (left-2*middle+right)/epsilon**2
                    slope = (right-left)/(2*epsilon)
                    if curvature > 0:
                        trial = float(np.clip(center-slope/curvature, -1., 1.))
                        optimized_action = min((optimized_action, trial), key=cost)
            candidates = (incumbent, optimized_action, float(np.clip(action_target, -1., 1.)))
            return min(candidates, key=cost)

        before = time.monotonic()
        s, u, raw, score = record_episode(controller, hanging_state(p, seed, exact=args.exact), p)
        path = args.out/f"episode-{seed}.npz"
        np.savez_compressed(path, states=s, actions=u, raw_actions=raw)
        row = dict(seed=seed, exact=args.exact, **score, wall_seconds=time.monotonic()-before,
                   recording=path.name, recording_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   native_prediction_calls=prediction_calls)
        rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    receipt = dict(schema="rage_mill.canonical15_development.v1", stage="development",
                   controller_qualified=False, reserved_episodes=0, cases=rows,
                   controller_sha256=hashlib.sha256((args.out/"controller.npz").read_bytes()).hexdigest(),
                   reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                   model_receipt_sha256=hashlib.sha256((args.model/"receipt.json").read_bytes()).hexdigest(),
                   wall_seconds=time.monotonic()-started, park_seconds=args.park_seconds,
                   tracking_cost=args.tracking_cost, native_bias=args.native_bias,
                   maximum_native_defect=float(np.max(np.abs(defects))),
                   maximum_affine_offset=float(np.max(np.abs(offsets))),
                   fixed_prefix_ticks=args.fixed_prefix_ticks, cart_target=args.cart_target,
                   native_value=args.native_value, lookahead=args.lookahead,
                   optimizer=args.optimizer, parking_decay=args.parking_decay,
                   balance_after=args.balance_after, riccati_precision=args.riccati_precision,
                   joint_lookahead=args.joint_lookahead)
    (args.out/"receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
