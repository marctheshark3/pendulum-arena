"""Bounded native-cadence fifteen-link reference repair, before control trials."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import casadi as ca
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--warm", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ticks", type=int, default=650)
    parser.add_argument("--cpu-seconds", type=float, default=180.)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--fixed-prefix", type=Path)
    parser.add_argument("--prefix-ticks", type=int, default=0)
    parser.add_argument("--feasibility-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model_receipt = json.loads((args.model/"receipt.json").read_text())
    assert model_receipt["passed"] and model_receipt["runtime"]["mujoco"] == "3.3.5"
    transition = ca.Function.load(str(args.model/"transition.casadi"))
    first = ca.Function.load(str(args.model/"first.casadi"))
    count, dimension = args.ticks, 32
    assert 550 <= count <= 1000
    with np.load(args.warm, allow_pickle=False) as saved:
        original_s, original_u = saved["states"].copy(), saved["actions"].copy()
    # Reflect the symmetric published development route into the canonical
    # +pi launch branch. This is an offline initializer, never a state reset.
    mirror = -1. if original_s[0, 1] < 0 else 1.
    warm_s = np.zeros((count+1, dimension))
    warm_u = np.zeros(count)
    prefix = min(400, len(original_u)) if len(original_u) < count else count
    warm_s[:prefix+1] = mirror*original_s[:prefix+1]
    warm_u[:prefix] = mirror*original_u[:prefix]
    for tick in range(prefix+1, min(prefix+101, count+1)):
        warm_s[tick] = warm_s[prefix] * (1-(tick-prefix)/100)
    fixed_s = fixed_u = None
    if args.fixed_prefix:
        assert 1 <= args.prefix_ticks < count-50
        with np.load(args.fixed_prefix, allow_pickle=False) as fixed:
            fixed_s = fixed["states"][:args.prefix_ticks+1].copy()
            fixed_u = fixed["actions"][:args.prefix_ticks].copy()
        warm_s[:args.prefix_ticks+1] = fixed_s
        warm_u[:args.prefix_ticks] = fixed_u
    initial = np.zeros(32)
    initial[1] = np.pi
    assert np.max(np.abs(warm_s[0]-initial)) <= 1e-8
    warm_s[0] = initial
    full_count = count
    if fixed_s is not None:
        # Optimize only the live suffix. Keeping fixed physical prefix nodes
        # as optimization variables leaves redundant constant dynamics rows
        # and asks the solver to erase representational native/map roundoff.
        count -= args.prefix_ticks
        warm_s = warm_s[args.prefix_ticks:].copy()
        warm_u = warm_u[args.prefix_ticks:].copy()
        initial = warm_s[0].copy()
    states = ca.MX.sym("states", dimension, count+1)
    actions = ca.MX.sym("actions", 1, count)
    z = ca.vertcat(ca.vec(states), ca.vec(actions))
    gaps = states[:, 1:]-transition.map(count)(states[:, :-1], actions)
    cumulative = ca.DM(np.tril(np.ones((15, 15))))
    hold_start = max(0, count-250)
    hold_width = count+1-hold_start
    hold_angles = cumulative@states[1:16, hold_start:]
    g = ca.vertcat(ca.vec(gaps), states[:, 0]-initial, states[:, -1], ca.vec(hold_angles))
    a, b = first.map(count)(states[:, :-1], actions)
    # Supply the verified implicit derivatives directly. Automatic AD through
    # the complete RK4 graph would recompile the expensive full derivative.
    block_a = ca.diagcat(*[a[:, tick*dimension:(tick+1)*dimension] for tick in range(count)])
    block_b = ca.diagcat(*[b[:, tick] for tick in range(count)])
    state_part = (ca.horzcat(-block_a, ca.MX(ca.Sparsity(dimension*count, dimension)))
                  + ca.horzcat(ca.MX(ca.Sparsity(dimension*count, dimension)), ca.DM.eye(dimension*count)))
    defect_jac = ca.horzcat(state_part, -block_b)
    boundary = ca.vertcat(states[:, 0]-initial, states[:, -1], ca.vec(hold_angles))
    full_jac = ca.vertcat(defect_jac, ca.jacobian(boundary, z))
    parameter = ca.MX.sym("p", 0)
    jac_g = ca.Function("canonical_nlp_jac_g", [z, parameter], [g, full_jac])
    objective = ca.MX(0.)
    if not args.feasibility_only:
        objective = 1e-4*ca.sumsqr(actions-ca.DM(warm_u.reshape(1, -1)))
        objective += 1e-7*ca.sumsqr(states-ca.DM(warm_s.T))
    options = {"jac_g": jac_g, "ipopt.hessian_approximation": "limited-memory",
               "ipopt.max_iter": args.iterations, "ipopt.max_cpu_time": args.cpu_seconds,
               "ipopt.tol": 1e-10, "ipopt.constr_viol_tol": 1e-11,
               "ipopt.acceptable_iter": 0, "ipopt.bound_relax_factor": 0.,
               "ipopt.nlp_scaling_method": "none", "ipopt.mu_strategy": "adaptive",
               "ipopt.linear_solver": "mumps", "ipopt.print_level": 5, "print_time": False}
    solver = ca.nlpsol("canonical_reference", "ipopt", {"x": z, "f": objective, "g": g}, options)
    lower_s = np.full((count+1, dimension), -80.)
    upper_s = -lower_s
    lower_s[:, 0], upper_s[:, 0] = -2.8, 2.8
    lower = np.r_[lower_s.ravel(), np.full(count, -.95)]
    upper = np.r_[upper_s.ravel(), np.full(count, .95)]
    equalities = dimension*count+2*dimension
    lower_g = np.r_[np.zeros(equalities), np.full(15*hold_width, -.12)]
    upper_g = np.r_[np.zeros(equalities), np.full(15*hold_width, .12)]
    vector = np.r_[warm_s.ravel(), warm_u]
    problem_built = time.monotonic()-started
    print(json.dumps(dict(event="start_repair", ticks=count, build_seconds=problem_built,
                         cpu_limit=args.cpu_seconds, iterations=args.iterations)), flush=True)
    solved = solver(x0=vector, lbx=lower, ubx=upper, lbg=lower_g, ubg=upper_g)
    vector = np.asarray(solved["x"]).ravel()
    result_s = vector[:dimension*(count+1)].reshape(count+1, dimension)
    result_u = vector[dimension*(count+1):]
    gfun = ca.Function("canonical_review", [z], [g])
    residual = np.asarray(gfun(vector)).ravel()
    defect = float(np.max(np.abs(residual[:dimension*count])))
    initial_error = float(np.max(np.abs(result_s[0]-initial)))
    terminal_error = float(np.max(np.abs(result_s[-1])))
    hold_max = float(np.max(np.abs(residual[equalities:])))
    bounds_ok = bool(np.all(vector >= lower-1e-12) and np.all(vector <= upper+1e-12))
    feasible = bool(defect <= 1e-11 and initial_error <= 1e-12 and terminal_error <= 1e-12
                    and hold_max < .15 and bounds_ok)
    prefix_defect = 0.
    if fixed_s is not None:
        predicted = np.asarray(transition.map(args.prefix_ticks)(fixed_s[:-1].T, fixed_u.reshape(1, -1))).T
        prefix_defect = float(np.max(np.abs(predicted-fixed_s[1:])))
        feasible = bool(feasible and prefix_defect <= 1e-11)
        result_s = np.vstack([fixed_s[:-1], result_s])
        result_u = np.r_[fixed_u, result_u]
        hold_max = float(np.max(np.abs(np.cumsum(result_s[-251:, 1:16], axis=1))))
        feasible = bool(feasible and hold_max < .15)
    np.savez_compressed(args.out/"reference.npz", states=result_s, actions=result_u)
    stats = solver.stats()
    receipt = dict(schema="rage_mill.canonical_reference_repair.v1", reference_only=True,
                   controller_qualified=False, reserved_episodes=0, ticks=full_count,
                   planner_map_feasible=feasible, maximum_dynamics_defect=max(defect, prefix_defect),
                   optimized_suffix_defect=defect, fixed_native_prefix_map_defect=prefix_defect,
                   initial_error=initial_error, terminal_error=terminal_error,
                   maximum_hold_angle=hold_max, planning_bounds_ok=bounds_ok,
                   status=stats["return_status"], iterations=stats["iter_count"],
                   model_receipt_sha256=hashlib.sha256((args.model/"receipt.json").read_bytes()).hexdigest(),
                   warm_sha256=hashlib.sha256(args.warm.read_bytes()).hexdigest(),
                   fixed_prefix_sha256=(hashlib.sha256(args.fixed_prefix.read_bytes()).hexdigest() if args.fixed_prefix else None),
                   fixed_prefix_ticks=args.prefix_ticks, feasibility_only=args.feasibility_only,
                   wall_seconds=time.monotonic()-started, build_seconds=problem_built)
    (args.out/"receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
