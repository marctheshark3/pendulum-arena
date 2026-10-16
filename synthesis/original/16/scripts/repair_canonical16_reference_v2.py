"""Bounded native-cadence sixteen-link reference repair, before control trials."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import casadi as ca
import numpy as np

from rage_mill.canonical16_trajectory import route_initializer


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--warm", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ticks", type=int, default=700)
    parser.add_argument("--cpu-seconds", type=float, default=180.)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--route-ticks", type=int, default=450)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model_receipt = json.loads((args.model/"receipt.json").read_text())
    assert model_receipt["passed"] and model_receipt["n_links"] == 16 and model_receipt["runtime"]["mujoco"] == "3.3.5"
    transition = ca.Function.load(str(args.model/"transition.casadi"))
    first = ca.Function.load(str(args.model/"first.casadi"))
    count, dimension = args.ticks, 34
    assert 550 <= count <= 1000
    with np.load(args.warm, allow_pickle=False) as saved:
        original_s, original_u = saved["states"].copy(), saved["actions"].copy()
    warm_s, warm_u = route_initializer(original_s, original_u, ticks=count, route_ticks=args.route_ticks)
    initial = warm_s[0]
    assert initial[1] == np.pi and np.count_nonzero(initial) == 1
    states = ca.MX.sym("states", dimension, count+1)
    actions = ca.MX.sym("actions", 1, count)
    z = ca.vertcat(ca.vec(states), ca.vec(actions))
    gaps = states[:, 1:]-transition.map(count)(states[:, :-1], actions)
    cumulative = ca.DM(np.tril(np.ones((16, 16))))
    hold_angles = cumulative@states[1:17, count-250:]
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
    lower_g = np.r_[np.zeros(equalities), np.full(16*251, -.12)]
    upper_g = np.r_[np.zeros(equalities), np.full(16*251, .12)]
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
    np.savez_compressed(args.out/"reference.npz", states=result_s, actions=result_u)
    stats = solver.stats()
    receipt = dict(schema="rage_mill.canonical16_reference_repair.v2", reference_only=True, n_links=16, route_ticks=args.route_ticks, development_episodes=0,
                   controller_qualified=False, reserved_episodes=0, ticks=count,
                   planner_map_feasible=feasible, maximum_dynamics_defect=defect,
                   initial_error=initial_error, terminal_error=terminal_error,
                   maximum_hold_angle=hold_max, planning_bounds_ok=bounds_ok,
                   status=stats["return_status"], iterations=stats["iter_count"],
                   model_receipt_sha256=hashlib.sha256((args.model/"receipt.json").read_bytes()).hexdigest(),
                   warm_sha256=hashlib.sha256(args.warm.read_bytes()).hexdigest(),
                   wall_seconds=time.monotonic()-started, build_seconds=problem_built)
    (args.out/"receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
