"""Bounded count-seventeen design components; never qualification admission.

Every subcommand writes fresh artifacts. A lifted sixteen route is only an
offline initializer. Native cold-map feasibility, local physical capture and
complete hanging-start episodes are reported separately.
"""
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

from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol, absolute_angles, hanging_state, record_episode
from rage_mill.canonical_dynamics import planning_functions
from rage_mill.canonical16_trajectory import lift_material, trajectory_feedback
from rage_mill.canonical16_dynamic_orbit import discrete_harmonic_initializer
from rage_mill.canonical16_orbit import refine_cyclic
from rage_mill.canonical16_orbit_bridge import delivered_actions, delivered_callback


N, D, M = 17, 18, 36


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def env17():
    if mujoco.__version__ != '3.3.5':
        raise ValueError('actual MuJoCo3.3.5 required')
    env = CanonicalEnv(CanonicalProtocol(n_links=N))
    if env.model.nq != D or env.model.nv != D:
        raise ValueError('actual seventeen-link q18/v18 plant required')
    return env


def weights(angle=100., rate=1., cart=.1, cart_rate=.1):
    total = np.tril(np.ones((N, N)))
    gram = total.T @ total
    q = np.eye(M)
    q[0, 0], q[D, D] = cart, cart_rate
    q[1:D, 1:D] += angle * gram
    q[D+1:, D+1:] = rate * gram + .01 * np.eye(N)
    return q


def model(args):
    """Construct own-count model and check every test on native F32 delivery."""
    env = env17()
    continuous, transition, first = planning_functions(env.model, jit=True)
    with np.load(args.donor, allow_pickle=False) as tape:
        warm = lift_material(tape['states'][:451], N)
        warm_u = delivered_actions(tape['actions'][:450])
    checks = []
    for tick in (0, 25, 100, 150, 200, 250, 300, 350, 400, 449):
        state, action = warm[tick], float(warm_u[tick])
        env.reset(state)
        env.data.ctrl[0] = 80. * action
        mujoco.mj_forward(env.model, env.data)
        actual_derivative = np.r_[env.data.qvel, env.data.qacc]
        continuous_gap = float(np.max(np.abs(actual_derivative - np.asarray(continuous(state, action)).ravel())))
        following, _ = env.step(action)
        transition_gap = float(np.max(np.abs(following - np.asarray(transition(state, action)).ravel())))
        a, b = map(np.asarray, first(state, action))
        rng = np.random.default_rng(3171000 + tick)
        direction = rng.normal(size=M)
        direction /= np.linalg.norm(direction)
        epsilon = 2.**-19
        actual = []
        for sign in (1., -1.):
            env.reset(state + sign * epsilon * direction)
            actual.append(env.step(action)[0])
        state_derivative_gap = float(np.max(np.abs((actual[0]-actual[1])/(2*epsilon)-a@direction)))
        # Native input differences use the actual unequal float32 endpoints.
        lo, hi = float(np.float32(action-1e-4)), float(np.float32(action+1e-4))
        env.reset(state); lower = env.step(lo)[0]
        env.reset(state); upper = env.step(hi)[0]
        input_derivative_gap = float(np.max(np.abs((upper-lower)/(hi-lo)-b[:, 0])))
        checks.append(dict(tick=tick, continuous_gap=continuous_gap, native_transition_gap=transition_gap,
                           native_state_derivative_gap=state_derivative_gap, native_input_derivative_gap=input_derivative_gap))
    a, b = map(np.asarray, first(np.zeros(M), 0.))
    for name, function in (('continuous', continuous), ('transition', transition), ('first', first)):
        function.save(str(args.out / (name+'.casadi')))
    np.savez_compressed(args.out/'design.npz', A=a, B=b, Q=weights(), gain=np.zeros(M), terminal_root=np.zeros((M, M)))
    np.savez_compressed(args.out/'lifted-initializer.npz', states=warm, actions=warm_u)
    passed = all(r['continuous_gap'] <= 1e-8 and r['native_transition_gap'] <= 1e-10
                 and r['native_state_derivative_gap'] <= 1e-4 and r['native_input_derivative_gap'] <= 1e-4 for r in checks)
    return dict(schema='rage_mill.canonical17_planning_model.v1', passed=passed, n_links=N,
                xml_sha256=env.xml_sha256, checks=checks, donor_only_offline_initializer=True,
                donor_sha256=sha(args.donor), native_cold_map_commands=50,
                maximum_native_transition_gap=max(r['native_transition_gap'] for r in checks))


def functions(directory):
    metadata = json.loads((directory/'receipt.json').read_text())
    if not metadata['passed'] or metadata['n_links'] != N:
        raise ValueError('verified own-count model required')
    return ca.Function.load(str(directory/'transition.casadi')), ca.Function.load(str(directory/'first.casadi'))


def orbit(args):
    transition, first = functions(args.model)
    env = env17()
    design = load(args.model/'design.npz')
    static_gain, _ = trajectory_feedback(np.zeros((251, M)), np.zeros(250), first, np.zeros((M, M)), design['Q'])
    profiles, candidates, native_maps = [], [], 0
    for index, frequency in enumerate((2.5, 5., 6.25)):
        states, actions, harmonic = discrete_harmonic_initializer(design['A'], design['B'], frequency)
        initializer = states.copy()
        states, residuals, newton = refine_cyclic(transition, first, states, actions, maximum_iterations=8,
                                                 maximum_halvings=6, tolerance=1e-10)
        native = np.empty_like(states)
        for tick, action in enumerate(actions):
            env.reset(states[tick]); following, _ = env.step(action)
            native[tick] = following-states[(tick+1) % len(actions)]
            native_maps += 1
        closed = bool(np.max(np.abs(native)) <= 1e-10 and newton['all_original_rows_closed'])
        tracked_s = states[np.arange(251) % len(actions)]
        tracked_u = actions[np.arange(250) % len(actions)]
        gains, offsets = trajectory_feedback(tracked_s, tracked_u, first, np.zeros((M, M)), design['Q'],
                                             defects=native[np.arange(250) % len(actions)])
        path = args.out/f'frequency-{frequency:g}.npz'
        np.savez_compressed(path, states=np.r_[states, states[:1]], actions=actions, gains=gains, offsets=offsets,
                            initializer=initializer, native_defects=native, analytic_residuals=residuals)
        row = dict(**harmonic, profile_index=index, closure_pass=closed,
                   maximum_native_gap=float(np.max(np.abs(native))), maximum_analytic_gap=float(np.max(np.abs(residuals))),
                   maximum_gain_norm=float(np.max(np.linalg.norm(gains, axis=1))), newton=newton,
                   artifact_path=str(path), artifact_sha256=sha(path))
        profiles.append(row)
        if closed:
            candidates.append((row['maximum_gain_norm'], index, tracked_s, tracked_u, gains, offsets))
        print(json.dumps(dict(event='orbit_profile', **row)), flush=True)
    selected = min(candidates, key=lambda row: row[:2]) if candidates else None
    cases = []
    if selected is not None:
        _, index, states, actions, gains, offsets = selected
        for case_index, epsilon in enumerate((1e-15, 1e-11)):
            direction = np.random.default_rng(3171010).uniform(-1., 1., M)
            direction /= np.max(np.abs(direction))
            initial = states[0]+epsilon*direction
            def controller(state, tick):
                if tick >= 250:
                    return 0.
                return actions[tick]-offsets[tick]-float(gains[tick]@(state-states[tick]))
            s, u, raw, score = record_episode(controller, initial, CanonicalProtocol(n_links=N))
            tape = args.out/f'local-{3171010+case_index}.npz'
            np.savez_compressed(tape, states=s, actions=u, raw_actions=raw, direction=direction)
            cases.append(dict(diagnostic_label=3171010+case_index, epsilon=epsilon, **score,
                              first250_hold=bool(len(u) >= 250 and np.all(np.abs(absolute_angles(s[1:251], N)) < .15)),
                              recording_path=str(tape), recording_sha256=sha(tape)))
    return dict(schema='rage_mill.canonical17_orbit_component.v1', n_links=N, profiles=profiles, cases=cases,
                selected_profile_index=None if selected is None else selected[1],
                static_finite250_maximum_gain_norm=float(np.max(np.linalg.norm(static_gain, axis=1))),
                selection_rule='minimum maximum finite250 gain norm among own-count native closed profiles',
                native_orbit_map_commands=native_maps, local_only=True, full_hanging_episodes=0)


def launch(args):
    _, first = functions(args.model)
    env = env17()
    initial = hanging_state(env.protocol, 3171020, exact=True)
    a, b = map(np.asarray, first(initial, 0.))
    total = np.tril(np.ones((N, N)))
    q = np.eye(M)*1e-8
    q[0, 0], q[D, D] = 10., 5.
    q[1:D, 1:D] += 10.*(total.T@total)
    q[D+1:, D+1:] += total.T@total
    p = solve_discrete_are(a, b, q, np.array([[1000.]]))
    gain = ((b.T@p@a)/(1000.+(b.T@p@b)[0, 0])).ravel()
    target = initial.copy(); target[0] = -.05
    np.savez_compressed(args.out/'design.npz', A=a, B=b, Q=q, P=p, gain=gain, target=target)
    cases = []
    for seed, exact in ((3171020, True), (3171021, False), (3171022, False)):
        state = env.reset(hanging_state(env.protocol, seed, exact=exact))
        states, actions, raw, times = [state], [], [], [0.]
        for tick in range(850):
            command = -float(gain@(state-target))
            state, delivered = env.step(command)
            states.append(state); actions.append(delivered); raw.append(command); times.append(float(env.data.time))
            if abs(state[0]) > 3.:
                break
        path = args.out/f'parking-{seed}.npz'
        np.savez_compressed(path, states=np.asarray(states), actions=np.asarray(actions, dtype=np.float32),
                            raw_actions=np.asarray(raw), times=np.asarray(times))
        cases.append(dict(seed=seed, exact=exact, ticks=len(actions), launch_error_norm=float(np.linalg.norm(state-target)),
                          maximum_cart_m=float(np.max(np.abs(np.asarray(states)[:, 0]))),
                          recording_path=str(path), recording_sha256=sha(path)))
    return dict(schema='rage_mill.canonical17_parking_component.v1', n_links=N, park_ticks=850,
                method='own-count hanging DARE angles10/rates1/cart10/cart-rate5/R1000', cases=cases,
                full_hanging_swingup_episodes=0, component_only=True)


def bridge(args):
    transition, first = functions(args.model)
    env = env17(); design = load(args.model/'design.npz')
    orbit_receipt = json.loads((args.orbit/'receipt.json').read_text())
    selected = orbit_receipt['profiles'][orbit_receipt['selected_profile_index']]
    if not selected['closure_pass']:
        raise ValueError('closed own-count orbit required')
    orbit_data = load(selected['artifact_path'])
    orbit_s, orbit_u = orbit_data['states'], orbit_data['actions']
    launch_data = load(args.launch/'parking-3171020.npz')
    initial = launch_data['states'][850].copy()
    donor = load(args.donor)
    warm_s = lift_material(donor['states'][:451], N)
    warm_u = delivered_actions(donor['actions'][:450])
    warm_s[:, 0] += initial[0]-warm_s[0, 0]
    phases = (np.arange(451)-450) % len(orbit_u)
    orbit_guess = orbit_s[phases]
    clock = np.clip((np.arange(451)-150)/300, 0., 1.)
    weight = 6*clock**5-15*clock**4+10*clock**3
    derivative = (30*clock**4-60*clock**3+30*clock**2)/6.
    warm_s = np.c_[(1-weight[:, None])*warm_s[:, :D]+weight[:, None]*orbit_guess[:, :D],
                   (1-weight[:, None])*warm_s[:, D:]+weight[:, None]*orbit_guess[:, D:]
                   +derivative[:, None]*(orbit_guess[:, :D]-warm_s[:, :D])]
    warm_s[0], warm_s[-1] = initial, orbit_s[0]
    warm_u = (1-weight[:-1])*warm_u+weight[:-1]*orbit_u[phases[:-1]]
    np.savez_compressed(args.out/'initializer.npz', states=warm_s, actions=warm_u)
    count = 450
    states = ca.MX.sym('bridge_states', M, count+1)
    raw_actions = ca.MX.sym('raw_actions', 1, count)
    deliver = delivered_callback(count)
    actions = deliver(raw_actions)
    z = ca.vertcat(ca.vec(states), ca.vec(raw_actions))
    gaps = states[:, 1:]-transition.map(count)(states[:, :-1], actions)
    boundary = ca.vertcat(states[:, 0]-initial, states[:, -1]-orbit_s[0])
    g = ca.vertcat(ca.vec(gaps), boundary)
    aa, bb = first.map(count)(states[:, :-1], actions)
    block_a = ca.diagcat(*[aa[:, tick*M:(tick+1)*M] for tick in range(count)])
    block_b = ca.diagcat(*[bb[:, tick] for tick in range(count)])
    state_part = ca.horzcat(-block_a, ca.MX(ca.Sparsity(M*count, M)))+ca.horzcat(ca.MX(ca.Sparsity(M*count, M)), ca.DM.eye(M*count))
    jac = ca.vertcat(ca.horzcat(state_part, -block_b), ca.jacobian(boundary, z))
    jac_g = ca.Function('bridge_surrogate_jac', [z, ca.MX.sym('p', 0)], [g, jac])
    objective = 1e-4*ca.sumsqr(raw_actions-ca.DM(warm_u.reshape(1, -1)))+1e-7*ca.sumsqr(states-ca.DM(warm_s.T))
    solver = ca.nlpsol('canonical17_bridge', 'ipopt', {'x': z, 'f': objective, 'g': g}, {
        'jac_g': jac_g, 'ipopt.hessian_approximation': 'limited-memory', 'ipopt.max_iter': 150,
        'ipopt.max_cpu_time': 180., 'ipopt.tol': 1e-9, 'ipopt.constr_viol_tol': 1e-9,
        'ipopt.acceptable_iter': 0, 'ipopt.bound_relax_factor': 0., 'ipopt.nlp_scaling_method': 'none',
        'ipopt.mu_strategy': 'adaptive', 'ipopt.linear_solver': 'mumps', 'ipopt.print_level': 5, 'print_time': False})
    lower_s = np.full((451, M), -80.); upper_s = -lower_s
    lower_s[:, 0], upper_s[:, 0] = -2.8, 2.8
    solved = solver(x0=np.r_[warm_s.ravel(), warm_u], lbx=np.r_[lower_s.ravel(), np.full(450, -.95)],
                    ubx=np.r_[upper_s.ravel(), np.full(450, .95)], lbg=np.zeros(M*450+2*M), ubg=np.zeros(M*450+2*M))
    vector = np.asarray(solved['x']).ravel()
    bridge_s = vector[:M*451].reshape(451, M)
    raw_u = vector[M*451:]; bridge_u = delivered_actions(raw_u)
    np.savez_compressed(args.out/'raw-solver.npz', states=bridge_s, actions=raw_u)
    bridge_s[0], bridge_s[-1] = initial, orbit_s[0]
    full_s = np.r_[bridge_s, orbit_s[np.arange(1, 251) % len(orbit_u)]]
    full_u = np.r_[bridge_u, orbit_u[np.arange(250) % len(orbit_u)]]
    native = np.empty((700, M))
    for tick, action in enumerate(full_u):
        env.reset(full_s[tick]); following, _ = env.step(action)
        native[tick] = following-full_s[tick+1]
    gains, offsets = trajectory_feedback(full_s, full_u, first, np.zeros((M, M)), design['Q'], defects=native)
    parking = load(args.launch/'design.npz')
    np.savez_compressed(args.out/'reference.npz', states=full_s, actions=full_u, gains=gains, offsets=offsets, native_defects=native)
    np.savez_compressed(args.out/'controller.npz', states=full_s[:651], actions=full_u[:650], gains=gains[:650], offsets=offsets[:650],
                        native_defects=native[:650], balance_gain=np.zeros(M), park_gain=parking['gain'], parking_target=parking['target'],
                        terminal_root=np.zeros((M, M)), park_ticks=np.int64(850))
    return dict(schema='rage_mill.canonical17_bridge.v1', n_links=N, reference_only=True,
                native_reference_feasible=bool(np.max(np.abs(native)) <= 1e-8),
                maximum_native_gap=float(np.max(np.abs(native))), maximum_gain_norm=float(np.max(np.linalg.norm(gains, axis=1))),
                full_reference_ticks=700, controller_route_ticks=650, parking_ticks=850,
                donor_sha256=sha(args.donor), orbit_sha256=selected['artifact_sha256'],
                launch_recording_sha256=sha(args.launch/'parking-3171020.npz'), controller_sha256=sha(args.out/'controller.npz'),
                optimizer_status=solver.stats()['return_status'], optimizer_iterations=solver.stats()['iter_count'],
                delivery='float32 values with explicitly disclosed straight-through optimizer input derivative')


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument('stage', choices=('model', 'orbit', 'launch', 'bridge'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--donor', type=Path)
    parser.add_argument('--orbit', type=Path)
    parser.add_argument('--launch', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cpu, wall = time.process_time(), time.monotonic()
    result = globals()[args.stage](args)
    result.update(controller_qualified=False, development_episodes=0, reserved_episodes=0,
                  CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall)
    write(args.out/'receipt.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
