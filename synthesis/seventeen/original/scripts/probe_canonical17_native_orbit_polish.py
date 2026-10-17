"""Native-value cyclic polish with unchanged bridge/actions and saved controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
import warnings

import casadi as ca
import numpy as np
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol, absolute_angles, score_tape
from rage_mill.canonical17_controller import Canonical17FiniteTrajectoryPolicy
from rage_mill.canonical16_orbit import gauge_system
from rage_mill.canonical16_trajectory import trajectory_feedback
from synthesize_canonical17 import load, sha, write


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    for name in ('model', 'reference', 'controller', 'saved', 'offset-source', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cpu, wall = time.process_time(), time.monotonic()
    first = ca.Function.load(str(args.model/'first.casadi'))
    reference, original, design = load(args.reference), load(args.controller), load(args.model/'design.npz')
    initial = load(args.saved)['states'][850].copy()
    protocol, env = CanonicalProtocol(n_links=17), CanonicalEnv(CanonicalProtocol(n_links=17))
    states = reference['states'][450:546].copy()
    actions = reference['actions'][450:546].copy()
    native_maps = 0

    def residuals(candidate):
        nonlocal native_maps
        values = np.empty_like(candidate)
        for tick, command in enumerate(actions):
            env.reset(candidate[tick]); following, _ = env.step(command)
            values[tick] = candidate[(tick+1) % 96]-following
        native_maps += 96
        return values

    original_states = states.copy()
    history = []
    for iteration in range(13):
        values = residuals(states)
        maximum = float(np.max(np.abs(values)))
        gauge = float(np.mean(states[:, 0]))
        history.append(dict(iteration=iteration, maximum_original_native_gap=maximum, mean_cart_gauge=gauge,
                            dropped_original_cart_row=float(values[-1, 0])))
        if maximum <= 1e-13 and abs(gauge) <= 1e-13 or iteration == 12:
            break
        matrix, right, dropped = gauge_system(first, states, actions, values)
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            step = spsolve(matrix, -right).reshape(states.shape)
        accepted = False
        for halving in range(9):
            alpha = 2.**-halving
            candidate = states+alpha*step
            if (not np.isfinite(candidate).all() or np.max(np.abs(absolute_angles(candidate, 17))) >= .145
                    or np.max(np.abs(candidate[:, 0])) > 2.8 or np.max(np.abs(candidate)) > 80):
                continue
            proposed = residuals(candidate)
            modified = proposed.ravel().copy(); modified[dropped] = np.mean(candidate[:, 0])
            if np.max(np.abs(modified)) < np.max(np.abs(right)):
                states = candidate; accepted = True
                history[-1].update(alpha=alpha, resulting_maximum_native_gap=float(np.max(np.abs(proposed))))
                break
        if not accepted:
            history[-1]['line_search_failed'] = True
            break
    final = residuals(states)
    np.savez_compressed(args.out/'polished-orbit.npz', states=np.r_[states, states[:1]], actions=actions,
                        source_states=np.r_[original_states, original_states[:1]], native_defects=-final)
    full_s = reference['states'].copy()
    full_s[450:] = states[np.arange(251) % 96]
    full_u = reference['actions'].copy()
    native = np.empty((700, 36))
    for tick, action in enumerate(full_u):
        env.reset(full_s[tick]); following, _ = env.step(action)
        native[tick] = following-full_s[tick+1]
    native_maps += 700
    gains, offsets = trajectory_feedback(full_s, full_u, first, np.zeros((36, 36)), design['Q'],
                                         control_cost=1000., defects=native)
    tables = {key: value.copy() for key, value in original.items()}
    tables.update(states=full_s[:651], gains=gains[:650], offsets=offsets[:650], native_defects=native[:650])
    table_path = args.out/'controller-polished.npz'
    np.savez_compressed(table_path, **tables)
    np.savez_compressed(args.out/'reference-polished.npz', states=full_s, actions=full_u, gains=gains,
                        offsets=offsets, native_defects=native)
    shutil.copyfile(args.controller, args.out/'controller-zero-change.npz')
    finite_boxes = bool(np.isfinite(full_s).all() and np.isfinite(full_u).all()
                        and np.max(np.abs(full_s[:, 0])) <= 2.8 and np.max(np.abs(full_s)) <= 80
                        and np.max(np.abs(full_u)) <= .95)
    strict_tail = bool(np.all(np.abs(absolute_angles(full_s[451:], 17)) < .15))
    if not finite_boxes or not strict_tail:
        raise ValueError('polished reference violates registered nominal physical boxes/angles')
    cases = []
    settings = [(True, 0.), (True, 1e-6)] + [(False, value) for value in
                (0., 1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2)]
    for profile_index, (zero, offset) in enumerate(settings):
        selected = original if zero else tables
        policy = Canonical17FiniteTrajectoryPolicy(
            **{key: selected[key] for key in ('states', 'actions', 'gains', 'offsets', 'balance_gain', 'park_gain', 'parking_target')},
            park_ticks=850)
        policy.next_tick = 850
        policy.launch_cart_offset = initial[0]-policy.states[0, 0]
        if offset != 0.:
            policy.launch_cart_offset += offset
        policy.launch_branch_offset = np.zeros(17)
        tape_s, tape_u, raw = [env.reset(initial)], [], []
        invalid = None
        for tick in range(850, 1500):
            command = policy(tape_s[-1].copy(), tick)
            try:
                following, delivered = env.step(command)
            except (ValueError, FloatingPointError) as error:
                invalid = f'{type(error).__name__}: {error}'
                break
            tape_s.append(following); tape_u.append(delivered); raw.append(command)
            if abs(following[0]) > 3.:
                break
        tape_s, tape_u, raw = np.asarray(tape_s), np.asarray(tape_u, dtype=np.float32), np.asarray(raw)
        score = score_tape(tape_s, tape_u, protocol, invalid_reason=invalid)
        passed = bool(len(tape_u) == 650 and score['longest_upright_ticks'] >= 250 and invalid is None
                      and all(score[key] for key in ('finite', 'force_ok', 'float32_delivery_ok', 'rail_ok')))
        parity = None
        if zero:
            control = load(args.offset_source/f'profile-{profile_index}.npz')
            parity = bool(all(control[key].dtype == value.dtype and control[key].shape == value.shape
                              and control[key].tobytes() == value.tobytes()
                              for key, value in (('states', tape_s), ('actions', tape_u), ('raw_actions', raw))))
            if not parity:
                raise ValueError('literal zero-change saved forecast differs')
        path = args.out/f'forecast-{profile_index}.npz'
        np.savez_compressed(path, states=tape_s, actions=tape_u, raw_actions=raw)
        row = dict(profile_index=profile_index, zero_change_control=zero, cart_error_offset_m=offset, **score,
                   forecast_pass=passed, zero_change_saved_forecast_byte_parity=parity,
                   recording_path=str(path), recording_sha256=sha(path))
        cases.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    passing = next((row for row in cases if not row['zero_change_control'] and row['forecast_pass']), None)
    result = dict(schema='rage_mill.canonical17_native_orbit_polish_discriminator.v1', n_links=17,
                  diagnostic_only=True, controller_qualified=False, actual_episode_calls=0, reset_RNG_draws=0,
                  development_episodes=0, reserved_episodes=0, Newton_iterations_max=12, halvings_max=8,
                  history=history, maximum_native_orbit_gap=float(np.max(np.abs(final))),
                  maximum_orbit_state_change=float(np.max(np.abs(states-original_states))),
                  native_component_commands=native_maps, native_forecast_commands=sum(row['ticks'] for row in cases),
                  bridge_states0_through449_literal_bytes=full_s[:450].tobytes() == reference['states'][:450].tobytes(),
                  all700_actions_literal_bytes=full_u.tobytes() == reference['actions'].tobytes(),
                  first449_native_defect_rows_literal_bytes=native[:449].tobytes() == reference['native_defects'][:449].tobytes(),
                  maximum_bridge_native_gap=float(np.max(np.abs(native[:450]))),
                  maximum_full_native_gap=float(np.max(np.abs(native))), nominal_finite_boxes=finite_boxes,
                  all250_nominal_tail_angles_strict=strict_tail, maximum_gain_norm=float(np.max(np.linalg.norm(gains, axis=1))),
                  source_maximum_tail_affine_offset=float(np.max(np.abs(reference['offsets'][450:]))),
                  polished_maximum_tail_affine_offset=float(np.max(np.abs(offsets[450:]))),
                  cases=cases, selected_profile_index=None if passing is None else passing['profile_index'],
                  controller_path=str(table_path), controller_sha256=sha(table_path),
                  CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall)
    write(args.out/'receipt.json', result)


if __name__ == '__main__':
    main()
