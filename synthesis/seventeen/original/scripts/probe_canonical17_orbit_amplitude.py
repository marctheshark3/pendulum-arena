"""Matched own-seventeen bent-orbit amplitude discriminator with zero control.

Every novel profile is independently native checked. The old selected orbit
and its exact saved starts are replayed first without a new reset draw.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import casadi as ca
import numpy as np
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol, absolute_angles, record_episode
from rage_mill.canonical16_orbit import cyclic_values, gauge_system
from rage_mill.canonical16_trajectory import trajectory_feedback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def refine(transition, first, states, actions):
    history = []
    for iteration in range(11):
        residuals = cyclic_values(transition, states, actions)
        maximum = float(np.max(np.abs(residuals)))
        history.append(dict(iteration=iteration, maximum_original_gap=maximum))
        if maximum <= 1e-12 or iteration == 10:
            break
        matrix, right, dropped = gauge_system(first, states, actions, residuals)
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            step = spsolve(matrix, -right).reshape(states.shape)
        accepted = False
        for halving in range(7):
            alpha = 2.**-halving
            candidate = states+alpha*step
            if (np.max(np.abs(np.cumsum(candidate[:, 1:18], axis=1))) >= .145
                    or np.max(np.abs(candidate[:, 0])) > 2.5 or np.max(np.abs(candidate)) > 80):
                continue
            values = cyclic_values(transition, candidate, actions)
            modified = values.ravel().copy(); modified[dropped] = np.mean(candidate[:, 0])
            if np.max(np.abs(modified)) < np.max(np.abs(right)):
                states = candidate; accepted = True
                history[-1].update(alpha=alpha)
                break
        if not accepted:
            history[-1]['line_search_failed'] = True
            break
    return states, cyclic_values(transition, states, actions), history


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    for name in ('model', 'baseline', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cpu, wall = time.process_time(), time.monotonic()
    transition = ca.Function.load(str(args.model/'transition.casadi'))
    first = ca.Function.load(str(args.model/'first.casadi'))
    design = load(args.model/'design.npz')
    baseline = load(args.baseline/'frequency-5.npz')
    direction = load(args.baseline/'local-3171010.npz')['direction']
    protocol, profiles = CanonicalProtocol(n_links=17), []
    env = CanonicalEnv(protocol)
    # Registration is literal and finite: zero change then six amplitudes.
    settings = [(5., .08, True)] + [(f, a, False) for f in (5., 6.25) for a in (.10, .12, .14)]
    for profile_index, (frequency, amplitude, zero) in enumerate(settings):
        if zero:
            cyclic_s, cyclic_u = baseline['states'][:-1].copy(), baseline['actions'].copy()
            gains, offsets = baseline['gains'], baseline['offsets']
            native = baseline['native_defects']
            residuals = baseline['analytic_residuals']
            history = []
        else:
            period = {5.: 10, 6.25: 8}[frequency]
            count = 100 if frequency == 5. else 96
            phase = np.exp(2j*np.pi*frequency*.02)
            response = np.linalg.solve(phase*np.eye(36)-design['A'], design['B'][:, 0])
            absolute = np.cumsum(response[1:18])
            force_amplitude = amplitude/np.max(np.abs(absolute))
            if force_amplitude > .95:
                raise ValueError('registered orbit force cap exceeded')
            half = (force_amplitude*np.cos(2*np.pi*frequency*.02*np.arange(period//2))).astype(np.float32)
            cyclic_u = np.tile(np.r_[half, -half], count//period).astype(np.float64)
            cyclic_s = np.real(np.exp(2j*np.pi*frequency*.02*np.arange(count))[:, None]*(force_amplitude*response)[None])
            cyclic_s, residuals, history = refine(transition, first, cyclic_s, cyclic_u)
            native = np.empty_like(cyclic_s)
            for tick, action in enumerate(cyclic_u):
                env.reset(cyclic_s[tick]); following, _ = env.step(action)
                native[tick] = following-cyclic_s[(tick+1) % count]
            tracked_s = cyclic_s[np.arange(251) % count]
            tracked_u = cyclic_u[np.arange(250) % count]
            gains, offsets = trajectory_feedback(tracked_s, tracked_u, first, np.zeros((36, 36)), design['Q'],
                                                 defects=native[np.arange(250) % count])
        count = len(cyclic_u)
        tracked_s, tracked_u = cyclic_s[np.arange(251) % count], cyclic_u[np.arange(250) % count]
        path = args.out/f'profile-{profile_index}.npz'
        np.savez_compressed(path, states=np.r_[cyclic_s, cyclic_s[:1]], actions=cyclic_u, gains=gains, offsets=offsets,
                            native_defects=native, analytic_residuals=residuals)
        closed = bool(np.max(np.abs(native)) <= 1e-10 and np.max(np.abs(residuals)) <= 1e-10
                      and np.max(np.abs(absolute_angles(cyclic_s, 17))) < .145)
        cases = []
        if closed:
            for case_index, epsilon in enumerate((1e-15, 1e-11)):
                initial = tracked_s[0]+epsilon*direction
                def controller(state, tick):
                    if tick >= 250:
                        return 0.
                    return tracked_u[tick]-offsets[tick]-float(gains[tick]@(state-tracked_s[tick]))
                s, u, raw, score = record_episode(controller, initial, protocol)
                tape = args.out/f'profile-{profile_index}-local-{3171200+2*profile_index+case_index}.npz'
                np.savez_compressed(tape, states=s, actions=u, raw_actions=raw)
                row = dict(diagnostic_label=3171200+2*profile_index+case_index, epsilon=epsilon, **score,
                           first250_hold=bool(len(u) >= 250 and np.all(np.abs(absolute_angles(s[1:251], 17)) < .15)),
                           recording_path=str(tape), recording_sha256=sha(tape))
                if zero:
                    saved = load(args.baseline/f'local-{3171010+case_index}.npz')
                    row['saved_zero_change_byte_parity'] = all(saved[key].tobytes() == value.tobytes()
                                                               for key, value in (('states', s), ('actions', u), ('raw_actions', raw)))
                    if not row['saved_zero_change_byte_parity']:
                        raise ValueError('zero-change local physical control differs from saved tape')
                cases.append(row)
        row = dict(profile_index=profile_index, frequency_Hz=frequency, nominal_maximum_angle_rad=amplitude,
                   zero_change_control=zero, closure_pass=closed, maximum_native_gap=float(np.max(np.abs(native))),
                   maximum_analytic_gap=float(np.max(np.abs(residuals))), actual_orbit_maximum_angle_rad=float(np.max(np.abs(absolute_angles(cyclic_s, 17)))),
                   maximum_gain_norm=float(np.max(np.linalg.norm(gains, axis=1))), newton_history=history, cases=cases,
                   both_local250_holds=bool(len(cases) == 2 and all(c['first250_hold'] for c in cases)),
                   artifact_path=str(path), artifact_sha256=sha(path))
        profiles.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    eligible = [row for row in profiles if row['closure_pass'] and row['both_local250_holds']]
    selected = min(eligible, key=lambda row: (row['maximum_gain_norm'], row['profile_index'])) if eligible else None
    receipt = dict(schema='rage_mill.canonical17_orbit_amplitude_discriminator.v1', n_links=17, diagnostic_only=True,
                   controller_qualified=False, full_hanging_episodes=0, development_episodes=0, reserved_episodes=0,
                   profiles=profiles, selected_profile_index=None if selected is None else selected['profile_index'],
                   selection_rule='native closed and both matched first250 holds, then minimum maximum finite250 gain norm',
                   CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall)
    with (args.out/'receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')


if __name__ == '__main__':
    main()
