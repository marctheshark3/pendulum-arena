"""Bounded native shadows from a saved actual launch; no new reset draws."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol, score_tape
from rage_mill.canonical17_controller import Canonical17FiniteTrajectoryPolicy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    for name in ('controller', 'saved', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cpu, wall = time.process_time(), time.monotonic()
    tables, saved = load(args.controller), load(args.saved)
    launch = int(tables['park_ticks'])
    if len(saved['states']) <= launch or launch+len(tables['actions']) != 1500:
        raise ValueError('literal saved launch and full remaining route required')
    initial = saved['states'][launch].copy()
    offsets = (0., 1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2)
    env, protocol, cases = CanonicalEnv(CanonicalProtocol(n_links=17)), CanonicalProtocol(n_links=17), []
    for profile_index, offset in enumerate(offsets):
        policy = Canonical17FiniteTrajectoryPolicy(
            **{key: tables[key] for key in ('states', 'actions', 'gains', 'offsets', 'balance_gain', 'park_gain', 'parking_target')},
            park_ticks=launch)
        policy.next_tick = launch
        policy.launch_cart_offset = initial[0]-policy.states[0, 0]
        if offset != 0.:
            policy.launch_cart_offset += offset
        policy.launch_branch_offset = np.zeros(17)
        states, actions, raw = [env.reset(initial)], [], []
        invalid = None
        for tick in range(launch, 1500):
            command = policy(states[-1].copy(), tick)
            try:
                following, delivered = env.step(command)
            except (ValueError, FloatingPointError) as error:
                invalid = f'{type(error).__name__}: {error}'
                break
            states.append(following); actions.append(delivered); raw.append(command)
            if abs(following[0]) > 3.:
                break
        states, actions, raw = np.asarray(states), np.asarray(actions, dtype=np.float32), np.asarray(raw)
        score = score_tape(states, actions, protocol, invalid_reason=invalid)
        passed = bool(len(actions) == 1500-launch and score['longest_upright_ticks'] >= 250
                      and all(score[key] for key in ('finite', 'force_ok', 'float32_delivery_ok', 'rail_ok')) and invalid is None)
        parity = None
        if profile_index == 0:
            parity = bool(states.dtype == saved['states'].dtype == np.dtype('float64')
                          and states.shape == saved['states'][launch:].shape
                          and states.tobytes() == saved['states'][launch:].tobytes()
                          and raw.dtype == saved['raw_actions'].dtype == np.dtype('float64')
                          and raw.shape == saved['raw_actions'][launch:].shape
                          and raw.tobytes() == saved['raw_actions'][launch:].tobytes()
                          and actions.tobytes() == saved['actions'][launch:].astype(np.float32).tobytes())
            if not parity:
                raise ValueError('zero-change shadow fails literal saved actual suffix parity')
        path = args.out/f'profile-{profile_index}.npz'
        np.savez_compressed(path, states=states, actions=actions, raw_actions=raw)
        row = dict(profile_index=profile_index, cart_error_offset_m=offset, **score,
                   forecast_pass=passed, zero_change_actual_suffix_byte_parity=parity,
                   recording_path=str(path), recording_sha256=sha(path))
        cases.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    selected = next((row for row in cases if row['forecast_pass']), None)
    receipt = dict(schema='rage_mill.canonical17_saved_offset_discriminator.v1', n_links=17,
                   diagnostic_only=True, controller_qualified=False, actual_episode_calls=0,
                   reset_RNG_draws=0, development_episodes=0, reserved_episodes=0,
                   cases=cases, selected_profile_index=None if selected is None else selected['profile_index'],
                   controller_sha256=sha(args.controller), saved_recording_sha256=sha(args.saved),
                   forecast_commands=sum(row['ticks'] for row in cases),
                   CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall)
    with (args.out/'receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')


if __name__ == '__main__':
    main()
