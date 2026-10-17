"""Single predeclared proximal-rate residual bracket on one saved launch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from rage_mill.canonical17_rate_residual import Canonical17RateResidualPolicy
from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol, score_tape
from synthesize_canonical17 import load, sha, write


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    for name in ('controller', 'saved', 'offset-source', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cpu, wall = time.process_time(), time.monotonic()
    tables, saved = load(args.controller), load(args.saved)
    initial = saved['states'][850].copy()
    protocol, env = CanonicalProtocol(n_links=17), CanonicalEnv(CanonicalProtocol(n_links=17))
    settings = [(0., 1., 0.), (0., 1., 1e-6)] + [(amplitude, sign, offset)
                for amplitude in (1e-4, 1e-3, 1e-2) for sign in (1., -1.) for offset in (0., 1e-6)]
    profiles = []
    for profile_index, (amplitude, sign, offset) in enumerate(settings):
        policy = Canonical17RateResidualPolicy(
            **{key: tables[key] for key in ('states', 'actions', 'gains', 'offsets', 'balance_gain', 'park_gain', 'parking_target')},
            park_ticks=850, residual_sign=sign, residual_amplitude=amplitude)
        policy.next_tick = 850
        policy.launch_cart_offset = initial[0]-policy.states[0, 0]
        if offset != 0.:
            policy.launch_cart_offset += offset
        policy.launch_branch_offset = np.zeros(17)
        states, actions, raw, base_raw, residuals, features, weights = [env.reset(initial)], [], [], [], [], [], []
        invalid = None
        for tick in range(850, 1500):
            command = policy(states[-1].copy(), tick)
            try:
                following, delivered = env.step(command)
            except (ValueError, FloatingPointError) as error:
                invalid = f'{type(error).__name__}: {error}'
                break
            states.append(following); actions.append(delivered); raw.append(command)
            base_raw.append(policy.last_base_raw); residuals.append(policy.last_residual)
            features.append(policy.last_feature); weights.append(policy.last_weight)
            if abs(following[0]) > 3.:
                break
        states, actions, raw = np.asarray(states), np.asarray(actions, dtype=np.float32), np.asarray(raw)
        score = score_tape(states, actions, protocol, invalid_reason=invalid)
        passed = bool(len(actions) == 650 and score['longest_upright_ticks'] >= 250 and invalid is None
                      and all(score[key] for key in ('finite', 'force_ok', 'float32_delivery_ok', 'rail_ok')))
        parity = None
        if amplitude == 0.:
            control = load(args.offset_source/f'profile-{profile_index}.npz')
            parity = bool(all(control[key].dtype == value.dtype and control[key].shape == value.shape
                              and control[key].tobytes() == value.tobytes()
                              for key, value in (('states', states), ('actions', actions), ('raw_actions', raw))))
            if not parity:
                raise ValueError('zero-amplitude literal saved control differs')
        path = args.out/f'profile-{profile_index}.npz'
        np.savez_compressed(path, states=states, actions=actions, raw_actions=raw, base_raw_actions=np.asarray(base_raw),
                            residual_actions=np.asarray(residuals), features=np.asarray(features), weights=np.asarray(weights))
        saturated = np.flatnonzero(np.abs(raw) > 1.)
        row = dict(profile_index=profile_index, residual_amplitude=amplitude, residual_sign=sign,
                   cart_error_offset_m=offset, **score, forecast_pass=passed,
                   zero_amplitude_saved_forecast_byte_parity=parity,
                   first_saturation_route_tick=None if not len(saturated) else int(saturated[0]),
                   maximum_absolute_raw_action=float(np.max(np.abs(raw), initial=0.)),
                   maximum_absolute_residual=float(np.max(np.abs(residuals), initial=0.)),
                   recording_path=str(path), recording_sha256=sha(path))
        profiles.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    selected = next((row for row in profiles if row['forecast_pass']), None)
    result = dict(schema='rage_mill.canonical17_proximal_rate_residual_discriminator.v1', n_links=17,
                  diagnostic_only=True, controller_qualified=False, actual_episode_calls=0, reset_RNG_draws=0,
                  development_episodes=0, reserved_episodes=0, maximum_forecast_commands=9100,
                  feature='observed_state[19]-current_reference[route_tick,19]',
                  residual='sign*amplitude*clip((route_tick-425)/25,0,1)*clip(feature/.01,-1,1)',
                  profiles=profiles, selected_profile_index=None if selected is None else selected['profile_index'],
                  controller_sha256=sha(args.controller), forecast_commands=sum(row['ticks'] for row in profiles),
                  CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall)
    write(args.out/'receipt.json', result)


if __name__ == '__main__':
    main()
