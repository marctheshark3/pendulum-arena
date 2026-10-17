"""Fixed saved-launch input-cost discriminator, retaining two literal controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import casadi as ca
import numpy as np

from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol, score_tape
from rage_mill.canonical17_controller import Canonical17FiniteTrajectoryPolicy
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
    saved = load(args.saved)
    initial = saved['states'][850].copy()
    protocol, env, profiles = CanonicalProtocol(n_links=17), CanonicalEnv(CanonicalProtocol(n_links=17)), []
    for cost_index, cost in enumerate((1000., 100., 1e4, 1e6)):
        if cost == 1000.:
            tables = {key: value.copy() for key, value in original.items()}
            table_path = args.out/'controller-R1000.npz'
            shutil.copyfile(args.controller, table_path)
        else:
            gains, offsets = trajectory_feedback(reference['states'], reference['actions'], first, np.zeros((36, 36)),
                                                 design['Q'], control_cost=cost, defects=reference['native_defects'])
            tables = {key: value.copy() for key, value in original.items()}
            tables['gains'], tables['offsets'] = gains[:650], offsets[:650]
            table_path = args.out/f'controller-R{cost:g}.npz'
            np.savez_compressed(table_path, **tables)
        for offset_index, offset in enumerate((0., 1e-6)):
            policy = Canonical17FiniteTrajectoryPolicy(
                **{key: tables[key] for key in ('states', 'actions', 'gains', 'offsets', 'balance_gain', 'park_gain', 'parking_target')},
                park_ticks=850)
            policy.next_tick = 850
            policy.launch_cart_offset = initial[0]-policy.states[0, 0]
            if offset != 0.:
                policy.launch_cart_offset += offset
            policy.launch_branch_offset = np.zeros(17)
            states, actions, raw = [env.reset(initial)], [], []
            invalid = None
            for tick in range(850, 1500):
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
            passed = bool(len(actions) == 650 and score['longest_upright_ticks'] >= 250 and invalid is None
                          and all(score[key] for key in ('finite', 'force_ok', 'float32_delivery_ok', 'rail_ok')))
            parity = None
            if cost == 1000.:
                control = load(args.offset_source/f'profile-{offset_index}.npz')
                parity = bool(all(control[key].dtype == value.dtype and control[key].shape == value.shape
                                  and control[key].tobytes() == value.tobytes()
                                  for key, value in (('states', states), ('actions', actions), ('raw_actions', raw))))
                if not parity:
                    raise ValueError('literal unchanged-R saved forecast control differs')
            tape = args.out/f'R{cost:g}-offset-{offset_index}.npz'
            np.savez_compressed(tape, states=states, actions=actions, raw_actions=raw)
            row = dict(profile_index=cost_index*2+offset_index, tracking_cost=cost, cart_error_offset_m=offset,
                       **score, forecast_pass=passed, zero_change_saved_forecast_byte_parity=parity,
                       maximum_gain_norm=float(np.max(np.linalg.norm(tables['gains'], axis=1))),
                       maximum_affine_offset=float(np.max(np.abs(tables['offsets']))),
                       controller_path=str(table_path), controller_sha256=sha(table_path),
                       recording_path=str(tape), recording_sha256=sha(tape))
            profiles.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
    selected = next((row for row in profiles if row['forecast_pass']), None)
    receipt = dict(schema='rage_mill.canonical17_tracking_cost_discriminator.v1', n_links=17,
                   diagnostic_only=True, controller_qualified=False, actual_episode_calls=0, reset_RNG_draws=0,
                   development_episodes=0, reserved_episodes=0, profiles=profiles,
                   selected_profile_index=None if selected is None else selected['profile_index'],
                   selection_rule='first complete in-rail650forecast with250uprightstates in literal registeredcost/offsetorder',
                   forecast_commands=sum(row['ticks'] for row in profiles),
                   CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall)
    write(args.out/'receipt.json', receipt)


if __name__ == '__main__':
    main()
