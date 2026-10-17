"""Single unchanged-duration bridge to the measured best local stronger orbit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from rage_mill.canonical_benchmark import absolute_angles
from synthesize_canonical17 import bridge, env17, load, sha, write


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    for name in ('model', 'orbit', 'launch', 'donor', 'out'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cpu, wall = time.process_time(), time.monotonic()
    source_receipt = json.loads((args.orbit/'receipt.json').read_text())
    selected = source_receipt['profiles'][6]
    if (selected['frequency_Hz'] != 6.25 or selected['nominal_maximum_angle_rad'] != .14
            or not selected['closure_pass'] or sha(selected['artifact_path']) != selected['artifact_sha256']):
        raise ValueError('literal registered stronger orbit source required')
    orbit = load(selected['artifact_path'])
    states, actions = orbit['states'], orbit['actions']
    if (states.shape != (97, 36) or actions.shape != (96,) or not np.isfinite(states).all()
            or not np.isfinite(actions).all() or np.max(np.abs(absolute_angles(states, 17))) >= .15
            or np.max(np.abs(actions)) > .95 or np.max(np.abs(states[:, 0])) > 2.8
            or np.max(np.abs(states)) > 80 or states[0].tobytes() != states[-1].tobytes()):
        raise ValueError('finite in-box literal cyclic own-count angle/force-compliant orbit required')
    env = env17()
    cold = np.empty((96, 36))
    for tick, action in enumerate(actions):
        env.reset(states[tick]); following, _ = env.step(action)
        cold[tick] = following-states[tick+1]
    np.savez_compressed(args.out/'orbit-native-check.npz', native_defects=cold)
    if np.max(np.abs(cold)) > 1e-10:
        raise ValueError('stronger orbit fails independent native cold closure')
    # This prospective selection wrapper never edits the failed amplitude
    # discriminator or substitutes its local result for full physical capture.
    selected_dir = args.out/'selected-orbit'; selected_dir.mkdir()
    selection = dict(source_receipt)
    selection['selected_profile_index'] = 6
    selection['new_selection_reason'] = 'single measured strongest local hold167 ticks; full-route discriminator only'
    write(selected_dir/'receipt.json', selection)
    original_orbit = args.orbit
    args.orbit = selected_dir
    result = bridge(args)
    initializer = load(args.out/'initializer.npz')
    initial_cold = np.empty((450, 36))
    for tick, action in enumerate(initializer['actions']):
        env.reset(initializer['states'][tick]); following, _ = env.step(action)
        initial_cold[tick] = following-initializer['states'][tick+1]
    np.savez_compressed(args.out/'initializer-native-check.npz', native_defects=initial_cold)
    reference = load(args.out/'reference.npz')
    finite = bool(np.isfinite(reference['states']).all() and np.isfinite(reference['actions']).all())
    boxes = bool(np.max(np.abs(reference['states'][:, 0])) <= 2.8
                 and np.max(np.abs(reference['states'])) <= 80 and np.max(np.abs(reference['actions'])) <= .95)
    tail = bool(np.all(np.abs(absolute_angles(reference['states'][451:], 17)) < .15))
    result.update(schema='rage_mill.canonical17_stronger_orbit_bridge.v1',
                  selected_profile_source_receipt_sha256=sha(original_orbit/'receipt.json'), selected_profile_index=6,
                  stronger_orbit_frequency_Hz=6.25, stronger_orbit_nominal_angle_rad=.14,
                  maximum_orbit_native_gap=float(np.max(np.abs(cold))),
                  maximum_initializer_native_gap=float(np.max(np.abs(initial_cold))),
                  reference_finite=finite, reference_planning_boxes_ok=boxes, all250_nominal_tail_angles_strict=tail,
                  native_reference_feasible=bool(finite and boxes and tail and result['maximum_native_gap'] <= 1e-8),
                  controller_qualified=False, development_episodes=0, reserved_episodes=0,
                  CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall)
    write(args.out/'receipt.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
