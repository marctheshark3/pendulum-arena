"""One prospectively selected 440-tick linear/Hermite initializer comparison.

Only the offline proposal changes. Both arms call the identical original
450-command native-delivery repair, bend blend, finite feedback and table
export. A duration of 450 returns literal donor array copies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from synthesize_canonical17 import bridge, load, sha, write


def retime(states, actions, *, method, route_ticks):
    if states.shape != (701, 34) or actions.shape != (700,) or method not in ('linear', 'hermite'):
        raise ValueError('literal 700-command sixteen offline donor and registered arm required')
    if route_ticks == 450:
        return states[:451].copy(), actions[:450].copy()
    if route_ticks != 440:
        raise ValueError('only prospectively selected 440-tick proposal permitted')
    sample = np.arange(route_ticks+1)*.02*450/route_ticks
    knots = np.arange(451)*.02
    ratio = 450/route_ticks
    if method == 'linear':
        first = np.array([np.interp(sample, knots, states[:451, column]) for column in range(34)]).T
        first[:, 17:] *= ratio
    else:
        spline = CubicHermiteSpline(knots, states[:451, :17], states[:451, 17:], axis=0)
        first = np.c_[spline(sample), spline(sample, 1)*ratio]
    # Continue the donor's actual bent-orbit initializer for the ten remaining
    # bridge knots; the unchanged own-count bend blend supplies the exact target.
    proposed_s = np.r_[first, states[451:461]]
    proposed_u = np.r_[np.interp(sample[:-1], np.arange(450)*.02, actions[:450]), actions[450:460]]
    return proposed_s, proposed_u


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    for name in ('model', 'orbit', 'launch', 'donor', 'out'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--method', choices=('linear', 'hermite'), required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cpu, wall = time.process_time(), time.monotonic()
    original_donor = args.donor
    donor = load(original_donor)
    for method in ('linear', 'hermite'):
        zero_s, zero_u = retime(donor['states'], donor['actions'], method=method, route_ticks=450)
        if zero_s.tobytes() != donor['states'][:451].tobytes() or zero_u.tobytes() != donor['actions'][:450].tobytes():
            raise ValueError('unchanged-duration proposal is not literal donor bytes')
    proposed_s, proposed_u = retime(donor['states'], donor['actions'], method=args.method, route_ticks=440)
    # The shared repair consumes 451 donor knots. Retain the source's remaining
    # 250 rows to satisfy the donor archive shape without using them in repair.
    warm = args.out/'retimed-donor-initializer.npz'
    np.savez_compressed(warm, states=np.r_[proposed_s, donor['states'][451:]],
                        actions=np.r_[proposed_u, donor['actions'][450:]])
    args.donor = warm
    receipt = bridge(args)
    receipt.update(schema='rage_mill.canonical17_retimed_bridge_comparison.v1',
                   original_donor_sha256=sha(original_donor), retiming_method=args.method,
                   donor_route_ticks=450, proposed_route_ticks=440, bridge_ticks_unchanged=450,
                   unchanged_duration_both_arms_literal_bytes=True,
                   arm_changes='offline q/v proposal only; same actions, repair, own-count orbit and finite feedback',
                   controller_qualified=False, development_episodes=0, reserved_episodes=0,
                   CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall)
    write(args.out/'receipt.json', receipt)
    print(json.dumps(receipt, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
