"""Eight fixed temporal phases of the retained stronger own-count orbit.

The 96-state archive is preserved: its actions repeat every eight commands,
but its solved states differ slightly across repeats. No orbit is refined,
compressed or replaced. This is classical local capture, not hanging success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import casadi as ca
import numpy as np

from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol, absolute_angles, score_tape
from rage_mill.canonical16_trajectory import trajectory_feedback


def pin(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def read(row):
    assert pin(row['path']) == row, 'prospective input changed'
    return Path(row['path']).read_bytes()


def load(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    return pin(path)


def record(controller, initial, protocol, out):
    """Literal source record_episode arithmetic, with an additive flushed log."""
    env = CanonicalEnv(protocol)
    states, actions, raw_actions = [env.reset(initial)], [], []
    times, invalid, attempts = [float(env.data.time)], None, 0
    with (out / 'journal.jsonl').open('x') as journal:
        journal.write(json.dumps(dict(initial=states[0].tolist(), time=times[0])) + '\n'); journal.flush()
        for tick in range(protocol.episode_ticks):
            try:
                raw = float(controller(states[-1].copy(), tick))
                attempts += 1
                state, delivered = env.step(raw)
            except (ValueError, FloatingPointError) as error:
                invalid = str(error)
                journal.write(json.dumps(dict(invalid_reason=invalid, completed_commands=len(actions))) + '\n'); journal.flush()
                break
            raw_actions.append(raw); actions.append(delivered); states.append(state)
            times.append(float(env.data.time))
            journal.write(json.dumps(dict(tick=tick, raw_action=raw, action=delivered,
                                           state=state.tolist(), time=times[-1])) + '\n'); journal.flush()
            if abs(state[0]) > protocol.rail_limit_m:
                break
    states, actions, raw_actions = np.asarray(states), np.asarray(actions), np.asarray(raw_actions)
    with (out / 'tape.npz').open('xb') as stream:
        np.savez_compressed(stream, states=states, actions=actions, raw_actions=raw_actions,
                            delivered_actions_f32=actions.astype(np.float32), times=np.asarray(times))
    return states, actions, raw_actions, score_tape(states, actions, protocol, invalid_reason=invalid), attempts


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    declaration = json.loads(args.declaration.read_text())
    for row in declaration['inputs'].values():
        read(row)
    args.out.mkdir(parents=True, exist_ok=False)
    cpu, wall = time.process_time(), time.monotonic()
    archive = load(declaration['inputs']['archive']['path'])
    design = load(declaration['inputs']['design']['path'])
    direction = load(declaration['inputs']['saved_direction']['path'])['direction']
    first = ca.Function.load(declaration['inputs']['first']['path'])
    cyclic_s, cyclic_u, cyclic_defects = archive['states'][:-1], archive['actions'], archive['native_defects']
    assert cyclic_s.shape == cyclic_defects.shape == (96, 36) and cyclic_u.shape == (96,)
    assert archive['states'][0].tobytes() == archive['states'][-1].tobytes()
    assert direction.shape == (36,) and np.isfinite(direction).all()
    assert np.isfinite(cyclic_s).all() and np.isfinite(cyclic_defects).all()
    assert float(np.max(np.abs(cyclic_defects))) <= 1e-10
    blocks, action_blocks = cyclic_s.reshape(12, 8, 36), cyclic_u.reshape(12, 8)
    periodicity = dict(actions_repeat8_bytes=all(value.tobytes() == action_blocks[0].tobytes() for value in action_blocks),
                       states_repeat8_bytes=all(value.tobytes() == blocks[0].tobytes() for value in blocks),
                       maximum_repeated_state_difference=float(np.max(np.abs(blocks - blocks[0]))),
                       maximum_shift8_state_difference=float(np.max(np.abs(cyclic_s - np.roll(cyclic_s, 8, axis=0)))),
                       phase_definition='start indices0..7 of actual96-state cyclic archive, preserved in full')
    assert periodicity['actions_repeat8_bytes']
    write(args.out / 'periodicity.json', periodicity)
    protocol, phases, native_attempts = CanonicalProtocol(n_links=17), [], 0
    for phase in range(8):
        directory = args.out / f'phase-{phase}'; directory.mkdir()
        state_indices = (np.arange(251) + phase) % 96
        action_indices = (np.arange(250) + phase) % 96
        states, actions = cyclic_s[state_indices], cyclic_u[action_indices]
        native = cyclic_defects[action_indices]
        gains, offsets = trajectory_feedback(states, actions, first, np.zeros((36, 36)), design['Q'],
                                             control_cost=1000., defects=native)
        zero_tables = None
        if phase == 0:
            zero_tables = (gains.tobytes() == archive['gains'].tobytes()
                           and offsets.tobytes() == archive['offsets'].tobytes())
            assert zero_tables, 'phase0 recomputed arithmetic differs from source; stop before physical calls'
        with (directory / 'tables.npz').open('xb') as stream:
            np.savez_compressed(stream, states=states, actions=actions, gains=gains, offsets=offsets,
                                native_defects=native, state_indices=state_indices, action_indices=action_indices,
                                Q=design['Q'], terminal_root=np.zeros((36, 36)))
        cases = []
        for case_index, epsilon in enumerate((1e-15, 1e-11)):
            label = 3171300 + 2 * phase + case_index
            case = directory / f'case-{label}'; case.mkdir()
            write(case / 'claim.json', dict(diagnostic_label=label, phase=phase, epsilon=epsilon,
                  initial_state_source='observed archived cyclic state + fixed saved perturbation vector',
                  fresh_reset_draws=0, hanging_start=False, declaration=pin(args.declaration)))
            initial = states[0] + epsilon * direction
            def controller(state, tick):
                if tick >= 250:
                    return 0.
                return actions[tick] - offsets[tick] - float(gains[tick] @ (state - states[tick]))
            s, u, raw, score, attempts = record(controller, initial, protocol, case)
            native_attempts += attempts
            assert native_attempts <= 24000
            parity = None
            if phase == 0:
                saved = load(declaration['inputs'][f'phase0_case{case_index}']['path'])
                parity = {key: saved[key].dtype == value.dtype and saved[key].shape == value.shape
                          and saved[key].tobytes() == value.tobytes()
                          for key, value in (('states', s), ('actions', u), ('raw_actions', raw))}
                assert all(parity.values()), 'phase0 source tape differs; no novel phase may proceed'
            row = dict(diagnostic_label=label, phase=phase, epsilon=epsilon, **score,
                       first250_hold=bool(len(u) >= 250 and np.all(np.abs(absolute_angles(s[1:251], 17)) < .15)),
                       native_command_attempts=attempts, phase0_saved_byte_parity=parity,
                       recording=pin(case / 'tape.npz'), journal=pin(case / 'journal.jsonl'),
                       component_only=True, hanging_start=False)
            write(case / 'receipt.json', row); cases.append(row)
            print(json.dumps(dict(event='phase_case', **row)), flush=True)
        row = dict(phase=phase, maximum_gain_norm=float(np.max(np.linalg.norm(gains, axis=1))),
                   maximum_affine_offset=float(np.max(np.abs(offsets))), cases=cases,
                   both_first250_hold=all(case['first250_hold'] for case in cases),
                   phase0_tables_byte_identical=zero_tables, tables=pin(directory / 'tables.npz'))
        phases.append(row)
        write(directory / 'receipt.json', row)
    eligible = [row for row in phases if row['both_first250_hold']]
    selected = min(eligible, key=lambda row: (row['maximum_gain_norm'], row['phase'])) if eligible else None
    write(args.out / 'receipt.json', dict(schema='rage_mill.canonical17_orbit_phase_component.v1',
        periodicity=periodicity, phases=phases, selected_phase=None if selected is None else selected['phase'],
        selection_rule='both matched first250 strict angular holds, then minimum maxgain and phaseindex',
        native_command_attempts=native_attempts, maximum_native_command_attempts=24000,
        fresh_reset_draws=0, additional_native_cold_commands=0, reused_archived_native_defects=True,
        CPU_seconds=time.process_time() - cpu, wall_seconds=time.monotonic() - wall,
        full_hanging_episodes=0, qualification_episodes=0, reserved_episodes=0,
        controller_qualified=False, learned_component=False, declaration=pin(args.declaration)))


if __name__ == '__main__':
    main()
