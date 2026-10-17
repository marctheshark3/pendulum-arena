"""Six saved-launch suffixes testing native affine-value action ranking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import casadi as ca
import numpy as np

from rage_mill.canonical17_controller import Canonical17FiniteTrajectoryPolicy
from rage_mill.canonical17_native_value import Canonical17NativeValueSuffix
from rage_mill.canonical16_affine_feedback import full_affine_feedback
from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol, score_tape
from synthesize_canonical17 import load, sha, write


def ledger_arrays(ledger):
    count = len(ledger)
    states, residuals = np.full((count, 36), np.nan), np.full((count, 36), np.nan)
    for index, row in enumerate(ledger):
        if 'state' in row:
            states[index] = row['state']
        if 'value_residual' in row:
            residuals[index] = row['value_residual']
    return dict(ticks=np.array([row['tick'] for row in ledger], dtype=np.int64),
                commands=np.array([row['command'] for row in ledger], dtype=np.float32),
                costs=np.array([np.nan if row['cost'] is None else row['cost'] for row in ledger]),
                states=states, value_residuals=residuals,
                errors=np.asarray([row['error'] or '' for row in ledger], dtype='U512'))


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    for name in ('model', 'reference', 'controller', 'saved', 'offset-source', 'out'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cpu, wall = time.process_time(), time.monotonic()
    reference, tables, design = load(args.reference), load(args.controller), load(args.model/'design.npz')
    saved = load(args.saved)
    initial = saved['states'][850].copy()
    first = ca.Function.load(str(args.model/'first.casadi'))
    feedback = full_affine_feedback(reference['states'], reference['actions'], first, np.zeros((36, 36)),
                                    design['Q'], control_cost=1000., defects=reference['native_defects'])
    gain_bytes = feedback.gains[:650].tobytes() == tables['gains'].tobytes()
    offset_bytes = feedback.offsets[:650].tobytes() == tables['offsets'].tobytes()
    if not gain_bytes or not offset_bytes:
        raise ValueError('full affine API does not reproduce original strong650 gains/offsets literally')
    np.savez_compressed(args.out/'full-affine-feedback.npz', gains=feedback.gains, offsets=feedback.offsets,
                        roots=feedback.roots, residuals=feedback.residuals, Q=design['Q'])
    protocol, env = CanonicalProtocol(n_links=17), CanonicalEnv(CanonicalProtocol(n_links=17))
    verified_native = np.empty((700, 36))
    for tick, action in enumerate(reference['actions']):
        env.reset(reference['states'][tick]); following, _ = env.step(action)
        verified_native[tick] = following-reference['states'][tick+1]
    if verified_native.tobytes() != reference['native_defects'].tobytes():
        raise ValueError('own700 native reference verification differs from source bias bytes')
    np.savez_compressed(args.out/'verified-native-defects.npz', native_defects=verified_native)
    profiles = []
    settings = [(None, 0.), (None, 1e-6)] + [(start, offset) for start in (0, 250) for offset in (0., 1e-6)]
    for profile_index, (activation, offset) in enumerate(settings):
        ordinary = Canonical17FiniteTrajectoryPolicy(
            **{key: tables[key] for key in ('states', 'actions', 'gains', 'offsets', 'balance_gain', 'park_gain', 'parking_target')},
            park_ticks=850)
        ordinary.next_tick = 850
        ordinary.launch_cart_offset = initial[0]-ordinary.states[0, 0]
        if offset != 0.:
            ordinary.launch_cart_offset += offset
        ordinary.launch_branch_offset = np.zeros(17)
        candidate_calls = 0

        def before_forecast():
            nonlocal candidate_calls
            if time.process_time()-cpu > 150. or time.monotonic()-wall > 270.:
                raise InterruptedError('prospective CPU/wall reserve exhausted; retain measured prefix')
            candidate_calls += 1
            if candidate_calls > 650*16:
                raise InterruptedError('registered per-profile candidate ceiling exhausted')

        policy = ordinary if activation is None else Canonical17NativeValueSuffix(
            ordinary, reference['states'], reference['actions'], feedback, active_route_start=activation,
            cart_offset=ordinary.launch_cart_offset, before_forecast=before_forecast)
        if activation is not None and (policy.forecast is env or policy.forecast.data is env.data):
            raise ValueError('candidate forecast must own a different native environment')
        states, actions, raw, original_raw = [env.reset(initial)], [], [], []
        selected_states, selected_ticks, selected_indices, baseline_indices, unique_counts, duplicates, agreements, ratios = [], [], [], [], [], [], [], []
        invalid = None
        for tick in range(850, 1500):
            try:
                command = policy(states[-1].copy(), tick)
                following, delivered = env.step(command)
            except (ValueError, FloatingPointError, InterruptedError) as error:
                invalid = f'{type(error).__name__}: {error}'
                break
            states.append(following); actions.append(delivered); raw.append(command)
            original_raw.append(command if activation is None else policy.last_original_raw)
            decision = None if activation is None else policy.last_decision
            if decision is not None:
                if (decision.predicted_state.dtype != following.dtype or decision.predicted_state.shape != following.shape
                        or decision.predicted_state.tobytes() != following.tobytes()
                        or np.float32(decision.action).tobytes() != np.float32(delivered).tobytes()):
                    raise ValueError('selected native candidate does not match actual shadow step literally')
                selected_states.append(decision.predicted_state); selected_ticks.append(tick-850)
                selected_indices.append(decision.selected_index); baseline_indices.append(decision.baseline_index)
                unique_counts.append(decision.unique_forecasts); duplicates.append(decision.duplicate_proposals)
                agreements.append(policy.last_baseline_delivery_agreement)
                ledger = policy.tracker.ledger
                denominator = ledger[decision.baseline_index]['cost']
                ratios.append(np.nan if denominator is None or denominator == 0. else ledger[decision.selected_index]['cost']/denominator)
            if abs(following[0]) > 3.:
                break
        states, actions, raw = np.asarray(states), np.asarray(actions, dtype=np.float32), np.asarray(raw)
        score = score_tape(states, actions, protocol, invalid_reason=invalid)
        passed = bool(len(actions) == 650 and score['longest_upright_ticks'] >= 250 and invalid is None
                      and all(score[key] for key in ('finite', 'force_ok', 'float32_delivery_ok', 'rail_ok')))
        control = load(args.offset_source/f'profile-{int(offset != 0.)}.npz')
        parity = None
        if activation is None:
            parity = bool(all(control[key].dtype == value.dtype and control[key].shape == value.shape
                              and control[key].tobytes() == value.tobytes()
                              for key, value in (('states', states), ('actions', actions), ('raw_actions', raw))))
            if not parity:
                raise ValueError('ordinary literal zero-change control differs from saved suffix')
        prefix = 0 if activation is None else activation
        prefix_parity = bool(states[:prefix+1].tobytes() == control['states'][:prefix+1].tobytes()
                             and actions[:prefix].tobytes() == control['actions'][:prefix].tobytes()
                             and raw[:prefix].tobytes() == control['raw_actions'][:prefix].tobytes())
        if not prefix_parity:
            raise ValueError('ordinary prefix before activation differs')
        path = args.out/f'profile-{profile_index}.npz'
        np.savez_compressed(path, states=states, actions=actions, raw_actions=raw, ordinary_raw_actions=np.asarray(original_raw),
                            selected_predicted_states=np.asarray(selected_states).reshape(-1, 36), selected_route_ticks=np.asarray(selected_ticks, dtype=np.int64),
                            selected_indices=np.asarray(selected_indices, dtype=np.int64), baseline_indices=np.asarray(baseline_indices, dtype=np.int64),
                            unique_counts=np.asarray(unique_counts, dtype=np.int64), duplicate_counts=np.asarray(duplicates, dtype=np.int64),
                            baseline_delivery_agreement=np.asarray(agreements, dtype=bool), selected_to_baseline_value_ratio=np.asarray(ratios))
        ledger_path = None
        ledger_count = 0
        if activation is not None:
            ledger_path = args.out/f'profile-{profile_index}-candidate-ledger.npz'
            np.savez_compressed(ledger_path, **ledger_arrays(policy.tracker.ledger))
            ledger_count = len(policy.tracker.ledger)
            write(args.out/f'profile-{profile_index}-candidate-errors.json', [
                {'ledger_index': index, 'error': entry['error']}
                for index, entry in enumerate(policy.tracker.ledger) if entry['error'] is not None])
        finite_ratios = np.asarray(ratios)[np.isfinite(ratios)]
        row = dict(profile_index=profile_index, active_route_start=activation, cart_error_offset_m=offset,
                   **score, forecast_pass=passed, ordinary_zero_change_byte_parity=parity,
                   ordinary_before_activation_prefix_byte_parity=prefix_parity,
                   selected_native_state_delivery_pairs=len(selected_ticks), all_selected_native_pairs_byte_exact=True,
                   native_candidate_commands=ledger_count, candidate_counter_calls=candidate_calls,
                   unique_forecasts_per_active_command_max=max(unique_counts, default=0),
                   baseline_delivery_disagreements=sum(not value for value in agreements),
                   baseline_arithmetic='historical tracker reference-dot-offset/cart error(state-reference)-offset; literal base reference-offset-dot/(state-offset)-reference',
                   median_selected_to_internal_baseline_value_ratio=None if not len(finite_ratios) else float(np.median(finite_ratios)),
                   recording_path=str(path), recording_sha256=sha(path),
                   candidate_ledger_path=None if ledger_path is None else str(ledger_path),
                   candidate_ledger_sha256=None if ledger_path is None else sha(ledger_path))
        profiles.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    selected = next((row for row in profiles if row['forecast_pass']), None)
    result = dict(schema='rage_mill.canonical17_native_affine_value_discriminator.v1', n_links=17,
                  diagnostic_only=True, controller_qualified=False, actual_episode_calls=0, reset_RNG_draws=0,
                  development_episodes=0, reserved_episodes=0, profiles=profiles,
                  full700_feedback_prefix_gains_literal_bytes=gain_bytes, full700_feedback_prefix_offsets_literal_bytes=offset_bytes,
                  source_native_bias700_literal_bytes=True, own_reference_verification_commands=700,
                  native_candidate_commands=sum(row['native_candidate_commands'] for row in profiles),
                  ordinary_control_commands=sum(row['ticks'] for row in profiles if row['active_route_start'] is None),
                  selected_step_verification_commands=sum(row['ticks'] for row in profiles if row['active_route_start'] is not None),
                  maximum_native_candidate_commands=41600, maximum_ordinary_control_commands=1300,
                  maximum_selected_step_verification_commands=2600, maximum_total_native_commands_including_reference=46200,
                  selected_profile_index=None if selected is None else selected['profile_index'],
                  historical_negative_context='same historical native-value module failed sixteen local1e-11 cases; this is own17 saved-full-route capture, not inference from lower value',
                  CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall)
    write(args.out/'receipt.json', result)


if __name__ == '__main__':
    main()
