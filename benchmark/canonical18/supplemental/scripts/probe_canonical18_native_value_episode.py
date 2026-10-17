"""Fresh actual hanging diagnostics of one declared native-value policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from rage_mill.canonical16_affine_feedback import AffineFeedback
from rage_mill.canonical18_controller import Canonical18FiniteTrajectoryPolicy
from rage_mill.canonical18_native_value_episode import Canonical18NativeValueEpisodePolicy
from rage_mill.canonical_benchmark import CanonicalProtocol, hanging_state, record_episode
from rage_mill.canonical_runtime_closure import runtime_binding_paths
from rage_mill.canonical_runtime_manifest import validate_runtime_manifest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def ledger_arrays(ledger):
    states, residuals = np.full((len(ledger), 38), np.nan), np.full((len(ledger), 38), np.nan)
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
    for name in ('controller', 'reference', 'feedback', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--active-route-start', type=int, choices=(0, 250), required=True)
    parser.add_argument('--cart-error-offset', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--exact', action='store_true')
    parser.add_argument('--runtime-closure', type=Path, required=True)
    args = parser.parse_args()
    if not 3181040 <= args.seed <= args.seed+args.episodes-1 <= 3181199:
        raise ValueError('only prospectively declared3181040..3181199 diagnostic labels permitted')
    runtime_pins = validate_runtime_manifest(json.loads(args.runtime_closure.read_text()))
    args.out.mkdir(parents=True, exist_ok=False)
    cpu, wall = time.process_time(), time.monotonic()
    tables, reference, value = load(args.controller), load(args.reference), load(args.feedback)
    feedback = AffineFeedback(value['gains'], value['offsets'], value['roots'], value['residuals'])
    if (feedback.gains[:650].tobytes() != tables['gains'].tobytes()
            or feedback.offsets[:650].tobytes() != tables['offsets'].tobytes()):
        raise ValueError('literal controller gain/offset prefix required')
    protocol, cases = CanonicalProtocol(n_links=18), []
    for seed in range(args.seed, args.seed+args.episodes):
        ordinary = Canonical18FiniteTrajectoryPolicy(
            **{key: tables[key] for key in ('states', 'actions', 'gains', 'offsets', 'balance_gain', 'park_gain', 'parking_target')},
            park_ticks=850)
        policy = Canonical18NativeValueEpisodePolicy(
            ordinary, reference['states'], reference['actions'], feedback,
            active_route_start=args.active_route_start, cart_error_offset_m=args.cart_error_offset)
        predicted, predicted_delivered, predicted_ticks = [], [], []
        selected_indices, baseline_indices, unique_counts, baseline_agreements, ordinary_raw = [], [], [], [], []

        def observed_controller(state, tick):
            command = policy(state, tick)
            ordinary_raw.append(command if policy.suffix is None else policy.suffix.last_original_raw)
            if policy.last_decision is not None:
                predicted.append(policy.last_decision.predicted_state.copy())
                predicted_delivered.append(policy.last_decision.action)
                predicted_ticks.append(tick)
                selected_indices.append(policy.last_decision.selected_index)
                baseline_indices.append(policy.last_decision.baseline_index)
                unique_counts.append(policy.last_decision.unique_forecasts)
                baseline_agreements.append(policy.suffix.last_baseline_delivery_agreement)
            return command

        states, actions, raw, score = record_episode(
            observed_controller, hanging_state(protocol, seed, exact=args.exact), protocol)
        predicted = np.asarray(predicted).reshape(-1, 38)
        predicted_delivered = np.asarray(predicted_delivered, dtype=np.float32)
        ticks = np.asarray(predicted_ticks, dtype=np.int64)
        actual = states[ticks+1]
        parity = bool(predicted.dtype == actual.dtype == np.dtype('float64') and predicted.shape == actual.shape
                      and predicted.tobytes() == actual.tobytes()
                      and predicted_delivered.shape == actions[ticks].shape
                      and predicted_delivered.tobytes() == actions[ticks].astype(np.float32).tobytes())
        if not parity:
            raise ValueError('selected native predictions differ from actual full-episode states')
        path = args.out/f'episode-{seed}.npz'
        np.savez_compressed(path, states=states, actions=actions, raw_actions=raw,
                            selected_predicted_states=predicted, selected_predicted_actions=predicted_delivered,
                            selected_episode_ticks=ticks, selected_indices=np.asarray(selected_indices, dtype=np.int64),
                            baseline_indices=np.asarray(baseline_indices, dtype=np.int64), unique_counts=np.asarray(unique_counts, dtype=np.int64),
                            baseline_delivery_agreement=np.asarray(baseline_agreements, dtype=bool),
                            ordinary_raw_actions=np.asarray(ordinary_raw))
        ledger = [] if policy.suffix is None else policy.suffix.tracker.ledger
        ledger_path = args.out/f'episode-{seed}-candidate-ledger.npz'
        np.savez_compressed(ledger_path, **ledger_arrays(ledger))
        write(args.out/f'episode-{seed}-candidate-errors.json', [
            {'ledger_index': index, 'error': entry['error']} for index, entry in enumerate(ledger) if entry['error'] is not None])
        row = dict(seed=seed, exact=args.exact, **score, selected_native_states=len(predicted),
                   all_selected_native_states_byte_exact=parity, native_candidate_commands=len(ledger),
                   maximum_candidates_per_active_command=max(unique_counts, default=0),
                   baseline_delivery_disagreements=sum(not value for value in baseline_agreements),
                   candidate_ledger_path=str(ledger_path), candidate_ledger_sha256=sha(ledger_path),
                   native_value_telemetry=policy.telemetry,
                   recording_path=str(path), recording_sha256=sha(path))
        cases.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    result = dict(schema='rage_mill.canonical18_native_value_hanging_diagnostic.v1', n_links=18,
                  diagnostic_only=True, controller_qualified=False, development_episodes=0, reserved_episodes=0,
                  active_route_start=args.active_route_start, cart_error_offset_m=args.cart_error_offset,
                  cases=cases, all_success=all(row['success'] for row in cases), controller_sha256=sha(args.controller),
                  reference_sha256=sha(args.reference), feedback_sha256=sha(args.feedback),
                  CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall)
    actual = {path: sha(path) for path in runtime_binding_paths(
        package_names=(), include_stdlib=False, include_loaded=True)}
    missing = {path: digest for path, digest in actual.items() if runtime_pins.get(path) != digest}
    coverage_path = args.out/'runtime-coverage.json'
    write(coverage_path, dict(runtime_closure=str(args.runtime_closure.resolve()),
        actual_loaded_file_sha256=actual, uncovered_or_changed=missing,
        all_runtime_pins_unchanged=all(sha(path) == digest for path, digest in runtime_pins.items()),
        native_commands=0, reset_draws=0))
    result['runtime_coverage'] = dict(path=str(coverage_path), sha256=sha(coverage_path))
    write(args.out/'receipt.json', result)
    if missing:
        raise ValueError('actual diagnostic runtime files were not prospectively pinned: '+repr(missing))


if __name__ == '__main__':
    main()
