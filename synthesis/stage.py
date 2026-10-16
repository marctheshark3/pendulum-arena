"""Rerun an original design stage in a new directory; no release/qualification runner."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from regenerate import ROOT, arrays, comparisons, digest, fresh, load_count, verify, write_json


NAMES = ('15-reference', '15-first-native-pass', '15-physical-reference', '15-prototype',
         '16-lift', '16-base-reference', '16-orbit', '16-launch', '16-bridge', '16-probe07', '16-parking')


def relocated(value):
    if isinstance(value, str):
        return value.replace('${SYNTHESIS}', str(ROOT))
    if isinstance(value, list):
        return [relocated(v) for v in value]
    if isinstance(value, dict):
        return {key: relocated(v) for key, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=NAMES)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--episode', type=Path)
    parser.add_argument('--replay-native', action='store_true',
                        help='For 15-physical-reference only: generate prefix states by native replay of the recorded controls')
    args = parser.parse_args()
    if args.replay_native and args.stage != '15-physical-reference':
        parser.error('--replay-native applies only to 15-physical-reference')
    verify()
    count = int(args.stage[:2])
    load_count(count)
    import numpy as np
    out = fresh(args.out)
    data = ROOT/f'data/{count}'
    product = out/'product'
    log = out/'stage.log'
    started = time.monotonic()
    commands = []

    def run(script, *arguments):
        command = [sys.executable, str(ROOT/f'original/{count}/scripts/{script}.py'), *map(str, arguments)]
        environment = dict(os.environ, PYTHONPATH=str(ROOT/f'original/{count}/src'))
        commands.append(command)
        with log.open('a') as output:
            output.write(json.dumps(dict(command=command))+'\n'); output.flush()
            completed = subprocess.run(command, env=environment, cwd=out, stdout=output,
                                       stderr=subprocess.STDOUT, timeout=420)
        if completed.returncode:
            raise RuntimeError(f'original stage exited {completed.returncode}; retained log: {log}')

    def need_model():
        if args.model is None:
            raise ValueError('--model must name the rebuilt count-specific planning directory')
        return args.model.resolve()

    comparison = None
    if args.stage == '15-reference':
        if args.inputs is None:
            raise ValueError('--inputs is the fetch-inputs directory')
        run('repair_canonical_reference', '--model', need_model(), '--warm',
            args.inputs.resolve()/'actual_minimax.npz', '--out', product,
            '--ticks', 650, '--cpu-seconds', 180, '--iterations', 150)
        comparison = comparisons(arrays(product/'reference.npz'), arrays(data/'repaired-reference.npz'))
    elif args.stage == '15-first-native-pass':
        reference = args.reference.resolve() if args.reference else data/'repaired-reference.npz'
        run('validate_canonical_candidate', '--reference', reference, '--model', need_model(),
            '--design', data/'design.npz', '--out', product, '--seed', 3156000, '--episodes', 1,
            '--exact', '--tracking-cost', 1000, '--native-bias', '--native-value',
            '--riccati-precision', 'double')
        comparison = comparisons(arrays(product/'episode-3156000.npz'), arrays(data/'first-native-pass.npz'))
    elif args.stage == '15-physical-reference':
        episode = args.episode.resolve() if args.episode else data/'first-native-pass.npz'
        tape = arrays(episode)
        if len(tape['actions']) != 1500:
            raise ValueError('a complete actual native episode is required')
        product.mkdir()
        states = tape['states'][:701].copy()
        if args.replay_native:
            from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol, hanging_state
            protocol = CanonicalProtocol(n_links=15)
            env = CanonicalEnv(protocol)
            state = env.reset(hanging_state(protocol, 0, exact=True))
            generated = [state]
            for action in tape['actions'][:700]:
                state, _ = env.step(action)
                generated.append(state)
            states = np.asarray(generated)
        value = dict(states=states, actions=tape['actions'][:700].copy())
        np.savez_compressed(product/'reference.npz', **value)
        comparison = comparisons(value, arrays(data/'physical-reference.npz'))
        write_json(product/'receipt.json', dict(source_sha256=digest(episode), prefix_ticks=700,
            native_replay=args.replay_native,
            scope=('Native replay of the retained original command prefix from exact hanging; not a new closed-loop controller result'
                   if args.replay_native else 'Explicit original manual extraction stage; recorded actual states, not optimizer states')))
    elif args.stage == '15-prototype':
        reference = args.reference.resolve() if args.reference else data/'physical-reference.npz'
        run('validate_canonical_candidate', '--reference', reference, '--model', need_model(),
            '--design', data/'design.npz', '--out', out/'prototype', '--seed', 3156000,
            '--episodes', 1, '--exact', '--park-seconds', 16, '--cart-target', 0,
            '--tracking-cost', 1000, '--native-bias', '--native-value', '--lookahead', 1,
            '--optimizer', 'bounded', '--parking-decay', 1.2, '--balance-after', 1500,
            '--riccati-precision', 'double', '--joint-lookahead', 2)
        run('prepare_canonical_controller', '--prototype', out/'prototype', '--design', data/'design.npz',
            '--out', product, '--bounded-planning')
        comparison = comparisons(arrays(product/'controller.npz'), arrays(data/'released-controller.npz'))
    elif args.stage == '16-lift':
        if args.inputs is None:
            raise ValueError('--inputs is the fetch-inputs directory')
        from rage_mill.canonical16_trajectory import DONOR14_SHA256, lift_material
        donor = args.inputs.resolve()/'fourteen_link_controller.json'
        if digest(donor) != DONOR14_SHA256:
            raise ValueError('fourteen donor differs from pinned source')
        value = json.loads(donor.read_text())
        states = np.asarray(value['search']['nominal_coordinate_states'], dtype=np.float64)
        actions = np.asarray(value['controller']['controls'], dtype=np.float64)
        mirror = -1 if states[0, 1] < 0 else 1
        result = dict(states=lift_material(mirror*states, 16), actions=mirror*actions)
        product.mkdir(); np.savez_compressed(product/'lifted14.npz', **result)
        comparison = comparisons(result, arrays(data/'lifted14.npz'))
        write_json(product/'receipt.json', dict(donor_sha256=DONOR14_SHA256, role='offline initializer only'))
    elif args.stage == '16-base-reference':
        reference = args.reference.resolve() if args.reference else data/'lifted14.npz'
        run('repair_canonical16_reference_v2', '--model', need_model(), '--warm', reference,
            '--out', product, '--ticks', 700, '--route-ticks', 450, '--cpu-seconds', 180, '--iterations', 150)
        comparison = comparisons(arrays(product/'reference.npz'), arrays(data/'base-reference.npz'))
    elif args.stage == '16-orbit':
        run('probe_canonical16_dynamic_orbit', '--model', need_model(), '--design', data/'design.npz', '--out', product)
        comparison = comparisons(arrays(product/'frequency-5/orbit.npz'), arrays(data/'orbit5.npz'))
    elif args.stage == '16-launch':
        from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol, hanging_state
        protocol = CanonicalProtocol(n_links=16)
        env = CanonicalEnv(protocol)
        state = env.reset(hanging_state(protocol, 0, exact=True))
        target = np.zeros(34); target[0], target[1] = -.05, np.pi
        gain = arrays(data/'parking-design.npz')['gain']
        states, actions, raw = [state], [], []
        for tick in range(800):
            command = -float(gain@(state-target))
            state, delivered = env.step(command)
            states.append(state); actions.append(delivered); raw.append(command)
        result = dict(states=np.asarray(states), actions=np.asarray(actions), raw_actions=np.asarray(raw))
        product.mkdir(); np.savez_compressed(product/'launch.npz', **result)
        original = arrays(data/'original-launch.npz')
        comparison = comparisons(result, dict(states=original['states'][:801],
            actions=original['actions'][:800], raw_actions=original['raw_actions'][:800]))
        write_json(product/'receipt.json', dict(exact_hanging=True, commands=800,
            scope='Reconstructed actual postparking input to bridge synthesis, not a full task episode'))
    elif args.stage == '16-bridge':
        # Relocation views of retained historical component evidence. These
        # preserve the original owner outcomes; they are not new owner receipts.
        views = out/'historical-input-views'
        orbit, launch = views/'orbit', views/'launch'
        for directory in (orbit, launch):
            (directory/'probe').mkdir(parents=True)
        orbit_meta = relocated(json.loads((ROOT/'evidence/16/fast-bent-orbit-discriminator-prepared01--probe--receipt.json').read_text()))
        launch_meta = relocated(json.loads((ROOT/'evidence/16/controller-diagnostic-prepared03--probe--receipt.json').read_text()))
        launch_meta['cases'][0]['recording'] = 'original-launch.npz'
        shutil.copyfile(data/'original-launch.npz', launch/'probe/original-launch.npz')
        write_json(orbit/'probe/receipt.json', orbit_meta)
        write_json(launch/'probe/receipt.json', launch_meta)
        for directory, name in ((orbit, 'fast-bent-orbit-discriminator-prepared01'),
                                (launch, 'controller-diagnostic-prepared03')):
            shutil.copyfile(ROOT/f'evidence/16/{name}--completion.json', directory/'completion.json')
        reference = args.reference.resolve() if args.reference else data/'base-reference.npz'
        run('repair_canonical16_early_orbit_bridge', '--model', need_model(), '--design', data/'design.npz',
            '--warm', reference, '--orbit-owner', orbit, '--launch-owner', launch, '--out', product)
        comparison = comparisons(arrays(product/'reference.npz'), arrays(data/'early-reference.npz'))
    elif args.stage == '16-probe07':
        reference = args.reference.resolve() if args.reference else data/'early-reference.npz'
        run('probe_canonical16_finite_controller', '--reference', reference, '--model', need_model(),
            '--design', data/'design.npz', '--out', product, '--seed', 3167106, '--episodes', 1,
            '--exact', '--park-seconds', 16, '--parking-design', data/'parking-design.npz',
            '--expected-parking-sha256', digest(data/'parking-design.npz'), '--parking-target-cart', -.05,
            '--native-bias', '--dot', 'float64', '--terminal-policy', 'zero', '--post-route', 'zero')
        comparison = comparisons(arrays(product/'controller.npz'), arrays(data/'probe07-controller.npz'))
    elif args.stage == '16-parking':
        run('probe_canonical16_parking', '--out', out/'native-hanging')
        run('probe_canonical16_settling_costs', '--saved', out/'native-hanging', '--out', product)
        comparison = comparisons(arrays(product/'angles10-design.npz'), arrays(data/'parking-design.npz'))
    receipt = dict(stage=args.stage, checkpoint_comparison=comparison, commands=commands,
        public_adapter_wall_seconds=time.monotonic()-started, reserved_episodes=0,
        scope='New design/diagnostic reproduction; no historical outcome or once-only label is replaced',
        original_receipt=str(product/'receipt.json') if (product/'receipt.json').exists() else None)
    write_json(out/'adapter.json', receipt)
    print(json.dumps(receipt, allow_nan=False))


if __name__ == '__main__':
    main()
