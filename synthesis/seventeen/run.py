"""Portable access to retained seventeen synthesis; never reserved evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
DATA = ROOT / 'retained'
SOURCE = ROOT / 'original'
STAGES = ('model', 'orbit', 'amplitude', 'launch', 'bridge', 'comparison')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def checked(root, relative):
    path = root / relative
    if (Path(relative).is_absolute() or '..' in Path(relative).parts
            or not path.is_file() or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError('invalid inventory path: ' + relative)
    return path


def verify():
    manifest = json.loads((ROOT / 'original-manifest.json').read_text())
    rows = [(ROOT, row) for row in manifest['files']]
    rows += [(REPO, row) for row in manifest['existing_repository_inputs']]
    for name in ('adapter-manifest.json', 'derivative-checkpoint-manifest.json'):
        path = ROOT / name
        if path.exists():
            rows += [(ROOT, row) for row in json.loads(path.read_text())['files']]
    for base, row in rows:
        path = checked(base, row['path'])
        if digest(path) != row['sha256'] or ('bytes' in row and path.stat().st_size != row['bytes']):
            raise ValueError('inventory mismatch: ' + row['path'])
    return dict(verified_files=len(rows), native_commands=0, reserved_episodes=0)


def load(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def compare(actual, expected):
    import numpy as np
    result = {}
    for name, value in expected.items():
        other = actual.get(name)
        same = other is not None and other.shape == value.shape and other.dtype == value.dtype
        numeric = same and value.size and value.dtype.kind in 'fiu' and np.isfinite(other).all() and np.isfinite(value).all()
        result[name] = dict(array_bytes_equal=bool(same and other.tobytes() == value.tobytes()),
                            maximum_absolute_difference=float(np.max(np.abs(other-value))) if numeric else None)
    return dict(all_array_bytes_equal=all(r['array_bytes_equal'] for r in result.values()), arrays=result)


def feedback(args, out):
    import numpy as np
    from rage_mill.canonical16_affine_feedback import full_affine_feedback
    reference = load(args.reference or DATA / 'bridge/reference.npz')
    if args.model:
        import casadi as ca
        first = ca.Function.load(str(args.model.resolve() / 'first.casadi'))
        aa, bb = map(np.asarray, first.map(700)(reference['states'][:-1].T, reference['actions'].reshape(1, -1)))
        design = load(args.model / 'design.npz')
    else:
        checkpoint = load(ROOT / 'derivative-checkpoint.npz')
        if digest(args.reference or DATA / 'bridge/reference.npz') != checkpoint['reference_sha256'].item():
            raise ValueError('numeric derivative checkpoint belongs to a different reference')
        aa, bb = checkpoint['A'], checkpoint['B']
        design = load(DATA / 'model/design.npz')

    class NumericFirst:
        def map(self, count):
            if count != 700:
                raise ValueError('retain the complete 700-command design')
            return lambda states, actions: (aa, bb)

    result = full_affine_feedback(reference['states'], reference['actions'], NumericFirst(),
                                  np.zeros((36, 36)), design['Q'], defects=reference['native_defects'])
    arrays = {name: getattr(result, name) for name in ('gains', 'offsets', 'roots', 'residuals')}
    np.savez_compressed(out / 'feedback.npz', **arrays)
    np.savez_compressed(out / 'derivatives.npz', A=aa, B=bb,
                        reference_sha256=np.array(digest(args.reference or DATA / 'bridge/reference.npz')))
    expected = load(REPO / 'benchmark/canonical17/candidate02/provenance/original/034-ff430e9ea49e-full-affine-feedback.npz')
    receipt = dict(scope='New full-horizon QR reconstruction, not an episode or qualification',
                   comparison=compare(arrays, {k: expected[k] for k in arrays}),
                   native_commands=0, reserved_episodes=0, model_source='rebuilt provider' if args.model else 'public numeric checkpoint')
    write(out / 'receipt.json', receipt)
    return receipt


def stage(args, out):
    product = out / 'product'
    script = 'synthesize_canonical17.py'
    arguments = [args.stage]
    model = args.model.resolve() if args.model else None
    if args.stage != 'model' and model is None:
        raise ValueError('--model must name the newly rebuilt seventeen model directory')
    donor = REPO / 'synthesis/data/16/early-reference.npz'
    if args.stage == 'model':
        arguments += ['--donor', str(donor)]
    else:
        arguments += ['--model', str(model)]
    if args.stage == 'amplitude':
        script, arguments = 'probe_canonical17_orbit_amplitude.py', arguments[1:]
        arguments += ['--baseline', str((args.baseline or DATA / 'orbit').resolve())]
    elif args.stage == 'bridge':
        script, arguments = 'synthesize_canonical17_strong_orbit.py', arguments[1:]
        orbit = (args.orbit or DATA / 'amplitude').resolve()
        receipt = json.loads((orbit / 'receipt.json').read_text())
        view = out / 'orbit-loading-view'
        view.mkdir()
        for index, row in enumerate(receipt['profiles']):
            path = orbit / f'profile-{index}.npz'
            if digest(path) != row['artifact_sha256']:
                raise ValueError('selected orbit input hash mismatch')
            row['artifact_path'] = str(path)
        write(view / 'receipt.json', receipt)
        arguments += ['--orbit', str(view), '--launch', str((args.launch or DATA / 'parking').resolve()), '--donor', str(donor)]
    elif args.stage == 'comparison':
        script, arguments = 'probe_canonical17_native_value.py', arguments[1:]
        arguments += ['--reference', str((args.reference or DATA / 'bridge/reference.npz').resolve()),
                      '--controller', str((args.controller or DATA / 'bridge/controller.npz').resolve()),
                      '--saved', str(DATA / 'stronger-failure/episode-3171033.npz'),
                      '--offset-source', str(DATA / 'ordinary-offsets')]
    command = [sys.executable, str(SOURCE / 'scripts' / script), *arguments, '--out', str(product)]
    environment = dict(os.environ, PYTHONPATH=str(SOURCE / 'src'))
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)

    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))

    write(out / 'invocation.json', dict(argv=command, CPU_cap_s_per_process=300,
                                       wall_cap_s=480, address_cap_bytes=8 * 1024**3,
                                       role='New public reproduction attempt; retained metadata is never launched'))
    timed_out = False
    with (out / 'worker.log').open('x') as log:
        child = subprocess.Popen(command, env=environment, cwd=out, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=limits)
        try:
            code = child.wait(timeout=480)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = dict(argv=command, returncode=code, wall_timeout=timed_out,
                   wall_seconds=time.monotonic()-started,
                   aggregate_child_CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
                   product=str(product), reserved_episodes=0)
    if (product / 'receipt.json').exists():
        receipt['product_receipt_sha256'] = digest(product / 'receipt.json')
    write(out / 'adapter.json', receipt)
    if code:
        raise RuntimeError(f'worker exited {code}; retained {out / "worker.log"}')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', choices=('verify', 'feedback', *STAGES))
    for name in ('out', 'model', 'orbit', 'launch', 'baseline', 'reference', 'controller'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    verified = verify()
    if args.command == 'verify':
        print(json.dumps(verified))
        return
    if args.out is None:
        parser.error('--out must name a fresh directory')
    if any(os.environ.get(name) != '1' for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')):
        parser.error('set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 before starting Python')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(SOURCE / 'src'))
    if args.command == 'feedback':
        result = feedback(args, out)
    else:
        args.stage = args.command
        result = stage(args, out)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
