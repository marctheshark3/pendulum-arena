"""Prepare, but never execute, one finite exact-copy media assembly owner."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time

from rage_mill.canonical_runtime_manifest import create_runtime_manifest, validate_runtime_manifest


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument('--expected-media-audit-sha256', required=True)
    parser.add_argument('--owner', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cpu, wall = time.process_time(), time.monotonic()
    source = Path(__file__).resolve(strict=True)
    worker = source.with_name('assemble_media.py')
    spec = importlib.util.spec_from_file_location('media_copy_definitions', worker)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.need(args.expected_media_audit_sha256 == m.AUDIT_SHA, 'authorized positive literal audit pin required')
    audit_row = m.pin(m.B / 'media-audit-owner01/result/receipt.json')
    m.need(audit_row['sha256'] == m.AUDIT_SHA, 'actual positive media audit changed')
    audit = m.read(audit_row)
    m.need(audit['passed'] is True and audit['links'] == 18 and audit['seed'] == 218500,
           'positive own18 media audit required')
    m.need(args.owner.parent == m.B and args.owner.resolve() == args.owner and not args.owner.exists()
           and not args.owner.is_symlink() and args.owner.name.startswith('public-media-provenance-owner'),
           'fresh direct evidence metadata owner required')
    m.need(args.out == m.B / 'public-media-provenance01' and not args.out.exists()
           and not args.out.is_symlink() and args.out != args.owner, 'fresh fixed outside-Arena package required')
    paths = [source, worker, Path(sys.executable).resolve(strict=True)]
    supervisor = m.B / 'synthesis-source01/supervise-stage-v2.py'
    m.need(m.pin(supervisor)['sha256'] == m.SUPERVISOR_SHA, 'original finite supervisor required')
    paths.append(supervisor)
    config = Path(sys.prefix) / 'pyvenv.cfg'
    if config.is_file():
        paths.append(config)
    for name in m.OWNERS:
        launch, _ = m.closed_owner(m.B / name)
        paths.extend(Path(path) for path in launch['source_hashes'])
        paths.extend(path for path in (m.B / name).rglob('*') if path.is_file())
    arena = m.B / 'arena18-authoritative01'
    paired = m.B / 'paired-view-source01'
    copies = {
        'canonical18.mp4': m.B / 'scientific-render-owner01/result/episode.mp4',
        'canonical18-telemetry.mp4': m.B / 'telemetry-render-owner02/result/canonical18-telemetry.mp4',
        'canonical18-poster.png': m.B / 'poster-owner01/result/canonical18-poster.png',
        'canonical18-poster.webp': m.B / 'poster-owner01/result/canonical18-poster.webp',
        'canonical18-scientific.render.json': m.B / 'scientific-render-owner01/result/receipt.json',
        'canonical18-scientific.adapter.json': m.B / 'scientific-render-owner01/result/adapter-provenance.json',
        'canonical18-telemetry.render.json': m.B / 'telemetry-render-owner02/result/render-proof/receipt.json',
        'canonical18-telemetry.frames.json': m.B / 'telemetry-render-owner02/result/render-proof/telemetry.json',
        'canonical18-poster.render.json': m.B / 'poster-owner01/result/receipt.json',
        'canonical18-silent-media-audit.json': m.B / 'media-audit-owner01/result/receipt.json',
        'canonical18-scientific.framemd5.txt': m.B / 'media-audit-owner01/result/scientific-framemd5.txt',
        'canonical18-telemetry.framemd5.txt': m.B / 'media-audit-owner01/result/telemetry-framemd5.txt',
        'canonical18-render-scientific-saved-only.py': m.B / 'media-source01/render_scientific_saved_only.py',
        'canonical18-scientific-renderer.py': m.ROOT / 'src/rage_mill/canonical_video.py',
        'canonical18-render-telemetry-video-v2.py': arena / 'scripts/render_telemetry_video_v2.py',
        'canonical18-audit-silent-media.py': m.B / 'media-source01/audit_silent_media.py',
        'canonical18-poster-from-silent.py': m.B / 'media-source02/poster_from_silent.py',
        'canonical18-assemble-media.py': worker,
        'canonical18-prepare-media-assembly.py': source,
        'canonical18-code-LICENSE.txt': m.ROOT / 'LICENSE',
        'canonical18-attribution.md': arena / 'docs/attribution.md',
        'canonical18-queen-eighteen.html': paired / 'queen-eighteen.html',
        'canonical18-queen-paired.js': paired / 'assets/queen-paired.js',
        'canonical18-queen-paired-controller.js': paired / 'assets/queen-paired-controller.js',
        'canonical18-queen-paired.css': paired / 'assets/queen-paired.css',
    }
    source_names = [name for name in copies if name.endswith(('.py', '.html', '.js', '.css', '.md', 'LICENSE.txt'))]
    paired_names = [name for name in source_names if name.startswith('canonical18-queen-')]
    paths.extend(copies.values())
    for path in paths:
        m.need(not Path(path).is_relative_to(args.owner) and not Path(path).is_relative_to(args.out),
               'fresh outputs must be disjoint from original inputs')
    manifest = create_runtime_manifest(package_names=(), extra_paths=paths)
    pins = validate_runtime_manifest(manifest)
    args.owner.mkdir(exist_ok=False)
    runtime = m.write(args.owner / 'runtime-closure.json', manifest)
    pins[runtime['path']] = runtime['sha256']
    inputs = m.write(args.owner / 'inputs.json', dict(
        schema='rage_mill.canonical18_public_media_inputs.v1', media_audit=audit_row,
        output_path=str(args.out), copies={k: str(v) for k, v in copies.items()},
        source_copy_names=source_names, paired_source_copy_names=paired_names,
        runtime_closure=runtime, source_hashes=dict(sorted(pins.items()))))
    pins[inputs['path']] = inputs['sha256']
    recipe = m.write(args.owner / 'launch.json', dict(
        schema='rage_mill.next_stage_supervisor_recipe.v1',
        argv=[str(m.ROOT / '.venv/bin/python'), str(worker), '--inputs', inputs['path'],
              '--expected-inputs-sha256', inputs['sha256'], '--out', str(args.out)],
        cwd=str(args.owner), environment=dict(PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1',
            CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'),
        CPU_cap_s=60, wall_cap_s=120, address_cap_bytes=8 * 1024**3,
        output_path=str(args.out), log=str(args.owner / 'worker.log'),
        receipt=str(args.owner / 'completion.json'), source_hashes=dict(sorted(pins.items())),
        runtime_closure=runtime, purpose='Exact copies of positively audited actual218500 scientific/silent/posters plus telemetry and official Queen pairing provenance',
        maximum_native_commands=0, maximum_controller_calls=0, maximum_reset_RNG_draws=0,
        maximum_media_render_decode_encode_calls=0, new_physical_episodes=0, new_reserved_labels=0,
        qualification_awarded=False))
    preparation = m.write(args.owner / 'preparation.json', dict(
        schema='rage_mill.canonical18_public_media_preparation.v1', metadata_only=True,
        launch=recipe, media_audit=audit_row, runtime_closure=runtime,
        native_calls=0, controller_calls=0, reset_RNG_draws=0, media_calls=0,
        CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall))
    print(json.dumps(dict(launch=recipe, preparation=preparation)), flush=True)


if __name__ == '__main__':
    main()
