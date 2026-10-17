"""Copy already audited actual218500 media and write additive provenance.

Standard library only. No renderer, codec, controller, model, random generator,
network client, or benchmark auditor is imported or executed by this worker.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

B = Path('/home/marctheshark/.t3/artifacts/rage-mill-canonical18-20261016')
ROOT = Path('/home/marctheshark/.t3/worktrees/rage-mill/t3-canonical18-swingup')
AUDIT_SHA = 'b7e73de5b9ac1e8f21864723de6329f09f098d0bba4fc2e32a21fc9223e12f17'
SUPERVISOR_SHA = 'a9f7ed1c727deb37fabcc877204558b82a6bd06e4ceeb306cfd2f2d84dcdc8b6'
OWNERS = ('scientific-render-owner01', 'telemetry-render-owner02', 'poster-owner01', 'media-audit-owner01')


def need(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    path = Path(path)
    need(path.is_absolute() and path.is_file() and not path.is_symlink()
         and path.resolve(strict=True) == path, 'original absolute regular file required')
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def read(row):
    need(pin(row['path']) == {k: row[k] for k in ('path', 'sha256', 'bytes')}, 'pinned bytes changed')
    return json.loads(Path(row['path']).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    return pin(path)


def closed_owner(owner):
    owner = Path(owner)
    need(owner.parent == B and owner.name in OWNERS, 'one original media owner required')
    launch_row, done_row = pin(owner / 'launch.json'), pin(owner / 'completion.json')
    launch, done = read(launch_row), read(done_row)
    need(done['schema'] == 'rage_mill.next_stage_supervised_completion.v1'
         and done['naturally_closed_zero'] is True and type(done['child_returncode']) is int
         and done['child_returncode'] == 0 and done['failure'] is None
         and done['recipe_unchanged'] is True and done['wall_timeout'] is False
         and done['interrupted'] is False and done['owned_group_cleanup'] == ['group_already_exited']
         and done['recipe_sha256'] == launch_row['sha256']
         and done['recipe_path'] == str(owner / 'launch.json')
         and done['log_path'] == str(owner / 'worker.log')
         and done['log_sha256'] == pin(owner / 'worker.log')['sha256']
         and done['supervisor_source_sha256'] == SUPERVISOR_SHA
         and set(done['source_hashes_unchanged']) == set(launch['source_hashes'])
         and all(x is True for x in done['source_hashes_unchanged'].values()),
         'natural zero/reaped/unchanged original media owner required')
    for path, digest in launch['source_hashes'].items():
        need(pin(path)['sha256'] == digest, 'original media owner source/runtime/evidence changed')
    return launch, dict(launch=launch_row, completion=done_row,
                       CPU_seconds=done['aggregate_child_CPU_seconds'], wall_seconds=done['wall_seconds'])


def copy_row(source, output, name):
    need(Path(name).name == name and name not in ('', '.', '..'), 'flat artifact filename required')
    original = pin(source)
    target = output / name
    with target.open('xb') as handle:
        handle.write(Path(source).read_bytes())
        handle.flush()
        os.fsync(handle.fileno())
    copied = pin(target)
    need(copied['sha256'] == original['sha256'] and copied['bytes'] == original['bytes'],
         'exact unchanged copy required')
    return dict(file=name, **original)


def loaded_rows():
    paths = {Path(sys.executable).resolve(strict=True)}
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).is_file():
            paths.add(Path(path).resolve(strict=True))
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[-1].startswith('/') and Path(fields[-1]).is_file():
            paths.add(Path(fields[-1]).resolve(strict=True))
    return {str(path): pin(path)['sha256'] for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--expected-inputs-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cpu, wall = time.process_time(), time.monotonic()
    input_row = pin(args.inputs)
    need(input_row['sha256'] == args.expected_inputs_sha256, 'literal prospective inputs required')
    plan = read(input_row)
    need(plan['schema'] == 'rage_mill.canonical18_public_media_inputs.v1', 'own18 media inputs required')
    output = args.out
    need(output == B / 'public-media-provenance01' and str(output) == plan['output_path']
         and not output.exists() and not output.is_symlink(), 'fresh fixed outside-Arena package required')
    for path, digest in plan['source_hashes'].items():
        need(pin(path)['sha256'] == digest, 'prospective original input/source/runtime changed')
    owners = {name: closed_owner(B / name)[1] for name in OWNERS}
    audit_row = pin(B / 'media-audit-owner01/result/receipt.json')
    need(audit_row['sha256'] == AUDIT_SHA and audit_row == plan['media_audit'], 'positive audit literal pin required')
    audit = read(audit_row)
    need(audit['schema'] == 'rage_mill.independent_silent_media_audit.v1' and audit['passed'] is True
         and audit['links'] == 18 and audit['seed'] == 218500
         and audit['telemetry_all1500_rows_exact'] is True and audit['uncovered_or_changed'] == {}
         and audit['native_calls'] == audit['controller_calls'] == audit['reset_RNG_draws'] == 0
         and audit['new_physical_episodes'] == audit['new_media_render_or_encode_calls'] == 0,
         'closed positive saved-media audit required')
    scientific = read(pin(B / 'scientific-render-owner01/result/receipt.json'))
    adapter = read(pin(B / 'scientific-render-owner01/result/adapter-provenance.json'))
    render = read(pin(B / 'telemetry-render-owner02/result/render-proof/receipt.json'))
    poster = read(pin(B / 'poster-owner01/result/receipt.json'))
    qualification = read(adapter['qualification_evidence'])
    need(qualification['passed'] is True and qualification['controller_qualified'] is True
         and qualification['n_links'] == 18 and qualification['all141_labels_consumed_once'] is True
         and qualification['replaced_actual_attempts'] == 0, 'all141 positive physics evidence required')
    for kind, dimensions in (('scientific', [1280, 720]), ('telemetry', [1280, 1080])):
        stream = audit['streams'][kind]
        read_video = pin(stream['video']['path'])
        need(read_video == stream['video'] and stream['frames'] == 1500 and stream['fps'] == 50
             and stream['seconds'] == 30 and stream['size'] == dimensions and stream['audio'] is False
             and stream['all1500_frame_PTS_verified'] is True, 'actual complete silent stream required')
    need(scientific['video'] == audit['streams']['scientific']['video']
         and render['output']['sha256'] == audit['streams']['telemetry']['video']['sha256']
         and render['n_links'] == 18 and render['seed'] == 218500 and render['park_ticks'] == 850
         and poster['passed'] is True and poster['PNG_and_WebP_lossless_RGB_byte_exact'] is True
         and poster['frame_index'] == 1499 and poster['actual_post_command'] == 1500
         and poster['source_video'] == audit['streams']['telemetry']['video'], 'same actual218500 assets required')
    rows = read(pin(B / 'telemetry-render-owner02/result/render-proof/telemetry.json'))
    need(len(rows) == 1500 and [row['post_command'] for row in rows] == list(range(1, 1501)),
         'all1500 actual telemetry rows required')
    starts = [row['post_command'] for row in rows if row['upright_streak_ticks'] == 1]
    earned = [row['post_command'] for row in rows if row['upright_streak_ticks'] == 250]
    need(earned and audit['first_hold_earned_post_command'] == earned[0]
         and audit['first_hold_start_post_command'] == earned[0] - 249
         and earned[0] - 249 in starts
         and max(row['best_upright_ticks'] for row in rows) == audit['strict_hold_ticks'],
         'audited strict hold indices must agree with copied rows')
    output.mkdir(exist_ok=False)
    assets = {name: copy_row(Path(path), output, name) for name, path in plan['copies'].items()}
    scientific_ref = assets['canonical18.mp4']
    telemetry_ref = assets['canonical18-telemetry.mp4']
    sources = {name: assets[name] for name in plan['source_copy_names']}
    shared = dict(n_links=18, seed=218500, frames=1500, fps=50, duration_seconds=30,
                  native_commands=0, controller_calls=0, reset_RNG_draws=0,
                  new_physical_resets=0, new_simulation_or_controller_inference=False,
                  qualification_awarded_by_renderer=False)
    telemetry = dict(schema='pendulum_arena.telemetry_video.v1', complete=True,
        video='canonical18-telemetry.mp4', video_sha256=telemetry_ref['sha256'], video_bytes=telemetry_ref['bytes'],
        scientific_video='canonical18.mp4', scientific_video_sha256=scientific_ref['sha256'],
        scientific_video_file=scientific_ref,
        source_recording_sha256=audit['actual_tape']['sha256'],
        source_receipt_sha256=audit['actual_episode']['sha256'],
        source_recording=audit['actual_tape'], source_receipt=audit['actual_episode'],
        dimensions=[1280, 1080], state_samples='post-command 1..1500', park_ticks=850,
        render_receipt=assets['canonical18-telemetry.render.json'],
        telemetry_rows=dict(**assets['canonical18-telemetry.frames.json'], rows=1500),
        renderer=assets['canonical18-render-telemetry-video-v2.py'],
        renderer_receipt_label_note='Original render receipt keeps its historical source_renderer.file label; its source hash binds the actual v2 file copied here.',
        source_verification=assets['canonical18-silent-media-audit.json'],
        scientific_render_receipt=assets['canonical18-scientific.render.json'],
        scientific_adapter=assets['canonical18-scientific.adapter.json'],
        source_files=sources, original_owners=owners,
        first_qualifying_hold_start_post_command=audit['first_hold_start_post_command'],
        first_250_hold_earned_post_command=audit['first_hold_earned_post_command'],
        longest_upright_ticks=audit['strict_hold_ticks'],
        claim_scope='Actual recorded held-out simulation presentation. 50 Hz is simulated cadence. No hardware, wall-clock realtime or qualification claim is awarded by rendering.',
        **shared)
    telemetry_pin = write(output / 'canonical18-telemetry.provenance.json', telemetry)
    credits = dict(
        project=dict(author='Marc / Rage Industries', website='https://neuralyogi.com/',
                     source='https://github.com/marctheshark3/rage-mill',
                     arena_source='https://github.com/marctheshark3/pendulum-arena',
                     code_license='MIT', license_file=assets['canonical18-code-LICENSE.txt'],
                     scope='MIT applies to project code. External sources, music and media retain their own rights and licenses.'),
        benchmark=dict(author='Corben Sorenson', source='https://github.com/corbensorenson/cartpole',
                       pinned_source='b84c6413974d5c0831cdf261ce15e58ba65cd3f5',
                       accepted_fourteen_source='3277b4173a9333819bf35acb848e384f52a227ec',
                       published_fourteen='https://github.com/corbensorenson/cartpole/releases/tag/fourteen-link-canonical-20261002',
                       role='Published canonical plant/task, fourteen-link benchmark and donor methodology; eighteen has its own qualification. No endorsement or independent reproduction is implied.'),
        inspiration=dict(author='Kache / @yacineMTB',
                         source='https://x.com/yacineMTB/status/2064148140899348779',
                         profile='https://github.com/yacineMTB',
                         role='Published six-pendulum demonstration and iterative learning/control inspiration. No matched canonical score or endorsement is assigned.'),
        queen=dict(artist='Queen', title="Don't Stop Me Now", official_video='https://www.youtube.com/watch?v=HgzGwKwLmgM',
                   rights='Queen recording, composition and official video retain their respective rightsholders. No license to extract, redistribute, rehost or remix audio is asserted.'))
    pairing = dict(schema='pendulum_arena.official_youtube_paired_view.v1', complete=True,
        n_links=18, seed=218500, page='/queen-eighteen.html',
        requested_wording='super star by queen', confirmed_artist='Queen', confirmed_title="Don't Stop Me Now",
        user_source_preference='youtube is fine',
        source=dict(platform='YouTube', video_id='HgzGwKwLmgM',
                    official_url='https://www.youtube.com/watch?v=HgzGwKwLmgM',
                    embed_url='https://www.youtube.com/embed/HgzGwKwLmgM',
                    delivery='Visible official YouTube player with official controls and branding; streamed separately by YouTube.'),
        playback=dict(user_gesture_required=True, autoplay_before_gesture=False,
                      audio_start_seconds=0, replay_duration_seconds=30,
                      pause_or_replay_end_pauses_Queen=True,
                      synchronization='Best-effort coordination of separate browser players; no sample-accurate alignment claim.',
                      blocked_player_fallback='Official YouTube link and replay-only playback remain available.'),
        edits=dict(audio_extracted=False, audio_downloaded=False, audio_rehosted=False,
                   audio_remixed=False, standalone_audio_derivative_created=False,
                   saved_MP4_modified=False, video_reencoded_for_pairing=False,
                   scientific_and_silent_downloads_preserved=True),
        legal_scope='Embedding is a separate official-player presentation. User preference and provenance do not grant a standalone audio redistribution license.',
        official_player_docs=['https://developers.google.com/youtube/iframe_api_reference',
                              'https://developers.google.com/youtube/player_parameters',
                              'https://developers.google.com/youtube/terms/required-minimum-functionality'],
        telemetry_video=telemetry_ref, scientific_video=scientific_ref,
        telemetry_provenance=dict(file='canonical18-telemetry.provenance.json', **telemetry_pin),
        staged_UI_sources={name: sources[name] for name in plan['paired_source_copy_names']},
        UI_review_scope='These pins identify the reviewed staging implementation; publication/browser QA is recorded separately. A later page change requires an additive provenance addendum.',
        deployment_status=dict(staging_is_donor=True, deployed_UI_source_pins_pending=True,
            paired_playback_browser_QA_pass_claimed=False,
            reported_mobile_presentation_edit='Parent changed only the actual Arena mobile h1 clamp from 58px/17vw/80px to 40px/14vw/70px after a measured 397px heading overflow on a 390px viewport. Original staged CSS remains unchanged. Final deployed CSS and real playback QA belong to a separate additive receipt.'),
        source_files=sources, credits=credits,
        native_commands=0, controller_calls=0, reset_RNG_draws=0, new_physical_resets=0,
        qualification_awarded=False)
    pairing_pin = write(output / 'canonical18-queen-paired-view.provenance.json', pairing)
    review = dict(schema='rage_mill.canonical18_media_review.v1', passed=True, complete=True,
        scope='Exact copies of already audited actual218500 media plus presentation provenance; no new qualification or external reproduction is awarded.',
        media_audit=assets['canonical18-silent-media-audit.json'],
        actual_receipt=audit['actual_episode'], actual_tape=audit['actual_tape'],
        scientific_video=scientific_ref, silent_telemetry_video=telemetry_ref,
        scientific_render_receipt=assets['canonical18-scientific.render.json'],
        scientific_adapter=assets['canonical18-scientific.adapter.json'],
        telemetry_render_receipt=assets['canonical18-telemetry.render.json'],
        telemetry_rows=assets['canonical18-telemetry.frames.json'],
        telemetry_provenance=dict(file='canonical18-telemetry.provenance.json', **telemetry_pin),
        queen_paired_view_provenance=dict(file='canonical18-queen-paired-view.provenance.json', **pairing_pin),
        poster=assets['canonical18-poster.webp'], lossless_final_frame=assets['canonical18-poster.png'],
        poster_provenance=assets['canonical18-poster.render.json'],
        first_hold_start_post_command=audit['first_hold_start_post_command'],
        first_hold_earned_post_command=audit['first_hold_earned_post_command'],
        strict_hold_ticks=audit['strict_hold_ticks'], original_owners=owners,
        source_files=sources, credits=credits, **shared)
    review_pin = write(output / 'canonical18-media-review.json', review)
    coverage = loaded_rows()
    missing = {path: digest for path, digest in coverage.items() if plan['source_hashes'].get(path) != digest}
    need(not missing, 'loaded original runtime source/library not prospectively covered')
    coverage_pin = write(output / 'assembly-runtime-coverage.json', dict(
        schema='rage_mill.canonical18_media_assembly_runtime_coverage.v1', passed=True,
        actual_loaded_file_sha256=coverage, uncovered_or_changed=missing,
        prospective_runtime=plan['runtime_closure']))
    result = write(output / 'assembly-receipt.json', dict(
        schema='rage_mill.canonical18_public_media_assembly.v1', passed=True,
        assembled_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        prospective_inputs=input_row, media_audit=audit_row, original_owners=owners,
        exact_copies=assets, telemetry_provenance=telemetry_pin, queen_pairing_provenance=pairing_pin,
        media_review=review_pin, runtime_coverage=coverage_pin,
        native_calls=0, controller_calls=0, reset_RNG_draws=0, new_physical_episodes=0,
        new_reserved_labels=0, new_render_decode_encode_calls=0, network_calls=0,
        qualification_awarded=False, CPU_seconds=time.process_time()-cpu,
        wall_seconds=time.monotonic()-wall))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
