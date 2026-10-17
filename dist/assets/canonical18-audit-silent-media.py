"""Independently check saved scientific/silent media, telemetry and frame clocks.

Reads and decodes existing files only. No renderer, encoder, simulation,
controller, reset generator or rage_mill package is imported.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

import numpy as np
from PIL import Image


def require(flag, message):
    if not flag:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    path = Path(path)
    return dict(path=str(path.resolve(strict=True)), sha256=sha(path), bytes=path.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text())


def closed_owner(path):
    path = Path(path); launch, completion = read(path / 'launch.json'), read(path / 'completion.json')
    require(completion['recipe_path'] == str(path / 'launch.json')
            and completion['recipe_sha256'] == sha(path / 'launch.json')
            and completion['naturally_closed_zero'] is True and completion['child_returncode'] == 0
            and completion['owned_group_cleanup'] and completion['recipe_unchanged'] is True
            and completion['failure'] is None and completion['interrupted'] is False
            and completion['wall_timeout'] is False
            and set(launch['source_hashes']) == set(completion['source_hashes_unchanged'])
            and all(flag is True for flag in completion['source_hashes_unchanged'].values())
            and all(sha(source) == digest for source, digest in launch['source_hashes'].items())
            and completion['log_path'] == str(path / 'worker.log')
            and completion['log_sha256'] == sha(path / 'worker.log'), 'original media owner is not closed with unchanged pins')
    return dict(launch=pin(path / 'launch.json'), completion=pin(path / 'completion.json'),
                CPU_seconds=completion['aggregate_child_CPU_seconds'], wall_seconds=completion['wall_seconds'])


def audit(inputs, output):
    count, seed, park = inputs['links'], inputs['seed'], inputs['park_ticks']
    require(type(count) is int and 11 <= count <= 20 and seed == 200000 + count * 1000 + 500
            and type(park) is int and 0 < park < 1500, 'own-count actual video identity required')
    pins = inputs['source_hashes']
    require(all(sha(path) == digest for path, digest in pins.items()), 'saved media audit input changed')
    episode = read(inputs['episode_receipt'])
    require(episode['stage'] == 'reserved' and episode['seed'] == seed and episode['exact'] is False
            and episode['identity']['protocol']['n_links'] == count
            and episode['identity']['policy_parameters']['park_ticks'] == park
            and episode['source_pins_unchanged'] is True and episode['reset_count_after_initialization'] == 0
            and episode['invalid_reason'] is None and episode['score']['success'] is True
            and episode['recording'] == pin(inputs['tape']), 'original successful unchanged actual episode/tape differs')
    with np.load(inputs['tape'], allow_pickle=False) as saved:
        states, actions, raw, times = (saved[key].copy() for key in ('states', 'actions', 'raw_actions', 'times'))
    require(states.shape == (1501, 2 * (count + 1)) and states.dtype == np.dtype('float64')
            and actions.shape == raw.shape == (1500,) and actions.dtype == np.dtype('float32')
            and raw.dtype == np.dtype('float64') and times.shape == (1501,) and times.dtype == np.dtype('float64')
            and all(np.isfinite(value).all() for value in (states, actions, raw, times))
            and np.all(np.abs(states[:, 0]) <= 3.) and np.all(np.abs(actions) <= 1.)
            and np.clip(raw.astype(np.float32), -1., 1.).tobytes() == actions.tobytes()
            and np.max(np.abs(times - np.arange(1501) * .02)) <= 1e-9, 'complete original38-state/F32/raw/native-clock tape required')
    science_owner, telemetry_owner = Path(inputs['scientific_owner']), Path(inputs['telemetry_owner'])
    owners = dict(scientific=closed_owner(science_owner), telemetry=closed_owner(telemetry_owner))
    science, telemetry_dir = science_owner / 'result', telemetry_owner / 'result'
    science_receipt, adapter = read(science / 'receipt.json'), read(science / 'adapter-provenance.json')
    render = read(telemetry_dir / 'render-proof/receipt.json')
    scientific, silent = science / 'episode.mp4', telemetry_dir / f'canonical{count}-telemetry.mp4'
    require(science_receipt['schema'] == 'rage_mill.canonical_saved_state_video.v1'
            and science_receipt['episode_receipt'] == pin(inputs['episode_receipt'])
            and science_receipt['recording'] == episode['recording'] and science_receipt['video'] == pin(scientific)
            and science_receipt['frames'] == 1500 and science_receipt['fps'] == 50
            and science_receipt['simulated_seconds'] == 30. and science_receipt['reset_count'] == 0
            and science_receipt['dynamics_reexecuted'] is False and science_receipt['interpolated_states'] is False
            and adapter['passed'] is True and adapter['reset_identity_reconstructions'] == 0
            and adapter['reset_RNG_draws'] == adapter['native_calls'] == adapter['controller_calls'] == 0,
            'scientific video source/count or disclosed zero-RNG adapter differs')
    require(render['schema'] == 'pendulum_arena.telemetry_video.v1' and render['passed'] is True
            and render['n_links'] == count and render['seed'] == seed and render['park_ticks'] == park
            and render['source_receipt']['sha256'] == sha(inputs['episode_receipt'])
            and render['source_recording']['sha256'] == sha(inputs['tape'])
            and render['output']['sha256'] == sha(silent)
            and render['width'] == 1280 and render['height'] == 1080 and render['rendered_frames'] == 1500
            and render['fps'] == 50 and render['audio'] is False and render['initial_frame_included'] is False
            and render['extra_held_frames'] == 0 and render['physics_steps_during_render'] == 0
            and render['controller_calls_during_render'] == 0
            and render['camera']['fixed_for_complete_recording'] is True
            and render['geometry'] == dict(link_count=count, serialized_link_length_m=float(format(3 / count, '.9g')),
                capsule_radius_m=.025, cart_pivot_z_m=.07, angle_coordinates='cumulative native relative hinge positions'),
            'standard silent telemetry render source/geometry/cadence differs')
    rows = read(telemetry_dir / 'render-proof/telemetry.json')
    require(len(rows) == 1500, 'all1500 post-command telemetry rows required')
    angles = np.cumsum(states[1:, 1:count + 1], axis=1)
    angles = np.where((angles >= -np.pi) & (angles < np.pi), angles, (angles + np.pi) % (2 * np.pi) - np.pi)
    maxima = np.max(np.abs(angles), axis=1)
    streak = best = 0; first_start = first_earned = None
    for index, maximum in enumerate(maxima):
        post = index + 1
        streak = streak + 1 if maximum < .15 else 0; best = max(best, streak)
        if streak == 250 and first_earned is None:
            first_start, first_earned = post - 249, post
        expected = dict(post_command=post, encoded_frame_index=index, encoded_pts_seconds=index / 50,
            display_time_seconds=post / 50, native_time_seconds=float(times[post]), cart_m=float(states[post, 0]),
            delivered_force_N=float(actions[index]) * 80., max_wrapped_absolute_angle_rad=float(maximum),
            upright_streak_ticks=streak, best_upright_ticks=best, hold_met=best >= 250,
            phase='HANGING SETTLE' if post <= park else 'SWING UP AND BALANCE')
        require(rows[index] == expected, 'independent actual state/action/strict-hold telemetry differs')
    require(best == episode['score']['longest_upright_ticks'] >= 250, 'actual full-episode hold differs')
    inspection = []
    for frame in render['inspection_frames']:
        path = telemetry_dir / 'render-proof' / frame['file']
        require(sha(path) == frame['sha256'], 'inspection PNG changed')
        with Image.open(path) as image:
            require(image.size == (1280, 1080) and image.format == 'PNG', 'inspection PNG dimensions differ')
        inspection.append(pin(path))
    commands = []

    def run(argv):
        start = time.monotonic()
        completed = subprocess.run(argv, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        commands.append(dict(argv=argv, wall_seconds=time.monotonic() - start,
                             stdout_sha256=hashlib.sha256(completed.stdout).hexdigest()))
        return completed.stdout

    streams = {}
    for name, path, size in (('scientific', scientific, (1280, 720)), ('telemetry', silent, (1280, 1080))):
        probe = json.loads(run(['/usr/bin/ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-show_format', '-of', 'json', str(path)]))
        video = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
        require(len(video) == 1 and not any(stream['codec_type'] == 'audio' for stream in probe['streams'])
                and (video[0]['width'], video[0]['height']) == size and video[0]['nb_read_frames'] == '1500'
                and video[0]['r_frame_rate'] == '50/1' and float(video[0]['duration']) == 30., 'actual media stream shape/count/audio differs')
        clocks = json.loads(run(['/usr/bin/ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
            '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(path)]))['frames']
        require(len(clocks) == 1500 and all(abs(float(frame['best_effort_timestamp_time']) - index / 50) <= 1e-9
                for index, frame in enumerate(clocks)), 'actual1500 frame PTS differs from0..29.98s')
        data = run(['/usr/bin/ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-threads', '2', '-i', str(path),
            '-map', '0:v:0', '-an', '-threads', '2', '-f', 'framemd5', '-'])
        require(len([line for line in data.splitlines() if line and not line.startswith(b'#')]) == 1500,
                'full1500 decoded frame hash rows required')
        (output / f'{name}-framemd5.txt').write_bytes(data)
        streams[name] = dict(video=pin(path), frames=1500, fps=50, seconds=30., size=list(size), audio=False,
                             all1500_frame_PTS_verified=True, decoded_frame_hashes=pin(output / f'{name}-framemd5.txt'))
    posters = {}
    if inputs.get('poster_png') is not None:
        rgb = run(['/usr/bin/ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-threads', '2',
            '-i', str(silent), '-map', '0:v:0', '-vf', r'select=eq(n\,1499)', '-frames:v', '1',
            '-pix_fmt', 'rgb24', '-threads', '1', '-f', 'rawvideo', '-'])
        final = np.frombuffer(rgb, dtype=np.uint8).reshape(1080, 1280, 3)
        with Image.open(inputs['poster_png']) as image:
            require(image.size == (1280, 1080) and image.format == 'PNG'
                    and np.array_equal(final, np.asarray(image.convert('RGB'))), 'lossless poster differs from final encoded frame1499')
        posters['lossless_final_frame'] = pin(inputs['poster_png'])
    if inputs.get('poster_webp') is not None:
        with Image.open(inputs['poster_webp']) as image:
            require(image.size == (1280, 1080) and image.format == 'WEBP', 'WebP share poster dimensions differ')
        posters['share_poster'] = pin(inputs['poster_webp'])
    require(not any(name.startswith(('mujoco', 'rage_mill')) for name in sys.modules), 'saved-media auditor must import no plant/controller package')
    require(all(sha(path) == digest for path, digest in pins.items()), 'saved media audit input changed')
    return dict(schema='rage_mill.independent_silent_media_audit.v1', passed=True, links=count, seed=seed,
                actual_episode=pin(inputs['episode_receipt']), actual_tape=pin(inputs['tape']), original_owners=owners,
                telemetry_all1500_rows_exact=True, strict_hold_ticks=best,
                first_hold_start_post_command=first_start, first_hold_earned_post_command=first_earned,
                inspection_PNGs=inspection, streams=streams, commands=commands,
                posters=posters,
                native_calls=0, controller_calls=0, reset_RNG_draws=0, new_physical_episodes=0,
                new_media_render_or_encode_calls=0, qualification_awarded=False,
                scope='Same-author separate saved-media implementation; no third-party reproduction claim',
                limitations='Every telemetry row and encoded frame clock is checked. Pixel-to-state geometry relies on reviewed renderer source plus separate visual inspection of actual PNGs; no second rendering is performed.')


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--expected-inputs-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); cpu, wall = time.process_time(), time.monotonic()
    require(sha(args.inputs) == args.expected_inputs_sha256 and not args.out.exists(), 'prospective saved-media input hash and fresh audit output required')
    inputs = read(args.inputs)
    require(inputs['source_hashes'].get(str(Path(__file__).resolve())) == sha(__file__), 'auditor source must be prospectively pinned')
    args.out.mkdir(parents=True, exist_ok=False)
    result = audit(inputs, args.out)
    loaded = {Path(sys.executable).resolve()}
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).is_file():
            loaded.add(Path(path).resolve())
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[-1].startswith('/') and Path(fields[-1]).is_file():
            loaded.add(Path(fields[-1]).resolve())
    actual = {str(path): sha(path) for path in sorted(loaded)}
    missing = {path: digest for path, digest in actual.items() if inputs['source_hashes'].get(path) != digest}
    require(not missing, 'saved-media actual loaded runtime was not prospectively pinned')
    result.update(actual_loaded_file_sha256=actual, uncovered_or_changed=missing)
    result.update(CPU_seconds=time.process_time() - cpu, wall_seconds=time.monotonic() - wall, producer_source=pin(__file__))
    with (args.out / 'receipt.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')


if __name__ == '__main__':
    main()
