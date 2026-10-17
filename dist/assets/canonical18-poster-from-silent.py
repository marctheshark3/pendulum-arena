"""Extract frame1499 from the unchanged qualified silent eighteen MP4.

Standalone saved-media worker: stdlib and explicit ffmpeg/ffprobe only. No
Pillow, font renderer, plant, policy, model or reset-RNG package is imported.
Inter-frame codec decoding may read the preceding compressed video prefix;
the select filter emits exactly the one recorded final frame, with no blending.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

WIDTH, HEIGHT, FRAME = 1280, 1080, 1499
SCHEMA = 'rage_mill.canonical18_saved_poster_inputs.v1'


def need(flag,message):
    if not flag:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def pin(path):
    path=Path(path)
    need(path.is_absolute() and path.resolve(strict=True)==path and not path.is_symlink() and path.is_file(),
         'canonical original absolute regular file required')
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)


def read(row):
    need(pin(row['path'])=={k:row[k] for k in ('path','sha256','bytes')},'pinned input changed')
    return json.loads(Path(row['path']).read_text(),parse_constant=lambda x:need(False,'nonfinite JSON'))


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    return pin(path)


def closed_owner(directory):
    directory=Path(directory)
    need(directory.is_absolute() and directory.resolve(strict=True)==directory and not directory.is_symlink(),
         'original canonical telemetry owner directory required')
    launch_row,done_row,log_row=(pin(directory/name) for name in ('launch.json','completion.json','worker.log'))
    launch,done=read(launch_row),read(done_row)
    need(done.get('schema')=='rage_mill.next_stage_supervised_completion.v1'
         and done.get('recipe_path')==launch_row['path'] and done.get('recipe_sha256')==launch_row['sha256']
         and type(done.get('child_returncode')) is int and done['child_returncode']==0
         and done.get('naturally_closed_zero') is True and done.get('recipe_unchanged') is True
         and done.get('failure') is None and done.get('interrupted') is False and done.get('wall_timeout') is False
         and done.get('owned_group_cleanup')==['group_already_exited']
         and done.get('log_path')==log_row['path'] and done.get('log_sha256')==log_row['sha256'],
         'naturally closed/reaped unchanged telemetry owner required')
    pins,stable=launch.get('source_hashes'),done.get('source_hashes_unchanged')
    need(isinstance(pins,dict) and pins and isinstance(stable,dict) and set(pins)==set(stable)
         and all(x is True for x in stable.values()) and all(pin(p)['sha256']==h for p,h in pins.items()),
         'original telemetry source/runtime/evidence pins changed')
    supervisors=[h for p,h in pins.items() if Path(p).name=='supervise-stage-v2.py']
    need(len(supervisors)==1 and done.get('supervisor_source_sha256')==supervisors[0],
         'original pinned bounded supervisor differs')
    for cap,field in [('CPU_cap_s','CPU_cap_s_per_process'),('wall_cap_s','wall_cap_s'),('address_cap_bytes','address_cap_bytes')]:
        need(type(launch.get(cap)) is int and launch[cap]>0 and done.get(field)==launch[cap],
             'original finite owner resource caps differ')
    return launch,dict(launch=launch_row,completion=done_row,log=log_row,
        CPU_seconds=done['aggregate_child_CPU_seconds'],wall_seconds=done['wall_seconds'])


def telemetry_input(directory,expected_sha):
    launch,owner=closed_owner(directory)
    need(launch.get('mode')=='telemetry' and launch.get('audio') is False
         and launch.get('output_path')==str(Path(directory)/'result'), 'original silent telemetry renderer required')
    episode_row=launch['actual_episode'];episode=read(episode_row)
    qualification_row=launch['qualification_evidence'];qualification=read(qualification_row)
    need(qualification.get('schema')=='rage_mill.canonical18_physics_qualification.v1'
         and qualification.get('passed') is True and qualification.get('controller_qualified') is True
         and qualification.get('n_links')==18 and qualification.get('all141_labels_consumed_once') is True
         and qualification.get('replaced_actual_attempts')==0 and qualification.get('reserved_episodes')==141
         and qualification.get('reserved_commands')==211500
         and qualification.get('freeze')==episode.get('freeze')
         and read(qualification['identity'])==episode.get('identity'),
         'same original all141 physics qualification required before poster')
    video_cohort=read(qualification['reserved_gates']['video']['actual_cohort'])
    need(video_cohort.get('episode_receipts')==[episode_row] and episode.get('stage')=='reserved'
         and episode.get('seed')==218500 and episode.get('exact') is False
         and episode.get('identity',{}).get('protocol',{}).get('n_links')==18
         and episode.get('source_pins_unchanged') is True and episode.get('reset_count_after_initialization')==0
         and episode.get('invalid_reason') is None and episode.get('score',{}).get('success') is True
         and episode.get('score',{}).get('ticks')==1500, 'actual complete qualified218500 required')
    video=pin(Path(directory)/'result/canonical18-telemetry.mp4')
    proof_row=pin(Path(directory)/'result/render-proof/receipt.json');proof=read(proof_row)
    need(video['sha256']==expected_sha and proof.get('schema')=='pendulum_arena.telemetry_video.v1'
         and proof.get('passed') is True and proof.get('n_links')==18 and proof.get('seed')==218500
         and proof.get('output',{}).get('sha256')==video['sha256']
         and proof.get('source_receipt',{}).get('sha256')==episode_row['sha256']
         and proof.get('source_recording',{}).get('sha256')==episode['recording']['sha256']
         and proof.get('width')==WIDTH and proof.get('height')==HEIGHT
         and proof.get('rendered_frames')==1500 and proof.get('fps')==50 and proof.get('audio') is False
         and proof.get('initial_frame_included') is False and proof.get('extra_held_frames')==0
         and proof.get('physics_steps_during_render')==proof.get('controller_calls_during_render')==0,
         'unchanged actual1500-frame50fps1280x1080 silent telemetry proof required')
    return launch,dict(owner=owner,video=video,render_proof=proof_row,episode=episode_row,qualification=qualification_row)


def validate_probe(probe):
    videos=[s for s in probe.get('streams',[]) if s.get('codec_type')=='video']
    need(len(videos)==1 and not any(s.get('codec_type')=='audio' for s in probe.get('streams',[])),
         'one silent video stream required')
    stream=videos[0]
    need(stream.get('width')==WIDTH and stream.get('height')==HEIGHT
         and stream.get('nb_frames')=='1500' and stream.get('r_frame_rate')==stream.get('avg_frame_rate')=='50/1',
         'actual1500-frame50fps1280x1080 stream required')
    for value in [stream.get('duration'),probe.get('format',{}).get('duration')]:
        duration=float(value);need(math.isfinite(duration) and abs(duration-30.)<=1e-6,'actual30s duration required')


def decode_argv(ffmpeg,video):
    return [ffmpeg,'-nostdin','-hide_banner','-loglevel','error','-threads','2','-i',video,
        '-map','0:v:0','-vf',r'select=eq(n\,1499)','-frames:v','1','-fps_mode','vfr',
        '-an','-pix_fmt','rgb24','-f','rawvideo','pipe:1']


def encode_argv(ffmpeg,output,kind):
    need(kind in ('png','webp'),'literal poster codec required')
    args=[ffmpeg,'-nostdin','-hide_banner','-loglevel','error','-threads','2','-f','rawvideo',
          '-pixel_format','rgb24','-video_size','1280x1080','-framerate','1','-i','pipe:0','-frames:v','1','-an']
    args+=['-c:v','png','-compression_level','9','-update','1'] if kind=='png' else [
        '-c:v','libwebp','-lossless','1','-compression_level','6','-pix_fmt','bgra']
    return args+['-threads','2','-n',output]


def actual_runtime():
    paths={str(Path(sys.executable).resolve())}
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).is_file():paths.add(str(Path(path).resolve()))
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts=line.split()
        if len(parts)>=6 and parts[-1].startswith('/') and Path(parts[-1]).is_file():
            paths.add(str(Path(parts[-1]).resolve()))
    return {p:sha(p) for p in sorted(paths)}


def run(args):
    cpu,wall=time.process_time(),time.monotonic()
    input_row=pin(args.inputs);need(input_row['sha256']==args.expected_inputs_sha256,'prospective inputs hash required')
    inputs=read(input_row);need(inputs.get('schema')==SCHEMA,'own18 poster inputs required')
    need(str(args.out)==inputs['output_path'] and args.out.is_absolute() and args.out.resolve()==args.out
         and not args.out.exists(),'prospectively bound fresh poster output required')
    pins=inputs['source_hashes'];need(all(pin(p)['sha256']==h for p,h in pins.items()),'prospective poster inputs changed')
    _,original=telemetry_input(Path(inputs['telemetry_owner']),inputs['video']['sha256'])
    need(original==inputs['original'],'closed original telemetry input changed')
    runtime=read(inputs['runtime_closure']);runtime_pins={r['path']:r['sha256'] for r in runtime['files']}
    need(all(pin(r['path'])==r for r in runtime['files']),'poster prospective runtime changed')
    ffmpeg,ffprobe=(inputs['tools'][name]['path'] for name in ('ffmpeg','ffprobe'))
    commands=[]
    def execute(argv,stdin=None):
        start=time.monotonic();result=subprocess.run(argv,input=stdin,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            check=True,timeout=120)
        commands.append(dict(argv=argv,wall_seconds=time.monotonic()-start,
            stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),stderr_sha256=hashlib.sha256(result.stderr).hexdigest()))
        return result.stdout
    probe=json.loads(execute([ffprobe,'-v','error','-show_streams','-show_format','-of','json',inputs['video']['path']]))
    validate_probe(probe)
    rgb=execute(decode_argv(ffmpeg,inputs['video']['path']))
    need(len(rgb)==WIDTH*HEIGHT*3,'exactly one final RGB frame required')
    args.out.mkdir(exist_ok=False)
    posters={}
    for kind in ('png','webp'):
        path=args.out/('canonical18-poster.'+kind)
        execute(encode_argv(ffmpeg,str(path),kind),rgb)
        decoded=execute([ffmpeg,'-nostdin','-hide_banner','-loglevel','error','-threads','2','-i',str(path),
            '-frames:v','1','-an','-pix_fmt','rgb24','-f','rawvideo','pipe:1'])
        need(decoded==rgb,'lossless poster RGB differs from actual final frame1499')
        posters[kind]=pin(path)
    loaded=actual_runtime();missing={p:h for p,h in loaded.items() if runtime_pins.get(p)!=h}
    need(not missing and all(pin(p)['sha256']==h for p,h in pins.items()),'poster runtime coverage/input stability failed')
    need(not any(name.startswith(('mujoco','rage_mill','PIL','numpy')) for name in sys.modules),
         'standalone poster worker imports no scientific/image renderer package')
    coverage=write(args.out/'runtime-coverage.json',dict(runtime_closure=inputs['runtime_closure'],
        actual_loaded_file_sha256=loaded,uncovered_or_changed=missing,all_runtime_pins_unchanged=True))
    receipt=write(args.out/'receipt.json',dict(schema='rage_mill.canonical18_actual_silent_frame_poster.v1',passed=True,
        original=original,source_video=inputs['video'],frame_index=1499,encoded_pts_seconds=29.98,
        actual_post_command=1500,width=WIDTH,height=HEIGHT,emitted_source_video_frames=1,
        selected_RGB_sha256=hashlib.sha256(rgb).hexdigest(),PNG_and_WebP_lossless_RGB_byte_exact=True,
        posters=posters,commands=commands,runtime_coverage=coverage,prospective_inputs=input_row,
        producer_source=pin(Path(__file__).resolve()),frame_interpolation=False,new_image_compositing=False,
        font_processing=False,Pillow_used=False,native_calls=0,controller_calls=0,reset_RNG_draws=0,
        new_physical_episodes=0,new_reserved_labels=0,qualification_awarded=False,
        source_video_modified=False,codec_note='H264 decoding may read the compressed prefix; select emits only final frame1499.',
        CPU_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall))
    print(json.dumps(receipt),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__,allow_abbrev=False)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--expected-inputs-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True);run(parser.parse_args())


if __name__=='__main__':main()
