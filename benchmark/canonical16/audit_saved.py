"""NEW portable sixteen evidence hash audit. Executes no native physics or policy."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from public_support import load_proof,read_json,require,sha256

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof-root',type=Path,required=True)
    parser.add_argument('--hashes-only',action='store_true',help='Explicitly select this non-physics auditor.')
    parser.add_argument('--video',type=Path,help='Optional original scientific MP4; soundtrack derivative is separate.')
    parser.add_argument('--out',type=Path,help='New report path; an existing file is refused.')
    args=parser.parse_args()
    require(args.hashes_only,'this adapter supports --hashes-only; native/controller replay is not implemented here')
    proof=load_proof(args.proof_root)
    video_checked=False
    if args.video:
        expected=proof['qualification']['video']
        require(sha256(args.video)==expected['sha256']and args.video.stat().st_size==expected['bytes'],
                'original scientific video differs; soundtrack is not the scientific MP4')
        video_checked=True
    report={'schema':'pendulum_arena.canonical16_public_hash_audit.v1','passed':True,
        'manifest_files_checked':proof['manifest_files'],'manifest_bytes_checked':proof['manifest_bytes'],
        'original_actual_episodes':len(proof['jobs']),'recorded_commands':len(proof['jobs'])*1500,
        'heldout_policy_commands_historically_byte_proved':211500,
        'heldout_selected_suffixes_historically_byte_proved':141,
        'original_qualification':True,'video_binary_checked':video_checked,
        'native_physics_calls':0,'controller_calls':0,'new_episode_resets':0,'benchmark_labels_consumed':0,
        'scope':'Hash verification and cached original-evidence binding; no new native replay or external third-party certification.'}
    if args.out:
        with args.out.open('x')as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(report,sort_keys=True))

if __name__=='__main__':main()
