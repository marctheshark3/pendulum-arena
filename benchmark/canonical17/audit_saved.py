"""Public hash/lineage inspection of completed own17 evidence; no physics/RNG."""
import argparse,hashlib,json
from pathlib import Path,PurePosixPath


def need(ok,message):
    if not ok:raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__,allow_abbrev=False);p.add_argument('--evidence-root',type=Path,required=True);p.add_argument('--expected-files-sha256',required=True);a=p.parse_args();root=a.evidence_root.resolve(strict=True)
    manifest=root/'files.json';need(digest(manifest)==a.expected_files_sha256,'published evidence manifest differs');data=json.loads(manifest.read_text());need(data['schema']=='pendulum-arena.evidence-files.v1','wrong evidence inventory');seen=set();index={}
    for row in data['files']:
        relative=row['path'];rel=PurePosixPath(relative);need(not rel.is_absolute()and'..'not in rel.parts and rel.as_posix()==relative and relative not in seen,'unsafe/duplicate public path');seen.add(relative);path=root/relative
        need(not path.is_symlink()and path.resolve(strict=True).is_relative_to(root)and path.stat().st_size==row['bytes']and digest(path)==row['sha256'],'evidence bytes differ');index[relative]=row
    paths=list(root.rglob('*'));need(not any(p.is_symlink()for p in paths)and{p.relative_to(root).as_posix()for p in paths if p.is_file()}==seen|{'files.json'},'unlisted/missing/symlink evidence file')
    audit=json.loads((root/'audit.json').read_text());release=json.loads((root/'metadata/verified-release.json').read_text());identity=root/'metadata/identity.json'
    need(audit['passed']and audit['actual_episodes']==262 and audit['recorded_commands']==393000 and audit['heldout_selected_shadow_suffixes_byte_proved']==141 and release['controller_qualified']and release['n_links']==17 and release['reserved_episodes']==141,'complete original own17 evidence required')
    need(digest(identity)=='61993d118c341001ef7973fce77a200f85dd86588b78919bc03d629e09c95268','different candidate identity')
    print(json.dumps(dict(schema='pendulum_arena.canonical17_completed_evidence_hash_audit.v1',passed=True,files=len(seen),actual_episode_metadata=262,full_release_hash_bound=True,native_physics_calls=0,controller_calls=0,reset_RNG_draws=0,new_reserved_labels=0,external_reproduction=False,scope='Hash/lineage inspection of actual supplied evidence; no native rerun or new qualification.')))


if __name__=='__main__':main()
