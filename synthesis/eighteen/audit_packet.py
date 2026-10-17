"""Read-only archive hash audit. No extraction or scientific imports."""
import argparse, hashlib, json, pathlib, tarfile

def audit(path):
    total=0
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers(); names=[m.name for m in members]
        if len(names)!=len(set(names)): raise ValueError('duplicate archive names')
        for member in members:
            relative=pathlib.PurePosixPath(member.name)
            if (not member.isfile() or relative.is_absolute() or '..' in relative.parts
                    or not relative.parts or member.mode & 0o111):
                raise ValueError('only safe nonexecutable regular archive members permitted')
            total+=member.size
            if member.size>64*1024**2 or total>128*1024**2: raise ValueError('archive size cap')
        def read(name):
            stream=archive.extractfile(name)
            if stream is None: raise ValueError('missing regular member')
            return stream.read()
        packet=json.loads(read('packet.json')); manifest=json.loads(read('original-manifest.json'))
        wanted=[r for r in manifest['files'] if r['packet']==packet['packet']]
        expected={'packet.json','original-manifest.json','README.md','LICENSE','CREDITS.md','audit_packet.py'}
        for row in wanted:
            expected.add(row['archive_path']); data=read(row['archive_path'])
            if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
                raise ValueError('original payload hash/size differs: '+row['archive_path'])
        if set(names)!=expected: raise ValueError('unexpected or missing archive members')
        return dict(passed=True,packet=packet['packet'],original_files=len(wanted),
                    archive_members=len(members),uncompressed_bytes=total,extracted=False,
                    native_commands=0,reset_RNG_draws=0,new_episodes=0)

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('archive');args=parser.parse_args()
    print(json.dumps(audit(args.archive),indent=2))
