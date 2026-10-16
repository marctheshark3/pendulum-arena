"""Deterministic curated sixteen source archive. Excludes original Git history."""
from pathlib import Path
import argparse,gzip,hashlib,io,json,tarfile
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'.artifacts/release/canonical16-source.tar.gz')
    args=parser.parse_args();source=ROOT/'benchmark/canonical16'
    files=sorted(p for p in source.rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.suffix not in ('.pyc','.pyo'))
    if any(p.is_symlink()for p in files):raise ValueError('source archive refuses symlinks')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('xb')as output:
        with gzip.GzipFile(filename='',mode='wb',fileobj=output,mtime=0,compresslevel=9)as compressed:
            with tarfile.open(fileobj=compressed,mode='w',format=tarfile.PAX_FORMAT)as archive:
                for path in files:
                    data=path.read_bytes();item=tarfile.TarInfo(path.relative_to(ROOT).as_posix())
                    item.size=len(data);item.mtime=0;item.mode=0o644;item.uid=item.gid=0;item.uname=item.gname=''
                    archive.addfile(item,io.BytesIO(data))
    print(json.dumps({'path':str(args.out),'sha256':hashlib.sha256(args.out.read_bytes()).hexdigest(),
                      'bytes':args.out.stat().st_size,'files':len(files)}))
if __name__=='__main__':main()
