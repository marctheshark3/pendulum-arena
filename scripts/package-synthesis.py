"""Deterministic standalone synthesis source/data archive; no private history or runtime binaries."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    files = sorted(path for path in (ROOT/'synthesis').rglob('*')
                   if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo'))
    if any(path.is_symlink() for path in files):
        raise ValueError('source archive refuses symlinks')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('xb') as output:
        with gzip.GzipFile(filename='', fileobj=output, mode='wb', mtime=0, compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as archive:
                for path in files:
                    data = path.read_bytes()
                    entry = tarfile.TarInfo(path.relative_to(ROOT).as_posix())
                    entry.size, entry.mtime, entry.mode = len(data), 0, 0o644
                    entry.uid = entry.gid = 0
                    entry.uname = entry.gname = ''
                    archive.addfile(entry, io.BytesIO(data))
    print(json.dumps(dict(path=str(args.out), bytes=args.out.stat().st_size,
        sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(), files=len(files))))


if __name__ == '__main__':
    main()
