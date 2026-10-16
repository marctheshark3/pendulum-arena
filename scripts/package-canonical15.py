"""Package the curated public benchmark directory without private Git history."""
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
    parser.add_argument("--out", type=Path, default=ROOT / ".artifacts/release/canonical15-source.tar.gz")
    args = parser.parse_args()
    source = ROOT / "benchmark"
    files = sorted(path for path in source.rglob("*") if path.is_file()
                   and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"))
    if any(path.is_symlink() for path in files):
        raise ValueError("source archive refuses symlinks")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in files:
                    data = path.read_bytes()
                    info = tarfile.TarInfo(path.relative_to(ROOT).as_posix())
                    info.size, info.mtime, info.mode = len(data), 0, 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    archive.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(json.dumps(dict(path=str(args.out), sha256=digest, bytes=args.out.stat().st_size, files=len(files)), sort_keys=True))


if __name__ == "__main__":
    main()
