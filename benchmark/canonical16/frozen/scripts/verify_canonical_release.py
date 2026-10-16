"""Verify saved canonical release tapes and video; never evaluate a controller."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rage_mill.canonical_evaluation import write_json_exclusive
from rage_mill.canonical_release import verify_release


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = verify_release(manifest["freeze"], manifest["reserved_cohorts"], manifest["video"])
    write_json_exclusive(args.out, result)


if __name__ == "__main__":
    main()
