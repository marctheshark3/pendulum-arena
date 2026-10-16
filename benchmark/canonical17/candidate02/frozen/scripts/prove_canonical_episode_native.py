"""Bounded native proof over saved states/actions; no RNG or controller run."""
from __future__ import annotations

import argparse
from pathlib import Path

from rage_mill.canonical_evaluation import pinned_file, require
from rage_mill.canonical_replay import prove_recorded_native_transitions


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--episode-receipt", type=Path, required=True)
    parser.add_argument("--episode-receipt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected = pinned_file(args.episode_receipt)
    require(selected["sha256"] == args.episode_receipt_sha256, "selected saved episode changed")
    prove_recorded_native_transitions(selected, args.out)


if __name__ == "__main__":
    main()
