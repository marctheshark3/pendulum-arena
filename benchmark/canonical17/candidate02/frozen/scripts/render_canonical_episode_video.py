"""Render a hash-selected recorded canonical episode, without a simulator."""
from __future__ import annotations

import argparse
from pathlib import Path

from rage_mill.canonical_evaluation import pinned_file, require
from rage_mill.canonical_video import render_saved_episode


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--episode-receipt", required=True, type=Path)
    parser.add_argument("--episode-receipt-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    row = pinned_file(args.episode_receipt)
    require(row["sha256"] == args.episode_receipt_sha256, "selected episode receipt changed")
    render_saved_episode(row, args.out)


if __name__ == "__main__":
    main()
