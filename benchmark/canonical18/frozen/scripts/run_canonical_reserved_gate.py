"""Execute one reserved gate under a separately reviewed pinned authority."""
import argparse
import json

from rage_mill.canonical_evaluation import pinned_file, require
from rage_mill.canonical_reserved import GATE_ORDER, run_reserved_gate


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--gate", choices=GATE_ORDER, required=True)
    args = parser.parse_args()
    authority = pinned_file(args.authority)
    require(authority["sha256"] == args.expected_authority_sha256, "authority hash differs")
    print(json.dumps(run_reserved_gate(authority, args.gate), allow_nan=False))


if __name__ == "__main__":
    main()
