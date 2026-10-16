"""Read-only ledger, actual tape, native proof, source and video release audit."""
import argparse
import json

from rage_mill.canonical_evaluation import pinned_file, read_pinned, require, write_json_exclusive
from rage_mill.canonical_reserved import verify_reserved_release


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = pinned_file(args.manifest)
    require(manifest["sha256"] == args.expected_manifest_sha256, "release manifest hash differs")
    document = json.loads(read_pinned(manifest))
    result = verify_reserved_release(document["campaign_authority"], document["reserved_cohorts"], document["video"])
    write_json_exclusive(args.out, {**result, "manifest": manifest})


if __name__ == "__main__":
    main()
