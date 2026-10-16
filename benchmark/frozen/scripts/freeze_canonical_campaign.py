"""Freeze clean complete source and a passing canonical15 candidate prospectively."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import importlib.metadata
from pathlib import Path
import shutil
import subprocess

from rage_mill.canonical_campaign import (
    candidate_identity, file_sha256, runtime_identity, source_identity, write_json,
)


ROOT = Path(__file__).resolve().parents[1]


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if git("status", "--porcelain"):
        raise ValueError("freeze requires the complete source committed and clean")
    identity = dict(candidate=candidate_identity(args.candidate), source=source_identity(ROOT),
                    runtime=runtime_identity())
    if (args.ledger/"reserved.claim.json").exists():
        raise ValueError("reserved labels have already been claimed; prospective freeze refused")
    development_claim = json.loads((args.ledger/"development.claim.json").read_text())
    if (development_claim["identities"] != identity
            or Path(development_claim["out"]).resolve() != args.development.parent.resolve()):
        raise ValueError("development consumption claim does not match this source and evidence")
    evidence = {}
    for stage, path in (("calibration", args.calibration), ("development", args.development)):
        data = json.loads(path.read_text())
        if (data["stage"] != stage or not data["complete"] or not data["passed"]
                or not data["identity_unchanged"] or data["identities"] != identity):
            raise ValueError(f"{stage} did not pass with these exact frozen identities")
        evidence[stage] = dict(path=str(path.resolve()), sha256=file_sha256(path), cohorts=data["cohorts"])
    expected = dict(dev20=dict(episodes=20, admitted=20, required=20),
                    dev100=dict(episodes=100, admitted=100, required=100))
    if evidence["development"]["cohorts"] != expected:
        raise ValueError("both complete disjoint development cohorts must pass")
    args.out.mkdir(parents=True, exist_ok=False)
    bundle = (args.out/"source.bundle").resolve()
    subprocess.run(["git", "bundle", "create", str(bundle), "HEAD"], cwd=ROOT, check=True)
    clone = (args.out/"source").resolve()
    subprocess.run(["git", "clone", "--no-hardlinks", str(bundle), str(clone)], check=True)
    commit = git("rev-parse", "HEAD")
    if (git("rev-parse", "HEAD", cwd=clone) != commit or git("status", "--porcelain", cwd=clone)
            or source_identity(clone) != identity["source"]):
        raise ValueError("fresh clone does not reproduce clean source identities")
    payload = args.out/"candidate"
    shutil.copytree(args.candidate, payload)
    if candidate_identity(payload) != identity["candidate"]:
        raise ValueError("frozen payload identity mismatch")
    requirements = "".join(f"{name}=={version}\n" for name, version in sorted(
        (distribution.metadata["Name"], distribution.version)
        for distribution in importlib.metadata.distributions()))
    (args.out/"requirements-evaluation.txt").write_text(requirements)
    receipt = dict(schema="rage_mill.canonical15_freeze.v1",
                   frozen_at_utc=datetime.now(timezone.utc).isoformat(), source_commit=commit,
                   source_bundle=str(bundle), source_bundle_sha256=file_sha256(bundle),
                   fresh_clone=str(clone), fresh_clone_clean=True,
                   frozen_candidate=str(payload.resolve()), identities=identity,
                   calibration_receipt=str(args.calibration.resolve()),
                   development_receipt=str(args.development.resolve()), evidence=evidence,
                   requirements_sha256=file_sha256(args.out/"requirements-evaluation.txt"),
                   consumption_ledger=str(args.ledger.resolve()), reserved_claim_absent=True,
                   ready_for_reserved=True, reserved_episodes_consumed_at_freeze=0,
                   controller_qualified=False)
    write_json(args.out/"freeze.json", receipt)
    for path in payload.iterdir():
        if path.is_file():
            path.chmod(0o444)
    bundle.chmod(0o444)
    print(json.dumps(dict(source_commit=commit, freeze=str((args.out/"freeze.json").resolve()),
                          ready_for_reserved=True)), flush=True)


if __name__ == "__main__":
    main()
