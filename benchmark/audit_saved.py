"""New relocation adapter: hash-check or cold-native-audit all 262 original saved episodes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

from public_support import (
    AUDIT_SHA256, FREEZE_SHA256, SOURCE_COMMIT, check_runtime, load_frozen_script,
    load_proof, require_hash, write_new_json,
)


def audit_job(arguments):
    return load_frozen_script("audit_canonical_release").check_case(*arguments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-root", required=True, type=Path, help="extracted proof pack root")
    parser.add_argument("--hashes-only", action="store_true", help="stdlib-only integrity check; no native physics claims")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-runtime-mismatch", action="store_true", help="exploratory only; still requires native MuJoCo 3.3.5")
    parser.add_argument("--video", type=Path, help="optionally verify the separately downloaded silent MP4")
    parser.add_argument("--out", type=Path, help="new JSON report; refuses to overwrite an existing file")
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        parser.error("--workers must be between 1 and 64")
    started = time.monotonic()
    proof = load_proof(args.proof_root)
    if args.video:
        require_hash(args.video, proof["video"]["video_sha256"])
    report = dict(schema="pendulum_arena.canonical15_public_saved_audit.v1", passed=True,
                  mode="hashes_only" if args.hashes_only else "native_cold_transitions",
                  source_commit=SOURCE_COMMIT, original_freeze_sha256=FREEZE_SHA256,
                  original_audit_sha256=AUDIT_SHA256, recording_sha256_checks=len(proof["jobs"]),
                  retained_forecast_sha256_checks=proof["forecast_count"],
                  cohort_receipt_sha256=proof["evidence"], video_binary_checked=bool(args.video),
                  policy_calls=0, creates_new_qualification=False,
                  note="Rechecks historical saved evidence. Does not rerun the controller or replace any original qualification receipt.")
    if not args.hashes_only:
        runtime, differences = check_runtime(proof["freeze"], args.allow_runtime_mismatch)
        candidate = proof["candidate"]
        arguments = [(str(candidate / "model.xml"), str(candidate / "policy.json"), cohort, row)
                     for cohort, row in proof["jobs"]]
        if args.workers == 1:
            rows = [audit_job(argument) for argument in arguments]
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                rows = list(pool.map(audit_job, arguments))
        report.update(frozen_runtime_match=not differences, runtime_differences=differences,
                      runtime=runtime, saved_episodes_checked=len(rows), cold_transition_checks=1500 * len(rows),
                      maximum_native_dynamics_defect=max(row["native_maximum_dynamics_defect"] for row in rows),
                      maximum_initial_state_gap=max(row["initial_state_gap"] for row in rows),
                      minimum_upright_seconds=min(row["longest_upright_seconds"] for row in rows),
                      maximum_cart_m=max(row["maximum_cart_m"] for row in rows),
                      cases=sorted(rows, key=lambda row: (row["gate"], row["seed"])))
    report["wall_seconds"] = time.monotonic() - started
    if args.out:
        write_new_json(args.out, report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("runtime", "cases")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
