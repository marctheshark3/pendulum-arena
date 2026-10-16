"""Once-only cohorts for an unchanged canonical15 controller and native runtime."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from rage_mill.canonical_benchmark import CanonicalProtocol, hanging_state, record_episode
from rage_mill.canonical_campaign import (
    audit_native_tape, candidate_identity, file_sha256, json_sha256, load_candidate,
    runtime_identity, source_identity, write_json,
)


ROOT = Path(__file__).resolve().parents[1]


def run_case(candidate, out, gate, seed, exact, identity):
    path = Path(out)/"cases"/f"{gate}-{seed}"
    path.mkdir(parents=True, exist_ok=False)
    write_json(path/"claim.json", dict(gate=gate, seed=seed, exact=exact, identities=identity))
    started = time.monotonic()
    policy = load_candidate(candidate)
    states, actions, raw, score = record_episode(
        policy, hanging_state(CanonicalProtocol(), seed, exact=exact))
    control_seconds = time.monotonic()-started
    np.savez_compressed(path/"episode.npz", states=states, actions=actions, raw_actions=raw)
    audit = audit_native_tape(states, actions, raw, seed, exact, score["invalid_reason"])
    forecasts = []
    for item in getattr(policy, "forecasts", []):
        index = item["profile"]
        forecast_path = path/f"forecast-{index:02d}.npz"
        np.savez_compressed(forecast_path, **{key: item[key] for key in ("states", "actions", "raw_actions")})
        forecasts.append({key: value for key, value in item.items()
                          if key not in ("states", "actions", "raw_actions")} | dict(
                              recording=forecast_path.name, recording_sha256=file_sha256(forecast_path)))
    if getattr(policy, "plan_states", None) is not None:
        start = policy.parameters["park_ticks"]
        audit["live_prediction_byte_match"] = bool(np.array_equal(states[start:], policy.plan_states))
        audit["admitted"] &= audit["live_prediction_byte_match"]
    row = dict(gate=gate, seed=seed, exact=exact, **audit,
               recording=str((path/"episode.npz").relative_to(out)),
               recording_sha256=file_sha256(path/"episode.npz"),
               control_wall_seconds=control_seconds, wall_seconds=time.monotonic()-started,
               native_prediction_calls=policy.native_prediction_calls,
               selected_profile=getattr(policy, "selected_profile", None), forecasts=forecasts)
    write_json(path/"receipt.json", row)
    return row


def require_receipt(path, identity, stage):
    data = json.loads(Path(path).read_text())
    if data["stage"] != stage or not data["passed"] or data["identities"] != identity:
        raise ValueError(f"missing passing {stage} evidence for these exact identities")
    return data


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("diagnostic", "calibration", "development", "reserved"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3156000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("worker count outside declared CPU bound")
    config = json.loads((ROOT/"configs/canonical15.json").read_text())
    identity = dict(candidate=candidate_identity(args.candidate), source=source_identity(ROOT),
                    runtime=runtime_identity())
    if args.stage == "diagnostic":
        if not 3156000 <= args.seed <= args.seed+args.episodes-1 <= 3156099:
            raise ValueError("diagnostics must stay in their declared range")
        gates = [dict(name="diagnostic", seed_start=args.seed, episodes=args.episodes, exact=False)]
    elif args.stage == "calibration":
        gates = [dict(name="calibration", seed_start=3156000, episodes=1, exact=True)]
    elif args.stage == "development":
        if args.calibration is None:
            raise ValueError("development requires exact source calibration")
        require_receipt(args.calibration, identity, "calibration")
        gates = config["development_gates"]
    else:
        if args.freeze is None:
            raise ValueError("reserved gates require the prospective source freeze")
        freeze = json.loads(args.freeze.read_text())
        if freeze["identities"] != identity or not freeze["ready_for_reserved"]:
            raise ValueError("freeze does not match this runtime/source/controller")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        if commit != freeze["source_commit"] or dirty:
            raise ValueError("reserved evaluation requires the clean frozen source clone")
        require_receipt(freeze["calibration_receipt"], identity, "calibration")
        require_receipt(freeze["development_receipt"], identity, "development")
        gates = config["reserved_gates"]
    if args.stage in ("development", "reserved"):
        if args.ledger is None:
            raise ValueError("formal gates require a shared once-only consumption ledger")
        args.ledger.mkdir(parents=True, exist_ok=True)
        with (args.ledger/f"{args.stage}.claim.json").open("x") as claim:
            json.dump(dict(stage=args.stage, out=str(args.out.resolve()), identities=identity,
                           gates=gates), claim, indent=2, allow_nan=False)
    args.out.mkdir(parents=True, exist_ok=False)
    jobs = [(g["name"], seed, g["exact"]) for g in gates
            for seed in range(g["seed_start"], g["seed_start"]+g["episodes"])]
    started = time.monotonic()
    receipt = dict(schema="rage_mill.canonical15_cohort.v1", stage=args.stage,
                   identities=identity, gates=gates, expected_cases=len(jobs), cases=[],
                   complete=False, passed=False, workers=args.workers)
    write_json(args.out/"claim.json", receipt)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, str(args.candidate.resolve()), str(args.out.resolve()),
                               gate, seed, exact, identity): (gate, seed, exact)
                   for gate, seed, exact in jobs}
        for future in as_completed(futures):
            gate, seed, exact = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = dict(gate=gate, seed=seed, exact=exact, admitted=False,
                           success=False, worker_error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
    after = dict(candidate=candidate_identity(args.candidate), source=source_identity(ROOT),
                 runtime=runtime_identity())
    receipt.update(cases=sorted(rows, key=lambda row: (row["gate"], row["seed"])),
                   complete=len(rows) == len(jobs), identity_unchanged=after == identity,
                   passed=bool(len(rows) == len(jobs) and all(row["admitted"] for row in rows)
                               and after == identity),
                   wall_seconds=time.monotonic()-started)
    receipt["cohorts"] = {g["name"]: dict(
        episodes=g["episodes"], admitted=sum(row["admitted"] for row in rows if row["gate"] == g["name"]),
        required=g.get("required_successes", g["episodes"])) for g in gates}
    write_json(args.out/"receipt.json", receipt)
    print(json.dumps(dict(stage=args.stage, passed=receipt["passed"], cohorts=receipt["cohorts"],
                          identity_sha256=json_sha256(identity)), allow_nan=False), flush=True)
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
