"""New public adapter: run one fresh diagnostic episode, never a qualification cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from public_support import (
    FREEZE_SHA256, SOURCE_COMMIT, check_runtime, install_frozen_path,
    load_proof, read_json, sha256, write_new_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="new diagnostic directory")
    parser.add_argument("--seed", type=int, default=3156500, help="reset only; never passed to the controller")
    parser.add_argument("--exact", action="store_true", help="use the exact hanging reset")
    parser.add_argument("--initial-state", type=Path, help="optional JSON list of 32 observed qpos/qvel values; diagnostic only")
    parser.add_argument("--allow-runtime-mismatch", action="store_true", help="exploratory only; still requires native MuJoCo 3.3.5")
    args = parser.parse_args()
    proof = load_proof(args.proof_root)
    runtime, differences = check_runtime(proof["freeze"], args.allow_runtime_mismatch)
    install_frozen_path()
    import numpy as np
    from rage_mill.canonical_benchmark import CanonicalProtocol, hanging_state, record_episode
    from rage_mill.canonical_campaign import load_candidate

    if args.initial_state:
        if args.exact or args.seed != 3156500:
            parser.error("--initial-state cannot be combined with --exact or an explicit --seed")
        initial = np.asarray(read_json(args.initial_state), dtype=float)
        if initial.shape != (32,) or not np.isfinite(initial).all():
            parser.error("--initial-state must contain exactly 32 finite numbers")
        reset = dict(kind="observed_state_json", sha256=sha256(args.initial_state))
    else:
        initial = hanging_state(CanonicalProtocol(), args.seed, exact=args.exact)
        reset = dict(kind="hanging", seed=args.seed, exact=args.exact)
    args.out.mkdir(parents=True, exist_ok=False)
    write_new_json(args.out / "diagnostic-claim.json", dict(source_commit=SOURCE_COMMIT, reset=reset,
                   runtime=runtime, frozen_runtime_match=not differences, creates_new_qualification=False))
    policy = load_candidate(proof["candidate"])
    started = time.monotonic()
    states, actions, raw_actions, score = record_episode(policy, initial)
    episode_path = args.out / "episode.npz"
    np.savez_compressed(episode_path, states=states, actions=actions, raw_actions=raw_actions)
    forecasts = []
    for item in policy.forecasts:
        path = args.out / f"forecast-{item['profile']:02d}.npz"
        np.savez_compressed(path, **{key: item[key] for key in ("states", "actions", "raw_actions")})
        forecasts.append({key: value for key, value in item.items() if key not in ("states", "actions", "raw_actions")}
                         | dict(recording=path.name, recording_sha256=sha256(path)))
    report = dict(schema="pendulum_arena.canonical15_public_diagnostic.v1", source_commit=SOURCE_COMMIT,
                  original_freeze_sha256=FREEZE_SHA256, reset=reset, score=score,
                  recording=episode_path.name, recording_sha256=sha256(episode_path), forecasts=forecasts,
                  selected_profile=policy.selected_profile, native_prediction_calls=policy.native_prediction_calls,
                  frozen_runtime_match=not differences, runtime_differences=differences,
                  control_wall_seconds=time.monotonic() - started, creates_new_qualification=False,
                  claim_scope="One fresh simulation diagnostic. No new qualification, real-time, hardware, disturbance-robustness, or global-priority claim.")
    write_new_json(args.out / "receipt.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "forecasts"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
