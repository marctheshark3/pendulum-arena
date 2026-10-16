"""Independent direct-MuJoCo audit of the saved canonical15 release episodes.

This does not run the policy or create replacement evaluation episodes. It
replays recorded one-command transitions in separate audit data and scores
angles independently of the campaign's environment/scorer implementation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from rage_mill.canonical_campaign import (
    candidate_identity, file_sha256, runtime_identity, source_identity, write_json,
)


ROOT = Path(__file__).resolve().parents[1]


def metrics(states, actions, raw):
    if (states.ndim != 2 or states.shape != (len(actions)+1, 32)
            or actions.ndim != 1 or raw.shape != actions.shape
            or not all(np.isfinite(value).all() for value in (states, actions, raw))):
        raise ValueError("invalid recorded trajectory arrays")
    if (np.any(np.abs(actions) > 1.)
            or not np.array_equal(actions, actions.astype(np.float32).astype(float))
            or not np.array_equal(actions, np.clip(raw.astype(np.float32), -1., 1.).astype(float))):
        raise ValueError("recorded commands differ from required float32 delivery")
    angles = np.cumsum(states[1:, 1:16], axis=1)
    outside = (angles < -np.pi) | (angles >= np.pi)
    angles[outside] = np.remainder(angles[outside]+np.pi, 2*np.pi)-np.pi
    upright = np.max(np.abs(angles), axis=1) < .15
    failures = np.flatnonzero(~upright)
    longest = int(max(np.diff(np.r_[-1, failures, len(actions)])-1))
    maximum_cart = float(np.max(np.abs(states[:, 0])))
    return dict(ticks=len(actions), longest_upright_ticks=longest,
                longest_upright_seconds=longest/50, maximum_cart_m=maximum_cart,
                rail_ok=maximum_cart <= 3., peak_normalized_action=float(np.max(np.abs(actions))) if len(actions) else 0.)


def check_case(xml, policy_path, cohort, row):
    base = Path(cohort).parent
    path = base/row["recording"]
    if file_sha256(path) != row["recording_sha256"]:
        raise ValueError("recording hash mismatch")
    if json.loads((path.parent/"receipt.json").read_text()) != row:
        raise ValueError("aggregate and individual episode receipt differ")
    with np.load(path, allow_pickle=False) as saved:
        states, actions, raw = (saved[key].copy() for key in ("states", "actions", "raw_actions"))
    result = metrics(states, actions, raw)
    if result["ticks"] != 1500 or not result["rail_ok"] or result["longest_upright_ticks"] < 250:
        raise ValueError("saved live episode fails the fixed task")
    initial = np.zeros(32)
    initial[1] = np.pi
    if not row["exact"]:
        rng = np.random.default_rng(row["seed"])
        initial[0] = rng.normal(0., 0.)
        initial[1:16] += rng.normal(0., .05, 15)
        initial[16:] = rng.normal(0., .05, 16)
        initial[16] = rng.normal(0., .05)
    initial_gap = float(np.max(np.abs(states[0]-initial)))
    if initial_gap > 1e-8:
        raise ValueError("initial state differs from published RNG/reset")
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    if (model.nq != 16 or model.nu != 1 or model.opt.timestep != .005
            or model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4):
        raise ValueError("native audit plant/cadence mismatch")
    maximum_defect = 0.
    for tick, action in enumerate(actions):
        mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:] = states[tick, :16], states[tick, 16:]
        mujoco.mj_forward(model, data)
        data.ctrl[0] = 80*float(action)
        warnings = tuple(item.number for item in data.warning)
        for _ in range(4):
            previous = float(data.time)
            mujoco.mj_step(model, data)
            if (not np.isfinite(previous) or not np.isfinite(data.time)
                    or abs(data.time-previous-.005) > 1e-10
                    or tuple(item.number for item in data.warning) != warnings
                    or not all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc))):
                raise ValueError("invalid direct native transition")
        maximum_defect = max(maximum_defect, float(np.max(np.abs(
            np.r_[data.qpos, data.qvel]-states[tick+1]))))
    if not np.isfinite(maximum_defect) or maximum_defect > 1e-8:
        raise ValueError("saved native transitions exceed source admission tolerance")
    policy = json.loads(Path(policy_path).read_text())
    forecasts = row["forecasts"]
    selected = row["selected_profile"]
    if not isinstance(selected, int) or [item["profile"] for item in forecasts] != list(range(selected+1)):
        raise ValueError("omitted or reordered internal planning forecasts")
    start = policy["controller"]["park_ticks"]
    for item in forecasts:
        if item["parameters"] != policy["forecast_profiles"][item["profile"]]:
            raise ValueError("internal planning search options differ from the frozen list")
        forecast_path = path.parent/item["recording"]
        if file_sha256(forecast_path) != item["recording_sha256"]:
            raise ValueError("internal forecast hash mismatch")
        with np.load(forecast_path, allow_pickle=False) as saved:
            fs, fa, fr = (saved[key].copy() for key in ("states", "actions", "raw_actions"))
        fm = metrics(fs, fa, fr)
        if not np.array_equal(fs[0], states[start]):
            raise ValueError("forecast did not start at the observed live launch state")
        accepted = (fm["ticks"] == 1500-start and fm["rail_ok"]
                    and fm["longest_upright_ticks"] >= 250 and item["score"]["invalid_reason"] is None)
        if accepted != item["accepted"] or accepted != (item["profile"] == selected):
            raise ValueError("planner did not select its first complete valid forecast")
        if item["profile"] == selected and not (
                np.array_equal(fs, states[start:]) and np.array_equal(fa, actions[start:])
                and np.array_equal(fr, raw[start:])):
            raise ValueError("live state/action route differs from the selected native forecast")
    return dict(gate=row["gate"], seed=row["seed"], exact=row["exact"], passed=True, **result,
                initial_state_gap=initial_gap, native_maximum_dynamics_defect=maximum_defect,
                live_prediction_byte_match=True, selected_profile=selected,
                retained_forecasts=len(forecasts), source_recording_sha256=file_sha256(path))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--reserved", required=True, type=Path)
    parser.add_argument("--video-proof", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    candidate = Path(freeze["frozen_candidate"])
    identity = dict(candidate=candidate_identity(candidate), source=source_identity(ROOT), runtime=runtime_identity())
    if identity != freeze["identities"] or file_sha256(freeze["source_bundle"]) != freeze["source_bundle_sha256"]:
        raise ValueError("frozen source/runtime/payload differs")
    declarations = dict(calibration=[("calibration", 3156000, 1, True)],
                        development=[("dev20", 3156200, 20, False), ("dev100", 3156300, 100, False)],
                        reserved=[("noisy20", 215000, 20, False), ("noisy100", 215100, 100, False),
                                  ("exact20", 215200, 20, True), ("video", 215500, 1, False)])
    jobs, evidence = [], {}
    for stage, receipt_path in (("calibration", Path(freeze["calibration_receipt"])),
                                 ("development", Path(freeze["development_receipt"])),
                                 ("reserved", args.reserved)):
        receipt = json.loads(receipt_path.read_text())
        if (receipt["stage"] != stage or not receipt["passed"] or not receipt["complete"]
                or not receipt["identity_unchanged"] or receipt["identities"] != identity):
            raise ValueError("cohort identities or admission failed")
        expected = {(name, seed, exact) for name, start, count, exact in declarations[stage]
                    for seed in range(start, start+count)}
        actual = [(row["gate"], row["seed"], row["exact"]) for row in receipt["cases"]]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("cohort has missing, replaced or duplicate labels")
        if stage != "reserved" and file_sha256(receipt_path) != freeze["evidence"][stage]["sha256"]:
            raise ValueError("pre-freeze development evidence changed")
        evidence[stage] = dict(path=str(receipt_path.resolve()), sha256=file_sha256(receipt_path))
        jobs.extend((str(receipt_path), row) for row in receipt["cases"])
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(check_case, str(candidate/"model.xml"), str(candidate/"policy.json"), cohort, row)
                   for cohort, row in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(dict(gate=row["gate"], seed=row["seed"], passed=True,
                                  native_maximum_dynamics_defect=row["native_maximum_dynamics_defect"])), flush=True)
    video = json.loads(args.video_proof.read_text())
    if (not video["passed"] or video["rendered_frames"] != 1500 or video["fps"] != 50
            or video["simulated_seconds"] != 30. or video["extra_held_frames"] != 0
            or video["initial_frame_included"] or file_sha256(video["video"]) != video["video_sha256"]
            or video["reserved_receipt_sha256"] != evidence["reserved"]["sha256"]):
        raise ValueError("video does not bind the required held-out thirty-second recording")
    video_row, = [row for row in rows if row["gate"] == "video"]
    if video_row["source_recording_sha256"] != video["source_recording_sha256"]:
        raise ValueError("video is not the reserved215500 recording")
    result = dict(schema="rage_mill.canonical15_independent_release_audit.v1", passed=True,
                  controller_qualified=True, native_actual_episodes=len(rows), native_actual_commands=1500*len(rows),
                  independent_cold_transition_checks=1500*len(rows), policy_calls_during_audit=0,
                  maximum_native_dynamics_defect=max(row["native_maximum_dynamics_defect"] for row in rows),
                  maximum_initial_state_gap=max(row["initial_state_gap"] for row in rows),
                  minimum_upright_seconds=min(row["longest_upright_seconds"] for row in rows),
                  maximum_cart_m=max(row["maximum_cart_m"] for row in rows),
                  freeze_sha256=file_sha256(args.freeze), source_commit=freeze["source_commit"],
                  identities=identity, evidence=evidence, video_proof_sha256=file_sha256(args.video_proof),
                  video_sha256=video["video_sha256"], cases=sorted(rows, key=lambda row: (row["gate"], row["seed"])),
                  wall_seconds=time.monotonic()-started,
                  claim_scope="matched fifteen-link published simulation task; no hardware or established global priority")
    write_json(args.out/"receipt.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("identities", "cases")}), flush=True)


if __name__ == "__main__":
    main()
