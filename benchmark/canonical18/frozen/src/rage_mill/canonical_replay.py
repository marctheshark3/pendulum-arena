"""Native proofs over recorded states/actions, with no RNG or controller calls.

This is a saved-input verification job, not another reset distribution or fresh
evaluation. The continuous replay starts from the recorded initial state. A
separate scratch plant evaluates each saved transition independently to measure
its native float32-delivery defect. Neither operation changes a live campaign.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import time

import numpy as np

from .canonical_benchmark import CanonicalEnv
from .canonical_evaluation import (
    pinned_file, read_pinned, require, runtime_metadata, validate_identity,
    value_sha256, write_json_exclusive,
)


def _inputs(episode_receipt):
    document = json.loads(read_pinned(episode_receipt))
    require(document.get("schema") == "rage_mill.canonical_episode_evidence.v1", "recorded canonical episode required")
    protocol = validate_identity(document["identity"])
    require(document.get("identity_sha256") == value_sha256(document["identity"]), "episode identity changed")
    with np.load(io.BytesIO(read_pinned(document["recording"])), allow_pickle=False) as data:
        states, actions = data["states"].copy(), data["actions"].copy()
    require(states.dtype == np.dtype("float64") and actions.dtype == np.dtype("float32")
            and states.ndim == 2 and states.shape == (len(actions) + 1, 2 * (protocol.n_links + 1))
            and actions.ndim == 1 and len(actions) <= 1500
            and np.isfinite(states).all() and np.isfinite(actions).all()
            and np.all(np.abs(actions) <= 1.), "finite complete recorded transition prefix required")
    return document, protocol, states, actions


def _summary(states, continuous, cold):
    literal_initial = np.zeros(states.shape[1]); literal_initial[1] = np.pi
    initial_gap = max(float(np.max(np.abs(states[0] - literal_initial))),
                      float(np.max(np.abs(continuous[0] - states[0]))))
    continuous_error = float(np.max(np.abs(continuous - states)))
    defect = float(np.max(np.abs(cold - states[1:]))) if len(cold) else 0.
    return {"maximum_dynamics_defect": defect, "initial_state_gap": initial_gap,
            "maximum_continuous_replay_error": continuous_error,
            "continuous_replay_identical": continuous.dtype == states.dtype
            and continuous.shape == states.shape and continuous.tobytes() == states.tobytes()}


def prove_recorded_native_transitions(episode_receipt, out, *, maximum_native_transitions=3000):
    document, protocol, states, actions = _inputs(episode_receipt)
    require(type(maximum_native_transitions) is int and maximum_native_transitions == 3000
            and 2 * len(actions) <= maximum_native_transitions, "bounded saved replay budget required")
    require(runtime_metadata() == document["identity"]["runtime"], "native replay runtime differs from recording")
    producer = pinned_file(__file__)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(out / "claim.json", {
        "purpose": "saved_recorded_initial_and_actions_only", "episode_receipt": episode_receipt,
        "recording": document["recording"], "maximum_native_transitions": maximum_native_transitions,
        "RNG_draws": 0, "controller_calls": 0, "benchmark_episodes": 0})
    continuous, cold = [], []
    failure = None
    wall, cpu = time.monotonic(), time.process_time()
    try:
        replay = CanonicalEnv(protocol)
        continuous.append(replay.reset(states[0]).copy())
        for action in actions:
            following, delivered = replay.step(action)
            require(np.float32(delivered).tobytes() == action.tobytes(), "continuous command delivery changed")
            continuous.append(following.copy())
        scratch = CanonicalEnv(protocol)
        for state, action in zip(states[:-1], actions):
            scratch.reset(state)
            following, delivered = scratch.step(action)
            require(np.float32(delivered).tobytes() == action.tobytes(), "one-step command delivery changed")
            cold.append(following.copy())
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    dimension = states.shape[1]
    continuous = np.asarray(continuous, dtype=np.float64).reshape(-1, dimension)
    cold = np.asarray(cold, dtype=np.float64).reshape(-1, dimension)
    with (out / "native-replay.npz").open("xb") as stream:
        np.savez_compressed(stream, continuous_states=continuous, cold_next_states=cold)
    complete = failure is None and continuous.shape == states.shape and cold.shape == states[1:].shape
    summary = _summary(states, continuous, cold) if complete else None
    # Rehash the immutable episode/input sources after native work as well.
    _inputs(episode_receipt)
    require(pinned_file(__file__) == producer, "native replay source changed during execution")
    return write_json_exclusive(out / "receipt.json", {
        "schema": "rage_mill.canonical_recorded_native_replay.v1", "episode_receipt": episode_receipt,
        "recording": document["recording"], "identity_sha256": value_sha256(document["identity"]),
        "proof_arrays": pinned_file(out / "native-replay.npz"), "claim": pinned_file(out / "claim.json"),
        "producer_source": producer, "runtime": runtime_metadata(),
        "continuous_transitions": max(0, len(continuous) - 1), "one_step_transitions": len(cold),
        "failure": failure, "complete": complete, "summary": summary,
        "RNG_draws": 0, "controller_calls": 0, "benchmark_episodes": 0,
        "wall_seconds": time.monotonic() - wall, "process_cpu_seconds": time.process_time() - cpu,
        "controller_qualified": False})


def audit_native_replay_proof(proof_row, *, episode_receipt):
    """Recompute proof summaries from pinned output arrays; no native calls."""
    proof = json.loads(read_pinned(proof_row))
    document, protocol, states, actions = _inputs(episode_receipt)
    require(proof.get("schema") == "rage_mill.canonical_recorded_native_replay.v1"
            and proof.get("episode_receipt") == episode_receipt
            and proof.get("recording") == document["recording"]
            and proof.get("identity_sha256") == value_sha256(document["identity"]), "native proof input lineage differs")
    require(proof.get("complete") is True and proof.get("failure") is None,
            "native proof failed or incomplete")
    require(proof.get("runtime") == document["identity"]["runtime"], "native proof runtime differs")
    require(read_pinned(proof["producer_source"]) == Path(__file__).read_bytes(), "native proof producer source differs")
    claim = json.loads(read_pinned(proof["claim"]))
    require(claim.get("episode_receipt") == episode_receipt and claim.get("recording") == document["recording"]
            and claim.get("maximum_native_transitions") == 3000
            and claim.get("purpose") == "saved_recorded_initial_and_actions_only", "native proof claim differs")
    require(all(type(proof.get(key)) is int and proof[key] == 0
                and type(claim.get(key)) is int and claim[key] == 0
                for key in ("RNG_draws", "controller_calls", "benchmark_episodes")),
            "saved proof cannot consume another controller/reset cohort")
    with np.load(io.BytesIO(read_pinned(proof["proof_arrays"])), allow_pickle=False) as data:
        require(set(data.files) == {"continuous_states", "cold_next_states"}, "native proof arrays required")
        continuous, cold = data["continuous_states"].copy(), data["cold_next_states"].copy()
    require(continuous.dtype == cold.dtype == states.dtype
            and continuous.shape == states.shape and cold.shape == states[1:].shape
            and np.isfinite(continuous).all() and np.isfinite(cold).all(), "complete finite native proof arrays required")
    require(proof.get("continuous_transitions") == proof.get("one_step_transitions") == len(actions),
            "native proof transition count differs")
    summary = _summary(states, continuous, cold)
    require(summary == proof.get("summary"), "cached native proof summary differs from actual arrays")
    return summary
