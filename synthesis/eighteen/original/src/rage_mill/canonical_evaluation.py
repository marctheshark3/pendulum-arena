"""Record and audit the published capsule benchmark without activating releases.

The development runner refuses every published reserved identifier. A future
reserved runner must be separately frozen and reviewed after its prerequisites
pass. Tape auditing and release verification never execute a controller or plant.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np

from .canonical_benchmark import (
    CanonicalEnv, CanonicalProtocol, SOURCE_COMMIT, canonical_xml,
    hanging_state, score_tape,
)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def value_sha256(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned_file(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), "regular non-symlink evidence file required")
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def read_pinned(row):
    require(isinstance(row, dict) and set(row) >= {"path", "sha256", "bytes"}, "pinned file row required")
    require(isinstance(row["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
            "SHA256 required")
    path = Path(row["path"])
    require(not path.is_symlink() and path.is_file(), "regular non-symlink evidence file required")
    data = path.read_bytes()
    require(type(row["bytes"]) is int and len(data) == row["bytes"]
            and hashlib.sha256(data).hexdigest() == row["sha256"], "pinned evidence bytes changed")
    return data


def write_json_exclusive(path, value):
    data = canonical_json(value) + b"\n"
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return pinned_file(path)


def runtime_metadata():
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "mujoco": importlib.metadata.version("mujoco"), "platform": platform.platform(),
        "executable": str(Path(sys.executable).resolve()),
        "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }


def reserved_gates(n_links):
    require(type(n_links) is int and 11 <= n_links <= 20, "published frontier count must be 11..20")
    base = 200000 + 1000 * n_links
    return {
        "noisy20": {"seed_start": base, "episodes": 20, "exact": False},
        "noisy100": {"seed_start": base + 100, "episodes": 100, "exact": False},
        "exact20": {"seed_start": base + 200, "episodes": 20, "exact": True},
        "video": {"seed_start": base + 500, "episodes": 1, "exact": False},
    }


def is_reserved(seed):
    # Refuse cross-count reserves as well as this controller's own reserve.
    return any(g["seed_start"] <= seed < g["seed_start"] + g["episodes"]
               for n in range(11, 21) for g in reserved_gates(n).values())


def make_identity(protocol, *, controller, configuration, source_files, policy_parameters):
    """Freeze concrete code/artifact/parameter/runtime identities before a run."""
    require(isinstance(protocol, CanonicalProtocol), "canonical protocol required")
    identity = {
        "schema": "rage_mill.canonical_evaluation_identity.v1", "source_commit": SOURCE_COMMIT,
        "protocol": asdict(protocol), "controller": pinned_file(controller),
        "configuration": pinned_file(configuration),
        "sources": [pinned_file(path) for path in source_files],
        "policy_parameters": policy_parameters, "runtime": runtime_metadata(),
        "native_xml_sha256": hashlib.sha256(canonical_xml(protocol).encode()).hexdigest(),
    }
    validate_identity(identity)
    return identity


def validate_identity(identity):
    require(isinstance(identity, dict) and identity.get("schema") == "rage_mill.canonical_evaluation_identity.v1",
            "canonical evaluation identity required")
    protocol = CanonicalProtocol(**identity["protocol"])
    require(identity.get("source_commit") == SOURCE_COMMIT, "different published source")
    require(identity.get("native_xml_sha256") == hashlib.sha256(canonical_xml(protocol).encode()).hexdigest(),
            "different canonical native XML")
    require(identity.get("runtime", {}).get("mujoco") == "3.3.5", "MuJoCo 3.3.5 identity required")
    require(isinstance(identity.get("policy_parameters"), dict), "explicit policy parameters required")
    sources = identity.get("sources")
    require(isinstance(sources, list) and sources, "controller and evaluator source pins required")
    rows = [identity["controller"], identity["configuration"], *sources]
    require(len({row["path"] for row in sources}) == len(sources), "duplicate source path")
    for row in rows:
        read_pinned(row)
    own_sources = {str(Path(__file__).resolve()),
                   str(Path(sys.modules[CanonicalProtocol.__module__].__file__).resolve())}
    require(own_sources <= {row["path"] for row in sources}, "canonical recorder and plant source pins required")
    canonical_json(identity)
    return protocol


def _journal(stream, value):
    stream.write(canonical_json(value) + b"\n")
    stream.flush()  # Keep a complete command prefix if the owner is interrupted.


def _record_episode(controller_factory, identity, seed, exact, out):
    """Internal recorder; only the development-only entry point invokes it."""
    protocol = validate_identity(identity)
    require(type(seed) is int and seed >= 0 and type(exact) is bool, "integer seed and boolean exact flag required")
    require(not is_reserved(seed), "reserved episodes need a separately reviewed activation runner")
    require(runtime_metadata() == identity["runtime"], "actual evaluation runtime differs from identity")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    claim = {"schema": "rage_mill.canonical_development_episode_claim.v1", "seed": seed,
             "exact": exact, "identity_sha256": value_sha256(identity), "stage": "development",
             "controller_qualified": False}
    write_json_exclusive(out / "claim.json", claim)
    states, actions, raw_actions, times = [], [], [], []
    invalid = None
    failure_state = np.empty((0, 2 * (protocol.n_links + 1)), dtype=np.float64)
    wall, cpu = time.monotonic(), time.process_time()
    with (out / "journal.jsonl").open("xb") as journal:
        # Claim precedes plant construction and initial RNG draws.
        try:
            controller = controller_factory()
            env = CanonicalEnv(protocol)
            initial = hanging_state(protocol, seed, exact=exact)
            states.append(env.reset(initial)); times.append(float(env.data.time))
            _journal(journal, {"initial": states[0].tolist(), "time": times[0]})
            for tick in range(protocol.episode_ticks):
                raw = float(controller(states[-1].copy(), tick))
                raw_actions.append(raw)
                state, delivered = env.step(raw)
                states.append(state.copy()); actions.append(delivered); times.append(float(env.data.time))
                _journal(journal, {"tick": tick, "raw_action": raw, "action": delivered,
                                   "state": state.tolist(), "time": times[-1]})
                if abs(state[0]) > protocol.rail_limit_m:
                    break
        except Exception as error:
            invalid = f"{type(error).__name__}: {error}"
            if "env" in locals():
                failure_state = np.asarray(env.state, dtype=np.float64).reshape(1, -1)
            attempted = np.float64(raw_actions[-1]).tobytes().hex() if len(raw_actions) > len(actions) else None
            _journal(journal, {"failure": invalid, "completed_commands": len(actions),
                               "attempted_raw_action_f64_hex": attempted})
    # Initialization failures retain their claim and diagnostic prefix but do
    # not fabricate an initial state or a score.
    s = np.asarray(states, dtype=np.float64).reshape(-1, 2 * (protocol.n_links + 1))
    u = np.asarray(actions, dtype=np.float32)
    raw = np.asarray(raw_actions, dtype=np.float64)
    stamp = np.asarray(times, dtype=np.float64)
    with (out / "tape.npz").open("xb") as stream:
        np.savez_compressed(stream, states=s, actions=u, raw_actions=raw,
                            times=stamp, failure_state=failure_state)
    score = score_tape(s, u, protocol, invalid_reason=invalid) if len(s) else None
    try:
        unchanged = all(pinned_file(row["path"]) == row for row in
                        [identity["controller"], identity["configuration"], *identity["sources"]])
    except (ValueError, OSError):
        unchanged = False
    receipt = {"schema": "rage_mill.canonical_episode_evidence.v1", "stage": "development",
               "seed": seed, "exact": exact, "identity": identity, "identity_sha256": value_sha256(identity),
               "recording": pinned_file(out / "tape.npz"), "journal": pinned_file(out / "journal.jsonl"),
               "claim": pinned_file(out / "claim.json"), "score": score, "invalid_reason": invalid,
               "source_pins_unchanged": unchanged, "wall_seconds": time.monotonic() - wall,
               "process_cpu_seconds": time.process_time() - cpu, "reset_count_after_initialization": 0,
               "initialized": bool(len(states)),
               "controller_qualified": False}
    return write_json_exclusive(out / "receipt.json", receipt)


def record_development_cohort(controller_factory, identity, *, seed_start, episodes, exact=False, out):
    """Fresh policy per episode, exclusive output, all failures kept; no reserves."""
    validate_identity(identity)
    require(type(seed_start) is int and seed_start >= 0 and type(episodes) is int and 1 <= episodes <= 100,
            "bounded declared development seed range required")
    require(type(exact) is bool and not any(is_reserved(s) for s in range(seed_start, seed_start + episodes)),
            "development cannot consume any published reserved identifier")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(out / "claim.json", {"schema": "rage_mill.canonical_development_cohort_claim.v1",
                         "seed_start": seed_start, "episodes": episodes,
                         "exact": exact, "identity_sha256": value_sha256(identity), "stage": "development"})
    rows = []
    for seed in range(seed_start, seed_start + episodes):
        rows.append(_record_episode(controller_factory, identity, seed, exact, out / f"episode-{seed}"))
    return write_json_exclusive(out / "receipt.json", {
        "schema": "rage_mill.canonical_cohort_evidence.v1", "stage": "development", "exact": exact,
        "seed_start": seed_start, "episodes": episodes, "episode_receipts": rows,
        "claim": pinned_file(out / "claim.json"),
        "identity": identity, "identity_sha256": value_sha256(identity), "controller_qualified": False,
    })


def audit_journal(data, *, states, actions, raw_actions, times, invalid_reason):
    """The flushed log and archive must describe the same actual command prefix."""
    def reject_constant(value):
        raise ValueError(f"nonfinite journal JSON constant: {value}")

    records = [json.loads(line, parse_constant=reject_constant) for line in data.splitlines()]
    require(all(isinstance(record, dict) for record in records), "journal records must be objects")
    expected = bool(len(states)) + len(actions) + (invalid_reason is not None)
    require(len(records) == expected, "journal prefix length differs from archive")

    def same(value, actual):
        values = value if isinstance(value, list) else [value]
        require(all(type(x) in (int, float) for x in values), "journal values must be numbers")
        converted = np.asarray(value, dtype=np.float64)
        target = np.asarray(actual, dtype=np.float64)
        require(converted.shape == target.shape and np.isfinite(converted).all()
                and converted.tobytes() == target.tobytes(), "journal value differs from actual archive")

    if len(states):
        initial = records.pop(0)
        require(set(initial) == {"initial", "time"}, "initial journal record required")
        same(initial["initial"], states[0]); same(initial["time"], times[0])
    for tick, record in enumerate(records[:len(actions)]):
        require(set(record) == {"tick", "raw_action", "action", "state", "time"}
                and type(record["tick"]) is int and record["tick"] == tick,
                "consecutive journal tick record required")
        same(record["raw_action"], raw_actions[tick]); same(record["action"], float(actions[tick]))
        same(record["state"], states[tick + 1]); same(record["time"], times[tick + 1])
    if invalid_reason is not None:
        failure = records[-1]
        attempted = (np.float64(raw_actions[-1]).tobytes().hex()
                     if len(raw_actions) > len(actions) else None)
        require(set(failure) == {"failure", "completed_commands", "attempted_raw_action_f64_hex"}
                and failure["failure"] == invalid_reason
                and type(failure["completed_commands"]) is int
                and failure["completed_commands"] == len(actions)
                and failure["attempted_raw_action_f64_hex"] == attempted,
                "journal failure or attempted command differs from archive")


def audit_episode(row, *, identity=None, expected_seed=None, expected_exact=None):
    """Rehash and rescore actual stored arrays, without plant/controller execution."""
    document = json.loads(read_pinned(row))
    require(document.get("schema") == "rage_mill.canonical_episode_evidence.v1", "canonical episode evidence required")
    actual_identity = document["identity"]
    protocol = validate_identity(actual_identity)
    require(identity is None or identity == actual_identity, "episode policy/source/runtime identity differs")
    require(document.get("identity_sha256") == value_sha256(actual_identity), "episode identity digest differs")
    seed, exact = document.get("seed"), document.get("exact")
    require(type(seed) is int and seed >= 0 and type(exact) is bool, "invalid episode reset identity")
    require(expected_seed is None or seed == expected_seed, "episode seed differs")
    require(expected_exact is None or exact is expected_exact, "episode distribution differs")
    require(document.get("source_pins_unchanged") is True
            and type(document.get("reset_count_after_initialization")) is int
            and document["reset_count_after_initialization"] == 0, "source change or phase reset")
    journal = read_pinned(document["journal"]); claim = json.loads(read_pinned(document["claim"]))
    stage = document.get("stage")
    require(stage in ("development", "reserved")
            and claim.get("schema") == f"rage_mill.canonical_{stage}_episode_claim.v1"
            and claim.get("stage") == stage and claim.get("seed") == seed and claim.get("exact") is exact
            and claim.get("identity_sha256") == value_sha256(actual_identity), "episode claim differs")
    if stage == "reserved":
        require(document.get("freeze") is not None and claim.get("freeze") == document["freeze"],
                "reserved episode ownership must bind its prospective freeze")
        read_pinned(document["freeze"])
    with np.load(io.BytesIO(read_pinned(document["recording"])), allow_pickle=False) as saved:
        require(set(saved.files) == {"states", "actions", "raw_actions", "times", "failure_state"}, "complete tape fields required")
        states, actions, raw, times = (saved[k].copy() for k in ("states", "actions", "raw_actions", "times"))
        failure_state = saved["failure_state"].copy()
    require(states.dtype == np.dtype("float64") and actions.dtype == np.dtype("float32")
            and raw.dtype == np.dtype("float64") and times.dtype == np.dtype("float64"), "native-state/action dtypes differ")
    require(failure_state.dtype == np.dtype("float64") and failure_state.ndim == 2
            and failure_state.shape[1] == 2 * (protocol.n_links + 1) and len(failure_state) <= 1,
            "native failure diagnostic state differs")
    require(document.get("initialized") is bool(len(states)), "initialization status differs")
    if not len(states):
        require(states.shape == (0, 2 * (protocol.n_links + 1)) and not len(actions)
                and raw.shape == times.shape == (0,) and document.get("score") is None
                and isinstance(document.get("invalid_reason"), str) and document["invalid_reason"],
                "failed initialization must retain an empty prefix and explicit failure")
        audit_journal(journal, states=states, actions=actions, raw_actions=raw,
                      times=times, invalid_reason=document["invalid_reason"])
        return {"document": document, "score": None, "states": states, "actions": actions,
                "raw_actions": raw, "times": times}
    require(len(states) and states[0].tobytes() == hanging_state(protocol, seed, exact=exact).tobytes(),
            "recording did not initialize from the specified canonical reset")
    require(times.shape == (len(states),) and np.isfinite(times).all()
            and np.max(np.abs(times - np.arange(len(states)) * protocol.control_dt)) <= 1e-9,
            "recorded cadence or initial clock differs")
    require(raw.ndim == 1 and len(actions) <= len(raw) <= len(actions) + 1, "raw command prefix differs")
    require(np.isfinite(raw[:len(actions)]).all()
            and np.array_equal(actions, np.clip(raw[:len(actions)].astype(np.float32), -1., 1.)),
            "delivered commands differ from recorded normalized float32 actions")
    score = score_tape(states, actions, protocol, invalid_reason=document.get("invalid_reason"))
    require(score == document.get("score"), "cached score differs from actual tape")
    audit_journal(journal, states=states, actions=actions, raw_actions=raw,
                  times=times, invalid_reason=document.get("invalid_reason"))
    return {"document": document, "score": score, "states": states, "actions": actions,
            "raw_actions": raw, "times": times}
