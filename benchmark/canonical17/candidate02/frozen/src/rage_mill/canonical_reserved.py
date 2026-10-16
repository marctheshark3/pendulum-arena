"""Prospective reserved execution with count-wide, once-only ownership.

Importing this module creates no files and draws no reset. There is no default
claim root or command-line activation. A separately reviewed, pinned authority
must bind a passing source freeze and an explicit permanent canonical ledger.
Directory reservations consume attempts even when a process dies before its
receipt is written. Do not delete, repair, relocate or retry those reservations.
"""
from __future__ import annotations

import fcntl
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from . import canonical_evaluation as evidence
from . import canonical_release as release
from .canonical_benchmark import CanonicalEnv, SOURCE_COMMIT, hanging_state, score_tape
from .canonical_evaluation import (
    canonical_json, pinned_file, read_pinned, require, reserved_gates,
    runtime_metadata, validate_identity, value_sha256, write_json_exclusive,
)
from .canonical_runtime_evidence import (
    audit_formal_runtime_owners, audit_runtime_execution, record_runtime_execution,
)

GATE_ORDER = ("noisy20", "noisy100", "exact20", "video")


def _absolute(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink(),
            "canonical absolute non-symlink path required")
    return path


def _sync(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _namespace(n_links):
    return {"schema": "rage_mill.canonical_reserved_namespace.v1",
            "source_commit": SOURCE_COMMIT, "n_links": n_links,
            "gates": reserved_gates(n_links)}


def _authority(row, *, execution):
    authority = json.loads(read_pinned(row))
    require(authority.get("schema") == "rage_mill.canonical_reserved_execution_authority.v1"
            and authority.get("activation_authorized") is True,
            "explicit prospective reserved execution authority required")
    # This rechecks source calibration/native proof, both complete development
    # cohorts and the clean bundle/clone. A cached qualification flag is unused.
    frozen = release.audit_freeze(authority["freeze"])
    identity = frozen["identity"]
    protocol = validate_identity(identity)
    require(authority.get("identity_sha256") == value_sha256(identity)
            and authority.get("namespace") == _namespace(protocol.n_links),
            "authority count, identity or reserved namespace differs")
    require(authority.get("runtime_closure") == frozen.get("runtime_closure")
            and isinstance(authority.get("runtime_closure"), dict),
            "freeze and authority require the same pinned runtime closure")
    audit_formal_runtime_owners(frozen)
    if execution:
        require(runtime_metadata() == identity["runtime"], "actual reserved runtime differs")
    review = json.loads(read_pinned(authority["source_review"]))
    require(review.get("schema") == "rage_mill.canonical_reserved_source_review.v1"
            and review.get("passed") is True and review.get("activation_authorized") is False,
            "separate pinned source-only review required")
    sources = review.get("sources")
    require(isinstance(sources, list) and sources, "reviewed runner sources required")
    for source in sources:
        read_pinned(source)
        require(source in identity["sources"], "reviewed runner source omitted from frozen identity")
    required_modules = (__name__, evidence.__name__, release.__name__,
                        "rage_mill.canonical_replay", "rage_mill.canonical_benchmark",
                        "rage_mill.canonical_runtime_evidence", "rage_mill.canonical_runtime_manifest",
                        "rage_mill.canonical_runtime_closure")
    required_paths = {str(Path(sys.modules[name].__file__).resolve()) for name in required_modules}
    require(required_paths <= {source["path"] for source in sources},
            "runner/recorder/plant/release/native-proof source review incomplete")
    entry = authority.get("controller_factory")
    require(isinstance(entry, dict) and set(entry) == {"module", "attribute", "source"}
            and isinstance(entry["module"], str)
            and all(part.isidentifier() for part in entry["module"].split("."))
            and isinstance(entry["attribute"], str) and entry["attribute"].isidentifier(),
            "literal frozen factory entry point required")
    read_pinned(entry["source"])
    require(entry["source"] in sources, "factory source not independently reviewed and frozen")
    claim_base, output_root = map(_absolute, (authority["claim_base"], authority["output_root"]))
    require(not (output_root == claim_base or output_root.is_relative_to(claim_base)
                 or claim_base.is_relative_to(output_root)), "ledger and episode output must be disjoint")
    canonical_json(authority)
    return authority, frozen, protocol


def _anchor(authority_row, authority, protocol):
    return {"schema": "rage_mill.canonical_reserved_campaign_claim.v1",
            "namespace": _namespace(protocol.n_links), "authority": authority_row,
            "freeze": authority["freeze"], "identity_sha256": authority["identity_sha256"],
            "output_root": authority["output_root"], "all_reserved_episodes": 141,
            "no_retry_or_rekey": True, "reset_calls_at_claim": 0}


def _root(authority, protocol):
    # Source+count, never controller/profile/authority: a failed campaign cannot
    # obtain new labels by changing a policy hash within this permanent base.
    return _absolute(authority["claim_base"]) / f"published-{SOURCE_COMMIT}" / f"count-{protocol.n_links}"


def _seed_claim(authority_row, authority, protocol, gate, seed):
    return {"schema": "rage_mill.canonical_reserved_label_claim.v1",
            "namespace": _namespace(protocol.n_links), "authority": authority_row,
            "freeze": authority["freeze"], "identity_sha256": authority["identity_sha256"],
            "gate": gate, "seed": seed, "exact": reserved_gates(protocol.n_links)[gate]["exact"],
            "output": str(_absolute(authority["output_root"]) / gate / f"episode-{seed}"),
            "no_retry": True, "reset_calls_at_claim": 0}


def _gate_claim(authority_row, authority, protocol, gate):
    return {"schema": "rage_mill.canonical_reserved_gate_claim.v1",
            "authority": authority_row, "freeze": authority["freeze"],
            "namespace": _namespace(protocol.n_links), "gate": gate,
            **reserved_gates(protocol.n_links)[gate],
            "output": str(_absolute(authority["output_root"]) / gate), "no_retry": True}


def _audit_storage(authority_row, authority, protocol, gate, cohort_row=None):
    root = _root(authority, protocol)
    _absolute(root)
    gate_directory = root / f"gate-{gate}"
    require(gate_directory.is_dir() and not gate_directory.is_symlink(),
            "regular reserved gate directory required")
    require(json.loads(read_pinned(pinned_file(root / "campaign.json")))
            == _anchor(authority_row, authority, protocol), "canonical campaign anchor differs; no rekey")
    gate_row = pinned_file(gate_directory / "claim.json")
    require(json.loads(read_pinned(gate_row)) == _gate_claim(authority_row, authority, protocol, gate),
            "reserved gate claim differs")
    seeds = reserved_gates(protocol.n_links)[gate]
    labels = {}
    for seed in range(seeds["seed_start"], seeds["seed_start"] + seeds["episodes"]):
        directory = root / f"seed-{seed}"
        require(directory.is_dir() and not directory.is_symlink(), "reserved label claim missing")
        label_row = pinned_file(directory / "claim.json")
        require(json.loads(read_pinned(label_row)) == _seed_claim(authority_row, authority, protocol, gate, seed),
                "reserved label ownership differs")
        labels[seed] = label_row
    if cohort_row is not None:
        cohort = json.loads(read_pinned(cohort_row))
        output = _absolute(authority["output_root"]) / gate
        _absolute(output)
        require(cohort_row["path"] == str(output / "receipt.json")
                and cohort.get("claim", {}).get("path") == str(output / "claim.json"),
                "cohort moved outside its exclusively owned output")
        claim = json.loads(read_pinned(cohort["claim"]))
        require(cohort.get("campaign_authority") == authority_row
                and cohort.get("campaign_claim") == pinned_file(root / "campaign.json")
                and cohort.get("ledger_gate_claim") == gate_row
                and claim.get("ledger_gate_claim") == gate_row,
                "cohort lacks its exact campaign ledger binding")
        for seed, episode_row in zip(labels, cohort["episode_receipts"]):
            episode_output = output / f"episode-{seed}"
            _absolute(episode_output)
            require(episode_row["path"] == str(episode_output / "receipt.json"),
                    "episode moved outside its exclusively owned output")
            episode = json.loads(read_pinned(episode_row))
            require(all(episode.get(field, {}).get("path") == str(episode_output / name)
                        for field, name in (("claim", "claim.json"), ("recording", "tape.npz"),
                                            ("journal", "journal.jsonl"))),
                    "episode files moved outside their exclusively owned output")
            claim = json.loads(read_pinned(episode["claim"]))
            require(claim.get("ledger_label_claim") == labels[seed],
                    "episode lacks its once-only label claim")
        runtime_report = cohort.get("runtime_execution")
        require(isinstance(runtime_report, dict)
                and runtime_report.get("path") == str(output / "runtime-receipt.json")
                and value_sha256(cohort["identity"]) == authority["identity_sha256"],
                "reserved cohort lacks its owned runtime report")
        audit_runtime_execution(runtime_report, identity=cohort["identity"],
                                runtime_closure=authority["runtime_closure"],
                                outputs=cohort["episode_receipts"])
    return gate_row, labels


def _claim_gate(authority_row, gate):
    """Burn a whole gate atomically before imports, construction or reset RNG."""
    authority, frozen, protocol = _authority(authority_row, execution=True)
    require(gate in GATE_ORDER, "one fixed reserved gate required")
    root = _root(authority, protocol)
    root.parent.mkdir(parents=True, exist_ok=True)
    require(root.parent.resolve() == root.parent, "canonical ledger parent cannot be a symlink")
    lock_path = root.parent / f"count-{protocol.n_links}.lock"
    require(not lock_path.is_symlink(), "ledger lock cannot be a symlink")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Recheck all actual prerequisite and full runtime bytes after waiting;
        # no cached pass is used at the point that burns labels.
        authority, frozen, protocol = _authority(authority_row, execution=True)
        if root.exists() or root.is_symlink():
            require(root.is_dir() and not root.is_symlink(), "canonical ledger root must be a directory")
            require((root / "campaign.json").is_file(), "crashed campaign reservation cannot be repaired")
            require(json.loads(read_pinned(pinned_file(root / "campaign.json")))
                    == _anchor(authority_row, authority, protocol), "canonical campaign already bound; no rekey")
        else:
            require(not _absolute(authority["output_root"]).exists(), "campaign output already exists")
            root.mkdir(); _sync(root.parent)
            write_json_exclusive(root / "campaign.json", _anchor(authority_row, authority, protocol))
            _sync(root)
        for previous in GATE_ORDER[:GATE_ORDER.index(gate)]:
            prior = pinned_file(_absolute(authority["output_root"]) / previous / "receipt.json")
            _audit_storage(authority_row, authority, protocol, previous, prior)
            details = reserved_gates(protocol.n_links)[previous]
            release.audit_cohort(prior, identity=frozen["identity"], count=details["episodes"],
                                 exact=details["exact"], seed_start=details["seed_start"],
                                 stage="reserved", freeze=authority["freeze"])
        gate_directory = root / f"gate-{gate}"
        require(not gate_directory.exists() and not gate_directory.is_symlink(),
                "reserved gate already consumed, including interrupted attempts; no retry")
        gate_output = _absolute(authority["output_root"]) / gate
        require(not gate_output.exists() and not gate_output.is_symlink(), "reserved output already exists")
        details = reserved_gates(protocol.n_links)[gate]
        labels = list(range(details["seed_start"], details["seed_start"] + details["episodes"]))
        require(all(not (root / f"seed-{seed}").exists() and not (root / f"seed-{seed}").is_symlink()
                    for seed in labels), "reserved labels already consumed; no replacement")
        gate_directory.mkdir(); _sync(root)  # The directory itself consumes the gate.
        write_json_exclusive(gate_directory / "claim.json", _gate_claim(authority_row, authority, protocol, gate))
        for seed in labels:
            directory = root / f"seed-{seed}"
            directory.mkdir(); _sync(root)
            write_json_exclusive(directory / "claim.json", _seed_claim(authority_row, authority, protocol, gate, seed))
            _sync(directory)
        _sync(gate_directory)
        gate_row, label_rows = _audit_storage(authority_row, authority, protocol, gate)
    return authority, frozen, protocol, gate_row, label_rows


def _controller(authority, identity):
    # Called only after the durable label claim, never by preflight/auditing.
    entry = authority["controller_factory"]
    module = importlib.import_module(entry["module"])
    require(pinned_file(module.__file__) == entry["source"], "imported controller module differs")
    factory = getattr(module, entry["attribute"])
    require(inspect.isfunction(factory) and factory.__module__ == module.__name__
            and factory.__name__ == entry["attribute"] and factory.__closure__ is None
            and Path(factory.__code__.co_filename).resolve() == Path(entry["source"]["path"]),
            "literal source-bound factory function required")
    controller = factory(identity)
    require(callable(controller), "factory must create a fresh callable")
    return controller


def _record_reserved_episode(authority, frozen, protocol, seed, exact, out, label_row):
    identity = frozen["identity"]
    validate_identity(identity)
    require(runtime_metadata() == identity["runtime"], "actual runtime changed before episode")
    read_pinned(label_row)
    out.mkdir(parents=True, exist_ok=False)
    claim = {"schema": "rage_mill.canonical_reserved_episode_claim.v1", "stage": "reserved",
             "seed": seed, "exact": exact, "identity_sha256": value_sha256(identity),
             "freeze": authority["freeze"], "ledger_label_claim": label_row,
             "controller_qualified": False}
    write_json_exclusive(out / "claim.json", claim)
    states, actions, raw_actions, times = [], [], [], []
    failure_state = np.empty((0, 2 * (protocol.n_links + 1)), dtype=np.float64)
    invalid = None
    wall, cpu = time.monotonic(), time.process_time()
    with (out / "journal.jsonl").open("xb") as journal:
        try:
            controller = _controller(authority, identity)
            env = CanonicalEnv(protocol)
            initial = hanging_state(protocol, seed, exact=exact)
            states.append(env.reset(initial)); times.append(float(env.data.time))
            evidence._journal(journal, {"initial": states[0].tolist(), "time": times[0]})
            for tick in range(protocol.episode_ticks):
                raw = float(controller(states[-1].copy(), tick)); raw_actions.append(raw)
                state, delivered = env.step(raw)
                states.append(state.copy()); actions.append(delivered); times.append(float(env.data.time))
                evidence._journal(journal, {"tick": tick, "raw_action": raw, "action": delivered,
                                           "state": state.tolist(), "time": times[-1]})
                if abs(state[0]) > protocol.rail_limit_m:
                    break
        except Exception as error:
            invalid = f"{type(error).__name__}: {error}"
            if "env" in locals():
                failure_state = np.asarray(env.state, dtype=np.float64).reshape(1, -1)
            attempted = np.float64(raw_actions[-1]).tobytes().hex() if len(raw_actions) > len(actions) else None
            evidence._journal(journal, {"failure": invalid, "completed_commands": len(actions),
                                       "attempted_raw_action_f64_hex": attempted})
    states = np.asarray(states, dtype=np.float64).reshape(-1, 2 * (protocol.n_links + 1))
    actions = np.asarray(actions, dtype=np.float32)
    raw_actions, times = np.asarray(raw_actions, dtype=np.float64), np.asarray(times, dtype=np.float64)
    with (out / "tape.npz").open("xb") as stream:
        np.savez_compressed(stream, states=states, actions=actions, raw_actions=raw_actions,
                            times=times, failure_state=failure_state)
    score = score_tape(states, actions, protocol, invalid_reason=invalid) if len(states) else None
    try:
        validate_identity(identity)
        unchanged = runtime_metadata() == identity["runtime"]
    except (ValueError, OSError):
        unchanged = False
    return write_json_exclusive(out / "receipt.json", {
        "schema": "rage_mill.canonical_episode_evidence.v1", "stage": "reserved",
        "seed": seed, "exact": exact, "identity": identity, "identity_sha256": value_sha256(identity),
        "freeze": authority["freeze"], "recording": pinned_file(out / "tape.npz"),
        "journal": pinned_file(out / "journal.jsonl"), "claim": pinned_file(out / "claim.json"),
        "score": score, "invalid_reason": invalid, "source_pins_unchanged": unchanged,
        "wall_seconds": time.monotonic() - wall, "process_cpu_seconds": time.process_time() - cpu,
        "reset_count_after_initialization": 0, "initialized": bool(len(states)), "controller_qualified": False})


def run_reserved_gate(authority_row, gate):
    """Execute one fixed gate once, only under separately issued authority.

    Every label in this gate is durably consumed before the first factory import
    or reset. An incomplete gate cannot be resumed. Failed tapes stay in place;
    later gates require actual complete prior passes and cannot replace them.
    """
    authority, frozen, protocol, gate_row, labels = _claim_gate(authority_row, gate)
    out = _absolute(authority["output_root"]) / gate
    out.mkdir(parents=True, exist_ok=False)
    details = reserved_gates(protocol.n_links)[gate]
    identity = frozen["identity"]
    write_json_exclusive(out / "claim.json", {
        "schema": "rage_mill.canonical_reserved_cohort_claim.v1", "stage": "reserved",
        **details, "identity_sha256": value_sha256(identity), "freeze": authority["freeze"],
        "ledger_gate_claim": gate_row})
    rows = []
    for seed, label_row in labels.items():
        rows.append(_record_reserved_episode(authority, frozen, protocol, seed, details["exact"],
                                             out / f"episode-{seed}", label_row))
    runtime_report = record_runtime_execution(out / "runtime-receipt.json", identity=identity,
                                              runtime_closure=authority["runtime_closure"], outputs=rows)
    cohort = write_json_exclusive(out / "receipt.json", {
        "schema": "rage_mill.canonical_cohort_evidence.v1", "stage": "reserved", **details,
        "identity": identity, "identity_sha256": value_sha256(identity), "freeze": authority["freeze"],
        "episode_receipts": rows, "claim": pinned_file(out / "claim.json"),
        "campaign_authority": authority_row,
        "campaign_claim": pinned_file(_root(authority, protocol) / "campaign.json"),
        "ledger_gate_claim": gate_row, "runtime_execution": runtime_report,
        "controller_qualified": False})
    # Keep the report and all tapes if coverage fails, then make owner failure
    # explicit. A later gate cannot consume labels through an uncovered result.
    audit_runtime_execution(runtime_report, identity=identity,
                            runtime_closure=authority["runtime_closure"], outputs=rows)
    return cohort


def verify_reserved_release(authority_row, cohorts, video_row):
    """Read-only ledger + native-proof/tape/source/video release verification."""
    authority, frozen, protocol = _authority(authority_row, execution=False)
    require(isinstance(cohorts, dict) and set(cohorts) == set(GATE_ORDER),
            "all four fixed reserved cohorts required")
    for gate in GATE_ORDER:
        _audit_storage(authority_row, authority, protocol, gate, cohorts[gate])
    result = release.verify_release(authority["freeze"], cohorts, video_row)
    return {**result, "campaign_authority": authority_row,
            "campaign_claim": pinned_file(_root(authority, protocol) / "campaign.json"),
            "once_only_label_claims_checked": 141}
