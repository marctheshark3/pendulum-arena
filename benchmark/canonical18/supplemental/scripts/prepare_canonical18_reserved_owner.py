"""Write one reserved or saved-proof recipe; never activate a benchmark claim.

All completed reserved predecessor owners are supplied explicitly. Generic
source/freeze audits run with execution=False; only the emitted worker can
activate the separately pinned authority, and only in reserved mode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rage_mill.canonical_evaluation import (
    pinned_file, read_pinned, require, reserved_gates, validate_identity,
    value_sha256, write_json_exclusive,
)
from rage_mill.canonical_release import audit_cohort
from rage_mill.canonical_reserved import GATE_ORDER, _anchor, _audit_storage, _authority, _root
from rage_mill.canonical_runtime_manifest import validate_runtime_manifest


def strict_path(value, *, exists=True, directory=False):
    path = Path(value)
    require(path.is_absolute() and path.resolve(strict=exists) == path and not path.is_symlink(),
            "canonical absolute non-symlink path required")
    if exists:
        require(path.is_dir() if directory else path.is_file(), "regular input path required")
    else:
        require(not path.exists() and not path.is_symlink(), "fresh owner/output path required")
        strict_path(path.parent, directory=True)
    return path


class PinClosure:
    """Bind execution roots while keeping historical inventories as data bytes."""
    def __init__(self, initial):
        self.pins, self.seen = dict(initial), set()

    def include(self, row, *, expand=True):
        strict_path(row["path"])
        data = read_pinned(row)
        require(row["path"] not in self.pins or self.pins[row["path"]] == row["sha256"],
                "conflicting transitive input pin")
        self.pins[row["path"]] = row["sha256"]
        if not expand:
            return
        if row["path"] in self.seen:
            return
        self.seen.add(row["path"])
        if Path(row["path"]).suffix == ".json":
            value = json.loads(data)
            # The initial cross-count inventory records immutable historical
            # paths. It is evidence, not a new transitive execution root.
            if not (isinstance(value, dict) and value.get("schema") in {
                    "rage_mill.canonical18_initial_inventory.v1",
                    "rage_mill.canonical18.initial_inventory.v1"}):
                self.walk(value)

    def walk(self, value):
        if isinstance(value, dict):
            if set(value) >= {"path", "sha256", "bytes"}:
                self.include(value)
            else:
                for path, digest in value.get("source_hashes", {}).items():
                    strict_path(path)
                    row = pinned_file(path)
                    require(row["sha256"] == digest, "predecessor source/runtime input changed")
                    # Existing owners already declare a complete flat map.
                    # Rehash its entries without expanding archived JSON.
                    self.include(row, expand=False)
                for key, item in value.items():
                    if key != "source_hashes":
                        self.walk(item)
        elif isinstance(value, list):
            for item in value:
                self.walk(item)

    def snapshot(self, snapshot):
        clone = strict_path(snapshot["fresh_clone"], directory=True)
        self.include(snapshot["bundle"])
        for expected in snapshot["inventory"]:
            relative = Path(expected["relative_path"])
            require(not relative.is_absolute() and ".." not in relative.parts,
                    "relative clone inventory path required")
            path = strict_path(clone / relative)
            row = pinned_file(path)
            require(row["sha256"] == expected["sha256"] and row["bytes"] == expected["bytes"],
                    "fresh-clone inventory bytes changed")
            # A clean clone inventories historical source bytes. Its JSON may
            # describe superseded execution paths and is not an execution root.
            self.include(row, expand=False)


def native_caps(mode, episodes):
    require(mode in ("reserved", "proof") and type(episodes) is int and episodes > 0,
            "typed positive gate size and owner mode required")
    proof = mode == "proof"
    return dict(maximum_outer_shadow_native_maps=650 * episodes,
        maximum_outer_shadow_inner_native_candidates=10400 * episodes,
        maximum_live_inner_native_candidates=0 if proof else 10400 * episodes,
        maximum_recorded_input_inner_native_candidates=10400 * episodes if proof else 0,
        maximum_total_inner_native_candidates=20800 * episodes,
        maximum_actual_physical_commands=0 if proof else 1500 * episodes,
        maximum_saved_native_proof_maps=3000 * episodes if proof else 0,
        maximum_native_commands=(24450 if proof else 22950) * episodes)


def worker_source(identity, name):
    rows = [row for row in identity["sources"] if Path(row["path"]).name == name]
    require(len(rows) == 1, "one frozen worker source required: " + name)
    read_pinned(rows[0])
    return rows[0]


def reserved_argv(launcher, worker, authority_row, gate):
    return [launcher, worker["path"], "--authority", authority_row["path"],
            "--expected-authority-sha256", authority_row["sha256"], "--gate", gate]


def checked_gate_owner(directory, *, authority_row, authority, identity, launcher, worker, gate, closure, supervisor):
    directory = strict_path(directory, directory=True)
    rows = {key: pinned_file(directory / name) for key, name in
            (("launch", "launch.json"), ("completion", "completion.json"))}
    launch, completion = (json.loads(read_pinned(rows[key])) for key in ("launch", "completion"))
    details = reserved_gates(18)[gate]
    require(launch.get("argv") == reserved_argv(launcher, worker, authority_row, gate)
            and launch.get("output_path") == str(Path(authority["output_root"]) / gate)
            and launch.get("runtime_closure") == authority["runtime_closure"]
            and launch.get("authority") == authority_row
            and launch.get("identity_sha256") == value_sha256(identity),
            "original reserved owner authority/gate/argv/output lineage differs")
    require(completion.get("recipe_path") == rows["launch"]["path"]
            and completion.get("recipe_sha256") == rows["launch"]["sha256"]
            and completion.get("naturally_closed_zero") is True
            and type(completion.get("child_returncode")) is int and completion["child_returncode"] == 0
            and completion.get("failure") is None and completion.get("interrupted") is False
            and completion.get("wall_timeout") is False and completion.get("recipe_unchanged") is True
            and completion.get("owned_group_cleanup")
            and completion.get("supervisor_source_sha256") == supervisor["sha256"],
            "naturally closed original reserved owner with same reviewed supervisor required")
    require(all(type(launch.get(key)) is int and launch[key] > 0 for key in ("CPU_cap_s", "wall_cap_s")),
            "bounded original reserved owner required")
    pins, unchanged = launch.get("source_hashes"), completion.get("source_hashes_unchanged")
    require(isinstance(pins, dict) and isinstance(unchanged, dict) and set(pins) == set(unchanged)
            and all(flag is True for flag in unchanged.values())
            and all(pins.get(path) == digest for path, digest in
                    validate_runtime_manifest(json.loads(read_pinned(authority["runtime_closure"])),
                                              require_complete=True).items())
            and all(pins.get(row["path"]) == row["sha256"] for row in
                    [identity["controller"], identity["configuration"], *identity["sources"]])
            and all(pins.get(row["path"]) == row["sha256"] for row in [authority_row, authority["freeze"]]),
            "original reserved owner prospective source/runtime pins incomplete")
    require(all(type(launch.get(key)) is int and launch[key] == value
                for key, value in native_caps("reserved", details["episodes"]).items()),
            "original reserved owner native caps differ")
    log = pinned_file(directory / "worker.log")
    require(completion.get("log_path") == log["path"] and completion.get("log_sha256") == log["sha256"],
            "original reserved owner log changed")
    for row in [*rows.values(), log]:
        closure.include(row)
    return dict(gate=gate, **rows, aggregate_child_CPU_seconds=completion["aggregate_child_CPU_seconds"],
                wall_seconds=completion["wall_seconds"], maximum_native_commands=22950 * details["episodes"])


def prepare(*, authority_path, authority_sha256, identity_path, identity_sha256,
            supervisor_path, gate, mode, cpu_seconds, wall_seconds, out, gate_owners):
    require(type(cpu_seconds) is int and cpu_seconds > 0
            and type(wall_seconds) is int and wall_seconds > 0, "positive integer CPU/wall caps required")
    require(gate in GATE_ORDER and mode in ("reserved", "proof"), "fixed gate and owner mode required")
    authority_row = pinned_file(strict_path(authority_path))
    identity_row = pinned_file(strict_path(identity_path))
    supervisor = pinned_file(strict_path(supervisor_path))
    require(authority_row["sha256"] == authority_sha256 and identity_row["sha256"] == identity_sha256,
            "explicit authority and identity input hashes required")
    authority, frozen, protocol = _authority(authority_row, execution=False)
    identity = json.loads(read_pinned(identity_row))
    require(protocol.n_links == 18 and validate_identity(identity).n_links == 18
            and identity == frozen["identity"] and value_sha256(identity) == authority["identity_sha256"],
            "same frozen own-count eighteen authority/identity required")
    runtime = authority["runtime_closure"]
    closure = PinClosure(validate_runtime_manifest(json.loads(read_pinned(runtime)), require_complete=True))
    for row in [authority_row, identity_row, supervisor, pinned_file(__file__),
                identity["controller"], identity["configuration"], *identity["sources"]]:
        closure.include(row)
    closure.walk(frozen)
    closure.snapshot(frozen["source_snapshot"])
    calibration = json.loads(read_pinned(frozen["runtime_owners"]["calibration"]["launch"]))
    calibration_completion = json.loads(read_pinned(frozen["runtime_owners"]["calibration"]["completion"]))
    require(supervisor["sha256"] == calibration_completion["supervisor_source_sha256"],
            "same previously reviewed bounded supervisor required")
    launcher = calibration["argv"][0]
    require(Path(launcher).is_absolute() and Path(launcher).is_file()
            and str(Path(launcher).resolve(strict=True)) == identity["runtime"]["executable"],
            "original formal-owner Python launcher required")
    # Preserve the original virtualenv argv for package discovery. Its real
    # executable was checked against the frozen runtime immediately above.
    closure.include(pinned_file(Path(launcher).resolve(strict=True)))
    actual_worker = worker_source(identity, "run_canonical_reserved_gate.py")
    proof_worker = worker_source(identity, "audit_canonical18_saved_policy.py")
    details = reserved_gates(18)[gate]
    required = set(GATE_ORDER[:GATE_ORDER.index(gate) + (mode == "proof")])
    supplied = {}
    for name, directory in gate_owners:
        require(name in GATE_ORDER and name not in supplied, "one explicit original owner per predecessor gate required")
        supplied[name] = directory
    require(set(supplied) == required, "explicit owners required for all prior gates and current proof gate only")
    previous, cohorts = [], {}
    for name in GATE_ORDER:
        if name not in required:
            continue
        previous.append(checked_gate_owner(supplied[name], authority_row=authority_row,
            authority=authority, identity=identity, launcher=launcher, worker=actual_worker,
            gate=name, closure=closure, supervisor=supervisor))
        cohort = pinned_file(Path(authority["output_root"]) / name / "receipt.json")
        _audit_storage(authority_row, authority, protocol, name, cohort)
        expected = reserved_gates(18)[name]
        audit_cohort(cohort, identity=identity, count=expected["episodes"], exact=expected["exact"],
                     seed_start=expected["seed_start"], stage="reserved", freeze=authority["freeze"])
        closure.include(cohort); cohorts[name] = cohort
    out = strict_path(out, exists=False)
    claim_base, campaign_output = Path(authority["claim_base"]), Path(authority["output_root"])
    require(not (out == claim_base or out.is_relative_to(claim_base) or claim_base.is_relative_to(out)
                 or out == campaign_output or out.is_relative_to(campaign_output) or campaign_output.is_relative_to(out)),
            "owner directory must be separate from permanent claims and actual episode outputs")
    if mode == "reserved":
        ledger = _root(authority, protocol)
        if ledger.exists() or ledger.is_symlink():
            strict_path(ledger, directory=True)
            campaign = pinned_file(ledger / "campaign.json")
            require(json.loads(read_pinned(campaign)) == _anchor(authority_row, authority, protocol),
                    "existing campaign differs; no rekey or repair")
            closure.include(campaign)
        else:
            require(not campaign_output.exists() and not campaign_output.is_symlink(),
                    "fresh campaign output required before first reservation")
        require(not (ledger / ("gate-" + gate)).exists()
                and not (ledger / ("gate-" + gate)).is_symlink(), "gate already consumed; no retry")
        for seed in range(details["seed_start"], details["seed_start"] + details["episodes"]):
            label = ledger / f"seed-{seed}"
            require(not label.exists() and not label.is_symlink(), "reserved label already consumed; no replacement")
        output = campaign_output / gate
        require(not output.exists() and not output.is_symlink() and output.resolve() == output,
                "fresh actual gate output required")
        argv = reserved_argv(launcher, actual_worker, authority_row, gate)
    else:
        output = out / "result"
        cohort = cohorts[gate]
        argv = [launcher, proof_worker["path"], "--identity", identity_row["path"],
            "--expected-identity-sha256", identity_row["sha256"], "--cohort", cohort["path"],
            "--expected-cohort-sha256", cohort["sha256"], "--runtime-closure", runtime["path"], "--out", str(output)]
    recipe = dict(schema="rage_mill.next_stage_supervisor_recipe.v1", argv=argv, cwd=str(out),
        environment=dict(PYTHONPATH=str(Path(actual_worker["path"]).parents[1] / "src"),
            PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        CPU_cap_s=cpu_seconds, wall_cap_s=wall_seconds, address_cap_bytes=8 * 1024**3,
        output_path=str(output), log=str(out / "worker.log"), receipt=str(out / "completion.json"),
        source_hashes=dict(sorted(closure.pins.items())), authority=authority_row,
        supervisor=supervisor,
        identity=identity_row, identity_sha256=value_sha256(identity), runtime_closure=runtime,
        gate=gate, mode=mode, gate_details=details, original_gate_owners=previous,
        purpose=("Actual once-only eighteen reserved gate" if mode == "reserved" else
                 "Frozen policy/native proof on original saved reserved tapes only"),
        physical_episodes=details["episodes"] if mode == "reserved" else 0,
        original_saved_episodes=details["episodes"] if mode == "proof" else 0,
        new_reserved_labels=details["episodes"] if mode == "reserved" else 0,
        **native_caps(mode, details["episodes"]))
    if mode == "proof":
        recipe.update(maximum_reset_RNG_draws=0, maximum_recorded_reset_reconstructions=0,
                      maximum_physical_resets=0, new_reserved_claims=0)
    out.mkdir(exist_ok=False)
    return write_json_exclusive(out / "launch.json", recipe)


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    for name in ("authority", "identity", "supervisor", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("authority", "identity"):
        parser.add_argument("--expected-" + name + "-sha256", required=True)
    parser.add_argument("--gate", choices=GATE_ORDER, required=True)
    parser.add_argument("--mode", choices=("reserved", "proof"), required=True)
    parser.add_argument("--cpu-seconds", type=int, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--gate-owner", nargs=2, action="append", default=[], metavar=("GATE", "DIRECTORY"))
    args = parser.parse_args()
    row = prepare(authority_path=args.authority, authority_sha256=args.expected_authority_sha256,
        identity_path=args.identity, identity_sha256=args.expected_identity_sha256,
        supervisor_path=args.supervisor, gate=args.gate, mode=args.mode,
        cpu_seconds=args.cpu_seconds, wall_seconds=args.wall_seconds, out=args.out, gate_owners=args.gate_owner)
    print(json.dumps({"launch": row, "executed": False, "claims_created": 0}))


if __name__ == "__main__":
    main()
