"""Package checked own-count prerequisites into a fresh source/runtime freeze.

This read-only evidence audit and source copier creates no authority and no
claims. The original absolute execution paths stay authoritative; the clean
clone inventories source bytes and cannot be used as a relocated runtime.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from rage_mill.canonical_evaluation import (
    pinned_file, read_pinned, reserved_gates, require, validate_identity,
    value_sha256, write_json_exclusive,
)
from rage_mill.canonical_release import audit_cohort, audit_exact_source_calibration, audit_freeze
from rage_mill.canonical_runtime_evidence import audit_formal_runtime_owners
from rage_mill.canonical18_evidence import audit_owner


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--calibration-owner", required=True, type=Path)
    parser.add_argument("--noisy20-owner", required=True, type=Path)
    parser.add_argument("--noisy100-owner", required=True, type=Path)
    parser.add_argument("--independent-reviews", required=True, nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    prepared = json.loads((args.candidate / "preparation.json").read_text())
    identity_row = prepared["identity"]
    identity = json.loads(read_pinned(identity_row))
    require(validate_identity(identity).n_links == 18, "own-count eighteen identity required")
    declaration_row = identity["policy_parameters"]["design_metadata"]["declaration"]
    declaration = json.loads(read_pinned(declaration_row))
    calibration = pinned_file(args.calibration_owner / "result/source-calibration.json")
    audit_exact_source_calibration(calibration, identity)
    development, owners, components = {}, {}, {}
    for stage, directory in (("calibration", args.calibration_owner),
                             ("noisy20", args.noisy20_owner), ("noisy100", args.noisy100_owner)):
        declared = ({"seed_start": declaration["exact_calibration_seed"], "episodes": 1}
                    if stage == "calibration" else
                    declaration["development20" if stage == "noisy20" else "development100"])
        checked = audit_owner(directory, identity=identity, runtime_closure=prepared["runtime_closure"],
                              seed_start=declared["seed_start"], episodes=declared["episodes"],
                              exact=stage == "calibration")
        owners[stage] = checked["owner"]
        components[stage] = {name: checked[name] for name in ("component", "cohort", "profile_decisions")}
        if stage != "calibration":
            declared = declaration["development20" if stage == "noisy20" else "development100"]
            cohort = pinned_file(directory / "result/episodes/receipt.json")
            audit_cohort(cohort, identity=identity, count=declared["episodes"], exact=False,
                         seed_start=declared["seed_start"], stage="development")
            development[stage] = {"seed_start": declared["seed_start"], "receipt": cohort}
    reviews = [pinned_file(path) for path in args.independent_reviews]
    for review in reviews:
        read_pinned(review)
    frozen = {
        "schema": "rage_mill.canonical_release_freeze.v1", "identity": identity,
        "identity_sha256": value_sha256(identity), "activation_authorized": False,
        "controller_qualified": False, "source_calibration": calibration,
        "development": development, "runtime_closure": prepared["runtime_closure"],
        "runtime_owners": owners, "reserved_gates": reserved_gates(18),
        "scientific_components": components,
        "requires_prerequisite_release": False, "independent_reviews": reviews,
    }
    # Reject failed/mutated owners before copying any source package.
    audit_formal_runtime_owners(frozen)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    source.mkdir()
    paths = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    ).decode().split("\0")
    for relative in sorted(set(filter(None, paths))):
        path = root / relative
        require(path.is_file() and not path.is_symlink(), "regular source inventory file required")
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    # Every identity-required external byte sequence gets a tracked package row.
    mapping = []
    for index, row in enumerate([identity["controller"], identity["configuration"], *identity["sources"]]):
        original = Path(row["path"])
        if original.is_relative_to(root):
            relative = original.relative_to(root)
        else:
            relative = Path("frozen/external") / f"{index:03d}-{row['sha256'][:12]}-{original.name}"
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(read_pinned(row))
        require(pinned_file(source / relative)["sha256"] == row["sha256"], "snapshot source changed")
        mapping.append({"original": row, "snapshot_relative_path": str(relative)})
    write_json_exclusive(source / "CANONICAL18_IDENTITY_SOURCE_MAPPING.json", {
        "schema": "rage_mill.canonical18_source_mapping.v1", "identity": identity_row,
        "execution_paths_relocated": False, "fresh_clone_scope": "source inventory only",
        "mapping": mapping,
    })
    git(source, "init", "-b", "canonical18-source")
    git(source, "add", "-f", ".")
    git(source, "-c", "user.name=Rage Industries", "-c", "user.email=marc@neuralyogi.com",
        "commit", "-m", "Freeze canonical eighteen source and prerequisite identity bytes")
    commit = git(source, "rev-parse", "HEAD")
    bundle = out / "source.bundle"
    git(source, "bundle", "create", str(bundle), "HEAD")
    clone = out / "fresh-clone"
    subprocess.run(["git", "clone", "--no-hardlinks", str(bundle), str(clone)], check=True)
    inventory = []
    for relative in git(clone, "ls-files").splitlines():
        row = pinned_file(clone / relative)
        inventory.append({"relative_path": relative, "sha256": row["sha256"], "bytes": row["bytes"]})
    frozen["source_snapshot"] = {"bundle": pinned_file(bundle), "fresh_clone": str(clone),
                                 "git_commit": commit, "inventory": inventory}
    freeze = write_json_exclusive(out / "freeze.json", frozen)
    audit_freeze(freeze)
    audit_formal_runtime_owners(frozen)
    receipt = write_json_exclusive(out / "preparation.json", {
        "schema": "rage_mill.canonical18_release_preparation.v1", "freeze": freeze,
        "identity": identity_row, "source_files": len(inventory), "all_required_bytes_preserved": True,
        "clean_source_clone_verified": True, "runtime_owners_reaudited": True,
        "activation_authorized": False, "reserved_claims_created": 0,
        "physical_episodes": 0, "native_commands": 0,
    })
    print(json.dumps({"freeze": freeze, "preparation": receipt}))


if __name__ == "__main__":
    main()
