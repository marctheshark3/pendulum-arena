"""Inspect preparatory own17 public source bytes; no policy or plant execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from public_support import ORIGINAL_IDENTITY_SHA256, inspect_tables, verify_export


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument("--expected-files-sha256", required=True)
    parser.add_argument("--tables", action="store_true", help="also inspect NPZ with NumPy; no MuJoCo needed")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    verified = verify_export(expected_files_sha256=args.expected_files_sha256)
    arrays = inspect_tables(expected_files_sha256=args.expected_files_sha256) if args.tables else None
    loaded = sorted(name for name in sys.modules if name == "rage_mill" or name.startswith("rage_mill."))
    if loaded or "mujoco" in sys.modules:
        raise ValueError("inspection must not import frozen factory or MuJoCo")
    report = dict(schema="pendulum_arena.canonical17_preparatory_source_audit.v1",
                  passed=True, preparation_only=True, qualification=False,
                  original_identity_sha256=ORIGINAL_IDENTITY_SHA256,
                  files_sha256=args.expected_files_sha256, files=verified["files"],
                  bytes=verified["bytes"], stored_arrays=arrays,
                  original_runtime_paths_unchanged=True, runtime_binary_hashes_verified=False,
                  original_identity_is_portable=False, loaded_rage_mill_modules=loaded,
                  factory_calls=0, controller_calls=0, native_models=0, native_commands=0,
                  reset_RNG_draws=0, new_reserved_labels=0)
    if args.out is not None:
        with args.out.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
