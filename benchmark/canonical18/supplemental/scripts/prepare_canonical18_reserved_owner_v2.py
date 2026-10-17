"""Preserve strict pins while resolving already-bound descriptive source names.

The original preparer stays immutable. Relative source_hashes are accepted only
for a canonical repository file already present in the concrete candidate
identity and this closure with the same digest. This is metadata preparation;
it never activates an authority, initializes a plant or consumes a claim.
"""
from __future__ import annotations

import json
from pathlib import Path
import runpy

from rage_mill.canonical_evaluation import pinned_file, read_pinned, require, validate_identity

ROOT = Path(__file__).resolve().parents[1]
_ORIGINAL = runpy.run_path(str(Path(__file__).with_name("prepare_canonical18_reserved_owner.py")),
                          run_name="canonical18_reserved_preparation_v1_metadata_only")
_ORIGINAL_PREPARE = _ORIGINAL["prepare"]
_ORIGINAL_MAIN = _ORIGINAL["main"]
strict_path = _ORIGINAL["strict_path"]


class PinClosure(_ORIGINAL["PinClosure"]):
    """Resolve only same-identity sources that are already absolutely pinned."""
    def __init__(self, initial, *, identity):
        require(validate_identity(identity).n_links == 18, "own-count frozen identity required")
        super().__init__(initial)
        self.identity_sources = {row["path"]: row["sha256"] for row in identity["sources"]}

    def source_pin_path(self, value, digest):
        require(isinstance(value, str) and bool(value), "literal source pin path required")
        relative = Path(value)
        if relative.is_absolute():
            # Absolute owner/runtime evidence keeps the original alias/symlink
            # rejection and does not acquire any new normalization rule.
            return str(strict_path(value))
        require(relative.parts and ".." not in relative.parts,
                "repository-relative source traversal is forbidden")
        path = ROOT / relative
        require(path.is_relative_to(ROOT) and path.resolve(strict=True) == path
                and not path.is_symlink() and path.is_file(),
                "canonical non-symlink repository source required")
        canonical = str(path)
        require(self.identity_sources.get(canonical) == digest
                and self.pins.get(canonical) == digest,
                "relative source must match a frozen identity source already absolutely bound")
        require(pinned_file(path)["sha256"] == digest, "already-bound relative source bytes changed")
        return canonical

    def walk(self, value):
        if isinstance(value, dict):
            if set(value) >= {"path", "sha256", "bytes"}:
                self.include(value)
            else:
                for path, digest in value.get("source_hashes", {}).items():
                    canonical = self.source_pin_path(path, digest)
                    row = pinned_file(canonical)
                    require(row["sha256"] == digest, "predecessor source/runtime input changed")
                    self.include(row, expand=False)
                for key, item in value.items():
                    if key != "source_hashes":
                        self.walk(item)
        elif isinstance(value, list):
            for item in value:
                self.walk(item)


def prepare(**kwargs):
    identity_row = pinned_file(strict_path(kwargs["identity_path"]))
    require(identity_row["sha256"] == kwargs["identity_sha256"],
            "explicit frozen identity hash required before relative source resolution")
    identity = json.loads(read_pinned(identity_row))
    require(validate_identity(identity).n_links == 18, "same frozen own-count identity required")

    class BoundPinClosure(PinClosure):
        def __init__(self, initial):
            super().__init__(initial, identity=identity)

    # Change only this process's helper binding. All original preparation,
    # authority, ledger, output and argv checks execute unchanged from source.
    globals_before = _ORIGINAL_PREPARE.__globals__.get("PinClosure")
    _ORIGINAL_PREPARE.__globals__["PinClosure"] = BoundPinClosure
    try:
        return _ORIGINAL_PREPARE(**kwargs)
    finally:
        _ORIGINAL_PREPARE.__globals__["PinClosure"] = globals_before


def main():
    previous = _ORIGINAL_MAIN.__globals__["prepare"]
    _ORIGINAL_MAIN.__globals__["prepare"] = prepare
    try:
        _ORIGINAL_MAIN()
    finally:
        _ORIGINAL_MAIN.__globals__["prepare"] = previous


if __name__ == "__main__":
    main()
