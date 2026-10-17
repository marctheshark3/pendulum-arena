"""Treat only a positively audited frozen clone's inventory as source data.

Original absolute execution inputs remain transitive. A fresh clone contains
historical JSON descriptions, and is an inventory rather than a portable
execution identity. Its exact frozen member bytes are pinned without expansion.
"""
from __future__ import annotations

import json
from pathlib import Path
import runpy

from rage_mill.canonical_evaluation import pinned_file, read_pinned, require, validate_identity
from rage_mill.canonical_reserved import _authority

_V3 = runpy.run_path(str(Path(__file__).with_name("prepare_canonical18_reserved_owner_v3.py")),
                     run_name="canonical18_reserved_preparation_v3_metadata_only")
_ORIGINAL_PREPARE, _ORIGINAL_MAIN = _V3["_ORIGINAL_PREPARE"], _V3["_ORIGINAL_MAIN"]
strict_path = _V3["strict_path"]


class PinClosure(_V3["PinClosure"]):
    def __init__(self, initial, *, identity, authority_row):
        authority, frozen, protocol = _authority(authority_row, execution=False)
        require(protocol.n_links == 18 and validate_identity(identity).n_links == 18
                and frozen["identity"] == identity,
                "same positively audited own-count authority and frozen identity required")
        snapshot = frozen["source_snapshot"]
        self.clone = strict_path(snapshot["fresh_clone"], directory=True)
        self.clone_rows = {}
        for expected in snapshot["inventory"]:
            relative = Path(expected["relative_path"])
            require(relative.parts and not relative.is_absolute() and ".." not in relative.parts,
                    "safe relative frozen inventory member required")
            path = strict_path(self.clone / relative)
            require(str(path) not in self.clone_rows, "distinct frozen inventory member required")
            self.clone_rows[str(path)] = dict(path=str(path), sha256=expected["sha256"],
                                              bytes=expected["bytes"])
        require(self.clone_rows, "positive frozen clone inventory required")
        super().__init__(initial, identity=identity)
        for row in (authority_row, authority["freeze"], pinned_file(__file__)):
            self.include(row, expand=False)

    def include(self, row, *, expand=True):
        path = strict_path(row["path"])
        if path.is_relative_to(self.clone):
            require(self.clone_rows.get(str(path)) == {key: row[key] for key in ("path", "sha256", "bytes")},
                    "clone data must be an exact positively frozen inventory member")
            # read_pinned in the parent rechecks actual bytes on every inclusion.
            return super().include(row, expand=False)
        return super().include(row, expand=expand)


def prepare(**kwargs):
    identity_row = pinned_file(strict_path(kwargs["identity_path"]))
    authority_row = pinned_file(strict_path(kwargs["authority_path"]))
    require(identity_row["sha256"] == kwargs["identity_sha256"]
            and authority_row["sha256"] == kwargs["authority_sha256"],
            "explicit frozen identity and authority SHA required")
    identity = json.loads(read_pinned(identity_row))

    class BoundPinClosure(PinClosure):
        def __init__(self, initial):
            super().__init__(initial, identity=identity, authority_row=authority_row)

    previous = _ORIGINAL_PREPARE.__globals__["PinClosure"]
    _ORIGINAL_PREPARE.__globals__["PinClosure"] = BoundPinClosure
    try:
        return _ORIGINAL_PREPARE(**kwargs)
    finally:
        _ORIGINAL_PREPARE.__globals__["PinClosure"] = previous


def main():
    previous = _ORIGINAL_MAIN.__globals__["prepare"]
    _ORIGINAL_MAIN.__globals__["prepare"] = prepare
    try:
        _ORIGINAL_MAIN()
    finally:
        _ORIGINAL_MAIN.__globals__["prepare"] = previous


if __name__ == "__main__":
    main()
