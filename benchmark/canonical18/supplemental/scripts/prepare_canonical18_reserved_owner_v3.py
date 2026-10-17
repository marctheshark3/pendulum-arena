"""Keep old permanent claim inventories as data under strict v2 source pins.

Runtime manifests from saved evidence inventory the immutable sixteen/seventeen
claim bytes. Those legacy claims describe older authorities, not execution
inputs for eighteen. Only exact known claim schemas at canonical permanent
ledger paths with their original published count16/count17 namespace are opaque.
Current eighteen owners and authorities retain the original transitive guards.
"""
from __future__ import annotations

import json
from pathlib import Path
import runpy

from rage_mill.canonical_evaluation import (
    pinned_file, read_pinned, require, reserved_gates, validate_identity,
)

_V2 = runpy.run_path(str(Path(__file__).with_name("prepare_canonical18_reserved_owner_v2.py")),
                     run_name="canonical18_reserved_preparation_v2_metadata_only")
_ORIGINAL_PREPARE, _ORIGINAL_MAIN = _V2["_ORIGINAL_PREPARE"], _V2["_ORIGINAL_MAIN"]
strict_path = _V2["strict_path"]
CLAIMS = Path("/home/marctheshark/.t3/artifacts/rage-mill-canonical-reserved-claims")
PUBLIC = "b84c6413974d5c0831cdf261ce15e58ba65cd3f5"
LEGACY_SCHEMAS = {
    "rage_mill.canonical_reserved_campaign_claim.v1",
    "rage_mill.canonical_reserved_gate_claim.v1",
    "rage_mill.canonical_reserved_label_claim.v1",
}


class PinClosure(_V2["PinClosure"]):
    def __init__(self, initial, *, identity):
        super().__init__(initial, identity=identity)
        # All additive metadata arithmetic is prospectively bound in recipes
        # produced by this class, although no helper changes the live identity.
        for name in ("prepare_canonical18_reserved_owner_v2.py", Path(__file__).name):
            self.include(pinned_file(Path(__file__).with_name(name)), expand=False)

    def include(self, row, *, expand=True):
        path = strict_path(row["path"])
        if expand and path.suffix == ".json" and path.is_relative_to(CLAIMS):
            relative = path.relative_to(CLAIMS)
            if (len(relative.parts) >= 3 and relative.parts[0] == "published-" + PUBLIC
                    and relative.parts[1] in ("count-16", "count-17")):
                value = json.loads(read_pinned(row))
                count = int(relative.parts[1].split("-")[1])
                namespace = dict(schema="rage_mill.canonical_reserved_namespace.v1",
                    source_commit=PUBLIC, n_links=count, gates=reserved_gates(count))
                require(isinstance(value, dict) and value.get("schema") in LEGACY_SCHEMAS
                    and value.get("namespace") == namespace,
                    "only original published sixteen/seventeen permanent claim inventories are opaque")
                remainder = relative.parts[2:]
                campaign = value["schema"] == "rage_mill.canonical_reserved_campaign_claim.v1"
                gate = value["schema"] == "rage_mill.canonical_reserved_gate_claim.v1"
                label = value["schema"] == "rage_mill.canonical_reserved_label_claim.v1"
                valid = campaign and remainder == ("campaign.json",)
                if gate:
                    valid = (len(remainder) == 2 and remainder[1] == "claim.json"
                        and value.get("gate") in reserved_gates(count)
                        and remainder[0] == "gate-" + value["gate"])
                if label:
                    seed = value.get("seed")
                    details = reserved_gates(count).get(value.get("gate"))
                    valid = (type(seed) is int and details is not None
                        and details["seed_start"] <= seed < details["seed_start"] + details["episodes"]
                        and remainder == (f"seed-{seed}", "claim.json"))
                require(valid, "legacy claim schema, published label and canonical ledger path differ")
                return super().include(row, expand=False)
        return super().include(row, expand=expand)


def prepare(**kwargs):
    identity_row = pinned_file(strict_path(kwargs["identity_path"]))
    require(identity_row["sha256"] == kwargs["identity_sha256"], "explicit frozen identity SHA required")
    identity = json.loads(read_pinned(identity_row))
    require(validate_identity(identity).n_links == 18, "own-count frozen identity required")

    class BoundPinClosure(PinClosure):
        def __init__(self, initial):
            super().__init__(initial, identity=identity)

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
