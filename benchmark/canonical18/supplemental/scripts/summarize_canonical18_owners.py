"""Hash and total every actual eighteen bounded owner, including failures."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rage_mill.canonical_evaluation import pinned_file, write_json_exclusive


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    owners = []
    for path in sorted(root.rglob("completion.json")):
        document = json.loads(path.read_text())
        recipe = document.get("recipe_path")
        # Copied historical/source-inventory receipts are never new owners.
        if (document.get("schema") != "rage_mill.next_stage_supervised_completion.v1"
                or not isinstance(recipe, str) or Path(recipe).parent != path.parent.resolve()):
            continue
        cpu, wall = document["aggregate_child_CPU_seconds"], document["wall_seconds"]
        if not all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                   for value in (cpu, wall)):
            raise ValueError("finite complete owner costs required")
        owners.append({
            "owner": str(path.parent.relative_to(root)), "launch": pinned_file(recipe),
            "completion": pinned_file(path), "aggregate_child_CPU_seconds": cpu,
            "wall_seconds": wall, "peak_rss_kib": document.get("peak_rss_kib"),
            "naturally_closed_zero": document.get("naturally_closed_zero"),
            "child_returncode": document.get("child_returncode"),
            "all_source_pins_unchanged": all(document.get("source_hashes_unchanged", {}).values()),
            "wall_timeout": document.get("wall_timeout"), "failure": document.get("failure"),
            "owned_group_cleanup": document.get("owned_group_cleanup"),
        })
    row = write_json_exclusive(args.out, {
        "schema": "rage_mill.canonical18_owner_cost_inventory.v1", "root": str(root),
        "owners": owners, "closed_owners": len(owners),
        "aggregate_child_CPU_seconds": sum(row["aggregate_child_CPU_seconds"] for row in owners),
        "sum_owner_wall_seconds": sum(row["wall_seconds"] for row in owners),
        "natural_zero_owners": sum(row["naturally_closed_zero"] is True for row in owners),
        "all_source_pins_unchanged": all(row["all_source_pins_unchanged"] for row in owners),
        "scope": "All original bounded completion receipts under this evidence root; copied receipts excluded. Includes negative experiments. Setup outside owners is reported separately. Wall times may overlap.",
        "physical_episodes_executed_by_this_inventory": 0, "native_commands": 0,
    })
    print(json.dumps({"receipt": row, "owners": len(owners)}))


if __name__ == "__main__":
    main()
