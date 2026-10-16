"""Preload MuJoCo's import dependencies before the metadata-only runtime freeze.

The original preparation and failed first closure stay unchanged. Importing
MuJoCo loads its GLFW/typing and native-library dependencies without constructing
a model, drawing reset noise or issuing a physics command. Both preparation
entrypoints belong to the new identity.
"""
from __future__ import annotations

from pathlib import Path
import runpy


def main():
    import mujoco  # noqa: F401 -- bind dependencies before manifest capture.

    source = Path(__file__).with_name("prepare_canonical17_native_value_candidate.py")
    entry = runpy.run_path(str(source), run_name="metadata_preparation_v2")
    prepare = entry["main"]
    prepare.__globals__["SCRIPTS"] = (*entry["SCRIPTS"], Path(__file__).name)
    prepare()


if __name__ == "__main__":
    main()
