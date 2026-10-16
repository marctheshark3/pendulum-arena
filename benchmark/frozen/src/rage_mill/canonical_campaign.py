"""Identity and independent native tape checks for the canonical campaign."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

from .canonical_benchmark import CanonicalEnv, CanonicalProtocol, hanging_state, score_tape
from .canonical_control import CanonicalPolicy, CanonicalPlanningPolicy


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def source_identity(root):
    root = Path(root)
    names = ("src/rage_mill/canonical_benchmark.py", "src/rage_mill/canonical_control.py",
             "src/rage_mill/canonical_campaign.py", "scripts/evaluate_canonical_campaign.py",
             "configs/canonical15.json", "pyproject.toml", "uv.lock")
    return {name: file_sha256(root/name) for name in names}


def runtime_identity():
    import mujoco
    import scipy

    if mujoco.__version__ != "3.3.5":
        raise ValueError("canonical evaluation requires native MuJoCo 3.3.5")
    threads = {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
    if any(value != "1" for value in threads.values()):
        raise ValueError("frozen canonical runtime requires one BLAS/OMP thread per worker")
    libraries = {}
    for package in (mujoco, np, scipy):
        base = Path(package.__file__).parent
        for directory in (base, base.parent/f"{base.name}.libs"):
            for path in sorted(directory.rglob("*.so*")):
                if path.is_file():
                    libraries[str(path.relative_to(base.parent))] = file_sha256(path)
    return dict(python=sys.version, python_executable_sha256=file_sha256(sys.executable),
                machine=platform.machine(), platform=platform.platform(),
                packages={name: importlib.metadata.version(name) for name in ("numpy", "mujoco", "scipy")},
                thread_environment=threads, shared_libraries=libraries)


def candidate_identity(candidate):
    path = Path(candidate)
    identity = {name: file_sha256(path/name) for name in
                ("controller.npz", "policy.json", "model.xml", "provenance.json")}
    recorded = json.loads((path/"identity.json").read_text())
    if identity != recorded:
        raise ValueError("candidate files changed after preparation")
    return identity


def load_candidate(candidate):
    candidate_identity(candidate)
    path = Path(candidate)
    with np.load(path/"controller.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    parameters = json.loads((path/"policy.json").read_text())
    cls = CanonicalPlanningPolicy if "forecast_profiles" in parameters else CanonicalPolicy
    return cls(arrays, **parameters)


def audit_native_tape(states, actions, raw_actions, seed, exact, invalid_reason=None):
    p = CanonicalProtocol()
    score = score_tape(states, actions, p, invalid_reason=invalid_reason)
    initial_gap = float(np.max(np.abs(states[0]-hanging_state(p, seed, exact=exact))))
    delivered = np.clip(np.asarray(raw_actions, dtype=np.float32), -1., 1.).astype(float)
    raw_delivery_ok = bool(np.array_equal(actions, delivered))
    env = CanonicalEnv(p)
    maximum_defect = 0.
    for tick, action in enumerate(actions):
        env.reset(states[tick])
        following, _ = env.step(action)
        maximum_defect = max(maximum_defect, float(np.max(np.abs(following-states[tick+1]))))
    integrity = bool(np.isfinite(maximum_defect) and maximum_defect <= 1e-8
                     and np.isfinite(initial_gap) and initial_gap <= 1e-8 and raw_delivery_ok)
    return dict(**score, initial_state_gap=initial_gap, native_maximum_dynamics_defect=maximum_defect,
                raw_delivery_ok=raw_delivery_ok, trajectory_integrity=integrity,
                admitted=bool(score["success"] and integrity))
