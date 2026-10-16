"""Read-only release evidence checks for the published count-aware benchmark.

These functions cannot evaluate reserved episodes or activate a controller.
They rehash and rescore stored tapes rather than accepting cached pass flags.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess

from .canonical_evaluation import (
    audit_episode, is_reserved, read_pinned, require, reserved_gates,
    validate_identity, value_sha256,
)
from .canonical_replay import audit_native_replay_proof


def _read(row):
    return json.loads(read_pinned(row))


def audit_cohort(row, *, identity, count, exact, seed_start, stage, freeze=None):
    document = _read(row)
    require(document.get("schema") == "rage_mill.canonical_cohort_evidence.v1", "canonical cohort required")
    require(document.get("identity") == identity
            and document.get("identity_sha256") == value_sha256(identity), "cohort identity differs")
    require(document.get("stage") == stage and type(document.get("episodes")) is int
            and document["episodes"] == count and type(document.get("seed_start")) is int
            and document["seed_start"] == seed_start and document.get("exact") is exact,
            "cohort size, stage or reset distribution differs")
    rows = document.get("episode_receipts")
    require(isinstance(rows, list) and len(rows) == count
            and len({x["path"] for x in rows}) == count, "one distinct episode receipt per seed required")
    claim = _read(document["claim"])
    require(claim.get("schema") == f"rage_mill.canonical_{stage}_cohort_claim.v1"
            and claim.get("stage") == stage and claim.get("seed_start") == seed_start
            and claim.get("episodes") == count and claim.get("exact") is exact
            and claim.get("identity_sha256") == value_sha256(identity), "cohort ownership claim differs")
    if stage == "reserved":
        require(document.get("freeze") == freeze, "reserved cohort must bind its prospective freeze")
        require(claim.get("freeze") == freeze,
                "reserved cohort ownership claim differs")
    outcomes = []
    for seed, episode in zip(range(seed_start, seed_start + count), rows):
        checked = audit_episode(episode, identity=identity, expected_seed=seed, expected_exact=exact)
        require(checked["document"].get("stage") == stage, "episode stage differs")
        require(checked["score"] is not None and checked["score"]["success"] is True,
                "physical episode failed; no replacement allowed")
        if stage == "reserved":
            require(checked["document"].get("freeze") == freeze, "episode predates or differs from freeze")
        outcomes.append({"seed": seed, "receipt": episode, "score": checked["score"],
                         "recording": checked["document"]["recording"]})
    return outcomes


def audit_exact_source_calibration(row, identity):
    source = _read(row)
    require(source.get("schema") == "rage_mill.canonical_exact_source_calibration.v1",
            "exact source calibration evidence required")
    require(source.get("identity_sha256") == value_sha256(identity), "calibration policy differs")
    require(source.get("trajectory_integrity") is True and source.get("is_feasible") is True,
            "calibration trajectory integrity and feasibility required")
    for field in ("maximum_dynamics_defect", "initial_state_gap"):
        value = source.get(field)
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1e-8,
                "published source calibration defect/initial gap exceeds 1e-8")
    episode = audit_episode(source["episode_receipt"], identity=identity, expected_exact=True)
    require(episode["score"] is not None and episode["score"]["success"] is True,
            "source calibration did not complete physical task")
    require(isinstance(source.get("native_replay_proof"), dict), "pinned native source replay proof required")
    native = audit_native_replay_proof(source["native_replay_proof"], episode_receipt=source["episode_receipt"])
    require(native["continuous_replay_identical"] is True
            and native["maximum_dynamics_defect"] <= 1e-8 and native["initial_state_gap"] <= 1e-8,
            "actual native source replay failed calibration")
    require(all(source[field] == native[field] for field in ("maximum_dynamics_defect", "initial_state_gap")),
            "source calibration metadata differs from actual native proof")
    return source


def source_calibration_from_proof(episode_receipt, native_replay_proof):
    """Construct evidence from checked actual arrays, without another rollout."""
    episode = audit_episode(episode_receipt, expected_exact=True)
    require(episode["score"] is not None and episode["score"]["success"] is True,
            "complete exact physical calibration required")
    native = audit_native_replay_proof(native_replay_proof, episode_receipt=episode_receipt)
    require(native["continuous_replay_identical"] is True
            and native["maximum_dynamics_defect"] <= 1e-8 and native["initial_state_gap"] <= 1e-8,
            "actual native source replay failed calibration")
    return {"schema": "rage_mill.canonical_exact_source_calibration.v1",
            "identity_sha256": episode["document"]["identity_sha256"],
            "episode_receipt": episode_receipt, "native_replay_proof": native_replay_proof,
            "trajectory_integrity": True, "is_feasible": True,
            "maximum_dynamics_defect": native["maximum_dynamics_defect"],
            "initial_state_gap": native["initial_state_gap"]}


def audit_source_snapshot(snapshot, identity):
    """Check a real bundle, its clean fresh clone and tracked source inventory."""
    read_pinned(snapshot["bundle"])
    clone = Path(snapshot["fresh_clone"]).resolve()
    commit = snapshot.get("git_commit")
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit), "frozen Git commit required")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(clone), *args], text=True).strip()

    require(git("rev-parse", "HEAD") == commit, "fresh clone commit differs")
    require(git("status", "--porcelain=v1", "--untracked-files=all") == "", "fresh clone must remain clean")
    origin = subprocess.run(["git", "-C", str(clone), "config", "--get", "remote.origin.url"],
                            capture_output=True, text=True)
    require(origin.returncode == 0
            and Path(origin.stdout.strip()).resolve() == Path(snapshot["bundle"]["path"]).resolve(),
            "fresh clone must originate from this frozen local source bundle")
    heads = git("bundle", "list-heads", str(Path(snapshot["bundle"]["path"]).resolve()))
    require(any(line.split()[0] == commit for line in heads.splitlines()), "bundle does not contain frozen head")
    subprocess.run(["git", "-C", str(clone), "bundle", "verify", snapshot["bundle"]["path"]],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    inventory = snapshot.get("inventory")
    require(isinstance(inventory, list) and inventory, "frozen source inventory required")
    seen, digests = set(), set()
    for row in inventory:
        relative = Path(row["relative_path"])
        require(not relative.is_absolute() and relative.parts and ".." not in relative.parts
                and str(relative) not in seen, "safe distinct snapshot path required")
        seen.add(str(relative))
        git("ls-files", "--error-unmatch", "--", str(relative))
        read_pinned({"path": str(clone / relative), "sha256": row["sha256"], "bytes": row["bytes"]})
        digests.add(row["sha256"])
    required = [identity["controller"], identity["configuration"], *identity["sources"]]
    require(all(row["sha256"] in digests for row in required), "frozen clone omits controller/config/source bytes")


def audit_freeze(row):
    """Require own-count gates; a lower-count dependency must be explicit."""
    frozen = _read(row)
    require(frozen.get("schema") == "rage_mill.canonical_release_freeze.v1", "canonical source freeze required")
    identity = frozen["identity"]
    protocol = validate_identity(identity)
    require(frozen.get("identity_sha256") == value_sha256(identity), "freeze identity differs")
    require(frozen.get("reserved_gates") == reserved_gates(protocol.n_links), "reserved identifiers changed")
    require(frozen.get("activation_authorized") is False, "freeze evidence grants no automatic activation")
    audit_source_snapshot(frozen["source_snapshot"], identity)
    audit_exact_source_calibration(frozen["source_calibration"], identity)
    dev = frozen["development"]
    require(set(dev) == {"noisy20", "noisy100"}, "both development cohorts required")
    used = set()
    for label, size in (("noisy20", 20), ("noisy100", 100)):
        description = dev[label]
        start = description.get("seed_start")
        require(type(start) is int and start >= 0, "declared development seed required")
        cohort = set(range(start, start + size))
        require(not cohort & used and not any(is_reserved(seed) for seed in cohort),
                "development cohorts overlap each other or reserved identifiers")
        used |= cohort
        audit_cohort(description["receipt"], identity=identity, count=size,
                     exact=False, seed_start=start, stage="development")
    dependency = frozen.get("requires_prerequisite_release", False)
    require(type(dependency) is bool, "explicit boolean prerequisite policy required")
    if dependency:
        prior = _read(frozen["prerequisite_release"])
        require(prior.get("schema") == "rage_mill.canonical_release_verification.v1"
                and prior.get("n_links") == protocol.n_links - 1, "previous count's completed release required")
        # Revalidate the actual previous evidence, never its cached pass flag.
        verify_release(prior["freeze"], prior["reserved_cohorts"], prior["video"])
    elif frozen.get("prerequisite_release") is not None:
        # Optional historical provenance is not a physical benchmark gate.
        # In particular, another session's different receipt format must not
        # prevent an independently qualified own-count release.
        read_pinned(frozen["prerequisite_release"])
    return frozen


def video_stream(path):
    result = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height,duration:format=duration",
        "-of", "json", str(path)], text=True)
    data = json.loads(result)
    require(len(data.get("streams", [])) == 1, "one actual video stream required")
    return data["streams"][0], data.get("format", {})


def audit_video(row, *, episode_receipt, identity):
    video = _read(row)
    checked = audit_episode(episode_receipt, identity=identity)
    require(checked["score"] is not None and checked["score"]["success"] is True, "video episode failed")
    require(video.get("schema") == "rage_mill.canonical_saved_state_video.v1"
            and video.get("episode_receipt") == episode_receipt
            and video.get("recording") == checked["document"]["recording"]
            and video.get("identity_sha256") == value_sha256(identity), "video depicts different evidence")
    require(video.get("frames") == 1500 and video.get("fps") == 50
            and video.get("simulated_seconds") == 30. and type(video.get("reset_count")) is int
            and video["reset_count"] == 0
            and video.get("dynamics_reexecuted") is False
            and video.get("interpolated_states") is False, "video cadence, duration or state provenance differs")
    renderer = Path(__file__).with_name("canonical_video.py")
    renderer_bytes = read_pinned(video["renderer_source"])
    require(renderer_bytes == renderer.read_bytes(), "different saved-state renderer source")
    read_pinned(video["encoder_log"])
    read_pinned(video["video"])
    stream, container = video_stream(video["video"]["path"])
    require(int(stream["nb_read_frames"]) == 1500 and stream["r_frame_rate"] == "50/1"
            and abs(float(container["duration"]) - 30.) <= 1e-6, "actual video frame count/rate/duration differs")
    return video


def verify_release(freeze_row, reserved_cohorts, video_row):
    """Return qualification evidence only after every actual reserved tape passes."""
    frozen = audit_freeze(freeze_row)
    identity = frozen["identity"]
    n = identity["protocol"]["n_links"]
    expected = reserved_gates(n)
    require(isinstance(reserved_cohorts, dict) and set(reserved_cohorts) == set(expected),
            "all four reserved cohorts required")
    results = {}
    for name, gate in expected.items():
        results[name] = audit_cohort(reserved_cohorts[name], identity=identity,
                                    count=gate["episodes"], exact=gate["exact"],
                                    seed_start=gate["seed_start"], stage="reserved", freeze=freeze_row)
    video_episode = results["video"][0]["receipt"]
    audit_video(video_row, episode_receipt=video_episode, identity=identity)
    return {"schema": "rage_mill.canonical_release_verification.v1", "n_links": n,
            "controller_qualified": True, "freeze": freeze_row, "reserved_cohorts": reserved_cohorts,
            "video": video_row, "reserved_episodes": 141, "all_actual_tapes_rescored": True,
            "identity_sha256": value_sha256(identity), "scope": "published simulated benchmark only"}
