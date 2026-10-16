"""Saved-only seventeen prerequisite checks shared by DEV and release prep.

These guards rehash and compare actual tapes, selected predictions and bounded
owner records. They construct no plant, invoke no policy and consume no labels.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np

from .canonical_benchmark import score_tape
from .canonical_evaluation import (
    audit_episode, pinned_file, read_pinned, require, validate_identity,
)
from .canonical_release import audit_cohort, audit_exact_source_calibration
from .canonical_runtime_evidence import audit_runtime_execution
from .canonical_runtime_manifest import validate_runtime_manifest


def _read(row):
    return json.loads(read_pinned(row))


def _zero(value):
    return type(value) is int and value == 0


def audit_component(row, *, identity, seed_start, episodes, exact):
    """Require complete own-count physical success and actual prediction parity."""
    protocol = validate_identity(identity)
    require(protocol.n_links == 17 and type(exact) is bool
            and type(seed_start) is int and seed_start >= 0
            and type(episodes) is int and (episodes == 1 if exact else episodes in (20, 100)),
            "formal own-count seventeen component required")
    component = _read(row)
    mode = "exact-calibration" if exact else ("noisy20" if episodes == 20 else "noisy100")
    require(component.get("schema") == "rage_mill.canonical_fixed_candidate_component.v1"
            and component.get("mode") == mode
            and all(component.get(name) is True for name in
                    ("scientific_pass", "all_physical_pass", "all_shadow_live_parity")),
            "passing physical and shadow/live component required")
    cohort_row = component["cohort"]
    outcomes = audit_cohort(cohort_row, identity=identity, count=episodes, exact=exact,
                            seed_start=seed_start, stage="development")
    cohort = _read(cohort_row)
    require(component.get("scores") == [outcome["score"] for outcome in outcomes],
            "component cached scores differ from actual cohort")
    parameters = identity["policy_parameters"]
    launch = parameters.get("selection_tick")
    budget = parameters.get("maximum_forecast_commands")
    offsets = parameters.get("cart_error_offsets_m")
    require(type(launch) is int and 0 <= launch < protocol.episode_ticks
            and type(budget) is int and budget > 0
            and isinstance(offsets, list) and offsets and offsets[0] == 0.
            and len(set(offsets)) == len(offsets) and np.isfinite(offsets).all(),
            "prospectively fixed finite profile horizon and offsets required")
    horizon = protocol.episode_ticks - launch
    require(budget == len(offsets) * horizon, "prospective forecast command budget differs")
    decisions_row = component["profile_decisions"]
    document = _read(decisions_row)
    decisions = document.get("decisions")
    require(document.get("schema") == "rage_mill.canonical_profile_decisions.v1"
            and document.get("cohort") == cohort_row
            and document.get("selection_state") == "observed_launch_state_only"
            and all(_zero(document.get(key)) for key in
                    ("additional_physical_episodes", "additional_reset_draws"))
            and isinstance(decisions, list) and len(decisions) == episodes,
            "profile decision/cohort lineage differs")
    for index, decision in enumerate(decisions):
        episode_row = cohort["episode_receipts"][index]
        actual = audit_episode(episode_row, identity=identity,
                               expected_seed=seed_start + index, expected_exact=exact)
        require(decision.get("seed") == seed_start + index
                and decision.get("episode_receipt") == episode_row
                and decision.get("factory_error") is None
                and decision.get("selected_offset_m") in offsets
                and all(_zero(decision.get(key)) for key in
                        ("additional_physical_episodes", "additional_reset_draws")),
                "selected prediction episode or causal scope differs")
        with np.load(io.BytesIO(read_pinned(decision["selected_prediction"])), allow_pickle=False) as saved:
            require(set(saved.files) == {"states", "actions", "raw_actions"},
                    "exact selected prediction fields required")
            predicted = {key: saved[key].copy() for key in saved.files}
        shapes = {"states": (horizon + 1, 36), "actions": (horizon,), "raw_actions": (horizon,)}
        dtypes = {"states": np.dtype("float64"), "actions": np.dtype("float32"),
                  "raw_actions": np.dtype("float64")}
        parity = {}
        for key in shapes:
            value, suffix = predicted[key], actual[key][launch:]
            parity[key] = {"dtype": value.dtype == suffix.dtype, "shape": value.shape == suffix.shape,
                           "bytes": value.tobytes() == suffix.tobytes()}
            require(value.shape == shapes[key] and value.dtype == dtypes[key]
                    and np.isfinite(value).all() and all(parity[key].values()),
                    f"complete selected shadow/live {key} bytes differ")
        require(predicted["states"][-1].tobytes() == actual["states"][1500].tobytes()
                and decision.get("selected_prediction_parity") == parity
                and decision.get("all_shadow_live_parity") is True
                and decision.get("final_state1500_byte_identical") is True,
                "final state1500 or cached selected parity differs")
        forecasts = decision.get("forecasts")
        require(isinstance(forecasts, list) and 1 <= len(forecasts) <= len(offsets),
                "bounded forecast history required")
        attempts = 0
        for profile_index, forecast in enumerate(forecasts):
            commands = forecast.get("predicted_commands")
            require(type(commands) is int and 0 <= commands <= horizon
                    and forecast.get("cart_error_offset_m") == offsets[profile_index]
                    and forecast.get("accepted") is (profile_index == len(forecasts) - 1),
                    "registered zero-first profile order or forecast command count differs")
            attempts += commands
        local_score = score_tape(predicted["states"], predicted["actions"], protocol)
        require(forecasts[-1].get("cart_error_offset_m") == decision["selected_offset_m"]
                and forecasts[-1].get("predicted_commands") == horizon
                and forecasts[-1].get("local_score") == local_score
                and local_score["longest_upright_ticks"] >= protocol.hold_ticks
                and all(local_score[key] for key in
                        ("finite", "force_ok", "float32_delivery_ok", "rail_ok"))
                and decision.get("completed_forecast_commands") == attempts <= budget
                and decision.get("maximum_forecast_command_attempts") == budget,
                "accepted forecast score or fixed work budget differs")
    calibration = component.get("source_calibration")
    if exact:
        require(isinstance(calibration, dict), "actual exact source calibration required")
        source = audit_exact_source_calibration(calibration, identity)
        require(source["episode_receipt"] == cohort["episode_receipts"][0],
                "calibration and actual component episode differ")
        outputs = [source["episode_receipt"], source["native_replay_proof"]]
    else:
        require(calibration is None, "noisy component cannot replace exact calibration")
        outputs = [cohort_row]
    return {"component": row, "document": component, "cohort": cohort_row,
            "source_calibration": calibration, "profile_decisions": decisions_row,
            "runtime_outputs": outputs}


def audit_owner(directory, *, identity, runtime_closure, seed_start, episodes, exact):
    """Require naturally closed prospective ownership and complete runtime pins."""
    directory = Path(directory).resolve(strict=True)
    result = audit_component(pinned_file(directory / "result/receipt.json"), identity=identity,
                             seed_start=seed_start, episodes=episodes, exact=exact)
    output = directory / "result"
    require(result["cohort"]["path"] == str(output / "episodes/receipt.json")
            and result["profile_decisions"]["path"] == str(output / "profile-decisions.json"),
            "component cohort or profile decisions escaped actual owner")
    cohort = _read(result["cohort"])
    require(all(episode["path"] == str(output / f"episodes/episode-{seed}/receipt.json")
                for seed, episode in zip(range(seed_start, seed_start + episodes),
                                         cohort["episode_receipts"])),
            "actual episode moved outside its original owner directory")
    if exact:
        require(result["source_calibration"]["path"] == str(output / "source-calibration.json")
                and result["runtime_outputs"][1]["path"] == str(output / "native-proof/receipt.json"),
                "actual source calibration or native proof escaped owner")
    rows = {key: pinned_file(directory / name) for key, name in (
        ("launch", "launch.json"), ("completion", "completion.json"),
        ("runtime_receipt", "result/runtime-receipt.json"))}
    launch, completion = _read(rows["launch"]), _read(rows["completion"])
    require(runtime_closure in identity["sources"], "runtime closure must belong to identity")
    runtime_files = validate_runtime_manifest(_read(runtime_closure), require_complete=True)
    pins = launch.get("source_hashes")
    require(launch.get("schema") == "rage_mill.next_stage_supervisor_recipe.v1"
            and launch.get("runtime_closure") == runtime_closure
            and launch.get("output_path") == str(directory / "result")
            and isinstance(pins, dict)
            and all(pins.get(path) == digest for path, digest in runtime_files.items())
            and all(pins.get(row["path"]) == row["sha256"] for row in
                    [identity["controller"], identity["configuration"], *identity["sources"]]),
            "prospective owner identity/runtime/source pins incomplete")
    argv = launch.get("argv")
    require(isinstance(argv, list) and len(argv) >= 2
            and argv[1] in {row["path"] for row in identity["sources"]}
            and Path(argv[1]).name == "evaluate_canonical17_profile_candidate.py"
            and pins.get(argv[1]) == pinned_file(argv[1])["sha256"],
            "prospectively source-bound seventeen worker required")
    for field in ("CPU_cap_s", "wall_cap_s"):
        value = launch.get(field)
        require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                "finite positive owner caps required")
    require(completion.get("schema") == "rage_mill.next_stage_supervised_completion.v1"
            and completion.get("recipe_path") == rows["launch"]["path"]
            and completion.get("recipe_sha256") == rows["launch"]["sha256"]
            and type(completion.get("child_returncode")) is int and completion["child_returncode"] == 0
            and completion.get("naturally_closed_zero") is True
            and completion.get("failure") is None and completion.get("interrupted") is False
            and completion.get("wall_timeout") is False and completion.get("recipe_unchanged") is True,
            "owner failed or lacks natural-zero recipe closure")
    unchanged = completion.get("source_hashes_unchanged")
    require(isinstance(unchanged, dict) and set(unchanged) == set(pins)
            and all(value is True for value in unchanged.values())
            and all(pinned_file(path)["sha256"] == digest for path, digest in pins.items()),
            "owner full source/runtime pin closure failed")
    require(all(Path(row["path"]).is_relative_to(output)
                and str(Path(row["path"]).resolve(strict=True)) == row["path"]
                for row in result["runtime_outputs"]), "component outputs escaped owner directory")
    audit_runtime_execution(rows["runtime_receipt"], identity=identity,
                            runtime_closure=runtime_closure, outputs=result["runtime_outputs"])
    return {**result, "owner": rows}
