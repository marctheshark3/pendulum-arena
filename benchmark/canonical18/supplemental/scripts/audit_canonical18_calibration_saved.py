#!/usr/bin/env python3
"""Audit closed exact calibration bytes and eligibility without new dynamics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--expected-inputs-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.run and sha(args.inputs) == args.expected_inputs_sha256 and not args.out.exists(),
            "prospectively declared saved-only calibration audit required")
    cpu, wall = time.process_time(), time.perf_counter()
    inputs = json.loads(args.inputs.read_text())
    require(all(sha(path) == digest for path,digest in inputs["source_hashes"].items()),
            "saved calibration audit input changed")
    import numpy as np
    from rage_mill.canonical_evaluation import audit_journal, read_pinned, validate_identity

    def same(a,b):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

    def read(row):
        return json.loads(read_pinned(row))

    def load(row):
        if isinstance(row,dict):
            read_pinned(row)
        with np.load(row["path"] if isinstance(row,dict) else row, allow_pickle=False) as saved:
            return {key:saved[key].copy() for key in saved.files}

    owner = Path(inputs["calibration_owner"])
    launch, completion = (json.loads((owner/name).read_text()) for name in ("launch.json","completion.json"))
    require(completion["recipe_sha256"] == sha(owner/"launch.json")
            and completion["child_returncode"] == inputs["expected_owner_returncode"]
            and completion["recipe_unchanged"] is True and completion["failure"] is None
            and completion["wall_timeout"] is False and completion["interrupted"] is False
            and bool(completion["owned_group_cleanup"])
            and set(launch["source_hashes"]) == set(completion["source_hashes_unchanged"])
            and all(flag is True for flag in completion["source_hashes_unchanged"].values())
            and all(sha(path) == digest for path,digest in launch["source_hashes"].items()),
            "closed calibration owner recipe/source history differs")
    component = json.loads((owner/"result/receipt.json").read_text())
    cohort = read(component["cohort"])
    seed = inputs["seed"]
    identity = cohort["identity"]
    require(validate_identity(identity).n_links == 18, "own-count eighteen calibration identity required")
    require(component["mode"] == "exact-calibration" and component["scientific_pass"] is True
            and component["all_physical_pass"] is True and component["all_shadow_live_parity"] is True
            and cohort["seed_start"] == seed and cohort["episodes"] == 1 and cohort["exact"] is True
            and cohort["stage"] == "development" and len(cohort["episode_receipts"]) == 1
            and identity == json.loads(Path(inputs["identity"]).read_text()),
            "fixed own18 actual exact calibration cohort differs")
    row = cohort["episode_receipts"][0]
    episode = read(row)
    require(row["path"] == str(owner/f"result/episodes/episode-{seed}/receipt.json")
            and episode["identity"] == identity and episode["seed"] == seed and episode["exact"] is True
            and episode["stage"] == "development" and episode["initialized"] is True
            and episode["reset_count_after_initialization"] == 0
            and episode["invalid_reason"] is None and episode["source_pins_unchanged"] is True,
            "one complete owned calibration episode required")
    tape = load(episode["recording"])
    require(set(tape) == {"states","actions","raw_actions","times","failure_state"},
            "complete calibration recording fields required")
    shapes = dict(states=(1501,38), actions=(1500,),raw_actions=(1500,),times=(1501,),failure_state=(0,38))
    for key,shape in shapes.items():
        require(tape[key].shape == shape and tape[key].dtype == np.dtype("float32" if key=="actions" else "float64")
                and np.isfinite(tape[key]).all(), "complete finite calibration tape differs: "+key)
    states,actions,raw = (tape[key] for key in ("states","actions","raw_actions"))
    initial = np.zeros(38);initial[1]=np.pi
    require(same(states[0],initial) and same(actions,np.clip(raw.astype(np.float32),-1.,1.))
            and np.max(np.abs(tape["times"]-np.arange(1501)*.02)) <= 1e-9,
            "literal exact hanging, immediate float32 action or clock differs")
    cumulative = np.cumsum(states[1:,1:19],axis=1)
    wrapped = cumulative.copy()
    outside = (wrapped < -np.pi) | (wrapped >= np.pi)
    wrapped[outside] = (wrapped[outside]+np.pi)%(2*np.pi)-np.pi
    upright = np.all(np.abs(wrapped)<.15,axis=1)
    longest = streak = 0
    for passed in upright:
        streak = streak+1 if passed else 0
        longest = max(longest,streak)
    maximum_cart = float(np.max(np.abs(states[:,0])))
    score = dict(success=bool(maximum_cart<=3. and np.max(np.abs(actions))<=1. and longest>=250),
                 complete=True,finite=True,force_ok=bool(np.max(np.abs(actions))<=1.),
                 float32_delivery_ok=True,rail_ok=bool(maximum_cart<=3.),ticks=1500,
                 longest_upright_ticks=longest,longest_upright_seconds=longest*.02,
                 maximum_cart_m=maximum_cart,invalid_reason=None)
    require(score["success"] and score == episode["score"] and component["scores"] == [score],
            "independent post-command cumulative strict-angle score differs")
    audit_journal(read_pinned(episode["journal"]),states=states,actions=actions,raw_actions=raw,
                  times=tape["times"],invalid_reason=None)
    decisions = read(component["profile_decisions"])["decisions"]
    require(len(decisions)==1 and decisions[0]["episode_receipt"]==row
            and decisions[0]["selected_offset_m"]==0. and decisions[0]["all_shadow_live_parity"] is True
            and decisions[0]["final_state1500_byte_identical"] is True,
            "single zero-offset outer decision differs")
    decision = decisions[0]
    prediction = load(decision["selected_prediction"])
    require(set(prediction)=={"states","actions","raw_actions"}
            and all(same(prediction[key],tape[key][850:]) for key in prediction)
            and same(prediction["states"][-1],states[1500]),
            "full selected651state650action/raw final suffix differs")
    forecasts = decision["forecasts"]
    require(len(forecasts)==1 and forecasts[0]["accepted"] is True
            and forecasts[0]["predicted_commands"]==650 and forecasts[0]["cart_error_offset_m"]==0.
            and decision["completed_forecast_commands"]==decision["maximum_forecast_command_attempts"]==650,
            "full bounded zero-only forecast differs")
    counters = {}
    for name in ("shadow","live"):
        report = forecasts[0][f"native_value_{name}_telemetry"]
        require(report["active_commands"]==650 and len(report["steps"])==650
                and report["native_forecast_errors"]==0,
                "full native candidate report differs")
        calls = agreements = 0
        for tick,step in enumerate(report["steps"]):
            count = step["unique_forecasts"]
            require(type(count) is int and 1<=count<=16 and step["episode_tick"]==tick+850
                    and step["route_tick"]==tick and type(step["baseline_delivery_agreement"]) is bool
                    and calls<=step["selected_index"]<calls+count and calls<=step["baseline_index"]<calls+count
                    and step["previous_selected_state_byte_parity"] is (None if tick==0 else True)
                    and 0.<=step["selected_cost"]<=step["baseline_cost"]
                    and np.isfinite(step["baseline_cost"]), "native decision block or retained baseline cost differs")
            calls += count;agreements += int(step["baseline_delivery_agreement"])
        require(calls==report["native_candidate_commands"]<=10400
                and report["baseline_delivery_agreements"]==agreements
                and report["baseline_delivery_disagreements"]==650-agreements,
                "candidate native counts or baseline disagreement telemetry differs")
        counters[name]=dict(native_candidate_commands=calls,baseline_delivery_disagreements=650-agreements,
                            native_select_CPU_seconds=report["native_select_CPU_seconds"],
                            native_select_wall_seconds=report["native_select_wall_seconds"])
    source = read(component["source_calibration"])
    proof = read(source["native_replay_proof"])
    native = load(proof["proof_arrays"])
    require(source["episode_receipt"]==row and proof["episode_receipt"]==row
            and proof["complete"] is True and proof["failure"] is None
            and proof["continuous_transitions"]==proof["one_step_transitions"]==1500
            and same(native["continuous_states"],states) and same(native["cold_next_states"],states[1:])
            and source["initial_state_gap"]==source["maximum_dynamics_defect"]==0.
            and proof["summary"]["continuous_replay_identical"] is True,
            "complete saved native continuous/cold source calibration differs")
    runtime = json.loads((owner/"result/runtime-receipt.json").read_text())
    manifest = json.loads(read_pinned(runtime["runtime_closure"]))
    pins = {r["path"]:r["sha256"] for r in manifest["files"]}
    pins.update({r["path"]:r["sha256"] for r in identity["sources"]})
    uncovered = {path:digest for path,digest in runtime["actual_loaded_file_sha256"].items() if pins.get(path)!=digest}
    require(uncovered == runtime["uncovered_or_changed"]
            and sorted(uncovered)==sorted(inputs["expected_uncovered_paths"])
            and runtime["runtime_pins_complete"] is (not bool(uncovered))
            and runtime["all_runtime_pins_unchanged"] is True,
            "actual imported runtime coverage or failure classification differs")
    qualified_owner = bool(completion["naturally_closed_zero"] and completion["child_returncode"]==0 and not uncovered)
    total = 1500+650+3000+sum(counters[name]["native_candidate_commands"] for name in counters)
    require(total<=25950 and all(sha(path)==digest for path,digest in inputs["source_hashes"].items()),
            "original total native work exceeded declared cap or saved input changed")
    args.out.mkdir()
    result = dict(schema="rage_mill.canonical18_closed_calibration_saved_independent_audit.v1",
        audit_passed=True,seed=seed,score=score,all1500journal_samples_match=True,
        selected_full651state650action_raw_suffix_final_byte_exact=True,
        native_continuous1500_and_cold1500_byte_exact=True,maximum_dynamics_defect=0.,initial_state_gap=0.,
        native_value_counts=counters,original_total_native_commands_including_proof=total,
        original_native_ceiling=25950,original_owner_returncode=completion["child_returncode"],
        original_owner_naturally_closed_zero=completion["naturally_closed_zero"],original_owner_reaped=True,
        original_owner_CPU_seconds=completion["aggregate_child_CPU_seconds"],
        original_owner_wall_seconds=completion["wall_seconds"],original_owner_peak_rss_kib=completion["peak_rss_kib"],
        runtime_uncovered_files=uncovered,owner_eligible_as_qualification_predecessor=qualified_owner,
        classification="eligible_source_calibration" if qualified_owner else "physical_native_parity_pass_but_failed_runtime_owner",
        source_hashes_unchanged=True,native_calls=0,controller_calls=0,inference_calls=0,reset_RNG_draws=0,
        new_physical_episodes=0,development_episodes=0,reserved_episodes=0,controller_qualified=False,
        CPU_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall)
    with (args.out/"receipt.json").open("x") as stream:
        json.dump(result,stream,indent=2);stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
