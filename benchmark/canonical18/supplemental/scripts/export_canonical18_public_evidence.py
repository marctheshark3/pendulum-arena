"""Stage qualified canonical eighteen source/evidence/media; never publish or run physics.

Usage: python scripts/export_canonical18_public_evidence.py --inputs INPUTS.json
       --expected-inputs-sha256 SHA256 --out /fresh/absolute/staging

Inputs schema rage_mill.canonical18_public_evidence_inputs.v1 contains pinned
identity, release_verification, combined_qualification, four actual_gate_owners
and four saved_policy_proof_owners (launch/completion/runtime_receipt pins),
source_checkout, license, timestamps(runAt/frozenAt/firstPublicAt/registeredAt),
public_media {silent,review,receipts:[pins], music?:{video,authorization,provenance}},
additional_artifacts:[pins], additional_source_files:[explicit own18 source pins],
development_native_proofs:[generic proof pins],
and expected_identity_sha256. No future-public/registry clock is assigned.

source_checkout binds the original qualification checkout. The output copies
only explicit benchmark identity sources and artifacts, never private Git or
runtime binaries. Original absolute-path JSON remains unchanged inventory and
is not a portable execution identity. Public auditors use relative paths only.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

GATES = ("noisy20", "noisy100", "exact20", "video")
EXPECTED_SOURCE = "b84c6413974d5c0831cdf261ce15e58ba65cd3f5"
MAX_FILES, MAX_BYTES = 30000, 3 * 1024**3


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def pin(path):
    path = Path(path).resolve(strict=True)
    need(path.is_file() and not path.is_symlink(), "regular input required")
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)


def read(row):
    need(isinstance(row, dict) and set(row) >= {"path", "sha256", "bytes"}, "actual pinned input required")
    need(all(pin(row["path"])[key] == row[key] for key in ("path", "sha256", "bytes")), "pinned input changed")
    return Path(row["path"]).read_bytes()


def document(row):
    return json.loads(read(row), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
    return pin(path)


def utc(value, *, required=False):
    if value is None and not required:
        return
    need(isinstance(value, str) and value.endswith(("Z", "+00:00")), "genuine UTC instant required")
    datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


def disjoint_output(out, roots):
    need(out.is_absolute() and out.resolve() == out and not out.exists(), "fresh canonical absolute output required")
    for root in roots:
        root = Path(root).resolve(strict=True)
        need(not (out == root or out.is_relative_to(root) or root.is_relative_to(out)),
             "output overlaps immutable input/owner/claim tree")


def owner(row, *, identity, runtime, outputs):
    from rage_mill.canonical_runtime_evidence import audit_runtime_execution
    need(isinstance(row, dict) and set(row) == {"launch", "completion", "runtime_receipt"},
         "complete bounded owner pins required")
    launch, done = document(row["launch"]), document(row["completion"])
    need(done.get("naturally_closed_zero") is True and done.get("child_returncode") == 0
         and done.get("recipe_sha256") == row["launch"]["sha256"]
         and done.get("recipe_path") == row["launch"]["path"]
         and done.get("recipe_unchanged") is True and done.get("failure") is None
         and done.get("wall_timeout") is False and done.get("interrupted") is False
         and done.get("owned_group_cleanup") == ["group_already_exited"], "owner failed/not naturally reaped")
    pins = launch["source_hashes"]
    need(set(done["source_hashes_unchanged"]) == set(pins)
         and all(value is True for value in done["source_hashes_unchanged"].values()), "owner pin closure differs")
    for path, expected in pins.items():
        need(digest(path) == expected, "original owner source/runtime changed")
    need(launch.get("runtime_closure") == runtime, "prospective owner runtime required")
    for key in ("CPU_cap_s", "wall_cap_s"):
        need(type(launch.get(key)) in (int, float) and math.isfinite(launch[key]) and launch[key] > 0,
             "finite positive owner cap required")
    for key in ("aggregate_child_CPU_seconds", "wall_seconds"):
        need(type(done.get(key)) in (int, float) and math.isfinite(done[key]) and done[key] >= 0,
             "finite complete actual owner cost required")
    audit_runtime_execution(row["runtime_receipt"], identity=identity, runtime_closure=runtime, outputs=outputs)
    return launch, dict(aggregate_child_CPU_seconds=done["aggregate_child_CPU_seconds"],
                        wall_seconds=done["wall_seconds"], owner=row)


class Copier:
    def __init__(self, root):
        self.root, self.mapping, self.total = root, [], 0

    def file(self, original, relative):
        row = pin(original); target = self.root / relative
        need(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "safe public destination required")
        if target.exists():
            need(pin(target)["sha256"] == row["sha256"], "conflicting public copy")
        else:
            self.total += row["bytes"]
            need(self.total <= MAX_BYTES and len(self.mapping) < MAX_FILES, "bounded evidence inventory exceeded")
            need(Path(original).suffix in (".json", ".jsonl", ".npz", ".xml", ".log", ".py", ".md", ".txt")
                 or Path(original).name == "LICENSE", "binary/unrelated evidence file refused")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(read(row))
        self.mapping.append(dict(original=row, relative_path=relative))
        return dict(path=relative, sha256=row["sha256"], bytes=row["bytes"])

    def tree(self, original, relative):
        original = Path(original).resolve(strict=True)
        for path in sorted(original.rglob("*")):
            need(not path.is_symlink(), "symlink evidence tree refused")
            if path.is_file():
                self.file(path, (Path(relative) / path.relative_to(original)).as_posix())


def archive(source, destination, prefix):
    with destination.open("xb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(source.rglob("*")):
                    need(not path.is_symlink(), "archive refuses symlinks")
                    if path.is_file():
                        data = path.read_bytes(); item = tarfile.TarInfo((Path(prefix) / path.relative_to(source)).as_posix())
                        item.size = len(data); item.mtime = 0; item.mode = 0o644
                        item.uid = item.gid = 0; item.uname = item.gname = ""
                        tar.addfile(item, io.BytesIO(data))
    return pin(destination)


def assemble(inputs, out):
    import numpy as np
    from rage_mill.canonical_evaluation import audit_episode, reserved_gates, value_sha256
    from rage_mill.canonical_reserved import verify_reserved_release
    from rage_mill.canonical_replay import audit_native_replay_proof
    from rage_mill.canonical18_evidence import audit_component
    need(inputs.get("schema") == "rage_mill.canonical18_public_evidence_inputs.v1", "own18 assembly inputs required")
    verified = document(inputs["release_verification"])
    need(verified.get("schema") == "rage_mill.canonical_release_verification.v1"
         and verified.get("controller_qualified") is True and verified.get("n_links") == 18
         and verified.get("reserved_episodes") == verified.get("once_only_label_claims_checked") == 141,
         "completed actual typed own18 release required; no premature qualification")
    rerun = verify_reserved_release(verified["campaign_authority"], verified["reserved_cohorts"], verified["video"])
    need(all(verified.get(key) == value for key, value in rerun.items()), "completed release differs on read-only re-audit")
    frozen, authority = document(verified["freeze"]), document(verified["campaign_authority"])
    identity = frozen["identity"]; runtime = frozen["runtime_closure"]
    need(identity["protocol"]["n_links"] == 18 and set(inputs["actual_gate_owners"]) == set(GATES)
         and set(inputs["saved_policy_proof_owners"]) == set(GATES), "all4 actual and proof owners required")
    identity_row = inputs["identity"]
    need(document(identity_row) == identity and identity_row["sha256"] == inputs["expected_identity_sha256"]
         and identity["source_commit"] == EXPECTED_SOURCE, "original own18 identity differs")
    combined = document(inputs["combined_qualification"])
    need(combined.get("passed") is True and combined.get("controller_qualified") is True
         and combined.get("n_links") == 18 and combined.get("all141_labels_consumed_once") is True
         and combined.get("replaced_actual_attempts") == 0
         and combined.get("identity") == identity_row and combined.get("freeze") == verified["freeze"]
         and set(combined.get("reserved_gates", {})) == set(GATES),
         "positive own18 combined all141 once-only qualification required")
    checkout = Path(inputs["source_checkout"]).resolve(strict=True)
    need(checkout.is_dir() and (checkout / "src/rage_mill").is_dir(), "actual original source checkout required")
    stamps = inputs["timestamps"]
    need(set(stamps) == {"runAt", "frozenAt", "firstPublicAt", "registeredAt"}, "separate actual source/publication clocks required")
    for key,value in stamps.items():
        utc(value, required=key == "frozenAt")
    need(stamps["firstPublicAt"] is None and stamps["registeredAt"] is None,
         "offline export cannot assign future publication or registry clocks")
    source_commit = frozen["source_snapshot"]["git_commit"]
    clock = subprocess.run(["git", "show", "-s", "--format=%H%n%ct", source_commit],
                           cwd=checkout, capture_output=True, text=True, check=True, timeout=15).stdout.splitlines()
    need(len(clock) == 2 and clock[0] == source_commit,
         "actual original source-freeze Git object required")
    actual_frozen = datetime.datetime.fromtimestamp(int(clock[1]), datetime.timezone.utc)
    need(datetime.datetime.fromisoformat(stamps["frozenAt"].replace("Z", "+00:00")) == actual_frozen,
         "frozenAt must equal original source-freeze commit clock")
    jobs, costs, proof_docs, proofs, native_defects = [], {}, {}, {}, []
    calibration = document(frozen["source_calibration"])
    cal = audit_episode(calibration["episode_receipt"], identity=identity, expected_exact=True)
    native_defects.append(audit_native_replay_proof(calibration["native_replay_proof"], episode_receipt=calibration["episode_receipt"])["maximum_dynamics_defect"])
    descriptions = [("calibration", frozen["runtime_owners"]["calibration"], None),
                    ("development20", frozen["runtime_owners"]["noisy20"], frozen["development"]["noisy20"]["receipt"]),
                    ("development100", frozen["runtime_owners"]["noisy100"], frozen["development"]["noisy100"]["receipt"])]
    cohorts = {}
    for label, owned, cohort in descriptions:
        launch = document(owned["launch"]); component = json.loads((Path(launch["output_path"]) / "receipt.json").read_text())
        cohort = component["cohort"] if cohort is None else cohort
        body = document(cohort)
        audit_component(pin(Path(launch["output_path"]) / "receipt.json"), identity=identity,
                        seed_start=body["seed_start"], episodes=body["episodes"], exact=body["exact"])
        expected = [calibration["episode_receipt"], calibration["native_replay_proof"]] if label == "calibration" else [cohort]
        _, costs[label] = owner(owned, identity=identity, runtime=runtime, outputs=expected)
        cohorts[label] = cohort
    for gate in GATES:
        cohort = verified["reserved_cohorts"][gate]; body = document(cohort)
        gate_summary = combined["reserved_gates"][gate]
        need(gate_summary.get("actual_cohort") == cohort
             and gate_summary.get("episodes") == {"noisy20":20,"noisy100":100,"exact20":20,"video":1}[gate],
             "combined qualification gate does not bind the same exact actual cohort")
        _, costs[gate] = owner(inputs["actual_gate_owners"][gate], identity=identity, runtime=runtime, outputs=body["episode_receipts"])
        proof_owner = inputs["saved_policy_proof_owners"][gate]
        launch = document(proof_owner["launch"]); row = pin(Path(launch["output_path"]) / "receipt.json"); proof = document(row)
        need(proof.get("schema") == "rage_mill.canonical18_saved_policy_cohort_proof.v1"
             and proof.get("passed") is True and proof.get("gate") == gate and proof.get("cohort") == cohort
             and document(proof["identity"]) == identity and proof.get("episodes_proved") == body["episodes"]
             and proof.get("rawF64_commands_byte_proved") == proof.get("deliveredF32_commands_byte_proved") == 1500 * body["episodes"]
             and proof.get("all_fullsuffix_and_finalstate1500_byte_identical") is True
             and proof.get("physical_episodes") == proof.get("new_reserved_labels") == proof.get("reset_RNG_draws") == 0,
             "complete actual saved-policy cohort proof required")
        _, costs[gate + "-proof"] = owner(proof_owner, identity=identity, runtime=runtime, outputs=[row])
        need(len(proof["episode_proofs"]) == body["episodes"], "one saved proof per actual label required")
        for episode_row, proof_row in zip(body["episode_receipts"], proof["episode_proofs"]):
            episode = audit_episode(episode_row, identity=identity); p = document(proof_row)
            need(p.get("schema") == "rage_mill.canonical18_saved_policy_episode_proof.v1"
                 and p.get("passed") is True and p.get("episode") == episode_row
                 and p.get("seed") == episode["document"]["seed"]
                 and p.get("recording") == episode["document"]["recording"]
                 and p.get("rawF64_commands_byte_proved") == p.get("deliveredF32_commands_byte_proved") == 1500
                 and p.get("selected_native_next_state_delivery_pairs_byte_proved") == 650
                 and p.get("selected_full651state650action_raw_suffix_byte_identical") is True
                 and p.get("final_state1500_byte_identical") is True, "complete fullsuffix/final-state proof required")
            with np.load(io.BytesIO(read(p["recomputed_commands"])), allow_pickle=False) as arrays:
                need(len(arrays.files) == 4 and set(arrays.files) == {"raw_actions", "actions", "selected_next_states", "selected_next_actions"},
                     "exact recomputed command/native-pair fields required")
                for key, expected in (("raw_actions", episode["raw_actions"]), ("actions", episode["actions"]),
                                      ("selected_next_states", episode["states"][851:]),
                                      ("selected_next_actions", episode["actions"][850:])):
                    need(arrays[key].shape == expected.shape and arrays[key].dtype == expected.dtype
                         and arrays[key].tobytes() == expected.tobytes(), "saved command/native-pair dtype/shape/bytes differ")
            with np.load(io.BytesIO(read(p["selected_prediction"])), allow_pickle=False) as arrays:
                need(len(arrays.files) == 3 and set(arrays.files) == {"states", "actions", "raw_actions"},
                     "exact selected fullsuffix fields required")
                for key in ("states", "actions", "raw_actions"):
                    need(arrays[key].shape == episode[key][850:].shape and arrays[key].dtype == episode[key].dtype
                         and arrays[key].tobytes() == episode[key][850:].tobytes(), "complete selected suffix bytes differ")
            native = audit_native_replay_proof(p["native_transition_proof"], episode_receipt=episode_row)
            need(native == p.get("native_transition_summary") and native["continuous_replay_identical"]
                 and native["maximum_dynamics_defect"] <= 1e-8, "saved native proof/parent summary failed")
            native_defects.append(native["maximum_dynamics_defect"])
            counter = p["cost_counts"]
            need(counter["policy_calls"] == 1500 and counter["outer_shadow_native_steps"] == 650
                 and counter["saved_native_proof_maps"] == 3000
                 and 0 <= counter["shadow_inner_native_candidates"] <= 10400
                 and 0 <= counter["recorded_input_inner_native_candidates"] <= 10400
                 and p["total_native_maps"] == sum(v for k, v in counter.items() if k != "policy_calls") <= 24450,
                 "full saved native costs exceed/mismatch declared bounds")
            telemetry = document(p["native_value_telemetry"])
            for name, counts in (("shadow", counter["shadow_inner_native_candidates"]), ("recorded_input", counter["recorded_input_inner_native_candidates"])):
                report = telemetry[name]; steps = report["steps"]
                need(len(steps) == report["active_commands"] == 650 and report["native_forecast_errors"] == 0
                     and report["native_candidate_commands"] == counts
                     and sum(s["unique_forecasts"] for s in steps) == counts
                     and all(1 <= s["unique_forecasts"] <= 16 for s in steps)
                     and report["baseline_delivery_agreements"] + report["baseline_delivery_disagreements"] == 650,
                     "bounded shadow/live telemetry incomplete")
        proofs[gate], proof_docs[gate], cohorts[gate] = row, proof, cohort
    for label, cohort in cohorts.items():
        body = document(cohort)
        for row in body["episode_receipts"]:
            checked = audit_episode(row, identity=identity)
            need(checked["score"]["success"] and checked["score"]["ticks"] == 1500, "actual physical episode failed")
            jobs.append((label, row, checked))
    need(len(jobs) == 262 and sum(document(r)["episodes_proved"] for r in proofs.values()) == 141, "full262+141 coverage required")
    # Optional DEV120 cold/continuous proofs extend the baseline142 episode
    # coverage only after every supplied proof binds an actual recorded DEV tape.
    dev_by_path = {row['path']:row for label,row,_ in jobs if label.startswith('development')}
    dev_native, seen_dev = [], set()
    for row in inputs.get('development_native_proofs', []):
        proof = document(row); episode = proof.get('episode_receipt')
        need(isinstance(episode,dict) and episode.get('path') in dev_by_path
             and episode == dev_by_path[episode['path']] and episode['path'] not in seen_dev,
             'DEV native proof must identify a distinct original development episode')
        summary = audit_native_replay_proof(row, episode_receipt=episode)
        need(summary['continuous_replay_identical'] is True and summary['maximum_dynamics_defect'] <= 1e-8,
             'DEV native saved proof failed')
        seen_dev.add(episode['path']); native_defects.append(summary['maximum_dynamics_defect'])
        dev_native.append((row,proof))
    if inputs.get('require_all_development_native_proofs', False):
        need(seen_dev == set(dev_by_path) and len(seen_dev) == 120,
             'all120 native development proofs required by this export declaration')
    native_episodes = 142 + len(seen_dev)
    video = document(verified['video'])
    media = inputs['public_media']
    read(media['silent']); review = document(media['review'])
    need(review.get('passed') is True, 'positive independent public media review required')
    need(video.get('frames') == 1500 and video.get('fps') == 50
         and video.get('simulated_seconds') == 30.
         and video.get('dynamics_reexecuted') is False and video.get('interpolated_states') is False
         and video.get('reset_count') == 0,
         'scientific render must retain exactly1500 actual post-command frames at50fps/30s')
    def contains_pin(value,target):
        if isinstance(value,dict):
            if all(value.get(key)==target[key] for key in ('path','sha256','bytes')):
                return True
            return any(contains_pin(item,target) for item in value.values())
        return isinstance(value,list) and any(contains_pin(item,target) for item in value)
    need(contains_pin(review,video['video']) and contains_pin(review,media['silent']),
         'independent review must bind actual scientific and silent presentation files')
    for row in media.get('receipts',[]): read(row)
    if media.get('music') is not None:
        music = media['music']; authorization = document(music['authorization'])
        provenance = document(music['provenance']); read(music['video'])
        need(authorization.get('requested_text') == 'super star by queen'
             and authorization.get('authorized_for_publication') is True
             and isinstance(authorization.get('verified_track_title'),str)
             and authorization['verified_track_title'].strip()
             and isinstance(authorization.get('audio_source_sha256'),str)
             and len(authorization['audio_source_sha256']) == 64
             and contains_pin(provenance,music['video'])
             and contains_pin(review,music['video']),
             'music needs precise requested-track identification, source permission, provenance and review')
    protected = [checkout, authority['output_root'], authority['claim_base'],
                 Path(verified['freeze']['path']).parent, Path(verified['video']['path']).parent]
    protected.extend(Path(detail['owner']['launch']['path']).parent for detail in costs.values())
    protected.extend(document(detail['owner']['launch'])['output_path'] for detail in costs.values())
    disjoint_output(out,protected)
    out.mkdir(parents=True)
    evidence = out/'canonical18-evidence'; evidence.mkdir(); copier = Copier(evidence)
    for name,row in (('identity',identity_row),('freeze',verified['freeze']),
                     ('verified-release',inputs['release_verification']),
                     ('combined-qualification',inputs['combined_qualification']),
                     ('authority',verified['campaign_authority'])):
        copier.file(row['path'],'metadata/'+name+'.json')
    if verified.get('manifest'): copier.file(verified['manifest']['path'],'metadata/release-manifest.json')
    for index,row in enumerate([authority['source_review'],*frozen.get('independent_reviews',[])]):
        read(row); copier.file(row['path'],f'reviews/{index:03d}-'+Path(row['path']).name)
    for label,owned,_ in descriptions:
        copier.tree(document(owned['launch'])['output_path'],label)
    for gate in GATES:
        copier.tree(Path(authority['output_root'])/gate,'reserved/'+gate)
        copier.tree(document(inputs['saved_policy_proof_owners'][gate]['launch'])['output_path'],
                    'policy-proof/'+gate)
    for index,(row,proof) in enumerate(dev_native):
        prefix = 'development-native/'+str(document(proof['episode_receipt'])['seed'])
        copier.file(row['path'],prefix+'/receipt.json')
        for key in ('claim','proof_arrays'):
            copier.file(proof[key]['path'],prefix+'/'+Path(proof[key]['path']).name)
    claim_root = Path(authority['claim_base'])/('published-'+identity['source_commit'])/'count-18'
    expected_labels = set(range(218000,218020))|set(range(218100,218200))|set(range(218200,218220))|{218500}
    claims = list(claim_root.glob('seed-*/claim.json'))
    need(len(claims)==141 and {int(p.parent.name.removeprefix('seed-')) for p in claims}==expected_labels,
         'published own18 permanent claims must be exactly all141 original labels')
    copier.tree(claim_root,'claims')
    for label,detail in costs.items():
        for key,row in detail['owner'].items(): copier.file(row['path'],'owners/'+label+'/'+key+'.json')
        copier.file(document(detail['owner']['launch'])['log'],'owners/'+label+'/worker.log')
    copier.file(verified['video']['path'],'render/receipt.json')
    if video.get('encoder_log'): copier.file(video['encoder_log']['path'],'render/encoder.log')
    copier.file(media['review']['path'],'media/review.json')
    for index,row in enumerate(media.get('receipts',[])):
        copier.file(row['path'],f'media/receipt-{index:03d}-'+Path(row['path']).name)
    for index,row in enumerate(inputs.get('additional_artifacts',[])):
        read(row); copier.file(row['path'],f'additional/{index:03d}-'+Path(row['path']).name)
    write(evidence/'original-path-mapping.json',dict(schema='pendulum_arena.original_path_mapping.v1',
          files=copier.mapping,original_JSON_rewritten=False,portable_execution_identity=False))
    stats = dict(minimum_upright_seconds=min(x[2]['score']['longest_upright_seconds'] for x in jobs),
                 maximum_cart_abs_m=max(x[2]['score']['maximum_cart_m'] for x in jobs),
                 maximum_delivered_force_N=max(float(np.max(np.abs(x[2]['actions'].astype(np.float64))))*80 for x in jobs))
    per_episode = [dict(stage=label,seed=checked['document']['seed'],episode_receipt=row,
                        score=checked['score']) for label,row,checked in jobs]
    write(evidence/'episode-metrics.json',dict(schema='pendulum_arena.canonical18_episode_metrics.v1',
          actual_episodes=262,recorded_commands=393000,episodes=per_episode,all262_saved_metrics=stats))
    audit = dict(schema='pendulum_arena.canonical18_public_audit.v1',passed=True,controller_qualified=True,
        author_attested=True,actual_episodes=262,recorded_commands=393000,reserved_episodes=141,
        once_only_label_claims=141,heldout_selected_shadow_suffixes_byte_proved=141,
        heldout_raw_F64_commands_byte_proved=211500,heldout_delivered_F32_commands_byte_proved=211500,
        heldout_selected_native_next_state_delivery_pairs_byte_proved=91650,
        all262_saved_metrics=stats,native_audited_episodes=native_episodes,
        native_audited_commands=1500*native_episodes,
        native_development_episodes=len(seen_dev),maximum_dynamics_defect=max(native_defects),
        calibration_initial_state_gap=calibration['initial_state_gap'],owner_costs=costs,
        aggregate_child_CPU_seconds=sum(c['aggregate_child_CPU_seconds'] for c in costs.values()),
        sum_owner_wall_seconds=sum(c['wall_seconds'] for c in costs.values()),owner_wall_times_may_overlap=True,
        owner_cost_scope='Source calibration,development120,reserved141 and reserved saved-policy proofs only; other engineering/failure/preparation/render/review costs remain separate.',
        actual_inner_candidate_counts_available=False,
        candidate_count_scope='Saved-policy causal reconstruction counts are separate proof work, not original generic recorder counters.',
        original_identity_sha256=identity_row['sha256'],timestamps=stamps,
        physics_steps_by_export=0,controller_calls_by_export=0,new_reserved_labels_by_export=0,
        original_absolute_paths_preserved=True,portable_execution_identity=False,
        learned_controller=False,hardware_result=False,wall_clock_realtime=False,global_priority_claim=False)
    write(evidence/'audit.json',audit)
    write(evidence/'README.md.json',dict(scope='Author-attested recorded simulation evidence; audit_saved.py inspects relative file hashes and lineage only, never dynamics or control.',
          native_coverage='Exact source calibration+reserved141, plus '+str(len(seen_dev))+' original development tapes with supplied checked cold/continuous proof arrays.',
          music='Separate optional derivative; scientific and silent copies remain independent.'))
    auditor = public_auditor_source()
    (evidence/'audit_saved.py').write_text(auditor)
    evidence_manifest = manifest(evidence,'pendulum_arena.canonical18_public_evidence_files.v1')
    source = out/'source/benchmark/canonical18'; source.mkdir(parents=True)
    source_copy = Copier(source)
    source_copy.file(identity_row['path'],'metadata/identity.json')
    source_copy.file(identity['controller']['path'],'payload/controller.npz')
    source_copy.file(identity['configuration']['path'],'payload/configuration.json')
    mappings = {row['original']['path'] for row in source_copy.mapping}
    for index,row in enumerate(identity['sources']):
        if row['path'] in mappings: continue
        path=Path(row['path'])
        if path.is_relative_to(checkout) and path.parts[-3:-1]==('src','rage_mill'):
            relative='frozen/'+path.relative_to(checkout).as_posix()
        elif path.is_relative_to(checkout) and path.parent==checkout/'scripts':
            relative='frozen/'+path.relative_to(checkout).as_posix()
        elif path.name in ('model.xml','runtime-closure.json'):
            relative='payload/'+path.name
        else:
            relative=f'provenance/original/{index:03d}-{row["sha256"][:12]}-'+path.name
        need(not relative.endswith(('.so','.dylib','.dll','.bundle')) and '.git/' not in relative,
             'runtime binary or private Git source excluded')
        source_copy.file(row['path'],relative); mappings.add(row['path'])
    for row in inputs.get('additional_source_files',[]):
        read(row); path=Path(row['path'])
        need(path.is_relative_to(checkout) and 'canonical18' in path.name.lower()
             and path.suffix in ('.py','.json','.md')
             and path.relative_to(checkout).parts[0] in ('src','scripts','tests','configs','docs'),
             'additional source must be an explicit own18 benchmark file in original checkout')
        if row['path'] not in mappings:
            source_copy.file(row['path'],'supplemental/'+path.relative_to(checkout).as_posix())
            mappings.add(row['path'])
    license_text=read(inputs['license']).decode()
    need('MIT License' in license_text and 'Permission is hereby granted' in license_text,'project MIT license required')
    source_copy.file(inputs['license']['path'],'LICENSE')
    source_copy.file(checkout/'src/rage_mill/__init__.py','frozen/src/rage_mill/__init__.py')
    write(source/'original-path-mapping.json',dict(schema='pendulum_arena.canonical18_source_inventory.v1',
          original_identity=identity_row,files=source_copy.mapping,
          original_execution_paths_relocated=False,native_runtime_binaries_included=False,
          private_Git_history_included=False,portable_execution_identity=False))
    (source/'audit_source.py').write_text(auditor)
    (source/'audit_saved.py').write_text(auditor)
    write(source/'qualified-release-binding.json',dict(original_identity=identity_row,
          verified_release=inputs['release_verification'],combined_qualification=inputs['combined_qualification'],
          evidence_files_sha256=evidence_manifest['sha256'],native_audited_episodes=native_episodes,
          original_absolute_execution_paths_immutable=True,local_inventory_is_execution_identity=False))
    (source/'README.md').write_text(source_readme(identity_row['sha256'],stats,native_episodes))
    source_manifest=manifest(source,'pendulum_arena.canonical18_public_source_files.v1')
    media_root=out/'canonical18-media';media_root.mkdir()
    for name,row in (('canonical18.mp4',video['video']),('canonical18-telemetry.mp4',media['silent'])):
        with (media_root/name).open('xb') as stream: stream.write(read(row))
    media_copy=Copier(media_root)
    media_copy.file(media['review']['path'],'review.json')
    for index,row in enumerate(media.get('receipts',[])):
        media_copy.file(row['path'],f'provenance/{index:03d}-'+Path(row['path']).name)
    if media.get('music') is not None:
        music=media['music']
        with (media_root/'canonical18-telemetry-music.mp4').open('xb') as stream: stream.write(read(music['video']))
        for key in ('authorization','provenance'):
            media_copy.file(music[key]['path'],'music/'+key+'.json')
    (media_root/'audit_saved.py').write_text(auditor)
    media_manifest=manifest(media_root,'pendulum_arena.canonical18_public_media_files.v1')
    source_archive=archive(out/'source',out/'canonical18-source.tar.gz','')
    evidence_archive=archive(evidence,out/'canonical18-evidence.tar.gz','canonical18-evidence')
    media_archive=archive(media_root,out/'canonical18-media.tar.gz','canonical18-media')
    receipt=dict(schema='rage_mill.canonical18_public_evidence_assembly.v1',passed=True,
          actual_release=inputs['release_verification'],combined_qualification=inputs['combined_qualification'],
          source_manifest=source_manifest,evidence_manifest=evidence_manifest,media_manifest=media_manifest,
          source_archive=source_archive,evidence_archive=evidence_archive,media_archive=media_archive,
          actual_episodes=262,recorded_commands=393000,reserved_episodes=141,
          native_audited_episodes=native_episodes,native_audited_commands=1500*native_episodes,
          all262_saved_metrics=stats,scientific_video=pin(media_root/'canonical18.mp4'),
          silent_telemetry_video=pin(media_root/'canonical18-telemetry.mp4'),music_included=media.get('music') is not None,
          assembledAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),timestamps=stamps,
          private_Git_or_runtime_binaries_copied=False,original_candidate_export_modified=False,
          Site_files_modified=False,external_writes=0,physics_steps=0,controller_calls=0,new_reserved_labels=0,
          scope='Qualified original-path evidence staged only. Generic saved tape guards may reconstruct recorded reset identities; no physical reset, native dynamics or controller inference occurs.')
    write(out/'assets.json',dict(schema='rage_mill.canonical18_staged_assets.v1',files=[source_archive,evidence_archive,media_archive,
          source_manifest,evidence_manifest,media_manifest,pin(media_root/'canonical18.mp4'),pin(media_root/'canonical18-telemetry.mp4')]))
    return write(out/'receipt.json',receipt)


def manifest(root,schema):
    paths=list(root.rglob('*'))
    need(not any(path.is_symlink() for path in paths),'manifest refuses every symlink')
    rows=[dict(path=p.relative_to(root).as_posix(),sha256=digest(p),bytes=p.stat().st_size)
          for p in sorted(paths) if p.is_file()]
    need(len(rows)<=MAX_FILES and sum(row['bytes'] for row in rows)<=MAX_BYTES,'public inventory exceeds bounds')
    return write(root/'files.json',dict(schema=schema,files=rows,qualification_authority=False,
          source_runtime_identity_is_portable=False))


def source_readme(identity_sha,stats,native_episodes):
    return f'''# Canonical eighteen: completed recorded simulation evidence

The original qualification binds identity `{identity_sha}`, exact hanging source
calibration, fresh disjoint development20+100 and all141 published reserved
labels consumed exactly once, including actual218500. All262 actual episodes
are retained and rescored. Their minimum strict upright hold is
{stats['minimum_upright_seconds']} seconds. Cold and continuous native proof
coverage is {native_episodes} original episodes / {1500*native_episodes} commands.

The controller is classical native one-step affine-value action search using
observed q19/v19 state, its own700-step zero-terminal value tables,850 parking
commands and650 live route commands. Its single causal launch forecast starts
from the observed state. At most16 distinct delivered float32 action candidates
are evaluated per active command.50Hz describes simulated cadence. No learned,
hardware,20ms wall-clock,global-priority or external reproduction claim follows.

`metadata/identity.json`, every copied frozen source and all mapped provenance
retain their original bytes and absolute-path identities. Runtime manifests
retain file hashes as inventory; native runtime binaries and private Git are
excluded. This archive is not a portable execution identity. It exposes no
label activation, policy execution or path-rebinding loader.

Inspect these relative public bytes with Python3.11+ standard library:

    PYTHONDONTWRITEBYTECODE=1 python3 audit_source.py --root . --expected-files-sha256 PUBLISHED_SOURCE_FILES_SHA256

Add `--tables` in an existing NumPy environment to inspect stored38-state
arrays without importing any factory or MuJoCo. The evidence archive has its
own `files.json` and `audit_saved.py`; use its externally published manifest
hash. Source/hash inspection does not requalify or rerun the controller.

Project source and these adapters use the copied MIT license. Credit
Corben Sorenson/cartpole at b84c6413974d5c0831cdf261ce15e58ba65cd3f5 for
the published fourteen-link benchmark, mechanics, reset order and strict
scoring. The unchanged count-generic Rage Mill affine/predictive helpers are
included by exact hash. External packages retain their own licenses and are
not vendored. Scientific and silent media remain separate from optional music.
Marc's soundtrack request is exactly “super star by queen”; no alternative
track is substituted when title or source permission remains unresolved.
'''


def public_auditor_source():
    return r'''"""Hash-bound relative public inventory inspection; no model/controller/RNG."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,sys
sys.dont_write_bytecode=True

def need(ok,message):
    if not ok: raise ValueError(message)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1048576),b''):h.update(data)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(__doc__,allow_abbrev=False)
    parser.add_argument('--root',type=Path,default=Path(__file__).parent)
    parser.add_argument('--expected-files-sha256',required=True)
    parser.add_argument('--tables',action='store_true')
    args=parser.parse_args();root=args.root.resolve(strict=True)
    need(digest(root/'files.json')==args.expected_files_sha256,'manifest hash differs')
    public=json.loads((root/'files.json').read_text());rows=public['files']
    need(public['schema'] in ('pendulum_arena.canonical18_public_source_files.v1',
         'pendulum_arena.canonical18_public_evidence_files.v1','pendulum_arena.canonical18_public_media_files.v1')
         and len(rows)<=30000,'wrong or unbounded public inventory')
    seen=set();total=0
    for row in rows:
        text=row['path'];relative=PurePosixPath(text)
        need(text and '\\' not in text and not relative.is_absolute() and '..' not in relative.parts
             and relative.as_posix()==text and text not in seen and text!='files.json','unsafe duplicate path')
        current=root
        for part in relative.parts:
            current=current/part;need(not current.is_symlink(),'symlink archive entry')
        need(current.is_file() and current.resolve().is_relative_to(root)
             and current.stat().st_size==row['bytes'] and digest(current)==row['sha256'],'file differs')
        seen.add(text);total+=row['bytes']
    need(total<=3*1024**3,'public byte cap exceeded')
    paths=list(root.rglob('*'))
    need(not any(path.is_symlink() for path in paths)
         and {path.relative_to(root).as_posix() for path in paths if path.is_file()}==seen|{'files.json'},
         'unlisted or missing public file')
    if (root/'metadata/identity.json').is_file():
        identity=json.loads((root/'metadata/identity.json').read_text())
        need(identity['protocol']['n_links']==18 and identity['source_commit']=='b84c6413974d5c0831cdf261ce15e58ba65cd3f5','wrong own-count identity')
    if (root/'qualified-release-binding.json').is_file():
        binding=json.loads((root/'qualified-release-binding.json').read_text())
        need(digest(root/'metadata/identity.json')==binding['original_identity']['sha256'],'original source identity differs')
        mapping=json.loads((root/'original-path-mapping.json').read_text())['files']
        originals={row['original']['path']:row for row in mapping}
        for row in [identity['controller'],identity['configuration'],*identity['sources']]:
            item=originals.get(row['path'])
            need(item and item['original']==row,'original identity source omitted')
            path=root/item['relative_path']
            need(digest(path)==row['sha256'] and path.stat().st_size==row['bytes'],'original source mapping differs')
    if (root/'audit.json').is_file():
        audit=json.loads((root/'audit.json').read_text());release=json.loads((root/'metadata/verified-release.json').read_text())
        combined=json.loads((root/'metadata/combined-qualification.json').read_text())
        need(audit['passed'] and audit['controller_qualified'] and audit['actual_episodes']==262
             and audit['recorded_commands']==393000 and audit['reserved_episodes']==141
             and audit['heldout_selected_shadow_suffixes_byte_proved']==141
             and release['controller_qualified'] and release['n_links']==18 and release['reserved_episodes']==141
             and combined['controller_qualified'] and combined['all141_labels_consumed_once'],
             'complete original qualification lineage differs')
        need(digest(root/'metadata/identity.json')==audit['original_identity_sha256'],'original evidence identity differs')
    if args.tables:
        import numpy as np
        with np.load(root/'payload/controller.npz',allow_pickle=False) as data:
            shapes={'states':(651,38),'actions':(650,),'gains':(650,38),'offsets':(650,),
                    'balance_gain':(38,),'park_gain':(38,),'parking_target':(38,),
                    'native_defects':(650,38),'terminal_root':(38,38),'park_ticks':(),
                    'value_states':(701,38),'value_actions':(700,),'value_gains':(700,38),
                    'value_offsets':(700,),'value_roots':(701,38,38),'value_residuals':(701,38),
                    'value_Q':(38,38),'value_native_defects':(700,38)}
            need(set(data.files)==set(shapes),'unexpected full700 table fields')
            for key,shape in shapes.items():
                need(data[key].shape==shape and data[key].dtype==np.dtype('int64' if key=='park_ticks' else 'float64')
                     and np.isfinite(data[key]).all(),'table shape/dtype differs')
            need(data['park_ticks'].item()==850 and not np.any(data['value_roots'][-1])
                 and not np.any(data['value_residuals'][-1]) and not np.any(data['terminal_root']),
                 'parking or zero-terminal value differs')
            for short,full in [('states','value_states'),('actions','value_actions'),('gains','value_gains'),
                               ('offsets','value_offsets'),('native_defects','value_native_defects')]:
                need(data[short].tobytes()==data[full][:len(data[short])].tobytes(),'literal650/700 prefix differs')
            actions=data['value_actions']
            need(np.max(np.abs(actions))<=1 and actions.tobytes()==actions.astype(np.float32).astype(np.float64).tobytes(),
                 'actions are not deliveredfloat32')
            c=np.tril(np.ones((18,18)));g=c.T@c;q=np.eye(38);q[0,0]=q[19,19]=.1
            q[1:19,1:19]+=100*g;q[20:,20:]=g+.01*np.eye(18)
            need(data['value_Q'].tobytes()==q.tobytes(),'own18Q differs')
    need('mujoco' not in sys.modules and not any(name=='rage_mill' or name.startswith('rage_mill.') for name in sys.modules),
         'public inspection imported execution modules')
    print(json.dumps(dict(schema='pendulum_arena.canonical18_public_saved_hash_audit.v1',passed=True,
         files=len(seen),bytes=total,native_calls=0,controller_calls=0,reset_RNG_draws=0,new_reserved_labels=0,
         original_identity_is_portable=False,qualification_reexecuted=False,external_reproduction=False)))

if __name__=='__main__':main()
'''


def main():
    parser=argparse.ArgumentParser(__doc__,allow_abbrev=False)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--expected-inputs-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();row=pin(args.inputs)
    need(row['sha256']==args.expected_inputs_sha256,'prospective export inputs SHA required')
    inputs=document(row)
    # Refuse prequalification inputs before imports, copies or output creation.
    release=document(inputs.get('release_verification'))
    need(release.get('controller_qualified') is True and release.get('n_links')==18
         and release.get('reserved_episodes')==141 and release.get('once_only_label_claims_checked')==141,
         'actual qualified own18 release is mandatory; staging is not admission')
    qualification=document(inputs.get('combined_qualification'))
    need(qualification.get('controller_qualified') is True and qualification.get('passed') is True,
         'positive combined qualification is mandatory before staging')
    print(json.dumps(assemble(inputs,args.out)),flush=True)


if __name__=='__main__':main()
