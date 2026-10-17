"""Collect a saved-only all141 PHYSICS qualification before media rendering.

Requires four closed FULL independent gate audits, the unchanged own18 source
freeze, its passing independent bounded review and all original permanent
claims. A positive result authorizes no new episode and is not the typed media
release: verify_reserved_release remains a separate post-scientific-media gate.

Preparation interface:
--prepare --identity PATH --expected-identity-sha256 SHA --authority PATH
--expected-authority-sha256 SHA --initial-inventory PATH --freeze-review-owner DIR
--gate-audit-owner noisy20=DIR --gate-audit-owner noisy100=DIR
--gate-audit-owner exact20=DIR --gate-audit-owner video=DIR
--supervisor PATH --out FRESH_OWNER
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from canonical18_saved_qualification_support_v2 import (
    CLAIMS, ROOT, audit_coverage, closed_owner, finish_run, load_source, pinned_file,
    prepare_owner, read, read_pinned, require, start_run,
)
from rage_mill.canonical_evaluation import reserved_gates, validate_identity, value_sha256
from rage_mill.canonical_release import audit_source_snapshot
from rage_mill.canonical_reserved import _audit_storage
from rage_mill.canonical_runtime_evidence import audit_formal_runtime_owners

SCHEMA = 'rage_mill.canonical18_physics_qualification_inputs.v2'
OWNER_SCHEMA = 'rage_mill.canonical18_physics_qualification_owner.v2'
GATES = tuple(reserved_gates(18))
GATE_HELPER = ROOT / 'scripts/prepare_canonical18_reserved_saved_audit_v2.py'
AUDITOR = ROOT / 'scripts/audit_canonical18_reserved_saved.py'


def parse_gate_owners(values):
    result = {}
    for value in values:
        gate, separator, directory = value.partition('=')
        require(separator and gate in GATES and gate not in result and directory,
                'one distinct gate=absolute-owner argument per gate required')
        result[gate] = str(Path(directory).resolve(strict=True))
    require(set(result) == set(GATES), 'all four gate audit owners required')
    return result


def historical_claims(inventory_row, identity):
    inventory = read(inventory_row)
    require(inventory.get('ledger') == str(CLAIMS) and inventory.get('count18_namespace_exists') is False,
            'original predevelopment ledger inventory required')
    rows = inventory.get('historical_claim_files')
    require(isinstance(rows, list) and len(rows) == 294 and len({r['path'] for r in rows}) == 294,
            'all294 original sixteen/seventeen claim files required')
    base = CLAIMS / ('published-' + identity['source_commit'])
    expected = set()
    for count in (16, 17):
        root = base / f'count-{count}'
        require(root.is_dir() and not root.is_symlink(), 'immutable historical namespace missing')
        for path in root.rglob('*'):
            require(not path.is_symlink(), 'historical claim symlink refused')
            if path.is_file():
                expected.add(str(path))
        expected.add(str(base / f'count-{count}.lock'))
    require({r['path'] for r in rows} == expected and all(pinned_file(r['path']) == r for r in rows),
            'historical claim inventory changed, removed or extended')
    return dict(initial_inventory=inventory_row, historical_claim_files_verified=294,
                count16_labels=141, count17_labels=141, all_bytes_unchanged=True)


def validate_gate_result(result, *, gate, identity_row, authority_row, cohort_row):
    details = reserved_gates(18)[gate]; count = details['episodes']
    require(result.get('schema') == 'rage_mill.canonical18_independent_reserved_saved_audit.v1'
            and result.get('passed') is True and result.get('audit_scope') == 'full'
            and result.get('identity') == identity_row and result.get('authority') == authority_row
            and result.get('gate') == gate and result.get('cohort') == cohort_row
            and result.get('all_actual1500_strict_pass') is True
            and result.get('source_hashes_unchanged') is True
            and result.get('controller_qualified') is False,
            'complete own18 FULL independent saved audit required')
    require(all(type(result.get(key)) is int and result[key] == 0 for key in
            ('additional_native_calls', 'additional_controller_calls', 'additional_RNG_draws',
             'new_reserved_labels', 'physical_episodes')), 'saved audit must create no execution or attempts')
    episodes, proofs = result.get('episodes'), result.get('saved_episode_proofs')
    require(isinstance(episodes, list) and isinstance(proofs, list)
            and len(episodes) == len(proofs) == count and result.get('saved_proof_owner') is not None
            and result.get('saved_proof_cohort') is not None, 'one full saved proof per actual episode required')
    require(len({item['episode']['path'] for item in episodes}) == count
            and len({item['proof']['path'] for item in proofs}) == count, 'duplicate actual/proof rows refused')
    for index, (actual, proof) in enumerate(zip(episodes, proofs)):
        seed = details['seed_start'] + index
        score = actual.get('score', {})
        require(actual.get('seed') == proof.get('seed') == seed
                and score.get('success') is True and score.get('ticks') == 1500
                and score.get('complete') is True and score.get('finite') is True
                and score.get('rail_ok') is True and score.get('force_ok') is True
                and score.get('float32_delivery_ok') is True
                and type(score.get('longest_upright_ticks')) is int and score['longest_upright_ticks'] >= 250
                and proof.get('rawF64_commands_byte_proved') == 1500
                and proof.get('deliveredF32_commands_byte_proved') == 1500
                and proof.get('selected_native_pairs_byte_proved') == 650
                and proof.get('full_suffix_and_final1500_byte_identical') is True,
                'full1500 strict actual and complete raw/F32/selected/suffix/final proof required')
        native = proof.get('native_transition_summary', {})
        require(native.get('maximum_dynamics_defect') == native.get('maximum_continuous_replay_error') == 0.
                and native.get('continuous_replay_identical') is True,
                'zero-defect cold and continuous native saved proof required')
    require(result.get('saved_proof_total_native_maps') == sum(p['total_native_maps'] for p in proofs)
            and result.get('saved_proof_maximum_native_maps') == count*24450
            and 0 < result['saved_proof_total_native_maps'] <= count*24450,
            'complete saved-proof cost map totals/caps required')
    return details


def gate_summary(inputs, gate, *, identity, authority):
    directory = Path(inputs['gate_audit_owners'][gate])
    launch, audit_owner = closed_owner(directory)
    require(launch.get('schema') == 'rage_mill.canonical18_reserved_saved_audit_owner.v2'
            and launch.get('mode') == 'full' and launch.get('gate') == gate
            and launch.get('argv', [None, None])[1] == str(GATE_HELPER)
            and launch.get('output_path') == str(directory / 'result'), 'own bounded full gate audit owner required')
    original_inputs = read(launch['inputs'])
    require(original_inputs.get('identity') == inputs['identity']
            and original_inputs.get('authority') == inputs['authority']
            and original_inputs.get('worker') == pinned_file(AUDITOR), 'audit original source/input lineage differs')
    row = pinned_file(directory / 'result/receipt.json'); result = read(row)
    cohort_row = pinned_file(Path(authority['output_root']) / gate / 'receipt.json')
    details = validate_gate_result(result, gate=gate, identity_row=inputs['identity'],
                                  authority_row=inputs['authority'], cohort_row=cohort_row)
    require(result.get('prospective_inputs') == launch['inputs']
            and result.get('bounded_producer_source') == pinned_file(GATE_HELPER)
            and result.get('producer_source') == pinned_file(AUDITOR), 'saved audit producer/input pins differ')
    audit_coverage(result, launch)
    _audit_storage(inputs['authority'], authority, validate_identity(identity), gate, cohort_row)
    cohort = read(cohort_row)
    require(cohort.get('episode_receipts') == [item['episode'] for item in result['episodes']]
            and cohort.get('identity') == identity and cohort.get('freeze') == authority['freeze']
            and all(cohort.get(key) == value for key, value in details.items()),
            'audit must bind exact ordered complete actual cohort')
    # Recheck original actual and proof owners, not only cached audit summaries.
    actual_launch, actual_owner = closed_owner(Path(original_inputs['actual_owner']))
    proof_launch, proof_owner = closed_owner(Path(original_inputs['proof_owner']))
    require(actual_owner['launch'] == result['actual_owner']['launch']
            and actual_owner['completion'] == result['actual_owner']['completion']
            and proof_owner['launch'] == result['saved_proof_owner']['launch']
            and proof_owner['completion'] == result['saved_proof_owner']['completion']
            and actual_launch.get('authority') == proof_launch.get('authority') == inputs['authority']
            and actual_launch.get('gate') == proof_launch.get('gate') == gate,
            'original actual/proof owner identities differ from independent audit')
    proof_cohort = read(result['saved_proof_cohort'])
    require(proof_cohort.get('cohort') == cohort_row and proof_cohort.get('identity') == inputs['identity']
            and proof_cohort.get('passed') is True and proof_cohort.get('episodes_proved') == details['episodes']
            and proof_cohort.get('episode_proofs') == [p['proof'] for p in result['saved_episode_proofs']],
            'all independent saved proof rows must match original proof cohort')
    return dict(actual_cohort=cohort_row, episodes=details['episodes'], actual_owner=actual_owner,
        saved_policy_native_proof=result['saved_proof_cohort'], saved_proof_owner=proof_owner,
        independent_saved_audit=row, independent_saved_audit_owner=audit_owner,
        rawF64_and_deliveredF32_commands_each=1500*details['episodes'],
        selected_native_pairs=650*details['episodes'], full651state650action_raw_suffixes=details['episodes'],
        saved_proof_native_maps=result['saved_proof_total_native_maps'],
        saved_proof_native_map_cap=result['saved_proof_maximum_native_maps'],
        minimum_strict_hold_ticks=min(item['score']['longest_upright_ticks'] for item in result['episodes']))


def audit_freeze_review(inputs, *, identity, authority):
    frozen = read(authority['freeze'])
    require(frozen.get('schema') == 'rage_mill.canonical_release_freeze.v1'
            and frozen.get('identity') == identity and frozen.get('identity_sha256') == value_sha256(identity)
            and frozen.get('reserved_gates') == reserved_gates(18)
            and frozen.get('activation_authorized') is False
            and frozen.get('requires_prerequisite_release') is False
            and frozen.get('runtime_closure') == authority['runtime_closure'], 'own18 source/runtime freeze differs')
    for source in [identity['controller'], identity['configuration'], *identity['sources']]:
        read_pinned(source)
    audit_source_snapshot(frozen['source_snapshot'], identity)
    audit_formal_runtime_owners(frozen)
    directory = Path(inputs['freeze_review_owner'])
    launch, owner = closed_owner(directory)
    require(Path(launch.get('argv', ['', ''])[1]).name == 'audit_canonical18_release_readiness.py'
            and launch.get('phase') == 'freeze' and launch.get('output_path') == str(directory / 'output'),
            'original bounded independent freeze review required')
    row = pinned_file(directory / 'output/receipt.json'); review = read(row)
    require(review.get('schema') == 'rage_mill.canonical18_independent_readiness_review.v1'
            and review.get('passed') is True and review.get('phase') == 'freeze'
            and review.get('identity') == inputs['identity'] and review.get('identity_sha256') == value_sha256(identity)
            and review.get('freeze_review', {}).get('freeze') == authority['freeze']
            and review.get('source_hashes_unchanged') is True,
            'complete positive original-candidate independent freeze review required')
    original_inputs = read(launch['inputs']) if launch.get('inputs') else read(pinned_file(directory / 'inputs.json'))
    require(original_inputs.get('identity') == inputs['identity'] and original_inputs.get('freeze') == authority['freeze']
            and original_inputs.get('initial_inventory') == inputs['initial_inventory'], 'freeze review original inputs differ')
    # The existing readiness worker uses its own coverage schema.
    coverage = read(review['runtime_coverage'])
    from rage_mill.canonical_runtime_manifest import validate_runtime_manifest
    runtime_pins = validate_runtime_manifest(read(launch['runtime_closure']))
    loaded = coverage.get('actual_loaded_file_sha256')
    require(coverage.get('runtime_closure') == launch['runtime_closure']
            and coverage.get('runtime_complete') is True and coverage.get('uncovered_or_changed') == {}
            and isinstance(loaded, dict) and loaded and all(runtime_pins.get(p) == h for p, h in loaded.items()),
            'independent freeze-review runtime coverage differs')
    require(review.get('source_and_historical_claim_review', {}).get('historical_claim_files_verified') == 294,
            'independent review omitted immutable historical claims')
    return frozen, dict(receipt=row, owner=owner, source_bundle=frozen['source_snapshot']['bundle'],
                        source_commit=frozen['source_snapshot']['git_commit'], original_execution_paths_preserved=True)


def collect(inputs):
    identity, authority = read(inputs['identity']), read(inputs['authority'])
    require(validate_identity(identity).n_links == 18
            and authority.get('schema') == 'rage_mill.canonical_reserved_execution_authority.v1'
            and authority.get('activation_authorized') is True
            and authority.get('identity_sha256') == value_sha256(identity)
            and authority.get('namespace') == dict(schema='rage_mill.canonical_reserved_namespace.v1',
                source_commit=identity['source_commit'], n_links=18, gates=reserved_gates(18))
            and authority.get('claim_base') == str(CLAIMS), 'original authorized own18 namespace required')
    history = historical_claims(inputs['initial_inventory'], identity)
    frozen, freeze_review = audit_freeze_review(inputs, identity=identity, authority=authority)
    require(set(inputs.get('gate_audit_owners', {})) == set(GATES)
            and len(set(inputs['gate_audit_owners'].values())) == 4, 'four distinct FULL gate audits required')
    summaries = {gate: gate_summary(inputs, gate, identity=identity, authority=authority) for gate in GATES}
    ledger = CLAIMS / ('published-' + identity['source_commit']) / 'count-18'
    expected_labels = {f'seed-{seed}' for details in reserved_gates(18).values()
                       for seed in range(details['seed_start'], details['seed_start']+details['episodes'])}
    require(ledger.is_dir() and not ledger.is_symlink()
            and {p.name for p in ledger.iterdir()} == expected_labels | {'campaign.json', *('gate-'+g for g in GATES)}
            and len(expected_labels) == 141, 'exactly all141 original own18 labels, campaign and four gates required')
    for path in ledger.rglob('*'):
        require(not path.is_symlink(), 'claim aliases refused')
    require(all({p.name for p in directory.iterdir()} == {'claim.json'}
                for directory in ledger.iterdir() if directory.is_dir()), 'extra/replacement claim artifacts refused')
    labels = [pinned_file(ledger / name / 'claim.json') for name in sorted(expected_labels)]
    return dict(schema='rage_mill.canonical18_physics_qualification.v1', passed=True,
        controller_qualified=True, n_links=18, identity=inputs['identity'], identity_sha256=value_sha256(identity),
        freeze=authority['freeze'], campaign_authority=inputs['authority'],
        all141_labels_consumed_once=True, replaced_actual_attempts=0, reserved_episodes=141,
        reserved_commands=211500, reserved_gates=summaries, historical_claim_preservation=history,
        once_only_labels=labels, source_freeze_review=freeze_review,
        source_calibration=frozen['source_calibration'], development=frozen['development'],
        qualification_scope='Physics and saved causal/native evidence only; actual218500 may now be rendered.',
        typed_release_verified=False, scientific_media_verified=False,
        limitations='No hardware or 20ms wall-time claim. No new physics, inference, reset reconstruction, RNG or reserved attempt. Typed release verification follows scientific media.',
        owner_cost_totals={kind: {field: sum(value[kind][field] for value in summaries.values())
            for field in ('CPU_seconds', 'wall_seconds')} for kind in
            ('actual_owner', 'saved_proof_owner', 'independent_saved_audit_owner')})


def prepare(args):
    args._preparation_cpu, args._preparation_wall = time.process_time(), time.monotonic()
    identity, authority = pinned_file(args.identity), pinned_file(args.authority)
    require(identity['sha256'] == args.expected_identity_sha256
            and authority['sha256'] == args.expected_authority_sha256, 'prospective identity/authority hashes required')
    owners = parse_gate_owners(args.gate_audit_owner)
    for directory in [*owners.values(), str(args.freeze_review_owner.resolve(strict=True))]:
        closed_owner(Path(directory))
    candidate, auth = read(identity), read(authority)
    require(validate_identity(candidate).n_links == 18, 'own18 candidate required')
    inventory = pinned_file(args.initial_inventory)
    historical_claims(inventory, candidate)
    trees = [*owners.values(), str(args.freeze_review_owner.resolve(strict=True)), auth['output_root']]
    frozen = read(auth['freeze'])
    trees.append(frozen['source_snapshot']['fresh_clone'])
    ledger = CLAIMS / ('published-' + candidate['source_commit']) / 'count-18'
    require(ledger.is_dir(), 'all actual reserved claims must already exist; no activation')
    trees.append(str(ledger))
    inputs = dict(schema=SCHEMA, identity=identity, authority=authority, initial_inventory=inventory,
        freeze_review_owner=str(args.freeze_review_owner.resolve(strict=True)), gate_audit_owners=owners,
        historical_claim_files=read(inventory)['historical_claim_files'], _input_trees=trees)
    # Preload pure definitions, including the exact bounded gate helper whose
    # producer source and runtime lineage the collector verifies later.
    load_source(GATE_HELPER, 'canonical18_gate_helper_prospective')
    load_source(AUDITOR, 'canonical18_saved_auditor_prospective')
    return prepare_owner(args, inputs, schema=OWNER_SCHEMA, source=Path(__file__).resolve(),
                         extra_sources=(GATE_HELPER, AUDITOR, Path('/usr/bin/git')), cpu=180, wall=300)


def run(args):
    cpu, wall = time.process_time(), time.monotonic()
    row, inputs = start_run(args, SCHEMA)
    result = collect(inputs)
    finish_run(args, inputs, row, result, cpu=cpu, wall=wall)


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--run', action='store_true')
    for name in ('identity', 'authority', 'initial-inventory', 'freeze-review-owner', 'supervisor', 'inputs', 'out'):
        parser.add_argument('--' + name, type=Path, required=name == 'out')
    for name in ('expected-identity-sha256', 'expected-authority-sha256', 'expected-inputs-sha256'):
        parser.add_argument('--' + name)
    parser.add_argument('--gate-audit-owner', action='append', default=[])
    args = parser.parse_args(); require(args.prepare != args.run, 'one preparation or bounded run required')
    if args.prepare:
        require(all(getattr(args, key) is not None for key in ('identity', 'authority', 'initial_inventory',
            'freeze_review_owner', 'supervisor', 'expected_identity_sha256', 'expected_authority_sha256')),
            'complete prospective physics collector inputs required')
        prepare(args)
    else:
        require(args.inputs is not None and args.expected_inputs_sha256, 'pinned prospective inputs required'); run(args)


if __name__ == '__main__':
    main()
