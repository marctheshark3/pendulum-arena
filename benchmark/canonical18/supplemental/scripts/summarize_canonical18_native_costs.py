"""Census original own18 native work from immutable owner/file evidence.

This is metadata inspection only: no NumPy, MuJoCo, policy, reset or numerical
module is imported. One held native map consists of four mj_step calls (each
already uses MuJoCo RK4). Analytical CasADi/solver evaluations are not native
MuJoCo maps. Unknown counters remain null; declared ceilings remain separate.
Copied owner/proof receipts are inventory, never additional executed work.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET
import zipfile

CATEGORIES = (
    'actual_physical_commands', 'outer_shadow_native_maps',
    'live_inner_native_candidates', 'shadow_inner_native_candidates',
    'recorded_input_inner_native_candidates', 'cold_native_proof_maps',
    'continuous_native_proof_maps', 'component_native_maps',
)
ZERO_WORKERS = {
    'audit_canonical18_candidate_metadata.py',
    'audit_canonical18_native_value_episode_saved.py',
    'audit_canonical18_reserved_saved.py',
    'audit_canonical18_development_telemetry.py',
    'verify_canonical_release.py', 'verify_canonical_reserved_release.py',
    'render_canonical_episode_video.py', 'render_telemetry_video.py',
    'export_canonical18_public_evidence.py',
    'summarize_canonical18_owners.py', 'summarize_canonical18_native_costs.py',
}


def need(value, message):
    if not value:
        raise ValueError(message)


def nonnegative(value):
    need(type(value) is int and value >= 0, 'nonnegative integer work counter required')
    return value


class Reader:
    def __init__(self):
        self.cache = {}

    def pin(self, path):
        path = Path(path).resolve(strict=True)
        need(path.is_file(), 'regular metadata input required')
        stat = path.stat(); key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.cache:
            with path.open('rb') as stream:
                self.cache[key] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return dict(path=str(path), bytes=stat.st_size, sha256=self.cache[key])

    def read(self, row):
        need(isinstance(row, dict) and {'path','sha256'} <= set(row), 'pinned original input required')
        actual = self.pin(row['path'])
        need(actual['path'] == row['path'] and actual['sha256'] == row['sha256']
             and ('bytes' not in row or actual['bytes'] == row['bytes']), 'pinned original input changed')
        return json.loads(Path(row['path']).read_text())

    def path(self, path):
        return self.read(self.pin(path))


def npz_shape(path, field):
    """Read a bounded numeric .npy header; never load arrays or execute NumPy."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist(); name = field+'.npy'
        need(names.count(name) == 1, 'one expected NPZ field required')
        with archive.open(name) as stream:
            need(stream.read(6) == b'\x93NUMPY', 'NPY field header required')
            version = stream.read(2)
            need(version in (b'\x01\x00', b'\x02\x00', b'\x03\x00'), 'supported NPY format required')
            size = struct.unpack('<H' if version == b'\x01\x00' else '<I',
                                 stream.read(2 if version == b'\x01\x00' else 4))[0]
            need(size <= 65536, 'bounded NPY header required')
            header = ast.literal_eval(stream.read(size).decode('latin1'))
            need(isinstance(header, dict) and isinstance(header['shape'], tuple)
                 and 'O' not in header['descr'], 'numeric NPY shape required')
            return header['shape']


def argv_value(argv, name):
    return argv[argv.index(name)+1] if name in argv else None


def result_counts(values=None):
    return {key: 0 if values is None else values.get(key, 0) for key in CATEGORIES}


def unknown_counts():
    return {key: None for key in CATEGORIES}


def original_owners(root, reader):
    """Original absolute recipe/receipt relation excludes publication copies."""
    owners = []; ignored = 0
    for path in sorted(root.rglob('launch.json')):
        try:
            launch = reader.path(path)
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
        completion_path = path.parent/'completion.json'
        if (launch.get('receipt') != str(completion_path.resolve())
                or not isinstance(launch.get('argv'), list)):
            ignored += 1; continue
        completion = None
        if completion_path.exists():
            try:
                candidate = reader.path(completion_path)
            except json.JSONDecodeError:
                candidate = None  # An in-progress writer is not a closed owner.
            if (candidate is not None
                    and candidate.get('schema') == 'rage_mill.next_stage_supervised_completion.v1'
                    and candidate.get('recipe_path') == str(path.resolve())
                    and candidate.get('recipe_sha256') == reader.pin(path)['sha256']):
                completion = candidate
        owners.append((path, launch, completion))
    return owners, ignored


def native_bound(launch):
    for key in ('maximum_total_native_commands_including_proof','maximum_total_native_commands',
                'maximum_native_commands','maximum_native_held_map_commands',
                'maximum_benchmark_component_commands'):
        if type(launch.get(key)) is int and launch[key] >= 0:
            return dict(value=launch[key], field=key)
    return dict(value=None, field=None)


def classify(owner, launch, completion, reader):
    """Return disjoint counters with explicit measured/source-derived bases."""
    argv = launch['argv']; names = {Path(value).name for value in argv if isinstance(value,str)}
    output = Path(launch['output_path']) if launch.get('output_path') else owner/'output'
    receipt_path = output/'receipt.json'
    receipt = reader.path(receipt_path) if receipt_path.is_file() else None
    evidence = [reader.pin(receipt_path)] if receipt is not None else []
    schema = receipt.get('schema') if receipt else None
    counts = result_counts(); details = {}; basis = 'recorded counters'
    def read_row(row):
        value = reader.read(row); evidence.append(reader.pin(row['path'])); return value
    def proof_counts(row):
        proof = read_row(row)
        need(proof.get('schema') == 'rage_mill.canonical_recorded_native_replay.v1', 'original native proof required')
        need(Path(row['path']).is_relative_to(output), 'a copied/referenced predecessor proof is not new work')
        return nonnegative(proof['one_step_transitions']), nonnegative(proof['continuous_transitions'])
    if schema == 'rage_mill.canonical18_plant_parity.v1':
        pairs = nonnegative(receipt['transition_count'])
        derivative = nonnegative(receipt['native_derivative_held_map_calls'])
        counts['component_native_maps'] = 2*pairs+derivative
        details = dict(paired_local_and_published_maps=2*pairs, derivative_native_maps=derivative,
                       transition_pairs=pairs, map_implementations_per_pair=2)
        basis = 'instrumented derivative counter plus pinned paired public/local source loops'
    elif schema == 'rage_mill.canonical18_planning_model.v1':
        counts['component_native_maps'] = nonnegative(receipt['native_cold_map_commands'])
    elif schema == 'rage_mill.canonical18_orbit_component.v1':
        counts['component_native_maps'] = nonnegative(receipt['native_orbit_map_commands'])
    elif schema == 'rage_mill.canonical18_parking_component.v1':
        counts['component_native_maps'] = sum(nonnegative(case['ticks']) for case in receipt['cases'])
        details = dict(parking_component_commands=counts['component_native_maps'], full_task_episodes=0)
    elif schema == 'rage_mill.canonical18_bridge.v1':
        counts['component_native_maps'] = nonnegative(receipt['full_reference_ticks'])
        basis = 'pinned completed synthesis source: exactly one native cold map for each stored reference action'
    elif schema == 'rage_mill.canonical18_saved_model_independent_audit.v1':
        counts['component_native_maps'] = nonnegative(receipt['native_held_map_calls'])
    elif (not receipt and names & {'audit_canonical18_saved_model.py','audit_canonical18_saved_model_v2.py'}):
        log_path = Path(completion['log_path']); text = log_path.read_text()
        evidence.append(reader.pin(log_path))
        need('ValueError: actual loaded source/native runtime was not prospectively frozen' in text,
             'failed numerical audit has no verified post-numerical marker')
        source = next(Path(v) for v in argv if Path(v).name in
                      ('audit_canonical18_saved_model.py','audit_canonical18_saved_model_v2.py'))
        source_text = source.read_text(); evidence.append(reader.pin(source))
        need('for index, (state, action) in enumerate(zip(states, actions)):' in source_text
             and 'for direction in directions:' in source_text
             and '[[0, 150, 300, 449]]' in source_text
             and source_text.index('for index, (state, action)') < source_text.index('actual loaded source/native runtime'),
             'bounded completed-loop source proof differs')
        input_path = argv_value(argv,'--inputs'); inputs = reader.path(input_path); evidence.append(reader.pin(input_path))
        component_path = inputs['independent_component_arrays']
        need(launch['source_hashes'].get(component_path) == reader.pin(component_path)['sha256'], 'component header must be prospectively pinned')
        components = npz_shape(component_path,'component_states')[0]
        counts['component_native_maps'] = (components+4)*7
        details = dict(components=components+4, native_maps_per_component=7,
                       completed_before_runtime_coverage_failure=True, numerical_success_does_not_repair_failed_owner=True)
        basis = 'post-loop failure traceback + pinned source: base1 + two directions*2 + action endpoints2 per component'
    elif schema == 'rage_mill.canonical18_native_value_hanging_diagnostic.v1':
        counts['actual_physical_commands'] = sum(nonnegative(case['ticks']) for case in receipt['cases'])
        counts['live_inner_native_candidates'] = sum(nonnegative(case['native_candidate_commands']) for case in receipt['cases'])
        need(all(case['native_value_telemetry']['native_forecast_errors'] == 0 for case in receipt['cases']),
             'failed forecast attempts cannot be converted to four completed mj_steps')
    elif schema == 'rage_mill.canonical_fixed_candidate_component.v1':
        cohort = read_row(receipt['cohort']); decisions = read_row(receipt['profile_decisions'])['decisions']
        counts['actual_physical_commands'] = sum(nonnegative(read_row(row)['score']['ticks']) for row in cohort['episode_receipts'])
        live_seen = set()
        for decision in decisions:
            counts['outer_shadow_native_maps'] += nonnegative(decision['completed_forecast_commands'])
            for forecast in decision['forecasts']:
                shadow = forecast.get('native_value_shadow_telemetry'); live = forecast.get('native_value_live_telemetry')
                need(isinstance(shadow,dict) and isinstance(live,dict), 'actual saved candidate counters unavailable')
                need(shadow['native_forecast_errors'] == live['native_forecast_errors'] == 0,
                     'failed forecast attempts do not establish complete native maps')
                counts['shadow_inner_native_candidates'] += nonnegative(shadow['native_candidate_commands'])
                # The same live reporting object can appear under multiple profile rows.
                key = decision['seed']
                if key not in live_seen:
                    counts['live_inner_native_candidates'] += nonnegative(live['native_candidate_commands']); live_seen.add(key)
        if receipt.get('source_calibration'):
            calibration = read_row(receipt['source_calibration'])
            cold,continuous = proof_counts(calibration['native_replay_proof'])
            counts['cold_native_proof_maps'] += cold; counts['continuous_native_proof_maps'] += continuous
    elif schema == 'rage_mill.canonical18_development_native_cohort_proof.v1':
        counts['cold_native_proof_maps'] = nonnegative(receipt['cold_native_maps'])
        counts['continuous_native_proof_maps'] = nonnegative(receipt['continuous_native_maps'])
        details = dict(original_saved_development_episodes=receipt['episodes_proved'], physical_episodes_added=0)
    elif schema == 'rage_mill.canonical18_saved_policy_cohort_proof.v1':
        for row in receipt['episode_proofs']:
            proof = read_row(row); counter = proof['cost_counts']
            need(Path(row['path']).is_relative_to(output), 'copied predecessor policy proof is not new work')
            counts['outer_shadow_native_maps'] += nonnegative(counter['outer_shadow_native_steps'])
            counts['shadow_inner_native_candidates'] += nonnegative(counter['shadow_inner_native_candidates'])
            counts['recorded_input_inner_native_candidates'] += nonnegative(counter['recorded_input_inner_native_candidates'])
            cold,continuous = proof_counts(proof['native_transition_proof'])
            need(cold+continuous == counter['saved_native_proof_maps'], 'proof native counter mismatch')
            counts['cold_native_proof_maps'] += cold; counts['continuous_native_proof_maps'] += continuous
        need(sum(counts.values()) == receipt['total_native_maps'], 'proof cohort total differs')
        details = dict(causal_saved_input_reexecution=True, original_actual_inner_counters_reconstructed=False)
    elif schema == 'rage_mill.canonical_cohort_evidence.v1' and receipt.get('stage') == 'reserved':
        episodes = [read_row(row) for row in receipt['episode_receipts']]
        counts['actual_physical_commands'] = sum(nonnegative(episode['score']['ticks']) for episode in episodes)
        if all(episode['score']['success'] and episode['score']['ticks'] == 1500 for episode in episodes):
            counts['outer_shadow_native_maps'] = sum(nonnegative(episode['identity']['policy_parameters']['forecast_horizon_ticks']) for episode in episodes)
        else:
            counts['outer_shadow_native_maps'] = None
        counts['live_inner_native_candidates'] = counts['shadow_inner_native_candidates'] = None
        details = dict(original_inner_candidate_counters_retained=False,
                       causal_proof_counts_cannot_replace_actual_unknown_counters=True)
        basis = 'actual episode ticks and successful fixed-source launch horizon; original inner counts unrecorded'
    elif 'pytest' in argv:
        tests = [Path(value.split('::')[0]) for value in argv if '.py' in value and not value.startswith('--')]
        benchmark = [path for path in tests if path.name == 'test_canonical18_benchmark.py']
        log_path = Path(completion['log_path']); log = log_path.read_text(); evidence.append(reader.pin(log_path))
        junit = next((value.split('=',1)[1] for value in argv if value.startswith('--junitxml=')),None)
        if junit:
            report = ET.parse(junit).getroot(); evidence.append(reader.pin(junit))
            suites = [node for node in report.iter('testsuite')]
            passing = bool(suites) and all(int(node.get('errors','0')) == int(node.get('failures','0'))
                                          == int(node.get('skipped','0')) == 0 for node in suites)
        else:
            passing = bool(re.search(r'\b\d+ passed\b',log)) and not re.search(r'\b\d+ skipped\b',log)
        need(completion['child_returncode'] == 0 and passing,
             'test skip/failure leaves native command count unresolved')
        if benchmark:
            source = benchmark[0]
            if not source.is_absolute():
                source = next(Path(path) for path in launch['source_hashes'] if Path(path).name == source.name)
            evidence.append(reader.pin(source)); tree = ast.parse(source.read_text())
            function = next(node for node in tree.body if isinstance(node,ast.FunctionDef)
                            and node.name == 'test_eighteen_uses_own_38_state_native_serialized_capsules')
            need(not any(isinstance(node,(ast.For,ast.While)) for node in ast.walk(function)), 'benchmark test loop requires reviewed accounting')
            steps = [node for node in ast.walk(function) if isinstance(node,ast.Call)
                     and isinstance(node.func,ast.Attribute) and node.func.attr == 'step'
                     and isinstance(node.func.value,ast.Name) and node.func.value.id == 'env']
            counts['component_native_maps'] = len(steps)
            details = dict(repeated_unit_component_commands=len(steps), fresh_qualification_episodes=0,
                           benchmark_test_passed_without_skip=True)
        else:
            need(launch.get('maximum_native_commands') == 0, 'unreviewed suite has no verified zero-native contract')
        basis = 'passing pinned source suite; benchmark actual two-step unit repeated per owner, other suites mock-only'
    elif (names & ZERO_WORKERS or (receipt and receipt.get('schema') == 'rage_mill.canonical_saved_state_video.v1')):
        need(native_bound(launch)['value'] in (0,None), 'read-only source unexpectedly declares native work')
        basis = 'pinned known saved-only/metadata/render source; no native dynamics invocation'
    else:
        return unknown_counts(), dict(reason='No supported source/receipt classifier; no zero inferred'), evidence, 'unknown'
    return counts, details, evidence, basis


def census(root):
    root = Path(root).resolve(strict=True); reader = Reader()
    originals, ignored = original_owners(root,reader); rows = []; active = []
    for path,launch,closed in originals:
        common = dict(owner=str(path.parent.relative_to(root)),launch=reader.pin(path),
                      declared_native_map_ceiling=native_bound(launch),argv=launch['argv'])
        if closed is None:
            active.append({**common,'status':'active_or_unclosed','native_held_maps':None,
                           'native_mj_step_calls':None,'reason':'No closed completion snapshot; incomplete work is not inferred.'})
            continue
        completion_path = path.parent/'completion.json'
        pins = launch.get('source_hashes',{})
        mismatches=[]
        for source,digest in pins.items():
            try:
                if reader.pin(source)['sha256'] != digest: mismatches.append(source)
            except (OSError,ValueError): mismatches.append(source)
        log_matches = None
        if closed.get('log_sha256') and closed.get('log_path'):
            log_matches = reader.pin(closed['log_path'])['sha256'] == closed['log_sha256']
        try:
            need(log_matches is not False, 'closed owner log bytes changed')
            irrelevant_test_changes = []
            if mismatches and 'pytest' in launch['argv']:
                selected = [Path(value.split('::')[0]) for value in launch['argv']
                            if '.py' in value and not value.startswith('--')]
                test_sources = []
                for selected_path in selected:
                    actual = (selected_path if selected_path.is_absolute() else
                              next(Path(source) for source in pins if Path(source).name == selected_path.name))
                    need(reader.pin(actual)['sha256'] == pins.get(str(actual)), 'executed test source changed')
                    test_sources.append(actual.read_text())
                # Broad preparation manifests can include unused sibling tools.
                # Keep their current mismatch visible, but do not turn an
                # unchanged mock-only executed suite into unknown native work.
                for changed in mismatches:
                    need(Path(changed).name not in {p.name for p in selected}
                         and all(Path(changed).stem not in text and str(Path(changed)) not in text
                                 for text in test_sources)
                         and closed.get('source_hashes_unchanged',{}).get(changed) is True,
                         'changed relevant source prevents test accounting')
                    irrelevant_test_changes.append(changed)
            elif mismatches:
                raise ValueError('prospective source/input bytes no longer match')
            counts,details,evidence,basis = classify(path.parent,launch,closed,reader)
            if irrelevant_test_changes:
                details['unused_sibling_preparation_pins_changed_after_test_closure'] = irrelevant_test_changes
                details['executed_test_sources_still_hash_exact'] = True
        except (ValueError,KeyError,StopIteration,OSError,zipfile.BadZipFile) as error:
            counts,details,evidence,basis = unknown_counts(),dict(reason=str(error)),[],'unknown'
        exact = sum(counts.values()) if all(value is not None for value in counts.values()) else None
        lower = sum(value for value in counts.values() if value is not None)
        cpu,wall = closed['aggregate_child_CPU_seconds'],closed['wall_seconds']
        need(all(type(value) in (float,int) and math.isfinite(value) and value>=0 for value in (cpu,wall)), 'finite actual owner costs required')
        rows.append({**common,'completion':reader.pin(completion_path),'naturally_closed_zero':closed.get('naturally_closed_zero'),
            'child_returncode':closed.get('child_returncode'),'aggregate_child_CPU_seconds':cpu,'wall_seconds':wall,
            'source_pin_count':len(pins),'source_pin_mismatches':mismatches,'original_log_hash_matches':log_matches,
            'work':counts,'native_held_maps':exact,'known_native_held_maps_lower_bound':lower,
            'native_mj_step_calls':None if exact is None else 4*exact,
            'known_native_mj_step_calls_lower_bound':4*lower,'count_basis':basis,'details':details,
            'evidence':evidence})
    totals={key:sum(row['work'][key] for row in rows if row['work'][key] is not None) for key in CATEGORIES}
    unknown=[row['owner'] for row in rows if row['native_held_maps'] is None]
    lower=sum(row['known_native_held_maps_lower_bound'] for row in rows)
    return dict(schema='rage_mill.canonical18_native_cost_inventory.v1',root=str(root),
        collector_source=reader.pin(__file__),
        snapshotAt=datetime.now(timezone.utc).isoformat(),original_closed_owners=len(rows),owners=rows,
        active_or_unclosed_owners=active,copied_nonoriginal_launch_files_excluded=ignored,
        exact_closed_owner_native_held_maps=None if unknown else lower,
        exact_all_started_owner_native_held_maps=None if unknown or active else lower,
        known_closed_native_held_maps_lower_bound=lower,
        exact_closed_owner_mj_step_calls=None if unknown else 4*lower,
        known_closed_mj_step_calls_lower_bound=4*lower,unknown_closed_owners=unknown,
        known_category_totals=totals,
        unknown_category_owners={key:[row['owner'] for row in rows if row['work'][key] is None] for key in CATEGORIES},
        aggregate_closed_owner_CPU_seconds=sum(row['aggregate_child_CPU_seconds'] for row in rows),
        sum_closed_owner_wall_seconds=sum(row['wall_seconds'] for row in rows),
        wall_times_may_overlap=True,
        count_contract='A completed native held map is four5ms mj_step calls; RK4 evaluations inside mj_step are not separately counted. Analytical CasADi/optimizer evaluations are excluded from native map totals and remain part of measured owner CPU/wall cost.',
        duplication_contract='One original absolute launch/completion pair per owner; publication copies and referenced predecessor receipts are not additional executed work. Identical repeated experiments at different original owners are charged again.',
        scope='All original supervised owners found under this evidence root, including failures. Active/unclosed work is unknown and excluded from closed totals. Ad hoc/setup/preparation work outside supervised owners, and any uninstrumented hidden partial native attempts, are not assigned zero cost.',
        census_native_calls=0,census_controller_calls=0,census_reset_RNG_draws=0,census_new_physical_episodes=0)


def main():
    parser=argparse.ArgumentParser(__doc__,allow_abbrev=False)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=census(args.root)
    with args.out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(receipt=Reader().pin(args.out),closed_owners=result['original_closed_owners'],
          known_native_held_maps_lower_bound=result['known_closed_native_held_maps_lower_bound'],
          unknown_closed_owners=result['unknown_closed_owners'],active_or_unclosed_owners=len(result['active_or_unclosed_owners']))))


if __name__=='__main__':main()
