"""V2 saved-only ownership with exact historical-data boundaries.

No activation, policy, model or reset entry point is called by this module.
These sources deliberately remain outside the immutable candidate identity.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import time

from rage_mill.canonical_evaluation import (
    pinned_file, read_pinned, require, reserved_gates, validate_identity, value_sha256, write_json_exclusive,
)
from rage_mill.canonical_runtime_closure import runtime_binding_paths
from rage_mill.canonical_runtime_manifest import create_runtime_manifest, validate_runtime_manifest

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = Path('/home/marctheshark/.t3/artifacts/rage-mill-canonical-reserved-claims')
PUBLIC = 'b84c6413974d5c0831cdf261ce15e58ba65cd3f5'
LEGACY_SCHEMAS = {
    'rage_mill.canonical_reserved_campaign_claim.v1',
    'rage_mill.canonical_reserved_gate_claim.v1',
    'rage_mill.canonical_reserved_label_claim.v1',
}


def read(row):
    return json.loads(read_pinned(row))


def strict_path(value, *, directory=False, fresh=False):
    path = Path(value)
    require(path.is_absolute() and not path.is_symlink() and path.resolve(strict=not fresh) == path,
            'canonical absolute non-symlink path required')
    require(not path.exists() if fresh else path.is_dir() if directory else path.is_file(),
            'fresh output or regular input required')
    if fresh:
        require(path.parent.is_dir(), 'existing output parent required')
    return path


class PinClosure:
    def __init__(self, identity=None, *, authority_row):
        self.pins, self.seen = {}, set()
        self.identity_sources = {row['path']: row for row in (identity or {}).get('sources', [])}
        authority, frozen = saved_authority_metadata(authority_row, identity)
        snapshot = frozen['source_snapshot']
        self.clone = strict_path(snapshot['fresh_clone'], directory=True)
        self.clone_rows = {}
        for expected in snapshot['inventory']:
            relative = Path(expected['relative_path'])
            require(relative.parts and not relative.is_absolute() and '..' not in relative.parts,
                    'safe frozen inventory member path required')
            path = strict_path(self.clone / relative)
            require(str(path) not in self.clone_rows, 'distinct frozen inventory member required')
            row = dict(path=str(path), sha256=expected['sha256'], bytes=expected['bytes'])
            read_pinned(row)
            self.clone_rows[str(path)] = row
        require(self.clone_rows, 'positive exact frozen clone inventory required')
        self.include(authority_row, expand=False)
        self.include(authority['freeze'], expand=False)

    def legacy_data(self, row):
        path = strict_path(row['path'])
        if path.suffix != '.json' or not path.is_relative_to(CLAIMS):
            return False
        parts = path.relative_to(CLAIMS).parts
        if len(parts) < 3 or parts[0] != 'published-' + PUBLIC or parts[1] not in ('count-16', 'count-17'):
            return False
        value = read(row)
        count = int(parts[1].split('-')[1]); remainder = parts[2:]
        require(isinstance(value, dict) and value.get('schema') in LEGACY_SCHEMAS
                and value.get('namespace') == dict(schema='rage_mill.canonical_reserved_namespace.v1',
                    source_commit=PUBLIC, n_links=count, gates=reserved_gates(count)),
                'only exact published16/17 permanent claim schemas/namespaces are historical data')
        schema = value['schema']
        valid = schema == 'rage_mill.canonical_reserved_campaign_claim.v1' and remainder == ('campaign.json',)
        if schema == 'rage_mill.canonical_reserved_gate_claim.v1':
            valid = value.get('gate') in reserved_gates(count) and remainder == ('gate-'+value['gate'], 'claim.json')
        if schema == 'rage_mill.canonical_reserved_label_claim.v1':
            seed = value.get('seed'); details = reserved_gates(count).get(value.get('gate'))
            valid = (type(seed) is int and details is not None
                and details['seed_start'] <= seed < details['seed_start']+details['episodes']
                and remainder == (f'seed-{seed}', 'claim.json'))
        require(valid, 'legacy claim path/schema/published label differ')
        return True

    def include(self, row, *, expand=True):
        path = strict_path(row['path'])
        if path.is_relative_to(self.clone):
            require(self.clone_rows.get(str(path)) == {key:row[key] for key in ('path','sha256','bytes')},
                    'clone data must match an exact positively frozen inventory member')
            expand = False
        elif self.legacy_data(row):
            expand = False
        data = read_pinned(row)
        require(row['path'] not in self.pins or self.pins[row['path']] == row['sha256'], 'input pin conflict')
        self.pins[row['path']] = row['sha256']
        if not expand or row['path'] in self.seen:
            return
        self.seen.add(row['path'])
        if Path(row['path']).suffix == '.json':
            value = json.loads(data)
            if not (isinstance(value, dict) and 'initial_inventory' in value.get('schema', '')):
                self.walk(value)

    def walk(self, value):
        if isinstance(value, dict):
            if set(value) >= {'path', 'sha256', 'bytes'}:
                self.include(value)
                return
            for path, digest in value.get('source_hashes', {}).items():
                if not Path(path).is_absolute():
                    relative = Path(path)
                    require(relative.parts and '..' not in relative.parts,
                            'safe descriptive relative source inventory required')
                    absolute = str(ROOT / relative)
                    row = self.identity_sources.get(absolute)
                    require(row is not None and row['sha256'] == digest,
                            'relative descriptive pin must map exactly to frozen original identity source')
                    self.include(row, expand=False)
                    continue
                row = pinned_file(path)
                require(row['sha256'] == digest, 'prospective owner pin changed')
                self.include(row, expand=False)
            for key, item in value.items():
                if key != 'source_hashes':
                    self.walk(item)
        elif isinstance(value, list):
            for item in value:
                self.walk(item)

    def tree(self, directory):
        directory = strict_path(directory, directory=True)
        for path in sorted(directory.rglob('*')):
            require(not path.is_symlink(), 'symlink in input evidence tree refused')
            if path.is_file() and '.git' not in path.relative_to(directory).parts:
                self.include(pinned_file(path))


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def guards():
    import numpy as np
    import mujoco
    import rage_mill.canonical_benchmark as benchmark
    import rage_mill.canonical_evaluation as evaluation
    import rage_mill.canonical_replay as replay

    def forbidden(*args, **kwargs):
        raise AssertionError('saved-only qualification forbids model/native/controller/reset/RNG execution')

    benchmark.CanonicalEnv = replay.CanonicalEnv = forbidden
    benchmark.hanging_state = benchmark.record_episode = forbidden
    evaluation.hanging_state = evaluation.record_development_cohort = evaluation._record_episode = forbidden
    np.random.default_rng = np.random.RandomState = forbidden
    mujoco.MjModel = mujoco.MjData = mujoco.mj_step = mujoco.mj_forward = forbidden
    # Policy modules are intentionally not imported. Block their public
    # entry points as well if some other caller already loaded one.
    for name, module in tuple(sys.modules.items()):
        if name.startswith('rage_mill.canonical18'):
            for attribute in ('controller_factory', 'controller_from_arrays'):
                if hasattr(module, attribute):
                    setattr(module, attribute, forbidden)


def unchanged(pins):
    require(isinstance(pins, dict) and pins and all(pinned_file(p)['sha256'] == h for p, h in pins.items()),
            'complete prospective input/runtime pin closure changed')


def closed_owner(directory):
    directory = strict_path(directory, directory=True)
    rows = {key: pinned_file(directory / name) for key, name in
            (('launch', 'launch.json'), ('completion', 'completion.json'), ('log', 'worker.log'))}
    launch, done = read(rows['launch']), read(rows['completion'])
    require(done.get('schema') == 'rage_mill.next_stage_supervised_completion.v1'
            and done.get('recipe_path') == rows['launch']['path']
            and done.get('recipe_sha256') == rows['launch']['sha256']
            and type(done.get('child_returncode')) is int and done['child_returncode'] == 0
            and done.get('naturally_closed_zero') is True and done.get('failure') is None
            and done.get('interrupted') is False and done.get('wall_timeout') is False
            and done.get('recipe_unchanged') is True
            and done.get('owned_group_cleanup') == ['group_already_exited']
            and done.get('log_path') == rows['log']['path'] and done.get('log_sha256') == rows['log']['sha256'],
            'naturally closed zero/reaped original owner required')
    pins, stable = launch.get('source_hashes'), done.get('source_hashes_unchanged')
    require(isinstance(pins, dict) and isinstance(stable, dict) and set(pins) == set(stable)
            and all(flag is True for flag in stable.values()), 'complete unchanged owner closure required')
    unchanged(pins)
    supervisor = launch.get('supervisor')
    if supervisor is None:
        choices = [p for p in pins if Path(p).name == 'supervise-stage-v2.py']
        require(len(choices) == 1, 'one pinned bounded supervisor required')
        supervisor = pinned_file(choices[0])
    require(pins.get(supervisor['path']) == supervisor['sha256']
            and done.get('supervisor_source_sha256') == supervisor['sha256'], 'supervisor identity differs')
    for key, done_key in (('CPU_cap_s', 'CPU_cap_s_per_process'), ('wall_cap_s', 'wall_cap_s'),
                          ('address_cap_bytes', 'address_cap_bytes')):
        require(type(launch.get(key)) is int and launch[key] > 0 and done.get(done_key) == launch[key],
                'finite prospective resource caps differ')
    for key in ('aggregate_child_CPU_seconds', 'wall_seconds', 'peak_rss_kib'):
        require(type(done.get(key)) in (int, float) and math.isfinite(done[key]) and done[key] >= 0,
                'measured total owner cost required')
    return launch, dict(**rows, naturally_closed_zero=True, all_pins_unchanged=True,
        source_pin_count=len(pins), CPU_seconds=done['aggregate_child_CPU_seconds'],
        wall_seconds=done['wall_seconds'], peak_rss_kib=done['peak_rss_kib'])


def saved_authority_metadata(authority_row, identity):
    """Bind a positive independent freeze review without generic reset audits.

    This deliberately never invokes canonical_reserved._authority, audit_freeze,
    audit_episode or hanging_state. Full original owner pin closure and the
    independent review are read and rehashed; snapshot bytes are checked by the
    caller and the final collector also audits source Git/bundle metadata.
    """
    authority = read(authority_row)
    require(validate_identity(identity).n_links == 18
            and authority.get('schema') == 'rage_mill.canonical_reserved_execution_authority.v1'
            and authority.get('activation_authorized') is True
            and authority.get('claim_base') == str(CLAIMS)
            and authority.get('identity_sha256') == value_sha256(identity)
            and authority.get('namespace') == dict(schema='rage_mill.canonical_reserved_namespace.v1',
                source_commit=PUBLIC, n_links=18, gates=reserved_gates(18)),
            'original authorized own18 metadata required')
    frozen = read(authority['freeze'])
    require(frozen.get('schema') == 'rage_mill.canonical_release_freeze.v1'
            and frozen.get('identity') == identity and frozen.get('identity_sha256') == value_sha256(identity)
            and frozen.get('reserved_gates') == reserved_gates(18)
            and frozen.get('activation_authorized') is False
            and frozen.get('requires_prerequisite_release') is False
            and frozen.get('runtime_closure') == authority.get('runtime_closure'),
            'same original own18 source/runtime freeze required')
    review_row = authority.get('independent_freeze_review')
    require(isinstance(review_row, dict), 'positive independent freeze review pin required')
    review = read(review_row)
    directory = Path(review_row['path']).parent.parent
    launch, owner = closed_owner(directory)
    require(review_row['path'] == str(directory/'output/receipt.json')
            and launch.get('phase') == 'freeze'
            and Path(launch.get('argv',['',''])[1]).name == 'audit_canonical18_release_readiness.py'
            and review.get('schema') == 'rage_mill.canonical18_independent_readiness_review.v1'
            and review.get('passed') is True and review.get('phase') == 'freeze'
            and review.get('identity_sha256') == value_sha256(identity)
            and read(review['identity']) == identity
            and review.get('freeze_review',{}).get('freeze') == authority['freeze']
            and review.get('source_hashes_unchanged') is True,
            'unchanged closed independent review of this exact freeze required')
    coverage = read(review['runtime_coverage'])
    runtime_pins = validate_runtime_manifest(read(launch['runtime_closure']))
    actual = coverage.get('actual_loaded_file_sha256')
    require(coverage.get('runtime_closure') == launch['runtime_closure']
            and coverage.get('runtime_complete') is True and coverage.get('uncovered_or_changed') == {}
            and isinstance(actual,dict) and actual and all(runtime_pins.get(p)==h for p,h in actual.items()),
            'original independent freeze review runtime coverage required')
    return authority, frozen


def prepare_owner(args, inputs, *, schema, source, extra_sources=(), cpu=120, wall=240):
    start = getattr(args, '_preparation_wall', time.monotonic())
    started_cpu = getattr(args, '_preparation_cpu', time.process_time())
    out = strict_path(args.out, fresh=True)
    trees = inputs.pop('_input_trees', [])
    protected = [ROOT, CLAIMS, *(Path(p) for p in trees),
                 Path(inputs['identity']['path']).parent, Path(inputs['authority']['path']).parent]
    require(all(not (out == p or out.is_relative_to(p) or p.is_relative_to(out)) for p in protected),
            'fresh owner must be disjoint from checkout, ledger and immutable input roots')
    supervisor = pinned_file(strict_path(args.supervisor))
    closure = PinClosure(read(inputs['identity']), authority_row=inputs['authority']); closure.walk(inputs)
    for directory in trees:
        closure.tree(directory)
    for path in (source, Path(__file__).resolve(), *extra_sources,
                 Path(sys.executable).resolve(strict=True), Path(sys.prefix) / 'pyvenv.cfg', supervisor['path']):
        closure.include(pinned_file(path))
    # All consumer imports occur before runtime collection, but no execution
    # entry point, table validation, reset or numerical transition is called.
    import numpy
    import mujoco
    manifest = create_runtime_manifest(package_names=('numpy', 'mujoco'), extra_paths=list(closure.pins))
    pins = validate_runtime_manifest(manifest)
    out.mkdir(parents=False, exist_ok=False)
    runtime = write_json_exclusive(out / 'runtime-closure.json', manifest)
    pins[runtime['path']] = runtime['sha256']
    inputs.update(runtime_closure=runtime, output_path=str(out / 'result'), source_hashes=dict(sorted(pins.items())))
    input_row = write_json_exclusive(out / 'inputs.json', inputs)
    pins[input_row['path']] = input_row['sha256']
    launch = write_json_exclusive(out / 'launch.json', dict(
        schema=schema, argv=[str(Path(sys.executable).absolute()), str(source), '--run', '--inputs', input_row['path'],
            '--expected-inputs-sha256', input_row['sha256'], '--out', str(out / 'result')],
        cwd=str(out), environment=dict(PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1',
            CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            GIT_OPTIONAL_LOCKS='0'), CPU_cap_s=cpu, wall_cap_s=wall, address_cap_bytes=8*1024**3,
        output_path=str(out / 'result'), log=str(out / 'worker.log'), receipt=str(out / 'completion.json'),
        source_hashes=dict(sorted(pins.items())), runtime_closure=runtime, supervisor=supervisor,
        inputs=input_row, maximum_native_commands=0, maximum_model_constructions=0, controller_calls=0,
        reset_RNG_draws=0, new_physical_episodes=0, new_reserved_claims=0,
        gate=inputs.get('gate'), mode=inputs.get('mode'),
        purpose='Saved-only own18 ' + inputs['schema']))
    preparation = write_json_exclusive(out / 'preparation.json', dict(
        schema=schema + '.preparation', metadata_only=True, launch=launch, inputs=input_row,
        runtime_closure=runtime, supervisor=supervisor, native_commands=0, controller_calls=0,
        reset_RNG_draws=0, CPU_seconds=time.process_time()-started_cpu, wall_seconds=time.monotonic()-start))
    print(json.dumps(dict(preparation=preparation, launch=launch)), flush=True)
    return launch


def start_run(args, schema):
    row = pinned_file(args.inputs)
    require(row['sha256'] == args.expected_inputs_sha256, 'prospective inputs hash required')
    inputs = read(row)
    require(inputs.get('schema') == schema, 'own-count prospective input schema differs')
    unchanged(inputs['source_hashes'])
    validate_runtime_manifest(read(inputs['runtime_closure']))
    require(str(args.out) == inputs.get('output_path') and args.out.is_absolute()
            and args.out.resolve() == args.out and not args.out.exists() and not args.out.is_symlink(),
            'prospectively bound create-only saved audit output required')
    guards()
    return row, inputs


def finish_run(args, inputs, row, result, *, cpu, wall):
    runtime_pins = validate_runtime_manifest(read(inputs['runtime_closure']))
    actual = {path: pinned_file(path)['sha256'] for path in runtime_binding_paths(
        package_names=(), include_stdlib=False, include_loaded=True)}
    missing = {path: digest for path, digest in actual.items() if runtime_pins.get(path) != digest}
    require(not missing, 'actual loaded runtime coverage failed: ' + json.dumps(missing))
    unchanged(inputs['source_hashes'])
    args.out.mkdir(parents=False, exist_ok=False)
    coverage = write_json_exclusive(args.out / 'runtime-coverage.json', dict(
        schema='rage_mill.canonical18_saved_qualification_runtime.v1', runtime_closure=inputs['runtime_closure'],
        prospective_inputs=row, actual_loaded_file_sha256=actual, uncovered_or_changed=missing,
        runtime_complete=True, all_runtime_pins_unchanged=True, execution_guards_active=True,
        native_commands=0, model_constructions=0, controller_calls=0, reset_RNG_draws=0))
    result.update(prospective_inputs=row, runtime_closure=inputs['runtime_closure'], runtime_coverage=coverage,
        bounded_producer_source=pinned_file(sys.argv[0]), source_hashes_unchanged=True,
        additional_native_calls=0, additional_controller_calls=0, additional_RNG_draws=0,
        physical_episodes=0, new_reserved_labels=0, CPU_seconds=time.process_time()-cpu,
        wall_seconds=time.monotonic()-wall)
    receipt = write_json_exclusive(args.out / 'receipt.json', result)
    print(json.dumps(dict(receipt=receipt, passed=True)), flush=True)
    return receipt


def audit_coverage(result, launch):
    runtime = launch['runtime_closure']
    require(result.get('runtime_closure') == runtime, 'saved audit runtime lineage differs')
    runtime_pins = validate_runtime_manifest(read(runtime))
    coverage = read(result['runtime_coverage'])
    actual = coverage.get('actual_loaded_file_sha256')
    require(coverage.get('runtime_closure') == runtime
            and coverage.get('prospective_inputs') == launch.get('inputs')
            and coverage.get('runtime_complete') is True and coverage.get('uncovered_or_changed') == {}
            and coverage.get('all_runtime_pins_unchanged') is True and coverage.get('execution_guards_active') is True
            and isinstance(actual, dict) and actual
            and all(runtime_pins.get(p) == h and pinned_file(p)['sha256'] == h for p, h in actual.items())
            and all(type(coverage.get(k)) is int and coverage[k] == 0 for k in
                    ('native_commands', 'model_constructions', 'controller_calls', 'reset_RNG_draws')),
            'complete guarded saved-only runtime coverage required')
