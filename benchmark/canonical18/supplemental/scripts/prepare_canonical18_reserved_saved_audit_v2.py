"""Prepare/run one bounded saved-only reserved gate audit after owners close.

--prepare --mode actual-only|full --identity PATH --expected-identity-sha256 SHA
--authority PATH --expected-authority-sha256 SHA --gate noisy20|noisy100|exact20|video
--actual-owner DIR [--proof-owner DIR] --supervisor PATH --out FRESH_OWNER

Execute only the emitted launch.json through the separately pinned supervisor.
Neither preparation nor execution consumes a label or runs physics/policy/RNG.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
from types import SimpleNamespace

from canonical18_saved_qualification_support_v2 import (
    ROOT, closed_owner, finish_run, load_source, pinned_file, prepare_owner,
    read, require, start_run,
)
from rage_mill.canonical_evaluation import reserved_gates, validate_identity, value_sha256

SCHEMA = 'rage_mill.canonical18_reserved_saved_audit_inputs.v2'
OWNER_SCHEMA = 'rage_mill.canonical18_reserved_saved_audit_owner.v2'
WORKER = ROOT / 'scripts/audit_canonical18_reserved_saved.py'


def worker_args(inputs):
    return SimpleNamespace(identity=Path(inputs['identity']['path']),
        expected_identity_sha256=inputs['identity']['sha256'], authority=Path(inputs['authority']['path']),
        expected_authority_sha256=inputs['authority']['sha256'], gate=inputs['gate'],
        actual_owner=Path(inputs['actual_owner']), expected_actual_sha256=inputs['actual_launch']['sha256'],
        proof_owner=Path(inputs['proof_owner']) if inputs.get('proof_owner') else None,
        expected_proof_sha256=inputs['proof_launch']['sha256'] if inputs.get('proof_launch') else None)


def prepare(args):
    args._preparation_cpu, args._preparation_wall = time.process_time(), time.monotonic()
    require(args.mode in ('actual-only', 'full') and (args.proof_owner is not None) == (args.mode == 'full'),
            'full mode requires one closed proof owner; actual-only accepts none')
    identity, authority = pinned_file(args.identity), pinned_file(args.authority)
    require(identity['sha256'] == args.expected_identity_sha256
            and authority['sha256'] == args.expected_authority_sha256, 'explicit actual identity/authority hashes required')
    candidate, auth = read(identity), read(authority)
    require(validate_identity(candidate).n_links == 18 and auth['identity_sha256'] == value_sha256(candidate),
            'own eighteen identity and authority required')
    actual_launch, actual_owner = closed_owner(args.actual_owner)
    require(actual_launch.get('mode') == 'reserved' and actual_launch.get('gate') == args.gate
            and actual_launch.get('authority') == authority, 'original actual gate owner differs')
    inputs = dict(schema=SCHEMA, mode=args.mode, identity=identity, authority=authority, gate=args.gate,
        actual_owner=str(args.actual_owner.resolve(strict=True)), actual_launch=actual_owner['launch'],
        proof_owner=None, proof_launch=None, worker=pinned_file(WORKER),
        _input_trees=[str(args.actual_owner.resolve(strict=True)), str(Path(auth['output_root']) / args.gate)])
    if args.proof_owner is not None:
        proof_launch, proof_owner = closed_owner(args.proof_owner)
        require(proof_launch.get('mode') == 'proof' and proof_launch.get('gate') == args.gate
                and proof_launch.get('authority') == authority, 'original saved proof gate owner differs')
        inputs.update(proof_owner=str(args.proof_owner.resolve(strict=True)), proof_launch=proof_owner['launch'])
        inputs['_input_trees'].append(str(args.proof_owner.resolve(strict=True)))
    # Imports only. The unchanged auditor is called solely by the bounded worker.
    load_source(WORKER, 'canonical18_reserved_saved_prospective')
    return prepare_owner(args, inputs, schema=OWNER_SCHEMA, source=Path(__file__).resolve(),
                         extra_sources=(WORKER,), cpu=120, wall=240)


def run(args):
    cpu, wall = time.process_time(), time.monotonic()
    row, inputs = start_run(args, SCHEMA)
    require(inputs['mode'] in ('actual-only', 'full')
            and (inputs.get('proof_owner') is not None) == (inputs['mode'] == 'full'), 'declared audit scope differs')
    require(pinned_file(WORKER) == inputs['worker'], 'original own18 saved auditor source changed')
    worker = load_source(WORKER, 'canonical18_reserved_saved_bounded')
    result = worker.audit(worker_args(inputs))
    result.update(audit_scope=inputs['mode'], producer_source=inputs['worker'])
    finish_run(args, inputs, row, result, cpu=cpu, wall=wall)


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--run', action='store_true')
    for name in ('identity', 'authority', 'actual-owner', 'proof-owner', 'supervisor', 'inputs', 'out'):
        parser.add_argument('--' + name, type=Path, required=name == 'out')
    for name in ('expected-identity-sha256', 'expected-authority-sha256', 'expected-inputs-sha256'):
        parser.add_argument('--' + name)
    parser.add_argument('--gate', choices=tuple(reserved_gates(18)))
    parser.add_argument('--mode', choices=('actual-only', 'full'))
    args = parser.parse_args(); require(args.prepare != args.run, 'one prepare or bounded run required')
    if args.prepare:
        require(all(getattr(args, key) is not None for key in ('identity', 'authority', 'actual_owner',
            'supervisor', 'gate', 'mode', 'expected_identity_sha256', 'expected_authority_sha256')),
            'complete prospective gate audit arguments required')
        prepare(args)
    else:
        require(args.inputs is not None and args.expected_inputs_sha256, 'pinned inputs required'); run(args)


if __name__ == '__main__':
    main()
