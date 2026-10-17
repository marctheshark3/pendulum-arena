"""Use frozen scientific forward geometry with a saved-only, zero-RNG audit.

The original canonical_video frame and encoder implementation stays unchanged.
Its two audit callbacks are explicitly replaced in this rendering process by
saved-array/journal/source checks which never reconstruct a noisy reset.
The additional adapter provenance discloses this substitution.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time

import numpy as np

from rage_mill import canonical_benchmark as benchmark
from rage_mill import canonical_evaluation as evaluation
from rage_mill import canonical_video as renderer


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument('--episode-receipt', type=Path, required=True)
    parser.add_argument('--episode-receipt-sha256', required=True)
    parser.add_argument('--qualification-evidence', type=Path, required=True)
    parser.add_argument('--qualification-evidence-sha256', required=True)
    parser.add_argument('--links', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cpu, wall = time.process_time(), time.monotonic()
    require = evaluation.require
    row = evaluation.pinned_file(args.episode_receipt)
    authority = evaluation.pinned_file(args.qualification_evidence)
    require(row['sha256'] == args.episode_receipt_sha256
            and authority['sha256'] == args.qualification_evidence_sha256,
            'prospective actual episode and completed qualification evidence hashes required')
    evaluation.read_pinned(authority)
    expected_seed = 200000 + args.links * 1000 + 500
    calls = []

    def forbidden(*unused_args, **unused_kwargs):
        raise AssertionError('saved-state rendering forbids native/model/policy/reset/RNG execution')

    benchmark.CanonicalEnv = benchmark.hanging_state = forbidden
    evaluation.hanging_state = np.random.default_rng = forbidden

    def saved_only_audit(selected, *, identity=None, **unused):
        require(selected == row, 'scientific renderer changed the selected actual episode')
        document = json.loads(evaluation.read_pinned(selected))
        actual_identity = document['identity']
        protocol = evaluation.validate_identity(actual_identity)
        require(protocol.n_links == args.links and (identity is None or identity == actual_identity)
                and document.get('schema') == 'rage_mill.canonical_episode_evidence.v1'
                and document.get('stage') == 'reserved' and document.get('seed') == expected_seed
                and document.get('exact') is False and document.get('initialized') is True
                and document.get('identity_sha256') == evaluation.value_sha256(actual_identity)
                and document.get('source_pins_unchanged') is True
                and type(document.get('reset_count_after_initialization')) is int
                and document['reset_count_after_initialization'] == 0
                and document.get('invalid_reason') is None,
                'original complete unchanged own-count held-out episode required')
        claim = json.loads(evaluation.read_pinned(document['claim']))
        require(claim.get('schema') == 'rage_mill.canonical_reserved_episode_claim.v1'
                and claim.get('seed') == expected_seed and claim.get('exact') is False
                and claim.get('identity_sha256') == document['identity_sha256']
                and claim.get('freeze') == document.get('freeze'), 'original video claim lineage differs')
        evaluation.read_pinned(document['freeze'])
        evaluation.read_pinned(claim['ledger_label_claim'])
        with np.load(io.BytesIO(evaluation.read_pinned(document['recording'])), allow_pickle=False) as source:
            require(set(source.files) == {'states', 'actions', 'raw_actions', 'times', 'failure_state'},
                    'complete actual video tape fields required')
            arrays = {key: source[key].copy() for key in source.files}
        dimension = 2 * (args.links + 1)
        for key, shape in dict(states=(1501, dimension), actions=(1500,), raw_actions=(1500,),
                               times=(1501,), failure_state=(0, dimension)).items():
            require(arrays[key].shape == shape
                    and arrays[key].dtype == np.dtype('float32' if key == 'actions' else 'float64')
                    and np.isfinite(arrays[key]).all(), 'complete finite actual video tape differs: ' + key)
        delivered = np.clip(arrays['raw_actions'].astype(np.float32), -1., 1.)
        require(delivered.tobytes() == arrays['actions'].tobytes()
                and np.max(np.abs(arrays['times'] - np.arange(1501) * .02)) <= 1e-9,
                'raw/F32 delivery or saved native clock differs')
        score = benchmark.score_tape(arrays['states'], arrays['actions'], protocol)
        require(score == document.get('score') and score['success'], 'complete original video score differs')
        evaluation.audit_journal(evaluation.read_pinned(document['journal']),
                                 **{key: arrays[key] for key in ('states', 'actions', 'raw_actions', 'times')},
                                 invalid_reason=None)
        calls.append('saved arrays/journal/claims/source only; no reset identity reconstruction')
        return dict(document=document, score=score, **arrays)

    renderer.audit_episode = saved_only_audit
    result = renderer.render_saved_episode(row, args.out)
    require(len(calls) == 2, 'both unchanged renderer audit seams must use saved-only validation')
    evaluation.write_json_exclusive(args.out / 'adapter-provenance.json', {
        'schema': 'rage_mill.saved_only_scientific_render_adapter.v1', 'passed': True,
        'scientific_receipt': result, 'actual_episode': row, 'qualification_evidence': authority,
        'adapter_source': evaluation.pinned_file(__file__), 'forward_geometry_renderer_source': evaluation.pinned_file(renderer.__file__),
        'adapter_change': 'Only the renderer module audit_episode callback is replaced in memory by source/journal/claim/array checks. Original frame_image and encoder loop execute unchanged.',
        'saved_only_checks': calls, 'reset_identity_reconstructions': 0,
        'native_calls': 0, 'model_constructions': 0, 'controller_calls': 0, 'reset_RNG_draws': 0,
        'new_physical_episodes': 0, 'new_reserved_labels': 0, 'qualification_awarded': False,
        'CPU_seconds': time.process_time() - cpu, 'wall_seconds': time.monotonic() - wall})


if __name__ == '__main__':
    main()
