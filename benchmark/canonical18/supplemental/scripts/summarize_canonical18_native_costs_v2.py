"""Add reviewed zero-native metadata owners to the immutable v1 cost census.

All actual, candidate, replay and partial/unknown classifications delegate to
v1 unchanged. This wrapper only recognizes explicitly reviewed saved-only
programs and hash-pinned additive producer reviews. It imports standard library
modules only and never runs a producer, test suite, policy, simulator or reset.

Optional --reviewed-zero-producers JSON has schema
rage_mill.canonical18_reviewed_zero_native_producers.v1 and producers rows:
{source:{path,sha256,bytes}, native_held_maps:0, rationale:NONEMPTY,
 expected_receipt_schema:STRING_OR_NULL}. The producer must be an actual argv
entry and a matching prospective owner source pin. Reviews do not fill unknown
ad hoc/preparation costs and cannot replace measured nonzero/partial counters.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).with_name('summarize_canonical18_native_costs.py')
ZERO_SOURCE_WORKERS = {
    'audit_canonical18_release_readiness.py': ('rage_mill.canonical18_independent_readiness_review.v1',),
    'prepare_canonical18_release.py': ('rage_mill.canonical18_release_preparation.v1',),
    'prepare_canonical18_reserved_saved_audit.py': ('rage_mill.canonical18_independent_reserved_saved_audit.v1',),
    'collect_canonical18_physics_qualification.py': ('rage_mill.canonical18_physics_qualification.v1',),
}
PURE_METADATA_TESTS = {'test_canonical18_native_costs.py', 'test_canonical18_native_costs_v2.py',
                       'test_canonical18_public_export.py', 'test_canonical18_saved_qualification.py'}


def load_base():
    spec=importlib.util.spec_from_file_location('canonical18_immutable_native_census_v1',BASE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def read_reviews(path,base):
    if path is None: return None,[]
    reader=base.Reader();row=reader.pin(path);document=reader.read(row)
    base.need(document.get('schema')=='rage_mill.canonical18_reviewed_zero_native_producers.v1'
              and isinstance(document.get('producers'),list),'explicit reviewed-zero producer registry required')
    entries=[];seen=set()
    for item in document['producers']:
        source=item.get('source')
        base.need(isinstance(source,dict) and {'path','sha256','bytes'} <= set(source)
                  and Path(source['path']).suffix=='.py','explicit Python producer source pin required')
        actual=reader.pin(source['path'])
        base.need(all(actual[key]==source[key] for key in ('path','sha256','bytes'))
                  and source['path'] not in seen and type(item.get('native_held_maps')) is int
                  and item['native_held_maps']==0 and isinstance(item.get('rationale'),str)
                  and item['rationale'].strip()
                  and (item.get('expected_receipt_schema') is None
                       or isinstance(item['expected_receipt_schema'],str)),
                  'exact distinct producer hash, zero-native review and rationale required')
        seen.add(source['path']);entries.append(item)
    return row,entries


def pinned_programs(launch,reader):
    programs={}
    for value in launch['argv']:
        path=Path(value)
        if value.endswith('.py') and path.is_absolute() and path.is_file():
            row=reader.pin(path)
            if launch.get('source_hashes',{}).get(row['path'])==row['sha256']:
                programs[row['path']]=row
    return programs


def zero_classification(owner,launch,completion,reader,base,reviews):
    output=Path(launch.get('output_path',owner/'output'))
    receipt_path=output/'receipt.json'
    if not receipt_path.is_file() and (output/'preparation.json').is_file():
        receipt_path=output/'preparation.json'
    receipt=reader.path(receipt_path) if receipt_path.is_file() else None
    evidence=[reader.pin(receipt_path)] if receipt is not None else []
    programs=pinned_programs(launch,reader)
    def admit(reason,source=None,extra=None):
        if source is not None:evidence.append(source)
        return base.result_counts(),dict(reviewed_zero_native_reason=reason,
            referenced_predecessor_work_is_not_new_execution=True,qualification_pass_inferred=False,
            **(extra or {})),evidence,'source-reviewed zero native held-map execution'
    for original,row in programs.items():
        name=Path(original).name
        if name in ZERO_SOURCE_WORKERS and receipt and receipt.get('schema') in ZERO_SOURCE_WORKERS[name]:
            zero_field='additional_native_calls' if name in ('prepare_canonical18_reserved_saved_audit.py',
                                                             'collect_canonical18_physics_qualification.py') else 'native_commands'
            base.need(type(receipt.get(zero_field)) is int and receipt[zero_field]==0,
                      'saved-only original producer must explicitly report zero new native work')
            if name!='prepare_canonical18_release.py':
                coverage=reader.read(receipt['runtime_coverage']);evidence.append(reader.pin(receipt['runtime_coverage']['path']))
                base.need(coverage.get('runtime_complete') is True and coverage.get('uncovered_or_changed')=={}
                          and type(coverage.get('native_commands')) is int and coverage['native_commands']==0,
                          'guarded saved-only producer runtime coverage must pass')
                loaded=coverage.get('actual_loaded_file_sha256')
                base.need(isinstance(loaded,dict) and loaded
                          and all(launch['source_hashes'].get(path)==digest
                                  and reader.pin(path)['sha256']==digest for path,digest in loaded.items()),
                          'actual loaded source/native dependencies must remain prospectively bound')
            return admit('Reviewed readiness, source copying or guarded saved qualification source; no native dynamics.',row,
                         dict(saved_reset_identity_reconstruction_may_occur=name=='prepare_canonical18_release.py'))
    # Guarded pytest wrappers report failures too. Late coverage failure does
    # not create dynamics: guards are installed before tests are invoked.
    if receipt and receipt.get('schema')=='rage_mill.canonical18_saved_qualification_source_tests.v1':
        base.need(all(type(receipt.get(key)) is int and receipt[key]==0
                      for key in ('native_calls','controller_calls','reset_RNG_draws')),
                  'guarded source test zero-execution report differs')
        choices=[]
        for original,row in programs.items():
            text=Path(original).read_text()
            if ('from canonical18_saved_qualification_support import guards' in text
                    and 'guards()' in text and 'pytest.main(' in text
                    and text.index('guards()')<text.index('pytest.main(')):
                choices.append(row)
        base.need(len(choices)==1,'one exact guard-before-pytest source wrapper required')
        return admit('Pinned native/RNG guards precede source-only tests; retained late failures remain failures.',choices[0],
                     dict(source_test_passed=receipt.get('passed'),child_returncode=completion['child_returncode']))
    if 'pytest' in launch['argv']:
        selected=[Path(v.split('::')[0]) for v in launch['argv'] if '.py' in v and not v.startswith('--')]
        if selected and all(path.name in PURE_METADATA_TESTS for path in selected):
            text=Path(completion['log_path']).read_text()
            base.need(completion.get('child_returncode')==0 and base.re.search(r'\b\d+ passed\b',text)
                      and not base.re.search(r'\b\d+ skipped\b',text), 'complete passing metadata-only suite required')
            for selected_path in selected:
                choices=[path for path in launch['source_hashes'] if Path(path).name==selected_path.name]
                base.need(len(choices)==1 and reader.pin(choices[0])['sha256']==launch['source_hashes'][choices[0]],
                          'exact reviewed metadata test source required')
                evidence.append(reader.pin(choices[0]))
            return admit('Explicit reviewed metadata/export/census rejection fixtures use no physics.')
    for review in reviews:
        source=review['source']
        if source['path'] in programs and programs[source['path']]==source:
            expected=review.get('expected_receipt_schema')
            base.need(expected is None or receipt is not None and receipt.get('schema')==expected,
                      'reviewed producer output schema differs')
            return admit(review['rationale'],source,dict(explicit_producer_review=review,
                retained_failure_does_not_become_success=completion['child_returncode']!=0))
    return None


def census(root,reviewed_zero_producers=None):
    base=load_base();review_row,reviews=read_reviews(reviewed_zero_producers,base)
    original=base.classify
    def classify(owner,launch,completion,reader):
        try:
            old=original(owner,launch,completion,reader)
        except ValueError as error:
            old=(base.unknown_counts(),dict(reason=str(error)),[],'unknown')
        # Never overwrite any measured nonzero work or partial original actual
        # attribution (notably reserved inner-candidate unknowns).
        if (all(value is not None for value in old[0].values())
                or any(value is not None for value in old[0].values())):
            return old
        return zero_classification(owner,launch,completion,reader,base,reviews) or old
    base.classify=classify
    result=base.census(root)
    result['schema']='rage_mill.canonical18_native_cost_inventory.v2'
    result['base_collector_source']=result['collector_source']
    result['collector_source']=base.Reader().pin(__file__)
    result['reviewed_zero_producer_registry']=review_row
    result['reviewed_zero_native_metadata_owners']=[row['owner'] for row in result['owners']
        if row['count_basis']=='source-reviewed zero native held-map execution']
    result['v2_scope']='Only additive reviewed-zero metadata classifications. Immutable v1 numerical categories, original recorded/proof attribution, stale75 pin disclosures, active unknowns and external unmeasured setup/preparation scope are preserved.'
    return result


def main():
    parser=argparse.ArgumentParser(__doc__,allow_abbrev=False)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reviewed-zero-producers',type=Path)
    args=parser.parse_args();result=census(args.root,args.reviewed_zero_producers)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    base=load_base()
    print(json.dumps(dict(receipt=base.Reader().pin(args.out),closed_owners=result['original_closed_owners'],
          exact_closed_native_maps=result['exact_closed_owner_native_held_maps'],
          known_native_maps_lower_bound=result['known_closed_native_held_maps_lower_bound'],
          unknown_closed_owners=result['unknown_closed_owners'],
          active_or_unclosed_owners=len(result['active_or_unclosed_owners']))))


if __name__=='__main__':main()
