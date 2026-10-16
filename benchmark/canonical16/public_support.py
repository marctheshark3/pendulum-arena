"""NEW public relocation/hash adapters; not part of the historical controller freeze."""
from __future__ import annotations
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
FROZEN=HERE/'frozen'
IDENTITY_SHA256='36076035602ee2cab11af5e777a977409823ad4a11b52eba0c998c853f93e607'
FREEZE_SHA256='28d97cd6dd8545bdaa5b6c4b9100903fd295e14342f32419cde73a791d1e9571'
QUALIFICATION_SHA256='48b865a129f6352e50cf850e750a7fd6f2199af64e6d500dcadc431568dd7c01'
FILES_SHA256='2caafd9fe7e56af6a74b005d907c56b85e1582b7d5cb83234764e8ac2a47bff1'
SOURCE_MANIFEST_SHA256='97366571a60e69a7936ba77033f4d58e1952570fdff714fc239c6de332bb1af1'
COHORTS=(('calibration','calibration/episodes/receipt.json',3167109,1,True,'development'),
 ('development20','development20/episodes/receipt.json',3168000,20,False,'development'),
 ('development100','development100/episodes/receipt.json',3168100,100,False,'development'),
 ('noisy20','reserved/noisy20/receipt.json',216000,20,False,'reserved'),
 ('noisy100','reserved/noisy100/receipt.json',216100,100,False,'reserved'),
 ('exact20','reserved/exact20/receipt.json',216200,20,True,'reserved'),
 ('video','reserved/video/receipt.json',216500,1,False,'reserved'))

def require(ok,message):
    if not ok:raise ValueError(message)

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):digest.update(part)
    return digest.hexdigest()

def read_json(path):
    def invalid(value):raise ValueError(f'nonfinite JSON constant:{value}')
    return json.loads(Path(path).read_text(),parse_constant=invalid)

def safe_join(base,relative):
    require(isinstance(relative,str)and relative and not Path(relative).is_absolute(),'relative public path required')
    parts=Path(relative).parts
    require('..'not in parts,'path traversal refused')
    base=Path(base).resolve(strict=True)
    path=base/relative
    require(not path.is_symlink()and path.resolve(strict=True).is_relative_to(base),'symlink/path escape refused')
    return path

def verify_file(path,expected,size=None):
    require(Path(path).is_file()and sha256(path)==expected,f'file digest differs:{path}')
    require(size is None or Path(path).stat().st_size==size,f'file size differs:{path}')
    return Path(path)

def verify_source():
    manifest_path=HERE/'frozen-manifest.json'
    verify_file(manifest_path,SOURCE_MANIFEST_SHA256)
    manifest=read_json(manifest_path)
    require(manifest['schema']=='pendulum_arena.canonical16_curated_source.v1','wrong source manifest')
    source_by_hash={}
    for name,item in manifest['files'].items():
        path=safe_join(FROZEN,name);verify_file(path,item['sha256'],item['bytes'])
        source_by_hash.setdefault((item['sha256'],item['bytes']),[]).append(path)
    for name,item in manifest['new_adapter_files'].items():verify_file(safe_join(HERE,name),item['sha256'])
    return manifest,source_by_hash

def load_proof(proof_root):
    """Verify every public byte first. Original absolute paths remain untouched."""
    root=Path(proof_root).resolve(strict=True)
    source_manifest,source_by_hash=verify_source()
    manifest_path=root/'files.json';verify_file(manifest_path,FILES_SHA256)
    files=read_json(manifest_path)
    require(files['schema']=='pendulum-arena.evidence-files.v1','wrong evidence manifest')
    by_hash={};seen=set();byte_count=0
    for item in files['files']:
        name=item['path'];require(name not in seen,'duplicate manifest path');seen.add(name)
        path=safe_join(root,name);verify_file(path,item['sha256'],item['bytes'])
        by_hash.setdefault((item['sha256'],item['bytes']),[]).append(path);byte_count+=item['bytes']
    def resolve(row):
        require(isinstance(row,dict)and {'path','sha256','bytes'}<=row.keys(),'original pinned row required')
        choices=by_hash.get((row['sha256'],row['bytes']),[])
        require(bool(choices),f'original evidence row not exported:{row["sha256"]}')
        return choices[0]
    identity_path=root/'metadata/identity.json';freeze_path=root/'metadata/freeze.json'
    qualification_path=root/'metadata/qualification.json'
    verify_file(identity_path,IDENTITY_SHA256);verify_file(freeze_path,FREEZE_SHA256)
    verify_file(qualification_path,QUALIFICATION_SHA256)
    identity,freeze,qualification=map(read_json,(identity_path,freeze_path,qualification_path))
    require(freeze['identity']==identity and qualification['controller_qualified']is True
            and qualification['actual_reserved_episodes']==qualification['actual_reserved_successes']==141,
            'historical qualification/identity mismatch')
    require(qualification['identity']['sha256']==IDENTITY_SHA256 and qualification['freeze']['sha256']==FREEZE_SHA256,
            'historical qualification binding differs')
    require(source_manifest['original_identity_sha256']==IDENTITY_SHA256
            and source_manifest['original_freeze_sha256']==FREEZE_SHA256,'source/evidence binding differs')
    require(len(identity['sources'])==22,'expected21Python source rows+runtime manifest')
    for item in identity['sources']:
        require(bool(source_by_hash.get((item['sha256'],item['bytes']))),'missing exact frozen source')
    for key in ('controller','configuration'):resolve(identity[key])
    jobs=[]
    for label,path,start,count,exact,stage in COHORTS:
        cohort=read_json(root/path)
        require(cohort['identity']==identity and cohort['stage']==stage and cohort['seed_start']==start
                and cohort['episodes']==count and cohort['exact']is exact
                and len(cohort['episode_receipts'])==count,'cohort labels/identity differ')
        for index,row in enumerate(cohort['episode_receipts']):
            episode=read_json(resolve(row))
            require(episode['identity']==identity and episode['stage']==stage
                    and episode['seed']==start+index and episode['exact']is exact
                    and episode['score']['success']is True and episode['score']['ticks']==1500
                    and episode['reset_count_after_initialization']==0
                    and episode['source_pins_unchanged']is True,'incomplete/different historical episode')
            for key in ('recording','journal','claim'):resolve(episode[key])
            jobs.append((label,episode,row))
    require(len(jobs)==262 and len({(label,e['seed'])for label,e,_ in jobs})==262,'262 distinct original episodes required')
    proof=read_json(root/'policy-proof/receipt.json')
    require(proof['passed']is True and proof['episodes_audited']==141
            and proof['raw_and_float32_commands_byte_proved']==211500
            and proof['actual_benchmark_episodes']==0 and proof['new_benchmark_labels']==0,
            'original heldout command proof differs')
    for case in proof['cases']:
        require(case['passed']and case['checked_command_prefix']==1500
                and case['all_selected_shadow_live_suffix_bytes']and case['final_state1500_byte_equal'],
                'heldout fullsuffix proof failed')
        for key in ('episode_receipt','selected_prediction','recomputed_commands','command_journal'):resolve(case[key])
    calibration=read_json(root/'calibration/source-calibration.json')
    require(calibration['initial_state_gap']==0. and calibration['maximum_dynamics_defect']==0.
            and calibration['trajectory_integrity']and calibration['is_feasible'],'exact source calibration differs')
    resolve(calibration['episode_receipt']);resolve(calibration['native_replay_proof'])
    return dict(root=root,identity=identity,freeze=freeze,qualification=qualification,jobs=jobs,
                proof=proof,manifest=files,manifest_files=len(seen),manifest_bytes=byte_count,
                source_by_hash=source_by_hash,resolve=resolve)

def controller_loading_view(proof):
    """NEW local view only; no historical JSON, bytes or identity hash is replaced."""
    view=copy.deepcopy(proof['identity'])
    for key in ('controller','configuration'):
        item=view[key];item['path']=str(proof['resolve'](item).resolve())
    for item in view['sources']:
        choices=proof['source_by_hash'][(item['sha256'],item['bytes'])]
        item['path']=str(choices[0].resolve())
    return view

def load_controller(proof_root,*,allow_runtime_mismatch=False):
    """Construct frozen policy from a verified NEW loading view; never step it here.

    A caller can use the returned policy in an explicitly new diagnostic. This
    constructor is not a historical qualification or a new reserved activation.
    """
    proof=load_proof(proof_root)
    frozen_path=str((FROZEN/'src').resolve())
    require(not any(name=='rage_mill'or name.startswith('rage_mill.')for name in sys.modules),
            'run public loader in a fresh interpreter; existing rage_mill imports refused')
    sys.path.insert(0,frozen_path)
    evaluation=importlib.import_module('rage_mill.canonical_evaluation')
    current=evaluation.runtime_metadata();expected=proof['identity']['runtime']
    differences={key:{'expected':value,'actual':current.get(key)}for key,value in expected.items()if current.get(key)!=value}
    require(current.get('mujoco')=='3.3.5'and all(value=='1'for value in current.get('threads',{}).values()),
            'MuJoCo3.3.5 and one BLAS/OMP thread required')
    require(not differences or allow_runtime_mismatch,
            'runtime differs; explicit allow_runtime_mismatch permits exploratory loading only')
    view=controller_loading_view(proof)
    module=importlib.import_module('rage_mill.canonical16_profile_policy')
    policy=module.controller_factory(view)
    return policy,{'original_identity_sha256':IDENTITY_SHA256,'loading_view_is_new':True,
                   'runtime_differences':differences,'historical_evidence_modified':False,
                   'native_steps_executed_by_loader':0,'runtime_binary_hashes_verified':False,'qualification':False}
