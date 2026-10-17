"""Hash-bound relative public inventory inspection; no model/controller/RNG."""
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
