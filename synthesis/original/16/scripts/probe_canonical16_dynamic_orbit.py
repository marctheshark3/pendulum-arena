"""Registered three-frequency discrete fast bent-orbit and four-case native discriminator."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import casadi as ca
import mujoco
import numpy as np
from rage_mill.canonical_benchmark import CanonicalEnv,CanonicalProtocol,absolute_angles,record_episode
from rage_mill.canonical16_orbit import refine_cyclic
from rage_mill.canonical16_dynamic_orbit import discrete_harmonic_initializer
from rage_mill.canonical16_trajectory import trajectory_feedback


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('model','design','out'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    model_meta=json.loads((args.model/'receipt.json').read_text())
    if mujoco.__version__!='3.3.5' or not model_meta['passed'] or model_meta['n_links']!=16:
        raise ValueError('verified canonical16 planning model and MuJoCo3.3.5 required')
    with np.load(args.design,allow_pickle=False) as s:design={k:s[k].copy() for k in s.files}
    args.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    transition=ca.Function.load(str(args.model/'transition.casadi'));first=ca.Function.load(str(args.model/'first.casadi'))
    q=design['Q'];env=CanonicalEnv(CanonicalProtocol(n_links=16));profiles=[];candidates=[];native_maps=0
    if env.xml_sha256!=model_meta['xml_sha256']:raise ValueError('actual canonical16 plant identity differs from verified model')
    static_gains,_=trajectory_feedback(np.zeros((251,34)),np.zeros(250),first,np.zeros((34,34)),q,control_cost=1000.)
    static_max=float(np.max(np.linalg.norm(static_gains,axis=1)))
    np.savez_compressed(args.out/'static-finite250.npz',gains=static_gains,Q=q)
    for index,frequency in enumerate((2.5,5.,6.25)):
        directory=args.out/f'frequency-{frequency:g}';directory.mkdir(exist_ok=False)
        states,actions,harmonic=discrete_harmonic_initializer(design['A'],design['B'],frequency)
        initial=states.copy();row=dict(**harmonic,profile_index=index,closure_pass=False)
        newton={};analytic=np.full_like(states,np.nan);native=np.full_like(states,np.nan);gains=np.empty((0,34));offsets=np.empty(0)
        try:
            states,analytic,newton=refine_cyclic(transition,first,states,actions,maximum_iterations=8,maximum_halvings=6,tolerance=1e-10)
            for tick,u in enumerate(actions):
                env.reset(states[tick]);following,_=env.step(u);native_maps+=1
                native[tick]=states[(tick+1)%len(actions)]-following
            angles=np.cumsum(states[:,1:17],axis=1)
            analytic_max=float(np.max(np.abs(analytic)));native_max=float(np.max(np.abs(native)))
            spatial_bend=float(np.max(np.ptp(angles,axis=1)))
            bounds=bool(np.max(np.abs(angles))<.12 and np.max(np.abs(states[:,0]))<=2.5 and np.max(np.abs(actions))<=.5)
            closed=bool(newton['all_original_rows_closed'] and analytic_max<=1e-10 and native_max<=1e-10 and bounds)
            row.update(closure_pass=closed,maximum_analytic_held_map_residual=analytic_max,maximum_native_held_map_residual=native_max,
                       dropped_original_cart_row_analytic=float(analytic[-1,0]),dropped_original_cart_row_native=float(native[-1,0]),
                       gauge_mean_cart=float(np.mean(states[:,0])),max_absolute_angle=float(np.max(np.abs(angles))),
                       max_spatial_bend_rad=spatial_bend,max_cart_m=float(np.max(np.abs(states[:,0]))),planning_bounds_ok=bounds,newton=newton)
            if closed:
                tracked_s=states[np.arange(251)%len(actions)];tracked_u=actions[np.arange(250)%len(actions)]
                defects=-native[np.arange(250)%len(actions)]
                gains,offsets=trajectory_feedback(tracked_s,tracked_u,first,np.zeros((34,34)),q,control_cost=1000.,defects=defects)
                row.update(maximum_gain_norm=float(np.max(np.linalg.norm(gains,axis=1))),gain_ratio_to_static=float(np.max(np.linalg.norm(gains,axis=1))/static_max),maximum_affine_offset=float(np.max(np.abs(offsets))))
                candidates.append(dict(index=index,states=tracked_s,actions=tracked_u,gains=gains,offsets=offsets,row=row))
        except Exception as exc:
            row['exception']=str(exc);row['exception_type']=type(exc).__name__
        np.savez_compressed(directory/'orbit.npz',states=np.r_[states,states[:1]],actions=actions,
                            harmonic_initializer=initial,analytic_residuals=analytic,native_residuals=native,gains=gains,offsets=offsets)
        row.update(artifact_path=str(directory/'orbit.npz'),artifact_sha256=sha(directory/'orbit.npz'))
        (directory/'receipt.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');profiles.append(row)
        print(json.dumps(dict(event='profile_closed',**row),allow_nan=False),flush=True)
    cases=[];selected=None
    if candidates:
        selected=min(candidates,key=lambda v:(v['row']['maximum_gain_norm'],v['index']))
        # Selection is registered gain magnitude only, never a stability claim.
        for case_index,(direction_seed,epsilon) in enumerate(((3167000,1e-15),(3167001,1e-15),(3167000,1e-11),(3167001,1e-11))):
            direction=np.random.default_rng(direction_seed).uniform(-1.,1.,34);direction/=np.max(np.abs(direction))
            initial=selected['states'][0]+epsilon*direction
            def controller(state,tick):
                if tick>=250:return 0.
                return selected['actions'][tick]-selected['offsets'][tick]-float(selected['gains'][tick]@(state-selected['states'][tick]))
            states,actions,raw,score=record_episode(controller,initial,CanonicalProtocol(n_links=16))
            path=args.out/f'local-{3167310+case_index}.npz';np.savez_compressed(path,states=states,actions=actions,raw_actions=raw,initial=initial,direction=direction)
            tracking_hold=bool(len(actions)>=250 and np.all(np.abs(absolute_angles(states[1:251],16))<.15))
            row=dict(diagnostic_label=3167310+case_index,direction_seed=direction_seed,epsilon=epsilon,**score,
                     all_first250_tracking_goals=tracking_hold,post_tracking_force='literal_zero',recording_path=str(path),recording_sha256=sha(path))
            cases.append(row);print(json.dumps(dict(event='local_case',**row),allow_nan=False),flush=True)
    receipt=dict(schema='rage_mill.canonical16_fast_bent_orbit_discriminator.v1',diagnostic_only=True,
                 controller_qualified=False,development_episodes=0,reserved_episodes=0,n_links=16,
                 registered_frequencies_Hz=[2.5,5.,6.25],registered_angle_amplitude=.08,registered_force_amplitude_cap=.5,registered_cart_amplitude_cap_m=.75,
                 orbit_gauge='replace last cart-closure row by mean cart0; ALL original rows independently checked',
                 Newton_iterations_max=8,Newton_halvings_max=6,profiles=profiles,static_finite250_maximum_gain_norm=static_max,
                 selection_rule='minimum maximum 250step zero-terminal QR gain norm among all-row analytic/native closed candidates; index tie-break',
                 selected_profile_index=None if selected is None else selected['index'],cases=cases,
                 all4_native_task_pass=bool(len(cases)==4 and all(v['success'] for v in cases)),
                 any_first250_tracking_hold=bool(any(v['all_first250_tracking_goals'] for v in cases)),
                 native_orbit_map_commands=native_maps,native_local_commands=sum(v['ticks'] for v in cases),
                 runtime=dict(mujoco=mujoco.__version__,numpy=np.__version__,casadi=ca.__version__),
                 model_receipt_sha256=sha(args.model/'receipt.json'),design_sha256=sha(args.design),wall_seconds=time.monotonic()-start)
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(event='component_closed',all4_native_task_pass=receipt['all4_native_task_pass'],any_first250_tracking_hold=receipt['any_first250_tracking_hold'],wall_seconds=receipt['wall_seconds'])),flush=True)


if __name__=='__main__':main()
