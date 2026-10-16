"""Bounded delivered-F32 bridge to the reviewed native five-hertz hold orbit.

The NLP's input derivative is explicitly straight-through. This optimizer
surrogate is never evidence of physical feasibility: every assembled row is
independently evaluated by actual native float32 command delivery afterward.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import casadi as ca
import mujoco
import numpy as np
from rage_mill.canonical_benchmark import CanonicalEnv,CanonicalProtocol,hanging_state,absolute_angles
from rage_mill.canonical16_orbit_bridge import bridge_initializer,delivered_actions,delivered_callback
from rage_mill.canonical16_trajectory import trajectory_feedback


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def closed_owner(directory):
    value=json.loads((directory/'completion.json').read_text())
    if (not value['naturally_closed_zero'] or value['failure'] is not None or value['wall_timeout']
            or value['interrupted'] or not all(value['source_hashes_unchanged'].values())):
        raise ValueError('closed unmodified owned source required')
    return value


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('model','design','warm','orbit-owner','launch-owner','out'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();started=time.monotonic()
    closed_owner(args.orbit_owner);closed_owner(args.launch_owner)
    orbit_receipt=json.loads((args.orbit_owner/'probe/receipt.json').read_text())
    launch_receipt=json.loads((args.launch_owner/'probe/receipt.json').read_text())
    selected=orbit_receipt['profiles'][orbit_receipt['selected_profile_index']]
    if (orbit_receipt['schema']!='rage_mill.canonical16_fast_bent_orbit_discriminator.v1'
            or selected['frequency_Hz']!=5. or not selected['closure_pass']
            or launch_receipt['n_links']!=16 or launch_receipt['park_seconds']!=16.
            or len(launch_receipt['cases'])!=1 or not launch_receipt['cases'][0]['exact']
            or not orbit_receipt['cases'][0]['all_first250_tracking_goals']
            or not orbit_receipt['cases'][1]['all_first250_tracking_goals']):
        raise ValueError('reviewed physical five-second5Hz target and exact16s native launch required')
    orbit_path=Path(selected['artifact_path'])
    launch_record=args.launch_owner/'probe'/launch_receipt['cases'][0]['recording']
    if sha(orbit_path)!=selected['artifact_sha256'] or sha(launch_record)!=launch_receipt['cases'][0]['recording_sha256']:
        raise ValueError('orbit or native launch bytes differ')
    with np.load(orbit_path,allow_pickle=False) as s:orbit_s=s['states'].copy();orbit_u=s['actions'].copy()
    if orbit_s.shape!=(101,34) or orbit_u.shape!=(100,) or orbit_s[-1].tobytes()!=orbit_s[0].tobytes():
        raise ValueError('exact retained cyclic5Hz arrays required')
    for case in orbit_receipt['cases'][:2]:
        tape=Path(case['recording_path'])
        if sha(tape)!=case['recording_sha256']:raise ValueError('physical orbit support tape differs')
        with np.load(tape,allow_pickle=False) as s:
            if not np.all(np.abs(absolute_angles(s['states'][1:251],16))<.15):raise ValueError('actual five-second support missing')
    with np.load(launch_record,allow_pickle=False) as s:
        initial=s['states'][800].copy()
        if s['states'][0].tobytes()!=hanging_state(CanonicalProtocol(n_links=16),0,exact=True).tobytes():
            raise ValueError('launch owner must start at actual canonical hanging reset')
    with np.load(args.warm,allow_pickle=False) as s:warm_s=s['states'][:451].copy();warm_u=s['actions'][:450].copy()
    with np.load(args.design,allow_pickle=False) as s:Q=s['Q'].copy()
    model_meta=json.loads((args.model/'receipt.json').read_text())
    if mujoco.__version__!='3.3.5' or not model_meta['passed'] or model_meta['n_links']!=16:
        raise ValueError('verified16 model and native3.3.5 runtime required')
    env=CanonicalEnv(CanonicalProtocol(n_links=16))
    if env.xml_sha256!=model_meta['xml_sha256']:raise ValueError('actual native model identity differs')
    warm_s,warm_u=bridge_initializer(warm_s,warm_u,orbit_s,orbit_u,initial,blend_start=150)
    args.out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.out/'initializer.npz',states=warm_s,actions=warm_u,actual_phase_initial=initial,orbit_phase0=orbit_s[0])
    transition=ca.Function.load(str(args.model/'transition.casadi'));first=ca.Function.load(str(args.model/'first.casadi'))
    count,dimension=450,34
    states=ca.MX.sym('bridge_states',dimension,count+1);raw_actions=ca.MX.sym('bridge_raw_actions',1,count)
    deliver=delivered_callback(count);actions=deliver(raw_actions)
    z=ca.vertcat(ca.vec(states),ca.vec(raw_actions))
    gaps=states[:,1:]-transition.map(count)(states[:,:-1],actions)
    boundary=ca.vertcat(states[:,0]-initial,states[:,-1]-orbit_s[0])
    g=ca.vertcat(ca.vec(gaps),boundary)
    aa,bb=first.map(count)(states[:,:-1],actions)
    block_a=ca.diagcat(*[aa[:,tick*dimension:(tick+1)*dimension] for tick in range(count)])
    block_b=ca.diagcat(*[bb[:,tick] for tick in range(count)])
    state_part=(ca.horzcat(-block_a,ca.MX(ca.Sparsity(dimension*count,dimension)))
                +ca.horzcat(ca.MX(ca.Sparsity(dimension*count,dimension)),ca.DM.eye(dimension*count)))
    jac=ca.vertcat(ca.horzcat(state_part,-block_b),ca.jacobian(boundary,z))
    jac_g=ca.Function('orbit_bridge_surrogate_jac_g',[z,ca.MX.sym('p',0)],[g,jac])
    objective=1e-4*ca.sumsqr(raw_actions-ca.DM(warm_u.reshape(1,-1)))+1e-7*ca.sumsqr(states-ca.DM(warm_s.T))
    options={'jac_g':jac_g,'ipopt.hessian_approximation':'limited-memory','ipopt.max_iter':150,'ipopt.max_cpu_time':180.,
             'ipopt.tol':1e-9,'ipopt.constr_viol_tol':1e-9,'ipopt.acceptable_iter':0,'ipopt.bound_relax_factor':0.,
             'ipopt.nlp_scaling_method':'none','ipopt.mu_strategy':'adaptive','ipopt.linear_solver':'mumps','ipopt.print_level':5,'print_time':False}
    solver=ca.nlpsol('canonical16_orbit_bridge','ipopt',{'x':z,'f':objective,'g':g},options)
    lower_s=np.full((451,34),-80.);upper_s=-lower_s;lower_s[:,0]=-2.8;upper_s[:,0]=2.8
    lower=np.r_[lower_s.ravel(),np.full(450,-.95)];upper=np.r_[upper_s.ravel(),np.full(450,.95)]
    build=time.monotonic()-started;print(json.dumps(dict(event='start_orbit_bridge',bridge_ticks=450,tail_ticks=250,build_seconds=build,optimizer_CPU_limit=180,iterations=150,input_jacobian='disclosed straight-through B at deliveredF32actions')),flush=True)
    solved=solver(x0=np.r_[warm_s.ravel(),warm_u],lbx=lower,ubx=upper,lbg=np.zeros(34*450+68),ubg=np.zeros(34*450+68))
    vector=np.asarray(solved['x']).ravel();bridge_s=vector[:34*451].reshape(451,34);raw_u=vector[34*451:];bridge_u=delivered_actions(raw_u)
    raw_initial_error=float(np.max(np.abs(bridge_s[0]-initial)));raw_terminal_error=float(np.max(np.abs(bridge_s[-1]-orbit_s[0])))
    raw_s=bridge_s.copy();bridge_s[0]=initial;bridge_s[-1]=orbit_s[0]
    tail_indices=np.arange(251)%len(orbit_u)
    full_s=np.r_[bridge_s,orbit_s[tail_indices[1:]]];full_u=np.r_[bridge_u,orbit_u[np.arange(250)%len(orbit_u)]]
    cold_gaps=np.empty((700,34));analytic_gaps=full_s[1:]-np.asarray(transition.map(700)(full_s[:-1].T,full_u.reshape(1,-1))).T
    for tick,u in enumerate(full_u):
        env.reset(full_s[tick]);following,_=env.step(u);cold_gaps[tick]=following-full_s[tick+1]
    coldmax=float(np.max(np.abs(cold_gaps)));analyticmax=float(np.max(np.abs(analytic_gaps)))
    boxes=bool(np.max(np.abs(full_s[:,0]))<=2.8 and np.max(np.abs(full_s))<=80 and np.max(np.abs(full_u))<=.95)
    tail_goals=bool(np.all(np.abs(absolute_angles(full_s[451:],16))<.15))
    literal_start=bool(full_s[0].tobytes()==initial.tobytes());literal_seam=bool(full_s[450].tobytes()==orbit_s[0].tobytes())
    tail_bytes=bool(full_s[451:].tobytes()==orbit_s[tail_indices[1:]].tobytes() and full_u[450:].tobytes()==orbit_u[np.arange(250)%100].tobytes())
    f32_bytes=bool(full_u.tobytes()==delivered_actions(full_u).tobytes())
    feasible=bool(np.isfinite(full_s).all() and np.isfinite(full_u).all() and coldmax<=1e-8 and boxes and tail_goals and literal_start and literal_seam and tail_bytes and f32_bytes)
    gains,offsets=trajectory_feedback(full_s,full_u,first,np.zeros((34,34)),Q,control_cost=1000.,defects=cold_gaps)
    np.savez_compressed(args.out/'reference.npz',states=full_s,actions=full_u,gains=gains,offsets=offsets,native_cold_gaps=cold_gaps,analytic_gaps=analytic_gaps)
    np.savez_compressed(args.out/'raw-solver.npz',states=raw_s,actions=raw_u)
    stats=solver.stats();receipt=dict(schema='rage_mill.canonical16_F32_early_orbit_bridge_repair.v2',reference_only=True,controller_qualified=False,
        development_episodes=0,reserved_episodes=0,actual_episode_calls=0,native_cold_map_commands=700,n_links=16,
        blend_start_tick=150,blend_end_tick=450,bridge_ticks=450,orbit_tail_ticks=250,full_reference_ticks=700,parking_ticks=800,full_episode_ticks=1500,
        phase_initial_source='owned actual16s postpark state in unchanged absolute native world frame',
        input_values='actual clipped normalized float32, converted to float64 for native/model calls',
        input_derivative='explicit straight-through identity/B surrogate for optimizer proposal only; not true derivative of quantization',
        native_reference_feasible=feasible,maximum_native_cold_gap=coldmax,maximum_analytic_delivered_gap=analyticmax,
        raw_solver_initial_error=raw_initial_error,raw_solver_orbit_endpoint_error=raw_terminal_error,
        offline_literal_boundary_substitution=True,all_cold_rows_recomputed_after_boundary_substitution=True,
        literal_actual_phase_initial=literal_start,literal_orbit_phase0_seam=literal_seam,exact_retained250tail=tail_bytes,
        final_orbit_saved_index=int(tail_indices[-1]),float32_action_bytes=f32_bytes,planning_boxes_ok=boxes,all250_nominal_tail_goals=tail_goals,
        maximum_gain_norm=float(np.max(np.linalg.norm(gains,axis=1))),maximum_bridge_gain_norm=float(np.max(np.linalg.norm(gains[:450],axis=1))),
        maximum_critical150_300_gain_norm=float(np.max(np.linalg.norm(gains[150:300],axis=1))),maximum_tail_gain_norm=float(np.max(np.linalg.norm(gains[450:],axis=1))),
        maximum_affine_offset=float(np.max(np.abs(offsets))),gain_terminal_policy='zero over complete700-step reference',
        status=stats['return_status'],iterations=stats['iter_count'],model_receipt_sha256=sha(args.model/'receipt.json'),design_sha256=sha(args.design),
        warm_sha256=sha(args.warm),orbit_owner_path=str(args.orbit_owner),orbit_owner_completion_sha256=sha(args.orbit_owner/'completion.json'),orbit_artifact_sha256=sha(orbit_path),
        launch_owner_path=str(args.launch_owner),launch_owner_completion_sha256=sha(args.launch_owner/'completion.json'),launch_recording_sha256=sha(launch_record),
        reference_sha256=sha(args.out/'reference.npz'),build_seconds=build,wall_seconds=time.monotonic()-started)
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(receipt,allow_nan=False),flush=True)


if __name__=='__main__':main()
