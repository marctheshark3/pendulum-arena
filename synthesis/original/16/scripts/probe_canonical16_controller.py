"""Bounded native canonical16 diagnostic; never development/reserved admission."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import casadi as ca
import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are
from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol, hanging_state, record_episode
from rage_mill.canonical16_trajectory import trajectory_feedback
from rage_mill.canonical16_controller import Canonical16TrajectoryPolicy


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('reference','model','design','out'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--seed',type=int,default=3167100)
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--exact',action='store_true')
    p.add_argument('--park-seconds',type=float,default=0.)
    p.add_argument('--tracking-cost',type=float,default=1000.)
    p.add_argument('--native-bias',action='store_true')
    p.add_argument('--parking-design',type=Path)
    p.add_argument('--expected-parking-sha256')
    p.add_argument('--parking-target-cart',type=float)
    p.add_argument('--align-launch-branches',action='store_true')
    p.add_argument('--dot',choices=('float64','longdouble'),default='float64')
    args=p.parse_args()
    if not 3167100<=args.seed<=args.seed+args.episodes-1<=3167199:
        raise ValueError('only predeclared diagnostic3167100..3167199 seeds permitted')
    if mujoco.__version__!='3.3.5':raise ValueError('MuJoCo3.3.5 required')
    with np.load(args.reference,allow_pickle=False) as s:states,actions=s['states'].copy(),s['actions'].copy()
    with np.load(args.design,allow_pickle=False) as s:design={k:s[k].copy() for k in s.files}
    protocol=CanonicalProtocol(n_links=16)
    if (states.shape!=(len(actions)+1,34) or len(actions)>1000 or not np.isfinite(states).all()
            or not np.isfinite(actions).all() or np.any(np.abs(actions)>1)
            or not 0<=args.park_seconds<=30-len(actions)*.02 or args.tracking_cost<=0):
        raise ValueError('valid16 reference and parking leaving full reference within30seconds required')
    metadata=json.loads((args.model/'receipt.json').read_text())
    if not metadata['passed'] or metadata['n_links']!=16:raise ValueError('verified16 planning model required')
    args.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    first=ca.Function.load(str(args.model/'first.casadi')); env=CanonicalEnv(protocol)
    defects=np.zeros((len(actions),34))
    if args.native_bias:
        for tick,u in enumerate(actions):
            env.reset(states[tick]);following,_=env.step(u);defects[tick]=following-states[tick+1]
    gains,offsets=trajectory_feedback(states,actions,first,design['terminal_root'],design['Q'],control_cost=args.tracking_cost,defects=defects)
    initial=hanging_state(protocol,args.seed,exact=True)
    parking_target=initial.copy()
    if args.parking_design is not None:
        if (not args.expected_parking_sha256 or sha(args.parking_design)!=args.expected_parking_sha256
                or args.parking_target_cart is None or not np.isfinite(args.parking_target_cart)):
            raise ValueError('external parking design requires exact expected SHA and explicit finite cart target')
        with np.load(args.parking_design,allow_pickle=False) as s:park_gain=s['gain'].copy()
        if park_gain.shape!=(34,) or not np.isfinite(park_gain).all():raise ValueError('finite34 parking gain required')
        parking_target[0]=args.parking_target_cart
    else:
        if args.expected_parking_sha256 or args.parking_target_cart is not None:
            raise ValueError('parking identity/target require an actual design')
        a,b=map(np.asarray,first(initial,0));q=np.eye(34)*1e-8;q[0,0]=10.;q[17,17]=5.
        park_p=solve_discrete_are(a,b,q,np.array([[1000.]]));park_gain=((b.T@park_p@a)/(1000+(b.T@park_p@b)[0,0])).ravel()
    park_ticks=round(args.park_seconds/.02)
    np.savez_compressed(args.out/'controller.npz',states=states,actions=actions,gains=gains,offsets=offsets,balance_gain=design['gain'],park_gain=park_gain,parking_target=parking_target,park_ticks=park_ticks,native_defects=defects)
    rows=[]
    for seed in range(args.seed,args.seed+args.episodes):
        controller=Canonical16TrajectoryPolicy(states=states,actions=actions,gains=gains,offsets=offsets,
                    balance_gain=design['gain'],park_gain=park_gain,parking_target=parking_target,
                    park_ticks=park_ticks,align_launch_branches=args.align_launch_branches,dot_scope=args.dot)
        start=time.monotonic()
        s,u,raw,score=record_episode(controller,hanging_state(protocol,seed,exact=args.exact),protocol)
        path=args.out/f'episode-{seed}.npz';np.savez_compressed(path,states=s,actions=u,raw_actions=raw)
        row=dict(seed=seed,exact=args.exact,**score,recording=path.name,recording_sha256=sha(path),wall_seconds=time.monotonic()-start)
        rows.append(row);print(json.dumps(row,allow_nan=False),flush=True)
    receipt=dict(schema='rage_mill.canonical16_controller_diagnostic.v1',n_links=16,diagnostic_only=True,
                 controller_qualified=False,development_episodes=0,reserved_episodes=0,cases=rows,
                 reference_sha256=sha(args.reference),design_sha256=sha(args.design),model_receipt_sha256=sha(args.model/'receipt.json'),
                 controller_sha256=sha(args.out/'controller.npz'),native_bias=args.native_bias,
                 maximum_native_defect=float(np.max(np.abs(defects))),maximum_affine_offset=float(np.max(np.abs(offsets))),
                 maximum_gain_norm=float(np.max(np.linalg.norm(gains,axis=1))),dot=args.dot,park_seconds=args.park_seconds,
                 tracking_cost=args.tracking_cost,parking_design_sha256=None if args.parking_design is None else sha(args.parking_design),
                 parking_target_cart=float(parking_target[0]),align_launch_branches=args.align_launch_branches,wall_seconds=time.monotonic()-started)
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
