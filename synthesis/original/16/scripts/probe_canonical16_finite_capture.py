"""Test finite holding against the saved infinite-horizon capture baseline."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from scipy.linalg import qr
from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol, score_tape


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_gains(a,b,q,horizon):
    dimension=a.shape[0];root=np.zeros((dimension,dimension));qroot=np.linalg.cholesky(q).T
    gains=np.empty((horizon,dimension))
    for tick in reversed(range(horizon)):
        stack=np.block([[np.sqrt(1000.)*np.ones((1,1)),np.zeros((1,dimension))],
                        [root@b,root@a],[np.zeros((dimension,1)),qroot]])
        upper=qr(stack,mode='economic')[1]
        gains[tick]=upper[0,1:]/upper[0,0];root=upper[1:,1:]
    if not np.isfinite(gains).all():raise ValueError('nonfinite finite-horizon gains')
    return gains


def main():
    parser=argparse.ArgumentParser(__doc__,allow_abbrev=False)
    parser.add_argument('--design',required=True,type=Path)
    parser.add_argument('--expected-design-sha256',required=True)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    if args.out.exists()or mujoco.__version__!='3.3.5'or sha(args.design)!=args.expected_design_sha256:
        raise ValueError('fresh output and pinned MuJoCo3.3.5 design required')
    args.out.mkdir(parents=True);wall,cpu=time.monotonic(),time.process_time()
    with np.load(args.design,allow_pickle=False)as d:a,b,q,k=d['A'],d['B'],d['Q'],d['K']
    if a.shape!=(34,34)or b.shape!=(34,1)or q.shape!=(34,34)or k.shape!=(34,):
        raise ValueError('canonical16 capture arrays required')
    finite250=finite_gains(a,b,q,250);finite300=finite_gains(a,b,q,300);short=finite_gains(a,b,q,100)
    profiles=[('infinite_baseline_250',np.tile(k,(250,1))),('finite250',finite250),
              ('finite300',finite300),('receding100_for250',np.tile(short[0],(250,1)))]
    np.savez_compressed(args.out/'gains.npz',**{name:g for name,g in profiles})
    env=CanonicalEnv(CanonicalProtocol(n_links=16));results=[]
    for name,gains in profiles:
        rows=[]
        for seed in(3167000,3167001):
            direction=np.random.default_rng(seed).uniform(-1.,1.,34);direction/=np.max(np.abs(direction))
            for amplitude in(1e-15,1e-11):
                state=env.reset(direction*amplitude);states=[state];actions=[];raw=[];invalid=None
                for tick in range(1500):
                    command=-float(gains[tick]@state)if tick<len(gains)else 0.
                    try:state,delivered=env.step(command)
                    except ValueError as error:invalid=str(error);break
                    states.append(state);actions.append(delivered);raw.append(command)
                    if abs(state[0])>3.:invalid='rail escape';break
                path=args.out/f'{name}-{seed}-{amplitude:.0e}.npz'
                s,u=np.asarray(states),np.asarray(actions)
                np.savez_compressed(path,states=s,actions=u,raw_actions=np.asarray(raw))
                score=score_tape(s,u,env.protocol,invalid_reason=invalid)
                score['local_capture_success']=score.pop('success')
                rows.append(dict(diagnostic_seed=seed,initial_amplitude=amplitude,**score,
                    maximum_raw_command=float(np.max(np.abs(raw)))if raw else 0.,
                    recording=dict(path=str(path),sha256=sha(path))))
        result=dict(profile=name,controller_ticks=len(gains),after_horizon_action=0.,
            maximum_gain=float(np.max(np.abs(gains))),initial_gain_norm=float(np.linalg.norm(gains[0])),
            cases=rows,all_four_local_cases_pass=all(r['local_capture_success']for r in rows))
        results.append(result);print(json.dumps(result,allow_nan=False),flush=True)
    receipt=dict(schema='rage_mill.canonical16_finite_capture_component.v1',profiles=results,
        source_design=dict(path=str(args.design),sha256=sha(args.design)),
        scope='Local upright matched initial states, finite maintenance then zero force; no hanging-start qualification',
        CPU_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall,
        development_gate_episodes=0,reserved_episodes=0,controller_qualified=False)
    with(args.out/'receipt.json').open('x')as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    if sha(args.design)!=args.expected_design_sha256:raise ValueError('design changed')


if __name__=='__main__':main()
