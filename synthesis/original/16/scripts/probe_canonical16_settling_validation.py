"""Fresh diagnostic check of the selected hanging controller; no swing-up gate."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol, hanging_state


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__,allow_abbrev=False)
    parser.add_argument('--design',required=True,type=Path)
    parser.add_argument('--expected-design-sha256',required=True)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    if (args.out.exists() or mujoco.__version__!='3.3.5'
            or sha(args.design)!=args.expected_design_sha256):
        raise ValueError('fresh output, MuJoCo3.3.5 and selected frozen design required')
    with np.load(args.design,allow_pickle=False)as design:gain=design['gain'].copy()
    if gain.shape!=(34,)or not np.isfinite(gain).all():raise ValueError('canonical16gain required')
    args.out.mkdir(parents=True);wall,cpu=time.monotonic(),time.process_time()
    protocol=CanonicalProtocol(n_links=16);env=CanonicalEnv(protocol)
    target=np.zeros(34);target[0]=-.05;target[1]=np.pi
    rows=[]
    for seed in range(3167210,3167230):
        state=env.reset(hanging_state(protocol,seed));states=[state];actions=[];raw=[];metrics=[];invalid=None
        for tick in range(1000):
            command=-float(gain@(state-target))
            try:state,delivered=env.step(command)
            except ValueError as error:invalid=str(error);break
            states.append(state);actions.append(delivered);raw.append(command)
            if tick+1 in(600,700,800,1000):
                error=state-target
                metrics.append(dict(seconds=(tick+1)*.02,error_norm=float(np.linalg.norm(error)),
                    maximum_absolute_angle_error=float(np.max(np.abs(np.cumsum(error[1:17])))),
                    maximum_absolute_rate=float(np.max(np.abs(np.cumsum(state[18:]))))))
            if abs(state[0])>3.:invalid='rail escape';break
        path=args.out/f'parking-{seed}.npz'
        np.savez_compressed(path,states=np.asarray(states),actions=np.asarray(actions,dtype=np.float32),raw_actions=np.asarray(raw))
        rows.append(dict(seed=seed,ticks=len(actions),valid=invalid is None,invalid_reason=invalid,
            measurements=metrics,maximum_cart_m=float(np.max(np.abs(np.asarray(states)[:,0]))),
            recording=dict(path=str(path),sha256=sha(path))))
    stages=[]
    for seconds in(12.,14.,16.,20.):
        sample=[m for row in rows for m in row['measurements']if m['seconds']==seconds]
        stages.append(dict(seconds=seconds,cases=len(sample),
            median_error_norm=float(np.median([m['error_norm']for m in sample])) if sample else None,
            worst_error_norm=max((m['error_norm']for m in sample),default=None),
            worst_absolute_angle_error=max((m['maximum_absolute_angle_error']for m in sample),default=None)))
    passed=all(r['valid']and r['ticks']==1000 for r in rows)and stages[2]['worst_error_norm']<=1e-5 and stages[2]['worst_absolute_angle_error']<=1e-6
    receipt=dict(schema='rage_mill.canonical16_fresh_settling_component.v1',passed=passed,
        acceptance='all20 complete20s inrail, worst16sstateerror<=1e-5 and cumulativeangleerror<=1e-6',
        cases=rows,stages=stages,target=target.tolist(),design=dict(path=str(args.design),sha256=sha(args.design)),
        CPU_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall,
        stage='hanging_settling_component_only',development_gate_episodes=0,reserved_episodes=0,controller_qualified=False)
    with(args.out/'receipt.json').open('x')as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=passed,stages=stages,receipt=str(args.out/'receipt.json'))),flush=True)
    if sha(args.design)!=args.expected_design_sha256:raise ValueError('design changed')


if __name__=='__main__':main()
