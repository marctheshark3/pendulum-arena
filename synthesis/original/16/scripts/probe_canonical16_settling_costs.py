"""Matched angular-cost bracket on ten saved hanging diagnostic initial states."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument('--saved', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() or mujoco.__version__ != '3.3.5':
        raise ValueError('fresh output and MuJoCo3.3.5 required')
    inputs = json.loads((args.saved/'receipt.json').read_text())
    if len(inputs['cases']) != 10 or [r['seed'] for r in inputs['cases']] != list(range(3167200,3167210)):
        raise ValueError('exact ten-case saved diagnostic input required')
    pins = {str(args.saved/'receipt.json'):sha(args.saved/'receipt.json'),
            str(args.saved/'design.npz'):inputs['design']['sha256'], str(Path(__file__)):sha(__file__)}
    initial = []
    for case in inputs['cases']:
        item = case['recording']; pins[item['path']] = item['sha256']
        if sha(item['path']) != item['sha256']:
            raise ValueError('saved initial-state recording changed')
        with np.load(item['path'], allow_pickle=False) as tape:
            initial.append(tape['states'][0].copy())
    if sha(args.saved/'design.npz') != inputs['design']['sha256']:
        raise ValueError('native hanging derivative changed')
    with np.load(args.saved/'design.npz', allow_pickle=False) as data:
        a,b=data['A'],data['B']
    args.out.mkdir(parents=True)
    wall,cpu=time.monotonic(),time.process_time()
    env=CanonicalEnv(CanonicalProtocol(n_links=16))
    target=np.zeros(34);target[0]=-.05;target[1]=np.pi
    total=np.tril(np.ones((16,16)))
    profiles=[('incumbent',0.,0.,1000.),('angles10',10.,1.,1000.),
              ('angles100',100.,1.,1000.),('angles100_R100',100.,1.,100.),('angles100_R10',100.,1.,10.)]
    results=[]
    for name,angle,rate,r in profiles:
        q=np.eye(34)*1e-8;q[0,0]=10.;q[17,17]=5.
        q[1:17,1:17]+=angle*(total.T@total);q[18:,18:]+=rate*(total.T@total)
        p=solve_discrete_are(a,b,q,np.array([[r]]));gain=((b.T@p@a)/(r+(b.T@p@b)[0,0])).ravel()
        np.savez_compressed(args.out/(name+'-design.npz'),A=a,B=b,Q=q,P=p,gain=gain,R=r)
        rows=[]
        for offset,saved in enumerate(initial):
            state=env.reset(saved);states=[state];actions=[];raw=[];invalid=None
            for tick in range(800):
                command=-float(gain@(state-target))
                try: state,delivered=env.step(command)
                except ValueError as error: invalid=str(error);break
                states.append(state);actions.append(delivered);raw.append(command)
                if abs(state[0])>3.: invalid='rail escape';break
            error=state-target
            record=args.out/(name+f'-31672{offset:02d}.npz')
            np.savez_compressed(record,states=np.asarray(states),actions=np.asarray(actions,dtype=np.float32),raw_actions=np.asarray(raw))
            rows.append(dict(saved_seed=3167200+offset,ticks=len(actions),valid=invalid is None,
                invalid_reason=invalid,final_error_norm=float(np.linalg.norm(error)),
                maximum_final_absolute_angle_error=float(np.max(np.abs(total@error[1:17]))),
                maximum_cart_m=float(np.max(np.abs(np.asarray(states)[:,0]))),
                maximum_raw_command=float(np.max(np.abs(raw))) if raw else 0.,
                recording=dict(path=str(record),sha256=sha(record))))
        result=dict(profile=name,angle_cost=angle,rate_cost=rate,control_cost=r,
                    gain_norm=float(np.linalg.norm(gain)),cases=rows,
                    all_valid=all(row['valid'] for row in rows),
                    median_final_error_norm=float(np.median([row['final_error_norm'] for row in rows])),
                    worst_final_absolute_angle_error=max(row['maximum_final_absolute_angle_error'] for row in rows))
        results.append(result);print(json.dumps(result,allow_nan=False),flush=True)
    baseline=results[0]['median_final_error_norm']
    for result in results:
        result['median_error_ratio']=result['median_final_error_norm']/baseline
        result['passes_prospective_settling_screen']=bool(result['all_valid'] and result['median_error_ratio']<=.25
            and result['worst_final_absolute_angle_error']<=.01)
    eligible=[r for r in results if r['passes_prospective_settling_screen']]
    selected=min(eligible,key=lambda r:r['median_final_error_norm'])['profile'] if eligible else 'incumbent'
    unchanged={p:sha(p)==digest for p,digest in pins.items()}
    receipt=dict(schema='rage_mill.canonical16_saved_settling_cost_component.v1',profiles=results,
        selected_profile=selected,source_hashes=pins,source_hashes_unchanged=unchanged,
        selection_rule='all10valid inrail, median16s error<=.25incumbent, worstcumulativeangleerror<=.01; chooseminimalmedian amongpassing elsekeepincumbent',
        CPU_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall,
        fresh_random_draws=0,development_gate_episodes=0,reserved_episodes=0,controller_qualified=False)
    with (args.out/'receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    if not all(unchanged.values()):raise ValueError('saved inputs changed')
    print(json.dumps(dict(selected=selected,receipt=str(args.out/'receipt.json'))))


if __name__=='__main__':main()
