"""Calibrate canonical16 planning mechanics on a frozen successful14 route lift."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import casadi as ca
import mujoco
import numpy as np
from rage_mill.canonical_benchmark import CanonicalEnv, CanonicalProtocol
from rage_mill.canonical_dynamics import planning_functions
from rage_mill.canonical16_trajectory import DONOR14_SHA256, lift_material


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--donor', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.donor = args.donor.resolve(); args.out = args.out.resolve()
    assert hashlib.sha256(args.donor.read_bytes()).hexdigest() == DONOR14_SHA256
    assert mujoco.__version__ == '3.3.5'
    source = json.loads(args.donor.read_text())
    assert source['controller']['final_trajectory_diagnostics']['is_exactly_feasible']
    states14 = np.asarray(source['search']['nominal_coordinate_states'], dtype=np.float64)
    actions14 = np.asarray(source['controller']['controls'], dtype=np.float64)
    assert states14.shape == (451,30) and actions14.shape == (450,)
    assert np.array_equal(actions14, actions14.astype(np.float32).astype(float))
    mirror = -1 if states14[0,1] < 0 else 1
    states = lift_material(mirror*states14, 16); actions=mirror*actions14
    assert states[0,1] == np.pi and np.count_nonzero(states[0]) == 1
    args.out.mkdir(parents=True, exist_ok=False); os.chdir(args.out)
    start=time.monotonic()
    env = CanonicalEnv(CanonicalProtocol(n_links=16))
    continuous, transition, first = planning_functions(env.model, jit=True)
    build=time.monotonic()-start
    print(json.dumps(dict(event='built_model',build_seconds=build)),flush=True)
    rows=[]
    for tick in (0,10,100,200,250,300,350,400,449):
        x,u=states[tick],actions[tick]
        env.reset(x);env.data.ctrl[0]=80*u;mujoco.mj_forward(env.model,env.data)
        native_dx=np.r_[env.data.qvel,env.data.qacc]
        expected=np.asarray(continuous(x,u)).ravel()
        following,delivered=env.step(u); predicted=np.asarray(transition(x,delivered)).ravel()
        a,b=map(np.asarray,first(x,u));direction=np.random.default_rng(3167100+tick).normal(size=34);direction/=np.linalg.norm(direction)
        eps=1e-5
        finite=(np.asarray(transition(x+eps*direction,u)).ravel()-np.asarray(transition(x-eps*direction,u)).ravel())/(2*eps)
        row=dict(tick=tick,continuous_error=float(np.max(np.abs(native_dx-expected))),
                 native_transition_error=float(np.max(np.abs(following-predicted))),
                 directional_error=float(np.max(np.abs(a@direction-finite))),
                 lifted_warm_defect=float(np.max(np.abs(states[tick+1]-predicted))))
        assert all(np.isfinite(v) for v in row.values())
        assert row['continuous_error']<1e-8 and row['native_transition_error']<1e-10 and row['directional_error']<1e-5
        rows.append(row)
    before=time.monotonic()
    for _ in range(100): first(states[300],actions[300])
    first_seconds=(time.monotonic()-before)/100
    for name,fn in [('continuous',continuous),('transition',transition),('first',first)]:fn.save(str(args.out/f'{name}.casadi'))
    np.savez_compressed(args.out/'lifted14.npz',states=states,actions=actions)
    receipt=dict(schema='rage_mill.canonical16_planning_model.v1',passed=True,n_links=16,
                 donor_path=str(args.donor),donor_sha256=DONOR14_SHA256,source_route_role='offline_initializer_only',
                 native_commands=len(rows),checks=rows,build_seconds=build,first_seconds=first_seconds,
                 runtime=dict(mujoco=mujoco.__version__,casadi=ca.__version__,numpy=np.__version__),
                 diagnostic_ad_seed_namespace=3167100,controller_calls=0,development_episodes=0,reserved_episodes=0,
                 xml_sha256=env.xml_sha256,wall_seconds=time.monotonic()-start,
                 generated_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.suffix in ('.casadi','.npz')})
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(receipt,allow_nan=False),flush=True)

if __name__=='__main__':main()
