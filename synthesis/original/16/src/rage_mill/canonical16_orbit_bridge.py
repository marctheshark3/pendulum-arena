"""Offline initializer and declared delivered-action NLP interface."""
from __future__ import annotations
import numpy as np


def bridge_initializer(route_states,route_actions,orbit_states,orbit_actions,initial,*,blend_start=300):
    route_states,route_actions,orbit_states,orbit_actions,initial=map(np.asarray,(route_states,route_actions,orbit_states,orbit_actions,initial))
    if (route_states.shape!=(451,34) or route_actions.shape!=(450,) or initial.shape!=(34,)
            or orbit_states.ndim!=2 or orbit_states.shape[1]!=34 or len(orbit_states)!=len(orbit_actions)+1
            or type(blend_start) is not int or not 0<=blend_start<450
            or not all(np.isfinite(v).all() for v in (route_states,route_actions,orbit_states,orbit_actions,initial))):
        raise ValueError('finite450-step route, cyclic16 orbit and actual phase initial required')
    count=len(orbit_actions)
    if orbit_states[-1].tobytes()!=orbit_states[0].tobytes():raise ValueError('literal cyclic orbit closure endpoint required')
    state=route_states.copy();action=route_actions.copy()
    state[:,0]+=initial[0]-state[0,0]
    phases=(np.arange(451)-450)%count
    orbit=orbit_states[phases]
    clock=np.clip((np.arange(451)-blend_start)/(450-blend_start),0.,1.)
    weight=6*clock**5-15*clock**4+10*clock**3
    derivative=(30*clock**4-60*clock**3+30*clock**2)/((450-blend_start)*.02)
    q=(1-weight[:,None])*state[:,:17]+weight[:,None]*orbit[:,:17]
    v=((1-weight[:,None])*state[:,17:]+weight[:,None]*orbit[:,17:]
       +derivative[:,None]*(orbit[:,:17]-state[:,:17]))
    result=np.c_[q,v];result[0]=initial;result[-1]=orbit_states[0]
    action=(1-weight[:-1])*action+weight[:-1]*orbit_actions[phases[:-1]]
    return result,action


def delivered_actions(actions):
    return np.clip(np.asarray(actions).astype(np.float32),-1.,1.).astype(np.float64)


def delivered_callback(count):
    """Values are realF32delivery; NLP inputJacB is a disclosed surrogate."""
    import casadi as ca
    class Float32Delivery(ca.Callback):
        def __init__(self):
            ca.Callback.__init__(self);self.construct('actual_float32_delivery',{})
        def get_n_in(self):return 1
        def get_n_out(self):return 1
        def get_sparsity_in(self,index):return ca.Sparsity.dense(1,count)
        def get_sparsity_out(self,index):return ca.Sparsity.dense(1,count)
        def eval(self,inputs):return [delivered_actions(np.asarray(inputs[0]))]
        def has_jacobian(self):return True
        def get_jacobian(self,name,inames,onames,opts):
            # This identity is the declared straight-through proposal derivative,
            # never a claim that quantization has an ordinary derivative.
            x=ca.MX.sym('input',1,count);y=ca.MX.sym('output',1,count)
            return ca.Function(name,[x,y],[ca.DM.eye(count)],inames,onames)
    return Float32Delivery()
