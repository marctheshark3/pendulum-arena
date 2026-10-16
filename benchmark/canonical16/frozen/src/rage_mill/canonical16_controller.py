"""Portable binary64 full-state policy for canonical16 diagnostics.

This policy makes no admission decisions. Parking and route phase boundaries
change only feedback coordinates; they never modify the native physical state.
"""
from __future__ import annotations
import numpy as np


class Canonical16TrajectoryPolicy:
    def __init__(self, *, states, actions, gains, offsets, balance_gain,
                 park_gain, parking_target, park_ticks=0, align_launch_branches=False,
                 dot_scope='float64'):
        self.states=np.asarray(states,dtype=np.float64)
        self.actions=np.asarray(actions,dtype=np.float64)
        self.gains=np.asarray(gains,dtype=np.float64)
        self.offsets=np.asarray(offsets,dtype=np.float64)
        self.balance_gain=np.asarray(balance_gain,dtype=np.float64)
        self.park_gain=np.asarray(park_gain,dtype=np.float64)
        self.parking_target=np.asarray(parking_target,dtype=np.float64)
        count=len(self.actions)
        if (self.actions.shape!=(count,) or self.states.shape!=(count+1,34) or self.gains.shape!=(count,34)
                or self.offsets.shape!=(count,) or self.balance_gain.shape!=(34,)
                or self.park_gain.shape!=(34,) or self.parking_target.shape!=(34,)
                or type(park_ticks) is not int or park_ticks<0 or park_ticks+count>1500
                or dot_scope not in ('float64','longdouble')
                or not all(np.isfinite(v).all() for v in (self.states,self.actions,self.gains,
                    self.offsets,self.balance_gain,self.park_gain,self.parking_target))):
            raise ValueError('finite canonical16 arrays and complete-episode phase timing required')
        self.park_ticks=park_ticks
        self.align_launch_branches=bool(align_launch_branches)
        self.dot_scope=dot_scope
        self.launch_cart_offset=None
        self.launch_branch_offset=None
        self.next_tick=0

    def _dot(self,a,b):
        if self.dot_scope=='float64':return float(np.dot(a,b))
        return float(np.sum(np.asarray(a,dtype=np.longdouble)*np.asarray(b,dtype=np.longdouble),dtype=np.longdouble))

    def __call__(self,state,tick):
        state=np.asarray(state,dtype=np.float64)
        if (state.shape!=(34,) or not np.isfinite(state).all() or type(tick) is not int
                or tick!=self.next_tick or not 0<=tick<1500):
            raise ValueError('one finite physical state for each consecutive episode tick required')
        self.next_tick+=1
        if tick<self.park_ticks:
            error=state-self.parking_target
            if self.align_launch_branches:error[1:17]=(error[1:17]+np.pi)%(2*np.pi)-np.pi
            return -self._dot(self.park_gain,error)
        if self.launch_cart_offset is None:
            self.launch_cart_offset=state[0]-self.states[0,0]
            self.launch_branch_offset=(2*np.pi*np.rint((state[1:17]-self.states[0,1:17])/(2*np.pi))
                                       if self.align_launch_branches else np.zeros(16))
        centered=state.copy()
        centered[0]-=self.launch_cart_offset
        centered[1:17]-=self.launch_branch_offset
        route_tick=tick-self.park_ticks
        if route_tick<len(self.actions):
            return self.actions[route_tick]-self.offsets[route_tick]-self._dot(self.gains[route_tick],centered-self.states[route_tick])
        return -self._dot(self.balance_gain,centered)
