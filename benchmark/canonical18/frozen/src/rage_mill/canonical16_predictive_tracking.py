"""Bounded native one-step tracking with complete affine value feedback.

The only environment reference held here is an explicitly supplied forecast
clone. Physical world coordinates are preserved in every forecast; cart
translation applies only to error coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .canonical16_affine_feedback import AffineFeedback


def _delivered(value):
    return float(np.clip(np.float32(value),-1.,1.))


def _key(value):
    return np.float32(value).tobytes()


def _dot(row,state):
    return math.fsum(float(a)*float(b)for a,b in zip(row,state,strict=True))


def _ulp(value):
    value=np.float32(value)
    return max(abs(float(np.nextafter(value,np.float32(math.inf)))-float(value)),
               abs(float(value)-float(np.nextafter(value,np.float32(-math.inf)))))


@dataclass(frozen=True)
class TrackingDecision:
    action: float
    predicted_state: np.ndarray
    selected_index: int
    baseline_index: int
    unique_forecasts: int
    duplicate_proposals: int


class NativeValueTracker:
    def __init__(self,states,actions,feedback:AffineFeedback,forecast,*,control_cost=1000.,
                 cart_offset=0.,before_forecast:Callable|None=None):
        states,actions=np.asarray(states,dtype=np.float64),np.asarray(actions,dtype=np.float64)
        if (states.ndim!=2 or actions.ndim!=1 or len(states)!=len(actions)+1
                or len(actions)<1 or not np.isfinite(states).all()or not np.isfinite(actions).all()
                or not np.all(np.abs(actions)<=1)
                or not np.array_equal(actions,actions.astype(np.float32).astype(np.float64))):
            raise ValueError('finite reference with delivered float32 commands required')
        count,dimension=len(actions),states.shape[1]
        if (feedback.gains.shape!=(count,dimension)or feedback.offsets.shape!=(count,)
                or feedback.roots.shape!=(count+1,dimension,dimension)
                or feedback.residuals.shape!=(count+1,dimension)
                or not all(np.isfinite(value).all()for value in
                           (feedback.gains,feedback.offsets,feedback.roots,feedback.residuals))
                or isinstance(control_cost,bool)or not np.isfinite(control_cost)or control_cost<=0
                or not np.isfinite(cart_offset)):
            raise ValueError('finite complete affine feedback and positive control cost required')
        self.states,self.actions,self.feedback=states,actions,feedback
        self.forecast,self.control_cost=forecast,float(control_cost)
        self.cart_offset,self.before_forecast=float(cart_offset),before_forecast
        self.ledger=[]

    def select(self,state,tick):
        state=np.asarray(state,dtype=np.float64)
        if (state.shape!=self.states.shape[1:]or not np.isfinite(state).all()
                or type(tick)is not int or not 0<=tick<len(self.actions)):
            raise ValueError('finite physical world state and reference tick required')
        error=state-self.states[tick]
        error[0]-=self.cart_offset
        reference=float(self.actions[tick])
        baseline=_delivered(reference-float(self.feedback.gains[tick]@error)-float(self.feedback.offsets[tick]))
        seen=set()
        samples=[]
        duplicates=0

        def evaluate(proposal):
            nonlocal duplicates
            delivered=_delivered(proposal)
            fingerprint=_key(delivered)
            if fingerprint in seen:
                duplicates+=1
                return None
            if len(samples)>=16:
                return None
            if self.before_forecast is not None:
                self.before_forecast()
            seen.add(fingerprint)
            record=dict(index=len(self.ledger),tick=tick,command=delivered,cost=None,error=None)
            self.ledger.append(record)
            samples.append(record)
            try:
                self.forecast.reset(state) # Full physical world state, forecast clone only.
                following,actual_command=self.forecast.step(delivered)
                record['state']=following.copy()
                if _key(actual_command)!=fingerprint:
                    raise ValueError('forecast delivery mismatch')
                next_error=following-self.states[tick+1]
                next_error[0]-=self.cart_offset
                value_residual=np.array([_dot(row,next_error)+float(affine)
                                         for row,affine in zip(self.feedback.roots[tick+1],
                                                               self.feedback.residuals[tick+1],strict=True)])
                cost=math.fsum([self.control_cost*(delivered-reference)**2]
                               +[float(value)*float(value)for value in value_residual])
                if not math.isfinite(cost)or not np.isfinite(value_residual).all():
                    raise ValueError('nonfinite native tracking objective')
                record['cost'],record['value_residual']=cost,value_residual
            except InterruptedError:
                record['error']='interrupted_or_CPU_reserve'
                raise
            except Exception as failure:
                record['error']=str(failure)
            return record

        def valid():
            return [sample for sample in samples if sample['cost']is not None]

        baseline_record=evaluate(baseline)
        evaluate(reference)
        evaluate(.5*(baseline+reference))
        if not valid():
            raise ValueError('all initial tracking forecasts failed')
        h=max(abs(baseline-reference),8*_ulp(baseline))
        for _ in range(5):
            if len(samples)>=16:
                break
            best=min(valid(),key=lambda item:item['cost'])
            center=best['command']
            evaluate(center-h)
            evaluate(center+h)
            best=min(valid(),key=lambda item:item['cost'])
            left=sorted((item for item in valid()if item['command']<best['command']),key=lambda item:item['command'])
            right=sorted((item for item in valid()if item['command']>best['command']),key=lambda item:item['command'])
            if left and right:
                a,b,c=left[-1],best,right[0]
                x0,x1,x2=a['command'],b['command'],c['command']
                d1=(b['cost']-a['cost'])/(x1-x0)
                d2=((c['cost']-b['cost'])/(x2-x1)-d1)/(x2-x0)
                proposal=(x0+x1-d1/d2)/2 if d2>0 and math.isfinite(d2)else math.nan
                if not math.isfinite(proposal)or not x0<proposal<x2 or _key(_delivered(proposal))in seen:
                    neighbor=a if x1-x0>x2-x1 else c
                    proposal=x1+.3819660112501051*(neighbor['command']-x1)
                evaluate(proposal)
                h=max(.25*min(x1-x0,x2-x1),2*_ulp(best['command']))
            else:
                direction=-1. if not left else 1.
                evaluate(best['command']+direction*2*h)
                h*=2
        best=min(valid(),key=lambda item:item['cost'])
        if len(samples)>16 or (baseline_record['cost']is not None and best['cost']>baseline_record['cost']):
            raise AssertionError('tracking forecast cap/baseline retention failure')
        return TrackingDecision(best['command'],best['state'],best['index'],baseline_record['index'],len(samples),duplicates)
