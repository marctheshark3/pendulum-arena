"""Full square-root affine value recursion for canonical trajectory tracking.

This additive API retains every state-value root and affine residual. It uses
the existing independently authored recursion without modifying its sources.
Roots describe local quadratic costs; they do not certify native stability.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffineFeedback:
    gains: np.ndarray
    offsets: np.ndarray
    roots: np.ndarray
    residuals: np.ndarray


def full_affine_feedback(states,actions,first,terminal_root,Q,*,control_cost=1000.,defects=None):
    """Return u-error feedback and value ||roots[t]e + residuals[t]||².

    The value omits additive constants independent of the state/control. The
    supplied defect is F_native(reference[t], action[t])-reference[t+1].
    """
    from scipy.linalg import qr

    states=np.asarray(states,dtype=np.float64)
    actions=np.asarray(actions,dtype=np.float64)
    if (states.ndim!=2 or states.shape[1]<1 or actions.ndim!=1
            or len(states)!=len(actions)+1 or len(actions)<1
            or not np.isfinite(states).all()or not np.isfinite(actions).all()):
        raise ValueError('one finite reference state per command plus endpoint required')
    count,dimension=len(actions),states.shape[1]
    Q=np.asarray(Q,dtype=np.float64)
    terminal_root=np.asarray(terminal_root,dtype=np.float64)
    if (Q.shape!=(dimension,dimension)or terminal_root.shape!=Q.shape
            or not np.isfinite(Q).all()or not np.isfinite(terminal_root).all()
            or not np.array_equal(Q,Q.T)
            or isinstance(control_cost,bool)or not np.isfinite(control_cost)or control_cost<=0):
        raise ValueError('finite symmetric Q, square terminal root and positive control cost required')
    defects=np.zeros((count,dimension))if defects is None else np.asarray(defects,dtype=np.float64)
    if defects.shape!=(count,dimension)or not np.isfinite(defects).all():
        raise ValueError('one finite native transition defect per reference command required')
    a,b=first.map(count)(states[:-1].T,actions.reshape(1,-1))
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=(dimension,count*dimension)or b.shape!=(dimension,count)or not np.isfinite(a).all()or not np.isfinite(b).all():
        raise ValueError('finite scalar-input transition derivatives required')
    qroot=np.linalg.cholesky(Q).T
    root=terminal_root.copy()
    residual=np.zeros(dimension)
    gains=np.empty((count,dimension))
    offsets=np.empty(count)
    roots=np.empty((count+1,dimension,dimension))
    residuals=np.empty((count+1,dimension))
    roots[count],residuals[count]=root,residual
    for tick in reversed(range(count)):
        aa,bb=a[:,tick*dimension:(tick+1)*dimension],b[:,tick:tick+1]
        stack=np.block([[np.sqrt(control_cost)*np.ones((1,1)),np.zeros((1,dimension)),np.zeros((1,1))],
                        [root@bb,root@aa,(root@defects[tick]+residual)[:,None]],
                        [np.zeros((dimension,1)),qroot,np.zeros((dimension,1))]])
        upper=qr(stack,mode='economic')[1]
        gains[tick]=upper[0,1:dimension+1]/upper[0,0]
        offsets[tick]=upper[0,dimension+1]/upper[0,0]
        root=upper[1:dimension+1,1:dimension+1]
        residual=upper[1:dimension+1,dimension+1]
        roots[tick],residuals[tick]=root,residual
    if not all(np.isfinite(array).all()for array in (gains,offsets,roots,residuals)):
        raise ArithmeticError('nonfinite affine square-root feedback')
    return AffineFeedback(gains,offsets,roots,residuals)
