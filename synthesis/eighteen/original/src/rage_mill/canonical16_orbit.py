"""Small forced periodic orbit discriminator for the canonical benchmark.

All operations here are offline. Replacing one cyclic cart equation by a
translation gauge does not waive the original equation: every original
held-map row remains part of the final closure check.
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve


def harmonic_initializer(mass, gravity, damping, frequency, *, dt=.02, repeats=1):
    mass,gravity,damping=map(np.asarray,(mass,gravity,damping))
    d=mass.shape[0];m=2*d
    if (mass.shape!=(d,d) or gravity.shape!=(d,d) or damping.shape!=(d,d)
            or frequency not in (.25,.5,1.) or repeats not in (1,2)
            or not all(np.isfinite(a).all() for a in (mass,gravity,damping))):
        raise ValueError('finite compiled mechanics and registered harmonic frequency required')
    a=np.block([[np.zeros((d,d)),np.eye(d)],[np.linalg.solve(mass,gravity),-np.linalg.solve(mass,damping)]])
    b=np.r_[np.zeros(d),80*np.linalg.solve(mass,np.eye(d)[:,0])]
    omega=2*np.pi*frequency
    response=np.linalg.solve(1j*omega*np.eye(m)-a,b)
    absolute_response=np.cumsum(response[1:d])
    amplitude=min(.08/np.max(np.abs(absolute_response)),.5,.75/max(abs(response[0]),1e-300))
    period=round(1/(frequency*dt));count=period*repeats
    # Exact paired half-period float32 values, then repeated whole periods.
    first=(amplitude*np.cos(omega*dt*np.arange(period//2))).astype(np.float32)
    one=np.r_[first,-first].astype(np.float32)
    actions=np.tile(one,repeats).astype(np.float64)
    states=np.real(np.exp(1j*omega*dt*np.arange(count))[:,None]*(amplitude*response)[None])
    return states,actions,dict(frequency_Hz=frequency,period_ticks=period,repeats=repeats,
                              harmonic_force_amplitude_normalized=float(amplitude),
                              harmonic_absolute_angle_amplitude=float(amplitude*np.max(np.abs(absolute_response))),
                              harmonic_cart_amplitude=float(amplitude*abs(response[0])),
                              harmonic_spatial_angle_amplitude_variation=float(np.ptp(np.abs(amplitude*absolute_response))))


def cyclic_values(transition, states, actions):
    count=len(actions)
    following=np.asarray(transition.map(count)(states.T,actions.reshape(1,-1))).T
    return np.roll(states,-1,axis=0)-following


def gauge_system(first, states, actions, residuals):
    count,dimension=states.shape;total=count*dimension
    aa,_=first.map(count)(states.T,actions.reshape(1,-1));aa=np.asarray(aa)
    rr=[];cc=[];vv=[]
    for tick in range(count):
        rr.append(np.repeat(tick*dimension+np.arange(dimension),dimension))
        cc.append(np.tile(tick*dimension+np.arange(dimension),dimension))
        vv.append(-aa[:,tick*dimension:(tick+1)*dimension].ravel())
        rr.append(tick*dimension+np.arange(dimension))
        cc.append(((tick+1)%count)*dimension+np.arange(dimension))
        vv.append(np.ones(dimension))
    rows,cols,values=np.concatenate(rr),np.concatenate(cc),np.concatenate(vv)
    dropped=(count-1)*dimension
    keep=rows!=dropped
    rows=np.r_[rows[keep],np.full(count,dropped)]
    cols=np.r_[cols[keep],np.arange(count)*dimension]
    values=np.r_[values[keep],np.full(count,1/count)]
    matrix=sparse.coo_matrix((values,(rows,cols)),shape=(total,total)).tocsc()
    right=residuals.ravel().copy();right[dropped]=np.mean(states[:,0])
    return matrix,right,dropped


def refine_cyclic(transition,first,states,actions,*,maximum_iterations=8,maximum_halvings=6,tolerance=1e-10):
    states=np.asarray(states,dtype=np.float64).copy();actions=np.asarray(actions,dtype=np.float64)
    history=[];value_maps=0;first_maps=0
    for iteration in range(maximum_iterations+1):
        residuals=cyclic_values(transition,states,actions);value_maps+=1
        original_max=float(np.max(np.abs(residuals)));gauge=float(np.mean(states[:,0]))
        history.append(dict(iteration=iteration,original_max=original_max,mean_cart_gauge=gauge))
        if original_max<=tolerance and abs(gauge)<=tolerance:break
        if iteration==maximum_iterations:break
        jac,right,dropped=gauge_system(first,states,actions,residuals);first_maps+=1
        with warnings.catch_warnings():
            warnings.simplefilter('error',MatrixRankWarning)
            step=spsolve(jac,-right).reshape(states.shape)
        if not np.isfinite(step).all():raise ArithmeticError('nonfinite cyclic Newton correction')
        old=float(np.max(np.abs(right)));accepted=False
        for halving in range(maximum_halvings+1):
            alpha=2.**-halving;candidate=states+alpha*step
            if (np.max(np.abs(candidate[:,0]))>2.5 or np.max(np.abs(candidate[:,1:states.shape[1]//2].cumsum(axis=1)),initial=0.)>.12
                    or np.max(np.abs(candidate))>80):continue
            values=cyclic_values(transition,candidate,actions);value_maps+=1
            modified=values.ravel().copy();modified[dropped]=np.mean(candidate[:,0]);merit=float(np.max(np.abs(modified)))
            if merit<=(1-.01*alpha)*old or np.max(np.abs(values))<=tolerance:
                states=candidate;accepted=True;history[-1].update(alpha=alpha,modified_merit=merit);break
        if not accepted:history[-1]['line_search_failed']=True;break
    final=cyclic_values(transition,states,actions);value_maps+=1
    return states,final,dict(history=history,value_maps=value_maps,first_maps=first_maps,
                             dropped_row_index=(len(actions)-1)*states.shape[1],
                             dropped_original_cart_row=float(final[-1,0]),
                             mean_cart_gauge=float(np.mean(states[:,0])),
                             all_original_rows_closed=bool(np.max(np.abs(final))<=tolerance and abs(np.mean(states[:,0]))<=tolerance))
