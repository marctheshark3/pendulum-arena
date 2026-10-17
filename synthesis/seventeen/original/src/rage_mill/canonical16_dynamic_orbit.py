"""Discrete held-input harmonic guesses for the registered fast orbit test."""
from __future__ import annotations
import numpy as np


def discrete_harmonic_initializer(A,B,frequency,*,dt=.02):
    A,B=np.asarray(A),np.asarray(B)
    m=A.shape[0];d=m//2
    periods={2.5:20,5.:10,6.25:8};repetitions={2.5:5,5.:10,6.25:12}
    if (frequency not in periods or A.shape!=(m,m) or m%2 or m<4 or B.shape!=(m,1)
            or not np.isfinite(A).all() or not np.isfinite(B).all() or dt!=.02):
        raise ValueError('finite discrete held-input map and registered fast frequency required')
    omega=2*np.pi*frequency;phase=np.exp(1j*omega*dt)
    response=np.linalg.solve(phase*np.eye(m)-A,B[:,0])
    absolute_response=np.cumsum(response[1:d])
    amplitude=min(.08/np.max(np.abs(absolute_response)),.5,.75/max(abs(response[0]),1e-300))
    period,repeats=periods[frequency],repetitions[frequency];count=period*repeats
    first=(amplitude*np.cos(omega*dt*np.arange(period//2))).astype(np.float32)
    actions=np.tile(np.r_[first,-first].astype(np.float32),repeats).astype(np.float64)
    states=np.real(np.exp(1j*omega*dt*np.arange(count))[:,None]*(amplitude*response)[None])
    return states,actions,dict(initializer='discrete held-input (exp(i*omega*.02)I-A)^-1 B',
                              frequency_Hz=frequency,period_ticks=period,repeats=repeats,
                              harmonic_force_amplitude_normalized=float(amplitude),
                              harmonic_absolute_angle_amplitude=float(amplitude*np.max(np.abs(absolute_response))),
                              harmonic_cart_amplitude=float(amplitude*abs(response[0])),
                              harmonic_spatial_angle_amplitude_variation=float(np.ptp(np.abs(amplitude*absolute_response))))
