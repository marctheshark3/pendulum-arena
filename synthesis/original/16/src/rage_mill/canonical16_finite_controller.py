"""Explicit finite-route policy followed by zero delivered force.

The inherited parking/tracking arithmetic is unchanged. This alternative
makes no stability or task-pass claim; the complete native tape decides that.
"""
from __future__ import annotations
import numpy as np
from rage_mill.canonical16_controller import Canonical16TrajectoryPolicy


class Canonical16FiniteTrajectoryPolicy(Canonical16TrajectoryPolicy):
    def __call__(self,state,tick):
        if type(tick) is int and tick>=self.park_ticks+len(self.actions):
            state=np.asarray(state,dtype=np.float64)
            if (state.shape!=(34,) or not np.isfinite(state).all()
                    or tick!=self.next_tick or not 0<=tick<1500):
                raise ValueError('one finite physical state for each consecutive episode tick required')
            self.next_tick+=1
            return 0.
        return super().__call__(state,tick)
