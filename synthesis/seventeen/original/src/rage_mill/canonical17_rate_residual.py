"""One bounded proximal-rate residual over immutable seventeen-link feedback.

The measured first hinge rate is compared only with the current reference.
Parking, tables, native physical states and base feedback arithmetic remain
unchanged. Zero amplitude returns the literal base command.
"""
from __future__ import annotations

import numpy as np

from .canonical17_controller import Canonical17FiniteTrajectoryPolicy


class Canonical17RateResidualPolicy(Canonical17FiniteTrajectoryPolicy):
    def __init__(self, *, residual_sign, residual_amplitude, **kwargs):
        super().__init__(**kwargs)
        sign, amplitude = float(residual_sign), float(residual_amplitude)
        if sign not in (-1., 1.) or amplitude not in (0., 1e-4, 1e-3, 1e-2):
            raise ValueError('one registered sign and bounded proximal-rate amplitude required')
        self.residual_parameters = (sign, amplitude)
        self.last_base_raw = None
        self.last_residual = 0.
        self.last_feature = 0.
        self.last_weight = 0.

    def __call__(self, state, tick):
        base = super().__call__(state, tick)
        self.last_base_raw = float(base)
        self.last_residual = self.last_feature = self.last_weight = 0.
        sign, amplitude = self.residual_parameters
        route_tick = tick-self.park_ticks
        if amplitude == 0. or route_tick < 0 or route_tick >= len(self.actions):
            return base
        weight = float(np.clip((route_tick-425)/25, 0., 1.))
        if weight == 0.:
            return base
        feature = float(np.asarray(state, dtype=np.float64)[19]-self.states[route_tick, 19])
        residual = sign*amplitude*weight*float(np.clip(feature/.01, -1., 1.))
        self.last_feature, self.last_weight, self.last_residual = feature, weight, residual
        return base+residual
