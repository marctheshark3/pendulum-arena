"""Offline balance and trajectory feedback for the separate capsule plant."""
from __future__ import annotations

import numpy as np


def balance_design(model, *, digits=100, control_cost=1000.):
    """High precision is confined to design; native simulation stays binary64."""
    import mpmath

    ctx = mpmath.mp.clone()
    ctx.dps = digits
    mp = lambda x: ctx.mpf(float(x))
    n, d = model.nv-1, model.nv
    lengths = [mp(x) for x in model.body_pos[3:, 2]]+[2*mp(model.body_ipos[-1, 2])]
    masses = [mp(x) for x in model.body_mass[2:]]
    centers = [mp(x) for x in model.body_ipos[2:, 2]]
    inertias = []
    for quat, principal in zip(model.body_iquat[2:], model.body_inertia[2:]):
        w, x, y, z = map(mp, quat)
        row = (2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x))
        inertias.append(sum(row[i]**2*mp(principal[i]) for i in range(3)))
    mass = ctx.zeros(d)
    gravity = ctx.zeros(d)
    mass[0, 0] = mp(model.body_mass[1])+sum(masses)
    cumulative = ctx.eye(d)
    for i in range(n):
        for j in range(i+1):
            cumulative[i+1, j+1] = 1
        moment = centers[i]*masses[i]+lengths[i]*sum(masses[i+1:])
        mass[0, i+1] = mass[i+1, 0] = moment
        mass[i+1, i+1] = centers[i]**2*masses[i]+lengths[i]**2*sum(masses[i+1:])+inertias[i]
        gravity[i+1, i+1] = -mp(model.opt.gravity[2])*moment
        for j in range(i):
            coeff = lengths[j]*moment
            mass[j+1, i+1] = mass[i+1, j+1] = coeff
    mass = cumulative.T*mass*cumulative+ctx.diag([mp(v) for v in model.dof_armature])
    gravity = cumulative.T*gravity*cumulative
    inverse = ctx.inverse(mass)
    stiffness = inverse*gravity
    viscous = -inverse*ctx.diag([mp(v) for v in model.dof_damping])
    augmented = ctx.zeros(2*d+1)
    for i in range(d):
        augmented[i, d+i] = 1
        augmented[d+i, 2*d] = 80*inverse[i, 0]
        for j in range(d):
            augmented[d+i, j] = stiffness[i, j]
            augmented[d+i, d+j] = viscous[i, j]
    h = mp(model.opt.timestep)
    step = ctx.eye(2*d+1)
    power = ctx.eye(2*d+1)
    for order in range(1, 5):
        power *= h*augmented/order
        step += power
    discrete = step**4
    a, b = discrete[:2*d, :2*d], discrete[:2*d, 2*d:]
    total = ctx.zeros(n)
    for i in range(n):
        for j in range(i+1):
            total[i, j] = 1
    q = ctx.zeros(2*d)
    q[0, 0] = q[d, d] = ctx.mpf("0.1")
    position, velocity = 100*total.T*total+ctx.eye(n), total.T*total+ctx.mpf("0.01")*ctx.eye(n)
    for i in range(n):
        for j in range(n):
            q[i+1, j+1] = position[i, j]
            q[d+i+1, d+j+1] = velocity[i, j]
    current_a, current_g, p = a.copy(), b*b.T/mp(control_cost), q.copy()
    identity = ctx.eye(2*d)
    converged = False
    for iteration in range(30):
        middle = ctx.inverse(identity+current_g*p)
        next_p = p+current_a.T*p*middle*current_a
        next_g = current_g+current_a*middle*current_g*current_a.T
        next_a = current_a*middle*current_a
        change = ctx.norm(next_p-p)/max(ctx.mpf(1), ctx.norm(next_p))
        p, current_g, current_a = (next_p+next_p.T)/2, (next_g+next_g.T)/2, next_a
        if change < ctx.mpf("1e-40") and ctx.norm(current_a) < ctx.mpf("1e-30"):
            converged = True
            break
    if not converged:
        raise ValueError("offline Riccati design did not converge")
    gain = (b.T*p*a)/(mp(control_cost)+(b.T*p*b)[0])
    residual = a.T*p*a-p-a.T*p*b*gain+q
    relative = ctx.norm(residual)/ctx.norm(p)
    if relative > ctx.mpf("1e-30"):
        raise ValueError("offline Riccati residual too large")
    rounded = np.asarray(gain.tolist(), dtype=np.float64).reshape(-1)
    root = np.asarray(ctx.cholesky(p).T.tolist(), dtype=np.float64)
    radius = max(abs(x) for x in ctx.eig(a-b*ctx.matrix([rounded.tolist()]), left=False, right=False))
    if radius >= 1:
        raise ValueError("rounded feedback is unstable for the design model")
    return dict(gain=rounded, terminal_root=root, A=np.asarray(a.tolist(), dtype=float),
                B=np.asarray(b.tolist(), dtype=float), Q=np.asarray(q.tolist(), dtype=float),
                diagnostics=dict(design_digits=digits, iterations=iteration+1,
                                 relative_residual=str(relative), rounded_gain_spectral_radius=str(radius),
                                 gain_norm=float(np.linalg.norm(rounded)), nonlinear_certified=False))


def trajectory_feedback(states, actions, first, terminal_root, Q, *, control_cost=1000., defects=None,
                        return_values=False):
    """Square-root affine Riccati recursion, retaining the residual column."""
    from scipy.linalg import qr

    states, actions = np.asarray(states), np.asarray(actions)
    count, dimension = len(actions), states.shape[1]
    a, b = first.map(count)(states[:-1].T, actions.reshape(1, -1))
    a, b = np.asarray(a), np.asarray(b)
    qroot = np.linalg.cholesky(Q).T
    root = np.asarray(terminal_root).copy()
    residual = np.zeros(dimension)
    gains = np.empty((count, dimension))
    offsets = np.empty(count)
    roots = np.empty((count+1, dimension, dimension)) if return_values else None
    residuals = np.empty((count+1, dimension)) if return_values else None
    if return_values:
        roots[count], residuals[count] = root, residual
    defects = np.zeros((count, dimension)) if defects is None else np.asarray(defects)
    if defects.shape != (count, dimension) or not np.isfinite(defects).all():
        raise ValueError("one finite transition defect per reference command required")
    for tick in reversed(range(count)):
        aa, bb = a[:, tick*dimension:(tick+1)*dimension], b[:, tick:tick+1]
        stack = np.block([[np.sqrt(control_cost)*np.ones((1, 1)), np.zeros((1, dimension)), np.zeros((1, 1))],
                          [root@bb, root@aa, (root@defects[tick]+residual)[:, None]],
                          [np.zeros((dimension, 1)), qroot, np.zeros((dimension, 1))]])
        upper = qr(stack, mode="economic")[1]
        gains[tick] = upper[0, 1:dimension+1]/upper[0, 0]
        offsets[tick] = upper[0, dimension+1]/upper[0, 0]
        root = upper[1:dimension+1, 1:dimension+1]
        residual = upper[1:dimension+1, dimension+1]
        if return_values:
            roots[tick], residuals[tick] = root, residual
    if return_values:
        return gains, offsets, roots, residuals
    return gains, offsets


def trajectory_gains(states, actions, first, terminal_root, Q, *, control_cost=1000.):
    return trajectory_feedback(states, actions, first, terminal_root, Q, control_cost=control_cost)[0]


def trajectory_feedback_quad(states, actions, first, terminal_root, Q, *, control_cost=1000., defects=None):
    """Offline affine recursion with at least 100 bits of significand.

    This path deliberately refuses platforms whose longdouble is an 80-bit
    extended value. Fifteen's value matrices can have condition numbers near
    1e26; binary64 products cannot preserve the small directions of that value.
    Actual plant states, delivered commands and native physics remain binary64.
    """
    if np.finfo(np.longdouble).nmant < 100:
        raise ValueError("quad Riccati design needs at least 100 significand bits")
    dtype = np.longdouble
    states, actions = np.asarray(states), np.asarray(actions)
    count, dimension = len(actions), states.shape[1]
    aa, bb = first.map(count)(states[:-1].T, actions.reshape(1, -1))
    aa, bb = np.asarray(aa, dtype=dtype), np.asarray(bb, dtype=dtype)
    root = np.asarray(terminal_root, dtype=dtype)
    p = root.T@root
    linear = np.zeros(dimension, dtype=dtype)
    q = np.asarray(Q, dtype=dtype)
    defects = np.zeros((count, dimension)) if defects is None else np.asarray(defects)
    gains, offsets = np.empty((count, dimension)), np.empty(count)
    roots = np.empty((count+1, dimension, dimension))
    residuals = np.empty((count+1, dimension))
    roots[count], residuals[count] = root, linear
    for tick in reversed(range(count)):
        a, b = aa[:, tick*dimension:(tick+1)*dimension], bb[:, tick]
        pa, pb = p@a, p@b
        h = p@np.asarray(defects[tick], dtype=dtype)+linear
        denominator = dtype(control_cost)+b@pb
        numerator = b@pa
        gain, offset = numerator/denominator, (b@h)/denominator
        p = q+a.T@pa-np.outer(numerator, numerator)/denominator
        p = (p+p.T)/2
        linear = a.T@h-numerator*offset
        # Cholesky and forward substitution stay in the wider scalar type;
        # numpy/scipy's LAPACK wrappers downcast it to binary64.
        lower = np.zeros_like(p)
        residual = np.zeros(dimension, dtype=dtype)
        for i in range(dimension):
            diagonal = p[i, i]-lower[i, :i]@lower[i, :i]
            if diagonal <= 0 or not np.isfinite(diagonal):
                raise ValueError("quad Riccati value lost positive definiteness")
            lower[i, i] = np.sqrt(diagonal)
            lower[i+1:, i] = (p[i+1:, i]-lower[i+1:, :i]@lower[i, :i])/lower[i, i]
            residual[i] = (linear[i]-lower[i, :i]@residual[:i])/lower[i, i]
        gains[tick], offsets[tick] = gain, offset
        roots[tick], residuals[tick] = lower.T, residual
    return gains, offsets, roots, residuals


class TrajectoryPolicy:
    def __init__(self, states, actions, gains, balance_gain, offsets=None):
        self.states = np.asarray(states)
        self.actions = np.asarray(actions)
        self.gains = np.asarray(gains)
        self.balance_gain = np.asarray(balance_gain)
        self.offsets = np.zeros(len(actions)) if offsets is None else np.asarray(offsets)

    def __call__(self, state, tick):
        if tick < len(self.actions):
            return self.actions[tick]-self.offsets[tick]-float(self.gains[tick]@(state-self.states[tick]))
        return -float(self.balance_gain@state)


class CanonicalPolicy:
    """Frozen arrays plus deterministic one-command native value feedback.

    The forecast plant is separate from the live episode. Its state is copied
    for predictions; the live plant is never reset by this controller. Each
    episode must construct a new policy. No seed enters feedback.
    """
    def __init__(self, arrays, *, park_ticks, cart_target, native_value_after=180,
                 control_cost=1000., joint_horizon=1, joint_after=390,
                 optimizer_xtol=2e-8):
        from .canonical_benchmark import CanonicalEnv, CanonicalProtocol, hanging_state

        self.protocol = CanonicalProtocol()
        dimension = 2*(self.protocol.n_links+1)
        self.arrays = {key: np.array(value, copy=True) for key, value in arrays.items()}
        count = len(self.arrays["actions"])
        required = dict(states=(count+1, dimension), actions=(count,),
                        gains=(count, dimension), offsets=(count,),
                        roots=(count+1, dimension, dimension),
                        residuals=(count+1, dimension), balance_gain=(dimension,),
                        terminal_root=(dimension, dimension), park_gain=(dimension,),
                        Q=(dimension, dimension))
        for name, shape in required.items():
            value = self.arrays[name]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid frozen controller array: {name}")
            value.flags.writeable = False
        if type(park_ticks) is not int or not 0 <= park_ticks < self.protocol.episode_ticks:
            raise ValueError("integer parking duration inside the episode required")
        if (not np.isfinite(cart_target) or abs(cart_target) >= self.protocol.rail_limit_m
                or type(native_value_after) is not int or native_value_after < 0
                or not np.isfinite(control_cost) or control_cost <= 0):
            raise ValueError("invalid frozen policy parameters")
        self.park_ticks, self.cart_target = park_ticks, cart_target
        self.native_value_after, self.control_cost = native_value_after, control_cost
        if (type(joint_horizon) is not int or not 1 <= joint_horizon <= 4
                or type(joint_after) is not int or joint_after < native_value_after
                or not 0 < optimizer_xtol < .01):
            raise ValueError("invalid forecast optimization parameters")
        self.joint_horizon, self.joint_after = joint_horizon, joint_after
        self.optimizer_xtol = optimizer_xtol
        self.stage_root = np.linalg.cholesky(self.arrays["Q"]).T
        a = self.arrays
        self.route = TrajectoryPolicy(a["states"], a["actions"], a["gains"],
                                      a["balance_gain"], a["offsets"])
        self.initial = hanging_state(self.protocol, 0, exact=True)
        self.predictor = CanonicalEnv(self.protocol)
        self.launch_offset = None
        self.native_prediction_calls = 0
        self.last_tick = -1

    def __call__(self, state, tick):
        from scipy.optimize import minimize_scalar

        if tick != self.last_tick+1:
            raise ValueError("policy requires consecutive live command indices from zero")
        self.last_tick = tick
        a = self.arrays
        if tick < self.park_ticks:
            error = state-self.initial
            error[0] -= self.cart_target
            return -float(a["park_gain"]@error)
        if self.launch_offset is None:
            self.launch_offset = state[0]-a["states"][0, 0]
        centered = state.copy()
        centered[0] -= self.launch_offset
        route_tick = tick-self.park_ticks
        proposed = self.route(centered, route_tick)
        if route_tick < self.native_value_after:
            return proposed
        if self.joint_horizon > 1 and route_tick >= self.joint_after:
            return self._joint_action(state, route_tick, proposed)
        if route_tick < len(a["actions"]):
            target_action = a["actions"][route_tick]
            target = a["states"][route_tick+1]
            root, linear = a["roots"][route_tick+1], a["residuals"][route_tick+1]
        else:
            target_action, target = 0., np.zeros_like(state)
            root, linear = a["terminal_root"], np.zeros_like(state)

        def cost(action):
            # Predict in world coordinates; center the resulting cart error.
            self.predictor.reset(state)
            following, delivered = self.predictor.step(action)
            following[0] -= self.launch_offset
            self.native_prediction_calls += 1
            error = root@(following-target)+linear
            total = self.control_cost*(delivered-target_action)**2
            total += float(error@error)
            return total

        optimized = minimize_scalar(cost, method="bounded", bounds=(-1., 1.),
                                    options={"xatol": self.optimizer_xtol, "maxiter": 40})
        candidates = (float(np.clip(proposed, -1., 1.)), float(optimized.x),
                      float(np.clip(target_action, -1., 1.)))
        return min(candidates, key=cost)

    def _joint_action(self, state, route_tick, proposed):
        from scipy.optimize import least_squares

        a, horizon = self.arrays, self.joint_horizon

        def residual(controls):
            self.predictor.reset(state)
            vector = []
            for ahead, control in enumerate(controls):
                future_tick = route_tick+ahead
                following, delivered = self.predictor.step(control)
                self.native_prediction_calls += 1
                following[0] -= self.launch_offset
                if future_tick < len(a["actions"]):
                    target, target_u = a["states"][future_tick+1], a["actions"][future_tick]
                    root, linear = a["roots"][future_tick+1], a["residuals"][future_tick+1]
                else:
                    target, target_u = np.zeros(32), 0.
                    root, linear = a["terminal_root"], np.zeros(32)
                vector.append(np.sqrt(self.control_cost)*(delivered-target_u))
                if ahead+1 == horizon:
                    vector.extend(root@(following-target)+linear)
                else:
                    vector.extend(self.stage_root@(following-target))
            return np.array(vector)

        def jacobian(controls):
            columns = []
            for coordinate in range(horizon):
                left, right = controls.copy(), controls.copy()
                left[coordinate] -= 1e-4
                right[coordinate] += 1e-4
                columns.append((residual(right)-residual(left))/2e-4)
            return np.array(columns).T

        guess = np.array([a["actions"][route_tick+i] if route_tick+i < len(a["actions"]) else 0.
                          for i in range(horizon)])
        guess[0] = np.clip(proposed, -1., 1.)
        solution = least_squares(residual, np.clip(guess, -1., 1.), jac=jacobian,
                                 bounds=(-1., 1.), max_nfev=20,
                                 xtol=1e-10, ftol=1e-8, gtol=1e-6)
        return float(min((guess, solution.x), key=lambda v: np.linalg.norm(residual(v)))[0])


class CanonicalPlanningPolicy:
    """Bounded full-route native planning after the live hanging settle.

    The profile list and selection rule are fixed before any evaluation.
    Forecasts use only the current measured state and a separate native plant;
    neither a seed nor an evaluation label is available here. Every failed
    forecast is retained by the campaign recorder. A selected route is checked
    against each observed live state before its next command is delivered.
    This deterministic simulation method makes no real-time/hardware claim.
    """
    def __init__(self, arrays, *, controller, forecast_profiles):
        from .canonical_benchmark import CanonicalProtocol

        self.arrays, self.parameters = arrays, dict(controller)
        self.parking = CanonicalPolicy(arrays, **controller)
        self.protocol = CanonicalProtocol()
        if not forecast_profiles or len(forecast_profiles) > 32:
            raise ValueError("one through thirty-two frozen forecast profiles required")
        for profile in forecast_profiles:
            if set(profile) != {"joint_horizon", "optimizer_xtol"}:
                raise ValueError("forecast profiles can vary only their declared search options")
            CanonicalPolicy(arrays, **(self.parameters | profile))
        self.profiles = [dict(profile) for profile in forecast_profiles]
        self.forecasts = []
        self.selected_profile = None
        self.plan_states = self.plan_actions = self.plan_raw = None
        self.native_prediction_calls = 0
        self.last_tick = -1

    def _plan(self, state):
        from .canonical_benchmark import CanonicalEnv, score_tape

        remaining = self.protocol.episode_ticks-self.parameters["park_ticks"]
        for index, profile in enumerate(self.profiles):
            policy = CanonicalPolicy(self.arrays, **(self.parameters | profile | {"park_ticks": 0}))
            plant = CanonicalEnv(self.protocol)
            states, actions, raw_actions = [plant.reset(state)], [], []
            invalid = None
            for tick in range(remaining):
                try:
                    raw = float(policy(states[-1].copy(), tick))
                    following, delivered = plant.step(raw)
                except (ValueError, FloatingPointError) as exc:
                    invalid = str(exc)
                    break
                states.append(following)
                actions.append(delivered)
                raw_actions.append(raw)
                if abs(following[0]) > self.protocol.rail_limit_m:
                    break
            states, actions, raw_actions = map(np.asarray, (states, actions, raw_actions))
            score = score_tape(states, actions, self.protocol, invalid_reason=invalid)
            accepted = bool(len(actions) == remaining and invalid is None
                            and all(score[key] for key in ("finite", "force_ok", "float32_delivery_ok", "rail_ok"))
                            and score["longest_upright_ticks"] >= self.protocol.hold_ticks)
            self.native_prediction_calls += policy.native_prediction_calls+len(actions)
            self.forecasts.append(dict(profile=index, parameters=profile, score=score,
                                       accepted=accepted, states=states, actions=actions,
                                       raw_actions=raw_actions,
                                       native_prediction_calls=policy.native_prediction_calls+len(actions)))
            if accepted:
                self.selected_profile = index
                self.plan_states, self.plan_actions, self.plan_raw = states, actions, raw_actions
                return
        raise ValueError("bounded native route planning exhausted all frozen profiles")

    def __call__(self, state, tick):
        if tick != self.last_tick+1:
            raise ValueError("planning policy requires consecutive live command indices")
        self.last_tick = tick
        if tick < self.parameters["park_ticks"]:
            return self.parking(state, tick)
        if self.plan_states is None:
            self._plan(state)
        route_tick = tick-self.parameters["park_ticks"]
        if not np.array_equal(state, self.plan_states[route_tick]):
            raise ValueError("live native state differs from the selected predicted route")
        return float(self.plan_raw[route_tick])
