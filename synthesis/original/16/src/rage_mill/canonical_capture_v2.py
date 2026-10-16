"""Independent high precision upright design for the canonical serial plant.

The design uses compiled MuJoCo scalar constants, COM Jacobians and a discrete
Riccati doubling iteration. It does not import an external controller. The
returned controller and terminal root are binary64; high precision is confined
to design and clearly identified mathematical diagnostics.
"""
from __future__ import annotations

import math

import mpmath as mp
import numpy as np


def _exact(value):
    numerator, denominator = float(value).as_integer_ratio()
    return mp.mpf(numerator) / denominator


def _maximum(matrix):
    return max(abs(value) for value in matrix)


def _array(matrix):
    return np.array([[float(matrix[i, j]) for j in range(matrix.cols)]
                     for i in range(matrix.rows)], dtype=np.float64)


def _doubling(A, B, Q, R, *, maximum_iterations=40):
    """Solve the stabilizing DARE without binary64 eigenspace selection."""
    dimension = A.rows
    current_a, current_g, current_h = A.copy(), B * B.T / R, Q.copy()
    history = []
    for iteration in range(maximum_iterations):
        inverse = (mp.eye(dimension) + current_g * current_h) ** -1
        next_a = current_a * inverse * current_a
        next_g = current_g + current_a * inverse * current_g * current_a.T
        next_h = current_h + current_a.T * current_h * inverse * current_a
        next_g, next_h = (next_g + next_g.T) / 2, (next_h + next_h.T) / 2
        relative_change = _maximum(next_h - current_h) / max(mp.mpf(1), _maximum(next_h))
        history.append({"iteration": iteration + 1,
                        "relative_P_change": mp.nstr(relative_change, 24),
                        "doubling_A_max": mp.nstr(_maximum(next_a), 24)})
        current_a, current_g, current_h = next_a, next_g, next_h
        if relative_change < mp.mpf("1e-65") and _maximum(current_a) < mp.mpf("1e-35"):
            return current_h, history
    raise ArithmeticError("high precision Riccati doubling failed to converge")


def scalar_doubling_check():
    """Nontrivial analytic scalar DARE cross-check of the iteration."""
    with mp.workdps(100):
        a, b, q, r = (mp.mpf(value) for value in ("1.1", ".2", "1", "3"))
        matrix, _ = _doubling(mp.matrix([[a]]), mp.matrix([[b]]), mp.matrix([[q]]), r)
        coefficient = r * (1 - a * a) - q * b * b
        analytic = (-coefficient + mp.sqrt(coefficient**2 + 4 * b * b * q * r)) / (2 * b * b)
        error = abs(matrix[0, 0] - analytic)
        if error >= mp.mpf("1e-65"):
            raise ArithmeticError("scalar DARE analytic cross-check failed")
        return mp.nstr(error, 24)


def upright_design(env, *, decimal_digits=100):
    """Relative [q, v] design; normalized input is delivered at 80 newtons."""
    if decimal_digits < 100:
        raise ValueError("at least 100 decimal digits required")
    model, protocol = env.model, env.protocol
    n, d = protocol.n_links, protocol.n_links + 1
    if (model.nq != d or model.nv != d or model.nu != 1
            or not np.array_equal(model.jnt_axis[1:], np.tile([0., 1., 0.], (n, 1)))):
        raise ValueError("canonical scalar hinge geometry required")
    with mp.workdps(decimal_digits):
        masses = [_exact(value) for value in model.body_mass[2:]]
        centers = [_exact(value) for value in model.body_ipos[2:, 2]]
        lengths = [_exact(value) for value in model.body_pos[3:, 2]] + [2 * centers[-1]]
        # Principal inertial frames need not coincide with body coordinates.
        inertia = []
        for quaternion, principal in zip(model.body_iquat[2:], model.body_inertia[2:], strict=True):
            w, x, y, z = map(_exact, quaternion)
            body_y = (2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x))
            inertia.append(sum(component**2 * _exact(value)
                               for component, value in zip(body_y, principal, strict=True)))
        mass = mp.zeros(d)
        gravity = mp.zeros(d)
        mass[0, 0] = _exact(model.body_mass[1])
        for k in range(n):
            com = mp.zeros(d, 1)
            rotation = mp.zeros(d, 1)
            com[0] = 1
            for j in range(k + 1):
                com[j + 1] = sum(lengths[j:k]) + centers[k]
                rotation[j + 1] = 1
            mass += masses[k] * com * com.T + inertia[k] * rotation * rotation.T
            moment = masses[k] * centers[k] + lengths[k] * sum(masses[k + 1:])
            gravity += _exact(-model.opt.gravity[2]) * moment * rotation * rotation.T
        for index in range(d):
            mass[index, index] += _exact(model.dof_armature[index])
        damping = mp.diag([_exact(value) for value in model.dof_damping])
        continuous = mp.zeros(2 * d + 1)
        inverse_mass = mass ** -1
        for index in range(d):
            continuous[index, d + index] = 1
        acceleration_q, acceleration_v = inverse_mass * gravity, -inverse_mass * damping
        normalized_force = inverse_mass[:, 0] * _exact(protocol.force_limit_N)
        for i in range(d):
            for j in range(d):
                continuous[d + i, j] = acceleration_q[i, j]
                continuous[d + i, d + j] = acceleration_v[i, j]
            continuous[d + i, 2 * d] = normalized_force[i]
        h = _exact(model.opt.timestep)
        one_step = mp.eye(2 * d + 1)
        term = mp.eye(2 * d + 1)
        for order in range(1, 5):
            term = term * continuous * h / order
            one_step += term
        held_map = one_step ** protocol.frame_skip
        A, B = held_map[:2*d, :2*d], held_map[:2*d, 2*d:2*d+1]
        cumulative = mp.matrix([[int(j <= i) for j in range(n)] for i in range(n)])
        gram = cumulative.T * cumulative
        Q = mp.zeros(2*d)
        Q[0, 0], Q[d, d] = mp.mpf(".1"), mp.mpf(".1")
        for i in range(n):
            for j in range(n):
                Q[i+1, j+1] = 100 * gram[i, j] + int(i == j)
                Q[d+i+1, d+j+1] = gram[i, j] + mp.mpf(".01") * int(i == j)
        R = mp.mpf(1000)
        P, history = _doubling(A, B, Q, R)
        K = (B.T * P * A) / (R + (B.T * P * B)[0])
        residual = A.T * P * A - P - A.T * P * B * K + Q
        relative_residual = _maximum(residual) / max(mp.mpf(1), _maximum(P))
        if relative_residual >= mp.mpf("1e-50"):
            raise ArithmeticError("high precision DARE residual failed")
        root = mp.cholesky(P).T
        gain = _array(K).reshape(-1)
        rounded_k = mp.matrix([[_exact(value) for value in gain]])
        closed = A - B * rounded_k
        eigenvalues = mp.eig(closed, left=False, right=False)
        radius = max(abs(value) for value in eigenvalues)
        if not radius < 1:
            raise ArithmeticError("rounded binary64 gain fails high precision model stability check")
        arrays = {"K": gain, "terminal_root": _array(root), "Q": _array(Q),
                  "P": _array(P), "A": _array(A), "B": _array(B),
                  "mass": _array(mass), "gravity": _array(gravity), "damping": _array(damping)}
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ArithmeticError("nonfinite binary64 design")
        env.reset(np.zeros(2*d))
        native_mass = np.empty((d, d))
        env.mj.mj_fullM(model, native_mass, env.data.qM)
        mass_error = float(np.max(np.abs(native_mass - arrays["mass"])))
        if mass_error > 1e-12:
            raise ArithmeticError("independent COM mass does not match native upright mass")
        numpy_radius = float(np.max(np.abs(np.linalg.eigvals(arrays["A"] - arrays["B"] @ gain[None]))))
        diagnostics = dict(decimal_digits=decimal_digits,
                           design="independent COM Jacobian / RK4 held-input polynomial / Riccati doubling",
                           state="relative q(17), relative v(17); cart target zero",
                           input="normalized action; 80 N; float32 delivery in native episodes",
                           Q_description="cart q/v .1; angles 100*T.T*T+I; rates T.T*T+.01*I",
                           R=1000, iterations=history,
                           relative_DARE_residual=mp.nstr(relative_residual, 40),
                           rounded_gain_high_precision_spectral_radius=mp.nstr(radius, 40),
                           numpy_binary64_spectral_radius_DIAGNOSTIC_ONLY=numpy_radius,
                           native_mass_max_error=mass_error,
                           maximum_abs_gain=float(np.max(np.abs(gain))),
                           K_high_precision_decimal=[mp.nstr(K[0, i], decimal_digits) for i in range(2*d)],
                           xml_sha256=env.xml_sha256,
                           high_precision_stability_is_nonlinear_native_certificate=False)
        return arrays, diagnostics


def feedback_dots(gain, state):
    """Three arithmetic scopes; physics and command delivery remain unchanged."""
    gain, state = np.asarray(gain, dtype=np.float64), np.asarray(state, dtype=np.float64)
    products = [float(k) * float(x) for k, x in zip(gain, state, strict=True)]
    ordinary = -float(np.dot(gain, state))
    compensated = -math.fsum(products)
    extended = -float(np.dot(gain.astype(np.longdouble), state.astype(np.longdouble)))
    magnitude = math.fsum(abs(value) for value in products)
    return dict(binary64_numpy=ordinary, binary64_fsum_products=compensated,
                longdouble_diagnostic=extended), magnitude
