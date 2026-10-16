"""Differentiable mechanics of the compiled uniform capsule benchmark.

Planning is separate from native MuJoCo execution. The model retains capsule
COM inertia, the relative-joint armature and relative-joint damping. Its RK4
map and derivatives must be checked against the native plant before use.
"""
from __future__ import annotations

import numpy as np


def planning_functions(model, *, frame_skip=4, force_limit=80., jit=False):
    import casadi as ca
    import mujoco

    n, d = model.nv-1, model.nv
    if (model.nq != d or model.nu != 1 or model.neq or model.nbody != n+2
            or model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4
            or np.any(model.dof_frictionloss) or np.any(model.jnt_stiffness)
            or not np.array_equal(model.body_parentid[1:], np.arange(n+1))):
        raise ValueError("unconstrained passive serial capsule plant required")
    lengths = np.r_[model.body_pos[3:, 2], 2*model.body_ipos[-1, 2]]
    masses, centers = model.body_mass[2:], model.body_ipos[2:, 2]
    inertias = []
    for quat, principal in zip(model.body_iquat[2:], model.body_inertia[2:]):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        inertias.append(float(np.sum(rotation.reshape(3, 3)[1]**2*principal)))
    tail = np.cumsum(masses[::-1])[::-1] - masses
    moments = lengths*tail + centers*masses
    difference = np.eye(d)
    difference[2:, 1:-1] -= np.eye(n-1)
    armature = difference.T@np.diag(model.dof_armature)@difference
    damping = difference.T@np.diag(model.dof_damping)@difference
    transform = np.eye(d)
    transform[1:, 1:] = np.tril(np.ones((n, n)))
    x, u = ca.SX.sym("relative_state", 2*d), ca.SX.sym("normalized_force")
    q, v = ca.DM(transform)@x[:d], ca.DM(transform)@x[d:]
    mass = ca.SX(ca.DM(armature))
    mass[0, 0] += float(model.body_mass[1] + masses.sum())
    centrifugal = ca.SX.zeros(d)
    gravity = ca.SX.zeros(d)
    for i in range(n):
        ii = i+1
        mass[0, ii] += moments[i]*ca.cos(q[ii])
        mass[ii, 0] += moments[i]*ca.cos(q[ii])
        mass[ii, ii] += lengths[i]**2*tail[i]+centers[i]**2*masses[i]+inertias[i]
        centrifugal[0] -= moments[i]*ca.sin(q[ii])*v[ii]**2
        gravity[ii] = float(model.opt.gravity[2])*moments[i]*ca.sin(q[ii])
        for j in range(i+1, n):
            jj = j+1
            coeff = lengths[i]*moments[j]
            mass[ii, jj] += coeff*ca.cos(q[ii]-q[jj])
            mass[jj, ii] += coeff*ca.cos(q[ii]-q[jj])
            centrifugal[ii] += coeff*ca.sin(q[ii]-q[jj])*v[jj]**2
            centrifugal[jj] -= coeff*ca.sin(q[ii]-q[jj])*v[ii]**2
    applied = ca.vertcat(force_limit*u, ca.SX.zeros(n))
    acceleration = ca.solve(mass, applied-centrifugal-gravity-ca.DM(damping)@v)
    derivative = ca.vertcat(x[d:], ca.DM(difference)@acceleration)
    options = {"jit": True, "jit_serialize": "embed",
               "jit_options": {"flags": ["-O2", "-ffp-contract=off"]}} if jit else {}
    continuous = ca.Function("canonical_continuous", [x, u], [derivative], options)
    # Differentiate the implicit mass equation with acceleration held fixed.
    # Solve all derivative right-hand sides together. Differentiating the full
    # RK4 graph directly generated a 15 MB C file and an expensive compile.
    fixed_acceleration = ca.SX.sym("fixed_acceleration", d)
    residual = mass@fixed_acceleration-applied+centrifugal+gravity+ca.DM(damping)@v
    residual_first = ca.Function("canonical_residual_first", [x, u, fixed_acceleration],
                                 [ca.jacobian(residual, x)])
    columns = ca.horzcat(-residual_first(x, u, acceleration), ca.vertcat(force_limit, ca.DM.zeros(n)))
    acceleration_first = ca.DM(difference)@ca.solve(mass, columns)
    continuous_first = ca.Function("canonical_continuous_first", [x, u],
        [ca.vertcat(ca.horzcat(ca.DM.zeros(d, d), ca.DM.eye(d)), acceleration_first[:, :2*d]),
         ca.vertcat(ca.DM.zeros(d), acceleration_first[:, 2*d:])], options)
    x, u = ca.MX.sym("rk4_state", 2*d), ca.MX.sym("held_force")
    following = x
    state_first, input_first = ca.DM.eye(2*d), ca.DM.zeros(2*d, 1)
    h = float(model.opt.timestep)
    for _ in range(frame_skip):
        a = continuous(following, u)
        ja, ba = continuous_first(following, u)
        ax, au = ja@state_first, ja@input_first+ba
        b = continuous(following+h*a/2, u)
        jb, bb = continuous_first(following+h*a/2, u)
        bx, bu = jb@(state_first+h*ax/2), jb@(input_first+h*au/2)+bb
        c = continuous(following+h*b/2, u)
        jc, bc = continuous_first(following+h*b/2, u)
        cx, cu = jc@(state_first+h*bx/2), jc@(input_first+h*bu/2)+bc
        e = continuous(following+h*c, u)
        je, be = continuous_first(following+h*c, u)
        ex, eu = je@(state_first+h*cx), je@(input_first+h*cu)+be
        following = following+h*(a+2*b+2*c+e)/6
        state_first = state_first+h*(ax+2*bx+2*cx+ex)/6
        input_first = input_first+h*(au+2*bu+2*cu+eu)/6
    transition = ca.Function("canonical_rk4", [x, u], [following], options)
    first = ca.Function("canonical_first", [x, u], [state_first, input_first], options)
    return continuous, transition, first
