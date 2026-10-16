"""Separate reproduction of Corben Sorenson's published cart-pole benchmark.

This plant intentionally has its own protocol. It does not change the sealed
Rage Mill point-mass plant, actuator queue or qualification receipts. Physical
constants and rounding follow the pinned public configuration/XML generator;
the implementation here is independent. All hinges remain passive.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Callable
import xml.etree.ElementTree as ET

import numpy as np

SOURCE_COMMIT = "b84c6413974d5c0831cdf261ce15e58ba65cd3f5"
SOURCE_REPOSITORY = "https://github.com/corbensorenson/cartpole"


@dataclass(frozen=True)
class CanonicalProtocol:
    n_links: int = 15
    total_length_m: float = 3.0
    total_mass_kg: float = 1.0
    cart_mass_kg: float = 1.0
    force_limit_N: float = 80.0
    rail_limit_m: float = 3.0
    physics_dt_s: float = 0.005
    frame_skip: int = 4
    episode_seconds: float = 30.0
    upright_limit_rad: float = 0.15
    required_hold_seconds: float = 5.0
    cart_damping: float = 0.02
    total_hinge_damping: float = 0.015
    joint_armature: float = 0.0005
    capsule_radius_m: float = 0.025
    reset_noise_std: float = 0.05

    def __post_init__(self):
        if type(self.n_links) is not int or not 1 <= self.n_links <= 20:
            raise ValueError("canonical link count must be an integer in 1..20")
        # A different task must receive a different protocol, not silently use
        # this campaign's name and acceptance criteria.
        defaults = {k: f.default for k, f in self.__dataclass_fields__.items()}
        for key, value in asdict(self).items():
            if key != "n_links" and (isinstance(value, bool) or value != defaults[key]):
                raise ValueError(f"canonical benchmark field differs: {key}")

    @property
    def control_dt(self):
        return self.physics_dt_s * self.frame_skip

    @property
    def episode_ticks(self):
        return round(self.episode_seconds / self.control_dt)

    @property
    def hold_ticks(self):
        return round(self.required_hold_seconds / self.control_dt)


def _number(value):
    # The public generator serializes physical constants to nine significant
    # digits. In particular, do not replace the compiled rounded masses with
    # an exact rational model while claiming identical dynamics.
    return format(float(value), ".9g")


def canonical_xml(protocol: CanonicalProtocol = CanonicalProtocol()) -> str:
    p = protocol
    root = ET.Element("mujoco", model=f"canonical_{p.n_links}_link")
    ET.SubElement(root, "compiler", angle="radian", coordinate="local", inertiafromgeom="true")
    ET.SubElement(root, "option", timestep=_number(p.physics_dt_s), gravity="0 0 -9.81",
                  integrator="RK4", iterations="20")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", contype="0", conaffinity="0", friction="0 0 0")
    ET.SubElement(default, "joint", armature=_number(p.joint_armature))
    world = ET.SubElement(root, "worldbody")
    cart = ET.SubElement(world, "body", name="cart", pos="0 0 0")
    ET.SubElement(cart, "joint", name="slide", type="slide", axis="1 0 0", limited="true",
                  range=f"-{_number(p.rail_limit_m)} {_number(p.rail_limit_m)}",
                  damping=_number(p.cart_damping), frictionloss="0")
    ET.SubElement(cart, "geom", name="cart_geom", type="box", size="0.18 0.12 0.07",
                  mass=_number(p.cart_mass_kg))
    length = _number(p.total_length_m / p.n_links)
    body = cart
    for i in range(p.n_links):
        body = ET.SubElement(body, "body", name=f"link_{i+1}", pos=f"0 0 {'0.07' if i == 0 else length}")
        ET.SubElement(body, "joint", name=f"hinge_{i+1}", type="hinge", axis="0 1 0",
                      damping=_number(p.total_hinge_damping / p.n_links), frictionloss="0")
        ET.SubElement(body, "geom", name=f"link_{i+1}_geom", type="capsule",
                      fromto=f"0 0 0 0 0 {length}", size=_number(p.capsule_radius_m),
                      mass=_number(p.total_mass_kg / p.n_links))
    actuator = ET.SubElement(root, "actuator")
    ET.SubElement(actuator, "motor", name="cart_motor", joint="slide", gear="1", ctrllimited="true",
                  ctrlrange=f"-{_number(p.force_limit_N)} {_number(p.force_limit_N)}")
    return ET.tostring(root, encoding="unicode")


def hanging_state(protocol: CanonicalProtocol, seed: int, *, exact=False):
    """Relative joint coordinates, including the reference RNG draw order."""
    n = protocol.n_links
    q = np.zeros(n + 1)
    q[1] = np.pi
    if exact:
        return np.r_[q, np.zeros(n + 1)]
    rng = np.random.default_rng(seed)
    q[0] += rng.normal(0., 0.)
    q[1:] += rng.normal(0., protocol.reset_noise_std, n)
    velocity = rng.normal(0., protocol.reset_noise_std, n + 1)
    velocity[0] = rng.normal(0., protocol.reset_noise_std)
    return np.r_[q, velocity]


def absolute_angles(states: np.ndarray, n_links: int):
    # Accumulate the serial angles before wrapping, including the +/-pi seam.
    theta = np.cumsum(np.asarray(states)[..., 1:n_links+1], axis=-1)
    wrapped = (theta + np.pi) % (2*np.pi) - np.pi
    # Match the source's preservation of representable errors near upright.
    return np.where((theta >= -np.pi) & (theta < np.pi), theta, wrapped)


def score_tape(states, actions, protocol=CanonicalProtocol(), *, invalid_reason=None):
    """Five consecutive seconds anywhere, plus the complete in-rail episode."""
    states, actions = np.asarray(states), np.asarray(actions)
    dimension = 2 * (protocol.n_links + 1)
    if (states.ndim != 2 or states.shape[1] != dimension or actions.ndim != 1
            or len(states) != len(actions) + 1 or len(actions) > protocol.episode_ticks):
        raise ValueError("one initial state and one subsequent state per command required")
    finite = bool(np.isfinite(states).all() and np.isfinite(actions).all())
    force_ok = bool(np.all(np.abs(actions) <= 1.))
    delivery_ok = bool(np.array_equal(actions, actions.astype(np.float32).astype(actions.dtype)))
    rail_ok = bool(np.all(np.abs(states[:, 0]) <= protocol.rail_limit_m))
    upright = np.all(np.abs(absolute_angles(states[1:], protocol.n_links)) < protocol.upright_limit_rad, axis=1)
    streak = longest = 0
    for value in upright:
        streak = streak + 1 if value else 0
        longest = max(longest, streak)
    complete = len(actions) == protocol.episode_ticks
    return dict(success=bool(finite and force_ok and delivery_ok and rail_ok and complete
                             and invalid_reason is None and longest >= protocol.hold_ticks),
                complete=complete, finite=finite, force_ok=force_ok, float32_delivery_ok=delivery_ok,
                rail_ok=rail_ok, ticks=len(actions), longest_upright_ticks=longest,
                longest_upright_seconds=longest * protocol.control_dt,
                maximum_cart_m=float(np.max(np.abs(states[:, 0]))), invalid_reason=invalid_reason)


class CanonicalEnv:
    """Native reference cadence with checked integration and immediate action."""
    def __init__(self, protocol=CanonicalProtocol()):
        import mujoco
        self.mj, self.protocol = mujoco, protocol
        self.xml = canonical_xml(protocol)
        self.xml_sha256 = hashlib.sha256(self.xml.encode()).hexdigest()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.ticks = 0

    @property
    def state(self):
        return np.r_[self.data.qpos, self.data.qvel]

    def reset(self, state):
        state = np.asarray(state, dtype=np.float64)
        d = self.protocol.n_links + 1
        if state.shape != (2*d,) or not np.isfinite(state).all():
            raise ValueError("finite relative-coordinate initial state required")
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[:], self.data.qvel[:] = state[:d], state[d:]
        self.mj.mj_forward(self.model, self.data)
        self.ticks = 0
        return self.state

    def step(self, action):
        raw = np.asarray(action)
        if raw.size != 1 or not np.isfinite(raw).all():
            raise ValueError("one finite normalized action required")
        delivered = float(np.clip(np.float32(raw.reshape(-1)[0]), -1., 1.))
        self.data.ctrl[0] = delivered * self.protocol.force_limit_N
        warnings = tuple(w.number for w in self.data.warning)
        for _ in range(self.protocol.frame_skip):
            previous = self.data.time
            if not np.isfinite(previous):
                raise ValueError("non-finite native clock before integration")
            self.mj.mj_step(self.model, self.data)
            if (tuple(w.number for w in self.data.warning) != warnings
                    or not np.isfinite(self.data.time)
                    or abs(self.data.time - previous - self.protocol.physics_dt_s) > 1e-10
                    or not all(np.isfinite(x).all() for x in (self.data.qpos, self.data.qvel, self.data.qacc))):
                raise ValueError("invalid native integration or automatic reset")
        self.ticks += 1
        return self.state, delivered


def record_episode(controller: Callable, initial, protocol=CanonicalProtocol()):
    env = CanonicalEnv(protocol)
    states, actions, raw_actions = [env.reset(initial)], [], []
    invalid = None
    for tick in range(protocol.episode_ticks):
        try:
            raw = float(controller(states[-1].copy(), tick))
            state, delivered = env.step(raw)
        except (ValueError, FloatingPointError) as exc:
            invalid = str(exc)
            break
        raw_actions.append(raw)
        actions.append(delivered)
        states.append(state)
        if abs(state[0]) > protocol.rail_limit_m:
            break
    states, actions, raw_actions = np.asarray(states), np.asarray(actions), np.asarray(raw_actions)
    return states, actions, raw_actions, score_tape(states, actions, protocol, invalid_reason=invalid)
