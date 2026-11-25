"""
Real source/target drone pair: `rl-tools/l2f` (source) -> `gym-pybullet-drones` CF2X (target).

Both simulators are driven at 500 Hz by the same CTBR body-rate loop + quadX allocator
(docs/SIMULATOR_PHYSICS_AUDIT.md gotcha #1). The policy runs at 100 Hz (5 substeps per step).

The allocator is calibrated to the *source* airframe (mass, inertia, yaw torque constant), as if the
firmware were tuned in L2F. Everything else differs, which is the sim-to-sim dynamics gap:

                    L2F source (Crazyflie preset)          PyBullet target (cf2x.urdf)
    mass            0.0306 kg                              0.027 kg
    J               diag(9e-6, 1e-5, 1.7e-5)               diag(1.4e-5, 1.4e-5, 2.17e-5)
    thrust map      quadratic poly in normalized rpm       kf * RPM^2
    max force/rotor 0.110 N (T/W ~ 1.47)                   0.149 N (T/W 2.25)
    yaw torque/N    0.004665 m                             km/kf = 0.0251 m
    motor lag       rise 55 ms / fall 249 ms (internal)    tau_m (wrapper, default 30 ms)
    drag            none                                   PYB_GND_DRAG_DW blade drag + 0.04 Bullet damping
    integrator      RK4                                    Bullet symplectic Euler

Common state convention (FLU world, z up):
    p  world position (3,), R body->world rotation (3, 3), v world velocity (3,), w body rates (3,)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import math
import numpy as np

SUB_DT = 0.002
SUBSTEPS = 5
RL_DT = SUB_DT * SUBSTEPS
GRAVITY = 9.81

# Rotor layout shared by both airframes (motor 0 front-right, then clockwise seen from above).
ROTOR_XY = np.array([[+1.0, -1.0], [-1.0, -1.0], [-1.0, +1.0], [+1.0, +1.0]])
ROTOR_YAW = np.array([-1.0, +1.0, -1.0, +1.0])


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        q = [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


@dataclass
class DroneState:
    p: np.ndarray
    R: np.ndarray
    v: np.ndarray
    w: np.ndarray

    def copy(self) -> "DroneState":
        return DroneState(self.p.copy(), self.R.copy(), self.v.copy(), self.w.copy())

    @staticmethod
    def hover(p=(0.0, 0.0, 1.0)) -> "DroneState":
        return DroneState(np.array(p, dtype=np.float64), np.eye(3), np.zeros(3), np.zeros(3))


@dataclass
class CTBRConfig:
    """Body-rate loop + allocator, calibrated to the L2F source airframe."""

    mass: float = 0.0306
    J: tuple[float, float, float] = (9e-6, 1e-5, 1.7e-5)
    arm_xy: float = 0.028                    # rotor |x| = |y|
    yaw_torque_per_N: float = 0.004665
    c_min: float = 2.0                       # mass-normalized collective thrust range [m/s^2]
    c_max: float = 14.0                      # inside L2F's T/W ~ 1.47 envelope
    rate_max_xy: float = 8.0                 # rad/s
    rate_max_z: float = 3.0
    kp_rate: tuple[float, float, float] = (60.0, 60.0, 20.0)

    def __post_init__(self):
        M = np.zeros((4, 4))
        for i in range(4):
            sx, sy = ROTOR_XY[i]
            M[0, i] = 1.0
            M[1, i] = sy * self.arm_xy
            M[2, i] = -sx * self.arm_xy
            M[3, i] = ROTOR_YAW[i] * self.yaw_torque_per_N
        self.M_inv = np.linalg.inv(M)

    def decode(self, a: np.ndarray) -> tuple[float, np.ndarray]:
        a = np.clip(a, -1.0, 1.0)
        c = 0.5 * (self.c_min + self.c_max) + 0.5 * (self.c_max - self.c_min) * a[0]
        w_cmd = np.array([self.rate_max_xy * a[1], self.rate_max_xy * a[2], self.rate_max_z * a[3]])
        return c, w_cmd

    def encode(self, c: float, w_cmd: np.ndarray) -> np.ndarray:
        a0 = (c - 0.5 * (self.c_min + self.c_max)) / (0.5 * (self.c_max - self.c_min))
        a = np.array([a0, w_cmd[0] / self.rate_max_xy, w_cmd[1] / self.rate_max_xy, w_cmd[2] / self.rate_max_z])
        return np.clip(a, -1.0, 1.0)

    def rotor_forces(self, a: np.ndarray, w_meas: np.ndarray) -> np.ndarray:
        c, w_cmd = self.decode(a)
        J = np.asarray(self.J)
        alpha = np.asarray(self.kp_rate) * (w_cmd - w_meas)
        tau = J * alpha + np.cross(w_meas, J * w_meas)
        return self.M_inv @ np.array([self.mass * c, tau[0], tau[1], tau[2]])


# --------------------------------------------------------------------------------------------------
# Backends: 500 Hz "apply desired rotor forces, advance one substep"
# --------------------------------------------------------------------------------------------------


class L2FBackend:
    """rl-tools/l2f at dt = 2 ms. Desired forces are mapped to normalized rpm setpoints through the
    inverse of L2F's quadratic thrust curve; L2F integrates its own asymmetric motor lag."""

    name = "l2f"

    def __init__(self, seed: int = 0, overrides: dict | None = None):
        import l2f

        self._l2f = l2f
        self.device, self.rng, self.env, self.params = l2f.Device(), l2f.Rng(), l2f.Environment(), l2f.Parameters()
        l2f.initialize_environment(self.device, self.env)
        l2f.initialize_rng(self.device, self.rng, seed)
        l2f.sample_initial_parameters(self.device, self.env, self.params, self.rng)
        P = json.loads(l2f.parameters_to_json(self.device, self.env, self.params))
        P["integration"]["dt"] = SUB_DT
        P["mdp"]["termination"]["enabled"] = False
        self.base_params = P
        self.set_dynamics(overrides or {})
        self.s, self.ns = l2f.State(), l2f.State()
        l2f.initial_state(self.device, self.env, self.params, self.s)
        self.residual_fn = None          # optional (DroneState, action) -> (a_res_body, alpha_res_body)
        self.current_action = np.zeros(4)

    def set_dynamics(self, overrides: dict) -> None:
        """overrides: mass_scale, thrust_scale, tau_rise_scale, tau_fall_scale."""
        P = json.loads(json.dumps(self.base_params))
        d = P["dynamics"]
        d["mass"] *= overrides.get("mass_scale", 1.0)
        ts = overrides.get("thrust_scale", 1.0)
        d["rotor_thrust_coefficients"] = [[c * ts for c in row] for row in d["rotor_thrust_coefficients"]]
        d["rotor_time_constants_rising"] = [t * overrides.get("tau_rise_scale", 1.0) for t in d["rotor_time_constants_rising"]]
        d["rotor_time_constants_falling"] = [t * overrides.get("tau_fall_scale", 1.0) for t in d["rotor_time_constants_falling"]]
        self._l2f.parameters_from_json(self.device, self.env, json.dumps(P), self.params)
        self.thrust_poly = np.array(d["rotor_thrust_coefficients"][0])
        self.f_max = float(np.polyval(self.thrust_poly[::-1], 1.0))

    def force_to_rpm(self, f: np.ndarray) -> np.ndarray:
        c0, c1, c2 = self.thrust_poly
        f = np.clip(f, c0, self.f_max)
        return (-c1 + np.sqrt(c1 * c1 - 4 * c2 * (c0 - f))) / (2 * c2)

    def reset(self, st: DroneState, rpm: np.ndarray | None = None) -> None:
        """rpm: normalized rotor speeds in [0, 1] for the hidden actuator state (default: hover)."""
        self._l2f.initial_state(self.device, self.env, self.params, self.s)
        self.s.position = list(st.p)
        self.s.orientation = list(R_to_quat_wxyz(st.R))
        self.s.linear_velocity = list(st.v)
        self.s.angular_velocity = list(st.w)
        if rpm is None:
            rpm = self.force_to_rpm(np.full(4, self.base_params["dynamics"]["mass"] * GRAVITY / 4.0))
        self.s.rpm = list(rpm)

    def motor_taus(self) -> tuple[float, float]:
        d = json.loads(self._l2f.parameters_to_json(self.device, self.env, self.params))["dynamics"]
        return d["rotor_time_constants_rising"][0], d["rotor_time_constants_falling"][0]

    def snapshot(self):
        snap = self._l2f.State()
        snap.assign(self.s)
        return snap

    def restore(self, snap) -> None:
        self.s.assign(snap)

    def substep(self, f_des: np.ndarray) -> None:
        u = 2.0 * self.force_to_rpm(f_des) - 1.0
        self._l2f.step(self.device, self.env, self.params, self.s, list(u), self.ns, self.rng)
        if self.residual_fn is not None:
            # Swift-style augmented source: add learned body-frame [a_res, alpha_res] over the substep
            st = self.state_of(self.ns)
            a_res_B, alpha_res = self.residual_fn(st, self.current_action)
            self.ns.linear_velocity = list(st.v + SUB_DT * (st.R @ a_res_B))
            self.ns.angular_velocity = list(st.w + SUB_DT * alpha_res)
        self.s.assign(self.ns)

    @staticmethod
    def state_of(s) -> DroneState:
        return DroneState(np.array(s.position), quat_wxyz_to_R(np.array(s.orientation)), np.array(s.linear_velocity), np.array(s.angular_velocity))

    def state(self) -> DroneState:
        return self.state_of(self.s)


class PyBulletBackend:
    """gym-pybullet-drones CF2X, Physics.PYB_GND_DRAG_DW, pyb_freq = ctrl_freq = 500 Hz, with a
    first-order motor lag tau_m (absent in vanilla BaseAviary)."""

    name = "pybullet"

    def __init__(self, tau_m: float = 0.03, physics: str = "PYB_GND_DRAG_DW", tau_down: float | None = None):
        import pybullet as pb
        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
        from gym_pybullet_drones.utils.enums import DroneModel, Physics

        self.pb = pb
        self.env = CtrlAviary(drone_model=DroneModel.CF2X, physics=getattr(Physics, physics),
                              pyb_freq=500, ctrl_freq=500, gui=False, initial_xyzs=np.array([[0.0, 0.0, 1.0]]))
        self.env.reset(seed=0)
        self.kf = self.env.KF
        self.f_max = self.kf * self.env.MAX_RPM ** 2
        self.tau_m = tau_m
        self.tau_down = tau_m if tau_down is None else tau_down  # asymmetric spin-down (slow-motor variant)
        self.rpm = np.full(4, self.env.HOVER_RPM)

    def reset(self, st: DroneState) -> None:
        e, pb = self.env, self.pb
        q = R_to_quat_wxyz(st.R)
        pb.resetBasePositionAndOrientation(e.DRONE_IDS[0], st.p.tolist(), [q[1], q[2], q[3], q[0]], physicsClientId=e.CLIENT)
        pb.resetBaseVelocity(e.DRONE_IDS[0], st.v.tolist(), (st.R @ st.w).tolist(), physicsClientId=e.CLIENT)
        e._updateAndStoreKinematicInformation()
        self.rpm = np.full(4, e.HOVER_RPM)

    def substep(self, f_des: np.ndarray) -> None:
        rpm_sp = np.sqrt(np.clip(f_des, 0.0, self.f_max) / self.kf)
        if self.tau_m > 1e-6:
            tau = np.where(rpm_sp >= self.rpm, self.tau_m, self.tau_down)
            self.rpm = self.rpm + (1.0 - np.exp(-SUB_DT / tau)) * (rpm_sp - self.rpm)
        else:
            self.rpm = rpm_sp
        self.env.step(self.rpm.reshape(1, 4))

    def state(self) -> DroneState:
        e = self.env
        R = np.array(self.pb.getMatrixFromQuaternion(e.quat[0])).reshape(3, 3)
        return DroneState(e.pos[0].copy(), R, e.vel[0].copy(), R.T @ e.ang_v[0])


# --------------------------------------------------------------------------------------------------
# 100 Hz CTBR drone with optional interface gap (action delay FIFO, observation delay, ZOH dropout)
# --------------------------------------------------------------------------------------------------


@dataclass
class InterfaceConfig:
    action_delay: int = 0          # d_a ticks
    obs_delay: int = 0             # d_o ticks
    p_drop: float = 0.0            # Bernoulli observation dropout
    dropout_mode: str = "zoh"      # "zoh" (Lemma 1.1) or "zero" (naive masking)
    obs_noise: dict = field(default_factory=dict)  # per-field std: {"p": .., "v": .., "w": ..}
    obs_bias_fn: object = None     # optional DroneState -> position bias (3,): systematic, pose-correlated b(s)


class CTBRDrone:
    def __init__(self, backend, ctbr: CTBRConfig | None = None, iface: InterfaceConfig | None = None, seed: int = 0):
        self.backend = backend
        self.ctbr = ctbr or CTBRConfig()
        self.iface = iface or InterfaceConfig()
        self.np_rng = np.random.default_rng(seed)
        self.hover_action = self.ctbr.encode(GRAVITY, np.zeros(3))

    def reset(self, st: DroneState) -> DroneState:
        self.backend.reset(st)
        n = self.iface.action_delay + 1
        self.act_fifo = deque([self.hover_action.copy()] * n, maxlen=n)
        m = self.iface.obs_delay + 1
        self.obs_fifo = deque([st.copy()] * m, maxlen=m)
        self.last_obs = st.copy()
        self.applied = self.hover_action.copy()
        return self.observe()

    def step(self, a: np.ndarray) -> DroneState:
        """Queue policy action a_t, execute a_{t - d_a} for 10 ms, return the (delayed/dropped) obs."""
        self.act_fifo.append(np.asarray(a, dtype=np.float64))
        self.applied = self.act_fifo[0]
        self.backend.current_action = self.applied
        for _ in range(SUBSTEPS):
            w = self.backend.state().w
            self.backend.substep(self.ctbr.rotor_forces(self.applied, w))
        self.obs_fifo.append(self.backend.state())
        return self.observe()

    def true_state(self) -> DroneState:
        return self.backend.state()

    def in_flight(self) -> list[np.ndarray]:
        """Actions already queued but not yet executed, oldest first (a_{t-d}, ..., a_{t-1})."""
        return list(self.act_fifo)[1:]

    def predict_forward(self, actions) -> DroneState:
        """Exact source-model rollout of `actions` from the current state (needs backend.snapshot)."""
        snap = self.backend.snapshot()
        for a in actions:
            self.backend.current_action = a
            for _ in range(SUBSTEPS):
                self.backend.substep(self.ctbr.rotor_forces(a, self.backend.state().w))
        st = self.backend.state()
        self.backend.restore(snap)
        return st

    def observe(self) -> DroneState:
        y = self.obs_fifo[0].copy()
        if self.iface.obs_bias_fn is not None:
            y.p = y.p + self.iface.obs_bias_fn(y)
        for k, std in self.iface.obs_noise.items():
            setattr(y, k, getattr(y, k) + std * self.np_rng.standard_normal(3))
        self.dropped = self.iface.p_drop > 0 and self.np_rng.random() < self.iface.p_drop
        if self.dropped:
            if self.iface.dropout_mode == "zoh":
                return self.last_obs.copy()
            return DroneState(np.zeros(3), np.zeros((3, 3)), np.zeros(3), np.zeros(3))
        self.last_obs = y
        return y.copy()


def make_source(seed: int = 0, overrides: dict | None = None, iface: InterfaceConfig | None = None) -> CTBRDrone:
    return CTBRDrone(L2FBackend(seed, overrides), iface=iface, seed=seed)


def make_target(tau_m: float = 0.03, iface: InterfaceConfig | None = None, seed: int = 0,
                tau_down: float | None = None) -> CTBRDrone:
    return CTBRDrone(PyBulletBackend(tau_m, tau_down=tau_down), iface=iface, seed=seed)


# --------------------------------------------------------------------------------------------------
# Expert: geometric position controller -> CTBR action (privileged, exact current state)
# --------------------------------------------------------------------------------------------------


@dataclass
class GeometricExpert:
    ctbr: CTBRConfig = field(default_factory=CTBRConfig)
    kp: float = 6.0
    kd: float = 4.0
    k_att: float = 7.0

    def __call__(self, st: DroneState, p_ref, v_ref, a_ref) -> np.ndarray:
        a_des = self.kp * (p_ref - st.p) + self.kd * (v_ref - st.v) + a_ref + np.array([0.0, 0.0, GRAVITY])
        n = np.linalg.norm(a_des)
        z_des = a_des / max(n, 1e-6)
        x_c = np.array([1.0, 0.0, 0.0])
        y_des = np.cross(z_des, x_c)
        y_des /= np.linalg.norm(y_des)
        x_des = np.cross(y_des, z_des)
        R_des = np.column_stack([x_des, y_des, z_des])
        E = R_des.T @ st.R - st.R.T @ R_des
        e_R = 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]])
        c = float(a_des @ st.R[:, 2])
        return self.ctbr.encode(c, -self.k_att * e_R)


# --------------------------------------------------------------------------------------------------
# Reference courses (training course + held-out course)
# --------------------------------------------------------------------------------------------------


@dataclass
class Course:
    """Smooth periodic reference p(t) with analytic v(t), a(t); gates sampled along it."""

    kind: str = "figure8"
    scale: float = 2.0
    period: float = 5.0
    z0: float = 1.2
    n_gates: int = 6
    gate_radius: float = 0.25

    def ref(self, t: float):
        w = 2 * math.pi / self.period
        A = self.scale
        if self.kind == "figure8":
            p = np.array([A * math.sin(w * t), 0.5 * A * math.sin(2 * w * t), self.z0])
            v = np.array([A * w * math.cos(w * t), A * w * math.cos(2 * w * t), 0.0])
            a = np.array([-A * w * w * math.sin(w * t), -2 * A * w * w * math.sin(2 * w * t), 0.0])
        elif self.kind == "ellipse3d":  # held-out: different shape, direction, and altitude changes
            p = np.array([A * math.cos(w * t), 0.7 * A * math.sin(w * t), self.z0 + 0.4 * math.sin(3 * w * t)])
            v = np.array([-A * w * math.sin(w * t), 0.7 * A * w * math.cos(w * t), 1.2 * w * math.cos(3 * w * t)])
            a = np.array([-A * w * w * math.cos(w * t), -0.7 * A * w * w * math.sin(w * t), -3.6 * w * w * math.sin(3 * w * t)])
        else:
            raise ValueError(self.kind)
        return p, v, a

    def gate_times(self):
        return [(i + 0.5) * self.period / self.n_gates for i in range(self.n_gates)]

    def start_state(self) -> DroneState:
        p, v, _ = self.ref(0.0)
        return DroneState(p, np.eye(3), v, np.zeros(3))

    def peak_speed(self) -> float:
        return max(np.linalg.norm(self.ref(t)[1]) for t in np.linspace(0, self.period, 200))


def gate_pass_rate(course: Course, traj_p: np.ndarray, t0: float = 0.0) -> float:
    """Fraction of gates (over all laps flown) whose center the drone passes within gate_radius."""
    T = len(traj_p)
    passed, total = 0, 0
    lap = 0
    while True:
        times = [lap * course.period + tg for tg in course.gate_times()]
        if times[0] / RL_DT >= T:
            break
        for tg in times:
            k = int(round((tg - t0) / RL_DT))
            if k >= T:
                break
            g = course.ref(tg)[0]
            lo, hi = max(0, k - 25), min(T, k + 26)
            total += 1
            passed += int(np.min(np.linalg.norm(traj_p[lo:hi] - g, axis=1)) < course.gate_radius)
        lap += 1
    return passed / max(total, 1)
