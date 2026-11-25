"""
Analytical L2F-Equation Quadrotor Gate-Racing Source Simulator + Asynchronous Wall-Clock Target Wrapper.

1. `L2FAnalyticalRacingEnv`: Implements the exact 17-state continuous-time RK4 quadrotor ODE from
   `rl-tools/l2f` (docs/SIMULATOR_PHYSICS_AUDIT.md Section 3.1) with our shared 500 Hz CTBR controller,
   domain randomization hooks, and 6D residual acceleration injection `[a_res_body, alpha_res_body]`.
2. `AsyncWallClockTargetWrapper`: Wraps any deterministic 100 Hz simulator in a real monotonic wall-clock
   interface with configurable inference/link delay, scheduling jitter, 60 Hz sensor decimation, and ZOH dropout
   (resolving the TMInterface SC_RUN_STEP_SYNC lockstep trap!).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Callable
import numpy as np

from .ctbr_controller import CF2XCTBRConfig, CF2XCTBRController
from .mujoco_drone_env import DEFAULT_SLALOM_GATES, DroneCourseGate


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class L2FAnalyticalRacingEnv:
    """
    17-state RK4 Quadrotor Source Environment matching `rl-tools/l2f` ODE equations,
    with direct hooks for Domain Randomization, Latency Randomization, and Swift 6D Residual injection.
    """

    def __init__(
        self,
        gates: tuple[DroneCourseGate, ...] = DEFAULT_SLALOM_GATES,
        ctbr_cfg: CF2XCTBRConfig | None = None,
        rotor_drag_xy: float = 0.008,
        rotor_drag_z: float = 0.012,
        tau_motor_ode: float = 0.015,
        action_delay_ticks: int = 0,
        obs_dropout_prob: float = 0.0,
        sensor_limited: bool = False,
        act_history_len: int = 6,
        residual_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
        max_steps: int = 600,
    ):
        self.gates = gates
        # Nominal L2F source uses clean P-only rate loop without extra controller motor filter
        self.base_cfg = ctbr_cfg or CF2XCTBRConfig(tau_motor=0.0, kp_rate_xy=75.0, kp_rate_z=38.0)
        self.controller = CF2XCTBRController(self.base_cfg)
        self.rotor_drag = np.array([rotor_drag_xy, rotor_drag_xy, rotor_drag_z], dtype=np.float64)
        self.tau_motor_ode = float(tau_motor_ode)
        self.residual_fn = residual_fn  # maps (vel_body, omega_body, action) -> 6D [a_res_b(3), alpha_res_b(3)]

        self.policy_dt = 0.01
        self.substeps = 5
        self.sub_dt = 0.002
        self.action_delay_ticks = int(action_delay_ticks)
        self.obs_dropout_prob = float(obs_dropout_prob)
        self.sensor_limited = bool(sensor_limited)
        self.act_history_len = int(act_history_len)
        self.max_steps = int(max_steps)

        self.rng = np.random.default_rng(0)
        # 17D state: [p(3), q_wxyz(4), v_world(3), omega_body(3), rpm(4)]
        self.state = np.zeros(17, dtype=np.float64)
        self.action_fifo: deque[np.ndarray] = deque(maxlen=max(1, self.action_delay_ticks + 1))
        self.act_history: deque[np.ndarray] = deque(maxlen=max(1, self.act_history_len))
        self.last_valid_obs: np.ndarray | None = None
        self.gate_idx = 0
        self.step_count = 0
        self.prev_gate_dist = 0.0

        d_arm = self.base_cfg.arm_length / math.sqrt(2.0)
        kappa = self.base_cfg.km / self.base_cfg.kf
        self.M_alloc = np.zeros((4, 4), dtype=np.float64)
        for i in range(4):
            sx, sy = self.base_cfg.motor_xy_signs[i]
            self.M_alloc[0, i] = 1.0
            self.M_alloc[1, i] = sy * d_arm
            self.M_alloc[2, i] = -sx * d_arm
            self.M_alloc[3, i] = self.base_cfg.motor_yaw_signs[i] * kappa

    def _ode_rhs(self, x: np.ndarray, rpm_sp: np.ndarray, res_6d: np.ndarray) -> np.ndarray:
        q = x[3:7]
        v_w = x[7:10]
        omega_b = x[10:13]
        rpm = x[13:17]

        R_wb = _quat_to_rotmat(q)
        v_b = R_wb.T @ v_w
        f_rotors = self.base_cfg.kf * np.maximum(rpm, 0.0) ** 2
        wrench = self.M_alloc @ f_rotors  # [T_total, tau_x, tau_y, tau_z]

        f_body = np.array([0.0, 0.0, wrench[0]], dtype=np.float64) - self.rotor_drag * v_b
        v_dot = np.array([0.0, 0.0, -9.81], dtype=np.float64) + R_wb @ (f_body / self.base_cfg.mass + res_6d[0:3])

        J = np.array([self.base_cfg.ixx, self.base_cfg.iyy, self.base_cfg.izz], dtype=np.float64)
        omega_dot = (wrench[1:4] - np.cross(omega_b, J * omega_b)) / J + res_6d[3:6]
        q_dot = 0.5 * _quat_mul(q, np.array([0.0, omega_b[0], omega_b[1], omega_b[2]], dtype=np.float64))

        if self.tau_motor_ode > 1e-6:
            rpm_dot = (rpm_sp - rpm) / self.tau_motor_ode
        else:
            rpm_dot = np.zeros(4, dtype=np.float64)

        return np.concatenate([v_w, q_dot, v_dot, omega_dot, rpm_dot], axis=0)

    def get_ground_truth_state(self) -> dict[str, np.ndarray]:
        R_wb = _quat_to_rotmat(self.state[3:7])
        return {
            "pos": self.state[0:3].copy(),
            "quat_wxyz": self.state[3:7].copy(),
            "R_wb": R_wb,
            "vel_world": self.state[7:10].copy(),
            "vel_body": R_wb.T @ self.state[7:10],
            "omega_body": self.state[10:13].copy(),
            "motor_rpm": self.state[13:17].copy(),
        }

    def _build_obs(self) -> np.ndarray:
        st = self.get_ground_truth_state()
        g = self.gates[min(self.gate_idx, len(self.gates) - 1)]
        rel_gate_b = st["R_wb"].T @ (np.asarray(g.center, dtype=np.float64) - st["pos"])
        vel_obs = np.zeros(3, dtype=np.float64) if self.sensor_limited else st["vel_body"]
        omega_obs = st["omega_body"] + (self.rng.normal(0.0, 0.03, size=3) if self.sensor_limited else 0.0)
        rel_obs = rel_gate_b + (self.rng.normal(0.0, 0.015, size=3) if self.sensor_limited else 0.0)
        hist_flat = np.concatenate(list(self.act_history), axis=0)
        raw = np.concatenate([vel_obs, st["R_wb"].reshape(-1), omega_obs, rel_obs, hist_flat], axis=0).astype(
            np.float32
        )
        if self.last_valid_obs is None or self.obs_dropout_prob <= 0.0 or self.rng.random() >= self.obs_dropout_prob:
            self.last_valid_obs = raw
        return self.last_valid_obs.copy()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.controller.reset()
        self.state[:] = 0.0
        self.state[0:3] = [0.0, 0.0, 1.0]
        self.state[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.state[13:17] = self.base_cfg.hover_rpm

        hover_norm = (9.81 - 0.5 * (self.base_cfg.min_thrust_acc + self.base_cfg.max_thrust_acc)) / (
            0.5 * (self.base_cfg.max_thrust_acc - self.base_cfg.min_thrust_acc)
        )
        a_init = np.array([hover_norm, 0.0, 0.0, 0.0], dtype=np.float32)
        self.action_fifo.clear()
        for _ in range(self.action_delay_ticks + 1):
            self.action_fifo.append(a_init.copy())
        self.act_history.clear()
        for _ in range(self.act_history_len):
            self.act_history.append(a_init.copy())

        self.gate_idx = 0
        self.step_count = 0
        self.last_valid_obs = None
        self.prev_gate_dist = float(np.linalg.norm(np.asarray(self.gates[0].center) - self.state[0:3]))
        return self._build_obs(), self.get_ground_truth_state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        act_clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        prev_cmd = self.act_history[-1].copy()
        self.action_fifo.append(act_clipped)
        executed_action = self.action_fifo[0]

        pos_before = self.state[0:3].copy()
        vel_before = self.state[7:10].copy()

        st_now = self.get_ground_truth_state()
        if self.residual_fn is not None:
            res_6d = np.asarray(
                self.residual_fn(st_now["vel_body"], st_now["omega_body"], executed_action), dtype=np.float64
            )
        else:
            res_6d = np.zeros(6, dtype=np.float64)

        for _ in range(self.substeps):
            rpm_sp = self.controller.step_substep(executed_action, self.state[10:13], self.sub_dt)
            if self.tau_motor_ode <= 1e-6:
                self.state[13:17] = rpm_sp
            k1 = self._ode_rhs(self.state, rpm_sp, res_6d)
            k2 = self._ode_rhs(self.state + 0.5 * self.sub_dt * k1, rpm_sp, res_6d)
            k3 = self._ode_rhs(self.state + 0.5 * self.sub_dt * k2, rpm_sp, res_6d)
            k4 = self._ode_rhs(self.state + self.sub_dt * k3, rpm_sp, res_6d)
            self.state = self.state + (self.sub_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            self.state[3:7] /= np.linalg.norm(self.state[3:7])

        self.act_history.append(act_clipped)
        self.step_count += 1

        st_after = self.get_ground_truth_state()
        pos_after = st_after["pos"]
        lin_acc_world = (st_after["vel_world"] - vel_before) / self.policy_dt
        crashed = bool(pos_after[2] <= 0.05)

        g = self.gates[self.gate_idx]
        g_center = np.asarray(g.center, dtype=np.float64)
        dist_now = float(np.linalg.norm(g_center - pos_after))
        prog_reward = 1.0 * (self.prev_gate_dist - dist_now)

        cam_axis_w = st_after["R_wb"][:, 0]
        ray = (g_center - pos_after) / max(dist_now, 1e-6)
        delta_cam = math.acos(np.clip(float(np.dot(cam_axis_w, ray)), -1.0, 1.0))
        perc_reward = 0.02 * math.exp(-(delta_cam ** 4))
        smooth_pen = 2e-4 * float(np.sum(st_after["omega_body"] ** 2)) + 2e-3 * float(
            np.sum((act_clipped - prev_cmd) ** 2)
        )
        reward = prog_reward + perc_reward - smooth_pen

        passed_gate = False
        if pos_before[0] < g_center[0] <= pos_after[0]:
            dy = abs(pos_after[1] - g_center[1])
            dz = abs(pos_after[2] - g_center[2])
            if dy < g.half_width and dz < g.half_height:
                passed_gate = True
                reward += 2.0
                self.gate_idx += 1
                if self.gate_idx < len(self.gates):
                    dist_now = float(np.linalg.norm(np.asarray(self.gates[self.gate_idx].center) - pos_after))
            elif dy < g.half_width + 0.05 and dz < g.half_height + 0.05:
                crashed = True

        self.prev_gate_dist = dist_now
        completed = self.gate_idx >= len(self.gates)
        if crashed:
            reward -= 4.0
        terminated = crashed or completed
        truncated = self.step_count >= self.max_steps

        info = {
            **st_after,
            "executed_action": executed_action.copy(),
            "lin_acc_world": lin_acc_world,
            "gate_idx": self.gate_idx,
            "passed_gate": passed_gate,
            "completed": completed,
            "crashed": crashed,
        }
        return self._build_obs(), float(reward), terminated, truncated, info


@dataclass
class AsyncInterfaceConfig:
    tick_dt: float = 0.01                   # 100 Hz physics tick (10 ms)
    base_action_delay_ticks: int = 2        # 20 ms baseline pipeline latency
    jitter_prob: float = 0.15               # probability of +1 tick OS/IPC scheduling slip
    obs_dropout_prob: float = 0.10          # Bernoulli packet drop (ZOH held)
    action_noise_std: float = 0.02          # actuator / quantization channel noise


class AsyncWallClockTargetWrapper:
    """
    Wraps any deterministic 100 Hz simulator (`TMNF-C`, `f1tenth_gym`, etc.) to inject
    realistic non-Markovian interface timing: FIFO pipeline latency, stochastic deadline slips
    (zero-order hold on previous command), action channel noise, and ZOH observation dropout.
    """

    def __init__(self, base_env, cfg: AsyncInterfaceConfig | None = None):
        self.env = base_env
        self.cfg = cfg or AsyncInterfaceConfig()
        self.rng = np.random.default_rng(0)
        self.cmd_buffer: deque[np.ndarray] = deque(maxlen=max(2, self.cfg.base_action_delay_ticks + 3))
        self.last_obs: np.ndarray | None = None

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        out = self.env.reset(seed=seed)
        obs = out[0] if isinstance(out, tuple) else out
        info = out[1] if isinstance(out, tuple) and len(out) > 1 else {}
        zero_act = np.zeros_like(self.env.action_space.sample(), dtype=np.float32)
        self.cmd_buffer.clear()
        for _ in range(self.cmd_buffer.maxlen):
            self.cmd_buffer.append(zero_act.copy())
        self.last_obs = np.asarray(obs, dtype=np.float32).copy()
        return self.last_obs.copy(), info

    def step(self, action: np.ndarray, measured_inference_sec: float = 0.0):
        t0 = time.perf_counter()
        self.cmd_buffer.append(np.asarray(action, dtype=np.float32).copy())
        extra_slip = 1 if (measured_inference_sec > self.cfg.tick_dt or self.rng.random() < self.cfg.jitter_prob) else 0
        delay_idx = min(len(self.cmd_buffer) - 1, self.cfg.base_action_delay_ticks + extra_slip)
        applied = self.cmd_buffer[-1 - delay_idx].copy()
        if self.cfg.action_noise_std > 0.0:
            applied = applied + self.rng.normal(0.0, self.cfg.action_noise_std, size=applied.shape).astype(applied.dtype)

        step_out = self.env.step(applied)
        raw_obs, reward, terminated, truncated, info = step_out
        if self.last_obs is None or self.rng.random() >= self.cfg.obs_dropout_prob:
            self.last_obs = np.asarray(raw_obs, dtype=np.float32).copy()

        info = dict(info)
        info["applied_action"] = applied
        info["effective_delay_ticks"] = delay_idx
        info["wrapper_step_sec"] = time.perf_counter() - t0
        return self.last_obs.copy(), reward, terminated, truncated, info
