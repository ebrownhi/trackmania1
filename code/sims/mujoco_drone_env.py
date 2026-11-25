"""
Native 3D 6-DoF MuJoCo Quadrotor Gate-Racing Environment (Crazyflie 2.x / CF2X).

Serves two roles:
1. Immediate zero-download 3D rigid-body physics simulator (works with both `hw3/.venv` mujoco>=2.2
   and the new `pixi` mujoco>=3.1 environment).
2. Official Pivot Option B target simulator (docs/MAC_STACK_AND_PIVOTS.md) with independent
   RK4/Euler rigid-body contact dynamics, ellipsoidal fluid air drag, 3D physical gate frames,
   500 Hz CTBR inner-loop control, configurable motor lag tau_m, action delay FIFO, and ZOH dropout.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import mujoco
import numpy as np

from .ctbr_controller import CF2XCTBRConfig, CF2XCTBRController


@dataclass
class DroneCourseGate:
    center: tuple[float, float, float]
    yaw_rad: float = 0.0
    half_width: float = 0.35
    half_height: float = 0.35


DEFAULT_SLALOM_GATES: tuple[DroneCourseGate, ...] = (
    DroneCourseGate((2.0, 0.45, 1.15), yaw_rad=0.18),
    DroneCourseGate((4.2, -0.50, 1.40), yaw_rad=-0.22),
    DroneCourseGate((6.4, 0.50, 0.95), yaw_rad=0.22),
    DroneCourseGate((8.6, 0.00, 1.25), yaw_rad=0.00),
)


def build_cf2x_mjcf(
    gates: tuple[DroneCourseGate, ...] = DEFAULT_SLALOM_GATES,
    mass: float = 0.027,
    fluid_density: float = 1.225,
    fluid_viscosity: float = 1.8e-5,
    integrator: str = "RK4",
    timestep: float = 0.002,
) -> str:
    """Generate a self-contained MuJoCo MJCF XML for CF2X + 3D physical racing gates."""
    d_arm = 0.0397 / math.sqrt(2.0)
    kappa = 7.94e-12 / 3.16e-10  # km / kf = 0.0251266 m

    gate_xml_parts = []
    for idx, g in enumerate(gates):
        cx, cy, cz = g.center
        hw, hh = g.half_width, g.half_height
        bar_r = 0.02
        cyaw = math.cos(0.5 * g.yaw_rad)
        syaw = math.sin(0.5 * g.yaw_rad)
        gate_xml_parts.append(f"""
        <body name="gate_{idx}" pos="{cx:.4f} {cy:.4f} {cz:.4f}" quat="{cyaw:.5f} 0 0 {syaw:.5f}">
            <geom name="gate_{idx}_left" type="box" pos="0 {hw:.4f} 0" size="{bar_r} {bar_r} {hh:.4f}" rgba="0.9 0.25 0.15 1"/>
            <geom name="gate_{idx}_right" type="box" pos="0 {-hw:.4f} 0" size="{bar_r} {bar_r} {hh:.4f}" rgba="0.9 0.25 0.15 1"/>
            <geom name="gate_{idx}_top" type="box" pos="0 0 {hh:.4f}" size="{bar_r} {hw:.4f} {bar_r}" rgba="0.9 0.25 0.15 1"/>
            <geom name="gate_{idx}_bottom" type="box" pos="0 0 {-hh:.4f}" size="{bar_r} {hw:.4f} {bar_r}" rgba="0.9 0.25 0.15 1"/>
        </body>""")

    gates_xml = "\n".join(gate_xml_parts)
    return f"""
<mujoco model="cf2x_gate_racing">
    <option timestep="{timestep}" gravity="0 0 -9.81" density="{fluid_density}" viscosity="{fluid_viscosity}" integrator="{integrator}"/>
    <worldbody>
        <geom name="floor" type="plane" size="20 20 0.1" rgba="0.2 0.22 0.25 1"/>
        <body name="drone" pos="0 0 1.0">
            <freejoint name="root"/>
            <inertial pos="0 0 0" mass="{mass:.6f}" diaginertia="1.4e-5 1.4e-5 2.17e-5"/>
            <geom name="chassis" type="box" size="0.032 0.032 0.012" rgba="0.15 0.55 0.95 1" contype="1" conaffinity="1"/>
            <site name="m0" pos="{d_arm:.5f} {-d_arm:.5f} 0" size="0.005"/>
            <site name="m1" pos="{-d_arm:.5f} {-d_arm:.5f} 0" size="0.005"/>
            <site name="m2" pos="{-d_arm:.5f} {d_arm:.5f} 0" size="0.005"/>
            <site name="m3" pos="{d_arm:.5f} {d_arm:.5f} 0" size="0.005"/>
        </body>
        {gates_xml}
    </worldbody>
    <actuator>
        <general name="rotor0" site="m0" gear="0 0 1 0 0 {-kappa:.6f}" ctrllimited="true" ctrlrange="0 0.20"/>
        <general name="rotor1" site="m1" gear="0 0 1 0 0 {kappa:.6f}"  ctrllimited="true" ctrlrange="0 0.20"/>
        <general name="rotor2" site="m2" gear="0 0 1 0 0 {-kappa:.6f}" ctrllimited="true" ctrlrange="0 0.20"/>
        <general name="rotor3" site="m3" gear="0 0 1 0 0 {kappa:.6f}"  ctrllimited="true" ctrlrange="0 0.20"/>
    </actuator>
</mujoco>
"""


class MuJoCoCF2XRacingEnv:
    """
    100 Hz CTBR Quadrotor Gate-Racing Environment backed by 500 Hz 3D MuJoCo physics.

    Enforces all theoretical guardrails from AGENTS.md:
    - Action history buffer support `a_{t - d_max : t - 1}`
    - Zero-Order Hold (ZOH) observation dropout (never zero-masking!)
    - Gate-relative / body-frame observation features + optional oblique PnP bias `b(phi_rel(s))`
    - Smoothness-regularized Swift progress + perception reward
    """

    def __init__(
        self,
        gates: tuple[DroneCourseGate, ...] = DEFAULT_SLALOM_GATES,
        ctbr_cfg: CF2XCTBRConfig | None = None,
        mass_scale: float = 1.0,
        drag_linear_xy: float = 0.018,
        drag_linear_z: float = 0.025,
        drag_quad: float = 0.0045,
        action_delay_ticks: int = 0,
        obs_dropout_prob: float = 0.0,
        sensor_limited: bool = False,
        pnp_bias_scale: float = 0.0,
        act_history_len: int = 6,
        max_steps: int = 600,
    ):
        self.gates = gates
        self.base_cfg = ctbr_cfg or CF2XCTBRConfig(
            tau_motor=0.025,
            kp_rate_xy=62.0,
            kp_rate_z=30.0,
            kd_rate_xy=0.8,
            gyro_lpf_alpha=0.65,
        )
        self.controller = CF2XCTBRController(self.base_cfg)
        self.true_mass = self.base_cfg.mass * mass_scale
        self.drag_linear = np.array([drag_linear_xy, drag_linear_xy, drag_linear_z], dtype=np.float64)
        self.drag_quad = drag_quad

        xml = build_cf2x_mjcf(gates=gates, mass=self.true_mass, timestep=0.002)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.drone_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")

        self.policy_dt = 0.01
        self.substeps = 5
        self.sub_dt = 0.002
        self.action_delay_ticks = int(action_delay_ticks)
        self.obs_dropout_prob = float(obs_dropout_prob)
        self.sensor_limited = bool(sensor_limited)
        self.pnp_bias_scale = float(pnp_bias_scale)
        self.act_history_len = int(act_history_len)
        self.max_steps = int(max_steps)

        self.rng = np.random.default_rng(0)
        self.action_fifo: deque[np.ndarray] = deque(maxlen=max(1, self.action_delay_ticks + 1))
        self.act_history: deque[np.ndarray] = deque(maxlen=max(1, self.act_history_len))
        self.last_valid_obs: np.ndarray | None = None
        self.gate_idx = 0
        self.step_count = 0
        self.prev_gate_dist = 0.0

    def get_ground_truth_state(self) -> dict[str, np.ndarray]:
        pos = self.data.qpos[0:3].copy()
        quat_wxyz = self.data.qpos[3:7].copy()
        rot = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot, quat_wxyz)
        R_wb = rot.reshape(3, 3)  # body-to-world rotation matrix
        vel_world = self.data.qvel[0:3].copy()
        vel_body = R_wb.T @ vel_world
        omega_body = self.data.qvel[3:6].copy()  # MuJoCo freejoint angular velocity is in body frame
        return {
            "pos": pos,
            "quat_wxyz": quat_wxyz,
            "R_wb": R_wb,
            "vel_world": vel_world,
            "vel_body": vel_body,
            "omega_body": omega_body,
            "motor_rpm": self.controller.motor_rpm.copy(),
        }

    def _build_raw_obs(self) -> np.ndarray:
        st = self.get_ground_truth_state()
        R_wb = st["R_wb"]
        g = self.gates[min(self.gate_idx, len(self.gates) - 1)]
        rel_gate_world = np.asarray(g.center, dtype=np.float64) - st["pos"]
        rel_gate_body = R_wb.T @ rel_gate_world

        if self.sensor_limited:
            # Systematic oblique viewing-angle PnP bias b(phi_rel(s)) + zero-mean sensor noise
            dist = np.linalg.norm(rel_gate_body) + 1e-6
            view_angle = math.acos(np.clip(rel_gate_body[0] / dist, -1.0, 1.0))
            pnp_bias = self.pnp_bias_scale * np.array(
                [0.08 * dist * (view_angle ** 2), 0.05 * math.sin(2.0 * view_angle), -0.04 * view_angle],
                dtype=np.float64,
            )
            rel_gate_obs = rel_gate_body + pnp_bias + self.rng.normal(0.0, 0.015, size=3)
            omega_obs = st["omega_body"] + self.rng.normal(0.0, 0.03, size=3)
            # Sensor-limited mode hides direct clean linear velocity (requiring recurrent filtering / VIO)
            vel_obs = np.zeros(3, dtype=np.float64)
        else:
            rel_gate_obs = rel_gate_body
            omega_obs = st["omega_body"]
            vel_obs = st["vel_body"]

        hist_flat = np.concatenate(list(self.act_history), axis=0)
        return np.concatenate([vel_obs, R_wb.reshape(-1), omega_obs, rel_gate_obs, hist_flat], axis=0).astype(
            np.float32
        )

    def _observe_with_zoh_dropout(self) -> np.ndarray:
        raw = self._build_raw_obs()
        if self.last_valid_obs is None or self.obs_dropout_prob <= 0.0:
            self.last_valid_obs = raw
            return raw.copy()
        dropped = self.rng.random() < self.obs_dropout_prob
        if not dropped:
            self.last_valid_obs = raw
        # Lemma 1.1 Guardrail: Zero-Order Hold on dropout, never zero-masking!
        return self.last_valid_obs.copy()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, 1.0]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)

        self.controller.reset()
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
        self.prev_gate_dist = float(np.linalg.norm(np.asarray(self.gates[0].center) - self.data.qpos[0:3]))
        return self._observe_with_zoh_dropout(), self.get_ground_truth_state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        act_clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        prev_cmd = self.act_history[-1].copy()
        self.action_fifo.append(act_clipped)
        executed_action = self.action_fifo[0]

        pos_before = self.data.qpos[0:3].copy()
        vel_before = self.data.qvel[0:3].copy()

        crashed = False
        for _ in range(self.substeps):
            st = self.get_ground_truth_state()
            rpm = self.controller.step_substep(executed_action, st["omega_body"], self.sub_dt)
            rotor_forces = self.base_cfg.kf * (rpm ** 2)
            self.data.ctrl[:] = rotor_forces

            # Apply explicit body-frame rotor + parasitic quadratic drag
            v_b = st["vel_body"]
            f_drag_b = -self.drag_linear * v_b - self.drag_quad * np.linalg.norm(v_b) * v_b
            self.data.xfrc_applied[self.drone_body_id, 0:3] = st["R_wb"] @ f_drag_b
            mujoco.mj_step(self.model, self.data)

            if self.data.ncon > 0 or self.data.qpos[2] <= 0.05:
                crashed = True
                break

        self.act_history.append(act_clipped)
        self.step_count += 1

        st_after = self.get_ground_truth_state()
        pos_after = st_after["pos"]
        lin_acc_world = (st_after["vel_world"] - vel_before) / self.policy_dt

        # Gate passage check & Swift reward
        g = self.gates[self.gate_idx]
        g_center = np.asarray(g.center, dtype=np.float64)
        dist_now = float(np.linalg.norm(g_center - pos_after))
        prog_reward = 1.0 * (self.prev_gate_dist - dist_now)

        # Camera alignment reward (optical axis = +x body axis)
        cam_axis_w = st_after["R_wb"][:, 0]
        ray = (g_center - pos_after) / max(dist_now, 1e-6)
        delta_cam = math.acos(np.clip(float(np.dot(cam_axis_w, ray)), -1.0, 1.0))
        perc_reward = 0.02 * math.exp(-(delta_cam ** 4))

        # Action smoothness & body-rate penalty (Guardrail #1)
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
        return self._observe_with_zoh_dropout(), float(reward), terminated, truncated, info
