"""
Unified 6-Gate 3D Drone Racing Environment & Proposal Metrics Suite for P3 (16-831 Final Project).

Supports:
  - Both `source` (`rl-tools/l2f`) and `target` (`gym-pybullet-drones` CF2X, nominal or perturbed)
  - Training course (`figure8`, 6 gates) and zero-shot held-out course (`ellipse3d`, 6 gates with 3D altitude waves)
  - SE(3)-invariant body-frame spatial corridor + active-gate observations (Guardrail #6 — zero world coordinates,
    zero reference clock `t`)
  - Exact-state AND sensor-limited observations (`v_B` hidden, `obs_delay`, ZOH packet dropout `p_drop`, gyro/pos noise,
    and SE(3)-invariant gate-relative oblique PnP perception bias `b(rel_g)`)
  - All Proposal Section 3 metrics: completion, clean_completion (collision-free), gate_frac, median_lap_s,
    p90_lap_s, track_err_m, gate_err_m, crash_rate, smoothness, and target-to-source performance ratio.
"""

from __future__ import annotations

from collections import deque
import contextlib
import io
import math
import gymnasium as gym
import numpy as np

import common  # noqa: F401
from sims.drone_pair import (
    Course,
    DroneState,
    GRAVITY,
    GeometricExpert,
    InterfaceConfig,
    RL_DT,
    make_source,
    make_target,
)

EXACT_BASE_DIM = 24
SENSOR_BASE_DIM = 21


def equilibrium_R(a_world: np.ndarray) -> np.ndarray:
    """Equilibrium body attitude R whose thrust axis aligns with a_world + g e_z (zero yaw)."""
    a_tot = np.asarray(a_world, dtype=np.float64) + np.array([0.0, 0.0, GRAVITY])
    z_des = a_tot / max(np.linalg.norm(a_tot), 1e-6)
    x_c = np.array([1.0, 0.0, 0.0])
    y_des = np.cross(z_des, x_c)
    y_des /= max(np.linalg.norm(y_des), 1e-6)
    x_des = np.cross(y_des, z_des)
    return np.column_stack([x_des, y_des, z_des])


class FastCenterline:
    """Time-free spatial centerline sampled from a 6-gate Course for local projection & arc-length progress."""

    def __init__(self, course: Course, n_pts: int = 250):
        self.course = course
        ts = np.linspace(0.0, course.period, n_pts, endpoint=False)
        refs = [course.ref(float(t)) for t in ts]
        self.pts = np.array([r[0] for r in refs], dtype=np.float64)
        self.vels = np.array([r[1] for r in refs], dtype=np.float64)
        self.accs = np.array([r[2] for r in refs], dtype=np.float64)
        self.n_pts = n_pts
        diffs = np.linalg.norm(np.roll(self.pts, -1, axis=0) - self.pts, axis=1)
        self.loop_len = float(np.sum(diffs))
        self.cum_s = np.concatenate([[0.0], np.cumsum(diffs)[:-1]])
        self.ds_mean = self.loop_len / n_pts
        self.gates = np.array([course.ref(tg)[0] for tg in course.gate_times()], dtype=np.float64)
        self.tangents = np.array(
            [course.ref(tg)[1] / max(np.linalg.norm(course.ref(tg)[1]), 1e-6) for tg in course.gate_times()],
            dtype=np.float64,
        )
        self._win = np.arange(-15, 45)
        self.lk1_di = max(1, int(round(0.50 / self.ds_mean)))
        self.lk2_di = max(2, int(round(1.20 / self.ds_mean)))

    def project(self, p: np.ndarray, hint_idx: int) -> tuple[int, float, float]:
        """Project 3D point `p` onto local track window around `hint_idx` -> `(idx, arc_s_m, cross_err_m)`."""
        idxs = (hint_idx + self._win) % self.n_pts
        diff = self.pts[idxs] - p
        d2 = diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2
        j = int(np.argmin(d2))
        idx = int(idxs[j])
        return idx, float(self.cum_s[idx]), float(math.sqrt(d2[j]))


def oblique_pnp_gate_bias(rel_gate_body: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Systematic gate-relative PnP position bias b(r_g) in the drone body frame [m] (Guardrail #6).
    Depends solely on SE(3)-invariant relative gate coordinates r_g = R^T (g_active - p_true),
    so a GP fit on `figure8` generalizes zero-shot to held-out `ellipse3d`.
    """
    if scale <= 0.0:
        return np.zeros(3, dtype=np.float64)
    dist = float(np.linalg.norm(rel_gate_body)) + 1e-6
    cos_ang = float(np.clip(rel_gate_body[0] / dist, -1.0, 1.0))
    ang = math.acos(cos_ang)
    return scale * np.array(
        [
            0.055 * dist * (ang ** 2) + 0.03,
            0.075 * math.sin(2.0 * ang) + 0.035 * np.clip(rel_gate_body[1], -2.0, 2.0),
            -0.050 * dist * math.sin(ang) + 0.045,
        ],
        dtype=np.float64,
    )


def build_racing_obs(
    obs_state: DroneState,
    cl: FastCenterline,
    gate_idx: int,
    obs_hint: int,
    act_hist: deque,
    hist_len: int,
    sensor_limited: bool = False,
) -> tuple[np.ndarray, int]:
    """
    SE(3)-invariant body-frame actor observation (Guardrail #6 — zero world coordinates, zero clock `t`):
      - Exact-state (24 + 4*hist_len):
          [rel_near (3), rel_lk1 (3), rel_lk2 (3), 0.5*rel_g0 (3), g_tan0_B (3),
           0.33*v_B (3), Rt[:,2] (3), 0.25*w_B (3), act_hist (4*hist_len)]
      - Sensor-limited (21 + 4*hist_len, linear velocity `v_B` omitted; `obs_state` already carries
        `obs_delay`, ZOH packet dropout, position/gyro noise, and oblique PnP perception bias):
          [rel_near (3), rel_lk1 (3), rel_lk2 (3), 0.5*rel_g0 (3), g_tan0_B (3),
           Rt[:,2] (3), 0.25*w_B (3), act_hist (4*hist_len)]
    """
    Rt = obs_state.R.T
    idx, _, _ = cl.project(obs_state.p, obs_hint)
    p_near = cl.pts[idx]
    p_lk1 = cl.pts[(idx + cl.lk1_di) % cl.n_pts]
    p_lk2 = cl.pts[(idx + cl.lk2_di) % cl.n_pts]
    gi = gate_idx % len(cl.gates)
    g0 = cl.gates[gi]
    t0 = cl.tangents[gi]

    spatial = [
        Rt @ (p_near - obs_state.p),
        Rt @ (p_lk1 - obs_state.p),
        Rt @ (p_lk2 - obs_state.p),
        0.5 * (Rt @ (g0 - obs_state.p)),
        Rt @ t0,
    ]
    if sensor_limited:
        core = spatial + [Rt[:, 2], 0.25 * obs_state.w]
    else:
        core = spatial + [0.33 * (Rt @ obs_state.v), Rt[:, 2], 0.25 * obs_state.w]

    past = list(act_hist)[-hist_len:][::-1] if hist_len > 0 else []
    return np.concatenate(core + past, axis=0).astype(np.float32), idx


class GateRacingEnv(gym.Env):
    """Spatial 6-gate 3D drone racing Gymnasium environment over `L2FBackend` or `PyBulletBackend`."""

    def __init__(
        self,
        course_kind: str = "figure8",
        domain: str = "source",
        hist: int = 8,
        lam_da: float = 0.02,
        delays: tuple[int, ...] = (0,),
        dyn_sampler=None,
        sensor_limited: bool = False,
        obs_delay: int = 0,
        p_drop: float = 0.0,
        pnp_bias_scale: float = 0.0,
        gp_gate_bias_fn=None,
        residual_fn=None,
        tau_hat: float | None = None,
        kf_scale: float = 1.0,
        tau_m_target: float = 0.03,
        n_laps: int = 2,
        max_ep_s: float = 12.5,
        gate_pass_radius: float = 0.35,
        clean_pass_radius: float = 0.25,
        random_start_gate: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.course = Course(kind=course_kind, gate_radius=clean_pass_radius)
        self.cl = FastCenterline(self.course)
        self.gates = self.cl.gates
        self.tangents = self.cl.tangents
        self.n_gates = len(self.gates)
        self.total_target_gates = self.n_gates * n_laps
        self.n_laps = n_laps

        self.domain = domain
        self.hist = hist
        self.lam_da = lam_da
        self.delays = delays
        self.dyn_sampler = dyn_sampler
        self.sensor_limited = sensor_limited
        self.obs_delay = obs_delay
        self.p_drop = p_drop
        self.pnp_bias_scale = pnp_bias_scale
        self.gp_gate_bias_fn = gp_gate_bias_fn
        self.residual_fn = residual_fn
        self.tau_hat = tau_hat
        self.kf_scale = kf_scale
        self.tau_m_target = tau_m_target
        self.max_steps = int(round(max_ep_s / RL_DT))
        self.gate_pass_radius = gate_pass_radius
        self.clean_pass_radius = clean_pass_radius
        self.random_start_gate = random_start_gate

        base_dim = SENSOR_BASE_DIM if sensor_limited else EXACT_BASE_DIM
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (base_dim + 4 * hist,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self.drones = {}
        self.start_gate_offset = 0
        self.gate_cursor = 0

    def active_gate_idx(self) -> int:
        return (self.start_gate_offset + self.gate_cursor) % self.n_gates

    def _obs_bias_world(self, st: DroneState) -> np.ndarray:
        """
        Convert body-frame gate-relative perception bias b_body(rel_g0) into the equivalent perceived
        world-position offset Delta_p_world = -R * b_body so that R^T (target - p_obs) = R^T (target - p_true) + b_body!
        """
        g0 = self.gates[self.active_gate_idx()]
        rel_g0 = st.R.T @ (g0 - st.p)
        b_body = np.zeros(3, dtype=np.float64)
        if self.pnp_bias_scale > 0.0:
            b_body = b_body + oblique_pnp_gate_bias(rel_g0, self.pnp_bias_scale)
        if self.gp_gate_bias_fn is not None:
            b_body = b_body + self.gp_gate_bias_fn(rel_g0)
        return -(st.R @ b_body)

    def _get_drone(self, d: int):
        key = (d, self.obs_delay, self.p_drop, self.sensor_limited)
        if key not in self.drones:
            use_bias = self.sensor_limited and (self.pnp_bias_scale > 0.0 or self.gp_gate_bias_fn is not None)
            obs_noise = {"p": 0.02, "w": 0.04} if self.sensor_limited else {}
            iface = InterfaceConfig(
                action_delay=d,
                obs_delay=self.obs_delay if self.sensor_limited else 0,
                p_drop=self.p_drop if self.sensor_limited else 0.0,
                dropout_mode="zoh",
                obs_noise=obs_noise,
                obs_bias_fn=self._obs_bias_world if use_bias else None,
            )
            if self.domain == "target":
                with contextlib.redirect_stdout(io.StringIO()):
                    dr = make_target(tau_m=self.tau_m_target, iface=iface, seed=self._seed)
                dr.backend.env.KF *= self.kf_scale
            else:
                ov = {}
                if self.tau_hat is not None:
                    ov = {"tau_rise_scale": self.tau_hat / 0.055455, "tau_fall_scale": self.tau_hat / 0.249394}
                dr = make_source(seed=self._seed, overrides=ov, iface=iface)
                dr.backend.residual_fn = self.residual_fn
            self.drones[key] = dr
        else:
            dr = self.drones[key]
            use_bias = self.sensor_limited and (self.pnp_bias_scale > 0.0 or self.gp_gate_bias_fn is not None)
            dr.iface.obs_bias_fn = self._obs_bias_world if use_bias else None
        return self.drones[key]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        self.gate_cursor = 0
        if self.random_start_gate:
            g_start = int(r.integers(0, self.n_gates))
            t0 = (g_start * self.course.period / self.n_gates) % self.course.period
            p0, v0, a0 = self.course.ref(t0)
            dp = np.array([r.normal(0, 0.05), r.normal(0, 0.05), r.normal(0, 0.10)])
            st = DroneState(p0 + dp, equilibrium_R(a0), v0 + r.normal(0, 0.12, 3), r.normal(0, 0.12, 3))
            self.start_gate_offset = g_start
            self.true_hint = int(round((t0 / self.course.period) * self.cl.n_pts)) % self.cl.n_pts
        else:
            p0, v0, a0 = self.course.ref(0.0)
            st = DroneState(p0 + r.normal(0, 0.025, 3), equilibrium_R(a0), v0 + r.normal(0, 0.06, 3), np.zeros(3))
            self.start_gate_offset = 0
            self.true_hint = 0

        d = int(r.choice(self.delays))
        self.drone = self._get_drone(d)
        if self.dyn_sampler is not None and self.domain == "source":
            self.drone.backend.set_dynamics(self.dyn_sampler(r))

        self.obs_hint = self.true_hint
        self.obs_state = self.drone.reset(st)
        self.ah = deque(
            [self.drone.hover_action.astype(np.float64).copy()] * max(1, self.hist),
            maxlen=max(1, self.hist),
        )
        self.gates_passed = 0
        self.clean_gates_passed = 0
        self.frame_grazes = 0
        self.lap_gates_passed = 0
        self.gate_errors = []
        self.track_errors = []
        self.lap_times = []
        self.lap_start_step = 0
        self.k = 0
        _, self.prev_s, _ = self.cl.project(st.p, self.true_hint)
        obs, self.obs_hint = build_racing_obs(
            self.obs_state, self.cl, self.active_gate_idx(), self.obs_hint, self.ah, self.hist, self.sensor_limited
        )
        return obs, {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        prev_a = self.ah[-1]
        p_prev = self.drone.true_state().p.copy()

        self.obs_state = self.drone.step(a)
        s = self.drone.true_state()
        self.ah.append(a)
        self.k += 1

        self.true_hint, cur_s, cross_err = self.cl.project(s.p, self.true_hint)
        self.track_errors.append(cross_err)
        ds = cur_s - self.prev_s
        if ds < -0.5 * self.cl.loop_len:
            ds += self.cl.loop_len
        elif ds > 0.5 * self.cl.loop_len:
            ds -= self.cl.loop_len
        self.prev_s = cur_s

        da = a - prev_a
        reward = 1.0 - 0.8 * cross_err + 4.0 * ds - 0.002 * float(s.w @ s.w) - self.lam_da * float(da @ da)

        # Gate plane crossing check (advances gate_cursor cleanly without false crash on slightly wide turns)
        gi = self.active_gate_idx()
        g_center = self.gates[gi]
        g_tan = self.tangents[gi]
        s_before = float((p_prev - g_center) @ g_tan)
        s_after = float((s.p - g_center) @ g_tan)
        if s_before < 0.0 <= s_after and float(np.linalg.norm(s.p - g_center)) < 1.4:
            denom = abs(s_before) + abs(s_after)
            frac = abs(s_before) / denom if denom > 1e-6 else 1.0
            p_cross = p_prev + frac * (s.p - p_prev)
            radial_err = float(np.linalg.norm(p_cross - g_center))
            self.gate_errors.append(radial_err)
            self.gate_cursor += 1
            if radial_err <= self.gate_pass_radius:
                self.gates_passed += 1
                self.lap_gates_passed += 1
                if radial_err <= self.clean_pass_radius:
                    self.clean_gates_passed += 1
                    reward += 3.0
                else:
                    self.frame_grazes += 1
                    reward += 1.5
            else:
                reward -= 1.5

            if self.gate_cursor % self.n_gates == 0:
                if self.lap_gates_passed == self.n_gates:
                    self.lap_times.append((self.k - self.lap_start_step) * RL_DT)
                self.lap_start_step = self.k
                self.lap_gates_passed = 0

        crashed = bool(cross_err > 1.6 or s.p[2] < 0.08 or s.R[2, 2] < 0.0 or not np.all(np.isfinite(s.p)))
        done_course = self.gate_cursor >= self.total_target_gates
        completed = bool((not crashed) and (self.gates_passed >= self.total_target_gates))
        if crashed:
            reward -= 40.0
        elif completed:
            reward += 5.0

        obs_out, self.obs_hint = build_racing_obs(
            self.obs_state, self.cl, self.active_gate_idx(), self.obs_hint, self.ah, self.hist, self.sensor_limited
        )
        info = {
            "state": s,
            "a": a,
            "da_norm": float(np.linalg.norm(da)),
            "gates_passed": self.gates_passed,
            "clean_gates_passed": self.clean_gates_passed,
            "completed": completed,
            "collision_free": bool(completed and self.frame_grazes == 0),
            "crashed": crashed,
            "lap_times": list(self.lap_times),
            "gate_errors": list(self.gate_errors),
            "track_err": float(np.mean(self.track_errors)) if self.track_errors else 0.0,
        }
        return obs_out, float(reward), bool(crashed or done_course), self.k >= self.max_steps, info


class GeometricGateController:
    """
    Time-free spatial geometric controller (Proposal Section 3 classical baseline).
    Projects observed drone pose onto the spatial track centerline (zero reference clock `t`)
    and commands curvature-feedforward geometric tracking.
    """

    def __init__(self, kp: float = 6.5, kd: float = 4.0, lookahead_pts: int = 2):
        self.ex = GeometricExpert(kp=kp, kd=kd, k_att=7.0)
        self.lookahead_pts = lookahead_pts
        self._hint = 0

    def reset_episode(self):
        self._hint = 0

    def __call__(self, obs: np.ndarray, env: GateRacingEnv, step_idx: int) -> np.ndarray:
        if step_idx == 0:
            self._hint = env.obs_hint
        obs_st = env.obs_state
        idx, _, _ = env.cl.project(obs_st.p, self._hint)
        self._hint = idx
        j = (idx + self.lookahead_pts) % env.cl.n_pts
        return self.ex(obs_st, env.cl.pts[j], env.cl.vels[j], env.cl.accs[j])


def evaluate_racing(policy_fn, env_or_fn, n_eval: int = 10, seed0: int = 30_000) -> dict:
    """
    Evaluate `policy_fn(obs, env, ep_step) -> action` across `n_eval` episodes and compute all
    Proposal Section 3 metrics. Accepts either an existing `GateRacingEnv` or a builder `make_env_fn()`.
    """
    env = env_or_fn() if callable(env_or_fn) else env_or_fn
    completions, clean_completions, crashes = [], [], []
    gate_fracs, clean_gate_fracs, gate_errs, track_errs, smooths, all_lap_times, ep_returns = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for ep in range(n_eval):
        obs, _ = env.reset(seed=seed0 + ep)
        if hasattr(policy_fn, "reset_episode"):
            policy_fn.reset_episode()
        das, ret = [], 0.0
        step_idx = 0
        info = {}
        while True:
            a = policy_fn(obs, env, step_idx)
            obs, r, term, trunc, info = env.step(a)
            ret += r
            das.append(info["da_norm"])
            step_idx += 1
            if term or trunc:
                break

        completions.append(float(info["completed"]))
        clean_completions.append(float(info["collision_free"]))
        crashes.append(float(info["crashed"]))
        gate_fracs.append(info["gates_passed"] / env.total_target_gates)
        clean_gate_fracs.append(info["clean_gates_passed"] / env.total_target_gates)
        track_errs.append(info["track_err"])
        if info["gate_errors"]:
            gate_errs.extend(info["gate_errors"])
        if info["lap_times"]:
            all_lap_times.extend(info["lap_times"])
        smooths.append(float(np.mean(das)) if das else 0.0)
        ep_returns.append(ret)

    return {
        "completion": float(np.mean(completions)),
        "clean_completion": float(np.mean(clean_completions)),
        "gate_frac": float(np.mean(gate_fracs)),
        "clean_gate_frac": float(np.mean(clean_gate_fracs)),
        "median_lap_s": float(np.median(all_lap_times)) if all_lap_times else float("nan"),
        "p90_lap_s": float(np.percentile(all_lap_times, 90)) if all_lap_times else float("nan"),
        "track_err_m": float(np.mean(track_errs)),
        "gate_err_m": float(np.mean(gate_errs)) if gate_errs else float("nan"),
        "crash_rate": float(np.mean(crashes)),
        "smoothness": float(np.mean(smooths)),
        "return": float(np.mean(ep_returns)),
    }
