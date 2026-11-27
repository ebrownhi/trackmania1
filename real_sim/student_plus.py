"""
Exp 23 student upgrades on top of the Exp 21 track-free observer (kin_observer.py).

Diagnosis (Exp 23 part A): with the target's PnP gate bias switched off, the Exp 21 `kin_gru_rand` student already
completes the held-out course in PyBullet at every sensor level, so the *target* gap is the bias; in the L2F *source*
the bottleneck is the velocity estimate (out-of-distribution body-z velocity on the 3D course) and, at the hard sensor
level, the 2-tick-stale geometry. Fixes:

    F1  wider estimator training data -> `kin_observer.distill_kin(courses=track_pool())` (multi-track; exploration
        noise `act_noise` was tried first and did not help)
    F2  delay compensation: predict the observed pose forward by its staleness (sensor delay + ticks since the
        last fresh packet) using the estimated velocity and measured body rates (Smith-predictor style)
    F3  PnP-bias GP: b(r_g) fit on a few target calibration flights with ground-truth (mocap) inputs, then the
        observed position is corrected (p_true ≈ p_obs + R b) before the geometry is rebuilt

`PlusStudent(est, teacher, obs_delay, delay_comp, gp_hook)` covers every ablation (F2 / F3 off = Exp 21 behaviour
on the same estimator).
"""

from __future__ import annotations

import numpy as np
import torch

from kin_observer import KinStudent, kin_feat, teacher_input
from racing_env import build_racing_obs
from sims.drone_pair import RL_DT, DroneState

# Estimator-training track pool: figure8 + Exp 18 procedural circuits, excluding the held-out ellipse3d and its two
# near-duplicates (saddle_3d, banked_oval_3d: xy ellipses with a vertical wave).
TRAIN_TRACKS = ["figure8", "split_s_3d", "corkscrew_3d", "tilted_hairpin", "diving_chicane", "ascending_esses", "trefoil_3d"]


def track_pool(kinds=TRAIN_TRACKS):
    from exp18_bootstrapped_swift_and_10tracks import Procedural3DCourse
    from racing_env import FastCenterline
    from sims.drone_pair import Course
    out = []
    for k in kinds:
        c = Course(kind="figure8", gate_radius=0.25) if k == "figure8" else Procedural3DCourse(k)
        out.append((c, FastCenterline(c)))
    return out


def so3_exp(phi: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(phi))
    K = np.array([[0.0, -phi[2], phi[1]], [phi[2], 0.0, -phi[0]], [-phi[1], phi[0], 0.0]])
    if th < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1.0 - np.cos(th)) / th ** 2 * (K @ K)


class PlusStudent(KinStudent):
    def __init__(self, est, teacher, obs_delay: int = 0, delay_comp: bool = False, gp_hook=None):
        super().__init__(est, teacher)
        self.obs_delay, self.delay_comp, self.gp_hook = obs_delay, delay_comp, gp_hook

    def reset_episode(self):
        super().reset_episode()
        self.stale, self.hint, self.last_raw = 0, None, None

    def corrected_state(self, env) -> DroneState:
        """Observed state with the PnP bias removed (F3); position only, as the bias only shifts position."""
        st = env.obs_state.copy()
        if self.gp_hook is not None:
            g0 = env.gates[env.active_gate_idx()]
            b = self.gp_hook(st.R.T @ (g0 - st.p))
            st.p = st.p + st.R @ b
        return st

    @torch.no_grad()
    def __call__(self, obs, env, step_idx):
        raw = env.obs_state.p
        fresh = self.last_raw is None or not np.array_equal(raw, self.last_raw)
        self.last_raw = raw.copy()
        self.stale = 0 if fresh else self.stale + 1
        st = self.corrected_state(env)
        # velocity estimate from the bias-corrected kinematics (falls back to Exp 21 features when no GP)
        f, self.lp = kin_feat(st, self.lp) if fresh or self.lp is None else (self._repeat_feat(st), self.lp)
        self.last_feat = f
        v_hat = self._step_est(f)
        if self.hint is None:
            self.hint = env.obs_hint
        if self.delay_comp:
            s = (self.obs_delay + self.stale) * RL_DT
            st = DroneState(st.p + st.R @ (v_hat / 0.33) * s, st.R @ so3_exp(st.w * s), st.v, st.w)
        if self.delay_comp or self.gp_hook is not None:
            o_s, self.hint = build_racing_obs(st, env.cl, env.active_gate_idx(), self.hint, env.ah, env.hist,
                                              sensor_limited=True)
        else:
            o_s = obs
        return self.teacher.predict(teacher_input(o_s, v_hat), deterministic=True)[0]

    def _repeat_feat(self, st):
        """Dropped packet: same feature the Exp 21 estimator saw (zero displacement, fresh flag 0)."""
        f, _ = kin_feat(st, st.p)
        return f

    def _step_est(self, f):
        if self.recurrent:
            out, self.h = self.est.gru(self.est.inp(torch.as_tensor(f).view(1, 1, -1)), self.h)
            return self.est.head(out[0, 0]).numpy()
        from collections import deque
        from kin_observer import K_FF
        if self.stack is None:
            self.stack = deque([f] * K_FF, maxlen=K_FF)
        self.stack.append(f)
        return self.est.net(torch.as_tensor(np.concatenate(list(self.stack)[::-1]))).numpy()


def calibrate_bias(policy, env, n_flights: int = 3, seed: int = 0, n_fit: int = 450):
    """Fly `policy` in the target `env` for `n_flights` episodes; on fresh packets record ground-truth (mocap)
    inputs r_g = R_y^T (g - p_y) of the *true* measured pose y (the delayed state the sensor imaged) and labels
    b = R_y^T (p_y - p_obs) (bias + zero-mean noise). Fit an RBF + White GP on ground-truth inputs (Guardrail:
    observed-pose inputs would leak the noise into the regressor). Returns (fast hook, diagnostics)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from exp10_method_matrix import make_gp_gate_hook
    from racing_env import evaluate_racing  # noqa: F401  (policy interface)

    rng = np.random.default_rng(seed)
    X, Y = [], []
    for ep in range(n_flights):
        obs, _ = env.reset(seed=70_000 + 100 * seed + ep)
        if hasattr(policy, "reset_episode"):
            policy.reset_episode()
        while True:
            obs, _, term, trunc, _ = env.step(policy(obs, env, 0))
            if not env.drone.dropped:
                y = env.drone.obs_fifo[0]  # privileged mocap of the imaged pose (calibration only)
                g0 = env.gates[env.active_gate_idx()]
                X.append(y.R.T @ (g0 - y.p))
                Y.append(y.R.T @ (y.p - env.obs_state.p))
            if term or trunc:
                break
    X, Y = np.array(X), np.array(Y)
    idx = rng.choice(len(X), size=min(n_fit, len(X)), replace=False)
    gp = GaussianProcessRegressor(RBF(0.6) + WhiteKernel(1e-3), normalize_y=False, random_state=0).fit(X[idx], Y[idx])
    held = np.setdiff1d(np.arange(len(X)), idx)
    resid = Y[held] - gp.predict(X[held]) if len(held) else Y[:0]
    diag = {"n_samples": int(len(X)), "label_rms_m": float(np.sqrt(np.mean(np.sum(Y ** 2, 1)))),
            "heldout_resid_rms_m": float(np.sqrt(np.mean(np.sum(resid ** 2, 1)))) if len(resid) else float("nan")}
    return make_gp_gate_hook(gp), diag
