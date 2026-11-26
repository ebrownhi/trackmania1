"""
Track-free velocity observers for sensor-limited racing students (H3', Exp 21).

Why track-free: the Exp 10 students (action-cloning FF/GRU) and a GRU that regresses the full privileged state from
the sensor-limited *racing* observation all fail closed-loop (100% crashes even in the source). Instrumenting the
GRU estimator showed it is accurate on teacher states but collapses closed-loop: it learns a course-position ->
typical-velocity prior from the track-relative inputs (corridor / gate vectors) instead of differencing positions,
and the prior is wrong as soon as the student leaves the teacher's trajectory (same shortcut as Exp 12's gate trap).

Fix: the estimator only sees kinematic, track-free measurements
    [0.1 * R^T (p_obs,t - p_obs,t-1) / dt,  fresh-packet flag,  R^T e_z,  0.25 * w_obs]
and outputs 0.33 * v_B (the teacher's velocity slot). The student policy is the frozen exact-state teacher acting on
[observed geometry, estimated velocity, observed attitude/rates, action history].

Students:
    KinGRU    recurrent estimator                                      (proposal: recurrent sensor-limited student)
    KinFF     MLP on the last K kinematic frames                        (recurrence ablation)
    AlphaBeta classical alpha-beta filter on observed position          (hand-designed observer baseline)
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from asym_policy import N_PRIV, PrivStackRacingEnv
from racing_env import SENSOR_BASE_DIM
from sims.drone_pair import RL_DT

KIN_DIM = 10
K_FF = 8


def kin_feat(obs_state, last_p):
    fresh = last_p is None or not np.array_equal(obs_state.p, last_p)
    dp = np.zeros(3) if last_p is None else obs_state.R.T @ (obs_state.p - last_p) / RL_DT
    f = np.concatenate([0.1 * dp, [float(fresh)], obs_state.R.T[:, 2], 0.25 * obs_state.w]).astype(np.float32)
    return f, obs_state.p.copy()


class KinGRU(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(KIN_DIM, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 3)

    def forward_seq(self, x):
        out, _ = self.gru(self.inp(x))
        return self.head(out)


class KinFF(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(KIN_DIM * K_FF, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 3))

    def forward_seq(self, x):  # x: (B, T, KIN_DIM) -> causal K-frame stacks, first frame repeated as padding
        pad = x[:, :1].expand(-1, K_FF - 1, -1)
        xp = torch.cat([pad, x], dim=1)
        stacks = torch.stack([xp[:, i:i + x.shape[1]] for i in range(K_FF)][::-1], dim=2)  # newest first
        return self.net(stacks.flatten(2))


def teacher_input(obs: np.ndarray, v_slot: np.ndarray) -> np.ndarray:
    """Exact-state teacher obs from a PrivStackRacingEnv obs with the velocity slot replaced by `v_slot`."""
    base = np.concatenate([obs[:15], v_slot, obs[15:SENSOR_BASE_DIM]])
    return np.concatenate([base, obs[SENSOR_BASE_DIM:SENSOR_BASE_DIM + 32]]).astype(np.float32)


class KinStudent:
    """Frozen teacher + learned track-free velocity estimator (evaluate_racing policy interface)."""

    def __init__(self, est: nn.Module, teacher):
        self.est, self.teacher = est, teacher
        self.recurrent = isinstance(est, KinGRU)

    def reset_episode(self):
        self.h, self.lp, self.stack = None, None, None

    @torch.no_grad()
    def estimate(self, env) -> np.ndarray:
        f, self.lp = kin_feat(env.obs_state, self.lp)
        self.last_feat = f  # the exact input consumed, kept for DAgger training data
        if self.recurrent:
            out, self.h = self.est.gru(self.est.inp(torch.as_tensor(f).view(1, 1, -1)), self.h)
            return self.est.head(out[0, 0]).numpy()
        if self.stack is None:
            self.stack = deque([f] * K_FF, maxlen=K_FF)
        self.stack.append(f)
        return self.est.net(torch.as_tensor(np.concatenate(list(self.stack)[::-1]))).numpy()

    def __call__(self, obs, env, step_idx):
        return self.teacher.predict(teacher_input(obs, self.estimate(env)), deterministic=True)[0]


class AlphaBeta:
    """Frozen teacher + alpha-beta filter on observed position (skips repeated / dropped packets)."""

    def __init__(self, teacher, alpha: float = 0.5, beta: float = 0.1):
        self.teacher, self.a, self.b = teacher, alpha, beta

    def reset_episode(self):
        self.p, self.last = None, None

    def __call__(self, obs, env, step_idx):
        st = env.obs_state
        fresh = self.last is None or not np.array_equal(st.p, self.last)
        self.last = st.p.copy()
        if self.p is None:
            self.p, self.v = st.p.copy(), np.zeros(3)
        pp = self.p + self.v * RL_DT
        if fresh:
            r = st.p - pp
            self.p, self.v = pp + self.a * r, self.v + self.b / RL_DT * r
        else:
            self.p = pp
        return self.teacher.predict(teacher_input(obs, 0.33 * (st.R.T @ self.v)), deterministic=True)[0]


def distill_kin(teacher, recurrent: bool, obs_delay: int, p_drop: float, seed: int, dyn_sampler=None,
                delays=(0,), iters: int = 6, eps: int = 10, steps: int = 800) -> nn.Module:
    """On-policy DAgger for the velocity estimator: label = true 0.33 v_B, executed action = teacher (prob beta)
    or the student (teacher on the estimate)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    est = KinGRU() if recurrent else KinFF()
    opt = torch.optim.Adam(est.parameters(), lr=2e-3)
    env = PrivStackRacingEnv("figure8", "source", 8, obs_delay=obs_delay, p_drop=p_drop, delays=delays,
                             dyn_sampler=dyn_sampler, max_ep_s=6.0, seed=10_000 + seed)
    pol = KinStudent(est, teacher)
    SX, SY = [], []
    for it in range(iters):
        beta = 1.0 if it == 0 else max(0.0, 0.5 - 0.1 * it)
        for ep in range(eps):
            o, _ = env.reset(seed=10_000 * seed + 100 * it + ep)
            pol.reset_episode()
            X, Y = [], []
            while True:
                Y.append(o[-N_PRIV:][15:18].copy())
                v_hat = pol.estimate(env)
                X.append(pol.last_feat)
                if it == 0 or rng.random() < beta:
                    a = teacher.predict(teacher_input(o, o[-N_PRIV:][15:18]), deterministic=True)[0]
                else:
                    a = teacher.predict(teacher_input(o, v_hat), deterministic=True)[0]
                o, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
            if len(X) >= 40:
                SX.append(np.array(X, np.float32))
                SY.append(np.array(Y, np.float32))
        W, B, BURN = 80, 32, 20
        for _ in range(steps):
            xb, yb = [], []
            for _ in range(B):
                i = int(rng.integers(len(SX)))
                s0 = int(rng.integers(0, max(1, len(SX[i]) - W)))
                xs, ys = SX[i][s0:s0 + W], SY[i][s0:s0 + W]
                if len(xs) < W:  # short (crashed) episode: edge-pad to the window length
                    pad = ((0, W - len(xs)), (0, 0))
                    xs, ys = np.pad(xs, pad, mode="edge"), np.pad(ys, pad, mode="edge")
                xb.append(xs)
                yb.append(ys)
            xb, yb = torch.as_tensor(np.array(xb)), torch.as_tensor(np.array(yb))
            loss = ((est.forward_seq(xb) - yb)[:, BURN:] ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(est.parameters(), 2.0)
            opt.step()
    return est

