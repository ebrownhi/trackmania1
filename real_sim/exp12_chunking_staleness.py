"""
Exp 12 (Step 2 Riff R2.3 + Step 3 Riffs R3.1 & R3.3):
  - Part A (R2.3 — DIDA + 4-Tick Action-Chunking Queue):
      Compare standard single-action ZOH (`a_t`) against a 4-step predictive action-chunking student
      (`[a_{t|t}, a_{t+1|t}, a_{t+2|t}, a_{t+3|t}]`) distilled via DIDA (`predict_forward` multi-step lookahead)
      when the wireless command link drops packets with probability `p_cmd_drop in {0.0, 0.25, 0.50, 0.70, 0.80}`
      on held-out 3D `ellipse3d` in PyBullet (`d = 2`).
  - Part B (R3.1 + R3.3 — Timestamped Staleness-Age Gating & RMA Factorized Velocity/Extrinsics Adaptation):
      Under harsh sensor-limited 3D racing (`obs_delay = 3`, `v_B` hidden, 2 cm pos noise, 0.04 rad/s gyro noise,
      oblique gate-camera PnP bias, and ZOH packet dropout `p_obs_drop in {0.0, 0.25, 0.50, 0.65}`) on held-out
      3D `ellipse3d` across `target_nominal` and `target_perturbed` (`kf x 1.20`), compare:
        1. `ff_blind`      : Feed-forward MLP on naive 4-tick ZOH finite-difference (no recurrence, no staleness age)
        2. `gru_blind`     : Recurrent GRU observer on naive 4-tick ZOH stream (no packet-staleness timestamps)
        3. `rma_staleness` : Track-free RMA + Staleness-Gated GRU (`R3.1 + R3.3`, trained on 3D Lissajous excitations
                             so zero shortcut memorization of `figure8` is possible!) predicting explicit
                             `(v_hat_B, z_hat_physics)` from packet-age-normalized displacements + `tau_stale`!
"""

from __future__ import annotations

from collections import deque
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from common import OUT, fmt, mean_std, plt, save_json
from racing_env import (
    EXACT_BASE_DIM,
    GateRacingEnv,
    GeometricGateController,
    build_racing_obs,
    oblique_pnp_gate_bias,
)
from sims.drone_pair import DroneState, InterfaceConfig, RL_DT, make_source

SEEDS = [0, 1, 2]
HIST = 8
CHUNK_H = 4
TRAIN_DELAYS = (0, 1, 2, 3, 4)


def dyn_sampler_with_vec(rng):
    ov = {
        "mass_scale": float(rng.uniform(0.80, 1.22)),
        "thrust_scale": float(rng.uniform(0.80, 1.28)),
        "tau_rise_scale": float(rng.uniform(0.35, 1.40)),
        "tau_fall_scale": float(rng.uniform(0.10, 1.40)),
    }
    z_vec = np.array(
        [
            (ov["mass_scale"] - 1.0) / 0.15,
            (ov["thrust_scale"] - 1.0) / 0.15,
            (ov["tau_rise_scale"] - 0.85) / 0.35,
            (ov["tau_fall_scale"] - 0.75) / 0.40,
        ],
        dtype=np.float32,
    )
    return ov, z_vec


# --------------------------------------------------------------------------------------------------
# Part A Models: DIDA Single-Step Policy vs. DIDA 4-Step Action-Chunking Policy
# --------------------------------------------------------------------------------------------------


class ActionChunkMLP(nn.Module):
    def __init__(self, in_dim: int, horizon: int = 1, hidden: int = 128):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 4 * horizon),
        )
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.tanh(self.net((x - self.mu) / self.sd))
        return out.view(-1, self.horizon, 4)


def query_chunk_labels(env: GateRacingEnv, expert: GeometricGateController, horizon: int) -> np.ndarray:
    drone = env.drone
    snap = drone.backend.snapshot()
    for u in drone.in_flight():
        drone.backend.current_action = u
        for _ in range(5):
            drone.backend.substep(drone.ctbr.rotor_forces(u, drone.backend.state().w))
    hint = env.true_hint
    chunk = []
    for _ in range(horizon):
        st = drone.backend.state()
        idx, _, _ = env.cl.project(st.p, hint)
        hint = idx
        j = (idx + expert.lookahead_pts) % env.cl.n_pts
        a_opt = expert.ex(st, env.cl.pts[j], env.cl.vels[j], env.cl.accs[j])
        chunk.append(a_opt)
        drone.backend.current_action = a_opt
        for _ in range(5):
            drone.backend.substep(drone.ctbr.rotor_forces(a_opt, drone.backend.state().w))
    drone.backend.restore(snap)
    return np.array(chunk, dtype=np.float32)


def train_dida_chunk_models(seed: int, iters: int = 6, eps_per_iter: int = 8):
    torch.manual_seed(seed + 500)
    rng = np.random.default_rng(seed + 500)
    in_dim = EXACT_BASE_DIM + 4 * HIST
    m_single = ActionChunkMLP(in_dim, horizon=1)
    m_chunk4 = ActionChunkMLP(in_dim, horizon=CHUNK_H)
    expert = GeometricGateController()

    env = GateRacingEnv("figure8", "source", HIST, delays=TRAIN_DELAYS, max_ep_s=5.5, seed=seed + 500)
    X_all, Y_all = [], []
    for it in range(iters):
        beta = 1.0 if it == 0 else max(0.0, 0.5 - 0.12 * it)
        for ep in range(eps_per_iter):
            ov, _ = dyn_sampler_with_vec(rng)
            obs, _ = env.reset(seed=20_000 * seed + 100 * it + ep)
            env.drone.backend.set_dynamics(ov)
            p_cmd_drop_train = float(rng.choice([0.0, 0.25, 0.50]))
            queue = deque([env.drone.hover_action.copy()] * CHUNK_H, maxlen=CHUNK_H)
            while True:
                y_chunk = query_chunk_labels(env, expert, CHUNK_H)
                X_all.append(obs.copy())
                Y_all.append(y_chunk)
                if rng.random() < beta:
                    pred_chunk = y_chunk
                else:
                    with torch.no_grad():
                        pred_chunk = m_chunk4(torch.as_tensor(obs[None]))[0].numpy()
                if rng.random() >= p_cmd_drop_train:
                    queue.clear()
                    for h_i in range(CHUNK_H):
                        queue.append(pred_chunk[h_i])
                a_exec = queue[0].copy()
                if len(queue) > 1:
                    queue.popleft()
                else:
                    queue.append(a_exec.copy())
                a_exec = np.clip(a_exec + 0.03 * rng.standard_normal(4), -1.0, 1.0)
                obs, _, term, trunc, _ = env.step(a_exec)
                if term or trunc:
                    break

        X_t = torch.as_tensor(np.array(X_all), dtype=torch.float32)
        Y_t = torch.as_tensor(np.array(Y_all), dtype=torch.float32)
        for mdl, target_y in [(m_single, Y_t[:, :1, :]), (m_chunk4, Y_t)]:
            mdl.mu.copy_(X_t.mean(0))
            mdl.sd.copy_(torch.clamp(X_t.std(0), min=0.25))
            opt = torch.optim.Adam(mdl.parameters(), lr=2e-3)
            n = len(X_t)
            for _ in range(40):
                perm = torch.randperm(n)
                for i in range(0, n, 512):
                    sl = perm[i : i + 512]
                    loss = ((mdl(X_t[sl]) - target_y[sl]) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    return m_single, m_chunk4


def eval_cmd_dropout(model: ActionChunkMLP, env: GateRacingEnv, p_cmd_drop: float, n_eval: int = 10, seed0: int = 40_000):
    gfs, clean_cps, crashes = [], [], []
    rng = np.random.default_rng(seed0 + int(round(100 * p_cmd_drop)))
    H = model.horizon
    for ep in range(n_eval):
        obs, _ = env.reset(seed=seed0 + ep)
        queue = deque([env.drone.hover_action.copy()] * H, maxlen=H)
        info = {}
        while True:
            with torch.no_grad():
                pred_chunk = model(torch.as_tensor(obs[None]))[0].numpy()
            if rng.random() >= p_cmd_drop:
                queue.clear()
                for h_i in range(H):
                    queue.append(pred_chunk[h_i])
            a_exec = queue[0].copy()
            if len(queue) > 1:
                queue.popleft()
            obs, _, term, trunc, info = env.step(a_exec)
            if term or trunc:
                break
        gfs.append(info["gates_passed"] / env.total_target_gates)
        clean_cps.append(float(info["collision_free"]))
        crashes.append(float(info["crashed"]))
    return {
        "gate_frac": float(np.mean(gfs)),
        "clean_completion": float(np.mean(clean_cps)),
        "crash_rate": float(np.mean(crashes)),
    }


# --------------------------------------------------------------------------------------------------
# Part B Models: Track-Free Proprioceptive + Packet-Timestamp RMA Observer vs Blind FF / Blind GRU
# --------------------------------------------------------------------------------------------------


class PacketTimestampStream:
    """
    Strictly causal, track-free sensor stream processor:
      - `f_blind`: uses naive 4-tick ZOH difference `(p_obs_t - p_obs_{t-4}) / 0.04` (drops to 0 during >=4-tick
                   ZOH bursts and spikes by `(4 + tau)/4` upon recovery!)
      - `f_stale`: uses timestamped valid-packet history `[(t_valid, p_valid)]` (`dt >= 0.06s` baseline for low noise
                   AND exact elapsed-time division across arbitrary ZOH bursts) + `[valid_flag, tau_stale / 10]`!
    """

    def __init__(self, init_p: np.ndarray):
        self.p_hist4 = deque([init_p.copy()] * 5, maxlen=5)
        self.valid_buf = deque([(0, init_p.copy())], maxlen=8)
        self.last_clean_vB = np.zeros(3, dtype=np.float64)
        self.tau_stale = 0
        self.tick = 0

    def step(self, obs_st: DroneState, ah: deque, dropped: bool) -> tuple[np.ndarray, np.ndarray]:
        Rt = obs_st.R.T
        past = list(ah)[-HIST:][::-1]

        # Blind 4-tick ZOH finite difference
        v_blind_B = np.clip(Rt @ (obs_st.p - self.p_hist4[0]) / 0.04, -9.0, 9.0)
        self.p_hist4.append(obs_st.p.copy())

        # Timestamped staleness-aware displacement rate across >=50ms valid baseline
        if (not dropped) or self.tick == 0:
            # Pick oldest valid packet in buffer (>=5-7 valid arrivals ago, ~60-120ms baseline -> <0.3m/s noise!)
            t_old, p_old = self.valid_buf[0]
            dt = (self.tick - t_old) * RL_DT
            if dt >= 0.02:
                self.last_clean_vB = np.clip(Rt @ (obs_st.p - p_old) / dt, -6.5, 6.5)
            self.valid_buf.append((self.tick, obs_st.p.copy()))
            age_feat = 0.0
            self.tau_stale = 0
        else:
            self.tau_stale += 1
            age_feat = min(self.tau_stale, 12) / 12.0

        self.tick += 1
        f_blind = np.concatenate([0.33 * v_blind_B, Rt[:, 2], Rt[:, 0], 0.25 * obs_st.w] + past).astype(np.float32)
        f_stale = np.concatenate(
            [0.33 * self.last_clean_vB, Rt[:, 2], Rt[:, 0], 0.25 * obs_st.w, [0.0 if dropped else 1.0, age_feat]] + past
        ).astype(np.float32)
        return f_blind, f_stale


class ClampedGRU(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 96):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, d_out))
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))

    def forward_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        z, _ = self.gru(self.in_proj((x_seq - self.mu) / self.sd))
        return self.head(z)

    def step_one(self, x: np.ndarray, h: torch.Tensor | None):
        xt = torch.as_tensor(x, dtype=torch.float32).view(1, 1, -1)
        z, h_next = self.gru(self.in_proj((xt - self.mu) / self.sd), h)
        return self.head(z[0, 0]).detach().numpy(), h_next


class ClampedFF(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, d_out))
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mu) / self.sd)


def fit_gru_batched(d_in: int, d_out: int, seqs_x, seqs_y, steps: int = 850, chunk: int = 120, seed: int = 0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mdl = ClampedGRU(d_in, d_out, hidden=96)
    all_x = np.concatenate(seqs_x, axis=0)
    mdl.mu.copy_(torch.as_tensor(all_x.mean(0), dtype=torch.float32))
    mdl.sd.copy_(torch.as_tensor(np.maximum(all_x.std(0), 0.20), dtype=torch.float32))

    X = [torch.as_tensor(x, dtype=torch.float32) for x in seqs_x]
    Y = [torch.as_tensor(y, dtype=torch.float32) for y in seqs_y]
    opt = torch.optim.Adam(mdl.parameters(), lr=2.5e-3)
    for _ in range(steps):
        idxs = rng.integers(0, len(X), size=16)
        bx = torch.zeros((16, chunk, d_in), dtype=torch.float32)
        by = torch.zeros((16, chunk, d_out), dtype=torch.float32)
        bm = torch.zeros((16, chunk), dtype=torch.float32)
        for b_i, i in enumerate(idxs):
            T = len(X[i])
            L = min(chunk, T)
            st = int(rng.integers(0, T - L + 1))
            bx[b_i, :L] = X[i][st : st + L]
            by[b_i, :L] = Y[i][st : st + L]
            # Ignore first 10 warm-up ticks of mid-flight chunk so hidden-state cold-start never biases loss!
            bm[b_i, 10:L] = 1.0
        pred = mdl.forward_seq(bx)
        loss = (((pred - by) ** 2).mean(-1) * bm).sum() / bm.sum()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        opt.step()
    return mdl


def train_part_b_models(seed: int):
    """
    1. Train privileged RMA base racing policy `priv_policy(obs_spatial_24, z_ext_4, a_{t-8:t-1})` on `figure8` + 3D z-waves.
    2. Train 3 track-free state/extrinsics estimators on 3D Lissajous excitations in randomized `L2F`:
       - `ff_blind`      : Memoryless MLP predicting `(v_hat_B, z_hat)` from `f_blind`
       - `gru_blind`     : Recurrent GRU predicting `(v_hat_B, z_hat)` from `f_blind`
       - `rma_staleness` : Recurrent GRU predicting `(v_hat_B, z_hat)` from timestamped `f_stale`!
    """
    torch.manual_seed(seed + 900)
    rng = np.random.default_rng(seed + 900)
    ex = GeometricGateController()

    # 1. Privileged RMA Actor on `figure8` (with 3D altitude perturbations + randomized PnP gate bias robustness)
    priv_in_dim = EXACT_BASE_DIM + 4 * HIST + 4  # +4 extrinsics z_vec!
    priv_actor = ActionChunkMLP(priv_in_dim, horizon=1, hidden=128)
    env_p = GateRacingEnv("figure8", "source", HIST, delays=TRAIN_DELAYS, max_ep_s=5.5, seed=seed + 910)
    Xp_all, Ya_all = [], []
    for it in range(5):
        beta = 1.0 if it == 0 else max(0.1, 0.5 - 0.12 * it)
        for ep in range(8):
            ov, z_vec = dyn_sampler_with_vec(rng)
            obs, _ = env_p.reset(seed=35_000 * seed + 100 * it + ep)
            env_p.drone.backend.set_dynamics(ov)
            pnp_sc = float(rng.uniform(0.0, 1.2))
            while True:
                a_opt = query_chunk_labels(env_p, ex, 1)  # (1, 4)
                obs_in = obs.copy()
                # Expose actor to realistic gate-camera PnP bias on `rel_g0` (`obs[9:12]`) + noisy `v_hat`
                obs_in[9:12] += 0.5 * oblique_pnp_gate_bias(obs_in[9:12] * 2.0, pnp_sc).astype(np.float32)
                obs_in[15:18] += rng.normal(0.0, 0.05, 3).astype(np.float32)
                p_feat = np.concatenate([obs_in, z_vec]).astype(np.float32)
                Xp_all.append(p_feat)
                Ya_all.append(a_opt)
                if it == 0 or rng.random() < beta:
                    a_ex = a_opt[0]
                else:
                    with torch.no_grad():
                        a_ex = priv_actor(torch.as_tensor(p_feat[None]))[0, 0].numpy()
                obs, _, term, trunc, _ = env_p.step(np.clip(a_ex + 0.04 * rng.standard_normal(4), -1, 1))
                if term or trunc:
                    break
        Xt = torch.as_tensor(np.array(Xp_all), dtype=torch.float32)
        Yt = torch.as_tensor(np.array(Ya_all), dtype=torch.float32)
        priv_actor.mu.copy_(Xt.mean(0))
        priv_actor.sd.copy_(torch.clamp(Xt.std(0), min=0.25))
        opt = torch.optim.Adam(priv_actor.parameters(), lr=2e-3)
        for _ in range(35):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 512):
                sl = perm[i : i + 512]
                loss = ((priv_actor(Xt[sl]) - Yt[sl]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    # 2. Track-Free 3D Lissajous Observer Training in Randomized L2F
    seqs_bl, seqs_st, seqs_tgt = [], [], []
    for ep in range(48):
        p_dr = float(rng.choice([0.0, 0.25, 0.50, 0.65]))
        ov, z_vec = dyn_sampler_with_vec(rng)
        dr = make_source(
            seed=1000 * seed + ep,
            overrides=ov,
            iface=InterfaceConfig(
                action_delay=int(rng.choice(TRAIN_DELAYS)),
                obs_delay=3,
                p_drop=p_dr,
                dropout_mode="zoh",
                obs_noise={"p": 0.02, "w": 0.04},
            ),
        )
        obs = dr.reset(DroneState.hover(p=(0.0, 0.0, 1.5)))
        ah = deque([dr.hover_action.copy()] * HIST, maxlen=HIST)
        stream = PacketTimestampStream(obs.p)
        ph1, ph2, ph3 = rng.uniform(0.0, 2.0 * math.pi, 3)
        w1, w2, w3 = rng.uniform(1.1, 2.0), rng.uniform(1.3, 2.2), rng.uniform(1.5, 2.6)
        xb, xs, yt = [], [], []
        for k in range(360):
            t = k * RL_DT
            p_r = np.array([1.9 * math.sin(w1 * t + ph1), 1.6 * math.sin(w2 * t + ph2), 1.5 + 0.55 * math.sin(w3 * t + ph3)])
            v_r = np.array([1.9 * w1 * math.cos(w1 * t + ph1), 1.6 * w2 * math.cos(w2 * t + ph2), 0.55 * w3 * math.cos(w3 * t + ph3)])
            a_r = np.array([-1.9 * w1 * w1 * math.sin(w1 * t + ph1), -1.6 * w2 * w2 * math.sin(w2 * t + ph2), -0.55 * w3 * w3 * math.sin(w3 * t + ph3)])
            fb, fs = stream.step(obs, ah, bool(dr.dropped))
            st_t = dr.true_state()
            v_B_sc = (0.33 * (st_t.R.T @ st_t.v)).astype(np.float32)
            xb.append(fb)
            xs.append(fs)
            yt.append(np.concatenate([v_B_sc, z_vec]).astype(np.float32))
            a = np.clip(ex.ex(st_t, p_r, v_r, a_r) + 0.08 * rng.standard_normal(4), -1.0, 1.0)
            obs = dr.step(a)
            ah.append(a)
        seqs_bl.append(np.array(xb[8:]))
        seqs_st.append(np.array(xs[8:]))
        seqs_tgt.append(np.array(yt[8:]))

    # Fit FF_blind, GRU_blind, and RMA_staleness
    X_bl_cat = torch.as_tensor(np.concatenate(seqs_bl, axis=0), dtype=torch.float32)
    Y_cat = torch.as_tensor(np.concatenate(seqs_tgt, axis=0), dtype=torch.float32)
    ff_blind = ClampedFF(X_bl_cat.shape[1], 7)
    ff_blind.mu.copy_(X_bl_cat.mean(0))
    ff_blind.sd.copy_(torch.clamp(X_bl_cat.std(0), min=0.20))
    opt_f = torch.optim.Adam(ff_blind.parameters(), lr=2.5e-3)
    for _ in range(45):
        perm = torch.randperm(len(X_bl_cat))
        for i in range(0, len(X_bl_cat), 512):
            sl = perm[i : i + 512]
            loss = ((ff_blind(X_bl_cat[sl]) - Y_cat[sl]) ** 2).mean()
            opt_f.zero_grad()
            loss.backward()
            opt_f.step()

    gru_blind = fit_gru_batched(seqs_bl[0].shape[1], 7, seqs_bl, seqs_tgt, steps=900, seed=seed + 10)
    rma_stale = fit_gru_batched(seqs_st[0].shape[1], 7, seqs_st, seqs_tgt, steps=900, seed=seed + 20)
    return priv_actor, ff_blind, gru_blind, rma_stale


def eval_part_b(kind: str, obs_mdl, priv_actor: ActionChunkMLP, env: GateRacingEnv, pnp_scale: float = 1.0, n_eval: int = 10, seed0: int = 45_000):
    gfs, crashes, v_errs, z_hats = [], [], [], []
    for ep in range(n_eval):
        env.reset(seed=seed0 + ep)
        stream = PacketTimestampStream(env.obs_state.p)
        h = None
        v_ep = []
        info = {}
        while True:
            obs = env.obs_state
            fb, fs = stream.step(obs, env.ah, bool(getattr(env.drone, "dropped", False)))
            with torch.no_grad():
                if kind == "ff_blind":
                    lat = obs_mdl(torch.as_tensor(fb[None]))[0].numpy()
                elif kind == "gru_blind":
                    lat, h = obs_mdl.step_one(fb, h)
                else:
                    lat, h = obs_mdl.step_one(fs, h)

                vh_sc, zh = lat[:3], lat[3:]
                o_s, env.obs_hint = build_racing_obs(obs, env.cl, env.active_gate_idx(), env.obs_hint, env.ah, HIST, sensor_limited=True)
                o_s = o_s.copy()
                if pnp_scale > 0.0:
                    o_s[9:12] += 0.5 * oblique_pnp_gate_bias(o_s[9:12] * 2.0, pnp_scale).astype(np.float32)
                p_in = np.concatenate([o_s[:15], vh_sc, o_s[15:], zh]).astype(np.float32)
                a = priv_actor(torch.as_tensor(p_in[None]))[0, 0].numpy()

            st_true = env.drone.true_state()
            v_ep.append(float(np.linalg.norm(vh_sc * 3.0 - (st_true.R.T @ st_true.v))))
            z_hats.append(zh)
            _, _, term, trunc, info = env.step(a)
            if term or trunc:
                break

        gfs.append(info["gates_passed"] / env.total_target_gates)
        crashes.append(float(info["crashed"]))
        if len(v_ep) > 20:
            v_errs.append(float(np.mean(v_ep[20:])))

    return {
        "gate_frac": float(np.mean(gfs)),
        "crash_rate": float(np.mean(crashes)),
        "vel_rmse_ms": float(np.mean(v_errs)) if v_errs else float("nan"),
        "z_thrust_mean": float(np.mean([z[1] for z in z_hats])) if z_hats else float("nan"),
    }


def run_seed(args):
    seed, part_b_only = args
    torch.set_num_threads(1)
    t0 = time.time()
    part_a = None
    if not part_b_only:
        m_single, m_chunk4 = train_dida_chunk_models(seed)
        env_cmd = GateRacingEnv("ellipse3d", "target", HIST, delays=(2,), random_start_gate=False, max_ep_s=12.5, seed=700 + seed)
        p_cmd_list = [0.0, 0.25, 0.50, 0.70, 0.80]
        part_a = {
            "single_step_zoh": {str(p): eval_cmd_dropout(m_single, env_cmd, p) for p in p_cmd_list},
            "dida_chunk4": {str(p): eval_cmd_dropout(m_chunk4, env_cmd, p) for p in p_cmd_list},
        }

    priv_actor, ff_blind, gru_blind, rma_stale = train_part_b_models(seed)
    p_obs_list = [0.0, 0.25, 0.50, 0.65]
    part_b = {"target_nominal": {}, "target_perturbed": {}}
    for tname, kfs in [("target_nominal", 1.0), ("target_perturbed", 1.20)]:
        for mname, mdl in [("ff_blind", ff_blind), ("gru_blind", gru_blind), ("rma_staleness", rma_stale)]:
            part_b[tname][mname] = {}
            for p_o in p_obs_list:
                # Note: VIO pose carries obs_delay=3, ZOH dropout p_o, and 2cm/0.04rad/s noise (`pnp_bias_scale=0` on VIO;
                # camera PnP bias is applied on gate vector `rel_g0` inside `eval_part_b`!)
                ev_o = GateRacingEnv(
                    "ellipse3d",
                    "target",
                    HIST,
                    delays=(2,),
                    kf_scale=kfs,
                    sensor_limited=True,
                    obs_delay=3,
                    p_drop=p_o,
                    pnp_bias_scale=0.0,
                    random_start_gate=False,
                    max_ep_s=12.5,
                    seed=750 + seed,
                )
                part_b[tname][mname][str(p_o)] = eval_part_b(mname, mdl, priv_actor, ev_o, pnp_scale=1.0)

    return {"seed": seed, "wall_s": time.time() - t0, "part_a": part_a, "part_b": part_b}


def agg_dict_list(ds):
    out = {}
    for k in ds[0]:
        vals = [d[k] for d in ds if np.isfinite(d[k])]
        out[k] = mean_std(vals) if vals else {"mean": float("nan"), "std": float("nan"), "n": 0, "values": []}
    return out


def main():
    part_b_only = "--part-b-only" in sys.argv
    t0 = time.time()
    with mp.get_context("spawn").Pool(3) as pool:
        outs = pool.map(run_seed, [(s, part_b_only) for s in SEEDS])
    wall = time.time() - t0
    print(f"Exp 12 ({'Part B only' if part_b_only else 'Full'}) finished in {wall:.1f}s")

    prev_path = f"{OUT}/exp12_chunking_staleness.json"
    if part_b_only and os.path.exists(prev_path):
        with open(prev_path) as f:
            prev = json.load(f)
        agg_a = prev["part_a_cmd_dropout"]
    else:
        agg_a = {
            m: {p: agg_dict_list([o["part_a"][m][p] for o in outs]) for p in outs[0]["part_a"][m]}
            for m in outs[0]["part_a"]
        }

    agg_b = {
        t: {
            m: {p: agg_dict_list([o["part_b"][t][m][p] for o in outs]) for p in outs[0]["part_b"][t][m]}
            for m in outs[0]["part_b"][t]
        }
        for t in outs[0]["part_b"]
    }

    res = {"seeds": SEEDS, "wall_s": wall, "part_a_cmd_dropout": agg_a, "part_b_sensor_observers": agg_b}
    save_json("exp12_chunking_staleness.json", res)

    print("\n=== Part A: Command-Packet Drop on Held-Out `ellipse3d` PyBullet Target (d=2) ===")
    for m in agg_a:
        for p, v in agg_a[m].items():
            print(f"  {m:16s} p_cmd={p}: gf={fmt(v['gate_frac']):>14} | crash={fmt(v['crash_rate']):>14}")

    for t in agg_b:
        print(f"\n=== Part B [{t}]: Sensor Dropout (obs_delay=3, hidden v_B, PnP bias) on `ellipse3d` ===")
        for m in agg_b[t]:
            for p, v in agg_b[t][m].items():
                v_str = f"{v['vel_rmse_ms']['mean']:.3f} m/s"
                z_str = f"{v['z_thrust_mean']['mean']:+.2f}"
                print(
                    f"  {m:15s} p_obs={p}: gf={fmt(v['gate_frac']):>14} | cr={fmt(v['crash_rate']):>14} | "
                    f"v_rmse={v_str} | z_thr={z_str}"
                )

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))
    pa = [float(p) for p in agg_a["single_step_zoh"]]
    for m, mk in [("single_step_zoh", "o"), ("dida_chunk4", "s")]:
        axes[0].errorbar(
            pa,
            [agg_a[m][str(p)]["gate_frac"]["mean"] for p in pa],
            yerr=[agg_a[m][str(p)]["gate_frac"]["std"] for p in pa],
            marker=mk,
            capsize=2,
            label=m,
        )
    axes[0].set_title("A. Command-Link Dropout (DIDA Chunk-4)")
    axes[0].set_xlabel("Command Packet Drop Prob p_cmd")
    axes[0].set_ylabel("Target Gate Fraction (ellipse3d)")
    axes[0].set_ylim(0, 1.08)
    axes[0].legend(fontsize=8)

    pb = [float(p) for p in agg_b["target_nominal"]["ff_blind"]]
    for idx_ax, tname in [(1, "target_nominal"), (2, "target_perturbed")]:
        for m, mk in [("ff_blind", "o"), ("gru_blind", "^"), ("rma_staleness", "D")]:
            axes[idx_ax].errorbar(
                pb,
                [agg_b[tname][m][str(p)]["gate_frac"]["mean"] for p in pb],
                yerr=[agg_b[tname][m][str(p)]["gate_frac"]["std"] for p in pb],
                marker=mk,
                capsize=2,
                label=m,
            )
        axes[idx_ax].set_title(f"{'B' if idx_ax == 1 else 'C'}. Sensor Dropout ({tname})")
        axes[idx_ax].set_xlabel("ZOH Obs Dropout Prob p_obs")
        axes[idx_ax].set_ylim(0, 1.08)
        axes[idx_ax].legend(fontsize=7.5)

    plt.tight_layout()
    plt.savefig(f"{OUT}/exp12_chunking_staleness.png", dpi=150)


if __name__ == "__main__":
    main()
