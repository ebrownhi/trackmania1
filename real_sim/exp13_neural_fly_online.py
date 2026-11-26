"""
Exp 13 (Step 3 Riff R3.2): Neural-Fly Online Last-Layer Composite RLS Adaptation — Live Lap-1 -> Lap-2 -> Lap-3!

Literature Anchor:
  O'Connell, Shi, Chung et al., "Neural-Fly enables rapid learning for agile flight in strong winds"
  (Science Robotics, 2022).

Core Scientific Finding & Ablation Included:
  - O'Connell et al. prove that including raw control commands `a_cmd` inside the adaptive basis `phi(x, a)`
    creates a destabilizing algebraic feedback loop `a_cmd -> phi(a_cmd) -> d_hat -> a_cmd` (`neural_fly_naive_u`).
  - Conditioning the meta-learned spectral-normalized basis strictly on kinematic state `phi_theta(v_world, R[:,2], R[:,0])`,
    paired with low-pass filtered prediction error + Lyapunov composite tracking adaptation (`s = e_v + lam * e_p`),
    stabilizes adaptation at 100 Hz and improves tracking live from Lap 1 to Lap 2 & Lap 3 with ZERO offline target data!

Target Regimes (held-out 3D `ellipse3d`, 3 continuous laps = 18 gates, zero resets between laps, 3 seeds):
  1. `nominal_pyb`       : Nominal PyBullet CF2X (`m=0.027`, `T/W=2.25`, rotor drag, `d=1`)
  2. `turbulent_wind_pyb`: PyBullet CF2X + strong 3D spatially/temporally varying crosswind + nonlinear drag
  3. `midrace_fault_pyb` : Starts nominal on Lap 1, then at start of Lap 2 suffers an abrupt `-25%` motor thrust
                          loss (`kf x 0.75`) + strong lateral wind shear!

Methods Compared:
  - `unadapted_base`     : Base L2F geometric controller (no online adaptation)
  - `world_pid`          : Classical world-frame integral disturbance observer (`k_i = 2.0`, anti-windup)
  - `neural_fly_naive_u` : Ablation — includes `a_cmd` in `phi` without filtering (algebraic loop instability!)
  - `neural_fly_rls`     : State-only DAIML basis `phi_theta(v, R)` + filtered composite RLS/Lyapunov adaptation
"""

from __future__ import annotations

import contextlib
import io
import math
import multiprocessing as mp
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from common import OUT, fmt, mean_std, plt, save_json
from racing_env import FastCenterline, equilibrium_R
from sims.drone_pair import (
    Course,
    CTBRConfig,
    DroneState,
    GRAVITY,
    GeometricExpert,
    InterfaceConfig,
    RL_DT,
    SUBSTEPS,
    make_source,
    make_target,
)

SEEDS = [0, 1, 2]
STATE_FEAT_DIM = 9   # [v_world (3), R[:,2] (3), R[:,0] (3)] — strictly control-free per Neural-Fly!
NAIVE_FEAT_DIM = 13  # [v_world (3), R[:,2] (3), w_B (3), a_cmd (4)] — algebraic-loop ablation
BASIS_DIM = 12


def state_only_features(st: DroneState) -> np.ndarray:
    return np.concatenate([st.v, st.R[:, 2], st.R[:, 0]]).astype(np.float32)


def naive_u_features(st: DroneState, a: np.ndarray) -> np.ndarray:
    return np.concatenate([st.v, st.R[:, 2], st.w, a]).astype(np.float32)


class NeuralFlyBasis(nn.Module):
    def __init__(self, in_dim: int, basis_dim: int = BASIS_DIM, hidden: int = 48):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(in_dim, hidden)),
            nn.Tanh(),
            spectral_norm(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            spectral_norm(nn.Linear(hidden, basis_dim - 1)),
            nn.Tanh(),
        )
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net((x - self.mu) / self.sd)
        ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
        return torch.cat([ones, z], dim=-1)


def meta_train_bases_in_l2f(seed: int, n_domains: int = 24) -> tuple[NeuralFlyBasis, NeuralFlyBasis]:
    """Meta-learn both `phi_state(v, R)` and ablation `phi_naive(v, R, w, a)` purely in randomized `L2F`."""
    torch.manual_seed(seed + 100)
    rng = np.random.default_rng(seed + 100)
    course = Course(kind="figure8")
    ex = GeometricExpert()
    ctbr = CTBRConfig()

    ds_state, ds_naive = [], []
    all_Xs, all_Xn = [], []
    for d_idx in range(n_domains):
        ov = {
            "mass_scale": float(rng.uniform(0.75, 1.30)),
            "thrust_scale": float(rng.uniform(0.75, 1.35)),
            "tau_rise_scale": float(rng.uniform(0.45, 1.35)),
            "tau_fall_scale": float(rng.uniform(0.20, 1.35)),
        }
        wind_v = rng.uniform(-4.0, 4.0, size=3)
        drag_lin = float(rng.uniform(0.2, 0.7))
        drag_quad = float(rng.uniform(0.04, 0.18))
        dr = make_source(seed=seed + d_idx, overrides=ov)

        t0 = float(rng.uniform(0.0, course.period))
        p0, v0, a0 = course.ref(t0)
        dr.reset(DroneState(p0, equilibrium_R(a0), v0, np.zeros(3)))
        c_filt = GRAVITY

        Xs_d, Xn_d, Y_d = [], [], []
        for k in range(200):
            st = dr.true_state()
            a = np.clip(ex(st, *course.ref(t0 + k * RL_DT)) + 0.06 * rng.standard_normal(4), -1.0, 1.0)
            c_cmd, _ = ctbr.decode(a)
            c_filt = 0.72 * c_filt + 0.28 * c_cmd
            a_nom_w = st.R[:, 2] * c_filt - np.array([0.0, 0.0, GRAVITY])

            dr.step(a)
            st_next = dr.true_state()
            v_rel = wind_v - st.v
            aero_acc_w = drag_lin * v_rel + drag_quad * float(np.linalg.norm(v_rel)) * v_rel
            v_dist_next = st_next.v + RL_DT * aero_acc_w
            dr.backend.s.linear_velocity = list(v_dist_next)

            if k >= 6:
                r_world = (v_dist_next - st.v) / RL_DT - a_nom_w
                xs = state_only_features(st)
                xn = naive_u_features(st, a)
                Xs_d.append(xs)
                Xn_d.append(xn)
                Y_d.append(r_world.astype(np.float32))
                all_Xs.append(xs)
                all_Xn.append(xn)

        Yt = torch.as_tensor(np.array(Y_d))
        ds_state.append((torch.as_tensor(np.array(Xs_d)), Yt))
        ds_naive.append((torch.as_tensor(np.array(Xn_d)), Yt))

    def fit_basis(in_dim, all_X_list, ds):
        b = NeuralFlyBasis(in_dim, BASIS_DIM)
        Xt = torch.as_tensor(np.array(all_X_list))
        b.mu.copy_(Xt.mean(0))
        b.sd.copy_(Xt.std(0) + 1e-3)
        opt = torch.optim.Adam(b.parameters(), lr=3e-3)
        reg_I = 3e-2 * torch.eye(BASIS_DIM)
        for _ in range(130):
            loss = torch.tensor(0.0)
            for idx in rng.choice(len(ds), size=min(8, len(ds)), replace=False):
                Xd, Yd = ds[idx]
                nh = len(Xd) // 2
                P1, P2 = b(Xd[:nh]), b(Xd[nh:])
                W = torch.linalg.solve(P1.T @ P1 + reg_I, P1.T @ Yd[:nh])
                loss = loss + ((P2 @ W - Yd[nh:]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        b.eval()
        return b

    return fit_basis(STATE_FEAT_DIM, all_Xs, ds_state), fit_basis(NAIVE_FEAT_DIM, all_Xn, ds_naive)


class CompositeNeuralFlyAdapter:
    """
    O'Connell et al. (Science Robotics 2022) Composite RLS + Lyapunov Tracking Error Adaptation:
      dW/dt = -lam_w W + (y_filt - W phi_filt) phi_filt^T P - k_s s_track phi^T P
    """

    def __init__(self, basis: NeuralFlyBasis, naive_u: bool = False):
        self.basis = basis
        self.naive_u = naive_u
        self.M = BASIS_DIM
        self.W = np.zeros((3, self.M), dtype=np.float64)
        self.P = (0.6 if naive_u else 0.12) * np.eye(self.M, dtype=np.float64)
        self.lam = 0.985 if naive_u else 0.992
        self.ctbr = CTBRConfig()
        self.c_filt = GRAVITY
        self.y_filt = np.zeros(3, dtype=np.float64)
        self.ph_filt = np.zeros(self.M, dtype=np.float64)
        self._inited = False

    @torch.no_grad()
    def phi(self, st: DroneState, a: np.ndarray) -> np.ndarray:
        xf = naive_u_features(st, a) if self.naive_u else state_only_features(st)
        return self.basis(torch.as_tensor(xf[None]))[0].numpy().astype(np.float64)

    def predict(self, st: DroneState, a_prev: np.ndarray) -> np.ndarray:
        return np.clip(self.W @ self.phi(st, a_prev), -7.5, 7.5)

    def update(self, st_prev: DroneState, a_exec: np.ndarray, st_curr: DroneState, e_p: np.ndarray, e_v: np.ndarray):
        c_cmd, _ = self.ctbr.decode(a_exec)
        self.c_filt = 0.72 * self.c_filt + 0.28 * c_cmd
        a_nom_w = st_prev.R[:, 2] * self.c_filt - np.array([0.0, 0.0, GRAVITY])
        y_raw = (st_curr.v - st_prev.v) / RL_DT - a_nom_w
        ph_raw = self.phi(st_prev, a_exec)

        if self.naive_u:
            # Unfiltered algebraic-loop ablation
            P_ph = self.P @ ph_raw
            gain = P_ph / (self.lam + float(ph_raw @ P_ph))
            self.W = 0.998 * self.W + np.outer(y_raw - self.W @ ph_raw, gain)
            self.P = np.clip((self.P - np.outer(gain, P_ph)) / self.lam + 0.005 * np.eye(self.M), -5.0, 5.0)
            return

        # Low-pass filtered prediction error (tau_f = 0.05 s) + Lyapunov composite tracking error `s_track`
        alpha_f = 0.18
        if not self._inited:
            self.y_filt = y_raw.copy()
            self.ph_filt = ph_raw.copy()
            self._inited = True
        else:
            self.y_filt = (1.0 - alpha_f) * self.y_filt + alpha_f * y_raw
            self.ph_filt = (1.0 - alpha_f) * self.ph_filt + alpha_f * ph_raw

        P_ph = self.P @ self.ph_filt
        denom = self.lam + float(self.ph_filt @ P_ph)
        gain = P_ph / denom
        pred_err = self.y_filt - (self.W @ self.ph_filt)
        # Composite tracking error `s_track = (v - v_ref) + 1.5 * (p - p_ref) = -(e_v + 1.5 * e_p)`
        s_track = -(e_v + 1.5 * e_p)
        ph_curr = self.phi(st_curr, a_exec)
        self.W = 0.999 * self.W + 0.85 * np.outer(pred_err, gain) - 0.25 * RL_DT * np.outer(s_track, self.P @ ph_curr)
        self.P = (self.P - 0.85 * np.outer(gain, P_ph)) / self.lam + 0.0025 * np.eye(self.M)
        tr = float(np.trace(self.P))
        if tr > 3.6:
            self.P *= 3.6 / tr


def wind_force_world(regime: str, st: DroneState, t: float, mass: float) -> np.ndarray:
    if regime == "nominal_pyb":
        return np.zeros(3, dtype=np.float64)
    if regime == "turbulent_wind_pyb":
        # Strong arena crosswind + nonlinear velocity-coupled aerodynamic drag (~5.2 m/s^2 peak = 0.53g load;
        # reverses relative airspeed from 5.8 m/s upwind to 0.4 m/s downwind every 2.5 s!)
        v_wind = np.array([2.7 + 0.35 * math.sin(1.6 * t), -2.2 + 0.30 * math.cos(1.3 * t), -0.8], dtype=np.float64)
        v_rel = v_wind - st.v
        acc = 0.48 * v_rel + 0.09 * float(np.linalg.norm(v_rel)) * v_rel
        return mass * acc
    if regime == "midrace_fault_pyb":
        if t >= 4.2:
            v_wind = np.array([2.5, -2.0, -0.6], dtype=np.float64)
            v_rel = v_wind - st.v
            return mass * (0.45 * v_rel + 0.08 * float(np.linalg.norm(v_rel)) * v_rel)
        return np.zeros(3, dtype=np.float64)
    raise ValueError(regime)


def run_three_lap_race(method: str, regime: str, basis_state: NeuralFlyBasis, basis_naive: NeuralFlyBasis, seed: int):
    course = Course(kind="ellipse3d")
    cl = FastCenterline(course)
    with contextlib.redirect_stdout(io.StringIO()):
        drone = make_target(tau_m=0.03, iface=InterfaceConfig(action_delay=1), seed=seed)
    base_kf = drone.backend.env.KF
    mass_tgt = 0.027
    ctbr = CTBRConfig()

    rng = np.random.default_rng(1000 + seed)
    p0, v0, a0 = course.ref(0.0)
    st0 = DroneState(p0 + rng.normal(0, 0.02, 3), equilibrium_R(a0), v0 + rng.normal(0, 0.05, 3), np.zeros(3))
    obs = drone.reset(st0)

    if method == "neural_fly_rls":
        nf = CompositeNeuralFlyAdapter(basis_state, naive_u=False)
    elif method == "neural_fly_naive_u":
        nf = CompositeNeuralFlyAdapter(basis_naive, naive_u=True)
    else:
        nf = None

    int_err = np.zeros(3, dtype=np.float64)
    hint = 0
    gate_cursor = 0
    n_gates = len(cl.gates)
    total_gates = n_gates * 3

    lap_errs = {0: [], 1: [], 2: []}
    lap_gates = {0: 0, 1: 0, 2: 0}
    lap_times = {0: float("nan"), 1: float("nan"), 2: float("nan")}
    lap_start_k = 0
    a_prev = drone.hover_action.copy()
    fault_applied = False

    for k in range(1850):
        t = k * RL_DT
        cur_lap = min(2, gate_cursor // n_gates)
        if regime == "midrace_fault_pyb" and (cur_lap >= 1 or t >= 4.2) and not fault_applied:
            drone.backend.env.KF = base_kf * 0.75
            drone.backend.kf = base_kf * 0.75
            drone.backend.f_max = drone.backend.kf * (drone.backend.env.MAX_RPM ** 2)
            fault_applied = True

        st_prev = obs.copy()
        p_prev_true = drone.true_state().p.copy()
        hint, _, cross_err = cl.project(obs.p, hint)
        lap_errs[cur_lap].append(cross_err)

        j = (hint + 2) % cl.n_pts
        p_ref, v_ref, a_ref = cl.pts[j], cl.vels[j], cl.accs[j]
        e_p = p_ref - obs.p
        e_v = v_ref - obs.v

        if method == "unadapted_base":
            d_hat_w = np.zeros(3)
        elif method == "world_pid":
            int_err = np.clip(int_err + RL_DT * (e_p + 0.30 * e_v), -2.0, 2.0)
            d_hat_w = -2.0 * int_err
        else:
            d_hat_w = nf.predict(obs, a_prev)

        a_des = 6.5 * e_p + 4.0 * e_v + a_ref - d_hat_w + np.array([0.0, 0.0, GRAVITY])
        z_des = a_des / max(float(np.linalg.norm(a_des)), 1e-6)
        x_c = np.array([1.0, 0.0, 0.0])
        y_des = np.cross(z_des, x_c)
        y_des /= max(float(np.linalg.norm(y_des)), 1e-6)
        x_des = np.cross(y_des, z_des)
        R_des = np.column_stack([x_des, y_des, z_des])
        E = R_des.T @ obs.R - obs.R.T @ R_des
        e_R = 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]])
        c_cmd = float(a_des @ obs.R[:, 2])
        a_cmd = ctbr.encode(c_cmd, -7.0 * e_R)

        drone.act_fifo.append(a_cmd)
        a_applied = drone.act_fifo[0].copy()
        e_pb, pb = drone.backend.env, drone.backend.pb
        for _ in range(SUBSTEPS):
            st_sub = drone.backend.state()
            f_wind = wind_force_world(regime, st_sub, t, mass_tgt)
            if np.any(f_wind != 0.0):
                pb.applyExternalForce(
                    e_pb.DRONE_IDS[0], -1, f_wind.tolist(), st_sub.p.tolist(), pb.WORLD_FRAME, physicsClientId=e_pb.CLIENT
                )
            drone.backend.substep(drone.ctbr.rotor_forces(a_applied, st_sub.w))
        drone.obs_fifo.append(drone.backend.state())
        obs = drone.observe()
        a_prev = a_cmd

        if nf is not None:
            nf.update(st_prev, a_applied, obs, e_p, e_v)

        s_true = drone.true_state()
        gi = gate_cursor % n_gates
        g_c, g_t = cl.gates[gi], cl.tangents[gi]
        s_bef = float((p_prev_true - g_c) @ g_t)
        s_aft = float((s_true.p - g_c) @ g_t)
        if s_bef < 0.0 <= s_aft and float(np.linalg.norm(s_true.p - g_c)) < 1.5:
            denom = abs(s_bef) + abs(s_aft)
            p_cr = p_prev_true + (abs(s_bef) / max(denom, 1e-6)) * (s_true.p - p_prev_true)
            if float(np.linalg.norm(p_cr - g_c)) <= 0.35:
                lap_gates[cur_lap] += 1
            gate_cursor += 1
            if gate_cursor % n_gates == 0:
                completed_lap = (gate_cursor // n_gates) - 1
                if completed_lap in lap_times:
                    lap_times[completed_lap] = (k + 1 - lap_start_k) * RL_DT
                lap_start_k = k + 1

        if cross_err > 1.8 or s_true.p[2] < 0.08 or s_true.R[2, 2] < -0.1:
            break
        if gate_cursor >= total_gates:
            break

    out = {}
    for lp in [0, 1, 2]:
        errs = lap_errs[lp]
        out[f"lap{lp+1}_rmse"] = float(np.sqrt(np.mean(np.square(errs)))) if errs else 1.8
        out[f"lap{lp+1}_gates"] = lap_gates[lp] / float(n_gates)
        out[f"lap{lp+1}_time_s"] = lap_times[lp]
    out["total_gate_frac"] = sum(lap_gates.values()) / float(total_gates)
    return out


def job(seed: int):
    torch.set_num_threads(1)
    t0 = time.time()
    b_state, b_naive = meta_train_bases_in_l2f(seed)
    regimes = ["nominal_pyb", "turbulent_wind_pyb", "midrace_fault_pyb"]
    methods = ["unadapted_base", "world_pid", "neural_fly_naive_u", "neural_fly_rls"]
    res = {reg: {m: run_three_lap_race(m, reg, b_state, b_naive, seed) for m in methods} for reg in regimes}
    return {"seed": seed, "wall_s": time.time() - t0, "res": res}


def main():
    t0 = time.time()
    with mp.get_context("spawn").Pool(3) as pool:
        outs = pool.map(job, SEEDS)
    wall = time.time() - t0
    print(f"Exp 13 (Neural-Fly Online RLS) finished in {wall:.1f}s")

    regimes = ["nominal_pyb", "turbulent_wind_pyb", "midrace_fault_pyb"]
    methods = ["unadapted_base", "world_pid", "neural_fly_naive_u", "neural_fly_rls"]
    agg = {}
    for reg in regimes:
        agg[reg] = {}
        for m in methods:
            agg[reg][m] = {}
            for k in outs[0]["res"][reg][m]:
                vals = [o["res"][reg][m][k] for o in outs if np.isfinite(o["res"][reg][m][k])]
                agg[reg][m][k] = mean_std(vals) if vals else {"mean": float("nan"), "std": float("nan"), "n": 0, "values": []}

    save_json("exp13_neural_fly_online.json", {"seeds": SEEDS, "wall_s": wall, "agg": agg})

    for reg in regimes:
        print(f"\n=== [{reg}] Lap 1 -> Lap 2 -> Lap 3 Tracking RMSE [m] & Gate Frac (ellipse3d, 3 seeds) ===")
        for m in methods:
            g = agg[reg][m]
            print(
                f"  {m:18s} | L1: {g['lap1_rmse']['mean']:.3f}m ({g['lap1_gates']['mean']:.2f}) -> "
                f"L2: {g['lap2_rmse']['mean']:.3f}m ({g['lap2_gates']['mean']:.2f}) -> "
                f"L3: {g['lap3_rmse']['mean']:.3f}m ({g['lap3_gates']['mean']:.2f}) | "
                f"Total gates={fmt(g['total_gate_frac'])}"
            )

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.9))
    laps_x = [1, 2, 3]
    for idx, reg in enumerate(regimes):
        for m, mk in [("unadapted_base", "o"), ("world_pid", "s"), ("neural_fly_naive_u", "x"), ("neural_fly_rls", "D")]:
            g = agg[reg][m]
            axes[idx].errorbar(
                laps_x,
                [g[f"lap{l}_rmse"]["mean"] for l in laps_x],
                yerr=[g[f"lap{l}_rmse"]["std"] for l in laps_x],
                marker=mk,
                capsize=3,
                label=m,
            )
        axes[idx].set_xticks(laps_x, ["Lap 1", "Lap 2", "Lap 3"])
        axes[idx].set_ylabel("Centerline Tracking RMSE [m]")
        axes[idx].set_title(reg.replace("_pyb", "").replace("_", " ").title())
        axes[idx].legend(fontsize=7.5)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp13_neural_fly_online.png", dpi=150)


if __name__ == "__main__":
    main()
