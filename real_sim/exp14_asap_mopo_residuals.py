"""
Exp 14 (Step 4 Headline CMU Riff R4.1 + R4.2 + Step 5 Riff R5.1):
  CMU ASAP Delta-Action Simulator Alignment + MOPO Epistemic Uncertainty Penalty vs. Swift k-NN Delta-Dynamics!

Literature Anchors:
  - He et al. (CMU LeCAR Lab & NVIDIA), "ASAP: Aligning Simulation and Real-World Physics for Learning Agile
    Humanoid Whole-Body Skills" (RSS 2025)
  - Yu et al., "MOPO: Model-based Offline Policy Optimization" (NeurIPS 2020)
  - Kaufmann et al., "Champion-level drone racing using deep reinforcement learning (Swift)" (Nature 2023)

Scientific Questions Tested (3 seeds, equal 1,600 target calibration samples & equal 100k source PPO FT steps):
  1. Angular & Multi-Step Forward Model Fidelity (`Part A`):
     In Exp 6/8, Swift's 6D acceleration `k-NN` cut linear residual RMSE (0.99 -> 0.40 m/s^2) but left angular
     acceleration RMSE stuck at ~12 rad/s^2 because additive rigid-body accelerations bypass L2F's 500 Hz inner
     body-rate P-loop and motor lag. Does ASAP's delta-action model `a_eff = a + pi_Delta(v_B, w_B, a)`—acting
     *through* L2F's 500 Hz rate loop—slash both angular RMSE and 0.5 s open-loop rollout drift?
  2. Step 5 Cross-Domain Falsification (`Part A2`, R5.1):
     On a pure interface-delayed target with identical physics (`Delta f == 0, d = 3`), do both Swift `k-NN` and
     ASAP `pi_Delta` hallucinate phantom corrections at `d' = 0` and collapse to ~0 once Step-0 delay-aligned?
  3. Closed-Loop 6-Gate 3D Racing Fine-Tuning (`Part B`, `figure8` + held-out `ellipse3d`, `target_nominal` &
     `target_perturbed kf x 1.20, d = 3`):
     Compare `zeroshot_ppo`, `swift_knn_ft`, `swift_mopo_ft` (5-MLP acceleration ensemble + `-beta * sigma_ep`),
     `asap_delta_action_ft` (5-MLP delta-action alignment), and `asap_mopo_ft` (ASAP + `-beta * sigma_ep`)!
"""

from __future__ import annotations

import contextlib
import io
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn as nn

from common import OUT, fmt, mean_std, plt, save_json
from exp6_knn_residual import KNNResidual, OneStepSource, feats
from exp9_residual_finetune import scripted_sysid_action
from exp10_method_matrix import HIST, SB3PolicyCallable, TAUS, make_knn_residual_hook
from racing_env import GateRacingEnv, evaluate_racing
from sims.drone_pair import CTBRConfig, DroneState, InterfaceConfig, RL_DT, SUB_DT, SUBSTEPS, make_source, make_target

SEEDS = [0, 1, 2]
FT_STEPS = 100_000
CTBR = CTBRConfig()


# --------------------------------------------------------------------------------------------------
# 5-Member Neural Residual Ensemble (Supports both 6D Accel Residual & 4D ASAP Delta-Action + MOPO Uncertainty)
# --------------------------------------------------------------------------------------------------


class ResidualEnsemble:
    """Bootstrap/seed ensemble of 5 MLPs predicting mean correction + epistemic disagreement `sigma_ep(x)`."""

    def __init__(self, out_dim: int, max_mag: np.ndarray, n_members: int = 5, hidden: int = 64, seed: int = 0):
        self.out_dim = out_dim
        self.max_mag = np.asarray(max_mag, dtype=np.float64)
        self.n_members = n_members
        self.seed = seed
        self.models = []
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = [], [], [], [], [], []
        self.mu = np.zeros(10, dtype=np.float64)
        self.inv_sd = np.ones(10, dtype=np.float64)
        self.hidden = hidden

    def fit(self, Z: np.ndarray, Y: np.ndarray, epochs: int = 90, lr: float = 3e-3):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.mu = Z.mean(0).astype(np.float64)
        self.inv_sd = 1.0 / (Z.std(0).astype(np.float64) + 1e-4)
        Zs = ((Z - self.mu) * self.inv_sd).astype(np.float32)
        Y_sc = np.clip(Y / self.max_mag[None, :], -0.99, 0.99).astype(np.float32)

        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = [], [], [], [], [], []
        n = len(Zs)
        for m_i in range(self.n_members):
            torch.manual_seed(self.seed + 31 * m_i)
            net = nn.Sequential(
                nn.Linear(10, self.hidden),
                nn.Tanh(),
                nn.Linear(self.hidden, self.hidden),
                nn.Tanh(),
                nn.Linear(self.hidden, self.out_dim),
                nn.Tanh(),
            )
            boot_idx = rng.choice(n, size=n, replace=True)
            xb = torch.as_tensor(Zs[boot_idx])
            yb = torch.as_tensor(Y_sc[boot_idx])
            opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
            for _ in range(epochs):
                perm = torch.randperm(n)
                for i in range(0, n, 256):
                    sl = perm[i : i + 256]
                    loss = ((net(xb[sl]) - yb[sl]) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            # Export weights to contiguous NumPy arrays for 8-microsecond per-tick inference!
            with torch.no_grad():
                self.W1.append(net[0].weight.numpy().T.copy().astype(np.float64))
                self.b1.append(net[0].bias.numpy().copy().astype(np.float64))
                self.W2.append(net[2].weight.numpy().T.copy().astype(np.float64))
                self.b2.append(net[2].bias.numpy().copy().astype(np.float64))
                self.W3.append(net[4].weight.numpy().T.copy().astype(np.float64))
                self.b3.append(net[4].bias.numpy().copy().astype(np.float64))
        self.W1 = np.stack(self.W1, axis=0)  # (M, 10, H)
        self.b1 = np.stack(self.b1, axis=0)  # (M, H)
        self.W2 = np.stack(self.W2, axis=0)  # (M, H, H)
        self.b2 = np.stack(self.b2, axis=0)  # (M, H)
        self.W3 = np.stack(self.W3, axis=0)  # (M, H, out_dim)
        self.b3 = np.stack(self.b3, axis=0)  # (M, out_dim)
        return self

    def predict_one(self, z10: np.ndarray) -> tuple[np.ndarray, float]:
        """Fast vectorized 5-MLP ensemble evaluation -> `(mean_pred, epistemic_std_norm)`."""
        zs = (z10 - self.mu) * self.inv_sd  # (10,)
        h1 = np.tanh(np.einsum("j,mjh->mh", zs, self.W1) + self.b1)  # (M, H)
        h2 = np.tanh(np.einsum("mh,mhk->mk", h1, self.W2) + self.b2)  # (M, H)
        out = np.tanh(np.einsum("mk,mko->mo", h2, self.W3) + self.b3) * self.max_mag[None, :]  # (M, O)
        mean = out.mean(axis=0)
        # Normalized epistemic disagreement across ensemble members (MOPO uncertainty u(s, a))
        norm_std = float(np.sqrt(np.mean(np.var(out / self.max_mag[None, :], axis=0))))
        return mean, norm_std


def solve_asap_delta_actions(flights, tau_hat: float | None, d_hat: int) -> tuple[np.ndarray, np.ndarray]:
    """
    ASAP Stage 2 Inverse-Actuator Alignment:
    For each delay-aligned transition `(s_k, a_exec, s_{k+1})` in the calibration logs, find the corrective
    delta-action `da* in [-0.55, 0.55]^4` such that stepping `L2F(s_k, clip(a_exec + da*, -1, 1))` through L2F's
    500 Hz inner rate loop + motor lag reproduces the target's body-frame `(dv_B, dw)` transition!
    """
    fS = OneStepSource(tau_hat)
    be = fS.be
    Z_list, dA_list = [], []

    def sim_tick(s: DroneState, rpm0: np.ndarray, a_try: np.ndarray) -> np.ndarray:
        be.reset(s, rpm=rpm0)
        a_c = np.clip(a_try, -1.0, 1.0)
        for _ in range(SUBSTEPS):
            be.substep(CTBR.rotor_forces(a_c, be.state().w))
        ns = be.state()
        return np.array(
            [
                float((s.R.T @ (ns.v - s.v))[2]) / RL_DT,  # thrust axis accel
                float(ns.w[0] - s.w[0]) / RL_DT,
                float(ns.w[1] - s.w[1]) / RL_DT,
                float(ns.w[2] - s.w[2]) / RL_DT,
            ],
            dtype=np.float64,
        )

    for S, A in flights:
        a_exec_fn = lambda k: A[k - d_hat] if k - d_hat >= 0 else fS.hover_a
        rpm = be.force_to_rpm(np.full(4, be.base_params["dynamics"]["mass"] * 9.81 / 4.0))
        for k in range(len(A)):
            s, s1 = S[k], S[k + 1]
            ae = a_exec_fn(k)
            if k >= 10:
                y_tgt = np.array(
                    [
                        float((s.R.T @ (s1.v - s.v))[2]) / RL_DT,
                        float(s1.w[0] - s.w[0]) / RL_DT,
                        float(s1.w[1] - s.w[1]) / RL_DT,
                        float(s1.w[2] - s.w[2]) / RL_DT,
                    ],
                    dtype=np.float64,
                )
                # 3-iteration diagonal secant/Newton solve on decoupled 4D CTBR channels (c, wx, wy, wz)
                da = np.zeros(4, dtype=np.float64)
                eps = 0.05
                for _ in range(3):
                    y0 = sim_tick(s, rpm, ae + da)
                    err = y_tgt - y0
                    if float(np.max(np.abs(err))) < 0.05:
                        break
                    jac_diag = np.empty(4, dtype=np.float64)
                    for j in range(4):
                        e_vec = np.zeros(4)
                        e_vec[j] = eps
                        yj = sim_tick(s, rpm, ae + da + e_vec)
                        jac_diag[j] = (yj[j] - y0[j]) / eps
                    step = np.where(np.abs(jac_diag) > 1.0, err / jac_diag, 0.0)
                    da = np.clip(da + 0.8 * step, -0.55, 0.55)

                Z_list.append(np.concatenate([s.R.T @ s.v, s.w, ae]))
                dA_list.append(da)

            # Advance reconstructed rotor RPM filter using the aligned effective action `ae + da`!
            a_rpm = np.clip(ae + da, -1.0, 1.0) if k >= 10 else ae
            sp = be.force_to_rpm(CTBR.rotor_forces(a_rpm, s.w))
            for _ in range(SUBSTEPS):
                tau = np.where(sp >= rpm, fS.tau_r, fS.tau_f)
                al = np.exp(-SUB_DT / tau)
                rpm = al * rpm + (1.0 - al) * sp

    return np.array(Z_list, dtype=np.float64), np.array(dA_list, dtype=np.float64)


class AugmentedRacingEnv(GateRacingEnv):
    """
    Extends `GateRacingEnv("source")` to support:
      - `asap_ens`: frozen ASAP delta-action ensemble `a_eff = clip(a_applied + pi_Delta(s, a_applied), -1, 1)`
                    plus optional small body-xy drag residual!
      - `acc_ens`:  5-MLP 6D acceleration residual ensemble
      - `mopo_beta`: MOPO epistemic uncertainty penalty `-mopo_beta * sigma_ep(s, a)` added to training reward!
    """

    def __init__(self, *args, asap_ens: ResidualEnsemble | None = None, xy_drag_ens: ResidualEnsemble | None = None,
                 acc_ens: ResidualEnsemble | None = None, mopo_beta: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.asap_ens = asap_ens
        self.xy_drag_ens = xy_drag_ens
        self.acc_ens = acc_ens
        self.mopo_beta = mopo_beta
        self._last_sigma = 0.0

        if self.acc_ens is not None:
            cache = {}

            def acc_hook(st, a):
                if cache.get("key") is not a:
                    z10 = np.concatenate([st.R.T @ st.v, st.w, a])
                    pred, sig = self.acc_ens.predict_one(z10)
                    cache["key"] = a
                    cache["val"] = (pred[:3], pred[3:])
                    self._last_sigma = sig
                return cache["val"]

            self.residual_fn = acc_hook
        elif self.xy_drag_ens is not None:
            cache_xy = {}

            def xy_hook(st, a):
                if cache_xy.get("key") is not a:
                    z10 = np.concatenate([st.R.T @ st.v, st.w, a])
                    pred_xy, _ = self.xy_drag_ens.predict_one(z10)
                    a_res = np.array([pred_xy[0], pred_xy[1], 0.0], dtype=np.float64)
                    cache_xy["key"] = a
                    cache_xy["val"] = (a_res, np.zeros(3, dtype=np.float64))
                return cache_xy["val"]

            self.residual_fn = xy_hook

    def _get_drone(self, d: int):
        dr = super()._get_drone(d)
        if self.asap_ens is not None and not getattr(dr, "_asap_wrapped", False):
            orig_step = dr.step

            def asap_step(a: np.ndarray):
                dr.act_fifo.append(np.asarray(a, dtype=np.float64))
                a_app = dr.act_fifo[0]
                st0 = dr.backend.state()
                z10 = np.concatenate([st0.R.T @ st0.v, st0.w, a_app])
                da_pred, sig = self.asap_ens.predict_one(z10)
                self._last_sigma = sig
                a_eff = np.clip(a_app + da_pred, -1.0, 1.0)
                dr.applied = a_eff
                dr.backend.current_action = a_eff
                for _ in range(SUBSTEPS):
                    w = dr.backend.state().w
                    dr.backend.substep(dr.ctbr.rotor_forces(a_eff, w))
                dr.obs_fifo.append(dr.backend.state())
                return dr.observe()

            dr.step = asap_step
            dr._asap_wrapped = True
        return dr

    def step(self, action):
        self._last_sigma = 0.0
        obs, reward, term, trunc, info = super().step(action)
        if self.mopo_beta > 0.0:
            reward -= self.mopo_beta * self._last_sigma
        return obs, float(reward), term, trunc, info


def collect_target_and_heldout_flights(base_model, kf_scale: float, target_delay: int, seed: int, delta_f_zero: bool = False):
    """Collect 1 Step-0 multi-sine flight + 3 `figure8` calibration flights + 2 held-out `ellipse3d` test flights."""
    rng = np.random.default_rng(seed + 800)
    if delta_f_zero:
        cal_drone = make_source(iface=InterfaceConfig(action_delay=target_delay), seed=seed + 801)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            cal_drone = make_target(tau_m=0.03, iface=InterfaceConfig(action_delay=target_delay), seed=seed + 801)
        cal_drone.backend.env.KF *= kf_scale

    s0 = cal_drone.reset(DroneState.hover(p=(0.0, 0.0, 2.0)))
    S_id, A_id = [s0], []
    phase = float(rng.uniform(0, 2 * np.pi))
    for k in range(350):
        a = scripted_sysid_action(k, RL_DT, cal_drone.hover_action, phase)
        cal_drone.step(a)
        S_id.append(cal_drone.true_state())
        A_id.append(a)
    sysid_flights = [(S_id, np.array(A_id))]

    dom = "source" if delta_f_zero else "target"
    env_tr = GateRacingEnv("figure8", dom, HIST, delays=(target_delay,), kf_scale=kf_scale, max_ep_s=4.5, seed=seed + 810)
    flights = list(sysid_flights)
    for ep in range(3):
        obs, _ = env_tr.reset(seed=28_000 + 100 * seed + ep)
        S_ep, A_ep = [env_tr.drone.true_state()], []
        while True:
            a, _ = base_model.predict(obs, deterministic=True)
            a = np.clip(a + 0.08 * rng.standard_normal(4), -1.0, 1.0)
            obs, _, term, trunc, info = env_tr.step(a)
            S_ep.append(info["state"])
            A_ep.append(np.asarray(info["a"]))
            if term or trunc:
                break
        if len(A_ep) >= 20:
            flights.append((S_ep, np.array(A_ep)))

    # Held-out `ellipse3d` trajectories for forward-model fidelity test
    env_te = GateRacingEnv("ellipse3d", dom, HIST, delays=(target_delay,), kf_scale=kf_scale, random_start_gate=False, max_ep_s=5.0, seed=seed + 820)
    held_flights = []
    for ep in range(2):
        obs, _ = env_te.reset(seed=29_000 + 100 * seed + ep)
        S_ep, A_ep = [env_te.drone.true_state()], []
        while True:
            a, _ = base_model.predict(obs, deterministic=True)
            a = np.clip(a + 0.05 * rng.standard_normal(4), -1.0, 1.0)
            obs, _, term, trunc, info = env_te.step(a)
            S_ep.append(info["state"])
            A_ep.append(np.asarray(info["a"]))
            if term or trunc or len(A_ep) >= 350:
                break
        if len(A_ep) >= 60:
            held_flights.append((S_ep, np.array(A_ep)))

    return sysid_flights, flights, held_flights


def fit_all_residual_models(sysid_flights, flights, seed: int):
    # 1. Step-0 Joint (d_hat, tau_hat) scan
    best_id = None
    for tname, tau in TAUS.items():
        fS = OneStepSource(tau)
        for d in range(5):
            rs, Xs, Ufs = zip(*[fS.residuals(S, A, d, return_u_filt=True) for S, A in sysid_flights])
            r_id, X_id, uf_id = np.concatenate(rs), sum(Xs, []), np.concatenate(Ufs)
            Phi = np.column_stack([np.ones(len(X_id)), uf_id, [s.R.T @ s.v for s, _ in X_id]])
            th, *_ = np.linalg.lstsq(Phi, r_id[:, :3], rcond=None)
            rms = float(np.sqrt(np.mean((r_id[:, :3] - Phi @ th) ** 2)))
            if best_id is None or rms < best_id[0]:
                best_id = (rms, tname, tau, d)
    _, tname, tau_hat, d_hat = best_id

    # 2. Swift 6D acceleration k-NN + 5-MLP ensemble (for MOPO!)
    fS_best = OneStepSource(tau_hat)
    rs_all, Xs_all = zip(*[fS_best.residuals(S, A, d_hat) for S, A in flights])
    r_all, X_all = np.concatenate(rs_all), sum(Xs_all, [])
    Z_all = feats(X_all, "body10")
    knn = KNNResidual(k=5).fit(Z_all, r_all)
    acc_ens = ResidualEnsemble(6, max_mag=[8.0, 8.0, 12.0, 45.0, 45.0, 25.0], seed=seed + 10).fit(Z_all, r_all)

    # 3. ASAP 4D delta-action alignment ensemble + body-xy drag residual (since CTBR has no direct xy thruster!)
    Z_asap, dA_asap = solve_asap_delta_actions(flights, tau_hat, d_hat)
    asap_ens = ResidualEnsemble(4, max_mag=[0.55, 0.55, 0.55, 0.55], seed=seed + 20).fit(Z_asap, dA_asap)
    xy_ens = ResidualEnsemble(2, max_mag=[6.0, 6.0], seed=seed + 30).fit(Z_all, r_all[:, :2])

    return {
        "tau_name": tname,
        "tau_hat": tau_hat,
        "d_hat": int(d_hat),
        "knn": knn,
        "acc_ens": acc_ens,
        "asap_ens": asap_ens,
        "xy_ens": xy_ens,
        "Z_cal": Z_all,
    }


def eval_forward_fidelity(models_dict, held_flights, true_delay: int) -> dict:
    """Evaluate 1-step linear & angular accel RMSE AND 0.5s (50-tick) rollout drift on held-out `ellipse3d`."""
    tau_hat = models_dict["tau_hat"]
    d_hat = models_dict["d_hat"]
    knn_hook = make_knn_residual_hook(models_dict["knn"])
    acc_ens = models_dict["acc_ens"]
    asap_ens = models_dict["asap_ens"]
    xy_ens = models_dict["xy_ens"]

    ov = {"tau_rise_scale": tau_hat / 0.055455, "tau_fall_scale": tau_hat / 0.249394} if tau_hat else {}

    def step_sim(be, mode: str, a_exec: np.ndarray):
        if mode == "nominal_l2f":
            a_eff = a_exec
            be.residual_fn = None
        elif mode == "swift_knn":
            a_eff = a_exec
            be.residual_fn = knn_hook
        elif mode == "swift_ens":
            a_eff = a_exec
            be.residual_fn = lambda st, a: (acc_ens.predict_one(np.concatenate([st.R.T @ st.v, st.w, a]))[0][:3],
                                            acc_ens.predict_one(np.concatenate([st.R.T @ st.v, st.w, a]))[0][3:])
        elif mode == "asap_delta_a":
            st0 = be.state()
            z10 = np.concatenate([st0.R.T @ st0.v, st0.w, a_exec])
            da, _ = asap_ens.predict_one(z10)
            a_eff = np.clip(a_exec + da, -1.0, 1.0)
            r_xy, _ = xy_ens.predict_one(z10)
            be.residual_fn = lambda st, a, rxy=r_xy: (np.array([rxy[0], rxy[1], 0.0]), np.zeros(3))
        else:
            raise ValueError(mode)

        be.current_action = a_eff
        for _ in range(SUBSTEPS):
            be.substep(CTBR.rotor_forces(a_eff, be.state().w))

    modes = ["nominal_l2f", "swift_knn", "asap_delta_a"]
    out = {}
    for mode in modes:
        dr_sim = make_source(seed=0, overrides=ov if mode != "nominal_l2f" else {})
        be = dr_sim.backend
        lin_sq, ang_sq, drift_50 = [], [], []
        for S, A in held_flights:
            rpm = be.force_to_rpm(np.full(4, be.base_params["dynamics"]["mass"] * 9.81 / 4.0))
            for k in range(12, len(A)):
                s, s1 = S[k], S[k + 1]
                ae = A[k - d_hat] if mode != "nominal_l2f" else A[k - true_delay]
                be.reset(s, rpm=rpm)
                step_sim(be, mode, ae)
                pr = be.state()
                e_lin = (s1.v - pr.v) / RL_DT
                e_ang = (s1.w - pr.w) / RL_DT
                lin_sq.append(float(np.sum(e_lin ** 2) / 3.0))
                ang_sq.append(float(np.sum(e_ang ** 2) / 3.0))

                # Update reconstructed rpm using the mode's executed effective action `be.current_action`
                sp = be.force_to_rpm(CTBR.rotor_forces(be.current_action, s.w))
                tr, tf = be.motor_taus()
                for _ in range(SUBSTEPS):
                    tau = np.where(sp >= rpm, tr, tf)
                    al = np.exp(-SUB_DT / tau)
                    rpm = al * rpm + (1.0 - al) * sp

            # 0.5 s (50-tick) open-loop position rollout drift from k=15
            if len(A) >= 70:
                be.reset(S[15])
                for step_j in range(15, 65):
                    ae = A[step_j - (d_hat if mode != "nominal_l2f" else true_delay)]
                    step_sim(be, mode, ae)
                drift_50.append(float(np.linalg.norm(be.state().p - S[65].p)))

        out[mode] = {
            "lin_rmse": float(np.sqrt(np.mean(lin_sq))),
            "ang_rmse": float(np.sqrt(np.mean(ang_sq))),
            "drift_05s_m": float(np.mean(drift_50)) if drift_50 else float("nan"),
        }

    # Check MOPO epistemic uncertainty separation: in-distribution calibration vs OOD high-rate unseen queries
    Z_cal = models_dict["Z_cal"]
    in_sigs = [asap_ens.predict_one(Z_cal[i])[1] for i in range(min(150, len(Z_cal)))]
    rng = np.random.default_rng(42)
    ood_Z = Z_cal[:150].copy()
    ood_Z[:, 3:6] += rng.uniform(5.0, 8.0, size=(len(ood_Z), 3))  # unseen extreme body rates!
    ood_sigs = [asap_ens.predict_one(ood_Z[i])[1] for i in range(len(ood_Z))]
    out["mopo_uncertainty_separation"] = {
        "in_dist_sigma": float(np.mean(in_sigs)),
        "ood_sigma": float(np.mean(ood_sigs)),
        "ood_in_ratio": float(np.mean(ood_sigs) / max(np.mean(in_sigs), 1e-6)),
    }
    return out


def run_delta_f_zero_falsification(base_model, seed: int) -> dict:
    """Step 5 Riff R5.1: Verify both Swift k-NN and ASAP delta-action collapse to ~0 on a pure delay gap (`Delta f == 0, d = 3`)."""
    sysid_f, cal_f, _ = collect_target_and_heldout_flights(base_model, kf_scale=1.0, target_delay=3, seed=seed + 300, delta_f_zero=True)
    # Naive d' = 0 (unaligned) vs Step-0 aligned d_hat
    fS0 = OneStepSource(None)
    rs_naive, Xs_naive = zip(*[fS0.residuals(S, A, 0) for S, A in cal_f])
    Z_naive = feats(sum(Xs_naive, []), "body10")
    knn_naive = KNNResidual(k=5).fit(Z_naive, np.concatenate(rs_naive))
    Z_asap_naive, dA_naive = solve_asap_delta_actions(cal_f, None, 0)
    asap_naive_mag = float(np.mean(np.linalg.norm(dA_naive, axis=1)))
    knn_naive_mag = float(np.mean(np.linalg.norm(knn_naive.predict(Z_naive[:200])[:, :3], axis=1)))

    # Aligned via Step-0
    md = fit_all_residual_models(sysid_f, cal_f, seed + 300)
    _, dA_aligned = solve_asap_delta_actions(cal_f, md["tau_hat"], md["d_hat"])
    asap_aligned_mag = float(np.mean(np.linalg.norm(dA_aligned, axis=1)))
    knn_aligned_mag = float(np.mean(np.linalg.norm(md["knn"].predict(md["Z_cal"][:200])[:, :3], axis=1)))
    return {
        "d_hat": md["d_hat"],
        "knn_naive_ms2": knn_naive_mag,
        "knn_aligned_ms2": knn_aligned_mag,
        "asap_naive_da": asap_naive_mag,
        "asap_aligned_da": asap_aligned_mag,
    }


def run_seed(seed: int):
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    t0 = time.time()
    base_path = f"{OUT}/exp10_base_nom_s{seed}.zip"
    if not os.path.exists(base_path):
        # Fallback if base checkpoint wasn't saved yet
        venv_b = DummyVecEnv(
            [lambda i=i: GateRacingEnv("figure8", "source", HIST, lam_da=0.02, delays=(0,), max_ep_s=5.5, seed=1000 * seed + i) for i in range(8)]
        )
        bm = PPO(
            "MlpPolicy",
            venv_b,
            n_steps=512,
            batch_size=512,
            n_epochs=10,
            learning_rate=3e-4,
            seed=seed,
            verbose=0,
            policy_kwargs=dict(net_arch=[128, 128], log_std_init=-0.7),
        )
        bm.learn(450_000)
        bm.save(base_path)

    base_model = PPO.load(base_path, device="cpu")
    df0 = run_delta_f_zero_falsification(base_model, seed)

    targets = {
        "target_nominal": (1.0, 2),
        "target_perturbed": (1.20, 3),
    }
    fidelity = {}
    closed_loop = {}

    for tname, (kfs, d_true) in targets.items():
        sysid_f, cal_f, held_f = collect_target_and_heldout_flights(base_model, kfs, d_true, seed + (0 if kfs == 1.0 else 40))
        md = fit_all_residual_models(sysid_f, cal_f, seed + (0 if kfs == 1.0 else 40))
        fidelity[tname] = eval_forward_fidelity(md, held_f, d_true)
        fidelity[tname]["d_hat"] = md["d_hat"]

        # Train 4 fine-tuned PPO arms for equal 100k source steps from `base_path`
        n_envs = 8
        d_h, t_h = md["d_hat"], md["tau_hat"]
        knn_hk = make_knn_residual_hook(md["knn"])

        arms_env_builders = {
            "swift_knn_ft": lambda i: GateRacingEnv(
                "figure8", "source", HIST, lam_da=0.20, delays=(d_h,), residual_fn=knn_hk, tau_hat=t_h, max_ep_s=5.5, seed=5000 * seed + i
            ),
            "swift_mopo_ft": lambda i: AugmentedRacingEnv(
                "figure8", "source", HIST, lam_da=0.20, delays=(d_h,), acc_ens=md["acc_ens"], mopo_beta=0.35, tau_hat=t_h, max_ep_s=5.5, seed=5100 * seed + i
            ),
            "asap_delta_a_ft": lambda i: AugmentedRacingEnv(
                "figure8", "source", HIST, lam_da=0.20, delays=(d_h,), asap_ens=md["asap_ens"], xy_drag_ens=md["xy_ens"], mopo_beta=0.0, tau_hat=t_h, max_ep_s=5.5, seed=5200 * seed + i
            ),
            "asap_mopo_ft": lambda i: AugmentedRacingEnv(
                "figure8", "source", HIST, lam_da=0.20, delays=(d_h,), asap_ens=md["asap_ens"], xy_drag_ens=md["xy_ens"], mopo_beta=0.35, tau_hat=t_h, max_ep_s=5.5, seed=5300 * seed + i
            ),
        }

        trained_pols = {"zeroshot_ppo": SB3PolicyCallable(PPO.load(base_path, device="cpu"))}
        for arm_name, b_fn in arms_env_builders.items():
            venv = DummyVecEnv([lambda i=i, bf=b_fn: bf(i) for i in range(n_envs)])
            mdl = PPO.load(base_path, env=venv, device="cpu")
            mdl.learn(FT_STEPS, reset_num_timesteps=False)
            trained_pols[arm_name] = SB3PolicyCallable(mdl)

        closed_loop[tname] = {}
        for cname in ["figure8", "ellipse3d"]:
            ev_tgt = GateRacingEnv(cname, "target", HIST, delays=(d_true,), kf_scale=kfs, random_start_gate=False, max_ep_s=12.5, seed=950 + seed)
            closed_loop[tname][cname] = {
                aname: evaluate_racing(pol, ev_tgt, n_eval=10) for aname, pol in trained_pols.items()
            }

    return {"seed": seed, "wall_s": time.time() - t0, "df0": df0, "fidelity": fidelity, "closed_loop": closed_loop}


def agg_leaf(ds):
    out = {}
    for k in ds[0]:
        vals = [d[k] for d in ds if np.isfinite(d[k])]
        out[k] = mean_std(vals) if vals else {"mean": float("nan"), "std": float("nan"), "n": 0, "values": []}
    return out


def main():
    t0 = time.time()
    with mp.get_context("spawn").Pool(3) as pool:
        outs = pool.map(run_seed, SEEDS)
    wall = time.time() - t0
    print(f"Exp 14 (ASAP + MOPO vs Swift k-NN) finished in {wall:.1f}s")

    agg_df0 = agg_leaf([o["df0"] for o in outs])
    agg_fid = {}
    for tname in outs[0]["fidelity"]:
        agg_fid[tname] = {}
        for k in outs[0]["fidelity"][tname]:
            if isinstance(outs[0]["fidelity"][tname][k], dict):
                agg_fid[tname][k] = agg_leaf([o["fidelity"][tname][k] for o in outs])
            else:
                agg_fid[tname][k] = mean_std([o["fidelity"][tname][k] for o in outs])

    agg_cl = {
        tname: {
            cname: {
                aname: agg_leaf([o["closed_loop"][tname][cname][aname] for o in outs])
                for aname in outs[0]["closed_loop"][tname][cname]
            }
            for cname in outs[0]["closed_loop"][tname]
        }
        for tname in outs[0]["closed_loop"]
    }

    save_json(
        "exp14_asap_mopo_residuals.json",
        {"seeds": SEEDS, "ft_steps": FT_STEPS, "wall_s": wall, "delta_f_zero_control": agg_df0, "fidelity": agg_fid, "closed_loop": agg_cl},
    )

    print("\n=== Step 5 Control (Delta f == 0, d=3): Phantom Residual Falsification ===")
    print(
        f"  d_hat={agg_df0['d_hat']['mean']:.2f} | k-NN naive={agg_df0['knn_naive_ms2']['mean']:.3f} -> aligned={agg_df0['knn_aligned_ms2']['mean']:.4f} m/s^2 | "
        f"ASAP naive ||da||={agg_df0['asap_naive_da']['mean']:.3f} -> aligned={agg_df0['asap_aligned_da']['mean']:.4f}"
    )

    for tname in agg_fid:
        print(f"\n=== Forward Model Fidelity on Held-Out `ellipse3d` [{tname}] ===")
        for m in ["nominal_l2f", "swift_knn", "asap_delta_a"]:
            f = agg_fid[tname][m]
            print(
                f"  {m:15s} lin_RMSE={fmt(f['lin_rmse']):>13} m/s^2 | ang_RMSE={fmt(f['ang_rmse']):>13} rad/s^2 | "
                f"0.5s_drift={fmt(f['drift_05s_m']):>13} m"
            )
        u = agg_fid[tname]["mopo_uncertainty_separation"]
        print(
            f"  MOPO sigma_ep: in_dist={u['in_dist_sigma']['mean']:.4f} vs OOD={u['ood_sigma']['mean']:.4f} "
            f"({u['ood_in_ratio']['mean']:.2f}x separation)"
        )

    for tname in agg_cl:
        for cname in ["figure8", "ellipse3d"]:
            print(f"\n=== Closed-Loop 6-Gate Racing [{tname} | {cname}] ===")
            for aname, m in agg_cl[tname][cname].items():
                lap_s = m["median_lap_s"]["mean"]
                lap_str = f"{lap_s:5.2f}s" if np.isfinite(lap_s) else "  n/a "
                print(
                    f"  {aname:16s} gf={fmt(m['gate_frac']):>13} | cln={m['clean_completion']['mean']:.2f} | "
                    f"cr={m['crash_rate']['mean']:.2f} | trk={m['track_err_m']['mean']:.3f}m | lap={lap_str}"
                )

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    # Panel A: Angular RMSE comparison (Nominal vs Swift k-NN vs ASAP Delta-Action)
    f_modes = ["nominal_l2f", "swift_knn", "asap_delta_a"]
    xf = np.arange(len(f_modes))
    for j, tname in enumerate(["target_nominal", "target_perturbed"]):
        axes[0].bar(
            xf + (j - 0.5) * 0.34,
            [agg_fid[tname][m]["ang_rmse"]["mean"] for m in f_modes],
            0.34,
            yerr=[agg_fid[tname][m]["ang_rmse"]["std"] for m in f_modes],
            capsize=3,
            label=tname,
        )
    axes[0].set_xticks(xf, ["Nominal\nL2F", "Swift\nk-NN (6D)", "ASAP\nDelta-Action"])
    axes[0].set_ylabel("Held-Out Angular Accel RMSE [rad/s^2]")
    axes[0].set_title("A. Actuator/Angular Fidelity (ellipse3d)")
    axes[0].legend(fontsize=8)

    # Panels B & C: Closed-loop Held-Out `ellipse3d` Gate Completion on Nominal & Perturbed Targets
    arms = ["zeroshot_ppo", "swift_knn_ft", "swift_mopo_ft", "asap_delta_a_ft", "asap_mopo_ft"]
    xa = np.arange(len(arms))
    for idx_ax, tname in [(1, "target_nominal"), (2, "target_perturbed")]:
        g = agg_cl[tname]["ellipse3d"]
        axes[idx_ax].bar(
            xa,
            [g[a]["gate_frac"]["mean"] for a in arms],
            0.58,
            yerr=[g[a]["gate_frac"]["std"] for a in arms],
            capsize=3,
            color=["#888888", "#4C72B0", "#55A868", "#DD8452", "#C44E52"],
        )
        axes[idx_ax].set_xticks(xa, [a.replace("_ppo", "").replace("_ft", "").replace("_", "\n") for a in arms], fontsize=7.5)
        axes[idx_ax].set_ylim(0, 1.08)
        axes[idx_ax].set_ylabel("Gate Completion Fraction")
        axes[idx_ax].set_title(f"{'B' if idx_ax == 1 else 'C'}. Held-Out Racing ({tname})")

    plt.tight_layout()
    plt.savefig(f"{OUT}/exp14_asap_mopo_residuals.png", dpi=150)


if __name__ == "__main__":
    main()
