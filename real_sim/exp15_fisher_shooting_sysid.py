"""
Exp 15 (Step 0 Riffs R0.1 + R0.2):
  - Part A (R0.1 — Fisher-Optimal Safe Excitation Design):
      Optimize multi-harmonic CTBR excitation amplitudes/frequencies/phases purely in `L2F` BEFORE touching the
      target to maximize D-optimality `log det(Phi^T Phi)` of the joint `(delay d, tau_up, tau_down, thrust_gain)`
      information matrix subject to a `0.45 m` hover safety sphere.
      Evaluate on an Asymmetric + Perturbed PyBullet target (`tau_up = 20 ms, tau_down = 80 ms, kf x 1.20, d = 2`)
      across short calibration windows `T_cal in {0.5s, 1.0s, 2.0s, 3.5s}` vs. hand-picked multi-sine (Exp 9)
      and damped Gaussian exploration!
  - Part B (R0.2 — Multi-Step 10-Tick Shooting Parametric Alignment):
      Optimize continuous `L2F` physical parameters `(mass_scale, thrust_scale, tau_rise_scale, tau_fall_scale,
      lin_drag)` by minimizing 10-tick (`100 ms`) open-loop position + velocity + body-rate shooting error on
      calibration rollouts, and measure 3.0-second (`300-tick`) open-loop trajectory divergence on held-out
      3D `ellipse3d` (exact Exp 1 protocol) with and without k-NN residuals!
"""

from __future__ import annotations

import contextlib
import io
import math
import multiprocessing as mp
import time

import numpy as np
from scipy.optimize import minimize

from common import OUT, fmt, mean_std, plt, save_json
from exp6_knn_residual import KNNResidual, OneStepSource, feats
from exp9_residual_finetune import scripted_sysid_action
from sims.drone_pair import (
    Course,
    CTBRConfig,
    DroneState,
    GRAVITY,
    GeometricExpert,
    InterfaceConfig,
    L2FBackend,
    RL_DT,
    SUB_DT,
    SUBSTEPS,
    make_source,
    make_target,
)

SEEDS = [0, 1, 2]
CTBR = CTBRConfig()


# --------------------------------------------------------------------------------------------------
# Part A: Pre-Deployment Fisher D-Optimal Safe Excitation Synthesis in L2F
# --------------------------------------------------------------------------------------------------


def eval_excitation_params(params_flat: np.ndarray, k: int, dt: float, st_obs: DroneState, p0: np.ndarray) -> np.ndarray:
    """Gentle 0.4 Hz outer hover leash + optimized 3-harmonic CTBR excitation (`[1.7, 3.5, 6.2] Hz`)."""
    p = params_flat.reshape(4, 3, 2)
    freqs = np.array([1.7, 3.5, 6.2])
    t = k * dt
    da = np.zeros(4, dtype=np.float64)
    for ch in range(4):
        for h_i in range(3):
            amp, ph = p[ch, h_i]
            da[ch] += amp * math.sin(2.0 * math.pi * freqs[h_i] * t + ph)
    leash = GeometricExpert(kp=2.2, kd=1.8, k_att=4.5)(st_obs, p0, np.zeros(3), np.zeros(3))
    return np.clip(leash + da, -1.0, 1.0)


def design_fisher_optimal_excitation(seed: int = 0, n_candidates: int = 90, horizon_ticks: int = 200) -> np.ndarray:
    """
    Search purely in `L2F` for safe harmonic parameters `theta_exc` maximizing D-optimality
    `log det(1/N * Phi^T Phi)` of `[u_rise_sensitivity, u_fall_sensitivity, du_dt, v_B, w_B]`
    subject to `max ||p(t) - p0|| <= 0.50 m` and `R[2,2] >= 0.70`.
    """
    rng = np.random.default_rng(seed + 10)
    dr = make_source(seed=seed)
    be = dr.backend
    p0 = np.array([0.0, 0.0, 2.0])

    best_score = -1e18
    best_params = None

    for c_i in range(n_candidates):
        if best_params is None or c_i < 35:
            amps = np.column_stack(
                [
                    rng.uniform(0.06, 0.28, size=(4, 2)),
                    rng.uniform(0.05, 0.22, size=(4, 1)),
                ]
            )
            amps[3] *= 0.6
            phases = rng.uniform(0.0, 2.0 * math.pi, size=(4, 3))
            cand = np.stack([amps, phases], axis=-1).reshape(-1)
        else:
            cand = best_params + rng.normal(0.0, 0.05, size=best_params.shape)
            cand_3d = cand.reshape(4, 3, 2)
            cand_3d[:, :, 0] = np.clip(cand_3d[:, :, 0], 0.02, 0.32)
            cand = cand_3d.reshape(-1)

        obs_st = dr.reset(DroneState.hover(p=p0))
        rpm_fast = be.force_to_rpm(np.full(4, be.base_params["dynamics"]["mass"] * GRAVITY / 4.0))
        rpm_slow = rpm_fast.copy()
        safe = True
        rows = []
        u_prev = 0.0
        for k in range(horizon_ticks):
            a = eval_excitation_params(cand, k, RL_DT, obs_st, p0)
            obs_st = dr.step(a)
            st = dr.true_state()
            if float(np.linalg.norm(st.p - p0)) > 0.50 or st.R[2, 2] < 0.70:
                safe = False
                break

            sp = be.force_to_rpm(CTBR.rotor_forces(a, st.w))
            # Track spin-up vs spin-down filter bank sensitivities!
            is_rising = float(np.mean(sp >= rpm_fast))
            rpm_fast = 0.90 * rpm_fast + 0.10 * sp
            rpm_slow = 0.97 * rpm_slow + 0.03 * sp
            u_f = float(np.sum(rpm_fast ** 2))
            u_s = float(np.sum(rpm_slow ** 2))
            du = (u_f - u_prev) / RL_DT
            u_prev = u_f
            if k >= 8:
                rows.append(
                    np.concatenate(
                        [[u_f * is_rising, u_s * (1.0 - is_rising), du], st.R.T @ st.v, st.w]
                    )
                )

        if safe and len(rows) >= 50:
            Phi = np.array(rows, dtype=np.float64)
            Phi_s = Phi / (np.std(Phi, axis=0, keepdims=True) + 1e-4)
            cov = (Phi_s.T @ Phi_s) / len(Phi_s) + 1e-4 * np.eye(Phi.shape[1])
            _, logdet = np.linalg.slogdet(cov)
            if float(logdet) > best_score:
                best_score = float(logdet)
                best_params = cand.copy()

    return best_params


def identify_asymmetric_sysid(S: list[DroneState], A: np.ndarray) -> tuple[int, float, float]:
    """
    Jointly identify `(d_hat, tau_up_hat, tau_down_hat)` over candidate grid using asymmetric rotor filter!
    """
    tau_ups = [0.015, 0.020, 0.030, 0.045]
    tau_downs = [0.020, 0.040, 0.080, 0.120]
    be = L2FBackend()
    c0, c1, c2 = be.thrust_poly
    hover_rpm = be.force_to_rpm(np.full(4, be.base_params["dynamics"]["mass"] * GRAVITY / 4.0))
    hover_a = CTBR.encode(GRAVITY, np.zeros(3))

    # Precompute target world->body z-acceleration + angular rate changes
    N = len(A)
    best = None
    for d in range(5):
        a_exec = [A[k - d] if k - d >= 0 else hover_a for k in range(N)]
        for tu in tau_ups:
            al_u = math.exp(-SUB_DT / tu)
            for td in tau_downs:
                al_d = math.exp(-SUB_DT / td)
                rpm = hover_rpm.copy()
                uf_list, vB_list, az_meas = [], [], []
                for k in range(N):
                    s, s1 = S[k], S[k + 1]
                    sp = be.force_to_rpm(CTBR.rotor_forces(a_exec[k], s.w))
                    f_sub = 0.0
                    for _ in range(SUBSTEPS):
                        al = np.where(sp >= rpm, al_u, al_d)
                        rpm = al * rpm + (1.0 - al) * sp
                        f_sub += float(np.sum(c0 + c1 * rpm + c2 * rpm * rpm))
                    if k >= 6:
                        uf_list.append(f_sub / SUBSTEPS)
                        vB_list.append(s.R.T @ s.v)
                        az_meas.append(float((s.R.T @ ((s1.v - s.v) / RL_DT + np.array([0.0, 0.0, GRAVITY])))[2]))
                Phi = np.column_stack([np.ones(len(uf_list)), uf_list, vB_list])
                y = np.array(az_meas)
                th, *_ = np.linalg.lstsq(Phi, y, rcond=None)
                rms = float(np.sqrt(np.mean((y - Phi @ th) ** 2)))
                if best is None or rms < best[0]:
                    best = (rms, d, tu, td)
    return int(best[1]), float(best[2]), float(best[3])


def run_part_a_seed(seed: int) -> dict:
    fisher_params = design_fisher_optimal_excitation(seed=seed)
    rng = np.random.default_rng(seed + 200)
    # Asymmetric + Perturbed PyBullet Target: true d=2, tau_up=20ms, tau_down=80ms, kf x 1.20!
    d_true, tu_true, td_true = 2, 0.020, 0.080
    with contextlib.redirect_stdout(io.StringIO()):
        tgt = make_target(tau_m=tu_true, tau_down=td_true, iface=InterfaceConfig(action_delay=d_true), seed=seed + 10)
    tgt.backend.env.KF *= 1.20

    windows_s = [0.5, 1.0, 2.0, 3.5]
    strategies = ["gaussian_walk", "hand_multisine", "fisher_optimal"]
    out = {st: {} for st in strategies}

    for st_name in strategies:
        for w_s in windows_s:
            n_ticks = int(round(w_s / RL_DT))
            d_exact, tau_errs_ms, max_radii = [], [], []
            for trial in range(6):
                p0 = np.array([0.0, 0.0, 2.0])
                obs_cur = tgt.reset(DroneState.hover(p=p0))
                S, A = [tgt.true_state()], []
                phase = float(rng.uniform(0, 2.0 * math.pi))
                g_noise = np.zeros(4)
                leash_ex = GeometricExpert(kp=2.2, kd=1.8, k_att=4.5)
                max_r = 0.0
                for k in range(n_ticks):
                    if st_name == "gaussian_walk":
                        # Correlated colored Gaussian exploration (poor high-freq spin-up/down separation & larger drift!)
                        g_noise = 0.88 * g_noise + 0.12 * rng.normal(0.0, 0.45, 4)
                        a_l = leash_ex(obs_cur, p0, np.zeros(3), np.zeros(3))
                        a = np.clip(a_l + g_noise, -1.0, 1.0)
                    elif st_name == "hand_multisine":
                        a_l = leash_ex(obs_cur, p0, np.zeros(3), np.zeros(3))
                        da_ms = scripted_sysid_action(k, RL_DT, tgt.hover_action, phase) - tgt.hover_action
                        a = np.clip(a_l + da_ms, -1.0, 1.0)
                    else:
                        a = eval_excitation_params(fisher_params, k + 7 * trial, RL_DT, obs_cur, p0)
                    obs_cur = tgt.step(a)
                    st_now = tgt.true_state()
                    max_r = max(max_r, float(np.linalg.norm(st_now.p - p0)))
                    S.append(st_now)
                    A.append(a)

                dh, tuh, tdh = identify_asymmetric_sysid(S, np.array(A))
                d_exact.append(float(dh == d_true))
                tau_err = 1000.0 * math.sqrt(0.5 * ((tuh - tu_true) ** 2 + (tdh - td_true) ** 2))
                tau_errs_ms.append(tau_err)
                max_radii.append(max_r)

            out[st_name][str(w_s)] = {
                "d_exact_rate": float(np.mean(d_exact)),
                "tau_rmse_ms": float(np.mean(tau_errs_ms)),
                "max_radius_m": float(np.mean(max_radii)),
            }
    return out


# --------------------------------------------------------------------------------------------------
# Part B: Multi-Step 10-Tick Shooting Parametric Alignment vs. 1-Step Grid (Exp 1 3s Replay Protocol)
# --------------------------------------------------------------------------------------------------


def fit_multistep_shooting_params(cal_flights, d_hat: int, horizon_h: int = 10) -> dict:
    """
    Optimize `[thrust_to_mass_scale, tau_rise_s, tau_fall_s, lin_drag_xy, lin_drag_z]` via Powell minimization
    of 10-tick (`100 ms`) open-loop shooting rollout velocity + position error in `L2FBackend`!
    """
    be = L2FBackend()
    tr0, tf0 = be.motor_taus()
    hover_a = CTBR.encode(GRAVITY, np.zeros(3))

    # Subsample 28 disjoint 10-tick windows across calibration flights for fast ~1.5s optimization!
    windows = []
    for S, A in cal_flights:
        a_exec = [A[k - d_hat] if k - d_hat >= 0 else hover_a for k in range(len(A))]
        for k0 in range(12, len(A) - horizon_h, 15):
            windows.append((S[k0 : k0 + horizon_h + 1], a_exec[k0 : k0 + horizon_h]))
            if len(windows) >= 28:
                break
        if len(windows) >= 28:
            break

    def loss_fn(x):
        tm_scale = float(np.clip(x[0], 0.65, 2.20))
        tau_r = float(np.clip(x[1], 0.008, 0.120))
        tau_f = float(np.clip(x[2], 0.008, 0.250))
        drag_xy = float(np.clip(x[3], 0.0, 1.5))
        drag_z = float(np.clip(x[4], 0.0, 1.5))

        be.set_dynamics(
            {
                "thrust_scale": tm_scale,
                "tau_rise_scale": tau_r / tr0,
                "tau_fall_scale": tau_f / tf0,
            }
        )
        err_sum = 0.0
        for S_w, A_w in windows:
            be.reset(S_w[0])
            for step_i, ae in enumerate(A_w):
                for _ in range(SUBSTEPS):
                    st_s = be.state()
                    be.substep(CTBR.rotor_forces(ae, st_s.w))
                    # Apply linear body/world drag term
                    ns_st = be.state()
                    v_d = ns_st.v - SUB_DT * np.array([drag_xy * ns_st.v[0], drag_xy * ns_st.v[1], drag_z * ns_st.v[2]])
                    be.s.linear_velocity = list(v_d)
                st_pred = be.state()
                st_true = S_w[step_i + 1]
                err_sum += float(np.sum((st_pred.p - st_true.p) ** 2) * 4.0 + np.sum((st_pred.v - st_true.v) ** 2))
        return err_sum / max(len(windows), 1)

    res = minimize(
        loss_fn,
        x0=np.array([1.25, 0.030, 0.035, 0.15, 0.15]),
        method="Powell",
        options={"maxiter": 45, "ftol": 1e-3, "xtol": 1e-2},
    )
    x = res.x
    return {
        "thrust_scale": float(np.clip(x[0], 0.65, 2.20)),
        "tau_rise_s": float(np.clip(x[1], 0.008, 0.120)),
        "tau_fall_s": float(np.clip(x[2], 0.008, 0.250)),
        "drag_xy": float(np.clip(x[3], 0.0, 1.5)),
        "drag_z": float(np.clip(x[4], 0.0, 1.5)),
    }


def eval_3s_open_loop_replay(seed: int, kf_scale: float = 1.15) -> dict:
    """Compare 1s, 2s, and 3s open-loop replay drift on held-out `ellipse3d` (exact Exp 1 protocol!)."""
    rng = np.random.default_rng(seed + 400)
    ex = GeometricExpert()
    train_c = Course(kind="figure8")
    held_c = Course(kind="ellipse3d")

    with contextlib.redirect_stdout(io.StringIO()):
        tgt = make_target(tau_m=0.03, tau_down=0.07, iface=InterfaceConfig(action_delay=2), seed=seed + 401)
    tgt.backend.env.KF *= kf_scale

    # Collect 3 x 4s calibration flights on `figure8`
    cal_flights = []
    for ep in range(3):
        t0 = float(rng.uniform(0.0, train_c.period))
        p0, v0, _ = train_c.ref(t0)
        tgt.reset(DroneState(p0, np.eye(3), v0, np.zeros(3)))
        S, A = [tgt.true_state()], []
        for k in range(400):
            a = np.clip(ex(tgt.true_state(), *train_c.ref(t0 + k * RL_DT)) + 0.12 * rng.standard_normal(4), -1.0, 1.0)
            tgt.step(a)
            S.append(tgt.true_state())
            A.append(a)
        cal_flights.append((S, np.array(A)))

    d_hat, tu_grid, _ = identify_asymmetric_sysid(cal_flights[0][0], cal_flights[0][1])
    shoot_p = fit_multistep_shooting_params(cal_flights, d_hat)

    # Fit standard Step-0 k-NN AND Post-Shooting k-NN!
    fS_step0 = OneStepSource(tu_grid)
    rs0, Xs0 = zip(*[fS_step0.residuals(S, A, d_hat) for S, A in cal_flights])
    knn_step0 = KNNResidual(k=5).fit(feats(sum(Xs0, []), "body10"), np.concatenate(rs0))

    # Compute residual AFTER parametric shooting model
    be_sh = L2FBackend()
    tr0, tf0 = be_sh.motor_taus()
    be_sh.set_dynamics(
        {
            "thrust_scale": shoot_p["thrust_scale"],
            "tau_rise_scale": shoot_p["tau_rise_s"] / tr0,
            "tau_fall_scale": shoot_p["tau_fall_s"] / tf0,
        }
    )
    hover_a = CTBR.encode(GRAVITY, np.zeros(3))
    X_sh, R_sh = [], []
    for S, A in cal_flights:
        rpm = be_sh.force_to_rpm(np.full(4, be_sh.base_params["dynamics"]["mass"] * GRAVITY / 4.0))
        for k in range(len(A)):
            s, s1 = S[k], S[k + 1]
            ae = A[k - d_hat] if k - d_hat >= 0 else hover_a
            if k >= 10:
                be_sh.reset(s, rpm=rpm)
                for _ in range(SUBSTEPS):
                    be_sh.substep(CTBR.rotor_forces(ae, be_sh.state().w))
                    ns = be_sh.state()
                    be_sh.s.linear_velocity = list(
                        ns.v - SUB_DT * np.array([shoot_p["drag_xy"] * ns.v[0], shoot_p["drag_xy"] * ns.v[1], shoot_p["drag_z"] * ns.v[2]])
                    )
                pr = be_sh.state()
                r_lin = s.R.T @ (s1.v - pr.v) / RL_DT
                r_ang = (s1.w - pr.w) / RL_DT
                X_sh.append((s, ae))
                R_sh.append(np.concatenate([r_lin, r_ang]))
            sp = be_sh.force_to_rpm(CTBR.rotor_forces(ae, s.w))
            for _ in range(SUBSTEPS):
                tau = np.where(sp >= rpm, shoot_p["tau_rise_s"], shoot_p["tau_fall_s"])
                al = np.exp(-SUB_DT / tau)
                rpm = al * rpm + (1.0 - al) * sp
    knn_shoot = KNNResidual(k=5).fit(feats(X_sh, "body10"), np.array(R_sh))

    # Evaluate 3-second open-loop replay on 4 held-out `ellipse3d` trajectories!
    modes = ["unaligned_l2f", "step0_knn", "shooting_param_only", "shooting_plus_knn"]
    drifts = {m: {1.0: [], 2.0: [], 3.0: []} for m in modes}

    for ep in range(4):
        t0 = 0.25 * ep * held_c.period
        p0, v0, _ = held_c.ref(t0)
        st_init = DroneState(p0, np.eye(3), v0, np.zeros(3))
        tgt.reset(st_init)
        S_gt, A_gt = [tgt.true_state()], []
        for k in range(300):
            a = ex(tgt.true_state(), *held_c.ref(t0 + k * RL_DT))
            tgt.step(a)
            S_gt.append(tgt.true_state())
            A_gt.append(a)

        for mode in modes:
            if mode == "unaligned_l2f":
                sim = make_source(iface=InterfaceConfig(action_delay=0))
            elif mode == "step0_knn":
                sim = make_source(
                    overrides={"tau_rise_scale": tu_grid / tr0, "tau_fall_scale": tu_grid / tf0},
                    iface=InterfaceConfig(action_delay=d_hat),
                )
                sim.backend.residual_fn = lambda st, a: (
                    knn_step0.predict(feats([(st, a)], "body10"))[0][:3],
                    knn_step0.predict(feats([(st, a)], "body10"))[0][3:],
                )
            elif mode == "shooting_param_only":
                sim = make_source(
                    overrides={
                        "thrust_scale": shoot_p["thrust_scale"],
                        "tau_rise_scale": shoot_p["tau_rise_s"] / tr0,
                        "tau_fall_scale": shoot_p["tau_fall_s"] / tf0,
                    },
                    iface=InterfaceConfig(action_delay=d_hat),
                )
            else:
                sim = make_source(
                    overrides={
                        "thrust_scale": shoot_p["thrust_scale"],
                        "tau_rise_scale": shoot_p["tau_rise_s"] / tr0,
                        "tau_fall_scale": shoot_p["tau_fall_s"] / tf0,
                    },
                    iface=InterfaceConfig(action_delay=d_hat),
                )
                sim.backend.residual_fn = lambda st, a: (
                    knn_shoot.predict(feats([(st, a)], "body10"))[0][:3],
                    knn_shoot.predict(feats([(st, a)], "body10"))[0][3:],
                )

            sim.reset(st_init)
            for k in range(300):
                sim.act_fifo.append(A_gt[k])
                ae = sim.act_fifo[0]
                sim.backend.current_action = ae
                for _ in range(SUBSTEPS):
                    sim.backend.substep(sim.ctbr.rotor_forces(ae, sim.backend.state().w))
                    if "shooting" in mode:
                        ns = sim.backend.state()
                        sim.backend.s.linear_velocity = list(
                            ns.v - SUB_DT * np.array([shoot_p["drag_xy"] * ns.v[0], shoot_p["drag_xy"] * ns.v[1], shoot_p["drag_z"] * ns.v[2]])
                        )
                step_s = (k + 1) * RL_DT
                if step_s in drifts[mode]:
                    drifts[mode][step_s].append(float(np.linalg.norm(sim.true_state().p - S_gt[k + 1].p)))

    return {
        "fitted_params": shoot_p,
        "drifts": {m: {str(ts): float(np.mean(drifts[m][ts])) for ts in [1.0, 2.0, 3.0]} for m in modes},
    }


def job(seed: int):
    t0 = time.time()
    pa = run_part_a_seed(seed)
    pb = eval_3s_open_loop_replay(seed)
    return {"seed": seed, "wall_s": time.time() - t0, "part_a": pa, "part_b": pb}


def main():
    t0 = time.time()
    with mp.get_context("spawn").Pool(3) as pool:
        outs = pool.map(job, SEEDS)
    wall = time.time() - t0
    print(f"Exp 15 (Fisher Safe SysID + Multi-Step Shooting) finished in {wall:.1f}s")

    agg_a = {}
    for st in outs[0]["part_a"]:
        agg_a[st] = {}
        for w_s in outs[0]["part_a"][st]:
            agg_a[st][w_s] = {
                k: mean_std([o["part_a"][st][w_s][k] for o in outs]) for k in outs[0]["part_a"][st][w_s]
            }

    agg_b = {}
    for m in outs[0]["part_b"]["drifts"]:
        agg_b[m] = {
            ts: mean_std([o["part_b"]["drifts"][m][ts] for o in outs]) for ts in outs[0]["part_b"]["drifts"][m]
        }

    save_json(
        "exp15_fisher_shooting_sysid.json",
        {"seeds": SEEDS, "wall_s": wall, "part_a_fisher_excitation": agg_a, "part_b_shooting_replay": agg_b, "fitted_params": [o["part_b"]["fitted_params"] for o in outs]},
    )

    print("\n=== Part A: Asymmetric Target SysID (true d=2, tau_up=20ms, tau_down=80ms, kf=1.20) ===")
    for st in agg_a:
        for w_s, m in agg_a[st].items():
            print(
                f"  {st:16s} T={w_s}s: d_exact={fmt(m['d_exact_rate']):>13} | tau_RMSE={fmt(m['tau_rmse_ms']):>13} ms | max_r={m['max_radius_m']['mean']:.2f}m"
            )

    print("\n=== Part B: Held-Out `ellipse3d` Open-Loop Replay Drift [m] ===")
    for m, d in agg_b.items():
        print(f"  {m:20s}: 1s={fmt(d['1.0'])} | 2s={fmt(d['2.0'])} | 3s={fmt(d['3.0'])}")

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))
    ws = [float(w) for w in agg_a["fisher_optimal"]]
    for st, mk in [("gaussian_walk", "o"), ("hand_multisine", "s"), ("fisher_optimal", "D")]:
        axes[0].errorbar(
            ws,
            [agg_a[st][str(w)]["d_exact_rate"]["mean"] for w in ws],
            yerr=[agg_a[st][str(w)]["d_exact_rate"]["std"] for w in ws],
            marker=mk,
            capsize=2,
            label=st,
        )
        axes[1].errorbar(
            ws,
            [agg_a[st][str(w)]["tau_rmse_ms"]["mean"] for w in ws],
            yerr=[agg_a[st][str(w)]["tau_rmse_ms"]["std"] for w in ws],
            marker=mk,
            capsize=2,
            label=st,
        )
    axes[0].set_title("A. Exact Delay Recovery vs Flight Window")
    axes[0].set_xlabel("Target Excitation Window T_cal [s]")
    axes[0].set_ylabel("P(d_hat == d_true)")
    axes[0].set_ylim(-0.05, 1.08)
    axes[0].legend(fontsize=7.5)

    axes[1].set_title("B. Asymmetric Rotor-Lag RMSE [ms]")
    axes[1].set_xlabel("Target Excitation Window T_cal [s]")
    axes[1].set_ylabel("RMSE(tau_up, tau_down) [ms]")
    axes[1].legend(fontsize=7.5)

    ts_x = [1.0, 2.0, 3.0]
    for m, mk in [("unaligned_l2f", "o"), ("step0_knn", "^"), ("shooting_param_only", "s"), ("shooting_plus_knn", "D")]:
        axes[2].errorbar(
            ts_x,
            [agg_b[m][str(t)]["mean"] for t in ts_x],
            yerr=[agg_b[m][str(t)]["std"] for t in ts_x],
            marker=mk,
            capsize=2,
            label=m,
        )
    axes[2].set_title("C. Open-Loop 3s Replay Drift (ellipse3d)")
    axes[2].set_xlabel("Open-Loop Rollout Horizon [s]")
    axes[2].set_ylabel("Position Drift [m]")
    axes[2].legend(fontsize=7.5)

    plt.tight_layout()
    plt.savefig(f"{OUT}/exp15_fisher_shooting_sysid.png", dpi=150)


if __name__ == "__main__":
    main()
