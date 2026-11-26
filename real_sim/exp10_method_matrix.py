"""
Exp 10 (P3 Headline Method Matrix & Robustness Curves):
Full 16-831 Proposal Method Matrix on 6-Gate 3D Drone Racing (`rl-tools/l2f` -> `gym-pybullet-drones` CF2X).

Evaluates across:
  - Methods (equal source interaction budget per seed: `base_steps + ft_steps`):
      1. classical_expert   (Time-free spatial `GeometricGateController` baseline)
      2. nominal_ppo        (`base_steps` lam=0.02 + `ft_steps` warm-start lam=0.20 in nominal L2F)
      3. dr_ppo             (`base_steps` + `ft_steps` warm-start with dynamics + latency randomization d~U{0..4})
      4. ff_student         (Feed-forward MLP teacher-cloning baseline for sensor-limited regime)
      5. gru_student_fixed  (Recurrent GRU student DAgger-distilled on fixed nominal L2F — Thm 2.4 ablation)
      6. gru_student_rand   (Recurrent GRU student DAgger-distilled over randomized L2F + delays + dropout/bias)
      7. residual_ft_ppo    (`base_steps` PPO + 4-flight target calibration [Step-0 multi-sine SysID (d_hat, tau_hat)
                             + 6D body-frame k-NN dynamics residual + ground-truth-pose GP gate-bias residual]
                             + `ft_steps` PPO fine-tuning in residual-augmented L2F)
      8. target_oracle_ppo  (In-domain reference: `base_steps` + `ft_steps` PPO fine-tuned directly in PyBullet target)
  - Observation Regimes:
      * exact_state         (SE(3)-invariant body-frame corridor + gate vectors, v_B, attitude, w_B, a_{t-8:t-1})
      * sensor_limited      (v_B hidden, obs_delay=2, p_drop=0.20 ZOH dropout, pos/gyro noise, oblique PnP bias b(rel_g))
  - Targets & Courses:
      * source_l2f          (figure8 train + ellipse3d held-out)
      * target_nominal      (PyBullet CF2X, d=2, figure8 + ellipse3d held-out)
      * target_perturbed    (PyBullet CF2X, kf x 1.20, d=3, figure8 + ellipse3d held-out)
  - One-Factor Robustness Curves (on held-out `ellipse3d`):
      * Action delay d in {0, 2, 4, 6, 8}
      * Observation dropout p_drop in {0.0, 0.15, 0.30, 0.45}
      * Motor thrust scale kf_scale in {0.85, 1.00, 1.15, 1.30}

Usage:
  python exp10_method_matrix.py                 # full 3-seed matrix + sweeps
  python exp10_method_matrix.py --smoke         # 1-seed converged validation check
"""

from __future__ import annotations

import argparse
import contextlib
import io
import multiprocessing as mp
import os
import time

import torch
import torch.nn as nn

from common import OUT, fmt, mean_std, np, plt, save_json
from exp6_knn_residual import KNNResidual, OneStepSource, feats
from exp9_residual_finetune import scripted_sysid_action
from racing_env import (
    EXACT_BASE_DIM,
    SENSOR_BASE_DIM,
    GateRacingEnv,
    GeometricGateController,
    build_racing_obs,
    evaluate_racing,
)
from sims.drone_pair import DroneState, InterfaceConfig, RL_DT, make_target

HIST = 8
TRAIN_DELAYS = (0, 1, 2, 3, 4)
TAUS = {"l2f_native": None, "tau20ms": 0.02, "tau30ms": 0.03, "tau50ms": 0.05}


def dyn_sampler(rng):
    return {
        "mass_scale": rng.uniform(0.82, 1.18),
        "thrust_scale": rng.uniform(0.82, 1.22),
        "tau_rise_scale": rng.uniform(0.35, 1.4),
        "tau_fall_scale": rng.uniform(0.10, 1.4),
    }


# --------------------------------------------------------------------------------------------------
# Neural Student Architectures (Feed-Forward Baseline & Recurrent GRU Observer) for H3'
# --------------------------------------------------------------------------------------------------


class FFRacingStudent(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 4),
        )
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net((x - self.mu) / self.sd))


class GRURacingStudent(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 4)
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))

    def forward_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        z = self.in_proj((x_seq - self.mu) / self.sd)
        out, _ = self.gru(z)
        return torch.tanh(self.head(out))

    def step_one(self, x: np.ndarray, h: torch.Tensor | None):
        xt = torch.as_tensor(x, dtype=torch.float32).view(1, 1, -1)
        z = self.in_proj((xt - self.mu) / self.sd)
        out, h_next = self.gru(z, h)
        act = torch.tanh(self.head(out[0, 0])).detach().numpy()
        return act, h_next


class SB3PolicyCallable:
    def __init__(self, model):
        self.model = model

    def __call__(self, obs, env, step_idx):
        a, _ = self.model.predict(obs, deterministic=True)
        return a


class ZeroVelPaddedPPO:
    """Zero-velocity-padded exact-state PPO under sensor-limited obs (shows naive non-recurrent collapse)."""

    def __init__(self, model):
        self.model = model

    def __call__(self, obs, env, step_idx):
        # Insert 3 zeros where 0.33 * v_B sits in exact-state obs (indices 15:18)
        obs_pad = np.concatenate([obs[:15], np.zeros(3, dtype=np.float32), obs[15:]])
        a, _ = self.model.predict(obs_pad, deterministic=True)
        return a


class TorchStudentCallable:
    def __init__(self, net: nn.Module, recurrent: bool):
        self.net = net
        self.recurrent = recurrent
        self.h = None

    def reset_episode(self):
        self.h = None

    @torch.no_grad()
    def __call__(self, obs, env, step_idx):
        if self.recurrent:
            a, self.h = self.net.step_one(obs, self.h)
            return a
        xt = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return self.net(xt)[0].numpy()


def distill_sensor_student(
    teacher_model,
    seed: int,
    recurrent: bool,
    randomized: bool,
    dagger_iters: int = 6,
    eps_per_iter: int = 8,
    train_steps: int = 550,
    residual_fn=None,
    tau_hat: float | None = None,
    gp_gate_bias_fn=None,
    fixed_delay: int | None = None,
) -> nn.Module:
    """
    Distill an exact-state PPO teacher into a sensor-limited student (FF or GRU) via on-policy DAgger:
      - `randomized=False` & no residual: fixed nominal L2F source, obs_delay=2, p_drop=0.20, zero bias (Thm 2.4 trap)
      - `randomized=True`: randomized L2F dynamics + delays + randomized PnP bias scale (RMA-style H3' fix)
      - `residual_fn` / `gp_gate_bias_fn` provided: distill inside Swift residual+GP augmented L2F source (H4' + H3')
    """
    torch.manual_seed(seed + 100)
    rng = np.random.default_rng(seed + 100)
    in_dim = SENSOR_BASE_DIM + 4 * HIST
    student = GRURacingStudent(in_dim) if recurrent else FFRacingStudent(in_dim)

    delays = (fixed_delay,) if fixed_delay is not None else (TRAIN_DELAYS if randomized else (0,))
    env = GateRacingEnv(
        course_kind="figure8",
        domain="source",
        hist=HIST,
        delays=delays,
        dyn_sampler=dyn_sampler if randomized else None,
        sensor_limited=True,
        obs_delay=2,
        p_drop=0.20,
        pnp_bias_scale=1.0 if randomized else 0.0,
        gp_gate_bias_fn=gp_gate_bias_fn,
        residual_fn=residual_fn,
        tau_hat=tau_hat,
        max_ep_s=6.0,
        seed=10_000 + 100 * seed,
    )

    seqs_X, seqs_Y = [], []
    for it in range(dagger_iters):
        beta = 1.0 if it == 0 else max(0.0, 0.55 - 0.12 * it)
        for ep in range(eps_per_iter):
            if randomized:
                env.pnp_bias_scale = float(rng.uniform(0.0, 1.35))
            obs_s, _ = env.reset(seed=10_000 * seed + 100 * it + ep)
            h = None
            X_ep, Y_ep = [], []
            true_hint = env.true_hint
            while True:
                true_st = env.drone.true_state()
                obs_exact, true_hint = build_racing_obs(
                    true_st, env.cl, env.active_gate_idx(), true_hint, env.ah, HIST, sensor_limited=False
                )
                a_teach, _ = teacher_model.predict(obs_exact, deterministic=True)
                X_ep.append(obs_s.copy())
                Y_ep.append(a_teach.copy())

                if rng.random() < beta:
                    a_exec = a_teach
                    if recurrent:
                        with torch.no_grad():
                            _, h = student.step_one(obs_s, h)
                else:
                    with torch.no_grad():
                        if recurrent:
                            a_stu, h = student.step_one(obs_s, h)
                        else:
                            a_stu = student(torch.as_tensor(obs_s, dtype=torch.float32)[None])[0].numpy()
                    a_exec = np.clip(a_stu + 0.04 * rng.standard_normal(4), -1.0, 1.0)

                obs_s, _, term, trunc, _ = env.step(a_exec)
                if term or trunc:
                    break
            if len(X_ep) >= 16:
                seqs_X.append(np.array(X_ep, dtype=np.float32))
                seqs_Y.append(np.array(Y_ep, dtype=np.float32))

        all_x = np.concatenate(seqs_X, axis=0)
        student.mu.copy_(torch.as_tensor(all_x.mean(0)))
        student.sd.copy_(torch.as_tensor(all_x.std(0) + 1e-3))
        opt = torch.optim.Adam(student.parameters(), lr=2e-3)

        if recurrent:
            for _ in range(train_steps):
                idx = int(rng.integers(0, len(seqs_X)))
                xb = torch.as_tensor(seqs_X[idx])[None]
                yb = torch.as_tensor(seqs_Y[idx])[None]
                loss = ((student.forward_seq(xb) - yb) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 2.0)
                opt.step()
        else:
            all_y = torch.as_tensor(np.concatenate(seqs_Y, axis=0))
            all_xt = torch.as_tensor(all_x)
            n = len(all_xt)
            for _ in range(max(35, train_steps // 12)):
                perm = torch.randperm(n)
                for i in range(0, n, 512):
                    sl = perm[i : i + 512]
                    loss = ((student(all_xt[sl]) - all_y[sl]) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    return student


# --------------------------------------------------------------------------------------------------
# Swift Step-0 Calibration + Residual Fitting on Gate Racing (k-NN Dynamics + Ground-Truth GP Bias)
# --------------------------------------------------------------------------------------------------


def calibrate_racing_target(
    base_model,
    kf_scale: float,
    target_delay: int,
    pnp_bias_scale: float,
    seed: int,
):
    """
    Collect 1 Step-0 scripted multi-sine flight (3.5 s) + 3 base-policy racing rollouts (~12.5 s, ~1,600 samples):
      1. Identify `(d_hat, tau_hat)` via lag-filtered rotor thrust joint scan.
      2. Fit 6D SE(3)-invariant body-frame `k-NN` dynamics residual `[a_res_B, alpha_res]`.
      3. Fit GP perception-bias residual on ground-truth body-frame relative gate vector `rel_g_true` (Prop 2.7).
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    rng = np.random.default_rng(seed + 700)

    # 1. Step-0 scripted multi-sine flight from safe altitude
    with contextlib.redirect_stdout(io.StringIO()):
        cal_drone = make_target(tau_m=0.03, iface=InterfaceConfig(action_delay=target_delay), seed=seed + 701)
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

    # 2. 3 target racing rollouts (~1,250 steps) for on-track state-action k-NN & ground-truth-pose GP fitting
    env = GateRacingEnv(
        course_kind="figure8",
        domain="target",
        hist=HIST,
        delays=(target_delay,),
        sensor_limited=True,
        obs_delay=0,
        p_drop=0.0,
        pnp_bias_scale=pnp_bias_scale,
        kf_scale=kf_scale,
        max_ep_s=4.5,
        seed=seed + 710,
    )
    flights = list(sysid_flights)
    X_gp, Y_gp = [], []
    zero_v_pol = ZeroVelPaddedPPO(base_model)
    for ep in range(3):
        obs, _ = env.reset(seed=25_000 + 100 * seed + ep)
        S_ep, A_ep = [env.drone.true_state()], []
        step_k = 0
        while True:
            st_true = env.drone.true_state()
            st_obs = env.obs_state
            g_cur = env.gates[env.active_gate_idx()]
            rel_true = st_true.R.T @ (g_cur - st_true.p)
            # Observed body-frame shift (R_true^T (p_true - p_obs) == b_body(rel_true) + noise)
            b_meas = st_true.R.T @ (st_true.p - st_obs.p)
            X_gp.append(rel_true)
            Y_gp.append(b_meas)

            a = zero_v_pol(obs, env, step_k)
            a = np.clip(a + 0.08 * rng.standard_normal(4), -1.0, 1.0)
            obs, _, term, trunc, info = env.step(a)
            S_ep.append(info["state"])
            A_ep.append(np.asarray(info["a"]))
            step_k += 1
            if term or trunc:
                break
        if len(A_ep) >= 20:
            flights.append((S_ep, np.array(A_ep)))

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

    fS_best = OneStepSource(tau_hat)
    rs_all, Xs_all = zip(*[fS_best.residuals(S, A, d_hat) for S, A in flights])
    r_all, X_all = np.concatenate(rs_all), sum(Xs_all, [])
    knn = KNNResidual(k=5).fit(feats(X_all, "body10"), r_all)

    X_gp_arr, Y_gp_arr = np.array(X_gp), np.array(Y_gp)
    sub = rng.choice(len(X_gp_arr), size=min(450, len(X_gp_arr)), replace=False)
    gp = GaussianProcessRegressor(RBF(0.6) + WhiteKernel(1e-3), normalize_y=False, random_state=0)
    gp.fit(X_gp_arr[sub], Y_gp_arr[sub])
    held = np.setdiff1d(np.arange(len(X_gp_arr)), sub)
    eval_pts = X_gp_arr[held[:150]] if len(held) >= 20 else X_gp_arr[:100]
    gp_mean_abs = float(np.mean(np.linalg.norm(gp.predict(eval_pts), axis=1)))

    return {
        "knn": knn,
        "tau_hat": tau_hat,
        "tau_name": tname,
        "d_hat": int(d_hat),
        "gp": gp,
        "gp_mean_abs": gp_mean_abs,
    }


def make_knn_residual_hook(knn):
    """Exact NumPy fast-path for single-state `KNNResidual.predict` (100x faster than per-tick sklearn check_array)."""
    Zs_tr = np.ascontiguousarray(knn.knn._fit_X, dtype=np.float64)
    Y_tr = np.ascontiguousarray(knn.knn._y, dtype=np.float64)
    mu = np.asarray(knn.mu, dtype=np.float64)
    inv_sd = 1.0 / np.asarray(knn.sd, dtype=np.float64)
    k = int(knn.k)
    h = float(knn.h)
    inv_h2 = 1.0 / (h * h + 1e-12)
    cache = {}

    def fn(st, a):
        if cache.get("key") is not a:
            vB = st.R.T @ st.v
            z = np.empty(10, dtype=np.float64)
            z[:3] = vB
            z[3:6] = st.w
            z[6:] = a
            zs = (z - mu) * inv_sd
            diff = Zs_tr - zs
            d2 = np.einsum("ij,ij->i", diff, diff)
            idx = np.argpartition(d2, k)[:k]
            dk = np.sqrt(d2[idx])
            if np.any(dk < 1e-12):
                pred = Y_tr[idx[int(np.argmin(dk))]]
            else:
                w = 1.0 / dk
                pred = (w @ Y_tr[idx]) / float(np.sum(w))
            ex_d = float(np.mean(dk)) - h
            decay = float(np.exp(-(ex_d * ex_d) * inv_h2)) if ex_d > 0.0 else 1.0
            r = pred * decay
            cache["key"] = a
            cache["val"] = (r[:3], r[3:])
        return cache["val"]

    return fn


def make_gp_gate_hook(gp):
    """Exact NumPy fast-path for single-point RBF+WhiteKernel GP posterior mean (verified against sklearn)."""
    if gp is None:
        return None
    X_tr = np.ascontiguousarray(gp.X_train_, dtype=np.float64)
    alpha = np.ascontiguousarray(gp.alpha_, dtype=np.float64)
    inv_ell = 1.0 / np.asarray(gp.kernel_.k1.length_scale, dtype=np.float64)
    # Verify exact numerical equivalence against sklearn on initialization
    d0 = (X_tr[0] - X_tr) * inv_ell
    f0 = np.exp(-0.5 * np.einsum("ij,ij->i", d0, d0)) @ alpha
    if not np.allclose(f0, gp.predict(X_tr[:1])[0], atol=1e-8):
        return lambda rel_g0: gp.predict(rel_g0[None])[0]

    def fn(rel_g0: np.ndarray) -> np.ndarray:
        diff = (rel_g0 - X_tr) * inv_ell
        k_vec = np.exp(-0.5 * np.einsum("ij,ij->i", diff, diff))
        return k_vec @ alpha

    return fn


# --------------------------------------------------------------------------------------------------
# Per-Seed Worker: Train Full Method Suite & Evaluate Across Courses, Targets, and Sweeps
# --------------------------------------------------------------------------------------------------


def run_seed(args_tuple):
    seed, base_steps, ft_steps, dagger_iters, n_eval = args_tuple
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    t0 = time.time()
    n_envs = 8
    ppo_kw = dict(
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        seed=seed,
        verbose=0,
        policy_kwargs=dict(net_arch=[128, 128], log_std_init=-0.7),
    )

    # 1. Base Nominal PPO (`lam_da=0.02`) -> Warm-start `lam_da=0.20` (Guardrail #1)
    venv_nom_base = DummyVecEnv(
        [
            lambda i=i: GateRacingEnv("figure8", "source", HIST, lam_da=0.02, delays=(0,), max_ep_s=5.5, seed=1000 * seed + i)
            for i in range(n_envs)
        ]
    )
    base_nom = PPO("MlpPolicy", venv_nom_base, **ppo_kw)
    base_nom.learn(base_steps)
    base_path = f"{OUT}/exp10_base_nom_s{seed}.zip"
    base_nom.save(base_path)

    venv_nom_ft = DummyVecEnv(
        [
            lambda i=i: GateRacingEnv("figure8", "source", HIST, lam_da=0.20, delays=(0,), max_ep_s=5.5, seed=1100 * seed + i)
            for i in range(n_envs)
        ]
    )
    nom_ppo = PPO.load(base_path, env=venv_nom_ft, device="cpu")
    nom_ppo.learn(ft_steps, reset_num_timesteps=False)

    # 2. Domain + Latency Randomized PPO (equal total source steps: `base_steps + ft_steps`)
    venv_dr_base = DummyVecEnv(
        [
            lambda i=i: GateRacingEnv(
                "figure8", "source", HIST, lam_da=0.02, delays=TRAIN_DELAYS, dyn_sampler=dyn_sampler, max_ep_s=5.5, seed=2000 * seed + i
            )
            for i in range(n_envs)
        ]
    )
    dr_ppo = PPO("MlpPolicy", venv_dr_base, **ppo_kw)
    dr_ppo.learn(base_steps)
    dr_base_path = f"{OUT}/exp10_base_dr_s{seed}.zip"
    dr_ppo.save(dr_base_path)

    venv_dr_ft = DummyVecEnv(
        [
            lambda i=i: GateRacingEnv(
                "figure8", "source", HIST, lam_da=0.20, delays=TRAIN_DELAYS, dyn_sampler=dyn_sampler, max_ep_s=5.5, seed=2100 * seed + i
            )
            for i in range(n_envs)
        ]
    )
    dr_ppo = PPO.load(dr_base_path, env=venv_dr_ft, device="cpu")
    dr_ppo.learn(ft_steps, reset_num_timesteps=False)

    # 3. Swift Residual Fine-Tuned PPO (for both nominal target and perturbed target, equal source budget!)
    # Note: nominal sensor has zero PnP bias (Prop 2.7 null control), perturbed target has oblique PnP bias scale=1.0!
    cal_nom = calibrate_racing_target(
        PPO.load(base_path, device="cpu"), kf_scale=1.0, target_delay=2, pnp_bias_scale=0.0, seed=seed
    )
    venv_res_nom = DummyVecEnv(
        [
            lambda i=i: GateRacingEnv(
                "figure8",
                "source",
                HIST,
                lam_da=0.20,
                delays=(cal_nom["d_hat"],),
                residual_fn=make_knn_residual_hook(cal_nom["knn"]),
                tau_hat=cal_nom["tau_hat"],
                max_ep_s=5.5,
                seed=3000 * seed + i,
            )
            for i in range(n_envs)
        ]
    )
    res_ppo_nom = PPO.load(base_path, env=venv_res_nom, device="cpu")
    res_ppo_nom.learn(ft_steps, reset_num_timesteps=False)

    cal_pert = calibrate_racing_target(
        PPO.load(base_path, device="cpu"), kf_scale=1.20, target_delay=3, pnp_bias_scale=1.0, seed=seed + 50
    )
    venv_res_pert = DummyVecEnv(
        [
            lambda i=i: GateRacingEnv(
                "figure8",
                "source",
                HIST,
                lam_da=0.20,
                delays=(cal_pert["d_hat"],),
                residual_fn=make_knn_residual_hook(cal_pert["knn"]),
                tau_hat=cal_pert["tau_hat"],
                max_ep_s=5.5,
                seed=3100 * seed + i,
            )
            for i in range(n_envs)
        ]
    )
    res_ppo_pert = PPO.load(base_path, env=venv_res_pert, device="cpu")
    res_ppo_pert.learn(ft_steps, reset_num_timesteps=False)

    # Also fit a biased-sensor calibration on the nominal airframe so sensor-limited `target_nominal` (which has PnP bias)
    # gets its corresponding GP perception model!
    cal_nom_sens = calibrate_racing_target(
        PPO.load(base_path, device="cpu"), kf_scale=1.0, target_delay=2, pnp_bias_scale=1.0, seed=seed + 25
    )

    # 4. Target-Trained PPO Oracles (in-domain upper reference)
    venv_orc_nom = DummyVecEnv(
        [
            lambda i=i: GateRacingEnv("figure8", "target", HIST, lam_da=0.20, delays=(2,), kf_scale=1.0, max_ep_s=5.5, seed=4000 * seed + i)
            for i in range(4)
        ]
    )
    orc_ppo_nom = PPO.load(base_path, env=venv_orc_nom, device="cpu")
    orc_ppo_nom.learn(ft_steps, reset_num_timesteps=False)

    venv_orc_pert = DummyVecEnv(
        [
            lambda i=i: GateRacingEnv("figure8", "target", HIST, lam_da=0.20, delays=(3,), kf_scale=1.20, max_ep_s=5.5, seed=4100 * seed + i)
            for i in range(4)
        ]
    )
    orc_ppo_pert = PPO.load(base_path, env=venv_orc_pert, device="cpu")
    orc_ppo_pert.learn(ft_steps, reset_num_timesteps=False)

    # 5. Sensor-Limited Teacher-Student Distillation (FF baseline, GRU fixed-source, GRU randomized-source, GRU + Residual/GP)
    ff_stu = distill_sensor_student(dr_ppo, seed, recurrent=False, randomized=True, dagger_iters=dagger_iters)
    gru_fix = distill_sensor_student(nom_ppo, seed + 10, recurrent=True, randomized=False, dagger_iters=dagger_iters)
    gru_rand = distill_sensor_student(dr_ppo, seed + 20, recurrent=True, randomized=True, dagger_iters=dagger_iters)
    gru_res_nom = distill_sensor_student(
        res_ppo_nom,
        seed + 30,
        recurrent=True,
        randomized=False,
        dagger_iters=dagger_iters,
        residual_fn=make_knn_residual_hook(cal_nom_sens["knn"]),
        tau_hat=cal_nom_sens["tau_hat"],
        gp_gate_bias_fn=make_gp_gate_hook(cal_nom_sens["gp"]),
        fixed_delay=cal_nom_sens["d_hat"],
    )
    gru_res_pert = distill_sensor_student(
        res_ppo_pert,
        seed + 40,
        recurrent=True,
        randomized=False,
        dagger_iters=dagger_iters,
        residual_fn=make_knn_residual_hook(cal_pert["knn"]),
        tau_hat=cal_pert["tau_hat"],
        gp_gate_bias_fn=make_gp_gate_hook(cal_pert["gp"]),
        fixed_delay=cal_pert["d_hat"],
    )

    # --------------------------------------------------------------------------------------------------
    # Evaluations Across Courses, Observation Modes, and Targets (reusing cached envs!)
    # --------------------------------------------------------------------------------------------------
    env_cache = {}

    def get_eval_env(cname, dom, d_act, kfs, sensor_lim=False, obs_d=0, p_dr=0.0, pnp_sc=0.0):
        key = (cname, dom, d_act, kfs, sensor_lim, obs_d, p_dr, pnp_sc)
        if key not in env_cache:
            env_cache[key] = GateRacingEnv(
                course_kind=cname,
                domain=dom,
                hist=HIST,
                delays=(d_act,),
                kf_scale=kfs,
                sensor_limited=sensor_lim,
                obs_delay=obs_d,
                p_drop=p_dr,
                pnp_bias_scale=pnp_sc,
                random_start_gate=False,
                max_ep_s=12.5,
                seed=800 + seed,
            )
        return env_cache[key]

    exact_methods = {
        "classical_expert": lambda tgt: GeometricGateController(),
        "nominal_ppo": lambda tgt: SB3PolicyCallable(nom_ppo),
        "dr_ppo": lambda tgt: SB3PolicyCallable(dr_ppo),
        "residual_ft_ppo": lambda tgt: SB3PolicyCallable(res_ppo_pert if tgt == "target_perturbed" else res_ppo_nom),
        "target_oracle_ppo": lambda tgt: SB3PolicyCallable(orc_ppo_pert if tgt == "target_perturbed" else orc_ppo_nom),
    }

    sensor_methods = {
        "nominal_ppo_zerovel": lambda tgt: ZeroVelPaddedPPO(nom_ppo),
        "ff_student": lambda tgt: TorchStudentCallable(ff_stu, recurrent=False),
        "gru_student_fixed": lambda tgt: TorchStudentCallable(gru_fix, recurrent=True),
        "gru_student_rand": lambda tgt: TorchStudentCallable(gru_rand, recurrent=True),
        "residual_gp_gru": lambda tgt: TorchStudentCallable(
            gru_res_pert if tgt == "target_perturbed" else gru_res_nom, recurrent=True
        ),
    }

    matrix = {"exact_state": {}, "sensor_limited": {}}
    courses = ["figure8", "ellipse3d"]
    targets = {
        "source_l2f": ("source", 0, 1.0, 0.0),
        "target_nominal": ("target", 2, 1.0, 1.0),
        "target_perturbed": ("target", 3, 1.20, 1.0),
    }

    for cname in courses:
        matrix["exact_state"][cname] = {}
        matrix["sensor_limited"][cname] = {}
        for tname, (dom, d_act, kfs, pnp_sc) in targets.items():
            matrix["exact_state"][cname][tname] = {}
            ev_ex = get_eval_env(cname, dom, d_act, kfs, sensor_lim=False)
            for mname, p_builder in exact_methods.items():
                matrix["exact_state"][cname][tname][mname] = evaluate_racing(p_builder(tname), ev_ex, n_eval=n_eval)

            matrix["sensor_limited"][cname][tname] = {}
            is_tgt = dom == "target"
            ev_sn = get_eval_env(
                cname,
                dom,
                d_act,
                kfs,
                sensor_lim=True,
                obs_d=2 if is_tgt else 0,
                p_dr=0.20 if is_tgt else 0.0,
                pnp_sc=pnp_sc if is_tgt else 0.0,
            )
            for mname, p_builder in sensor_methods.items():
                matrix["sensor_limited"][cname][tname][mname] = evaluate_racing(p_builder(tname), ev_sn, n_eval=n_eval)

    # One-factor robustness sweeps on held-out `ellipse3d` in PyBullet target
    n_sw = max(6, n_eval // 2)
    sweeps = {"delay": {}, "dropout": {}, "kf_scale": {}}
    for mname in ["nominal_ppo", "dr_ppo", "residual_ft_ppo"]:
        pol_nom = exact_methods[mname]("target_nominal")
        sweeps["delay"][mname] = {
            str(d): evaluate_racing(pol_nom, get_eval_env("ellipse3d", "target", d, 1.0, False), n_eval=n_sw)
            for d in [0, 2, 4, 6, 8]
        }
        sweeps["kf_scale"][mname] = {
            str(kf): evaluate_racing(
                exact_methods[mname]("target_perturbed" if kf > 1.08 else "target_nominal"),
                get_eval_env("ellipse3d", "target", 2, kf, False),
                n_eval=n_sw,
            )
            for kf in [0.85, 1.0, 1.15, 1.30]
        }

    for mname in ["ff_student", "gru_student_fixed", "gru_student_rand", "residual_gp_gru"]:
        pol_sn = sensor_methods[mname]("target_nominal")
        sweeps["dropout"][mname] = {
            str(p): evaluate_racing(
                pol_sn,
                get_eval_env("ellipse3d", "target", 2, 1.0, True, obs_d=2, p_dr=p, pnp_sc=1.0),
                n_eval=n_sw,
            )
            for p in [0.0, 0.15, 0.30, 0.45]
        }

    return {
        "seed": seed,
        "wall_s": time.time() - t0,
        "cal_nom_null": {k: cal_nom[k] for k in ["tau_name", "d_hat", "gp_mean_abs"]},
        "cal_nom_sens": {k: cal_nom_sens[k] for k in ["tau_name", "d_hat", "gp_mean_abs"]},
        "cal_pert": {k: cal_pert[k] for k in ["tau_name", "d_hat", "gp_mean_abs"]},
        "matrix": matrix,
        "sweeps": sweeps,
    }


def aggregate_seed_dicts(dicts: list[dict]) -> dict:
    keys = list(dicts[0].keys())
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts]
        finite = [v for v in vals if np.isfinite(v)]
        if finite:
            out[k] = mean_std(finite)
        else:
            out[k] = {"mean": float("nan"), "std": float("nan"), "n": 0, "values": vals}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="1-seed converged validation check")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-steps", type=int, default=450_000)
    parser.add_argument("--ft-steps", type=int, default=100_000)
    args = parser.parse_args()

    if args.smoke:
        seeds = [0]
        base_steps, ft_steps, dagger_iters, n_eval = 360_000, 90_000, 5, 6
    else:
        seeds = list(range(args.seeds))
        base_steps, ft_steps, dagger_iters, n_eval = args.base_steps, args.ft_steps, 6, 10

    t0 = time.time()
    with mp.get_context("spawn").Pool(min(3, len(seeds))) as pool:
        outs = pool.map(run_seed, [(s, base_steps, ft_steps, dagger_iters, n_eval) for s in seeds])
    wall_total = time.time() - t0
    print(f"Exp 10 ({'SMOKE' if args.smoke else 'FULL'}) finished {len(seeds)} seed(s) in {wall_total:.1f}s")

    agg_matrix = {}
    for obs_mode in outs[0]["matrix"]:
        agg_matrix[obs_mode] = {}
        for cname in outs[0]["matrix"][obs_mode]:
            agg_matrix[obs_mode][cname] = {}
            for tname in outs[0]["matrix"][obs_mode][cname]:
                agg_matrix[obs_mode][cname][tname] = {}
                for mname in outs[0]["matrix"][obs_mode][cname][tname]:
                    m_agg = aggregate_seed_dicts([o["matrix"][obs_mode][cname][tname][mname] for o in outs])
                    ratios = []
                    for o in outs:
                        src_gf = o["matrix"][obs_mode][cname]["source_l2f"][mname]["gate_frac"]
                        tgt_gf = o["matrix"][obs_mode][cname][tname][mname]["gate_frac"]
                        ratios.append(tgt_gf / max(src_gf, 1e-3))
                    m_agg["tgt_src_ratio"] = mean_std(ratios)
                    agg_matrix[obs_mode][cname][tname][mname] = m_agg

    agg_sweeps = {}
    for sw_name in outs[0]["sweeps"]:
        agg_sweeps[sw_name] = {}
        for mname in outs[0]["sweeps"][sw_name]:
            agg_sweeps[sw_name][mname] = {
                xval: aggregate_seed_dicts([o["sweeps"][sw_name][mname][xval] for o in outs])
                for xval in outs[0]["sweeps"][sw_name][mname]
            }

    res = {
        "meta": {
            "seeds": seeds,
            "base_steps": base_steps,
            "ft_steps": ft_steps,
            "total_source_budget": base_steps + ft_steps,
            "wall_s": wall_total,
            "smoke": args.smoke,
        },
        "calibrations": {
            "nominal_null_bias": [o["cal_nom_null"] for o in outs],
            "nominal_biased_sensor": [o["cal_nom_sens"] for o in outs],
            "perturbed_target": [o["cal_pert"] for o in outs],
        },
        "matrix": agg_matrix,
        "sweeps": agg_sweeps,
    }
    out_json = "exp10_smoke.json" if args.smoke else "exp10_method_matrix.json"
    save_json(out_json, res)

    for obs_mode in ["exact_state", "sensor_limited"]:
        for cname in ["figure8", "ellipse3d"]:
            print(
                f"\n=== [{obs_mode.upper()} | course={cname}] gate_frac | comp | clean | crash | med_lap | trk_err | tgt/src ==="
            )
            for tname in ["source_l2f", "target_nominal", "target_perturbed"]:
                print(f"  -- {tname} --")
                for mname, m in agg_matrix[obs_mode][cname][tname].items():
                    lap_s = m["median_lap_s"]["mean"]
                    lap_str = f"{lap_s:5.2f}s" if np.isfinite(lap_s) else "  n/a "
                    print(
                        f"    {mname:20s} gf={fmt(m['gate_frac']):>13} | cp={m['completion']['mean']:.2f} | "
                        f"cln={m['clean_completion']['mean']:.2f} | cr={m['crash_rate']['mean']:.2f} | "
                        f"lap={lap_str} | te={m['track_err_m']['mean']:.2f}m | ratio={m['tgt_src_ratio']['mean']:.2f}"
                    )

    print("\nStep-0 Calibrations:", res["calibrations"])

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.8))
    ex_m = ["classical_expert", "nominal_ppo", "dr_ppo", "residual_ft_ppo", "target_oracle_ppo"]
    x0 = np.arange(len(ex_m))
    for j, tname in enumerate(["source_l2f", "target_nominal", "target_perturbed"]):
        axes[0, 0].bar(
            x0 + (j - 1) * 0.26,
            [agg_matrix["exact_state"]["ellipse3d"][tname][m]["gate_frac"]["mean"] for m in ex_m],
            0.26,
            yerr=[agg_matrix["exact_state"]["ellipse3d"][tname][m]["gate_frac"]["std"] for m in ex_m],
            capsize=2,
            label=tname,
        )
    axes[0, 0].set_xticks(x0, [m.replace("_ppo", "").replace("_", "\n") for m in ex_m], fontsize=7.5)
    axes[0, 0].set_ylim(0, 1.08)
    axes[0, 0].set_ylabel("Gate Completion Fraction")
    axes[0, 0].set_title("A. Exact-State Held-Out Course (ellipse3d)")
    axes[0, 0].legend(fontsize=7)

    sn_m = ["nominal_ppo_zerovel", "ff_student", "gru_student_fixed", "gru_student_rand", "residual_gp_gru"]
    x1 = np.arange(len(sn_m))
    for j, tname in enumerate(["source_l2f", "target_nominal", "target_perturbed"]):
        axes[0, 1].bar(
            x1 + (j - 1) * 0.26,
            [agg_matrix["sensor_limited"]["ellipse3d"][tname][m]["gate_frac"]["mean"] for m in sn_m],
            0.26,
            yerr=[agg_matrix["sensor_limited"]["ellipse3d"][tname][m]["gate_frac"]["std"] for m in sn_m],
            capsize=2,
            label=tname,
        )
    axes[0, 1].set_xticks(x1, [m.replace("_student", "").replace("_", "\n") for m in sn_m], fontsize=7.5)
    axes[0, 1].set_ylim(0, 1.08)
    axes[0, 1].set_ylabel("Gate Completion Fraction")
    axes[0, 1].set_title("B. Sensor-Limited Held-Out Course (ellipse3d)")
    axes[0, 1].legend(fontsize=7)

    # Panel C: Training Course (`figure8`) Exact-State comparison
    for j, tname in enumerate(["source_l2f", "target_nominal", "target_perturbed"]):
        axes[0, 2].bar(
            x0 + (j - 1) * 0.26,
            [agg_matrix["exact_state"]["figure8"][tname][m]["gate_frac"]["mean"] for m in ex_m],
            0.26,
            yerr=[agg_matrix["exact_state"]["figure8"][tname][m]["gate_frac"]["std"] for m in ex_m],
            capsize=2,
            label=tname,
        )
    axes[0, 2].set_xticks(x0, [m.replace("_ppo", "").replace("_", "\n") for m in ex_m], fontsize=7.5)
    axes[0, 2].set_ylim(0, 1.08)
    axes[0, 2].set_ylabel("Gate Completion Fraction")
    axes[0, 2].set_title("C. Exact-State Training Course (figure8)")
    axes[0, 2].legend(fontsize=7)

    # Row 1: Three One-Factor Robustness Curves on Held-Out `ellipse3d`
    d_vals = [0, 2, 4, 6, 8]
    for mname in ["nominal_ppo", "dr_ppo", "residual_ft_ppo"]:
        g = agg_sweeps["delay"][mname]
        axes[1, 0].errorbar(
            d_vals,
            [g[str(d)]["gate_frac"]["mean"] for d in d_vals],
            yerr=[g[str(d)]["gate_frac"]["std"] for d in d_vals],
            marker="o",
            capsize=2,
            label=mname,
        )
    axes[1, 0].set_xlabel("Action Delay d [10 ms ticks]")
    axes[1, 0].set_ylabel("Target Gate Fraction")
    axes[1, 0].set_ylim(0, 1.08)
    axes[1, 0].set_title("D. Delay Robustness (ellipse3d Target)")
    axes[1, 0].legend(fontsize=7.5)

    p_vals = [0.0, 0.15, 0.30, 0.45]
    for mname in ["ff_student", "gru_student_fixed", "gru_student_rand", "residual_gp_gru"]:
        g = agg_sweeps["dropout"][mname]
        axes[1, 1].errorbar(
            p_vals,
            [g[str(p)]["gate_frac"]["mean"] for p in p_vals],
            yerr=[g[str(p)]["gate_frac"]["std"] for p in p_vals],
            marker="s",
            capsize=2,
            label=mname,
        )
    axes[1, 1].set_xlabel("ZOH Observation Dropout Probability p_drop")
    axes[1, 1].set_ylabel("Target Gate Fraction")
    axes[1, 1].set_ylim(0, 1.08)
    axes[1, 1].set_title("E. Sensor Dropout Robustness (ellipse3d Target)")
    axes[1, 1].legend(fontsize=7.5)

    kf_vals = [0.85, 1.0, 1.15, 1.30]
    for mname in ["nominal_ppo", "dr_ppo", "residual_ft_ppo"]:
        g = agg_sweeps["kf_scale"][mname]
        axes[1, 2].errorbar(
            kf_vals,
            [g[str(kf)]["gate_frac"]["mean"] for kf in kf_vals],
            yerr=[g[str(kf)]["gate_frac"]["std"] for kf in kf_vals],
            marker="^",
            capsize=2,
            label=mname,
        )
    axes[1, 2].set_xlabel("Target Motor Thrust Scale kf_scale")
    axes[1, 2].set_ylabel("Target Gate Fraction")
    axes[1, 2].set_ylim(0, 1.08)
    axes[1, 2].set_title("F. Airframe Gain Robustness (ellipse3d Target)")
    axes[1, 2].legend(fontsize=7.5)

    plt.tight_layout()
    out_png = "exp10_smoke.png" if args.smoke else "exp10_method_matrix.png"
    plt.savefig(f"{OUT}/{out_png}", dpi=150)


if __name__ == "__main__":
    main()
