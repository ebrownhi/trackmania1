"""
Exp 18 (Step 4 Riff R4.3 + Step 5 Riff R5.2):
  - Part A (R4.3 — Active 2-Round Bootstrapped Swift vs. Single-Batch Swift vs. Direct Target RL at Equal 16s Budget):
      Compare 5 transfer strategies under a STRICTLY EQUAL 1,600-tick (`16.0 s`) target interaction budget on a
      challenging high-gain + delayed PyBullet target (`kf x 1.25, d = 3`):
        1. `zeroshot_base`          : 0 target ticks, 0 FT
        2. `direct_target_ppo_1600` : Fine-tune PPO directly in PyBullet for 1,600 steps (16s hardware budget!)
        3. `direct_res_policy_1600` : Train a lightweight 2-layer additive policy residual directly in PyBullet (1,600 steps)
        4. `swift_mopo_1round`      : 1,600 target ticks in 1 batch -> 5-MLP ensemble + 100k MOPO-PPO source FT
        5. `swift_mopo_2round`      : Active 2-round bootstrap (800 ticks -> 50k FT -> 800 on-policy ticks -> 50k FT,
                                      exact same 1,600 target ticks & 100k source FT steps!)
  - Part B (R5.2 — Zero-Shot Generalization Across a 10-Track Procedural 3D Racing Benchmark):
      Evaluate `zeroshot_base`, `direct_target_ppo_1600`, `swift_mopo_1round`, and `swift_mopo_2round` zero-shot
      across 10 diverse 3D Fourier racing circuits (`figure8`, `ellipse3d`, `split_s_3d`, `corkscrew_3d`,
      `tilted_hairpin`, `saddle_3d`, `diving_chicane`, `banked_oval_3d`, `ascending_esses`, `trefoil_3d`)!
"""

from __future__ import annotations

import contextlib
import io
import math
import multiprocessing as mp
import os
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from common import OUT, fmt, mean_std, plt, save_json
from exp6_knn_residual import OneStepSource, feats
from exp9_residual_finetune import scripted_sysid_action
from exp10_method_matrix import HIST, SB3PolicyCallable, TAUS
from exp14_asap_mopo_residuals import AugmentedRacingEnv, ResidualEnsemble
from racing_env import FastCenterline, GateRacingEnv, evaluate_racing
from sims.drone_pair import DroneState, InterfaceConfig, RL_DT, make_target

SEEDS = [0, 1, 2]
KF_HARD = 1.25
D_HARD = 3

TRACK_NAMES = [
    "figure8",
    "ellipse3d",
    "split_s_3d",
    "corkscrew_3d",
    "tilted_hairpin",
    "saddle_3d",
    "diving_chicane",
    "banked_oval_3d",
    "ascending_esses",
    "trefoil_3d",
]


class Procedural3DCourse:
    """10 thrust-feasible closed 3D Fourier racing circuits (`v_peak <= 3.55 m/s`, `||a + g e_z|| <= 13.2 m/s^2`)."""

    def __init__(self, kind: str, n_gates: int = 6):
        self.kind = kind
        self.n_gates = n_gates
        self.period = 5.4 if kind in ("trefoil_3d", "corkscrew_3d", "ascending_esses") else 5.0
        self.z0 = 1.25
        # Fourier harmonics: x(t) = sum ax_k cos(kwt) + bx_k sin(kwt), etc.
        specs = {
            "figure8": ([(0.0, 2.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 1.0)], [(0.0, 0.0), (0.0, 0.0)]),
            "ellipse3d": ([(2.0, 0.0)], [(0.0, 1.4)], [(0.0, 0.0), (0.0, 0.0), (0.0, 0.40)]),
            "split_s_3d": ([(1.9, 0.0), (0.0, 0.35)], [(0.0, 1.5), (0.25, 0.0)], [(0.45, 0.0), (0.0, 0.25)]),
            "corkscrew_3d": ([(1.8, 0.0), (0.25, 0.0)], [(0.0, 1.6), (0.0, 0.25)], [(0.0, 0.45), (0.25, 0.0)]),
            "tilted_hairpin": ([(0.0, 2.1)], [(0.0, 0.85), (0.0, 0.55)], [(0.0, 0.45)]),
            "saddle_3d": ([(1.95, 0.0)], [(0.0, 1.55)], [(0.0, 0.0), (0.42, 0.0)]),
            "diving_chicane": ([(0.0, 1.95)], [(0.0, 1.15), (0.40, 0.0)], [(0.35, 0.20), (0.0, -0.22)]),
            "banked_oval_3d": ([(2.1, 0.0)], [(0.0, 1.25)], [(0.38, -0.25)]),
            "ascending_esses": ([(0.0, 2.0)], [(0.45, 0.95), (0.0, 0.55)], [(-0.35, 0.25), (0.18, 0.0)]),
            "trefoil_3d": ([(1.55, 0.0), (-0.35, 0.0)], [(0.0, 1.55), (0.0, 0.35)], [(0.0, 0.0), (0.0, 0.0), (0.30, 0.0)]),
        }
        self.xh, self.yh, self.zh = specs[kind]

    def _eval_axis(self, harms, w: float, t: float) -> tuple[float, float, float]:
        p, v, a = 0.0, 0.0, 0.0
        for idx, (ac, bs) in enumerate(harms):
            k = idx + 1
            kw = k * w
            c, s = math.cos(kw * t), math.sin(kw * t)
            p += ac * c + bs * s
            v += -ac * kw * s + bs * kw * c
            a += -(kw * kw) * (ac * c + bs * s)
        return p, v, a

    def ref(self, t: float):
        w = 2.0 * math.pi / self.period
        px, vx, ax = self._eval_axis(self.xh, w, t)
        py, vy, ay = self._eval_axis(self.yh, w, t)
        pz, vz, az = self._eval_axis(self.zh, w, t)
        return (
            np.array([px, py, self.z0 + pz], dtype=np.float64),
            np.array([vx, vy, vz], dtype=np.float64),
            np.array([ax, ay, az], dtype=np.float64),
        )

    def gate_times(self):
        return [(i + 0.5) * self.period / self.n_gates for i in range(self.n_gates)]


def make_procedural_env(track_kind: str, domain: str = "target", delays=(D_HARD,), kf_scale: float = KF_HARD, seed: int = 0) -> GateRacingEnv:
    env = GateRacingEnv(
        course_kind="figure8",
        domain=domain,
        hist=HIST,
        delays=delays,
        kf_scale=kf_scale,
        random_start_gate=False,
        max_ep_s=13.5,
        seed=seed,
    )
    if track_kind not in ("figure8", "ellipse3d"):
        pc = Procedural3DCourse(track_kind)
        env.course = pc
        env.cl = FastCenterline(pc)
        env.gates = env.cl.gates
        env.tangents = env.cl.tangents
    elif track_kind == "ellipse3d":
        from sims.drone_pair import Course
        ec = Course(kind="ellipse3d")
        env.course = ec
        env.cl = FastCenterline(ec)
        env.gates = env.cl.gates
        env.tangents = env.cl.tangents
    return env


# --------------------------------------------------------------------------------------------------
# Direct Policy-Residual Wrapper (`1,600` target steps in PyBullet)
# --------------------------------------------------------------------------------------------------


class FrozenBasePlusResidualEnv(gym.Wrapper):
    def __init__(self, env: GateRacingEnv, base_model, res_scale: float = 0.35):
        super().__init__(env)
        self.base_model = base_model
        self.res_scale = res_scale
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs.copy()
        return obs, info

    def step(self, da_act):
        a_base, _ = self.base_model.predict(self._last_obs, deterministic=True)
        a_tot = np.clip(a_base + self.res_scale * np.asarray(da_act, dtype=np.float32), -1.0, 1.0)
        obs, r, term, trunc, info = self.env.step(a_tot)
        self._last_obs = obs.copy()
        return obs, r, term, trunc, info


class DirectResCallable:
    def __init__(self, base_model, res_model, res_scale: float = 0.35):
        self.base_model = base_model
        self.res_model = res_model
        self.res_scale = res_scale

    def __call__(self, obs, env, step_idx):
        ab, _ = self.base_model.predict(obs, deterministic=True)
        da, _ = self.res_model.predict(obs, deterministic=True)
        return np.clip(ab + self.res_scale * da, -1.0, 1.0)


def collect_sysid_flight(seed: int, n_ticks: int = 350):
    rng = np.random.default_rng(seed + 10)
    with contextlib.redirect_stdout(io.StringIO()):
        dr = make_target(tau_m=0.03, iface=InterfaceConfig(action_delay=D_HARD), seed=seed + 11)
    dr.backend.env.KF *= KF_HARD
    s0 = dr.reset(DroneState.hover(p=(0.0, 0.0, 2.0)))
    S, A = [s0], []
    ph = float(rng.uniform(0, 2.0 * math.pi))
    for k in range(n_ticks):
        a = scripted_sysid_action(k, RL_DT, dr.hover_action, ph)
        dr.step(a)
        S.append(dr.true_state())
        A.append(a)
    return [(S, np.array(A))]


def collect_policy_target_ticks(policy_model, total_ticks: int, seed: int):
    rng = np.random.default_rng(seed)
    env = GateRacingEnv("figure8", "target", HIST, delays=(D_HARD,), kf_scale=KF_HARD, max_ep_s=5.0, seed=seed)
    flights = []
    used = 0
    ep = 0
    while used < total_ticks:
        obs, _ = env.reset(seed=40_000 + 100 * seed + ep)
        S_ep, A_ep = [env.drone.true_state()], []
        while used < total_ticks:
            a, _ = policy_model.predict(obs, deterministic=True)
            a = np.clip(a + 0.07 * rng.standard_normal(4), -1.0, 1.0)
            obs, _, term, trunc, info = env.step(a)
            S_ep.append(info["state"])
            A_ep.append(np.asarray(info["a"]))
            used += 1
            if term or trunc:
                break
        if len(A_ep) >= 15:
            flights.append((S_ep, np.array(A_ep)))
        ep += 1
    return flights


def fit_swift_mopo_from_flights(sysid_flights, all_flights, seed: int):
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
    _, _, tau_hat, d_hat = best_id
    fS_b = OneStepSource(tau_hat)
    rs_all, Xs_all = zip(*[fS_b.residuals(S, A, d_hat) for S, A in all_flights])
    r_all, Z_all = np.concatenate(rs_all), feats(sum(Xs_all, []), "body10")
    ens = ResidualEnsemble(6, max_mag=[8.0, 8.0, 14.0, 45.0, 45.0, 25.0], seed=seed).fit(Z_all, r_all)
    return int(d_hat), tau_hat, ens


def run_seed(seed: int):
    torch.set_num_threads(1)
    t0 = time.time()
    base_path = f"{OUT}/exp10_base_nom_s{seed}.zip"
    if not os.path.exists(base_path):
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

    base_mdl = PPO.load(base_path, device="cpu")

    # 1. Direct Target PPO (`1,600` target steps = 16s hardware budget!)
    env_tgt_1 = DummyVecEnv(
        [lambda: GateRacingEnv("figure8", "target", HIST, lam_da=0.20, delays=(D_HARD,), kf_scale=KF_HARD, max_ep_s=5.5, seed=6000 + seed)]
    )
    direct_ppo = PPO.load(base_path, env=env_tgt_1, device="cpu")
    direct_ppo.n_steps = 400
    direct_ppo.batch_size = 200
    direct_ppo._setup_model()
    # Reload weights after resizing rollout buffer to 400 so 1,600 steps performs 4 genuine PPO updates!
    direct_ppo.policy.load_state_dict(base_mdl.policy.state_dict())
    direct_ppo.learn(1_600, reset_num_timesteps=False)

    # 2. Direct Target Policy-Residual RL (`1,600` target steps = 16s hardware budget!)
    env_res_tgt = DummyVecEnv(
        [
            lambda: FrozenBasePlusResidualEnv(
                GateRacingEnv("figure8", "target", HIST, lam_da=0.20, delays=(D_HARD,), kf_scale=KF_HARD, max_ep_s=5.5, seed=6100 + seed),
                PPO.load(base_path, device="cpu"),
                res_scale=0.35,
            )
        ]
    )
    res_head = PPO(
        "MlpPolicy",
        env_res_tgt,
        n_steps=400,
        batch_size=200,
        n_epochs=8,
        learning_rate=3e-4,
        seed=seed,
        verbose=0,
        policy_kwargs=dict(net_arch=[32, 32], log_std_init=-1.2),
    )
    # Zero-init last layer so residual starts at exact zero (`a = a_base`)!
    with torch.no_grad():
        res_head.policy.action_net.weight.zero_()
        res_head.policy.action_net.bias.zero_()
    res_head.learn(1_600)

    # 3. Single-Batch Swift + MOPO (`350` Step-0 + `1,250` base-policy ticks = `1,600` target ticks -> `100k` source FT)
    sysid_f = collect_sysid_flight(seed, n_ticks=350)
    cal_1batch = sysid_f + collect_policy_target_ticks(base_mdl, 1_250, seed + 100)
    d1, t1, ens1 = fit_swift_mopo_from_flights(sysid_f, cal_1batch, seed + 10)
    venv_1r = DummyVecEnv(
        [
            lambda i=i: AugmentedRacingEnv(
                "figure8", "source", HIST, lam_da=0.20, delays=(d1,), acc_ens=ens1, mopo_beta=0.35, tau_hat=t1, max_ep_s=5.5, seed=7000 * seed + i
            )
            for i in range(8)
        ]
    )
    swift_1r = PPO.load(base_path, env=venv_1r, device="cpu")
    swift_1r.learn(100_000, reset_num_timesteps=False)

    # 4. Active 2-Round Bootstrapped Swift + MOPO (`350` Step-0 + `450` R1 ticks -> `50k` FT -> `800` R2 ticks -> `50k` FT)
    #    Total target budget = `350 + 450 + 800 = 1,600` ticks! Total source FT = `50k + 50k = 100k` steps!
    cal_r1 = sysid_f + collect_policy_target_ticks(base_mdl, 450, seed + 200)
    d_r1, t_r1, ens_r1 = fit_swift_mopo_from_flights(sysid_f, cal_r1, seed + 20)
    venv_r1 = DummyVecEnv(
        [
            lambda i=i: AugmentedRacingEnv(
                "figure8", "source", HIST, lam_da=0.20, delays=(d_r1,), acc_ens=ens_r1, mopo_beta=0.35, tau_hat=t_r1, max_ep_s=5.5, seed=7100 * seed + i
            )
            for i in range(8)
        ]
    )
    swift_2r = PPO.load(base_path, env=venv_r1, device="cpu")
    swift_2r.learn(50_000, reset_num_timesteps=False)

    cal_r2_extra = collect_policy_target_ticks(swift_2r, 800, seed + 300)
    cal_r2_all = cal_r1 + cal_r2_extra
    d_r2, t_r2, ens_r2 = fit_swift_mopo_from_flights(sysid_f, cal_r2_all, seed + 30)
    venv_r2 = DummyVecEnv(
        [
            lambda i=i: AugmentedRacingEnv(
                "figure8", "source", HIST, lam_da=0.20, delays=(d_r2,), acc_ens=ens_r2, mopo_beta=0.35, tau_hat=t_r2, max_ep_s=5.5, seed=7200 * seed + i
            )
            for i in range(8)
        ]
    )
    swift_2r.set_env(venv_r2)
    swift_2r.learn(50_000, reset_num_timesteps=False)

    # Measure on-policy residual RMSE of `ens1` (1-round) vs `ens_r2` (2-round) on high-speed R2 trajectories!
    onpol_flights = collect_policy_target_ticks(swift_2r, 500, seed + 400)
    fS_eval = OneStepSource(t_r2)
    rs_op, Xs_op = zip(*[fS_eval.residuals(S, A, d_r2) for S, A in onpol_flights])
    r_op, Z_op = np.concatenate(rs_op), feats(sum(Xs_op, []), "body10")
    pred_1r = np.array([ens1.predict_one(z)[0][:3] for z in Z_op])
    pred_2r = np.array([ens_r2.predict_one(z)[0][:3] for z in Z_op])
    rmse_1r = float(np.sqrt(np.mean((r_op[:, :3] - pred_1r) ** 2)))
    rmse_2r = float(np.sqrt(np.mean((r_op[:, :3] - pred_2r) ** 2)))

    pols_part_a = {
        "zeroshot_base": SB3PolicyCallable(base_mdl),
        "direct_target_ppo_1600": SB3PolicyCallable(direct_ppo),
        "direct_res_policy_1600": DirectResCallable(base_mdl, res_head, 0.35),
        "swift_mopo_1round": SB3PolicyCallable(swift_1r),
        "swift_mopo_2round": SB3PolicyCallable(swift_2r),
    }

    # Evaluate Part A on `ellipse3d` AND Part B across ALL 10 procedural 3D tracks!
    tracks_eval = {}
    for trk in TRACK_NAMES:
        ev = make_procedural_env(trk, "target", delays=(D_HARD,), kf_scale=KF_HARD, seed=900 + seed)
        pol_subset = pols_part_a if trk in ("figure8", "ellipse3d") else {
            k: pols_part_a[k] for k in ["zeroshot_base", "direct_target_ppo_1600", "swift_mopo_1round", "swift_mopo_2round"]
        }
        tracks_eval[trk] = {mname: evaluate_racing(pol, ev, n_eval=6) for mname, pol in pol_subset.items()}

    return {
        "seed": seed,
        "wall_s": time.time() - t0,
        "onpolicy_lin_rmse": {"round1_batch": rmse_1r, "round2_bootstrap": rmse_2r},
        "tracks": tracks_eval,
    }


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
    print(f"Exp 18 (2-Round Bootstrapped Swift + 10-Track 3D Benchmark) finished in {wall:.1f}s")

    agg_rmse = {
        k: mean_std([o["onpolicy_lin_rmse"][k] for o in outs]) for k in outs[0]["onpolicy_lin_rmse"]
    }
    agg_tracks = {
        trk: {
            m: agg_leaf([o["tracks"][trk][m] for o in outs]) for m in outs[0]["tracks"][trk]
        }
        for trk in TRACK_NAMES
    }

    # Compute 10-track macro-average per seed for the 4 main methods!
    four_m = ["zeroshot_base", "direct_target_ppo_1600", "swift_mopo_1round", "swift_mopo_2round"]
    macro_10 = {}
    for m in four_m:
        seed_means = []
        for o in outs:
            seed_means.append(
                {
                    "gate_frac": float(np.mean([o["tracks"][trk][m]["gate_frac"] for trk in TRACK_NAMES])),
                    "clean_completion": float(np.mean([o["tracks"][trk][m]["clean_completion"] for trk in TRACK_NAMES])),
                    "crash_rate": float(np.mean([o["tracks"][trk][m]["crash_rate"] for trk in TRACK_NAMES])),
                    "track_err_m": float(np.mean([o["tracks"][trk][m]["track_err_m"] for trk in TRACK_NAMES])),
                }
            )
        macro_10[m] = agg_leaf(seed_means)

    save_json(
        "exp18_bootstrapped_swift_and_10tracks.json",
        {
            "seeds": SEEDS,
            "target_budget_ticks": 1600,
            "wall_s": wall,
            "onpolicy_lin_rmse": agg_rmse,
            "macro_10_tracks": macro_10,
            "per_track": agg_tracks,
        },
    )

    print(
        f"\n=== On-Policy High-Speed Linear Residual RMSE: 1-Batch={fmt(agg_rmse['round1_batch'])} vs "
        f"2-Round Bootstrap={fmt(agg_rmse['round2_bootstrap'])} m/s^2 ==="
    )

    print("\n=== Part A: Equal 16.0s (1,600-tick) Target Budget Comparison on `ellipse3d` (kf=1.25, d=3) ===")
    for m, d in agg_tracks["ellipse3d"].items():
        print(
            f"  {m:24s} gf={fmt(d['gate_frac']):>13} | cln={d['clean_completion']['mean']:.2f} | "
            f"cr={d['crash_rate']['mean']:.2f} | trk={d['track_err_m']['mean']:.3f}m"
        )

    print("\n=== Part B: 10-Track Procedural 3D Benchmark Macro-Average (kf=1.25, d=3, 3 seeds) ===")
    for m, d in macro_10.items():
        print(
            f"  {m:24s} gf={fmt(d['gate_frac']):>13} | cln={fmt(d['clean_completion']):>13} | "
            f"cr={fmt(d['crash_rate']):>13} | trk={d['track_err_m']['mean']:.3f}m"
        )

    print("\n  Per-track Clean Completion (zeroshot -> 1-round -> 2-round bootstrap):")
    for trk in TRACK_NAMES:
        z = agg_tracks[trk]["zeroshot_base"]["clean_completion"]["mean"]
        r1 = agg_tracks[trk]["swift_mopo_1round"]["clean_completion"]["mean"]
        r2 = agg_tracks[trk]["swift_mopo_2round"]["clean_completion"]["mean"]
        gf2 = agg_tracks[trk]["swift_mopo_2round"]["gate_frac"]["mean"]
        print(f"    {trk:18s}: zs={z:.2f} -> 1r={r1:.2f} -> 2r={r2:.2f} (gf={gf2:.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2))
    # Panel A: Equal 1,600-tick Target Budget Comparison on `ellipse3d`
    m_a = list(agg_tracks["ellipse3d"].keys())
    xa = np.arange(len(m_a))
    axes[0].bar(
        xa - 0.18,
        [agg_tracks["ellipse3d"][m]["gate_frac"]["mean"] for m in m_a],
        0.35,
        yerr=[agg_tracks["ellipse3d"][m]["gate_frac"]["std"] for m in m_a],
        capsize=2,
        label="Gate Fraction",
    )
    axes[0].bar(
        xa + 0.18,
        [agg_tracks["ellipse3d"][m]["clean_completion"]["mean"] for m in m_a],
        0.35,
        yerr=[agg_tracks["ellipse3d"][m]["clean_completion"]["std"] for m in m_a],
        capsize=2,
        label="Clean Completion",
    )
    axes[0].set_xticks(xa, [m.replace("_1600", "").replace("_", "\n") for m in m_a], fontsize=7.5)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("A. Equal 16s Target Budget (ellipse3d, kf=1.25)")
    axes[0].legend(fontsize=8)

    # Panel B: 10-Track Procedural 3D Clean Completion per Track
    xt = np.arange(len(TRACK_NAMES))
    for j, (m, lbl) in enumerate([("zeroshot_base", "Zero-Shot"), ("swift_mopo_1round", "Swift-MOPO 1-Rd"), ("swift_mopo_2round", "Swift-MOPO 2-Rd")]):
        axes[1].bar(
            xt + (j - 1) * 0.27,
            [agg_tracks[trk][m]["clean_completion"]["mean"] for trk in TRACK_NAMES],
            0.27,
            label=lbl,
        )
    axes[1].set_xticks(xt, [t.replace("_3d", "").replace("_", "\n") for t in TRACK_NAMES], fontsize=6.5)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Clean Completion Rate")
    axes[1].set_title("B. Zero-Shot 10-Track 3D Generalization (kf=1.25, d=3)")
    axes[1].legend(fontsize=7.5)

    plt.tight_layout()
    plt.savefig(f"{OUT}/exp18_bootstrapped_swift_and_10tracks.png", dpi=150)


if __name__ == "__main__":
    main()
