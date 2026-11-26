"""
Exp 16 (Step 1 Riffs R1.2 + R1.3):
  - Part A (R1.2 — Architectural EMA Bandwidth Filter `a_t = alpha * a_{t-1} + (1 - alpha) * u_theta(o_t)`):
      Test whether wrapping an unregularized (`lam_da = 0`) PPO actor in a structural first-order low-pass filter
      (`alpha in {0.0, 0.50, 0.70}`) enforces Bode phase-margin smoothness across delays `d in {0, 2, 4, 6, 8}`
      without reward shaping or custom actor losses!
  - Part B (R1.3 — Hybrid Residual RL over Classical `GeometricGateController` (`AC-MPC` / Johannink et al. Style)):
      Compare `a = clip(a_classical(o) + 0.45 * pi_theta(o), -1, 1)` against pure End-to-End PPO and pure
      `classical_expert` across sample budgets `{50k, 100k, 250k, 450k}` on 6-gate 3D racing (`figure8` train ->
      held-out `ellipse3d` across `source_l2f`, `target_nominal`, and `target_perturbed kf x 1.20, d = 3`)!
"""

from __future__ import annotations

import gymnasium as gym
import multiprocessing as mp
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from common import OUT, fmt, mean_std, plt, save_json
from exp11_caps_smoothness import eval_caps, make_caps_hover_env
from racing_env import GateRacingEnv, GeometricGateController, evaluate_racing

SEEDS = [0, 1, 2]
HIST = 8
TRAIN_DELAYS = (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------------------------------
# Part A: Architectural First-Order Low-Pass Actuator Layer (`alpha` EMA Wrapper)
# --------------------------------------------------------------------------------------------------


class EMAFilterWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, alpha: float):
        super().__init__(env)
        self.alpha = alpha
        self.a_filt = np.zeros(4, dtype=np.float64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.a_filt = self.env.unwrapped.drone.hover_action.astype(np.float64).copy()
        return obs, info

    def step(self, action):
        u = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.a_filt = self.alpha * self.a_filt + (1.0 - self.alpha) * u
        return self.env.step(self.a_filt)


def eval_ema_hover(model, alpha: float, domain: str, d: int, n_eval: int = 16) -> dict:
    base_env = make_caps_hover_env(0.0, 777, domain=domain, hist=HIST, delays=(d,), ep_s=4.0, noisy_zoh=False)
    env = EMAFilterWrapper(base_env, alpha) if alpha > 0.0 else base_env
    from ppo_lib import hf_power

    errs, crashes, das, hfs = [], [], [], []
    for ep in range(n_eval):
        o, _ = env.reset(seed=12_000 + ep)
        E, A, crashed = [], [], False
        while True:
            u, _ = model.predict(o, deterministic=True)
            o, _, term, trunc, info = env.step(u)
            E.append(info["e"])
            A.append(info["a"])
            if term:
                crashed = True
                break
            if trunc:
                break
        A_arr = np.array(A)
        crashes.append(float(crashed))
        errs.append(float(np.mean(E[len(E) // 2 :])) if not crashed else np.nan)
        das.append(float(np.mean(np.linalg.norm(np.diff(A_arr, axis=0), axis=1))) if len(A_arr) > 1 else 0.0)
        hfs.append(hf_power(A_arr) if len(A_arr) > 8 else 0.0)
    errs_arr = np.array(errs)
    return {
        "eff_err": float(np.mean(np.where(np.isfinite(errs_arr), errs_arr, 1.0))),
        "crash": float(np.mean(crashes)),
        "mean_da": float(np.mean(das)),
        "hf_power": float(np.mean(hfs)),
    }


def run_part_a_seed(seed: int, steps: int = 600_000) -> dict:
    torch.set_num_threads(1)
    alphas = [0.0, 0.50, 0.70]
    out = {}
    for al in alphas:
        def mk(i, a=al):
            base = make_caps_hover_env(0.0, 1000 * seed + i, domain="source", hist=HIST, delays=TRAIN_DELAYS)
            return EMAFilterWrapper(base, a) if a > 0.0 else base

        venv = DummyVecEnv([lambda i=i: mk(i) for i in range(8)])
        mdl = PPO(
            "MlpPolicy",
            venv,
            n_steps=512,
            batch_size=512,
            n_epochs=10,
            learning_rate=3e-4,
            seed=seed,
            verbose=0,
            policy_kwargs=dict(net_arch=[128, 128]),
        )
        mdl.learn(steps)
        out[f"ema_{al:.2f}"] = {
            dom: {str(d): eval_ema_hover(mdl, al, dom, d) for d in [0, 2, 4, 6, 8]}
            for dom in ["source", "target"]
        }
    return out


# --------------------------------------------------------------------------------------------------
# Part B: Hybrid Residual-RL Policy over `GeometricGateController` (`a = a_classical + 0.45 * pi_theta`)
# --------------------------------------------------------------------------------------------------


class ResidualOverClassicalEnv(gym.Wrapper):
    """Wraps `GateRacingEnv` so PPO outputs bounded residual corrections over `GeometricGateController`."""

    def __init__(self, env: GateRacingEnv, res_scale: float = 0.45):
        super().__init__(env)
        self.ctrl = GeometricGateController()
        self.res_scale = res_scale
        self._ep_step = 0
        # Append `a_classical (4)` to actor observation so residual actor knows the nominal command it is correcting!
        low = np.concatenate([env.observation_space.low, -np.ones(4, dtype=np.float32)])
        high = np.concatenate([env.observation_space.high, np.ones(4, dtype=np.float32)])
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self._last_a_cls = np.zeros(4, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.ctrl.reset_episode()
        self._ep_step = 0
        self._last_a_cls = self.ctrl(obs, self.env, 0).astype(np.float32)
        return np.concatenate([obs, self._last_a_cls]).astype(np.float32), info

    def step(self, res_action):
        da_pol = np.clip(np.asarray(res_action, dtype=np.float64), -1.0, 1.0)
        a_total = np.clip(self._last_a_cls.astype(np.float64) + self.res_scale * da_pol, -1.0, 1.0)
        obs, reward, term, trunc, info = self.env.step(a_total)
        # Small L2 regularization on residual magnitude encourages minimal intervention when classical suffices
        reward -= 0.015 * float(da_pol @ da_pol)
        self._ep_step += 1
        self._last_a_cls = self.ctrl(obs, self.env, self._ep_step).astype(np.float32)
        return np.concatenate([obs, self._last_a_cls]).astype(np.float32), float(reward), term, trunc, info


class ResidualPolicyCallable:
    def __init__(self, model, res_scale: float = 0.45):
        self.model = model
        self.res_scale = res_scale
        self.ctrl = GeometricGateController()

    def reset_episode(self):
        self.ctrl.reset_episode()

    def __call__(self, obs: np.ndarray, env: GateRacingEnv, step_idx: int) -> np.ndarray:
        a_cls = self.ctrl(obs, env, step_idx).astype(np.float32)
        obs_aug = np.concatenate([obs, a_cls]).astype(np.float32)
        da, _ = self.model.predict(obs_aug, deterministic=True)
        return np.clip(a_cls + self.res_scale * da, -1.0, 1.0)


class E2ECallable:
    def __init__(self, model):
        self.model = model

    def __call__(self, obs, env, step_idx):
        a, _ = self.model.predict(obs, deterministic=True)
        return a


def run_part_b_seed(seed: int) -> dict:
    torch.set_num_threads(1)
    budgets = [50_000, 100_000, 250_000, 450_000]
    deltas = [budgets[0]] + [budgets[i] - budgets[i - 1] for i in range(1, len(budgets))]

    def dyn_s(rng):
        return {"mass_scale": rng.uniform(0.85, 1.15), "thrust_scale": rng.uniform(0.85, 1.25)}

    venv_e2e = DummyVecEnv(
        [
            lambda i=i: GateRacingEnv("figure8", "source", HIST, lam_da=0.04, delays=(0, 1, 2, 3), dyn_sampler=dyn_s, max_ep_s=5.5, seed=2000 * seed + i)
            for i in range(8)
        ]
    )
    venv_res = DummyVecEnv(
        [
            lambda i=i: ResidualOverClassicalEnv(
                GateRacingEnv("figure8", "source", HIST, lam_da=0.04, delays=(0, 1, 2, 3), dyn_sampler=dyn_s, max_ep_s=5.5, seed=3000 * seed + i),
                res_scale=0.45,
            )
            for i in range(8)
        ]
    )

    ppo_kw = dict(
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        learning_rate=3e-4,
        seed=seed,
        verbose=0,
        policy_kwargs=dict(net_arch=[128, 128], log_std_init=-0.7),
    )
    m_e2e = PPO("MlpPolicy", venv_e2e, **ppo_kw)
    m_res = PPO("MlpPolicy", venv_res, **ppo_kw)

    eval_envs = {
        "source_l2f": GateRacingEnv("ellipse3d", "source", HIST, delays=(0,), random_start_gate=False, max_ep_s=12.5, seed=800 + seed),
        "target_nominal": GateRacingEnv("ellipse3d", "target", HIST, delays=(2,), kf_scale=1.0, random_start_gate=False, max_ep_s=12.5, seed=810 + seed),
        "target_perturbed": GateRacingEnv("ellipse3d", "target", HIST, delays=(3,), kf_scale=1.20, random_start_gate=False, max_ep_s=12.5, seed=820 + seed),
    }

    curve = {"classical_0k": {t: evaluate_racing(GeometricGateController(), ev, n_eval=8) for t, ev in eval_envs.items()}}
    curve["e2e_ppo"] = {}
    curve["residual_over_classical"] = {}

    for b, dl in zip(budgets, deltas):
        m_e2e.learn(dl, reset_num_timesteps=False)
        m_res.learn(dl, reset_num_timesteps=False)
        b_key = f"{b // 1000}k"
        curve["e2e_ppo"][b_key] = {t: evaluate_racing(E2ECallable(m_e2e), ev, n_eval=8) for t, ev in eval_envs.items()}
        curve["residual_over_classical"][b_key] = {
            t: evaluate_racing(ResidualPolicyCallable(m_res, 0.45), ev, n_eval=8) for t, ev in eval_envs.items()
        }

    return curve


def job(seed: int):
    t0 = time.time()
    pa = run_part_a_seed(seed)
    pb = run_part_b_seed(seed)
    return {"seed": seed, "wall_s": time.time() - t0, "part_a": pa, "part_b": pb}


def agg_leaf(ds):
    out = {}
    for k in ds[0]:
        vals = [d[k] for d in ds if np.isfinite(d[k])]
        out[k] = mean_std(vals) if vals else {"mean": float("nan"), "std": float("nan"), "n": 0, "values": []}
    return out


def main():
    t0 = time.time()
    with mp.get_context("spawn").Pool(3) as pool:
        outs = pool.map(job, SEEDS)
    wall = time.time() - t0
    print(f"Exp 16 (EMA Bandwidth + Residual Policy RL) finished in {wall:.1f}s")

    agg_a = {
        al: {
            dom: {d: agg_leaf([o["part_a"][al][dom][d] for o in outs]) for d in outs[0]["part_a"][al][dom]}
            for dom in outs[0]["part_a"][al]
        }
        for al in outs[0]["part_a"]
    }

    agg_b = {
        "classical_0k": {t: agg_leaf([o["part_b"]["classical_0k"][t] for o in outs]) for t in outs[0]["part_b"]["classical_0k"]},
        "e2e_ppo": {
            bk: {t: agg_leaf([o["part_b"]["e2e_ppo"][bk][t] for o in outs]) for t in outs[0]["part_b"]["e2e_ppo"][bk]}
            for bk in outs[0]["part_b"]["e2e_ppo"]
        },
        "residual_over_classical": {
            bk: {
                t: agg_leaf([o["part_b"]["residual_over_classical"][bk][t] for o in outs])
                for t in outs[0]["part_b"]["residual_over_classical"][bk]
            }
            for bk in outs[0]["part_b"]["residual_over_classical"]
        },
    }

    save_json("exp16_bandwidth_and_residual_policy.json", {"seeds": SEEDS, "wall_s": wall, "part_a_ema": agg_a, "part_b_residual_policy": agg_b})

    print("\n=== Part A: Architectural EMA Filter on PyBullet Target Hold Error [m] & HF Power ===")
    for al in agg_a:
        row = " | ".join(
            f"d={d}: {agg_a[al]['target'][d]['eff_err']['mean']:.3f}m (HF={agg_a[al]['target'][d]['hf_power']['mean']:.2f})"
            for d in ["0", "2", "4", "6", "8"]
        )
        print(f"  {al:10s} -> {row}")

    print("\n=== Part B: Sample-Efficiency & Held-Out `ellipse3d` Gate Fraction (Classical vs E2E vs Residual-RL) ===")
    for t in ["source_l2f", "target_nominal", "target_perturbed"]:
        cl0 = agg_b["classical_0k"][t]["gate_frac"]
        print(f"  [{t}] classical_0k: gf={fmt(cl0)} (cln={agg_b['classical_0k'][t]['clean_completion']['mean']:.2f})")
        for bk in ["50k", "100k", "250k", "450k"]:
            ge = agg_b["e2e_ppo"][bk][t]
            gr = agg_b["residual_over_classical"][bk][t]
            print(
                f"    @{bk:4s} | e2e_ppo: gf={fmt(ge['gate_frac']):>13} (cln={ge['clean_completion']['mean']:.2f}) | "
                f"res_over_cls: gf={fmt(gr['gate_frac']):>13} (cln={gr['clean_completion']['mean']:.2f})"
            )

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))
    ds_x = [0, 2, 4, 6, 8]
    for al, mk in [("ema_0.00", "o"), ("ema_0.50", "s"), ("ema_0.70", "D")]:
        g = agg_a[al]["target"]
        axes[0].errorbar(
            ds_x,
            [g[str(d)]["eff_err"]["mean"] for d in ds_x],
            yerr=[g[str(d)]["eff_err"]["std"] for d in ds_x],
            marker=mk,
            capsize=2,
            label=al,
        )
    axes[0].set_title("A. Architectural EMA Filter (Target Hold Err)")
    axes[0].set_xlabel("Action Delay d [ticks]")
    axes[0].set_ylabel("Effective Hold Error [m]")
    axes[0].legend(fontsize=8)

    b_ks = [50, 100, 250, 450]
    for idx_ax, tname in [(1, "source_l2f"), (2, "target_perturbed")]:
        cl_val = agg_b["classical_0k"][tname]["gate_frac"]["mean"]
        axes[idx_ax].axhline(cl_val, ls="--", color="#555555", label="classical_expert (0k)")
        for m, mk in [("e2e_ppo", "o"), ("residual_over_classical", "D")]:
            axes[idx_ax].errorbar(
                b_ks,
                [agg_b[m][f"{b}k"][tname]["gate_frac"]["mean"] for b in b_ks],
                yerr=[agg_b[m][f"{b}k"][tname]["gate_frac"]["std"] for b in b_ks],
                marker=mk,
                capsize=2,
                label=m,
            )
        axes[idx_ax].set_title(f"{'B' if idx_ax == 1 else 'C'}. Held-Out ellipse3d ({tname})")
        axes[idx_ax].set_xlabel("PPO Training Steps [x1000]")
        axes[idx_ax].set_ylabel("Gate Completion Fraction")
        axes[idx_ax].set_ylim(-0.02, 1.08)
        axes[idx_ax].legend(fontsize=7.5)

    plt.tight_layout()
    plt.savefig(f"{OUT}/exp16_bandwidth_and_residual_policy.png", dpi=150)


if __name__ == "__main__":
    main()
