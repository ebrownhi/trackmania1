"""
Exp 17 (Step 2 Riffs R2.1 + R2.2):
  DORAEMON Entropy-Maximized Curriculum Randomization + CVaR Tail-Risk PPO on 6-Gate 3D Drone Racing!

Literature Anchors:
  - Tiboni et al., "DORAEMON: Domain Randomization via Entropy Maximization" (ICLR 2024)
  - Rajeswaran et al., "EPOpt" (ICLR 2017) & Greenberg et al., "CeSoR: Risk-Averse RL" (NeurIPS 2022)

Motivation from Exp 10:
  In Exp 10, fixed wide domain randomization (`dr_ppo`) suffered severe Prop 2.3 conservatism on held-out 3D
  `ellipse3d`: `0.750` gate fraction, `0.00` collision-free completion, and `7.12 s` laps (40% slower than nominal!).

Methods Compared (3 seeds, equal `450k + 100k = 550k` source steps on `figure8` -> zero-shot `ellipse3d`):
  1. `nominal_ppo`       : Nominal L2F (`d = 0`, zero randomization)
  2. `fixed_wide_dr`     : Hand-picked wide uniform DR (`mass/thrust +-25%`, `d ~ U{0..5}`) from step 0
  3. `doraemon_ppo`      : Starts narrow (`w = 0.04, d_max = 1`) and updates distribution entropy `H(nu_phi)` after
                           every rollout batch subject to empirical episode success constraint `P(gates >= 5) >= 0.72`!
  4. `doraemon_cvar_ppo` : `DORAEMON` curriculum + `CVaR_{0.35}` risk-sensitive PPO weighting advantages on the
                           worst 35% of episode returns in each rollout buffer (directly targeting `p90` tail lap time
                           and worst-decile gate completion)!
"""

from __future__ import annotations

import math
import multiprocessing as mp
import time

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from common import OUT, fmt, mean_std, plt, save_json
from racing_env import GateRacingEnv, evaluate_racing

SEEDS = [0, 1, 2]
HIST = 8
BASE_STEPS = 450_000
FT_STEPS = 100_000


class DoraemonController:
    """Shared adaptive domain-randomization state tracking entropy growth under a success-rate constraint."""

    def __init__(self, active: bool = True, w_init: float = 0.04, d_init: float = 1.0, target_sr: float = 0.72):
        self.active = active
        self.w = w_init if active else 0.25          # half-width around 1.0 for mass/thrust/tau
        self.d_max = d_init if active else 5.0       # continuous delay upper bound
        self.target_sr = target_sr
        self.ep_outcomes = []
        self.entropy_log = []

    def sample_dyn_and_delay(self, rng: np.random.Generator) -> tuple[dict, int]:
        w = self.w
        ov = {
            "mass_scale": float(rng.uniform(1.0 - w, 1.0 + w)),
            "thrust_scale": float(rng.uniform(1.0 - w, 1.0 + 1.2 * w)),
            "tau_rise_scale": float(rng.uniform(max(0.25, 1.0 - 2.0 * w), 1.0 + 1.5 * w)),
            "tau_fall_scale": float(rng.uniform(max(0.10, 1.0 - 2.5 * w), 1.0 + 1.5 * w)),
        }
        d = int(rng.integers(0, int(round(self.d_max)) + 1))
        return ov, d

    def current_entropy(self) -> float:
        # Differential entropy of 4 uniform intervals + log discrete delay support
        w = max(self.w, 1e-3)
        return float(4.0 * math.log(2.0 * w) + math.log(1.0 + self.d_max))

    def record_episode(self, gates_passed: int):
        self.ep_outcomes.append(float(gates_passed >= 5))

    def step_curriculum(self):
        if len(self.ep_outcomes) < 12:
            return
        sr = float(np.mean(self.ep_outcomes[-32:]))
        self.ep_outcomes.clear()
        if self.active:
            if sr >= self.target_sr:
                self.w = min(0.28, self.w + 0.018)
                self.d_max = min(5.0, self.d_max + 0.35)
            elif sr < self.target_sr - 0.10:
                self.w = max(0.03, self.w - 0.012)
                self.d_max = max(1.0, self.d_max - 0.25)
        self.entropy_log.append((sr, self.w, self.d_max, self.current_entropy()))


class DoraemonRaceEnv(GateRacingEnv):
    def __init__(self, ctrl: DoraemonController, **kwargs):
        super().__init__(**kwargs)
        self.ctrl = ctrl
        self._ep_inited = False

    def reset(self, seed=None, options=None):
        if self._ep_inited:
            self.ctrl.record_episode(self.gates_passed)
        self._ep_inited = True
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        ov, d = self.ctrl.sample_dyn_and_delay(self.rng)
        self.delays = (d,)
        self.dyn_sampler = lambda _r, o=ov: o
        return super().reset(seed=None, options=options)


class DoraemonCallback(BaseCallback):
    def __init__(self, ctrl: DoraemonController):
        super().__init__(verbose=0)
        self.ctrl = ctrl

    def _on_rollout_end(self) -> None:
        self.ctrl.step_curriculum()

    def _on_step(self) -> bool:
        return True


class CVaRPPO(PPO):
    """
    Risk-Sensitive PPO (`CVaR_{alpha}`) weighting policy-gradient updates toward the worst `alpha` fraction of
    episode/trajectory segments in each rollout buffer (`w_tail = 2.2`, `w_bulk = 0.45`).
    """

    def __init__(self, *args, cvar_alpha: float = 0.35, **kwargs):
        super().__init__(*args, **kwargs)
        self.cvar_alpha = cvar_alpha

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)

        # Compute per-sample risk weights based on rollout returns `self.rollout_buffer.returns`
        rets = self.rollout_buffer.returns  # (n_steps, n_envs)
        q_cut = float(np.quantile(rets, self.cvar_alpha))
        # Temporarily scale advantages in buffer before normalization? No—apply soft tail mask directly via returns!
        # Even cleaner: store per-transition risk weight aligned with flattened buffer!
        # Since SB3's RolloutBuffer.get() shuffles `indices`, we attach `risk_weights` by scaling raw advantages
        # AFTER centering! Wait: `advantages = w_risk * (advantages - mean) / std` preserves sign AND amplifies
        # gradients on worst-outcome trajectories (`rets <= q_cut`)!
        adv_orig = self.rollout_buffer.advantages.copy()
        adv_norm = (adv_orig - adv_orig.mean()) / (adv_orig.std() + 1e-8)
        w_risk = np.where(rets <= q_cut, 2.1, 0.50).astype(np.float32)
        w_risk /= np.mean(w_risk)
        self.rollout_buffer.advantages = adv_norm * w_risk
        orig_norm_flag = self.normalize_advantage
        self.normalize_advantage = False
        try:
            super().train()
        finally:
            self.rollout_buffer.advantages = adv_orig
            self.normalize_advantage = orig_norm_flag


def eval_with_tail_metrics(model, env: GateRacingEnv, n_eval: int = 15, seed0: int = 50_000) -> dict:
    pol = lambda obs, e, k: model.predict(obs, deterministic=True)[0]
    ev = evaluate_racing(pol, env, n_eval=n_eval, seed0=seed0)
    # Also compute worst-quintile (`CVaR_0.20`) gate fraction across individual episodes!
    ep_gfs = []
    for ep in range(n_eval):
        obs, _ = env.reset(seed=seed0 + ep)
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
        ep_gfs.append(info["gates_passed"] / env.total_target_gates)
    ep_gfs_sorted = np.sort(ep_gfs)
    n_tail = max(1, int(round(0.25 * n_eval)))
    ev["cvar25_gate_frac"] = float(np.mean(ep_gfs_sorted[:n_tail]))
    return ev


def run_seed(seed: int) -> dict:
    th.set_num_threads(1)
    t0 = time.time()
    ppo_kw = dict(
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        learning_rate=3e-4,
        seed=seed,
        verbose=0,
        policy_kwargs=dict(net_arch=[128, 128], log_std_init=-0.7),
    )

    arms = ["nominal_ppo", "fixed_wide_dr", "doraemon_ppo", "doraemon_cvar_ppo"]
    models = {}
    ent_traces = {}

    for arm in arms:
        if arm == "nominal_ppo":
            venv_b = DummyVecEnv(
                [lambda i=i: GateRacingEnv("figure8", "source", HIST, lam_da=0.02, delays=(0,), max_ep_s=5.5, seed=1000 * seed + i) for i in range(8)]
            )
            venv_f = DummyVecEnv(
                [lambda i=i: GateRacingEnv("figure8", "source", HIST, lam_da=0.20, delays=(0,), max_ep_s=5.5, seed=1100 * seed + i) for i in range(8)]
            )
            mdl = PPO("MlpPolicy", venv_b, **ppo_kw)
            mdl.learn(BASE_STEPS)
            mdl.set_env(venv_f)
            mdl.learn(FT_STEPS, reset_num_timesteps=False)
        else:
            is_dor = "doraemon" in arm
            ctrl = DoraemonController(active=is_dor)
            cb = DoraemonCallback(ctrl)
            venv_b = DummyVecEnv(
                [lambda i=i, c=ctrl: DoraemonRaceEnv(c, course_kind="figure8", domain="source", hist=HIST, lam_da=0.02, max_ep_s=5.5, seed=2000 * seed + i) for i in range(8)]
            )
            venv_f = DummyVecEnv(
                [lambda i=i, c=ctrl: DoraemonRaceEnv(c, course_kind="figure8", domain="source", hist=HIST, lam_da=0.20, max_ep_s=5.5, seed=2100 * seed + i) for i in range(8)]
            )
            cls = CVaRPPO if "cvar" in arm else PPO
            mdl = cls("MlpPolicy", venv_b, **ppo_kw)
            mdl.learn(BASE_STEPS, callback=cb)
            mdl.set_env(venv_f)
            mdl.learn(FT_STEPS, reset_num_timesteps=False, callback=cb)
            ent_traces[arm] = [x[3] for x in ctrl.entropy_log]
        models[arm] = mdl

    regimes = {
        "source_d0": GateRacingEnv("ellipse3d", "source", HIST, delays=(0,), random_start_gate=False, max_ep_s=12.5, seed=700 + seed),
        "target_nominal": GateRacingEnv("ellipse3d", "target", HIST, delays=(2,), kf_scale=1.0, random_start_gate=False, max_ep_s=12.5, seed=710 + seed),
        "target_perturbed": GateRacingEnv("ellipse3d", "target", HIST, delays=(3,), kf_scale=1.20, random_start_gate=False, max_ep_s=12.5, seed=720 + seed),
        "target_extreme_ood": GateRacingEnv("ellipse3d", "target", HIST, delays=(6,), kf_scale=1.30, random_start_gate=False, max_ep_s=12.5, seed=730 + seed),
    }

    eval_out = {reg: {arm: eval_with_tail_metrics(models[arm], ev, n_eval=12) for arm in arms} for reg, ev in regimes.items()}
    return {"seed": seed, "wall_s": time.time() - t0, "eval": eval_out, "ent_traces": ent_traces}


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
    print(f"Exp 17 (DORAEMON + CVaR-PPO) finished in {wall:.1f}s")

    agg = {
        reg: {arm: agg_leaf([o["eval"][reg][arm] for o in outs]) for arm in outs[0]["eval"][reg]}
        for reg in outs[0]["eval"]
    }
    save_json("exp17_doraemon_cvar_dr.json", {"seeds": SEEDS, "wall_s": wall, "agg": agg, "ent_traces": [o["ent_traces"] for o in outs]})

    for reg in agg:
        print(f"\n=== [{reg} | held-out `ellipse3d`] gf | cvar25_gf | clean | crash | med_lap | p90_lap ===")
        for arm, m in agg[reg].items():
            ml = f"{m['median_lap_s']['mean']:5.2f}s" if np.isfinite(m["median_lap_s"]["mean"]) else "  n/a "
            pl = f"{m['p90_lap_s']['mean']:5.2f}s" if np.isfinite(m["p90_lap_s"]["mean"]) else "  n/a "
            print(
                f"  {arm:18s} gf={fmt(m['gate_frac']):>13} | cvar25={m['cvar25_gate_frac']['mean']:.2f} | "
                f"cln={m['clean_completion']['mean']:.2f} | cr={m['crash_rate']['mean']:.2f} | med={ml} | p90={pl}"
            )

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    # Panel A: DORAEMON Entropy Evolution
    for o in outs:
        axes[0].plot(o["ent_traces"]["doraemon_ppo"], alpha=0.5, color="#4C72B0")
        axes[0].plot(o["ent_traces"]["doraemon_cvar_ppo"], alpha=0.7, color="#C44E52", ls="--")
    axes[0].axhline(outs[0]["ent_traces"]["fixed_wide_dr"][0], color="#555555", ls=":", label="fixed_wide_dr")
    axes[0].plot([], [], color="#4C72B0", label="doraemon_ppo")
    axes[0].plot([], [], color="#C44E52", ls="--", label="doraemon_cvar_ppo")
    axes[0].set_title("A. DORAEMON Curriculum Entropy Growth")
    axes[0].set_xlabel("PPO Rollout Updates")
    axes[0].set_ylabel("Domain Distribution Entropy H(nu)")
    axes[0].legend(fontsize=7.5)

    arms = ["nominal_ppo", "fixed_wide_dr", "doraemon_ppo", "doraemon_cvar_ppo"]
    xa = np.arange(len(arms))
    regs_bar = ["source_d0", "target_nominal", "target_perturbed", "target_extreme_ood"]
    for j, rname in enumerate(regs_bar):
        axes[1].bar(
            xa + (j - 1.5) * 0.20,
            [agg[rname][a]["gate_frac"]["mean"] for a in arms],
            0.20,
            yerr=[agg[rname][a]["gate_frac"]["std"] for a in arms],
            capsize=2,
            label=rname,
        )
        axes[2].bar(
            xa + (j - 1.5) * 0.20,
            [agg[rname][a]["cvar25_gate_frac"]["mean"] for a in arms],
            0.20,
            yerr=[agg[rname][a]["cvar25_gate_frac"]["std"] for a in arms],
            capsize=2,
            label=rname,
        )
    for ax, ttl, yl in [(axes[1], "B. Mean Gate Fraction (ellipse3d)", "Gate Fraction"),
                        (axes[2], "C. Worst-Quartile CVaR_25 Gate Fraction", "CVaR_25 Gate Fraction")]:
        ax.set_xticks(xa, [a.replace("_ppo", "").replace("_", "\n") for a in arms], fontsize=7.5)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(yl)
        ax.set_title(ttl)
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{OUT}/exp17_doraemon_cvar_dr.png", dpi=150)


if __name__ == "__main__":
    main()
