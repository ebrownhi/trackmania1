"""
Exp 11 (Step 1 Riff R1.1): CAPS + L2C2 Actor-Loss Smoothness vs. Reward-Penalty Smoothness in PPO.

Motivation from Exp 7b/7c:
  - Adding `-lam * ||a_t - a_{t-1}||^2` (lam = 0.3) to the MDP reward from scratch stalled PPO on 2/3 seeds
    (`eff_err = 0.68 +- 0.45 m`) because early Gaussian exploration actions incur large negative rewards that
    depress the critic V(s) against moving toward the goal.
  - CAPS (Mysore et al., ICRA 2021) + L2C2 (Kobayashi, IROS 2022) remove smoothness from the reward (`lam_rew = 0`)
    and instead regularize the deterministic actor mean `mu_theta(o_t)` directly inside the actor gradient:
        L_CAPS = lam_T * E[ ||mu(o_t) - mu(o_{t-1})||^2 ] + lam_S * E[ ||mu(o_t) - mu(o_t + eps)||^2 ]
    preserving unbiased critic returns AND unpenalized stochastic exploration noise `sigma_expl`!

Cells (3 seeds x 1,000,000 steps in L2F HoverEnv with `hist=8` action buffer + latency DR `d ~ U{0..4}`):
  1. `unreg_lam0`      : Standard PPO, reward lam = 0.0
  2. `reward_lam0.3`   : Standard PPO, reward lam = 0.3 from scratch (reproduces Exp 7b exploration stall)
  3. `caps_temporal`   : CAPSPPO (`lam_T = 0.8, lam_S = 0.0`), reward lam = 0.0
  4. `caps_l2c2`       : CAPSPPO (`lam_T = 0.8, lam_S = 0.4, sigma_S = 0.04`), reward lam = 0.0

Evaluated zero-shot across:
  - `source`           : L2F clean obs, d in {0, 2, 4}
  - `target_clean`     : PyBullet CF2X clean obs, d in {0, 2, 4, 6}
  - `target_noisy_zoh` : PyBullet CF2X with pos/gyro sensor noise + 25% ZOH packet dropout, d in {0, 2, 4}
"""

from __future__ import annotations

from collections import deque
import contextlib
import io
import multiprocessing as mp
import sys
import time

import gymnasium as gym
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import DummyVecEnv

from common import OUT, fmt, mean_std, np, plt, save_json
from ppo_lib import hf_power
from sims.drone_pair import DroneState, InterfaceConfig, RL_DT, make_source, make_target

SEEDS = [0, 1, 2]
HIST = 8
TRAIN_DELAYS = (0, 1, 2, 3, 4)
TOTAL_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1_000_000
N_ENVS = 8

CELLS = {
    "unreg_lam0": {"rew_lam": 0.0, "lam_t": 0.0, "lam_s": 0.0, "sig_s": 0.0},
    "reward_lam0.3": {"rew_lam": 0.3, "lam_t": 0.0, "lam_s": 0.0, "sig_s": 0.0},
    "caps_temporal": {"rew_lam": 0.0, "lam_t": 0.8, "lam_s": 0.0, "sig_s": 0.0},
    "caps_l2c2": {"rew_lam": 0.0, "lam_t": 0.8, "lam_s": 0.4, "sig_s": 0.04},
}


class CAPSPPO(PPO):
    """PPO with CAPS temporal mean-action smoothness + L2C2 local spatial Lipschitz regularization."""

    def __init__(self, *args, lam_t: float = 0.0, lam_s: float = 0.0, sig_s: float = 0.04, state_dim: int = 15, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam_t = lam_t
        self.lam_s = lam_s
        self.sig_s = sig_s
        self.state_dim = state_dim

    def _actor_mean(self, obs: th.Tensor) -> th.Tensor:
        latent_pi = self.policy.mlp_extractor.forward_actor(obs)
        return self.policy.action_net(latent_pi)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # Extract chronological consecutive (o_{t-1}, o_t) pairs from rollout buffer where episode did NOT reset
        obs_curr_pairs, obs_prev_pairs = None, None
        if self.lam_t > 0.0:
            obs_arr = self.rollout_buffer.observations  # (n_steps, n_envs, obs_dim)
            ep_starts = self.rollout_buffer.episode_starts  # (n_steps, n_envs)
            valid = ep_starts[1:] < 0.5
            if np.any(valid):
                obs_curr_pairs = th.as_tensor(obs_arr[1:][valid], dtype=th.float32, device=self.device)
                obs_prev_pairs = th.as_tensor(obs_arr[:-1][valid], dtype=th.float32, device=self.device)

        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                values_pred = (
                    values
                    if self.clip_range_vf is None
                    else rollout_data.old_values + th.clamp(values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                caps_loss = th.tensor(0.0, device=self.device)
                if self.lam_t > 0.0 and obs_curr_pairs is not None and len(obs_curr_pairs) > 0:
                    n_p = len(obs_curr_pairs)
                    idx = th.randint(0, n_p, (min(self.batch_size, n_p),), device=self.device)
                    mu_t = self._actor_mean(obs_curr_pairs[idx])
                    mu_tm1 = self._actor_mean(obs_prev_pairs[idx])
                    caps_loss = caps_loss + self.lam_t * F.mse_loss(mu_t, mu_tm1)

                if self.lam_s > 0.0:
                    obs_mb = rollout_data.observations
                    noise = th.zeros_like(obs_mb)
                    noise[:, : self.state_dim] = self.sig_s * th.randn(
                        (obs_mb.shape[0], self.state_dim), device=self.device
                    )
                    mu_clean = self._actor_mean(obs_mb)
                    mu_pert = self._actor_mean(obs_mb + noise)
                    caps_loss = caps_loss + self.lam_s * F.mse_loss(mu_clean, mu_pert)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + caps_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)


def make_caps_hover_env(rew_lam=0.0, seed=0, domain="source", hist=HIST, delays=(0,), ep_s=3.0, noisy_zoh=False):
    class CapsHoverEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (15 + 4 * hist,), np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
            self.rng = np.random.default_rng(seed)
            self.T = int(ep_s / RL_DT)
            self.drones = {}

        def _drone(self, d):
            if d not in self.drones:
                iface = InterfaceConfig(
                    action_delay=d,
                    obs_delay=1 if noisy_zoh else 0,
                    p_drop=0.25 if noisy_zoh else 0.0,
                    dropout_mode="zoh",
                    obs_noise={"p": 0.04, "w": 0.08} if noisy_zoh else {},
                )
                if domain == "target":
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.drones[d] = make_target(iface=iface, seed=seed)
                else:
                    self.drones[d] = make_source(seed=seed, iface=iface)
            return self.drones[d]

        def _obs(self):
            s = self.obs_state
            Rt = s.R.T
            past = list(self.ah)[::-1]
            return np.concatenate([Rt @ (self.goal - s.p), Rt @ s.v, Rt[:, 2], Rt[:, 0], s.w] + past).astype(np.float32)

        def reset(self, seed=None, options=None):
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            r = self.rng
            self.drone = self._drone(int(r.choice(delays)))
            ax = r.normal(size=3)
            ax /= np.linalg.norm(ax)
            ang = r.uniform(0, 0.5)
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
            p0 = np.array([0.0, 0.0, 2.0])
            st = DroneState(p0, R, r.uniform(-0.5, 0.5, 3), r.uniform(-1, 1, 3))
            self.goal = p0 + r.uniform(-1, 1, 3)
            self.obs_state = self.drone.reset(st)
            self.ah = deque([self.drone.hover_action.astype(np.float64).copy()] * hist, maxlen=hist)
            self.k = 0
            return self._obs(), {}

        def step(self, a):
            a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
            self.obs_state = self.drone.step(a)
            s = self.drone.true_state()
            e = float(np.linalg.norm(self.goal - s.p))
            da = a - self.ah[-1]
            r = 1.0 - 0.6 * e - 0.05 * float(np.linalg.norm(s.v)) - 0.002 * float(s.w @ s.w) - rew_lam * float(da @ da)
            self.ah.append(a)
            self.k += 1
            crash = e > 2.0 or s.R[2, 2] < 0.0 or not np.all(np.isfinite(s.p))
            if crash:
                r -= 50.0
            return self._obs(), float(r), bool(crash), self.k >= self.T, {"e": e, "a": a, "state": s}

    return CapsHoverEnv()


def eval_caps(model, domain: str, d: int, noisy_zoh: bool = False, n_eval: int = 20):
    env = make_caps_hover_env(0.0, 777, domain=domain, hist=HIST, delays=(d,), ep_s=4.0, noisy_zoh=noisy_zoh)
    errs, crashes, das, hfs = [], [], [], []
    for ep in range(n_eval):
        o, _ = env.reset(seed=10_000 + ep)
        E, A, crashed = [], [], False
        while True:
            a, _ = model.predict(o, deterministic=True)
            a = np.clip(a, -1.0, 1.0)
            o, _, term, trunc, info = env.step(a)
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
    eff = float(np.mean(np.where(np.isfinite(errs_arr), errs_arr, 1.0)))
    return {
        "eff_err": eff,
        "crash": float(np.mean(crashes)),
        "mean_da": float(np.mean(das)),
        "hf_power": float(np.mean(hfs)),
        "learned": float(eff < 0.20),
    }


def train_job(args):
    th.set_num_threads(1)
    cell_name, seed = args
    cfg = CELLS[cell_name]
    t0 = time.time()
    venv = DummyVecEnv(
        [
            lambda i=i: make_caps_hover_env(cfg["rew_lam"], 1000 * seed + i, domain="source", hist=HIST, delays=TRAIN_DELAYS)
            for i in range(N_ENVS)
        ]
    )
    model = CAPSPPO(
        "MlpPolicy",
        venv,
        lam_t=cfg["lam_t"],
        lam_s=cfg["lam_s"],
        sig_s=cfg["sig_s"],
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
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    model.learn(TOTAL_STEPS)

    ev = {
        "source": {str(d): eval_caps(model, "source", d, False) for d in [0, 2, 4]},
        "target_clean": {str(d): eval_caps(model, "target", d, False) for d in [0, 2, 4, 6]},
        "target_noisy_zoh": {str(d): eval_caps(model, "target", d, True) for d in [0, 2, 4]},
    }
    return {"cell": cell_name, "seed": seed, "wall_s": time.time() - t0, "eval": ev}


def main():
    jobs = [(c, s) for c in CELLS for s in SEEDS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(12, len(jobs))) as pool:
        outs = pool.map(train_job, jobs)
    wall = time.time() - t0
    print(f"Exp 11 (CAPS/L2C2): {len(jobs)} PPO runs x {TOTAL_STEPS} steps in {wall:.1f}s")

    agg = {}
    for c in CELLS:
        o = [x for x in outs if x["cell"] == c]
        agg[c] = {}
        for dom in o[0]["eval"]:
            agg[c][dom] = {}
            for d in o[0]["eval"][dom]:
                agg[c][dom][d] = {m: mean_std([x["eval"][dom][d][m] for x in o]) for m in o[0]["eval"][dom][d]}

    res = {"total_steps": TOTAL_STEPS, "seeds": SEEDS, "cells": CELLS, "wall_s": wall, "agg": agg}
    save_json("exp11_caps_smoothness.json", res)

    for dom in ["source", "target_clean", "target_noisy_zoh"]:
        print(f"\n=== [{dom}] eff_err [m] | crash | mean||da|| | HF (>10Hz) | seeds_learned ===")
        for c in CELLS:
            for d in agg[c][dom]:
                m = agg[c][dom][d]
                print(
                    f"  {c:15s} d={d}: err={fmt(m['eff_err']):>14} | cr={fmt(m['crash']):>14} | "
                    f"da={fmt(m['mean_da']):>14} | hf={fmt(m['hf_power']):>14} | ok={m['learned']['mean']:.2f}"
                )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    d_clean = ["0", "2", "4", "6"]
    d_noisy = ["0", "2", "4"]
    for c in CELLS:
        g_cl = agg[c]["target_clean"]
        axes[0].errorbar(
            [int(d) for d in d_clean],
            [g_cl[d]["eff_err"]["mean"] for d in d_clean],
            yerr=[g_cl[d]["eff_err"]["std"] for d in d_clean],
            marker="o",
            capsize=2,
            label=c,
        )
        axes[1].errorbar(
            [int(d) for d in d_clean],
            [g_cl[d]["hf_power"]["mean"] for d in d_clean],
            yerr=[g_cl[d]["hf_power"]["std"] for d in d_clean],
            marker="s",
            capsize=2,
            label=c,
        )
        g_nz = agg[c]["target_noisy_zoh"]
        axes[2].errorbar(
            [int(d) for d in d_noisy],
            [g_nz[d]["eff_err"]["mean"] for d in d_noisy],
            yerr=[g_nz[d]["eff_err"]["std"] for d in d_noisy],
            marker="^",
            capsize=2,
            label=c,
        )
    axes[0].set_title("A. Target Clean Hold Error [m]")
    axes[1].set_title("B. Target Action HF Power (>10 Hz)")
    axes[2].set_title("C. Target Noisy + 25% ZOH Hold Error [m]")
    for a in axes:
        a.set_xlabel("Action Delay d [10 ms ticks]")
    axes[0].legend(fontsize=7.5)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp11_caps_smoothness.png", dpi=150)


if __name__ == "__main__":
    main()
