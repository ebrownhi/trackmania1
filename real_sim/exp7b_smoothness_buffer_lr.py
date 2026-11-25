"""
Exp 7b (H1' retest): Δa smoothness penalty *combined with* action buffer + latency randomization.

Exp 7 trained nominal PPO (a_{t-1} only, no delay) and the penalty alone gave no delay robustness. Here all
policies are trained with latency randomization d ~ U{0..4}; cells:
    buf8_lam0     a_{t-8:t-1} buffer, lambda_da = 0
    buf8_lam0.3   a_{t-8:t-1} buffer, lambda_da = 0.3
    prev_lam0.3   a_{t-1} only,       lambda_da = 0.3     (Theorem 2.2 ablation)
3 seeds x 1.5M steps. Zero-shot eval in L2F source and PyBullet target at d in {0..6}.
H1' predicts: lambda > 0 lowers chattering (mean||Δa||, HF power) and crash rate under delay at equal d=0 skill.

Usage: python exp7b_smoothness_buffer_lr.py [total_steps]
"""

import multiprocessing as mp
import sys
import time

from common import OUT, np, plt, save_json, mean_std, fmt

CELLS = {"buf8_lam0": (8, 0.0), "buf8_lam0.3": (8, 0.3), "prev_lam0.3": (1, 0.3)}
SEEDS = [0, 1, 2]
TRAIN_DELAYS = (0, 1, 2, 3, 4)
EVAL_DELAYS = [0, 1, 2, 3, 4, 6]
TOTAL_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1_500_000
N_ENVS = 8


def train_job(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from ppo_lib import evaluate_hover, make_hover_env

    torch.set_num_threads(1)
    cell, seed = args
    hist, lam = CELLS[cell]
    t0 = time.time()
    venv = DummyVecEnv([lambda i=i: make_hover_env(lam, 1000 * seed + i, hist=hist, delays=TRAIN_DELAYS)
                        for i in range(N_ENVS)])
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=512, n_epochs=10, learning_rate=3e-4, gamma=0.99,
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, seed=seed, verbose=0,
                policy_kwargs=dict(net_arch=[128, 128]))
    model.learn(TOTAL_STEPS)
    model.save(f"{OUT}/exp7b_{cell}_s{seed}.zip")
    ev = {dom: {d: evaluate_hover(model, lambda ep_s, d=d, dom=dom: make_hover_env(
        lam, 777, domain=dom, hist=hist, delays=(d,), ep_s=ep_s)) for d in EVAL_DELAYS} for dom in ["source", "target"]}
    return {"cell": cell, "seed": seed, "train_s": time.time() - t0, "eval": ev}


def main():
    jobs = [(c, s) for c in CELLS for s in SEEDS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 6) as pool:
        outs = pool.map(train_job, jobs)
    print(f"Exp 7b: {len(jobs)} PPO runs x {TOTAL_STEPS} steps in {time.time() - t0:.0f}s")

    res = {"total_steps": TOTAL_STEPS, "cells": {k: list(v) for k, v in CELLS.items()}, "train_delays": TRAIN_DELAYS,
           "eval_delays": EVAL_DELAYS, "agg": {}}
    for c in CELLS:
        o = [x for x in outs if x["cell"] == c]
        res["agg"][c] = {dom: {d: {m: mean_std([x["eval"][dom][d][m] for x in o]) for m in o[0]["eval"][dom][d]}
                               for d in EVAL_DELAYS} for dom in ["source", "target"]}
    save_json("exp7b_smoothness_buffer_lr.json", res)

    for dom in ["source", "target"]:
        print(f"\n[{dom}] eff. hold err [m] (crash=1 m) | crash | mean||Δa|| | HF power   (mean ± std, {len(SEEDS)} seeds)")
        for c in CELLS:
            for d in EVAL_DELAYS:
                m = res["agg"][c][dom][d]
                print(f"  {c:12s} d={d}: {fmt(m['eff_err']):>16} | {fmt(m['crash']):>16} | {fmt(m['mean_da']):>16} | "
                      f"{fmt(m['hf_power']):>16}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for c in CELLS:
        for dom, ls in [("source", "-"), ("target", "--")]:
            g = res["agg"][c][dom]
            for a, key in zip(ax, ["crash", "eff_err", "mean_da"]):
                a.errorbar(EVAL_DELAYS, [g[d][key]["mean"] for d in EVAL_DELAYS],
                           yerr=[g[d][key]["std"] for d in EVAL_DELAYS], ls=ls, marker="o", capsize=2,
                           label=f"{c} {dom}")
    for a, t in zip(ax, ["crash rate", "effective hold error [m]", "mean ||Δa||"]):
        a.set_xlabel("action delay d [ticks]")
        a.set_title(t)
    ax[0].legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp7b_smoothness_buffer_lr.png", dpi=140)


if __name__ == "__main__":
    main()
