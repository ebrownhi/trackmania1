"""
Exp 27 (H2', idea 3 of docs/FOLLOWUP_RIFFS_ROUND2.md): is the dynamics-randomization skill tax avoidable?

Exp 24: full DR (mass + thrust + motor lag + latency) costs source skill at 550k and only matches no-DR at >= 1.1M,
with ~33% slower laps. SimpleFlight (arXiv 2412.11764) / Gräfe et al. (arXiv 2603.05944): identify mass, randomize
only the poorly known thrust coefficient, and take mass compensation out of the network.

Arms (equal 550k budget, Exp 10/19 recipe: 450k at lam=0.02 + 100k at lam=0.20, 3 seeds):
    none, full          reused from Exp 19 (exp19_dr_factor_ablation.json; Exp 24 reproduced them exactly)
    hn_none             hover-normalized thrust action (hover_norm.py), no DR
    hn_thrust_lat       hover-normalized + thrust-coefficient-only DR + latency d~U{0..4}
    hn_full             hover-normalized + full Exp 19 DR (mass is known to the normalization; thrust, motor lag unknown)
    full_bigbatch       Exp 19 `full`, PPO rollout and minibatch 4x larger (n_steps 2048, batch 2048) at equal steps
Eval as Exp 19 / 24: figure8 + held-out ellipse3d x source_l2f (d=0) / target_nominal (d=2) /
target_perturbed (kf x 1.20, d=3), fixed start, 10 episodes.

Usage:  python exp27_hover_norm_dr.py [--seeds 3] [--workers 3]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

from common import OUT, fmt, np, plt, save_json
from exp10_method_matrix import HIST, TRAIN_DELAYS, SB3PolicyCallable, aggregate_seed_dicts
from exp19_dr_factor_ablation import COURSES, TARGETS, PartialDynSampler

ARMS = {  # name: (hover-normalized?, dynamics factors randomized, latency DR?, batch multiplier)
    "hn_none": (True, [], False, 1),
    "hn_thrust_lat": (True, ["thrust"], True, 1),
    "hn_full": (True, ["mass", "thrust", "motor_lag"], True, 1),
    "full_bigbatch": (False, ["mass", "thrust", "motor_lag"], True, 4),
}
REUSED = {"none": "none", "full": "full"}  # arm -> Exp 19 variant
ORDER = ["none", "full", "hn_none", "hn_thrust_lat", "hn_full", "full_bigbatch"]


def job(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from hover_norm import HoverNormRacingEnv
    from racing_env import GateRacingEnv, evaluate_racing

    torch.set_num_threads(1)
    arm, seed, base_steps, ft_steps, n_eval = args
    cache = f"{OUT}/exp27_job_{arm}_s{seed}.json"  # per-job result, so an interrupted sweep resumes
    if os.path.exists(cache):
        d = json.load(open(cache))
        return arm, seed, d["ev"], d["wall_s"]
    hn, keep, lat, bmul = ARMS[arm]
    Env = HoverNormRacingEnv if hn else GateRacingEnv
    sampler = PartialDynSampler(keep) if keep else None
    delays = TRAIN_DELAYS if lat else (0,)
    t0 = time.time()
    ppo_kw = dict(n_steps=512 * bmul, batch_size=512 * bmul, n_epochs=10, learning_rate=3e-4, gamma=0.99,
                  gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, seed=seed, verbose=0,
                  policy_kwargs=dict(net_arch=[128, 128], log_std_init=-0.7))

    def venv(lam, s0):
        return DummyVecEnv([lambda i=i: Env("figure8", "source", HIST, lam_da=lam, delays=delays, dyn_sampler=sampler,
                                            max_ep_s=5.5, seed=s0 + i) for i in range(8)])

    model = PPO("MlpPolicy", venv(0.02, 2000 * seed), **ppo_kw)  # same env seeds as Exp 19
    model.learn(base_steps)
    path = f"{OUT}/exp27_{arm}_s{seed}.zip"
    model.save(path)
    model = PPO.load(path, env=venv(0.20, 2100 * seed), device="cpu")
    model.learn(ft_steps, reset_num_timesteps=False)
    model.save(path)

    pol = SB3PolicyCallable(model)
    ev = {c: {t: evaluate_racing(pol, Env(c, dom, HIST, delays=(d,), kf_scale=kf, random_start_gate=False,
                                          max_ep_s=12.5, seed=800 + seed), n_eval=n_eval)
              for t, (dom, d, kf) in TARGETS.items()} for c in COURSES}
    with open(cache, "w") as f:
        json.dump({"ev": ev, "wall_s": time.time() - t0}, f)
    return arm, seed, ev, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--base-steps", type=int, default=450_000)
    ap.add_argument("--ft-steps", type=int, default=100_000)
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    arms = args.arms.split(",")

    jobs = [(a, s, args.base_steps, args.ft_steps, args.n_eval) for s in range(args.seeds) for a in arms]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        outs = pool.map(job, jobs, chunksize=1)
    print(f"Exp 27: {len(jobs)} runs in {time.time() - t0:.0f}s")

    agg = {a: {c: {t: aggregate_seed_dicts([o[2][c][t] for o in outs if o[0] == a]) for t in TARGETS}
               for c in COURSES} for a in arms}
    e19 = json.load(open(f"{OUT}/exp19_dr_factor_ablation.json"))["agg"]
    for a, v in REUSED.items():
        agg[a] = e19[v]
    shown = [a for a in ORDER if a in agg]
    save_json("exp27_hover_norm_dr.json", {"seeds": list(range(args.seeds)), "base_steps": args.base_steps,
                                           "ft_steps": args.ft_steps, "wall_s": time.time() - t0,
                                           "reused_from_exp19": REUSED, "agg": agg})

    for c in COURSES:
        print(f"\n[{c}] gate_frac | clean | crash | median lap  (mean ± std, {args.seeds} seeds)")
        for a in shown:
            row = " || ".join(f"{t}: {fmt(agg[a][c][t]['gate_frac'])} | {agg[a][c][t]['clean_completion']['mean']:.2f} | "
                              f"{agg[a][c][t]['crash_rate']['mean']:.2f} | {agg[a][c][t]['median_lap_s']['mean']:.2f}s"
                              for t in TARGETS)
            print(f"  {a:14s} {row}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(shown))
    for j, t in enumerate(TARGETS):
        ax[0].bar(x + (j - 1) * 0.27, [agg[a]["ellipse3d"][t]["gate_frac"]["mean"] for a in shown], 0.27,
                  yerr=[agg[a]["ellipse3d"][t]["gate_frac"]["std"] for a in shown], capsize=2, label=t)
        ax[1].bar(x + (j - 1) * 0.27, [agg[a]["ellipse3d"][t]["median_lap_s"]["mean"] for a in shown], 0.27, label=t)
    for k, m in enumerate(["gate fraction", "median lap [s]"]):
        ax[k].set_xticks(x, shown, rotation=30, ha="right")
        ax[k].set_title(f"held-out ellipse3d: {m}")
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp27_hover_norm_dr.png", dpi=140)


if __name__ == "__main__":
    main()
