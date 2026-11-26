"""
Exp 19 (proposal ablation: "remove individual randomization factors"):
Leave-one-out ablation of the Exp 10 `dr_ppo` recipe on the 6-gate racing task.

Exp 10 found that full dynamics + latency randomization *hurts* transfer (0.75 gates / 0.00 clean on held-out
`ellipse3d` in the target vs 1.00 / 0.40 for nominal PPO). This experiment asks which factor is responsible.

Variants (identical PPO recipe to Exp 10 `dr_ppo`: `base_steps` at lam_da=0.02, then `ft_steps` at lam_da=0.20):
    none          no randomization                                   (= Exp 10 nominal_ppo recipe)
    full          mass + thrust + motor lag + latency d~U{0..4}       (= Exp 10 dr_ppo recipe)
    no_mass       full minus mass_scale
    no_thrust     full minus thrust_scale
    no_motor_lag  full minus tau_rise_scale / tau_fall_scale
    no_latency    full dynamics, d = 0
    latency_only  d~U{0..4}, nominal dynamics
Evaluation: exact-state, fixed start, 2 laps, held-out `ellipse3d` (+ `figure8` train course), on
`source_l2f` (d=0), `target_nominal` (PyBullet, d=2) and `target_perturbed` (PyBullet, kf x 1.20, d=3) — as Exp 10.

Usage:  python exp19_dr_factor_ablation.py [--seeds 3] [--workers 10]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

from common import OUT, fmt, mean_std, np, plt, save_json
from exp10_method_matrix import HIST, TRAIN_DELAYS, SB3PolicyCallable, aggregate_seed_dicts, dyn_sampler
from racing_env import GateRacingEnv, evaluate_racing

DYN_KEYS = {"mass": ["mass_scale"], "thrust": ["thrust_scale"], "motor_lag": ["tau_rise_scale", "tau_fall_scale"]}
VARIANTS = {
    #               dynamics factors kept                  latency randomized
    "none": ([], False),
    "full": (["mass", "thrust", "motor_lag"], True),
    "no_mass": (["thrust", "motor_lag"], True),
    "no_thrust": (["mass", "motor_lag"], True),
    "no_motor_lag": (["mass", "thrust"], True),
    "no_latency": (["mass", "thrust", "motor_lag"], False),
    "latency_only": ([], True),
}
TARGETS = {"source_l2f": ("source", 0, 1.0), "target_nominal": ("target", 2, 1.0), "target_perturbed": ("target", 3, 1.20)}
COURSES = ["figure8", "ellipse3d"]
METRICS = ["gate_frac", "clean_completion", "crash_rate", "track_err_m", "median_lap_s", "smoothness"]


class PartialDynSampler:
    """Exp 10 `dyn_sampler` with the excluded factors pinned to nominal (1.0). Module-level so it pickles."""

    def __init__(self, keep):
        self.keep = {k for f in keep for k in DYN_KEYS[f]}

    def __call__(self, rng):
        s = dyn_sampler(rng)
        return {k: (v if k in self.keep else 1.0) for k, v in s.items()}


def job(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.set_num_threads(1)
    variant, seed, base_steps, ft_steps, n_eval = args
    keep, lat = VARIANTS[variant]
    sampler = PartialDynSampler(keep) if keep else None
    delays = TRAIN_DELAYS if lat else (0,)
    t0 = time.time()
    ppo_kw = dict(n_steps=512, batch_size=512, n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.0, seed=seed, verbose=0,
                  policy_kwargs=dict(net_arch=[128, 128], log_std_init=-0.7))

    def venv(lam, s0):
        return DummyVecEnv([lambda i=i: GateRacingEnv("figure8", "source", HIST, lam_da=lam, delays=delays,
                                                      dyn_sampler=sampler, max_ep_s=5.5, seed=s0 + i)
                            for i in range(8)])

    model = PPO("MlpPolicy", venv(0.02, 2000 * seed), **ppo_kw)
    model.learn(base_steps)
    path = f"{OUT}/exp19_{variant}_s{seed}.zip"
    model.save(path)
    model = PPO.load(path, env=venv(0.20, 2100 * seed), device="cpu")
    model.learn(ft_steps, reset_num_timesteps=False)
    model.save(path)

    pol = SB3PolicyCallable(model)
    ev = {c: {t: evaluate_racing(pol, GateRacingEnv(c, dom, HIST, delays=(d,), kf_scale=kf, random_start_gate=False,
                                                    max_ep_s=12.5, seed=800 + seed), n_eval=n_eval)
              for t, (dom, d, kf) in TARGETS.items()} for c in COURSES}
    return variant, seed, ev, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--base-steps", type=int, default=450_000)
    ap.add_argument("--ft-steps", type=int, default=100_000)
    ap.add_argument("--n-eval", type=int, default=10)
    args = ap.parse_args()

    jobs = [(v, s, args.base_steps, args.ft_steps, args.n_eval) for s in range(args.seeds) for v in VARIANTS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        outs = pool.map(job, jobs)
    print(f"Exp 19: {len(jobs)} runs in {time.time() - t0:.0f}s")

    agg = {v: {c: {t: aggregate_seed_dicts([o[2][c][t] for o in outs if o[0] == v]) for t in TARGETS}
               for c in COURSES} for v in VARIANTS}
    save_json("exp19_dr_factor_ablation.json", {
        "seeds": list(range(args.seeds)), "base_steps": args.base_steps, "ft_steps": args.ft_steps,
        "wall_s": time.time() - t0, "agg": agg})

    for c in COURSES:
        print(f"\n[{c}] gate_frac | clean_completion | crash  (mean ± std, {args.seeds} seeds)")
        for v in VARIANTS:
            row = " || ".join(f"{t}: {fmt(agg[v][c][t]['gate_frac'])} | {fmt(agg[v][c][t]['clean_completion'])} | "
                              f"{fmt(agg[v][c][t]['crash_rate'])}" for t in TARGETS)
            print(f"  {v:13s} {row}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    x = np.arange(len(VARIANTS))
    for j, t in enumerate(TARGETS):
        for k, m in enumerate(["gate_frac", "clean_completion"]):
            ax[k].bar(x + (j - 1) * 0.27, [agg[v]["ellipse3d"][t][m]["mean"] for v in VARIANTS], 0.27,
                      yerr=[agg[v]["ellipse3d"][t][m]["std"] for v in VARIANTS], capsize=2, label=t)
    for k, m in enumerate(["gate fraction", "collision-free completion"]):
        ax[k].set_xticks(x, list(VARIANTS), rotation=30, ha="right")
        ax[k].set_title(f"held-out ellipse3d: {m}")
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp19_dr_factor_ablation.png", dpi=140)


if __name__ == "__main__":
    main()
