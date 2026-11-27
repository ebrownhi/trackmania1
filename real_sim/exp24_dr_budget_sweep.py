"""
Exp 24 (H2' follow-up to Exp 19): does dynamics randomization *hurt transfer*, or does it only *under-train* at equal budget?

Exp 19 found full dynamics + latency DR below no-DR on the target, but DR also lowered source skill at the fixed 550k
budget. Here each (arm, seed) trains one continuing lam_da=0.02 trunk (same recipe and seeds as Exp 19). At trunk
checkpoints 450k / 900k / 1.8M a lam_da=0.20 fine-tune branch of 22% of the trunk is split off (the Exp 10/19
450k + 100k ratio), giving total source budgets 550k / 1.1M / 2.2M. The 550k point reproduces Exp 19 exactly.

Arms: none, full, latency_only, no_mass (Exp 19 definitions).
Eval as Exp 19: exact-state, fixed start, 2 laps, figure8 + held-out ellipse3d on source_l2f (d=0), target_nominal
(PyBullet, d=2), target_perturbed (kf x 1.20, d=3), 10 episodes. Transfer ratio = target gate_frac / source gate_frac.

Usage:  python exp24_dr_budget_sweep.py [--seeds 3] [--workers 4] [--arms none,full,latency_only,no_mass]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

from common import OUT, fmt, np, plt, save_json
from exp10_method_matrix import HIST, TRAIN_DELAYS, SB3PolicyCallable, aggregate_seed_dicts
from exp19_dr_factor_ablation import COURSES, TARGETS, VARIANTS, PartialDynSampler
from racing_env import GateRacingEnv, evaluate_racing

TRUNK = [450_000, 900_000, 1_800_000]
FT_FRAC = 100_000 / 450_000


def job(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.set_num_threads(1)
    arm, seed, trunk, n_eval = args
    keep, lat = VARIANTS[arm]
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
    out, done = {}, 0
    for tb in trunk:
        model.learn(tb - done, reset_num_timesteps=done == 0)
        done = tb
        trunk_path = f"{OUT}/exp24_{arm}_trunk{tb // 1000}k_s{seed}.zip"
        model.save(trunk_path)
        ft = int(round(FT_FRAC * tb))
        branch = PPO.load(trunk_path, env=venv(0.20, 2100 * seed), device="cpu")
        branch.learn(ft, reset_num_timesteps=False)
        total = tb + ft
        branch.save(f"{OUT}/exp24_{arm}_{total // 1000}k_s{seed}.zip")
        pol = SB3PolicyCallable(branch)
        out[total] = {c: {t: evaluate_racing(pol, GateRacingEnv(c, dom, HIST, delays=(d,), kf_scale=kf,
                                                                random_start_gate=False, max_ep_s=12.5,
                                                                seed=800 + seed), n_eval=n_eval)
                          for t, (dom, d, kf) in TARGETS.items()} for c in COURSES}
        print(f"  [{arm} s{seed}] {total // 1000}k done at {time.time() - t0:.0f}s: ellipse3d tgt_nom gf "
              f"{out[total]['ellipse3d']['target_nominal']['gate_frac']:.2f}", flush=True)
    return arm, seed, out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arms", default="none,full,latency_only,no_mass")
    ap.add_argument("--trunk", default=",".join(str(t) for t in TRUNK))
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    arms = args.arms.split(",")
    trunk = [int(t) for t in args.trunk.split(",")]

    # longest jobs first so the pool's tail is short
    jobs = [(a, s, trunk, args.n_eval) for s in range(args.seeds) for a in arms]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        outs = pool.map(job, jobs, chunksize=1)
    wall = time.time() - t0
    print(f"Exp 24: {len(jobs)} runs in {wall:.0f}s")

    budgets = sorted(outs[0][2])
    agg = {a: {str(b): {c: {t: aggregate_seed_dicts([o[2][b][c][t] for o in outs if o[0] == a]) for t in TARGETS}
                        for c in COURSES} for b in budgets} for a in arms}
    ratio = {a: {str(b): {c: {t: aggregate_seed_dicts([
        {"ratio": o[2][b][c][t]["gate_frac"] / max(o[2][b][c]["source_l2f"]["gate_frac"], 1e-6)}
        for o in outs if o[0] == a]) for t in ["target_nominal", "target_perturbed"]} for c in COURSES}
        for b in budgets} for a in arms}
    per_seed = {f"{o[0]}_s{o[1]}": {str(b): {c: {t: {k: o[2][b][c][t][k] for k in ["gate_frac", "clean_completion",
                                                                                   "crash_rate"]}
                                                 for t in TARGETS} for c in COURSES} for b in budgets} for o in outs}
    save_json(f"exp24_dr_budget_sweep{args.tag}.json", {"seeds": list(range(args.seeds)), "trunk": trunk,
                                                        "budgets": budgets, "wall_s": wall, "agg": agg,
                                                        "ratio": ratio, "per_seed": per_seed})

    for c in COURSES:
        print(f"\n[{c}] source gf | tgt_nom gf / clean | tgt_pert gf / clean | ratio nom  (mean ± std)")
        for a in arms:
            for b in budgets:
                g = agg[a][str(b)][c]
                print(f"  {a:13s} {b // 1000:5d}k  {fmt(g['source_l2f']['gate_frac'])} | "
                      f"{fmt(g['target_nominal']['gate_frac'])} / {g['target_nominal']['clean_completion']['mean']:.2f} | "
                      f"{fmt(g['target_perturbed']['gate_frac'])} / {g['target_perturbed']['clean_completion']['mean']:.2f} | "
                      f"{fmt(ratio[a][str(b)][c]['target_nominal']['ratio'])}")

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for a in arms:
        xs = [b / 1e6 for b in budgets]
        for k, (c, t, ttl) in enumerate([("figure8", "source_l2f", "source skill (figure8, L2F)"),
                                         ("ellipse3d", "target_nominal", "held-out ellipse3d, target_nominal"),
                                         ("ellipse3d", "target_perturbed", "held-out ellipse3d, target_perturbed")]):
            ax[k].errorbar(xs, [agg[a][str(b)][c][t]["gate_frac"]["mean"] for b in budgets],
                           yerr=[agg[a][str(b)][c][t]["gate_frac"]["std"] for b in budgets], marker="o", capsize=2,
                           label=a)
            ax[k].set_title(ttl, fontsize=9)
            ax[k].set_xlabel("total source steps [M]")
    ax[0].set_ylabel("gate fraction")
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp24_dr_budget_sweep{args.tag}.png", dpi=140)


if __name__ == "__main__":
    main()
