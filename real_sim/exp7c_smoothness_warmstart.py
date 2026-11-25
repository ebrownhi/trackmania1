"""
Exp 7c (H1'): smoothness penalty as a fine-tuning stage.

In Exp 7b, PPO with lambda_da = 0.3 from scratch failed to learn on 2/3 seeds (the penalty stalls early
exploration), while the one seed that learned transferred with a flat target error across delays. To compare
at equal skill, warm-start from the three Exp 7b buf8_lam0 policies (buffer + latency randomization) and
continue for FT_STEPS with lambda in {0 (control), 0.3}. Same eval as Exp 7b.
"""

import multiprocessing as mp
import sys
import time

from common import OUT, np, plt, save_json, mean_std, fmt
from exp7b_smoothness_buffer_lr import EVAL_DELAYS, N_ENVS, SEEDS, TRAIN_DELAYS

LAMS = [0.0, 0.3]
FT_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000


def job(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from ppo_lib import evaluate_hover, make_hover_env

    torch.set_num_threads(1)
    lam, seed = args
    t0 = time.time()
    venv = DummyVecEnv([lambda i=i: make_hover_env(lam, 4000 * seed + i, hist=8, delays=TRAIN_DELAYS)
                        for i in range(N_ENVS)])
    model = PPO.load(f"{OUT}/exp7b_buf8_lam0_s{seed}.zip", env=venv, device="cpu")
    model.learn(FT_STEPS, reset_num_timesteps=False)
    model.save(f"{OUT}/exp7c_warm_lam{lam}_s{seed}.zip")
    ev = {dom: {d: evaluate_hover(model, lambda ep_s, d=d, dom=dom: make_hover_env(
        lam, 777, domain=dom, hist=8, delays=(d,), ep_s=ep_s)) for d in EVAL_DELAYS} for dom in ["source", "target"]}
    return lam, seed, ev, time.time() - t0


def main():
    t0 = time.time()
    with mp.get_context("spawn").Pool(6) as pool:
        outs = pool.map(job, [(l, s) for l in LAMS for s in SEEDS])
    print(f"Exp 7c: {len(outs)} warm-start runs x {FT_STEPS} steps in {time.time() - t0:.0f}s")
    res = {"ft_steps": FT_STEPS, "agg": {}}
    for lam in LAMS:
        o = [x for x in outs if x[0] == lam]
        res["agg"][lam] = {dom: {d: {m: mean_std([x[2][dom][d][m] for x in o]) for m in o[0][2][dom][d]}
                                 for d in EVAL_DELAYS} for dom in ["source", "target"]}
    save_json("exp7c_smoothness_warmstart.json", res)
    for dom in ["source", "target"]:
        print(f"\n[{dom}] eff. hold err [m] | crash | mean||Δa|| | HF power  (mean ± std, {len(SEEDS)} seeds)")
        for lam in LAMS:
            for d in EVAL_DELAYS:
                m = res["agg"][lam][dom][d]
                print(f"  λ={lam:<4} d={d}: {fmt(m['eff_err']):>16} | {fmt(m['crash']):>16} | {fmt(m['mean_da']):>16} | "
                      f"{fmt(m['hf_power']):>16}")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for lam in LAMS:
        for dom, ls in [("source", "-"), ("target", "--")]:
            g = res["agg"][lam][dom]
            ax[0].errorbar(EVAL_DELAYS, [g[d]["eff_err"]["mean"] for d in EVAL_DELAYS],
                           yerr=[g[d]["eff_err"]["std"] for d in EVAL_DELAYS], ls=ls, marker="o", capsize=2,
                           label=f"λ={lam} {dom}")
            ax[1].errorbar(EVAL_DELAYS, [g[d]["hf_power"]["mean"] for d in EVAL_DELAYS],
                           yerr=[g[d]["hf_power"]["std"] for d in EVAL_DELAYS], ls=ls, marker="o", capsize=2)
    ax[0].set_title("effective hold error [m]")
    ax[1].set_title("action power above 10 Hz")
    for a in ax:
        a.set_xlabel("action delay d [ticks]")
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp7c_smoothness_warmstart.png", dpi=140)


if __name__ == "__main__":
    main()
