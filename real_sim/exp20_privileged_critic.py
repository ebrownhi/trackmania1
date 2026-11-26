"""
Exp 20 (proposal ablation: "remove privileged critic inputs"):
Sensor-limited PPO on the 6-gate racing task with an asymmetric (privileged) critic vs a symmetric critic.

Both arms use the identical deployable actor: sensor-limited obs (v_B hidden, obs_delay=2, 20% ZOH dropout,
pos/gyro noise) + the previous 3 sensor frames (so velocity is inferable by differencing) + 8-action history.
    asym_critic   critic additionally sees the exact-state base obs (true v_B, true gate vectors, true w_B)
    sym_critic    critic sees only what the actor sees (privileged block zeroed)   <- the ablation
PPO recipe identical to Exp 10 (`base_steps` at lam_da=0.02 + `ft_steps` at lam_da=0.20, 8 envs, same kwargs).
Evaluation identical to the Exp 10 sensor-limited cells: fixed start, 2 laps, figure8 + held-out ellipse3d on
`source_l2f` (obs_d=0, p_drop=0), `target_nominal` (d=2, obs_d=2, p_drop=0.2, PnP bias 1.0) and
`target_perturbed` (kf x 1.20, d=3, obs_d=2, p_drop=0.2, PnP bias 1.0).

Usage:  python exp20_privileged_critic.py [--seeds 3] [--workers 6]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

from common import OUT, fmt, np, plt, save_json
from exp10_method_matrix import HIST, SB3PolicyCallable, aggregate_seed_dicts
from racing_env import evaluate_racing

ARMS = {"asym_critic": True, "sym_critic": False}
TARGETS = {  # domain, action delay, kf_scale, obs_delay, p_drop, pnp bias
    "source_l2f": ("source", 0, 1.0, 0, 0.0, 0.0),
    "target_nominal": ("target", 2, 1.0, 2, 0.2, 1.0),
    "target_perturbed": ("target", 3, 1.20, 2, 0.2, 1.0),
}
COURSES = ["figure8", "ellipse3d"]


def job(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from asym_policy import AsymActorCriticPolicy, PrivStackRacingEnv

    torch.set_num_threads(1)
    arm, seed, base_steps, ft_steps, n_eval = args
    t0 = time.time()
    ppo_kw = dict(n_steps=512, batch_size=512, n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.0, seed=seed, verbose=0,
                  policy_kwargs=dict(net_arch=[128, 128], log_std_init=-0.7, critic_privileged=ARMS[arm]))

    def venv(lam, s0):
        return DummyVecEnv([lambda i=i: PrivStackRacingEnv("figure8", "source", HIST, lam_da=lam, delays=(0,),
                                                           obs_delay=2, p_drop=0.2, max_ep_s=5.5, seed=s0 + i)
                            for i in range(8)])

    model = PPO(AsymActorCriticPolicy, venv(0.02, 3000 * seed), **ppo_kw)
    model.learn(base_steps)
    path = f"{OUT}/exp20_{arm}_s{seed}.zip"
    model.save(path)
    model = PPO.load(path, env=venv(0.20, 3100 * seed), device="cpu")
    model.learn(ft_steps, reset_num_timesteps=False)
    model.save(path)

    pol = SB3PolicyCallable(model)
    ev = {c: {t: evaluate_racing(pol, PrivStackRacingEnv(c, dom, HIST, delays=(d,), kf_scale=kf, obs_delay=od,
                                                         p_drop=pd, pnp_bias_scale=pnp, random_start_gate=False,
                                                         max_ep_s=12.5, seed=800 + seed), n_eval=n_eval)
              for t, (dom, d, kf, od, pd, pnp) in TARGETS.items()} for c in COURSES}
    return arm, seed, ev, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--base-steps", type=int, default=450_000)
    ap.add_argument("--ft-steps", type=int, default=100_000)
    ap.add_argument("--n-eval", type=int, default=10)
    args = ap.parse_args()

    jobs = [(a, s, args.base_steps, args.ft_steps, args.n_eval) for s in range(args.seeds) for a in ARMS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        outs = pool.map(job, jobs)
    print(f"Exp 20: {len(jobs)} runs in {time.time() - t0:.0f}s")

    agg = {a: {c: {t: aggregate_seed_dicts([o[2][c][t] for o in outs if o[0] == a]) for t in TARGETS}
               for c in COURSES} for a in ARMS}
    save_json("exp20_privileged_critic.json", {
        "seeds": list(range(args.seeds)), "base_steps": args.base_steps, "ft_steps": args.ft_steps,
        "wall_s": time.time() - t0, "agg": agg,
        "per_seed": {f"{o[0]}_s{o[1]}": {c: {t: o[2][c][t]["gate_frac"] for t in TARGETS} for c in COURSES}
                     for o in outs}})

    for c in COURSES:
        print(f"\n[{c}] gate_frac | clean_completion | crash  (mean ± std, {args.seeds} seeds)")
        for a in ARMS:
            row = " || ".join(f"{t}: {fmt(agg[a][c][t]['gate_frac'])} | {fmt(agg[a][c][t]['clean_completion'])} | "
                              f"{fmt(agg[a][c][t]['crash_rate'])}" for t in TARGETS)
            print(f"  {a:12s} {row}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    x = np.arange(len(TARGETS))
    for k, c in enumerate(COURSES):
        for j, a in enumerate(ARMS):
            ax[k].bar(x + (j - 0.5) * 0.38, [agg[a][c][t]["gate_frac"]["mean"] for t in TARGETS], 0.38,
                      yerr=[agg[a][c][t]["gate_frac"]["std"] for t in TARGETS], capsize=2, label=a)
        ax[k].set_xticks(x, list(TARGETS))
        ax[k].set_title(f"sensor-limited PPO, {c}: gate fraction")
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp20_privileged_critic.png", dpi=140)


if __name__ == "__main__":
    main()
