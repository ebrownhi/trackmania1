"""
Exp 22 (proposal ablation "remove privileged critic inputs", fair rerun of Exp 20):
sensor-limited PPO on 6-gate racing with an asymmetric (privileged) vs symmetric critic, where the actor can infer
velocity track-free.

Exp 20 was uninformative: both arms failed to learn because the actor only saw the shortcut-prone track-relative
observation under the hard sensor. Here, per arm (identical except the critic input):
    actor   sensor-limited racing obs + last 8 Exp 21 kinematic frames + 8-action history  (KinStackRacingEnv)
    critic  asym: + privileged exact-state block | sym: privileged block masked
Init (same for both arms of a seed):
    scratch   random init (Exp 10 PPO recipe: 450k at lam=0.02 + 100k at lam=0.20)
    bc        actor warm-started by DAgger behaviour cloning of the exact-state Exp 10 teacher (labels = teacher on
              the true state), then the same PPO schedule. The critic is freshly initialized in both arms, so the
              arms differ only in what the critic sees while it learns.
Sensor level: mid (obs_delay 1, 10% dropout, 2 cm / 0.04 rad/s noise) by default. Eval as Exp 21: fixed start, 2 laps,
figure8 + held-out ellipse3d on source_l2f (d=0), target_nominal (d=2), target_perturbed (kf x 1.20, d=3), with the
Exp 10 oblique PnP gate bias in the targets.

Usage:  python exp22_privileged_critic_trackfree.py [--init scratch|bc] [--level mid] [--seeds 3] [--workers 5]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

from common import OUT, fmt, np, plt, save_json
from exp10_method_matrix import HIST, SB3PolicyCallable, aggregate_seed_dicts

ARMS = {"asym_critic": True, "sym_critic": False}
LEVELS = {"clean": (0, 0.0), "mid": (1, 0.1), "hard": (2, 0.2)}
TARGETS = {"source_l2f": ("source", 0, 1.0, 0.0), "target_nominal": ("target", 2, 1.0, 1.0),
           "target_perturbed": ("target", 3, 1.20, 1.0)}
COURSES = ["figure8", "ellipse3d"]
N_ACT = 4 * HIST


def _teacher_obs(o, n_sensor):
    from asym_policy import N_PRIV
    return np.concatenate([o[-N_PRIV:], o[n_sensor - N_ACT:n_sensor]]).astype(np.float32)


def bc_warmstart(model, teacher, level, seed, iters=8, eps=12, epochs=40):
    """DAgger-clone the exact-state teacher into the PPO actor mean (masked features), in place."""
    import torch
    import torch.nn as nn
    from trackfree_env import KinStackRacingEnv

    od, pd = LEVELS[level]
    rng = np.random.default_rng(seed + 500)
    env = KinStackRacingEnv("figure8", "source", HIST, obs_delay=od, p_drop=pd, max_ep_s=6.0, seed=7000 + seed)
    pol = model.policy
    params = list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    def actor_mean(x):
        return pol.action_net(pol.mlp_extractor.policy_net(pol.pi_features_extractor(x)))

    X, Y = [], []
    for it in range(iters):
        beta = 1.0 if it == 0 else max(0.0, 0.6 - 0.1 * it)
        for ep in range(eps):
            o, _ = env.reset(seed=50_000 + 1000 * seed + 100 * it + ep)
            while True:
                a_t = teacher.predict(_teacher_obs(o, env.n_sensor), deterministic=True)[0]
                X.append(o.copy())
                Y.append(a_t)
                if it == 0 or rng.random() < beta:
                    a = a_t
                else:
                    with torch.no_grad():
                        a = np.clip(actor_mean(torch.as_tensor(o)[None])[0].numpy(), -1, 1)
                o, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
        xt, yt = torch.as_tensor(np.array(X)), torch.as_tensor(np.array(Y))
        for _ in range(epochs):
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), 512):
                sl = perm[i:i + 512]
                loss = ((actor_mean(xt[sl]) - yt[sl]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 2.0)
                opt.step()
    return float(loss)


def job(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from asym_policy import AsymActorCriticPolicy
    from racing_env import evaluate_racing
    from trackfree_env import KinStackRacingEnv

    torch.set_num_threads(1)
    arm, seed, init, level, base_steps, ft_steps, n_eval = args
    od, pd = LEVELS[level]
    t0 = time.time()
    ppo_kw = dict(n_steps=512, batch_size=512, n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.0, seed=seed, verbose=0,
                  policy_kwargs=dict(net_arch=[128, 128], log_std_init=-0.7, critic_privileged=ARMS[arm]))

    def venv(lam, s0):
        return DummyVecEnv([lambda i=i: KinStackRacingEnv("figure8", "source", HIST, lam_da=lam, delays=(0,),
                                                          obs_delay=od, p_drop=pd, max_ep_s=5.5, seed=s0 + i)
                            for i in range(8)])

    model = PPO(AsymActorCriticPolicy, venv(0.02, 3000 * seed), **ppo_kw)
    info = {}
    if init == "bc":
        # Identical actor init for both arms: BC weights are cached per (seed, level) and loaded by every arm.
        import os
        path_bc = f"{OUT}/exp22_bc_actor_{level}_s{seed}.pt"
        keys = ("mlp_extractor.policy_net", "action_net")
        if not os.path.exists(path_bc):
            teacher = PPO.load(f"{OUT}/exp10_base_nom_s{seed}.zip", device="cpu")
            info["bc_loss"] = bc_warmstart(model, teacher, level, seed)
            sd = {k: v for k, v in model.policy.state_dict().items() if k.startswith(keys)}
            torch.save(sd, path_bc + f".tmp{os.getpid()}")
            os.replace(path_bc + f".tmp{os.getpid()}", path_bc)
        model.policy.load_state_dict(torch.load(path_bc), strict=False)
        info["bc_eval"] = {c: evaluate_racing(SB3PolicyCallable(model), KinStackRacingEnv(
            c, "source", HIST, obs_delay=od, p_drop=pd, random_start_gate=False, max_ep_s=12.5, seed=800 + seed),
            n_eval=3)["gate_frac"] for c in COURSES}
    model.learn(base_steps)
    path = f"{OUT}/exp22_{init}_{level}_{arm}_s{seed}.zip"
    model.save(path)
    model = PPO.load(path, env=venv(0.20, 3100 * seed), device="cpu")
    model.learn(ft_steps, reset_num_timesteps=False)
    model.save(path)

    pol = SB3PolicyCallable(model)
    ev = {c: {t: evaluate_racing(pol, KinStackRacingEnv(c, dom, HIST, delays=(d,), kf_scale=kf, obs_delay=od, p_drop=pd,
                                                        pnp_bias_scale=pnp, random_start_gate=False, max_ep_s=12.5,
                                                        seed=800 + seed), n_eval=n_eval)
              for t, (dom, d, kf, pnp) in TARGETS.items()} for c in COURSES}
    return arm, seed, ev, info, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["scratch", "bc"], default="bc")
    ap.add_argument("--level", choices=list(LEVELS), default="mid")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--base-steps", type=int, default=450_000)
    ap.add_argument("--ft-steps", type=int, default=100_000)
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    seeds = list(range(args.seed0, args.seed0 + args.seeds))

    if args.init == "bc":  # build the shared BC init once per seed before the arms start (no race between arms)
        with mp.get_context("spawn").Pool(min(args.workers, len(seeds))) as pool:
            pool.map(job, [("asym_critic", s, "bc", args.level, 0, 0, 1) for s in seeds])
    jobs = [(a, s, args.init, args.level, args.base_steps, args.ft_steps, args.n_eval) for s in seeds for a in ARMS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        outs = pool.map(job, jobs, chunksize=1)
    print(f"Exp 22 ({args.init}, {args.level}): {len(jobs)} runs in {time.time() - t0:.0f}s")

    agg = {a: {c: {t: aggregate_seed_dicts([o[2][c][t] for o in outs if o[0] == a]) for t in TARGETS}
               for c in COURSES} for a in ARMS}
    name = f"exp22_privileged_critic_trackfree_{args.init}_{args.level}{args.tag}"
    save_json(f"{name}.json", {
        "init": args.init, "level": args.level, "seeds": seeds, "base_steps": args.base_steps,
        "ft_steps": args.ft_steps, "wall_s": time.time() - t0, "agg": agg,
        "bc_info": {f"{o[0]}_s{o[1]}": o[3] for o in outs},
        "per_seed": {f"{o[0]}_s{o[1]}": {c: {t: [o[2][c][t]["gate_frac"], o[2][c][t]["crash_rate"]] for t in TARGETS}
                                         for c in COURSES} for o in outs}})

    for o in outs:
        if o[3]:
            print(f"  BC init {o[0]} s{o[1]}: {o[3]}")
    for c in COURSES:
        print(f"\n[{c}] gate_frac | crash  (mean ± std, {len(seeds)} seeds)")
        for a in ARMS:
            row = " || ".join(f"{t}: {fmt(agg[a][c][t]['gate_frac'])} | {fmt(agg[a][c][t]['crash_rate'])}" for t in TARGETS)
            print(f"  {a:12s} {row}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    x = np.arange(len(TARGETS))
    for k, c in enumerate(COURSES):
        for j, a in enumerate(ARMS):
            ax[k].bar(x + (j - 0.5) * 0.38, [agg[a][c][t]["gate_frac"]["mean"] for t in TARGETS], 0.38,
                      yerr=[agg[a][c][t]["gate_frac"]["std"] for t in TARGETS], capsize=2, label=a)
        ax[k].set_xticks(x, list(TARGETS))
        ax[k].set_title(f"{args.init}/{args.level}, {c}: gate fraction")
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", dpi=140)


if __name__ == "__main__":
    main()
