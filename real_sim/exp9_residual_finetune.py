"""
Exp 9 (H4' + Prop 2.7): Swift-style residual fine-tuning with PPO vs domain randomization, equal source budget.

Base policies: the lambda_da = 0.1 PPO waypoint-hover policies from Exp 7 (3 seeds; obs = body-frame goal
error, v_B, gravity/heading dirs, w, a_{t-1}). Each arm continues training from the same base for FT_STEPS:
    zeroshot     no fine-tuning
    nominal_ft   nominal L2F                                   (controls for extra training)
    dr_ft        L2F with dynamics randomization (mass, thrust, rotor lag)
    residual_ft  L2F + Step-0 SysID (d_hat, tau_hat) + delay-aligned body-frame k-NN dynamics residual
                 + GP observation-bias residual, all fit on N_CAL target episodes flown by the base policy
    oracle_ft    the target itself (upper reference: target-trained PPO)
Targets (all PyBullet CF2X, d = 0):
    nominal      dynamics gap only
    perturbed    larger dynamics gap: motors 15% stronger (physics kf x 1.15)
    biased       + systematic, position-correlated sensor bias b(p) (~6-10 cm, like oblique-view PnP error)
    noisy        + zero-mean i.i.d. position noise only (Prop 2.7 null: GP mean should collapse to ~0)
Metric: true hold error (mean ||goal - p_true|| over the last 2 s of 4 s episodes), crash rate, 20 episodes.
"""

import multiprocessing as mp
import sys
import time

from common import OUT, np, plt, save_json, mean_std, fmt

SEEDS = [0, 1, 2]
ARMS = ["zeroshot", "nominal_ft", "dr_ft", "residual_ft", "oracle_ft"]
TARGETS = ["nominal", "perturbed", "biased", "noisy"]
FT_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
N_CAL = 4            # target calibration episodes x 4 s = 1600 samples (Swift used ~800-1000 at 50 Hz)
LAM = 0.1
TAUS = {"l2f_native": None, "tau20ms": 0.02, "tau30ms": 0.03, "tau50ms": 0.05}


def bias_field(st):
    """Systematic, pose-correlated position bias b(p) of the target sensor [m]."""
    x, y, z = st.p
    return np.array([0.08 * np.sin(2.0 * y + 0.5), 0.08 * np.cos(1.7 * x), 0.06 * np.sin(1.5 * (z - 2.0) + x)])


class IIDNoise:
    def __init__(self, seed=0, std=0.04):
        self.rng, self.std = np.random.default_rng(seed), std

    def __call__(self, st):
        return self.std * self.rng.standard_normal(3)


def target_bias(target, seed=0):
    return {"nominal": None, "perturbed": None, "biased": bias_field, "noisy": IIDNoise(seed)}[target]


def target_kw(target, seed=0):
    return {"domain": "target", "obs_bias_fn": target_bias(target, seed), "kf_scale": 1.15 if target == "perturbed" else 1.0}


def dyn_sampler(rng):
    return {"mass_scale": rng.uniform(0.8, 1.2), "thrust_scale": rng.uniform(0.8, 1.25),
            "tau_rise_scale": rng.uniform(0.3, 1.5), "tau_fall_scale": rng.uniform(0.08, 1.5)}


def calibrate(model, target, seed):
    """Fly the base policy in the target (+ exploration noise), log client-side obs/actions and ground truth.
    Returns k-NN dynamics residual, tau_hat, d_hat, and a GP observation-bias model (or None)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from exp6_knn_residual import KNNResidual, OneStepSource, feats
    from ppo_lib import make_hover_env

    env = make_hover_env(LAM, 5000 + seed, ep_s=4.0, **target_kw(target, seed))
    rng = np.random.default_rng(seed)
    flights, Xgp, Ygp = [], [], []
    for ep in range(N_CAL):
        o, _ = env.reset(seed=20_000 + 100 * seed + ep)
        S_obs, S_true, A = [env.obs_state], [env.drone.true_state()], []
        while True:
            a, _ = model.predict(o, deterministic=True)
            a = np.clip(a + 0.1 * rng.standard_normal(4), -1, 1)
            o, r, term, trunc, info = env.step(a)
            S_obs.append(env.obs_state)
            S_true.append(info["state"])
            A.append(np.asarray(info["a"]))
            if term or trunc:
                break
        flights.append((S_obs, np.array(A)))
        # GP input = ground-truth (mocap) pose, the same quantity the augmented source applies b_hat to. Using the
        # *observed* pose as input leaks the noise into the regressor (x = p + n, y = n) and learns a spurious bias.
        Xgp += [st.p for st in S_true]
        Ygp += [so.p - st.p for so, st in zip(S_obs, S_true)]

    # Step 0 system ID: joint (d', tau) scan with an affine thrust/drag correction (Exp 6)
    best = None
    for tname, tau in TAUS.items():
        fS = OneStepSource(tau)
        for d in range(4):
            rs, Xs = zip(*[fS.residuals(S, A, d) for S, A in flights])
            r, X = np.concatenate(rs), sum(Xs, [])
            Phi = np.column_stack([np.ones(len(X)), [a[0] for _, a in X], [s.R.T @ s.v for s, _ in X]])
            th, *_ = np.linalg.lstsq(Phi, r[:, :3], rcond=None)
            rms = float(np.sqrt(np.mean((r[:, :3] - Phi @ th) ** 2)))
            if best is None or rms < best[0]:
                best = (rms, tname, tau, d, r, X)
    _, tname, tau_hat, d_hat, r, X = best
    knn = KNNResidual().fit(feats(X, "body10"), r)

    Xgp, Ygp = np.array(Xgp), np.array(Ygp)
    idx = rng.choice(len(Xgp), size=min(500, len(Xgp)), replace=False)
    gp = GaussianProcessRegressor(RBF(0.5) + WhiteKernel(1e-3), normalize_y=False, random_state=0)
    gp.fit(Xgp[idx], Ygp[idx])
    held = np.setdiff1d(np.arange(len(Xgp)), idx)  # report on points not used for fitting
    gp_mean_abs = float(np.mean(np.linalg.norm(gp.predict(Xgp[held]), axis=1)))
    return {"knn": knn, "tau_hat": tau_hat, "tau_name": tname, "d_hat": d_hat, "gp": gp, "gp_mean_abs": gp_mean_abs,
            "true_bias_abs": float(np.mean(np.linalg.norm(Ygp, axis=1)))}


def make_residual_fn(knn):
    from exp6_knn_residual import feats
    cache = {}

    def fn(st, a):
        if cache.get("key") is not a:  # one k-NN query per 10 ms tick (action object is fixed within a tick)
            cache["key"], cache["r"] = a, knn.predict(feats([(st, a)], "body10"))[0]
        return cache["r"][:3], cache["r"][3:]
    return fn


def make_gp_bias(gp):
    def fn(st):
        return gp.predict(st.p[None])[0]
    return fn


def job(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from ppo_lib import evaluate_hover, make_hover_env

    torch.set_num_threads(1)
    arm, target, seed = args
    t0 = time.time()
    base = f"{OUT}/exp7_ppo_lam{LAM}_s{seed}.zip"
    info = {}
    if arm == "zeroshot":
        model = PPO.load(base, device="cpu")
    else:
        if arm == "nominal_ft":
            kw = {}
        elif arm == "dr_ft":
            kw = {"dyn_sampler": dyn_sampler}
        elif arm == "residual_ft":
            cal = calibrate(PPO.load(base, device="cpu"), target, seed)
            info = {k: cal[k] for k in ["tau_name", "d_hat", "gp_mean_abs", "true_bias_abs"]}
            kw = {"residual": make_residual_fn(cal["knn"]), "tau_hat": cal["tau_hat"],
                  "obs_bias_model": make_gp_bias(cal["gp"]), "delays": (cal["d_hat"],)}
        else:  # oracle_ft
            kw = target_kw(target, seed)
        venv = DummyVecEnv([lambda i=i: make_hover_env(LAM, 3000 * seed + 17 * i, **kw) for i in range(8)])
        model = PPO.load(base, env=venv, device="cpu")
        model.learn(FT_STEPS, reset_num_timesteps=False)
    ev = evaluate_hover(model, lambda ep_s: make_hover_env(LAM, 999, ep_s=ep_s, **target_kw(target, 999)))
    return arm, target, seed, ev, info, time.time() - t0


def main():
    jobs = [(a, t, s) for t in TARGETS for a in ARMS for s in SEEDS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 5) as pool:
        outs = pool.map(job, jobs)
    print(f"Exp 9: {len(jobs)} jobs ({FT_STEPS} FT steps) in {time.time() - t0:.0f}s")

    res = {"ft_steps": FT_STEPS, "n_cal_samples": N_CAL * 400, "agg": {}, "calibration": {}}
    for t in TARGETS:
        res["agg"][t] = {}
        for a in ARMS:
            o = [x for x in outs if x[0] == a and x[1] == t]
            res["agg"][t][a] = {m: mean_std([x[3][m] for x in o]) for m in o[0][3]}
        res["calibration"][t] = [x[4] for x in outs if x[0] == "residual_ft" and x[1] == t]
    save_json("exp9_residual_finetune.json", res)

    for t in TARGETS:
        print(f"\n[target: {t}]  true hold error [m] (crash = 1 m) | crash rate   (mean ± std, {len(SEEDS)} seeds)")
        for a in ARMS:
            m = res["agg"][t][a]
            print(f"  {a:12s} {fmt(m['eff_err']):>16} | {fmt(m['crash']):>16}")
        for c in res["calibration"][t]:
            print(f"  calib: d_hat={c['d_hat']} tau_hat={c['tau_name']} |GP mean|={c['gp_mean_abs']:.3f} m "
                  f"(true |obs - p| = {c['true_bias_abs']:.3f} m)")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(ARMS))
    for j, t in enumerate(TARGETS):
        ax.bar(x + (j - 1.5) * 0.2, [res["agg"][t][a]["eff_err"]["mean"] for a in ARMS], 0.2,
               yerr=[res["agg"][t][a]["eff_err"]["std"] for a in ARMS], capsize=2, label=f"target: {t}")
    ax.set_xticks(x, ARMS)
    ax.set_ylabel("true hold error in target [m]")
    ax.set_title(f"Residual fine-tuning vs DR (equal {FT_STEPS // 1000}k source steps)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp9_residual_finetune.png", dpi=140)


if __name__ == "__main__":
    main()
