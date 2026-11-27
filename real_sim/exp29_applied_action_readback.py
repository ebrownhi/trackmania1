"""
Exp 29 (follow-up riff 5, GT Sophy): observe the APPLIED action instead of (or beside) the commanded one.

Part B — phantom residual (Thm 2.5 / Exp 6). The expert (+ noise) flies 6 x 5 s calibration flights on figure8 in
four delayed targets; the client logs states, commanded actions a_t and the actuator's applied-action readback:
    l2f_d3        L2F itself + fixed delay 3                (Delta f == 0: the true residual is exactly 0)
    l2f_jit14     L2F itself + per-command delay ~ U{1..4}  (Delta f == 0, jittered delay)
    pyb_d3        PyBullet CF2X + fixed delay 3             (real dynamics gap)
    pyb_jit14     PyBullet CF2X + jittered delay U{1..4}
Residual models (k-NN, body10 features), each with its one-step source model's rotor lag chosen by the Exp 9
affine-corrected scan over TAUS:
    naive         commanded actions, assumed delay 0
    scan          commanded actions, delay d_hat from the joint (d', tau) scan (Exp 6 / Exp 9 Step 0)
    applied       applied-action readback, delay 0 (no alignment needed)
Metrics on held-out flights (4 on figure8, 4 on held-out ellipse3d): mean predicted |r_lin| (phantom magnitude; only
meaningful for the Delta f == 0 targets, where the truth is 0) and Exp 8-style open-loop replay error: the augmented
L2F (residual + the method's delay) replays the method's action sequence from logged states; position error at 1 s.

Part A — racing PPO with commanded ("cmd"), applied ("app") or both ("both") action histories in the obs, trained
with a fixed delay d = 1 or with latency randomization d ~ U{0..2}; evaluated at mismatched delays d in
{0, 1, 2, 3, 4, 6} and under jitter U{1..4}. Exp 10 PPO recipe (450k at lam=0.02 + 100k at lam=0.20), 3 seeds.

Usage:  python exp29_applied_action_readback.py B
        python exp29_applied_action_readback.py A [--seeds 3] [--workers 3]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import multiprocessing as mp
import time

from common import OUT, fmt, mean_std, np, plt, save_json

TAUS = {"l2f_native": None, "tau20ms": 0.02, "tau30ms": 0.03, "tau50ms": 0.05}
D_SCAN = list(range(7))
B_TARGETS = ["l2f_d3", "l2f_jit14", "pyb_d3", "pyb_jit14"]
METHODS = ["naive", "scan", "applied"]


# ---------------------------------------------------------------- Part B -------------------------------------------
def make_b_target(name, seed):
    from applied_readback import JitterDrone
    from sims.drone_pair import InterfaceConfig, make_source, make_target

    with contextlib.redirect_stdout(io.StringIO()):
        maker = make_source if name.startswith("l2f") else make_target
        if name.endswith("d3"):
            return maker(seed=seed, iface=InterfaceConfig(action_delay=3))
        return JitterDrone(maker(seed=seed), (1, 4), seed=seed)


def fly_log(name, course, seed, n_flights, dur_s=5.0, noise=0.15):
    from sims.drone_pair import DroneState, GeometricExpert, RL_DT

    rng = np.random.default_rng(seed)
    ex = GeometricExpert()
    out = []
    for f in range(n_flights):
        dr = make_b_target(name, seed + f)
        t0 = float(rng.uniform(0, course.period))
        p, v, _ = course.ref(t0)
        obs = dr.reset(DroneState(p, np.eye(3), v, np.zeros(3)))
        S, A, AP = [obs], [], []
        for k in range(int(dur_s / RL_DT)):
            a = np.clip(ex(obs, *course.ref(t0 + k * RL_DT)) + noise * rng.standard_normal(4), -1, 1)
            obs = dr.step(a)
            S.append(obs)
            A.append(a)
            AP.append(np.asarray(dr.applied).copy())
        out.append((S, np.array(A), np.array(AP)))
    return out


def fit_method(method, flights):
    from exp6_knn_residual import KNNResidual, OneStepSource, feats

    ds = D_SCAN if method == "scan" else [0]
    best = None
    for tname, tau in TAUS.items():
        fS = OneStepSource(tau)
        for d in ds:
            rs, Xs = zip(*[fS.residuals(S, AP if method == "applied" else A, d) for S, A, AP in flights])
            r, X = np.concatenate(rs), sum(Xs, [])
            Phi = np.column_stack([np.ones(len(X)), [a[0] for _, a in X], [s.R.T @ s.v for s, _ in X]])
            th, *_ = np.linalg.lstsq(Phi, r[:, :3], rcond=None)
            rms = float(np.sqrt(np.mean((r[:, :3] - Phi @ th) ** 2)))
            if best is None or rms < best[0]:
                best = (rms, tname, tau, d, r, X)
    _, tname, tau, d, r, X = best
    return {"knn": KNNResidual().fit(feats(X, "body10"), r), "tau": tau, "tau_name": tname, "d": d}


def rotor_at(fS, S, seq, d, upto):
    """L2F rotor speeds at tick `upto`, reconstructed by running the source's own asymmetric rotor filter over the
    method's d-aligned action history (the same reconstruction OneStepSource.residuals uses)."""
    from sims.drone_pair import SUB_DT, SUBSTEPS, CTBRConfig

    ctbr = CTBRConfig()
    rpm = fS.be.force_to_rpm(np.full(4, fS.be.base_params["dynamics"]["mass"] * 9.81 / 4))
    for k in range(upto):
        a = seq[k - d] if k - d >= 0 else fS.hover_a
        sp = fS.be.force_to_rpm(ctbr.rotor_forces(a, S[k].w))
        for _ in range(SUBSTEPS):
            al = np.exp(-SUB_DT / np.where(sp >= rpm, fS.tau_r, fS.tau_f))
            rpm = al * rpm + (1 - al) * sp
    return rpm


def eval_method(method, fit, flights, windows=(20, 120, 220, 320), horizon=100):
    from exp6_knn_residual import OneStepSource, feats
    from exp9_residual_finetune import make_residual_fn
    from sims.drone_pair import InterfaceConfig, make_source

    fS = OneStepSource(fit["tau"])
    phantom = []
    for S, A, AP in flights:
        _, X = fS.residuals(S, AP if method == "applied" else A, fit["d"])
        phantom.append(np.linalg.norm(fit["knn"].predict(feats(X, "body10"))[:, :3], axis=1))
    ov = {}
    if fit["tau"] is not None:
        ov = {"tau_rise_scale": fit["tau"] / 0.055455, "tau_fall_scale": fit["tau"] / 0.249394}
    err, err_nores = [], []
    for S, A, AP in flights:
        seq = AP if method == "applied" else A
        for w0 in windows:
            if w0 + horizon >= len(seq):
                continue
            for use_res, sink in [(True, err), (False, err_nores)]:
                src = make_source(seed=0, overrides=ov, iface=InterfaceConfig(action_delay=fit["d"]))
                src.backend.residual_fn = make_residual_fn(fit["knn"]) if use_res else None
                src.reset(S[w0])
                src.backend.reset(S[w0], rpm=rotor_at(fS, S, seq, fit["d"], w0))  # hidden rotor state, not hover
                # pre-load the delay FIFO with the commands already in flight at w0
                for j in range(fit["d"]):
                    k = w0 - fit["d"] + j
                    src.act_fifo.append(np.asarray(seq[k] if k >= 0 else src.hover_action, dtype=np.float64))
                for k in range(w0, w0 + horizon):
                    src.step(seq[k])
                sink.append(float(np.linalg.norm(src.true_state().p - S[w0 + horizon].p)))
    return {"phantom_lin": float(np.mean(np.concatenate(phantom))), "replay_err_1s": float(np.mean(err)),
            "replay_err_1s_nores": float(np.mean(err_nores))}


def b_job(args):
    from sims.drone_pair import Course

    target, seed = args
    t0 = time.time()
    cal = fly_log(target, Course("figure8"), 1000 * seed, 6)
    test = (fly_log(target, Course("figure8"), 1000 * seed + 500, 4)
            + fly_log(target, Course("ellipse3d"), 1000 * seed + 700, 4))
    out = {}
    for m in METHODS:
        fit = fit_method(m, cal)
        out[m] = {"d": fit["d"], "tau_name": fit["tau_name"], **eval_method(m, fit, test)}
    return target, seed, out, time.time() - t0


def part_b(seeds, workers):
    jobs = [(t, s) for t in B_TARGETS for s in range(seeds)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        outs = pool.map(b_job, jobs, chunksize=1)
    print(f"Exp 29 B: {len(jobs)} jobs in {time.time() - t0:.0f}s")
    agg = {t: {m: {k: mean_std([o[2][m][k] for o in outs if o[0] == t])
                   for k in ["phantom_lin", "replay_err_1s", "replay_err_1s_nores"]} for m in METHODS} for t in B_TARGETS}
    picks = {t: {m: [(o[2][m]["d"], o[2][m]["tau_name"]) for o in outs if o[0] == t] for m in METHODS} for t in B_TARGETS}
    save_json("exp29_partB_phantom_residual.json", {"seeds": list(range(seeds)), "agg": agg, "picks": picks,
                                                    "wall_s": time.time() - t0})
    print("\n|r_hat_lin| [m/s^2] | replay err @1 s with residual / without [m]   (mean ± std)")
    for t in B_TARGETS:
        for m in METHODS:
            a = agg[t][m]
            print(f"  {t:10s} {m:8s} {fmt(a['phantom_lin']):>16} | {fmt(a['replay_err_1s']):>16} / "
                  f"{fmt(a['replay_err_1s_nores']):>16}   picks (d, tau): {picks[t][m]}")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(B_TARGETS))
    for j, m in enumerate(METHODS):
        for k, key in enumerate(["phantom_lin", "replay_err_1s"]):
            ax[k].bar(x + (j - 1) * 0.27, [agg[t][m][key]["mean"] for t in B_TARGETS], 0.27,
                      yerr=[agg[t][m][key]["std"] for t in B_TARGETS], capsize=2, label=m)
    ax[0].set_title("predicted residual |r_lin| [m/s²] (truth = 0 for l2f_*)")
    ax[1].set_title("open-loop replay error at 1 s [m]")
    for a in ax:
        a.set_xticks(x, B_TARGETS)
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp29_partB_phantom_residual.png", dpi=140)


# ---------------------------------------------------------------- Part A -------------------------------------------
ARMS = ["cmd", "app", "both"]
REGIMES = {"fixed1": (1,), "rand02": (0, 1, 2)}
TEST_D = [0, 1, 2, 3, 4, 6]


def a_job(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from applied_readback import ReadbackRacingEnv
    from exp10_method_matrix import HIST, SB3PolicyCallable
    from racing_env import evaluate_racing

    torch.set_num_threads(1)
    arm, regime, seed, base_steps, ft_steps, n_eval = args
    t0 = time.time()
    ppo_kw = dict(n_steps=512, batch_size=512, n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.0, seed=seed, verbose=0,
                  policy_kwargs=dict(net_arch=[128, 128], log_std_init=-0.7))

    def venv(lam, s0):
        return DummyVecEnv([lambda i=i: ReadbackRacingEnv("figure8", "source", HIST, lam_da=lam, delays=REGIMES[regime],
                                                          hist_mode=arm, max_ep_s=5.5, seed=s0 + i) for i in range(8)])

    model = PPO("MlpPolicy", venv(0.02, 5000 * seed), **ppo_kw)
    model.learn(base_steps)
    path = f"{OUT}/exp29_{arm}_{regime}_s{seed}.zip"
    model.save(path)
    model = PPO.load(path, env=venv(0.20, 5100 * seed), device="cpu")
    model.learn(ft_steps, reset_num_timesteps=False)
    model.save(path)

    pol = SB3PolicyCallable(model)

    def ev(course, dom, kf, d=None, jitter=None):
        env = ReadbackRacingEnv(course, dom, HIST, delays=(0 if jitter else d,), kf_scale=kf, hist_mode=arm,
                                jitter=jitter, random_start_gate=False, max_ep_s=12.5, seed=800 + seed)
        return evaluate_racing(pol, env, n_eval=n_eval)

    res = {"figure8_source_train": ev("figure8", "source", 1.0, d=REGIMES[regime][-1])}
    for dom, kf in [("source", 1.0), ("target", 1.0)]:
        name = "source_l2f" if dom == "source" else "target_nominal"
        for d in TEST_D:
            res[f"{name}_d{d}"] = ev("ellipse3d", dom, kf, d=d)
        res[f"{name}_jit14"] = ev("ellipse3d", dom, kf, jitter=(1, 4))
    res["target_perturbed_d3"] = ev("ellipse3d", "target", 1.20, d=3)
    res["target_perturbed_jit14"] = ev("ellipse3d", "target", 1.20, jitter=(1, 4))
    return arm, regime, seed, res, time.time() - t0


def part_a(seeds, workers, base_steps, ft_steps, n_eval):
    from exp10_method_matrix import aggregate_seed_dicts

    jobs = [(a, r, s, base_steps, ft_steps, n_eval) for s in range(seeds) for r in REGIMES for a in ARMS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        outs = pool.map(a_job, jobs, chunksize=1)
    print(f"Exp 29 A: {len(jobs)} runs in {time.time() - t0:.0f}s")
    cells = list(outs[0][3])
    agg = {r: {a: {c: aggregate_seed_dicts([o[3][c] for o in outs if o[0] == a and o[1] == r]) for c in cells}
               for a in ARMS} for r in REGIMES}
    save_json("exp29_partA_readback_racing.json", {"seeds": list(range(seeds)), "base_steps": base_steps,
                                                   "ft_steps": ft_steps, "n_eval": n_eval, "agg": agg,
                                                   "wall_s": time.time() - t0})
    for r in REGIMES:
        print(f"\n[train {r}] gate_frac / clean_completion / crash on held-out ellipse3d (mean over {seeds} seeds)")
        for c in cells:
            row = " || ".join(f"{a}: {agg[r][a][c]['gate_frac']['mean']:.2f}/{agg[r][a][c]['clean_completion']['mean']:.2f}/"
                              f"{agg[r][a][c]['crash_rate']['mean']:.2f}" for a in ARMS)
            print(f"  {c:26s} {row}")
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5), sharey=True)
    for i, r in enumerate(REGIMES):
        for k, dom in enumerate(["source_l2f", "target_nominal"]):
            for a in ARMS:
                m = [agg[r][a][f"{dom}_d{d}"]["gate_frac"]["mean"] for d in TEST_D]
                s = [agg[r][a][f"{dom}_d{d}"]["gate_frac"]["std"] for d in TEST_D]
                ax[i, k].errorbar(TEST_D, m, yerr=s, marker="o", capsize=2, label=a)
            ax[i, k].set_title(f"train {r} → {dom} (ellipse3d)")
            ax[i, k].set_xlabel("test action delay d [ticks]")
    ax[0, 0].set_ylabel("gate fraction")
    ax[1, 0].set_ylabel("gate fraction")
    ax[0, 0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp29_partA_readback_racing.png", dpi=140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["A", "B"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--base-steps", type=int, default=450_000)
    ap.add_argument("--ft-steps", type=int, default=100_000)
    ap.add_argument("--n-eval", type=int, default=6)
    args = ap.parse_args()
    if args.part == "B":
        part_b(args.seeds, args.workers)
    else:
        part_a(args.seeds, args.workers, args.base_steps, args.ft_steps, args.n_eval)


if __name__ == "__main__":
    main()
