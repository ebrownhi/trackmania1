"""
Exp 5b (H3' / Thm 2.4 retest): Exp 5 with a harder sensor channel, a larger DAgger budget, and a perturbed
target with a bigger dynamics gap.

Changes vs Exp 5 (everything else imported unchanged from exp5_recurrent_observer.py):
  - sensor: obs delay 3 ticks (was 1), dropout 20% (was 10%), velocity removed ("novel" mode only)
  - DAgger: 8 iterations x 12 episodes (was 5 x 8)
  - targets: nominal PyBullet CF2X and a perturbed CF2X with 15% stronger motors (kf x 1.15), so the
    Markovian gap ||Delta f|| that Theorem 2.4 amplifies by (1-K)/K is ~2.3x larger than in Exp 5.
"""

import multiprocessing as mp
import time

import exp5_recurrent_observer as E
from common import OUT, np, plt, save_json, mean_std
from sims.drone_pair import Course, make_target

E.Sensor.__init__.__defaults__ = (3, 0.2)
MODE = "novel"
SEEDS = [0, 1, 2]
TARGETS = ["target", "target_pert"]
KF_SCALE = 1.15

_make_env = E.make_env


def make_env(domain, rng, seed):
    if domain == "target_pert":
        d = make_target(seed=seed)
        d.backend.env.KF *= KF_SCALE  # physics uses env.KF; the wrapper's command->rpm map keeps nominal kf
        return d
    return _make_env(domain, rng, seed)


E.make_env = make_env


def probe_job(seed):
    course = Course()
    rng = np.random.default_rng(seed)
    data = {}
    plan = [("src_fixed", "src_fixed", 24), ("src_rand", "src_rand", 24), ("test_src", "src_fixed", 6),
            ("test_target", "target", 6), ("test_target_pert", "target_pert", 6)]
    for j, (key, dom, n) in enumerate(plan):
        eps = [E.rollout(dom, MODE, course, rng, 10_000 * seed + 1000 * j + i, 5.0) for i in range(n)]
        data[key] = ([e[0] for e in eps], [e[1] for e in eps])
    out = {}
    D = E.feat_dim(MODE) + 4
    for train in ["src_fixed", "src_rand"]:
        for arch, cls in [("FF", E.FFNet), ("GRU", E.GRUNet)]:
            m = cls(D, 3)
            E.train_seq(m, *data[train], steps=800, seed=seed)
            r = {}
            for test in ["test_src", "test_target", "test_target_pert"]:
                with E.torch.no_grad():
                    errs = [((m(E.torch.as_tensor(x, dtype=E.torch.float32)[None])[0].numpy() - v)[50:])
                            for x, v in zip(*data[test])]
                r[test] = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
            out[f"{arch}_{'fixed' if train == 'src_fixed' else 'rand'}"] = r
    return seed, out


def evaluate(model, seed):
    course = Course()
    ev = {}
    for dom in ["src_fixed"] + TARGETS:
        mets = [E.rollout(dom, MODE, course, np.random.default_rng(7000 + seed * 10 + e), 7000 + seed * 10 + e,
                          2 * course.period, student=model, act_noise=0.0, t0=0.0)[3] for e in range(3)]
        ev[dom] = {k: float(np.mean([m[k] for m in mets])) for k in mets[0]}
    return ev


def dagger_job(args):
    arch, train_dom, seed = args
    course = Course()
    rng = np.random.default_rng(1000 + seed)
    model = (E.GRUNet if arch == "GRU" else E.FFNet)(E.feat_dim(MODE) + 4, 4, squash=True)
    Xs, Ys = [], []
    for it in range(8):
        for ep in range(12):
            X, _, Y, _ = E.rollout(train_dom, MODE, course, rng, 50_000 * seed + 100 * it + ep, 6.0,
                                   student=None if it == 0 else model, act_noise=0.05)
            Xs.append(X)
            Ys.append(Y)
        E.train_seq(model, Xs, Ys, steps=600 if it < 7 else 1200, seed=seed + it)
    return arch, train_dom, seed, evaluate(model, seed)


def teacher_job(seed):
    return seed, evaluate(None, seed)


def main():
    t0 = time.time()
    with mp.get_context("spawn").Pool(4) as pool:
        probe = dict(pool.map(probe_job, SEEDS))
        teach = dict(pool.map(teacher_job, SEEDS))
        dag = pool.map(dagger_job, [(a, d, s) for a in ["FF", "GRU"] for d in ["src_fixed", "src_rand"] for s in SEEDS])
    print(f"Exp 5b done in {time.time() - t0:.0f}s")

    doms = ["src_fixed"] + TARGETS
    res = {"probe": {}, "teacher": {}, "closed_loop": {}, "rho_obs": {},
           "meta": {"obs_delay": 3, "p_drop": 0.2, "mode": MODE, "kf_scale_pert": KF_SCALE, "wall_s": time.time() - t0}}
    for key in probe[0]:
        res["probe"][key] = {t: mean_std([probe[s][key][t] for s in SEEDS]) for t in probe[0][key]}
    res["teacher"] = {dom: {k: mean_std([teach[s][dom][k] for s in SEEDS]) for k in teach[0][dom]} for dom in doms}
    per = {}
    for a in ["FF", "GRU"]:
        for d in ["src_fixed", "src_rand"]:
            name = f"{a}_{'fixed' if d == 'src_fixed' else 'rand'}"
            runs = {s: ev for aa, dd, s, ev in dag if (aa, dd) == (a, d)}
            per[name] = runs
            res["closed_loop"][name] = {dom: {k: mean_std([runs[s][dom][k] for s in SEEDS]) for k in runs[0][dom]}
                                        for dom in doms}
    for dom in TARGETS:
        res["rho_obs"][dom] = {}
        for tr in ["fixed", "rand"]:
            rhos = []
            for s in SEEDS:
                J_t, J_ff = -teach[s][dom]["eff_rmse"], -per[f"FF_{tr}"][s][dom]["eff_rmse"]
                J_rnn = -per[f"GRU_{tr}"][s][dom]["eff_rmse"]
                rhos.append((J_rnn - J_ff) / (J_t - J_ff) if abs(J_t - J_ff) > 0.02 else float("nan"))
            res["rho_obs"][dom][f"GRU_{tr}_vs_FF_{tr}"] = rhos
    save_json("exp5b_recurrent_observer_hard.json", res)

    print("\nPart A: body-velocity RMSE [m/s] (mean ± std, 3 seeds)")
    for key, r in res["probe"].items():
        print(f"  {key:10s} " + "  ".join(f"{t[5:]} {v['mean']:.3f} ± {v['std']:.3f}" for t, v in r.items()))
    print("\nPart B: eff_rmse [m] (crash = 1 m) / gates / crash")
    rows = {"teacher": res["teacher"], **res["closed_loop"]}
    for name, r in rows.items():
        print(f"  {name:10s} " + " | ".join(f"{dom} {r[dom]['eff_rmse']['mean']:.3f} ± {r[dom]['eff_rmse']['std']:.3f} / "
                                            f"{r[dom]['gates']['mean']:.2f} / {r[dom]['crash']['mean']:.2f}" for dom in doms))
    for dom, r in res["rho_obs"].items():
        for k, v in r.items():
            print(f"  rho_obs {dom} {k}: {np.round(v, 2).tolist()}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    keys = ["FF_fixed", "GRU_fixed", "FF_rand", "GRU_rand"]
    tests = ["test_src", "test_target", "test_target_pert"]
    x = np.arange(len(keys))
    for j, t in enumerate(tests):
        ax[0].bar(x + (j - 1) * 0.27, [res["probe"][k][t]["mean"] for k in keys], 0.27,
                  yerr=[res["probe"][k][t]["std"] for k in keys], capsize=2, label=t[5:])
    ax[0].set_xticks(x, keys)
    ax[0].set_ylabel("body-velocity RMSE [m/s]")
    ax[0].set_title("Observer probe (obs delay 3, 20% dropout, no velocity)")
    ax[0].legend(fontsize=7)
    for j, dom in enumerate(doms):
        ax[1].bar(x + (j - 1) * 0.27, [res["closed_loop"][k][dom]["eff_rmse"]["mean"] for k in keys], 0.27,
                  yerr=[res["closed_loop"][k][dom]["eff_rmse"]["std"] for k in keys], capsize=2, label=dom)
    ax[1].set_xticks(x, keys)
    ax[1].set_ylabel("effective RMSE [m] (crash = 1 m)")
    ax[1].set_title("Closed-loop students")
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp5b_recurrent_observer_hard.png", dpi=140)


if __name__ == "__main__":
    main()
