"""
Exp 4 (Theorem 2.2 / guardrail #2): action-history buffer a_{t-8:t-1} vs previous action a_{t-1} under delay.

For each action delay d, DAgger-distill MLP students in the L2F source from a teacher that sees the exact
future state s_{t+d} (a perfect source-model Smith predictor). Theorem 2.2: for d >= 2 the input
(o_t, a_{t-1}) is missing d-1 in-flight actions, so the a_{t-1} student cannot reproduce the prediction;
the buffer student can. Evaluate zero-shot in the source and in the PyBullet target at the same d.

Baselines: naive expert (acts on o_t, ignores delay), Smith expert (source-model forward prediction),
and a nominal buffer student trained at d = 0.
"""

import multiprocessing as mp
import time

from common import OUT, np, plt, save_json, mean_std
from sims.drone_pair import Course, InterfaceConfig, RL_DT, GeometricExpert, make_source, make_target

DELAYS = [0, 2, 4, 6, 8]
HISTS = {"a_prev": 1, "buffer8": 8}
SEEDS = [0, 1, 2]


def src(iface, s):
    return make_source(seed=s, iface=iface)


def tgt(iface, s):
    return make_target(iface=iface, seed=s)


def smith_policy(course):
    ex = GeometricExpert()

    def f(feat, ah, obs, t, env):
        fl = env.in_flight()
        # the target has no exact model, so the Smith expert always predicts with the *source* model
        return ex(_source_predictor(env).predict(obs, fl), *course.ref(t + len(fl) * RL_DT))
    return f


class _SourceShadow:
    def __init__(self):
        self.d = make_source()

    def predict(self, obs, actions):
        self.d.backend.reset(obs)
        return self.d.predict_forward(actions)


_SHADOW = {}


def _source_predictor(env):
    if "s" not in _SHADOW:
        _SHADOW["s"] = _SourceShadow()
    return _SHADOW["s"]


def summarize(runs):
    return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}


def train_job(args):
    from policy_lib import dagger, evaluate, mlp_policy
    d_train, hname, seed = args
    course = Course()
    t0 = time.time()
    model = dagger(src, course, HISTS[hname], lambda r: d_train, seed=seed)
    out = {"d_train": d_train, "hist": hname, "seed": seed, "train_s": time.time() - t0, "eval": {}}
    eval_delays = DELAYS if d_train == 0 and hname == "buffer8" else [d_train]
    for d in eval_delays:
        iface = InterfaceConfig(action_delay=d)
        out["eval"][d] = {"source": summarize(evaluate(mlp_policy(model), src, course, iface, seed=2000 + seed)),
                          "target": summarize(evaluate(mlp_policy(model), tgt, course, iface, seed=2000 + seed))}
    return out


def baseline_job(d):
    from policy_lib import evaluate, expert_policy
    course = Course()
    iface = InterfaceConfig(action_delay=d)
    return d, {
        "naive_expert": {"source": summarize(evaluate(expert_policy(course), src, course, iface, n_eps=6)),
                         "target": summarize(evaluate(expert_policy(course), tgt, course, iface, n_eps=6))},
        "smith_expert": {"source": summarize(evaluate(smith_policy(course), src, course, iface, n_eps=6)),
                         "target": summarize(evaluate(smith_policy(course), tgt, course, iface, n_eps=6))},
    }


def main():
    jobs = [(d, h, s) for d in DELAYS for h in HISTS for s in SEEDS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(14) as pool:
        base = dict(pool.map(baseline_job, DELAYS))
        outs = pool.map(train_job, jobs)
    print(f"Exp 4: {len(jobs)} DAgger students trained + evaluated in {time.time() - t0:.0f}s")

    res = {"delays": DELAYS, "baselines": base, "students": {}}
    for h in HISTS:
        res["students"][h] = {}
        for d in DELAYS:
            o = [x for x in outs if x["hist"] == h and x["d_train"] == d]
            res["students"][h][d] = {dom: {m: mean_std([x["eval"][d][dom][m] for x in o]) for m in o[0]["eval"][d][dom]}
                                     for dom in ["source", "target"]}
    nom = [x for x in outs if x["hist"] == "buffer8" and x["d_train"] == 0]
    res["students"]["nominal_d0_buffer8"] = {
        d: {dom: {m: mean_std([x["eval"][d][dom][m] for x in nom]) for m in nom[0]["eval"][d][dom]} for dom in ["source", "target"]}
        for d in DELAYS}
    save_json("exp4_action_buffer.json", res)

    for dom in ["source", "target"]:
        print(f"\n[{dom}]  gate pass rate (crash rate)  — students: mean over {len(SEEDS)} seeds")
        print(f"{'d':>3} | {'naive expert':>14} {'smith expert':>14} | {'nominal d=0':>14} {'a_prev':>14} {'buffer8':>14}")
        for d in DELAYS:
            b = base[d]
            cell = lambda m: f"{m['gates']:.2f} ({m['crash']:.2f})"
            scell = lambda m: f"{m['gates']['mean']:.2f} ({m['crash']['mean']:.2f})"
            print(f"{d:>3} | {cell(b['naive_expert'][dom]):>14} {cell(b['smith_expert'][dom]):>14} | "
                  f"{scell(res['students']['nominal_d0_buffer8'][d][dom]):>14} "
                  f"{scell(res['students']['a_prev'][d][dom]):>14} {scell(res['students']['buffer8'][d][dom]):>14}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for i, dom in enumerate(["source", "target"]):
        for name, key, mk in [("naive expert", "naive_expert", "k--"), ("Smith expert (source model)", "smith_expert", "k:")]:
            ax[i].plot(DELAYS, [base[d][key][dom]["gates"] for d in DELAYS], mk, label=name)
        for name, key, mk in [("nominal student (d=0)", "nominal_d0_buffer8", "^-"), ("a_{t-1} student", "a_prev", "s-"),
                              ("a_{t-8:t-1} student", "buffer8", "o-")]:
            m = [res["students"][key][d][dom]["gates"]["mean"] for d in DELAYS]
            s = [res["students"][key][d][dom]["gates"]["std"] for d in DELAYS]
            ax[i].errorbar(DELAYS, m, yerr=s, fmt=mk, capsize=3, label=name)
        ax[i].set_title(f"{dom} (L2F)" if dom == "source" else "target (PyBullet)")
        ax[i].set_xlabel("action delay d [10 ms ticks]")
    ax[0].set_ylabel("gate pass rate")
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp4_action_buffer.png", dpi=140)


if __name__ == "__main__":
    main()
