"""
Exp 25 (literature riff 1, docs/FOLLOWUP_RIFFS_ROUND2.md): does source (L2F) performance predict target (PyBullet)
performance across policies?

No training: every (method/arm, seed[, budget, sensor level]) already evaluated in Exp 10, 19, 21, 22, 23, 24 is one
policy, with its per-seed score in the source cell and the target cells of the same course. Duplicates are dropped
(Exp 19 `none`/`full` = Exp 10 `nominal_ppo`/`dr_ppo` = Exp 24 at 550k; Exp 23 `v0_exp21` = Exp 21 `kin_gru_rand`).

Metrics between the vector of source scores s and target scores r over policies:
    Pearson r     (Sim2Real Correlation Coefficient, Kadian et al. 2020)
    Spearman rho
    MMRV          mean maximum rank violation (SIMPLER, Li et al. 2024), as defined here:
                  MMRV = (1/N) sum_i max_j |r_i - r_j| * 1[(s_i - s_j)(r_i - r_j) < 0]
                  i.e. for each policy, the largest target-score margin by which the source ranks it the wrong way
                  against any other policy; 0 = ranking fully preserved, in target-metric units. (Definition written
                  from memory of the SIMPLER paper, not re-checked against its text.)
95% CIs: percentile bootstrap over policies (2000 resamples).

Usage:  python exp25_predictivity.py
"""

from __future__ import annotations

import json

from common import OUT, np, plt, save_json

METRICS = ["gate_frac", "clean_completion"]
COURSES = ["figure8", "ellipse3d"]
TARGETS = ["target_nominal", "target_perturbed"]
TARGET_DATA = {"residual_ft_ppo", "target_oracle_ppo", "residual_gp_gru"}  # trained partly on target data / bias fits


def load(name, root="agg"):
    return json.load(open(f"{OUT}/{name}.json"))[root]


def add(policies, fam, label, cell_fn):
    """cell_fn(course, target) -> metric dict {metric: {"values": [...]}} ; one policy per seed index."""
    n = len(cell_fn("figure8", "source_l2f")["gate_frac"]["values"])
    for k in range(n):
        p = {"family": fam, "label": f"{label}#s{k}", "scores": {}}
        for c in COURSES:
            for t in ["source_l2f"] + TARGETS:
                p["scores"][(c, t)] = {m: cell_fn(c, t)[m]["values"][k] for m in METRICS}
        policies.append(p)


def collect():
    exact, sensor = [], []
    e10 = load("exp10_method_matrix", "matrix")
    for meth in ["residual_ft_ppo", "target_oracle_ppo"]:
        add(exact, "target_data", f"exp10/{meth}", lambda c, t, m=meth: e10["exact_state"][c][t][m])
    e24 = load("exp24_dr_budget_sweep")
    for arm in e24:
        for b in e24[arm]:
            fam = "no_dr" if arm == "none" else ("latency_dr" if arm == "latency_only" else "dynamics_dr")
            add(exact, fam, f"exp24/{arm}@{int(b) // 1000}k", lambda c, t, a=arm, bb=b: e24[a][bb][c][t])
    e19 = load("exp19_dr_factor_ablation")
    for v in ["no_thrust", "no_motor_lag", "no_latency"]:
        add(exact, "dynamics_dr", f"exp19/{v}", lambda c, t, vv=v: e19[vv][c][t])

    for meth in e10["sensor_limited"]["figure8"]["source_l2f"]:
        fam = "target_data" if meth in TARGET_DATA else "exp10_student"
        add(sensor, fam, f"exp10/{meth}", lambda c, t, m=meth: e10["sensor_limited"][c][t][m])
    e21 = load("exp21_h3_observer_retest")
    for st in e21:
        for lv in e21[st]:
            add(sensor, "teacher_ref" if st == "teacher_exact" else "observer_student", f"exp21/{st}/{lv}",
                lambda c, t, s=st, l=lv: e21[s][l][c][t])
    for lv in ["clean", "mid", "hard"]:
        e22 = load(f"exp22_privileged_critic_trackfree_scratch_{lv}")
        for arm in e22:
            add(sensor, "ppo_student", f"exp22/{arm}/{lv}", lambda c, t, a=arm, d=e22: d[a][c][t])
    e23 = load("exp23_student_completion")
    for v in e23:
        if v == "v0_exp21":
            continue
        for lv in e23[v]:
            fam = "target_data" if "f3" in v else "observer_student"  # F3 = GP fit on target calibration flights
            add(sensor, fam, f"exp23/{v}/{lv}", lambda c, t, vv=v, l=lv: e23[vv][l][c][t])
    return exact, sensor


def pearson(s, r):
    if np.std(s) < 1e-12 or np.std(r) < 1e-12:
        return float("nan")
    return float(np.corrcoef(s, r)[0, 1])


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):  # average ranks for ties
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(s, r):
    return pearson(rankdata(s), rankdata(r))


def mmrv(s, r):
    ds, dr = s[:, None] - s[None, :], r[:, None] - r[None, :]
    viol = (ds * dr) < 0
    return float(np.mean(np.max(np.abs(dr) * viol, axis=1)))


def stats(s, r, rng, n_boot=2000):
    out = {"n": len(s), "src_mean": float(np.mean(s)), "tgt_mean": float(np.mean(r))}
    for name, f in [("pearson", pearson), ("spearman", spearman), ("mmrv", mmrv)]:
        v = f(s, r)
        bs = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(s), len(s))
            b = f(s[idx], r[idx])
            if np.isfinite(b):
                bs.append(b)
        lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (float("nan"), float("nan")))
        out[name] = {"value": v, "ci": [float(lo), float(hi)]}
    return out


def analyse(policies, families, rng):
    res = {}
    for c in COURSES:
        for t in TARGETS:
            for m in METRICS:
                sel = [p for p in policies if p["family"] in families]
                s = np.array([p["scores"][(c, "source_l2f")][m] for p in sel])
                r = np.array([p["scores"][(c, t)][m] for p in sel])
                res[f"{c}|{t}|{m}"] = stats(s, r, rng)
    return res


def main():
    rng = np.random.default_rng(0)
    exact, sensor = collect()
    groups = {
        "exact_all": (exact, {"no_dr", "latency_dr", "dynamics_dr", "target_data"}),
        "exact_source_only": (exact, {"no_dr", "latency_dr", "dynamics_dr"}),
        "exact_dynamics_dr": (exact, {"dynamics_dr"}),
        "sensor_all": (sensor, {"exp10_student", "observer_student", "ppo_student", "target_data", "teacher_ref"}),
        "sensor_source_only": (sensor, {"exp10_student", "observer_student", "ppo_student", "teacher_ref"}),
        "sensor_observer_students": (sensor, {"observer_student"}),
    }
    res = {g: analyse(pol, fams, rng) for g, (pol, fams) in groups.items()}
    counts = {g: sum(p["family"] in fams for p in pol) for g, (pol, fams) in groups.items()}
    save_json("exp25_predictivity.json", {
        "mmrv_definition": "(1/N) sum_i max_j |r_i - r_j| * 1[(s_i - s_j)(r_i - r_j) < 0]",
        "n_policies": counts, "results": res,
        "policies": {"exact": [p["label"] for p in exact], "sensor": [p["label"] for p in sensor]}})

    f = lambda d: f"{d['value']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"
    g = lambda d: f"{d['value']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"
    for grp in groups:
        print(f"\n[{grp}] n={counts[grp]} policies   Pearson | Spearman | MMRV   (95% bootstrap CI)")
        for key, v in res[grp].items():
            print(f"  {key:42s} {f(v['pearson'])} | {f(v['spearman'])} | {g(v['mmrv'])}  "
                  f"(src {v['src_mean']:.2f}, tgt {v['tgt_mean']:.2f})")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for row, (grp, (pol, fams)) in enumerate([("exact_all", groups["exact_all"]), ("sensor_all", groups["sensor_all"])]):
        sel = [p for p in pol if p["family"] in fams]
        fam_list = sorted({p["family"] for p in sel})
        for col, (c, t) in enumerate([(c, t) for c in COURSES for t in TARGETS]):
            ax = axes[row, col]
            for fam in fam_list:
                pts = [p for p in sel if p["family"] == fam]
                ax.scatter([p["scores"][(c, "source_l2f")]["gate_frac"] for p in pts],
                           [p["scores"][(c, t)]["gate_frac"] for p in pts], s=14, alpha=0.7, label=fam)
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            v = res[grp][f"{c}|{t}|gate_frac"]
            ax.set_title(f"{grp.split('_')[0]}: {c} → {t}\nρ={v['spearman']['value']:.2f}, MMRV={v['mmrv']['value']:.2f}",
                         fontsize=8)
            ax.set_xlabel("source gate frac")
            ax.set_ylabel("target gate frac")
        axes[row, 0].legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp25_predictivity.png", dpi=140)


if __name__ == "__main__":
    main()
