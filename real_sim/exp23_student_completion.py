"""
Exp 23: from gate fraction to full-course completion for sensor-limited students (builds on Exp 21).

Part A diagnosis (seeds 0-1, Exp 21 `kin_gru_rand`, held-out ellipse3d; scratch script, numbers in the report):
  * target without the PnP gate bias: the student already completes the course (completion 1.00, 0 crashes) at all
    three sensor levels -> the whole target gap is the bias; with the bias, completion 0 (gate fraction 0.76-0.86);
  * L2F source: the velocity estimate is the bottleneck (clean level: true velocity -> completion 1.00, estimate -> 0);
    on ellipse3d the estimator's body-z error triples (0.24-0.28 vs 0.09-0.13 m/s on figure8) because the planar
    training course under-covers vertical velocity; at the hard level the 2-tick-stale geometry also matters.

Part B ablation (student_plus.py), all on the same frozen Exp 10 nominal teacher, 3 seeds x 3 sensor levels:
    v0_exp21       Exp 21 `kin_gru_rand` estimator, unchanged                          (baseline)
    v1_f1          + F1 estimator re-trained on a multi-track pool (figure8 + 6 Exp 18 circuits, excluding ellipse3d and
                   its near-duplicates; the estimator is track-free, so no track geometry is learned), 16 eps/iter.
                   (AR(1) action-noise exploration was tried first, seed 0: no gain, 0.42 -> 0.35-0.40 source gates.)
    v2_f1f2        + F2 delay compensation (predict observed pose forward by its staleness)
    v3_f1f2f3      + F3 PnP-bias GP from 3 target calibration flights on the *training* course (figure8)
    v4_f3only      Exp 21 estimator + F3 only (isolates the bias fix)
Eval cells/metrics identical to Exp 21 (fixed start, 2 laps, figure8 + held-out ellipse3d; source_l2f, target_nominal
with d=2, target_perturbed with kf x 1.20 and d=3; target PnP bias scale 1; 10 episodes per cell).

Usage:  python exp23_student_completion.py [--seeds 3] [--workers 5] [--smoke]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

from common import OUT, fmt, np, plt, save_json
from exp10_method_matrix import TRAIN_DELAYS, aggregate_seed_dicts, dyn_sampler
from exp21_h3_observer_retest import COURSES, LEVELS, TARGETS

VARIANTS = ["v0_exp21", "v1_f1", "v2_f1f2", "v3_f1f2f3", "v4_f3only"]
METRICS = ["completion", "clean_completion", "gate_frac", "crash_rate"]


def job(args):
    import torch
    from stable_baselines3 import PPO
    from asym_policy import PrivStackRacingEnv
    from kin_observer import KinGRU, distill_kin
    from racing_env import evaluate_racing
    from student_plus import PlusStudent, calibrate_bias, track_pool

    torch.set_num_threads(1)
    level, seed, n_eval, iters, eps = args
    od, pd = LEVELS[level]
    t0 = time.time()
    teacher = PPO.load(f"{OUT}/exp10_base_nom_s{seed}.zip", device="cpu")
    est_kw = dict(recurrent=True, obs_delay=od, p_drop=pd, seed=100 + seed, dyn_sampler=dyn_sampler, delays=TRAIN_DELAYS,
                  iters=iters)
    base_path = f"{OUT}/exp21_kin_gru_rand_{level}_s{seed}.pt"
    est0 = KinGRU()
    if os.path.exists(base_path):
        est0.load_state_dict(torch.load(base_path))
    else:  # weights are not version-controlled: retrain with the identical Exp 21 recipe
        est0 = distill_kin(teacher, eps=10, **est_kw)
    est1 = distill_kin(teacher, eps=eps, courses=track_pool(), **est_kw)
    torch.save(est1.state_dict(), f"{OUT}/exp23_kin_gru_f1_{level}_s{seed}.pt")

    def env(course, tname, s_off=0):
        dom, d, kf, pnp = TARGETS[tname]
        return PrivStackRacingEnv(course, dom, 8, delays=(d,), kf_scale=kf, obs_delay=od, p_drop=pd, pnp_bias_scale=pnp,
                                  random_start_gate=False, max_ep_s=12.5, seed=800 + seed + s_off)

    # F3 calibration: per target, 3 flights on the training course flown by the F1+F2 student, mocap labels
    gp, cal = {}, {}
    for tname in ["target_nominal", "target_perturbed"]:
        hook, diag = calibrate_bias(PlusStudent(est1, teacher, od, delay_comp=True), env("figure8", tname, 50),
                                    n_flights=3, seed=seed)
        gp[tname], cal[tname] = hook, diag

    def policy(variant, tname):
        g = gp.get(tname)  # None in the source (no PnP bias there)
        return {"v0_exp21": PlusStudent(est0, teacher, od),
                "v1_f1": PlusStudent(est1, teacher, od),
                "v2_f1f2": PlusStudent(est1, teacher, od, delay_comp=True),
                "v3_f1f2f3": PlusStudent(est1, teacher, od, delay_comp=True, gp_hook=g),
                "v4_f3only": PlusStudent(est0, teacher, od, gp_hook=g)}[variant]

    ev = {v: {c: {t: evaluate_racing(policy(v, t), env(c, t), n_eval=n_eval) for t in TARGETS} for c in COURSES}
          for v in VARIANTS}
    return level, seed, ev, cal, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        jobs = [("clean", 0, 1, 1, 2)]
    else:
        jobs = [(lv, s, args.n_eval, 6, 16) for s in range(args.seeds) for lv in LEVELS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        outs = pool.map(job, jobs, chunksize=1)
    print(f"Exp 23: {len(jobs)} jobs in {time.time() - t0:.0f}s")
    levels = sorted({o[0] for o in outs}, key=list(LEVELS).index)

    agg = {v: {lv: {c: {t: aggregate_seed_dicts([o[2][v][c][t] for o in outs if o[0] == lv]) for t in TARGETS}
                    for c in COURSES} for lv in levels} for v in VARIANTS}
    cal = {f"{o[0]}_s{o[1]}": o[3] for o in outs}
    tag = "_smoke" if args.smoke else ""
    save_json(f"exp23_student_completion{tag}.json", {"seeds": sorted({o[1] for o in outs}), "f1": "multi-track, 16 eps/iter",
                                                      "n_eval": args.n_eval, "wall_s": time.time() - t0, "agg": agg,
                                                      "calibration": cal})
    for c in COURSES:
        print(f"\n[{c}] completion | clean_completion | gate_frac | crash  (mean ± std over seeds)")
        for lv in levels:
            for v in VARIANTS:
                row = " || ".join(f"{t}: " + " | ".join(f"{agg[v][lv][c][t][m]['mean']:.2f}" for m in METRICS) for t in TARGETS)
                print(f"  {lv:5s} {v:10s} {row}")
    for k, d in cal.items():
        print(f"  calib {k}: " + ", ".join(f"{t}: n={x['n_samples']} |b| rms {x['label_rms_m']:.3f} -> resid {x['heldout_resid_rms_m']:.3f} m"
                                         for t, x in d.items()))

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    x = np.arange(len(levels))
    for k, t in enumerate(TARGETS):
        for j, v in enumerate(VARIANTS):
            ax[k].bar(x + (j - 2) * 0.16, [agg[v][lv]["ellipse3d"][t]["completion"]["mean"] for lv in levels], 0.16,
                      yerr=[agg[v][lv]["ellipse3d"][t]["completion"]["std"] for lv in levels], capsize=2, label=v)
        ax[k].set_xticks(x, levels)
        ax[k].set_title(f"held-out ellipse3d, {t}: full-course completion")
        ax[k].set_xlabel("sensor level")
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp23_student_completion{tag}.png", dpi=140)


if __name__ == "__main__":
    main()
