"""
Exp 21 (H3' retest after the 2026-11-26 audit): sensor-limited racing students with track-free velocity observers.

Exp 10's sensor-limited half failed for every method (100% crashes even in the source on the training course). The
diagnosis (see kin_observer.py and REAL_SIM_RESULTS.md, Round 5) was (a) a course-position shortcut in learned
estimators and (b) a training sensor at the edge of feasibility. This experiment re-tests H3' with track-free
observers, over a sweep of sensor difficulty instead of a single cell.

Students (all deploy the frozen Exp 10 nominal teacher `exp10_base_nom_s{seed}`, except `teacher_exact`):
    teacher_exact   exact-state teacher                                 (upper reference, not deployable)
    alpha_beta      teacher + classical alpha-beta filter               (hand-designed observer baseline)
    kin_ff          teacher + MLP on last 8 kinematic frames            (recurrence ablation)
    kin_gru_fixed   teacher + GRU estimator, DAgger in fixed nominal L2F       (Thm 2.4: fixed-dynamics observer)
    kin_gru_rand    teacher + GRU estimator, DAgger in randomized L2F + d~U{0..4} (Thm 2.4: randomized observer)
Sensor levels (train = eval): clean (obs_delay 0, p_drop 0), mid (1, 0.1), hard (2, 0.2 — the Exp 10 setting);
all levels have 2 cm position / 0.04 rad/s gyro noise. Targets also have the Exp 10 oblique PnP gate bias (scale 1).
Eval: fixed start, 2 laps, figure8 + held-out ellipse3d on source_l2f (d=0), target_nominal (d=2),
target_perturbed (kf x 1.20, d=3), 10 episodes per cell.

Usage:  python exp21_h3_observer_retest.py [--seeds 3] [--workers 12]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

from common import OUT, fmt, np, plt, save_json
from exp10_method_matrix import TRAIN_DELAYS, aggregate_seed_dicts, dyn_sampler

STUDENTS = ["teacher_exact", "alpha_beta", "kin_ff", "kin_gru_fixed", "kin_gru_rand"]
LEVELS = {"clean": (0, 0.0), "mid": (1, 0.1), "hard": (2, 0.2)}
TARGETS = {"source_l2f": ("source", 0, 1.0, 0.0), "target_nominal": ("target", 2, 1.0, 1.0),
           "target_perturbed": ("target", 3, 1.20, 1.0)}
COURSES = ["figure8", "ellipse3d"]


class ExactTeacher:
    def __init__(self, teacher):
        self.teacher = teacher

    def __call__(self, obs, env, step_idx):
        from asym_policy import N_PRIV
        return self.teacher.predict(np.concatenate([obs[-N_PRIV:], obs[21:53]]).astype(np.float32),
                                    deterministic=True)[0]


def job(args):
    import torch
    from stable_baselines3 import PPO
    from asym_policy import PrivStackRacingEnv
    from kin_observer import AlphaBeta, KinStudent, distill_kin
    from racing_env import evaluate_racing

    torch.set_num_threads(1)
    student, level, seed, n_eval = args
    od, pd = LEVELS[level]
    t0 = time.time()
    teacher = PPO.load(f"{OUT}/exp10_base_nom_s{seed}.zip", device="cpu")
    if student == "teacher_exact":
        pol = ExactTeacher(teacher)
    elif student == "alpha_beta":
        pol = AlphaBeta(teacher)
    else:
        rand = student == "kin_gru_rand"
        est = distill_kin(teacher, recurrent=student != "kin_ff", obs_delay=od, p_drop=pd, seed=100 + seed,
                          dyn_sampler=dyn_sampler if rand else None, delays=TRAIN_DELAYS if rand else (0,))
        torch.save(est.state_dict(), f"{OUT}/exp21_{student}_{level}_s{seed}.pt")
        pol = KinStudent(est, teacher)
    ev = {c: {t: evaluate_racing(pol, PrivStackRacingEnv(c, dom, 8, delays=(d,), kf_scale=kf, obs_delay=od, p_drop=pd,
                                                         pnp_bias_scale=pnp, random_start_gate=False, max_ep_s=12.5,
                                                         seed=800 + seed), n_eval=n_eval)
              for t, (dom, d, kf, pnp) in TARGETS.items()} for c in COURSES}
    return student, level, seed, ev, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-eval", type=int, default=10)
    args = ap.parse_args()

    jobs = [(st, lv, s, args.n_eval) for s in range(args.seeds) for lv in LEVELS for st in STUDENTS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        outs = pool.map(job, jobs, chunksize=1)
    print(f"Exp 21: {len(jobs)} jobs in {time.time() - t0:.0f}s")

    agg = {st: {lv: {c: {t: aggregate_seed_dicts([o[3][c][t] for o in outs if o[0] == st and o[1] == lv])
                         for t in TARGETS} for c in COURSES} for lv in LEVELS} for st in STUDENTS}
    save_json("exp21_h3_observer_retest.json", {"seeds": list(range(args.seeds)), "n_eval": args.n_eval,
                                                "wall_s": time.time() - t0, "agg": agg})

    for c in COURSES:
        print(f"\n[{c}] gate_frac / crash  (mean ± std over {args.seeds} seeds)")
        for lv in LEVELS:
            for st in STUDENTS:
                row = " || ".join(f"{t}: {fmt(agg[st][lv][c][t]['gate_frac'])} / {agg[st][lv][c][t]['crash_rate']['mean']:.2f}"
                                  for t in TARGETS)
                print(f"  {lv:5s} {st:14s} {row}")

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    x = np.arange(len(LEVELS))
    for k, t in enumerate(TARGETS):
        for j, st in enumerate(STUDENTS):
            ax[k].bar(x + (j - 2) * 0.16, [agg[st][lv]["ellipse3d"][t]["gate_frac"]["mean"] for lv in LEVELS], 0.16,
                      yerr=[agg[st][lv]["ellipse3d"][t]["gate_frac"]["std"] for lv in LEVELS], capsize=2, label=st)
        ax[k].set_xticks(x, list(LEVELS))
        ax[k].set_title(f"held-out ellipse3d, {t}: gate fraction")
        ax[k].set_xlabel("sensor level")
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp21_h3_observer_retest.png", dpi=140)


if __name__ == "__main__":
    main()
