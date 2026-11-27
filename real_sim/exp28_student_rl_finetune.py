"""
Exp 28 (H3', idea 4 of docs/FOLLOWUP_RIFFS_ROUND2.md): RL fine-tuning of the distilled sensor-limited student
(Xing et al., CoRL 2024, arXiv 2403.12203).

Open gap after Exp 23: students complete the held-out course in the PyBullet targets, but collision-free completion
stays <= 0.50, and in the L2F source the hard sensor level completes nothing. Here the Exp 21 student (frozen Exp 10
nominal teacher + Exp 21 `kin_gru_rand` track-free GRU velocity estimator, no delay compensation; in the targets plus
the Exp 23 F3 PnP-bias GP) is fine-tuned with PPO (student_ft.py):
    actor  = the teacher MLP, acting on the estimator-filled observation (deployable; estimator frozen in the loop)
    critic = the teacher's value net on the privileged exact observation (asymmetric)
    init   = exact copy of the student (step-0 action identical), log_std = teacher's + LOG_STD_OFFSET
    train  = randomized L2F (Exp 10 dyn_sampler, action delay d~U{0..4}) on figure8 at the student's sensor level,
             Exp 10 fine-tune smoothness lam_da = 0.20, FT_STEPS steps, conservative PPO (lr 5e-5, clip 0.1, target_kl)
Arms (same estimator, same per-target GP, same eval envs and seeds):
    student      the un-fine-tuned student                         (= Exp 23 v4_f3only in the targets, v0 in the source)
    student_ft   the RL-fine-tuned student
Eval: Exp 21/23 cells (fixed start, 2 laps, figure8 + held-out ellipse3d; source_l2f, target_nominal d=2,
target_perturbed kf x 1.20 d=3; target PnP bias scale 1; 10 episodes per cell), 3 seeds x 3 sensor levels.

Usage:  python exp28_student_rl_finetune.py [--seeds 3] [--workers 3] [--ft-steps 300000] [--smoke]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

from common import OUT, np, plt, save_json
from exp10_method_matrix import TRAIN_DELAYS, aggregate_seed_dicts, dyn_sampler
from exp21_h3_observer_retest import COURSES, LEVELS, TARGETS

ARMS = ["student", "student_ft"]
METRICS = ["completion", "clean_completion", "gate_frac", "crash_rate"]
LOG_STD_OFFSET = -0.5  # teacher std ~0.30 -> ~0.18: exploration around an already-flying policy


def load_estimator(level, seed, teacher):
    import torch
    from kin_observer import KinGRU, distill_kin
    od, pd = LEVELS[level]
    est = KinGRU()
    path = f"{OUT}/exp21_kin_gru_rand_{level}_s{seed}.pt"
    if os.path.exists(path):
        est.load_state_dict(torch.load(path))
    else:  # weights are not version-controlled: retrain with the identical Exp 21 recipe
        est = distill_kin(teacher, recurrent=True, obs_delay=od, p_drop=pd, seed=100 + seed, dyn_sampler=dyn_sampler,
                          delays=TRAIN_DELAYS)
    est.eval()
    return est


def job(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from asym_policy import AsymActorCriticPolicy, PrivStackRacingEnv
    from racing_env import evaluate_racing
    from student_ft import TEACHER_DIM, CaptureTeacher, StudentFTEnv, init_from_teacher
    from student_plus import PlusStudent, calibrate_bias
    from exp10_method_matrix import SB3PolicyCallable

    torch.set_num_threads(1)
    level, seed, ft_steps, n_eval = args
    od, pd = LEVELS[level]
    t0 = time.time()
    teacher = PPO.load(f"{OUT}/exp10_base_nom_s{seed}.zip", device="cpu")
    est = load_estimator(level, seed, teacher)

    def student(gp=None):
        return PlusStudent(est, CaptureTeacher(teacher), od, gp_hook=gp)

    # ---- fine-tune in randomized L2F at the student's sensor level ----
    def mk(i):
        return lambda: StudentFTEnv("figure8", "source", 8, lam_da=0.20, delays=TRAIN_DELAYS, dyn_sampler=dyn_sampler,
                                    obs_delay=od, p_drop=pd, max_ep_s=5.5, seed=5000 * seed + 37 * i, student=student())
    venv = DummyVecEnv([mk(i) for i in range(8)])
    model = PPO(AsymActorCriticPolicy, venv, n_steps=512, batch_size=512, n_epochs=5, learning_rate=5e-5, gamma=0.99,
                gae_lambda=0.95, clip_range=0.1, target_kl=0.03, ent_coef=0.0, seed=seed, verbose=0, device="cpu",
                policy_kwargs=dict(net_arch=[128, 128], n_priv=TEACHER_DIM, critic_privileged=True))
    init_from_teacher(model, teacher, LOG_STD_OFFSET)
    if ft_steps > 0:
        model.learn(ft_steps)
    model.save(f"{OUT}/exp28_student_ft_{level}_s{seed}.zip")
    t_train = time.time() - t0

    # ---- evaluation (identical envs and seeds for both arms) ----
    def env_kw(tname):
        dom, d, kf, pnp = TARGETS[tname]
        return dict(delays=(d,), kf_scale=kf, obs_delay=od, p_drop=pd, pnp_bias_scale=pnp, random_start_gate=False,
                    max_ep_s=12.5)

    gp, cal = {}, {}
    for tname in ["target_nominal", "target_perturbed"]:  # Exp 23 F3: 3 calibration flights on the training course
        dom = TARGETS[tname][0]
        hook, diag = calibrate_bias(PlusStudent(est, teacher, od), PrivStackRacingEnv("figure8", dom, 8, seed=850 + seed,
                                                                                      **env_kw(tname)),
                                    n_flights=3, seed=seed)
        gp[tname], cal[tname] = hook, diag

    ev = {a: {c: {} for c in COURSES} for a in ARMS}
    for c in COURSES:
        for t in TARGETS:
            dom = TARGETS[t][0]
            g = gp.get(t)
            ev["student"][c][t] = evaluate_racing(PlusStudent(est, teacher, od, gp_hook=g),
                                                  PrivStackRacingEnv(c, dom, 8, seed=800 + seed, **env_kw(t)), n_eval=n_eval)
            ev["student_ft"][c][t] = evaluate_racing(SB3PolicyCallable(model),
                                                     StudentFTEnv(c, dom, 8, seed=800 + seed, student=student(g), **env_kw(t)),
                                                     n_eval=n_eval)
    return level, seed, ev, cal, {"train_s": t_train, "total_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--ft-steps", type=int, default=300_000)
    ap.add_argument("--levels", default="clean,mid,hard")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    levels = args.levels.split(",")
    if args.smoke:
        jobs = [("clean", 0, 4096, 1)]
    else:
        jobs = [(lv, s, args.ft_steps, args.n_eval) for s in range(args.seeds) for lv in levels]
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        outs = pool.map(job, jobs, chunksize=1)
    print(f"Exp 28: {len(jobs)} jobs ({jobs[0][2]} FT steps) in {time.time() - t0:.0f}s")
    levels = [lv for lv in LEVELS if any(o[0] == lv for o in outs)]

    agg = {a: {lv: {c: {t: aggregate_seed_dicts([o[2][a][c][t] for o in outs if o[0] == lv]) for t in TARGETS}
                    for c in COURSES} for lv in levels} for a in ARMS}
    tag = "_smoke" if args.smoke else ""
    save_json(f"exp28_student_rl_finetune{tag}.json", {
        "seeds": sorted({o[1] for o in outs}), "ft_steps": jobs[0][2], "log_std_offset": LOG_STD_OFFSET,
        "n_eval": jobs[0][3], "wall_s": time.time() - t0, "agg": agg,
        "calibration": {f"{o[0]}_s{o[1]}": o[3] for o in outs}, "timing": {f"{o[0]}_s{o[1]}": o[4] for o in outs},
        "per_seed": {f"{o[0]}_s{o[1]}": {a: {c: {t: {m: o[2][a][c][t][m] for m in METRICS} for t in TARGETS}
                                             for c in COURSES} for a in ARMS} for o in outs}})
    for c in COURSES:
        print(f"\n[{c}] completion | clean_completion | gate_frac | crash  (mean over seeds)")
        for lv in levels:
            for a in ARMS:
                row = " || ".join(f"{t}: " + " | ".join(f"{agg[a][lv][c][t][m]['mean']:.2f}" for m in METRICS) for t in TARGETS)
                print(f"  {lv:5s} {a:10s} {row}")

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    x = np.arange(len(levels))
    for k, t in enumerate(TARGETS):
        for j, a in enumerate(ARMS):
            for m, hatch in [("completion", ""), ("clean_completion", "//")]:
                off = (j - 0.5) * 0.38 + (0.0 if m == "completion" else 0.19) - 0.095
                ax[k].bar(x + off, [agg[a][lv]["ellipse3d"][t][m]["mean"] for lv in levels], 0.19,
                          yerr=[agg[a][lv]["ellipse3d"][t][m]["std"] for lv in levels], capsize=2, hatch=hatch,
                          color=f"C{j}", alpha=0.6 if hatch else 1.0, label=f"{a} {m}")
        ax[k].set_xticks(x, levels)
        ax[k].set_title(f"held-out ellipse3d, {t}")
        ax[k].set_xlabel("sensor level")
    ax[0].set_ylabel("completion (solid) / collision-free (hatched)")
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp28_student_rl_finetune{tag}.png", dpi=140)


if __name__ == "__main__":
    main()
