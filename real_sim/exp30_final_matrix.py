"""
Exp 30: the final drone method matrix, one protocol for every method (report table).

Exp 1-29 used slightly different baselines, env seeds and episode counts, and at 3 seeds x 10 episodes
collision-free completion is close to all-or-nothing per seed. This reruns the methods the report compares under one
protocol: 5 seeds, 30 eval episodes per cell, Exp 19 env seeds, Exp 10/19 recipe (450k at lam_da = 0.02, then 100k
at lam_da = 0.20; 550k source steps for every source-trained arm).

Exact-state arms (stage 1, independent):
    nominal         no randomization, d = 0                                           (H1 zero-shot baseline)
    latency_dr      d ~ U{0..4}                                                       (Exp 4b / 19 best variant)
    full_dr         mass + thrust + motor lag + latency DR                            (Exp 10 / 19 dr_ppo)
    hn_thrust_lat   hover-normalized thrust, thrust-only DR + latency                 (Exp 27 best compromise)
    motor_sysid     L2F with its motor model replaced by the target's 30 ms lag       (Exp 26 `only_motor`)
Stage 2 (start from this seed's 450k nominal checkpoint):
    residual_ft     Swift-style: 1 scripted + 3 racing target flights -> (d, tau) scan + body-frame k-NN residual,
                    100k FT in the residual-augmented source, one per target          (Exp 10 residual_ft_ppo)
    oracle          100k FT directly in the target, one per target                    (upper reference)
    student_<lvl>   sensor-limited track-free GRU student on the frozen 450k teacher: multi-track estimator,
                    delay compensation, 3-flight PnP-bias GP per target               (Exp 23 v3_f1f2f3)
    student_f3_<lvl> Exp 21 estimator recipe (figure8 only, no delay compensation) + the same GP; Exp 23's best
                    target variant (v4_f3only), here calibrated by its own flights
Plus classical_expert (no training). Source-cell evaluation of per-target arms uses the target_nominal policy.

Eval: figure8 (training) + held-out ellipse3d x source_l2f (d=0) / target_nominal (PyBullet, d=2) /
target_perturbed (kf x 1.20, d=3), fixed start, 2 laps, 30 episodes. Students at sensor levels clean / mid / hard
with the target PnP bias. Each job caches its result (exp30_job_*.json), so an interrupted run resumes.

Usage:  python exp30_final_matrix.py [--seeds 5] [--workers 12] [--n-eval 30] [--smoke]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import time

from common import OUT, np, save_json
from exp10_method_matrix import HIST, TRAIN_DELAYS, aggregate_seed_dicts
from exp19_dr_factor_ablation import COURSES, TARGETS, PartialDynSampler

STAGE1 = {  # arm: (hover-normalized?, randomized dynamics factors, latency DR?, matched physics factors)
    "nominal": (False, [], False, ()),
    "latency_dr": (False, [], True, ()),
    "full_dr": (False, ["mass", "thrust", "motor_lag"], True, ()),
    "hn_thrust_lat": (True, ["thrust"], True, ()),
    "motor_sysid": (False, [], False, ("motor",)),
}
LEVELS = {"clean": (0, 0.0), "mid": (1, 0.1), "hard": (2, 0.2)}
STAGE2 = ["residual_ft", "oracle"] + [f"student_{lv}" for lv in LEVELS] + [f"student_f3_{lv}" for lv in LEVELS]
CAL = {"target_nominal": (1.0, 2), "target_perturbed": (1.20, 3)}  # per-target arms: (kf scale, action delay)
METRICS = ["gate_frac", "completion", "clean_completion", "crash_rate", "median_lap_s", "track_err_m"]


def _ppo_kw(seed):
    return dict(n_steps=512, batch_size=512, n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.0, seed=seed, verbose=0,
                policy_kwargs=dict(net_arch=[128, 128], log_std_init=-0.7))


def _eval_exact(pol_for, Env, seed, n_eval, **env_kw):
    """pol_for(target name) -> policy; every course x target cell."""
    from racing_env import evaluate_racing

    with contextlib.redirect_stdout(io.StringIO()):
        return {c: {t: evaluate_racing(pol_for(t), Env(c, dom, HIST, delays=(d,), kf_scale=kf, random_start_gate=False,
                                                      max_ep_s=12.5, seed=800 + seed, **env_kw), n_eval=n_eval)
                    for t, (dom, d, kf) in TARGETS.items()} for c in COURSES}


def _cached(arm, seed, fn):
    path = f"{OUT}/exp30_job_{arm}_s{seed}.json"
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    out = fn()
    out["wall_s"] = time.time() - t0
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"  [{arm} s{seed}] done in {out['wall_s']:.0f}s", flush=True)
    return out


def stage1_job(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from exp10_method_matrix import SB3PolicyCallable
    from exp26_gap_attribution import make_variant_env_cls
    from hover_norm import HoverNormRacingEnv
    from racing_env import GateRacingEnv

    torch.set_num_threads(1)
    arm, seed, base_steps, ft_steps, n_eval = args
    hn, keep, lat, match = STAGE1[arm]

    def run():
        Env = HoverNormRacingEnv if hn else (make_variant_env_cls() if match else GateRacingEnv)
        kw = dict(match=match) if match else {}
        sampler = PartialDynSampler(keep) if keep else None
        delays = TRAIN_DELAYS if lat else (0,)

        def venv(lam, s0):
            return DummyVecEnv([lambda i=i: Env("figure8", "source", HIST, lam_da=lam, delays=delays, dyn_sampler=sampler,
                                                max_ep_s=5.5, seed=s0 + i, **kw) for i in range(8)])

        path = f"{OUT}/exp30_{arm}_s{seed}.zip"
        with contextlib.redirect_stdout(io.StringIO()):
            model = PPO("MlpPolicy", venv(0.02, 2000 * seed), **_ppo_kw(seed))
            model.learn(base_steps)
            model.save(f"{OUT}/exp30_{arm}_base_s{seed}.zip")  # stage 2 starts from the nominal one
            model = PPO.load(f"{OUT}/exp30_{arm}_base_s{seed}.zip", env=venv(0.20, 2100 * seed), device="cpu")
            model.learn(ft_steps, reset_num_timesteps=False)
            model.save(path)
        pol = SB3PolicyCallable(model)
        EvalEnv = HoverNormRacingEnv if hn else GateRacingEnv  # all arms are scored in the real L2F / PyBullet cells
        return {"ev": _eval_exact(lambda t: pol, EvalEnv, seed, n_eval)}

    return arm, seed, _cached(arm, seed, run)


def stage2_job(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from exp10_method_matrix import SB3PolicyCallable, calibrate_racing_target, make_knn_residual_hook
    from racing_env import GateRacingEnv

    torch.set_num_threads(1)
    arm, seed, ft_steps, n_eval = args
    base_path = f"{OUT}/exp30_nominal_base_s{seed}.zip"

    def run_per_target(make_venv):
        pols, diag = {}, {}
        for tname, (kf, d) in CAL.items():
            with contextlib.redirect_stdout(io.StringIO()):
                venv, diag[tname] = make_venv(tname, kf, d)
                model = PPO.load(base_path, env=venv, device="cpu")
                model.learn(ft_steps, reset_num_timesteps=False)
                model.save(f"{OUT}/exp30_{arm}_{tname}_s{seed}.zip")
            pols[tname] = SB3PolicyCallable(model)
        pol_for = lambda t: pols["target_perturbed" if t == "target_perturbed" else "target_nominal"]
        return {"ev": _eval_exact(pol_for, GateRacingEnv, seed, n_eval), "cal": diag}

    def residual_venv(tname, kf, d):
        off = 0 if tname == "target_nominal" else 50
        cal = calibrate_racing_target(PPO.load(base_path, device="cpu"), kf_scale=kf, target_delay=d,
                                      pnp_bias_scale=0.0, seed=seed + off)
        venv = DummyVecEnv([lambda i=i: GateRacingEnv("figure8", "source", HIST, lam_da=0.20, delays=(cal["d_hat"],),
                                                      residual_fn=make_knn_residual_hook(cal["knn"]),
                                                      tau_hat=cal["tau_hat"], max_ep_s=5.5, seed=3000 * seed + off + i)
                            for i in range(8)])
        return venv, {"d_hat": cal["d_hat"], "tau_hat": cal["tau_hat"]}

    def oracle_venv(tname, kf, d):
        off = 0 if tname == "target_nominal" else 100
        return DummyVecEnv([lambda i=i: GateRacingEnv("figure8", "target", HIST, lam_da=0.20, delays=(d,), kf_scale=kf,
                                                      max_ep_s=5.5, seed=4000 * seed + off + i) for i in range(4)]), {}

    if arm == "residual_ft":
        return arm, seed, _cached(arm, seed, lambda: run_per_target(residual_venv))
    if arm == "oracle":
        return arm, seed, _cached(arm, seed, lambda: run_per_target(oracle_venv))
    f3 = arm.startswith("student_f3_")
    level = arm.split("_")[-1]
    return arm, seed, _cached(arm, seed, lambda: _student(level, seed, n_eval, base_path, f3_only=f3))


def _student(level, seed, n_eval, base_path, f3_only=False):
    from stable_baselines3 import PPO
    from asym_policy import PrivStackRacingEnv
    from exp10_method_matrix import dyn_sampler
    from exp21_h3_observer_retest import TARGETS as S_TARGETS
    from kin_observer import distill_kin
    from racing_env import evaluate_racing
    from student_plus import PlusStudent, calibrate_bias, track_pool

    od, pd = LEVELS[level]
    teacher = PPO.load(base_path, device="cpu")
    pool_kw = dict(eps=10) if f3_only else dict(eps=16, courses=track_pool())
    est = distill_kin(teacher, recurrent=True, obs_delay=od, p_drop=pd, seed=100 + seed, dyn_sampler=dyn_sampler,
                      delays=TRAIN_DELAYS, iters=6, **pool_kw)
    dc = not f3_only

    def env(course, tname, s_off=0):
        dom, d, kf, pnp = S_TARGETS[tname]
        return PrivStackRacingEnv(course, dom, HIST, delays=(d,), kf_scale=kf, obs_delay=od, p_drop=pd,
                                  pnp_bias_scale=pnp, random_start_gate=False, max_ep_s=12.5, seed=800 + seed + s_off)

    gp, cal = {}, {}
    for tname in CAL:
        gp[tname], cal[tname] = calibrate_bias(PlusStudent(est, teacher, od, delay_comp=dc), env("figure8", tname, 50),
                                               n_flights=3, seed=seed)
    ev = {c: {t: evaluate_racing(PlusStudent(est, teacher, od, delay_comp=dc, gp_hook=gp.get(t)), env(c, t),
                                 n_eval=n_eval) for t in S_TARGETS} for c in COURSES}
    return {"ev": ev, "cal": cal}


def classical_ev(seed, n_eval):
    from racing_env import GateRacingEnv, GeometricGateController

    return _cached("classical_expert", seed, lambda: {"ev": _eval_exact(lambda t: GeometricGateController(),
                                                                         GateRacingEnv, seed, n_eval)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--base-steps", type=int, default=450_000)
    ap.add_argument("--ft-steps", type=int, default=100_000)
    ap.add_argument("--smoke", action="store_true", help="tiny budgets, seed 99, to check the pipeline end to end")
    args = ap.parse_args()
    seeds = [99] if args.smoke else list(range(args.seeds))
    base, ft, n_eval = (2048, 1024, 1) if args.smoke else (args.base_steps, args.ft_steps, args.n_eval)

    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        outs = pool.map(stage1_job, [(a, s, base, ft, n_eval) for s in seeds for a in STAGE1], chunksize=1)
        print(f"Exp 30 stage 1: {len(outs)} jobs in {time.time() - t0:.0f}s", flush=True)
        outs += pool.map(stage2_job, [(a, s, ft, n_eval) for s in seeds for a in STAGE2], chunksize=1)
    outs += [("classical_expert", s, classical_ev(s, n_eval)) for s in seeds]
    print(f"Exp 30: {len(outs)} jobs in {time.time() - t0:.0f}s")
    if args.smoke:
        for f in os.listdir(OUT):
            if f.startswith("exp30_") and "_s99" in f:
                os.remove(os.path.join(OUT, f))
        return

    arms = list(STAGE1) + STAGE2 + ["classical_expert"]
    agg = {a: {c: {t: aggregate_seed_dicts([o[2]["ev"][c][t] for o in outs if o[0] == a]) for t in TARGETS}
               for c in COURSES} for a in arms}
    cal = {f"{o[0]}_s{o[1]}": o[2]["cal"] for o in outs if "cal" in o[2]}
    save_json("exp30_final_matrix.json", {"seeds": seeds, "n_eval": n_eval, "base_steps": base, "ft_steps": ft,
                                          "wall_s": time.time() - t0, "calibration": cal, "agg": agg})
    for c in COURSES:
        print(f"\n[{c}] gate_frac / clean completion / crash  (mean over {len(seeds)} seeds x {n_eval} episodes)")
        for a in arms:
            row = " || ".join(f"{t}: {agg[a][c][t]['gate_frac']['mean']:.2f} / {agg[a][c][t]['clean_completion']['mean']:.2f}"
                              f" / {agg[a][c][t]['crash_rate']['mean']:.2f}" for t in TARGETS)
            print(f"  {a:16s} {row}")


if __name__ == "__main__":
    main()
