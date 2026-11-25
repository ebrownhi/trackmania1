"""
Complete Multi-Seed Parallel Empirical Verification Suite for 16-831 Final Project Proposal:
"One Method, Two Racing Worlds: Sim-to-Sim Transfer in Drone and Trackmania Racing"

Generates:
  - results.json
  - exp1_divergence.png
  - exp2_randomization_delay.png
  - exp3_recurrent_ood.png
  - exp4_knn_residual.png
"""

import os
import json
import time
import argparse
import multiprocessing as mp
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs import DroneSlalomEnv, TrackmaniaCircuitEnv
from models_and_learners import (
    KNNResidual, GPObsResidual,
    train_drone_policy, eval_drone_policy,
    train_tm_policy, eval_tm_policy,
    distill_drone_student, distill_tm_student,
    make_drone_knn_features,
)


def agg(vals):
    arr = np.array(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "sem": float(np.std(arr) / max(1.0, np.sqrt(len(arr)))),
        "values": [float(v) for v in arr],
    }


@torch.no_grad()
def collect_drone_residual_data(env, policy, seed: int, n_eps: int = 6, delay: int = 1):
    gen = torch.Generator().manual_seed(seed + 8000)
    p_tgt = env.default_params(n_eps, alpha=1.0, randomize=False)
    p_src = env.default_params(n_eps, alpha=0.0, randomize=False)
    s = env.reset(n_eps, random_checkpoints=False, perturb_init=True, generator=gen)
    act_buf = torch.zeros((n_eps, 6, 2))

    Z_markov_list, Z_hist_list, Y_acc_list = [], [], []
    X_gp_list, Y_gp_list = [], []

    for t in range(330):
        obs = env.get_obs(s, sensor_limited=False)
        a_cmd = policy(obs, act_buf[:, : policy.act_hist_len, :])
        if n_eps >= 3:
            noise = 0.26 * torch.randn(a_cmd.shape, generator=gen)
            noise[0] = 0.0
            a_cmd = torch.clamp(a_cmd + noise, -1.0, 1.0)
        else:
            noise = 0.22 * torch.randn(a_cmd.shape, generator=gen)
            a_cmd = torch.clamp(a_cmd + noise, -1.0, 1.0)

        new_buf = torch.cat([a_cmd.unsqueeze(1), act_buf[:, :-1, :]], dim=1)
        a_app = new_buf[:, delay, :]
        a_lag = 0.55 * new_buf[:, delay + 1, :] + 0.30 * new_buf[:, delay + 2, :] + 0.15 * new_buf[:, delay + 3, :]

        s_tgt_next = env.step_physics(s, a_app, p_tgt)
        s_src_next = env.step_physics(s, a_app, p_src)

        dv_tgt = (s_tgt_next[:, [3, 4, 5, 7]] - s[:, [3, 4, 5, 7]]) / env.dt
        dv_src = (s_src_next[:, [3, 4, 5, 7]] - s[:, [3, 4, 5, 7]]) / env.dt
        y_res = dv_tgt - dv_src

        zm = make_drone_knn_features(s, a_app, a_lag, use_history=False)
        zh = make_drone_knn_features(s, a_app, a_lag, use_history=True)
        Z_markov_list.append(zm)
        Z_hist_list.append(zh)
        Y_acc_list.append(y_res)

        x_pos = s[:, 0]
        phi = s[:, 6]
        ahead = env.gate_x.unsqueeze(0) > (x_pos.unsqueeze(1) - 0.12)
        idx = torch.clamp(5 - ahead.long().sum(dim=1), 0, 4)
        dx = env.gate_x[idx] - x_pos
        bias_y = 0.065 * dx * torch.sin(phi) + 0.025 * dx + 0.015 * torch.randn(dx.shape, generator=gen)
        bias_z = 0.045 * dx * (1.0 - torch.cos(phi)) + 0.032 * dx + 0.015 * torch.randn(dx.shape, generator=gen)
        X_gp_list.append(torch.stack([dx, phi], dim=-1))
        Y_gp_list.append(torch.stack([bias_y, bias_z], dim=-1))

        s = s_tgt_next
        act_buf = new_buf

    return (
        torch.cat(Z_markov_list, dim=0),
        torch.cat(Z_hist_list, dim=0),
        torch.cat(Y_acc_list, dim=0),
        torch.cat(X_gp_list, dim=0),
        torch.cat(Y_gp_list, dim=0),
    )


@torch.no_grad()
def collect_tm_residual_data(env, policy, seed: int, n_eps: int = 6, true_delay: int = 2, act_noise: float = 0.01):
    gen = torch.Generator().manual_seed(seed + 9000)
    params = env.default_params(n_eps, randomize_dynamics=False)
    s = env.reset(n_eps, random_checkpoints=False, perturb_init=True, generator=gen)
    act_buf = torch.zeros((n_eps, 7, 2))
    scale_vec = torch.tensor([[30.0, 3.0, 2.0, 0.35]])

    S_list, Buf_list, Acc_meas_list = [], [], []
    for t in range(420):
        obs = env.get_obs(s, sensor_limited=False)
        a_cmd = policy(obs, act_buf[:, : policy.act_hist_len, :])
        new_buf = torch.cat([a_cmd.unsqueeze(1), act_buf[:, :-1, :]], dim=1)
        a_exec = torch.clamp(
            new_buf[:, true_delay, :] + act_noise * torch.randn((n_eps, 2), generator=gen), -1.0, 1.0
        )
        dvx_m, dvy_m, dr_m, _ = env.compute_body_accels(s, a_exec, params)
        acc_meas = torch.stack([dvx_m, dvy_m, dr_m], dim=-1)
        s_next = env.step_physics(s, a_exec, params)

        if t >= 5:
            S_list.append(s.clone())
            Buf_list.append(new_buf.clone())
            Acc_meas_list.append(acc_meas)

        s = s_next
        act_buf = new_buf

    S_all = torch.cat(S_list, dim=0)
    Buf_all = torch.cat(Buf_list, dim=0)
    Acc_meas = torch.cat(Acc_meas_list, dim=0)
    p_all = env.default_params(S_all.shape[0], randomize_dynamics=False)

    delay_scan_rms = {}
    res_by_d = {}
    feat_by_d = {}
    for d_cand in range(5):
        a_cand = Buf_all[:, d_cand, :]
        dvx_s, dvy_s, dr_s, _ = env.compute_body_accels(S_all, a_cand, p_all)
        acc_src = torch.stack([dvx_s, dvy_s, dr_s], dim=-1)
        res = Acc_meas - acc_src
        rms = torch.sqrt((res ** 2).mean()).item()
        delay_scan_rms[d_cand] = rms
        res_by_d[d_cand] = res
        feat_by_d[d_cand] = torch.cat([S_all[:, 3:7] / scale_vec, a_cand], dim=-1)

    return delay_scan_rms, feat_by_d, res_by_d


def run_single_seed(s: int):
    torch.set_num_threads(1)
    ts = time.time()
    d_env = DroneSlalomEnv()
    tm_env = TrackmaniaCircuitEnv()

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    delays = [0, 1, 2, 3, 4]

    d_nom = train_drone_policy(d_env, seed=s, act_hist_len=1, iters=52)
    d_dr = train_drone_policy(d_env, seed=s, act_hist_len=1, dyn_rand=True, iters=52)
    d_lr1 = train_drone_policy(d_env, seed=s, act_hist_len=1, dyn_rand=True, lat_rand=True, iters=56)
    d_lr4 = train_drone_policy(d_env, seed=s, act_hist_len=4, dyn_rand=True, lat_rand=True, iters=56)
    d_orc = train_drone_policy(d_env, seed=s, act_hist_len=4, target_oracle=True, fixed_delay=1, iters=56)

    tm_nom = train_tm_policy(tm_env, seed=s, act_hist_len=1, iters=52)
    tm_dr = train_tm_policy(tm_env, seed=s, act_hist_len=1, dyn_rand=True, iters=52)
    tm_lr1 = train_tm_policy(tm_env, seed=s, act_hist_len=1, dyn_rand=False, lat_rand=True, iters=56)
    tm_lr4 = train_tm_policy(tm_env, seed=s, act_hist_len=4, dyn_rand=False, lat_rand=True, iters=56)
    tm_orc = train_tm_policy(tm_env, seed=s, act_hist_len=4, fixed_delay=2, iters=56)

    T_d, T_tm = 340, 480
    with torch.no_grad():
        p_src = d_env.default_params(1, alpha=0.0, randomize=False)
        p_tgt = d_env.default_params(1, alpha=1.0, randomize=False)
        s_src = d_env.reset(1, random_checkpoints=False, perturb_init=False)
        act_buf = torch.zeros((1, 6, 2))
        d_tape, pos_src = [], []
        for t in range(T_d):
            pos_src.append(s_src[0, :3].clone().numpy())
            obs = d_env.get_obs(s_src, sensor_limited=False)
            a = d_nom(obs, act_buf[:, :1, :])
            d_tape.append(a.clone())
            act_buf = torch.cat([a.unsqueeze(1), act_buf[:, :-1, :]], dim=1)
            s_src = d_env.step_physics(s_src, a, p_src)
        pos_src = np.array(pos_src)

        drone_divs, drone_trajs = {}, {"src": pos_src}
        for d_val in [0, 2]:
            s_t = d_env.reset(1, random_checkpoints=False, perturb_init=False)
            q = [torch.zeros((1, 2)) for _ in range(d_val)]
            pos_t = []
            for t in range(T_d):
                pos_t.append(s_t[0, :3].clone().numpy())
                q.insert(0, d_tape[t])
                a_app = q.pop()
                s_t = d_env.step_physics(s_t, a_app, p_tgt)
            pos_t = np.array(pos_t)
            drone_divs[d_val] = np.linalg.norm(pos_t - pos_src, axis=-1)
            drone_trajs[f"tgt_d{d_val}"] = pos_t

        tm_p = tm_env.default_params(1, randomize_dynamics=False)
        s_src = tm_env.reset(1, random_checkpoints=False, perturb_init=False)
        act_buf = torch.zeros((1, 6, 2))
        tm_tape, xy_src = [], []
        for t in range(T_tm):
            xy_src.append(s_src[0, 7:9].clone().numpy())
            obs = tm_env.get_obs(s_src, sensor_limited=False)
            a = tm_nom(obs, act_buf[:, :1, :])
            tm_tape.append(a.clone())
            act_buf = torch.cat([a.unsqueeze(1), act_buf[:, :-1, :]], dim=1)
            s_src = tm_env.step_physics(s_src, a, tm_p)
        xy_src = np.array(xy_src)

        tm_divs, tm_trajs = {}, {"src": xy_src}
        for d_val in [0, 1, 2, 3]:
            s_t = tm_env.reset(1, random_checkpoints=False, perturb_init=False)
            q = [torch.zeros((1, 2)) for _ in range(d_val)]
            xy_t = []
            for t in range(T_tm):
                xy_t.append(s_t[0, 7:9].clone().numpy())
                q.insert(0, tm_tape[t])
                a_app = q.pop()
                s_t = tm_env.step_physics(s_t, a_app, tm_p)
            xy_t = np.array(xy_t)
            tm_divs[d_val] = np.linalg.norm(xy_t - xy_src, axis=-1)
            tm_trajs[f"tgt_d{d_val}"] = xy_t

    d_alpha_res = {a: eval_drone_policy(d_env, d_nom, alpha=a, delay=0, seed=100 + s)["score"] for a in alphas}
    tm_alpha_res = {a: eval_tm_policy(tm_env, tm_nom, delay=0, act_noise=0.0, seed=100 + s)["score"] for a in alphas}

    d_del_sc, d_del_gp = {}, {}
    tm_del_sc, tm_del_cl, tm_t_d0 = {}, {}, {}
    for name, pol in [("Nominal", d_nom), ("DynRand", d_dr), ("LatRand_a1", d_lr1), ("LatRand_buf4", d_lr4)]:
        d_del_sc[name], d_del_gp[name] = {}, {}
        for d in delays:
            ev = eval_drone_policy(d_env, pol, alpha=1.0, delay=d, seed=200 + s)
            d_del_sc[name][d] = ev["score"]
            d_del_gp[name][d] = ev["gate_pass_rate"]

    for name, pol in [("Nominal", tm_nom), ("DynRand", tm_dr), ("LatRand_a1", tm_lr1), ("LatRand_buf4", tm_lr4)]:
        tm_del_sc[name], tm_del_cl[name] = {}, {}
        for d in delays:
            ev = eval_tm_policy(tm_env, pol, delay=d, act_noise=0.015, seed=300 + s)
            tm_del_sc[name][d] = ev["score"]
            tm_del_cl[name][d] = ev["clean_completion"]
            if d == 0:
                tm_t_d0[name] = ev["finish_time_s"]

    d_ff = distill_drone_student(d_env, d_nom, seed=s, recurrent=False, randomize_source=False)
    d_gru_fix = distill_drone_student(d_env, d_nom, seed=s, recurrent=True, randomize_source=False)
    d_gru_rnd = distill_drone_student(d_env, d_lr4, seed=s, recurrent=True, randomize_source=True)

    e3_d = {
        "Teacher": {
            "src": eval_drone_policy(d_env, d_nom, alpha=0.0, delay=0, seed=400 + s)["score"],
            "tgt": eval_drone_policy(d_env, d_lr4, alpha=1.0, delay=1, seed=400 + s)["score"],
        }
    }
    for nm, st, is_g in [("FF_Student", d_ff, False), ("GRU_FixedSrc", d_gru_fix, True), ("GRU_RandSrc", d_gru_rnd, True)]:
        es = eval_drone_policy(d_env, st, alpha=0.0, delay=0, sensor_limited=True, pnp_bias=0.0, obs_noise=0.0, is_gru=is_g, seed=400 + s)
        et = eval_drone_policy(d_env, st, alpha=1.0, delay=1, sensor_limited=True, pnp_bias=1.0, obs_noise=0.02, is_gru=is_g, seed=400 + s)
        e3_d[nm] = {"src": es["score"], "tgt": et["score"], "probe_src": es["vel_probe_rmse"], "probe_tgt": et["vel_probe_rmse"]}

    tm_ff = distill_tm_student(tm_env, tm_nom, seed=s, recurrent=False, randomize_interface=False)
    tm_gru_fix = distill_tm_student(tm_env, tm_nom, seed=s, recurrent=True, randomize_interface=False)
    tm_gru_rnd = distill_tm_student(tm_env, tm_lr4, seed=s, recurrent=True, randomize_interface=True)

    e3_tm = {
        "Teacher": {
            "src": eval_tm_policy(tm_env, tm_nom, delay=0, seed=500 + s)["score"],
            "tgt": eval_tm_policy(tm_env, tm_lr4, delay=2, act_noise=0.02, seed=500 + s)["score"],
        }
    }
    for nm, st, is_g in [("FF_Student", tm_ff, False), ("GRU_FixedSrc", tm_gru_fix, True), ("GRU_RandSrc", tm_gru_rnd, True)]:
        es = eval_tm_policy(tm_env, st, delay=0, sensor_limited=True, cam_bias=0.0, obs_noise=0.0, obs_dropout=0.0, is_gru=is_g, seed=500 + s)
        et = eval_tm_policy(tm_env, st, delay=2, act_noise=0.02, sensor_limited=True, cam_bias=1.0, obs_noise=0.025, obs_dropout=0.12, is_gru=is_g, seed=500 + s)
        e3_tm[nm] = {"src": es["score"], "tgt": et["score"], "probe_src": es["vel_probe_rmse"], "probe_tgt": et["vel_probe_rmse"]}

    scan_rms, feat_d, res_d = collect_tm_residual_data(tm_env, tm_nom, seed=s, n_eps=6, true_delay=2, act_noise=0.01)
    d_hat = int(min(scan_rms, key=scan_rms.get))
    N_tm = feat_d[0].shape[0]
    split_tm = int(0.8 * N_tm)

    knn_tm_naive = KNNResidual(k=10, bandwidth=0.80)
    knn_tm_naive.fit(feat_d[0][:split_tm], res_d[0][:split_tm])
    rms_naive = torch.sqrt((knn_tm_naive.predict(feat_d[0][split_tm:]) ** 2).mean()).item()

    knn_tm_align = KNNResidual(k=10, bandwidth=0.80)
    knn_tm_align.fit(feat_d[d_hat][:split_tm], res_d[d_hat][:split_tm])
    rms_align = torch.sqrt((knn_tm_align.predict(feat_d[d_hat][split_tm:]) ** 2).mean()).item()

    tm_ft_naive = train_tm_policy(
        tm_env, seed=s + 70, act_hist_len=1, fixed_delay=0,
        knn_res=knn_tm_naive, knn_aligned=False, init_policy=tm_nom, iters=26
    )
    tm_base_b4 = train_tm_policy(tm_env, seed=s + 79, act_hist_len=4, fixed_delay=0, iters=26)
    tm_ft_align = train_tm_policy(
        tm_env, seed=s + 80, act_hist_len=4, fixed_delay=d_hat,
        knn_res=knn_tm_align, knn_aligned=True, init_policy=tm_base_b4, iters=26
    )
    e4_tm_sc = {
        "Nominal_ZeroShot": eval_tm_policy(tm_env, tm_nom, delay=2, act_noise=0.015, seed=600 + s)["score"],
        "Naive_KNN_FT": eval_tm_policy(tm_env, tm_ft_naive, delay=2, act_noise=0.015, seed=600 + s)["score"],
        "Aligned_KNN_FT": eval_tm_policy(tm_env, tm_ft_align, delay=2, act_noise=0.015, seed=600 + s)["score"],
        "Oracle_Ref": eval_tm_policy(tm_env, tm_orc, delay=2, act_noise=0.015, seed=600 + s)["score"],
    }

    Zm, Zh, Y_acc, X_gp, Y_gp = collect_drone_residual_data(d_env, d_nom, seed=s, n_eps=6, delay=1)
    Zm_val, Zh_val, Y_val, _, _ = collect_drone_residual_data(d_env, d_lr4, seed=s + 50, n_eps=2, delay=1)

    knn_d_markov = KNNResidual(k=10, bandwidth=0.80)
    knn_d_markov.fit(Zm, Y_acc)
    knn_d_hist = KNNResidual(k=10, bandwidth=0.80)
    knn_d_hist.fit(Zh, Y_acc)

    e4_d_rmse = {
        "Zero_Res": torch.sqrt((Y_val[:, :3] ** 2).mean()).item(),
        "Markov_KNN": torch.sqrt(((knn_d_markov.predict(Zm_val)[:, :3] - Y_val[:, :3]) ** 2).mean()).item(),
        "History_KNN": torch.sqrt(((knn_d_hist.predict(Zh_val)[:, :3] - Y_val[:, :3]) ** 2).mean()).item(),
    }

    d_base_short = train_drone_policy(d_env, seed=s + 90, act_hist_len=4, fixed_delay=1, iters=26)
    d_ft_markov = train_drone_policy(
        d_env, seed=s + 91, act_hist_len=4, fixed_delay=1,
        knn_res=knn_d_markov, knn_use_hist=False, init_policy=d_base_short, iters=26
    )
    d_ft_hist = train_drone_policy(
        d_env, seed=s + 92, act_hist_len=4, fixed_delay=1,
        knn_res=knn_d_hist, knn_use_hist=True, init_policy=d_base_short, iters=26
    )
    d_dr_eq = train_drone_policy(d_env, seed=s + 93, act_hist_len=4, dyn_rand=True, lat_rand=True, iters=52)

    e4_d_sc = {
        "Nominal_ZeroShot": eval_drone_policy(d_env, d_nom, alpha=1.0, delay=1, seed=700 + s)["score"],
        "Domain_Rand": eval_drone_policy(d_env, d_dr_eq, alpha=1.0, delay=1, seed=700 + s)["score"],
        "ResFT_Markov_KNN": eval_drone_policy(d_env, d_ft_markov, alpha=1.0, delay=1, seed=700 + s)["score"],
        "ResFT_History_KNN": eval_drone_policy(d_env, d_ft_hist, alpha=1.0, delay=1, seed=700 + s)["score"],
        "Target_Oracle": eval_drone_policy(d_env, d_orc, alpha=1.0, delay=1, seed=700 + s)["score"],
    }

    gp_obs = GPObsResidual(lengthscale=1.1)
    gp_obs.fit(X_gp, Y_gp)
    e4_gp = {
        "No_GP": eval_drone_policy(
            d_env, d_gru_rnd, alpha=1.0, delay=1, sensor_limited=True, pnp_bias=1.25, obs_noise=0.02, gp_obs_corr=None, is_gru=True, seed=750 + s
        )["score"],
        "With_GP": eval_drone_policy(
            d_env, d_gru_rnd, alpha=1.0, delay=1, sensor_limited=True, pnp_bias=1.25, obs_noise=0.02, gp_obs_corr=gp_obs, is_gru=True, seed=750 + s
        )["score"],
    }

    print(f"  [Worker Seed {s}] finished in {time.time() - ts:.2f}s", flush=True)
    return {
        "seed": s,
        "drone_divs": drone_divs,
        "tm_divs": tm_divs,
        "drone_trajs": drone_trajs if s == 0 else None,
        "tm_trajs": tm_trajs if s == 0 else None,
        "d_alpha_res": d_alpha_res,
        "tm_alpha_res": tm_alpha_res,
        "d_del_sc": d_del_sc,
        "d_del_gp": d_del_gp,
        "tm_del_sc": tm_del_sc,
        "tm_del_cl": tm_del_cl,
        "tm_t_d0": tm_t_d0,
        "e3_d": e3_d,
        "e3_tm": e3_tm,
        "scan_rms": scan_rms,
        "rms_naive": rms_naive,
        "rms_align": rms_align,
        "e4_tm_sc": e4_tm_sc,
        "e4_d_rmse": e4_d_rmse,
        "e4_d_sc": e4_d_sc,
        "e4_gp": e4_gp,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--outdir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t_start = time.time()
    seeds = list(range(args.seeds))

    print(f"=== Running 16-831 Parallel Verification Suite across {len(seeds)} seeds ===")
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(5, len(seeds))) as pool:
        seed_outs = pool.map(run_single_seed, seeds)

    seed_outs.sort(key=lambda x: x["seed"])
    d_env = DroneSlalomEnv()

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    delays = [0, 1, 2, 3, 4]

    d_d0 = np.array([o["drone_divs"][0] for o in seed_outs])
    d_d2 = np.array([o["drone_divs"][2] for o in seed_outs])
    t_d0 = np.array([o["tm_divs"][0] for o in seed_outs])
    t_d1 = np.array([o["tm_divs"][1] for o in seed_outs])
    t_d2 = np.array([o["tm_divs"][2] for o in seed_outs])
    t_d3 = np.array([o["tm_divs"][3] for o in seed_outs])

    exp1_summary = {
        "drone_1s_div_d0_m": agg(d_d0[:, 100]),
        "drone_2s_div_d0_m": agg(d_d0[:, 200]),
        "drone_3s_div_d0_m": agg(d_d0[:, 300]),
        "drone_3s_div_d2_m": agg(d_d2[:, 300]),
        "tm_3s_div_d0_m": agg(t_d0[:, 300]),
        "tm_45s_div_d0_m": agg(t_d0[:, 450]),
        "tm_3s_div_d1_m": agg(t_d1[:, 300]),
        "tm_3s_div_d2_m": agg(t_d2[:, 300]),
        "tm_45s_div_d2_m": agg(t_d2[:, 450]),
        "tm_45s_div_d3_m": agg(t_d3[:, 450]),
    }

    methods2 = ["Nominal", "DynRand", "LatRand_a1", "LatRand_buf4"]
    methods3 = ["Teacher", "FF_Student", "GRU_FixedSrc", "GRU_RandSrc"]

    results = {
        "meta": {
            "num_seeds": len(seeds),
            "wall_time_sec": round(time.time() - t_start, 2),
        },
        "exp1_open_loop_divergence": exp1_summary,
        "exp2_h1_h2": {
            "drone_alpha_sweep": {str(a): agg([o["d_alpha_res"][a] for o in seed_outs]) for a in alphas},
            "tm_alpha_sweep": {str(a): agg([o["tm_alpha_res"][a] for o in seed_outs]) for a in alphas},
            "drone_delay_score": {m: {str(d): agg([o["d_del_sc"][m][d] for o in seed_outs]) for d in delays} for m in methods2},
            "drone_delay_gate_pass": {m: {str(d): agg([o["d_del_gp"][m][d] for o in seed_outs]) for d in delays} for m in methods2},
            "tm_delay_score": {m: {str(d): agg([o["tm_del_sc"][m][d] for o in seed_outs]) for d in delays} for m in methods2},
            "tm_delay_clean_rate": {m: {str(d): agg([o["tm_del_cl"][m][d] for o in seed_outs]) for d in delays} for m in methods2},
            "tm_d0_finish_time_s": {m: agg([o["tm_t_d0"][m] for o in seed_outs]) for m in methods2},
        },
        "exp3_recurrent_ood": {
            "drone": {
                m: {k: agg([o["e3_d"][m][k] for o in seed_outs]) for k in seed_outs[0]["e3_d"][m].keys()}
                for m in methods3
            },
            "trackmania": {
                m: {k: agg([o["e3_tm"][m][k] for o in seed_outs]) for k in seed_outs[0]["e3_tm"][m].keys()}
                for m in methods3
            },
        },
        "exp4_knn_residual": {
            "tm_delay_sysid_rms": {str(d): agg([o["scan_rms"][d] for o in seed_outs]) for d in range(5)},
            "tm_fitted_knn_rms": {
                "Naive_d0": agg([o["rms_naive"] for o in seed_outs]),
                "Aligned_dhat": agg([o["rms_align"] for o in seed_outs]),
            },
            "tm_transfer_scores": {
                k: agg([o["e4_tm_sc"][k] for o in seed_outs])
                for k in ["Nominal_ZeroShot", "Naive_KNN_FT", "Aligned_KNN_FT", "Oracle_Ref"]
            },
            "drone_accel_pred_rmse": {
                k: agg([o["e4_d_rmse"][k] for o in seed_outs])
                for k in ["Zero_Res", "Markov_KNN", "History_KNN"]
            },
            "drone_transfer_scores": {
                k: agg([o["e4_d_sc"][k] for o in seed_outs])
                for k in ["Nominal_ZeroShot", "Domain_Rand", "ResFT_Markov_KNN", "ResFT_History_KNN", "Target_Oracle"]
            },
            "drone_gp_obs_scores": {
                k: agg([o["e4_gp"][k] for o in seed_outs]) for k in ["No_GP", "With_GP"]
            },
        },
    }

    json_path = os.path.join(args.outdir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== ALL {len(seeds)} SEEDS COMPLETED IN {time.time() - t_start:.2f}s ===")
    print(f"Saved results to: {json_path}")


if __name__ == "__main__":
    main()
