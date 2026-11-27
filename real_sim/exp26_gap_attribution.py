"""
Exp 26 (idea 2 of docs/FOLLOWUP_RIFFS_ROUND2.md): per-component attribution of the L2F -> PyBullet gap.

The target is OUR PyBullet wrapper (CF2X, PYB_GND_DRAG_DW, 500 Hz, plus a 30 ms first-order motor lag added by
PyBulletBackend), not stock gym-pybullet-drones. Six physical mismatches can be removed from the L2F source exactly,
through its parameter JSON plus a drag hook (L2FBackendPB below):

    mass       0.0306 -> 0.027 kg      (the CTBR loop still assumes 0.0306, as it does when deployed in the target)
    inertia    J = (9, 10, 17) e-6 -> (14, 14, 21.7) e-6 kg m^2   (the rate loop still assumes the L2F J)
    thrust     L2F quadratic curve, 0.110 N/motor ceiling (T/W 1.47) -> kf (r * MAX_RPM)^2, 0.149 N (T/W 2.25)
    motor      rotor lag 55 ms rise / 249 ms fall -> 30 ms / 30 ms
    yaw        rotor torque constant 0.004665 -> km/kf = 0.0251 N m / N (5.4x)
    drag       none -> PyBullet's Forster drag  F_B = R^T(-k (.) sum(omega) (.) v_world), k = (9.18, 9.18, 10.3) e-7
Anything left after matching all six (integrator, PyBullet rigid-body details, ground effect/downwash, which are
negligible at 1.2 m with one drone) is the residual.

Part A (training-free): k-step open-loop prediction error. Fly the Exp 10 nominal PPO teachers (+ exploration noise)
in the target on figure8 and held-out ellipse3d, log (state, rotor rpm, applied action). From many start points,
reset every source variant to the logged state + equivalent rotor state and replay the logged actions; error at
10 ticks (0.1 s) and 50 ticks (0.5 s). Exact Shapley attribution over all 2^6 = 64 subsets of matched factors,
bootstrap CIs over start points.

Part B (retrained): Exp 10 nominal PPO recipe (450k @ lam 0.02 + 100k @ lam 0.20) in selected source variants,
evaluated in target_nominal (d = 2, as in Exp 10/19) on figure8 + held-out ellipse3d. Action delay d = 2 is the one
interface mismatch of target_nominal, so it is a seventh factor here.

Usage:
    python exp26_gap_attribution.py partA [--workers 4]
    python exp26_gap_attribution.py partB [--workers 4] [--seeds 3] [--variants a,b,...]
    python exp26_gap_attribution.py plotB      # figure from the saved Part B JSON (+ Exp 19 `none` as plain L2F)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import math
import multiprocessing as mp
import sys
import time

from common import OUT, fmt, mean_std, np, plt, save_json
from sims.drone_pair import (GRAVITY, SUB_DT, SUBSTEPS, CTBRDrone, DroneState, InterfaceConfig, L2FBackend,
                             make_target)

FACTORS = ["mass", "inertia", "thrust", "motor", "yaw", "drag"]
# PyBullet CF2X constants (gym-pybullet-drones cf2x.urdf, printed by BaseAviary)
PB = dict(M=0.027, J=(1.4e-5, 1.4e-5, 2.17e-5), KF=3.16e-10, KM=7.94e-12, MAX_RPM=21702.64377525105,
          DRAG=np.array([9.1785e-7, 9.1785e-7, 1.0311e-6]), TAU=0.03)
PB["F_MAX"] = PB["KF"] * PB["MAX_RPM"] ** 2


# --------------------------------------------------------------------------------------------------
# PyBullet-like L2F source: each factor in `match` set to the PyBullet value
# --------------------------------------------------------------------------------------------------


class L2FBackendPB(L2FBackend):
    def __init__(self, seed: int = 0, match=()):
        self.match = frozenset(match)
        unknown = self.match - set(FACTORS)
        assert not unknown, unknown
        super().__init__(seed)
        P = json.loads(json.dumps(self.base_params))
        d = P["dynamics"]
        if "mass" in self.match:
            d["mass"] = PB["M"]
        if "inertia" in self.match:
            d["J"] = [[PB["J"][0], 0.0, 0.0], [0.0, PB["J"][1], 0.0], [0.0, 0.0, PB["J"][2]]]
            d["J_inv"] = [[1.0 / PB["J"][0], 0.0, 0.0], [0.0, 1.0 / PB["J"][1], 0.0], [0.0, 0.0, 1.0 / PB["J"][2]]]
        if "thrust" in self.match:
            d["rotor_thrust_coefficients"] = [[0.0, 0.0, PB["F_MAX"]] for _ in range(4)]
        if "motor" in self.match:
            d["rotor_time_constants_rising"] = [PB["TAU"]] * 4
            d["rotor_time_constants_falling"] = [PB["TAU"]] * 4
        if "yaw" in self.match:
            d["rotor_torque_constants"] = [PB["KM"] / PB["KF"]] * 4
        self.base_params = P
        self.set_dynamics({})
        self.l2f_mass = d["mass"]

    def rotor_thrust(self, r: np.ndarray) -> np.ndarray:
        c0, c1, c2 = self.thrust_poly
        return c0 + c1 * r + c2 * r * r

    def substep(self, f_des: np.ndarray) -> None:
        super().substep(f_des)
        if "drag" in self.match:
            st = self.state_of(self.s)
            f = np.clip(self.rotor_thrust(np.clip(np.array(self.s.rpm), 0.0, 1.0)), 0.0, None)
            omega_sum = float(np.sum(2.0 * np.pi / 60.0 * np.sqrt(f / PB["KF"])))   # PyBullet-equivalent rotor speed
            F_B = st.R.T @ (-PB["DRAG"] * omega_sum * st.v)
            self.s.linear_velocity = list(st.v + SUB_DT * (st.R @ F_B) / self.l2f_mass)

    def rpm_from_pybullet(self, rpm_pb: np.ndarray) -> np.ndarray:
        """Normalized L2F rotor state producing the same per-rotor thrust as PyBullet rpm `rpm_pb`."""
        return self.force_to_rpm(PB["KF"] * np.asarray(rpm_pb) ** 2)


def make_source_pb(match=(), iface: InterfaceConfig | None = None, seed: int = 0) -> CTBRDrone:
    return CTBRDrone(L2FBackendPB(seed, match), iface=iface, seed=seed)


# --------------------------------------------------------------------------------------------------
# Part A — k-step open-loop prediction error, exact Shapley attribution
# --------------------------------------------------------------------------------------------------

K_TICKS = (10, 50)
N_STARTS = 80          # start points per logged flight


def log_target_flights(seed: int):
    """Fly the Exp 10 nominal teacher (+ AR(1) exploration noise) in the target (d = 0, physics only)."""
    from stable_baselines3 import PPO
    from racing_env import GateRacingEnv

    model = PPO.load(f"{OUT}/exp10_base_nom_s{seed}.zip", device="cpu")
    rng = np.random.default_rng(100 + seed)
    flights = []
    for course in ["figure8", "ellipse3d"]:
        env = GateRacingEnv(course, "target", 8, delays=(0,), random_start_gate=False, max_ep_s=12.5, seed=900 + seed)
        o, _ = env.reset(seed=4000 + seed)
        S, RPM, A = [env.drone.true_state()], [env.drone.backend.rpm.copy()], []
        n = np.zeros(4)
        while True:
            a, _ = model.predict(o, deterministic=True)
            n = 0.9 * n + math.sqrt(1 - 0.81) * np.array([0.10, 0.06, 0.06, 0.06]) * rng.standard_normal(4)
            o, _, term, trunc, _ = env.step(np.clip(a + n, -1, 1))
            A.append(env.drone.applied.copy())
            S.append(env.drone.true_state())
            RPM.append(env.drone.backend.rpm.copy())
            if term or trunc:
                break
        flights.append({"course": course, "S": S, "RPM": RPM, "A": A})
    return flights


def predict_errors(match, flights, seed: int = 0):
    """Per-start position / velocity / attitude errors after each k in K_TICKS for one source variant."""
    drone = make_source_pb(match, seed=seed)
    be = drone.backend
    rng = np.random.default_rng(7)
    out = {k: [] for k in K_TICKS}
    kmax = max(K_TICKS)
    for fl in flights:
        S, RPM, A = fl["S"], fl["RPM"], fl["A"]
        hi = len(A) - kmax
        if hi <= 0:
            continue
        starts = rng.choice(hi, size=min(N_STARTS, hi), replace=False)
        for t0 in starts:
            drone.reset(S[t0])
            be.reset(S[t0], rpm=be.rpm_from_pybullet(RPM[t0]))
            for k in range(1, kmax + 1):
                drone.step(A[t0 + k - 1])
                if k in out:
                    st, ref = drone.true_state(), S[t0 + k]
                    dR = st.R.T @ ref.R
                    ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
                    out[k].append((float(np.linalg.norm(st.p - ref.p)), float(np.linalg.norm(st.v - ref.v)), ang,
                                   fl["course"]))
    return out


def subset_job(args):
    import torch
    torch.set_num_threads(1)
    match, flights = args
    with contextlib.redirect_stdout(io.StringIO()):
        errs = predict_errors(match, flights)
    return tuple(sorted(match)), errs


def shapley(E: dict, factors=FACTORS):
    """phi_i = sum_S |S|!(n-|S|-1)!/n! (E(S) - E(S + i)): error removed by matching factor i."""
    n = len(factors)
    phi = {}
    for f in factors:
        rest = [g for g in factors if g != f]
        tot = 0.0
        for r in range(n):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in itertools.combinations(rest, r):
                key = tuple(sorted(S))
                tot += w * (E[key] - E[tuple(sorted(S + (f,)))])
        phi[f] = tot
    return phi


def part_a(workers: int):
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(workers, 3)) as pool:
        flights = sum(pool.map(_log_job, [0, 1, 2]), [])
    print(f"logged {len(flights)} target flights ({sum(len(f['A']) for f in flights)} ticks) in {time.time() - t0:.0f}s",
          flush=True)
    subsets = [tuple(sorted(S)) for r in range(len(FACTORS) + 1) for S in itertools.combinations(FACTORS, r)]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = dict(pool.map(subset_job, [(S, flights) for S in subsets], chunksize=1))
    print(f"Part A: {len(subsets)} variants in {time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(0)
    summary = {}
    for k in K_TICKS:
        n_pts = len(res[()][k])
        for ci, name in [(0, "pos_m"), (1, "vel_mps"), (2, "att_deg")]:
            for course in ["all", "figure8", "ellipse3d"]:
                idx = [i for i, e in enumerate(res[()][k]) if course == "all" or e[3] == course]
                arr = {S: np.array([res[S][k][i][ci] for i in idx]) for S in subsets}
                E = {S: float(arr[S].mean()) for S in subsets}
                phi = shapley(E)
                boots = []
                for _ in range(200):  # paired bootstrap over start points
                    b = rng.integers(0, len(idx), len(idx))
                    boots.append(shapley({S: float(arr[S][b].mean()) for S in subsets}))
                summary[f"{name}@{k}_{course}"] = {
                    "E_none": E[()], "E_all": E[tuple(sorted(FACTORS))],
                    "single": {f: E[(f,)] for f in FACTORS},
                    "all_but": {f: E[tuple(sorted(set(FACTORS) - {f}))] for f in FACTORS},
                    "shapley": phi,
                    "shapley_ci": {f: [float(np.percentile([b[f] for b in boots], 2.5)),
                                       float(np.percentile([b[f] for b in boots], 97.5))] for f in FACTORS},
                    "n_starts": len(idx)}
        del n_pts
    save_json("exp26_partA_prediction_attribution.json", {"factors": FACTORS, "k_ticks": list(K_TICKS),
                                                          "wall_s": time.time() - t0, "summary": summary})
    for key in ["pos_m@50_all", "vel_mps@10_all", "att_deg@10_all", "pos_m@50_ellipse3d"]:
        s = summary[key]
        gap = s["E_none"] - s["E_all"]
        print(f"\n[{key}] L2F {s['E_none']:.4f} -> all matched {s['E_all']:.4f} (residual {100 * s['E_all'] / s['E_none']:.0f}%)")
        for f in sorted(FACTORS, key=lambda f: -s["shapley"][f]):
            lo, hi = s["shapley_ci"][f]
            print(f"  {f:8s} shapley {s['shapley'][f]:+.4f} [{lo:+.4f}, {hi:+.4f}]  = {100 * s['shapley'][f] / gap:5.1f}% of gap"
                  f" | alone {s['single'][f]:.4f} | all-but {s['all_but'][f]:.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, key, lab in zip(ax, ["pos_m@50_all", "vel_mps@10_all", "att_deg@10_all"],
                           ["position error @0.5 s [m]", "velocity error @0.1 s [m/s]", "attitude error @0.1 s [deg]"]):
        s = summary[key]
        order = sorted(FACTORS, key=lambda f: -s["shapley"][f])
        vals = [s["shapley"][f] for f in order]
        err = np.array([[s["shapley"][f] - s["shapley_ci"][f][0], s["shapley_ci"][f][1] - s["shapley"][f]] for f in order]).T
        a.bar(order + ["residual"], vals + [s["E_all"]], yerr=np.hstack([err, [[0], [0]]]), capsize=2,
              color=["C0"] * len(order) + ["0.6"])
        a.set_title(f"{lab}\nL2F {s['E_none']:.3f} → matched {s['E_all']:.3f}", fontsize=9)
        a.tick_params(axis="x", rotation=30)
    ax[0].set_ylabel("Shapley share of the error")
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp26_partA_prediction_attribution.png", dpi=140)


def _log_job(seed):
    import torch
    torch.set_num_threads(1)
    with contextlib.redirect_stdout(io.StringIO()):
        return log_target_flights(seed)


# --------------------------------------------------------------------------------------------------
# Part B — retrain PPO in source variants, evaluate in target_nominal
# --------------------------------------------------------------------------------------------------

def make_variant_env_cls():
    from racing_env import GateRacingEnv

    class PBSourceRacingEnv(GateRacingEnv):
        """GateRacingEnv whose source is L2FBackendPB(match) (exact-state; target domain unchanged)."""

        def __init__(self, *args, match=(), **kw):
            self.match = tuple(match)
            super().__init__(*args, **kw)

        def _get_drone(self, d: int):
            if self.domain != "source":
                return super()._get_drone(d)
            key = ("pb", d)
            if key not in self.drones:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.drones[key] = make_source_pb(self.match, iface=InterfaceConfig(action_delay=d), seed=self._seed)
            return self.drones[key]

    return PBSourceRacingEnv


# variant -> (matched physics factors, training action delay)
VARIANTS_B = {
    "l2f": ((), 0),
    "all_physics": (tuple(FACTORS), 0),
    "all_physics_d2": (tuple(FACTORS), 2),
    "l2f_d2": ((), 2),
}


def parse_variant(name: str):
    """Extra variants by name: 'only_<f>[+<g>]' or 'allbut_<f>' (d = 0), suffix '_d2' for delay 2."""
    if name in VARIANTS_B:
        return VARIANTS_B[name]
    d = 2 if name.endswith("_d2") else 0
    base = name[:-3] if d else name
    if base.startswith("only_"):
        return tuple(base[5:].split("+")), d
    if base.startswith("allbut_"):
        drop = set(base[7:].split("+"))
        return tuple(f for f in FACTORS if f not in drop), d
    raise ValueError(name)


EVAL_CELLS = {"own_source": None, "target_nominal": ("target", 2, 1.0)}


def train_job(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from exp10_method_matrix import HIST, SB3PolicyCallable
    from racing_env import evaluate_racing

    torch.set_num_threads(1)
    variant, seed, base_steps, ft_steps, n_eval = args
    match, d_train = parse_variant(variant)
    Env = make_variant_env_cls()
    t0 = time.time()
    ppo_kw = dict(n_steps=512, batch_size=512, n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.0, seed=seed, verbose=0,
                  policy_kwargs=dict(net_arch=[128, 128], log_std_init=-0.7))

    def venv(lam, s0):
        return DummyVecEnv([lambda i=i: Env("figure8", "source", HIST, lam_da=lam, delays=(d_train,), match=match,
                                            max_ep_s=5.5, seed=s0 + i) for i in range(8)])

    path = f"{OUT}/exp26_{variant}_s{seed}.zip"
    with contextlib.redirect_stdout(io.StringIO()):
        model = PPO("MlpPolicy", venv(0.02, 2000 * seed), **ppo_kw)
        model.learn(base_steps)
        model.save(path)
        model = PPO.load(path, env=venv(0.20, 2100 * seed), device="cpu")
        model.learn(ft_steps, reset_num_timesteps=False)
        model.save(path)
    pol = SB3PolicyCallable(model)
    ev = {}
    for c in ["figure8", "ellipse3d"]:
        ev[c] = {}
        with contextlib.redirect_stdout(io.StringIO()):
            ev[c]["own_source"] = evaluate_racing(pol, Env(c, "source", HIST, delays=(d_train,), match=match,
                                                           random_start_gate=False, max_ep_s=12.5, seed=800 + seed),
                                                  n_eval=n_eval)
            ev[c]["target_nominal"] = evaluate_racing(pol, Env(c, "target", HIST, delays=(2,), random_start_gate=False,
                                                               max_ep_s=12.5, seed=800 + seed), n_eval=n_eval)
    return variant, seed, ev, time.time() - t0


def part_b(workers: int, seeds: int, variants: list[str], base_steps: int, ft_steps: int, n_eval: int, tag: str):
    from exp10_method_matrix import aggregate_seed_dicts

    jobs = [(v, s, base_steps, ft_steps, n_eval) for s in range(seeds) for v in variants]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        outs = pool.map(train_job, jobs, chunksize=1)
    print(f"Part B: {len(jobs)} runs in {time.time() - t0:.0f}s", flush=True)
    agg = {v: {c: {cell: aggregate_seed_dicts([o[2][c][cell] for o in outs if o[0] == v]) for cell in EVAL_CELLS}
               for c in ["figure8", "ellipse3d"]} for v in variants}
    per_seed = {f"{o[0]}_s{o[1]}": {c: {cell: {m: o[2][c][cell][m] for m in ["gate_frac", "clean_completion",
                                                                              "track_err_m", "crash_rate"]}
                                        for cell in EVAL_CELLS} for c in o[2]} for o in outs}
    save_json(f"exp26_partB_retrain{tag}.json", {"seeds": list(range(seeds)), "base_steps": base_steps,
                                                 "ft_steps": ft_steps, "variants": {v: parse_variant(v) for v in variants},
                                                 "wall_s": time.time() - t0, "agg": agg, "per_seed": per_seed})
    for c in ["figure8", "ellipse3d"]:
        print(f"\n[{c}] gate_frac | clean | track err [m] | crash   (own source || target_nominal)")
        for v in variants:
            a = agg[v][c]
            row = " || ".join(f"{fmt(a[cell]['gate_frac'])} | {a[cell]['clean_completion']['mean']:.2f} | "
                              f"{fmt(a[cell]['track_err_m'])} | {a[cell]['crash_rate']['mean']:.2f}" for cell in EVAL_CELLS)
            print(f"  {v:18s} {row}")


def plot_b():
    """Part B figure: target vs own-source tracking error per variant (+ the Exp 19 `none` run as plain L2F)."""
    b = json.load(open(f"{OUT}/exp26_partB_retrain.json"))
    e19 = json.load(open(f"{OUT}/exp19_dr_factor_ablation.json"))["agg"]["none"]
    rows = {"l2f (Exp 19 none)": {c: {"own_source": e19[c]["source_l2f"], "target_nominal": e19[c]["target_nominal"]}
                                  for c in ["figure8", "ellipse3d"]}}
    rows.update({v: b["agg"][v] for v in ["only_motor", "allbut_motor", "all_physics", "all_physics_d2"]})
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(rows))
    for a, c in zip(ax, ["figure8", "ellipse3d"]):
        for j, cell in enumerate(["own_source", "target_nominal"]):
            a.bar(x + (j - 0.5) * 0.38, [rows[v][c][cell]["track_err_m"]["mean"] for v in rows], 0.38,
                  yerr=[rows[v][c][cell]["track_err_m"]["std"] for v in rows], capsize=2, label=cell)
        a.set_xticks(x, list(rows), rotation=25, ha="right", fontsize=8)
        a.set_title(f"{c}: tracking error [m] (lower is better)")
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp26_partB_retrain.png", dpi=140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["partA", "partB", "plotB"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants", default="l2f,all_physics,all_physics_d2,l2f_d2")
    ap.add_argument("--base-steps", type=int, default=450_000)
    ap.add_argument("--ft-steps", type=int, default=100_000)
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.part == "partA":
        part_a(args.workers)
    elif args.part == "plotB":
        plot_b()
    else:
        part_b(args.workers, args.seeds, args.variants.split(","), args.base_steps, args.ft_steps, args.n_eval, args.tag)


if __name__ == "__main__":
    main()
