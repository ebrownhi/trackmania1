"""
Exp 8 (H4'): does the Swift-style residual-augmented source actually close the L2F -> PyBullet gap?

Models of the target, all built from the same 6 x 5 s calibration flights of Exp 6 (true delay 3):
  M0  L2F nominal
  M1  L2F + parametric tau_hat (Step 0 system ID: d_hat, tau_hat)
  M2  M1 + delay-aligned body-frame k-NN linear-acceleration residual
  M3  M2 + k-NN angular-acceleration residual (6D, gotcha #4)
  M2w M1 + world-frame k-NN linear residual (guardrail #6 ablation)

(a) Fidelity: open-loop replay of executed target actions from many start points on held-out flights
    (training course + held-out 3D ellipse course); position divergence at 0.5 s and 1.0 s.
(b) Closed loop: a Smith-predictive expert in the PyBullet target with action delay 6 that uses each model
    to forward-predict through the in-flight actions (rotor state reconstructed online from the executed
    commands). Better model -> better prediction -> better tracking.
"""

import multiprocessing as mp

from common import OUT, np, plt, save_json, mean_std, fmt
from exp6_knn_residual import CTBR, KNNResidual, OneStepSource, fly, feats
from sims.drone_pair import (Course, DroneState, GeometricExpert, InterfaceConfig, RL_DT, SUB_DT, SUBSTEPS,
                             gate_pass_rate, make_target)

D_CAL = 3
TAU_HAT = 0.03          # recovered by Exp 6's joint scan (d_hat = 3, tau_hat = 30 ms)
D_CLOSED = 6
MODELS = ["M0_nominal", "M1_tau", "M2_tau_knn_lin", "M3_tau_knn_6d", "M2w_tau_knn_world"]


def calibrate():
    tgt = lambda iface, s: make_target(iface=iface, seed=s)
    cal = fly(tgt, Course(), 100, 6)
    fS = OneStepSource(TAU_HAT)
    rs, Xs = zip(*[fS.residuals(S, A, D_CAL) for S, A in cal])
    Y, X = np.concatenate(rs), sum(Xs, [])
    return {
        "body10": KNNResidual().fit(feats(X, "body10"), Y),
        "world10": KNNResidual().fit(feats(X, "world10"), Y),
    }


class Model:
    """A forward model of the target built on a private L2F backend."""

    def __init__(self, name, knns):
        self.name = name
        self.src = OneStepSource(None if name == "M0_nominal" else TAU_HAT)
        self.be = self.src.be
        if name.startswith("M2") or name.startswith("M3"):
            knn = knns["world10" if name == "M2w_tau_knn_world" else "body10"]
            kind = "world10" if name == "M2w_tau_knn_world" else "body10"
            use_ang = name.startswith("M3")
            cache = {}

            def residual_fn(st, a):
                key = id(a)  # constant within one 10 ms tick: query k-NN once per tick
                if cache.get("key") != key:
                    r = knn.predict(feats([(st, a)], kind))[0]
                    cache["key"], cache["r"] = key, (r[:3], r[3:] if use_ang else np.zeros(3))
                return cache["r"]
            self.be.residual_fn = residual_fn
        self.hover_rpm = self.be.force_to_rpm(np.full(4, self.be.base_params["dynamics"]["mass"] * 9.81 / 4))

    def advance_rotors(self, rpm, a, w):
        sp = self.be.force_to_rpm(CTBR.rotor_forces(a, w))
        for _ in range(SUBSTEPS):
            tau = np.where(sp >= rpm, self.src.tau_r, self.src.tau_f)
            al = np.exp(-SUB_DT / tau)
            rpm = al * rpm + (1 - al) * sp
        return rpm

    def rollout(self, s0, rpm0, actions):
        self.be.reset(s0, rpm=rpm0)
        out = []
        for a in actions:
            a = np.array(a)  # fresh object per tick so the residual cache refreshes
            self.be.current_action = a
            for _ in range(SUBSTEPS):
                self.be.substep(CTBR.rotor_forces(a, self.be.state().w))
            out.append(self.be.state())
        return out


def fidelity_job(args):
    name, course_kind = args
    knns = calibrate()
    m = Model(name, knns)
    tgt = lambda iface, s: make_target(iface=iface, seed=s)
    flights = fly(tgt, Course(kind=course_kind), 700 if course_kind == "figure8" else 900, 3)
    div = {0.5: [], 1.0: []}
    hover_a = CTBR.encode(9.81, np.zeros(3))
    for S, A in flights:
        exe = [A[k - D_CAL] if k >= D_CAL else hover_a for k in range(len(A))]
        rpm = m.hover_rpm.copy()
        for k in range(len(A)):
            if k >= 20 and k % 25 == 0 and k + 100 < len(A):
                traj = m.rollout(S[k], rpm, exe[k:k + 100])
                for h in div:
                    n = int(round(h / RL_DT))
                    div[h].append(np.linalg.norm(traj[n - 1].p - S[k + n].p))
            rpm = m.advance_rotors(rpm, exe[k], S[k].w)
    return name, course_kind, {f"{h}s": mean_std(v) for h, v in div.items()}


def closed_loop_job(args):
    name, course_kind, seed = args
    knns = calibrate()
    m = None if name == "naive_expert" else Model(name, knns)
    course = Course(kind=course_kind)
    ex = GeometricExpert()
    env = make_target(iface=InterfaceConfig(action_delay=D_CLOSED), seed=seed)
    rng = np.random.default_rng(seed)
    p, v, _ = course.ref(0.0)
    obs = env.reset(DroneState(p + rng.normal(0, 0.05, 3), np.eye(3), v + rng.normal(0, 0.15, 3), np.zeros(3)))
    rpm = None if m is None else m.hover_rpm.copy()
    T = int(2 * course.period / RL_DT)
    P, err, crashed = [], [], False
    for k in range(T):
        t = k * RL_DT
        if m is None:
            a = ex(obs, *course.ref(t))
        else:
            fl = env.in_flight()
            s_pred = m.rollout(obs, rpm, fl)[-1]
            a = ex(s_pred, *course.ref(t + len(fl) * RL_DT))
            rpm = m.advance_rotors(rpm, env.act_fifo[1] if D_CLOSED > 0 else a, obs.w)
        obs = env.step(a)
        st = env.true_state()
        P.append(st.p)
        e = np.linalg.norm(st.p - course.ref(t + RL_DT)[0])
        err.append(e)
        if e > 1.0 or st.p[2] < 0.05:
            crashed = True
            break
    return name, course_kind, seed, {"completion": len(P) / T, "crash": float(crashed),
                                     "gates": gate_pass_rate(course, np.array(P)) * len(P) / T,
                                     "rmse": float(np.sqrt(np.mean(np.square(err))))}


def main():
    courses = ["figure8", "ellipse3d"]
    with mp.get_context("spawn").Pool(3) as pool:
        fid = pool.map(fidelity_job, [(n, c) for n in MODELS for c in courses])
        cl = pool.map(closed_loop_job, [(n, c, s) for n in ["naive_expert"] + MODELS for c in courses for s in range(3)])

    res = {"fidelity": {}, "closed_loop": {}, "meta": {"d_cal": D_CAL, "tau_hat": TAU_HAT, "d_closed": D_CLOSED}}
    for n, c, r in fid:
        res["fidelity"].setdefault(n, {})[c] = r
    for n in ["naive_expert"] + MODELS:
        for c in courses:
            runs = [r for (nn, cc, s, r) in cl if nn == n and cc == c]
            res["closed_loop"].setdefault(n, {})[c] = {k: mean_std([r[k] for r in runs]) for k in runs[0]}
    save_json("exp8_residual_model.json", res)

    print("Exp 8a: open-loop replay divergence of target flights [m] (mean ± std over windows)")
    print(f"{'model':22s} {'fig8 0.5s':>16} {'fig8 1.0s':>16} {'held-out 0.5s':>16} {'held-out 1.0s':>16}")
    for n in MODELS:
        f = res["fidelity"][n]
        print(f"{n:22s} {fmt(f['figure8']['0.5s']):>16} {fmt(f['figure8']['1.0s']):>16} "
              f"{fmt(f['ellipse3d']['0.5s']):>16} {fmt(f['ellipse3d']['1.0s']):>16}")
    print(f"\nExp 8b: Smith-predictive expert in PyBullet with action delay {D_CLOSED} (3 seeds, 2 laps)")
    print(f"{'predictor':22s} {'fig8 rmse':>16} {'gates':>6} {'crash':>6} | {'held-out rmse':>16} {'gates':>6} {'crash':>6}")
    for n in ["naive_expert"] + MODELS:
        a, b = res["closed_loop"][n]["figure8"], res["closed_loop"][n]["ellipse3d"]
        print(f"{n:22s} {fmt(a['rmse']):>16} {a['gates']['mean']:>6.2f} {a['crash']['mean']:>6.2f} | "
              f"{fmt(b['rmse']):>16} {b['gates']['mean']:>6.2f} {b['crash']['mean']:>6.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(MODELS))
    for i, c in enumerate(courses):
        ax[0].bar(x + (i - 0.5) * 0.4, [res["fidelity"][n][c]["1.0s"]["mean"] for n in MODELS], 0.4, label=c)
    ax[0].set_xticks(x, [n.replace("_", "\n", 1) for n in MODELS], fontsize=7)
    ax[0].set_ylabel("replay divergence at 1 s [m]")
    ax[0].set_title("Model fidelity on target flights")
    ax[0].legend(fontsize=8)
    names = ["naive_expert"] + MODELS
    x = np.arange(len(names))
    for i, c in enumerate(courses):
        ax[1].bar(x + (i - 0.5) * 0.4, [res["closed_loop"][n][c]["rmse"]["mean"] for n in names], 0.4,
                  yerr=[res["closed_loop"][n][c]["rmse"]["std"] for n in names], label=c)
    ax[1].set_xticks(x, [n.replace("_", "\n", 1) for n in names], fontsize=7)
    ax[1].set_ylabel("tracking RMSE [m]")
    ax[1].set_title(f"Smith predictor in target, delay {D_CLOSED}")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp8_residual_model.png", dpi=140)


if __name__ == "__main__":
    main()
