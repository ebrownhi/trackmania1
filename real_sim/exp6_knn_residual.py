"""
Exp 6 (Theorem 2.5, guardrails #5-#6): delay estimation and delay-aligned body-frame k-NN dynamics residuals.

Calibration data: the expert (plus exploration noise) flies 6 x 5 s on the training figure-8 in a target
with an unknown action delay d = 3. The client logs (o_t, a_t) only. Two targets:
  - "l2f_delayed":  the source itself + delay  (Delta f == 0, the Trackmania-like control case),
  - "pybullet":     gym-pybullet-drones CF2X + delay (Delta f != 0, real dynamics gap).

Steps:
  1. Delay scan  d_hat = argmin_d' RMS(r_d'),  r_d' = [v,w]_{t+1} - f_S(s_t, a_{t-d'})  (6D body-frame).
     f_S is a one-tick L2F rollout from the logged state; L2F's hidden rotor speeds are reconstructed by
     running L2F's own asymmetric rotor filter over the d'-aligned command history.
  2. Fit k-NN (k = 5) residuals: naive (d' = 0) vs aligned (d_hat), and body-frame vs world-frame features.
  3. Test on held-out flights (training figure-8 and held-out 3D ellipse course): residual-model RMSE vs
     the true residual (computed with the true delay), and mean |r_hat| (phantom magnitude).
"""

import math

from sklearn.neighbors import KNeighborsRegressor

from common import OUT, np, plt, save_json
from sims.drone_pair import (Course, CTBRConfig, DroneState, GeometricExpert, InterfaceConfig, L2FBackend, RL_DT,
                             SUB_DT, SUBSTEPS, make_source, make_target)

D_TRUE = 3
D_SCAN = list(range(7))
CTBR = CTBRConfig()


def fly(make_env, course, seed, n_flights, dur_s=5.0, noise=0.15):
    rng = np.random.default_rng(seed)
    ex = GeometricExpert()
    flights = []
    for f in range(n_flights):
        env = make_env(InterfaceConfig(action_delay=D_TRUE), seed + f)
        t0 = float(rng.uniform(0, course.period))
        p, v, _ = course.ref(t0)
        obs = env.reset(DroneState(p, np.eye(3), v, np.zeros(3)))
        S, A = [obs], []
        for k in range(int(dur_s / RL_DT)):
            a = np.clip(ex(obs, *course.ref(t0 + k * RL_DT)) + noise * rng.standard_normal(4), -1, 1)
            obs = env.step(a)
            S.append(obs)
            A.append(a)
        flights.append((S, np.array(A)))
    return flights


class OneStepSource:
    """f_S(s_t, a_exec) with L2F rotor state reconstructed from the aligned command history.

    tau=None keeps L2F's own asymmetric rotor lag; a scalar tau replaces it with a symmetric first-order lag
    (the parametric tau_m of guardrail Nuance #2), both inside the one-tick rollout and in the reconstruction."""

    def __init__(self, tau: float | None = None):
        self.be = L2FBackend()
        tr, tf = self.be.motor_taus()
        if tau is not None:
            self.be.set_dynamics({"tau_rise_scale": tau / tr, "tau_fall_scale": tau / tf})
        self.tau_r, self.tau_f = self.be.motor_taus()
        self.hover_a = CTBR.encode(9.81, np.zeros(3))

    def residuals(self, S, A, d, return_u_filt: bool = False):
        """6D body-frame residuals for every tick t >= 10 of one flight, assuming delay d."""
        a_exec = lambda k: A[k - d] if k - d >= 0 else self.hover_a
        rpm = self.be.force_to_rpm(np.full(4, self.be.base_params["dynamics"]["mass"] * 9.81 / 4))
        c0, c1, c2 = self.be.thrust_poly
        out, X, Uf = [], [], []
        for k in range(len(A)):
            s, s1 = S[k], S[k + 1]
            if k >= 10:
                self.be.reset(s, rpm=rpm)
                self.be.current_action = a_exec(k)
                for _ in range(SUBSTEPS):
                    self.be.substep(CTBR.rotor_forces(a_exec(k), self.be.state().w))
                pr = self.be.state()
                r_lin = s.R.T @ (s1.v - pr.v) / RL_DT
                r_ang = (s1.w - pr.w) / RL_DT
                out.append(np.concatenate([r_lin, r_ang]))
                X.append((s, a_exec(k)))
            # advance the reconstructed rotor state with L2F's own closed-form asymmetric filter
            sp = self.be.force_to_rpm(CTBR.rotor_forces(a_exec(k), s.w))
            f_sub = 0.0
            for _ in range(SUBSTEPS):
                tau = np.where(sp >= rpm, self.tau_r, self.tau_f)
                al = np.exp(-SUB_DT / tau)
                rpm = al * rpm + (1 - al) * sp
                f_sub += float(np.sum(c0 + c1 * rpm + c2 * rpm * rpm))
            if k >= 10:
                Uf.append(f_sub / SUBSTEPS)
        if return_u_filt:
            return np.array(out), X, np.array(Uf)
        return np.array(out), X


def feats(X, kind):
    rows = []
    for s, a in X:
        vB = s.R.T @ s.v
        if kind == "body7":
            rows.append(np.concatenate([vB, s.w, a[:1]]))
        elif kind == "body10":
            rows.append(np.concatenate([vB, s.w, a]))
        elif kind == "world10":
            rows.append(np.concatenate([s.p, s.v, a]))
    return np.array(rows)


class KNNResidual:
    """Standardized k-NN with decay to zero outside the calibration support."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, Z, Y):
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        Zs = (Z - self.mu) / self.sd
        self.knn = KNeighborsRegressor(self.k, weights="distance").fit(Zs, Y)
        d, _ = self.knn.kneighbors(Zs, n_neighbors=self.k + 1)
        self.h = np.percentile(d[:, 1:].mean(1), 95)
        return self

    def predict(self, Z):
        Zs = (Z - self.mu) / self.sd
        d, _ = self.knn.kneighbors(Zs)
        decay = np.exp(-np.maximum(d.mean(1) - self.h, 0.0) ** 2 / (self.h ** 2))
        return self.knn.predict(Zs) * decay[:, None]


def run_target(name, make_env, fS):
    train_c, held_c = Course(), Course(kind="ellipse3d")
    cal = fly(make_env, train_c, 100, 6)
    tests = {"train_course": fly(make_env, train_c, 500, 3), "heldout_course": fly(make_env, held_c, 900, 3)}

    # Step 0 system ID: joint scan over delay d' and parametric rotor lag tau, each with an affine
    # least-squares thrust/drag correction r_lin ~ theta^T [1, c, v_B] so the airframe-gain mismatch
    # does not mask the delay. Report the raw delay-only scan too (tau = L2F native, no correction).
    raw_scan, joint = [], {}
    for tau_name, src_model in fS.items():
        for d in D_SCAN:
            rs, Xs = zip(*[src_model.residuals(S, A, d) for S, A in cal])
            r, X = np.concatenate(rs), sum(Xs, [])
            if tau_name == "l2f_native":
                raw_scan.append(float(np.sqrt(np.mean(r[:, :3] ** 2))))
            Phi = np.column_stack([np.ones(len(X)), [a[0] for _, a in X], [s.R.T @ s.v for s, _ in X]])
            theta, *_ = np.linalg.lstsq(Phi, r[:, :3], rcond=None)
            joint[(tau_name, d)] = float(np.sqrt(np.mean((r[:, :3] - Phi @ theta) ** 2)))
    tau_hat, d_hat = min(joint, key=joint.get)
    fS = fS[tau_hat]

    res = {"delay_scan_rms_lin_raw": raw_scan,
           "joint_scan": {f"{k[0]}|d={k[1]}": v for k, v in joint.items()},
           "d_hat": d_hat, "tau_hat": tau_hat, "d_true": D_TRUE, "models": {}}
    test_true = {}
    for tname, fl in tests.items():
        rs, Xs = zip(*[fS.residuals(S, A, D_TRUE) for S, A in fl])
        test_true[tname] = (np.concatenate(rs), sum(Xs, []))

    for align_name, d_fit in [("naive_d0", 0), ("aligned_dhat", d_hat)]:
        rs, Xs = zip(*[fS.residuals(S, A, d_fit) for S, A in cal])
        Y, X = np.concatenate(rs), sum(Xs, [])
        for fk in ["body7", "body10", "world10"]:
            m = KNNResidual().fit(feats(X, fk), Y)
            key = f"{align_name}/{fk}"
            res["models"][key] = {}
            for tname, (Yt, Xt) in test_true.items():
                pred = m.predict(feats(Xt, fk))
                res["models"][key][tname] = {
                    "rmse_lin": float(np.sqrt(np.mean((pred[:, :3] - Yt[:, :3]) ** 2))),
                    "rmse_ang": float(np.sqrt(np.mean((pred[:, 3:] - Yt[:, 3:]) ** 2))),
                    "mean_abs_pred_lin": float(np.mean(np.linalg.norm(pred[:, :3], axis=1))),
                }
    for tname, (Yt, _) in test_true.items():
        res["models"].setdefault("zero_residual", {})[tname] = {
            "rmse_lin": float(np.sqrt(np.mean(Yt[:, :3] ** 2))),
            "rmse_ang": float(np.sqrt(np.mean(Yt[:, 3:] ** 2))),
            "mean_abs_pred_lin": 0.0,
            "true_mean_abs_lin": float(np.mean(np.linalg.norm(Yt[:, :3], axis=1))),
        }
    return res


TAUS = {"l2f_native": None, "tau10ms": 0.01, "tau20ms": 0.02, "tau30ms": 0.03, "tau50ms": 0.05, "tau80ms": 0.08}


def main():
    fS = {k: OneStepSource(t) for k, t in TAUS.items()}
    targets = {
        "l2f_delayed (Δf≡0)": lambda iface, s: make_source(seed=s, iface=iface),
        "pybullet (Δf≠0)": lambda iface, s: make_target(iface=iface, seed=s),
    }
    out = {name: run_target(name, mk, fS) for name, mk in targets.items()}
    save_json("exp6_knn_residual.json", out)

    for name, r in out.items():
        print(f"\n=== {name}: true delay {r['d_true']} ===")
        print("raw delay scan (L2F rotors, no correction) RMS [m/s^2]: " + "  ".join(f"{d}:{v:.3f}" for d, v in zip(D_SCAN, r["delay_scan_rms_lin_raw"])))
        for tn in TAUS:
            print(f"  joint scan {tn:>10}: " + "  ".join(f"{d}:{r['joint_scan'][f'{tn}|d={d}']:.3f}" for d in D_SCAN))
        print(f"-> d_hat = {r['d_hat']}, tau_hat = {r['tau_hat']}")
        print(f"{'model':24s} {'train-course lin/ang RMSE':>28} {'|r_hat|':>8} {'held-out lin/ang RMSE':>24} {'|r_hat|':>8}")
        for key, m in r["models"].items():
            a, b = m["train_course"], m["heldout_course"]
            print(f"{key:24s} {a['rmse_lin']:>14.3f} / {a['rmse_ang']:<11.2f} {a['mean_abs_pred_lin']:>8.3f} "
                  f"{b['rmse_lin']:>10.3f} / {b['rmse_ang']:<11.2f} {b['mean_abs_pred_lin']:>8.3f}")
        print(f"(true residual mean |r| on held-out: {r['models']['zero_residual']['heldout_course']['true_mean_abs_lin']:.3f} m/s^2)")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, r in out.items():
        ax[0].plot(D_SCAN, [r["joint_scan"][f"{r['tau_hat']}|d={d}"] for d in D_SCAN], "o-",
                   label=f"{name}, τ̂={r['tau_hat']}")
    ax[0].axvline(D_TRUE, ls="--", c="gray")
    ax[0].set_xlabel("candidate delay d'")
    ax[0].set_ylabel("RMS lin. residual after affine fit [m/s²]")
    ax[0].set_title("Delay scan (true d = 3)")
    ax[0].legend(fontsize=8)
    keys = ["zero_residual", "naive_d0/body7", "aligned_dhat/body7", "aligned_dhat/body10", "aligned_dhat/world10"]
    r = out["pybullet (Δf≠0)"]["models"]
    x = np.arange(len(keys))
    ax[1].bar(x - 0.2, [r[k]["train_course"]["rmse_lin"] for k in keys], 0.4, label="training course")
    ax[1].bar(x + 0.2, [r[k]["heldout_course"]["rmse_lin"] for k in keys], 0.4, label="held-out course")
    ax[1].set_xticks(x, [k.replace("/", "\n") for k in keys], fontsize=7)
    ax[1].set_ylabel("residual prediction RMSE [m/s²]")
    ax[1].set_title("PyBullet k-NN residual (test flights)")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp6_knn_residual.png", dpi=140)


if __name__ == "__main__":
    main()
