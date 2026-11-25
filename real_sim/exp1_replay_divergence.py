"""
Exp 1 (Proposal Step 1): open-loop scripted-action replay, L2F source vs gym-pybullet-drones target.

1. Fly the expert closed-loop in the source on the training course and record its CTBR actions.
2. Replay those exact actions open-loop from the same start state in:
     - source again (determinism check, must be 0),
     - target variants that add one gap source at a time:
         PYB (no drag, no motor lag) -> + blade drag/ground effect -> + motor lag tau_m = 30 ms.
3. Report 3D position divergence at 0.5 / 1 / 2 / 3 s, averaged over perturbed start states.
"""

from common import OUT, np, plt, save_json, mean_std, fmt
from sims.drone_pair import (Course, CTBRDrone, DroneState, GeometricExpert, L2FBackend, PyBulletBackend, RL_DT,
                             make_source)

T_HORIZON = 3.0
N_STARTS = 5


def record_source_actions(course, start):
    src = make_source()
    ex = GeometricExpert()
    src.reset(start)
    A, P = [], []
    for k in range(int(T_HORIZON / RL_DT)):
        pr, vr, ar = course.ref(k * RL_DT)
        a = ex(src.true_state(), pr, vr, ar)
        src.step(a)
        A.append(a)
        P.append(src.true_state().p)
    return np.array(A), np.array(P)


def replay(drone: CTBRDrone, start, A):
    drone.reset(start)
    P = []
    for a in A:
        drone.step(a)
        P.append(drone.true_state().p)
    return np.array(P)


def main():
    course = Course()
    rng = np.random.default_rng(0)
    variants = {
        "source_replay": lambda: make_source(),
        "target_PYB_nolag": lambda: CTBRDrone(PyBulletBackend(tau_m=0.0, physics="PYB")),
        "target_DRAG_nolag": lambda: CTBRDrone(PyBulletBackend(tau_m=0.0, physics="PYB_GND_DRAG_DW")),
        "target_DRAG_lag30ms": lambda: CTBRDrone(PyBulletBackend(tau_m=0.03, physics="PYB_GND_DRAG_DW")),
    }
    drones = {k: f() for k, f in variants.items()}
    checkpoints = [0.5, 1.0, 2.0, 3.0]
    div = {k: {c: [] for c in checkpoints} for k in variants}
    curves = {k: [] for k in variants}
    for i in range(N_STARTS):
        s0 = course.start_state()
        if i > 0:
            s0.p = s0.p + rng.normal(0, 0.05, 3)
            s0.v = s0.v + rng.normal(0, 0.1, 3)
        A, P_src = record_source_actions(course, s0)
        for k, d in drones.items():
            P = replay(d, s0, A)
            e = np.linalg.norm(P - P_src, axis=1)
            curves[k].append(e)
            for c in checkpoints:
                div[k][c].append(e[int(round(c / RL_DT)) - 1])

    res = {k: {f"{c}s": mean_std(v) for c, v in d.items()} for k, d in div.items()}
    res["meta"] = {"course": course.kind, "peak_speed_mps": course.peak_speed(), "n_starts": N_STARTS}
    save_json("exp1_replay_divergence.json", res)

    print(f"Exp 1: open-loop replay divergence [m] (figure-8, v_peak {course.peak_speed():.2f} m/s, {N_STARTS} starts)")
    print(f"{'variant':24s}" + "".join(f"{c:>18}" for c in ["0.5s", "1s", "2s", "3s"]))
    for k in variants:
        print(f"{k:24s}" + "".join(f"{fmt(res[k][f'{c}s']):>18}" for c in checkpoints))

    t = np.arange(1, int(T_HORIZON / RL_DT) + 1) * RL_DT
    plt.figure(figsize=(6.5, 4))
    for k in variants:
        c = np.array(curves[k])
        plt.plot(t, c.mean(0), label=k)
        plt.fill_between(t, c.min(0), c.max(0), alpha=0.15)
    plt.axhline(course.gate_radius, ls="--", c="gray", lw=0.8, label="gate radius")
    plt.yscale("symlog", linthresh=1e-3)
    plt.xlabel("time [s]")
    plt.ylabel("‖p_target − p_source‖ [m]")
    plt.title("Exp 1: open-loop replay of source expert actions")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp1_replay_divergence.png", dpi=140)


if __name__ == "__main__":
    main()
