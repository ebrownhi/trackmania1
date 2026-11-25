"""
Exp 2 (Prop. 2.1 / H1 guardrail): discrete delay stability limit, measured in the real simulators.

A vertical-velocity hold loop  c_t = g + k (0 - vz_t),  executed with action delay d ticks, is exactly
the linear plant of Prop. 2.1:  e_{t+1} = e_t - g_eff e_{t-d},  g_eff = k * dt * (thrust gain).
Theory: Schur-stable iff g_eff < g_crit(d) = 2 sin(pi / (2(2d+1))).

For each d we bisect the largest stable g_eff in:
  - PyBullet target, tau_m = 0      (should match the pure-delay formula),
  - PyBullet target, tau_m = 30 ms  (motor lag adds phase -> lower limit; compared with a linear
                                     delay + first-order-lag model at 500 Hz),
  - L2F source (asymmetric 55/249 ms motor lag; reported, no closed form).
We also run the bang-bang (relay) controller c = g + U sign(-vz) to show the delay-driven limit cycle.
"""

import math

from common import OUT, np, plt, save_json
from sims.drone_pair import (CTBRDrone, DroneState, GRAVITY, InterfaceConfig, L2FBackend, PyBulletBackend, RL_DT,
                             SUB_DT, SUBSTEPS, CTBRConfig)

DELAYS = [0, 1, 2, 3, 4, 5]
CFG = CTBRConfig()
MASS_TGT = 0.027


def g_crit_pure(d):
    return 2 * math.sin(math.pi / (2 * (2 * d + 1)))


def linear_model_unstable(g_eff, d, tau_m, n=1200):
    """Same loop at 500 Hz substeps with first-order force lag; returns True if it diverges."""
    alpha = 1.0 - math.exp(-SUB_DT / tau_m) if tau_m > 0 else 1.0
    e, f = 1.0, 0.0
    fifo = [0.0] * d
    peak0 = None
    hist = []
    for t in range(n):
        fifo.append(-g_eff / RL_DT * e)
        u = fifo.pop(0)
        for _ in range(SUBSTEPS):
            f += alpha * (u - f)
            e += SUB_DT * f
        hist.append(abs(e))
    hist = np.array(hist)
    return hist[-200:].max() > hist[:200].max()


def bisect(unstable, lo=0.01, hi=2.5, iters=14):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if unstable(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def sim_vz_loop(make_backend, gain_factor, d, g_eff=None, relay_U=None, T=8.0, vz0=0.05):
    drone = CTBRDrone(make_backend(), iface=InterfaceConfig(action_delay=d))
    st = DroneState.hover((0.0, 0.0, 20.0))
    st.v = np.array([0.0, 0.0, vz0])
    drone.reset(st)
    vz = []
    for _ in range(int(T / RL_DT)):
        s = drone.true_state()
        z_err = 0.5 * np.array([s.R[2, 1] - s.R[1, 2], s.R[0, 2] - s.R[2, 0], 0.0])  # level attitude
        if relay_U is not None:
            c = GRAVITY / gain_factor - relay_U * np.sign(s.v[2])
        else:  # hover feed-forward corrected for the airframe so only the loop dynamics remain
            c = GRAVITY / gain_factor - (g_eff / (RL_DT * gain_factor)) * s.v[2]
        drone.step(CFG.encode(c, -7.0 * z_err))
        vz.append(drone.true_state().v[2])
    return np.array(vz)


def measured_unstable(make_backend, gain_factor, d):
    def f(g_eff):
        vz = sim_vz_loop(make_backend, gain_factor, d, g_eff=g_eff)
        # stable <=> error has decayed below 20% of the initial 0.05 m/s after 8 s (saturated or
        # limit-cycling loops never do)
        tail = vz[-int(1.0 / RL_DT):]
        return tail.max() - tail.min() > 0.2 * 0.05
    return f


def relay_model_period(d, h=1.0, n=4000):
    x, fifo, xs = 0.3 * h, [0.0] * d, []
    for _ in range(n):
        fifo.append(-np.sign(x))
        x += h * fifo.pop(0)
        xs.append(x)
    return limit_cycle(np.array(xs))[0]


def limit_cycle(vz):
    tail = vz[len(vz) // 2:]
    s = np.sign(tail)
    cross = np.where(np.diff(s) != 0)[0]
    period = 2 * np.mean(np.diff(cross)) * RL_DT if len(cross) > 2 else float("nan")
    return period, float(tail.max() - tail.min())


def main():
    sims = {
        "pybullet_tau0": (lambda: PyBulletBackend(tau_m=0.0), CFG.mass / MASS_TGT, 0.0),
        "pybullet_tau30ms": (lambda: PyBulletBackend(tau_m=0.03), CFG.mass / MASS_TGT, 0.03),
        "l2f_source": (lambda: L2FBackend(), 1.0, None),
    }
    res = {"delays": DELAYS, "theory_pure_delay": [g_crit_pure(d) for d in DELAYS],
           "theory_delay_plus_lag30ms": [bisect(lambda g: linear_model_unstable(g, d, 0.03)) for d in DELAYS]}
    for name, (mk, gf, _) in sims.items():
        res[name] = [bisect(measured_unstable(mk, gf, d), iters=10) for d in DELAYS]

    # Relay (bang-bang) limit cycle vs delay, PyBullet tau_m = 0
    relay = {"period_s": [], "p2p_vz": [], "discrete_model_period_s": [relay_model_period(d) for d in DELAYS]}
    for d in DELAYS:
        per, amp = limit_cycle(sim_vz_loop(sims["pybullet_tau0"][0], sims["pybullet_tau0"][1], d, relay_U=2.0, T=4.0))
        relay["period_s"].append(per)
        relay["p2p_vz"].append(amp)
    res["relay_U2_pybullet_tau0"] = relay
    save_json("exp2_delay_stability.json", res)

    print("Exp 2: critical loop gain g_crit(d)  [largest stable g_eff = k*dt*thrust_gain]")
    hdr = ["d", "theory 2sin(.)", "measured PyB tau=0", "theory+lag30ms", "measured PyB 30ms", "measured L2F"]
    print("  ".join(f"{h:>18}" for h in hdr))
    for i, d in enumerate(DELAYS):
        row = [d, res["theory_pure_delay"][i], res["pybullet_tau0"][i], res["theory_delay_plus_lag30ms"][i],
               res["pybullet_tau30ms"][i], res["l2f_source"][i]]
        print("  ".join(f"{v:>18.3f}" if isinstance(v, float) else f"{v:>18d}" for v in row))
    print("\nRelay c = g - 2 sign(vz), PyBullet tau=0: limit-cycle period [s] / peak-to-peak vz [m/s]")
    for d, p, a, pm in zip(DELAYS, relay["period_s"], relay["p2p_vz"], relay["discrete_model_period_s"]):
        print(f"  d={d}: period {p / RL_DT:.1f} ticks (discrete model {pm / RL_DT:.1f}, 4(d+1) = {4 * (d + 1)}), p2p {a:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(DELAYS, res["theory_pure_delay"], "k-", label="theory: 2 sin(π/(2(2d+1)))")
    ax[0].plot(DELAYS, res["pybullet_tau0"], "o", label="PyBullet, τm=0")
    ax[0].plot(DELAYS, res["theory_delay_plus_lag30ms"], "k--", label="linear model, delay + 30 ms lag")
    ax[0].plot(DELAYS, res["pybullet_tau30ms"], "s", label="PyBullet, τm=30 ms")
    ax[0].plot(DELAYS, res["l2f_source"], "^", label="L2F (55/249 ms lag)")
    ax[0].set_xlabel("action delay d [ticks of 10 ms]")
    ax[0].set_ylabel("critical gain g_crit")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=7)
    ax[0].set_title("Delay stability limit (Prop. 2.1)")
    ax[1].plot(DELAYS, relay["p2p_vz"], "o-")
    ax[1].set_xlabel("action delay d")
    ax[1].set_ylabel("relay limit-cycle p2p v_z [m/s]")
    ax[1].set_title("Bang-bang chattering grows with delay")
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp2_delay_stability.png", dpi=140)


if __name__ == "__main__":
    main()
