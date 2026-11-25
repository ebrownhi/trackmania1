"""
Exp 3 (Lemma 1.1 / guardrail #3): observation dropout via zero-order hold vs. naive zero-masking.

The privileged expert flies the training course in the PyBullet target from observations with
obs delay d_o = 1 and Bernoulli dropout p in {0, .05, .1, .2, .3, .5}, handled either by
  - ZOH:  o_t = o_{t-1} when dropped, or
  - zero: o_t = 0 when dropped.
Lemma 1.1 predicts (a) ZOH is equivalent to extra staleness with E[stale] = d_o + p / (1 - p),
and (b) zero-masking injects O(p * ||h(s)||) impulses. We measure tracking RMSE, gate pass rate,
crash rate and the empirical mean observation staleness.
"""

from common import OUT, np, plt, save_json, mean_std, fmt
from sims.drone_pair import Course, GeometricExpert, InterfaceConfig, RL_DT, gate_pass_rate, make_target

P_DROPS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
SEEDS = range(5)
LAPS = 2
CRASH_ERR = 1.0


def run(course, p_drop, mode, seed):
    iface = InterfaceConfig(obs_delay=1, p_drop=p_drop, dropout_mode=mode)
    drone = make_target(iface=iface, seed=seed)
    ex = GeometricExpert()
    obs = drone.reset(course.start_state())
    P, err, stale, age = [], [], [], 0
    crashed = False
    for k in range(int(LAPS * course.period / RL_DT)):
        pr, vr, ar = course.ref(k * RL_DT)
        obs = drone.step(ex(obs, pr, vr, ar))
        age = age + 1 if drone.dropped else 0
        stale.append(1 + age)
        st = drone.true_state()
        P.append(st.p)
        e = np.linalg.norm(st.p - course.ref((k + 1) * RL_DT)[0])
        err.append(e)
        if e > CRASH_ERR or st.p[2] < 0.05 or not np.all(np.isfinite(st.p)):
            crashed = True
            break
    err = np.array(err)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))) if not crashed else float("nan"),
        "crash": float(crashed),
        "gates": gate_pass_rate(course, np.array(P)) * len(P) / int(LAPS * course.period / RL_DT),
        "stale": float(np.mean(stale)),
    }


def main():
    course = Course()
    res = {}
    for mode in ["zoh", "zero"]:
        res[mode] = {}
        for p in P_DROPS:
            runs = [run(course, p, mode, s) for s in SEEDS]
            res[mode][str(p)] = {k: mean_std([r[k] for r in runs]) for k in runs[0]}
            res[mode][str(p)]["stale_theory"] = 1 + p / (1 - p)
    save_json("exp3_dropout_zoh.json", res)

    print("Exp 3: expert in PyBullet target, obs delay 1, Bernoulli dropout (5 seeds, 2 laps)")
    print(f"{'p_drop':>7} | {'ZOH rmse [m]':>16} {'crash':>6} {'gates':>6} {'stale (thy)':>13} | {'ZERO crash':>10} {'gates':>6}")
    for p in P_DROPS:
        z, n = res["zoh"][str(p)], res["zero"][str(p)]
        print(f"{p:>7} | {fmt(z['rmse']):>16} {z['crash']['mean']:>6.2f} {z['gates']['mean']:>6.2f} "
              f"{z['stale']['mean']:>6.2f} ({z['stale_theory']:.2f}) | {n['crash']['mean']:>10.2f} {n['gates']['mean']:>6.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for mode, mk in [("zoh", "o-"), ("zero", "s--")]:
        ax[0].plot(P_DROPS, [res[mode][str(p)]["gates"]["mean"] for p in P_DROPS], mk, label=mode)
    ax[0].set_xlabel("dropout probability p")
    ax[0].set_ylabel("gate pass rate")
    ax[0].legend()
    ax[0].set_title("ZOH vs zero-masking")
    ax[1].plot(P_DROPS, [res["zoh"][str(p)]["stale"]["mean"] for p in P_DROPS], "o", label="measured")
    ax[1].plot(P_DROPS, [res["zoh"][str(p)]["stale_theory"] for p in P_DROPS], "k-", label="d_o + p/(1-p)")
    ax[1].set_xlabel("dropout probability p")
    ax[1].set_ylabel("mean obs staleness [ticks]")
    ax[1].legend()
    ax[1].set_title("Lemma 1.1 staleness")
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp3_dropout_zoh.png", dpi=140)


if __name__ == "__main__":
    main()
