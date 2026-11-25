"""
Exp 7 (H1' / guardrail #1): action-smoothness penalty in nominal PPO, L2F source -> PyBullet target.

Task (learnable by PPO in minutes on CPU): fly to and hold a random waypoint within 1 m from a perturbed
start (random velocity, tilt, body rates), 3 s episodes at 100 Hz, CTBR actions through the shared 500 Hz
rate loop. Observation: body-frame position error, body velocity, gravity / heading directions, body rates,
previous action a_{t-1}. Reward:
    r_t = 1 - 0.6 ||e_p|| - 0.05 ||v|| - 0.002 ||w||^2 - lambda_da ||a_t - a_{t-1}||^2,  crash -> terminate.
PPO is trained in the source only (no randomization) for lambda_da in {0, 0.1, 1.0}, 3 seeds each.

Zero-shot evaluation in source and target at action delay d in {0..4} on 20 fixed starts (4 s):
hold error (mean ||e_p|| over the last 2 s), crash rate, mean ||Δa||, and high-frequency action power
(fraction of action variance above 10 Hz).
"""

import multiprocessing as mp
import sys
import time

from common import OUT, np, plt, save_json, mean_std, fmt

LAMBDAS = [0.0, 0.1, 1.0]
SEEDS = [0, 1, 2]
DELAYS = [0, 1, 2, 3, 4]
TOTAL_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 600_000
N_ENVS = 8
EP_S = 3.0
EVAL_S = 4.0
N_EVAL = 20


def make_gym_env(lam, seed, target=False, delay=0, ep_s=EP_S):
    import gymnasium as gym
    from sims.drone_pair import DroneState, InterfaceConfig, RL_DT, make_source, make_target

    class HoverEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (19,), np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
            iface = InterfaceConfig(action_delay=delay)
            self.drone = make_target(iface=iface, seed=seed) if target else make_source(seed=seed, iface=iface)
            self.rng = np.random.default_rng(seed)
            self.T = int(ep_s / RL_DT)

        def _obs(self):
            s = self.obs_state
            Rt = s.R.T
            return np.concatenate([Rt @ (self.goal - s.p), Rt @ s.v, Rt[:, 2], Rt[:, 0], s.w, self.a_prev]).astype(np.float32)

        def reset(self, seed=None, options=None):
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            r = self.rng
            ax = r.normal(size=3)
            ax /= np.linalg.norm(ax)
            ang = r.uniform(0, 0.5)
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
            p0 = np.array([0.0, 0.0, 2.0])
            st = DroneState(p0, R, r.uniform(-0.5, 0.5, 3), r.uniform(-1, 1, 3))
            self.goal = p0 + r.uniform(-1, 1, 3)
            self.obs_state = self.drone.reset(st)
            self.a_prev = self.drone.hover_action.astype(np.float64).copy()
            self.k = 0
            return self._obs(), {}

        def step(self, a):
            a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
            self.obs_state = self.drone.step(a)
            s = self.drone.true_state()
            e = np.linalg.norm(self.goal - s.p)
            da = a - self.a_prev
            r = 1.0 - 0.6 * e - 0.05 * np.linalg.norm(s.v) - 0.002 * float(s.w @ s.w) - lam * float(da @ da)
            self.a_prev = a
            self.k += 1
            crash = e > 2.0 or s.R[2, 2] < 0.0 or not np.all(np.isfinite(s.p))
            if crash:
                r -= 50.0
            return self._obs(), float(r), bool(crash), self.k >= self.T, {"e": e, "a": a}

    return HoverEnv()


def train_job(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.set_num_threads(1)
    lam, seed = args
    t0 = time.time()
    venv = DummyVecEnv([lambda i=i: make_gym_env(lam, 1000 * seed + i) for i in range(N_ENVS)])
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=512, n_epochs=10, learning_rate=3e-4, gamma=0.99,
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, seed=seed, verbose=0,
                policy_kwargs=dict(net_arch=[128, 128]))
    model.learn(TOTAL_STEPS)
    path = f"{OUT}/exp7_ppo_lam{lam}_s{seed}.zip"
    model.save(path)
    ev = evaluate(model, lam)
    return {"lam": lam, "seed": seed, "train_s": time.time() - t0, "path": path, "eval": ev}


def hf_power(A, cutoff_hz=10.0, dt=0.01):
    X = A - A.mean(0)
    F = np.abs(np.fft.rfft(X, axis=0)) ** 2
    f = np.fft.rfftfreq(len(X), dt)
    tot = F[1:].sum()
    return float(F[f > cutoff_hz].sum() / tot) if tot > 0 else 0.0


def evaluate(model, lam):
    out = {}
    for dom in ["source", "target"]:
        out[dom] = {}
        for d in DELAYS:
            env = make_gym_env(lam, 777, target=(dom == "target"), delay=d, ep_s=EVAL_S)
            errs, crashes, das, hfs = [], [], [], []
            for ep in range(N_EVAL):
                o, _ = env.reset(seed=10_000 + ep)
                E, A, crashed = [], [], False
                while True:
                    a, _ = model.predict(o, deterministic=True)
                    o, r, term, trunc, info = env.step(a)
                    E.append(info["e"])
                    A.append(info["a"])
                    if term:
                        crashed = True
                        break
                    if trunc:
                        break
                A = np.array(A)
                crashes.append(float(crashed))
                errs.append(float(np.mean(E[len(E) // 2:])) if not crashed else np.nan)
                das.append(float(np.mean(np.linalg.norm(np.diff(A, axis=0), axis=1))))
                hfs.append(hf_power(A))
            out[dom][d] = {"hold_err": float(np.nanmean(errs)) if not all(np.isnan(errs)) else float("nan"),
                           "crash": float(np.mean(crashes)), "mean_da": float(np.mean(das)),
                           "hf_power": float(np.mean(hfs))}
    return out


def main():
    jobs = [(lam, s) for lam in LAMBDAS for s in SEEDS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(4) as pool:
        outs = pool.map(train_job, jobs)
    print(f"Exp 7: {len(jobs)} PPO runs x {TOTAL_STEPS} steps in {time.time() - t0:.0f}s "
          f"(mean {np.mean([o['train_s'] for o in outs]):.0f}s per run incl. eval)")

    res = {"total_steps": TOTAL_STEPS, "lambdas": LAMBDAS, "delays": DELAYS, "runs": outs, "agg": {}}
    for lam in LAMBDAS:
        o = [x for x in outs if x["lam"] == lam]
        res["agg"][lam] = {dom: {d: {m: mean_std([x["eval"][dom][d][m] for x in o]) for m in o[0]["eval"][dom][d]}
                                 for d in DELAYS} for dom in ["source", "target"]}
    save_json("exp7_smoothness_ppo.json", res)

    for dom in ["source", "target"]:
        print(f"\n[{dom}] hold error [m] / crash rate / mean||Δa|| / HF power (>10 Hz)   mean ± std over {len(SEEDS)} seeds")
        for lam in LAMBDAS:
            for d in DELAYS:
                m = res["agg"][lam][dom][d]
                print(f"  λ={lam:<4} d={d}: err {fmt(m['hold_err']):>16}  crash {fmt(m['crash']):>16}  "
                      f"Δa {fmt(m['mean_da']):>16}  HF {fmt(m['hf_power']):>16}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for lam in LAMBDAS:
        for dom, ls in [("source", "-"), ("target", "--")]:
            g = res["agg"][lam][dom]
            ax[0].errorbar(DELAYS, [g[d]["crash"]["mean"] for d in DELAYS], yerr=[g[d]["crash"]["std"] for d in DELAYS],
                           ls=ls, marker="o", capsize=2, label=f"λ={lam} {dom}")
            ax[1].errorbar(DELAYS, [g[d]["hold_err"]["mean"] for d in DELAYS], yerr=[g[d]["hold_err"]["std"] for d in DELAYS],
                           ls=ls, marker="o", capsize=2)
            ax[2].errorbar(DELAYS, [g[d]["mean_da"]["mean"] for d in DELAYS], yerr=[g[d]["mean_da"]["std"] for d in DELAYS],
                           ls=ls, marker="o", capsize=2)
    for a, t in zip(ax, ["crash rate", "hold error [m] (non-crashed)", "mean ||Δa||"]):
        a.set_xlabel("action delay d [ticks]")
        a.set_title(t)
    ax[0].legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp7_smoothness_ppo.png", dpi=140)


if __name__ == "__main__":
    main()
