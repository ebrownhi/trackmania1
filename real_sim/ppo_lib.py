"""PPO waypoint-hover task shared by Exp 7b (H1' retest) and Exp 9 (H4' residual fine-tuning).

Same task and reward as Exp 7 (fly to and hold a random waypoint within 1 m from a perturbed start, 100 Hz
CTBR), generalized with: action-history length, per-episode delay sampling, dynamics randomization,
a Swift-style residual-augmented source (k-NN dynamics + GP observation bias), and target variants.
"""

import common  # noqa: F401  (puts code/ on sys.path)
import numpy as np


def hf_power(A, cutoff_hz=10.0, dt=0.01):
    X = A - A.mean(0)
    F = np.abs(np.fft.rfft(X, axis=0)) ** 2
    f = np.fft.rfftfreq(len(X), dt)
    tot = F[1:].sum()
    return float(F[f > cutoff_hz].sum() / tot) if tot > 0 else 0.0


def make_hover_env(lam=0.1, seed=0, domain="source", hist=1, delays=(0,), dyn_sampler=None, ep_s=3.0,
                   residual=None, obs_bias_fn=None, obs_bias_model=None, tau_hat=None, kf_scale=1.0):
    """domain: 'source' (L2F) or 'target' (PyBullet CF2X).
    delays: tuple of action delays sampled uniformly per episode.
    dyn_sampler(rng) -> L2F overrides dict, resampled per episode (dynamics randomization).
    residual: callable (DroneState, action) -> (a_res_B, alpha_res) injected into L2F (augmented source).
    obs_bias_fn: true systematic position bias of the sensor, b(s) (target sensor model).
    obs_bias_model: learned GP mean b_hat(o) added to source observations (augmented source).
    kf_scale: perturbed target with stronger/weaker motors (physics kf scaled, command map kept nominal)."""
    import gymnasium as gym
    from sims.drone_pair import DroneState, InterfaceConfig, RL_DT, make_source, make_target

    class HoverEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (15 + 4 * hist,), np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
            self.rng = np.random.default_rng(seed)
            self.T = int(ep_s / RL_DT)
            self.drones = {}
            self.bias = obs_bias_fn if obs_bias_fn is not None else obs_bias_model

        def _drone(self, d):
            if d not in self.drones:
                iface = InterfaceConfig(action_delay=d, obs_bias_fn=self.bias)
                if domain == "target":
                    self.drones[d] = make_target(iface=iface, seed=seed)
                    self.drones[d].backend.env.KF *= kf_scale
                else:
                    ov = {}
                    if tau_hat is not None:  # parametric tau_hat from Step 0 system ID (both rise and fall)
                        ov = {"tau_rise_scale": tau_hat / 0.055455, "tau_fall_scale": tau_hat / 0.249394}
                    self.drones[d] = make_source(seed=seed, overrides=ov, iface=iface)
                    self.drones[d].backend.residual_fn = residual
            return self.drones[d]

        def _obs(self):
            s = self.obs_state
            Rt = s.R.T
            past = list(self.ah)[::-1]
            return np.concatenate([Rt @ (self.goal - s.p), Rt @ s.v, Rt[:, 2], Rt[:, 0], s.w] + past).astype(np.float32)

        def reset(self, seed=None, options=None):
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            r = self.rng
            self.drone = self._drone(int(r.choice(delays)))
            if dyn_sampler is not None:
                self.drone.backend.set_dynamics(dyn_sampler(r))
            ax = r.normal(size=3)
            ax /= np.linalg.norm(ax)
            ang = r.uniform(0, 0.5)
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
            p0 = np.array([0.0, 0.0, 2.0])
            st = DroneState(p0, R, r.uniform(-0.5, 0.5, 3), r.uniform(-1, 1, 3))
            self.goal = p0 + r.uniform(-1, 1, 3)
            self.obs_state = self.drone.reset(st)
            from collections import deque
            self.ah = deque([self.drone.hover_action.astype(np.float64).copy()] * hist, maxlen=hist)
            self.k = 0
            return self._obs(), {}

        def step(self, a):
            a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
            self.obs_state = self.drone.step(a)
            s = self.drone.true_state()
            e = np.linalg.norm(self.goal - s.p)
            da = a - self.ah[-1]
            r = 1.0 - 0.6 * e - 0.05 * np.linalg.norm(s.v) - 0.002 * float(s.w @ s.w) - lam * float(da @ da)
            self.ah.append(a)
            self.k += 1
            crash = e > 2.0 or s.R[2, 2] < 0.0 or not np.all(np.isfinite(s.p))
            if crash:
                r -= 50.0
            return self._obs(), float(r), bool(crash), self.k >= self.T, {"e": e, "a": a, "state": s}

    return HoverEnv()


def evaluate_hover(model, env_fn, n_eval=20, eval_s=4.0, seed0=10_000):
    """env_fn(ep_s) -> env. Hold error = mean true ||goal - p|| over the last half of the episode."""
    env = env_fn(eval_s)
    errs, crashes, das, hfs = [], [], [], []
    for ep in range(n_eval):
        o, _ = env.reset(seed=seed0 + ep)
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
    errs = np.array(errs)
    return {"hold_err": float(np.nanmean(errs)) if np.any(np.isfinite(errs)) else float("nan"),
            "crash": float(np.mean(crashes)), "mean_da": float(np.mean(das)), "hf_power": float(np.mean(hfs)),
            "eff_err": float(np.mean(np.where(np.isfinite(errs), errs, 1.0)))}
