"""Students, features, DAgger and evaluation shared by the learning experiments (Exp 4-6)."""

from collections import deque

import torch
import torch.nn as nn

from common import np
from sims.drone_pair import Course, DroneState, GeometricExpert, InterfaceConfig, RL_DT, gate_pass_rate

torch.set_num_threads(1)
LOOKAHEAD_S = 0.3
CRASH_ERR = 1.0


def features(obs: DroneState, course: Course, t: float) -> np.ndarray:
    """Body-frame, course-relative observation (no world position: guardrail #6)."""
    Rt = obs.R.T
    p0, v0, a0 = course.ref(t)
    p1 = course.ref(t + LOOKAHEAD_S)[0]
    return np.concatenate([Rt @ (p0 - obs.p), Rt @ (v0 - obs.v), Rt @ a0, Rt @ (p1 - obs.p),
                           Rt[:, 2], Rt[:, 0], obs.w])


OBS_DIM = 21
ACT_DIM = 4


class MLPStudent(nn.Module):
    def __init__(self, hist: int, hidden: int = 128):
        super().__init__()
        self.hist = hist
        self.net = nn.Sequential(nn.Linear(OBS_DIM + ACT_DIM * hist, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, ACT_DIM))
        self.register_buffer("mu", torch.zeros(OBS_DIM + ACT_DIM * hist))
        self.register_buffer("sd", torch.ones(OBS_DIM + ACT_DIM * hist))

    def forward(self, x):
        return torch.tanh(self.net((x - self.mu) / self.sd))


def student_input(feat, act_hist: deque, hist: int):
    """act_hist holds the most recent actions, newest last."""
    past = list(act_hist)[-hist:][::-1]  # a_{t-1}, a_{t-2}, ...
    return np.concatenate([feat] + past)


def fit(model: MLPStudent, X, Y, epochs=60, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    Y = torch.as_tensor(np.asarray(Y), dtype=torch.float32)
    model.mu.copy_(X.mean(0))
    model.sd.copy_(X.std(0) + 1e-3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            loss = ((model(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return float(loss)


def perturbed_start(course: Course, rng, t0: float) -> DroneState:
    p, v, _ = course.ref(t0)
    return DroneState(p + rng.normal(0, 0.05, 3), np.eye(3), v + rng.normal(0, 0.15, 3), np.zeros(3))


def dagger(make_env, course: Course, hist: int, delay_sampler, iters=6, eps_per_iter=6, ep_len_s=6.0,
           act_noise=0.05, seed=0, label_fn=None):
    """DAgger in the source. The teacher label for a_t is the expert evaluated on the exact future state
    s_{t+d} reached after the in-flight actions (a perfect Smith predictor in the source model)."""
    rng = np.random.default_rng(seed)
    ex = GeometricExpert()
    model = MLPStudent(hist)
    X, Y = [], []
    for it in range(iters):
        beta = 1.0 if it == 0 else 0.0
        for ep in range(eps_per_iter):
            d = delay_sampler(rng)
            env = make_env(InterfaceConfig(action_delay=d), int(rng.integers(1 << 30)))
            t0 = float(rng.uniform(0, course.period))
            obs = env.reset(perturbed_start(course, rng, t0))
            ah = deque([env.hover_action.copy()] * max(hist, 1), maxlen=max(hist, 1))
            for k in range(int(ep_len_s / RL_DT)):
                t = t0 + k * RL_DT
                s_future = env.predict_forward(env.in_flight())
                pr, vr, ar = course.ref(t + d * RL_DT)
                y = ex(s_future, pr, vr, ar) if label_fn is None else label_fn(env, s_future, t + d * RL_DT)
                x = student_input(features(obs, course, t), ah, hist)
                X.append(x)
                Y.append(y)
                if rng.random() < beta:
                    a = y
                else:
                    with torch.no_grad():
                        a = model(torch.as_tensor(x, dtype=torch.float32)[None])[0].numpy()
                a = np.clip(a + act_noise * rng.standard_normal(4), -1, 1)
                obs = env.step(a)
                ah.append(a)
                st = env.true_state()
                if np.linalg.norm(st.p - course.ref(t + RL_DT)[0]) > 1.5:
                    break
        fit(model, X, Y, seed=seed + it)
    return model


@torch.no_grad()
def evaluate(policy_fn, make_env, course: Course, iface: InterfaceConfig, n_eps=3, laps=2, seed=1000, hist=8):
    """policy_fn(feat, act_hist, obs, t, env) -> action. Returns per-episode metrics."""
    rng = np.random.default_rng(seed)
    out = []
    for ep in range(n_eps):
        env = make_env(iface, seed + ep)
        obs = env.reset(perturbed_start(course, rng, 0.0))
        ah = deque([env.hover_action.copy()] * max(hist, 1), maxlen=max(hist, 1))
        T = int(laps * course.period / RL_DT)
        P, err, dA = [], [], []
        crashed = False
        for k in range(T):
            t = k * RL_DT
            a = np.clip(policy_fn(features(obs, course, t), ah, obs, t, env), -1, 1)
            dA.append(np.linalg.norm(a - ah[-1]))
            obs = env.step(a)
            ah.append(a)
            st = env.true_state()
            P.append(st.p)
            e = np.linalg.norm(st.p - course.ref(t + RL_DT)[0])
            err.append(e)
            if e > CRASH_ERR or st.p[2] < 0.05 or not np.all(np.isfinite(st.p)):
                crashed = True
                break
        out.append({
            "completion": len(P) / T,
            "crash": float(crashed),
            "gates": gate_pass_rate(course, np.array(P)) * len(P) / T,
            "rmse": float(np.sqrt(np.mean(np.square(err)))),
            "smooth": float(np.mean(dA)),
        })
    return out


def mlp_policy(model: MLPStudent):
    def f(feat, ah, obs, t, env):
        x = student_input(feat, ah, model.hist)
        return model(torch.as_tensor(x, dtype=torch.float32)[None])[0].numpy()
    return f


def expert_policy(course: Course):
    ex = GeometricExpert()

    def f(feat, ah, obs, t, env):
        return ex(obs, *course.ref(t))
    return f
