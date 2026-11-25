"""
Exp 5 (Theorem 2.4 / H3' / guardrail #4): recurrent-observer OOD bias, L2F source -> PyBullet target.

Sensor-limited observation (own channel, obs delay 1 tick, 10% ZOH dropout + validity bit):
    noisy position (sigma 2 cm), exact attitude, noisy body rates (0.05 rad/s), and velocity either
      - "novel": removed entirely, or
      - "vio":   noisy (0.1 m/s) + slowly drifting bias (stationary std ~0.2 m/s) VIO velocity.

Part A (supervised probe, the clean test of Thm 2.4): estimate true body-frame velocity R^T v from the
sensor stream + action history with
      FF   = MLP over the last 4 sensor frames + actions,
      GRU  = recurrent estimator over the whole stream,
  each trained on (fixed) nominal L2F or on L2F with randomized dynamics (mass, thrust gain, rotor lag).
  Evaluate RMSE on held-out source flights and on PyBullet target flights.
Part B (closed loop): DAgger-distill sensor-limited students (FF_fixed, FF_rand, GRU_fixed, GRU_rand) from
  the privileged GeometricExpert (exact state) in L2F and evaluate zero-shot in PyBullet.
  J = -(effective RMSE), effective RMSE = rmse * completion + 1.0 m * (1 - completion);
  rho_obs = (J_GRU - J_FF) / (J_teacher - J_FF).
"""

import multiprocessing as mp
import time
from collections import deque

import torch
import torch.nn as nn

from common import OUT, np, plt, save_json, mean_std
from sims.drone_pair import Course, DroneState, GeometricExpert, RL_DT, gate_pass_rate, make_source, make_target

torch.set_num_threads(1)
SEEDS = [0, 1, 2]
MODES = ["novel", "vio"]
K_WIN = 4
LOOKAHEAD_S = 0.3
CRASH_ERR = 1.0


# ------------------------------------------------------------------------------------------------
# Sensor channel and features
# ------------------------------------------------------------------------------------------------

class Sensor:
    def __init__(self, mode, rng, delay=1, p_drop=0.1):
        self.mode, self.rng, self.delay, self.p_drop = mode, rng, delay, p_drop

    def reset(self, st):
        self.fifo = deque([st.copy()] * (self.delay + 1), maxlen=self.delay + 1)
        self.drift = np.zeros(3)
        self.last = self._read(st)
        return self.last, 1.0

    def _read(self, st):
        r = self.rng
        y = DroneState(st.p + 0.02 * r.standard_normal(3), st.R.copy(), st.v.copy(), st.w + 0.05 * r.standard_normal(3))
        self.drift = 0.995 * self.drift + 0.02 * r.standard_normal(3)
        y.v = st.v + self.drift + 0.1 * r.standard_normal(3) if self.mode == "vio" else np.zeros(3)
        return y

    def observe(self, st_true):
        self.fifo.append(st_true.copy())
        if self.rng.random() < self.p_drop:
            return self.last, 0.0
        self.last = self._read(self.fifo[0])
        return self.last, 1.0


def sensor_features(y, valid, course, t, mode):
    Rt = y.R.T
    p0, v0, a0 = course.ref(t)
    p1 = course.ref(t + LOOKAHEAD_S)[0]
    f = [Rt @ (p0 - y.p), Rt @ v0, Rt @ a0, Rt @ (p1 - y.p), Rt[:, 2], Rt[:, 0], y.w]
    if mode == "vio":
        f.append(Rt @ y.v)
    f.append([valid])
    return np.concatenate(f)


def feat_dim(mode):
    return 22 + (3 if mode == "vio" else 0)


def sample_dyn(rng):
    return {"mass_scale": rng.uniform(0.8, 1.2), "thrust_scale": rng.uniform(0.8, 1.25),
            "tau_rise_scale": rng.uniform(0.3, 1.5), "tau_fall_scale": rng.uniform(0.08, 1.5)}


def make_env(domain, rng, seed):
    if domain == "src_fixed":
        return make_source(seed=seed)
    if domain == "src_rand":
        return make_source(seed=seed, overrides=sample_dyn(rng))
    return make_target(seed=seed)


def perturbed_start(course, rng, t0):
    p, v, _ = course.ref(t0)
    return DroneState(p + rng.normal(0, 0.05, 3), np.eye(3), v + rng.normal(0, 0.15, 3), np.zeros(3))


# ------------------------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------------------------

class Norm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sd", torch.ones(d))

    def set(self, X):
        self.mu.copy_(X.mean(0))
        self.sd.copy_(X.std(0) + 1e-3)

    def forward(self, x):
        return (x - self.mu) / self.sd


class FFNet(nn.Module):
    """MLP over a window of the last K_WIN (features, prev action) frames."""

    def __init__(self, d_in, d_out, hidden=128, squash=False):
        super().__init__()
        self.norm = Norm(d_in * K_WIN)
        self.net = nn.Sequential(nn.Linear(d_in * K_WIN, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, d_out))
        self.squash = squash

    def forward(self, seq):  # seq: [B, T, d_in] -> [B, T, d_out]
        B, T, D = seq.shape
        pad = seq[:, :1].expand(B, K_WIN - 1, D)
        s = torch.cat([pad, seq], 1)
        win = torch.cat([s[:, k:k + T] for k in range(K_WIN)][::-1], -1)  # newest first
        y = self.net(self.norm(win))
        return torch.tanh(y) if self.squash else y


class GRUNet(nn.Module):
    def __init__(self, d_in, d_out, hidden=64, squash=False):
        super().__init__()
        self.norm = Norm(d_in)
        self.inp = nn.Linear(d_in, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, d_out))
        self.squash = squash

    def forward(self, seq, h=None, return_h=False):
        z, h = self.gru(torch.tanh(self.inp(self.norm(seq))), h)
        y = self.out(z)
        y = torch.tanh(y) if self.squash else y
        return (y, h) if return_h else y


def train_seq(model, seqs_x, seqs_y, steps=400, lr=3e-3, seed=0, chunk=200):
    """Train on variable-length episodes, sampling random chunks (truncated BPTT from episode start of chunk)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    allx = torch.as_tensor(np.concatenate(seqs_x), dtype=torch.float32)
    model.norm.set(allx if isinstance(model, GRUNet) else allx.repeat(1, K_WIN))
    X = [torch.as_tensor(x, dtype=torch.float32) for x in seqs_x]
    Y = [torch.as_tensor(y, dtype=torch.float32) for y in seqs_y]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(steps):
        idx = rng.integers(len(X), size=16)
        bx, by, bm = [], [], []
        for i in idx:
            T = len(X[i])
            L = min(chunk, T)
            s = int(rng.integers(0, T - L + 1))
            px = torch.zeros(chunk, X[i].shape[1]); py = torch.zeros(chunk, Y[i].shape[1]); pm = torch.zeros(chunk)
            px[:L], py[:L], pm[:L] = X[i][s:s + L], Y[i][s:s + L], 1.0
            bx.append(px); by.append(py); bm.append(pm)
        bx, by, bm = torch.stack(bx), torch.stack(by), torch.stack(bm)
        pred = model(bx)
        loss = (((pred - by) ** 2).mean(-1) * bm).sum() / bm.sum()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return float(loss)


class StepRunner:
    """Run an FFNet / GRUNet one step at a time."""

    def __init__(self, model):
        self.m, self.h, self.buf = model, None, deque(maxlen=K_WIN)

    @torch.no_grad()
    def __call__(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        if isinstance(self.m, GRUNet):
            y, self.h = self.m(x[None, None], self.h, return_h=True)
            return y[0, 0].numpy()
        if not self.buf:
            self.buf.extend([x] * K_WIN)
        self.buf.append(x)
        return self.m(torch.stack(list(self.buf))[None])[0, -1].numpy()


# ------------------------------------------------------------------------------------------------
# Rollouts
# ------------------------------------------------------------------------------------------------

def rollout(domain, mode, course, rng, seed, dur_s, student=None, act_noise=0.1, t0=None):
    """Fly one episode. Control = expert on exact state (student None) or student on sensor stream.
    Returns per-step inputs X (sensor feat + prev action), probe targets V (true body vel), expert labels Y,
    and metrics."""
    env = make_env(domain, rng, seed)
    ex = GeometricExpert()
    sens = Sensor(mode, np.random.default_rng(seed + 77))
    t0 = float(rng.uniform(0, course.period)) if t0 is None else t0
    st0 = perturbed_start(course, rng, t0)
    env.reset(st0)
    y, valid = sens.reset(env.true_state())
    a_prev = env.hover_action.copy()
    runner = StepRunner(student) if student is not None else None
    X, V, Y, P, err = [], [], [], [], []
    T = int(dur_s / RL_DT)
    crashed = False
    for k in range(T):
        t = t0 + k * RL_DT
        st = env.true_state()
        x = np.concatenate([sensor_features(y, valid, course, t, mode), a_prev])
        a_exp = ex(st, *course.ref(t))
        X.append(x); V.append(st.R.T @ st.v); Y.append(a_exp)
        a = a_exp if runner is None else runner(x)
        a = np.clip(a + act_noise * rng.standard_normal(4), -1, 1)
        env.step(a)
        a_prev = a
        st = env.true_state()
        y, valid = sens.observe(st)
        P.append(st.p)
        e = np.linalg.norm(st.p - course.ref(t + RL_DT)[0])
        err.append(e)
        if e > CRASH_ERR or st.p[2] < 0.05 or not np.all(np.isfinite(st.p)):
            crashed = True
            break
    comp = len(P) / T
    rmse = float(np.sqrt(np.mean(np.square(err))))
    met = {"completion": comp, "crash": float(crashed), "rmse": rmse,
           "eff_rmse": rmse * comp + CRASH_ERR * (1 - comp),
           "gates": gate_pass_rate(course, np.array(P), t0=0.0) * comp if t0 == 0.0 else float("nan")}
    return np.array(X), np.array(V), np.array(Y), met


# ------------------------------------------------------------------------------------------------
# Part A: supervised velocity probe
# ------------------------------------------------------------------------------------------------

def probe_job(args):
    mode, seed = args
    course = Course()
    rng = np.random.default_rng(seed)
    data = {}
    for j, (dom, n) in enumerate([("src_fixed", 24), ("src_rand", 24), ("test_src", 6), ("test_tgt", 6)]):
        d = {"test_src": "src_fixed", "test_tgt": "target"}.get(dom, dom)
        eps = [rollout(d, mode, course, rng, 10_000 * seed + 1000 * j + i, 5.0) for i in range(n)]
        data[dom] = ([e[0] for e in eps], [e[1] for e in eps])
    out = {}
    D = feat_dim(mode) + 4
    for train in ["src_fixed", "src_rand"]:
        for arch, cls in [("FF", FFNet), ("GRU", GRUNet)]:
            m = cls(D, 3)
            train_seq(m, *data[train], steps=600, seed=seed)
            r = {}
            for test in ["test_src", "test_tgt"]:
                with torch.no_grad():
                    errs = [((m(torch.as_tensor(x, dtype=torch.float32)[None])[0].numpy() - v)[50:]) for x, v in zip(*data[test])]
                r[test] = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
            out[f"{arch}_{'fixed' if train == 'src_fixed' else 'rand'}"] = r
    # trivial reference: predict the reference velocity in body frame (what a no-memory policy "knows")
    return mode, seed, out


# ------------------------------------------------------------------------------------------------
# Part B: closed-loop DAgger students
# ------------------------------------------------------------------------------------------------

def dagger_job(args):
    mode, arch, train_dom, seed = args
    course = Course()
    rng = np.random.default_rng(1000 + seed)
    D = feat_dim(mode) + 4
    model = (GRUNet if arch == "GRU" else FFNet)(D, 4, squash=True)
    Xs, Ys = [], []
    for it in range(5):
        for ep in range(8):
            X, _, Y, _ = rollout(train_dom, mode, course, rng, 50_000 * seed + 100 * it + ep, 6.0,
                                 student=None if it == 0 else model, act_noise=0.05)
            Xs.append(X); Ys.append(Y)
        train_seq(model, Xs, Ys, steps=500 if it < 4 else 900, seed=seed + it)
    ev = {}
    for dom in ["src_fixed", "target"]:
        mets = [rollout(dom, mode, course, np.random.default_rng(7000 + seed * 10 + e), 7000 + seed * 10 + e,
                        2 * course.period, student=model, act_noise=0.0, t0=0.0)[3] for e in range(3)]
        ev[dom] = {k: float(np.mean([m[k] for m in mets])) for k in mets[0]}
    return mode, arch, train_dom, seed, ev


def teacher_job(seed):
    course = Course()
    ev = {}
    for dom in ["src_fixed", "target"]:
        mets = [rollout(dom, "novel", course, np.random.default_rng(7000 + seed * 10 + e), 7000 + seed * 10 + e,
                        2 * course.period, student=None, act_noise=0.0, t0=0.0)[3] for e in range(3)]
        ev[dom] = {k: float(np.mean([m[k] for m in mets])) for k in mets[0]}
    return seed, ev


def main():
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        probe = pool.map(probe_job, [(m, s) for m in MODES for s in SEEDS])
        print(f"Part A done in {time.time() - t0:.0f}s", flush=True)
        teach = dict(pool.map(teacher_job, SEEDS))
        jobs = [(m, a, d, s) for m in MODES for a in ["FF", "GRU"] for d in ["src_fixed", "src_rand"] for s in SEEDS]
        dag = pool.map(dagger_job, jobs)
    print(f"Part B done in {time.time() - t0:.0f}s", flush=True)

    res = {"probe": {}, "closed_loop": {}, "teacher": {}, "rho_obs": {}}
    for m in MODES:
        res["probe"][m] = {}
        for key in probe[0][2]:
            res["probe"][m][key] = {t: mean_std([o[key][t] for mm, s, o in probe if mm == m]) for t in ["test_src", "test_tgt"]}
    for dom in ["src_fixed", "target"]:
        res["teacher"][dom] = {k: mean_std([teach[s][dom][k] for s in SEEDS]) for k in teach[0][dom]}
    for m in MODES:
        res["closed_loop"][m] = {}
        for a in ["FF", "GRU"]:
            for d in ["src_fixed", "src_rand"]:
                name = f"{a}_{'fixed' if d == 'src_fixed' else 'rand'}"
                runs = {s: ev for mm, aa, dd, s, ev in dag if (mm, aa, dd) == (m, a, d)}
                res["closed_loop"][m][name] = {dom: {k: mean_std([runs[s][dom][k] for s in SEEDS]) for k in runs[0][dom]}
                                               for dom in ["src_fixed", "target"]}
                res["closed_loop"][m][name]["_per_seed_target_eff"] = [runs[s]["target"]["eff_rmse"] for s in SEEDS]
        # rho_obs per seed with J = -eff_rmse in the target; FF reference = FF trained on the same source
        res["rho_obs"][m] = {}
        for tr in ["fixed", "rand"]:
            rhos = []
            for i, s in enumerate(SEEDS):
                J_t = -teach[s]["target"]["eff_rmse"]
                J_ff = -res["closed_loop"][m][f"FF_{tr}"]["_per_seed_target_eff"][i]
                J_rnn = -res["closed_loop"][m][f"GRU_{tr}"]["_per_seed_target_eff"][i]
                rhos.append((J_rnn - J_ff) / (J_t - J_ff) if abs(J_t - J_ff) > 1e-3 else float("nan"))
            res["rho_obs"][m][f"GRU_{tr}_vs_FF_{tr}"] = rhos
        res["meta"] = {"wall_s": time.time() - t0, "seeds": SEEDS}
    save_json("exp5_recurrent_observer.json", res)

    print("\nPart A: body-velocity estimation RMSE [m/s] (mean ± std over 3 seeds)")
    for m in MODES:
        print(f"  [{m}]")
        for key, r in res["probe"][m].items():
            a, b = r["test_src"], r["test_tgt"]
            print(f"    {key:10s} source {a['mean']:.3f} ± {a['std']:.3f}   target {b['mean']:.3f} ± {b['std']:.3f}   "
                  f"(OOD ratio {b['mean'] / a['mean']:.2f}x)")
    print("\nPart B: closed loop, target (PyBullet) — eff_rmse [m] / gates / crash")
    t = res["teacher"]["target"]
    print(f"  teacher (exact state): {t['eff_rmse']['mean']:.3f} ± {t['eff_rmse']['std']:.3f} / {t['gates']['mean']:.2f} / {t['crash']['mean']:.2f}")
    for m in MODES:
        print(f"  [{m}]")
        for name, r in res["closed_loop"][m].items():
            tg, sc = r["target"], r["src_fixed"]
            print(f"    {name:9s} tgt {tg['eff_rmse']['mean']:.3f} ± {tg['eff_rmse']['std']:.3f} / {tg['gates']['mean']:.2f} / "
                  f"{tg['crash']['mean']:.2f}    (src {sc['eff_rmse']['mean']:.3f} / {sc['gates']['mean']:.2f})")
        for k, v in res["rho_obs"][m].items():
            print(f"    rho_obs {k}: {np.round(v, 2).tolist()}")

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    keys = ["FF_fixed", "GRU_fixed", "FF_rand", "GRU_rand"]
    x = np.arange(len(keys))
    for i, m in enumerate(MODES):
        for j, test in enumerate(["test_src", "test_tgt"]):
            vals = [res["probe"][m][k][test]["mean"] for k in keys]
            errs = [res["probe"][m][k][test]["std"] for k in keys]
            ax[0].bar(x + (2 * i + j - 1.5) * 0.2, vals, 0.2, yerr=errs, capsize=2, label=f"{m} / {test[5:]}")
    ax[0].set_xticks(x, keys)
    ax[0].set_ylabel("body-velocity RMSE [m/s]")
    ax[0].set_title("Part A: observer probe")
    ax[0].legend(fontsize=7)
    for i, m in enumerate(MODES):
        vals = [res["closed_loop"][m][k]["target"]["eff_rmse"]["mean"] for k in keys]
        errs = [res["closed_loop"][m][k]["target"]["eff_rmse"]["std"] for k in keys]
        ax[1].bar(x + (i - 0.5) * 0.35, vals, 0.35, yerr=errs, capsize=2, label=m)
    ax[1].axhline(res["teacher"]["target"]["eff_rmse"]["mean"], c="k", ls="--", label="exact-state teacher")
    ax[1].set_xticks(x, keys)
    ax[1].set_ylabel("target effective RMSE [m] (crash = 1 m)")
    ax[1].set_title("Part B: closed loop in PyBullet")
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp5_recurrent_observer.png", dpi=140)


if __name__ == "__main__":
    main()
