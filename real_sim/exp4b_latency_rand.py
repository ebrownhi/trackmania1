"""
Exp 4b (Theorem 2.2 / H2'): action-history buffer and latency randomization, regime where delay matters in
the target too.

Regime: training figure-8 (v_peak 3.55 m/s), action delays d in {0, 4, 8, 12, 16} ticks (0-160 ms). The
naive expert (acts on o_t) survives the PyBullet target up to d ~ 10 and crashes at d >= 12-14, so the
large-delay cells are genuinely hard in the target. (The faster 4.4 m/s course was rejected: the expert
already crashes at d = 0 in L2F because it exceeds L2F's T/W ~ 1.47.)

Cells (3 seeds each, DAgger in L2F, teacher = expert on the exact future state s_{t+d}):
  fixed-d:      a_prev (a_{t-1}) vs buffer16 (a_{t-16:t-1}) trained and evaluated at the same d
  nominal:      buffer16 trained at d = 0, evaluated at all d
  latency-rand: d ~ U{0..16} per episode (student does not know d):
                  a_prev | buffer16 | buffer16 + obs-frame stack (o_t, o_{t-1}) | same + dynamics DR
Baselines: naive expert; Smith expert with exact source prediction (source only); Smith expert with a
source-model shadow whose hidden rotor rpm is reconstructed from the executed-command history (source and
target).

Metrics per episode (2 laps): crash (tracking error > 1 m), gate pass rate, RMSE until crash.
All aggregates are mean ± std over seeds (std, not SEM).
"""

import multiprocessing as mp
import time
from collections import deque

from common import OUT, np, plt, save_json

DELAYS = [0, 4, 8, 12, 16]
D_MAX = 16
SEEDS = [0, 1, 2]
N_WORKERS = 4
N_EVAL_EPS = 3


# ------------------------------------------------------------------------------------------------
# Student with optional observation-frame stack (local copy; policy_lib is shared with other agents)
# ------------------------------------------------------------------------------------------------

def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


def make_student(in_dim, hidden=128):
    torch = _torch()
    nn = torch.nn

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                     nn.Linear(hidden, 4))
            self.register_buffer("mu", torch.zeros(in_dim))
            self.register_buffer("sd", torch.ones(in_dim))

        def forward(self, x):
            return torch.tanh(self.net((x - self.mu) / self.sd))

    return Student()


def build_input(feat_hist: deque, act_hist: deque, hist: int, stack: int):
    f = list(feat_hist)[-stack:][::-1]          # o_t, o_{t-1}, ...
    a = list(act_hist)[-hist:][::-1]            # a_{t-1}, a_{t-2}, ...
    return np.concatenate(f + a)


def fit(model, X, Y, epochs=60, lr=2e-3, seed=0):
    torch = _torch()
    torch.manual_seed(seed)
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    Y = torch.as_tensor(np.asarray(Y), dtype=torch.float32)
    model.mu.copy_(X.mean(0))
    model.sd.copy_(X.std(0) + 1e-3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 512):
            idx = perm[i:i + 512]
            loss = ((model(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()


def sample_dr(rng):
    return {"mass_scale": rng.uniform(0.85, 1.15), "thrust_scale": rng.uniform(0.85, 1.25),
            "tau_rise_scale": rng.uniform(0.4, 1.2), "tau_fall_scale": rng.uniform(0.15, 1.2)}


def dagger(course, hist, stack, delay_sampler, dyn_dr=False, iters=6, eps_per_iter=6, ep_len_s=6.0,
           act_noise=0.05, seed=0):
    from policy_lib import OBS_DIM, features, perturbed_start
    from sims.drone_pair import GeometricExpert, InterfaceConfig, RL_DT, make_source
    torch = _torch()
    rng = np.random.default_rng(seed)
    ex = GeometricExpert()
    model = make_student(OBS_DIM * stack + 4 * hist)
    X, Y = [], []
    for it in range(iters):
        for ep in range(eps_per_iter):
            d = delay_sampler(rng)
            env = make_source(seed=int(rng.integers(1 << 30)), overrides=sample_dr(rng) if dyn_dr else None,
                              iface=InterfaceConfig(action_delay=d))
            t0 = float(rng.uniform(0, course.period))
            obs = env.reset(perturbed_start(course, rng, t0))
            f0 = features(obs, course, t0)
            fh = deque([f0] * stack, maxlen=stack)
            ah = deque([env.hover_action.copy()] * hist, maxlen=hist)
            for k in range(int(ep_len_s / RL_DT)):
                t = t0 + k * RL_DT
                s_future = env.predict_forward(env.in_flight())
                y = ex(s_future, *course.ref(t + d * RL_DT))
                x = build_input(fh, ah, hist, stack)
                X.append(x)
                Y.append(y)
                if it == 0:
                    a = y
                else:
                    with torch.no_grad():
                        a = model(torch.as_tensor(x, dtype=torch.float32)[None])[0].numpy()
                a = np.clip(a + act_noise * rng.standard_normal(4), -1, 1)
                obs = env.step(a)
                ah.append(a)
                fh.append(features(obs, course, t + RL_DT))
                if np.linalg.norm(env.true_state().p - course.ref(t + RL_DT)[0]) > 1.5:
                    break
        fit(model, X, Y, seed=seed + it)
    return model


# ------------------------------------------------------------------------------------------------
# Evaluation (local copy with frame stack + per-episode policy state)
# ------------------------------------------------------------------------------------------------

def make_env(domain, d, seed):
    from sims.drone_pair import InterfaceConfig, make_source, make_target
    iface = InterfaceConfig(action_delay=d)
    return make_source(seed=seed, iface=iface) if domain == "source" else make_target(iface=iface, seed=seed)


def evaluate(policy_factory, course, domain, d, seed, hist=16, stack=1, n_eps=N_EVAL_EPS, laps=2):
    """policy_factory() -> fresh per-episode callable(x, obs, t, env)."""
    from policy_lib import CRASH_ERR, features, perturbed_start
    from sims.drone_pair import RL_DT, gate_pass_rate
    rng = np.random.default_rng(seed)
    out = []
    for ep in range(n_eps):
        env = make_env(domain, d, seed + ep)
        obs = env.reset(perturbed_start(course, rng, 0.0))
        pol = policy_factory()
        fh = deque([features(obs, course, 0.0)] * stack, maxlen=stack)
        ah = deque([env.hover_action.copy()] * hist, maxlen=hist)
        T = int(laps * course.period / RL_DT)
        P, err = [], []
        crashed = False
        for k in range(T):
            t = k * RL_DT
            a = np.clip(pol(build_input(fh, ah, hist, stack), obs, t, env), -1, 1)
            obs = env.step(a)
            ah.append(a)
            fh.append(features(obs, course, t + RL_DT))
            st = env.true_state()
            P.append(st.p)
            e = np.linalg.norm(st.p - course.ref(t + RL_DT)[0])
            err.append(e)
            if e > CRASH_ERR or st.p[2] < 0.05 or not np.all(np.isfinite(st.p)):
                crashed = True
                break
        out.append({"crash": float(crashed), "gates": gate_pass_rate(course, np.array(P)) * len(P) / T,
                    "rmse": float(np.sqrt(np.mean(np.square(err))))})
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def student_factory(model):
    torch = _torch()

    def factory():
        def f(x, obs, t, env):
            with torch.no_grad():
                return model(torch.as_tensor(x, dtype=torch.float32)[None])[0].numpy()
        return f
    return factory


def naive_factory(course):
    from sims.drone_pair import GeometricExpert
    ex = GeometricExpert()
    return lambda: (lambda x, obs, t, env: ex(obs, *course.ref(t)))


def smith_exact_factory(course):
    """Exact source-model prediction from the env's own hidden state (only valid in the source)."""
    from sims.drone_pair import GeometricExpert, RL_DT
    ex = GeometricExpert()

    def factory():
        def f(x, obs, t, env):
            fl = env.in_flight()
            return ex(env.predict_forward(fl), *course.ref(t + len(fl) * RL_DT))
        return f
    return factory


def smith_recon_factory(course):
    """Source-model shadow; hidden L2F rotor rpm is reconstructed by running L2F's asymmetric rotor filter
    over the executed-command history (as in exp6 OneStepSource). Usable in any target."""
    from sims.drone_pair import CTBRConfig, GeometricExpert, GRAVITY, RL_DT, SUB_DT, SUBSTEPS, make_source
    ex = GeometricExpert()
    ctbr = CTBRConfig()
    shadow = make_source()
    be = shadow.backend
    tau_r, tau_f = be.motor_taus()
    rpm_hover = be.force_to_rpm(np.full(4, be.base_params["dynamics"]["mass"] * GRAVITY / 4.0))

    def factory():
        state = {"rpm": rpm_hover.copy()}

        def f(x, obs, t, env):
            # advance rotor estimate with the action executed during the last tick
            sp = be.force_to_rpm(ctbr.rotor_forces(env.applied, obs.w))
            rpm = state["rpm"]
            for _ in range(SUBSTEPS):
                tau = np.where(sp >= rpm, tau_r, tau_f)
                al = np.exp(-SUB_DT / tau)
                rpm = al * rpm + (1 - al) * sp
            state["rpm"] = rpm
            fl = env.in_flight()
            be.reset(obs, rpm=rpm)
            return ex(shadow.predict_forward(fl), *course.ref(t + len(fl) * RL_DT))
        return f
    return factory


# ------------------------------------------------------------------------------------------------
# Jobs
# ------------------------------------------------------------------------------------------------

STUDENTS = {
    # name: (hist, stack, delay mode, dyn_dr)
    "a_prev": (1, 1, "fixed", False),
    "buffer16": (16, 1, "fixed", False),
    "nominal_d0_buffer16": (16, 1, "d0", False),
    "rand_a_prev": (1, 1, "rand", False),
    "rand_buffer16": (16, 1, "rand", False),
    "rand_buffer16_stack2": (16, 2, "rand", False),
    "rand_buffer16_stack2_dynDR": (16, 2, "rand", True),
}


def student_job(args):
    from sims.drone_pair import Course
    name, d_train, seed = args
    hist, stack, mode, dyn = STUDENTS[name]
    course = Course()
    t0 = time.time()
    sampler = (lambda r: int(r.integers(0, D_MAX + 1))) if mode == "rand" else (lambda r: d_train)
    model = dagger(course, hist, stack, sampler, dyn_dr=dyn, seed=seed)
    eval_ds = [d_train] if mode == "fixed" else DELAYS
    ev = {d: {dom: evaluate(student_factory(model), course, dom, d, 3000 + 10 * seed, hist=hist, stack=stack)
              for dom in ["source", "target"]} for d in eval_ds}
    return {"name": name, "d_train": d_train, "seed": seed, "train_s": time.time() - t0, "eval": ev}


def baseline_job(args):
    from sims.drone_pair import Course
    name, d, dom = args
    course = Course()
    fac = {"naive_expert": naive_factory, "smith_exact": smith_exact_factory, "smith_recon": smith_recon_factory}[name]
    return name, d, dom, evaluate(fac(course), course, dom, d, 5000, n_eps=6)


def agg(vals):
    v = np.asarray(vals, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std()), "values": v.tolist()}


def main():
    t0 = time.time()
    bjobs = [(n, d, dom) for n in ["naive_expert", "smith_exact", "smith_recon"] for d in DELAYS
             for dom in ["source", "target"] if not (n == "smith_exact" and dom == "target")]
    sjobs = [(n, d, s) for n in ["a_prev", "buffer16"] for d in DELAYS if d > 0 for s in SEEDS]
    sjobs += [(n, 0, s) for n in STUDENTS if STUDENTS[n][2] in ("d0", "rand") for s in SEEDS]
    sjobs.sort(key=lambda j: -j[1] if STUDENTS[j[0]][2] == "fixed" else -D_MAX)  # longest first
    with mp.get_context("spawn").Pool(N_WORKERS) as pool:
        souts = pool.map(student_job, sjobs, chunksize=1)
        bouts = pool.map(baseline_job, bjobs, chunksize=1)
    wall = time.time() - t0

    res = {"delays": DELAYS, "d_max": D_MAX, "seeds": SEEDS, "wall_s": wall, "baselines": {}, "students": {}}
    for n, d, dom, m in bouts:
        res["baselines"].setdefault(n, {}).setdefault(dom, {})[d] = m
    for name, (hist, stack, mode, dyn) in STUDENTS.items():
        res["students"][name] = {}
        for dom in ["source", "target"]:
            res["students"][name][dom] = {}
            for d in DELAYS:
                if mode == "fixed":
                    o = [x for x in souts if x["name"] == name and x["d_train"] == d]
                    if d == 0:  # fixed-d students at d = 0 are identical to the nominal student
                        o = [x for x in souts if x["name"] == "nominal_d0_buffer16"]
                else:
                    o = [x for x in souts if x["name"] == name]
                res["students"][name][dom][d] = {m: agg([x["eval"][d][dom][m] for x in o]) for m in ["crash", "gates", "rmse"]}
    save_json("exp4b_latency_rand.json", res)

    print(f"Exp 4b: {len(sjobs)} students + {len(bjobs)} baseline cells in {wall:.0f}s on {N_WORKERS} workers")
    cell = lambda m: f"{m['crash']['mean']:.2f}±{m['crash']['std']:.2f} / {m['gates']['mean']:.2f}"
    bcell = lambda m: f"{m['crash']:.2f} / {m['gates']:.2f}"
    for dom in ["source", "target"]:
        print(f"\n[{dom}] crash rate ± std / gate pass rate   (baselines: 6 eps; students: 3 seeds x {N_EVAL_EPS} eps)")
        names = ["naive_expert", "smith_exact", "smith_recon"]
        print(f"{'d':>3} " + "".join(f"{n:>16}" for n in names if n in res["baselines"] and dom in res["baselines"][n]))
        for d in DELAYS:
            print(f"{d:>3} " + "".join(f"{bcell(res['baselines'][n][dom][d]):>16}" for n in names
                                       if n in res["baselines"] and dom in res["baselines"][n]))
        print(f"{'d':>3} " + "".join(f"{n:>28}" for n in STUDENTS))
        for d in DELAYS:
            print(f"{d:>3} " + "".join(f"{cell(res['students'][n][dom][d]):>28}" for n in STUDENTS))

    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, dom in enumerate(["source", "target"]):
        for i, metric in enumerate(["crash", "rmse"]):
            a = ax[i, j]
            for n, mk in [("naive_expert", "k--"), ("smith_exact", "k:"), ("smith_recon", "k-.")]:
                if dom in res["baselines"].get(n, {}):
                    a.plot(DELAYS, [res["baselines"][n][dom][d][metric] for d in DELAYS], mk, label=n)
            for n, mk in zip(STUDENTS, ["s-", "o-", "^-", "s--", "o--", "D--", "v--"]):
                m = [res["students"][n][dom][d][metric]["mean"] for d in DELAYS]
                s = [res["students"][n][dom][d][metric]["std"] for d in DELAYS]
                a.errorbar(np.array(DELAYS) + 0.15 * list(STUDENTS).index(n), m, yerr=s, fmt=mk, capsize=2, ms=4, label=n)
            a.set_title(f"{dom}: {metric}")
            if i == 1:
                a.set_xlabel("action delay d [10 ms ticks]")
    ax[0, 1].legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp4b_latency_rand.png", dpi=140)


if __name__ == "__main__":
    main()
