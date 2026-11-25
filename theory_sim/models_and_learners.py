"""
Fast TorchScript-Fused Neural Policy Learners, Evaluation Loops, Online DAgger Distillation,
and Swift-style Detached k-NN Dynamics + GP Observation Residuals (~9s per seed!).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from envs import drone_step_jit, drone_obs_jit, tm_step_jit, tm_obs_jit


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 2, act_hist_len: int = 1, hidden: int = 48):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_hist_len = act_hist_len
        in_dim = obs_dim + act_dim * act_hist_len
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, act_dim)
        if act_hist_len > 0:
            with torch.no_grad():
                self.fc1.weight[:, obs_dim:] *= 0.20

    def forward(self, obs: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        if self.act_hist_len > 0:
            inp = torch.cat([obs, act_hist.reshape(B, -1)], dim=-1)
        else:
            inp = obs
        h = torch.tanh(self.fc1(inp))
        h = torch.tanh(self.fc2(h))
        return torch.tanh(self.fc3(h))


class GRUStudentPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 2, hidden: int = 48, aux_vel_dim: int = 3):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self.gru = nn.GRUCell(obs_dim + act_dim, hidden)
        self.fc1 = nn.Linear(hidden, 36)
        self.fc2 = nn.Linear(36, act_dim)
        self.vel_probe = nn.Linear(hidden, aux_vel_dim)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros((batch_size, self.hidden), dtype=torch.float32)

    def step(self, obs: torch.Tensor, prev_act: torch.Tensor, h: torch.Tensor):
        inp = torch.cat([obs, prev_act], dim=-1)
        h_next = self.gru(inp, h)
        act = torch.tanh(self.fc2(torch.tanh(self.fc1(h_next))))
        vel_hat = self.vel_probe(h_next.detach())
        return act, h_next, vel_hat


@torch.jit.script
def knn_predict_jit(
    Z_q: torch.Tensor,
    Z_norm: torch.Tensor,
    Y: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    k: int,
    bw: float,
) -> torch.Tensor:
    Zq_norm = (Z_q - mean) / std
    dists = torch.cdist(Zq_norm, Z_norm)
    d_knn, idx = torch.topk(dists, k=k, largest=False, dim=-1)
    weights = torch.softmax(-(d_knn * d_knn) / (2.0 * bw * bw), dim=-1)
    Y_nn = Y[idx]
    pred = (weights.unsqueeze(-1) * Y_nn).sum(dim=1)
    conf = torch.exp(-(d_knn[:, 0:1] * d_knn[:, 0:1]) / 18.0)
    return pred * conf


class KNNResidual:
    def __init__(self, k: int = 10, bandwidth: float = 0.80, max_pts: int = 450):
        self.k = k
        self.bandwidth = bandwidth
        self.max_pts = max_pts
        self.Z_norm = torch.zeros((1, 1))
        self.Y = torch.zeros((1, 4))
        self.mean = torch.zeros((1, 1))
        self.std = torch.ones((1, 1))
        self.k_eff = 1
        self.active = False

    def fit(self, Z: torch.Tensor, Y: torch.Tensor):
        if Z.shape[0] > self.max_pts:
            perm = torch.linspace(0, Z.shape[0] - 1, self.max_pts).long()
            Z = Z[perm]
            Y = Y[perm]
        self.mean = Z.mean(dim=0, keepdim=True).detach()
        self.std = Z.std(dim=0, keepdim=True).clamp(min=1e-3).detach()
        self.Z_norm = ((Z - self.mean) / self.std).detach()
        self.Y = Y.detach()
        self.k_eff = min(self.k, self.Z_norm.shape[0])
        self.active = True

    def predict(self, Z_q: torch.Tensor) -> torch.Tensor:
        return knn_predict_jit(Z_q, self.Z_norm, self.Y, self.mean, self.std, self.k_eff, self.bandwidth)


class GPObsResidual:
    def __init__(self, lengthscale: float = 1.1, noise_var: float = 1e-3):
        self.ls = lengthscale
        self.noise_var = noise_var
        self.X_norm = None
        self.alpha = None
        self.x_mean = None
        self.x_std = None

    def _rbf(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        dists_sq = torch.cdist(A, B) ** 2
        return torch.exp(-dists_sq / (2.0 * self.ls ** 2))

    def fit(self, X: torch.Tensor, Y: torch.Tensor, max_pts: int = 180):
        if X.shape[0] > max_pts:
            perm = torch.linspace(0, X.shape[0] - 1, max_pts).long()
            X = X[perm]
            Y = Y[perm]
        self.x_mean = X.mean(dim=0, keepdim=True)
        self.x_std = X.std(dim=0, keepdim=True).clamp(min=1e-3)
        self.X_norm = ((X - self.x_mean) / self.x_std).detach()
        K = self._rbf(self.X_norm, self.X_norm) + self.noise_var * torch.eye(X.shape[0])
        self.alpha = torch.linalg.solve(K, Y.detach())

    def __call__(self, dx: torch.Tensor, phi: torch.Tensor):
        Xq = torch.stack([dx, phi], dim=-1)
        Xq_norm = (Xq - self.x_mean) / self.x_std
        K_star = self._rbf(Xq_norm, self.X_norm)
        pred = K_star @ self.alpha
        return pred[:, 0], pred[:, 1]


def make_drone_knn_features(s: torch.Tensor, act_applied: torch.Tensor, act_lag_summary: torch.Tensor, use_history: bool):
    feat_s = torch.cat([s[:, 3:6] / 5.0, s[:, 6:7] * 1.5, s[:, 7:8] / 8.0], dim=-1)
    if not use_history:
        return torch.cat([feat_s, act_applied], dim=-1)
    else:
        return torch.cat([feat_s, act_applied, act_lag_summary], dim=-1)


@torch.jit.script
def drone_rollout_loss_jit(
    s0: torch.Tensor,
    delays: torch.Tensor,
    gust_yz: torch.Tensor,
    mass: torch.Tensor,
    t_scale: torch.Tensor,
    tau_m: torch.Tensor,
    k_w: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    c_blade: torch.Tensor,
    use_rk2: bool,
    gate_x: torch.Tensor,
    gate_y: torch.Tensor,
    gate_z: torch.Tensor,
    knot_x: torch.Tensor,
    knot_y: torch.Tensor,
    knot_z: torch.Tensor,
    w1: torch.Tensor, b1: torch.Tensor,
    w2: torch.Tensor, b2: torch.Tensor,
    w3: torch.Tensor, b3: torch.Tensor,
    act_hist_len: int,
    horizon: int,
    has_knn: bool,
    knn_use_hist: bool,
    Z_norm: torch.Tensor, Y_knn: torch.Tensor, z_mean: torch.Tensor, z_std: torch.Tensor, k_eff: int,
) -> torch.Tensor:
    B = s0.shape[0]
    s = s0
    act_buf = torch.zeros((B, 8, 2))
    b_idx = torch.arange(B)

    with torch.no_grad():
        zero_ext = torch.zeros((B, 4))
        for _ in range(5):
            obs_w = drone_obs_jit(s, gate_x, gate_y, gate_z)
            hist_w = act_buf[:, :act_hist_len, :].reshape(B, 2 * act_hist_len)
            inp_w = torch.cat([obs_w, hist_w], dim=-1)
            a_w = torch.tanh(F.linear(torch.tanh(F.linear(torch.tanh(F.linear(inp_w, w1, b1)), w2, b2)), w3, b3))
            act_buf = torch.cat([a_w.unsqueeze(1), act_buf[:, :-1, :]], dim=1)
            s = drone_step_jit(s, act_buf[b_idx, delays], mass, t_scale, tau_m, k_w, c1, c2, c_blade, use_rk2, zero_ext)

    total_loss = torch.zeros(1)
    for t in range(horizon):
        obs = drone_obs_jit(s, gate_x, gate_y, gate_z)
        hist = act_buf[:, :act_hist_len, :].reshape(B, 2 * act_hist_len)
        inp = torch.cat([obs, hist], dim=-1)
        h1 = torch.tanh(F.linear(inp, w1, b1))
        h2 = torch.tanh(F.linear(h1, w2, b2))
        a_cmd = torch.tanh(F.linear(h2, w3, b3))

        new_buf = torch.cat([a_cmd.unsqueeze(1), act_buf[:, :-1, :]], dim=1)
        a_applied = new_buf[b_idx, delays]

        ext_acc = torch.zeros((B, 4))
        ext_acc[:, 1] = gust_yz[:, 0]
        ext_acc[:, 2] = gust_yz[:, 1]
        if has_knn:
            with torch.no_grad():
                feat_s = torch.cat([s[:, 3:6] / 5.0, s[:, 6:7] * 1.5, s[:, 7:8] / 8.0], dim=-1)
                if knn_use_hist:
                    a_lag = (
                        0.55 * new_buf[b_idx, delays + 1]
                        + 0.30 * new_buf[b_idx, delays + 2]
                        + 0.15 * new_buf[b_idx, delays + 3]
                    )
                    zf = torch.cat([feat_s, a_applied, a_lag], dim=-1)
                else:
                    zf = torch.cat([feat_s, a_applied], dim=-1)
                knn_c = knn_predict_jit(zf, Z_norm, Y_knn, z_mean, z_std, k_eff, 0.80)
            ext_acc = ext_acc + knn_c

        s_next = drone_step_jit(s, a_applied, mass, t_scale, tau_m, k_w, c1, c2, c_blade, use_rk2, ext_acc)

        x, y, z = s_next[:, 0], s_next[:, 1], s_next[:, 2]
        vx, vy, vz = s_next[:, 3], s_next[:, 4], s_next[:, 5]
        phi, wx = s_next[:, 6], s_next[:, 7]

        xc = torch.clamp(x, 0.0, 28.79)
        idx = torch.clamp((xc / 4.8).long(), 0, 5)
        u = (xc - knot_x[idx]) / 4.8
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h01 = -2.0 * u**3 + 3.0 * u**2
        dh = 6.0 * u * (1.0 - u) / 4.8

        y_ref = knot_y[idx] * h00 + knot_y[idx + 1] * h01
        z_ref = knot_z[idx] * h00 + knot_z[idx + 1] * h01
        vy_ref = vx * (knot_y[idx + 1] - knot_y[idx]) * dh
        vz_ref = vx * (knot_z[idx + 1] - knot_z[idx]) * dh

        gate_focus = 1.0 + 15.0 * torch.exp(-(((x.unsqueeze(1) - gate_x.unsqueeze(0)) / 0.42) ** 2)).sum(dim=1)
        pos_err = (y - y_ref) ** 2 + 1.3 * (z - z_ref) ** 2
        vel_err = (vy - vy_ref) ** 2 + (vz - vz_ref) ** 2
        att_pen = 3.5 * F.relu(torch.abs(phi) - 0.95) ** 2 + 0.003 * (wx ** 2)
        act_smooth = 0.005 * ((a_cmd - act_buf[:, 0, :]) ** 2).sum(dim=-1) + 0.002 * (a_cmd ** 2).sum(dim=-1)

        active = (x < 25.2).float()
        step_l = active * (4.8 * gate_focus * pos_err + 0.32 * vel_err + att_pen + act_smooth)
        total_loss = total_loss + step_l.mean()

        s = 0.93 * s_next + 0.07 * s_next.detach()
        act_buf = 0.93 * new_buf + 0.07 * new_buf.detach()

    return total_loss / float(horizon)


@torch.jit.script
def tm_rollout_loss_jit(
    s0: torch.Tensor,
    delays: torch.Tensor,
    gust_lat: torch.Tensor,
    m: torch.Tensor,
    Iz: torch.Tensor,
    mu_f: torch.Tensor,
    mu_r: torch.Tensor,
    p_scale: torch.Tensor,
    v_opt_grid: torch.Tensor,
    wall_w: float,
    w1: torch.Tensor, b1: torch.Tensor,
    w2: torch.Tensor, b2: torch.Tensor,
    w3: torch.Tensor, b3: torch.Tensor,
    act_hist_len: int,
    horizon: int,
    has_knn: bool,
    knn_aligned: bool,
    Z_norm: torch.Tensor, Y_knn: torch.Tensor, z_mean: torch.Tensor, z_std: torch.Tensor, k_eff: int,
) -> torch.Tensor:
    B = s0.shape[0]
    s = s0
    act_buf = torch.zeros((B, 7, 2))
    b_idx = torch.arange(B)
    scale_vec = torch.tensor([[30.0, 3.0, 2.0, 0.35]])

    with torch.no_grad():
        zero_ext = torch.zeros((B, 3))
        for _ in range(5):
            obs_w = tm_obs_jit(s, wall_w)
            hist_w = act_buf[:, :act_hist_len, :].reshape(B, 2 * act_hist_len)
            inp_w = torch.cat([obs_w, hist_w], dim=-1)
            a_w = torch.tanh(F.linear(torch.tanh(F.linear(torch.tanh(F.linear(inp_w, w1, b1)), w2, b2)), w3, b3))
            act_buf = torch.cat([a_w.unsqueeze(1), act_buf[:, :-1, :]], dim=1)
            s = tm_step_jit(s, act_buf[b_idx, delays], m, Iz, mu_f, mu_r, p_scale, wall_w, zero_ext)

    total_loss = torch.zeros(1)
    for t in range(horizon):
        obs = tm_obs_jit(s, wall_w)
        hist = act_buf[:, :act_hist_len, :].reshape(B, 2 * act_hist_len)
        inp = torch.cat([obs, hist], dim=-1)
        h1 = torch.tanh(F.linear(inp, w1, b1))
        h2 = torch.tanh(F.linear(h1, w2, b2))
        a_cmd = torch.tanh(F.linear(h2, w3, b3))

        new_buf = torch.cat([a_cmd.unsqueeze(1), act_buf[:, :-1, :]], dim=1)
        a_applied = new_buf[b_idx, delays]

        ext_acc = torch.zeros((B, 3))
        ext_acc[:, 1] = gust_lat
        if has_knn:
            a_feat = a_applied if knn_aligned else a_cmd
            zf = torch.cat([s[:, 3:7].detach() / scale_vec, a_feat], dim=-1)
            knn_c = knn_predict_jit(zf, Z_norm, Y_knn, z_mean, z_std, k_eff, 0.80)
            if knn_aligned:
                knn_c = knn_c.detach()
            ext_acc = ext_acc + knn_c

        s_next = tm_step_jit(s, a_applied, m, Iz, mu_f, mu_r, p_scale, wall_w, ext_acc)

        s_arc, ey, epsi, vx, vy = s_next[:, 0], s_next[:, 1], s_next[:, 2], s_next[:, 3], s_next[:, 4]
        ds = (s_arc - s[:, 0]) * 100.0

        idx_f = torch.clamp(s_arc / 0.5, 0.0, 349.99)
        i0 = idx_f.long()
        frac = idx_f - i0.float()
        v_opt = v_opt_grid[i0] * (1.0 - frac) + v_opt_grid[i0 + 1] * frac

        speed_track = 0.06 * ((vx - v_opt) ** 2) + 0.30 * F.relu(vx - (v_opt + 0.8)) ** 2 - 0.16 * ds
        corridor_loss = (
            2.60 * (ey ** 2)
            + 26.0 * F.relu(torch.abs(ey) - (wall_w - 0.16)) ** 2
            + 65.0 * F.relu(torch.abs(ey) - wall_w) ** 2
        )
        stab_loss = 2.80 * (epsi ** 2) + 0.030 * (vy ** 2)
        act_smooth = 0.004 * ((a_cmd - act_buf[:, 0, :]) ** 2).sum(dim=-1)

        active = (s_arc < 152.0).float()
        step_l = active * (speed_track + corridor_loss + stab_loss + act_smooth)
        total_loss = total_loss + step_l.mean()

        s = 0.93 * s_next + 0.07 * s_next.detach()
        act_buf = 0.93 * new_buf + 0.07 * new_buf.detach()

    return total_loss / float(horizon)


def train_drone_policy(
    env,
    seed: int,
    act_hist_len: int = 1,
    dyn_rand: bool = False,
    lat_rand: bool = False,
    fixed_delay: int = 0,
    target_oracle: bool = False,
    knn_res: KNNResidual = None,
    knn_use_hist: bool = False,
    init_policy: MLPPolicy = None,
    iters: int = 50,
    batch_size: int = 44,
    horizon: int = 65,
    lr: float = 4.5e-3,
):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    policy = MLPPolicy(obs_dim=10, act_dim=2, act_hist_len=act_hist_len, hidden=48)
    if init_policy is not None:
        policy.load_state_dict(init_policy.state_dict())

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    has_knn = knn_res is not None and knn_res.active
    Z_norm = knn_res.Z_norm if has_knn else torch.zeros((2, 7))
    Y_knn = knn_res.Y if has_knn else torch.zeros((2, 4))
    z_mean = knn_res.mean if has_knn else torch.zeros((1, 7))
    z_std = knn_res.std if has_knn else torch.ones((1, 7))
    k_eff = knn_res.k_eff if has_knn else 1

    for _ in range(iters):
        if target_oracle:
            params = env.default_params(batch_size, alpha=1.0, randomize=False)
        elif dyn_rand:
            params = env.default_params(batch_size, alpha=0.0, randomize=True, generator=gen)
        else:
            params = env.default_params(batch_size, alpha=0.0, randomize=False)

        if lat_rand:
            delays = torch.randint(0, 5, (batch_size,), generator=gen)
        else:
            delays = torch.full((batch_size,), fixed_delay, dtype=torch.long)

        s0 = env.reset(batch_size, random_checkpoints=True, perturb_init=True, generator=gen)
        gust_yz = 0.9 * (torch.rand((batch_size, 2), generator=gen) - 0.5)

        opt.zero_grad()
        loss = drone_rollout_loss_jit(
            s0, delays, gust_yz,
            params["mass"], params["t_scale"], params["tau_m"],
            params["k_w"], params["c1"], params["c2"], params["c_blade"], params["use_rk2"],
            env.gate_x, env.gate_y, env.gate_z, env.knot_x, env.knot_y, env.knot_z,
            policy.fc1.weight, policy.fc1.bias,
            policy.fc2.weight, policy.fc2.bias,
            policy.fc3.weight, policy.fc3.bias,
            act_hist_len, horizon,
            has_knn, knn_use_hist, Z_norm, Y_knn, z_mean, z_std, k_eff,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
        opt.step()

    return policy


def train_tm_policy(
    env,
    seed: int,
    act_hist_len: int = 1,
    dyn_rand: bool = False,
    lat_rand: bool = False,
    fixed_delay: int = 0,
    knn_res: KNNResidual = None,
    knn_aligned: bool = True,
    init_policy: MLPPolicy = None,
    iters: int = 50,
    batch_size: int = 44,
    horizon: int = 65,
    lr: float = 4.5e-3,
):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 2000)
    policy = MLPPolicy(obs_dim=8, act_dim=2, act_hist_len=act_hist_len, hidden=48)
    if init_policy is not None:
        policy.load_state_dict(init_policy.state_dict())

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    has_knn = knn_res is not None and knn_res.active
    Z_norm = knn_res.Z_norm if has_knn else torch.zeros((2, 6))
    Y_knn = knn_res.Y if has_knn else torch.zeros((2, 3))
    z_mean = knn_res.mean if has_knn else torch.zeros((1, 6))
    z_std = knn_res.std if has_knn else torch.ones((1, 6))
    k_eff = knn_res.k_eff if has_knn else 1

    for _ in range(iters):
        params = env.default_params(batch_size, randomize_dynamics=dyn_rand, generator=gen)
        if lat_rand:
            delays = torch.randint(0, 5, (batch_size,), generator=gen)
        else:
            delays = torch.full((batch_size,), fixed_delay, dtype=torch.long)

        s0 = env.reset(batch_size, random_checkpoints=True, perturb_init=True, generator=gen)
        gust_lat = 1.4 * (torch.rand((batch_size,), generator=gen) - 0.5)

        opt.zero_grad()
        loss = tm_rollout_loss_jit(
            s0, delays, gust_lat,
            params["m"], params["Iz"], params["mu_f"], params["mu_r"], params["p_scale"],
            env.v_opt_grid, env.wall_w,
            policy.fc1.weight, policy.fc1.bias,
            policy.fc2.weight, policy.fc2.bias,
            policy.fc3.weight, policy.fc3.bias,
            act_hist_len, horizon,
            has_knn, knn_aligned, Z_norm, Y_knn, z_mean, z_std, k_eff,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
        opt.step()

    return policy


@torch.jit.script
def _eval_drone_mlp_core(
    s: torch.Tensor,
    gust_seq: torch.Tensor,
    mass: torch.Tensor, t_scale: torch.Tensor, tau_m: torch.Tensor, k_w: torch.Tensor,
    c1: torch.Tensor, c2: torch.Tensor, c_blade: torch.Tensor, use_rk2: bool,
    gate_x: torch.Tensor, gate_y: torch.Tensor, gate_z: torch.Tensor,
    gate_r: float, finish_x: float, max_steps: int, delay: int,
    w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor, w3: torch.Tensor, b3: torch.Tensor,
    act_hist_len: int,
):
    n_ep = s.shape[0]
    max_d = max(5, delay + 1)
    act_buf = torch.zeros((n_ep, max_d + 2, 2))
    crashed = torch.zeros(n_ep, dtype=torch.bool)
    gate_passed = torch.zeros((n_ep, 5), dtype=torch.bool)
    gate_err = torch.full((n_ep, 5), 0.90)
    finish_step = torch.full((n_ep,), float(max_steps))
    finished = torch.zeros(n_ep, dtype=torch.bool)
    smooth_acc = torch.zeros(n_ep)

    for t in range(max_steps):
        obs = drone_obs_jit(s, gate_x, gate_y, gate_z)
        hist_in = act_buf[:, :act_hist_len, :].reshape(n_ep, 2 * act_hist_len)
        inp = torch.cat([obs, hist_in], dim=-1)
        a_cmd = torch.tanh(F.linear(torch.tanh(F.linear(torch.tanh(F.linear(inp, w1, b1)), w2, b2)), w3, b3))

        smooth_acc = smooth_acc + ((a_cmd - act_buf[:, 0, :]) ** 2).sum(dim=-1).sqrt()
        new_buf = torch.cat([a_cmd.unsqueeze(1), act_buf[:, :-1, :]], dim=1)
        a_applied = new_buf[:, delay, :]
        s_next = drone_step_jit(s, a_applied, mass, t_scale, tau_m, k_w, c1, c2, c_blade, use_rk2, gust_seq[t])

        newly_crashed = (
            (torch.abs(s_next[:, 1]) > 1.75)
            | (torch.abs(s_next[:, 2]) > 1.50)
            | (torch.abs(s_next[:, 6]) > 1.52)
            | (s_next[:, 3] < 0.5)
        )
        crashed = crashed | (newly_crashed & (~finished))

        x0, x1 = s[:, 0], s_next[:, 0]
        y0, y1 = s[:, 1], s_next[:, 1]
        z0, z1 = s[:, 2], s_next[:, 2]
        for gi in range(5):
            gx, gy_g, gz_g = gate_x[gi], gate_y[gi], gate_z[gi]
            crossed = (x0 < gx) & (x1 >= gx) & (~crashed)
            if crossed.any():
                frac = (gx - x0) / torch.clamp(x1 - x0, min=1e-5)
                yc = y0 + frac * (y1 - y0)
                zc = z0 + frac * (z1 - z0)
                err = torch.sqrt((yc - gy_g) ** 2 + (zc - gz_g) ** 2)
                gate_err[:, gi] = torch.where(crossed, err, gate_err[:, gi])
                gate_passed[:, gi] = torch.where(crossed, err <= gate_r, gate_passed[:, gi])

        just_fin = (s_next[:, 0] >= finish_x) & (~finished) & (~crashed)
        finish_step = torch.where(just_fin, torch.full_like(finish_step, float(t + 1)), finish_step)
        finished = finished | just_fin
        s = s_next
        act_buf = new_buf

    return gate_passed, gate_err, finish_step, crashed, smooth_acc


@torch.no_grad()
def eval_drone_policy(
    env,
    policy,
    alpha: float = 1.0,
    delay: int = 0,
    n_episodes: int = 18,
    seed: int = 999,
    sensor_limited: bool = False,
    pnp_bias: float = 0.0,
    obs_noise: float = 0.0,
    gp_obs_corr=None,
    is_gru: bool = False,
    gust_std: float = 0.45,
):
    gen = torch.Generator().manual_seed(seed)
    params = env.default_params(n_episodes, alpha=alpha, randomize=False)
    s = env.reset(n_episodes, random_checkpoints=False, perturb_init=True, generator=gen)
    s[0] = env.reset(1, random_checkpoints=False, perturb_init=False)[0]

    gust_seq = torch.zeros((env.max_steps, n_episodes, 4), dtype=torch.float32)
    g_cur = torch.zeros((n_episodes, 4), dtype=torch.float32)
    for t in range(env.max_steps):
        if gust_std > 0.0 and t % 12 == 0:
            gy = gust_std * torch.randn((n_episodes,), generator=gen)
            gz = 0.7 * gust_std * torch.randn((n_episodes,), generator=gen)
            gy[0], gz[0] = 0.0, 0.0
            g_cur[:, 1] = 0.5 * g_cur[:, 1] + 0.5 * gy
            g_cur[:, 2] = 0.5 * g_cur[:, 2] + 0.5 * gz
        gust_seq[t] = g_cur

    if (not sensor_limited) and (not is_gru):
        gate_passed, gate_err, finish_step, crashed, smooth_acc = _eval_drone_mlp_core(
            s, gust_seq,
            params["mass"], params["t_scale"], params["tau_m"], params["k_w"],
            params["c1"], params["c2"], params["c_blade"], params["use_rk2"],
            env.gate_x, env.gate_y, env.gate_z, env.gate_r, env.finish_x, env.max_steps, delay,
            policy.fc1.weight, policy.fc1.bias, policy.fc2.weight, policy.fc2.bias, policy.fc3.weight, policy.fc3.bias,
            policy.act_hist_len,
        )
        vel_probe_errs = []
    else:
        max_d = max(5, delay + 1)
        act_buf = torch.zeros((n_episodes, max_d + 2, 2), dtype=torch.float32)
        h = policy.init_hidden(n_episodes) if is_gru else None

        crashed = torch.zeros(n_episodes, dtype=torch.bool)
        gate_passed = torch.zeros((n_episodes, 5), dtype=torch.bool)
        gate_err = torch.full((n_episodes, 5), 0.90, dtype=torch.float32)
        finish_step = torch.full((n_episodes,), env.max_steps, dtype=torch.float32)
        finished = torch.zeros(n_episodes, dtype=torch.bool)
        smooth_acc = torch.zeros(n_episodes)
        vel_probe_errs = []

        for t in range(env.max_steps):
            obs = env.get_obs(
                s,
                sensor_limited=sensor_limited,
                pnp_bias_scale=pnp_bias,
                obs_noise=obs_noise,
                gp_obs_corr=gp_obs_corr,
                generator=gen,
            )
            if is_gru:
                a_cmd, h, vel_hat = policy.step(obs, act_buf[:, 0, :], h)
                true_vel = torch.stack([(s[:, 3] - 7.0) / 2.5, s[:, 4] / 3.0, s[:, 5] / 2.5], dim=-1)
                active_mask = (~finished) & (~crashed)
                if active_mask.any():
                    vel_probe_errs.append(((vel_hat[active_mask] - true_vel[active_mask]) ** 2).mean().item())
            else:
                hist_in = act_buf[:, : policy.act_hist_len, :]
                a_cmd = policy(obs, hist_in)

            smooth_acc += ((a_cmd - act_buf[:, 0, :]) ** 2).sum(dim=-1).sqrt()
            new_buf = torch.cat([a_cmd.unsqueeze(1), act_buf[:, :-1, :]], dim=1)
            a_applied = new_buf[:, delay, :]
            s_next = env.step_physics(s, a_applied, params, ext_accel=gust_seq[t])

            newly_crashed = (
                (torch.abs(s_next[:, 1]) > 1.75)
                | (torch.abs(s_next[:, 2]) > 1.50)
                | (torch.abs(s_next[:, 6]) > 1.52)
                | (s_next[:, 3] < 0.5)
            )
            crashed = crashed | (newly_crashed & (~finished))

            x0, x1 = s[:, 0], s_next[:, 0]
            y0, y1 = s[:, 1], s_next[:, 1]
            z0, z1 = s[:, 2], s_next[:, 2]
            for gi in range(5):
                gx, gy_g, gz_g = env.gate_x[gi], env.gate_y[gi], env.gate_z[gi]
                crossed = (x0 < gx) & (x1 >= gx) & (~crashed)
                if crossed.any():
                    frac = (gx - x0) / torch.clamp(x1 - x0, min=1e-5)
                    yc = y0 + frac * (y1 - y0)
                    zc = z0 + frac * (z1 - z0)
                    err = torch.sqrt((yc - gy_g) ** 2 + (zc - gz_g) ** 2)
                    gate_err[crossed, gi] = err[crossed]
                    gate_passed[crossed, gi] = err[crossed] <= env.gate_r

            just_fin = (s_next[:, 0] >= env.finish_x) & (~finished) & (~crashed)
            finish_step[just_fin] = t + 1
            finished = finished | just_fin
            s = s_next
            act_buf = new_buf

    gate_pass_rate = gate_passed.float().mean().item()
    all_gates_clean = (gate_passed.all(dim=1) & (~crashed)).float().mean().item()
    mean_gate_err = gate_err.mean().item()
    mean_time = (finish_step * env.dt).mean().item()
    crash_rate = crashed.float().mean().item()

    speed_bonus = torch.clamp((4.15 - finish_step * env.dt) / 0.65, 0.0, 1.0)
    ep_score = (
        60.0 * gate_passed.float().mean(dim=1)
        + 25.0 * torch.clamp(1.0 - gate_err.mean(dim=1) / 0.55, 0.0, 1.0)
        + 15.0 * speed_bonus * (~crashed).float()
    )
    return {
        "score": ep_score.mean().item(),
        "gate_pass_rate": gate_pass_rate,
        "clean_completion": all_gates_clean,
        "gate_err_m": mean_gate_err,
        "finish_time_s": mean_time,
        "crash_rate": crash_rate,
        "smoothness": (smooth_acc / env.max_steps).mean().item(),
        "vel_probe_rmse": float(np.sqrt(np.mean(vel_probe_errs))) * 2.7 if vel_probe_errs else 0.0,
    }


@torch.jit.script
def _eval_tm_mlp_core(
    s: torch.Tensor,
    gust_seq: torch.Tensor,
    noise_seq: torch.Tensor,
    m: torch.Tensor, Iz: torch.Tensor, mu_f: torch.Tensor, mu_r: torch.Tensor, p_scale: torch.Tensor,
    wall_w: float, crash_w: float, track_len: float, dt: float, max_steps: int, delay: int,
    w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor, w3: torch.Tensor, b3: torch.Tensor,
    act_hist_len: int,
):
    n_ep = s.shape[0]
    max_d = max(5, delay + 1)
    act_buf = torch.zeros((n_ep, max_d + 2, 2))
    wall_hit = torch.zeros(n_ep, dtype=torch.bool)
    crashed = torch.zeros(n_ep, dtype=torch.bool)
    finished = torch.zeros(n_ep, dtype=torch.bool)
    finish_time = torch.full((n_ep,), float(max_steps) * dt)
    ey_sq_acc = torch.zeros(n_ep)
    active_steps = torch.zeros(n_ep)

    for t in range(max_steps):
        obs = tm_obs_jit(s, wall_w)
        hist_in = act_buf[:, :act_hist_len, :].reshape(n_ep, 2 * act_hist_len)
        inp = torch.cat([obs, hist_in], dim=-1)
        a_cmd = torch.tanh(F.linear(torch.tanh(F.linear(torch.tanh(F.linear(inp, w1, b1)), w2, b2)), w3, b3))
        a_noisy = torch.clamp(a_cmd + noise_seq[t], -1.0, 1.0)

        new_buf = torch.cat([a_noisy.unsqueeze(1), act_buf[:, :-1, :]], dim=1)
        a_applied = new_buf[:, delay, :]
        s_next = tm_step_jit(s, a_applied, m, Iz, mu_f, mu_r, p_scale, wall_w, gust_seq[t])

        act_m = (~finished) & (~crashed)
        ey_abs = torch.abs(s_next[:, 1])
        wall_hit = wall_hit | ((ey_abs > wall_w) & act_m)
        crashed = crashed | ((ey_abs > crash_w) & act_m)

        ey_sq_acc = ey_sq_acc + torch.where(act_m, s_next[:, 1] ** 2, torch.zeros_like(ey_abs))
        active_steps = active_steps + act_m.float()

        just_fin = (s_next[:, 0] >= track_len) & (~finished) & (~crashed)
        if just_fin.any():
            frac = (track_len - s[:, 0]) / torch.clamp(s_next[:, 0] - s[:, 0], min=1e-4)
            ft_cand = (float(t) + frac) * dt
            finish_time = torch.where(just_fin, ft_cand, finish_time)
        finished = finished | just_fin
        s = s_next
        act_buf = new_buf

    return s, finished, wall_hit, crashed, finish_time, ey_sq_acc, active_steps


@torch.no_grad()
def eval_tm_policy(
    env,
    policy,
    delay: int = 0,
    act_noise: float = 0.0,
    obs_dropout: float = 0.0,
    n_episodes: int = 18,
    seed: int = 999,
    sensor_limited: bool = False,
    cam_bias: float = 0.0,
    obs_noise: float = 0.0,
    is_gru: bool = False,
):
    gen = torch.Generator().manual_seed(seed)
    params = env.default_params(n_episodes, randomize_dynamics=False)
    s = env.reset(n_episodes, random_checkpoints=False, perturb_init=True, generator=gen)
    s[0] = env.reset(1, random_checkpoints=False, perturb_init=False)[0]

    gust_seq = torch.zeros((env.max_steps, n_episodes, 3), dtype=torch.float32)
    g_cur = torch.zeros((n_episodes, 3), dtype=torch.float32)
    for t in range(env.max_steps):
        if t % 15 == 0:
            gl = 0.55 * torch.randn((n_episodes,), generator=gen)
            gl[0] = 0.0
            g_cur[:, 1] = 0.5 * g_cur[:, 1] + 0.5 * gl
        gust_seq[t] = g_cur

    if (not sensor_limited) and (not is_gru) and (obs_dropout == 0.0):
        if act_noise > 0.0:
            noise_seq = act_noise * torch.randn((env.max_steps, n_episodes, 2), generator=gen)
        else:
            noise_seq = torch.zeros((env.max_steps, n_episodes, 2))
        s, finished, wall_hit, crashed, finish_time, ey_sq_acc, active_steps = _eval_tm_mlp_core(
            s, gust_seq, noise_seq,
            params["m"], params["Iz"], params["mu_f"], params["mu_r"], params["p_scale"],
            env.wall_w, env.crash_w, env.track_len, env.dt, env.max_steps, delay,
            policy.fc1.weight, policy.fc1.bias, policy.fc2.weight, policy.fc2.bias, policy.fc3.weight, policy.fc3.bias,
            policy.act_hist_len,
        )
        vel_probe_errs = []
    else:
        max_d = max(5, delay + 1)
        act_buf = torch.zeros((n_episodes, max_d + 2, 2), dtype=torch.float32)
        h = policy.init_hidden(n_episodes) if is_gru else None
        last_obs = None
        wall_hit = torch.zeros(n_episodes, dtype=torch.bool)
        crashed = torch.zeros(n_episodes, dtype=torch.bool)
        finished = torch.zeros(n_episodes, dtype=torch.bool)
        finish_time = torch.full((n_episodes,), env.max_steps * env.dt, dtype=torch.float32)
        ey_sq_acc = torch.zeros(n_episodes)
        active_steps = torch.zeros(n_episodes)
        vel_probe_errs = []

        for t in range(env.max_steps):
            raw_obs = env.get_obs(
                s, sensor_limited=sensor_limited, cam_bias_scale=cam_bias, obs_noise=obs_noise, generator=gen
            )
            if obs_dropout > 0.0 and last_obs is not None:
                drop_mask = torch.rand((n_episodes, 1), generator=gen) < obs_dropout
                obs = torch.where(drop_mask, last_obs, raw_obs)
            else:
                obs = raw_obs
            last_obs = obs

            if is_gru:
                a_cmd, h, vel_hat = policy.step(obs, act_buf[:, 0, :], h)
                true_slip = torch.stack([s[:, 4] / 3.0, s[:, 5] / 2.0], dim=-1)
                active_m = (~finished) & (~crashed)
                if active_m.any():
                    vel_probe_errs.append(((vel_hat[active_m] - true_slip[active_m]) ** 2).mean().item())
            else:
                hist_in = act_buf[:, : policy.act_hist_len, :]
                a_cmd = policy(obs, hist_in)

            if act_noise > 0.0:
                a_noisy = torch.clamp(a_cmd + act_noise * torch.randn(a_cmd.shape, generator=gen), -1.0, 1.0)
            else:
                a_noisy = a_cmd

            new_buf = torch.cat([a_noisy.unsqueeze(1), act_buf[:, :-1, :]], dim=1)
            a_applied = new_buf[:, delay, :]
            s_next = env.step_physics(s, a_applied, params, ext_accel=gust_seq[t])

            act_m = (~finished) & (~crashed)
            ey_abs = torch.abs(s_next[:, 1])
            wall_hit = wall_hit | ((ey_abs > env.wall_w) & act_m)
            crashed = crashed | ((ey_abs > env.crash_w) & act_m)

            ey_sq_acc += torch.where(act_m, s_next[:, 1] ** 2, torch.zeros_like(ey_abs))
            active_steps += act_m.float()

            just_fin = (s_next[:, 0] >= env.track_len) & (~finished) & (~crashed)
            if just_fin.any():
                frac = (env.track_len - s[:, 0]) / torch.clamp(s_next[:, 0] - s[:, 0], min=1e-4)
                finish_time[just_fin] = (t + frac[just_fin]) * env.dt
            finished = finished | just_fin
            s = s_next
            act_buf = new_buf

    completion = finished.float().mean().item()
    clean_completion = (finished & (~wall_hit)).float().mean().item()
    lat_rmse = torch.sqrt(ey_sq_acc / torch.clamp(active_steps, min=1.0)).mean().item()
    mean_time = finish_time.mean().item()

    prog_frac = torch.clamp(s[:, 0] / env.track_len, 0.0, 1.0)
    speed_score = torch.clamp((5.75 - finish_time) / 0.65, 0.0, 1.0)
    lat_acc_score = torch.clamp(1.0 - torch.sqrt(ey_sq_acc / torch.clamp(active_steps, min=1.0)) / 0.75, 0.0, 1.0)
    ep_score = (
        30.0 * prog_frac * (~crashed).float()
        + 40.0 * (finished & (~wall_hit)).float()
        + 15.0 * lat_acc_score * (~crashed).float()
        + 15.0 * speed_score * finished.float()
    )
    return {
        "score": ep_score.mean().item(),
        "completion": completion,
        "clean_completion": clean_completion,
        "finish_time_s": mean_time,
        "lat_rmse_m": lat_rmse,
        "crash_rate": crashed.float().mean().item(),
        "vel_probe_rmse": float(np.sqrt(np.mean(vel_probe_errs))) * 3.0 if vel_probe_errs else 0.0,
    }


def distill_drone_student(
    env,
    teacher: MLPPolicy,
    seed: int,
    recurrent: bool = True,
    randomize_source: bool = False,
    iters: int = 34,
    batch_size: int = 40,
    horizon: int = 60,
):
    torch.manual_seed(seed + 500)
    gen = torch.Generator().manual_seed(seed + 5000)
    if recurrent:
        student = GRUStudentPolicy(obs_dim=7, act_dim=2, hidden=48, aux_vel_dim=3)
    else:
        student = MLPPolicy(obs_dim=7, act_dim=2, act_hist_len=1, hidden=48)

    opt = torch.optim.Adam(student.parameters(), lr=4.5e-3)
    max_d = 4

    for it in range(iters):
        beta = max(0.0, 1.0 - it / 14.0)
        if randomize_source:
            params = env.default_params(batch_size, alpha=0.0, randomize=True, generator=gen)
            delays = torch.randint(0, 3, (batch_size,), generator=gen)
            pnp_scale = float(torch.rand(1, generator=gen).item() * 1.35)
            obs_noise = 0.02
        else:
            params = env.default_params(batch_size, alpha=0.0, randomize=False)
            delays = torch.zeros(batch_size, dtype=torch.long)
            pnp_scale = 0.0
            obs_noise = 0.0

        s = env.reset(batch_size, random_checkpoints=True, perturb_init=True, generator=gen)
        act_buf = torch.zeros((batch_size, max_d + 2, 2), dtype=torch.float32)
        h = student.init_hidden(batch_size) if recurrent else None
        b_idx = torch.arange(batch_size)
        gust = torch.zeros((batch_size, 4))
        gust[:, 1:3] = 0.6 * (torch.rand((batch_size, 2), generator=gen) - 0.5)

        with torch.no_grad():
            for _ in range(6):
                t_obs = env.get_obs(s, sensor_limited=False)
                a_targ = teacher(t_obs, act_buf[:, : teacher.act_hist_len, :])
                s_obs = env.get_obs(s, sensor_limited=True, pnp_bias_scale=pnp_scale, obs_noise=obs_noise, generator=gen)
                if recurrent:
                    _, h, _ = student.step(s_obs, act_buf[:, 0, :], h)
                act_buf = torch.cat([a_targ.unsqueeze(1), act_buf[:, :-1, :]], dim=1)
                s = env.step_physics(s, act_buf[b_idx, delays], params, ext_accel=gust)

        act_loss = 0.0
        probe_loss = 0.0
        for t in range(horizon):
            with torch.no_grad():
                t_obs = env.get_obs(s, sensor_limited=False)
                a_targ = teacher(t_obs, act_buf[:, : teacher.act_hist_len, :])

            s_obs = env.get_obs(
                s, sensor_limited=True, pnp_bias_scale=pnp_scale, obs_noise=obs_noise, generator=gen
            )
            if recurrent:
                a_pred, h, vel_hat = student.step(s_obs, act_buf[:, 0, :], h)
                true_vel = torch.stack([(s[:, 3] - 7.0) / 2.5, s[:, 4] / 3.0, s[:, 5] / 2.5], dim=-1).detach()
                probe_loss = probe_loss + F.mse_loss(vel_hat, true_vel)
                if t % 25 == 24:
                    h = h.detach()
            else:
                a_pred = student(s_obs, act_buf[:, :1, :])

            act_loss = act_loss + F.mse_loss(a_pred, a_targ)

            a_exec = beta * a_targ + (1.0 - beta) * a_pred.detach()
            new_buf = torch.cat([a_exec.unsqueeze(1), act_buf[:, :-1, :]], dim=1)
            a_applied = new_buf[b_idx, delays]
            with torch.no_grad():
                s = env.step_physics(s, a_applied, params, ext_accel=gust)
            act_buf = new_buf.detach()

        opt.zero_grad()
        ((act_loss + 0.25 * probe_loss) / horizon).backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
        opt.step()

    return student


def distill_tm_student(
    env,
    teacher: MLPPolicy,
    seed: int,
    recurrent: bool = True,
    randomize_interface: bool = False,
    iters: int = 34,
    batch_size: int = 40,
    horizon: int = 60,
):
    torch.manual_seed(seed + 600)
    gen = torch.Generator().manual_seed(seed + 6000)
    if recurrent:
        student = GRUStudentPolicy(obs_dim=6, act_dim=2, hidden=48, aux_vel_dim=2)
    else:
        student = MLPPolicy(obs_dim=6, act_dim=2, act_hist_len=1, hidden=48)

    opt = torch.optim.Adam(student.parameters(), lr=4.5e-3)
    max_d = 4

    for it in range(iters):
        beta = max(0.0, 1.0 - it / 14.0)
        params = env.default_params(batch_size, randomize_dynamics=False)
        if randomize_interface:
            delays = torch.randint(0, 4, (batch_size,), generator=gen)
            cam_bias = float(torch.rand(1, generator=gen).item() * 1.30)
            obs_noise = 0.03
            p_drop = 0.12
        else:
            delays = torch.zeros(batch_size, dtype=torch.long)
            cam_bias = 0.0
            obs_noise = 0.0
            p_drop = 0.0

        s = env.reset(batch_size, random_checkpoints=True, perturb_init=True, generator=gen)
        act_buf = torch.zeros((batch_size, max_d + 2, 2), dtype=torch.float32)
        h = student.init_hidden(batch_size) if recurrent else None
        last_obs = None
        b_idx = torch.arange(batch_size)

        with torch.no_grad():
            for _ in range(6):
                t_obs = env.get_obs(s, sensor_limited=False)
                a_targ = teacher(t_obs, act_buf[:, : teacher.act_hist_len, :])
                s_obs = env.get_obs(s, sensor_limited=True, cam_bias_scale=cam_bias, obs_noise=obs_noise, generator=gen)
                if recurrent:
                    _, h, _ = student.step(s_obs, act_buf[:, 0, :], h)
                last_obs = s_obs
                act_buf = torch.cat([a_targ.unsqueeze(1), act_buf[:, :-1, :]], dim=1)
                s = env.step_physics(s, act_buf[b_idx, delays], params)

        act_loss = 0.0
        probe_loss = 0.0
        for t in range(horizon):
            with torch.no_grad():
                t_obs = env.get_obs(s, sensor_limited=False)
                a_targ = teacher(t_obs, act_buf[:, : teacher.act_hist_len, :])

            raw_obs = env.get_obs(
                s, sensor_limited=True, cam_bias_scale=cam_bias, obs_noise=obs_noise, generator=gen
            )
            if p_drop > 0.0 and last_obs is not None:
                drop_m = torch.rand((batch_size, 1), generator=gen) < p_drop
                s_obs = torch.where(drop_m, last_obs, raw_obs)
            else:
                s_obs = raw_obs
            last_obs = s_obs.detach()

            if recurrent:
                a_pred, h, vel_hat = student.step(s_obs, act_buf[:, 0, :], h)
                true_slip = torch.stack([s[:, 4] / 3.0, s[:, 5] / 2.0], dim=-1).detach()
                probe_loss = probe_loss + F.mse_loss(vel_hat, true_slip)
                if t % 25 == 24:
                    h = h.detach()
            else:
                a_pred = student(s_obs, act_buf[:, :1, :])

            act_loss = act_loss + F.mse_loss(a_pred, a_targ)

            a_exec = beta * a_targ + (1.0 - beta) * a_pred.detach()
            new_buf = torch.cat([a_exec.unsqueeze(1), act_buf[:, :-1, :]], dim=1)
            a_applied = new_buf[b_idx, delays]
            with torch.no_grad():
                s = env.step_physics(s, a_applied, params)
            act_buf = new_buf.detach()

        opt.zero_grad()
        ((act_loss + 0.25 * probe_loss) / horizon).backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
        opt.step()

    return student
