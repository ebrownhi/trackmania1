"""
TorchScript-Accelerated Analytical Racing Micro-Simulators (16-831 Empirical Verification):
1. DroneSlalomEnv: Underactuated Quadrotor with CTBR [thrust, roll_rate] at 100 Hz policy + 500 Hz
   inner rate P-loop flying a high-speed 3D lateral-vertical 5-gate slalom (4th-order roll->y chain).
   - Source (L2F-like): Euler 500 Hz, tau_m = 0, nominal mass/thrust, linear drag only, nominal rate P-gain.
   - Target (PyBullet-like): RK2 500 Hz, motor spin-up lag tau_m = 0.03s, +12% mass, -10% thrust scale,
     quadratic + rotor blade drag, detuned inner rate gain, configurable delay d, and PnP-biased partial obs.
2. TrackmaniaCircuitEnv: 100 Hz Ground Racing Car on a 5-corner technical circuit with nonlinear
   Pacejka lateral tire grip saturation & speed-dependent understeer.
   - Source (TMNF-C-like) & Target (Live-TMNF-like) share IDENTICAL tick-for-tick physics (Delta f === 0).
   - Target interface gap: configurable action delay d in {0..4} ticks (0-40 ms), action noise sigma_a,
     observation dropout p_drop, and camera-rendered partial obs hiding slip velocity & yaw rate.
"""

import math
import torch
import torch.nn.functional as F


@torch.jit.script
def drone_step_jit(
    s: torch.Tensor,
    action: torch.Tensor,
    mass: torch.Tensor,
    t_scale: torch.Tensor,
    tau_m: torch.Tensor,
    k_w: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    c_blade: torch.Tensor,
    use_rk2: bool,
    ext_accel: torch.Tensor,
) -> torch.Tensor:
    a = torch.clamp(action, -1.0, 1.0)
    T_cmd = 13.5 + 11.0 * a[:, 0]
    wx_cmd = 14.0 * a[:, 1]

    x, y, z = s[:, 0], s[:, 1], s[:, 2]
    vx, vy, vz = s[:, 3], s[:, 4], s[:, 5]
    phi, wx = s[:, 6], s[:, 7]
    T_act, tau_wx = s[:, 8], s[:, 9]

    dt = 0.002
    g = 9.81
    sin_th0 = 0.29552
    cos_th0 = 0.95534

    has_lag = tau_m > 1e-5
    alpha_m = torch.where(has_lag, 1.0 - torch.exp(-dt / torch.clamp(tau_m, min=1e-4)), torch.ones_like(tau_m))

    for _ in range(5):
        T_act = T_act + alpha_m * (t_scale * T_cmd - T_act)
        tau_wx = tau_wx + alpha_m * (k_w * (wx_cmd - wx) - tau_wx)
        wx_dot = tau_wx + ext_accel[:, 3]

        if not use_rk2:
            cp, sp = torch.cos(phi), torch.sin(phi)
            spd = torch.sqrt(vx * vx + vy * vy + vz * vz + 1e-6)
            vh = torch.sqrt(vx * vx + vy * vy + 1e-6)
            drag_f = c1 + c2 * spd + c_blade * T_act * (vh / (spd + 0.5))

            ax = (T_act * cp * sin_th0) / mass - drag_f * vx + ext_accel[:, 0]
            ay = (T_act * sp) / mass - drag_f * vy + ext_accel[:, 1]
            az = (T_act * cp * cos_th0) / mass - g - (c1 + c2 * spd) * vz + ext_accel[:, 2]

            x = x + dt * vx
            y = y + dt * vy
            z = z + dt * vz
            vx = vx + dt * ax
            vy = vy + dt * ay
            vz = vz + dt * az
            phi = phi + dt * wx
            wx = wx + dt * wx_dot
        else:
            cp, sp = torch.cos(phi), torch.sin(phi)
            spd = torch.sqrt(vx * vx + vy * vy + vz * vz + 1e-6)
            vh = torch.sqrt(vx * vx + vy * vy + 1e-6)
            drag_f1 = c1 + c2 * spd + c_blade * T_act * (vh / (spd + 0.5))

            ax1 = (T_act * cp * sin_th0) / mass - drag_f1 * vx + ext_accel[:, 0]
            ay1 = (T_act * sp) / mass - drag_f1 * vy + ext_accel[:, 1]
            az1 = (T_act * cp * cos_th0) / mass - g - (c1 + c2 * spd) * vz + ext_accel[:, 2]

            vxm = vx + 0.5 * dt * ax1
            vym = vy + 0.5 * dt * ay1
            vzm = vz + 0.5 * dt * az1
            phim = phi + 0.5 * dt * wx
            wxm = wx + 0.5 * dt * wx_dot

            cpm, spm = torch.cos(phim), torch.sin(phim)
            spdm = torch.sqrt(vxm * vxm + vym * vym + vzm * vzm + 1e-6)
            vhm = torch.sqrt(vxm * vxm + vym * vym + 1e-6)
            drag_f2 = c1 + c2 * spdm + c_blade * T_act * (vhm / (spdm + 0.5))

            ax2 = (T_act * cpm * sin_th0) / mass - drag_f2 * vxm + ext_accel[:, 0]
            ay2 = (T_act * spm) / mass - drag_f2 * vym + ext_accel[:, 1]
            az2 = (T_act * cpm * cos_th0) / mass - g - (c1 + c2 * spdm) * vzm + ext_accel[:, 2]

            x = x + dt * vxm
            y = y + dt * vym
            z = z + dt * vzm
            vx = vx + dt * ax2
            vy = vy + dt * ay2
            vz = vz + dt * az2
            phi = phi + dt * wxm
            wx = wx + dt * wx_dot

    return torch.stack([x, y, z, vx, vy, vz, phi, wx, T_act, tau_wx], dim=-1)


@torch.jit.script
def drone_obs_jit(s: torch.Tensor, gate_x: torch.Tensor, gate_y: torch.Tensor, gate_z: torch.Tensor) -> torch.Tensor:
    x, y, z = s[:, 0], s[:, 1], s[:, 2]
    vx, vy, vz = s[:, 3], s[:, 4], s[:, 5]
    phi, wx = s[:, 6], s[:, 7]

    ahead = gate_x.unsqueeze(0) > (x.unsqueeze(1) - 0.12)
    passed = 5 - ahead.long().sum(dim=1)
    idx = torch.clamp(passed, 0, 4)
    idx_next = torch.clamp(passed + 1, 0, 4)

    dx = gate_x[idx] - x
    dy = gate_y[idx] - y
    dz = gate_z[idx] - z
    dy_next = gate_y[idx_next] - gate_y[idx]
    dz_next = gate_z[idx_next] - gate_z[idx]

    return torch.stack(
        [
            dx / 4.8,
            dy * 1.4,
            dz * 1.8,
            dy_next * 0.9,
            dz_next * 1.2,
            (vx - 7.0) / 2.5,
            vy / 3.0,
            vz / 2.5,
            phi * 1.5,
            wx / 8.0,
        ],
        dim=-1,
    )


@torch.jit.script
def tm_curvature_jit(s_arc: torch.Tensor) -> torch.Tensor:
    u1 = torch.clamp((s_arc - 24.0) / 10.5, -1.0, 1.0)
    k1 = 0.026 * (1.0 - u1 * u1) ** 2
    u2 = torch.clamp((s_arc - 55.0) / 8.5, -1.0, 1.0)
    k2 = -0.043 * (1.0 - u2 * u2) ** 2
    u3 = torch.clamp((s_arc - 73.0) / 8.5, -1.0, 1.0)
    k3 = 0.045 * (1.0 - u3 * u3) ** 2
    u4 = torch.clamp((s_arc - 92.0) / 8.5, -1.0, 1.0)
    k4 = -0.042 * (1.0 - u4 * u4) ** 2
    u5 = torch.clamp((s_arc - 124.0) / 12.0, -1.0, 1.0)
    k5 = 0.046 * (1.0 - u5 * u5) ** 2
    return k1 + k2 + k3 + k4 + k5


@torch.jit.script
def tm_body_accels_jit(
    s: torch.Tensor,
    action: torch.Tensor,
    m: torch.Tensor,
    Iz: torch.Tensor,
    mu_f: torch.Tensor,
    mu_r: torch.Tensor,
    p_scale: torch.Tensor,
    wall_w: float,
) -> torch.Tensor:
    a = torch.clamp(action, -1.0, 1.0)
    u_steer = a[:, 0]
    u_long = a[:, 1]

    ey = s[:, 1]
    vx = s[:, 3]
    vy = s[:, 4]
    r = s[:, 5]
    delta = s[:, 6]

    delta_max = 0.45 / (1.0 + (vx / 33.0) ** 2)
    delta_cmd = delta_max * u_steer
    d_delta = (delta_cmd - delta) / 0.007
    delta_eff = delta + 0.005 * d_delta

    vx_safe = torch.clamp(vx, min=4.0)
    lf = 1.35
    lr = 1.42
    alpha_f = delta_eff - torch.atan2(vy + lf * r, vx_safe)
    alpha_r = -torch.atan2(vy - lr * r, vx_safe)

    g = 9.81
    L = lf + lr
    Fz_f = m * g * (lr / L)
    Fz_r = m * g * (lf / L)

    throttle = F.relu(u_long)
    brake = F.relu(-u_long)
    Fx = p_scale * 16500.0 * throttle / (1.0 + vx_safe / 36.0) - 22000.0 * brake

    lat_scale = torch.sqrt(torch.clamp(1.0 - 0.32 * brake * brake, min=0.58))
    B, C, E = 13.5, 1.52, 0.42
    Baf = B * alpha_f
    Bar = B * alpha_r
    Fy_f = (mu_f * Fz_f * lat_scale) * torch.sin(C * torch.atan(Baf - E * (Baf - torch.atan(Baf))))
    Fy_r = (mu_r * Fz_r * lat_scale) * torch.sin(C * torch.atan(Bar - E * (Bar - torch.atan(Bar))))

    wall_pen = F.relu(torch.abs(ey) - wall_w)
    Fx_drag = 0.52 * vx_safe * vx + 11500.0 * wall_pen

    cos_d = torch.cos(delta_eff)
    sin_d = torch.sin(delta_eff)

    dvx = (Fx - Fy_f * sin_d - Fx_drag) / m + vy * r
    dvy = (Fy_f * cos_d + Fy_r) / m - vx * r
    dr = (lf * Fy_f * cos_d - lr * Fy_r) / Iz
    return torch.stack([dvx, dvy, dr, d_delta], dim=-1)


@torch.jit.script
def tm_step_jit(
    s: torch.Tensor,
    action: torch.Tensor,
    m: torch.Tensor,
    Iz: torch.Tensor,
    mu_f: torch.Tensor,
    mu_r: torch.Tensor,
    p_scale: torch.Tensor,
    wall_w: float,
    ext_accel: torch.Tensor,
) -> torch.Tensor:
    acc = tm_body_accels_jit(s, action, m, Iz, mu_f, mu_r, p_scale, wall_w)
    dvx = acc[:, 0] + ext_accel[:, 0]
    dvy = acc[:, 1] + ext_accel[:, 1]
    dr = acc[:, 2] + ext_accel[:, 2]
    d_delta = acc[:, 3]

    dt = 0.01
    s_arc, ey, epsi, vx, vy, r, delta, X, Y, psi = (
        s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5], s[:, 6], s[:, 7], s[:, 8], s[:, 9]
    )

    vx_n = torch.clamp(vx + dt * dvx, min=3.0, max=45.0)
    vy_n = torch.clamp(vy + dt * dvy, min=-12.0, max=12.0)
    r_n = torch.clamp(r + dt * dr, min=-5.0, max=5.0)
    delta_n = delta + dt * d_delta

    vxm = 0.5 * (vx + vx_n)
    vym = 0.5 * (vy + vy_n)
    rm = 0.5 * (r + r_n)

    kap = tm_curvature_jit(s_arc)
    denom = torch.clamp(1.0 - kap * ey, min=0.35)
    ds = (vxm * torch.cos(epsi) - vym * torch.sin(epsi)) / denom
    dey = vxm * torch.sin(epsi) + vym * torch.cos(epsi)
    depsi = rm - kap * ds

    s_arc_n = s_arc + dt * ds
    ey_n = ey + dt * dey
    epsi_n = epsi + dt * depsi

    X_n = X + dt * (vxm * torch.cos(psi) - vym * torch.sin(psi))
    Y_n = Y + dt * (vxm * torch.sin(psi) + vym * torch.cos(psi))
    psi_n = psi + dt * rm

    return torch.stack([s_arc_n, ey_n, epsi_n, vx_n, vy_n, r_n, delta_n, X_n, Y_n, psi_n], dim=-1)


@torch.jit.script
def tm_obs_jit(s: torch.Tensor, wall_w: float) -> torch.Tensor:
    s_arc, ey, epsi, vx, vy, r = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5]
    k_now = tm_curvature_jit(s_arc + 1.5) * 22.0
    k_near = tm_curvature_jit(s_arc + 8.0) * 22.0
    k_far = tm_curvature_jit(s_arc + 20.0) * 22.0
    return torch.stack(
        [ey / wall_w, epsi * 3.2, (vx - 26.0) / 8.0, vy / 3.0, r / 2.0, k_now, k_near, k_far],
        dim=-1,
    )


class DroneSlalomEnv:
    def __init__(self):
        self.dt = 0.01
        self.m0 = 1.0
        self.k_w_nom = 36.0
        self.c1_nom = 0.36

        self.gate_x = torch.tensor([4.8, 9.6, 14.4, 19.2, 24.0], dtype=torch.float32)
        self.gate_y = torch.tensor([0.65, -0.65, 0.68, -0.65, 0.58], dtype=torch.float32)
        self.gate_z = torch.tensor([0.34, -0.32, 0.36, -0.32, 0.26], dtype=torch.float32)
        self.gate_r = 0.32
        self.finish_x = 24.6
        self.max_steps = 415

        self.knot_x = torch.tensor([0.0, 4.8, 9.6, 14.4, 19.2, 24.0, 28.8], dtype=torch.float32)
        self.knot_y = torch.tensor([0.0, 0.65, -0.65, 0.68, -0.65, 0.58, 0.58], dtype=torch.float32)
        self.knot_z = torch.tensor([0.0, 0.34, -0.32, 0.36, -0.32, 0.26, 0.26], dtype=torch.float32)

    def ref_yz(self, x: torch.Tensor):
        xc = torch.clamp(x, 0.0, 28.79)
        idx = torch.clamp((xc / 4.8).long(), 0, 5)
        u = (xc - self.knot_x[idx]) / 4.8
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h01 = -2.0 * u**3 + 3.0 * u**2
        dh = 6.0 * u * (1.0 - u) / 4.8
        y_ref = self.knot_y[idx] * h00 + self.knot_y[idx + 1] * h01
        z_ref = self.knot_z[idx] * h00 + self.knot_z[idx + 1] * h01
        dy_dx = (self.knot_y[idx + 1] - self.knot_y[idx]) * dh
        dz_dx = (self.knot_z[idx + 1] - self.knot_z[idx]) * dh
        return y_ref, z_ref, dy_dx, dz_dx

    def default_params(self, batch_size: int, alpha: float = 0.0, randomize: bool = False, generator=None):
        if not randomize:
            mass = torch.full((batch_size,), self.m0 * (1.0 + 0.12 * alpha))
            t_scale = torch.full((batch_size,), 1.0 - 0.10 * alpha)
            tau_m = torch.full((batch_size,), 0.038 * alpha)
            k_w = torch.full((batch_size,), self.k_w_nom * (1.0 - 0.24 * alpha))
            c1 = torch.full((batch_size,), self.c1_nom)
            c2 = torch.full((batch_size,), 0.035 * alpha)
            c_blade = torch.full((batch_size,), 0.020 * alpha)
            use_rk2 = alpha > 0.0
        else:
            mass = self.m0 * (0.86 + 0.28 * torch.rand((batch_size,), generator=generator))
            t_scale = 0.87 + 0.23 * torch.rand((batch_size,), generator=generator)
            tau_m = 0.035 * torch.rand((batch_size,), generator=generator)
            k_w = torch.full((batch_size,), self.k_w_nom) * (0.74 + 0.34 * torch.rand((batch_size,), generator=generator))
            c1 = torch.full((batch_size,), self.c1_nom)
            c2 = 0.038 * torch.rand((batch_size,), generator=generator)
            c_blade = 0.022 * torch.rand((batch_size,), generator=generator)
            use_rk2 = True
        return {
            "mass": mass,
            "t_scale": t_scale,
            "tau_m": tau_m,
            "k_w": k_w,
            "c1": c1,
            "c2": c2,
            "c_blade": c_blade,
            "use_rk2": use_rk2,
        }

    def reset(self, batch_size: int, random_checkpoints: bool = False, perturb_init: bool = False, generator=None):
        s = torch.zeros((batch_size, 10), dtype=torch.float32)
        if random_checkpoints:
            x0 = 20.5 * torch.rand((batch_size,), generator=generator)
            yr, zr, dydx, dzdx = self.ref_yz(x0)
            s[:, 0] = x0
            s[:, 1] = yr
            s[:, 2] = zr
            s[:, 3] = 7.0
            s[:, 4] = 7.0 * dydx
            s[:, 5] = 7.0 * dzdx
        else:
            s[:, 3] = 6.2
        s[:, 8] = 9.81 / 0.95534
        if perturb_init:
            s[:, 1] += (torch.rand((batch_size,), generator=generator) - 0.5) * 0.30
            s[:, 2] += (torch.rand((batch_size,), generator=generator) - 0.5) * 0.24
            s[:, 4] += (torch.rand((batch_size,), generator=generator) - 0.5) * 1.10
            s[:, 5] += (torch.rand((batch_size,), generator=generator) - 0.5) * 0.90
            s[:, 6] += (torch.rand((batch_size,), generator=generator) - 0.5) * 0.32
            s[:, 7] += (torch.rand((batch_size,), generator=generator) - 0.5) * 2.00
        return s

    def step_physics(self, s: torch.Tensor, action: torch.Tensor, params: dict, ext_accel: torch.Tensor = None):
        if ext_accel is None:
            ext_accel = torch.zeros((s.shape[0], 4), dtype=torch.float32)
        return drone_step_jit(
            s, action,
            params["mass"], params["t_scale"], params["tau_m"],
            params["k_w"], params["c1"], params["c2"], params["c_blade"],
            params["use_rk2"], ext_accel
        )

    def get_obs(
        self,
        s: torch.Tensor,
        sensor_limited: bool = False,
        pnp_bias_scale: float = 0.0,
        obs_noise: float = 0.0,
        gp_obs_corr=None,
        generator=None,
    ):
        if not sensor_limited:
            return drone_obs_jit(s, self.gate_x, self.gate_y, self.gate_z)

        x, y, z = s[:, 0], s[:, 1], s[:, 2]
        phi, wx = s[:, 6], s[:, 7]
        ahead = self.gate_x.unsqueeze(0) > (x.unsqueeze(1) - 0.12)
        passed = 5 - ahead.long().sum(dim=1)
        idx = torch.clamp(passed, 0, 4)
        idx_next = torch.clamp(passed + 1, 0, 4)

        dx = self.gate_x[idx] - x
        dy = self.gate_y[idx] - y
        dz = self.gate_z[idx] - z
        dy_next = self.gate_y[idx_next] - self.gate_y[idx]
        dz_next = self.gate_z[idx_next] - self.gate_z[idx]

        if pnp_bias_scale > 0.0:
            bias_y = pnp_bias_scale * (0.065 * dx * torch.sin(phi) + 0.025 * dx)
            bias_z = pnp_bias_scale * (0.045 * dx * (1.0 - torch.cos(phi)) + 0.032 * dx)
        else:
            bias_y = 0.0
            bias_z = 0.0

        dy_obs = dy + bias_y
        dz_obs = dz + bias_z
        if obs_noise > 0.0:
            dy_obs = dy_obs + obs_noise * torch.randn(dy.shape, generator=generator)
            dz_obs = dz_obs + obs_noise * torch.randn(dz.shape, generator=generator)
        if gp_obs_corr is not None:
            corr_y, corr_z = gp_obs_corr(dx, phi)
            dy_obs = dy_obs - corr_y
            dz_obs = dz_obs - corr_z

        return torch.stack(
            [dx / 4.8, dy_obs * 1.4, dz_obs * 1.8, dy_next * 0.9, dz_next * 1.2, phi * 1.5, wx / 8.0],
            dim=-1,
        )


class TrackmaniaCircuitEnv:
    def __init__(self):
        self.dt = 0.01
        self.track_len = 146.0
        self.wall_w = 0.85
        self.crash_w = 2.05
        self.max_steps = 540

        self.m0 = 1200.0
        self.Iz0 = 1580.0
        self.mu_f0 = 2.32
        self.mu_r0 = 2.22

        s_grid = torch.linspace(0.0, 175.0, 351)
        kap = tm_curvature_jit(s_grid).abs()
        v_lat = torch.clamp(torch.sqrt(20.6 / (kap + 1e-4)), max=35.0)
        v_prof = v_lat.clone()
        ds = 0.5
        a_brk = 14.0
        for i in range(349, -1, -1):
            v_allow = math.sqrt(float(v_prof[i + 1] ** 2) + 2.0 * a_brk * ds)
            if v_allow < v_prof[i]:
                v_prof[i] = v_allow
        self.v_opt_grid = v_prof

    def opt_speed(self, s_arc: torch.Tensor) -> torch.Tensor:
        idx_f = torch.clamp(s_arc / 0.5, 0.0, 349.99)
        i0 = idx_f.long()
        frac = idx_f - i0.float()
        return self.v_opt_grid[i0] * (1.0 - frac) + self.v_opt_grid[i0 + 1] * frac

    def curvature(self, s_arc: torch.Tensor) -> torch.Tensor:
        return tm_curvature_jit(s_arc)

    def default_params(self, batch_size: int, randomize_dynamics: bool = False, generator=None):
        if not randomize_dynamics:
            return {
                "m": torch.full((batch_size,), self.m0),
                "Iz": torch.full((batch_size,), self.Iz0),
                "mu_f": torch.full((batch_size,), self.mu_f0),
                "mu_r": torch.full((batch_size,), self.mu_r0),
                "p_scale": torch.ones((batch_size,)),
            }
        else:
            return {
                "m": self.m0 * (0.82 + 0.36 * torch.rand((batch_size,), generator=generator)),
                "Iz": self.Iz0 * (0.82 + 0.36 * torch.rand((batch_size,), generator=generator)),
                "mu_f": self.mu_f0 * (0.68 + 0.42 * torch.rand((batch_size,), generator=generator)),
                "mu_r": self.mu_r0 * (0.68 + 0.42 * torch.rand((batch_size,), generator=generator)),
                "p_scale": 0.85 + 0.30 * torch.rand((batch_size,), generator=generator),
            }

    def reset(self, batch_size: int, random_checkpoints: bool = False, perturb_init: bool = False, generator=None):
        s = torch.zeros((batch_size, 10), dtype=torch.float32)
        if random_checkpoints:
            s0 = 126.0 * torch.rand((batch_size,), generator=generator)
            v0 = self.opt_speed(s0)
            kap0 = tm_curvature_jit(s0)
            s[:, 0] = s0
            s[:, 3] = v0
            s[:, 5] = v0 * kap0
            s[:, 6] = torch.clamp(2.77 * kap0, -0.35, 0.35)
        else:
            s[:, 0] = 0.0
            s[:, 3] = 28.0
        if perturb_init:
            s[:, 1] += (torch.rand((batch_size,), generator=generator) - 0.5) * 0.42
            s[:, 2] += (torch.rand((batch_size,), generator=generator) - 0.5) * 0.14
            s[:, 3] += (torch.rand((batch_size,), generator=generator) - 0.5) * 2.2
            s[:, 4] += (torch.rand((batch_size,), generator=generator) - 0.5) * 1.2
            s[:, 5] += (torch.rand((batch_size,), generator=generator) - 0.5) * 0.6
        return s

    def compute_body_accels(self, s: torch.Tensor, action: torch.Tensor, params: dict):
        acc = tm_body_accels_jit(
            s, action, params["m"], params["Iz"], params["mu_f"], params["mu_r"], params["p_scale"], self.wall_w
        )
        return acc[:, 0], acc[:, 1], acc[:, 2], acc[:, 3]

    def step_physics(self, s: torch.Tensor, action: torch.Tensor, params: dict, ext_accel: torch.Tensor = None):
        if ext_accel is None:
            ext_accel = torch.zeros((s.shape[0], 3), dtype=torch.float32)
        return tm_step_jit(
            s, action, params["m"], params["Iz"], params["mu_f"], params["mu_r"], params["p_scale"], self.wall_w, ext_accel
        )

    def get_obs(
        self,
        s: torch.Tensor,
        sensor_limited: bool = False,
        cam_bias_scale: float = 0.0,
        obs_noise: float = 0.0,
        generator=None,
    ):
        if not sensor_limited:
            return tm_obs_jit(s, self.wall_w)

        s_arc, ey, epsi, vx = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        k_now = tm_curvature_jit(s_arc + 1.5) * 22.0
        k_near = tm_curvature_jit(s_arc + 8.0) * 22.0
        k_far = tm_curvature_jit(s_arc + 20.0) * 22.0

        ey_obs = ey + cam_bias_scale * (0.18 * k_now)
        epsi_obs = epsi + cam_bias_scale * (0.035 * k_now)
        if obs_noise > 0.0:
            ey_obs = ey_obs + obs_noise * torch.randn(ey.shape, generator=generator)
            epsi_obs = epsi_obs + (0.2 * obs_noise) * torch.randn(epsi.shape, generator=generator)
        return torch.stack(
            [ey_obs / self.wall_w, epsi_obs * 3.2, (vx - 26.0) / 8.0, k_now, k_near, k_far],
            dim=-1,
        )
