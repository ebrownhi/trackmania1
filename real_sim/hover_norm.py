"""
Hover-normalized collective-thrust action (Exp 27; SimpleFlight arXiv 2412.11764, Gräfe et al. arXiv 2603.05944).

Default CTBR action: c = 8 + 6 * a0 [m/s^2, mass-normalized to the fixed CTBRConfig mass], so hover needs
a0 = (g * m_true / m_cfg - 8) / 6 — a different value on every airframe, which the policy must learn to compensate.
Hover-normalized: c = c_hover * (1 + K * a0) with c_hover = g * m_true / m_cfg, so a0 = 0 hovers on ANY airframe of
known mass. K = 6 / g keeps the default slope at nominal mass, and c is clipped to the default envelope
[c_min, c_max] / g * c_hover, so only the hover offset changes.

Knowledge assumed: the airframe's true MASS (source: the sampled mass_scale; target: the PyBullet CF2X mass).
The thrust coefficient (L2F thrust_scale, PyBullet kf_scale) stays unknown to the normalization.
"""

from __future__ import annotations

import json

import numpy as np

from racing_env import GateRacingEnv
from sims.drone_pair import GRAVITY, CTBRConfig

K_SLOPE = 6.0 / GRAVITY


def true_mass(drone) -> float:
    be = drone.backend
    if hasattr(be, "_l2f"):  # L2FBackend: current (possibly randomized) dynamics
        return float(json.loads(be._l2f.parameters_to_json(be.device, be.env, be.params))["dynamics"]["mass"])
    return float(be.env.M)   # PyBullet CtrlAviary


def hover_norm_to_raw(a_pol: np.ndarray, c_hover: float, ctbr: CTBRConfig) -> np.ndarray:
    a_pol = np.clip(np.asarray(a_pol, dtype=np.float64), -1.0, 1.0)
    c = c_hover * (1.0 + K_SLOPE * a_pol[0])
    c = float(np.clip(c, ctbr.c_min / GRAVITY * c_hover, ctbr.c_max / GRAVITY * c_hover))
    a_raw = a_pol.copy()
    a_raw[0] = (c - 0.5 * (ctbr.c_min + ctbr.c_max)) / (0.5 * (ctbr.c_max - ctbr.c_min))
    return np.clip(a_raw, -1.0, 1.0)


class HoverNormDrone:
    """Proxy around a CTBRDrone: step() takes hover-normalized actions; hover_action is 0 in policy space."""

    def __init__(self, inner):
        self._inner = inner
        self.c_hover = GRAVITY

    @property
    def hover_action(self):
        return np.zeros(4)

    def reset(self, st):
        self.c_hover = GRAVITY * true_mass(self._inner) / self._inner.ctbr.mass
        return self._inner.reset(st)

    def step(self, a_pol):
        return self._inner.step(hover_norm_to_raw(a_pol, self.c_hover, self._inner.ctbr))

    def __getattr__(self, name):
        return getattr(self._inner, name)


class HoverNormRacingEnv(GateRacingEnv):
    """GateRacingEnv whose policy acts in hover-normalized thrust (action history is stored in policy space)."""

    def _get_drone(self, d: int):
        dr = super()._get_drone(d)
        if not isinstance(dr, HoverNormDrone):
            key = (d, self.obs_delay, self.p_drop, self.sensor_limited)
            dr = self.drones[key] = HoverNormDrone(dr)
        return dr
