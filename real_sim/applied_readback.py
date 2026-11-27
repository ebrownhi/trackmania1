"""
Applied-action readback (Exp 29; GT Sophy feeds "the game's view of the most recent steering/throttle/brake").

`CTBRDrone.applied` is the action the actuator layer is executing this tick (a_{t-d} under a fixed delay d). A
policy's usual action history is the *commanded* one, which says which action is in effect only if the policy
knows d. The readback history says it directly, whatever the delay.

    JitterDrone          CTBRDrone whose per-command delay is drawn from {d_min..d_max}; the actuator zero-order-
                         holds the newest command that has arrived (an older command never overrides a newer one).
                         A fixed delay d (d_min = d_max = d) reproduces CTBRDrone's FIFO exactly.
    ReadbackRacingEnv    GateRacingEnv whose action-history block is the commanded history ("cmd", the default
                         behaviour), the applied history ("app"), or both concatenated ("both"); optional jitter.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np

from racing_env import EXACT_BASE_DIM, SENSOR_BASE_DIM, GateRacingEnv
from sims.drone_pair import SUBSTEPS, CTBRDrone


class JitterDrone(CTBRDrone):
    def __init__(self, base: CTBRDrone, d_range: tuple[int, int], seed: int = 0):
        super().__init__(base.backend, base.ctbr, base.iface, seed)
        self.d_range = d_range
        self.jit_rng = np.random.default_rng(seed + 7919)

    def reset(self, st):
        obs = super().reset(st)
        self.k = 0
        self.pending = []  # (issue index, arrival tick, action)
        return obs

    def step(self, a):
        d = int(self.jit_rng.integers(self.d_range[0], self.d_range[1] + 1))
        self.pending.append((self.k, self.k + d, np.asarray(a, dtype=np.float64)))
        arrived = [p for p in self.pending if p[1] <= self.k]
        if arrived:
            newest = max(arrived, key=lambda p: p[0])
            self.applied = newest[2]
            self.pending = [p for p in self.pending if p[0] > newest[0]]
        self.backend.current_action = self.applied
        for _ in range(SUBSTEPS):
            w = self.backend.state().w
            self.backend.substep(self.ctbr.rotor_forces(self.applied, w))
        self.obs_fifo.append(self.backend.state())
        self.k += 1
        return self.observe()


class ReadbackRacingEnv(GateRacingEnv):
    def __init__(self, *args, hist_mode: str = "cmd", jitter: tuple[int, int] | None = None, **kw):
        super().__init__(*args, **kw)
        assert hist_mode in ("cmd", "app", "both")
        self.hist_mode, self.jitter = hist_mode, jitter
        self.base_dim = SENSOR_BASE_DIM if self.sensor_limited else EXACT_BASE_DIM
        n = self.base_dim + 4 * self.hist * (2 if hist_mode == "both" else 1)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)
        self.jitter_drones = {}

    def _get_drone(self, d: int):
        base = super()._get_drone(d)
        if self.jitter is None:
            return base
        if d not in self.jitter_drones:
            self.jitter_drones[d] = JitterDrone(base, self.jitter, seed=self._seed)
        return self.jitter_drones[d]

    def _wrap(self, o: np.ndarray) -> np.ndarray:
        if self.hist_mode == "cmd":
            return o
        app = np.concatenate(list(self.app_hist)[-self.hist:][::-1]).astype(np.float32)
        if self.hist_mode == "app":
            return np.concatenate([o[:self.base_dim], app])
        return np.concatenate([o, app])

    def reset(self, seed=None, options=None):
        o, info = super().reset(seed=seed, options=options)
        self.app_hist = deque([self.drone.hover_action.astype(np.float64).copy()] * max(1, self.hist),
                              maxlen=max(1, self.hist))
        return self._wrap(o), info

    def step(self, action):
        o, r, term, trunc, info = super().step(action)
        self.app_hist.append(np.asarray(self.drone.applied, dtype=np.float64).copy())
        info["applied"] = self.app_hist[-1].copy()
        return self._wrap(o), r, term, trunc, info
