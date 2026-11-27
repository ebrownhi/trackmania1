"""
Racing env whose observation gives a sensor-limited PPO actor a track-free route to velocity (Exp 22).

Observation = [sensor-limited racing obs (21 + 4*hist) | last K_FF Exp 21 kinematic frames, newest first (10 each) |
privileged exact-state base obs (24)]. The privileged block is the last N_PRIV columns, so `AsymActorCriticPolicy`
(asym_policy.py) masks it for the actor and keeps or masks it for the critic.

Exp 20's actor only saw track-relative geometry (+ 3 stacked frames), from which velocity is recoverable mainly via
a course-position shortcut; here the kinematic frames (body-frame position differences, fresh-packet flag, attitude,
rates) let an MLP infer velocity without the track (cf. `kin_ff` in Exp 21).
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np

from asym_policy import N_PRIV
from kin_observer import K_FF, KIN_DIM, kin_feat
from racing_env import GateRacingEnv, build_racing_obs

KIN_STACK_DIM = K_FF * KIN_DIM


class KinStackRacingEnv(GateRacingEnv):
    def __init__(self, *args, **kw):
        kw["sensor_limited"] = True
        super().__init__(*args, **kw)
        self.n_sensor = self.observation_space.shape[0]
        n = self.n_sensor + KIN_STACK_DIM + N_PRIV
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)

    def _priv(self) -> np.ndarray:
        o, _ = build_racing_obs(self.drone.true_state(), self.cl, self.active_gate_idx(), self.true_hint, self.ah,
                                self.hist, sensor_limited=False)
        return o[:N_PRIV]

    def _wrap(self, o: np.ndarray) -> np.ndarray:
        f, self.last_p = kin_feat(self.obs_state, self.last_p)
        if self.kin is None:
            self.kin = deque([f] * K_FF, maxlen=K_FF)
        self.kin.append(f)
        return np.concatenate([o, *list(self.kin)[::-1], self._priv()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        o, info = super().reset(seed=seed, options=options)
        self.last_p, self.kin = None, None
        return self._wrap(o), info

    def step(self, action):
        o, r, term, trunc, info = super().step(action)
        return self._wrap(o), r, term, trunc, info
