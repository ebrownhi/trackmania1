"""
Asymmetric actor-critic for sensor-limited racing PPO (proposal: "a privileged critic that may retain exact state").

`PrivStackRacingEnv` observation = [sensor-limited obs (21 + 4*hist) | previous K sensor base frames (21 each) |
privileged exact-state base obs (24)]. The privileged block is the *last* N_PRIV columns.

`AsymActorCriticPolicy` uses separate, parameter-free masking feature extractors:
    actor  always sees the privileged block zeroed           (deployable: sensor-only)
    critic sees it when `critic_privileged=True`, zeroed otherwise (the ablation)
Masking (rather than slicing) keeps one features_dim, which SB3's MlpExtractor requires.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from racing_env import EXACT_BASE_DIM, SENSOR_BASE_DIM, GateRacingEnv, build_racing_obs

N_PRIV = EXACT_BASE_DIM
K_PREV = 3


class PrivStackRacingEnv(GateRacingEnv):
    def __init__(self, *args, k_prev: int = K_PREV, **kw):
        kw["sensor_limited"] = True
        super().__init__(*args, **kw)
        self.k_prev = k_prev
        n = self.observation_space.shape[0] + k_prev * SENSOR_BASE_DIM + N_PRIV
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)

    def _priv(self) -> np.ndarray:
        o, _ = build_racing_obs(self.drone.true_state(), self.cl, self.active_gate_idx(), self.true_hint, self.ah,
                                self.hist, sensor_limited=False)
        return o[:N_PRIV]

    def _wrap(self, o: np.ndarray) -> np.ndarray:
        out = np.concatenate([o, *list(self.prev)[::-1], self._priv()]).astype(np.float32)
        self.prev.append(o[:SENSOR_BASE_DIM].copy())
        return out

    def reset(self, seed=None, options=None):
        o, info = super().reset(seed=seed, options=options)
        self.prev = deque([o[:SENSOR_BASE_DIM].copy()] * self.k_prev, maxlen=self.k_prev)
        return self._wrap(o), info

    def step(self, action):
        o, r, term, trunc, info = super().step(action)
        return self._wrap(o), r, term, trunc, info


class MaskExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, n_priv: int, zero_priv: bool):
        super().__init__(observation_space, features_dim=int(observation_space.shape[0]))
        mask = th.ones(self.features_dim)
        if zero_priv:
            mask[-n_priv:] = 0.0
        self.register_buffer("mask", mask)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return obs * self.mask


class AsymActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, n_priv: int = N_PRIV, critic_privileged: bool = True, **kw):
        kw["share_features_extractor"] = False
        super().__init__(*args, **kw)
        self.n_priv, self.critic_privileged = n_priv, critic_privileged
        self.pi_features_extractor = MaskExtractor(self.observation_space, n_priv, zero_priv=True)
        self.features_extractor = self.pi_features_extractor
        self.vf_features_extractor = MaskExtractor(self.observation_space, n_priv, zero_priv=not critic_privileged)

    def _get_constructor_parameters(self):
        d = super()._get_constructor_parameters()
        d.update(n_priv=self.n_priv, critic_privileged=self.critic_privileged)
        return d
