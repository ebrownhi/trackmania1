"""
RL fine-tuning of the distilled sensor-limited student (Exp 28; Xing et al., CoRL 2024, arXiv 2403.12203:
privileged RL teacher -> imitation student -> RL fine-tuning of the student).

The Exp 21/23 student is a frozen exact-state teacher MLP acting on a teacher-format observation whose velocity slot
(and, with F2/F3, geometry) comes from the track-free GRU estimator. Fine-tuning keeps the estimator frozen in the
loop and trains the teacher MLP itself with PPO:

    StudentFTEnv  observation = [x_est (56) | x_exact (56)]
                  x_est   = the teacher input the student builds from the sensor stream (deployable)
                  x_exact = the teacher input built from the privileged exact state (critic only)
    actor   = AsymActorCriticPolicy with n_priv = 56: sees x_est only (x_exact masked)
    critic  = sees both blocks (asymmetric, privileged)
    init    = actor <- teacher policy net on the x_est columns; critic <- teacher value net on the x_exact columns;
              all other input columns zero, so at step 0 the deterministic action equals the student's exactly
              (tests/test_asym_policy.py::test_student_ft_init_reproduces_student).
"""

from __future__ import annotations

import numpy as np
import torch

from asym_policy import N_PRIV, PrivStackRacingEnv
from racing_env import SENSOR_BASE_DIM

TEACHER_DIM = 24 + 32  # exact base obs + 8-action history


class CaptureTeacher:
    """Wraps the frozen teacher; remembers the last teacher input so the student's own logic builds x_est."""

    def __init__(self, teacher):
        self.teacher, self.x = teacher, None

    def predict(self, x, deterministic=True):
        """Records the input only: in StudentFTEnv the student's own action is discarded (PPO acts), so the teacher
        forward pass is skipped (it was ~half the step cost)."""
        self.x = np.asarray(x, np.float32)
        return np.zeros(4, np.float32), None


def exact_teacher_obs(o: np.ndarray) -> np.ndarray:
    return np.concatenate([o[-N_PRIV:], o[SENSOR_BASE_DIM:SENSOR_BASE_DIM + 32]]).astype(np.float32)


class StudentFTEnv(PrivStackRacingEnv):
    """Racing env whose obs is [x_est | x_exact]. `student`: a stateful student (e.g. PlusStudent) owned by this env
    alone, whose `.teacher` is a CaptureTeacher; it is stepped exactly once per env step."""

    def __init__(self, *args, student=None, **kw):
        super().__init__(*args, **kw)
        self.student = student  # a PlusStudent / KinStudent whose `.teacher` is a CaptureTeacher
        import gymnasium as gym
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2 * TEACHER_DIM,), np.float32)

    def _ft_obs(self, o):
        self.student(o, self, self.k)  # steps the estimator once; CaptureTeacher records x_est
        return np.concatenate([self.student.teacher.x, exact_teacher_obs(o)]).astype(np.float32)

    def reset(self, seed=None, options=None):
        o, info = super().reset(seed=seed, options=options)
        self.student.reset_episode()
        return self._ft_obs(o), info

    def step(self, action):
        o, r, term, trunc, info = super().step(action)
        return self._ft_obs(o), r, term, trunc, info


@torch.no_grad()
def init_from_teacher(model, teacher, log_std_offset: float = 0.0):
    """Copy the teacher's actor onto the x_est columns and its critic onto the x_exact columns (zeros elsewhere)."""
    src, dst = teacher.policy, model.policy
    for net_name, cols in [("policy_net", slice(0, TEACHER_DIM)), ("value_net", slice(TEACHER_DIM, 2 * TEACHER_DIM))]:
        s, d = getattr(src.mlp_extractor, net_name), getattr(dst.mlp_extractor, net_name)
        d[0].weight.zero_()
        d[0].weight[:, cols] = s[0].weight
        d[0].bias.copy_(s[0].bias)
        d[2].load_state_dict(s[2].state_dict())
    dst.action_net.load_state_dict(src.action_net.state_dict())
    dst.value_net.load_state_dict(src.value_net.state_dict())
    dst.log_std.copy_(src.log_std + log_std_offset)
