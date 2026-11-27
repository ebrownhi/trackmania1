"""Applied-action readback (Exp 29): the readback must equal what the actuator FIFO executes."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "real_sim"))

from applied_readback import JitterDrone, ReadbackRacingEnv  # noqa: E402
from sims.drone_pair import DroneState, InterfaceConfig, make_source  # noqa: E402


def _actions(n, seed=0):
    return np.clip(np.random.default_rng(seed).normal(0, 0.3, (n, 4)), -1, 1)


def test_applied_history_equals_fifo_executed_actions():
    d = 3
    env = ReadbackRacingEnv("figure8", "source", 8, delays=(d,), hist_mode="both", random_start_gate=False, seed=0)
    o, _ = env.reset(seed=1)
    A = _actions(20)
    for k, a in enumerate(A):
        o, *_ = env.step(a)
        expected = A[k - d] if k >= d else env.drone.hover_action
        assert np.allclose(env.app_hist[-1], expected)
    app_block = o[-32:].reshape(8, 4)  # newest first
    assert np.allclose(app_block[0], A[len(A) - 1 - d])
    assert np.allclose(o[24:56].reshape(8, 4)[0], A[-1])  # commanded block keeps the newest command


def test_app_mode_layout_and_fixed_jitter_matches_fifo():
    env = ReadbackRacingEnv("figure8", "source", 8, delays=(0,), hist_mode="app", seed=0)
    o, _ = env.reset(seed=2)
    assert o.shape == env.observation_space.shape == (24 + 32,)
    st = DroneState.hover(p=np.array([0.0, 0.0, 1.5]))
    ref = make_source(seed=0, iface=InterfaceConfig(action_delay=2))
    jit = JitterDrone(make_source(seed=0), (2, 2), seed=0)
    ref.reset(st), jit.reset(st)
    for a in _actions(25, seed=3):
        ref.step(a), jit.step(a)
        assert np.allclose(ref.applied, jit.applied)
    assert np.allclose(ref.true_state().p, jit.true_state().p)


def test_jitter_never_applies_an_older_command_after_a_newer_one():
    jit = JitterDrone(make_source(seed=0), (1, 4), seed=5)
    jit.reset(DroneState.hover(p=np.array([0.0, 0.0, 1.5])))
    A = _actions(60, seed=4)
    last_idx = -1
    for a in A:
        jit.step(a)
        hit = [i for i, b in enumerate(A) if np.array_equal(b, jit.applied)]
        if hit:
            assert hit[0] >= last_idx
            last_idx = hit[0]
