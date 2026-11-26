"""Asymmetric actor-critic: the deployable actor must never see the privileged exact-state block."""

import os
import sys

import numpy as np
import torch as th

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "real_sim"))

from stable_baselines3 import PPO  # noqa: E402

from asym_policy import N_PRIV, AsymActorCriticPolicy, PrivStackRacingEnv  # noqa: E402


def _model(critic_privileged):
    env = PrivStackRacingEnv("figure8", "source", 8, obs_delay=2, p_drop=0.2, seed=0)
    return env, PPO(AsymActorCriticPolicy, env, n_steps=64, batch_size=64, seed=0, device="cpu",
                    policy_kwargs=dict(critic_privileged=critic_privileged))


def test_priv_block_is_exact_state():
    env, _ = _model(True)
    o, _ = env.reset(seed=1)
    for _ in range(5):
        o, *_ = env.step(env.action_space.sample() * 0.1)
    v_B = env.drone.true_state().R.T @ env.drone.true_state().v
    assert np.allclose(o[-N_PRIV:][15:18], 0.33 * v_B, atol=1e-5)  # privileged block carries true body velocity
    assert o.shape == env.observation_space.shape


def test_actor_blind_critic_sees_priv_only_when_enabled():
    for priv in [True, False]:
        env, m = _model(priv)
        o, _ = env.reset(seed=2)
        o2 = o.copy()
        o2[-N_PRIV:] += 5.0
        x = th.as_tensor(np.stack([o, o2]))
        with th.no_grad():
            mu = m.policy.get_distribution(x).distribution.mean
            v = m.policy.predict_values(x)
        assert th.allclose(mu[0], mu[1]), "actor output changed with privileged input"
        assert (not th.allclose(v[0], v[1])) == priv


def test_save_load_keeps_critic_mode(tmp_path):
    _, m = _model(False)
    m.save(tmp_path / "m.zip")
    m2 = PPO.load(tmp_path / "m.zip", device="cpu")
    assert m2.policy.critic_privileged is False
    assert float(m2.policy.vf_features_extractor.mask[-1]) == 0.0


def test_kin_feat_body_velocity_and_fresh_flag():
    from kin_observer import kin_feat
    from sims.drone_pair import DroneState

    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # yaw 90 deg
    st = DroneState(np.array([0.0, 0.02, 1.0]), R, np.zeros(3), np.zeros(3))
    f, _ = kin_feat(st, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(f[:3], 0.1 * np.array([2.0, 0.0, 0.0]))  # 2 m/s along world y = body x
    assert f[3] == 1.0
    f2, _ = kin_feat(st, st.p.copy())
    assert f2[3] == 0.0 and np.allclose(f2[:3], 0.0)  # repeated (dropped) packet


def test_kin_ff_is_causal():
    from kin_observer import KIN_DIM, KinFF

    th.manual_seed(0)
    net = KinFF()
    x = th.randn(1, 20, KIN_DIM)
    x2 = x.clone()
    x2[:, 12:] += 3.0
    with th.no_grad():
        y, y2 = net.forward_seq(x), net.forward_seq(x2)
    assert th.allclose(y[:, :12], y2[:, :12]) and not th.allclose(y[:, 12:], y2[:, 12:])
