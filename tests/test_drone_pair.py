"""Fast regression suite for the real L2F -> gym-pybullet-drones pair (run: pixi run --manifest-path code/pixi.toml test)."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "real_sim"))

from sims.drone_pair import (Course, DroneState, GeometricExpert, InterfaceConfig, R_to_quat_wxyz, RL_DT,  # noqa: E402
                             gate_pass_rate, make_source, make_target, quat_wxyz_to_R)

MAKERS = {"l2f": lambda **kw: make_source(**kw), "pybullet": lambda **kw: make_target(**kw)}


@pytest.fixture(scope="module", params=list(MAKERS))
def drone(request):
    return MAKERS[request.param]()


def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q *= np.sign(q[0])
        q2 = R_to_quat_wxyz(quat_wxyz_to_R(q))
        assert np.allclose(q, q2 * np.sign(q2[0]), atol=1e-9)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_body_rate_sign_convention(drone, axis):
    """+omega_cmd on each axis must produce +body rate on that axis (FLU), in both simulators."""
    drone.reset(DroneState.hover())
    a = drone.hover_action.copy()
    a[1 + axis] = 0.25
    for _ in range(10):
        drone.step(a)
    w = drone.true_state().w
    assert w[axis] > 0.2
    assert np.all(np.abs(np.delete(w, axis)) < 0.1 * w[axis])


def test_source_hover_is_steady():
    d = make_source()
    d.reset(DroneState.hover())
    for _ in range(100):
        d.step(d.hover_action)
    s = d.true_state()
    assert abs(s.p[2] - 1.0) < 0.01 and abs(s.v[2]) < 0.01


def test_target_airframe_gap_sign():
    """Allocator is calibrated to the heavier L2F airframe, so hover thrust makes the CF2X climb."""
    d = make_target()
    d.reset(DroneState.hover())
    for _ in range(100):
        d.step(d.hover_action)
    assert d.true_state().v[2] > 0.5


def test_expert_tracks_course_in_both(drone):
    c, ex = Course(), GeometricExpert()
    drone.reset(c.start_state())
    P, err = [], []
    for k in range(int(c.period / RL_DT)):
        drone.step(ex(drone.true_state(), *c.ref(k * RL_DT)))
        p = drone.true_state().p
        P.append(p)
        err.append(np.linalg.norm(p - c.ref((k + 1) * RL_DT)[0]))
    assert np.sqrt(np.mean(np.square(err))) < 0.3
    assert gate_pass_rate(c, np.array(P)) == 1.0


def test_source_replay_is_deterministic():
    c, ex = Course(), GeometricExpert()
    d = make_source()
    d.reset(c.start_state())
    A, P = [], []
    for k in range(100):
        a = ex(d.true_state(), *c.ref(k * RL_DT))
        d.step(a)
        A.append(a)
        P.append(d.true_state().p)
    d2 = make_source()
    d2.reset(c.start_state())
    for a, p in zip(A, P):
        d2.step(a)
        assert np.array_equal(d2.true_state().p, p)


def test_action_delay_fifo_and_predict_forward():
    """With delay d, a_t executes d ticks later; predict_forward(in_flight) equals the true future state."""
    d = make_source(iface=InterfaceConfig(action_delay=3))
    c, ex = Course(), GeometricExpert()
    d.reset(c.start_state())
    for k in range(20):
        d.step(ex(d.true_state(), *c.ref(k * RL_DT)))
    pred = d.predict_forward(d.in_flight())
    assert len(d.in_flight()) == 3
    for _ in range(3):
        d.step(d.hover_action)  # the next 3 executed actions are the in-flight ones
    assert np.allclose(pred.p, d.true_state().p, atol=1e-9)


def test_zoh_dropout_holds_last_obs():
    d = make_source(iface=InterfaceConfig(p_drop=0.5, dropout_mode="zoh"), seed=3)
    prev = d.reset(DroneState.hover())
    held = 0
    for _ in range(50):
        o = d.step(d.hover_action)
        if d.dropped:
            held += 1
            assert np.array_equal(o.p, prev.p)
        prev = o
    assert held > 10


def test_prop_2_1_delay_limit_pybullet():
    """Measured critical gain of the v_z loop in PyBullet (tau_m = 0) matches 2 sin(pi / (2(2d+1)))."""
    from exp2_delay_stability import CFG, MASS_TGT, PyBulletBackend, g_crit_pure, measured_unstable

    mk = lambda: PyBulletBackend(tau_m=0.0)
    for d in [1, 3]:
        g = g_crit_pure(d)
        unstable = measured_unstable(mk, CFG.mass / MASS_TGT, d)
        assert not unstable(0.93 * g) and unstable(1.07 * g)
