"""Hover-normalized thrust action (Exp 27): a0 = 0 must hover on any airframe of known mass."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "real_sim"))

from hover_norm import HoverNormDrone, hover_norm_to_raw  # noqa: E402
from sims.drone_pair import GRAVITY, CTBRConfig, DroneState, make_source, make_target  # noqa: E402


def test_zero_action_is_default_hover_at_nominal_mass():
    ctbr = CTBRConfig()
    assert np.allclose(hover_norm_to_raw(np.zeros(4), GRAVITY, ctbr), ctbr.encode(GRAVITY, np.zeros(3)))


def _vz_after(drone, steps=50):
    drone.reset(DroneState.hover(p=np.array([0.0, 0.0, 1.5])))
    for _ in range(steps):
        drone.step(np.zeros(4))
    return drone.true_state().v[2]


def test_zero_action_hovers_on_heavier_source_and_on_target():
    src = HoverNormDrone(make_source(seed=0))
    src.backend.set_dynamics({"mass_scale": 1.15})
    tgt = HoverNormDrone(make_target())
    for d in (src, tgt):
        assert abs(_vz_after(d)) < 0.15  # after 0.5 s at a0 = 0
    # the default (non-normalized) hover action sinks the heavier source airframe
    raw = make_source(seed=0)
    raw.backend.set_dynamics({"mass_scale": 1.15})
    raw.reset(DroneState.hover(p=np.array([0.0, 0.0, 1.5])))
    for _ in range(50):
        raw.step(raw.hover_action)
    assert raw.true_state().v[2] < -0.3
