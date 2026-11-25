"""
Shared 500 Hz Collective Thrust + Body Rate (CTBR) Inner-Loop Controller & QuadX Mixer.

Addresses Drone Gotcha #1 & #2 from docs/SIMULATOR_PHYSICS_AUDIT.md:
- Neither `l2f` nor `gym-pybullet-drones` exposes pure-Python CTBR natively.
- Vanilla `gym-pybullet-drones` has zero motor spin-up lag (tau_m = 0) across all Physics modes.

This module maps normalized policy CTBR commands:
    action = [u_thrust, u_wx, u_wy, u_wz] in [-1, 1]^4
to 4 individual Crazyflie 2.x (CF2X) motor RPMs (or normalized [-1, 1]^4 commands for L2F),
with configurable P/PID body-rate gains, gyro low-pass filtering, and 1st-order motor lag tau_m.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class CF2XCTBRConfig:
    """Physical and inner-loop parameters for Crazyflie 2.x (CF2X)."""

    mass: float = 0.027                     # kg
    gravity: float = 9.81                   # m/s^2
    arm_length: float = 0.0397              # m (rotor distance from CoM; dx=dy=arm/sqrt(2))
    kf: float = 3.16e-10                    # N / RPM^2
    km: float = 7.94e-12                    # N*m / RPM^2
    thrust_to_weight: float = 2.25          # max total thrust / (m * g)
    min_thrust_acc: float = 2.0             # m/s^2 minimum collective mass-normalized thrust
    max_thrust_acc: float = 21.5            # m/s^2 maximum collective mass-normalized thrust
    max_rate_xy: float = 10.0               # rad/s max roll/pitch body rate command
    max_rate_z: float = 6.0                 # rad/s max yaw body rate command

    # Inner body-rate PID gains (outputting desired angular acceleration in rad/s^2)
    kp_rate_xy: float = 70.0
    kp_rate_z: float = 35.0
    ki_rate_xy: float = 0.0
    kd_rate_xy: float = 0.0
    ixx: float = 1.4e-5                     # kg*m^2
    iyy: float = 1.4e-5                     # kg*m^2
    izz: float = 2.17e-5                    # kg*m^2

    # Actuator & sensor filtering
    tau_motor: float = 0.0                  # s (0.0 = instantaneous; e.g. 0.025s for target motor lag)
    gyro_lpf_alpha: float = 1.0             # 1.0 = un-filtered gyro; <1.0 = 1st-order gyro LPF

    # Motor ordering & spin convention:
    # Default matches gym-pybullet-drones cf2x.urdf:
    #   Motor 0: (+x, -y) Front-Right (CCW / torque -1)
    #   Motor 1: (-x, -y) Back-Right  (CW  / torque +1)
    #   Motor 2: (-x, +y) Back-Left   (CCW / torque -1)
    #   Motor 3: (+x, +y) Front-Left  (CW  / torque +1)
    motor_xy_signs: tuple[tuple[float, float], ...] = (
        (+1.0, -1.0),
        (-1.0, -1.0),
        (-1.0, +1.0),
        (+1.0, +1.0),
    )
    motor_yaw_signs: tuple[float, float, float, float] = (-1.0, +1.0, -1.0, +1.0)

    @property
    def max_rpm(self) -> float:
        return math.sqrt((self.thrust_to_weight * self.mass * self.gravity) / (4.0 * self.kf))

    @property
    def hover_rpm(self) -> float:
        return math.sqrt((self.mass * self.gravity) / (4.0 * self.kf))


class CF2XCTBRController:
    """Stateful 500 Hz inner-loop body-rate controller + QuadX control allocator."""

    def __init__(self, cfg: CF2XCTBRConfig | None = None):
        self.cfg = cfg or CF2XCTBRConfig()
        self.omega_filt = np.zeros(3, dtype=np.float64)
        self.rate_int = np.zeros(3, dtype=np.float64)
        self.prev_rate_err = np.zeros(3, dtype=np.float64)
        self.motor_rpm = np.full(4, self.cfg.hover_rpm, dtype=np.float64)

        # Precompute 4x4 wrench-to-rotor-thrust allocation matrix inverse:
        # [T_total, tau_x, tau_y, tau_z]^T = M @ [f_0, f_1, f_2, f_3]^T
        d_arm = self.cfg.arm_length / math.sqrt(2.0)
        kappa = self.cfg.km / self.cfg.kf
        M = np.zeros((4, 4), dtype=np.float64)
        for i in range(4):
            sx, sy = self.cfg.motor_xy_signs[i]
            # Roll torque tau_x = r_y * f_i ; Pitch torque tau_y = -r_x * f_i
            M[0, i] = 1.0
            M[1, i] = sy * d_arm
            M[2, i] = -sx * d_arm
            M[3, i] = self.cfg.motor_yaw_signs[i] * kappa
        self.M_inv = np.linalg.inv(M)

    def reset(self, init_rpm: float | None = None) -> None:
        rpm0 = self.cfg.hover_rpm if init_rpm is None else init_rpm
        self.omega_filt[:] = 0.0
        self.rate_int[:] = 0.0
        self.prev_rate_err[:] = 0.0
        self.motor_rpm[:] = rpm0

    def decode_ctbr_action(self, action: np.ndarray) -> tuple[float, np.ndarray]:
        """Map normalized [-1, 1]^4 policy action to (c_cmd [m/s^2], omega_cmd [rad/s])."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        c_mid = 0.5 * (self.cfg.min_thrust_acc + self.cfg.max_thrust_acc)
        c_half = 0.5 * (self.cfg.max_thrust_acc - self.cfg.min_thrust_acc)
        c_cmd = c_mid + c_half * float(a[0])
        omega_cmd = np.array(
            [
                self.cfg.max_rate_xy * a[1],
                self.cfg.max_rate_xy * a[2],
                self.cfg.max_rate_z * a[3],
            ],
            dtype=np.float64,
        )
        return c_cmd, omega_cmd

    def step_substep(self, action: np.ndarray, omega_body_meas: np.ndarray, dt_sub: float) -> np.ndarray:
        """
        Advance one inner-loop tick (e.g., dt_sub = 0.002s at 500 Hz) and return 4 motor RPMs.
        """
        c_cmd, omega_cmd = self.decode_ctbr_action(action)
        a_lpf = self.cfg.gyro_lpf_alpha
        self.omega_filt = a_lpf * np.asarray(omega_body_meas, dtype=np.float64) + (1.0 - a_lpf) * self.omega_filt

        err = omega_cmd - self.omega_filt
        self.rate_int = np.clip(self.rate_int + err * dt_sub, -2.0, 2.0)
        d_err = (err - self.prev_rate_err) / max(dt_sub, 1e-6)
        self.prev_rate_err = err

        kp = np.array([self.cfg.kp_rate_xy, self.cfg.kp_rate_xy, self.cfg.kp_rate_z])
        ki = np.array([self.cfg.ki_rate_xy, self.cfg.ki_rate_xy, 0.0])
        kd = np.array([self.cfg.kd_rate_xy, self.cfg.kd_rate_xy, 0.0])
        ang_acc_des = kp * err + ki * self.rate_int + kd * d_err

        J = np.array([self.cfg.ixx, self.cfg.iyy, self.cfg.izz])
        gyro_coriolis = np.cross(self.omega_filt, J * self.omega_filt)
        tau_des = J * ang_acc_des + gyro_coriolis

        wrench = np.array([self.cfg.mass * c_cmd, tau_des[0], tau_des[1], tau_des[2]], dtype=np.float64)
        f_max = self.cfg.kf * (self.cfg.max_rpm ** 2)
        f_rotors = np.clip(self.M_inv @ wrench, 0.0, f_max)
        rpm_cmd = np.sqrt(f_rotors / self.cfg.kf)

        if self.cfg.tau_motor > 1e-6:
            alpha_m = 1.0 - math.exp(-dt_sub / self.cfg.tau_motor)
            self.motor_rpm = self.motor_rpm + alpha_m * (rpm_cmd - self.motor_rpm)
        else:
            self.motor_rpm = rpm_cmd

        return self.motor_rpm.copy()

    def rpm_to_normalized_action(self, rpm: np.ndarray) -> np.ndarray:
        """Convert physical RPM in [0, max_rpm] to normalized [-1, 1]^4 action (for L2F)."""
        return np.clip(2.0 * (rpm / self.cfg.max_rpm) - 1.0, -1.0, 1.0)
