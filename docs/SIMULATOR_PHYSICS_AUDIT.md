# Simulator Physics, Control Interfaces & Prior-Work Equations Audit

**Project:** *One Method, Two Racing Worlds: Sim-to-Sim Transfer in Drone and Trackmania Racing* (CMU 16-831)  
**Scope:** Exact ODEs, URDF constants, control abstractions, observation/reward definitions, and source-code-level simulator gotchas across **Swift**, **`L2F`**, **`gym-pybullet-drones`**, **`TMNF-C`**, **`TMInterface`**, and **`Linesight`**.

---

## 1. Seven Critical Source-Code Gotchas (Read Before Coding Wrappers!)

1. **No Native Pure-Python CTBR in Either Drone Simulator:**
   * **`L2F` (`rl-tools/l2f`)** exposes **direct 4-motor normalized RPM commands** $\mathbf{a} \in [-1, 1]^4$.
   * **`gym-pybullet-drones` (`BaseRLAviary`)** only implements `ActionType.{RPM, PID (3D pos), VEL (4D vel), ONE_D_RPM, ONE_D_PID}`. Note that `gym_pybullet_drones/control/CTBRControl.py` is **not** a Python rate controller—it is a **UDP socket client for an external Betaflight SITL firmware binary**, and `DSLPIDControl` is a full Euler-attitude controller rather than an acro body-rate loop.
   * **Fix:** Implement explicit Python/C++ inner body-rate PID + quadX mixer wrappers around `ActionType.RPM` in **both** simulators.
2. **Inverted Motor-Lag Fidelity & Default Zero Drag in `gym-pybullet-drones`:**
   * `L2F` integrates a 1st-order motor filter $\dot{\omega}_i = (\omega_{i,\text{sp}} - \omega_i)/\tau_m$ ($\tau_m > 0$) and rotor drag.
   * `gym-pybullet-drones` defaults to `physics=Physics.PYB`, which applies $F_i = k_f \text{RPM}_i^2$ **instantaneously with zero drag, zero ground effect, and zero motor lag ($\tau_m = 0$)**! Even when `Physics.PYB_GND_DRAG_DW` *(exact enum spelling—NOT `PYB_DRAG_DW_GND`!)* is enabled, **motor spin-up lag is still absent ($\tau_m = 0$)** in `BaseAviary._physics()`, and `_downwash()` is `0` when `NUM_DRONES == 1`. Also note Bullet silently sets `linearDamping=0.04, angularDamping=0.04` on URDF bodies unless overridden via `p.changeDynamics`.
   * **Fix:** Pass `physics=Physics.PYB_GND_DRAG_DW` **and** wrap `gym-pybullet-drones` motor commands in an explicit 1st-order motor lag filter ($\tau_{m,\text{target}} \neq \tau_{m,\text{source}}$, e.g., asymmetric $\tau_{\text{up}} < \tau_{\text{down}}$).
3. **Non-Integer Clock Sub-Stepping (`240 Hz` vs. `100 Hz` / `50 Hz`):**
   * `L2F` defaults to `100 Hz` ($10\text{ ms}$), whereas `gym-pybullet-drones` defaults to `PYB_FREQ = 240 Hz`. Because $240/100 = 2.4$ and $240/50 = 4.8$ are non-integers, `CTRL_FREQ = 48` introduces a **4% clock drift** during open-loop scripted action replay!
   * **Fix:** Set `pyb_freq = 500` (or `200`) and `ctrl_freq = 100` (or `50`) so `pyb_freq / ctrl_freq` is an exact integer.
4. **3D Linear Acceleration `k-NN` Misses Rotational Gaps on `cf2x`:**
   * Swift's physical quadrotor ran a tightly tuned multi-kHz Betaflight loop where high-speed translational drag ($100\text{ km/h}$) dominated, so 3D $\mathbf{a}_{\text{res}} \in \mathbb{R}^3$ sufficed. On a $27\text{ g}$ Crazyflie (`cf2x`) transferring `L2F` $\to$ `gym-pybullet-drones`, inertia $\mathbf{J}$ (watch out for stripped `3.85e-6` vs. lumped `1.4e-5` presets in `L2F`!), $\tau_m$, and inner rate-PID tracking create **angular acceleration residuals** $\boldsymbol{\alpha}_{\text{res}} \in \mathbb{R}^3$.
   * **Fix:** Regress **6D linear + angular acceleration residuals** $[\mathbf{a}_{\text{res}}, \boldsymbol{\alpha}_{\text{res}}] \in \mathbb{R}^6$ conditioned on body-frame $(\mathbf{v}_B, \boldsymbol{\omega}_B, c, \boldsymbol{\omega}_{\text{cmd}}, \mathbf{a}_{t-1})$.
5. **`TMInterface` (`SC_RUN_STEP_SYNC`) Has ZERO Natural Timing/Latency Gap:**
   * `on_run_step` sends `SC_RUN_STEP_SYNC` and blocks (`WaitForResponse`) on every $10\text{ ms}$ physics tick until Python replies—even at `set speed 100`. Unperturbed `TMNF-C` $\to$ `TMInterface` has **zero natural latency and zero jitter**.
   * **Fix:** Keep 0-tick lockstep as a unit-test parity check, and define the evaluation target via **`AsyncWallClockSocketServer`** (or explicit FIFO latency + sensor decimation/dropout in `on_run_step`).
6. **`TMNF-C` Hardcodes C Vehicle Physics Constants (No Python Physics Kwargs):**
   * `TMNF-C` exposes actions, observations, and state snapshots/rewinds to Python, *not* car mass/grip/drag kwargs. Testing dynamics/disturbance randomization in Trackmania (H2) requires either patching 3 scalar multipliers (`mass_scale`, `grip_scale`, `drag_scale`) into `TMNF-C`'s C struct (`cmake --build`) or injecting external velocity/wrench disturbances in the Python wrapper.
7. **`Linesight` Is Off-Policy Multimodal IQN @ 10–20 Hz, Not 100 Hz State-Only PPO:**
   * `Linesight` uses `64x64` grayscale images + `state_float` (`zone_centers` virtual checkpoints in car frame, gear, RPM, wheel contact/slip) with **12 discrete actions** held for `ms_per_action = 50 ms` (`20 Hz`) or `100 ms` (`10 Hz`). Unify `ms_per_action` and the `state_float` observation builder across `TMNF-C` and `TMInterface`.

---

## 2. Swift (Kaufmann et al., *Nature* 2023) Reference Equations

* **Actor Observation ($\mathbf{o}_t^{\text{actor}} \in \mathbb{R}^{31}$ at $50\text{ Hz}$):**
  $$\mathbf{o}_t^{\text{actor}} = \big[\,\hat{\mathbf{p}}_t \in \mathbb{R}^3,\; \hat{\mathbf{v}}_t \in \mathbb{R}^3,\; \text{vec}(\hat{\mathbf{R}}_t) \in \mathbb{R}^9,\; \delta\hat{\mathbf{p}}_{\text{gate}, 1..4} \in \mathbb{R}^{12},\; \mathbf{a}_{t-1} \in \mathbb{R}^4\,\big]^\top$$
  *(Per Guardrail #2 in `AGENTS.md`, extend $\mathbf{a}_{t-1}$ to action buffer $\mathbf{a}_{t-d_{\max}:t-1}$ for delayed regimes, and express state features in gate-relative / body coordinates so held-out tracks generalize).*
* **Action ($\mathbf{a}_t \in \mathbb{R}^4$, CTBR):** $[c_t,\; \omega_{x,t},\; \omega_{y,t},\; \omega_{z,t}]^\top$ (mass-normalized collective thrust + desired body angular rates).
* **Reward Function:**
  $$r_t = \lambda_1 (d_{t-1} - d_t) + \lambda_2 \exp\!\left(-\lambda_3 \delta_{\text{cam}, t}^4\right) - \left(\lambda_4 \|\boldsymbol{\omega}_t\|^2 + \lambda_5 \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2\right) + r_t^{\text{crash}}$$
  *(Keep $\lambda_2 > 0$ even for exact-state teachers so teacher trajectories maintain gate visibility for sensor-limited student distillation!)*
* **Target Calibration Budget:** **800–1,000 samples** ($16\text{–}20\text{ s}$ at $50\text{ Hz}$, i.e., $3\text{–}5$ target laps) used to fit $k$-NN ($k=5$) dynamics residuals and 9 independent 1D RBF-mixture GPs for spatially/temporally correlated odometry drift.

---

## 3. Exact Drone ODE Comparison: `L2F` vs. `gym-pybullet-drones` (`CF2X`)

### 3.1 `L2F` Continuous-Time State ($\mathbb{R}^{17}$, RK4 @ $\Delta t = 0.01\text{ s}$)
State $\mathbf{x} = [\mathbf{p}^\top, \mathbf{q}^\top, \mathbf{v}^\top, \boldsymbol{\omega}_B^\top, \boldsymbol{\omega}_m^\top]^\top$:
$$\begin{aligned}
\dot{\mathbf{p}} &= \mathbf{v}, \qquad \dot{\mathbf{q}} = \frac{1}{2}\mathbf{q} \otimes [0,\, \boldsymbol{\omega}_B^\top]^\top \\
\dot{\mathbf{v}} &= \mathbf{g} + \frac{1}{m}\mathbf{R}(\mathbf{q})\!\left(\sum_{i=1}^4 [0, 0, T_i(\omega_{m,i})]^\top - \mathbf{D}_{\text{rotor}}\mathbf{v}_B\right) \\
\dot{\boldsymbol{\omega}}_B &= \mathbf{J}^{-1}\!\left(\sum_{i=1}^4 \big(\mathbf{r}_{\text{rot},i} \times [0, 0, T_i]^\top + [0, 0, d_i \kappa T_i]^\top\big) - \boldsymbol{\omega}_B \times (\mathbf{J}\boldsymbol{\omega}_B)\right) \\
\dot{\omega}_{m,i} &= \frac{1}{\tau_m}\big(\omega_{m,i,\text{sp}} - \omega_{m,i}\big)
\end{aligned}$$

### 3.2 `gym-pybullet-drones` `cf2x.urdf` Constants & `Physics.PYB_GND_DRAG_DW`
* **URDF Constants:** $m = 0.027\text{ kg}$, $J_{xx} = J_{yy} = 1.4 \times 10^{-5}\text{ kg}\cdot\text{m}^2$, $J_{zz} = 2.17 \times 10^{-5}\text{ kg}\cdot\text{m}^2$, arm $l = 0.0397\text{ m}$, $k_f = 3.16 \times 10^{-10}\text{ N/RPM}^2$, $k_m = 7.94 \times 10^{-12}\text{ N}\cdot\text{m/RPM}^2$, `thrust2weight` $= 2.25$ ($\text{MAX\_RPM} \approx 21,667$), $k_{D,xy} = 9.1785 \times 10^{-7}$, $k_{D,z} = 1.0311 \times 10^{-6}$, $r_P = 0.02313\text{ m}$, $k_G = 11.3686$.
* **Forces Applied Under `Physics.PYB_GND_DRAG_DW`:**
  * Rotor thrust & torque (instantaneous, $\tau_m = 0$ unless wrapped!): $F_i = k_f \text{RPM}_i^2$, $\tau_{z,i} = (-1)^i k_m \text{RPM}_i^2$.
  * Blade drag: $\mathbf{F}_{\text{drag}, B} = -\operatorname{diag}(k_{D,xy}, k_{D,xy}, k_{D,z}) \left(\sum_{i=0}^3 \frac{2\pi \text{RPM}_i}{60}\right)(\mathbf{R}^\top \dot{\mathbf{p}})$.
  * Ground effect: $\Delta F_{\text{gnd}, i} = \operatorname{clip}\!\left(k_G \left(\frac{r_P}{4 z_i}\right)^2 k_f \text{RPM}_i^2,\, 0,\, 0.25 F_{\max}\right)$.
  * Symplectic Euler step + URDF default `linearDamping=0.04, angularDamping=0.04` + full 3D gate contact restitution.

---

## 4. Trackmania Feature & Stepping Alignment Checklist

1. **Shared Observation Builder:** Write a single shared Python helper mapping raw car state (`pos, rot, vel, ang_vel, gear, rpm, wheel_contact, wheel_slip`) + `.npy` track `zone_centers` into car-local coordinates identically for both `TMNF-C` and `TMInterface`.
2. **Action Hold Alignment (`ms_per_action`):** Match the action hold period (`10 ms` = 100 Hz vs. `50 ms` = 20 Hz) and action space (continuous `[steer, gas, brake]` vs. `Linesight`'s 12 discrete actions) identically across `TMNF-C` and `TMInterface`.
3. **Target Timing Modes:** Always report (a) **0-tick synchronous lockstep** (confirming $0.0$ physics divergence) alongside (b) **asynchronous / delayed target** (`AsyncWallClockSocketServer` on Mac; FIFO-delayed / unlocked `TMInterface` on Windows).
