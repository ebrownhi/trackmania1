# Mathematical Framework & Hypothesis Stress-Test (H1–H4)

> **Status (2026-11-25):** the claims below have been tested on the real `l2f` → `gym-pybullet-drones` pair — see
> `docs/REAL_SIM_RESULTS.md` for which hold, which are partial, and which were corrected (relay period 4d+2; Δa penalty
> as a fine-tuning stage; GP input must be ground-truth pose; dynamics DR did not help).


**Project:** *One Method, Two Racing Worlds: Sim-to-Sim Transfer in Drone and Trackmania Racing* (CMU 16-831)  
**Purpose:** Formal theoretical foundation, sim-to-sim return gap decomposition, pre-implementation failure-mode proofs, and refined falsifiable hypotheses (**H1'–H4'**).

---

## 1. Unified Delayed-POMDP Sim-to-Sim Formulation

We model both source–target pipelines under a single **Delayed, Latent-Actuator Partially Observable MDP (Delayed-POMDP)** at discrete control step $\Delta t > 0$ ($10\text{ ms}$ at $100\text{ Hz}$) and discount $\gamma \in (0, 1)$.

### 1.1 State, Latent Actuator, Interface, and Sensor Operators
* **Observable Plant State $s_t \in \mathcal{S} \subseteq \mathbb{R}^{n_s}$ and Action $a_t \in \mathcal{A} \subseteq \mathbb{R}^{n_a}$:**
  * *Drone:* $s_t = (p_W, R_{WB}, v_B, \omega_B, g_{\text{next}}(s_t))$; $a_t = (c_t, \omega_{\text{cmd}, t}) \in \mathbb{R}^4$ (CTBR).
  * *Trackmania:* $s_t = (p_W, R_{WB}, v_B, \omega_B, w_{\text{wheel}}, \text{gear}, \text{rpm}, r_{\text{route}}(s_t))$; $a_t = (\delta_{\text{steer}}, u_{\text{gas}}, u_{\text{brake}})$.
* **Unobserved Inner-Loop / Actuator State $z_{\text{hid}, t} \in \mathcal{Z}$:**
  * Physical rotor speeds $\Omega_t \in \mathbb{R}^4$ governed by 1st-order motor lag $\tau_m$ plus low-level rate-PID integral/filter states (present in Drone target; absent in Trackmania).
* **Delayed & Noisy Action Channel:**
  * Given integer action delay $d_{a,t} \in \{0, \dots, d_{a,\max}\}$ ($\mathbb{E}[d_{a,t}] = \bar{d}_a$, capturing fixed latency plus stepping jitter) and channel noise $\eta_t$ ($\mathbb{E}\|\eta_t\| \le \sigma_\eta$), the plant executes:
    $$\tilde{a}_t = \Pi_{\mathcal{A}}\!\left(a_{t - d_{a,t}} + \eta_t\right)$$
* **Canonical Transition Split (Markovian Mismatch vs. Latent Lag):**
  * **Source $\mathcal{M}_S$:** $s_{t+1} = f_S(s_t, a_t)$ (with $d_a \equiv 0, \eta \equiv 0$).
  * **Target $\mathcal{M}_T$:** $z_{\text{hid}, t+1} = g_{\text{hid}}(z_{\text{hid}, t}, s_t, \tilde{a}_t)$ and $s_{t+1} = f_T(s_t, \tilde{a}_t, z_{\text{hid}, t})$.
  * Using the quasi-steady-state actuator equilibrium $z^*_{\text{eq}}(s, a) = g_{\text{hid}}(z^*_{\text{eq}}(s, a), s, a)$, target mismatch decomposes uniquely into:
    $$\Delta f(s_t, \tilde{a}_t, z_{\text{hid}, t}) = \underbrace{\Delta f_{\text{markov}}(s_t, \tilde{a}_t)}_{f_T(s_t, \tilde{a}_t, z^*_{\text{eq}}(s_t, \tilde{a}_t)) - f_S(s_t, \tilde{a}_t)} + \underbrace{\Delta f_{\text{hid}}(s_t, \tilde{a}_t, z_{\text{hid}, t})}_{f_T(s_t, \tilde{a}_t, z_{\text{hid}, t}) - f_T(s_t, \tilde{a}_t, z^*_{\text{eq}}(s_t, \tilde{a}_t))}$$
* **Delayed, Biased, Dropped Sensor Channel:**
  * Underlying sensor reading $y_\tau = h(s_\tau) + b(s_\tau) + \epsilon_\tau$, separating **pose-correlated systematic bias** $b(s_\tau)$ from zero-mean i.i.d. noise $\epsilon_\tau$ ($\mathbb{E}\|\epsilon_\tau\| \le \sigma_\epsilon$).
  * Under observation latency $d_{o,t}$ ($\mathbb{E}[d_{o,t}] = \bar{d}_o$) and Bernoulli dropout $m_t \sim \operatorname{Bern}(p_{\text{drop}})$ with **Zero-Order Hold (ZOH)**:
    $$o_t = (1 - m_t)\, y_{t - d_{o,t}} + m_t \, o_{t-1}$$

---

### 1.2 Key Structural Lemmas & Unified Return Gap Bound

#### Lemma 1.1 (Dropout-to-Geometric-Staleness Equivalence)
Under ZOH $o_t = (1 - m_t) y_{t - d_{o,t}} + m_t o_{t-1}$ with i.i.d. $m_t \sim \operatorname{Bern}(p_{\text{drop}})$, let $K_t = \min\{k \ge 0 : m_{t-k} = 0\} \sim \operatorname{Geom}(1 - p_{\text{drop}})$. Then $o_t = y_{t - \tilde{d}_{o,t}}$ with effective observation staleness:
$$\tilde{d}_{o,t} = d_{o, t - K_t} + K_t, \qquad \mathbb{E}[\tilde{d}_{o,t}] = \bar{d}_o + \frac{p_{\text{drop}}}{1 - p_{\text{drop}}}.$$
Conversely, naive zero-masking $o^{\text{zero}}_t = (1 - m_t) y_{t - d_{o,t}}$ injects large impulses $\Omega(p_{\text{drop}} L_\pi \|h(s_t)\|)$ proportional to raw coordinate magnitude.

#### Theorem 1.2 (Unified Sim-to-Sim Return Gap Decomposition)
Under Lipschitz value/dynamics/policy regularity ($L_V, L_{Q,a}, L_{f,a}, L_h, L_\pi$) and 1st-order actuator contraction $1 - \alpha = e^{-\Delta t / \tau_m}$, the Sim-to-Sim return gap satisfies:
$$\big|J_T(\pi) - J_S(\pi)\big| \le \frac{1}{1-\gamma}\Big(\mathcal{E}_{\text{markov}}(\pi) + \mathcal{E}_{\text{lag}}(\pi) + \mathcal{E}_{\text{delay}}(\pi) + \mathcal{E}_{\text{obs}}(\pi)\Big)$$
where over the target discounted visitation distribution $\bar{d}^\pi_T$:
1. **Markovian Dynamics Term** *(Drone $> 0$; Trackmania $\equiv 0$)*:
   $$\mathcal{E}_{\text{markov}}(\pi) = \gamma L_V \, \mathbb{E}_{\bar{d}^\pi_T}\!\left[\|\Delta f_{\text{markov}}(s_t, \tilde{a}_t)\|\right]$$
2. **Unmodeled Actuator Lag Term** *(Drone $> 0$; Trackmania $\equiv 0$)*:
   $$\mathcal{E}_{\text{lag}}(\pi) \le \gamma L_V L_{f,z} L_\Omega \left(\frac{e^{-\Delta t / \tau_m}}{1 - \gamma e^{-\Delta t / \tau_m}}\right) \mathbb{E}_{\bar{d}^\pi_T}\!\left[\|\tilde{a}_t - \tilde{a}_{t-1}\|\right] \approx \mathcal{O}\!\left(\frac{\tau_m}{\Delta t}\mathbb{E}\|\Delta a_t\|\right)$$
3. **Interface Delay, Jitter & Dropout Term** *(Present in Trackmania & Perturbed Drone)*:
   $$\mathcal{E}_{\text{delay}}(\pi) = \gamma L_V L_{f,a} \Big(\bar{d}_a \, \mathbb{E}_{\bar{d}^\pi_T}\|a_t - a_{t-1}\| + \sigma_\eta\Big) + L_{Q,a} L_\pi L_h \left(\bar{d}_o + \frac{p_{\text{drop}}}{1 - p_{\text{drop}}}\right) \mathbb{E}_{\bar{d}^\pi_T}\|s_t - s_{t-1}\|$$
4. **Partial Observability, Bias & Noise Term**:
   $$\mathcal{E}_{\text{obs}}(\pi) = L_{Q,a} L_\pi \, \mathbb{E}_{\bar{d}^\pi_T}\!\left[\|b(s_{t - \tilde{d}_{o,t}})\| + \sigma_\epsilon\right]$$

> **Action-Smoothness Amplification Law:** Both unmodeled motor lag $\mathcal{E}_{\text{lag}}$ and interface delay $\mathcal{E}_{\text{delay}}$ are **linearly multiplied by policy action chattering $\mathbb{E}\|a_t - a_{t-1}\|$**. Unregularized bang-bang PPO policies ($L_\pi \to \infty$) convert $10\text{--}30\text{ ms}$ delays into $\mathcal{O}(1)$ destabilizing kicks.

---

## 2. Stress-Test of Hypotheses H1–H4: Six Traps & Exact Fixes

### 2.1 H1 Stress-Test: Schur Stability Collapse vs. Course Speed Envelope
* **Proposition 2.1 (Discrete Delay Schur Stability Limit):** For linearized error regulation $x_{t+1} = x_t - \kappa \Delta t \, u_{t-d}$ under linear feedback $u_t = K_\pi x_t$ ($g = \kappa K_\pi \Delta t$), closed-loop Schur stability holds iff:
  $$0 < g < g_{\text{crit}}(d) = 2\sin\!\left(\frac{\pi}{2(2d + 1)}\right)$$
  Allowable gain collapses **$4.5\times$** over 3 ticks: $g_{\text{crit}}(0) = 2.0 \to 1.0\ (d=1) \to 0.618\ (d=2) \to 0.445\ (d=3)$ (verified within $3\%$ in PyBullet Exp 2). Saturating bang-bang control $u_t = -u_{\max}\operatorname{sign}(x_{t-d})$ locks into permanent discrete relay limit cycles of period **$(4d + 2)\Delta t$** (verified exactly in Exp 2).
* **Airframe Envelope & Smoothness Curriculum Caveat:** On the real `L2F` Crazyflie preset ($T/W \approx 1.47$), practical course peak speed is $v_{\text{peak}} \approx 3.55\text{ m/s}$ (airframe mass/thrust/yaw-torque mismatch dominates over high-speed drag). Moreover, applying a large smoothness penalty ($\lambda_{\Delta a} = 0.3$) **from scratch** can stall PPO exploration (2/3 seeds in Exp 7b); instead, **warm-starting from $\lambda_{\Delta a} = 0$ (or ramping $\lambda_{\Delta a}$)** reliably halves action chattering and lowers target error across delays (Exp 7c).

### 2.2 H2 Stress-Test: Non-Markovian `previous_action` Trap & Dynamics-DR Limits
* **Theorem 2.2 (`a_{t-1}` Insufficient Statistic for $d \ge 2$):** Exact Smith-predictor state forward-propagation requires $s_{t+d_a} = f_S^{(d)}(s_{t-d_o}; a_{t-d}, \dots, a_{t-1})$ where $d = d_a + d_o$. Feeding only $(s_{t-d_o}, a_{t-1})$ omits $d - 1$ in-flight actions for all $d \ge 2$, forcing latency randomization to attenuate gain rather than predict state! Moreover, online identification of unknown $d \sim p(d)$ requires at least one state-transition pair $(o_{t-1}, o_t)$ or recurrence.
* **Proposition 2.3 (Conservatism & Asymmetric Gap Mismatch):** Maximizing $\mathbb{E}_\xi[J_\xi(\pi)]$ induces boundary conservatism $\Delta_{\text{conserv}}(\xi_0) \ge 0$ at $d=0$ (seen in Exp 4b: $d=0$ gate rate $1.00 \to 0.73$). Furthermore, on systematic directional airframe shifts (`L2F` $30.6\text{ g}$ $\to$ `CF2X` $27.0\text{ g}$), zero-mean wide-prior dynamics DR fails to close steady-state offset and **degrades** hold precision relative to nominal/residual fine-tuning (Exp 4b, Exp 9).
* **Fix:** Feed action buffer $a_{t-d_{\max}:t-1}$ for latency robustness, keep `a_{t-1}`-only as a targeted ablation, and rely on **residual fine-tuning** (rather than wide memoryless dynamics DR) to correct systematic directional physics offsets.

### 2.3 H3 Stress-Test: Recurrent Observer OOD Drift under Hard Sensor Channels
* **Theorem 2.4 (Observer Bias Amplification on Nominal $\mathcal{M}_S$):** A recurrent student distilled strictly on fixed nominal $\mathcal{M}_S$ learns an internal predictor–corrector $\hat{s}_t = A_S \hat{s}_{t-1} + B_S a_{t-1} + K(o_t - C \hat{s}_{t \mid t-1})$ with small smoothing gain $K \ll 1$. Under target physics shift $\Delta f$, steady-state estimation MSE (for $A_S = C = I$) is:
  $$\mathbb{E}\|e_\infty\|^2 = \left(\frac{1 - K}{K}\right)^2 \|\Delta f\|^2 + \left(\frac{K}{2 - K}\right) d_s \sigma_\epsilon^2.$$
  * **Empirical Real-Sim Scope (Exp 5, 5b):** Under mild position noise ($2\text{ cm}$), finite-differencing already yields usable velocity; under a **hard sensor channel** (`obs_delay = 3`, $20\%$ dropout, no velocity) and larger target gap (`kf * 1.15`), feed-forward students crash ($100\%$), fixed-source estimators degrade ($0.14 \to 0.21\text{ m/s}$), and randomized GRU distillation recovers $\rho_{\text{obs}} \approx 0.7$ on the nominal target and reduces perturbed-target crash rate ($100\% \to 67\%$), though `GRU_fixed` did not underperform memoryless `FF` in closed loop because `FF` already fails under delay+dropout.
* **Fix:** Distill recurrent students via **on-policy DAgger over randomized $(\xi, d, b)$** under realistic sensor delay/dropout/bias, and measure the normalized **Observability Gap Recovery Ratio**:
  $$\rho_{\text{obs}} = \frac{J_T(\pi^{\text{RNN-sens}}) - J_T(\pi^{\text{FF-sens}})}{J_T(\pi^{\text{exact}}) - J_T(\pi^{\text{FF-sens}})}.$$

### 2.4 H4 Stress-Test: Joint $(\hat{d}, \hat{\tau}_m)$ Lag-Filtered SysID, Body-Frame `k-NN` & Ground-Truth-Pose `GP`
* **Theorem 2.5 (Latency-Induced Phantom `k-NN` Residual & Joint SysID):** With loop delay $d \ge 1$, fitting $r^{\text{naive}}_{\text{dyn}, t} = \frac{1}{\Delta t}(\Pi_v \hat{s}_{t+1} - \Pi_v f_S(\hat{s}_t, a_t))$ on client-logged $a_t$ instead of executed $a_{t-d}$ learns a **spurious phantom residual** ($17\times$ larger in Exp 6's $\Delta f \equiv 0$ control!).
  * **Joint Lag-Filtered De-Confounding Fix:** When both delay $d$ and actuator lag $\tau_m$ differ between source and target, delay-only scans fail (Exp 6), and regressing thrust-scale residuals against *unfiltered* white-noise commands $a_{t-d}$ biases $\hat{d}$ by $+1$ tick because a $30\text{ ms}$ motor filter weights $a_{t-1}$ more heavily than $a_t$ over $[t, t+\Delta t]$! Always run **Step-0 scripted multi-sine excitation** and scan $(d', \tau')$ jointly while regressing affine scale terms against the **reconstructed lag-filtered rotor thrust** $\bar{u}_{\text{filt}, t}(\tau', d')$.
* **Propositions 2.6 & 2.7 (Body-Frame `k-NN` & Ground-Truth-Pose `GP` Identifiability):**
  * Parameterize `k-NN` strictly in **SE(3)-invariant body frame** $(v_B, \omega_B, a_t)$ with distance decay outside rollout support (world-frame features gave $0\%$ held-out course improvement vs. $-60\%$ RMSE in body frame in Exp 6/8!).
  * **Crucial GP Input Condition (Exp 9):** Fit the GP observation residual $y_i = o_i - h(s_i^T)$ as a function of **ground-truth (mocap) pose** $\phi(s_i^T)$—*never* noisy observed pose $\phi(o_i)$ (regressing $n_i$ on $p_i + n_i$ leaks noise into the regressor and learns a spurious $0.16\text{--}0.25\text{ m}$ bias under the i.i.d. null!).

---

## 3. Summary Table: Original vs. Real-Sim Refined Hypotheses (H1'–H4')

| ID | Original Proposal Hypothesis | Real-Sim Verified & Refined Hypothesis |
| :--- | :--- | :--- |
| **H1 $\to$ H1'** | Exact-state nominal PPO transfers almost losslessly in TM (small latency loss) and degrades sharply on Drone. | Exact-state PPO transfers losslessly when $\Delta f \equiv 0, d = 0$, degrades on the Drone airframe/actuator gap, and benefits across delays when a $-\lambda_{\Delta a}\|a_t - a_{t-1}\|^2$ smoothness penalty is applied via **warm-start / ramp** alongside an action buffer (halving chattering without stalling from-scratch exploration). |
| **H2 $\to$ H2'** | Dynamics DR improves Drone and leaves TM unchanged; latency DR improves both. | Latency randomization **with an action-history buffer** $a_{t-d_{\max}:t-1}$ prevents delayed crashes in target transfer (at a modest $d=0$ conservatism cost), whereas zero-mean wide-prior dynamics DR fails to close systematic directional airframe offsets. |
| **H3 $\to$ H3'** | Recurrent students improve sensor-limited transfer by a similar fraction in both domains. | Under delayed, dropped, velocity-deprived sensor channels, feed-forward students fail while recurrent GRU students distilled via **randomized DAgger** recover a large fraction $\rho_{\text{obs}}$ of the observability gap and resist OOD observer drift on perturbed targets. |
| **H4 $\to$ H4'** | Residual fine-tuning closes more Drone gap than DR at equal source budget, and its TM dynamics residual is near zero. | With **Step-0 scripted multi-sine $(\hat{d}, \hat{\tau}_m)$ SysID**, delay-aligned **body-frame `k-NN`** dynamics residuals + **ground-truth-pose `GP`** bias residuals close substantially more of the Drone transfer gap than DR at equal source compute, while collapsing to $\approx 0$ when $\Delta f \equiv 0$ (and under the zero-mean sensor-noise null). |
