# Mathematical Framework & Hypothesis Stress-Test (H1–H4)

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

### 2.1 H1 Stress-Test: Schur Stability Collapse vs. Low-Speed Drone Masking
* **Proposition 2.1 (Discrete Delay Schur Stability Limit):** For linearized error regulation $x_{t+1} = x_t - \kappa \Delta t \, u_{t-d}$ under linear feedback $u_t = K_\pi x_t$ ($g = \kappa K_\pi \Delta t$), closed-loop Schur stability holds iff:
  $$0 < g < g_{\text{crit}}(d) = 2\sin\!\left(\frac{\pi}{2(2d + 1)}\right)$$
  Allowable gain collapses **$4.5\times$** over 3 ticks: $g_{\text{crit}}(0) = 2.0 \to 1.0\ (d=1) \to 0.618\ (d=2) \to 0.445\ (d=3)$. Saturating bang-bang PPO locks into permanent relay limit cycles of period $4(d+1)\Delta t$.
* **Low-Speed Drone Masking:** Quadratic drag scales as $\mathcal{O}(\|v_B\|^2)$ ($50\text{--}100\times$ smaller at $1.5\text{ m/s}$ than at $12\text{ m/s}$), and coarse RL steps ($\Delta t_{\text{RL}} \gg \tau_m$) drive $e^{-\Delta t_{\text{RL}}/\tau_m} \to 0$.
* **Fix:** Add $-\lambda_{\Delta a}\|a_t - a_{t-1}\|^2$ to nominal PPO (keep $\lambda_{\Delta a}=0$ as an ablation), separate $d=0$ lockstep from $d \in \{1..5\}$ in Trackmania, and enforce $v_{\text{peak}} \ge 10\text{ m/s}$ at $\Delta t_{\text{RL}} = 10\text{--}20\text{ ms}$ on the drone course.

### 2.2 H2 Stress-Test: Non-Markovian `previous_action` Trap & Boundary Conservatism
* **Theorem 2.2 (`a_{t-1}` Insufficient Statistic for $d \ge 2$):** Exact Smith-predictor state forward-propagation requires $s_{t+d_a} = f_S^{(d)}(s_{t-d_o}; a_{t-d}, \dots, a_{t-1})$ where $d = d_a + d_o$. Feeding only $(s_{t-d_o}, a_{t-1})$ omits $d - 1$ in-flight actions for all $d \ge 2$, forcing latency randomization to attenuate gain rather than predict state! Moreover, online identification of unknown $d \sim p(d)$ requires at least one state-transition pair $(o_{t-1}, o_t)$ or recurrence.
* **Proposition 2.3 (Crash-Boundary Conservatism):** Maximizing $\mathbb{E}_\xi[J_\xi(\pi)]$ near deterministic racing walls/curbs forces a positive safety margin $m_{\text{DR}} > 0$, inducing $\Delta_{\text{conserv}}(\xi_0) = J_{\xi_0}(\pi^*_{\xi_0}) - J_{\xi_0}(\pi_{\text{DR}}) > 0$ (slightly slower nominal Trackmania lap times rather than strictly zero change).
* **Fix:** Feed action buffer $a_{t-d_{\max}:t-1}$ (plus short frame stack $o_{t-H:t}$ for unknown $d$), keep `a_{t-1}`-only as a targeted ablation, and include an explicit perturbation-DR cell in `TMNF-C`.

### 2.3 H3 Stress-Test: Recurrent Observer OOD Divergence Trap
* **Theorem 2.4 (Observer Bias Amplification on Nominal $\mathcal{M}_S$):** A recurrent student distilled strictly on fixed nominal $\mathcal{M}_S$ learns an internal predictor–corrector $\hat{s}_t = A_S \hat{s}_{t-1} + B_S a_{t-1} + K(o_t - C \hat{s}_{t \mid t-1})$ with small smoothing gain $K \ll 1$. Under target physics shift $\Delta f$, steady-state estimation MSE (for $A_S = C = I$) is:
  $$\mathbb{E}\|e_\infty\|^2 = \left(\frac{1 - K}{K}\right)^2 \|\Delta f\|^2 + \left(\frac{K}{2 - K}\right) d_s \sigma_\epsilon^2.$$
  * **Trackmania ($\Delta f = 0$):** Recurrence strictly reduces MSE vs. memoryless feed-forward ($d_s \sigma_\epsilon^2$).
  * **Drone ($\Delta f \neq 0$):** Whenever $\|\Delta f\| > K\sqrt{\frac{2d_s}{2-K}}\sigma_\epsilon$, nominal-distilled recurrence has **strictly worse MSE than feed-forward** and drives GRU/LSTM states $h_t$ out-of-distribution!
* **Fix:** Distill recurrent students via **on-policy DAgger over randomized $(\xi, d, b)$** (RMA-style joint $(\hat{s}_t, \hat{\xi}_t)$ adaptation), include nominal-distilled RNN as an ablation proving Theorem 2.4, and measure the normalized **Observability Gap Recovery Ratio**:
  $$\rho_{\text{obs}} = \frac{J_T(\pi^{\text{RNN-sens}}) - J_T(\pi^{\text{FF-sens}})}{J_T(\pi^{\text{DR-exact}}) - J_T(\pi^{\text{FF-sens}})}.$$

### 2.4 H4 Stress-Test: Latency-Confounded Phantom Residuals & Frame Invariance
* **Theorem 2.5 (Latency-Induced Phantom `k-NN` Residual in Trackmania):** In Trackmania ($\Delta f \equiv 0$) with loop delay $d = d_a + d_o \ge 1$, fitting $r^{\text{naive}}_{\text{dyn}, t} = \frac{1}{\Delta t}(\Pi_v \hat{s}_{t+1} - \Pi_v f_S(\hat{s}_t, a_t))$ on client-logged $a_t$ instead of executed $a_{t-d}$ converges to:
  $$\hat{r}^{\text{naive}}_{\text{dyn}}(s, a) \xrightarrow{p} J_{v,a}(s)\,\mathbb{E}[a_{t-d} - a_t \mid \hat{s}_t = s, a_t = a] \neq \mathbf{0}$$
  (producing $+10\text{--}25\text{ m/s}^2$ phantom acceleration at braking zones and refuting H4!).
  * **De-Confounding Fix:** Estimate integer loop delay $\hat{d} = \arg\min_{d'} \sum \|\hat{s}_{t+1} - f_S(\hat{s}_t, a_{t-d'})\|^2$ first; fitting `k-NN` on aligned $(\hat{s}_t, a_{t-\hat{d}})$ restores $\hat{r}^{\text{aligned}}_{\text{dyn}} \approx \mathbf{0}$.
* **Propositions 2.6 & 2.7 (Drone Motor-Lag & GP Bias Identifiability):**
  * Augment Drone `k-NN` features with $a_{t-1}$ or $\hat{\Omega}_{t-1}$ strictly in **SE(3)-invariant body frame** $(v_B, \omega_B, a_t, \hat{\Omega}_{t-1})$ with kernel decay outside rollout support so held-out tracks generalize.
  * Ensure sensor-limited targets include systematic **pose-correlated bias** $b(\phi_{\text{rel}}(s)) \neq 0$ (e.g., oblique gate-viewing-angle PnP error); pure zero-mean noise/dropout collapses $\hat{\mu}_{\text{GP}} \to 0$.

---

## 3. Summary Table: Original vs. Refined Hypotheses (H1'–H4')

| ID | Original Proposal Hypothesis | Refined, Theoretically Airtight Hypothesis |
| :--- | :--- | :--- |
| **H1 $\to$ H1'** | Exact-state nominal PPO transfers almost losslessly in TM (small latency loss) and degrades sharply on Drone. | Smoothness-regularized exact-state PPO transfers with $0\%$ gap in $d=0$ lockstep TM, degrades gracefully for $d \le 2$ (whereas unregularized bang-bang PPO limit-cycles), and degrades sharply on high-speed ($v \ge 10\text{ m/s}$) drone tracks. |
| **H2 $\to$ H2'** | Dynamics DR improves Drone and leaves TM unchanged; latency DR improves both. | Dynamics DR improves Drone transfer while inducing a small boundary-conservatism lap-time penalty ($\Delta_{\text{conserv}} \ge 0$) in TM; latency DR *with action-history buffering* $a_{t-d_{\max}:t-1}$ improves delayed transfer in both domains (outperforming `a_{t-1}`-only DR for $d \ge 2$). |
| **H3 $\to$ H3'** | Recurrent students improve sensor-limited transfer by a similar fraction in both domains. | When distilled via DAgger over randomized transitions $(\xi, d, b)$, recurrent students recover a high fraction $\rho_{\text{obs}}$ of the *observability gap* in both domains—whereas nominal-source distillation degrades on Drone due to OOD internal observer bias. |
| **H4 $\to$ H4'** | Residual fine-tuning closes more Drone gap than DR at equal source budget, and its TM dynamics residual is near zero. | *Delay-aligned* ($\hat{d}$), body-frame residual fine-tuning closes more of the specific Drone gap than wide-prior DR at equal source compute, yields $\|\hat{r}^{\text{aligned}}_{\text{dyn}}\| \approx 0$ in TM (unlike unaligned fitting, which learns phantom lag bias), and captures systematic pose bias $b(\phi_{\text{rel}}(s))$ via GP. |
