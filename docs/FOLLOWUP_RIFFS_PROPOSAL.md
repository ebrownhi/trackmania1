# Follow-Up Research Proposal: Next-Wave Riffs Across Every Pipeline Step

**Project:** *One Method, Two Racing Worlds: Sim-to-Sim Transfer in Drone and Trackmania Racing* (CMU 16-831)  
**Date:** 2026-11-25 · **Baseline Code & Results:** [REAL_SIM_RESULTS.md](../docs/REAL_SIM_RESULTS.md), [drone_pair.py](../code/sims/drone_pair.py), [real_sim/](../real_sim)

---

## Motivation: Why Our Exp 1–10 Findings Point Directly to High-Impact Extensions

Our baseline real-simulator campaign (`rl-tools/l2f` 2.0.18 → `gym-pybullet-drones` 2.2.0 `CF2X`) validated the core Swift transfer pipeline, **and** exposed five concrete scientific bottlenecks that make ideal "Act II" experiments for the final report:

1. **Smoothness Exploration Stall (`Exp 7b/7c`):** Adding a reward-level action-rate penalty ($\lambda_{\Delta a} = 0.3 \|a_t - a_{t-1}\|^2$) from scratch stalled PPO on 2/3 seeds, requiring a two-stage warm-start curriculum.
2. **Directional Mismatch Defeats Zero-Mean DR (`Exp 4b, 9`):** Symmetric zero-mean dynamics randomization degraded target hold accuracy (`−29%` to `−65%` of the gap) because `L2F → PyBullet` is a *systematic directional shift* (`T/W: 1.47 → 2.25`, `km/kf: 5.4×` higher).
3. **Sub-Tick Actuator Mismatch Limits Acceleration `k-NN` (`Exp 6, 8`):** Delay-aligned SE(3) `k-NN` cut held-out linear acceleration residual RMSE by `60%` (`0.99 → 0.40 m/s²`), **but angular acceleration RMSE stayed flat (`12 → 12 rad/s²`)** because `500 Hz` rate-loop + rotor-saturation coupling is non-additive in 100 Hz rigid-body acceleration space.
4. **Out-of-Support Residual Exploitation (`Exp 6, 9`):** Fine-tuning PPO inside a residual-augmented simulator risks exploiting inaccurate residuals when the fine-tuned policy flies faster or tighter racing lines than the 4 calibration flights covered.
5. **Recurrent Observer OOD Variance (`Exp 5b`):** Under `obs_delay = 3`, `20%` ZOH dropout, and hidden velocity, `GRU_rand` dramatically beat `FF` (`0.40 m` vs `0.84 m`), yet still showed `67%` crash variance when motor gain shifted (`kf × 1.15`).

Below are **14 literature-grounded experimental riffs**, organized by the exact steps of our proposal (`Step 0` through `Step 5`), each mapped directly to existing files in our repository so any teammate or agent can implement and benchmark them on Mac CPU in hours.

---

## Quick Prioritization Matrix (Best Return-on-Effort for the Final Report)

| Riff ID | Pipeline Step | Core Paper Anchor | Mac Runtime | Expected Payoff / Story Value |
|---|---|---|---|---|
| **R4.1** | Step 4 (Residuals) | **ASAP** (He et al., CMU/NVIDIA, *RSS 2025*) | ~10 min | **Highest:** Directly fixes `Exp 6/8`'s `12 rad/s²` angular residual limit via $\Delta a$ action-space alignment |
| **R1.1** | Step 1 (Smoothness) | **CAPS** (Mysore et al., *ICRA 2021*) + **L2C2** (*IROS 2022*) | ~8 min | Fixes `Exp 7b`'s 2/3-seed PPO exploration stall without needing a two-stage warm-start |
| **R3.2** | Step 3 (Adaptation) | **Neural-Fly** (O'Connell et al., *Sci. Robot. 2022*) | ~5 min | **Zero-retraining** online adaptation during Lap 1 so **Lap 2 beats Lap 1 live** in PyBullet |
| **R4.2** | Step 4 (Residuals) | **MOPO** (Yu et al., *NeurIPS 2020*) | ~8 min | Replaces ad-hoc `k-NN` decay with principled epistemic uncertainty pessimism $-\beta \sigma(s,a)$ |
| **R2.3** | Step 2 (Latency) | **DIDA** (Liotet et al., *ICML 2022*) + Action Chunking | ~6 min | Emits $H$-step action queues `[a_t..a_{t+3}]` to beat ZOH under burst packet loss (Drone + TMNF!) |
| **R3.1** | Step 3 (Observer) | **RMA** (Kumar et al., *RSS 2021*) + **DreamWaQ** (*ICRA 2023*) | ~10 min | Disentangles velocity estimation $\hat{v}_B$ from physics extrinsics $\hat{z}_t$, cutting `GRU` OOD variance |
| **R2.1** | Step 2 (DR) | **DORAEMON** (Tiboni et al., *ICLR 2024*) + **CVaR-PPO** | ~12 min | Auto-expands DR entropy under a completion constraint; directly targets `p90` tail lap time |

---

## Step 0 Riffs — System Identification & Pre-Training Physics Alignment

*Baseline achieved:* `scripted_sysid_action` multi-sine + lag-filtered rotor thrust scan (`exp9_residual_finetune.py`) recovers exact $(\hat{d}, \hat{\tau}_m)$ on `3/3` nominal and perturbed seeds from one 3.5 s flight.

### Riff R0.1 — Optimal Fisher-Information Excitation Design (Safe Active SysID)
- **Literature Anchor:** Ljung (*System Identification*), Brunke et al. (*Annual Reviews in Control* 2022, Safe Learning in Robotics).
- **Idea:** Our current Step-0 trajectory (`scripted_sysid_action`) uses hand-picked frequencies (`1.1–4.3 Hz`). Instead, optimize amplitudes and phases $\phi \in \mathbb{R}^{4 \times K}$ in `L2F` *before* touching the target to maximize D-optimality $\log \det \big(\Phi^\top \Phi\big)$ of the joint $(\text{delay } d, \tau_{\text{rise}}, \tau_{\text{fall}}, k_f/m, k_m/k_f)$ regressor while keeping predicted position inside a $0.5\text{ m}$ safety sphere.
- **Hypothesis:** Cuts required target calibration data from `1,600` ticks (`16 s`) down to **`200–300` ticks (`2–3 s`)** while distinguishing *asymmetric* motor spin-up vs. spin-down ($\tau_{\text{rise}} \neq \tau_{\text{fall}}$).
- **Where to plug in:** Drop-in replacement for `scripted_sysid_action` in [exp9_residual_finetune.py](../real_sim/exp9_residual_finetune.py).

### Riff R0.2 — Multi-Step Differentiable Shooting / Sub-Tick Rotor Alignment
- **Literature Anchor:** Neural ODE / Multi-Step Shooting SysID ( Forgione & Piga, *Automatica* 2021; Agilicious Foehn et al., *Sci. Robot.* 2022).
- **Idea:** One-step (`10 ms`) finite-difference residuals amplify gyro differentiation noise and miss hidden `500 Hz` rotor states across ticks. Instead, optimize parametric `L2F` overrides `(mass_scale, thrust_scale, torque_scale, tau_rise, tau_fall, linear_drag)` by minimizing **10-tick (`100 ms`) open-loop rollout position/attitude shooting error** via CMA-ES or finite-difference Gauss–Newton using `L2FBackend.snapshot()/restore()`.
- **Hypothesis:** Closes `>75%` of `Exp 1`'s `3.72 m` 3-second replay divergence *purely parametrically* before non-parametric `k-NN` even runs, leaving a smaller, smoother residual for `k-NN`.
- **Where to plug in:** Uses existing `L2FBackend.set_dynamics()` in [drone_pair.py](../code/sims/drone_pair.py).

---

## Step 1 Riffs — Nominal Policy & Smoothness Regularization (Beyond H1')

*Baseline achieved:* Reward penalty $\lambda_{\Delta a}\|a_t - a_{t-1}\|^2$ halves chattering and cuts target hold error `0.398 → 0.296 m` at `d=2` *only* when warm-started (`Exp 7c`); training from scratch stalls on 2/3 seeds (`Exp 7b`).

### Riff R1.1 — CAPS + L2C2: Actor-Loss Smoothness Instead of Reward Shaping
- **Literature Anchor:** Mysore et al., *"Regularizing Action Policies for Smooth Control with Reinforcement Learning (CAPS)"* (*ICRA 2021*); Kobayashi, *"L2C2: Locally Lipschitz Continuous Constraint towards Stable and Smooth RL"* (*IROS 2022*).
- **Why Reward Smoothness Stalled in `Exp 7b`:** Putting $-\lambda_{\Delta a}\|a_t - a_{t-1}\|^2$ inside the MDP reward contaminates the **critic's value function** $V^\pi(s)$: early random Gaussian exploration incurs huge negative step rewards, teaching the critic that *moving at all* has lower value than standing still. **CAPS** removes smoothness from the reward entirely and adds two auxiliary losses directly to the **actor update**:
  $$\mathcal{L}_{\text{CAPS}}(\theta) = \lambda_T \|\mu_\theta(o_t) - \mu_\theta(o_{t-1})\|_2^2 + \lambda_S \|\mu_\theta(o_t) - \mu_\theta(o_t + \epsilon)\|_2^2, \quad \epsilon \sim \mathcal{N}(0, \sigma_o^2 I)$$
  Notice two major wins: (1) $\mu_\theta$ is the **deterministic policy mean**, so Gaussian exploration noise $\sigma_{\text{expl}}$ is *never* penalized during early exploration, eliminating the 2/3-seed stall! (2) Spatial smoothness $\lambda_S$ bounds the actor's local Lipschitz constant $\|\nabla_o \mu_\theta\|$, preventing sensor noise and ZOH step jumps from triggering actuator spikes!
- **Hypothesis:** Single-stage CAPS-PPO from scratch matches or beats two-stage warm-start `Exp 7c` on `3/3` seeds with `50%` lower high-frequency power (`>10 Hz`) and higher robustness to sensor noise.
- **Where to plug in:** Custom 25-line SB3 `PPO` subclass overriding `train()` in `real_sim/`.

### Riff R1.2 — Architectural Bandwidth Limiting ($\Delta a$-Rate Policy + Learnable EMA Filter)
- **Literature Anchor:** Song et al., *"Reaching the limit in autonomous racing: Optimal control versus RL"* (*Science Robotics* 2023).
- **Idea:** Instead of commanding raw CTBR $a_t \in [-1, 1]^4$, have the actor output bounded *slew rates* $\dot{a}_t \in [-r_{\max}, r_{\max}]^4$ integrated through a fixed or learnable low-pass filter $a_t = \alpha a_{t-1} + (1 - \alpha) u_\theta(o_t)$ inside the policy graph. By Bode's gain-phase relationship (connected to our Proposition 2.1!), bounding policy bandwidth structurally guarantees loop stability up to delay $d$ regardless of reward scale.

### Riff R1.3 — Hybrid Actor-Critic MPC / Residual Policy over Geometric Controller
- **Literature Anchor:** Romero, Song & Scaramuzza, *"Actor-Critic Model Predictive Control: Differentiable Optimization meets RL"* (*IEEE T-RO 2025*); Johannink et al., *"Residual Reinforcement Learning"* (*ICRA 2019*).
- **Idea:** Our `GeometricGateController` (`classical_expert`) already achieves `12/12` gates safely on nominal `figure8` (`~4.1 s` laps) but lacks time-optimality and struggles under unmodeled delays/biases. Train PPO to output a bounded corrective action $a_t = \text{clip}(a_{\text{classical}}(o_t) + 0.35 \, \pi_\theta(o_t), -1, 1)$. Compare sample efficiency (`50k` steps vs `450k` steps) and target crash rates against pure end-to-end PPO.

---

## Step 2 Riffs — Domain & Latency Randomization (Beyond H2')

*Baseline achieved:* Uniform latency randomization (`d ~ U{0..4}`) + action buffer eliminates delay crashes (`Exp 4b`), whereas hand-tuned uniform dynamics randomization hurts transfer (`Exp 9`) and incurs conservatism (`1.00 → 0.73` gates at `d=0`).

### Riff R2.1 — DORAEMON: Entropy-Maximized Curriculum Randomization
- **Literature Anchor:** Tiboni et al., *"DORAEMON: Domain Randomization via Entropy Maximization"* (*ICLR 2024*).
- **Idea:** Hand-picked uniform ranges (`mass_scale ∈ [0.82, 1.18]`, `d ∈ {0..4}`) either under-cover the target (`kf × 1.20`) or over-regularize `d = 0`. Parameterize the source randomization distribution as Beta/Gaussian distributions $\nu_\phi(\xi)$ over `(mass_scale, thrust_scale, torque_scale, tau_scale, action_delay)` and update $\phi$ every PPO rollout to **maximize distribution entropy $\mathcal{H}(\nu_\phi)$ subject to $\mathbb{P}_{\xi \sim \nu_\phi}[\text{clean completion}] \ge 0.80$**.
- **Hypothesis:** Automatically discovers which physical axes `L2F` can tolerate wide variation on (e.g., mass/thrust) vs. tight limits (e.g., high delay near `L2F`'s `T/W = 1.47` ceiling), reducing `d = 0` conservatism while expanding zero-shot coverage to `kf_scale = 1.30`.

### Riff R2.2 — Risk-Sensitive Tail Optimization (`CVaR-PPO` / `EPOpt`)
- **Literature Anchor:** Rajeswaran et al., *"EPOpt: Learning Robust Neural Network Policies Using Model Ensembles"* (*ICLR 2017*); Greenberg et al., *"Efficient Risk-Averse Reinforcement Learning (CeSoR)"* (*NeurIPS 2022*).
- **Idea:** Proposal Section 3 explicitly tracks **`p90` tail lap time** and **crash rate**, yet standard DR-PPO optimizes *expected* return across all sampled domains—allowing the policy to sacrifice the hardest `15%` of high-delay / high-`kf` rollouts to score higher on easy `d = 0` rollouts. Weight PPO policy-gradient updates exclusively (or heavily) on the **bottom $\alpha = 25\%$ (`CVaR_{0.25}`) of episode returns** in each batch.
- **Hypothesis:** Directly tightens `p90_lap_s` and cuts high-delay/high-gain crash rates in `Exp 10`'s robustness sweeps.

### Riff R2.3 — DIDA + Multi-Step Action Chunking for Burst Packet Loss
- **Literature Anchor:** Liotet et al., *"Delayed Imitation with Dataset Aggregation (DIDA)"* (*ICML 2022*); Zhao et al., *"Action Chunking"* (*RSS 2023*).
- **Idea:** Two complementary latency breakthroughs relevant to **both Drone wireless links and Trackmania `TMInterface`**:
  1. **DIDA:** Instead of RL credit assignment through `d = 4..8` delayed ticks, train an undelayed `d = 0` PPO expert $\pi^*$ on `s_{t+d}` (accessible via `CTBRDrone.predict_forward(in_flight)`!) and distill a delay-conditioned policy $\pi_\theta(o_t, a_{t-H:t-1}, d)$ in minutes.
  2. **Open-Loop Action-Chunk Queue (`H = 4` ticks):** Have the 100 Hz policy transmit a *packet of 4 planned actions* $[a_{t|t}, a_{t+1|t}, a_{t+2|t}, a_{t+3|t}]$ every tick. If the wireless link or `TMInterface` drops packets $t+1, t+2$, the onboard 500 Hz controller executes $a_{t+1|t}, a_{t+2|t}$ instead of stale ZOH $a_{t|t}$!
- **Hypothesis:** Pushes tolerable packet-drop / command-drop rate from `p = 0.30` past **`p = 0.70`** on hairline `figure8` apexes.

---

## Step 3 Riffs — Privileged Teacher–Student & Implicit Adaptation (Beyond H3')

*Baseline achieved:* `GRURacingStudent` distilled under randomized dynamics + ZOH dropout recovers sensor-limited flight where `FF` fails (`Exp 5b, 10`), but black-box GRU hidden states conflate *kinematic state estimation* ($\hat{v}_B$) with *dynamics identification* ($m, k_f, \tau_m$).

### Riff R3.1 — RMA + DreamWaQ: Factorized Velocity & Extrinsics Latent Estimation
- **Literature Anchor:** Kumar et al., *"RMA: Rapid Motor Adaptation for Legged Robots"* (*RSS 2021*); Nahrendra et al., *"DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination"* (*ICRA 2023*).
- **Idea:** Replace monolithic GRU action cloning with a two-part **explicitly factorized** architecture:
  1. **Phase 1 (Privileged PPO Policy):** Train $\pi_{\text{priv}}(o_t^{\text{sens}}, v_B, z_t)$ where $z_t = E_\psi(m, k_f, k_m, \tau_m, d) \in \mathbb{R}^6$ is an extrinsics latent vector encoding the true simulator parameters.
  2. **Phase 2 (Adaptation Encoder):** Freeze $\pi_{\text{priv}}$ and train a causal GRU/TCN encoder $g_\phi(o_{t-K:t}^{\text{sens}}, a_{t-K:t-1}) \to (\hat{v}_B, \hat{z}_t)$ with explicit multi-task supervision $\|\hat{v}_B - v_B\|^2 + \|\hat{z}_t - z_t\|^2$!
- **Why this is cleaner than monolithic `GRU_rand`:** We can **plot $\hat{v}_B(t)$ vs. true PyBullet $v_B(t)$** and **run PCA/t-SNE on $\hat{z}_t$** across `L2F`, `PyBullet nominal`, and `PyBullet kf × 1.20` to *visually prove* in the final report that the recurrent observer identifies the target airframe within `0.4 seconds` of takeoff!

### Riff R3.2 — Neural-Fly Online Last-Layer Adaptation (Lap 2 Beats Lap 1 Live!)
- **Literature Anchor:** O'Connell, Shi, Chung et al., *"Neural-Fly enables rapid learning for agile flight in strong winds"* (*Science Robotics* 2022).
- **Idea:** Swift's `k-NN` + `100k` PPO fine-tuning is **offline** (collect 4 target flights → retrain PPO in `L2F` for minutes → redeploy). Inspired by **Neural-Fly**, train a deep SE(3)-invariant basis network $\phi_\theta(v_B, \omega_B, a_t) \in \mathbb{R}^{16}$ across randomized `L2F` airframes so that dynamics residuals are linear in the basis: $\Delta \dot{v}_B \approx W \phi_\theta(x_t)$. At deployment in `PyBullet` (even under time-varying wind or mid-flight rotor degradation `kf × 0.85` on Lap 2!), run **Recursive Least Squares (RLS) with forgetting factor $\lambda = 0.995$** at `100 Hz` (`16 × 16` matrix update = `20 microseconds`/tick!) feeding into an online disturbance-compensating CTBR wrapper.
- **Hypothesis:** **Zero offline target retraining**—the drone adapts *within the first 2 gates of Lap 1* and achieves measurably faster, cleaner Lap 2 times than Lap 1!

### Riff R3.3 — Staleness-Age ($\tau_{\text{stale}}$) Aware Recurrent Gating
- **Literature Anchor:** Lemma 1.1 of our [THEORY_AND_HYPOTHESES.md](../docs/THEORY_AND_HYPOTHESES.md) ($\mathbb{E}[\text{staleness}] = d_o + \frac{p}{1-p}$); Continuous-Time / ODE-RNNs (Rubanova et al., *NeurIPS 2019*).
- **Idea:** In real communication stacks (`Crazyradio` CRTP packet sequence numbers, `TMInterface` frame timestamps), the receiver *knows* whether packet $t$ was freshly received ($\delta_t = 1, \tau_{\text{stale}} = 0$) or held via ZOH ($\delta_t = 0, \tau_{\text{stale}} = \tau_{\text{stale},t-1} + 1$). Appending `[dropped_flag, tau_stale_ticks]` (2 scalars!) to the sensor-limited student input lets the GRU ignore frozen ZOH observations during dropout bursts and roll forward purely on internal action-driven kinematics until a fresh packet arrives.

---

## Step 4 Riffs — Residual Physics & Perception Fine-Tuning (Beyond H4' & Prop 2.7)

*Baseline achieved:* `k-NN` acceleration residual + ground-truth-pose GP bias closes `+69%` (nominal), `+32%` (perturbed), and `+56%` (biased sensor) of the gap (`Exp 9`), but angular acceleration RMSE stays flat (`12 rad/s²`) and OOD residual behavior relies on heuristic distance decay.

### Riff R4.1 — **Headline CMU Riff:** ASAP Delta-Action Alignment vs. Swift Delta-Dynamics
- **Literature Anchor:** He, Gao, Xiao, Yuan, Shi, Liu et al. (CMU LeCAR Lab & NVIDIA), *"ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills"* (*RSS 2025*).
- **Core Insight Connected to `Exp 6/8`:** Swift injects an additive rigid-body acceleration residual $\dot{v}_B \leftarrow \dot{v}_{B,\text{L2F}} + \hat{r}_v(s, a)$. That *bypasses* `L2F`'s `500 Hz` rate controller, rotor thrust polynomial saturation (`T/W = 1.47`), and asymmetric spin-up/down filter—which is why angular residuals stayed stuck at `12 rad/s²`! **ASAP** instead learns a small frozen **delta-action network** $\Delta a_t = \pi_\Delta(v_B, \omega_B, a_t) \in [-0.4, 0.4]^4$ so that stepping `L2F` with $a_t + \Delta a_t$ reproduces PyBullet's next state $s_{t+1}^{\text{PyB}}$!
  - Because $a_t + \Delta a_t$ flows *through* `L2F`'s `500 Hz` rate loop and motor allocator, $\pi_\Delta$ naturally captures thrust-gain scaling (`kf × 1.20`), yaw-to-thrust coupling (`km/kf`), and rate-damping mismatch!
  - Once $\pi_\Delta$ is frozen inside `GateRacingEnv("source")`, we fine-tune PPO for the exact same `100k` steps and deploy $\pi_{\text{PPO}}$ directly to `PyBullet` *without* $\pi_\Delta$.
- **Hypothesis:** Direct head-to-head table in our CMU 16-831 report (**Swift Delta-Dynamics `k-NN` vs. CMU ASAP Delta-Action $\pi_\Delta$** at identical `1,600` target samples and `100k` source FT steps): ASAP reduces multi-step angular & position rollout error beyond `k-NN` and closes an even larger fraction of the `perturbed` (`kf × 1.20`) racing gap!

### Riff R4.2 — MOPO Pessimistic Residual Fine-Tuning (Epistemic Uncertainty Penalty)
- **Literature Anchor:** Yu et al., *"MOPO: Model-based Offline Policy Optimization"* (*NeurIPS 2020*).
- **Idea:** Replace `KNNResidual` with either a **5-member MLP Residual Ensemble** $\{r_{\psi_i}(s, a)\}_{i=1}^5$ or **Sparse GP Dynamics Residual**, giving both a mean residual $\bar{r}(s,a)$ and an exact epistemic disagreement $\sigma_{\text{ep}}(s,a) = \sqrt{\frac{1}{5}\sum_i \|r_{\psi_i}(s,a) - \bar{r}(s,a)\|^2}$. During `100k`-step PPO fine-tuning in the residual-augmented `L2F` source, modify the step reward per MOPO:
  $$\tilde{R}(s_t, a_t) = R_{\text{race}}(s_t, a_t) - \beta \, \sigma_{\text{ep}}(s_t, a_t)$$
- **Why it matters:** Without $-\beta \sigma_{\text{ep}}$, PPO fine-tuning can push the drone into aggressive high-rate states unseen in the 4 calibration flights where residual extrapolation is wrong. With $-\beta \sigma_{\text{ep}}$, PPO maximizes a rigorous lower bound on true target return!

### Riff R4.3 — Iterative / 2-Round Bootstrapped Swift vs. Direct Policy-Residual RL
- **Literature Anchor:** Kaufmann et al. (*Nature* 2023); Silver et al., *"Residual Policy Learning"* (2018).
- **Idea:** Split the `4`-flight (`1,600`-sample) target budget into **2 flights of Round-1 calibration** (`800` samples → `50k` FT) + **2 flights of Round-2 on-policy calibration** (`800` samples collected by the faster Round-1 policy → `50k` FT), testing whether *active distribution alignment* beats single-batch `1,600`-sample calibration at *strictly equal* target + source budgets! Compare both against training a tiny `2-layer` residual policy $\pi_{\text{res}}(o)$ directly in PyBullet for just `4,000` target steps (`40 s` of flight).

---

## Step 5 Riffs — Cross-Domain Synthesis & Unpausing Trackmania (`TMNF-C` → `TMInterface`)

*Current state:* `TMNF-C` is already patched for macOS (`code/patches/tmnf-c-macos.patch`) and passing `6/6` `ctest` tests in `1.06 s`, and `AsyncWallClockTargetWrapper` is implemented in `code/sims/l2f_and_async_wrappers.py`.

### Riff R5.1 — The $\Delta f \equiv 0$ Cross-Domain Falsification Experiment
- **Idea:** When Trackmania unpauses, wire `TMNF-C` (`source`, deterministic physics) → `TMNF-C + AsyncWallClockTargetWrapper` (or Windows `TMInterface` live game: `d = 2..4` tick latency, jitter, frame drops, noisy speed/gear/slip observations). Run **both Swift `k-NN` dynamics residual and ASAP $\Delta a$ residual** alongside our Step-0 delay alignment:
  - Because `TMNF-C` matches live Trackmania physics tick-for-tick ($\Delta f \equiv 0$), Theorem 2.5 predicts **naive ($d'=0$) residual fitting hallucinates a large phantom tire/braking residual**, whereas **delay-aligned ($d'=\hat{d}$) dynamics residual collapses to $\approx 0$**, proving quantitatively that the same unified diagnostic pipeline automatically separates *physics gaps* (Drone) from *interface gaps* (Trackmania)!

### Riff R5.2 — Adaptive Procedural Track Curriculum for Zero-Shot Track Generalization
- **Literature Anchor:** Hanover, Kaufmann, Scaramuzza et al., *"Autonomous Drone Racing: A Survey"* (*IEEE T-RO 2024*) & UZH Adaptive Track Shaping (2024–2025).
- **Idea:** In `Exp 10`, training on a single flat `figure8` transfers surprisingly well to `ellipse3d` thanks to SE(3)-invariant local centerline + gate observations (`FastCenterline`), but extreme 3D hairpin climbs still stress vertical authority. Parameterize random closed 3D Fourier courses $p(t) = \sum_{k=1}^3 (a_k \cos k\omega t + b_k \sin k\omega t)$ filtered for `L2F` thrust feasibility ($\|\ddot{p}(t) + g e_z\| \le 0.90 \, T_{\max}/m$), and evaluate zero-shot transfer across a **10-track benchmark** (Split-S, 3D corkscrew, ascending hairpin, chicane) in PyBullet!

---

## Empirical Execution Status — All 14 Drone Riffs Completed (`Exp 11–18`, 2026-11-26!)

Every drone riff in `Steps 0–5` above has now been implemented and evaluated across 3 seeds (`mean ± std`) on `rl-tools/l2f` → `gym-pybullet-drones` CF2X (full tables in [REAL_SIM_RESULTS.md](../docs/REAL_SIM_RESULTS.md)):

1. ✅ **[exp11_caps_smoothness.py](../real_sim/exp11_caps_smoothness.py) (`R1.1` CAPS + L2C2 Smoothness):** `3/3` seeds learn in one stage (`0%` crashes across 10 conditions), `44×` lower `‖Δa‖`, and `0.079 ± 0.026 m` target error (`2.3×` better than two-stage warm-start `Exp 7c`).
2. ✅ **[exp12_chunking_staleness.py](../real_sim/exp12_chunking_staleness.py) (`R2.3 + R3.1 + R3.3` DIDA Chunk-4 + Track-Free RMA):** `0.975 ± 0.030` gate completion at `80%` command-packet loss; uncovered & fixed the clockwise-`figure8` → counterclockwise-`ellipse3d` observer shortcut trap (`4.33 → 1.31 m/s`) and online thrust-shift detection (`z_thr = +0.58…+0.71`).
3. ✅ **[exp13_neural_fly_online.py](../real_sim/exp13_neural_fly_online.py) (`R3.2` Neural-Fly Live 100 Hz Composite RLS):** `1.000 ± 0.000` (`18/18` gates, zero offline target data) on held-out 3D `ellipse3d` in `0.5g` crosswind (`−67%` RMSE vs unadapted, `−43%` vs PID) and mid-race `-25%` motor loss (`−45%` vs PID), plus experimental verification of O'Connell et al.'s algebraic-loop instability (`0.167` gates).
4. ✅ **[exp14_asap_mopo_residuals.py](../real_sim/exp14_asap_mopo_residuals.py) (`R4.1 + R4.2 + R5.1` CMU ASAP $\Delta a$ + MOPO Ensemble vs. Swift `k-NN`):** `swift_mopo_ft` achieved `1.00` collision-free completion across all `4/4` `(target × course)` cells (`4.97–5.12 s` laps); `asap_delta_a` achieved best `0.5 s` drift (`0.107 m`) and `1.00` clean completion within `L2F`'s motor envelope (`target_nominal`) while revealing the source-actuator clipping limit (`target_perturbed`, `T/W = 2.70 > 1.47`), and Step-0 alignment collapsed $\Delta f \equiv 0$ phantom residuals `11×` (`0.190 → 0.0174 m/s²`).
5. ✅ **[exp15_fisher_shooting_sysid.py](../real_sim/exp15_fisher_shooting_sysid.py) (`R0.1 + R0.2` Fisher Safe Excitation + 10-Tick Parametric Shooting):** ⚠️ (audit) correct-cell `(d̂, τ_up, τ_down)` recovery on a 4×4 grid that contains the truth — but a Gaussian random walk does equally well, so no Fisher advantage is shown, and all strategies exceed the 0.5 m safety sphere in the target at ≥1 s; parametric-only shooting cut 3 s drift `18%` vs `65%` for 6D `k-NN` because $J$, `km/kf`, and downwash lie outside `L2F`'s JSON parameter set.
6. ✅ **[exp16_bandwidth_and_residual_policy.py](../real_sim/exp16_bandwidth_and_residual_policy.py) (`R1.2 + R1.3` EMA Bandwidth Filter + Residual-over-Classical RL):** 5-line EMA wrapper (`α=0.50–0.70`) cut PyBullet hold error `53–73%` across `d=0..8` (`0.092 m` @ `d=8`); Residual-over-Classical gave `15–24×` higher gate completion at `50k` steps, with unconstrained E2E crossing over near `L2F` saturation (`450k+`, Song et al., *Sci. Robot.* 2023).
7. ✅ **[exp17_doraemon_cvar_dr.py](../real_sim/exp17_doraemon_cvar_dr.py) (`R2.1 + R2.2` DORAEMON Curriculum DR + CVaR-PPO):** `doraemon_ppo` eliminated `source_d0` crashes (`47% → 0%`, `gf = 0.972`) and beat `fixed_wide_dr` by `+16..+32` gate points (`3.1 s` faster `p90` tail laps), while `CVaR_{0.35}` confirmed Proposition 2.3's risk–speed conservatism trade-off.
8. ✅ **[exp18_bootstrapped_swift_and_10tracks.py](../real_sim/exp18_bootstrapped_swift_and_10tracks.py) (`R4.3 + R5.2` 2-Round Bootstrapped Swift + 10-Track 3D Benchmark):** At equal `16 s` (`1,600`-tick) target budget (`kf × 1.25, d=3`), 2-round active bootstrap cut high-speed residual RMSE `33%` (`0.360 → 0.241 m/s²`), cut 10-track macro tracking error `50%` (`0.135 → 0.067 m`), and achieved `1.00` clean completion on `9/10` unseen 3D circuits (`0.900 ± 0.000` macro clean vs `0.472` zero-shot and `0.50` direct `16 s` target PPO).
