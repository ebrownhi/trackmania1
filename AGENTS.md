# AGENTS.md — Onboarding Guide for Coding Agents

**Project:** *One Method, Two Racing Worlds: Sim-to-Sim Transfer in Drone and Trackmania Racing*  
**Course:** CMU 16-831 Introduction to Robot Learning (Fall 2026)  
**Proposal PDF:** `16831_Final_Project.pdf` (in this directory)  
**Primary Dev Machine:** Apple M5 Pro (`arm64`, 15 CPU cores, 24 GB Unified RAM, macOS 26.6.2, Apple Clang 21.0.0)

Welcome! Read this file **before** writing simulator wrappers, training scripts, or environment configs. We conducted an upfront theoretical and systems audit to identify hidden mathematical failure modes and macOS `arm64` build traps *before* implementation.

---

## 1. Directory Map & Documentation Index

* `16831_Final_Project.pdf` — Original 4-person project proposal defining hypotheses **H1–H4**, metrics, baselines, and ablations.
* `AGENTS.md` *(this file)* — High-level operational rules, architectural invariants, and mandatory implementation guardrails.
* `docs/THEORY_AND_HYPOTHESES.md` — Formal **Delayed-POMDP Sim-to-Sim Framework**, proofs (Theorem 1.2 Return Gap Bound, Schur delay stability, RNN OOD bias amplification, phantom `k-NN` residuals), and refined hypotheses **H1'–H4'**.
* `docs/MAC_STACK_AND_PIVOTS.md` — Local M5 Pro toolchain audit, `pixi`/`uv` bootstrap commands, `TMInterface` lockstep finding, `AsyncWallClockSocketServer` architecture, and time-boxed Pivot Triggers.
* `docs/SIMULATOR_PHYSICS_AUDIT.md` — Exact ODEs, URDF constants, frame conventions, and source-code gotchas across Swift, `L2F`, `gym-pybullet-drones`, `TMNF-C`, and `Linesight`/`TMInterface`.
* `docs/EMPIRICAL_THEORY_VERIFICATION.md` — Gemini toy-simulator verification (`theory_sim/`); error bars are SEM, see corrections in the next doc.
* `docs/REAL_SIM_RESULTS.md` — Verified results on the real `l2f` → `gym-pybullet-drones` pair (`real_sim/`, `code/sims/drone_pair.py`, outputs in `results/real_sim/`).

---

## 2. Strict Workspace & Environment Rules

1. **NEVER modify sibling homework folders:**
   * Do **not** edit or install packages into `../hw3/.venv`, `../hw3/.venv-fresh`, or `../16831-F26-HW-upstream-REFERENCE-ONLY/`.
   * Those environments are pinned to legacy course versions (`torch==1.12.1`, `gym==0.25.1`, `mujoco==2.2.1`) that conflict with `gymnasium>=1.0` and modern PyTorch/MPS.
2. **Use a dedicated Python 3.11 environment inside `Final Project/`:**
   * Prefer **`pixi` (`conda-forge`)** (`brew install cmake ninja pixi`) so `pybullet`, `mujoco`, `pytorch`, `scipy`, `scikit-learn`, and `numba` install from prebuilt `osx-arm64` binaries in seconds.
   * **Apple Clang 21 + `pybullet` Trap:** PyPI has *no* `macosx_arm64` wheel for `pybullet`, and bare `pip install pybullet` fails on Clang 21 (`fdopen` zlib macro clash + `-Wincompatible-function-pointer-types`). Either install `pybullet` from `conda-forge` via `pixi` or pass:
     `CFLAGS="-Dfdopen=fdopen -Wno-error=incompatible-function-pointer-types -Wno-error=int-conversion"`.
   * Prefer the actively maintained Gymnasium fork `learnsyslab/gym-pybullet-drones` over legacy `utiasDSL` `gym==0.21` pins.
3. **Reusable 16-831 Patterns (Read-Only Reference):**
   * **GAE($\lambda$) & Advantage Standardization:** `../16831-F26-HW/hw2/rob831/agents/pg_agent.py`
   * **Normalized State-Delta ($\Delta s$) Residual Regression:** `../hw3/hw4/rob831/hw4_part1/models/ff_model.py`
   * **3-Seed Parallel Runner & Provenance Logging:** `../hw3/hw3/RUN.md`

---

## 3. System Architecture: Two Matched Source–Target Pipelines

| Dimension | Branch 1: Drone Racing *(Dynamics & Actuator Gap)* | Branch 2: Trackmania Racing *(Interface & Timing Gap)* |
| :--- | :--- | :--- |
| **Source Sim (Fast Train)** | **`L2F` (`rl-tools/l2f`)** (100 Hz policy $\to$ 500 Hz inner rate loop, Apple `Accelerate`) | **`TMNF-C` (`adonis-singh/TMNF-C`)** (C11 CPU `-DTMNF_CUDA=OFF`, 100 Hz vectorized Gymnasium) |
| **Target Sim (Zero-Shot & Residual Eval)** | **`gym-pybullet-drones`** (`CF2X`, `Physics.PYB_GND_DRAG_DW`, `pyb_freq=500`, custom motor-lag $\tau_m$ + 500 Hz rate PID) *(Pivot B: `Crazyflow` / `MuJoCo`)* | **Mac Target:** `AsyncWallClockTMNFTarget` (100 Hz socket server + latency/jitter/dropout)<br>**Windows Final Eval:** Live `TmForever.exe` + `TMInterface` |
| **Physics Gap ($\Delta f_{\text{markov}}, \Delta f_{\text{hid}}$)** | **Non-zero:** RK4 vs. Euler, quadratic/rotor drag, thrust curve, 1st-order motor lag $\tau_m$, rate-PID tracking | **Identically Zero ($\Delta f \equiv 0$):** `TMNF-C` matches `TmForever.exe 2.11.26` tick-for-tick at 100 Hz |
| **Interface Gap ($d_a, d_o, \eta, m_t$)** | Optional injected latency, noise, and sensor dropout | **Primary Gap:** Asynchronous wall-clock stepping, action/obs latency, 60 Hz aliasing, dropout |

### Critical Simulator Implementation Notes
* **Shared CTBR Wrapper Required on Drone:** Neither `l2f` nor `gym-pybullet-drones` exposes Swift's **Collective Thrust + Body Rates (`[c, ω_x, ω_y, ω_z]`)** natively (both take 4-motor RPMs). Wrap *both* in an explicit 500 Hz inner rate P/PID + mixer loop stepped $5\times$ per 100 Hz RL step. Verify frame conventions (`FLU` vs `NWU`/`NED`: `+thrust -> +a_z`, `+ω_x -> same roll sign`) via unit test before any RL training!
* **The `TMInterface` Synchronous Lockstep Trap:** By default, `Linesight` / `TMInterface`'s `OnRunStep` **pauses `TmForever.exe` at every 100 Hz tick** waiting for Python. Because `TMNF-C` is tick-for-tick identical to `TmForever.exe`, paused lockstep evaluation with exact `SimStateData` produces **literally zero transfer gap**! Always evaluate Trackmania against an **asynchronous wall-clock / latency-injected target** (`AsyncTMNFTarget` on Mac; unlocked/perturbed `TMInterface` on Windows), keeping $d=0$ lockstep strictly as a sanity-check baseline.

---

## 4. Six Mandatory Theoretical Guardrails (Do NOT Violate!)

Our mathematical stress-test (`docs/THEORY_AND_HYPOTHESES.md`) proved six ways a literal/naive implementation breaks **H1–H4**. Follow these invariants across all code:

1. **Action-Smoothness Penalty Required in Nominal PPO (H1 Guardrail):**
   * Discrete delay stability collapses $4.5\times$ by $d=3$ ticks ($g_{\text{crit}}(d) = 2\sin(\frac{\pi}{2(2d+1)})$). Unregularized PPO saturates $\tanh$ into bang-bang chattering that limit-cycles into Trackmania walls even at $d=1\text{--}2$. Always include $-\lambda_{\Delta a}\|a_t - a_{t-1}\|^2$ in nominal rewards (and ablate $\lambda_{\Delta a} = 0$ explicitly). Also ensure the drone course reaches $v_{\text{peak}} \ge 10\text{ m/s}$ at $\Delta t_{\text{RL}} = 10\text{--}20\text{ ms} \lesssim \tau_m$ so quadratic drag and motor lag are active.
2. **Use an Action History Buffer $a_{t - d_{\max} : t - 1}$, Not Just `previous_action` $a_{t-1}$ (H2 Guardrail):**
   * For total loop delay $d = d_a + d_o \ge 2$, $(s_{t-d_o}, a_{t-1})$ is **non-Markovian** (missing $d-1$ in-flight actions) and cannot learn Smith-predictor state forward-propagation (Theorem 2.2). Standardize feed-forward actors on `(obs, route, a_{t - d_max : t - 1})` ($d_{\max} \approx 6\text{--}8$), keeping `a_{t-1}`-only as an ablation.
3. **Implement Observation Dropout via Zero-Order Hold (ZOH), NEVER Zero-Masking (Lemma 1.1):**
   * Setting dropped observations to `0.0` injects huge $\mathcal{O}(p_{\text{drop}} L_\pi \|h(s_t)\|)$ impulses. Always hold the last valid reading $o_t = (1 - m_t) y_{t - d_o} + m_t o_{t-1}$ and optionally append the validity bit $(1 - m_t)$.
4. **Distill Recurrent Students via On-Policy DAgger Over Randomized Transitions (H3 Guardrail):**
   * Distilling a GRU/LSTM student on **fixed nominal source dynamics $\mathcal{M}_S$** trains an $f_S$-locked observer whose small smoothing gain $K \ll 1$ **amplifies steady-state Drone physics mismatch by $\frac{1-K}{K}$** and drives hidden states $h_t$ out-of-distribution (Theorem 2.4), making recurrent students *worse* than feed-forward on Drone! Always distill recurrent students inside the **domain- and latency-randomized** simulator (RMA-style joint filtering + implicit SysID), and report the normalized **Observability Gap Recovery Ratio** $\rho_{\text{obs}}$.
5. **Time-Align Target Rollouts by Estimated Delay $\hat{d}$ Before Fitting `k-NN` Dynamics Residuals (H4 Guardrail):**
   * In Trackmania ($\Delta f \equiv 0$) with loop delay $d \ge 1$, regressing $\Delta v / \Delta t$ naively on client-logged $(\hat{s}_t, a_t)$ instead of executed $a_{t-d}$ learns a **massive phantom acceleration residual** $J_{v,a}(s)\mathbb{E}[a_{t-d} - a_t \mid s, a] \neq 0$ (e.g., $+10\text{--}25\text{ m/s}^2$ at braking zones!), falsifying H4 (Theorem 2.5). Always estimate integer delay $\hat{d} = \arg\min_{d'} \sum \|\hat{s}_{t+1} - f_S(\hat{s}_t, a_{t-d'})\|^2$ first (via scripted-action replay) and fit `k-NN` on aligned $(\hat{s}_t, a_{t-\hat{d}})$.
6. **SE(3)-Invariant Body-Frame `k-NN` Features + Systematic Sensor Bias $b(s)$ for `GP` (H4 Guardrail):**
   * Never pass world position $p_W$ into `k-NN` dynamics residuals—it will fail completely on the held-out course. Use strictly body-frame features $(v_B, \omega_B, a_t, a_{t-1} \text{ or } \hat{\Omega}_{t-1})$ with kernel decay to $0$ outside rollout support.
   * Swift's GP perception residual collapses to $\hat{\mu}_{\text{GP}} \to 0$ under pure zero-mean i.i.d. noise/dropout (Prop. 2.7). Sensor-limited targets must include **relative-pose-correlated systematic bias** $b(\phi_{\text{rel}}(s))$ (e.g., oblique gate-viewing-angle PnP bias or perspective bias).

---

## 5. Summary of Pivot Triggers (See `docs/MAC_STACK_AND_PIVOTS.md`)

* **Gate 1 (Day 1, 2h cap — Drone Target):** If `pybullet` build or aero stability snags $>2\text{h}$ on M5 Pro $\implies$ **Pivot B:** switch Drone target to **`learnsyslab/crazyflow`** (native `osx-arm64`, built-in 500 Hz `Control.body_rate` CTBR) or native **`mujoco`** (`bitcraze_crazyflie_2`).
* **Gate 2 (Day 3, 48h cap — `TMNF-C` Track Snapshots):** `TMNF-C` builds and passes synthetic `ctest` on Mac with zero game files, while campaign tracks (`A01`–`A03`) need a one-time 10-min Windows snapshot export. If Windows snapshots stall $>48\text{h}$ $\implies$ **Sub-Pivot A2:** run ground racing on **`f1tenth_gym`** (100 Hz Pacejka bicycle, 100% asset-free `osx-arm64`).
* **Gate 3 (Day 9 — Live Windows `TMInterface`):** Use **`AsyncWallClockSocketServer`** (two-process 100 Hz wall-clock server over UDP/Unix socket on Mac) for all daily Trackmania sweeps, and fall back to it as primary target if Windows `TMInterface` integration flakes out.
