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
* `docs/WORK_WRITEUP.md` — **Start here:** status of every drone hypothesis, what was done in each round, the Exp 10 audit and the Round 5–6 fixes (Exp 19–24), and what is still open.
* `docs/FOLLOWUP_RIFFS_ROUND2.md` — Ranked next ideas from the 2026-11-26 literature scan (drone tiers 1–2, Trackmania gap ladder).
* `docs/REAL_SIM_RESULTS.md` — Verified results on the real `l2f` → `gym-pybullet-drones` pair (`real_sim/`, `code/sims/drone_pair.py`, outputs in `results/real_sim/`).
* `docs/FOLLOWUP_RIFFS_PROPOSAL.md` — Literature-backed Phase-2 research proposal with 14 concrete extensions across Steps 0–5 (`ASAP`, `CAPS`/`L2C2`, `Neural-Fly`, `MOPO`, `DORAEMON`, `DIDA`, `RMA`/`DreamWaQ`).

---

## 2. Strict Workspace & Environment Rules

1. **NEVER modify sibling homework folders:**
   * Do **not** edit or install packages into `../hw3/.venv`, `../hw3/.venv-fresh`, or `../16831-F26-HW-upstream-REFERENCE-ONLY/`.
   * Those environments are pinned to legacy course versions (`torch==1.12.1`, `gym==0.25.1`, `mujoco==2.2.1`) that conflict with `gymnasium>=1.0` and modern PyTorch/MPS.
2. **Use a dedicated Python 3.12 `pixi` environment inside `Final Project/code/`:**
   * Run `pixi install` then `pixi run setup-sims` (`code/setup_sims.sh`), which uses `conda-forge` `pybullet`/`pytorch`/`mujoco` and builds the `l2f` wheel cleanly via Homebrew `python@3.12` (avoiding spaces-in-path `CFLAGS` issues). Run `pixi run test` to execute the fast regression suite.
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

1. **Apply Action-Smoothness Penalty via Warm-Start / Ramp, Not Large From-Scratch Penalty (H1' Guardrail):**
   * Discrete delay stability collapses $4.5\times$ by $d=3$ ticks ($g_{\text{crit}}(d) = 2\sin(\frac{\pi}{2(2d+1)})$; relay limit-cycle period is $(4d+2)\Delta t$). However, large $\lambda_{\Delta a} = 0.3$ *from scratch* can stall PPO exploration (Exp 7b). Train base PPO with small/zero $\lambda_{\Delta a}$ (`0.0`–`0.05`) and ramp/warm-start $\lambda_{\Delta a} \in [0.1, 0.3]$ during fine-tuning (Exp 7c). Note `L2F` Crazyflie's $T/W \approx 1.47$ caps course speed near $3.55\text{ m/s}$.
2. **Use an Action History Buffer $a_{t - d_{\max} : t - 1}$, Not Just `previous_action` $a_{t-1}$ (H2' Guardrail):**
   * For total loop delay $d = d_a + d_o \ge 2$, $(s_{t-d_o}, a_{t-1})$ is **non-Markovian** (Theorem 2.2). Standardize actors on `(obs, route, a_{t - d_max : t - 1})` ($d_{\max} \approx 8\text{--}16$), keeping `a_{t-1}`-only as an ablation. Note zero-mean dynamics DR does not fix directional airframe offsets (Exp 9).
3. **Implement Observation Dropout via Zero-Order Hold (ZOH), NEVER Zero-Masking (Lemma 1.1):**
   * Setting dropped observations to `0.0` crashes every seed for $p_{\text{drop}} \ge 0.2$ (Exp 3). Always hold the last valid reading $o_t = (1 - m_t) y_{t - d_o} + m_t o_{t-1}$.
4. **Distill Recurrent Students via On-Policy DAgger Over Randomized Transitions (H3' Guardrail):**
   * Always distill GRU students inside the **domain- and latency-randomized** simulator with sufficient DAgger rollouts under realistic sensor delay/dropout/bias, and report **std across seeds** (never SEM!) alongside $\rho_{\text{obs}}$.
5. **Run Step-0 Scripted Multi-Sine Joint $(\hat{d}, \hat{\tau}_m)$ Lag-Filtered SysID Before `k-NN` Fitting (H4' Guardrail):**
   * On real `L2F` $\to$ `PyBullet`, delay-only scans fail without matching $\tau_m$ (Exp 6), and regressing thrust-scale residuals against raw unfiltered white-noise commands $a_{t-d}$ biases $\hat{d}$ by $+1$ tick (Exp 9). Always use **Step-0 scripted multi-sine excitation** and regress affine scale terms against the **reconstructed lag-filtered rotor thrust** $\bar{u}_{\text{filt}}(\tau', d')$.
6. **SE(3)-Invariant Body-Frame `k-NN` Features + Ground-Truth-Pose `GP` Input (H4' Guardrail):**
   * Never pass world position $p_W$ into `k-NN` dynamics residuals (gave $0\%$ held-out gain vs. $-60\%$ RMSE in body frame in Exp 6/8).
   * Always fit the GP perception residual $o_i - s_i^T$ on **ground-truth (mocap) pose** $\phi(s_i^T)$—conditioning on noisy observed pose $\phi(o_i)$ leaks sensor noise into the regressor and learns a spurious bias under the i.i.d. null (Exp 9)!

---

## 5. Summary of Pivot Triggers (See `docs/MAC_STACK_AND_PIVOTS.md`)

* **Gate 1 (Day 1, 2h cap — Drone Target):** If `pybullet` build or aero stability snags $>2\text{h}$ on M5 Pro $\implies$ **Pivot B:** switch Drone target to **`learnsyslab/crazyflow`** (native `osx-arm64`, built-in 500 Hz `Control.body_rate` CTBR) or native **`mujoco`** (`bitcraze_crazyflie_2`).
* **Gate 2 (Day 3, 48h cap — `TMNF-C` Track Snapshots):** `TMNF-C` builds and passes synthetic `ctest` on Mac with zero game files, while campaign tracks (`A01`–`A03`) need a one-time 10-min Windows snapshot export. If Windows snapshots stall $>48\text{h}$ $\implies$ **Sub-Pivot A2:** run ground racing on **`f1tenth_gym`** (100 Hz Pacejka bicycle, 100% asset-free `osx-arm64`).
* **Gate 3 (Day 9 — Live Windows `TMInterface`):** Use **`AsyncWallClockSocketServer`** (two-process 100 Hz wall-clock server over UDP/Unix socket on Mac) for all daily Trackmania sweeps, and fall back to it as primary target if Windows `TMInterface` integration flakes out.
