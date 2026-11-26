# Real-Simulator Results: L2F → gym-pybullet-drones (Drone Branch)

**Date:** 2026-11-25 (round 2 same day) · **Scope:** drone only (Trackmania paused) · **Code:** `real_sim/`, `code/sims/drone_pair.py` · **Outputs:** `results/real_sim/` (JSON + PNG per experiment)

All experiments use the *real* simulators: `rl-tools/l2f` 2.0.18 (source, Crazyflie preset, dt = 2 ms) and `gym-pybullet-drones` CF2X (target, `PYB_GND_DRAG_DW`, 500 Hz, + 30 ms motor lag). Both are driven by one shared 500 Hz CTBR rate loop calibrated to the source airframe; the policy runs at 100 Hz. The gap is real: 12% mass mismatch, different inertia / thrust map / yaw torque, motor lag 55/249 ms vs 30 ms, drag only in the target. Students are DAgger-distilled from a privileged geometric expert unless stated; PPO only in Exp 7. Error bars are **std across seeds** (not SEM).

Run any experiment with `../code/.pixi/envs/default/bin/python real_sim/<script>.py` from `Final Project/real_sim/`.

## Verdict summary

| Claim | Experiment | Verdict |
|---|---|---|
| Proposal Step 1: open-loop divergence | Exp 1 | ✅ 0.87 m @ 1 s, 3.7 m @ 3 s; source replay exactly 0 |
| Prop 2.1 delay stability limit | Exp 2 | ✅ PyBullet matches `2 sin(π/(2(2d+1)))` within 3%; delay+lag model within 4% |
| Relay limit-cycle period `4(d+1)` | Exp 2 | ⚠️ **Corrected:** measured and exact discrete model give **4d+2 ticks** |
| Lemma 1.1 ZOH vs zero-mask dropout | Exp 3 | ✅ ZOH: 100% gates up to p=0.5; zero-mask crashes every seed at p ≥ 0.2; staleness = d_o + p/(1-p) |
| Thm 2.2 action buffer (H2 guardrail) | Exp 4, 4b | ✅ buffer16 survives d=8–12 where a_{t-1} crashes; randomized-latency buffer: 0 crashes in target at all d |
| H2' dynamics DR helps | Exp 4b, 9 | ❌ no gain in Exp 4b; DR fine-tuning *hurt* target hold error on every target in Exp 9 (−29…−65% of gap) |
| H1' smoothness penalty (PPO) | Exp 7, 7b, 7c | ⚠️ partial: penalty alone (Exp 7) ❌; with buffer + latency DR, λ=0 learns bang-bang control whose target error is 4.5× its source error; λ=0.3 warm-start halves chattering and lowers target error at every d (0.26→0.18 m at d=0, 0.50→0.27 m at d=1); λ=0.3 from scratch fails to learn on 2/3 seeds |
| Thm 2.4 recurrent-observer OOD | Exp 5, 5b | ⚠️ partial: with a 2.3× larger gap, fixed-dynamics estimators degrade (GRU 0.21, FF 0.23 m/s) and randomized ones hold (0.14, 0.15 m/s); 'GRU_fixed worse than FF' not observed |
| H3' recurrent students + randomized distillation | Exp 5, 5b | ⚠️ partial: under a hard sensor (obs delay 3, 20% dropout, no velocity) FF students crash everywhere and GRUs recover ρ_obs ≈ 0.7 on the nominal target; on the perturbed target GRU_fixed crashes 100% vs GRU_rand 67% (high seed variance) |
| Nuance #1 (velocity removal unobservable) | Exp 5 | ❌ not reproduced (2 cm position noise still gives ~0.1 m/s velocity) |
| Thm 2.5 phantom residual / delay alignment | Exp 6 | ✅ Δf≡0 control: d̂=3 exact, naive 0.12 vs aligned 0.007 m/s² (17×) |
| Guardrail #5 + Nuance #2 joint (d̂, τ̂) SysID | Exp 6 | ✅ needed on PyBullet: delay-only scan fails (d̂=0); joint scan recovers d̂=3, τ̂=30 ms exactly |
| Guardrail #6 body-frame vs world-frame k-NN | Exp 6, 8 | ✅ world-frame gives 0% held-out improvement; body-frame −60% residual RMSE |
| H4' residual model closes gap | Exp 8, 9, 10, 14 | ✅ model fidelity (replay 0.48 → 0.064 m); ✅ hover FT closes +69% (nominal) and +32% (perturbed) vs DR −29…−65%; ✅ **6-gate 3D racing (`Exp 10, 14`)**: `residual_ft_ppo` & `swift_mopo_ft` hit **`1.000 ± 0.000` gate completion and `1.00` collision-free completion** on held-out `ellipse3d` (vs `0.33–0.43` clean for zero-shot nominal PPO and `0.00` for DR) |
| Prop 2.7 GP perception residual | Exp 9, 10 | ✅ biased sensor: GP recovers systematic bias and FT closes +56% of gap; zero-bias / i.i.d. noise null: GP mean collapses to **0.0008–0.003 m** (`Exp 9, 10`) when conditioned on ground-truth pose |
| Riff R1.1: CAPS + L2C2 actor-loss smoothness | Exp 11 | ✅ **Eliminates `Exp 7b` 2/3-seed stall** (`3/3` seeds learn from scratch, `0%` crashes across all 10 conditions), cuts chattering `44×` (`‖Δa‖: 1.091 → 0.025`), and beats even 2-stage warm-start `Exp 7c` (`0.079 ± 0.026 m` vs `0.182 m` @ `d=0`, `0.084 m` @ `d=6`) |
| Riff R2.3 + R3.1/R3.3: DIDA 4-step chunking & RMA | Exp 12 | ✅ **DIDA Chunk-4** holds **`1.000 → 0.975 ± 0.030`** gates (`0` crashes) up to **`80%` command-packet loss**; track-free **RMA/GRU** `2–3×`s FF gate completion (`0.24 vs 0.10`), cuts `v_B` RMSE `33%` (`1.97 → 1.31 m/s`), detects target thrust shift (`z_thr = +0.58…+0.71`), and exposes a **gate-coordinate shortcut trap** (`4.33 m/s` sign-inversion on `ellipse3d` when observers see track coordinates) |
| Riff R3.2: Neural-Fly live 100 Hz composite RLS | Exp 13 | ✅ **Zero offline target data:** **`1.000 ± 0.000` (`18/18` gates, 3 laps)** on 3D `ellipse3d` in `0.5g` turbulent crosswind (`0.112 m` RMSE: `−67%` vs unadapted, `−43%` vs `PID`) and mid-race `-25%` motor loss (`−45%` vs `PID`); confirms **algebraic-loop divergence** (`0.167` gates) when `a_cmd` enters basis $\phi$ |
| Riff R4.1 + R4.2: CMU ASAP $\Delta a$ vs Swift + MOPO | Exp 14 | ✅ **`swift_mopo_ft`** (`5-MLP` ensemble + $-\beta\sigma_{\text{ep}}$, `2.3–3.3×` OOD separation) hits **`1.00` clean completion across all `4/4` `(target × course)` cells**; **`ASAP` $\Delta a$** gets best `0.5 s` drift (`0.107 m`) & `1.00` clean completion within `L2F`'s motor envelope (`target_nominal`), but clips at `a=+1` when target exceeds source ceiling (`kf × 1.20`, `T/W = 2.70 > 1.47`) |

## Key results

**Exp 1 — replay divergence [m]** (5 perturbed starts, figure-8, v_peak 3.55 m/s)

| | 0.5 s | 1 s | 2 s | 3 s |
|---|---|---|---|---|
| source replay | 0.000 | 0.000 | 0.000 | 0.000 |
| target PYB (no drag, no lag) | 0.207 | 0.832 | 2.49 | 4.77 |
| target + drag | 0.254 | 0.930 | 2.32 | 3.89 |
| target + drag + 30 ms lag | 0.228 | 0.866 | 2.21 | 3.72 |

Airframe (mass/thrust) mismatch dominates; drag and lag shift it modestly.

**Exp 2 — critical loop gain g_crit(d)** (v_z hold loop = exact plant of Prop 2.1)

| d | theory | PyB τm=0 | delay+30 ms lag model | PyB τm=30 ms | L2F |
|---|---|---|---|---|---|
| 1 | 1.000 | 0.994 | 0.770 | 0.755 | 0.636 |
| 2 | 0.618 | 0.612 | 0.465 | 0.454 | 0.386 |
| 3 | 0.445 | 0.437 | 0.339 | 0.330 | 0.279 |
| 5 | 0.285 | 0.276 | 0.224 | 0.215 | 0.181 |

**Exp 4b — action buffer under delay** (3 seeds; crash rate / gate rate)

| d | src a_{t-1} | src buffer16 | tgt a_{t-1} | tgt buffer16 |
|---|---|---|---|---|
| 8 | 0.67 / 0.67 | 0.00 / 0.97 | 0.11 / 0.22 | 0.00 / 0.72 |
| 12 | 1.00 / 0.14 | 0.11 / 0.79 | 1.00 / 0.11 | 0.00 / 0.66 |

Latency-randomized (d ~ U{0..16}): buffer16 0 crashes in target at every d; a_{t-1} 11–44%; nominal d=0 student 100% crash at d=16. Cost: d=0 gates 1.00 → ~0.73 (conservatism, Prop 2.3).

**Exp 6/8 — residual models**: on PyBullet the aligned body10 k-NN cuts held-out linear residual RMSE 0.99 → 0.40 m/s²; angular residual barely improves (12 → 12 rad/s²; likely hidden rotor-state dominated). Closed-loop Smith predictor at d=6: nominal-model 42% gates, residual-model 100%, naive expert 100% (target's fast motors tolerate d=6).

## Round 2 (P1–P2 of the plan)

**Why the target tolerates delay better than the source (P1.3 diagnosis).** Naive expert, crash at d=6–10: L2F crashes;
PyBullet does not, even with L2F's exact motor lag (55/249 ms), with L2F's thrust ceiling, or with an inertia-matched
rate loop. Speeding up L2F's motors makes L2F robust. So delay fragility is a property of L2F's actuator model near
saturation, not something the target lacks; we keep the default target and test target delay effects at d ≥ 12.

**Exp 7b/7c — H1' smoothness (PPO hover, buffer a_{t-8:t-1}, latency DR d~U{0..4}, 3 seeds)**

| policy | target eff. err d=0 | d=2 | d=4 | d=6 | mean‖Δa‖ (tgt d=0) | HF power |
|---|---|---|---|---|---|---|
| λ=0 (1.5M + 0.5M steps) | 0.264 ± 0.038 | 0.398 ± 0.094 | 0.475 ± 0.152 | 0.614 ± 0.236 | 1.39 | 0.94 |
| λ=0 → λ=0.3 warm-start | 0.182 ± 0.120 | 0.296 ± 0.017 | 0.450 ± 0.013 | 0.452 ± 0.082 | 0.70 | 0.47 |
| λ=0.3 from scratch, the 1/3 seeds that learned | 0.083 | 0.079 | 0.070 | 0.087 | — | — |

Source hold error is 0.03–0.05 m for all λ at d ≤ 3: the penalty's benefit is almost entirely in transfer.

**Exp 5b — H3' hard sensor (obs delay 3, 20% dropout, no velocity; perturbed target = kf × 1.15)**

| student | target eff. RMSE | target_pert eff. RMSE | crash (pert) |
|---|---|---|---|
| teacher (exact state) | 0.210 | 0.389 | 0.00 |
| FF fixed / rand | 0.842 / 0.869 | 0.887 / 0.908 | 1.00 / 1.00 |
| GRU fixed | 0.411 ± 0.014 | 0.867 ± 0.023 | 1.00 |
| GRU rand | 0.403 ± 0.134 | 0.697 ± 0.199 | 0.67 |

**Exp 9 — H4' PPO residual fine-tuning (200k FT steps per arm, 1,600 target calibration samples, 3 seeds)**

True hold error [m] in the target (% of zero-shot → oracle gap closed):

| target | zero-shot | nominal FT | DR FT | residual FT | oracle FT |
|---|---|---|---|---|---|
| nominal | 0.137 ± 0.070 | 0.138 (−2%) | 0.170 (−29%) | **0.059 ± 0.036 (+69%)** | 0.025 |
| perturbed (kf × 1.15) | 0.064 ± 0.013 | 0.069 (−10%) | 0.095 (−65%) | **0.049 ± 0.015 (+32%)** | 0.017 |
| biased sensor b(p) | 0.159 ± 0.054 | 0.169 (−14%) | 0.189 (−43%) | **0.120 ± 0.041 (+56%)** | 0.090 |
| noisy sensor (null) | 0.142 ± 0.076 | 0.131 (+17%) | 0.161 (−28%) | **0.066 ± 0.021 (+115%)** | 0.076 |

Step-0 SysID with scripted multi-sine excitation + lag-filtered rotor thrust (`rerun_exp9_perturbed.py`) recovers exact `d̂ = 0, τ̂ = 30 ms` on 3/3 perturbed-target seeds (previously, regressing thrust-scale residuals on raw white-noise commands `a_t` biased `d̂ = 1` on 2/3 seeds and degraded `residual_ft` to 0.076 m / −26%).

## Corrections to earlier docs

1. `EMPIRICAL_THEORY_VERIFICATION.md` (Gemini toy suite) reproduces its means exactly (`results/theory_sim_rerun/`), but its "±" values are SEM; true seed std ≈ the mean (e.g. buffer vs a_{t-1} at d=4: 16.7 ± 11.6 vs 10.1 ± 10.8 — not separable). It also omits that toy GRU_RandSrc scored *worst* on the drone target (0.13 vs FF 1.73).
2. Relay limit-cycle period is 4d+2 ticks, not 4(d+1).
3. "v_peak ≥ 10 m/s" is infeasible on these airframes (L2F Crazyflie T/W ≈ 1.47); courses use 3.5 m/s.
4. L2F's reported `hovering_throttle_relative` (0.726) is inconsistent with its thrust curve; true hover ≈ 0.806 normalized rpm.
5. Guardrail #1 as written ("always include the Δa penalty") is incomplete: from scratch it can stall PPO (2/3 seeds at λ=0.3); adding it as a fine-tuning stage is reliable.
6. Prop 2.7's GP collapse requires the GP input to be independent of the sensor noise (fit on ground-truth / mocap pose).

## Round 3 — P3 Proposal Method Matrix (`Exp 10`) & Next-Wave Extensions (`Exp 11–14`)

### Exp 10 — Unified 6-Gate 3D Racing Method Matrix & Robustness Curves (`550k` source steps, 3 seeds)
Trained on 6-gate `figure8` (`2` laps = `12` gates, SE(3)-invariant local corridor + gate observations in [racing_env.py](file:///Users/ebrownhi/Desktop/Robot%20Learning/Final%20Project/real_sim/racing_env.py)) and evaluated zero-shot on **held-out 3D `ellipse3d`** (`±0.4 m` vertical waves, opposite circuit chirality) across `source_l2f`, `target_nominal` (`PyBullet CF2X`, `d=2`), and `target_perturbed` (`kf × 1.20, d=3`). Step-0 calibrations hit **`9/9` exact `(d̂, τ̂ = 30 ms)`** with **`0.8–2.8 mm`** GP null collapse under zero sensor bias.

**Held-Out 3D Course (`ellipse3d`, Exact-State, `mean ± std` over 3 seeds):**

| Method (`550k` budget) | `source_l2f` Gate Frac (Clean) | `target_nominal` Gate Frac | `target_nominal` Clean Comp | `target_nominal` Med Lap / Err | `target_perturbed` Gate Frac | `target_perturbed` Clean Comp | `target_perturbed` Med Lap / Err |
|---|---|---|---|---|---|---|---|
| `classical_expert` | `0.242 ± 0.000` (`0.00`) | `1.000 ± 0.000` | `1.00` | `4.51 s` / `0.11 m` | `0.900 ± 0.000` | `0.00` | `4.19 s` / `0.28 m` |
| `nominal_ppo` | `0.797 ± 0.230` (`0.37`) | `1.000 ± 0.000` | `0.40` | `5.02 s` / `0.13 m` | `1.000 ± 0.000` | `0.43` | `4.91 s` / `0.15 m` |
| `dr_ppo` | `0.675 ± 0.301` (`0.33`) | `0.750 ± 0.136` | `0.00` | `7.12 s` / `0.19 m` | `0.667 ± 0.245` | `0.00` | `6.95 s` / `0.21 m` |
| **`residual_ft_ppo`** | **`0.933 ± 0.094` (`0.67`)** | **`1.000 ± 0.000`** | **`1.00`** | **`5.08 s` / `0.11 m`** | **`1.000 ± 0.000`** | **`1.00`** | **`5.04 s` / `0.10 m`** |
| `target_oracle_ppo` | `0.503 ± 0.359` (`0.33`) | `1.000 ± 0.000` | `1.00` | `4.96 s` / `0.10 m` | `1.000 ± 0.000` | `1.00` | `4.70 s` / `0.09 m` |

**One-Factor Robustness Sweeps on Held-Out `ellipse3d` (`gate_frac` / `clean_completion`):**
- **Action delay `d ∈ {0, 2, 4, 6, 8}`:** `nominal_ppo` clean completion drops from `1.00` (`d=0`) to `0.33–0.50` (`d=2..8`, `gf = 0.95 @ d=8`); `dr_ppo` stays stuck at `0.72–0.75 / 0.00`; **`residual_ft_ppo` holds `1.00 ± 0.00 / 1.00` across every delay `d = 0..8`**.
- **Motor thrust scale `kf ∈ {0.85, 1.00, 1.15, 1.30}`:** `nominal_ppo` clean completion is `0.33–0.67`; `dr_ppo` is `0.64–0.75 / 0.00`; **`residual_ft_ppo` holds `1.00 ± 0.00` gate completion across `0.85..1.30` (`1.00` clean up to `1.15`, `0.94` at `1.30`)**.

---

### Exp 11 — CAPS + L2C2 Actor-Loss Smoothness vs. Reward Smoothness (`1.0M` steps from scratch, 3 seeds)
Directly resolves `Exp 7b`'s 2/3-seed exploration stall by moving temporal (`CAPS`, Mysore et al., *ICRA 2021*) and local spatial Lipschitz (`L2C2`, Kobayashi, *IROS 2022*) penalties onto the deterministic actor mean $\mu_\theta(o_t)$ ([exp11_caps_smoothness.py](file:///Users/ebrownhi/Desktop/Robot%20Learning/Final%20Project/real_sim/exp11_caps_smoothness.py)):

| Policy (`1.0M` single-stage) | Seeds Learned | Source `‖Δa‖` (`d=0`) | Target Clean `d=0` [m] | Target Clean `d=2` [m] | Target Clean `d=4` [m] | Target Clean `d=6` [m] | Target Noisy+25% ZOH `d=4` [m] |
|---|---|---|---|---|---|---|---|
| `unreg_lam0` | `3/3` | `1.091 ± 0.015` | `0.257 ± 0.077` | `0.411 ± 0.142` | `0.536 ± 0.189` | `0.456 ± 0.196` | `0.387 ± 0.112` |
| `reward_lam0.3` (reward pen.) | `0/3` (`62%` cr) | `0.433 ± 0.049` | `0.880 ± 0.302` | `0.929 ± 0.166` | `0.899 ± 0.227` | `0.936 ± 0.114` | `0.889 ± 0.158` |
| `caps_temporal` (`λ_T=0.8`) | **`3/3` (`0%` cr)** | `0.044 ± 0.020` | `0.134 ± 0.072` | `0.279 ± 0.047` | `0.251 ± 0.066` | `0.182 ± 0.053` | `0.167 ± 0.051` |
| **`caps_l2c2` (`λ_T=0.8, λ_S=0.4`)** | **`3/3` (`0%` cr)** | **`0.025 ± 0.001`** | **`0.079 ± 0.026`** | **`0.255 ± 0.042`** | **`0.220 ± 0.051`** | **`0.084 ± 0.050`** | **`0.098 ± 0.039`** |

---

### Exp 12 — DIDA 4-Step Action Chunking, Track-Free RMA Observer & Shortcut Trap (3 seeds)
([exp12_chunking_staleness.py](file:///Users/ebrownhi/Desktop/Robot%20Learning/Final%20Project/real_sim/exp12_chunking_staleness.py))
- **Part A — Command-Link Packet Drop (`ellipse3d` PyBullet, `d=2`):** `dida_chunk4` (`[a_{t|t}..a_{t+3|t}]` shift-register queue) holds **`1.000 ± 0.000`** gate completion at `0–25%` drop, **`0.983 ± 0.024`** at `50%` drop, and **`0.975 ± 0.030` (`0%` crash)** even at **`80%` wireless command loss** (beating `single_step_zoh` `0.942 ± 0.036`).
- **Part B — Harsh Sensor Channel (`obs_delay=3`, hidden `v_B`, `15 cm` oblique PnP bias, `p_obs ∈ {0..0.65}`):**
  - **Causal Shortcut Discovery:** Observers given gate coordinates `(g0 - p)` memorized clockwise `figure8` velocities and inverted sign (`v_RMSE = 4.33 m/s`) on counterclockwise `ellipse3d`; removing gate coordinates (`PacketTimestampStream`) restored genuine kinematic estimation (`v_RMSE = 1.31 m/s` vs `1.97 m/s` FF, `2–3×` higher gate fraction `0.24–0.26` vs `0.10–0.12` FF, and automatic online thrust-shift detection `z_thr = +0.58` nominal / `+0.71` perturbed vs `0.00` source).

---

### Exp 13 — Neural-Fly Live 100 Hz Composite RLS Adaptation + Algebraic-Loop Ablation (3 seeds)
Meta-trains spectral-normalized features $\phi_\theta$ *purely in `L2F`* and adapts linear weights $\hat{W}_t$ online at `100 Hz` across **3 continuous laps (`18` gates, zero target pre-calibration)** on held-out 3D `ellipse3d` ([exp13_neural_fly_online.py](file:///Users/ebrownhi/Desktop/Robot%20Learning/Final%20Project/real_sim/exp13_neural_fly_online.py)):

| Regime (`ellipse3d`, 3 laps) | `unadapted_base` RMSE (Gates) | `world_pid` RMSE (Gates) | `neural_fly_naive_u` (Algebraic Loop) | **`neural_fly_rls` (State-Only Basis)** |
|---|---|---|---|---|
| `nominal_pyb` | `0.122 m` (`1.000`) | `0.071 m` (`1.000`) | `0.30 → 1.80 m` (`0.370 ± 0.105`) | **`0.057 → 0.073 m` (`1.000 ± 0.000`)** |
| `turbulent_wind_pyb` (`~0.5g`) | `0.340 m` (`0.611`) | `0.195 m` (`0.944`) | `0.51 → 1.48 m` (`0.167 ± 0.045`) | **`0.108 → 0.112 m` (`1.000 ± 0.000`, `−67%`)** |
| `midrace_fault_pyb` (`kf × 0.75` @ Lap 2) | `0.276 m` L2 (`0.778`) | `0.196 m` L2 (`1.000`) | `0.58 → 0.92 m` (`0.407 ± 0.094`) | **`0.107 m` L2 (`1.000 ± 0.000`, `−45%` vs PID)** |

---

### Exp 14 — CMU ASAP Delta-Action Alignment + MOPO Uncertainty vs. Swift `k-NN` (3 seeds)
Compares action-space alignment (`ASAP`, He et al., *RSS 2025*) against acceleration-space `k-NN` (`Swift`) and a 5-MLP residual ensemble with epistemic pessimism $-\beta \sigma_{\text{ep}}$ (`MOPO`, Yu et al., *NeurIPS 2020*) at identical `1,600` target samples and `100k` source PPO FT steps ([exp14_asap_mopo_residuals.py](file:///Users/ebrownhi/Desktop/Robot%20Learning/Final%20Project/real_sim/exp14_asap_mopo_residuals.py)):

| Arm (`100k` FT) | `nom / figure8` Clean (Err / Lap) | `nom / ellipse3d` Clean (Err / Lap) | `pert / figure8` Clean (Err / Lap) | `pert / ellipse3d` Clean (Err / Lap) |
|---|---|---|---|---|
| `zeroshot_ppo` | `0.73` (`0.087 m` / `5.41 s`) | `0.63` (`0.123 m` / `5.48 s`) | `0.33` (`0.132 m` / `5.23 s`) | `0.33` (`0.148 m` / `5.26 s`) |
| `swift_knn_ft` | `1.00` (`0.059 m` / `5.13 s`) | `0.67` (`0.102 m` / `5.17 s`) | `1.00` (`0.064 m` / `4.99 s`) | `1.00` (`0.094 m` / `5.12 s`) |
| **`swift_mopo_ft`** | **`1.00` (`0.051 m` / `5.03 s`)** | **`1.00` (`0.100 m` / `5.12 s`)** | **`1.00` (`0.056 m` / `4.92 s`)** | **`1.00` (`0.092 m` / `4.97 s`)** |
| `asap_delta_a_ft` | **`1.00`** (`0.112 m` / `5.30 s`) | **`1.00`** (`0.125 m` / `5.36 s`) | `0.67` (`0.140 m` / `5.15 s`) | `0.00` (`0.141 m` / `5.30 s` — *motor clip!*) |

- **Forward-model & $\Delta f \equiv 0$ falsification:** On `target_nominal`, `asap_delta_a` achieves the lowest `0.5 s` open-loop drift (`0.107 ± 0.031 m` vs `0.124 m` `k-NN` and `0.190 m` nominal), while on `target_perturbed` (`T/W = 2.70 > L2F`'s `1.47` ceiling) action deltas clip at `a = +1.0`. Ensemble epistemic $\sigma_{\text{ep}}$ separates OOD states by **`2.31–3.30×`**. On the $\Delta f \equiv 0, d=3$ control, Step-0 alignment (`d̂ = 3.00` exact) collapses phantom `k-NN` residuals **`11×` (`0.190 → 0.0174 m/s²`)**.

## Setup & reproducibility notes
- Reproducible environment & build scripts are captured in [code/pixi.toml](file:///Users/ebrownhi/Desktop/Robot%20Learning/Final%20Project/code/pixi.toml) and [code/setup_sims.sh](file:///Users/ebrownhi/Desktop/Robot%20Learning/Final%20Project/code/setup_sims.sh); run `pixi run --manifest-path code/pixi.toml test` (`17/17` tests passing in ~5 s).
