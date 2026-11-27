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
| H2' dynamics DR helps | Exp 4b, 9, 10, 19, 24 | ⚠️ **no transfer gain, no transfer harm:** the Exp 10/19 penalty is under-training at 550k (Exp 24: full DR matches no-DR at ≥1.1M, `1.00 / 1.00` in both targets; ceiling effect), at a ~33% lap-time cost. DR does improve held-out-course robustness under L2F's tight thrust limits (`0.96` vs `0.34` gates at 2.2M). Latency-only DR is best at 550k |
| H1' smoothness penalty (PPO) | Exp 7, 7b, 7c | ⚠️ partial: penalty alone (Exp 7) ❌; with buffer + latency DR, λ=0 learns bang-bang control whose target error is 4.5× its source error; λ=0.3 warm-start halves chattering and lowers target error at every d (0.26→0.18 m at d=0, 0.50→0.27 m at d=1); λ=0.3 from scratch fails to learn on 2/3 seeds |
| Thm 2.4 recurrent-observer OOD | Exp 5, 5b | ⚠️ partial: with a 2.3× larger gap, fixed-dynamics estimators degrade (GRU 0.21, FF 0.23 m/s) and randomized ones hold (0.14, 0.15 m/s); 'GRU_fixed worse than FF' not observed |
| H3' recurrent students + randomized distillation | Exp 5, 5b, 10, 21, 23 | ✅ supported (Exp 21) and now reaching **full-course completion** (Exp 23). With a 3-flight PnP-bias GP, the track-free GRU student completes held-out `ellipse3d` in the PyBullet target at every sensor level (nominal `1.00 / 0.93 / 0.53`, perturbed `0.77 / 0.60 / 0.60` at clean / mid / hard, vs `0` before). Multi-track estimator training lifts L2F-source completion `0.33 → 0.60` (clean). Open: collision-free completion `≤ 0.50`; the source at hard stays at `0` (teacher ceiling) |
| Privileged critic (proposal ablation) | Exp 20, 22 | ✅ with a track-free-capable actor (Exp 22): asym critic > masked critic in `64/66` seed × cell pairs. Asym learns the training course (`0.98–1.00` gates) at every sensor level; sym never exceeds `0.45`. Held-out targets `0.30–0.51` vs `0.07–0.08`. It fixes learnability, not transfer (Exp 21 students transfer better) |
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
| Riff R0.1 + R0.2: Fisher safe SysID & 10-tick shooting | Exp 15 | ⚠️ (audit-corrected) all excitation strategies — Fisher, hand multi-sine, *and* a Gaussian random walk — pick the right `(d̂, τ_up, τ_down)` cell of a 4×4 grid containing the truth on `17–18/18` trials, so Fisher shows no measurable advantage, and every strategy leaves the `0.5 m` safety sphere in the target at `≥1 s`; parametric-only 10-tick shooting cuts 3 s drift only `18%` (`7.69 → 6.33 m`) vs **`−65%` (`2.69 m`) for 6D `k-NN`** because $J$, `km/kf`, and downwash lie outside `L2F`'s exposed parameters |
| Riff R1.2 + R1.3: Architectural EMA & Residual-over-Classical RL | Exp 16 | ✅ **5-line EMA filter (`α=0.50–0.70`, `λ_rew=0`)** cuts PyBullet hold error **`53–73%`** across `d=0..8` (`0.092 m` @ `d=8` vs `0.311 m` unfiltered); **Residual-over-Classical** gives **`15–24×` more gates at `50k` steps** (`1.000` vs `0.066` E2E on `target_nominal`), while unconstrained E2E overtakes near `L2F` saturation at `450k+` (reproducing Song et al., *Sci. Robot.* 2023) |
| Riff R2.1 + R2.2: DORAEMON curriculum DR + CVaR-PPO | Exp 17 | ✅ **`doraemon_ppo`** eliminates `source_d0` crashes (`47% → 0%`, `gf: 0.736 → 0.972`) and beats `fixed_wide_dr` across all PyBullet targets (`0.917–1.000` vs `0.671–0.762` gates, up to **`3.1 s` faster `p90` tail laps**); `CVaR_{0.35}` tail-weighting over-brakes (`0.66–0.74` gates), confirming Prop 2.3's risk-speed trade-off |
| Riff R4.3 + R5.2: 2-Round Bootstrapped Swift & 10-Track 3D Suite | Exp 18 | ✅ At equal `16 s` (`1,600`-tick) target budget (`kf × 1.25, d=3`), **2-round active bootstrap** cuts high-speed residual RMSE **`33%` (`0.360 → 0.241 m/s²`)**, halves **10-track macro error (`0.135 → 0.067 m`)**, and hits **`1.00` clean completion on `9/10` unseen 3D circuits** (`0.900 ± 0.000` macro clean vs `0.472` zero-shot & `0.50` direct `16 s` target PPO) |

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
Trained on 6-gate `figure8` (`2` laps = `12` gates, SE(3)-invariant local corridor + gate observations in [racing_env.py](../real_sim/racing_env.py)) and evaluated zero-shot on **held-out 3D `ellipse3d`** (`±0.4 m` vertical waves, opposite circuit chirality) across `source_l2f`, `target_nominal` (`PyBullet CF2X`, `d=2`), and `target_perturbed` (`kf × 1.20, d=3`). Step-0 calibrations hit **`9/9` exact `(d̂, τ̂ = 30 ms)`** with **`0.8–2.8 mm`** GP null collapse under zero sensor bias.

> **Audit note (2026-11-26):** the table below is the *exact-state* half only. In the *sensor-limited* half every method crashes on ~100% of episodes, even on the training course in the source. Those rows, the diagnosis and the fixed retest (Exp 21) are in **Round 5** below.

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
Directly resolves `Exp 7b`'s 2/3-seed exploration stall by moving temporal (`CAPS`, Mysore et al., *ICRA 2021*) and local spatial Lipschitz (`L2C2`, Kobayashi, *IROS 2022*) penalties onto the deterministic actor mean $\mu_\theta(o_t)$ ([exp11_caps_smoothness.py](../real_sim/exp11_caps_smoothness.py)):

| Policy (`1.0M` single-stage) | Seeds Learned | Source `‖Δa‖` (`d=0`) | Target Clean `d=0` [m] | Target Clean `d=2` [m] | Target Clean `d=4` [m] | Target Clean `d=6` [m] | Target Noisy+25% ZOH `d=4` [m] |
|---|---|---|---|---|---|---|---|
| `unreg_lam0` | `3/3` | `1.091 ± 0.015` | `0.257 ± 0.077` | `0.411 ± 0.142` | `0.536 ± 0.189` | `0.456 ± 0.196` | `0.387 ± 0.112` |
| `reward_lam0.3` (reward pen.) | `0/3` (`62%` cr) | `0.433 ± 0.049` | `0.880 ± 0.302` | `0.929 ± 0.166` | `0.899 ± 0.227` | `0.936 ± 0.114` | `0.889 ± 0.158` |
| `caps_temporal` (`λ_T=0.8`) | **`3/3` (`0%` cr)** | `0.044 ± 0.020` | `0.134 ± 0.072` | `0.279 ± 0.047` | `0.251 ± 0.066` | `0.182 ± 0.053` | `0.167 ± 0.051` |
| **`caps_l2c2` (`λ_T=0.8, λ_S=0.4`)** | **`3/3` (`0%` cr)** | **`0.025 ± 0.001`** | **`0.079 ± 0.026`** | **`0.255 ± 0.042`** | **`0.220 ± 0.051`** | **`0.084 ± 0.050`** | **`0.098 ± 0.039`** |

---

### Exp 12 — DIDA 4-Step Action Chunking, Track-Free RMA Observer & Shortcut Trap (3 seeds)
([exp12_chunking_staleness.py](../real_sim/exp12_chunking_staleness.py))
- **Part A — Command-Link Packet Drop (`ellipse3d` PyBullet, `d=2`):** `dida_chunk4` (`[a_{t|t}..a_{t+3|t}]` shift-register queue) holds **`1.000 ± 0.000`** gate completion at `0–25%` drop, **`0.983 ± 0.024`** at `50%` drop, and **`0.975 ± 0.030` (`0%` crash)** even at **`80%` wireless command loss** (beating `single_step_zoh` `0.942 ± 0.036`).
- **Part B — Harsh Sensor Channel (`obs_delay=3`, hidden `v_B`, `15 cm` oblique PnP bias, `p_obs ∈ {0..0.65}`):**
  - **Causal Shortcut Discovery:** Observers given gate coordinates `(g0 - p)` memorized clockwise `figure8` velocities and inverted sign (`v_RMSE = 4.33 m/s`) on counterclockwise `ellipse3d`; removing gate coordinates (`PacketTimestampStream`) restored genuine kinematic estimation (`v_RMSE = 1.31 m/s` vs `1.97 m/s` FF, `2–3×` higher gate fraction `0.24–0.26` vs `0.10–0.12` FF, and automatic online thrust-shift detection `z_thr = +0.58` nominal / `+0.71` perturbed vs `0.00` source).

---

### Exp 13 — Neural-Fly Live 100 Hz Composite RLS Adaptation + Algebraic-Loop Ablation (3 seeds)
Meta-trains spectral-normalized features $\phi_\theta$ *purely in `L2F`* and adapts linear weights $\hat{W}_t$ online at `100 Hz` across **3 continuous laps (`18` gates, zero target pre-calibration)** on held-out 3D `ellipse3d` ([exp13_neural_fly_online.py](../real_sim/exp13_neural_fly_online.py)):

| Regime (`ellipse3d`, 3 laps) | `unadapted_base` RMSE (Gates) | `world_pid` RMSE (Gates) | `neural_fly_naive_u` (Algebraic Loop) | **`neural_fly_rls` (State-Only Basis)** |
|---|---|---|---|---|
| `nominal_pyb` | `0.122 m` (`1.000`) | `0.071 m` (`1.000`) | `0.30 → 1.80 m` (`0.370 ± 0.105`) | **`0.057 → 0.073 m` (`1.000 ± 0.000`)** |
| `turbulent_wind_pyb` (`~0.5g`) | `0.340 m` (`0.611`) | `0.195 m` (`0.944`) | `0.51 → 1.48 m` (`0.167 ± 0.045`) | **`0.108 → 0.112 m` (`1.000 ± 0.000`, `−67%`)** |
| `midrace_fault_pyb` (`kf × 0.75` @ Lap 2) | `0.276 m` L2 (`0.778`) | `0.196 m` L2 (`1.000`) | `0.58 → 0.92 m` (`0.407 ± 0.094`) | **`0.107 m` L2 (`1.000 ± 0.000`, `−45%` vs PID)** |

---

### Exp 14 — CMU ASAP Delta-Action Alignment + MOPO Uncertainty vs. Swift `k-NN` (3 seeds)
Compares action-space alignment (`ASAP`, He et al., *RSS 2025*) against acceleration-space `k-NN` (`Swift`) and a 5-MLP residual ensemble with epistemic pessimism $-\beta \sigma_{\text{ep}}$ (`MOPO`, Yu et al., *NeurIPS 2020*) at identical `1,600` target samples and `100k` source PPO FT steps ([exp14_asap_mopo_residuals.py](../real_sim/exp14_asap_mopo_residuals.py)):

| Arm (`100k` FT) | `nom / figure8` Clean (Err / Lap) | `nom / ellipse3d` Clean (Err / Lap) | `pert / figure8` Clean (Err / Lap) | `pert / ellipse3d` Clean (Err / Lap) |
|---|---|---|---|---|
| `zeroshot_ppo` | `0.73` (`0.087 m` / `5.41 s`) | `0.63` (`0.123 m` / `5.48 s`) | `0.33` (`0.132 m` / `5.23 s`) | `0.33` (`0.148 m` / `5.26 s`) |
| `swift_knn_ft` | `1.00` (`0.059 m` / `5.13 s`) | `0.67` (`0.102 m` / `5.17 s`) | `1.00` (`0.064 m` / `4.99 s`) | `1.00` (`0.094 m` / `5.12 s`) |
| **`swift_mopo_ft`** | **`1.00` (`0.051 m` / `5.03 s`)** | **`1.00` (`0.100 m` / `5.12 s`)** | **`1.00` (`0.056 m` / `4.92 s`)** | **`1.00` (`0.092 m` / `4.97 s`)** |
| `asap_delta_a_ft` | **`1.00`** (`0.112 m` / `5.30 s`) | **`1.00`** (`0.125 m` / `5.36 s`) | `0.67` (`0.140 m` / `5.15 s`) | `0.00` (`0.141 m` / `5.30 s` — *motor clip!*) |

- **Forward-model & $\Delta f \equiv 0$ falsification:** On `target_nominal`, `asap_delta_a` achieves the lowest `0.5 s` open-loop drift (`0.107 ± 0.031 m` vs `0.124 m` `k-NN` and `0.190 m` nominal), while on `target_perturbed` (`T/W = 2.70 > L2F`'s `1.47` ceiling) action deltas clip at `a = +1.0`. Ensemble epistemic $\sigma_{\text{ep}}$ separates OOD states by **`2.31–3.30×`**. On the $\Delta f \equiv 0, d=3$ control, Step-0 alignment (`d̂ = 3.00` exact) collapses phantom `k-NN` residuals **`11×` (`0.190 → 0.0174 m/s²`)**.

---

### Exp 15 — Fisher-Optimal Safe Excitation & 10-Tick Multi-Step Parametric Shooting (3 seeds)
([exp15_fisher_shooting_sysid.py](../real_sim/exp15_fisher_shooting_sysid.py))
- **Part A — Asymmetric Rotor-Lag + Delay SysID (`true d = 2, τ_up = 20 ms, τ_down = 80 ms, kf × 1.20`), corrected by the 2026-11-26 audit:** all three excitation strategies identify `d̂ = 2` on `18/18` trials at every window and pick the correct `(τ_up, τ_down)` cell on `17/18` (`0.5 s`) and `17–18/18` (`≥1.0 s`) trials. Fisher-optimal and an unplanned Gaussian random walk tie at `1.0 s` (`18/18`); hand multi-sine misses one trial. **The data does not show an advantage for Fisher design.** Two further caveats: (i) `identify_asymmetric_sysid` searches a discrete `4×4` grid (`τ_up ∈ {15, 20, 30, 45} ms`, `τ_down ∈ {20, 40, 80, 120} ms`) that contains the truth, so `0.000 ms` "RMSE" means "right cell", not continuous precision; (ii) the `0.50 m` safety sphere is enforced only while designing the excitation in `L2F` — in the PyBullet target every strategy exceeds it at `≥1.0 s` (Fisher `0.89 m`, Gaussian `0.88 m`, hand `0.81 m` at `1.0 s`; `1.3–1.4 m` at `3.5 s`). Only the `0.5 s` window (`0.36–0.37 m`) stays inside it.
- **Part B — 10-Tick Parametric Shooting vs. Non-Parametric `k-NN` (3 s Replay on Held-Out `ellipse3d`):** Optimizing scalar `L2F` overrides `(thrust_scale, τ_rise, τ_fall, drag_xy, drag_z)` over 10-tick (`100 ms`) windows trims 3 s replay drift by only **`18%` (`7.69 → 6.33 m`)**, whereas **6D SE(3) `step0_knn` cuts drift by `65%` (`7.69 → 2.69 ± 0.59 m`, and `0.94 → 0.326 m` @ `1 s`)**, proving that non-parametric state-action residuals are indispensable when structural mismatches ($J$, `km/kf = 5.4×`, downwash) lie outside the source simulator's exposed parameter vector.

---

### Exp 16 — Architectural EMA Bandwidth Filter & Residual-RL over Classical Control (3 seeds)
([exp16_bandwidth_and_residual_policy.py](../real_sim/exp16_bandwidth_and_residual_policy.py))
- **Part A — Structural Low-Pass Actuator Wrapper (`a_t = α a_{t-1} + (1−α) u_θ(o_t)`, `λ_rew = 0`, `600k` steps):**
  - `ema_0.00` (unfiltered): Target hold error `0.307 m @ d=0`, `0.393 m @ d=2`, `0.436 m @ d=4`, `0.424 m @ d=6`, `0.311 m @ d=8`.
  - **`ema_0.50` (`16 Hz` cutoff):** Cuts target hold error **`53–73%`** across all delays (`0.137 m @ d=0`, `0.186 m @ d=2`, `0.128 m @ d=4`, `0.116 m @ d=6`, `0.144 m @ d=8`).
  - **`ema_0.70` (`7.6 Hz` cutoff):** Achieves **`0.104 m @ d=6` and `0.092 m @ d=8` (`−70%`)**, directly confirming Proposition 2.1's Bode bandwidth–delay trade-off purely architecturally.
- **Part B — Hybrid Residual-RL over `GeometricGateController` (`a = clip(a_cls + 0.45 π_θ, −1, 1)`) on Held-Out `ellipse3d`:**
  - **Low-sample regime (`50k` steps):** `res_over_cls` reaches **`1.000 ± 0.000` gates (`0.67` clean) on `target_nominal`** (vs `0.066` `e2e_ppo`, **`15×` gain**), **`0.750` vs `0.031` (`24×`)** on `target_perturbed`, and lifts `source_l2f` from `0.250` (`classical_0k`) to **`0.448` @ `250k`** (vs `0.111` `e2e_ppo`).
  - **Asymptotic crossover (`450k+` steps):** Near `L2F`'s `T/W = 1.47` ceiling on `ellipse3d`, unconstrained `e2e_ppo` (`0.583` @ `450k` → `0.80+` @ `550k`) overtakes bounded `res_over_cls` (`0.153`), reproducing Song et al. (*Science Robotics* 2023): reference-tracking priors give `10–20×` early sample efficiency, while unconstrained RL reaches higher asymptotic agility when the classical prior saturates actuators.

---

### Exp 17 — DORAEMON Entropy-Maximized Curriculum DR & CVaR-PPO (`550k` steps, 3 seeds)
([exp17_doraemon_cvar_dr.py](../real_sim/exp17_doraemon_cvar_dr.py))

| Arm (`ellipse3d` held-out) | `source_d0` Gate Frac (Crash) | `target_nominal` Gate Frac (`p90` Lap) | `target_perturbed` Gate Frac (`p90` Lap) | `target_extreme_ood` (`d=6, kf=1.30`) Gate Frac (`CVaR_25`) |
|---|---|---|---|---|
| `nominal_ppo` | `0.736 ± 0.187` (`0.47` cr) | `1.000 ± 0.000` (`5.36 s`) | `1.000 ± 0.000` (`5.23 s`) | `1.000 ± 0.000` (`1.00`) |
| `fixed_wide_dr` | `0.875 ± 0.037` (`0.00` cr) | `0.762 ± 0.070` (`7.56 s`) | `0.671 ± 0.193` (`9.13 s`) | `0.685 ± 0.182` (`0.67`) |
| **`doraemon_ppo`** | **`0.972 ± 0.039` (`0.00` cr)** | **`0.917 ± 0.068` (`6.16 s`)** | **`0.935 ± 0.047` (`6.00 s`)** | **`1.000 ± 0.000` (`1.00`)** |
| `doraemon_cvar_ppo` (`α=0.35`) | `0.743 ± 0.044` (`0.06` cr) | `0.688 ± 0.183` (`6.71 s`) | `0.667 ± 0.324` (`6.37 s`) | `0.660 ± 0.253` (`0.54`) |

---

### Exp 18 — Active 2-Round Bootstrapped Swift & 10-Track Procedural 3D Benchmark (`16 s` Target Budget, 3 seeds)
Compares model-free direct target RL against 1-round and 2-round active bootstrapped `Swift + MOPO` at **strictly equal `1,600`-tick (`16.0 s`) target budget** on a high-gain delayed target (`kf × 1.25, d = 3`), then evaluates zero-shot across **10 procedural 3D circuits** ([exp18_bootstrapped_swift_and_10tracks.py](../real_sim/exp18_bootstrapped_swift_and_10tracks.py)):

- **On-policy high-speed residual RMSE:** `0.360 ± 0.129 m/s²` (`1-batch`) → **`0.241 ± 0.089 m/s²` (`2-round bootstrap`, `−33%`)**.
- **Equal `16 s` target budget on `ellipse3d` (`kf × 1.25, d = 3`):** `zeroshot_base` `0.17` clean (`0.162 m` err), `direct_res_policy_1600` `0.06` clean (`0.163 m`), `direct_target_ppo_1600` `0.50` clean (`0.151 m`) vs. **`swift_mopo_1round` `1.00` clean (`0.101 m`)** and **`swift_mopo_2round` `1.00` clean (`0.098 m`)**.
- **10-Track Procedural 3D Benchmark Macro-Average (`figure8`, `ellipse3d`, `split_s_3d`, `corkscrew_3d`, `tilted_hairpin`, `saddle_3d`, `diving_chicane`, `banked_oval_3d`, `ascending_esses`, `trefoil_3d`):**
  - `zeroshot_base`: `cln = 0.472 ± 0.267`, `trk = 0.135 m`
  - `direct_target_ppo_1600`: `cln = 0.472 ± 0.281`, `trk = 0.129 m`
  - `swift_mopo_1round`: **`cln = 0.900 ± 0.000` (`1.00` clean on `9/10` 3D circuits)**, `trk = 0.076 m`
  - **`swift_mopo_2round`**: **`cln = 0.900 ± 0.000` (`1.00` clean on `9/10` 3D circuits)**, **`trk = 0.067 m` (`−50%` tracking error across 10 unseen 3D tracks!)**

## Round 5 — Audit follow-ups (2026-11-26): proposal ablations & the sensor-limited half of Exp 10

### Exp 19 — Leave-one-out randomization-factor ablation (proposal: "remove individual randomization factors")
([exp19_dr_factor_ablation.py](../real_sim/exp19_dr_factor_ablation.py), 3 seeds × 7 variants, identical recipe to Exp 10 `dr_ppo`: `450k` at `λ=0.02` + `100k` at `λ=0.20`.)
The `none` and `full` arms reproduce Exp 10's `nominal_ppo` and `dr_ppo` exactly (same seeds, deterministic), so the rows are directly comparable to Exp 10.

| Variant | `source_l2f` figure8 gates | held-out `ellipse3d` → `target_nominal` gates | clean completion | track err | median lap |
|---|---|---|---|---|---|
| `none` | `1.000` | `1.000` | `0.40` | `0.133 m` | `5.02 s` |
| `full` (mass + thrust + motor lag + latency) | `0.778` | `0.750` | `0.00` | `0.188 m` | `7.12 s` |
| `no_mass` | `0.972` | `0.917` | `0.33` | `0.152 m` | `5.61 s` |
| `no_thrust` | `0.917` | `0.806` | `0.00` | `0.169 m` | `6.97 s` |
| `no_motor_lag` | `0.861` | `0.775` | `0.00` | `0.175 m` | `6.42 s` |
| `no_latency` (dynamics only) | `0.972` | `0.894` | `0.00` | `0.128 m` | `6.20 s` |
| **`latency_only`** | **`1.000`** | **`1.000`** | **`0.67`** | **`0.112 m`** | `5.63 s` |

- **Latency randomization alone is the best variant** (`1.00` gates in source and both targets on the held-out course, `0` crashes, lowest tracking error); its clean-completion edge over `none` (`0.67` vs `0.40`) is within 3-seed noise (std ≈ `0.45`, clean completion is nearly binary per seed).
- **The Exp 10 DR penalty comes from the dynamics factors, cumulatively:** every variant that keeps them is below `none`; removing any single one recovers only part of the loss (mass the most).
- **Interpretation caveat:** dynamics DR also lowers *source* skill at this fixed `550k` budget (`0.78` gates on the training course). Its own source→target drop is small, so "DR hurts transfer" here is mostly "DR under-trains at equal budget", not a transfer failure. A budget sweep would be needed to separate the two.

### Exp 20 — Privileged (asymmetric) vs symmetric critic for sensor-limited PPO (proposal: "remove privileged critic inputs")
([exp20_privileged_critic.py](../real_sim/exp20_privileged_critic.py), [asym_policy.py](../real_sim/asym_policy.py), 3 seeds × 2 arms, Exp 10 recipe; the actor is provably blind to the privileged block — `tests/test_asym_policy.py`.)
Both arms **fail to learn**: `0.14 ± 0.02` (asym) vs `0.09 ± 0.00` (sym) gates and `100%` crashes even on the training course in the source. The privileged critic gives a small, consistent edge, but around a policy that cannot fly. The ablation is therefore *uninformative* in this sensor regime (see the diagnosis below) rather than a negative result for privileged critics. Its actor received the same track-relative, hard-sensor observation that Exp 21 found shortcut-prone. **Resolved in Exp 22 (Round 6):** with track-free kinematic inputs in the actor, the privileged critic decisively helps.

### Exp 10 sensor-limited half — previously unreported, and why it failed
The Round 3 write-up showed only the exact-state rows of Exp 10. The sensor-limited rows (from `exp10_method_matrix.json`) are:

| Method | course | `source_l2f` gates / crash | `target_nominal` gates / crash | `target_perturbed` gates / crash |
|---|---|---|---|---|
| `nominal_ppo_zerovel` | figure8 | `0.08` / `1.00` | `0.08` / `1.00` | `0.08` / `1.00` |
| `nominal_ppo_zerovel` | ellipse3d | `0.08` / `1.00` | `0.08` / `1.00` | `0.04` / `1.00` |
| `ff_student` | figure8 | `0.09` / `1.00` | `0.09` / `1.00` | `0.09` / `1.00` |
| `ff_student` | ellipse3d | `0.08` / `1.00` | `0.11` / `1.00` | `0.06` / `1.00` |
| `gru_student_fixed` | figure8 | `0.13` / `1.00` | `0.12` / `1.00` | `0.11` / `1.00` |
| `gru_student_fixed` | ellipse3d | `0.04` / `1.00` | `0.03` / `1.00` | `0.01` / `1.00` |
| `gru_student_rand` | figure8 | `0.17` / `1.00` | `0.13` / `0.97` | `0.09` / `1.00` |
| `gru_student_rand` | ellipse3d | `0.09` / `1.00` | `0.05` / `1.00` | `0.06` / `1.00` |
| `residual_gp_gru` | figure8 | `0.10` / `1.00` | `0.09` / `1.00` | `0.12` / `1.00` |
| `residual_gp_gru` | ellipse3d | `0.06` / `1.00` | `0.06` / `1.00` | `0.08` / `1.00` |

Every sensor-limited method crashes on essentially every episode, **including on the training course in the source**, so
these rows measure a training failure, not transfer, and Exp 10 cannot be used to judge H3'. Diagnosis (seed-0 teacher, scratch scripts):

1. **Not a pipeline bug.** The sensor-limited observation equals the exact one minus `v_B`, plus the configured 2 cm / 0.04 rad/s noise.
   A feed-forward student distilled by the same DAgger loop onto *exact* observations flies `1.00` gates / `0` crashes everywhere.
2. **The information is there.** The teacher plus a two-line alpha-beta filter on observed position flies `1.00` / `0` under the clean sensor.
   The teacher is also insensitive to velocity error (`0.48 m/s` white or `0.64 m/s` correlated error costs ≤ `2%` of gates).
3. **The Exp 10 training sensor (obs delay 2, 20% dropout) is at the edge of feasibility** for this course at 3.5 m/s:
   the teacher given the *true* velocity reaches `0.27` gates / `100%` crashes, and a hand-built delay-compensating observer about `0.50` / `50%`.
4. **Learned students take a course-position shortcut.** With 3× the DAgger data and a windowed-BPTT trainer, the action-cloning GRU improves
   from `0.13` / `100%` to `0.53` / `40%` (the hand-built observer's ceiling), but every learned estimator that sees the track-relative racing
   observation is accurate on teacher states (≈`0.1 m/s`) and collapses closed-loop: it reports `v ≈ 0` while the true speed is `1.5 m/s`.
   It learned "position on the figure-8 → typical teacher velocity", which is wrong once the student leaves the teacher's trajectory.
   Removing the action history from its input (a copycat test) did not help; removing the track-relative inputs did.
5. **Fix:** a *track-free* velocity estimator (body-frame position differences, fresh-packet flag, attitude, rates → `v_B`) feeding the frozen teacher
   ([kin_observer.py](../real_sim/kin_observer.py)). Single-seed check: `1.00` gates / `0` crashes on the training course in the source (clean and mid sensor)
   and on held-out `ellipse3d` in the PyBullet target at **all three** sensor levels. The 3-seed retest is Exp 21.

### Exp 21 — H3' retest with track-free velocity observers (3 seeds × 5 students × 3 sensor levels)
([exp21_h3_observer_retest.py](../real_sim/exp21_h3_observer_retest.py), [kin_observer.py](../real_sim/kin_observer.py).) Each student deploys the frozen
Exp 10 nominal teacher on an estimated velocity; `teacher_exact` is the non-deployable reference. Sensor levels: clean (no obs delay or dropout),
mid (delay 1, 10% dropout), hard (delay 2, 20% dropout, the Exp 10 setting); all have 2 cm / 0.04 rad/s noise, and the targets add the Exp 10
oblique PnP gate bias (which `teacher_exact` does not see). Training course `figure8`: every observer flies `1.00` gates / `0` crashes in the source
at the clean level, and `≥0.94` at mid.

**Held-out `ellipse3d`, gate fraction / crash rate (mean ± std over 3 seeds):**

| Sensor | Student | `source_l2f` | `target_nominal` | `target_perturbed` |
|---|---|---|---|---|
| clean | `teacher_exact` | `1.00 ± 0.00` / `0.00` | `1.00 ± 0.00` / `0.00` | `1.00 ± 0.00` / `0.00` |
| clean | `alpha_beta` | `0.47 ± 0.26` / `0.67` | `0.69 ± 0.16` / `0.27` | `0.57 ± 0.08` / `0.37` |
| clean | `kin_ff` | `0.20 ± 0.11` / `1.00` | `0.47 ± 0.11` / `0.57` | `0.34 ± 0.11` / `0.90` |
| clean | `kin_gru_fixed` | `0.51 ± 0.34` / `0.63` | `0.69 ± 0.05` / `0.00` | `0.61 ± 0.10` / `0.23` |
| clean | `kin_gru_rand` | `0.63 ± 0.26` / `0.53` | `0.79 ± 0.01` / `0.00` | `0.68 ± 0.05` / `0.00` |
| mid | `teacher_exact` | `1.00 ± 0.00` / `0.00` | `1.00 ± 0.00` / `0.00` | `1.00 ± 0.00` / `0.00` |
| mid | `alpha_beta` | `0.38 ± 0.23` / `0.73` | `0.59 ± 0.12` / `0.17` | `0.55 ± 0.17` / `0.37` |
| mid | `kin_ff` | `0.13 ± 0.04` / `1.00` | `0.36 ± 0.06` / `0.73` | `0.19 ± 0.03` / `0.77` |
| mid | `kin_gru_fixed` | `0.47 ± 0.30` / `0.63` | `0.71 ± 0.04` / `0.03` | `0.43 ± 0.10` / `0.53` |
| mid | `kin_gru_rand` | `0.51 ± 0.30` / `0.67` | `0.78 ± 0.04` / `0.00` | `0.59 ± 0.04` / `0.27` |
| hard | `teacher_exact` | `1.00 ± 0.00` / `0.00` | `1.00 ± 0.00` / `0.00` | `1.00 ± 0.00` / `0.00` |
| hard | `alpha_beta` | `0.16 ± 0.05` / `1.00` | `0.44 ± 0.17` / `0.43` | `0.38 ± 0.13` / `0.57` |
| hard | `kin_ff` | `0.14 ± 0.03` / `1.00` | `0.28 ± 0.08` / `0.80` | `0.17 ± 0.09` / `1.00` |
| hard | `kin_gru_fixed` | `0.18 ± 0.09` / `1.00` | `0.62 ± 0.12` / `0.27` | `0.26 ± 0.11` / `0.83` |
| hard | `kin_gru_rand` | `0.18 ± 0.02` / `1.00` | `0.76 ± 0.10` / `0.00` | `0.56 ± 0.02` / `0.27` |

- **Exp 10's failure is fixed.** At the Exp 10 (hard) setting, `kin_gru_rand` reaches `0.76` gates / `0` crashes on held-out `ellipse3d` in the
  nominal target and `0.56` / `0.27` in the perturbed target, vs `0.05` / `1.00` and `0.06` / `1.00` for Exp 10's `gru_student_rand`.
- **Recurrence matters (proposal ablation):** both GRU estimators beat the 8-frame feed-forward estimator in every target cell at every level:
  the randomized GRU by `0.32–0.48` gates, the fixed GRU by `0.09–0.35` (e.g. hard / nominal target: `0.76` vs `0.28`). FF crashes in `57–100%` of target episodes.
- **Randomized-dynamics observers generalize better (Thm 2.4):** `kin_gru_rand` ≥ `kin_gru_fixed` in all six target cells, with the gap growing with
  sensor difficulty and dynamics shift (hard / perturbed target: `0.56` vs `0.26` gates, `0.27` vs `0.83` crashes).
- **Learned beats classical, only with randomization:** `kin_gru_rand` beats the alpha-beta filter in all six target cells (hard / nominal: `0.76` vs `0.44`).
  `kin_gru_fixed` loses to it in two (mid and hard, perturbed target), consistent with Thm 2.4.
- **Still open:** (i) no observer matches the exact teacher (`1.00`). Part of the target gap is the PnP gate bias, which no estimator here corrects.
  (ii) In the L2F *source*, held-out performance is weak for every observer (`≤0.63`, `0.18` at hard), although the teacher scores `1.00` there.
  L2F's lower thrust margin (T/W 1.47 vs 2.25) makes the teacher far less tolerant of estimate error than in PyBullet, so target/source ratios exceed 1.
  (iii) Gate *completion* (all 12 gates) is still rare for students; these are gate-fraction results.

## Round 6 — Remaining drone gaps (2026-11-26, three parallel sub-studies)

### Exp 22 — Privileged vs symmetric critic with a track-free-capable actor (fair rerun of Exp 20)
([exp22_privileged_critic_trackfree.py](../real_sim/exp22_privileged_critic_trackfree.py), [trackfree_env.py](../real_sim/trackfree_env.py); from scratch, Exp 10 recipe `450k` at `λ=0.02` + `100k` at `λ=0.20`; 5 seeds at mid, 3 at clean and hard.)
The actor sees the sensor-limited racing obs **plus the last 8 Exp 21 kinematic frames**, so velocity is inferable without the course-position shortcut. Arms differ only in the critic input (`asym`: + privileged exact-state block; `sym`: masked). A behaviour-cloning warm start was piloted and rejected: it degraded under PPO.

| Level | Arm | figure8 source | ellipse3d source | ellipse3d target_nominal | ellipse3d target_perturbed |
|---|---|---|---|---|---|
| clean (3) | asym | `1.00±0.00` / `0.00` | `0.27±0.11` / `0.40` | `0.51±0.18` / `0.00` | `0.41±0.11` / `0.10` |
| clean (3) | sym | `0.28±0.14` / `0.70` | `0.11±0.03` / `1.00` | `0.08±0.00` / `1.00` | `0.01±0.00` / `1.00` |
| mid (5) | asym | `0.98±0.04` / `0.00` | `0.23±0.10` / `0.56` | `0.42±0.17` / `0.14` | `0.24±0.15` / `0.58` |
| mid (5) | sym | `0.34±0.11` / `0.44` | `0.08±0.01` / `0.94` | `0.07±0.03` / `0.98` | `0.06±0.04` / `0.92` |
| hard (3) | asym | `1.00±0.00` / `0.00` | `0.50±0.26` / `0.33` | `0.30±0.08` / `0.13` | `0.29±0.03` / `0.07` |
| hard (3) | sym | `0.28±0.10` / `0.57` | `0.12±0.02` / `0.70` | `0.07±0.02` / `0.90` | `0.06±0.03` / `0.97` |

(gate fraction / crash rate, mean ± std over seeds)

- **The privileged critic is what makes sensor-limited PPO learnable:** asym > sym in `64/66` seed × cell pairs. Every asym seed learns the training course in the source (`0.91–1.00` gates, `0` crashes); no sym seed exceeds `0.45`.
- **Exp 20's failure came from the actor's observation:** with track-free kinematic inputs, the asym arm goes from `0.14` to `0.98–1.00` source training-course gates. Exp 20 also used a different sensor level, so this is attribution by elimination, not a one-factor test.
- **Caveat:** these directly trained students transfer worse than Exp 21's teacher + GRU-estimator students (mid / nominal target, held-out: `0.42` vs `0.78`), and held-out source performance stays weak (`0.23–0.50`). The privileged critic fixes learnability, not transfer.

### Exp 23 — From gate fraction to full-course completion for sensor-limited students (3 seeds × 3 sensor levels)
([exp23_student_completion.py](../real_sim/exp23_student_completion.py), [student_plus.py](../real_sim/student_plus.py).)
**Diagnosis** (Exp 21 `kin_gru_rand`, held-out `ellipse3d`):
- **Target: the whole gap is the PnP gate bias.** With the bias switched off, the Exp 21 student already completes the course in PyBullet at every sensor level (completion `1.00`, `0` crashes). With it on, completion is `0` (gates `0.76–0.86`).
- **Source: the velocity estimate is the bottleneck.** At the clean level, true velocity gives completion `1.00` and the estimate gives `0`. Its body-z error triples on `ellipse3d` (`0.24–0.28` vs `0.09–0.13 m/s`) because the planar training course under-covers vertical motion.
- **Source at hard is a teacher ceiling.** Even the teacher with *true* velocity reaches only `0.29–0.47` gates (stale geometry), which caps any observer.

**Fixes (ablated):**
- **F1:** estimator DAgger on a multi-track pool (figure8 + 6 Exp 18 circuits, excluding `ellipse3d` and its near-duplicates; the estimator is track-free, so no track geometry is learned).
- **F2:** delay compensation (predict the observed pose forward by its staleness).
- **F3:** PnP-bias GP fit from 3 calibration flights per target on the *training* course with ground-truth inputs, used to correct the observed position (bias `0.29–0.45 m` RMS → `0.05–0.07 m` held-out residual).

AR(1) action-noise exploration was tried for F1 first and gave no gain.

**Held-out `ellipse3d`, completion (mean ± std) / gate fraction / crash rate:**

| Sensor | Variant | `source_l2f` | `target_nominal` | `target_perturbed` |
|---|---|---|---|---|
| clean | Exp 21 `kin_gru_rand` | `0.33 ± 0.47` / `0.63` / `0.53` | `0.00` / `0.79` / `0.00` | `0.00` / `0.68` / `0.00` |
| clean | + F1 | `0.60 ± 0.33` / `0.90` / `0.10` | `0.00` / `0.75` / `0.00` | `0.00` / `0.63` / `0.13` |
| clean | + F1 + F2 + F3 | `0.60 ± 0.33` / `0.90` / `0.10` | `1.00 ± 0.00` / `1.00` / `0.00` | `0.70 ± 0.16` / `0.97` / `0.00` |
| clean | F3 only | `0.33 ± 0.47` / `0.63` / `0.53` | **`1.00 ± 0.00`** / `1.00` / `0.00` | **`0.77 ± 0.21`** / `0.98` / `0.00` |
| mid | Exp 21 `kin_gru_rand` | `0.27 ± 0.38` / `0.51` / `0.67` | `0.00` / `0.78` / `0.00` | `0.00` / `0.59` / `0.27` |
| mid | + F1 + F2 | `0.50 ± 0.41` / `0.74` / `0.40` | `0.03` / `0.69` / `0.07` | `0.00` / `0.55` / `0.17` |
| mid | F3 only | `0.27 ± 0.38` / `0.51` / `0.67` | **`0.93 ± 0.09`** / `0.99` / `0.03` | **`0.60 ± 0.33`** / `0.96` / `0.00` |
| hard | Exp 21 `kin_gru_rand` | `0.00` / `0.18` / `1.00` | `0.00` / `0.76` / `0.00` | `0.00` / `0.56` / `0.27` |
| hard | + F1 + F2 | `0.00` / `0.29` / `0.87` | `0.03` / `0.64` / `0.07` | `0.00` / `0.39` / `0.43` |
| hard | + F1 + F2 + F3 | `0.00` / `0.29` / `0.87` | `0.13 ± 0.12` / `0.82` / `0.00` | `0.07 ± 0.09` / `0.78` / `0.07` |
| hard | F3 only | `0.00` / `0.18` / `1.00` | **`0.53 ± 0.41`** / `0.94` / `0.00` | **`0.60 ± 0.33`** / `0.96` / `0.00` |

- **F3, the perception-bias residual, is what turns gate fraction into completion in the target:** completion `0 → 0.53–1.00` (nominal) and `0 → 0.60–0.77` (perturbed) at all levels, with `0–3%` crashes. This is Swift's GP perception residual (Prop 2.7) working on a racing task.
- **F1 is the source fix:** clean completion `0.33 → 0.60` and crashes `0.53 → 0.10`; mid `0.27 → 0.43`. It is neutral to slightly negative in the target.
- **F2 helps the source but hurts the target at mid and hard** (velocity-extrapolation error), so the full stack is *worse* than F3 alone in the targets. The best deployable student differs by domain: F1 + F2 in the source, F3 alone in the target.
- **Still open:**
  - collision-free completion stays modest (`≤ 0.50` in the target);
  - the source at hard stays at completion `0` (teacher ceiling under stale geometry);
  - seed variance on completion is high;
  - F3 assumes mocap ground truth for 3 calibration flights, as in Exp 9/10.

### Exp 24 — DR budget sweep: under-training vs transfer harm (H2' follow-up to Exp 19)
([exp24_dr_budget_sweep.py](../real_sim/exp24_dr_budget_sweep.py), 4 arms × 3 seeds; one continuing `λ=0.02` trunk per run, with `λ=0.20` fine-tune branches (22% of trunk) at total budgets `550k / 1.1M / 2.2M`; the 550k point reproduces Exp 19 exactly.)

| Arm | Budget | source figure8 gates | source ellipse3d gates | held-out `target_nominal` gates / clean | held-out `target_perturbed` gates / clean | median lap (nom) |
|---|---|---|---|---|---|---|
| `none` | 550k / 1.1M / 2.2M | `1.00 / 1.00 / 1.00` | `0.80 / 0.33 / 0.34` | `1.00/0.40`, `1.00/1.00`, `1.00/1.00` | `1.00/0.40`, `0.99/0.87`, `1.00/1.00` | `5.02 / 4.45 / 4.21 s` |
| `full` | 550k / 1.1M / 2.2M | `0.78 / 1.00 / 1.00` | `0.67 / 0.90 / 0.96` | `0.75/0.00`, `1.00/1.00`, `1.00/1.00` | `0.67/0.00`, `1.00/1.00`, `1.00/1.00` | `7.12 / 5.83 / 5.60 s` |
| `latency_only` | 550k / 1.1M / 2.2M | `1.00 / 1.00 / 1.00` | `1.00 / 0.86 / 0.34` | `1.00/0.67`, `1.00/0.67`, `1.00/1.00` | `0.97/0.33`, `1.00/0.67`, `1.00/0.77` | `5.63 / 4.74 / 4.39 s` |
| `no_mass` | 550k / 1.1M / 2.2M | `0.97 / 1.00 / 1.00` | `0.86 / 1.00 / 0.72` | `0.92/0.33`, `1.00/0.67`, `1.00/0.67` | `0.86/0.00`, `1.00/0.67`, `1.00/0.33` | `5.61 / 5.29 / 5.45 s` |

- **The Exp 10/19 DR penalty is under-training, not transfer harm:** once full DR's source skill catches up (`1.00` on figure8 at ≥1.1M), it matches no-DR in both PyBullet targets (`1.00` gates / `1.00` clean completion on held-out `ellipse3d`).
- **No measurable DR transfer *benefit* either:** every arm saturates both targets at ≥1.1M (ceiling effect).
- **Where DR does help:** on the hardest condition, held-out `ellipse3d` in the low-thrust-margin L2F source. Full DR improves with budget (`0.67 → 0.90 → 0.96`), while no-DR and latency-only *degrade* (`0.80 → 0.34`, `1.00 → 0.34`). Without dynamics DR, longer training overfits the training course under L2F's tight thrust limits.
- **Persistent cost:** DR policies fly laps about `33%` slower (`5.60` vs `4.21 s` at 2.2M), a robustness-for-speed trade-off (cf. Prop 2.3 / Exp 17).
- **Caveats:** 3 seeds; clean completion is nearly binary per seed; the targets are too easy at ≥1.1M to rank arms.

## Round 7 — In progress (2026-11-26; usage limit hit mid-round)

### Exp 25 — Does source performance predict target performance? (sim→target predictivity)
([exp25_predictivity.py](../real_sim/exp25_predictivity.py); no training; 51 exact-state PPO policies and 118 sensor-limited students from Exp 10/19/21–24; Pearson, Spearman, and MMRV (definition from memory of SIMPLER); 95% bootstrap CIs over policies.)
- **Training course:** the L2F ranking predicts the PyBullet ranking (Spearman `0.84–0.88`, MMRV ≤ `0.02` gates).
- **Held-out course:** no predictivity for exact-state PPO (Spearman `≈0.04`, CI spans 0). The L2F held-out course is the harder environment (mean gates `0.76` vs `0.94`), driven by L2F thrust-limit overfitting. Dynamics-DR policies stay partly predictive (`0.47`).
- **Target-side corrections break predictivity by construction:** adding the F3 GP students raises clean-completion MMRV from `0.00` to `0.49`.
- **Caveats:** target ceiling compresses correlations; checkpoints sharing a trunk aren't independent.

Exp 26 (gap attribution), 27 (hover-normalized thrust DR), 28 (student RL fine-tuning) and 29 (applied-action readback) were started but stopped unfinished when the usage limit was reached; any partial scripts/outputs in `real_sim/exp26–29*` and `results/real_sim/exp26–29*` are uncommitted and unverified.

## Setup & reproducibility notes
- Reproducible environment & build scripts are captured in [code/pixi.toml](../code/pixi.toml) and [code/setup_sims.sh](../code/setup_sims.sh); run `pixi run --manifest-path code/pixi.toml test` (`17/17` tests passing in ~5 s).
