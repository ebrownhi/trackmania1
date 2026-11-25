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
| H4' residual model closes gap | Exp 8, 9 | ✅ model fidelity (replay 0.48 → 0.064 m); ✅ PPO residual fine-tuning closes 69% of the zero-shot→oracle gap on the nominal target vs DR −29%; ❌ on the perturbed target (d̂ misidentified on 2/3 seeds) |
| Prop 2.7 GP perception residual | Exp 9 | ✅ biased sensor: GP recovers the bias (0.093 vs 0.093 m) and FT closes 56% of the gap; noisy null: GP mean collapses to 0.003 m — **only if the GP input is ground-truth pose** (observed-pose input leaks noise and learns a spurious 0.16–0.25 m bias) |

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
| perturbed (kf × 1.15) | 0.064 ± 0.013 | 0.069 (−10%) | 0.095 (−65%) | 0.076 (−26%) | 0.017 |
| biased sensor b(p) | 0.159 ± 0.054 | 0.169 (−14%) | 0.189 (−43%) | **0.120 ± 0.041 (+56%)** | 0.090 |
| noisy sensor (null) | 0.142 ± 0.076 | 0.131 (+17%) | 0.161 (−28%) | **0.066 ± 0.021 (+115%)** | 0.076 |

Step-0 SysID on 1,600 samples recovers τ̂ = 20–30 ms (true 30) and d̂ = 0 on 10/12 calibrations; both misses
(d̂ = 1) include 2/3 perturbed-target seeds, where residual FT then fails.

## Corrections to earlier docs

1. `EMPIRICAL_THEORY_VERIFICATION.md` (Gemini toy suite) reproduces its means exactly (`results/theory_sim_rerun/`), but its "±" values are SEM; true seed std ≈ the mean (e.g. buffer vs a_{t-1} at d=4: 16.7 ± 11.6 vs 10.1 ± 10.8 — not separable). It also omits that toy GRU_RandSrc scored *worst* on the drone target (0.13 vs FF 1.73).
2. Relay limit-cycle period is 4d+2 ticks, not 4(d+1).
3. "v_peak ≥ 10 m/s" is infeasible on these airframes (L2F Crazyflie T/W ≈ 1.47); courses use 3.5 m/s.
4. L2F's reported `hovering_throttle_relative` (0.726) is inconsistent with its thrust curve; true hover ≈ 0.806 normalized rpm.
5. Guardrail #1 as written ("always include the Δa penalty") is incomplete: from scratch it can stall PPO (2/3 seeds at λ=0.3); adding it as a fine-tuning stage is reliable.
6. Prop 2.7's GP collapse requires the GP input to be independent of the sensor noise (fit on ground-truth / mocap pose).

## Caveats / next steps

- H4 RL fine-tuning (Exp 9) is on a hover task; the racing-course version is part of the P3 method matrix.
- The target is *more* delay-tolerant than the source (30 ms vs 55/249 ms motors), so delay effects show mainly at d ≥ 8.
- Next: H1' retest combining Δa penalty (λ≈0.3) + buffer + latency randomization; more DAgger budget for GRU_rand; PPO fine-tuning in the augmented source.
- Reproducibility: l2f had to be built with Homebrew Python (spaces in path break the conda build flags) and needs `external/json/nlohmann/json.hpp`; gym-pybullet-drones installed `--no-deps`. Not yet captured in `code/pixi.toml`.
