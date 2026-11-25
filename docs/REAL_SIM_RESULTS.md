# Real-Simulator Results: L2F → gym-pybullet-drones (Drone Branch)

**Date:** 2026-11-25 · **Scope:** drone only (Trackmania paused) · **Code:** `real_sim/`, `code/sims/drone_pair.py` · **Outputs:** `results/real_sim/` (JSON + PNG per experiment)

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
| H2' dynamics DR helps | Exp 4b | ❌ no measurable gain |
| H1' smoothness penalty (PPO) | Exp 7 | ❌ not supported as tested (penalty alone gave no delay robustness; λ=1 under-trained) |
| Thm 2.4 recurrent-observer OOD | Exp 5 | ⚠️ weak: GRU_fixed degrades most OOD (2.1×) but only to FF level; gap too small |
| H3' GRU_rand best closed-loop | Exp 5 | ❌ not supported |
| Nuance #1 (velocity removal unobservable) | Exp 5 | ❌ not reproduced (2 cm position noise still gives ~0.1 m/s velocity) |
| Thm 2.5 phantom residual / delay alignment | Exp 6 | ✅ Δf≡0 control: d̂=3 exact, naive 0.12 vs aligned 0.007 m/s² (17×) |
| Guardrail #5 + Nuance #2 joint (d̂, τ̂) SysID | Exp 6 | ✅ needed on PyBullet: delay-only scan fails (d̂=0); joint scan recovers d̂=3, τ̂=30 ms exactly |
| Guardrail #6 body-frame vs world-frame k-NN | Exp 6, 8 | ✅ world-frame gives 0% held-out improvement; body-frame −60% residual RMSE |
| H4' residual model closes gap | Exp 8 | ✅ model fidelity (replay 0.48 → 0.064 m train course, 0.75 → 0.27 m held-out); ⚠️ closed-loop only repairs a model-based controller, does not beat naive at d=6 |

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

## Corrections to earlier docs

1. `EMPIRICAL_THEORY_VERIFICATION.md` (Gemini toy suite) reproduces its means exactly (`results/theory_sim_rerun/`), but its "±" values are SEM; true seed std ≈ the mean (e.g. buffer vs a_{t-1} at d=4: 16.7 ± 11.6 vs 10.1 ± 10.8 — not separable). It also omits that toy GRU_RandSrc scored *worst* on the drone target (0.13 vs FF 1.73).
2. Relay limit-cycle period is 4d+2 ticks, not 4(d+1).
3. "v_peak ≥ 10 m/s" is infeasible on these airframes (L2F Crazyflie T/W ≈ 1.47); courses use 3.5 m/s.
4. L2F's reported `hovering_throttle_relative` (0.726) is inconsistent with its thrust curve; true hover ≈ 0.806 normalized rpm.

## Caveats / next steps

- Students imitate a feedback expert, so source-model dependence only enters through delay prediction; a proper H4 test needs RL fine-tuning in the residual-augmented source (hook exists: `L2FBackend.residual_fn`).
- The target is *more* delay-tolerant than the source (30 ms vs 55/249 ms motors), so delay effects show mainly at d ≥ 8.
- Next: H1' retest combining Δa penalty (λ≈0.3) + buffer + latency randomization; more DAgger budget for GRU_rand; PPO fine-tuning in the augmented source.
- Reproducibility: l2f had to be built with Homebrew Python (spaces in path break the conda build flags) and needs `external/json/nlohmann/json.hpp`; gym-pybullet-drones installed `--no-deps`. Not yet captured in `code/pixi.toml`.
