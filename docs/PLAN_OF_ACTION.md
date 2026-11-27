# Plan of Action — Updated 2026-11-25

**Project:** One Method, Two Racing Worlds (CMU 16-831) · **Current scope:** drone branch only (Trackmania paused for a few days)
**Results so far:** `docs/REAL_SIM_RESULTS.md` · **Code:** `real_sim/`, `code/sims/drone_pair.py` · **Outputs:** `results/real_sim/`

---

## 1. Done

### Environment
- [x] `l2f` 2.0.18 built and installed in `code/.pixi` (Homebrew-Python wheel workaround for spaces in path; `nlohmann/json.hpp` fetched).
- [x] `gym-pybullet-drones` 2.2.0 installed editable (`--no-deps`, conda-forge pybullet/numpy kept) + `stable-baselines3`, `pettingzoo`, `transforms3d`, `control`.
- [x] Re-ran Gemini's toy suite (`results/theory_sim_rerun/`): means reproduce, but quoted ± are SEM and most method gaps are within seed noise.

### Shared real-sim infrastructure (`code/sims/drone_pair.py`)
- [x] One 500 Hz CTBR rate loop + quadX allocator driving both sims (100 Hz policy); frame/sign checks pass in both.
- [x] L2F backend (dt = 2 ms, parameter overrides for DR, state snapshot/restore, hidden-rotor reset, residual-injection hook).
- [x] PyBullet CF2X backend (`PYB_GND_DRAG_DW`, 500 Hz, wrapper motor lag τm).
- [x] Interface gap: action delay FIFO, obs delay, ZOH vs zero dropout, obs noise.
- [x] Privileged geometric expert, training figure-8 + held-out 3D ellipse course, gate-pass metric.

### Experiments (all on the real pair)
| # | Claim | Status |
|---|---|---|
| 1 | Open-loop replay divergence | ✅ done |
| 2 | Prop 2.1 delay stability + relay limit cycle | ✅ confirmed (relay period corrected to 4d+2) |
| 3 | Lemma 1.1 ZOH vs zero-mask | ✅ confirmed |
| 4/4b | Thm 2.2 action buffer, H2' latency/dynamics DR | ✅ buffer confirmed; ❌ dynamics DR no gain |
| 5 | Thm 2.4 / H3' recurrent observer | ⚠️ weak / ❌ closed-loop |
| 6 | Thm 2.5 delay alignment, joint (d̂, τ̂) SysID, body vs world k-NN | ✅ confirmed |
| 7 | H1' smoothness penalty (PPO) | ❌ not supported as tested |
| 8 | H4' residual model fidelity + closed loop | ✅ fidelity; ⚠️ closed-loop only partial |

---

## 2. Next steps (prioritized)

### P1 — Close the gaps in the weak/negative results ✅ done 2026-11-25 (see REAL_SIM_RESULTS.md, Round 2)
1. [x] **H1' retest (Exp 7b/7c).** → partial support; warm-start penalty works, from-scratch stalls. PPO with Δa penalty λ ∈ {0, 0.3} *combined with* action buffer + latency randomization; longer budget (≥2M steps, vectorized L2F). Success: λ>0 lowers crash rate at d=2–4 vs λ=0 with equal d=0 performance.
2. [x] **H3' retest (Exp 5b).** → partial support. 3–4× DAgger budget for GRU students; a harder sensor channel (larger obs delay 2–3, 20% dropout, pose-correlated bias b(φ) per Prop 2.7) so the observability gap is large enough for ρ_obs to be meaningful.
3. [x] **Regime fix** → diagnosed instead: fragility is L2F's actuator model; target kept, delay tested at d ≥ 12. The PyBullet CF2X (30 ms motors) is more delay-tolerant than L2F. Add a target variant with slower/asymmetric motor lag (e.g. τ_up 40 ms / τ_down 80 ms) so d = 2–6 matters in the target, then re-run Exp 4b / 8b there.

### P2 — The headline H4 experiment ✅ done 2026-11-25 (Exp 9; racing-course version moves to P3)
4. [x] **Swift-style residual fine-tuning with RL (Exp 9).** → 69% gap closed on nominal target, DR negative. Start from the nominal PPO policy, fine-tune in the residual-augmented L2F (`L2FBackend.residual_fn` + τ̂ + d̂), evaluate in PyBullet. Compare at equal source budget against: nominal, dynamics DR, latency DR, target-trained PPO oracle.
5. [x] **GP observation residual (Exp 9).** → works; needs ground-truth-pose input. Add pose-correlated systematic sensor bias in the target and fit the GP perception residual (Prop 2.7: zero-mean noise alone gives μ_GP → 0, so include this as the null control).

### P3 — Proposal method matrix & reporting ✅ done 2026-11-25 (`Exp 10` + `Exp 9` fix)
6. [x] **Full matrix on one consistent task (`real_sim/exp10_method_matrix.py`, `real_sim/racing_env.py`):** `{classical_expert, nominal_ppo, dr_ppo, ff_student, gru_student_fixed, gru_student_rand, residual_ft_ppo, target_oracle_ppo}` across `{figure8 train, ellipse3d held-out}` × `{source_l2f, target_nominal, target_perturbed}`, 3 seeds (`550k` source budget). `residual_ft_ppo` hits `1.000 ± 0.000` gate completion and `1.00` collision-free completion on held-out `ellipse3d` (`9/9` exact Step-0 calibrations).
7. [x] **Proposal metrics everywhere:** completion, collision-free completion, median/`p90` lap time, centerline & gate tracking error, crash rate, action smoothness, target/source ratio, and one-factor robustness sweeps (`delay`, `dropout`, `kf_scale`).
8. [x] **Baselines:** time-free spatial `GeometricGateController`, nominal source PPO, in-domain PyBullet PPO oracle, and feed-forward teacher cloning.
9. [x] **Update docs:** updated `THEORY_AND_HYPOTHESES.md`, `EMPIRICAL_THEORY_VERIFICATION.md`, `REAL_SIM_RESULTS.md`, and `FOLLOWUP_RIFFS_PROPOSAL.md`.

### P3+ — Phase-2 Literature Riffs ✅ ALL DONE 2026-11-26 (`docs/FOLLOWUP_RIFFS_PROPOSAL.md`, `Exp 11–18`)
- [x] **`Exp 11` (`R1.1` CAPS + L2C2 smoothness):** Eliminated `Exp 7b` 2/3-seed stall (`3/3` learned from scratch, `0%` crashes, `44×` smoother, `0.079 m` target error beating two-stage warm-start `0.182 m`).
- [x] **`Exp 12` (`R2.3 + R3.1/R3.3` DIDA Chunk-4 + Track-Free RMA):** `0.975 ± 0.030` gates at `80%` command-packet drop; uncovered & fixed gate-coordinate shortcut trap (`4.33 → 1.31 m/s` velocity RMSE on `ellipse3d`).
- [x] **`Exp 13` (`R3.2` Neural-Fly Live 100 Hz Composite RLS):** `1.000 ± 0.000` (`18/18` gates, zero target pre-calibration) in `0.5g` crosswind (`−67%` RMSE) and mid-race `-25%` motor loss (`−45%` vs PID), plus algebraic-loop ablation.
- [x] **`Exp 14` (`R4.1 + R4.2 + R5.1` CMU ASAP $\Delta a$ + MOPO Ensemble vs. Swift `k-NN`):** `swift_mopo_ft` achieved `1.00` clean completion across all `4/4` cells; identified `ASAP`'s source-actuator saturation boundary (`kf × 1.20`) and `11×` $\Delta f \equiv 0$ phantom-residual collapse.
- [x] **`Exp 15` (`R0.1 + R0.2` Fisher Safe Excitation + 10-Tick Parametric Shooting):** ⚠️ audit: Fisher, hand multi-sine and a Gaussian random walk all pick the right cell of a 4×4 `(τ_up, τ_down)` grid that contains the truth (17–18/18 trials), so no Fisher advantage is shown, and all exceed the 0.5 m sphere in the target at ≥1 s; showed parametric-only shooting closes `18%` vs `65%` for 6D `k-NN` due to unexposed $J$, `km/kf`, and downwash.
- [x] **`Exp 16` (`R1.2 + R1.3` Architectural EMA Bandwidth + Residual-over-Classical RL):** `53–73%` lower target hold error across `d=0..8` via 5-line EMA wrapper; `15–24×` early sample-efficiency gain (`50k` steps) with Song et al. (*Sci. Robot.* 2023) asymptotic E2E crossover near saturation.
- [x] **`Exp 17` (`R2.1 + R2.2` DORAEMON Curriculum DR + CVaR-PPO):** `doraemon_ppo` eliminated `source_d0` crashes (`47% → 0%`) and beat `fixed_wide_dr` by `+16..+32` gate pts (`3.1 s` faster `p90` laps); verified Prop 2.3 risk-speed conservatism on `CVaR_{0.35}`.
- [x] **`Exp 18` (`R4.3 + R5.2` 2-Round Bootstrapped Swift + 10-Track 3D Benchmark):** `−33%` on-policy residual RMSE (`0.360 → 0.241 m/s²`), `−50%` 10-track macro tracking error (`0.135 → 0.067 m`), and `1.00` clean completion across `9/10` unseen 3D circuits at a `16 s` target budget.

### P3++ — Audit follow-ups (2026-11-26, `docs/REAL_SIM_RESULTS.md` Round 5)
- [x] **Exp 10 sensor-limited rows documented.** Every method crashed ~100%, even in the source on the training course. Diagnosed as a course-position shortcut in learned estimators plus a near-infeasible training sensor.
- [x] **Exp 21 — H3' retest with track-free velocity observers** (`real_sim/kin_observer.py`, 3 seeds × 5 students × 3 sensor levels). The randomized GRU reaches `0.76` gates / `0` crashes on held-out `ellipse3d` in the nominal target at the Exp 10 sensor setting (Exp 10: `0.05` / `1.00`). GRU > FF (recurrence ablation) and randomized > fixed (Thm 2.4) in every target cell.
- [x] **Exp 19 — leave-one-out DR factor ablation.** Latency-only randomization is the best variant; the DR penalty comes from the dynamics factors cumulatively and mostly reflects lower source skill at equal budget.
- [x] **Exp 20 — privileged vs symmetric critic.** Both arms failed to learn with the shortcut-prone observation, so the ablation is uninformative. **Open:** rerun with a track-free actor.
- [x] **Exp 15 claims corrected.** "0.000 ms" is a grid-cell hit, Fisher shows no advantage over a Gaussian walk, and every strategy exceeds the 0.5 m safety sphere in the target at ≥ 1 s.
- [x] Doc links made relative (they work on GitHub now); +5 regression tests (`tests/test_asym_policy.py`).
- [x] **Round 6 (2026-11-26, three parallel sub-studies):**
  - **Exp 22 (privileged critic):** decisively helps with a track-free actor (`64/66` pairs).
  - **Exp 23 (full-course completion):** a PnP-bias GP gives completion in both targets at every sensor level; multi-track estimator training fixes the source at clean.
  - **Exp 24 (DR budget sweep):** the DR penalty is under-training; no transfer gain or harm at ≥1.1M; DR improves held-out L2F robustness.
- [x] **Round 7 (2026-11-27, Exp 25–29, five parallel sub-studies):**
  - **Exp 25:** predictivity holds on the training course only.
  - **Exp 26:** the gap is mostly motor lag.
  - **Exp 27:** hover-normalized thrust removes most of the DR skill tax.
  - **Exp 28:** student RL fine-tuning in the source hurts transfer.
  - **Exp 29:** the readback fixes the phantom residual but is not a better policy input.
- [ ] **Residual drone items (non-blocking):** collision-free completion for students (`≤ 0.50`); the source at hard sensor (teacher ceiling under stale geometry); harder targets so DR arms can be ranked (ceiling effect at ≥1.1M).

### P4 — Housekeeping ✅ done 2026-11-25
10. [x] **Reproducible env:** `pixi run setup-sims` (`code/setup_sims.sh`) clones pinned l2f / gym-pybullet-drones / TMNF-C, applies `code/patches/tmnf-c-macos.patch`, builds the l2f wheel, installs `--no-deps` extras.
11. [x] **Tests:** `pixi run test` — 15 tests in ~15 s (`tests/test_drone_pair.py`): sign conventions, determinism, delay FIFO/predict_forward, ZOH, Prop 2.1 limit.
12. [x] **Version control:** git repo `Final Project/` → private GitHub `ebrownhi/trackmania1` (`main`).

### P5 — Resume Trackmania (after drone P1–P2)
13. Run TMNF-C `ctest` and wire its Python bindings; wrap with the existing `AsyncWallClockTargetWrapper`.
14. Re-run the transferable experiments (Exp 2 delay limit, Exp 3 dropout, Exp 4b buffer, Exp 6 phantom residual) on the Δf ≡ 0 platform; Windows TMInterface only for final validation.

---

## 3. Open decisions for the team
- **Course speed:** 3.5 m/s peak is the practical limit on the L2F Crazyflie (T/W ≈ 1.47). Accept, or raise L2F thrust_scale to emulate a racing quad?
- **Policy learning:** keep DAgger students for mechanism tests and use PPO only for the H1'/H4 headline runs, or move everything to PPO?
- **Target fidelity:** keep the 30 ms symmetric motor lag as the "official" target, or adopt the slower asymmetric variant from P1.3?
- **Trackmania restart date** and who owns the Windows snapshot export.
