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

### P1 — Close the gaps in the weak/negative results (≈1–2 days)
1. **H1' retest (Exp 7b).** PPO with Δa penalty λ ∈ {0, 0.3} *combined with* action buffer + latency randomization; longer budget (≥2M steps, vectorized L2F). Success: λ>0 lowers crash rate at d=2–4 vs λ=0 with equal d=0 performance.
2. **H3' retest (Exp 5b).** 3–4× DAgger budget for GRU students; a harder sensor channel (larger obs delay 2–3, 20% dropout, pose-correlated bias b(φ) per Prop 2.7) so the observability gap is large enough for ρ_obs to be meaningful.
3. **Regime fix: make the target delay-sensitive.** The PyBullet CF2X (30 ms motors) is more delay-tolerant than L2F. Add a target variant with slower/asymmetric motor lag (e.g. τ_up 40 ms / τ_down 80 ms) so d = 2–6 matters in the target, then re-run Exp 4b / 8b there.

### P2 — The headline H4 experiment (≈2 days)
4. **Swift-style residual fine-tuning with RL.** Start from the nominal PPO policy, fine-tune in the residual-augmented L2F (`L2FBackend.residual_fn` + τ̂ + d̂), evaluate in PyBullet. Compare at equal source budget against: nominal, dynamics DR, latency DR, target-trained PPO oracle.
5. **GP observation residual.** Add pose-correlated systematic sensor bias in the target and fit the GP perception residual (Prop 2.7: zero-mean noise alone gives μ_GP → 0, so include this as the null control).

### P3 — Proposal method matrix & reporting (≈2 days)
6. **Full matrix on one consistent task:** {nominal PPO, DR, teacher–student, residual FT} × {exact-state, sensor-limited} × {nominal, perturbed target}, ≥3 seeds, held-out course.
7. **Proposal metrics everywhere:** completion, collision-free completion, median/tail lap time, tracking error, crash rate, action smoothness, target/source ratio; one-factor robustness curves.
8. **Baselines:** classical controller (geometric expert), target-trained PPO reference, feed-forward teacher cloning.
9. **Update docs:** replace claims in `THEORY_AND_HYPOTHESES.md` / `EMPIRICAL_THEORY_VERIFICATION.md` that the real-sim results don't support; keep `REAL_SIM_RESULTS.md` current.

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
