# Follow-up riffs, round 2 — literature scan (2026-11-26)

Four parallel literature agents searched the areas around the proposal's papers (Swift, L2F, gym-pybullet-drones,
Linesight / TMNF-C): drone-racing RL, fast simulators and quadrotor learning, general sim-to-real transfer, and
racing-game / delayed RL. This file merges their reports into one ranked list, excluding everything already done
(see `FOLLOWUP_RIFFS_PROPOSAL.md` and Exp 1–24 in `REAL_SIM_RESULTS.md`).

**Caveat:** agents opened every cited arXiv abs page, but for most papers read only the abstract. Mechanisms marked
"(abstract)" are partly from memory and should be checked before relying on them.

## One finding that changes the Trackmania plan

Linesight's own live-game loop pauses the game at every action step (`request_speed(0)` → grab frame → infer → send
inputs → resume; `trackmania_rl/tmi_interaction/game_instance_manager.py`; 10 ms ticks, 5 ticks per action = 20 Hz).
In that mode inference latency costs no game time. So the "interface gap" behind H1 exists only if the policy runs
in real time, as tmrl/rtgym does; otherwise the remaining gap is rendered observations plus state-readout mismatch.
This suggests running Trackmania as a **gap ladder** (idea T1 below) instead of a single live-game evaluation.

## Tier 1 — small effort, reuses existing pipelines (drone side)

| # | Idea | Source | Experiment in our setup | Bears on |
| --- | --- | --- | --- | --- |
| 1 | **Sim→target predictivity metrics** (SRCC, Spearman, rank-violation MMRV) | Kadian et al. [1912.06321](https://arxiv.org/abs/1912.06321); SIMPLER [2405.05941](https://arxiv.org/abs/2405.05941) | Correlate source vs target scores across all existing checkpoints (nominal, DR, budget sweep, students, residual FT). No training. Does residual FT or latency DR raise predictivity? | H1, H2 (evaluation) |
| 2 | **Per-component gap attribution** | Eschmann et al. motor-delay SysID [2404.07837](https://arxiv.org/abs/2404.07837); RAPTOR [2509.11481](https://arxiv.org/abs/2509.11481); NeuroBEM [2106.08015](https://arxiv.org/abs/2106.08015) | Move L2F one factor at a time toward PyBullet (motor lag, drag, thrust/torque constants), retrain cheaply, report the gap removed per factor. Optionally use PyBullet's physics modes (PYB → PYB_GND_DRAG_DW) as a graded series of targets. | H1 (attribution table) |
| 3 | **Hover-normalized thrust + T/W-only or per-axis DR** | SimpleFlight [2412.11764](https://arxiv.org/abs/2412.11764); Gräfe et al. [2603.05944](https://arxiv.org/abs/2603.05944) | Thrust = T_hover·(1 + k·a); DR only on thrust coefficient / T/W vs full DR at equal budget; also 4× PPO batch at equal steps. Tests whether the Exp 24 DR skill tax comes from randomizing parameters that didn't need it. | H2 |
| 4 | **Spread-out and replayed-state starts** | Song et al. IROS 2021 [2103.08624](https://arxiv.org/abs/2103.08624) (full text read) | Reset at hover along every segment, then from states logged in earlier rollouts; variant seeded with PyBullet-logged states. Measure target gates and recovery after velocity/attitude kicks. | H1, H2 |
| 5 | **Observe the applied action, not the commanded one** | GT Sophy, Nature 2022 ([PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)) | Replace the commanded-action history with the action actually applied (drone: motor-command readback) under a 1-step misaligned delay. Candidate fix for the phantom residual (Thm 2.5). | H2, Thm 2.5 |
| 6 | **RAPTOR foundation policy as zero-shot baseline** | RAPTOR [2509.11481](https://arxiv.org/abs/2509.11481), `pip install foundation-policy` | Run the L2F-trained generalist GRU in gym-pybullet-drones (RPM actions, FLU frame, motor order FR-BR-BL-FL) for hover and gate-to-gate. No racing lookahead, so expect slow laps. Risk: frame and motor-order mapping. | H1 (generalist reference) |
| 7 | **Speed–robustness Pareto: gate safety margin × DR level** | Song 2021 safety reward; Ferede et al. [2504.21586](https://arxiv.org/abs/2504.21586) | 4×4 sweep; plot L2F lap time vs PyBullet success. Turns H2 into "how much DR, at what speed cost" (Exp 24 already found a 33% lap-time cost). | H2 (new H5) |

## Tier 2 — medium effort, new methods

| # | Idea | Source | Experiment | Bears on |
| --- | --- | --- | --- | --- |
| 8 | **RL fine-tuning of the distilled student** | Xing et al. CoRL 2024 [2403.12203](https://arxiv.org/abs/2403.12203) | PPO on the Exp 21/23 student in randomized L2F with the teacher's critic as init. Target: the open collision-free completion gap (≤ 0.50). | H3 |
| 9 | **Posterior DR from target rollouts** | BayesSim [1906.01728](https://arxiv.org/abs/1906.01728); SimOpt [1810.05687](https://arxiv.org/abs/1810.05687) | Infer a posterior over motor lag, thrust, mass, drag from ~10 PyBullet rollouts; DR from the posterior vs uniform at equal budget. Same recipe for Trackmania latency/jitter/drop rate. | H2 (both pairs) |
| 10 | **Learned actuator / interface network** | Hwangbo et al. [1901.08652](https://arxiv.org/abs/1901.08652) | Net(command history) → realised thrust/torques fitted on PyBullet, inserted into L2F; vs k-NN state residual. Tests whether the gap sits at the actuator. | H4 |
| 11 | **Off-dynamics reward correction** (DARC / PAR) | [2006.13916](https://arxiv.org/abs/2006.13916); [2405.15369](https://arxiv.org/abs/2405.15369) | Classifier (or representation) penalty from the same few target rollouts; vs bootstrapped Swift residual. A clean negative result is likely where the target needs behaviour the source can't produce. | H4 |
| 12 | **Grounded action transformation** (GARAT / RGAT) | [2008.01594](https://arxiv.org/abs/2008.01594); [2008.01279](https://arxiv.org/abs/2008.01279) | Trajectory-distribution-matched action transformer vs the existing ASAP delta-action (Exp 14), same data. | H4 |
| 13 | **Many-specialist → recurrent student distillation** | RAPTOR (two-stage meta-imitation) | 8–16 PPO specialists on fixed sampled dynamics, distilled into a small GRU; vs DR-PPO at equal samples. Could avoid the DR skill tax. | H2, H3 |
| 14 | **Embodiment-descriptor conditioning** | [2606.10857](https://arxiv.org/abs/2606.10857) (abstract only) | Append T/W, motor lag, inertia ratio to the obs under DR; feed CF2X nominal values at test. vs plain DR and RMA. | H2 |
| 15 | **Multi-task racing + recovery + velocity tracking** | Xing et al. RA-L 2024 [2412.12442](https://arxiv.org/abs/2412.12442) | Auxiliary recovery tasks; target success under perturbations vs racing-only. | H1 (new H6) |
| 16 | **Hybrid offline-target + online-source RL (H2O)** / **simulation-guided fine-tuning (SGFT)** | [2206.13464](https://arxiv.org/abs/2206.13464); [2502.02705](https://arxiv.org/abs/2502.02705) | ~20 target episodes as an offline set (H2O, needs SAC) or source critic as a shaping term for short target fine-tuning (SGFT). | H4 |

## Trackmania — ideas for when the branch restarts

| # | Idea | Source | Experiment | Where it runs |
| --- | --- | --- | --- | --- |
| T1 | **Gap ladder** | Linesight game loop; [tmrl](https://github.com/trackmania-rl/tmrl) | One frozen TMNF-C policy at three rungs: (a) paused stepping, state features (dynamics only, tests H4's near-zero residual); (b) paused + rendered frames; (c) real time at speed 1, rtgym-style. 3 tracks × 3 rungs × 20 runs. | Windows + TMInterface |
| T2 | **Sub-step delay + step-timing model** | tmrl (`start_obs_capture=0.04`, `time_step_duration=0.05`, timeouts) | Train with 0–4 engine-tick delay + timeouts vs integer-step delay; test at rung (c). Also applies to the drone (sub-ticks of the 2 ms physics step). | TMNF-C (macOS) |
| T3 | **Control-rate × delay sweep; drop stalled transitions** | GT Sophy | `ms_per_action` {20, 50, 100} × delay {0, 50, 100, 200 ms}. | TMNF-C (macOS) |
| T4 | **Step-length jitter with dt as input** | MOSEAC [2402.14961](https://arxiv.org/abs/2402.14961) (abstract) | Randomize action duration 40–80 ms plus rare 150 ms stalls; compare at rung (c). | TMNF-C + live game |
| T5 | **Linesight observation/return design for the PPO student** | Linesight `config_files/config.py` | 40 upcoming reference-line points, 5 previous actions, 7 s fixed-horizon "mini-race" return; gives a fair Linesight-style baseline. | TMNF-C (macOS) |
| T6 | **Delay-aware critics / belief forecasting** | AD-RL [2402.03141](https://arxiv.org/abs/2402.03141); DFBT [2505.00546](https://arxiv.org/abs/2505.00546) (abstracts) | Auxiliary short-delay critic for delay ≥ 2; direct current-state forecasting from delayed obs + actions in flight, supervised exactly by deterministic TMNF-C. | TMNF-C (macOS) |
| T7 | **Observation canonicalisation** (RCAN-style) | [1812.07252](https://arxiv.org/abs/1812.07252) | Translator from randomized observations to canonical ones; isolates the observation part of the interface gap. | TMNF-C + live game |

## Not recommended for this project

Need images, a GPU-native or differentiable simulator, or heavy solvers we don't have: Dream to Fly (pixels),
perception-aware teacher-student, differentiable-sim / BPTT fine-tuning (Heeg et al., DiffAero, Learning on the Fly),
Actor-Critic MPC (Romero et al. T-RO 2025), ReDRAW latent world-model residual, temporal skip connections for CNN latency.

## Notes on the agents' claims

- "gym-pybullet-drones applies RPM instantly" is true of stock gym-pybullet-drones, but our `PyBulletBackend` adds a
  30 ms first-order motor lag, so idea 2's attribution must start from our wrapper, not the stock env.
- SimpleFlight's "rotation matrix beats quaternion" is partly already in place: our racing obs use the body-frame
  gravity direction `R^T e_z`, not the full matrix.
- One agent found no published L2F → gym-pybullet-drones sim-to-sim study, so the setup appears novel.

## Suggested order

1. **Idea 1** (predictivity metrics) — no training, uses existing checkpoints (weights are local only).
2. **Ideas 2 and 3** — together they explain *where* the gap comes from and whether DR's skill tax is avoidable;
   both make clean report figures.
3. **Idea 8** — the most direct attack on the open collision-free completion gap.
4. **Idea 5** — cheap, and the only one aimed at the phantom residual; reusable for Trackmania.
5. Trackmania: plan around **T1** (gap ladder) and do **T2, T3, T5** in TMNF-C on macOS first.
