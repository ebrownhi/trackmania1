# Drone Sim-to-Sim: Audit & Rounds 5–6 Write-up

As of 2026-11-27. Also published as a Claude Doc: https://claude.ai/artifact/2wU1Kkw6ug4a9N2WvB26UE

The drone branch (L2F source → PyBullet target) now has a documented result for every proposal hypothesis, ablation and
baseline; Trackmania and the final report have not started. This write-up covers the work Claude did: setup and
Rounds 1–2, an audit of the Rounds 3–4 claims, the Round 5–6 fixes and new experiments, and Round 7's gap attribution (Exp 25–29). Full numbers live in
`docs/REAL_SIM_RESULTS.md`; the task list is `docs/PLAN_OF_ACTION.md`.

## Where each hypothesis stands

Three of the four drone-side predictions have a clear answer; H2 came out as "no transfer gain, no harm". Numbers are on
the held-out 3D course (`ellipse3d`) unless noted, 3 seeds.

| Hypothesis (drone part) | Verdict | Key number | Experiments |
| --- | --- | --- | --- |
| H1: nominal PPO degrades sharply on the drone | Partly | Zero-shot PPO passes all gates in the target but completes cleanly only 0.40 of the time | Exp 1, 10 |
| H1: where the drone gap comes from | Mostly motor lag | L2F's 55/249 ms motor model: 56% of prediction error, ~170% of the closed-loop gap; matching physics removes the training-course gap (0.036 → 0.036 m); 2-tick delay adds nothing | Exp 26 |
| H2: dynamics randomization helps the drone | No gain, no harm | Matches no-DR at 1.1M+ steps (1.00 / 1.00); 33% slower laps; better on held-out L2F (0.96 vs 0.34 gates). Hover-normalized thrust removes most of the skill tax at 550k (held-out L2F 0.68 → 0.97) | Exp 4b, 9, 10, 19, 24, 27 |
| H2: latency randomization helps | Yes | Best variant at equal budget (1.00 gates, 0.67 clean vs 0.40) | Exp 4b, 19 |
| H3: recurrent students improve sensor-limited transfer | Yes, after a fix | Students complete the held-out course in the target at every sensor level (nominal 1.00 / 0.93 / 0.53); source RL fine-tuning does not help (worse in 11/24 target pairs) | Exp 10, 21, 23, 28 |
| H4: residual fine-tuning beats randomization | Yes | 1.00 clean completion in both targets vs 0.00 for DR at 550k | Exp 9, 10, 14, 18 |
| Ablation: privileged critic | Helps learning, not transfer | Wins 64/66 seed × cell pairs | Exp 20, 22 |
| Ablation: recurrence | Helps | GRU beats feed-forward in every target cell (by 0.32–0.48 gates) | Exp 21 |
| Ablation: individual DR factors | Harm is cumulative | Removing any one factor only partly recovers | Exp 19 |
| Source→target predictivity | Training course only | Spearman 0.84–0.88 on the training course, ≈0.04 held out | Exp 25 |
| Applied-action readback | For model fitting, not the policy | Removes the phantom residual even under jitter (0.008 vs 0.071 m/s²); commanded history is the better policy input | Exp 29 |

The cross-domain halves of H1–H4 ("near-lossless", "unchanged", "similar fraction in both domains", "near-zero residual")
need Trackmania.

## What was done

Seven rounds of work landed over three days; Rounds 3–4 (Exp 10–18) were done by another contributor and audited, not authored, by Claude.

| Date | Round | What | Commit |
| --- | --- | --- | --- |
| 2026-11-27 | Audit 2 + Exp 30 | Audited Exp 11–18; final method matrix (5 seeds × 30 episodes, one protocol) | see git log |
| 2026-11-27 | Round 7 (Exp 25–29) | Five parallel sub-studies: predictivity, gap attribution, hover-normalized DR, student RL fine-tuning, applied-action readback | see git log |
| 2026-11-26 | Literature scan | Four agents; ranked ideas in `docs/FOLLOWUP_RIFFS_ROUND2.md` | `4d9111c` |
| 2026-11-26 | Round 6 (Exp 22–24) | Three parallel sub-studies: privileged critic, student full-course completion, DR budget sweep | `fe947fd` |
| 2026-11-26 | Round 5 (Exp 19–21) | Proposal ablations, sensor-limited failure diagnosed and fixed, doc corrections | `56249b4` |
| 2026-11-26 | Audit | Checked the "drone side is done" claim against the proposal and the repo | — |
| 2026-11-26 | Rounds 3–4 (Exp 10–18) | Method matrix and literature riffs (other contributor) | `8c7008b`, `0cd12fc` |
| 2026-11-25 | Round 2 (Exp 5b, 7b, 7c, 9) | H1'/H3'/H4' retests, PPO residual fine-tuning, GP bias residual | `8ad500d` |
| 2026-11-25 | Round 1 (Exp 1–8) | Shared L2F/PyBullet CTBR infrastructure and the theory experiments | `ccd68e7` |
| 2026-11-25 | Setup | l2f + gym-pybullet-drones installed, pixi env, `setup_sims.sh`, tests, git repo | `ccd68e7` |

## Audit of the "drone side is done" claim

The claim was mostly right but overstated; every gap it hid is now fixed or marked open.

- **Held up:** Exp 1–18 all had scripts, JSON and plots; Exp 10 was a full 3-seed run with every proposal metric and
  baseline; TMNF-C builds on macOS and its `ctest` passes 6/6; no final report existed.
- **Sensor-limited half of Exp 10 was undocumented:** every sensor-limited method crashed ~100%, even on the training
  course in the source. Fixed in Exp 21–23.
- **Two proposal ablations were missing:** individual randomization factors and the privileged critic. Done in Exp 19, 20, 22.
- **Exp 15 overclaimed:** "0.000 ms RMSE" was a hit on a 4×4 grid that contains the truth; a Gaussian random walk does as
  well as Fisher-optimal excitation; every strategy leaves the 0.5 m safety sphere in the target at ≥ 1 s. Corrected in the docs.
- **Exp 11–18, second audit (2026-11-27):** all numbers match their JSON, but six of seven headline claims go further than the data. Not supported: chunking's packet-loss tolerance (Exp 12: loss barely hurts the baseline), residual RL over a classical controller (Exp 16B: worse than the controller alone), the 2-round bootstrap (Exp 18: same completion as 1-round), and MOPO over k-NN (Exp 14: one seed). Omitted from the write-ups: nominal PPO beats every DR arm in Exp 17, and CAPS doubles source hold error in Exp 11. Table in `REAL_SIM_RESULTS.md`, *Audit of Exp 11–18*.
- **Housekeeping:** Rounds 3–4 were unpushed at the time (now pushed); doc links were absolute `file:///Users/...` paths
  (now relative); the GitHub repo is public, though described as private.

## Why the sensor-limited students failed, and the fix

The students learned a course-position shortcut instead of estimating velocity; a track-free estimator fixed it.

1. **No pipeline bug:** the sensor observation equals the exact one minus velocity plus 2 cm noise; a student distilled
   onto exact observations flies 1.00 gates, 0 crashes.
2. **The information is there:** the teacher plus a two-line alpha-beta filter flies 1.00 / 0 at the clean sensor, and
   tolerates 0.5–0.6 m/s velocity error.
3. **The Exp 10 training sensor is near-infeasible:** at 2-tick delay and 20% dropout, even the teacher with true
   velocity reaches 0.27 gates; a hand-built delay-compensating observer about 0.50.
4. **Learned estimators took a shortcut:** accurate on teacher states (~0.1 m/s) but closed-loop they reported v ≈ 0
   while the drone flew 1.5 m/s — they had learned "position on the figure-8 → typical velocity". Removing the action
   history did not help; removing track-relative inputs did.
5. **Fix (Exp 21):** a GRU that sees only body-frame position differences, a fresh-packet flag, attitude and rates,
   feeding the frozen teacher. Held-out target at the Exp 10 sensor: 0.76 gates / 0 crashes vs 0.05 / 1.00.
6. **To full completion (Exp 23):** the remaining target gap was the PnP gate bias; a GP fit from 3 calibration flights
   turns gate fraction into completion at every sensor level.

## Rounds 5–6 results

Each new experiment answered one open question; all used 3 seeds (Exp 22: 5 at mid).

| Exp | Question | Answer |
| --- | --- | --- |
| 19 | Which randomization factor causes the DR penalty? | None alone: the dynamics factors add up; latency-only is the best variant (1.00 gates, 0.67 clean) |
| 20 | Does a privileged critic help (Exp 10 inputs)? | Uninformative: both arms failed to learn with the shortcut-prone observation |
| 21 | Do recurrent, randomized students work with track-free inputs? | Yes: GRU > FF and randomized > fixed in every target cell; 0.76 gates / 0 crashes at the hard sensor |
| 22 | Privileged critic with a track-free actor? | Decisively helps learning (64/66 pairs; 0.98–1.00 vs ≤ 0.45 training gates) but not transfer (0.42 vs Exp 21's 0.78) |
| 23 | Can students complete the full course? | Yes in the targets, via a 3-flight PnP-bias GP (nominal 1.00 / 0.93 / 0.53 by sensor level); source improves 0.33 → 0.60 with multi-track estimator training |
| 24 | Is the DR penalty under-training or transfer harm? | Under-training: full DR matches no-DR at ≥ 1.1M steps; DR helps held-out L2F (0.96 vs 0.34) at a 33% lap-time cost |

## What is still open

Nothing open blocks the report on the drone side; Trackmania and the report itself are the real remaining work.

- Collision-free completion for students stays ≤ 0.50.
- The hard sensor in the L2F source still completes nothing (teacher ceiling under stale geometry).
- Targets are too easy at ≥ 1.1M steps to rank DR variants (ceiling effect).
- Trackmania: TMNF-C builds; no Python bindings or experiments yet. Half the proposal's claims need it; the proposal allows
  reporting an integration failure and completing the matrix on the drone platform instead.
- Final report: not started; the material is in `docs/REAL_SIM_RESULTS.md`.
- Literature scan: done (`docs/FOLLOWUP_RIFFS_ROUND2.md`); tier-1 ideas 1–5 tested in Round 7. Remaining ideas there (posterior DR, actuator network, DARC, GARAT, Trackmania gap ladder) are untested.
- Student fine-tuning in a target-informed (residual-augmented) source, the next step after Exp 28's negative result.

## Where everything lives

All code, results and docs are in https://github.com/ebrownhi/trackmania1 (currently public).

- Results and verdicts: `docs/REAL_SIM_RESULTS.md` (Rounds 1–6); plan: `docs/PLAN_OF_ACTION.md`.
- New scripts: `real_sim/exp19_dr_factor_ablation.py` … `exp29_applied_action_readback.py`; libraries `asym_policy.py`,
  `kin_observer.py`, `student_plus.py`, `trackfree_env.py`, `hover_norm.py`, `student_ft.py`, `applied_readback.py`.
- Tests: `cd code && pixi run test` (31 pass). Simulators: `pixi run setup-sims`.
- Trained weights (`results/**/*.zip`, `*.pt`) are gitignored and exist only on the local machine.
