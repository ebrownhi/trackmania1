# Empirical Theory Verification Report (5-Seed Mac TorchScript Suite)

**Project:** *One Method, Two Racing Worlds: Sim-to-Sim Transfer in Drone and Trackmania Racing* (CMU 16-831)  
**Runnable Suite:** `../theory_sim/` (`envs.py`, `models_and_learners.py`, `run_all_experiments.py`, `results.json`)  
**Execution:** 5 deterministic seeds (`0..4`) run in parallel on Apple M5 Pro CPU in **87.8 seconds**.

---

## 1. Summary of Verified Findings & Two New Nuances Discovered Empirically

Running end-to-end neural policy optimization, DAgger GRU distillation, and Swift $k$-NN/GP residual fine-tuning across matched analytical micro-simulators (`DroneSlalomEnv`: 3D underactuated CTBR quadrotor @ 100 Hz / 500 Hz inner loop; `TrackmaniaCircuitEnv`: 100 Hz Pacejka dynamic bicycle car with $\Delta f \equiv 0$) confirmed all theoretical predictions from `THEORY_AND_HYPOTHESES.md` **and** uncovered two important practical nuances for full simulator implementation:

1. **Proposal Step 1 Confirmed (`Exp 1` — Open-Loop Replay):**
   * **Drone ($\Delta f \neq 0, d = 0$):** Open-loop 3D position divergence grows super-linearly: **$1.55 \pm 0.01\text{ m}$ at $1.0\text{ s}$** ($4.8\times$ gate radius), **$5.07 \pm 0.03\text{ m}$ at $2.0\text{ s}$**, and **$9.60 \pm 0.05\text{ m}$ at $3.0\text{ s}$** as motor/rate-loop lag integrates into roll error $\Delta \phi(t)$ and rotates collective thrust.
   * **Trackmania ($\Delta f \equiv 0$):** Bitwise exact **$0.000000 \pm 0.000000\text{ m}$** across the entire lap at $d = 0$, vs. corner-locked divergence spikes ($0.054\text{ m}$ at $d=1$, $0.107\text{ m}$ at $d=2$, peaking $\sim 0.26\text{ m}$ in S-chicanes) under pure interface latency.

2. **H1 & H2 Confirmed — Action History Buffer Beats `previous_action` $a_{t-1}$ (`Exp 2`):**
   * **H1:** Drone Nominal collapses by **$-86.7\%$** under target dynamics mismatch ($53.93 \to 7.19$), while Trackmania Nominal transfers losslessly at $d=0$ ($40.16$), dips gently at $d=1\text{--}2$ ($38.14, 32.74$), and hits a **$-78.9\%$ pilot-induced oscillation (PIO) cliff at $d=4$** ($8.48 \pm 3.18$) once phase lag pushes tires past the Pacejka grip peak.
   * **H2 Action-Buffer Effect:** Replacing Swift's single `previous_action` ($a_{t-1}$) with a 4-step ring buffer $a_{t-4:t-1}$ improves Drone $d=4$ latency-randomized transfer by **$+65.6\%$** ($10.12 \to \mathbf{16.75 \pm 5.20}$; gate pass rate $16.2\% \to 26.4\%$) and retains **$92.2\%$** of $d=0$ Trackmania score at $d=4$ ($37.02 \pm 0.59$) with **$1.9\times$ lower seed variance** than $a_{t-1}$-only ($36.14 \pm 1.14$).

3. **H3 Confirmed + New Nuance #1: Drone VIO Velocity Observability Cliff (`Exp 3`):**
   * **OOD Observer Drift Confirmed:** Detached linear probes on the GRU hidden state show that distilling a student on fixed nominal source dynamics (`GRU_FixedSrc`) causes target velocity/slip estimation error to jump **$1.89\times$ on Drone** ($1.17 \to 2.21\text{ m/s}$) and **$2.57\times$ on Trackmania** ($0.87 \to 2.23\text{ m/s}$).
   * **Trackmania RMA Recovery:** Distilling over randomized transitions (`GRU_RandSrc`) jumps Trackmania target score ($d=2$, $12\%$ ZOH dropout, camera bias) from $11.63$ (FF) and $22.76$ (`GRU_FixedSrc`) to **$38.48 \pm 0.52$ ($97.8\%$ of Exact-State Teacher $39.33$!)**, cutting hidden-state slip probe RMSE **$3.7\times$** to $0.60\text{ m/s}$.
   * **NEW NUANCE #1 (Keep Noisy VIO Velocity on Drone!):** When *all* 3D linear velocities $(v_x, v_y, v_z)$ are completely deleted on Drone alongside unknown thrust/mass mismatch + uncalibrated PnP bias, velocity and pose bias become jointly unobservable over short gate horizons (`GRU_RandSrc` cuts probe error $-24.8\%$, yet $1.66\text{ m/s}$ error still clips $0.32\text{ m}$ gates). **Rule for `gym-pybullet-drones` / `MuJoCo`:** Match physical Swift [1] by feeding **noisy/drifting VIO velocity + biased PnP gate corners** in the sensor-limited student rather than zeroing velocity out completely!

4. **H4 Confirmed + New Nuance #2: Delay Alignment & Small-Sample `k-NN` Dimensionality (`Exp 4`):**
   * **Trackmania Delay-Confounding Proof ($d = 2$, 6 Target Laps):**
     * 1-lap System-ID scan $\hat{d} = \arg\min_{d'} \text{RMS}(\dot{v}_{\text{meas}} - f_S(s_t, a_{t-d'}))$ recovers exact $\hat{d} = 2$ on **5/5 seeds** ($0.240\text{ m/s}^2$ vs. $2.160\text{ m/s}^2$ at $d'=0$).
     * Naive unaligned `k-NN` learns a massive phantom tire residual (**$1.546 \pm 0.169\text{ m/s}^2$**), whereas **delay-aligned `k-NN` collapses $26.8\times$ to $\mathbf{0.058 \pm 0.008\text{ m/s}^2} \approx 0$** and lifts target fine-tuning score to **$39.24 \pm 0.37$ ($99.5\%$ of Oracle $39.42$)**!
   * **Drone Residual Fine-Tuning Beats Domain Randomization:** At equal source budget + 6 target flights (~20s), body-frame $7\text{D}$ `k-NN` cuts held-out 3D acceleration RMSE by **$-56.1\%$** ($2.89 \to 1.27\text{ m/s}^2$) and achieves **$20.84 \pm 3.16$**, beating `Nominal` ($8.06 \pm 3.07$, $+158\%$) and `Domain Randomization` ($18.62 \pm 8.53$, with **$2.7\times$ lower seed variance**).
   * **NEW NUANCE #2 (Parametric $\hat{\tau}_m$ + Low-D `k-NN` Beats High-D History `k-NN`):** At tiny target sample budgets ($N \approx 450\text{--}1000$), concatenating raw action history directly into `k-NN` ($7\text{D} \to 9\text{D}$) slightly dilutes neighbor density ($\mathcal{O}(N^{-2/(4+D)})$: RMSE $1.265 \to 1.297\text{ m/s}^2$, score $20.84 \to 17.55$). **Best practice:** Fit scalar motor lag $\hat{\tau}_m$ parametrically during the Step 0 scripted replay (just like $\hat{d}$!) and keep nonparametric `k-NN` strictly low-dimensional ($\le 7\text{D}$ body-frame)!

---

## 2. Reproducing the Verification Suite

```bash
MPLCONFIGDIR=/tmp/mpl_cache "../hw3/.venv/bin/python3" theory_sim/run_all_experiments.py
# Or using the new Final Project/code pixi environment:
pixi run --manifest-path code/pixi.toml python theory_sim/run_all_experiments.py
```
