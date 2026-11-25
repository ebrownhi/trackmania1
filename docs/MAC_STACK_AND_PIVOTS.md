# Mac (`arm64` M5 Pro) Stack, Execution Playbook & Pivot Matrix

**Machine Audited:** Apple M5 Pro (`arm64`, 15 CPU cores, 24 GB Unified Memory, macOS 26.6.2, Apple Clang 21.0.0)  
**Mandate:** Maximize fast, self-contained execution on macOS while maintaining clean, pre-registered pivot paths if cross-OS or C++ simulator plumbing hits friction.

---

## 1. Local Toolchain & Simulator Compatibility Summary

### 1.1 Installed vs. Needed System Tools
* **Already Installed:** Homebrew (`/opt/homebrew/bin/brew`), `python@3.12` (3.12.14), `python@3.14`, `ffmpeg`, `sdl2`/`sdl3`, `docker`/`colima`/`lima`, `uv 0.13.5` (`../hw3/.bootstrap/bin/uv`), and standalone `cpython-3.10.22` (`../hw3/.python/`).
* **One-Time Setup Needed:** `cmake`, `ninja`, and `pixi`:
  ```bash
  brew install cmake ninja pixi
  ```
* **Environment Isolation:** Do **not** touch `../hw3/.venv` (`torch==1.12.1`, `gym==0.25.1`). Initialize a clean Python 3.11 environment inside `Final Project/code/`:
  ```bash
  mkdir -p code && cd code
  pixi init . -c conda-forge
  pixi add python=3.11 cmake ninja pkg-config \
    pytorch numpy scipy scikit-learn gymnasium matplotlib tensorboard tqdm pytest \
    pybullet mujoco numba
  pixi run pip install l2f
  ```

### 1.2 Simulator-by-Simulator `osx-arm64` Audit

| Simulator | Role | `osx-arm64` Status | Key Engineering Notes |
| :--- | :--- | :--- | :--- |
| **`rl-tools/l2f`** | Drone Source | **100% Native** | Links `-framework Accelerate` on Apple Silicon automatically. Default action space is 4-motor RPM $[-1,1]^4$; wrap with a 500 Hz inner CTBR rate loop stepped $5\times$ per 100 Hz policy step. |
| **`learnsyslab/gym-pybullet-drones`** | Drone Target (Primary) | **Native via `conda-forge` (`pixi`)** | Avoid PyPI `pybullet` sdist on Clang 21 (fails with `fdopen` macro + pointer-type errors unless custom `CFLAGS` passed). Pass `physics=Physics.PYB_GND_DRAG_DW` (note exact spelling!), set `pyb_freq=500` (exact integer multiple of 100/50 Hz), add a 1st-order motor lag filter ($\tau_m > 0$, missing in vanilla `BaseAviary`), and wrap `ActionType.RPM` in a 500 Hz CTBR inner rate PID. |
| **`learnsyslab/crazyflow` / `mujoco`** | Drone Target (Pivot B) | **100% Native Wheels / `pixi`** | `pip install mujoco` ships official DeepMind `arm64` wheels. `crazyflow` provides **built-in 500 Hz `Control.body_rate` (CTBR)** firmware emulation on Crazyflie 2.x! |
| **`adonis-singh/TMNF-C`** | Trackmania Source | **100% Native C11 CPU** | Build with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTMNF_CUDA=OFF && cmake --build build -j12`. Synthetic `ctest` fixtures run **with zero game files**. Real Nadeo tracks (`A01`–`A03`) load from static binary snapshots exported once (~10 min) on a teammate's Windows PC. |
| **Live `TMNF` + `TMInterface`** | Trackmania Live Target | **Windows / x86 Wine Only** | Keep off Mac. **Crucial Lockstep Catch:** Default `TMInterface` `OnRunStep` pauses `TmForever.exe` synchronously at every tick, yielding $0.0$ gap vs. `TMNF-C` unless unlocked or perturbed! |
| **`f1tenth_gym`** | Ground Racing (Sub-Pivot A2) | **100% Native (`osx-arm64`)** | 7-state nonlinear Pacejka Magic Formula bicycle + 1080-beam LiDAR at 100 Hz with **zero proprietary game assets**. |

---

## 2. Recommended Mac-First Execution Strategy (Option 0: 95% on Mac)

1. **Drone Branch (100% on M5 Pro Mac):**
   * Train PPO, Domain/Latency Randomization, DAgger Recurrent Student, and Swift `k-NN` + `GP` Residuals in `L2F` (100 Hz CTBR $\to$ 500 Hz rate loop) and evaluate zero-shot + calibration rollouts in headless `gym-pybullet-drones` (`CF2X`).
2. **Trackmania Branch (95% on M5 Pro Mac + Two Short Windows Touchpoints):**
   * **Touchpoint 1 (Day 1–2, ~15 min on Windows):** Teammate runs `TMNF-C`'s snapshot exporter on `TmForever.exe` for `A01-Race`, `A02-Race`, and `A03-Race` (held-out) and commits the binary snapshots.
   * **All Daily Training & Evaluation on Mac:** Train against vectorized synchronous `TMNF-C` and evaluate against **`AsyncWallClockSocketServer`** (below) on Mac.
   * **Touchpoint 2 (Day 9+, Half-Day on Windows):** Validate exported `.pt` policies in live `TmForever.exe` + `TMInterface` (`Linesight` reference comparison + demo videos).

---

## 3. Asynchronous Wall-Clock Target Architecture (Option 0 Proxy & Pivot A)

Because synchronous lockstep `TMInterface` pauses the game engine at each 10 ms tick (erasing timing/latency gaps), our primary Mac Trackmania target—and official fallback target under Proposal Section 3—is a **two-process Real-Time Wall-Clock Socket Server**:

```text
┌──────────────────────────────────────────────┐        UDP / Unix Domain Socket         ┌──────────────────────────────────────────────┐
│  Process 1: Wall-Clock Physics Target Server │ ◄────── Action Packet (seq, u_cmd) ──── │  Process 2: RL Policy Client (PyTorch/MPS)   │
│  • Pinned monotonic 100 Hz loop (10.0 ms)    │                                         │  • Receives obs packet, runs MLP / GRU       │
│  • Steps 1 env of TMNF-C (or f1tenth_gym)    │ ──────► Obs Packet (t_wall, o_sensor) ─►│  • Real forward-pass latency + OS scheduler  │
│  • Zero-Order Hold (ZOH) on missed deadlines │        • Configurable 60 Hz decimation, │    jitter + configurable link delay (0-50ms) │
│  • Logs exact applied u_t & deadline slips   │          quantization, bias & dropout   │                                              │
└──────────────────────────────────────────────┘                                         └──────────────────────────────────────────────┘
```

* Physics inside Process 1 remains exact 100 Hz `TMNF-C` ($\Delta f_{\text{markov}} \equiv 0$), while the IPC boundary introduces **authentic real-time clock skew, non-integer inference latency, zero-order-hold tick slips, multi-rate sensor aliasing, and packet dropout**—cleanly testing **H1'–H4'** with 100% single-machine reproducibility.

---

## 4. Time-Boxed Pivot Triggers

| Gate | Component | Pass Criterion | Pivot Trigger & Action |
| :--- | :--- | :--- | :--- |
| **Day 1 (2h cap)** | `gym-pybullet-drones` on M5 Pro | Headless `CF2X` steps $>2,000\text{ Hz}$ and tracks step CTBR commands cleanly. | Build/aero snags $>2\text{h}$ $\implies$ **Pivot B:** Switch Drone target to **`learnsyslab/crazyflow`** (`Control.body_rate`, 500 Hz) or native **`mujoco`** (`bitcraze_crazyflie_2`). |
| **Day 1 (2h cap)** | `l2f` + `TMNF-C` builds | `l2f` steps vectorized envs in Python 3.11 and `TMNF-C` passes `ctest` synthetic suite. | `l2f` binding issue $\implies$ build `rl-tools/l2f` from source with `-framework Accelerate` (or use analytical PyTorch L2F ODE verified against `l2f`). |
| **Day 3 (48h cap)** | `TMNF-C` Track Snapshots | Windows teammate delivers `A01`–`A03` binary snapshots and Mac loads them in `TMNF-C`. | Snapshots delayed $>48\text{h}$ $\implies$ **Sub-Pivot A2:** Prototype ground branch immediately on **`f1tenth_gym`** + `AsyncWallClockSocketServer` so RL sweeps never block. |
| **Day 4 (Theory Gate)** | Scripted Divergence & Residual $R^2$ | Drone shows clear open-loop divergence + body-frame `k-NN` $R^2 > 0.6$; Ground shows $0.0$ sync gap, clear async gap, and $\|\hat{r}^{\text{aligned}}_{\text{dyn}}\| \approx 0$. | Zero gap or inverted sign $\implies$ enforce `AsyncWallClockSocketServer` on Ground and verify `FLU`/`NWU` axis sign unit tests on Drone before any PPO training. |
| **Day 9 (Live Gate)** | Windows `TMInterface` Eval | `.pt` checkpoints drive `A01`/`A02` in live `TmForever.exe` $>24\text{h}$ before report freeze. | Windows `TMInterface` flakes out $\implies$ **Pivot A:** Report Mac `AsyncWallClockSocketServer` (`TMNF-C`) as primary reproducible target per Proposal Sec. 3's contingency clause. |
