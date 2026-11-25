# One Method, Two Racing Worlds — Sim-to-Sim Transfer (CMU 16-831, Fall 2026)

Drone branch: `rl-tools/l2f` (source) → `gym-pybullet-drones` CF2X (target). Trackmania (`TMNF-C`) is paused.

## Setup (macOS arm64)
```bash
brew install pixi python@3.12          # python@3.12 is only used to build the l2f wheel
cd code
pixi install
pixi run setup-sims                   # clones pinned l2f / gym-pybullet-drones / TMNF-C, builds + installs
pixi run test                         # ~15 s regression suite
```

## Layout
- `code/sims/drone_pair.py` — shared CTBR rate loop, L2F + PyBullet backends, delay/dropout interface, expert, courses
- `real_sim/` — experiments on the real simulators (`expN_*.py`), outputs in `results/real_sim/`
- `theory_sim/` — earlier toy micro-simulators (see corrections in `docs/REAL_SIM_RESULTS.md`)
- `docs/` — theory, simulator audit, results, and `PLAN_OF_ACTION.md`
- `AGENTS.md` — guardrails and onboarding for coding agents
