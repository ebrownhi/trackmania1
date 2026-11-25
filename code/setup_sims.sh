#!/usr/bin/env bash
# Reproducible install of the third-party simulators into this pixi env (osx-arm64).
#   pixi run --manifest-path code/pixi.toml setup-sims
#
# Why this is a script and not plain pixi dependencies:
#   - l2f has no osx-arm64 wheel, and building it with the pixi Python fails because the project path contains
#     spaces ("Robot Learning/Final Project") and Python's baked-in CFLAGS are unquoted. We build the wheel with
#     a Python whose prefix has no spaces (same cp312 ABI) and install it into the pixi env.
#   - gym-pybullet-drones pins numpy>=2.5 / pybullet>=3.2.7, which would force a pybullet source build that fails
#     on Apple Clang 21. We install it --no-deps on top of the conda-forge pybullet/numpy, which work fine.
set -euo pipefail

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT="$CODE/external"
PY="$CODE/.pixi/envs/default/bin/python"
BUILD_PY="${L2F_BUILD_PYTHON:-/opt/homebrew/bin/python3.12}"
WHEEL_DIR="${TMPDIR:-/tmp}/l2f_wheel"

# repo dir | url | pinned commit
REPOS=(
  "l2f|https://github.com/rl-tools/l2f.git|3c06519920394af8d8505df99881b639de8e13ac"
  "gym-pybullet-drones|https://github.com/learnsyslab/gym-pybullet-drones.git|781e4340fd1a83d643ae8f874b9ec0172d0a99d3"
  "TMNF-C|https://github.com/adonis-singh/TMNF-C.git|eb6be045970e1f490aa02b92b2b39c560782503a"
)
RL_TOOLS_COMMIT=48488c58df20d9a62f53faa8d8bc24f20590e8e9
JSON_URL=https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp

mkdir -p "$EXT"
for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url commit <<<"$entry"
  if [ ! -d "$EXT/$name/.git" ]; then
    echo ">> cloning $name"
    git clone --quiet "$url" "$EXT/$name"
    git -C "$EXT/$name" checkout --quiet "$commit"
  fi
done

if [ ! -d "$EXT/l2f/external/rl-tools/include" ]; then
  git -C "$EXT/l2f" submodule update --init external/rl-tools
  git -C "$EXT/l2f/external/rl-tools" checkout --quiet "$RL_TOOLS_COMMIT"
fi
if [ ! -f "$EXT/l2f/external/json/nlohmann/json.hpp" ]; then
  mkdir -p "$EXT/l2f/external/json/nlohmann"
  curl -fsSL -o "$EXT/l2f/external/json/nlohmann/json.hpp" "$JSON_URL"
fi

# TMNF-C macOS port (Linux-only futex / CPU-affinity code guarded); skip if already applied
if git -C "$EXT/TMNF-C" apply --check "$CODE/patches/tmnf-c-macos.patch" 2>/dev/null; then
  git -C "$EXT/TMNF-C" apply "$CODE/patches/tmnf-c-macos.patch"
fi

if ! "$PY" -c "import l2f" 2>/dev/null; then
  echo ">> building l2f wheel with $BUILD_PY"
  rm -rf "$WHEEL_DIR" && mkdir -p "$WHEEL_DIR"
  (cd "$WHEEL_DIR" && "$BUILD_PY" -m pip wheel --quiet --no-deps -w "$WHEEL_DIR" "$EXT/l2f")
  "$PY" -m pip install --quiet --no-deps "$WHEEL_DIR"/l2f-*.whl
fi

"$PY" -m pip install --quiet --no-deps -e "$EXT/gym-pybullet-drones"
"$PY" -m pip install --quiet transforms3d==0.4.2 pettingzoo==1.27.0 stable-baselines3==2.9.2 control==0.10.2 "numpy<2.2"

(cd "$CODE" && "$PY" -c "import l2f, gym_pybullet_drones, stable_baselines3; print('setup-sims OK')")
