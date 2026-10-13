#!/bin/sh
# Build this fork's wheels for the ring: mlx-metal once, then mlx per Python.
#
#   ring/build_wheels.sh OUTDIR PYTHON [PYTHON ...]
#
# The tag at HEAD names the local version (v0.32.3+ring.2 -> 0.32.3+ring.2).
# Each PYTHON must be an arm64 interpreter: a universal2 one tags the wheels
# universal2 although they hold arm64 code only.
set -eu
cd "$(dirname "$0")/.."
tag=$(git describe --tags --exact-match)
out=${1:?usage: ring/build_wheels.sh OUTDIR PYTHON [PYTHON ...]}
shift
[ $# -gt 0 ] || { echo "usage: ring/build_wheels.sh OUTDIR PYTHON [PYTHON ...]" >&2; exit 2; }
for python in "$@"; do
  "$python" -c 'import sys, sysconfig; sys.exit(not sysconfig.get_platform().endswith("arm64"))' ||
    { echo "$python is not an arm64 interpreter" >&2; exit 2; }
done

# libmlx gets the JACCL backend only for macOS 26.2 and later
# (mlx/distributed/jaccl/CMakeLists.txt), as in upstream's releases.
export MACOSX_DEPLOYMENT_TARGET=26.2
export MLX_LOCAL_VERSION="${tag#*+}"
export CMAKE_ARGS="-G Ninja -DMLX_BUILD_METAL=ON -DCMAKE_COMPILE_WARNING_AS_ERROR=ON"
mkdir -p "$out"
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

build() {  # PYTHON BACKEND|FRONTEND
  venv="$scratch/venv-$2-$(basename "$1")"
  uv venv -q --python "$1" "$venv"
  uv pip install -q --python "$venv/bin/python" build setuptools
  "$venv/bin/python" setup.py clean --all >/dev/null
  rm -rf dist
  env "MLX_BUILD_${2}_PACKAGE=1" "$venv/bin/python" -m build -w
  mv dist/*.whl "$out/"
}

build "$1" BACKEND
for python in "$@"; do
  build "$python" FRONTEND
done

# Without JACCL the build loads but mx.distributed.init(backend="jaccl") fails.
unzip -o -q "$out"/mlx_metal-*.whl "mlx/lib/libmlx.dylib" -d "$scratch/check"
otool -L "$scratch/check/mlx/lib/libmlx.dylib" | grep -q libjaccl.dylib ||
  { echo "libmlx does not link JACCL" >&2; exit 1; }
