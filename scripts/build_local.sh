#!/usr/bin/env bash

# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build Knowhere locally inside a self-contained Python virtualenv so it can be
# opened / debugged in CLion. All Conan tooling is installed into the venv, so
# nothing pollutes your global Python environment.
#
# Usage:
#   scripts/build_local.sh [--debug|--release] [--no-ut] [--clean]
#                          [--diskann] [--configure-only]
#
# Common examples:
#   scripts/build_local.sh                 # Debug build with unit tests
#   scripts/build_local.sh --release       # optimized build
#   scripts/build_local.sh --clean         # wipe venv + build/ first

set -euo pipefail

# --------------------------- configuration ---------------------------------
BUILD_TYPE="Debug"
WITH_UT="True"
WITH_DISKANN="False"
DO_CLEAN=0
CONFIGURE_ONLY=0

CONAN_VERSION="1.61.0"
CONAN_REMOTE_NAME="default-conan-local"
CONAN_REMOTE_URL="https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local"

# --------------------------- arg parsing -----------------------------------
usage() {
    grep '^#' "$0" | sed -E 's/^# ?//' | awk 'NR>15'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)          BUILD_TYPE="Debug" ;;
        --release)        BUILD_TYPE="Release" ;;
        --no-ut)          WITH_UT="False" ;;
        --diskann)        WITH_DISKANN="True" ;;
        --clean)          DO_CLEAN=1 ;;
        --configure-only) CONFIGURE_ONLY=1 ;;
        -h|--help)        usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
    shift
done

# --------------------------- paths / platform ------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv-knowhere"
BUILD_DIR="${REPO_ROOT}/build"

cd "${REPO_ROOT}"

case "$(uname -s)" in
    Darwin*) IS_MAC=1; LIBCXX="libc++" ;;
    *)       IS_MAC=0; LIBCXX="libstdc++11" ;;
esac

# macOS toolchain setup. This mirrors how milvus (internal/core/build.sh) and
# knowhere (scripts/python_deps.sh) build on macOS: use Homebrew LLVM/Clang
# (NOT Apple clang) plus Homebrew libomp for OpenMP. Apple clang ships no
# OpenMP runtime, and its very new releases also trip other issues; the tested
# path is Homebrew LLVM.
OPENMP_ARGS=()     # extra -D flags so find_package(OpenMP) locates libomp
CONAN_EXTRA_ARGS=()  # macOS-only -s/-o overrides for conan install
if [[ "${IS_MAC}" -eq 1 ]]; then
    # Detect the newest installed Homebrew LLVM, same order as milvus
    # scripts/setenv.sh (17 -> 16 -> 15 -> 14). Conan 1.61 accepts these.
    LLVM_VER=""
    for v in 17 16 15 14; do
        if brew ls --versions "llvm@${v}" >/dev/null 2>&1; then LLVM_VER="${v}"; break; fi
    done
    if [[ -z "${LLVM_VER}" ]]; then
        echo "==> No Homebrew llvm@{17,16,15,14} found; installing llvm@17"
        brew install llvm@17
        LLVM_VER="17"
    fi
    LLVM_PREFIX="$(brew --prefix "llvm@${LLVM_VER}")"

    if ! brew list libomp >/dev/null 2>&1; then
        echo "==> Installing libomp via Homebrew (required for OpenMP)"
        brew install libomp
    fi
    OMP_PREFIX="$(brew --prefix libomp)"

    # Use Homebrew clang + libomp, matching milvus scripts/setenv.sh exactly.
    export CC="${LLVM_PREFIX}/bin/clang"
    export CXX="${LLVM_PREFIX}/bin/clang++"
    export ASM="${LLVM_PREFIX}/bin/clang"
    export CFLAGS="-Wno-deprecated-declarations -I${OMP_PREFIX}/include ${CFLAGS:-}"
    export CXXFLAGS="${CFLAGS}"
    export LDFLAGS="-L${OMP_PREFIX}/lib ${LDFLAGS:-}"

    # -s compiler=clang <LLVM_VER>: the default profile detects apple-clang,
    #   which would mismatch the Homebrew CC/CXX we just exported.
    # -o abseil:shared=True: abseil's static archives are built in GNU `ar`
    #   format (with a '/' symbol-table member) that Apple's ld cannot link
    #   ("archive member '/' not a mach-o file"). Building abseil shared avoids
    #   this; this is exactly what milvus does on macOS (internal/core/conanfile.py).
    CONAN_EXTRA_ARGS=(
        -s compiler=clang -s compiler.version="${LLVM_VER}"
        -o abseil:shared=True
    )

    OPENMP_ARGS=(
        "-DOpenMP_C_FLAGS=-Xclang -fopenmp -I${OMP_PREFIX}/include"
        "-DOpenMP_C_LIB_NAMES=omp"
        "-DOpenMP_CXX_FLAGS=-Xclang -fopenmp -I${OMP_PREFIX}/include"
        "-DOpenMP_CXX_LIB_NAMES=omp"
        "-DOpenMP_omp_LIBRARY=${OMP_PREFIX}/lib/libomp.dylib"
    )
fi

echo "==> Knowhere local build"
echo "    repo root : ${REPO_ROOT}"
echo "    build type: ${BUILD_TYPE}"
echo "    with_ut   : ${WITH_UT}"
echo "    libcxx    : ${LIBCXX}"
if [[ "${IS_MAC}" -eq 1 ]]; then
    echo "    compiler  : Homebrew llvm@${LLVM_VER} (${CC})"
fi

# --------------------------- clean -----------------------------------------
if [[ "${DO_CLEAN}" -eq 1 ]]; then
    echo "==> Cleaning venv and build directories"
    rm -rf "${VENV_DIR}" "${BUILD_DIR}"
fi

# --------------------------- pick a compatible python ----------------------
# Conan 1.61 imports the stdlib `imp` module, which was removed in Python 3.12,
# so the venv MUST be built with Python <= 3.11. Allow override via $PYTHON.
find_python() {
    if [[ -n "${PYTHON:-}" ]] && "${PYTHON}" -c 'import sys; exit(0 if sys.version_info[:2] <= (3,11) else 1)' 2>/dev/null; then
        echo "${PYTHON}"; return 0
    fi
    local candidates=(python3.11 python3.10 python3.9 python3.8)
    if [[ "${IS_MAC}" -eq 1 ]]; then
        for v in 3.11 3.10 3.9; do
            candidates+=("/opt/homebrew/opt/python@${v}/bin/python${v}" "/usr/local/opt/python@${v}/bin/python${v}")
        done
    fi
    for py in "${candidates[@]}"; do
        if command -v "${py}" >/dev/null 2>&1 && "${py}" -c 'import sys; exit(0 if sys.version_info[:2] <= (3,11) else 1)' 2>/dev/null; then
            echo "${py}"; return 0
        fi
    done
    return 1
}

PYTHON_BIN="$(find_python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ "${IS_MAC}" -eq 1 ]]; then
        echo "==> No Python <=3.11 found; installing python@3.11 via Homebrew"
        brew install python@3.11
        PYTHON_BIN="$(find_python || true)"
    fi
fi
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "ERROR: Conan 1.61 needs Python <=3.11 (the 'imp' module was removed in 3.12)." >&2
    echo "       Install one (e.g. 'brew install python@3.11') or set PYTHON=/path/to/python3.11." >&2
    exit 1
fi
echo "    using python: ${PYTHON_BIN} ($("${PYTHON_BIN}" --version 2>&1))"

# --------------------------- virtualenv ------------------------------------
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "==> Creating virtualenv at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --quiet --upgrade pip

if ! conan --version 2>/dev/null | grep -q "${CONAN_VERSION}"; then
    echo "==> Installing conan ${CONAN_VERSION} into venv"
    python -m pip install --quiet "conan==${CONAN_VERSION}"
fi
echo "    $(conan --version)"

# Pin CMake < 4 inside the venv. CMake 4.x de-duplicates the repeated
# "-Xarch_x86_64" token that abseil emits for its randen HWAES flags, which
# leaves "-msse4.1" unguarded and makes it get applied to the arm64 target
# (error: unsupported option '-msse4.1' for target 'arm64-...'). CMake 3.x
# keeps both tokens, so the flag stays correctly gated to x86_64.
if ! cmake --version 2>/dev/null | grep -qE 'version 3\.'; then
    echo "==> Installing cmake<4 into venv (avoids abseil -msse4.1/arm64 build error)"
    python -m pip install --quiet "cmake<4"
fi
echo "    using cmake: $(command -v cmake) ($(cmake --version | head -1))"

# --------------------------- conan profile ---------------------------------
# A fresh conan (in a clean venv) has no default profile; create one.
if ! conan profile show default >/dev/null 2>&1; then
    echo "==> Creating default conan profile"
    conan profile new default --detect --force
fi

# --------------------------- conan remote ----------------------------------
if ! conan remote list | grep -q "${CONAN_REMOTE_NAME}"; then
    echo "==> Adding conan remote ${CONAN_REMOTE_NAME}"
    conan remote add "${CONAN_REMOTE_NAME}" "${CONAN_REMOTE_URL}"
fi

# --------------------------- build -----------------------------------------
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "==> conan install (${BUILD_TYPE})"
conan install .. \
    --build=missing \
    -o with_ut="${WITH_UT}" \
    -o with_diskann="${WITH_DISKANN}" \
    "${CONAN_EXTRA_ARGS[@]}" \
    -s compiler.libcxx="${LIBCXX}" \
    -s build_type="${BUILD_TYPE}"

# Locate the conan-generated toolchain and derive the matching cmake build dir.
TOOLCHAIN="$(find "${BUILD_DIR}" -name conan_toolchain.cmake 2>/dev/null | head -n1 || true)"
if [[ -z "${TOOLCHAIN}" ]]; then
    echo "ERROR: conan_toolchain.cmake not found after conan install" >&2
    exit 1
fi
TC_DIR="$(dirname "${TOOLCHAIN}")"
if [[ "$(basename "${TC_DIR}")" == "generators" ]]; then
    CMAKE_BUILD_DIR="$(dirname "${TC_DIR}")"
else
    CMAKE_BUILD_DIR="${TC_DIR}"
fi

# We drive cmake directly (instead of `conan build`) so the OpenMP hint flags
# get applied. This mirrors exactly what CLion will do on reload.
echo "==> cmake configure (${CMAKE_BUILD_DIR})"
cmake -S "${REPO_ROOT}" -B "${CMAKE_BUILD_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DWITH_UT="${WITH_UT}" \
    -DWITH_DISKANN="${WITH_DISKANN}" \
    "${OPENMP_ARGS[@]}"

if [[ "${CONFIGURE_ONLY}" -eq 0 ]]; then
    echo "==> cmake build"
    cmake --build "${CMAKE_BUILD_DIR}" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
fi

# Build a single-line CMake options string for CLion, quoting the OpenMP
# flags (they contain spaces) exactly as CLion expects them.
CLION_OPTS="-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN} -DWITH_UT=${WITH_UT} -DWITH_DISKANN=${WITH_DISKANN}"
if [[ "${IS_MAC}" -eq 1 ]]; then
    CLION_OPTS="${CLION_OPTS} \"-DOpenMP_C_FLAGS=-Xclang -fopenmp -I${OMP_PREFIX}/include\" -DOpenMP_C_LIB_NAMES=omp \"-DOpenMP_CXX_FLAGS=-Xclang -fopenmp -I${OMP_PREFIX}/include\" -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_omp_LIBRARY=${OMP_PREFIX}/lib/libomp.dylib"
fi

MAC_TOOLCHAIN_HINT=""
if [[ "${IS_MAC}" -eq 1 ]]; then
    MAC_TOOLCHAIN_HINT="
CLion Toolchain (Settings > Build, Execution, Deployment > Toolchains):
  The deps above were built with Homebrew LLVM. For a matching build, add a
  toolchain whose C/C++ compilers point at:
    C compiler  : ${CC}
    C++ compiler: ${CXX}
  (Apple clang also works for knowhere itself since the prebuilt deps are
   libc++/ABI-compatible, as long as you pass the OpenMP flags below.)
"
fi

cat <<EOF

============================================================================
 Done.
============================================================================
Conan toolchain file (point CLion at this):
  ${TOOLCHAIN}
${MAC_TOOLCHAIN_HINT}
CLion setup (Settings > Build, Execution, Deployment > CMake > add profile):
  Build type   : ${BUILD_TYPE}
  Build dir    : ${CMAKE_BUILD_DIR}
  CMake options (paste the whole line, incl. the OpenMP flags):
    ${CLION_OPTS}

To reuse the venv in your shell:  source "${VENV_DIR}/bin/activate"
============================================================================
EOF
