#!/usr/bin/env bash
set -euo pipefail

# Build and run the independent retained-headroom parameter probe against the
# exact OpenFHE source revision used by CROSS.

readonly EXPECTED_COMMIT="1306d14f8c26bb6150d3e6ad54f28dfe1007689e"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROBE_SOURCE="${SCRIPT_DIR}/openfhe_151_cd2_security_probe.cpp"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

fail() {
  echo "$*" >&2
  exit 2
}

OPENFHE_SOURCE_DIR="${OPENFHE_SOURCE_DIR:-${1:-}}"
if [[ -z "${OPENFHE_SOURCE_DIR}" ]]; then
  fail "usage: $0 /path/to/openfhe-v1.5.1 [/path/to/openfhe-build]"
fi
if [[ ! -d "${OPENFHE_SOURCE_DIR}" ]]; then
  fail "OpenFHE source directory does not exist: ${OPENFHE_SOURCE_DIR}"
fi
OPENFHE_SOURCE_DIR="$(cd -- "${OPENFHE_SOURCE_DIR}" && pwd -P)"
OPENFHE_BUILD_DIR="${OPENFHE_BUILD_DIR:-${2:-${OPENFHE_SOURCE_DIR}/build-cross-reference}}"
PROBE_BINARY="${PROBE_BINARY:-${OPENFHE_BUILD_DIR}/openfhe_151_cd2_security_probe}"

if [[ "$(git -C "${OPENFHE_SOURCE_DIR}" rev-parse --is-inside-work-tree)" != "true" ]]; then
  fail "OpenFHE source directory is not a Git worktree: ${OPENFHE_SOURCE_DIR}"
fi
source_toplevel="$(git -C "${OPENFHE_SOURCE_DIR}" rev-parse --show-toplevel)"
source_toplevel="$(cd -- "${source_toplevel}" && pwd -P)"
if [[ "${source_toplevel}" != "${OPENFHE_SOURCE_DIR}" ]]; then
  fail "OpenFHE source directory must be the worktree root: ${source_toplevel}"
fi

actual_commit="$(git -C "${OPENFHE_SOURCE_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  fail "expected OpenFHE commit ${EXPECTED_COMMIT}, got ${actual_commit}"
fi

worktree_status="$(
  git -C "${OPENFHE_SOURCE_DIR}" status \
    --porcelain=v1 \
    --untracked-files=all \
    --ignore-submodules=none
)"
if [[ -n "${worktree_status}" ]]; then
  echo "OpenFHE source tree must be clean; found:" >&2
  echo "${worktree_status}" >&2
  exit 2
fi

submodule_status="$(git -C "${OPENFHE_SOURCE_DIR}" submodule status --recursive)"
bad_submodules=""
while IFS= read -r status_line; do
  if [[ -n "${status_line}" && "${status_line:0:1}" != " " ]]; then
    bad_submodules+="${status_line}"$'\n'
  fi
done <<< "${submodule_status}"
if [[ -n "${bad_submodules}" ]]; then
  echo "OpenFHE recursive submodules must be initialized at pinned commits:" >&2
  printf '%s' "${bad_submodules}" >&2
  exit 2
fi

cmake \
  -S "${OPENFHE_SOURCE_DIR}" \
  -B "${OPENFHE_BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED=ON \
  -DBUILD_STATIC=OFF \
  -DBUILD_UNITTESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DWITH_OPENMP=OFF \
  -DGIT_SUBMOD_AUTO=OFF
cmake --build "${OPENFHE_BUILD_DIR}" --target OPENFHEpke --parallel

"${CXX:-c++}" \
  -std=c++17 \
  -O2 \
  -DOPENFHE_VERSION=1.5.1 \
  -DMATHBACKEND=4 \
  -I"${OPENFHE_SOURCE_DIR}/third-party/include" \
  -I"${OPENFHE_SOURCE_DIR}/third-party/cereal/include" \
  -I"${OPENFHE_SOURCE_DIR}/src/core/include" \
  -I"${OPENFHE_SOURCE_DIR}/src/binfhe/include" \
  -I"${OPENFHE_BUILD_DIR}/src/core" \
  -I"${OPENFHE_SOURCE_DIR}/src/core/lib" \
  -I"${OPENFHE_SOURCE_DIR}/src/pke/include" \
  -I"${OPENFHE_SOURCE_DIR}/src/pke/lib" \
  "${PROBE_SOURCE}" \
  -L"${OPENFHE_BUILD_DIR}/lib" \
  -Wl,-rpath,"${OPENFHE_BUILD_DIR}/lib" \
  -lOPENFHEpke \
  -lOPENFHEcore \
  -o "${PROBE_BINARY}"

probe_output="$(mktemp "${TMPDIR:-/tmp}/openfhe-151-cd2-probe.XXXXXX")"
trap 'rm -f "${probe_output}"' EXIT

"${PROBE_BINARY}" | tee "${probe_output}"
(
  cd -- "${REPOSITORY_ROOT}"
  "${PYTHON:-python3}" -m jaxite_word.he_params \
    --verify-openfhe "${probe_output}"
)
