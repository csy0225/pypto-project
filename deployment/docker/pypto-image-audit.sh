#!/usr/bin/env bash
# Immutable identity/capability audit for a built vllm-pypto image.
set -euo pipefail

check_repo() {
  local name=$1
  local path=$2
  local expected=$3
  local actual
  local dirty

  actual=$(git -C "$path" rev-parse HEAD)
  [ "$actual" = "$expected" ] || {
    echo "[audit] $name pin mismatch: expected=$expected actual=$actual" >&2
    return 1
  }
  dirty=$(git -C "$path" status --porcelain | wc -l)
  [ "$dirty" = 0 ] || {
    echo "[audit] $name worktree is dirty: entries=$dirty" >&2
    return 1
  }
  printf '[audit] pin %-10s %s clean\n' "$name" "$actual"
}

: "${PYPTO_IMAGE_PYPTO_COMMIT:?missing image pypto pin}"
: "${PYPTO_IMAGE_PYPTO_LIB_COMMIT:?missing image pypto-lib pin}"
: "${PYPTO_IMAGE_PTO_ISA_COMMIT:?missing image pto-isa pin}"
: "${PYPTO_IMAGE_PTOAS_COMMIT:?missing image PTOAS pin}"
: "${PYPTO_IMAGE_SIMPLER_COMMIT:?missing image simpler pin}"

check_repo pypto /workspace/pypto "$PYPTO_IMAGE_PYPTO_COMMIT"
check_repo pypto-lib /workspace/pypto-lib "$PYPTO_IMAGE_PYPTO_LIB_COMMIT"
check_repo pto-isa /workspace/pto-isa "$PYPTO_IMAGE_PTO_ISA_COMMIT"
check_repo PTOAS /workspace/PTOAS "$PYPTO_IMAGE_PTOAS_COMMIT"
check_repo simpler /workspace/pypto/runtime "$PYPTO_IMAGE_SIMPLER_COMMIT"

for repo in \
  /workspace/pypto \
  /workspace/pypto-lib \
  /workspace/pto-isa \
  /workspace/PTOAS \
  /workspace/pypto/runtime; do
  if git -C "$repo" config --get-regexp \
    '^(remote\..*\.url|submodule\..*\.url)$' 2>/dev/null \
    | grep -Eqs 'oauth2:|github_pat_|ghp_'; then
    echo "[audit] credential-bearing Git URL remains in $repo" >&2
    exit 1
  fi
done
echo "[audit] git credential scrub: PASS"

python - <<'PY'
import os

from models.step3p5 import config
from pypto.runtime.runner import RunConfig

expected_profile = os.environ.get(
    "PYPTO_STEP3P5_ATTN_TASK_PROFILE",
    "portable",
)
assert config.ATTN_TASK_PROFILE == expected_profile, (
    config.ATTN_TASK_PROFILE,
    expected_profile,
)
required = (
    os.environ.get("PYPTO_REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN", "0")
    == "1"
)
available = (
    "l2_swimlane_reuse_dep_gen"
    in RunConfig.__dataclass_fields__
)
assert available or not required, (
    "required l2_swimlane_reuse_dep_gen is unavailable"
)
print("[audit] attention profile:", config.ATTN_TASK_PROFILE)
print("[audit] prepared swimlane reuse capability:", available)
PY

ptoas --version
echo "[audit] build jobs: ${PYPTO_IMAGE_BUILD_JOBS:-unknown} (resource only)"
echo "IMAGE_IMMUTABLE_AUDIT=PASS"
