#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <root> <trusted-normal-seal-authority-sha256>" >&2
  exit 2
fi

ROOT=$1
AUTHORITY=${NORMAL_SEAL_AUTHORITY:-$ROOT/campaign/normal_seal_authority.json}
TRUSTED_SHA=$2
SCRIPTS=${SCRIPTS:-$ROOT/scripts}
VALIDATOR=$SCRIPTS/validate_five_layer_case.py
BATCHES=(1 2 4 7 8 16)

test -d "$ROOT/campaign"
test -s "$AUTHORITY"
test -s "$VALIDATOR"
[[ "$TRUSTED_SHA" =~ ^[0-9a-f]{64}$ ]]
test "$(sha256sum "$AUTHORITY" | awk '{print $1}')" = "$TRUSTED_SHA"

for source in baseline candidate; do
  for round in 1 2 3; do
    for batch in "${BATCHES[@]}"; do
      run="$ROOT/campaign/runs/${source}-r${round}-normal-bs${batch}-64k"
      python3 "$VALIDATOR" \
        --run "$run" \
        --seal-runtime-evidence \
        --seal-authority "$AUTHORITY" \
        --seal-authority-sha256 "$TRUSTED_SHA"
    done
  done
done

echo FIVE_LAYER_0162_NORMAL_RUNTIME_EVIDENCE_SEAL=PASS
