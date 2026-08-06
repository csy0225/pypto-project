#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1

if [ "$#" -ne 4 ] && [ "$#" -ne 5 ]; then
  echo \
    "usage: $0 <root> <image@digest> <candidate-dfx-source> <candidate-route-source> [matched-policy.json]" \
    >&2
  exit 2
fi

ROOT=$1
IMG=$2
SOURCE=$3
ROUTE_SOURCE=$4
MATCHED_POLICY=${5:-}
SCRIPTS=${CAMPAIGN_SCRIPTS:-$ROOT/scripts}
NC=/mnt/persist/k8s-install/containerd/bin/nerdctl
ANALYSIS_NAME=${ROUTE_ANALYSIS_NAME:-dfx_analysis_route_v1}
SIDECAR_DIR=${ROUTE_SIDECAR_DIR:-route-sidecar}
REPORT_PREFIX=${ROUTE_REPORT_PREFIX:-dfx_campaign_route_v1}
REQUIRE_PUBLICATION_READY=${ROUTE_REQUIRE_PUBLICATION_READY:-1}
ROUTE_PROFILE=${ROUTE_PROFILE:-row16}
ROUTE_SOURCE_ROLE=reference
MATCHED_POLICY_SHA=
BATCHES=(1 2 4 7 8 16)

[[ "$ANALYSIS_NAME" =~ ^[A-Za-z0-9._-]+$ ]]
[[ "$SIDECAR_DIR" =~ ^[A-Za-z0-9._-]+$ ]]
[[ "$REPORT_PREFIX" =~ ^[A-Za-z0-9._-]+$ ]]
[[ "$ROUTE_PROFILE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "invalid ROUTE_PROFILE=$ROUTE_PROFILE" >&2
  exit 33
}
case "$REQUIRE_PUBLICATION_READY" in
  0|1) ;;
  *)
    echo \
      "invalid ROUTE_REQUIRE_PUBLICATION_READY=$REQUIRE_PUBLICATION_READY" \
      >&2
    exit 31
    ;;
esac

test -d "$ROOT"
test -d "$SOURCE"
test -d "$ROUTE_SOURCE"
test -d "$SCRIPTS"
test -s "$SOURCE/SOURCE_SHA256SUMS"
test -s "$ROUTE_SOURCE/SOURCE_SHA256SUMS"
test -s "$SCRIPTS/SCRIPTS_SHA256SUMS"
test -x "$SCRIPTS/container_reanalyze_dfx_with_route.sh"
test -s "$SCRIPTS/analyze_dfx_campaign.py"
test -s "$SCRIPTS/audit_route_dfx_source_compatibility.py"
test -s "$SCRIPTS/verify_exact_tree_manifest.py"
python3 "$SCRIPTS/verify_exact_tree_manifest.py" --root "$SOURCE"
python3 "$SCRIPTS/verify_exact_tree_manifest.py" --root "$ROUTE_SOURCE"
python3 "$SCRIPTS/verify_exact_tree_manifest.py" \
  --root "$SCRIPTS" \
  --manifest SCRIPTS_SHA256SUMS \
  --symlink-manifest -

if [ -n "$MATCHED_POLICY" ]; then
  test -s "$MATCHED_POLICY"
  MATCHED_POLICY=$(realpath "$MATCHED_POLICY")
  mapfile -t POLICY_FIELDS < <(
    PYTHONPATH="$SCRIPTS" python3 - "$MATCHED_POLICY" <<'PY'
import sys

from validate_five_layer_case import load_matched_dfx_policy

policy = load_matched_dfx_policy(sys.argv[1])
candidate = policy["sources"]["candidate"]
print(policy["policy_sha256"])
print(policy["authority"]["image_ref"])
print(candidate["profile"])
print(candidate["source_role"])
print(candidate["source_manifest_sha256"])
print(candidate["decode_fwd_sha256"])
PY
  )
  test "${#POLICY_FIELDS[@]}" -eq 6
  MATCHED_POLICY_SHA=${POLICY_FIELDS[0]}
  test "${POLICY_FIELDS[1]}" = "$IMG"
  ROUTE_PROFILE=${POLICY_FIELDS[2]}
  ROUTE_SOURCE_ROLE=${POLICY_FIELDS[3]}
  test \
    "$(sha256sum "$SOURCE/SOURCE_SHA256SUMS" | awk '{print $1}')" \
    = "${POLICY_FIELDS[4]}"
  test \
    "$(sha256sum "$SOURCE/models/step3p5/decode_fwd.py" | awk '{print $1}')" \
    = "${POLICY_FIELDS[5]}"
  test \
    "$(sha256sum "$ROUTE_SOURCE/models/step3p5/decode_fwd.py" | awk '{print $1}')" \
    = "${POLICY_FIELDS[5]}"
else
  case "$ROUTE_PROFILE" in
    row16) ROUTE_SOURCE_ROLE=reference ;;
    shared-split) ROUTE_SOURCE_ROLE=candidate ;;
    *)
      echo "generic ROUTE_PROFILE requires a matched policy" >&2
      exit 34
      ;;
  esac
fi

COMPATIBILITY=$ROOT/campaign/${REPORT_PREFIX}_source_compatibility.json
python3 "$SCRIPTS/audit_route_dfx_source_compatibility.py" \
  --campaign "$ROOT/campaign" \
  --dfx-source "$SOURCE" \
  --route-source "$ROUTE_SOURCE" \
  --sidecar-dir "$SIDECAR_DIR" \
  --image-ref "$IMG" \
  --round 1 \
  --profile "$ROUTE_PROFILE" \
  --source-role "$ROUTE_SOURCE_ROLE" \
  --out "$COMPATIBILITY"
COMPATIBILITY_SHA=$(sha256sum "$COMPATIBILITY" | awk '{print $1}')

for batch in "${BATCHES[@]}"; do
  DFX_RUN=$ROOT/campaign/runs/candidate-r1-dfx-bs${batch}-64k
  ROUTE_RUN=$ROOT/campaign/$SIDECAR_DIR/candidate-route-bs${batch}-64k
  OUT=$DFX_RUN/runtime/bs${batch}/$ANALYSIS_NAME
  VALIDATION=$ROUTE_RUN/runtime/route_artifact_validation.json
  test -d "$DFX_RUN"
  test -d "$ROUTE_RUN/runtime"
  test -s "$ROUTE_RUN/runtime/recv_meta_sidecar.pt"
  python3 -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] is True' \
    "$VALIDATION"
  python3 - "$VALIDATION" "$ROUTE_RUN/runtime/recv_meta_sidecar.pt" \
    "$ROUTE_PROFILE" "$ROUTE_SOURCE_ROLE" <<'PY'
import hashlib
import json
import sys

validation_path, sidecar_path, expected_profile, expected_role = sys.argv[1:]
value = json.loads(open(validation_path, encoding="utf-8").read())
assert value.get("profile") == expected_profile, value.get("profile")
assert value.get("source_role") == expected_role
assert value.get("artifacts", {}).get("recv_meta_sidecar.pt") == (
    hashlib.sha256(open(sidecar_path, "rb").read()).hexdigest()
)
PY
  VALIDATION_SHA=$(sha256sum "$VALIDATION" | awk '{print $1}')
  if [ -e "$OUT" ]; then
    test "$(cat "$OUT/route_validation_sha256.txt")" = "$VALIDATION_SHA"
    test \
      "$(cat "$OUT/source_compatibility_sha256.txt")" \
      = "$COMPATIBILITY_SHA"
    test -s "$OUT/analysis/moe_dfx_report.json"
    test -s "$OUT/analysis/moe_critical_path_report.md"
    echo "[route-aware-dfx] verified existing BS$batch"
    continue
  fi
  sudo -n install -d \
    -o "$(id -u)" \
    -g "$(id -g)" \
    -m 0755 \
    "$OUT"
  printf '%s\n' "$VALIDATION_SHA" > "$OUT/route_validation_sha256.txt"
  printf '%s\n' \
    "$COMPATIBILITY_SHA" \
    > "$OUT/source_compatibility_sha256.txt"
  set +e
  sudo -n "$NC" run --rm --net host \
    --security-opt apparmor=unconfined \
    --env ROUTE_ANALYSIS_BATCH="$batch" \
    --env ROUTE_ANALYSIS_PROFILE="$ROUTE_PROFILE" \
    -v "$SOURCE":/workspace/pypto-lib:ro \
    -v "$SCRIPTS":/campaign-scripts:ro \
    -v "$DFX_RUN":/dfx-run:ro \
    -v "$ROUTE_RUN/runtime":/route:ro \
    -v "$OUT":/out \
    "$IMG" \
    bash /campaign-scripts/container_reanalyze_dfx_with_route.sh \
    2>&1 | tee "$OUT/container.log"
  RC=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$RC" > "$OUT/container.rc"
  if [ "$RC" -ne 0 ]; then
    exit "$RC"
  fi
  test -s "$OUT/analysis/moe_dfx_report.json"
  echo "[route-aware-dfx] pass BS$batch"
done

AGGREGATE_ARGS=(
  --campaign "$ROOT/campaign"
  --round 1
  --candidate-analysis-dir "$ANALYSIS_NAME/analysis"
  --report-prefix "$REPORT_PREFIX"
  --source-compatibility "$COMPATIBILITY"
)
if [ -n "$MATCHED_POLICY" ]; then
  test "$(sha256sum "$MATCHED_POLICY" | awk '{print $1}')" = \
    "$MATCHED_POLICY_SHA"
  AGGREGATE_ARGS+=(--matched-policy "$MATCHED_POLICY")
fi
if [ "$REQUIRE_PUBLICATION_READY" = 1 ]; then
  AGGREGATE_ARGS+=(--require-publication-ready)
fi
python3 "$SCRIPTS/analyze_dfx_campaign.py" "${AGGREGATE_ARGS[@]}"
echo FIVE_LAYER_0162_ROUTE_AWARE_DFX_ANALYSIS=PASS
