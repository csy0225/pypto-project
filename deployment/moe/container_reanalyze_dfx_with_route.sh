#!/usr/bin/env bash
set -Eeuo pipefail

: "${ROUTE_ANALYSIS_BATCH:?missing ROUTE_ANALYSIS_BATCH}"
: "${ROUTE_ANALYSIS_PROFILE:?missing ROUTE_ANALYSIS_PROFILE}"
case "$ROUTE_ANALYSIS_BATCH" in
  1|2|4|7|8|16) ;;
  *)
    echo "invalid ROUTE_ANALYSIS_BATCH=$ROUTE_ANALYSIS_BATCH" >&2
    exit 31
    ;;
esac

export PYTHONDONTWRITEBYTECODE=1
export SOURCE_KIND=candidate
[[ "$ROUTE_ANALYSIS_PROFILE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "invalid ROUTE_ANALYSIS_PROFILE=$ROUTE_ANALYSIS_PROFILE" >&2
  exit 35
}
cd /workspace/pypto-lib
/usr/local/python3.11.14/bin/python3 \
  /campaign-scripts/verify_exact_tree_manifest.py \
  --root /workspace/pypto-lib \
  > /out/source_verify.log

mapfile -t BUILD_DIRS < <(
  find /dfx-run/runtime/build_output \
    -type d -name dfx_outputs -printf '%h\n' \
    | sort -u
)
if [ "${#BUILD_DIRS[@]}" -ne 1 ]; then
  printf 'expected one DFX build dir, got %s\n' "${#BUILD_DIRS[@]}" >&2
  printf '%s\n' "${BUILD_DIRS[@]}" >&2
  exit 32
fi
test -s /route/recv_meta_sidecar.pt
test -s /route/route_artifact_validation.json

/usr/local/python3.11.14/bin/python3 \
  tools/step3p5/analyze_five_layer_moe_dfx.py \
  --build-dir "${BUILD_DIRS[0]}" \
  --out /out/analysis \
  --skip-critical-path \
  --recv-meta-sidecar /route/recv_meta_sidecar.pt \
  --profile "$ROUTE_ANALYSIS_PROFILE"

test -s /out/analysis/moe_dfx_report.json
test -s /out/analysis/moe_critical_path_report.md
echo FIVE_LAYER_ROUTE_AWARE_DFX_ANALYSIS=PASS
