#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1

if [ "$#" -ne 4 ] && [ "$#" -ne 5 ]; then
  echo \
    "usage: $0 <root> <image@digest> <baseline-source> <candidate-source> [matched-policy.json]" \
    >&2
  exit 2
fi

ROOT=$1
IMG=$2
BASELINE_SOURCE=$3
CANDIDATE_SOURCE=$4
MATCHED_POLICY=${5:-}
MATCHED_CANDIDATE_ANALYSIS_DIR=${DFX_MATCHED_CANDIDATE_ANALYSIS_DIR:-dfx_analysis_route_v1/analysis}
MATCHED_SOURCE_COMPATIBILITY=${DFX_MATCHED_SOURCE_COMPATIBILITY:-$ROOT/campaign/dfx_campaign_route_v1_source_compatibility.json}
SCRIPTS=${CAMPAIGN_SCRIPTS:-$ROOT/scripts}
RUNNER=$SCRIPTS/run_0162_five_layer_matrix.sh
VALIDATOR=$SCRIPTS/validate_five_layer_case.py
AGGREGATOR=$SCRIPTS/analyze_dfx_campaign.py
SCRIPTS_MANIFEST=$SCRIPTS/SCRIPTS_SHA256SUMS
ROUNDS=${DFX_ROUNDS:-1}
ITERS=${DFX_ITERS:-1}
WARMUP=${DFX_WARMUP:-2}
BATCHES=(1 2 4 7 8 16)
ANALYZER_REL=tools/step3p5/analyze_five_layer_moe_dfx.py
CORRECTNESS=$ROOT/campaign/matrix_correctness_report.json
GOLDEN=$ROOT/campaign/golden/heterogeneous-64k

test -d "$ROOT"
test -d "$BASELINE_SOURCE"
test -d "$CANDIDATE_SOURCE"
test -x "$RUNNER"
test -s "$VALIDATOR"
test -s "$AGGREGATOR"
test -s "$SCRIPTS_MANIFEST"
test -s "$SCRIPTS/verify_exact_tree_manifest.py"
[[ "$ROUNDS" =~ ^[1-9][0-9]*$ ]]
[[ "$ITERS" =~ ^[1-9][0-9]*$ ]]
[[ "$WARMUP" =~ ^[0-9]+$ ]]
python3 "$SCRIPTS/verify_exact_tree_manifest.py" \
  --root "$SCRIPTS" \
  --manifest SCRIPTS_SHA256SUMS \
  --symlink-manifest -
SCRIPTS_MANIFEST_SHA=$(sha256sum "$SCRIPTS_MANIFEST" | awk '{print $1}')

BASELINE_PROFILE=baseline
CANDIDATE_PROFILE=candidate
MATCHED_POLICY_SHA=
BASELINE_POLICY_ID=
CANDIDATE_POLICY_ID=
BASELINE_SOURCE_MANIFEST_SHA=
CANDIDATE_SOURCE_MANIFEST_SHA=
BASELINE_DECODE_SHA=
CANDIDATE_DECODE_SHA=
BASELINE_RELEASE_ENFORCED=
CANDIDATE_RELEASE_ENFORCED=
if [ -n "$MATCHED_POLICY" ]; then
  test "$ROUNDS" -eq 1
  [[ "$MATCHED_CANDIDATE_ANALYSIS_DIR" =~ ^[A-Za-z0-9._/-]+$ ]]
  case "$MATCHED_CANDIDATE_ANALYSIS_DIR" in
    /*|*..*)
      echo "invalid matched candidate analysis dir" >&2
      exit 24
      ;;
  esac
  test -s "$MATCHED_POLICY"
  MATCHED_POLICY=$(realpath "$MATCHED_POLICY")
  python3 "$SCRIPTS/verify_exact_tree_manifest.py" \
    --root "$BASELINE_SOURCE"
  python3 "$SCRIPTS/verify_exact_tree_manifest.py" \
    --root "$CANDIDATE_SOURCE"
  mapfile -t POLICY_FIELDS < <(
    PYTHONPATH="$SCRIPTS" python3 - \
      "$MATCHED_POLICY" "$BASELINE_SOURCE" "$CANDIDATE_SOURCE" \
      "$IMG" "$SCRIPTS_MANIFEST" "$VALIDATOR" "$GOLDEN" \
      "$ROUNDS" "$ITERS" "$WARMUP" <<'PY'
import ast
import hashlib
import json
import sys
from pathlib import Path

from validate_five_layer_case import load_matched_dfx_policy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


(
    policy_path,
    baseline_text,
    candidate_text,
    image_ref,
    scripts_manifest_text,
    validator_text,
    golden_text,
    rounds_text,
    iters_text,
    warmup_text,
) = sys.argv[1:]
policy = load_matched_dfx_policy(policy_path)
roots = {
    "baseline": Path(baseline_text).resolve(),
    "candidate": Path(candidate_text).resolve(),
}
authority = policy["authority"]
if authority["image_ref"] != image_ref:
    raise AssertionError("matched authority image does not match campaign image")


def verify_reference(reference: dict[str, str], expected: Path | None = None) -> None:
    path = Path(reference["path"]).resolve()
    if expected is not None and path != expected.resolve():
        raise AssertionError(f"authority path mismatch: {path} != {expected}")
    if sha256(path) != reference["sha256"]:
        raise AssertionError(f"authority evidence hash mismatch: {path}")


verify_reference(authority["selection_report"])
verify_reference(authority["source_audit"])
verify_reference(authority["normal_correctness_report"])
verify_reference(authority["normal_performance_report"])
verify_reference(authority["normal_campaign_spec"])
verify_reference(authority["normal_counterbalance_spec"])
verify_reference(authority["normal_seal_authority"])
verify_reference(
    authority["scripts_manifest"],
    Path(scripts_manifest_text),
)
if sha256(Path(validator_text)) != authority["validator_sha256"]:
    raise AssertionError("authority validator hash mismatch")
golden = Path(golden_text)
for batch in (1, 2, 4, 7, 8, 16):
    verify_reference(
        authority["golden_manifests"][str(batch)],
        golden / f"bs{batch}" / "manifest.json",
    )
capture = authority["capture"]
if (
    capture["dfx_rounds"] != int(rounds_text)
    or capture["dfx_iters"] != int(iters_text)
    or capture["dfx_warmup"] != int(warmup_text)
):
    raise AssertionError("authority DFX capture protocol mismatch")

print(policy["policy_sha256"])
for source_kind in ("baseline", "candidate"):
    source_policy = policy["sources"][source_kind]
    root = roots[source_kind]
    manifest_sha = sha256(root / "SOURCE_SHA256SUMS")
    decode_sha = sha256(root / "models/step3p5/decode_fwd.py")
    analyzer_path = root / "tools/step3p5/analyze_five_layer_moe_dfx.py"
    analyzer_tree = ast.parse(
        analyzer_path.read_text(encoding="utf-8"),
        filename=str(analyzer_path),
    )
    analyze_defs = [
        node
        for node in analyzer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "analyze"
    ]
    if len(analyze_defs) != 1:
        raise AssertionError(
            f"{source_kind} analyzer must define exactly one analyze()"
        )
    analyze_args = analyze_defs[0].args
    parameter_names = {
        argument.arg
        for argument in (
            *analyze_args.posonlyargs,
            *analyze_args.args,
            *analyze_args.kwonlyargs,
        )
    }
    if "profile" not in parameter_names:
        raise AssertionError(
            f"{source_kind} analyzer does not accept an explicit profile"
        )
    if manifest_sha != source_policy["source_manifest_sha256"]:
        raise AssertionError(
            f"{source_kind} source manifest does not match matched policy"
        )
    if decode_sha != source_policy["decode_fwd_sha256"]:
        raise AssertionError(
            f"{source_kind} decode hash does not match matched policy"
        )
    if sha256(analyzer_path) != authority["analyzer_sha256"]:
        raise AssertionError(
            f"{source_kind} analyzer does not match authority"
        )
    print(source_policy["profile"])
    print(source_policy["policy_id"])
    print(manifest_sha)
    print(decode_sha)
    print(str(source_policy["release_enforced"]).lower())
PY
  )
  test "${#POLICY_FIELDS[@]}" -eq 11
  MATCHED_POLICY_SHA=${POLICY_FIELDS[0]}
  BASELINE_PROFILE=${POLICY_FIELDS[1]}
  BASELINE_POLICY_ID=${POLICY_FIELDS[2]}
  BASELINE_SOURCE_MANIFEST_SHA=${POLICY_FIELDS[3]}
  BASELINE_DECODE_SHA=${POLICY_FIELDS[4]}
  BASELINE_RELEASE_ENFORCED=${POLICY_FIELDS[5]}
  CANDIDATE_PROFILE=${POLICY_FIELDS[6]}
  CANDIDATE_POLICY_ID=${POLICY_FIELDS[7]}
  CANDIDATE_SOURCE_MANIFEST_SHA=${POLICY_FIELDS[8]}
  CANDIDATE_DECODE_SHA=${POLICY_FIELDS[9]}
  CANDIDATE_RELEASE_ENFORCED=${POLICY_FIELDS[10]}
fi

verify_matched_policy_sha() {
  if [ -n "$MATCHED_POLICY" ]; then
    test "$(sha256sum "$MATCHED_POLICY" | awk '{print $1}')" = \
      "$MATCHED_POLICY_SHA" || {
      echo "matched DFX policy changed during campaign" >&2
      exit 23
    }
  fi
}

python3 -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] is True' \
  "$CORRECTNESS"
test -d "$GOLDEN"

BASELINE_ANALYZER_SHA=$(
  sha256sum "$BASELINE_SOURCE/$ANALYZER_REL" | awk '{print $1}'
)
CANDIDATE_ANALYZER_SHA=$(
  sha256sum "$CANDIDATE_SOURCE/$ANALYZER_REL" | awk '{print $1}'
)
test "$BASELINE_ANALYZER_SHA" = "$CANDIDATE_ANALYZER_SHA" || {
  echo "baseline/candidate DFX analyzers differ" >&2
  exit 22
}

mkdir -p "$ROOT/campaign"
SPEC=$ROOT/campaign/dfx_campaign_spec.txt
if [ -n "$MATCHED_POLICY" ]; then
  SPEC_CONTENT=$(
    printf '%s\n' \
      "image=$IMG" \
      "baseline_source=$BASELINE_SOURCE" \
      "candidate_source=$CANDIDATE_SOURCE" \
      "campaign_scripts=$SCRIPTS" \
      "campaign_scripts_manifest_sha256=$SCRIPTS_MANIFEST_SHA" \
      "batches=1,2,4,7,8,16" \
      "context_len_per_sequence=65536" \
      "rounds=$ROUNDS" \
      "warmup=$WARMUP" \
      "measured_iters=$ITERS" \
      "capture=separate-warm-dep-gen-then-l2-swimlane" \
      "l2_swimlane_reuse_dep_gen=required" \
      "analyzer_sha256=$BASELINE_ANALYZER_SHA" \
      "matched_policy=$MATCHED_POLICY" \
      "matched_policy_sha256=$MATCHED_POLICY_SHA" \
      "baseline_profile=$BASELINE_PROFILE" \
      "baseline_policy_id=$BASELINE_POLICY_ID" \
      "baseline_source_manifest_sha256=$BASELINE_SOURCE_MANIFEST_SHA" \
      "baseline_decode_fwd_sha256=$BASELINE_DECODE_SHA" \
      "baseline_release_enforced=$BASELINE_RELEASE_ENFORCED" \
      "candidate_profile=$CANDIDATE_PROFILE" \
      "candidate_policy_id=$CANDIDATE_POLICY_ID" \
      "candidate_source_manifest_sha256=$CANDIDATE_SOURCE_MANIFEST_SHA" \
      "candidate_decode_fwd_sha256=$CANDIDATE_DECODE_SHA" \
      "candidate_release_enforced=$CANDIDATE_RELEASE_ENFORCED" \
      "candidate_analysis_dir=$MATCHED_CANDIDATE_ANALYSIS_DIR" \
      "source_compatibility=$MATCHED_SOURCE_COMPATIBILITY" \
      "publication_ready=required" \
      "order=per-bs-and-round:baseline,candidate"
  )
else
  SPEC_CONTENT=$(
    printf '%s\n' \
      "image=$IMG" \
      "baseline_source=$BASELINE_SOURCE" \
      "candidate_source=$CANDIDATE_SOURCE" \
      "campaign_scripts=$SCRIPTS" \
      "campaign_scripts_manifest_sha256=$SCRIPTS_MANIFEST_SHA" \
      "batches=1,2,4,7,8,16" \
      "context_len_per_sequence=65536" \
      "rounds=$ROUNDS" \
      "warmup=$WARMUP" \
      "measured_iters=$ITERS" \
      "capture=separate-warm-dep-gen-then-l2-swimlane" \
      "l2_swimlane_reuse_dep_gen=required" \
      "analyzer_sha256=$BASELINE_ANALYZER_SHA" \
      "order=per-bs-and-round:baseline,candidate"
  )
fi
if [ -e "$SPEC" ]; then
  test "$(cat "$SPEC")" = "$SPEC_CONTENT" || {
    echo "DFX campaign spec mismatch: $SPEC" >&2
    exit 20
  }
else
  printf '%s\n' "$SPEC_CONTENT" > "$SPEC"
fi

run_case() {
  local source_kind=$1
  local round_id=$2
  local batch=$3
  local source
  local dfx_profile
  local run_name
  local run_dir
  verify_matched_policy_sha
  if [ "$source_kind" = baseline ]; then
    source=$BASELINE_SOURCE
    dfx_profile=$BASELINE_PROFILE
  else
    source=$CANDIDATE_SOURCE
    dfx_profile=$CANDIDATE_PROFILE
  fi
  run_name="${source_kind}-r${round_id}-dfx-bs${batch}-64k"
  run_dir=$ROOT/campaign/runs/$run_name
  if [ -e "$run_dir" ]; then
    if [ -n "$MATCHED_POLICY" ]; then
      if MOE_DFX_PROFILE="$dfx_profile" \
        MOE_DFX_MATCHED_POLICY="$MATCHED_POLICY" \
        python3 "$VALIDATOR" --run "$run_dir"; then
        echo "[dfx-campaign] verified existing $run_name"
        return
      fi
    elif python3 "$VALIDATOR" --run "$run_dir"; then
      echo "[dfx-campaign] verified existing $run_name"
      return
    fi
    echo "[dfx-campaign] refusing incomplete or invalid run: $run_dir" >&2
    exit 21
  fi
  echo "[dfx-campaign] start $run_name"
  if [ -n "$MATCHED_POLICY" ]; then
    MOE_DFX_PROFILE="$dfx_profile" \
      MOE_DFX_MATCHED_POLICY="$MATCHED_POLICY" \
      CAMPAIGN_SCRIPTS=$SCRIPTS \
      MATRIX_ITERS=$ITERS MATRIX_WARMUP=$WARMUP \
      bash "$RUNNER" \
        "$ROOT" "$IMG" "$source" "$source_kind" \
        "$round_id" dfx "$batch"
    MOE_DFX_PROFILE="$dfx_profile" \
      MOE_DFX_MATCHED_POLICY="$MATCHED_POLICY" \
      python3 "$VALIDATOR" --run "$run_dir"
  else
    CAMPAIGN_SCRIPTS=$SCRIPTS \
      MATRIX_ITERS=$ITERS MATRIX_WARMUP=$WARMUP \
      bash "$RUNNER" \
        "$ROOT" "$IMG" "$source" "$source_kind" \
        "$round_id" dfx "$batch"
    python3 "$VALIDATOR" --run "$run_dir"
  fi
  echo "[dfx-campaign] pass $run_name"
}

for batch in "${BATCHES[@]}"; do
  for round_id in $(seq 1 "$ROUNDS"); do
    run_case baseline "$round_id" "$batch"
    run_case candidate "$round_id" "$batch"
  done
done

for round_id in $(seq 1 "$ROUNDS"); do
  verify_matched_policy_sha
  if [ -n "$MATCHED_POLICY" ]; then
    test -s "$MATCHED_SOURCE_COMPATIBILITY"
    python3 "$AGGREGATOR" \
      --campaign "$ROOT/campaign" \
      --round "$round_id" \
      --matched-policy "$MATCHED_POLICY" \
      --candidate-analysis-dir "$MATCHED_CANDIDATE_ANALYSIS_DIR" \
      --source-compatibility "$MATCHED_SOURCE_COMPATIBILITY" \
      --require-publication-ready \
      --report-prefix "dfx_campaign_formal_r${round_id}"
  else
    python3 "$AGGREGATOR" \
      --campaign "$ROOT/campaign" \
      --round "$round_id" \
      --report-prefix "dfx_campaign_diagnostic_r${round_id}"
  fi
done

if [ -n "$MATCHED_POLICY" ]; then
  echo FIVE_LAYER_0162_DFX_FORMAL_CAMPAIGN=PASS
else
  echo FIVE_LAYER_0162_DFX_DIAGNOSTIC_CAMPAIGN=PASS
fi
