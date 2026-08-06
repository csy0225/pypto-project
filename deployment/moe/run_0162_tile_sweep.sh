#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <root> <image@digest> <source-root>" >&2
  exit 2
fi

ROOT=$1
IMG=$2
SOURCE_ROOT=$3
SCRIPTS=${CAMPAIGN_SCRIPTS:-$ROOT/scripts}
VARIANTS_TEXT=${TILE_VARIANTS:-mm-n32-r16,mm-n64-r8,mm-n64-r16,mm-n64-r32}
BATCHES_TEXT=${TILE_BATCHES:-1,2,4,7,8,16}
EXPECTED_VARIANTS=mm-n32-r16,mm-n64-r8,mm-n64-r16,mm-n64-r32
EXPECTED_BATCHES=1,2,4,7,8,16

test -d "$ROOT"
test -d "$SOURCE_ROOT"
test -d "$SCRIPTS"
test -x "$SCRIPTS/run_0162_tile_sweep_case.sh"
test -x "$SCRIPTS/run_0162_tile_compile_probe.sh"
test -s "$SCRIPTS/audit_tile_sweep_source.py"
test -s "$SCRIPTS/audit_tile_compile_resources.py"
test "$VARIANTS_TEXT" = "$EXPECTED_VARIANTS"
test "$BATCHES_TEXT" = "$EXPECTED_BATCHES"

IFS=, read -r -a VARIANTS <<< "$VARIANTS_TEXT"
IFS=, read -r -a BATCHES <<< "$BATCHES_TEXT"
test "${#VARIANTS[@]}" -gt 0
test "${#BATCHES[@]}" -gt 0

for variant in "${VARIANTS[@]}"; do
  [[ "$variant" =~ ^mm-n(32|64)-r(8|16|32)$ ]]
  test -d "$SOURCE_ROOT/$variant"
done
for batch in "${BATCHES[@]}"; do
  case "$batch" in
    1|2|4|7|8|16) ;;
    *) echo "invalid batch=$batch" >&2; exit 3 ;;
  esac
done

COMPILE_PASS_VARIANTS=()
for variant in "${VARIANTS[@]}"; do
  set +e
  bash "$SCRIPTS/run_0162_tile_compile_probe.sh" \
    "$ROOT" "$IMG" "$SOURCE_ROOT/$variant" "$variant"
  PROBE_RC=$?
  set -e
  if [ "$PROBE_RC" -eq 0 ]; then
    COMPILE_PASS_VARIANTS+=("$variant")
    continue
  fi
  FAILED_RUN=$ROOT/campaign/tile-compile/$variant
  test -s "$FAILED_RUN/container.rc"
  test "$(cat "$FAILED_RUN/container.rc")" != 0
  test -s "$FAILED_RUN/container.log"
  echo "[tile-compile] classified NO-GO $variant rc=$PROBE_RC"
done
test "${#COMPILE_PASS_VARIANTS[@]}" -gt 0
printf '%s\n' "${COMPILE_PASS_VARIANTS[@]}" \
  | grep -Fx "mm-n64-r16" >/dev/null
COMPILE_PASS_TEXT=$(
  IFS=,
  printf '%s' "${COMPILE_PASS_VARIANTS[*]}"
)

mkdir -p "$ROOT/campaign/tile"
AUDIT_OUT=$ROOT/campaign/tile_source_audit.json
python3 "$SCRIPTS/audit_tile_sweep_source.py" \
  --source-root "$SOURCE_ROOT" \
  --out "$AUDIT_OUT"

SPEC=$ROOT/campaign/tile_sweep_spec.txt
SPEC_CONTENT=$(
  printf '%s\n' \
    "schema=step3p5.moe.tile-sweep.v1" \
    "image=$IMG" \
    "source_root=$SOURCE_ROOT" \
    "variants=$VARIANTS_TEXT" \
    "compile_pass_variants=$COMPILE_PASS_TEXT" \
    "batches=$BATCHES_TEXT" \
    "context_len_per_sequence=65536" \
    "blocks_per_sequence=512" \
    "case_iters=${CASE_ITERS:-8}" \
    "case_warmup=${CASE_WARMUP:-3}" \
    "golden=heterogeneous-64k"
)
if [ -e "$SPEC" ]; then
  test "$(cat "$SPEC")" = "$SPEC_CONTENT" || {
    echo "tile sweep spec mismatch: $SPEC" >&2
    exit 20
  }
else
  printf '%s\n' "$SPEC_CONTENT" > "$SPEC"
fi

for variant in "${COMPILE_PASS_VARIANTS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    CASE_ITERS=${CASE_ITERS:-8} CASE_WARMUP=${CASE_WARMUP:-3} \
      bash "$SCRIPTS/run_0162_tile_sweep_case.sh" \
        "$ROOT" "$IMG" "$SOURCE_ROOT/$variant" "$variant" "$batch"
  done
done

python3 - \
  "$ROOT" "$IMG" "$SOURCE_ROOT" "$VARIANTS_TEXT" "$BATCHES_TEXT" \
  "$COMPILE_PASS_TEXT" <<'PY'
import json
import hashlib
import statistics
import sys
from pathlib import Path

(
    root_text,
    image,
    source_root_text,
    variants_text,
    batches_text,
    compile_pass_text,
) = sys.argv[1:]
root = Path(root_text).resolve()
source_root = Path(source_root_text).resolve()
variants = tuple(variants_text.split(","))
batches = tuple(int(item) for item in batches_text.split(","))
compile_pass_variants = tuple(compile_pass_text.split(","))
assert variants == (
    "mm-n32-r16",
    "mm-n64-r8",
    "mm-n64-r16",
    "mm-n64-r32",
)
assert batches == (1, 2, 4, 7, 8, 16)
assert len(set(variants)) == len(variants)
assert len(set(batches)) == len(batches)
assert compile_pass_variants
assert set(compile_pass_variants) <= set(variants)
assert "mm-n64-r16" in compile_pass_variants
rows = []
resource_audits = {}
compile_gate = {}
for variant in variants:
    run = root / "campaign" / "tile-compile" / variant
    container_rc = int((run / "container.rc").read_text().strip())
    record = {
        "run": str(run),
        "container_rc": container_rc,
        "image": (run / "image_ref.txt").read_text().strip(),
        "source_manifest_sha256": (
            run / "source_manifest_sha256.txt"
        ).read_text().strip(),
        "container_log_sha256": hashlib.sha256(
            (run / "container.log").read_bytes()
        ).hexdigest(),
    }
    assert record["image"] == image
    if variant in compile_pass_variants:
        assert container_rc == 0
        resource_path = run / "resource_audit.json"
        resource = json.loads(resource_path.read_text())
        assert resource["passed"] is True
        assert resource["variant"] == variant
        resource_audits[variant] = {
            "path": str(resource_path),
            "sha256": hashlib.sha256(resource_path.read_bytes()).hexdigest(),
            "maxima_bytes": resource["maxima_bytes"],
            "limits_bytes": resource["limits_bytes"],
        }
        record["status"] = "PASS"
        record["resource_audit"] = resource_audits[variant]
    else:
        assert container_rc != 0
        record["status"] = "NO_GO"
        errors = sorted(
            run.glob(
                "runtime/build_output/**/report/codegen_errors.txt"
            )
        )
        record["codegen_errors"] = [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in errors
        ]
    compile_gate[variant] = record

for variant in compile_pass_variants:
    for batch in batches:
        out = root / "campaign" / "tile" / variant / f"bs{batch}"
        report = json.loads(
            (out / "runtime" / "five_layer_moe_report.json").read_text()
        )
        timing = report["timing"]
        rows.append(
            {
                "variant": variant,
                "batch": batch,
                "p50_ms": timing["p50_ms"],
                "p99_ms": timing["p99_ms"],
                "max_ms": timing["max_ms"],
                "source_manifest_sha256": (
                    out / "source_manifest_sha256.txt"
                ).read_text().strip(),
                "decode_fwd_sha256": report["source"]["decode_fwd_sha256"],
                "hidden_l3_exact": report["comparisons"]["hidden_l3"]["exact"],
                "hidden_l4_exact": report["comparisons"]["hidden_l4"]["exact"],
                "resource_audit_passed": json.loads(
                    (out / "resource_audit.json").read_text()
                )["passed"],
                "profile": (
                    out / "MOE_TILE_SWEEP_PROFILE.txt"
                ).read_text().splitlines(),
            }
        )

summary = {
    "schema": "step3p5.moe.tile-sweep-report.v1",
    "image": image,
    "source_root": str(source_root),
    "context_len_per_sequence": 65536,
    "blocks_per_sequence": 512,
    "batches": list(batches),
    "variants": list(variants),
    "compile_pass_variants": list(compile_pass_variants),
    "compile_no_go_variants": [
        variant for variant in variants
        if variant not in compile_pass_variants
    ],
    "compile_gate": compile_gate,
    "passed": False,
    "correctness_passed": all(
        row["hidden_l3_exact"] and row["hidden_l4_exact"] for row in rows
    ),
    "resource_passed": all(
        row["resource_audit_passed"] for row in rows
    ),
    "compile_resource_audits": resource_audits,
    "dfx_status": "PENDING_FORMAL_MATCHED_SOURCE_CAPTURE",
    "rows": rows,
    "variant_summary": {},
}
for variant in variants:
    variant_rows = [row for row in rows if row["variant"] == variant]
    if variant not in compile_pass_variants:
        summary["variant_summary"][variant] = {
            "status": "COMPILE_NO_GO",
        }
        continue
    values = [row["p50_ms"] for row in variant_rows]
    summary["variant_summary"][variant] = {
        "status": "RUNTIME_PASS",
        "p50_mean_ms": statistics.fmean(values),
        "p50_by_batch_ms": {
            str(row["batch"]): row["p50_ms"] for row in variant_rows
        },
        "max_by_batch_ms": {
            str(row["batch"]): row["max_ms"] for row in variant_rows
        },
    }

out = root / "campaign" / "tile" / "tile_sweep_report.json"
summary["passed"] = bool(
    summary["correctness_passed"]
    and summary["resource_passed"]
    and len(rows) == len(compile_pass_variants) * len(batches)
    and len(compile_gate) == len(variants)
)
if out.exists() and out.read_text() != json.dumps(summary, indent=2, sort_keys=True) + "\n":
    raise SystemExit(f"refusing to replace different report: {out}")
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
if not summary["passed"]:
    raise SystemExit("MOE_TILE_SWEEP=FAIL")
print("MOE_TILE_SWEEP=PASS")
PY
