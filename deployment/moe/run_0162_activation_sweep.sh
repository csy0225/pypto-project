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
BATCHES=(1 2 4 7 8 16)
VARIANTS=(act-n64 act-n128 act-n256)

test -d "$ROOT"
test -d "$SOURCE_ROOT"
test -d "$SCRIPTS"
test -x "$SCRIPTS/run_0162_activation_sweep_case.sh"
mkdir -p "$ROOT/campaign/activation"

SPEC=$ROOT/campaign/activation/sweep_spec.txt
SPEC_CONTENT=$(
  printf '%s\n' \
    "schema=step3p5.moe.activation-sweep.v1" \
    "image=$IMG" \
    "source_root=$SOURCE_ROOT" \
    "variants=act-n64,act-n128,act-n256" \
    "batches=1,2,4,7,8,16" \
    "context_len_per_sequence=65536" \
    "blocks_per_sequence=512"
)
if [ -e "$SPEC" ]; then
  test "$(cat "$SPEC")" = "$SPEC_CONTENT" || {
    echo "activation sweep spec mismatch: $SPEC" >&2
    exit 20
  }
else
  printf '%s\n' "$SPEC_CONTENT" > "$SPEC"
fi

for variant in "${VARIANTS[@]}"; do
  SOURCE=$SOURCE_ROOT/$variant
  test -d "$SOURCE"
  test -s "$SOURCE/ACTIVATION_SWEEP_PROFILE.txt"
  test -s "$SOURCE/SOURCE_SHA256SUMS"
  (cd "$SOURCE" && sha256sum -c SOURCE_SHA256SUMS >/dev/null)
  for batch in "${BATCHES[@]}"; do
    bash "$SCRIPTS/run_0162_activation_sweep_case.sh" \
      "$ROOT" "$IMG" "$SOURCE" "$variant" "$batch"
  done
done

python3 - "$ROOT" "$IMG" "$SOURCE_ROOT" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root, image, source_root = map(Path, sys.argv[1:])
variants = ("act-n64", "act-n128", "act-n256")
batches = (1, 2, 4, 7, 8, 16)
rows = []
for variant in variants:
    source = source_root / variant
    for batch in batches:
        out = root / "campaign" / "activation" / variant / f"bs{batch}"
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
                "profile": (source / "ACTIVATION_SWEEP_PROFILE.txt").read_text(),
            }
        )
summary = {
    "schema": "step3p5.moe.activation-sweep-report.v1",
    "image": str(image),
    "source_root": str(source_root),
    "context_len_per_sequence": 65536,
    "blocks_per_sequence": 512,
    "batches": list(batches),
    "variants": list(variants),
    "correctness_passed": all(
        item["hidden_l3_exact"] and item["hidden_l4_exact"] for item in rows
    ),
    "rows": rows,
}
for variant in variants:
    values = [item["p50_ms"] for item in rows if item["variant"] == variant]
    summary.setdefault("variant_summary", {})[variant] = {
        "p50_mean_ms": statistics.fmean(values),
        "p50_by_batch_ms": {
            str(item["batch"]): item["p50_ms"]
            for item in rows
            if item["variant"] == variant
        },
    }
(root / "campaign" / "activation" / "sweep_report.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
assert summary["correctness_passed"]
print("MOE_ACTIVATION_SWEEP=PASS")
PY
