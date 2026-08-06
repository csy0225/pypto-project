#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <root> <image@digest>" >&2
  exit 2
fi

ROOT=$1
IMG=$2
CAMPAIGN=$ROOT/campaign
SCRIPTS=$ROOT/scripts
VALIDATOR=$SCRIPTS/validate_five_layer_case.py
ANALYZER=$SCRIPTS/analyze_matrix_correctness.py
NC=/mnt/persist/k8s-install/containerd/bin/nerdctl
BATCHES=(1 2 4 7 8 16)
SOURCES=(baseline candidate)
ROUNDS=(1 2)

test -d "$ROOT"
test -d "$CAMPAIGN"
test -s "$VALIDATOR"
test -s "$ANALYZER"
test "$(cat "$CAMPAIGN/image_ref.txt")" = "$IMG"

for batch in "${BATCHES[@]}"; do
  for round_id in "${ROUNDS[@]}"; do
    for source_kind in "${SOURCES[@]}"; do
      python3 "$VALIDATOR" \
        --run "$CAMPAIGN/runs/${source_kind}-r${round_id}-normal-bs${batch}-64k"
    done
  done
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
if [ -s "$CAMPAIGN/correctness.log" ] \
  && [ ! -e "$CAMPAIGN/matrix_correctness_report.json" ]; then
  FAILURE=$CAMPAIGN/failures/correctness-apparmor-resolver-$STAMP
  mkdir -p "$FAILURE"
  cp -a "$CAMPAIGN/correctness.log" "$FAILURE/"
  cp -a "$CAMPAIGN/normal_campaign_spec.txt" "$FAILURE/"
  cp -a "$CAMPAIGN/image_ref.txt" "$FAILURE/"
  (
    cd "$FAILURE"
    sha256sum correctness.log normal_campaign_spec.txt image_ref.txt \
      > EVIDENCE_SHA256SUMS
  )
  printf '%s\n' \
    "schema=step3p5.five-layer-moe-failure.v1" \
    "stage=correctness-container-launch" \
    "cause=nerdctl rejected current-directory apparmor_parser resolution" \
    "device_runs_preserved=24/24" \
    "authoritative_date=2026-08-05" \
    "host_clock_note=$CAMPAIGN/CLOCK_SKEW_NOTE.txt" \
    > "$FAILURE/MANIFEST.txt"
fi

CORRECTNESS=$CAMPAIGN/matrix_correctness_report.json
GOLDEN=$CAMPAIGN/golden/heterogeneous-64k
if [ ! -e "$GOLDEN" ]; then
  LOG=$CAMPAIGN/correctness_retry_$STAMP.log
  cd /
  set +e
  sudo -n "$NC" run --rm --net host \
    --security-opt apparmor=unconfined \
    -v "$CAMPAIGN":/campaign \
    -v "$SCRIPTS":/campaign-scripts:ro \
    "$IMG" \
    /usr/local/python3.11.14/bin/python3 \
    /campaign-scripts/analyze_matrix_correctness.py \
    --campaign /campaign \
    2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  set -e
  if [ "$RC" -ne 0 ]; then
    FAILURE=$CAMPAIGN/failures/correctness-retry-$STAMP
    mkdir -p "$FAILURE"
    cp -a "$LOG" "$FAILURE/"
    printf '%s\n' "$RC" > "$FAILURE/container.rc"
    (
      cd "$FAILURE"
      sha256sum "$(basename "$LOG")" container.rc > EVIDENCE_SHA256SUMS
    )
    exit "$RC"
  fi
fi

python3 -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] is True' \
  "$CORRECTNESS"
test -d "$GOLDEN"
for batch in "${BATCHES[@]}"; do
  test -s "$GOLDEN/bs${batch}/hidden_l3.pt"
  test -s "$GOLDEN/bs${batch}/hidden_l4.pt"
  test -s "$GOLDEN/bs${batch}/manifest.json"
done

if [ ! -e "$GOLDEN/SHA256SUMS" ]; then
  sudo -n bash -c '
    set -Eeuo pipefail
    cd "$1"
    find . -type f ! -name SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum \
      > SHA256SUMS
  ' _ "$GOLDEN"
fi
(
  cd "$GOLDEN"
  sha256sum -c SHA256SUMS
)
sudo -n chmod -R a-w "$GOLDEN"

echo FIVE_LAYER_0162_CORRECTNESS_FINALIZE=PASS
