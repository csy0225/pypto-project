#!/usr/bin/env bash
# Exact-lease landing for the r10 pypto-lib release commit.
#
# Default behavior is descriptive only.  --check performs all immutable gate,
# source identity, remote-ref, fast-forward, and authenticated push dry-run
# checks without changing a ref.  --run repeats those checks and only then
# advances csy0225/pypto-lib:stepfun/develop with an exact lease.
set -Eeuo pipefail

REMOTE_SOURCE=${REMOTE_SOURCE:-0162}
CAMPAIGN_ROOT=${CAMPAIGN_ROOT:-/mnt/persist/chensiyu/workspace/moe-fusion-release-20260825}
SOURCE_REPO=${SOURCE_REPO:-$CAMPAIGN_ROOT/source/pypto-lib}
BRANCH=refs/heads/stepfun/develop
STAGING_BRANCH=refs/heads/release/r9-moe-fusion-20260825
R10_IMAGE=hub.i.basemind.com/stepcast/vllm-pypto@sha256:8510f30e1f2a2f2edcaa834c831165b349a4aca1212b655ca2a02ed6b3e9907b
R10_CONFIG=sha256:38ebba41d6aa0c49940c03e2e7c6fa42d85b61d631c143d38944683d0c657b5f
R9_IMAGE=hub.i.basemind.com/stepcast/vllm-pypto@sha256:b637f00c66d4dc976c053c617d2e19e6d6d66f68f4bef30250984da7a71690f6
R9_CONFIG=sha256:f6c8f72eecad0a9d40d0c4ea55afaab09dd4e2f5fe54d6a091e332465e421dae
OLD_PYPTO_LIB=bf3ff4400082f74b35fbdb5b3e0f5f4bf51ce373
NEW_PYPTO_LIB=fe641929dbf959d887ad111f3bd7cac0b73fa34b
NEW_PYPTO_LIB_TREE=5d8f7e647cab301ee5bb2f0175fec4d91bfa71e8

MODE=describe
ROUTE_DIR=
OUTER_DIR=
BS_DIR=
ABA_DIR=
CONFIRM_LEASE=
CONFIRM_TARGET=

usage() {
  cat <<EOF
Usage:
  $0 --describe
  $0 --check --route-dir PATH --outer-dir PATH --bs-dir PATH --aba-dir PATH
  $0 --run   --route-dir PATH --outer-dir PATH --bs-dir PATH --aba-dir PATH \\
     --confirm-lease $OLD_PYPTO_LIB --confirm-target $NEW_PYPTO_LIB

Modes:
  --describe  Print the immutable identity and required gates. No network write.
  --check     Validate every gate/ref and run an authenticated push --dry-run.
  --run       Repeat --check, close the lease race, then exact-lease push once.

The final verdict is copied to:
  $REMOTE_SOURCE:$CAMPAIGN_ROOT/git-sync-r10-<timestamp>/verdict.json

Only pypto-lib moves. The other four stepfun/develop refs must remain exactly:
  pypto      519b588a7a6461cac0e443e853accf29479c1d15
  pto-isa    cd4a3d3f7a1a27fcfe536f617e9bca3008929664
  PTOAS      307d0484a9e7d5e36f01b253d2bebe4d2f45fe81
  simpler    85a82c454074c069315ed6485033c3c2b136e562
EOF
}

while (($#)); do
  case "$1" in
    --describe)
      MODE=describe
      shift
      ;;
    --check)
      MODE=check
      shift
      ;;
    --run)
      MODE=run
      shift
      ;;
    --route-dir)
      ROUTE_DIR=${2:?missing --route-dir value}
      shift 2
      ;;
    --outer-dir)
      OUTER_DIR=${2:?missing --outer-dir value}
      shift 2
      ;;
    --bs-dir)
      BS_DIR=${2:?missing --bs-dir value}
      shift 2
      ;;
    --aba-dir)
      ABA_DIR=${2:?missing --aba-dir value}
      shift 2
      ;;
    --confirm-lease)
      CONFIRM_LEASE=${2:?missing --confirm-lease value}
      shift 2
      ;;
    --confirm-target)
      CONFIRM_TARGET=${2:?missing --confirm-target value}
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "FAIL: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ $MODE == describe ]]; then
  usage
  exit 0
fi

declare -A OPTION_NAME=(
  [ROUTE_DIR]=--route-dir
  [OUTER_DIR]=--outer-dir
  [BS_DIR]=--bs-dir
  [ABA_DIR]=--aba-dir
)
for value in ROUTE_DIR OUTER_DIR BS_DIR ABA_DIR; do
  [[ -n ${!value} ]] || {
    echo "FAIL: ${OPTION_NAME[$value]} is required for $MODE" >&2
    exit 2
  }
done
if [[ $MODE == run ]]; then
  [[ $CONFIRM_LEASE == "$OLD_PYPTO_LIB" ]] || {
    echo "FAIL: --confirm-lease must equal $OLD_PYPTO_LIB" >&2
    exit 2
  }
  [[ $CONFIRM_TARGET == "$NEW_PYPTO_LIB" ]] || {
    echo "FAIL: --confirm-target must equal $NEW_PYPTO_LIB" >&2
    exit 2
  }
fi

normalize_remote_path() {
  local value=$1
  if [[ $value == *:/* ]]; then
    printf '%s\n' "${value#*:}"
  else
    printf '%s\n' "$value"
  fi
}

ROUTE_DIR=$(normalize_remote_path "$ROUTE_DIR")
OUTER_DIR=$(normalize_remote_path "$OUTER_DIR")
BS_DIR=$(normalize_remote_path "$BS_DIR")
ABA_DIR=$(normalize_remote_path "$ABA_DIR")

STAMP=$(date +%Y%m%d-%H%M%S)
if [[ $MODE == run ]]; then
  REMOTE_OUT=$CAMPAIGN_ROOT/git-sync-r10-$STAMP
else
  REMOTE_OUT=$CAMPAIGN_ROOT/git-sync-r10-preflight-$STAMP
fi
LOCAL_ROOT=$(mktemp -d)
EVIDENCE=$LOCAL_ROOT/evidence
BARE=$LOCAL_ROOT/pypto-lib.git
LOG=$LOCAL_ROOT/sync.log
REMOTE_ROWS_BEFORE=$LOCAL_ROOT/remote_before.tsv
REMOTE_ROWS_AFTER=$LOCAL_ROOT/remote_after.tsv
mkdir -p "$EVIDENCE"

cleanup() {
  python3 - "$LOCAL_ROOT" <<'PY'
import shutil
import sys

shutil.rmtree(sys.argv[1], ignore_errors=True)
PY
}
trap cleanup EXIT

exec 9>"${TMPDIR:-/tmp}/step3p5-r10-stepfun-sync.lock"
flock -n 9 || {
  echo "FAIL: another local r10 sync runner is active" >&2
  exit 73
}

copy_remote_evidence() {
  local remote_path=$1
  local local_path=$2
  local label=$3
  ssh -o BatchMode=yes "$REMOTE_SOURCE" "test -s '$remote_path'" || {
    echo "FAIL: missing $label: $REMOTE_SOURCE:$remote_path" >&2
    return 80
  }
  scp -q "$REMOTE_SOURCE:$remote_path" "$local_path"
}

copy_remote_evidence \
  "$ROUTE_DIR/route_gate.json" \
  "$EVIDENCE/route_gate.json" \
  "route gate"
copy_remote_evidence \
  "$OUTER_DIR/outer_admission.json" \
  "$EVIDENCE/outer_admission.json" \
  "outer DFX gate"
copy_remote_evidence \
  "$BS_DIR/six_batch_r9_r10_verdict.json" \
  "$EVIDENCE/six_batch_r9_r10_verdict.json" \
  "six-BS gate"
copy_remote_evidence \
  "$ABA_DIR/aba_admission.json" \
  "$EVIDENCE/aba_admission.json" \
  "immutable A/B/A gate"

python3 - \
  "$EVIDENCE" \
  "$R9_IMAGE" "$R9_CONFIG" \
  "$R10_IMAGE" "$R10_CONFIG" \
  "$LOCAL_ROOT/prerequisites.json" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
r9_image, r9_config, r10_image, r10_config = sys.argv[2:6]
out = Path(sys.argv[6])


def load(name):
    path = root / name
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), name
    return path, value


def all_true(value):
    return isinstance(value, dict) and bool(value) and all(
        item is True for item in value.values()
    )


paths = {}
route_path, route = load("route_gate.json")
outer_path, outer = load("outer_admission.json")
bs_path, bs = load("six_batch_r9_r10_verdict.json")
aba_path, aba = load("aba_admission.json")
for label, path in (
    ("route", route_path),
    ("outer", outer_path),
    ("six_batch", bs_path),
    ("aba", aba_path),
):
    paths[label] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }

checks = {
    "route_schema": route.get("schema")
    == "step3p5.r10-route-sidecar-gate.v1",
    "route_pass": route.get("pass") is True,
    "route_checks": all_true(route.get("checks")),
    "route_image": route.get("image_ref") == r10_image,
    "outer_schema": outer.get("schema")
    == "step3p5.r10-packed-nz-outer-swimlane-verdict.v1",
    "outer_pass": outer.get("pass") is True,
    "outer_image": outer.get("image_ref") == r10_image,
    "outer_hidden_exact": outer.get("hidden_l3_exact") is True
    and outer.get("hidden_l4_exact") is True
    and outer.get("hidden_l3_sha_matches_golden") is True
    and outer.get("hidden_l4_sha_matches_golden") is True,
    "outer_eight_rank_dfx": all(
        outer.get(name) == 8
        for name in (
            "chip_swimlane_records",
            "deps_records",
            "name_maps",
            "critical_path_reports",
            "merged_swimlanes",
            "dfx_protocol_rank_count",
        )
    ),
    "outer_analyzer": outer.get("analyzer_pass") is True
    and outer.get("analyzer_gate_pass") is True
    and outer.get("analyzer_blockers") == []
    and outer.get("resource_grid_exact") is True,
    "six_batch_schema": bs.get("schema")
    == "step3p5.r10-six-batch-r9-r10-verdict.v1",
    "six_batch_pass": bs.get("pass") is True,
    "six_batch_counts": bs.get("summary", {}).get("exact_batches") == 6
    and bs.get("summary", {}).get("healthy_arm_batches") == 12
    and bs.get("summary", {}).get("max_tp_spread") == 0.0
    and bs.get("summary", {}).get("max_inactive_abs") == 0.0,
    "six_batch_cases": len(bs.get("cases", [])) == 6
    and {row.get("active_batch") for row in bs.get("cases", [])}
    == {1, 2, 4, 7, 8, 16}
    and all(row.get("passed") is True for row in bs.get("cases", [])),
    "aba_schema": aba.get("schema")
    == "step3p5.r10-immutable-image-aba-admission.v1",
    "aba_pass": aba.get("pass") is True,
    "aba_checks": all_true(aba.get("checks")),
    "aba_arm_order": set(aba.get("arms", {})) == {"A1", "B", "A2"},
    "aba_arm_identities": (
        aba.get("arms", {}).get("A1", {}).get("image_manifest") == r9_image
        and aba.get("arms", {}).get("A1", {}).get("image_config") == r9_config
        and aba.get("arms", {}).get("B", {}).get("image_manifest") == r10_image
        and aba.get("arms", {}).get("B", {}).get("image_config") == r10_config
        and aba.get("arms", {}).get("A2", {}).get("image_manifest") == r9_image
        and aba.get("arms", {}).get("A2", {}).get("image_config") == r9_config
    ),
    "aba_candidate_improves": (
        math.isfinite(float(aba.get("candidate_minus_midpoint_p50_ms")))
        and float(aba.get("candidate_minus_midpoint_p50_ms")) < 0.0
        and math.isfinite(float(aba.get("candidate_minus_midpoint_p50_pct")))
        and float(aba.get("candidate_minus_midpoint_p50_pct")) < 0.0
    ),
}
payload = {
    "schema": "step3p5.r10-sync-prerequisites.v1",
    "pass": all(checks.values()),
    "checks": checks,
    "artifacts": paths,
    "aba_summary": {
        name: aba.get(name)
        for name in (
            "baseline_midpoint_p50_ms",
            "baseline_bracket_p50_ms",
            "candidate_minus_midpoint_p50_ms",
            "candidate_minus_midpoint_p50_pct",
        )
    },
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
if not payload["pass"]:
    failed = [name for name, passed in checks.items() if not passed]
    raise SystemExit(f"release prerequisites failed: {failed}")
PY

ssh -o BatchMode=yes "$REMOTE_SOURCE" \
  "set -e
   head=\$(git -C '$SOURCE_REPO' rev-parse HEAD)
   tree=\$(git -C '$SOURCE_REPO' rev-parse HEAD^{tree})
   status=\$(git -C '$SOURCE_REPO' status --porcelain=v1 --untracked-files=all)
   parent=\$(git -C '$SOURCE_REPO' rev-parse '$NEW_PYPTO_LIB^')
   test \"\$head\" = '$NEW_PYPTO_LIB'
   test \"\$tree\" = '$NEW_PYPTO_LIB_TREE'
   test -z \"\$status\"
   git -C '$SOURCE_REPO' merge-base --is-ancestor '$OLD_PYPTO_LIB' '$NEW_PYPTO_LIB'
   test \"\$parent\" = '$OLD_PYPTO_LIB'
   printf 'head=%s\\ntree=%s\\nparent=%s\\nclean=true\\nfast_forward=true\\n' \
     \"\$head\" \"\$tree\" \"\$parent\"" \
  >"$LOCAL_ROOT/source_identity.log"

declare -a REPOS=(pto-isa PTOAS simpler pypto pypto-lib)
declare -A EXPECTED_BEFORE=(
  [pto-isa]=cd4a3d3f7a1a27fcfe536f617e9bca3008929664
  [PTOAS]=307d0484a9e7d5e36f01b253d2bebe4d2f45fe81
  [simpler]=85a82c454074c069315ed6485033c3c2b136e562
  [pypto]=519b588a7a6461cac0e443e853accf29479c1d15
  [pypto-lib]=$OLD_PYPTO_LIB
)
declare -A EXPECTED_AFTER=(
  [pto-isa]=cd4a3d3f7a1a27fcfe536f617e9bca3008929664
  [PTOAS]=307d0484a9e7d5e36f01b253d2bebe4d2f45fe81
  [simpler]=85a82c454074c069315ed6485033c3c2b136e562
  [pypto]=519b588a7a6461cac0e443e853accf29479c1d15
  [pypto-lib]=$NEW_PYPTO_LIB
)

remote_sha() {
  local repository=$1
  GIT_TERMINAL_PROMPT=0 \
    git ls-remote "https://github.com/csy0225/${repository}.git" "$BRANCH" |
    awk 'NR == 1 {print $1}'
}

: >"$REMOTE_ROWS_BEFORE"
for repository in "${REPOS[@]}"; do
  live=$(remote_sha "$repository")
  [[ -n $live ]] || {
    echo "FAIL: $repository has no $BRANCH" >&2
    exit 81
  }
  [[ $live == "${EXPECTED_BEFORE[$repository]}" ]] || {
    echo \
      "FAIL: $repository lease changed: expected=${EXPECTED_BEFORE[$repository]} live=$live" \
      >&2
    exit 82
  }
  printf '%s\t%s\t%s\t%s\n' \
    "$repository" "$live" \
    "${EXPECTED_BEFORE[$repository]}" "${EXPECTED_AFTER[$repository]}" \
    >>"$REMOTE_ROWS_BEFORE"
done

git init --bare -q "$BARE"
git -C "$BARE" remote add target https://github.com/csy0225/pypto-lib.git
git -C "$BARE" remote add source "$REMOTE_SOURCE:$SOURCE_REPO"
GIT_TERMINAL_PROMPT=0 git -C "$BARE" fetch -q --no-tags target \
  "+$BRANCH:refs/remotes/target/stepfun-develop" \
  "+$STAGING_BRANCH:refs/remotes/target/r10-staging"
git -C "$BARE" fetch -q --no-tags source \
  "$NEW_PYPTO_LIB:refs/remotes/source/r10-release"

[[ $(git -C "$BARE" rev-parse refs/remotes/target/stepfun-develop) == "$OLD_PYPTO_LIB" ]]
[[ $(git -C "$BARE" rev-parse refs/remotes/target/r10-staging) == "$NEW_PYPTO_LIB" ]]
[[ $(git -C "$BARE" rev-parse refs/remotes/source/r10-release) == "$NEW_PYPTO_LIB" ]]
[[ $(git -C "$BARE" rev-parse "$NEW_PYPTO_LIB^{tree}") == "$NEW_PYPTO_LIB_TREE" ]]
[[ $(git -C "$BARE" rev-parse "$NEW_PYPTO_LIB^") == "$OLD_PYPTO_LIB" ]]
git -C "$BARE" merge-base --is-ancestor "$OLD_PYPTO_LIB" "$NEW_PYPTO_LIB"

{
  echo "mode=$MODE"
  echo "branch=$BRANCH"
  echo "lease=$OLD_PYPTO_LIB"
  echo "target=$NEW_PYPTO_LIB"
  echo "tree=$NEW_PYPTO_LIB_TREE"
  echo "staging_branch=$STAGING_BRANCH"
  echo "prerequisites_sha256=$(sha256sum "$LOCAL_ROOT/prerequisites.json" | awk '{print $1}')"
} >"$LOCAL_ROOT/run_contract.txt"

set +e
GIT_TERMINAL_PROMPT=0 git -C "$BARE" push --dry-run \
  --force-with-lease="$BRANCH:$OLD_PYPTO_LIB" \
  target "$NEW_PYPTO_LIB:$BRANCH" \
  2>&1 | tee "$LOCAL_ROOT/push_dry_run.log"
dry_run_rc=${PIPESTATUS[0]}
set -e
[[ $dry_run_rc -eq 0 ]] || {
  echo "FAIL: authenticated push dry-run rc=$dry_run_rc" >&2
  exit "$dry_run_rc"
}

before_push_sha=
push_rc=
executed=false
if [[ $MODE == run ]]; then
  # Re-hash the exact gate files on 0162 immediately before the mutating push.
  python3 - \
    "$LOCAL_ROOT/prerequisites.json" \
    "$REMOTE_SOURCE" \
    "$ROUTE_DIR/route_gate.json" \
    "$OUTER_DIR/outer_admission.json" \
    "$BS_DIR/six_batch_r9_r10_verdict.json" \
    "$ABA_DIR/aba_admission.json" <<'PY'
import json
import subprocess
import sys

prerequisites = json.load(open(sys.argv[1], encoding="utf-8"))
remote = sys.argv[2]
remote_paths = sys.argv[3:]
for label, remote_path in zip(
    ("route", "outer", "six_batch", "aba"), remote_paths
):
    actual = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", remote, "sha256sum", remote_path],
        text=True,
    ).split()[0]
    expected = prerequisites["artifacts"][label]["sha256"]
    if actual != expected:
        raise SystemExit(
            f"{label} evidence changed before push: "
            f"expected={expected} actual={actual}"
        )
PY

  # Close the dry-run-to-push race for every release pin.
  for repository in "${REPOS[@]}"; do
    live=$(remote_sha "$repository")
    [[ $live == "${EXPECTED_BEFORE[$repository]}" ]] || {
      echo "FAIL: $repository moved after dry-run: $live" >&2
      exit 84
    }
  done
  before_push_sha=$(remote_sha pypto-lib)
  [[ $before_push_sha == "$OLD_PYPTO_LIB" ]]

  set +e
  GIT_TERMINAL_PROMPT=0 git -C "$BARE" push \
    --force-with-lease="$BRANCH:$OLD_PYPTO_LIB" \
    target "$NEW_PYPTO_LIB:$BRANCH" \
    2>&1 | tee "$LOCAL_ROOT/push.log"
  push_rc=${PIPESTATUS[0]}
  set -e
  [[ $push_rc -eq 0 ]] || {
    echo "FAIL: exact-lease push rc=$push_rc" >&2
    exit "$push_rc"
  }
  executed=true
fi

: >"$REMOTE_ROWS_AFTER"
for repository in "${REPOS[@]}"; do
  live=$(remote_sha "$repository")
  expected=${EXPECTED_BEFORE[$repository]}
  if [[ $MODE == run ]]; then
    expected=${EXPECTED_AFTER[$repository]}
  fi
  [[ $live == "$expected" ]] || {
    echo \
      "FAIL: $repository post-$MODE mismatch: expected=$expected live=$live" \
      >&2
    exit 85
  }
  printf '%s\t%s\t%s\t%s\n' \
    "$repository" "$live" \
    "${EXPECTED_BEFORE[$repository]}" "${EXPECTED_AFTER[$repository]}" \
    >>"$REMOTE_ROWS_AFTER"
done

cp "$0" "$LOCAL_ROOT/runner.sh"
python3 - \
  "$MODE" "$executed" "$dry_run_rc" "${push_rc:-}" \
  "$REMOTE_ROWS_BEFORE" "$REMOTE_ROWS_AFTER" \
  "$LOCAL_ROOT/prerequisites.json" \
  "$LOCAL_ROOT/verdict.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    mode,
    executed_text,
    dry_run_rc_text,
    push_rc_text,
    before_path,
    after_path,
    prerequisites_path,
    destination,
) = sys.argv[1:]
order = ["pto-isa", "PTOAS", "simpler", "pypto", "pypto-lib"]
before = {}
for line in Path(before_path).read_text(encoding="utf-8").splitlines():
    name, live, expected_before, expected_after = line.split("\t")
    before[name] = {
        "live": live,
        "expected_before": expected_before,
        "expected_after": expected_after,
    }
after = {}
for line in Path(after_path).read_text(encoding="utf-8").splitlines():
    name, live, expected_before, expected_after = line.split("\t")
    after[name] = {
        "live": live,
        "expected_before": expected_before,
        "expected_after": expected_after,
    }
executed = executed_text == "true"
repositories = []
for name in order:
    old = before[name]["live"]
    target = before[name]["expected_after"]
    verified = after[name]["live"]
    repositories.append(
        {
            "repository": name,
            "old_stepfun_develop": old,
            "new_stepfun_develop": target,
            "fast_forward": True,
            "backup_ref": None,
            "verified_remote_sha": verified,
            "changed": old != target,
        }
    )
prerequisites = json.loads(Path(prerequisites_path).read_text(encoding="utf-8"))
if mode == "run":
    schema = "step3p5.r10-five-repo-sync.v1"
    refs_pass = all(
        row["new_stepfun_develop"] == row["verified_remote_sha"]
        for row in repositories
    )
    pass_value = (
        executed
        and int(dry_run_rc_text) == 0
        and int(push_rc_text) == 0
        and prerequisites.get("pass") is True
        and refs_pass
    )
else:
    schema = "step3p5.r10-five-repo-sync-preflight.v1"
    refs_pass = all(
        before[name]["live"] == before[name]["expected_before"]
        for name in order
    )
    pass_value = (
        not executed
        and int(dry_run_rc_text) == 0
        and prerequisites.get("pass") is True
        and refs_pass
    )
payload = {
    "schema": schema,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "mode": mode,
    "executed": executed,
    "branch": "refs/heads/stepfun/develop",
    "order": order,
    "exact_lease": {
        "repository": "pypto-lib",
        "expected_old": (
            "bf3ff4400082f74b35fbdb5b3e0f5f4bf51ce373"
        ),
        "target": "fe641929dbf959d887ad111f3bd7cac0b73fa34b",
        "dry_run_rc": int(dry_run_rc_text),
        "push_rc": int(push_rc_text) if push_rc_text else None,
    },
    "prerequisites": prerequisites,
    "repositories": repositories,
    "ready_to_execute": pass_value if mode == "check" else None,
    "pass": pass_value,
}
Path(destination).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if not pass_value:
    raise SystemExit(f"{mode} verdict failed")
PY

ssh -o BatchMode=yes "$REMOTE_SOURCE" "test ! -e '$REMOTE_OUT' && mkdir -p '$REMOTE_OUT'"
scp -q \
  "$LOCAL_ROOT/prerequisites.json" \
  "$LOCAL_ROOT/run_contract.txt" \
  "$LOCAL_ROOT/push_dry_run.log" \
  "$LOCAL_ROOT/remote_before.tsv" \
  "$LOCAL_ROOT/remote_after.tsv" \
  "$LOCAL_ROOT/source_identity.log" \
  "$LOCAL_ROOT/verdict.json" \
  "$LOCAL_ROOT/runner.sh" \
  "$REMOTE_SOURCE:$REMOTE_OUT/"
if [[ -s $LOCAL_ROOT/push.log ]]; then
  scp -q "$LOCAL_ROOT/push.log" "$REMOTE_SOURCE:$REMOTE_OUT/"
fi
scp -q "$EVIDENCE"/*.json "$REMOTE_SOURCE:$REMOTE_OUT/"
ssh -o BatchMode=yes "$REMOTE_SOURCE" \
  "cd '$REMOTE_OUT' &&
   find . -type f ! -name artifacts.sha256 -print0 |
     LC_ALL=C sort -z |
     xargs -0 sha256sum > artifacts.sha256 &&
   cat verdict.json"

if [[ $MODE == run ]]; then
  printf '%s\n' "$REMOTE_OUT" |
    ssh -o BatchMode=yes "$REMOTE_SOURCE" \
      "cat > '$CAMPAIGN_ROOT/.latest-git-sync-r10'"
  echo "R10_EXACT_LEASE_SYNC_PASS OUT=$REMOTE_SOURCE:$REMOTE_OUT"
else
  echo "R10_EXACT_LEASE_SYNC_PREFLIGHT_PASS OUT=$REMOTE_SOURCE:$REMOTE_OUT"
fi
