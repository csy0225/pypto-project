#!/usr/bin/env python3
"""Validate one immutable five-layer, 64K-per-sequence matrix run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


TP = 8
NUM_LAYERS = 5
STORAGE_BATCH = 16
BLOCK_SIZE = 128
HEAD_DIM = 128
BF16_BYTES = 2
CONTEXT_LEN = 65536
BLOCKS_PER_SEQUENCE = CONTEXT_LEN // BLOCK_SIZE
MATCHED_DFX_POLICY_SCHEMA = "step3p5.moe.matched-dfx-policy.v1"
MATCHED_DFX_POLICY_ENV = "MOE_DFX_MATCHED_POLICY"
MATCHED_DFX_SOURCE_KINDS = ("baseline", "candidate")
NORMAL_SEAL_AUTHORITY_SCHEMA = (
    "step3p5.five-layer-moe-normal-seal-authority.v1"
)
RUNTIME_SEAL_FIELDS = (
    "runtime_marker_sha256",
    "kv_key_sha256_by_rank",
)
IMAGE_PYPTO_COMMIT = "8e92b46808f9f7c09b6431ad4691503f09c12ee5"
IMAGE_PYPTO_LIB_COMMIT = "491267c45875e9b1e0071eed224e2e73526799e2"
IMAGE_ATTN_PROFILE = "a2a3"
IMAGE_AUDIT_SCHEMA = "step3p5.moe.pre-mount-image-audit.v1"
CAPABILITY_SCHEMA = "step3p5.moe-image-capability.v1"
DFX_RAW_SCHEMA = "step3p5.moe.dfx-raw-evidence.v1"
RUN_RE = re.compile(
    r"^(baseline|candidate)-r([0-9]+)-(normal|dfx)-"
    r"bs(1|2|4|7|8|16)-64k$"
)
MERGED_SWIMLANE_RE = re.compile(r"^merged_swimlane_.+\.json$")
DFX_POLICIES = {
    "baseline": {
        "policy_id": "campaign-baseline-56b3d477-row32-fused-v1",
        "decode_fwd_sha256": (
            "3553664cbe5bba2453b17b992c9c8a5489deb0df8f88b98d4a93a1aa45544ff0"
        ),
        "release_enforced": False,
    },
    "candidate": {
        "policy_id": (
            "campaign-candidate-65b0b8bf-row16-graph-wide-split-v1"
        ),
        "decode_fwd_sha256": (
            "65b0b8bf139aa40a5cf67317148dc16193ff22a81b394fbfe86e31ea05623e08"
        ),
        "release_enforced": True,
    },
    "row16": {
        "policy_id": "shared-experiment-reference-row16-v1",
        "source_kind": "baseline",
        "source_role": "reference",
        "decode_fwd_sha256": (
            "65b0b8bf139aa40a5cf67317148dc16193ff22a81b394fbfe86e31ea05623e08"
        ),
        "release_enforced": True,
    },
    "shared-split": {
        "policy_id": "shared-experiment-5-5-16-v1",
        "source_kind": "candidate",
        "source_role": "candidate",
        "decode_fwd_sha256": (
            "572ea2a2b0ceab8952cbd8d7c1f383351fb877e11b01b4d68a60697f4508576b"
        ),
        "release_enforced": True,
    },
}
DFX_DIAGNOSTIC_BLOCKERS = {
    "expert_aic_duration_release_failed",
    "expert_routed_compute_coverage_failed",
    "expert_activation_aiv_release_failed",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--seal-runtime-evidence",
        action="store_true",
        help=(
            "add marker/key hashes to a legacy normal validation only when "
            "all pre-existing fields match"
        ),
    )
    parser.add_argument(
        "--matched-policy",
        default="",
        help=(
            "optional matched-source DFX policy; defaults to "
            f"${MATCHED_DFX_POLICY_ENV}"
        ),
    )
    parser.add_argument(
        "--seal-authority",
        default="",
        help=(
            "pinned normal-capture authority manifest required by "
            "--seal-runtime-evidence"
        ),
    )
    parser.add_argument(
        "--seal-authority-sha256",
        default="",
        help=(
            "trusted SHA256 of --seal-authority; the manifest is not "
            "self-authorizing"
        ),
    )
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    _require(path.is_file(), f"missing artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _hex_sha(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value)
    )


def _safe_identifier(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value)
    )


def _strict_equal(
    actual: object,
    expected: object,
    *,
    path: str = "$",
) -> None:
    _require(
        type(actual) is type(expected),
        f"type mismatch at {path}: "
        f"{type(actual).__name__} != {type(expected).__name__}",
    )
    if isinstance(expected, dict):
        _require(
            set(actual) == set(expected),
            f"key mismatch at {path}",
        )
        for key in sorted(expected):
            _strict_equal(
                actual[key],
                expected[key],
                path=f"{path}.{key}",
            )
        return
    if isinstance(expected, list):
        _require(
            len(actual) == len(expected),
            f"length mismatch at {path}",
        )
        for index, item in enumerate(expected):
            _strict_equal(
                actual[index],
                item,
                path=f"{path}[{index}]",
            )
        return
    _require(actual == expected, f"value mismatch at {path}")


def _validate_pre_mount_image_audit(
    run: Path,
    *,
    image_ref: str,
) -> dict[str, str]:
    """Validate the host-side audit that ran before any source was mounted."""
    log_path = run / "image_audit.log"
    invocation_path = run / "image_audit_invocation.json"
    _require(
        log_path.is_file() and not log_path.is_symlink(),
        f"missing pre-mount image audit log: {log_path}",
    )
    log = log_path.read_text(encoding="utf-8")
    _require(
        log.count("IMAGE_IMMUTABLE_AUDIT=PASS") == 1,
        f"pre-mount image audit did not pass exactly once: {log_path}",
    )
    expected_lines = (
        "[audit] attention profile: a2a3",
        "[audit] prepared swimlane reuse capability:",
        "[audit] git credential scrub: PASS",
    )
    for line in expected_lines:
        _require(
            line in log,
            f"pre-mount image audit is missing marker {line!r}",
        )
    for component, commit in (
        ("pypto", IMAGE_PYPTO_COMMIT),
        ("pypto-lib", IMAGE_PYPTO_LIB_COMMIT),
    ):
        pattern = re.compile(
            rf"(?m)^\[audit\] pin {re.escape(component)}\s+"
            rf"{re.escape(commit)}\s+clean$"
        )
        _require(
            pattern.search(log) is not None,
            f"pre-mount image audit is missing {component} pin",
        )
    build_match = re.search(r"\[audit\] build jobs: ([0-9]+)", log)
    _require(
        build_match is not None and int(build_match.group(1)) >= 1,
        "pre-mount image audit has invalid build jobs",
    )
    invocation = _json(invocation_path)
    _require(
        set(invocation)
        == {
            "audit_log_sha256",
            "image_ref",
            "passed",
            "phase",
            "schema",
            "source_mount",
        },
        "pre-mount image audit invocation keys are incomplete",
    )
    _require(
        invocation.get("schema") == IMAGE_AUDIT_SCHEMA
        and invocation.get("passed") is True
        and invocation.get("phase") == "pre-source-mount"
        and invocation.get("image_ref") == image_ref,
        "pre-mount image audit invocation contract failed",
    )
    _require(
        invocation.get("source_mount") is False,
        "pre-mount image audit source_mount must be false",
    )
    audit_sha = _sha256(log_path)
    _require(
        invocation.get("audit_log_sha256") == audit_sha,
        "pre-mount image audit log hash mismatch",
    )
    return {
        "image_audit_log_sha256": audit_sha,
        "image_audit_invocation_sha256": _sha256(invocation_path),
    }


def _validate_capability_report(
    run: Path,
    *,
    image_ref: str,
    mode: str,
) -> dict[str, str]:
    """Validate the capability probe executed inside the immutable image."""
    path = run / "capability_report.json"
    report = _json(path)
    _require(
        report.get("schema") == CAPABILITY_SCHEMA,
        f"unsupported capability report schema: {path}",
    )
    _require(
        report.get("image_ref") == image_ref,
        "capability report image identity mismatch",
    )
    _require(
        report.get("attention_profile") == IMAGE_ATTN_PROFILE
        and report.get("pypto_git_head") == IMAGE_PYPTO_COMMIT,
        "capability report attention/PyPTO identity mismatch",
    )
    image_commits = report.get("image_commits")
    _require(
        isinstance(image_commits, dict)
        and image_commits.get("pypto") == IMAGE_PYPTO_COMMIT
        and image_commits.get("pypto_lib") == IMAGE_PYPTO_LIB_COMMIT,
        "capability report image commit mismatch",
    )
    reuse = report.get("reuse_capability")
    _require(isinstance(reuse, dict), "capability reuse report is missing")
    _require(
        reuse.get("fields_available") is True
        and reuse.get("reuse_config_constructed") is True,
        "capability report cannot construct prepared swimlane reuse",
    )
    if mode == "dfx":
        _require(
            reuse.get("required") is True
            and reuse.get("environment_present") is True
            and reuse.get("environment_value") == "1",
            "formal DFX capability requirement is not enabled",
        )
    else:
        _require(
            reuse.get("required") is False
            and reuse.get("environment_value") in (None, "0", "1"),
            "normal capability requirement is malformed",
        )
    return {"capability_report_sha256": _sha256(path)}


def _validate_dfx_raw_evidence(
    batch_dir: Path,
    *,
    dfx_artifacts: Any,
) -> dict[str, Any]:
    """Validate the complete eight-rank raw DFX tree and merged traces."""
    raw_root = batch_dir / "dfx_raw"
    _require(
        raw_root.is_dir() and not raw_root.is_symlink(),
        f"missing raw DFX tree: {raw_root}",
    )
    expected_dispatches = {f"rank{rank}/d0" for rank in range(TP)}
    actual_dispatches = {
        path.relative_to(raw_root).as_posix()
        for path in raw_root.glob("rank*/d*")
        if path.is_dir()
    }
    _require(
        actual_dispatches == expected_dispatches,
        "DFX raw tree must contain exactly rank0..rank7/d0 dispatches",
    )
    dep_paths = sorted(raw_root.rglob("deps.json"))
    swim_paths = sorted(raw_root.rglob("l2_swimlane_records.json"))
    _require(
        {
            path.relative_to(raw_root).as_posix() for path in dep_paths
        }
        == {f"rank{rank}/d0/deps.json" for rank in range(TP)},
        "DFX raw deps.json set is not exactly eight ranks",
    )
    _require(
        {
            path.relative_to(raw_root).as_posix() for path in swim_paths
        }
        == {
            f"rank{rank}/d0/l2_swimlane_records.json"
            for rank in range(TP)
        },
        "DFX raw swimlane set is not exactly eight ranks",
    )

    _require(
        isinstance(dfx_artifacts, dict),
        "case report DFX artifact metadata is missing",
    )
    _require(
        dfx_artifacts.get("dep_gen_preserved_after_swim") is True,
        "DFX swim capture did not preserve dep-gen artifacts",
    )
    dep_hashes = dfx_artifacts.get("dep_gen_artifacts")
    swim_hashes = dfx_artifacts.get("swimlane_artifacts")
    _require(isinstance(dep_hashes, dict), "DFX dep artifact hashes are missing")
    _require(
        isinstance(swim_hashes, dict),
        "DFX swimlane artifact hashes are missing",
    )
    expected_dep_hashes = {
        f"dfx_outputs/{relative}": _sha256(raw_root / relative)
        for relative in (
            path.relative_to(raw_root).as_posix() for path in dep_paths
        )
    }
    expected_swim_hashes = {
        f"dfx_outputs/{relative}": _sha256(raw_root / relative)
        for relative in (
            path.relative_to(raw_root).as_posix() for path in swim_paths
        )
    }
    _require(
        dep_hashes == expected_dep_hashes,
        "DFX dep artifact hash map does not match the raw tree",
    )
    _require(
        swim_hashes == expected_swim_hashes,
        "DFX swimlane artifact hash map does not match the raw tree",
    )

    by_rank: dict[str, dict[str, str]] = {}
    for rank in range(TP):
        dispatch = raw_root / f"rank{rank}" / "d0"
        required = {
            name: dispatch / name
            for name in (
                "deps.json",
                "name_map.json",
                "l2_swimlane_records.json",
                "critical_path_report.md",
            )
        }
        for name, path in required.items():
            _require(
                path.is_file() and not path.is_symlink(),
                f"DFX raw rank{rank} missing {name}",
            )
        merged = sorted(
            path
            for path in dispatch.glob("merged_swimlane_*.json")
            if path.is_file() and not path.is_symlink()
        )
        _require(
            len(merged) == 1
            and MERGED_SWIMLANE_RE.fullmatch(merged[0].name) is not None,
            f"DFX raw rank{rank} must contain exactly one merged swimlane",
        )
        merged_value = json.loads(merged[0].read_text(encoding="utf-8"))
        trace_events = (
            merged_value.get("traceEvents")
            if isinstance(merged_value, dict)
            else None
        )
        _require(
            isinstance(trace_events, list) and bool(trace_events),
            f"DFX raw rank{rank} merged swimlane is malformed",
        )
        timed_events = [
            event
            for event in trace_events
            if isinstance(event, dict)
            and event.get("ph") == "X"
            and isinstance(event.get("name"), str)
            and bool(event["name"])
            and isinstance(event.get("ts"), (int, float))
            and isinstance(event.get("dur"), (int, float))
            and event["dur"] > 0
        ]
        _require(
            bool(timed_events),
            f"DFX raw rank{rank} merged swimlane has no timed task events",
        )
        by_rank[str(rank)] = {
            name: _sha256(path) for name, path in required.items()
        }
        by_rank[str(rank)]["merged_swimlane"] = {
            "name": merged[0].name,
            "sha256": _sha256(merged[0]),
        }
    return {
        "schema": DFX_RAW_SCHEMA,
        "rank_dispatches": sorted(expected_dispatches),
        "rank_count": TP,
        "merged_swimlane_count": TP,
        "sha256_by_rank": by_rank,
    }


def _run_evidence_sha256(run: Path) -> dict[str, str]:
    evidence: dict[str, str] = {}
    validation = run / "artifact_validation.json"
    for path in sorted(run.rglob("*")):
        _require(
            not path.is_symlink(),
            f"seal evidence must not contain symlinks: {path}",
        )
        if path == validation or path.is_dir():
            continue
        _require(
            path.is_file(),
            f"seal evidence contains a non-regular path: {path}",
        )
        relative = path.relative_to(run).as_posix()
        evidence[relative] = _sha256(path)
    return evidence


def _seal_authority_record(
    *,
    run: Path,
    authority_path: str,
    authority_sha256: str,
) -> dict[str, Any]:
    raw_path = Path(authority_path)
    _require(
        raw_path.is_file() and not raw_path.is_symlink(),
        f"missing regular seal authority: {raw_path}",
    )
    path = raw_path.resolve()
    _require(
        _hex_sha(authority_sha256),
        "invalid trusted seal authority SHA256",
    )
    _require(
        _sha256(path) == authority_sha256,
        "seal authority SHA256 mismatch",
    )
    value = _json(path)
    _require(
        set(value) == {"schema", "campaign_root", "runs"},
        "seal authority keys are incomplete",
    )
    _require(
        value.get("schema") == NORMAL_SEAL_AUTHORITY_SCHEMA,
        "unsupported seal authority schema",
    )
    _require(
        value.get("campaign_root") == str(run.parent.parent.resolve()),
        "seal authority campaign root mismatch",
    )
    runs = value.get("runs")
    _require(isinstance(runs, dict), "seal authority runs are missing")
    record = runs.get(run.name)
    _require(
        isinstance(record, dict),
        f"seal authority does not cover {run.name}",
    )
    _require(
        set(record)
        == {
            "legacy_validation_sha256",
            "sealed_validation_sha256",
            "evidence_sha256",
        },
        f"seal authority record is malformed: {run.name}",
    )
    for key in ("legacy_validation_sha256", "sealed_validation_sha256"):
        _require(
            _hex_sha(record.get(key)),
            f"invalid seal authority {run.name}.{key}",
        )
    evidence = record.get("evidence_sha256")
    _require(
        isinstance(evidence, dict) and bool(evidence),
        f"seal authority evidence is empty: {run.name}",
    )
    for relative, digest in evidence.items():
        _require(
            isinstance(relative, str)
            and bool(relative)
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            f"invalid seal authority evidence path: {relative!r}",
        )
        _require(
            _hex_sha(digest),
            f"invalid seal authority evidence hash: {relative}",
        )
    actual_evidence = _run_evidence_sha256(run)
    _strict_equal(
        actual_evidence,
        evidence,
        path=f"$.runs.{run.name}.evidence_sha256",
    )
    return record


def _evidence_reference(
    value: object,
    *,
    name: str,
) -> dict[str, str]:
    _require(isinstance(value, dict), f"invalid authority.{name}")
    _require(
        set(value) == {"path", "sha256"},
        f"authority.{name} must contain path and sha256",
    )
    path = value.get("path")
    digest = value.get("sha256")
    _require(
        isinstance(path, str) and bool(path.strip()),
        f"invalid authority.{name}.path",
    )
    _require(_hex_sha(digest), f"invalid authority.{name}.sha256")
    return {"path": path, "sha256": digest}


def _matched_authority(value: object) -> dict[str, Any]:
    _require(isinstance(value, dict), "matched policy authority is missing")
    expected_keys = {
        "image_ref",
        "image_pypto_commit",
        "selection_report",
        "source_audit",
        "scripts_manifest",
        "normal_capture_scripts_manifest",
        "normal_seal_authority",
        "normal_correctness_report",
        "normal_performance_report",
        "normal_campaign_spec",
        "normal_counterbalance_spec",
        "analyzer_sha256",
        "validator_sha256",
        "golden_manifests",
        "workload",
        "capture",
    }
    _require(
        set(value) == expected_keys,
        "matched policy authority keys are incomplete",
    )
    image_ref = value.get("image_ref")
    _require(
        isinstance(image_ref, str)
        and re.fullmatch(r".+@sha256:[0-9a-f]{64}", image_ref) is not None,
        "invalid authority.image_ref",
    )
    _require(
        value.get("image_pypto_commit") == IMAGE_PYPTO_COMMIT,
        "authority image PyPTO commit mismatch",
    )
    for key in ("analyzer_sha256", "validator_sha256"):
        _require(_hex_sha(value.get(key)), f"invalid authority.{key}")

    golden = value.get("golden_manifests")
    _require(
        isinstance(golden, dict)
        and set(golden) == {str(batch) for batch in (1, 2, 4, 7, 8, 16)},
        "authority golden manifests must cover the six-BS matrix",
    )
    normalized_golden = {
        batch: _evidence_reference(item, name=f"golden_manifests.{batch}")
        for batch, item in golden.items()
    }

    workload = value.get("workload")
    _require(
        workload
        == {
            "layers": [0, 1, 2, 3, 4],
            "batches": [1, 2, 4, 7, 8, 16],
            "context_len_per_sequence": CONTEXT_LEN,
            "blocks_per_sequence": BLOCKS_PER_SEQUENCE,
            "hidden_outputs": ["hidden_l3", "hidden_l4"],
            "hidden_exact": True,
        },
        "authority workload contract mismatch",
    )
    capture = value.get("capture")
    _require(isinstance(capture, dict), "authority capture is missing")
    expected_capture_keys = {
        "normal_rounds",
        "normal_iters",
        "normal_warmup",
        "counterbalanced",
        "dfx_rounds",
        "dfx_iters",
        "dfx_warmup",
        "l2_swimlane_reuse_dep_gen",
    }
    _require(
        set(capture) == expected_capture_keys,
        "authority capture keys are incomplete",
    )
    _require(
        isinstance(capture.get("normal_rounds"), int)
        and capture["normal_rounds"] >= 3,
        "authority normal rounds must be at least three",
    )
    _require(
        isinstance(capture.get("normal_iters"), int)
        and capture["normal_iters"] >= 30,
        "authority normal iterations must be at least 30",
    )
    _require(
        isinstance(capture.get("normal_warmup"), int)
        and capture["normal_warmup"] >= 5,
        "authority normal warmup must be at least five",
    )
    _require(
        capture.get("counterbalanced") is True,
        "authority normal campaign must be counterbalanced",
    )
    _require(
        capture.get("dfx_rounds") == 1
        and isinstance(capture.get("dfx_iters"), int)
        and capture["dfx_iters"] >= 1
        and isinstance(capture.get("dfx_warmup"), int)
        and capture["dfx_warmup"] >= 2,
        "authority DFX capture contract mismatch",
    )
    _require(
        capture.get("l2_swimlane_reuse_dep_gen") is True,
        "authority requires L2 swimlane dep-gen reuse",
    )

    return {
        "image_ref": image_ref,
        "image_pypto_commit": value["image_pypto_commit"],
        "selection_report": _evidence_reference(
            value.get("selection_report"),
            name="selection_report",
        ),
        "source_audit": _evidence_reference(
            value.get("source_audit"),
            name="source_audit",
        ),
        "scripts_manifest": _evidence_reference(
            value.get("scripts_manifest"),
            name="scripts_manifest",
        ),
        "normal_capture_scripts_manifest": _evidence_reference(
            value.get("normal_capture_scripts_manifest"),
            name="normal_capture_scripts_manifest",
        ),
        "normal_seal_authority": _evidence_reference(
            value.get("normal_seal_authority"),
            name="normal_seal_authority",
        ),
        "normal_correctness_report": _evidence_reference(
            value.get("normal_correctness_report"),
            name="normal_correctness_report",
        ),
        "normal_performance_report": _evidence_reference(
            value.get("normal_performance_report"),
            name="normal_performance_report",
        ),
        "normal_campaign_spec": _evidence_reference(
            value.get("normal_campaign_spec"),
            name="normal_campaign_spec",
        ),
        "normal_counterbalance_spec": _evidence_reference(
            value.get("normal_counterbalance_spec"),
            name="normal_counterbalance_spec",
        ),
        "analyzer_sha256": value["analyzer_sha256"],
        "validator_sha256": value["validator_sha256"],
        "golden_manifests": normalized_golden,
        "workload": dict(workload),
        "capture": dict(capture),
    }


def load_matched_dfx_policy(path: str | Path) -> dict[str, Any]:
    policy_path = Path(path).resolve()
    value = _json(policy_path)
    _require(
        value.get("schema") == MATCHED_DFX_POLICY_SCHEMA,
        f"{policy_path}: unsupported matched DFX policy schema",
    )
    sources = value.get("sources")
    _require(
        isinstance(sources, dict)
        and set(sources) == set(MATCHED_DFX_SOURCE_KINDS),
        f"{policy_path}: sources must be exactly baseline and candidate",
    )

    normalized: dict[str, dict[str, Any]] = {}
    for source_kind in MATCHED_DFX_SOURCE_KINDS:
        source = sources.get(source_kind)
        _require(
            isinstance(source, dict),
            f"{policy_path}: missing sources.{source_kind}",
        )
        _require(
            _safe_identifier(source.get("profile")),
            f"{policy_path}: invalid sources.{source_kind}.profile",
        )
        expected_scalars = {
            "source_kind": source_kind,
            "source_role": source_kind,
        }
        for key, expected in expected_scalars.items():
            _require(
                source.get(key) == expected,
                f"{policy_path}: sources.{source_kind}.{key} "
                f"must be {expected!r}",
            )
        _require(
            _safe_identifier(source.get("policy_id")),
            f"{policy_path}: invalid sources.{source_kind}.policy_id",
        )
        for key in ("source_manifest_sha256", "decode_fwd_sha256"):
            _require(
                _hex_sha(source.get(key)),
                f"{policy_path}: invalid sources.{source_kind}.{key}",
            )
        _require(
            isinstance(source.get("release_enforced"), bool),
            f"{policy_path}: sources.{source_kind}.release_enforced "
            "must be boolean",
        )
        if source_kind == "candidate":
            _require(
                source["release_enforced"] is True,
                f"{policy_path}: candidate release gate must be enforced",
            )
        normalized[source_kind] = dict(source)

    _require(
        normalized["baseline"]["policy_id"]
        != normalized["candidate"]["policy_id"],
        f"{policy_path}: baseline/candidate policy IDs must differ",
    )
    return {
        "schema": MATCHED_DFX_POLICY_SCHEMA,
        "policy_path": str(policy_path),
        "policy_sha256": _sha256(policy_path),
        "authority": _matched_authority(value.get("authority")),
        "sources": normalized,
    }


def resolve_dfx_policy(
    *,
    source_kind: str,
    profile: str,
    matched_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    if matched_policy is None:
        _require(
            profile in DFX_POLICIES,
            f"unsupported DFX profile: {profile}",
        )
        return DFX_POLICIES[profile]

    policy = matched_policy["sources"][source_kind]
    expected_profile = policy["profile"]
    _require(
        profile == expected_profile,
        f"matched DFX profile for {source_kind} must be "
        f"{expected_profile!r}, got {profile!r}",
    )
    _require(
        policy.get("source_kind") == source_kind,
        "matched DFX policy source kind mismatch",
    )
    return policy


def validate_matched_policy_binding(
    *,
    source_kind: str,
    profile: str,
    policy: dict[str, Any],
    source: dict[str, Any],
    source_policy: dict[str, Any],
    expert_release: dict[str, Any],
    admission: dict[str, Any],
) -> None:
    _require(
        profile == policy["profile"],
        "matched DFX report profile does not match policy",
    )
    _require(
        policy.get("source_kind") == source_kind,
        "matched DFX source kind does not match policy",
    )
    _require(
        source.get("source_manifest_sha256")
        == policy["source_manifest_sha256"],
        "matched DFX source manifest does not match policy",
    )
    _require(
        source.get("decode_fwd_sha256") == policy["decode_fwd_sha256"],
        "matched DFX decode hash does not match policy",
    )
    expected_source_policy = {
        "policy_id": policy["policy_id"],
        "source_role": policy["source_role"],
        "enforce_candidate_release_gate": policy["release_enforced"],
        "decode_sha256_prefix": policy["decode_fwd_sha256"][:8],
    }
    for key, expected in expected_source_policy.items():
        _require(
            source_policy.get(key) == expected,
            f"matched DFX source policy {key} does not match policy",
        )
    _require(
        expert_release.get("release_enforced")
        is policy["release_enforced"],
        "matched DFX expert release enforcement does not match policy",
    )
    _require(
        expert_release.get("profile") == profile,
        "matched DFX expert release profile does not match policy",
    )
    _require(
        admission.get("profile") == profile,
        "matched DFX admission profile does not match policy",
    )
    _require(
        admission.get("expert_release_enforced")
        is policy["release_enforced"],
        "matched DFX admission release enforcement does not match policy",
    )
    for contract_name, contract in (
        ("expert release", expert_release),
        ("admission", admission),
    ):
        nested_policy = contract.get("source_policy")
        _require(
            isinstance(nested_policy, dict),
            f"matched DFX {contract_name} source policy is missing",
        )
        for key, expected in expected_source_policy.items():
            _require(
                nested_policy.get(key) == expected,
                f"matched DFX {contract_name} source policy {key} "
                "does not match",
            )


def _write_stable(
    path: Path,
    value: dict[str, Any],
    *,
    seal_runtime_evidence: bool = False,
    seal_authority_record: dict[str, Any] | None = None,
) -> None:
    content_bytes = _canonical_json_bytes(value)
    content = content_bytes.decode("utf-8")
    if seal_runtime_evidence:
        _require(
            seal_authority_record is not None,
            "runtime evidence seal requires a pinned authority record",
        )
        _require(
            path.is_file() and not path.is_symlink(),
            f"runtime evidence seal requires an existing regular file: {path}",
        )
        existing_bytes = path.read_bytes()
        existing = _json(path)
        legacy = dict(value)
        for field in RUNTIME_SEAL_FIELDS:
            _require(field in legacy, f"missing runtime seal field: {field}")
            legacy.pop(field)
        legacy_bytes = _canonical_json_bytes(legacy)
        expected_legacy_sha = seal_authority_record[
            "legacy_validation_sha256"
        ]
        expected_sealed_sha = seal_authority_record[
            "sealed_validation_sha256"
        ]
        _require(
            _sha256_bytes(legacy_bytes) == expected_legacy_sha,
            f"recomputed legacy validation is not authority-bound: {path}",
        )
        _require(
            _sha256_bytes(content_bytes) == expected_sealed_sha,
            f"recomputed sealed validation is not authority-bound: {path}",
        )
        existing_sha = _sha256_bytes(existing_bytes)
        if existing_sha == expected_legacy_sha:
            _strict_equal(existing, legacy)
            path.write_text(content, encoding="utf-8")
            return
        if existing_sha == expected_sealed_sha:
            _strict_equal(existing, value)
            return
        # Compare the object before the hash error so type confusion cannot
        # be hidden behind Python's bool/int equality semantics.
        try:
            _strict_equal(existing, legacy)
        except AssertionError as exc:
            raise AssertionError(
                f"refusing non-additive validation seal: {path}: {exc}"
            ) from exc
        raise AssertionError(
            f"refusing non-additive validation seal: {path}"
        )
    if path.exists():
        _require(
            path.read_text(encoding="utf-8") == content,
            f"refusing to replace different validation evidence: {path}",
        )
        return
    path.write_text(content, encoding="utf-8")


def _validate_map(
    path: Path,
    *,
    rank: int,
    batch: int,
    nonce: str,
) -> str:
    value = _json(path)
    scheduler_blocks = batch * BLOCKS_PER_SEQUENCE
    physical_blocks = scheduler_blocks + STORAGE_BATCH - 1
    slots_per_layer = physical_blocks * BLOCK_SIZE
    bytes_per_layer = slots_per_layer * HEAD_DIM * BF16_BYTES
    pool_bytes = 2 * NUM_LAYERS * bytes_per_layer

    expected_scalars = {
        "version": 3,
        "rank": rank,
        "tp_world_size": TP,
        "num_layers": NUM_LAYERS,
        "scheduler_num_blocks": scheduler_blocks,
        "physical_num_blocks": physical_blocks,
        "padding_block_count": STORAGE_BATCH - 1,
        "reserve_start": scheduler_blocks,
        "block_size": BLOCK_SIZE,
        "num_kv_heads": 1,
        "head_dim": HEAD_DIM,
        "dtype": "bfloat16",
        "pool_bytes": pool_bytes,
    }
    for key, expected in expected_scalars.items():
        _require(
            value.get(key) == expected,
            f"{path}: {key}={value.get(key)!r}, expected={expected!r}",
        )
    _require(
        value.get("padding_block_ids")
        == list(range(scheduler_blocks, physical_blocks)),
        f"{path}: invalid padding block IDs",
    )
    _require(
        value.get("groups")
        == [{"group_id": 0, "layer_indices": list(range(NUM_LAYERS))}],
        f"{path}: invalid five-layer group",
    )
    session = value.get("ipc_session")
    _require(isinstance(session, dict), f"{path}: missing ipc_session")
    _require(
        session.get("session_nonce") == nonce,
        f"{path}: IPC nonce mismatch",
    )
    _require(
        session.get("producer_rank") == rank,
        f"{path}: producer rank mismatch",
    )

    entries = value.get("map")
    _require(isinstance(entries, dict), f"{path}: missing map")
    expected_keys = {
        f"L{layer}.{which}"
        for layer in range(NUM_LAYERS)
        for which in ("K", "V")
    }
    _require(
        set(entries) == expected_keys,
        f"{path}: map keys do not describe exactly L0-L4 K/V",
    )
    for layer in range(NUM_LAYERS):
        for section_index, which in enumerate(("K", "V")):
            key = f"L{layer}.{which}"
            entry = entries[key]
            _require(isinstance(entry, dict), f"{path}: invalid {key}")
            expected_entry = {
                "layer_idx": layer,
                "which": which,
                "group_id": 0,
                "num_blocks": physical_blocks,
                "block_size": BLOCK_SIZE,
                "num_slots": slots_per_layer,
                "num_kv_heads": 1,
                "head_dim": HEAD_DIM,
                "dtype": "bfloat16",
                "nbytes": bytes_per_layer,
                "shape": [physical_blocks, BLOCK_SIZE, 1, HEAD_DIM],
                "flat_shape": [slots_per_layer, HEAD_DIM],
                "offset": (
                    section_index * NUM_LAYERS * bytes_per_layer
                    + layer * bytes_per_layer
                ),
            }
            for field, expected in expected_entry.items():
                _require(
                    entry.get(field) == expected,
                    f"{path}: {key}.{field}={entry.get(field)!r}, "
                    f"expected={expected!r}",
                )
    _require(
        value.get("sections")
        == {
            "K": {
                "offset": 0,
                "nbytes": NUM_LAYERS * bytes_per_layer,
            },
            "V": {
                "offset": NUM_LAYERS * bytes_per_layer,
                "nbytes": NUM_LAYERS * bytes_per_layer,
            },
        },
        f"{path}: invalid K/V section layout",
    )
    return _sha256(path)


def _validate_runtime_markers(
    runtime: Path,
    *,
    rank: int,
    batch: int,
    nonce: str,
) -> dict[str, str]:
    map_name = f"pypto_kvpool_map.json.rank{rank}"
    key_name = f"pypto_kvpool.key.rank{rank}"
    done_name = f"{map_name}.done"
    ready_name = f"ready.rank{rank}"
    map_path = runtime / map_name
    key_path = runtime / key_name
    done_path = runtime / done_name
    ready_path = runtime / ready_name
    map_value = _json(map_path)
    done = _json(done_path)
    ready = _json(ready_path)

    map_session = map_value.get("ipc_session")
    _require(
        isinstance(map_session, dict),
        f"{map_path}: missing ipc_session",
    )
    _require(
        done.get("ipc_session") == map_session,
        f"{done_path}: ipc_session does not match KV map",
    )
    _require(
        map_session.get("session_nonce") == nonce,
        f"{done_path}: IPC nonce mismatch",
    )
    _require(
        map_session.get("producer_rank") == rank,
        f"{done_path}: producer rank mismatch",
    )

    expected_map_path = f"/out/runtime/{map_name}"
    expected_key_path = f"/out/runtime/{key_name}"
    expected_done_path = f"/out/runtime/{done_name}"
    _require(
        done.get("ready_schema_version") == 1,
        f"{done_path}: ready schema mismatch",
    )
    _require(
        done.get("key_path") == expected_key_path,
        f"{done_path}: key path mismatch",
    )
    _require(
        done.get("map_path") == expected_map_path,
        f"{done_path}: map path mismatch",
    )
    _require(
        key_path.is_file() and key_path.stat().st_size > 0,
        f"{key_path}: missing or empty KV key artifact",
    )

    ready_kv = ready.get("kv")
    _require(
        ready.get("rank") == rank,
        f"{ready_path}: exporter rank mismatch",
    )
    _require(
        isinstance(ready_kv, dict),
        f"{ready_path}: missing KV readiness",
    )
    scheduler_blocks = batch * BLOCKS_PER_SEQUENCE
    physical_blocks = scheduler_blocks + STORAGE_BATCH - 1
    pool_bytes = (
        2
        * NUM_LAYERS
        * physical_blocks
        * BLOCK_SIZE
        * HEAD_DIM
        * BF16_BYTES
    )
    expected_ready = {
        "ok": True,
        "rank": rank,
        "num_layers": NUM_LAYERS,
        "scheduler_num_blocks": scheduler_blocks,
        "physical_num_blocks": physical_blocks,
        "pool_bytes": pool_bytes,
        "map_path": expected_map_path,
        "ready_path": expected_done_path,
    }
    for key, expected in expected_ready.items():
        _require(
            ready_kv.get(key) == expected,
            f"{ready_path}: kv.{key}={ready_kv.get(key)!r}, "
            f"expected={expected!r}",
        )

    return {
        key_name: _sha256(key_path),
        done_name: _sha256(done_path),
        ready_name: _sha256(ready_path),
    }


def main() -> int:
    args = _parse_args()
    run = Path(args.run).resolve()
    match = RUN_RE.fullmatch(run.name)
    _require(match is not None, f"invalid run name: {run.name}")
    source_kind, round_text, mode, batch_text = match.groups()
    round_id = int(round_text)
    batch = int(batch_text)
    runtime = run / "runtime"
    batch_dir = runtime / f"bs{batch}"
    _require(
        not args.seal_runtime_evidence or mode == "normal",
        "runtime evidence sealing is only allowed for normal captures",
    )
    _require(
        not args.seal_runtime_evidence
        or (bool(args.seal_authority) and bool(args.seal_authority_sha256)),
        "runtime evidence sealing requires authority path and trusted SHA256",
    )
    _require(
        args.seal_runtime_evidence
        or (not args.seal_authority and not args.seal_authority_sha256),
        "seal authority arguments require --seal-runtime-evidence",
    )

    _require(
        (run / "container.rc").read_text(encoding="utf-8").strip()
        == "0",
        f"{run}: container did not exit cleanly",
    )
    image_ref = (run / "image_ref.txt").read_text(
        encoding="utf-8"
    ).strip()
    nonce = (run / "run_nonce.txt").read_text(
        encoding="utf-8"
    ).strip()
    _require(_hex_sha(nonce), f"{run}: invalid run nonce")
    _require(
        image_ref.startswith(
            "hub.i.basemind.com/stepcast/vllm-pypto@sha256:"
        ),
        f"{run}: image is not digest-pinned",
    )
    pre_mount_audit = _validate_pre_mount_image_audit(
        run,
        image_ref=image_ref,
    )
    capability = _validate_capability_report(
        run,
        image_ref=image_ref,
        mode=mode,
    )

    matrix = _json(runtime / "matrix_report.json")
    report = _json(batch_dir / "report.json")
    expected_identity = {
        "source_kind": source_kind,
        "round": round_id,
        "mode": mode,
        "image_ref": image_ref,
        "image_pypto_commit": IMAGE_PYPTO_COMMIT,
    }
    for artifact_name, artifact in (
        ("matrix", matrix),
        ("report", report),
    ):
        for key, expected in expected_identity.items():
            _require(
                artifact.get(key) == expected,
                f"{artifact_name}: {key} identity mismatch",
            )
    _require(matrix.get("batches") == [batch], "matrix is not single-BS")
    _require(
        matrix.get("context_len_per_sequence") == CONTEXT_LEN,
        "matrix context contract mismatch",
    )
    _require(
        matrix.get("blocks_per_sequence") == BLOCKS_PER_SEQUENCE,
        "matrix block contract mismatch",
    )
    _require(
        matrix.get("reports") == {str(batch): f"bs{batch}/report.json"},
        "matrix report path mismatch",
    )

    scheduler_blocks = batch * BLOCKS_PER_SEQUENCE
    physical_blocks = scheduler_blocks + STORAGE_BATCH - 1
    expected_workload = {
        "active_batch": batch,
        "context_len_per_sequence": CONTEXT_LEN,
        "blocks_per_sequence": BLOCKS_PER_SEQUENCE,
        "active_total_context_tokens": batch * CONTEXT_LEN,
        "allocated_scheduler_blocks": scheduler_blocks,
        "allocated_physical_blocks": physical_blocks,
        "kv_num_layers": NUM_LAYERS,
        "allocated_kv_rows_per_rank": (
            NUM_LAYERS * physical_blocks * BLOCK_SIZE
        ),
        "allocated_kv_pool_bytes_per_rank": (
            2
            * NUM_LAYERS
            * physical_blocks
            * BLOCK_SIZE
            * HEAD_DIM
            * BF16_BYTES
        ),
    }
    workload = report.get("workload")
    _require(isinstance(workload, dict), "missing workload report")
    for key, expected in expected_workload.items():
        _require(
            workload.get(key) == expected,
            f"workload {key}={workload.get(key)!r}, expected={expected!r}",
        )
    _require(
        len(workload.get("input_tokens", [])) == batch,
        "input token count does not match active batch",
    )

    source = report.get("source")
    _require(isinstance(source, dict), "missing source provenance")
    for key in (
        "source_manifest_sha256",
        "decode_fwd_sha256",
        "five_layer_program_sha256",
        "five_layer_holder_sha256",
    ):
        _require(_hex_sha(source.get(key)), f"invalid source hash: {key}")

    hidden_hashes: dict[str, str] = {}
    health = report.get("health")
    _require(isinstance(health, dict), "missing hidden health")
    for name in ("hidden_l3", "hidden_l4"):
        path = batch_dir / f"{name}.pt"
        digest = _sha256(path)
        hidden_hashes[path.name] = digest
        _require(
            report.get("files", {}).get(path.name) == digest,
            f"{path}: report hash mismatch",
        )
        item = health.get(name)
        _require(isinstance(item, dict), f"missing health for {name}")
        _require(
            item.get("shape") == [TP, batch, 4096],
            f"{name}: shape mismatch",
        )
        _require(item.get("dtype") == "torch.bfloat16", f"{name}: dtype")
        _require(item.get("finite") is True, f"{name}: non-finite")
        _require(item.get("tp_spread_max") == 0.0, f"{name}: TP spread")
        _require(
            item.get("nonzero_rank_rows")
            == item.get("expected_nonzero_rank_rows")
            == TP * batch,
            f"{name}: missing nonzero active rank/rows",
        )

    dfx_gate: dict[str, Any] = {}
    dfx_raw_evidence: dict[str, Any] = {}
    if mode == "normal":
        _require(report.get("comparisons") == {}, "normal run used golden")
        _require(report.get("dfx") == {}, "normal run unexpectedly has DFX")
    else:
        matched_policy_text = (
            args.matched_policy
            or os.environ.get(MATCHED_DFX_POLICY_ENV, "")
        )
        matched_policy = (
            load_matched_dfx_policy(matched_policy_text)
            if matched_policy_text
            else None
        )
        comparisons = report.get("comparisons")
        _require(
            isinstance(comparisons, dict)
            and set(comparisons) == {"hidden_l3", "hidden_l4"}
            and all(
                isinstance(item, dict) and item.get("exact") is True
                for item in comparisons.values()
            ),
            "DFX hidden outputs are not exact to frozen golden",
        )
        _require(
            (batch_dir / "dfx_analysis" / "moe_dfx_report.json").is_file(),
            "missing DFX JSON report",
        )
        _require(
            (
                batch_dir
                / "dfx_analysis"
                / "moe_critical_path_report.md"
            ).is_file(),
            "missing DFX critical-path report",
        )
        dfx_report = _json(
            batch_dir / "dfx_analysis" / "moe_dfx_report.json"
        )
        dfx_raw_evidence = _validate_dfx_raw_evidence(
            batch_dir,
            dfx_artifacts=report.get("dfx"),
        )
        _require(
            str(dfx_report.get("schema", "")).startswith(
                "step3p5.five-layer-moe-dfx.v"
            ),
            "unsupported DFX report schema",
        )
        dfx_profile = os.environ.get("MOE_DFX_PROFILE", source_kind)
        policy = resolve_dfx_policy(
            source_kind=source_kind,
            profile=dfx_profile,
            matched_policy=matched_policy,
        )
        _require(
            policy.get("source_kind", source_kind) == source_kind,
            "DFX profile does not match the matrix source kind",
        )
        source_policy = dfx_report.get("source_policy")
        _require(
            isinstance(source_policy, dict),
            "DFX report is missing source policy",
        )
        _require(
            dfx_report.get("profile") == dfx_profile
            and source_policy.get("source_role")
            == policy.get("source_role", source_kind),
            "DFX source profile mismatch",
        )
        _require(
            source_policy.get("policy_id") == policy["policy_id"],
            "DFX source policy ID mismatch",
        )
        _require(
            source.get("decode_fwd_sha256")
            == policy["decode_fwd_sha256"],
            "DFX source decode hash does not match frozen policy",
        )
        _require(
            bool(source_policy.get("enforce_candidate_release_gate"))
            == policy["release_enforced"],
            "DFX release policy mismatch",
        )

        rank_contract = dfx_report.get("rank_contract")
        expected_rank_tags = [f"rank{rank}/d0" for rank in range(TP)]
        _require(
            isinstance(rank_contract, dict)
            and rank_contract.get("exact") is True
            and rank_contract.get("expected")
            == rank_contract.get("actual")
            == expected_rank_tags,
            "DFX rank contract failed",
        )
        structural = dfx_report.get("structural_contracts")
        _require(
            isinstance(structural, dict)
            and structural.get("pass") is True,
            "DFX structural contract failed",
        )
        expected_ranks = set(expected_rank_tags)
        for name in ("task_id", "combine_dependency"):
            contracts = structural.get(name)
            _require(
                isinstance(contracts, dict)
                and set(contracts) == expected_ranks
                and all(
                    isinstance(item, dict) and item.get("pass") is True
                    for item in contracts.values()
                ),
                f"DFX {name} contract failed",
            )
        slice_contract = dfx_report.get("slice_contract")
        _require(
            isinstance(slice_contract, dict)
            and slice_contract.get("expected_equals_observed") is True
            and slice_contract.get("errors") == [],
            "DFX physical slice contract failed",
        )
        routed_profiles = dfx_report.get("routed_slice_profiles")
        _require(
            isinstance(routed_profiles, dict)
            and routed_profiles.get("pass") is True,
            "DFX routed physical-slice profile failed",
        )

        expert_release = dfx_report.get("expert_kernel_release")
        admission = dfx_report.get("admission")
        _require(
            isinstance(expert_release, dict)
            and isinstance(admission, dict),
            "DFX release contracts are missing",
        )
        _require(
            expert_release.get("release_enforced")
            is policy["release_enforced"],
            "DFX expert release enforcement mismatch",
        )
        if matched_policy is not None:
            validate_matched_policy_binding(
                source_kind=source_kind,
                profile=str(dfx_report.get("profile")),
                policy=policy,
                source=source,
                source_policy=source_policy,
                expert_release=expert_release,
                admission=admission,
            )
        blocker_codes = {
            str(item.get("code"))
            for item in admission.get("blockers", [])
            if isinstance(item, dict)
        }
        _require(
            not blocker_codes
            or (
                policy["release_enforced"]
                and blocker_codes <= DFX_DIAGNOSTIC_BLOCKERS
            ),
            f"DFX analyzer integrity blockers: {sorted(blocker_codes)}",
        )
        release_readiness = admission.get("release_readiness")
        _require(
            isinstance(release_readiness, dict)
            and release_readiness.get("publication_allowed") is False,
            "DFX publication status is malformed",
        )
        dfx_gate = {
            "profile": dfx_profile,
            "policy_id": policy["policy_id"],
            "analyzer_gate_pass": admission.get("analyzer_gate_pass"),
            "analyzer_blocker_codes": sorted(blocker_codes),
            "expert_release_gate_pass": (
                expert_release.get("release_gate_pass")
                if policy["release_enforced"]
                else None
            ),
            "expert_release_gate_status": (
                expert_release.get("release_gate_status")
                if policy["release_enforced"]
                else "NOT_APPLICABLE"
            ),
            "release_readiness_status": release_readiness.get("status"),
            "recv_meta_publication_evidence_ready": (
                release_readiness.get(
                    "recv_meta_publication_evidence_ready"
                )
            ),
        }
        if matched_policy is not None:
            dfx_gate["matched_policy"] = {
                "schema": matched_policy["schema"],
                "policy_sha256": matched_policy["policy_sha256"],
                "policy_id": policy["policy_id"],
                "profile": policy["profile"],
                "source_kind": policy["source_kind"],
                "source_role": policy["source_role"],
                "source_manifest_sha256": policy[
                    "source_manifest_sha256"
                ],
                "decode_fwd_sha256": policy["decode_fwd_sha256"],
                "release_enforced": policy["release_enforced"],
            }

    map_hashes = {}
    runtime_hashes = {}
    for rank in range(TP):
        map_path = runtime / f"pypto_kvpool_map.json.rank{rank}"
        map_hashes[str(rank)] = _validate_map(
            map_path,
            rank=rank,
            batch=batch,
            nonce=nonce,
        )
        runtime_hashes.update(
            _validate_runtime_markers(
                runtime,
                rank=rank,
                batch=batch,
                nonce=nonce,
            )
        )

    result = {
        "schema": "step3p5.five-layer-moe-case-validation.v1",
        "passed": True,
        "run": run.name,
        "source_kind": source_kind,
        "round": round_id,
        "mode": mode,
        "active_batch": batch,
        "context_len_per_sequence": CONTEXT_LEN,
        "active_total_context_tokens": batch * CONTEXT_LEN,
        "image_ref": image_ref,
        "run_nonce": nonce,
        "pre_mount_image_audit": pre_mount_audit,
        "capability_report": capability,
        "workload": expected_workload,
        "hidden_sha256": hidden_hashes,
        "kv_map_sha256_by_rank": map_hashes,
        "runtime_marker_sha256": {
            name: digest
            for name, digest in runtime_hashes.items()
            if not name.startswith("pypto_kvpool.key.")
        },
        "kv_key_sha256_by_rank": {
            str(rank): runtime_hashes[
                f"pypto_kvpool.key.rank{rank}"
            ]
            for rank in range(TP)
        },
    }
    if mode == "dfx":
        result["dfx_gate"] = dfx_gate
        result["dfx_raw_evidence"] = dfx_raw_evidence
    output = run / "artifact_validation.json"
    seal_record = (
        _seal_authority_record(
            run=run,
            authority_path=args.seal_authority,
            authority_sha256=args.seal_authority_sha256,
        )
        if args.seal_runtime_evidence
        else None
    )
    _write_stable(
        output,
        result,
        seal_runtime_evidence=args.seal_runtime_evidence,
        seal_authority_record=seal_record,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "report": str(output),
                "runtime_evidence_sealed": args.seal_runtime_evidence,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
