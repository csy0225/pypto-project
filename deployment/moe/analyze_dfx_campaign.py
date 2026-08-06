#!/usr/bin/env python3
"""Aggregate source-aware L0-L4 MoE DFX evidence across the BS matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

from analyze_matrix_performance import validate_normal_run
from validate_five_layer_route_case import (
    ROUTE_ARTIFACT_NAMES,
    recompute_route_tensor_contracts,
    validate_route_validation_record,
)
from validate_five_layer_case import (
    DFX_POLICIES,
    _seal_authority_record,
    _validate_map as validate_kv_map,
    _validate_runtime_markers as validate_runtime_markers,
    load_matched_dfx_policy,
    validate_matched_policy_binding,
)
from verify_exact_tree_manifest import verify_exact_tree


BATCHES = (1, 2, 4, 7, 8, 16)
SOURCES = ("baseline", "candidate")
HIDDEN_NAMES = ("hidden_l3", "hidden_l4")
CONTEXT_LEN = 65536
SOURCE_COMPATIBILITY_SCHEMA = (
    "step3p5.five-layer-moe-route-dfx-source-compatibility.v1"
)
FINAL_SELECTION_SCHEMA = "step3p5.moe.final-selection.v1"
MATCHED_SOURCE_AUDIT_SCHEMA = "step3p5.moe.matched-source-audit.v1"
CRITICAL_DECODE = "models/step3p5/decode_fwd.py"
CRITICAL_PROGRAM = (
    "tests/step3p5/harnesses/_five_layer_moe_program.py"
)
CRITICAL_HOLDER = "tools/step3p5/five_layer_moe_holder.py"
CRITICAL_FILES = (
    "COMMIT",
    CRITICAL_DECODE,
    CRITICAL_PROGRAM,
    "tests/step3p5/harnesses/_stage_five_layer_moe.py",
    CRITICAL_HOLDER,
    "tools/step3p5/analyze_five_layer_moe_dfx.py",
)
ROUTE_ADDITIONS = (
    "tests/step3p5/harnesses/_five_layer_moe_route_program.py",
    "tests/step3p5/harnesses/_stage_five_layer_moe_route.py",
    "tests/step3p5/unit/test_five_layer_moe_route_contract.py",
    "tools/step3p5/five_layer_moe_route_holder.py",
)
EXPECTED_CHANGED_PROVENANCE = (
    "DFX_SOURCE_PROVENANCE.txt",
    "PARENT_SOURCE_SHA256SUMS",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument(
        "--candidate-analysis-dir",
        default="dfx_analysis",
        help="candidate analysis directory under runtime/bsN",
    )
    parser.add_argument("--out", default="")
    parser.add_argument("--report-prefix", default="dfx_campaign")
    parser.add_argument(
        "--require-publication-ready",
        action="store_true",
    )
    parser.add_argument(
        "--source-compatibility",
        default="",
        help="route/DFX source compatibility report",
    )
    parser.add_argument(
        "--matched-policy",
        default="",
        help="optional dual-role tile control/winner DFX policy",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected JSON object")
    return value


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise AssertionError(f"missing text artifact: {path}")
    return path.read_text(encoding="utf-8").strip()


def _stable_write(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"refusing to replace different report: {path}")
    path.write_text(content, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex_sha(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value)
    )


def _validate_hidden_against_golden(
    *,
    batch_dir: Path,
    batch: int,
    image_ref: str,
    golden_reference: dict[str, str],
) -> None:
    manifest_path = Path(golden_reference["path"]).resolve()
    if (
        not manifest_path.is_file()
        or _sha256(manifest_path) != golden_reference["sha256"]
    ):
        raise AssertionError(f"BS{batch}: golden manifest authority mismatch")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema") != "step3p5.five-layer-moe-golden.v3"
        or manifest.get("source_kind") != "baseline"
        or manifest.get("active_batch") != batch
        or manifest.get("context_len_per_sequence") != CONTEXT_LEN
        or manifest.get("image_ref") != image_ref
    ):
        raise AssertionError(f"BS{batch}: golden manifest semantic mismatch")
    golden_dir = manifest_path.parent
    for name in HIDDEN_NAMES:
        actual_path = batch_dir / f"{name}.pt"
        golden_path = golden_dir / f"{name}.pt"
        if manifest.get("files", {}).get(golden_path.name) != _sha256(
            golden_path
        ):
            raise AssertionError(f"BS{batch}: golden hidden hash mismatch")
        actual = torch.load(
            actual_path,
            map_location="cpu",
            weights_only=True,
        )
        golden = torch.load(
            golden_path,
            map_location="cpu",
            weights_only=True,
        )
        if (
            not isinstance(actual, torch.Tensor)
            or not isinstance(golden, torch.Tensor)
            or tuple(actual.shape) != (8, batch, 4096)
            or not torch.equal(actual, golden)
        ):
            raise AssertionError(
                f"BS{batch}: {name} entity is not golden-exact"
            )


def _run_name(source: str, round_id: int, batch: int) -> str:
    return f"{source}-r{round_id}-dfx-bs{batch}-64k"


def _analysis_dir(
    source: str,
    candidate_analysis_dir: str,
) -> str:
    if source == "candidate":
        return candidate_analysis_dir
    return "dfx_analysis"


def _record(
    campaign: Path,
    *,
    source: str,
    round_id: int,
    batch: int,
    candidate_analysis_dir: str,
    matched_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run = campaign / "runs" / _run_name(source, round_id, batch)
    runtime = run / "runtime"
    batch_dir = runtime / f"bs{batch}"
    validation = _load_json(run / "artifact_validation.json")
    case = _load_json(batch_dir / "report.json")
    matrix = _load_json(runtime / "matrix_report.json")
    analysis_name = _analysis_dir(source, candidate_analysis_dir)
    dfx = _load_json(
        batch_dir / analysis_name / "moe_dfx_report.json"
    )

    if validation.get("passed") is not True:
        raise AssertionError(f"{run.name}: artifact validation failed")
    if _read_text(run / "container.rc") != "0":
        raise AssertionError(f"{run.name}: container did not exit cleanly")
    host_image_ref = _read_text(run / "image_ref.txt")
    host_nonce = _read_text(run / "run_nonce.txt")
    if not _hex_sha(host_nonce):
        raise AssertionError(f"{run.name}: invalid run nonce")
    if case.get("source_kind") != source:
        raise AssertionError(f"{run.name}: source kind mismatch")
    if case.get("mode") != "dfx":
        raise AssertionError(f"{run.name}: not a DFX run")
    if int(case.get("round", -1)) != round_id:
        raise AssertionError(f"{run.name}: round mismatch")
    expected_validation = {
        "schema": "step3p5.five-layer-moe-case-validation.v1",
        "passed": True,
        "run": run.name,
        "source_kind": source,
        "round": round_id,
        "mode": "dfx",
        "active_batch": batch,
        "context_len_per_sequence": CONTEXT_LEN,
        "active_total_context_tokens": batch * CONTEXT_LEN,
        "image_ref": host_image_ref,
        "run_nonce": host_nonce,
    }
    for key, expected in expected_validation.items():
        if validation.get(key) != expected:
            raise AssertionError(
                f"{run.name}: artifact validation {key} mismatch"
            )
    if case.get("image_ref") != host_image_ref:
        raise AssertionError(f"{run.name}: host/case image mismatch")
    timing = case.get("timing")
    workload = case.get("workload")
    if not isinstance(timing, dict) or not isinstance(workload, dict):
        raise AssertionError(f"{run.name}: timing/workload provenance missing")
    expected_iters = (
        matched_policy["authority"]["capture"]["dfx_iters"]
        if matched_policy is not None
        else 1
    )
    expected_warmup = (
        matched_policy["authority"]["capture"]["dfx_warmup"]
        if matched_policy is not None
        else 2
    )
    if (
        timing.get("iters") != expected_iters
        or timing.get("warmup") != expected_warmup
    ):
        raise AssertionError(f"{run.name}: DFX timing protocol mismatch")
    if (
        workload.get("active_batch") != batch
        or workload.get("context_len_per_sequence") != CONTEXT_LEN
        or workload.get("blocks_per_sequence") != 512
        or workload.get("active_total_context_tokens")
        != batch * CONTEXT_LEN
    ):
        raise AssertionError(f"{run.name}: DFX workload contract mismatch")
    comparisons = case.get("comparisons")
    if not (
        isinstance(comparisons, dict)
        and set(comparisons) == set(HIDDEN_NAMES)
        and all(
            isinstance(comparisons[name], dict)
            and comparisons[name].get("exact") is True
            for name in HIDDEN_NAMES
        )
    ):
        raise AssertionError(f"{run.name}: hidden outputs are not exact")

    policy = (
        matched_policy["sources"][source]
        if matched_policy is not None
        else DFX_POLICIES[source]
    )
    expected_profile = policy.get("profile", source)
    source_policy = dfx.get("source_policy")
    if (
        dfx.get("profile") != expected_profile
        or not isinstance(source_policy, dict)
        or source_policy.get("policy_id") != policy["policy_id"]
    ):
        raise AssertionError(f"{run.name}: DFX source policy mismatch")
    if case["source"]["decode_fwd_sha256"] != policy["decode_fwd_sha256"]:
        raise AssertionError(f"{run.name}: decode source hash mismatch")
    if (
        matched_policy is not None
        and case["source"]["source_manifest_sha256"]
        != policy["source_manifest_sha256"]
    ):
        raise AssertionError(f"{run.name}: source manifest policy mismatch")
    if dfx.get("structural_contracts", {}).get("pass") is not True:
        raise AssertionError(f"{run.name}: structural DFX contract failed")
    if (
        dfx.get("slice_contract", {}).get("expected_equals_observed")
        is not True
    ):
        raise AssertionError(f"{run.name}: physical slice contract failed")
    if dfx.get("routed_slice_profiles", {}).get("pass") is not True:
        raise AssertionError(
            f"{run.name}: routed physical-slice profile failed"
        )

    hidden_hashes = {
        f"{name}.pt": _sha256(batch_dir / f"{name}.pt")
        for name in HIDDEN_NAMES
    }
    if validation.get("hidden_sha256") != hidden_hashes:
        raise AssertionError(f"{run.name}: hidden entity hash mismatch")
    for name, digest in hidden_hashes.items():
        if case.get("files", {}).get(name) != digest:
            raise AssertionError(f"{run.name}: case hidden hash mismatch: {name}")
    kv_hashes = {}
    runtime_hashes = {}
    for rank in range(8):
        kv_hashes[str(rank)] = validate_kv_map(
            runtime / f"pypto_kvpool_map.json.rank{rank}",
            rank=rank,
            batch=batch,
            nonce=host_nonce,
        )
        runtime_hashes.update(
            validate_runtime_markers(
                runtime,
                rank=rank,
                batch=batch,
                nonce=host_nonce,
            )
        )
    marker_hashes = {
        name: digest
        for name, digest in runtime_hashes.items()
        if not name.startswith("pypto_kvpool.key.")
    }
    key_hashes = {
        str(rank): runtime_hashes[
            f"pypto_kvpool.key.rank{rank}"
        ]
        for rank in range(8)
    }
    if validation.get("kv_map_sha256_by_rank") != kv_hashes:
        raise AssertionError(f"{run.name}: KV map entity hash mismatch")
    if validation.get("runtime_marker_sha256") != marker_hashes:
        raise AssertionError(
            f"{run.name}: runtime marker entity hash mismatch"
        )
    if validation.get("kv_key_sha256_by_rank") != key_hashes:
        raise AssertionError(
            f"{run.name}: KV key entity hash mismatch"
        )
    expected_validated_workload = {
        key: workload[key]
        for key in (
            "active_batch",
            "context_len_per_sequence",
            "blocks_per_sequence",
            "active_total_context_tokens",
            "allocated_scheduler_blocks",
            "allocated_physical_blocks",
            "kv_num_layers",
            "allocated_kv_rows_per_rank",
            "allocated_kv_pool_bytes_per_rank",
        )
    }
    if validation.get("workload") != expected_validated_workload:
        raise AssertionError(f"{run.name}: validated workload mismatch")
    if matched_policy is not None:
        _validate_hidden_against_golden(
            batch_dir=batch_dir,
            batch=batch,
            image_ref=host_image_ref,
            golden_reference=matched_policy["authority"][
                "golden_manifests"
            ][str(batch)],
        )

    admission = dfx["admission"]
    expert = dfx["expert_kernel_release"]
    readiness = admission["release_readiness"]
    if matched_policy is not None:
        validation_policy = validation.get("dfx_gate", {}).get(
            "matched_policy"
        )
        if not isinstance(validation_policy, dict):
            raise AssertionError(
                f"{run.name}: matched policy validation is missing"
            )
        expected_validation = {
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
        for key, expected in expected_validation.items():
            if validation_policy.get(key) != expected:
                raise AssertionError(
                    f"{run.name}: matched validation {key} mismatch"
                )
        validate_matched_policy_binding(
            source_kind=source,
            profile=str(dfx.get("profile")),
            policy=policy,
            source=case["source"],
            source_policy=source_policy,
            expert_release=expert,
            admission=admission,
        )
    blocker_codes = [
        str(item.get("code"))
        for item in admission.get("blockers", [])
        if isinstance(item, dict)
    ]
    expected_matrix = {
        "schema": "step3p5.five-layer-moe-64k-matrix.v1",
        "source_kind": source,
        "round": round_id,
        "mode": "dfx",
        "image_ref": host_image_ref,
        "image_pypto_commit": case.get("image_pypto_commit"),
        "batches": [batch],
        "context_len_per_sequence": CONTEXT_LEN,
        "blocks_per_sequence": 512,
        "reports": {str(batch): f"bs{batch}/report.json"},
        "report": case,
    }
    if matrix != expected_matrix:
        raise AssertionError(f"{run.name}: matrix report identity mismatch")
    critical_path = batch_dir / analysis_name / "moe_critical_path_report.md"
    if not critical_path.is_file():
        raise AssertionError(f"{run.name}: critical-path report is missing")
    evidence_paths = {
        "container.rc": run / "container.rc",
        "image_ref.txt": run / "image_ref.txt",
        "run_nonce.txt": run / "run_nonce.txt",
        "artifact_validation.json": run / "artifact_validation.json",
        "runtime/matrix_report.json": runtime / "matrix_report.json",
        f"runtime/bs{batch}/report.json": batch_dir / "report.json",
        f"runtime/bs{batch}/{analysis_name}/moe_dfx_report.json": (
            batch_dir / analysis_name / "moe_dfx_report.json"
        ),
        f"runtime/bs{batch}/{analysis_name}/moe_critical_path_report.md": (
            critical_path
        ),
        **{
            f"runtime/bs{batch}/{name}": batch_dir / name
            for name in hidden_hashes
        },
        **{
            f"runtime/pypto_kvpool_map.json.rank{rank}": (
                runtime / f"pypto_kvpool_map.json.rank{rank}"
            )
            for rank in range(8)
        },
        **{
            f"runtime/{relative}": runtime / relative
            for relative in runtime_hashes
        },
    }
    result = {
        "run": run.name,
        "analysis_dir": analysis_name,
        "policy_id": policy["policy_id"],
        "source_manifest_sha256": case["source"][
            "source_manifest_sha256"
        ],
        "decode_fwd_sha256": case["source"]["decode_fwd_sha256"],
        "image_ref": case.get("image_ref"),
        "image_pypto_commit": case.get("image_pypto_commit"),
        "workload": case.get("workload"),
        "timing": case.get("timing"),
        "run_nonce": host_nonce,
        "runtime_marker_sha256": marker_hashes,
        "kv_key_sha256_by_rank": key_hashes,
        "evidence_sha256": {
            relative: _sha256(path)
            for relative, path in evidence_paths.items()
        },
        "hidden_exact": True,
        "analyzer_gate_pass": admission.get("analyzer_gate_pass"),
        "analyzer_blocker_codes": blocker_codes,
        "expert_release_enforced": expert.get("release_enforced"),
        "expert_release_gate_pass": expert.get("release_gate_pass"),
        "expert_release_gate_status": expert.get("release_gate_status"),
        "route_evidence_ready": readiness.get(
            "recv_meta_publication_evidence_ready"
        ),
        "release_readiness_status": readiness.get("status"),
        "publication_allowed_by_analyzer": readiness.get(
            "publication_allowed"
        ),
    }
    if matched_policy is not None:
        result["matched_policy_sha256"] = matched_policy["policy_sha256"]
        result["profile"] = policy["profile"]
    return result


def _validate_evidence_reference(
    reference: dict[str, str],
    *,
    name: str,
) -> dict[str, str]:
    path = Path(reference["path"]).resolve()
    if not path.is_file():
        raise AssertionError(f"authority evidence is missing: {name}")
    digest = _sha256(path)
    if digest != reference["sha256"]:
        raise AssertionError(f"authority evidence hash mismatch: {name}")
    return {"path": str(path), "sha256": digest}


def _validate_matched_authority(
    *,
    matched_policy: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    authority = matched_policy["authority"]
    if {item.get("image_ref") for item in records} != {
        authority["image_ref"]
    }:
        raise AssertionError(
            "matched authority image does not match all 12 DFX runs"
        )
    if {item.get("image_pypto_commit") for item in records} != {
        authority["image_pypto_commit"]
    }:
        raise AssertionError(
            "matched authority PyPTO commit does not match all 12 DFX runs"
        )
    capture = authority["capture"]
    for item in records:
        timing = item.get("timing")
        workload = item.get("workload")
        if not isinstance(timing, dict) or not isinstance(workload, dict):
            raise AssertionError("matched DFX timing/workload provenance missing")
        if (
            timing.get("iters") != capture["dfx_iters"]
            or timing.get("warmup") != capture["dfx_warmup"]
        ):
            raise AssertionError("matched DFX capture protocol mismatch")
        if (
            workload.get("context_len_per_sequence") != CONTEXT_LEN
            or workload.get("blocks_per_sequence") != 512
        ):
            raise AssertionError("matched DFX workload authority mismatch")

    evidence = {
        key: _validate_evidence_reference(authority[key], name=key)
        for key in (
            "selection_report",
            "source_audit",
            "scripts_manifest",
            "normal_capture_scripts_manifest",
            "normal_seal_authority",
            "normal_correctness_report",
            "normal_performance_report",
            "normal_campaign_spec",
            "normal_counterbalance_spec",
        )
    }
    selection = _load_json(Path(evidence["selection_report"]["path"]))
    source_audit = _load_json(Path(evidence["source_audit"]["path"]))
    candidate_policy = matched_policy["sources"]["candidate"]
    expected_selection = {
        "schema": FINAL_SELECTION_SCHEMA,
        "passed": True,
        "image_ref": authority["image_ref"],
        "selected_variant": candidate_policy["profile"],
        "selected_decode_fwd_sha256": candidate_policy[
            "decode_fwd_sha256"
        ],
        "source_audit_sha256": evidence["source_audit"]["sha256"],
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            raise AssertionError(
                f"matched selection semantic mismatch: {key}"
            )
    if (
        selection.get("tile_decision") != "KEEP_CONTROL"
        or selection.get("tile_selected_variant") != "mm-n64-r16"
        or selection.get("activation_decision") != "KEEP_ACT_N64"
        or selection.get("activation_selected_variant")
        != candidate_policy["profile"]
    ):
        raise AssertionError("matched selection decision is not authoritative")
    rejected = selection.get("rejected_variants")
    if not (
        isinstance(rejected, list)
        and {
            "mm-n64-r32",
            "lineage-only",
            "full-execution-patch",
        }
        <= set(rejected)
    ):
        raise AssertionError("matched selection rejection evidence is incomplete")

    if (
        source_audit.get("schema") != MATCHED_SOURCE_AUDIT_SCHEMA
        or source_audit.get("passed") is not True
        or source_audit.get("image_ref") != authority["image_ref"]
        or source_audit.get("selected_variant")
        != candidate_policy["profile"]
        or source_audit.get("selected_raw_source_manifest_sha256")
        != selection.get("selected_raw_source_manifest_sha256")
    ):
        raise AssertionError("matched source audit semantic mismatch")
    audited_sources = source_audit.get("sources")
    if not (
        isinstance(audited_sources, dict)
        and set(audited_sources) == set(SOURCES)
    ):
        raise AssertionError("matched source audit must cover both sources")
    for source_kind in SOURCES:
        item = audited_sources[source_kind]
        policy = matched_policy["sources"][source_kind]
        if not isinstance(item, dict):
            raise AssertionError(
                f"matched source audit is malformed: {source_kind}"
            )
        expected = {
            "profile": policy["profile"],
            "source_kind": source_kind,
            "source_role": policy["source_role"],
            "source_manifest_sha256": policy[
                "source_manifest_sha256"
            ],
            "decode_fwd_sha256": policy["decode_fwd_sha256"],
            "exact_tree": True,
        }
        for key, expected_value in expected.items():
            if item.get(key) != expected_value:
                raise AssertionError(
                    f"matched source audit {source_kind}.{key} mismatch"
                )
        root_text = item.get("root")
        if not isinstance(root_text, str) or not root_text:
            raise AssertionError(
                f"matched source audit {source_kind}.root is missing"
            )
        source_root = Path(root_text).resolve()
        verify_exact_tree(source_root)
        if (
            _sha256(source_root / "SOURCE_SHA256SUMS")
            != policy["source_manifest_sha256"]
            or _sha256(source_root / CRITICAL_DECODE)
            != policy["decode_fwd_sha256"]
        ):
            raise AssertionError(
                f"matched source audit tree drift: {source_kind}"
            )

    scripts_manifest_path = Path(evidence["scripts_manifest"]["path"])
    script_entries = verify_exact_tree(
        scripts_manifest_path.parent,
        manifest_name=scripts_manifest_path.name,
        symlink_manifest_name=None,
    )
    required_scripts = {
        "analyze_dfx_campaign.py",
        "analyze_matrix_performance.py",
        "validate_five_layer_case.py",
        "validate_five_layer_route_case.py",
        "verify_exact_tree_manifest.py",
    }
    if not required_scripts <= set(script_entries):
        raise AssertionError("authority scripts manifest is incomplete")
    validator = Path(__file__).with_name("validate_five_layer_case.py")
    if _sha256(validator) != authority["validator_sha256"]:
        raise AssertionError("matched authority validator hash mismatch")
    if (
        script_entries["validate_five_layer_case.py"]
        != authority["validator_sha256"]
        or script_entries["analyze_dfx_campaign.py"] != _sha256(Path(__file__))
    ):
        raise AssertionError("authority scripts content mismatch")
    normal_capture_manifest_path = Path(
        evidence["normal_capture_scripts_manifest"]["path"]
    )
    normal_capture_entries = verify_exact_tree(
        normal_capture_manifest_path.parent,
        manifest_name=normal_capture_manifest_path.name,
        symlink_manifest_name=None,
    )
    required_capture_scripts = {
        "analyze_matrix_correctness.py",
        "container_five_layer_matrix.sh",
        "five_layer_matrix.py",
        "run_0162_five_layer_matrix.sh",
        "run_0162_normal_campaign.sh",
        "run_0162_normal_counterbalance.sh",
        "create_normal_seal_authority.py",
        "run_0162_normal_evidence_seal.sh",
        "validate_five_layer_case.py",
        "verify_exact_tree_manifest.py",
    }
    if not required_capture_scripts <= set(normal_capture_entries):
        raise AssertionError(
            "authority normal capture scripts manifest is incomplete"
        )

    normal_correctness = _load_json(
        Path(evidence["normal_correctness_report"]["path"])
    )
    if (
        normal_correctness.get("schema")
        != "step3p5.five-layer-moe-64k-correctness.v1"
        or normal_correctness.get("passed") is not True
        or normal_correctness.get("batches") != list(BATCHES)
        or normal_correctness.get("context_len_per_sequence") != CONTEXT_LEN
        or normal_correctness.get("gates")
        != {
            "health": True,
            "fresh_process_raw_exact": True,
            "baseline_equals_candidate": True,
            "batch_extension_invariance": True,
        }
    ):
        raise AssertionError("authority normal correctness report failed")
    normal_performance_path = Path(
        evidence["normal_performance_report"]["path"]
    )
    normal_performance = _load_json(normal_performance_path)
    if (
        normal_performance.get("schema")
        != "step3p5.five-layer-moe-64k-performance.v2"
        or normal_performance.get("measurement_integrity_passed") is not True
        or normal_performance.get("correctness_report_passed") is not True
        or normal_performance.get(
            "hidden_hash_exact_across_selected_rounds"
        )
        is not True
        or normal_performance.get("image_ref") != authority["image_ref"]
        or normal_performance.get("rounds") != [1, 2, 3]
        or set(normal_performance.get("batches", {}))
        != {str(batch) for batch in BATCHES}
    ):
        raise AssertionError("authority normal performance report failed")
    normal_run_nonces: set[str] = set()
    seal_authority_path = evidence["normal_seal_authority"]["path"]
    seal_authority_sha256 = evidence["normal_seal_authority"]["sha256"]
    for batch in BATCHES:
        item = normal_performance["batches"][str(batch)]
        if (
            item.get("context_len_per_sequence") != CONTEXT_LEN
            or item.get("active_total_context_tokens")
            != batch * CONTEXT_LEN
            or set(item.get("paired_rounds", {})) != {"1", "2", "3"}
        ):
            raise AssertionError(
                f"authority normal performance BS{batch} mismatch"
            )
        for source_kind in SOURCES:
            rounds = item.get(source_kind, {}).get("rounds")
            if not isinstance(rounds, list) or len(rounds) != 3:
                raise AssertionError(
                    f"authority normal {source_kind} BS{batch} rounds"
                )
            reported_by_round = {
                record.get("round"): record
                for record in rounds
                if isinstance(record, dict)
            }
            if set(reported_by_round) != {1, 2, 3}:
                raise AssertionError(
                    f"authority normal {source_kind} BS{batch} "
                    "run identity mismatch"
                )
            expected_policy = matched_policy["sources"][source_kind]
            for round_id in (1, 2, 3):
                record = reported_by_round[round_id]
                source = record.get("source")
                if (
                    not isinstance(source, dict)
                    or source.get("source_manifest_sha256")
                    != expected_policy["source_manifest_sha256"]
                    or source.get("decode_fwd_sha256")
                    != expected_policy["decode_fwd_sha256"]
                ):
                    raise AssertionError(
                        f"authority normal source drift: "
                        f"{source_kind} BS{batch}"
                    )
                actual = validate_normal_run(
                    normal_performance_path.parent,
                    source_kind,
                    round_id,
                    batch,
                )
                normal_run = (
                    normal_performance_path.parent
                    / "runs"
                    / actual["run"]
                )
                seal_record = _seal_authority_record(
                    run=normal_run,
                    authority_path=seal_authority_path,
                    authority_sha256=seal_authority_sha256,
                )
                validation_path = normal_run / "artifact_validation.json"
                if (
                    _sha256(validation_path)
                    != seal_record["sealed_validation_sha256"]
                ):
                    raise AssertionError(
                        f"authority normal validation seal drift: "
                        f"{source_kind} r{round_id} BS{batch}"
                    )
                if record != actual:
                    raise AssertionError(
                        f"authority normal evidence drift: "
                        f"{source_kind} r{round_id} BS{batch}"
                    )
                nonce = actual["run_nonce"]
                if nonce in normal_run_nonces:
                    raise AssertionError(
                        f"duplicate authority normal run nonce: {nonce}"
                    )
                normal_run_nonces.add(nonce)
    if len(normal_run_nonces) != len(BATCHES) * len(SOURCES) * 3:
        raise AssertionError(
            "authority normal run nonces are not unique across 36 runs"
        )
    normal_spec = Path(
        evidence["normal_campaign_spec"]["path"]
    ).read_text(encoding="utf-8")
    counterbalance_spec = Path(
        evidence["normal_counterbalance_spec"]["path"]
    ).read_text(encoding="utf-8")
    required_normal_lines = {
        f"image={authority['image_ref']}",
        (
            "campaign_scripts_manifest_sha256="
            f"{evidence['normal_capture_scripts_manifest']['sha256']}"
        ),
        "batches=1,2,4,7,8,16",
        "context_len_per_sequence=65536",
        "warmup=5",
        "measured_iters=30",
        "order=per-bs:baseline-r1,candidate-r1,baseline-r2,candidate-r2",
    }
    required_counterbalance_lines = {
        f"image={authority['image_ref']}",
        (
            "campaign_scripts_manifest_sha256="
            f"{evidence['normal_capture_scripts_manifest']['sha256']}"
        ),
        "batches=1,2,4,7,8,16",
        "context_len_per_sequence=65536",
        "round=3",
        "warmup=5",
        "measured_iters=30",
        "order=per-bs:candidate-r3,baseline-r3",
    }
    if (
        not required_normal_lines
        <= set(normal_spec.splitlines())
        or not required_counterbalance_lines
        <= set(counterbalance_spec.splitlines())
    ):
        raise AssertionError("authority normal capture specs mismatch")

    golden: dict[str, dict[str, str]] = {}
    for batch in BATCHES:
        key = str(batch)
        reference = _validate_evidence_reference(
            authority["golden_manifests"][key],
            name=f"golden_manifests.{key}",
        )
        manifest = _load_json(Path(reference["path"]))
        if (
            manifest.get("schema") != "step3p5.five-layer-moe-golden.v3"
            or manifest.get("source_kind") != "baseline"
            or manifest.get("active_batch") != batch
            or manifest.get("context_len_per_sequence") != CONTEXT_LEN
            or manifest.get("image_ref") != authority["image_ref"]
        ):
            raise AssertionError(
                f"matched authority golden manifest mismatch: BS{batch}"
            )
        golden[key] = reference
    return {
        "image_ref": authority["image_ref"],
        "image_pypto_commit": authority["image_pypto_commit"],
        **evidence,
        "selection": selection,
        "source_audit_semantic_passed": True,
        "scripts_exact_tree_passed": True,
        "normal_capture_scripts_exact_tree_passed": True,
        "analyzer_sha256": authority["analyzer_sha256"],
        "validator_sha256": authority["validator_sha256"],
        "route_validator_sha256": script_entries[
            "validate_five_layer_route_case.py"
        ],
        "normal_analyzer_sha256": script_entries[
            "analyze_matrix_performance.py"
        ],
        "golden_manifests": golden,
        "workload": authority["workload"],
        "capture": authority["capture"],
    }


def _audit_compatibility_sources(
    compatibility: dict[str, Any],
) -> dict[str, Any]:
    dfx_root = Path(str(compatibility.get("dfx_source_root", ""))).resolve()
    route_root = Path(
        str(compatibility.get("route_source_root", ""))
    ).resolve()
    dfx_entries = verify_exact_tree(dfx_root)
    route_entries = verify_exact_tree(route_root)
    dfx_manifest_sha = _sha256(dfx_root / "SOURCE_SHA256SUMS")
    route_manifest_sha = _sha256(route_root / "SOURCE_SHA256SUMS")
    if (
        compatibility.get("dfx_source_manifest_sha256")
        != dfx_manifest_sha
        or compatibility.get("route_source_manifest_sha256")
        != route_manifest_sha
    ):
        raise AssertionError(
            "compatibility source manifest does not match exact source tree"
        )

    dfx_paths = set(dfx_entries)
    route_paths = set(route_entries)
    additions = sorted(route_paths - dfx_paths)
    removals = sorted(dfx_paths - route_paths)
    changed = sorted(
        path
        for path in dfx_paths & route_paths
        if dfx_entries[path] != route_entries[path]
    )
    if removals:
        raise AssertionError(f"compatibility route source removals: {removals}")
    if additions != sorted(ROUTE_ADDITIONS):
        raise AssertionError(
            f"compatibility route source additions mismatch: {additions}"
        )
    if changed != sorted(EXPECTED_CHANGED_PROVENANCE):
        raise AssertionError(
            f"compatibility source provenance drift mismatch: {changed}"
        )
    critical = compatibility.get("critical_hashes")
    if not isinstance(critical, dict) or set(critical) != set(CRITICAL_FILES):
        raise AssertionError("compatibility critical hashes are incomplete")
    expected_critical = {
        path: dfx_entries[path] for path in CRITICAL_FILES
    }
    if critical != expected_critical:
        raise AssertionError("compatibility critical hashes differ from source")
    route_hashes = compatibility.get("route_addition_hashes")
    expected_route_hashes = {
        path: route_entries[path] for path in ROUTE_ADDITIONS
    }
    if route_hashes != expected_route_hashes:
        raise AssertionError(
            "compatibility route addition hashes differ from source"
        )
    return {
        "dfx_source_root": str(dfx_root),
        "route_source_root": str(route_root),
        "dfx_source_manifest_sha256": dfx_manifest_sha,
        "route_source_manifest_sha256": route_manifest_sha,
        "critical_hashes": expected_critical,
        "route_addition_hashes": expected_route_hashes,
    }


def _validate_source_compatibility(
    *,
    campaign: Path,
    compatibility_path: Path,
    compatibility: dict[str, Any],
    round_id: int,
    candidate_analysis_dir: str,
    candidate_records: list[dict[str, Any]],
    matched_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    if (
        compatibility.get("schema") != SOURCE_COMPATIBILITY_SCHEMA
        or compatibility.get("passed") is not True
    ):
        raise AssertionError("route/DFX source compatibility failed")
    if compatibility.get("round") != round_id:
        raise AssertionError("compatibility round does not match campaign")
    if compatibility.get("context_len_per_sequence") != CONTEXT_LEN:
        raise AssertionError("compatibility context contract mismatch")

    expected_profile = (
        matched_policy["sources"]["candidate"]["profile"]
        if matched_policy is not None
        else None
    )
    compatibility_profile = compatibility.get("profile")
    if matched_policy is not None:
        if compatibility_profile != expected_profile:
            raise AssertionError(
                "compatibility profile does not match candidate policy"
            )
    elif compatibility_profile not in DFX_POLICIES:
        raise AssertionError("unsupported compatibility profile")
    expected_source_role = (
        matched_policy["sources"]["candidate"]["source_role"]
        if matched_policy is not None
        else compatibility.get("source_role")
    )
    if compatibility.get("source_role") != expected_source_role:
        raise AssertionError(
            "compatibility source role does not match candidate policy"
        )

    source_audit = _audit_compatibility_sources(compatibility)
    if (
        matched_policy is not None
        and source_audit["critical_hashes"][
            "tools/step3p5/analyze_five_layer_moe_dfx.py"
        ]
        != matched_policy["authority"]["analyzer_sha256"]
    ):
        raise AssertionError("matched authority analyzer hash mismatch")

    candidate_manifests = {
        item["source_manifest_sha256"] for item in candidate_records
    }
    if (
        len(candidate_manifests) != 1
        or compatibility.get("dfx_source_manifest_sha256")
        not in candidate_manifests
    ):
        raise AssertionError(
            "compatibility DFX manifest does not match campaign"
        )
    route_manifest = compatibility.get("route_source_manifest_sha256")
    if not _hex_sha(route_manifest) or route_manifest in candidate_manifests:
        raise AssertionError("compatibility route manifest is invalid")

    critical_hashes = source_audit["critical_hashes"]

    compatibility_batches = compatibility.get("batches")
    if not (
        isinstance(compatibility_batches, dict)
        and set(compatibility_batches) == {
            str(batch) for batch in BATCHES
        }
    ):
        raise AssertionError(
            "compatibility batches must be exactly the six-BS matrix"
        )

    compatibility_sha = _sha256(compatibility_path)
    image_refs: set[str] = set()
    program_hashes: set[str] = set()
    holder_hashes: set[str] = set()
    decode_hashes: set[str] = set()
    sidecar_dir = Path(str(compatibility.get("sidecar_dir", "")))
    if (
        sidecar_dir.is_absolute()
        or not sidecar_dir.parts
        or ".." in sidecar_dir.parts
    ):
        raise AssertionError("compatibility sidecar directory is invalid")
    for batch in BATCHES:
        batch_text = str(batch)
        run = (
            campaign
            / "runs"
            / _run_name("candidate", round_id, batch)
        )
        batch_dir = run / "runtime" / f"bs{batch}"
        case = _load_json(batch_dir / "report.json")
        source = case.get("source")
        workload = case.get("workload")
        if not isinstance(source, dict) or not isinstance(workload, dict):
            raise AssertionError(
                f"BS{batch}: compatibility campaign provenance missing"
            )
        image_ref = case.get("image_ref")
        if not isinstance(image_ref, str) or "@sha256:" not in image_ref:
            raise AssertionError(
                f"BS{batch}: compatibility image digest missing"
            )
        image_refs.add(image_ref)
        decode_hashes.add(str(source.get("decode_fwd_sha256")))
        program_hashes.add(str(source.get("five_layer_program_sha256")))
        holder_hashes.add(str(source.get("five_layer_holder_sha256")))

        item = compatibility_batches[batch_text]
        if not isinstance(item, dict):
            raise AssertionError(
                f"BS{batch}: compatibility batch record is malformed"
            )
        expected_batch_fields = {
            "dfx_run": _run_name("candidate", round_id, batch),
            "route_run": f"candidate-route-bs{batch}-64k",
            "input_tokens": workload.get("input_tokens"),
        }
        for key, expected in expected_batch_fields.items():
            if item.get(key) != expected:
                raise AssertionError(
                    f"BS{batch}: compatibility {key} mismatch"
                )
        for key in (
            "sidecar_sha256",
            "route_validation_sha256",
            "route_report_sha256",
        ):
            if not _hex_sha(item.get(key)):
                raise AssertionError(
                    f"BS{batch}: compatibility {key} is malformed"
                )
        if (
            item.get("profile") != compatibility_profile
            or item.get("source_role") != expected_source_role
        ):
            raise AssertionError(
                f"BS{batch}: compatibility route profile/role mismatch"
            )

        route_runtime = (
            campaign
            / sidecar_dir
            / item["route_run"]
            / "runtime"
        )
        sidecar_path = route_runtime / "recv_meta_sidecar.pt"
        validation_path = (
            route_runtime / "route_artifact_validation.json"
        )
        route_report_path = (
            route_runtime / "five_layer_moe_route_report.json"
        )
        for name, path, expected_sha in (
            ("sidecar", sidecar_path, item["sidecar_sha256"]),
            (
                "route validation",
                validation_path,
                item["route_validation_sha256"],
            ),
            (
                "route report",
                route_report_path,
                item["route_report_sha256"],
            ),
        ):
            if not path.is_file() or _sha256(path) != expected_sha:
                raise AssertionError(
                    f"BS{batch}: {name} artifact hash mismatch"
                )

        validation = _load_json(validation_path)
        route_report = _load_json(route_report_path)
        expected_validation = {
            "passed": True,
            "profile": compatibility_profile,
            "source_role": expected_source_role,
            "active_batch": batch,
            "context_len_per_sequence": CONTEXT_LEN,
            "image_ref": compatibility["image_ref"],
            "source_manifest_sha256": source_audit[
                "route_source_manifest_sha256"
            ],
            "decode_fwd_sha256": critical_hashes[CRITICAL_DECODE],
        }
        for key, expected in expected_validation.items():
            if validation.get(key) != expected:
                raise AssertionError(
                    f"BS{batch}: route validation {key} mismatch"
                )
        validate_route_validation_record(
            validation,
            active_batch=batch,
            profile=str(compatibility_profile),
            source_role=str(expected_source_role),
            decode_fwd_sha256=critical_hashes[CRITICAL_DECODE],
            image_ref=str(compatibility["image_ref"]),
            source_manifest_sha256=source_audit[
                "route_source_manifest_sha256"
            ],
            input_tokens=list(workload.get("input_tokens", [])),
        )
        golden_dir = (
            Path(
                matched_policy["authority"]["golden_manifests"][
                    batch_text
                ]["path"]
            ).resolve().parent
            if matched_policy is not None
            else campaign / "golden" / "heterogeneous-64k" / f"bs{batch}"
        )
        recomputed_route = recompute_route_tensor_contracts(
            runtime=route_runtime,
            golden_dir=golden_dir,
            active_batch=batch,
            image_ref=str(compatibility["image_ref"]),
        )
        for key, expected in recomputed_route.items():
            if validation.get(key) != expected:
                raise AssertionError(
                    f"BS{batch}: recomputed route {key} mismatch"
                )
        for artifact_name in ROUTE_ARTIFACT_NAMES:
            artifact_path = route_runtime / artifact_name
            if (
                not artifact_path.is_file()
                or _sha256(artifact_path)
                != validation["artifacts"][artifact_name]
            ):
                raise AssertionError(
                    f"BS{batch}: route artifact hash mismatch: "
                    f"{artifact_name}"
                )
        if (
            validation.get("artifacts", {}).get(
                "recv_meta_sidecar.pt"
            )
            != item["sidecar_sha256"]
        ):
            raise AssertionError(
                f"BS{batch}: route validation sidecar hash mismatch"
            )

        provenance = route_report.get("provenance")
        if not isinstance(provenance, dict):
            raise AssertionError(
                f"BS{batch}: route report provenance is missing"
            )
        route_source = provenance.get("source")
        input_contract = provenance.get("input_contract")
        if not isinstance(route_source, dict) or not isinstance(
            input_contract,
            dict,
        ):
            raise AssertionError(
                f"BS{batch}: route report source/input provenance missing"
            )
        route_workload = input_contract.get("workload")
        if not isinstance(route_workload, dict):
            raise AssertionError(
                f"BS{batch}: route report workload is missing"
            )
        expected_route_source = {
            "source_tree_manifest_sha256": source_audit[
                "route_source_manifest_sha256"
            ],
            "decode_fwd_sha256": critical_hashes[CRITICAL_DECODE],
            "formal_program_sha256": critical_hashes[CRITICAL_PROGRAM],
            "route_program_sha256": source_audit[
                "route_addition_hashes"
            ][
                "tests/step3p5/harnesses/"
                "_five_layer_moe_route_program.py"
            ],
            "route_stage_sha256": source_audit[
                "route_addition_hashes"
            ][
                "tests/step3p5/harnesses/"
                "_stage_five_layer_moe_route.py"
            ],
            "route_holder_sha256": source_audit[
                "route_addition_hashes"
            ]["tools/step3p5/five_layer_moe_route_holder.py"],
        }
        if provenance.get("image_digest") != compatibility["image_ref"]:
            raise AssertionError(
                f"BS{batch}: route report image mismatch"
            )
        for key, expected in expected_route_source.items():
            if route_source.get(key) != expected:
                raise AssertionError(
                    f"BS{batch}: route report {key} mismatch"
                )
        if (
            input_contract.get("input_tokens")
            != workload.get("input_tokens")
            or route_workload.get("active_batch") != batch
            or route_workload.get("context_len") != CONTEXT_LEN
        ):
            raise AssertionError(
                f"BS{batch}: route report workload/input mismatch"
            )
        report_sidecar_sha = route_report.get("artifacts", {}).get(
            "recv_meta_sidecar.pt"
        )
        if (
            report_sidecar_sha is not None
            and report_sidecar_sha != item["sidecar_sha256"]
        ):
            raise AssertionError(
                f"BS{batch}: route report sidecar hash mismatch"
            )

        analysis_dir = batch_dir / candidate_analysis_dir
        lineage_dir = (
            analysis_dir.parent
            if analysis_dir.name == "analysis"
            else analysis_dir
        )
        compatibility_pointer = (
            lineage_dir / "source_compatibility_sha256.txt"
        )
        route_validation_pointer = (
            lineage_dir / "route_validation_sha256.txt"
        )
        if (
            not compatibility_pointer.is_file()
            or compatibility_pointer.read_text(
                encoding="utf-8"
            ).strip()
            != compatibility_sha
        ):
            raise AssertionError(
                f"BS{batch}: route analysis compatibility lineage mismatch"
            )
        if (
            not route_validation_pointer.is_file()
            or route_validation_pointer.read_text(
                encoding="utf-8"
            ).strip()
            != item["route_validation_sha256"]
        ):
            raise AssertionError(
                f"BS{batch}: route validation lineage mismatch"
            )

    if len(image_refs) != 1 or compatibility.get("image_ref") not in image_refs:
        raise AssertionError(
            "compatibility image digest does not match campaign"
        )
    expected_critical = {
        CRITICAL_DECODE: decode_hashes,
        CRITICAL_PROGRAM: program_hashes,
        CRITICAL_HOLDER: holder_hashes,
    }
    for key, observed in expected_critical.items():
        if len(observed) != 1 or critical_hashes.get(key) not in observed:
            raise AssertionError(
                f"compatibility critical hash mismatch: {key}"
            )

    return {
        "report": str(compatibility_path),
        "report_sha256": compatibility_sha,
        "dfx_source_manifest_sha256": compatibility[
            "dfx_source_manifest_sha256"
        ],
        "route_source_manifest_sha256": route_manifest,
        "dfx_source_root": source_audit["dfx_source_root"],
        "route_source_root": source_audit["route_source_root"],
        "profile": compatibility_profile,
        "source_role": expected_source_role,
        "round": round_id,
        "image_ref": compatibility["image_ref"],
        "batches": list(BATCHES),
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Five-layer MoE DFX campaign",
        "",
        "| BS | candidate analyzer | expert gate | route sidecar | release status |",
        "|---:|:---:|:---:|:---:|:---|",
    ]
    for batch in BATCHES:
        candidate = result["batches"][str(batch)]["candidate"]
        lines.append(
            f"| {batch} | {candidate['analyzer_gate_pass']} | "
            f"{candidate['expert_release_gate_status']} | "
            f"{candidate['route_evidence_ready']} | "
            f"{candidate['release_readiness_status']} |"
        )
    lines.extend(
        [
            "",
            "Diagnostic capture: "
            + (
                "PASS"
                if result["diagnostic_capture_passed"]
                else "FAIL"
            ),
            "",
            "Publication ready: "
            + ("PASS" if result["publication_ready"] else "BLOCKED"),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    if args.round <= 0:
        raise ValueError("--round must be positive")
    campaign = Path(args.campaign).resolve()
    out = Path(args.out).resolve() if args.out else campaign
    out.mkdir(parents=True, exist_ok=True)
    matched_policy = (
        load_matched_dfx_policy(args.matched_policy)
        if args.matched_policy
        else None
    )

    batches: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for batch in BATCHES:
        batches[str(batch)] = {}
        for source in SOURCES:
            item = _record(
                campaign,
                source=source,
                round_id=args.round,
                batch=batch,
                candidate_analysis_dir=args.candidate_analysis_dir,
                matched_policy=matched_policy,
            )
            batches[str(batch)][source] = item
            records.append(item)

    dfx_run_nonces = [item["run_nonce"] for item in records]
    if len(set(dfx_run_nonces)) != len(records):
        raise AssertionError(
            "DFX run nonces are not unique across the 12-run matrix"
        )
    candidate_records = [
        batches[str(batch)]["candidate"] for batch in BATCHES
    ]
    matched_authority: dict[str, Any] | None = None
    candidate_manifests = {
        item["source_manifest_sha256"] for item in candidate_records
    }
    if len(candidate_manifests) != 1:
        raise AssertionError("candidate DFX source manifest changed across BS")
    if matched_policy is not None:
        policy_hashes = {
            item.get("matched_policy_sha256") for item in records
        }
        if policy_hashes != {matched_policy["policy_sha256"]}:
            raise AssertionError(
                "matched DFX policy changed across the 12 runs"
            )
        for source in SOURCES:
            source_records = [
                batches[str(batch)][source] for batch in BATCHES
            ]
            source_manifests = {
                item["source_manifest_sha256"] for item in source_records
            }
            expected_manifest = matched_policy["sources"][source][
                "source_manifest_sha256"
            ]
            if source_manifests != {expected_manifest}:
                raise AssertionError(
                    f"{source} DFX source does not match matched policy"
                )
        matched_authority = _validate_matched_authority(
            matched_policy=matched_policy,
            records=records,
        )
    diagnostic_capture_passed = all(
        item["hidden_exact"] for item in records
    )
    candidate_analyzer_gate_passed = all(
        item["analyzer_gate_pass"] is True
        for item in candidate_records
    )
    candidate_kernel_release_passed = all(
        item["expert_release_gate_pass"] is True
        for item in candidate_records
    )
    candidate_route_evidence_ready = all(
        item["route_evidence_ready"] is True
        for item in candidate_records
    )
    source_compatibility: dict[str, Any] | None = None
    source_compatibility_passed = False
    if args.source_compatibility:
        compatibility_path = Path(args.source_compatibility).resolve()
        compatibility = _load_json(compatibility_path)
        source_compatibility = _validate_source_compatibility(
            campaign=campaign,
            compatibility_path=compatibility_path,
            compatibility=compatibility,
            round_id=args.round,
            candidate_analysis_dir=args.candidate_analysis_dir,
            candidate_records=candidate_records,
            matched_policy=matched_policy,
        )
        source_compatibility_passed = True
    source_compatibility_required = (
        candidate_route_evidence_ready or matched_policy is not None
    )
    source_compatibility_ready = (
        not source_compatibility_required or source_compatibility_passed
    )
    publication_authority_ready = bool(
        matched_policy is not None
        and matched_authority is not None
        and source_compatibility is not None
    )
    publication_ready = all(
        (
            diagnostic_capture_passed,
            candidate_analyzer_gate_passed,
            candidate_kernel_release_passed,
            candidate_route_evidence_ready,
            source_compatibility_ready,
            publication_authority_ready,
        )
    )
    publication_authority: dict[str, Any] | None = None
    if publication_authority_ready:
        dfx_spec_path = campaign / "dfx_campaign_spec.txt"
        dfx_spec = _read_text(dfx_spec_path)
        required_dfx_lines = {
            f"image={matched_authority['image_ref']}",
            "batches=1,2,4,7,8,16",
            "context_len_per_sequence=65536",
            "rounds=1",
            "warmup=2",
            "measured_iters=1",
            "capture=separate-warm-dep-gen-then-l2-swimlane",
            "l2_swimlane_reuse_dep_gen=required",
            f"matched_policy_sha256={matched_policy['policy_sha256']}",
            "publication_ready=required",
        }
        if not required_dfx_lines <= set(dfx_spec.splitlines()):
            raise AssertionError("DFX campaign capture spec mismatch")
        publication_authority = {
            "matched_policy": {
                "path": matched_policy["policy_path"],
                "sha256": matched_policy["policy_sha256"],
            },
            "source_compatibility": {
                "path": source_compatibility["report"],
                "sha256": source_compatibility["report_sha256"],
            },
            "scripts_manifest": matched_authority["scripts_manifest"],
            "normal_capture_scripts_manifest": matched_authority[
                "normal_capture_scripts_manifest"
            ],
            "normal_seal_authority": matched_authority[
                "normal_seal_authority"
            ],
            "selection_report": matched_authority["selection_report"],
            "source_audit": matched_authority["source_audit"],
            "normal_correctness_report": matched_authority[
                "normal_correctness_report"
            ],
            "normal_performance_report": matched_authority[
                "normal_performance_report"
            ],
            "normal_campaign_spec": matched_authority[
                "normal_campaign_spec"
            ],
            "normal_counterbalance_spec": matched_authority[
                "normal_counterbalance_spec"
            ],
            "dfx_campaign_spec": {
                "path": str(dfx_spec_path),
                "sha256": _sha256(dfx_spec_path),
            },
            "dfx_runs": {
                item["run"]: {
                    "run_nonce": item["run_nonce"],
                    "evidence_sha256": item["evidence_sha256"],
                }
                for item in records
            },
            "aggregator_sha256": _sha256(Path(__file__)),
            "validator_sha256": matched_authority["validator_sha256"],
            "route_validator_sha256": matched_authority[
                "route_validator_sha256"
            ],
            "normal_analyzer_sha256": matched_authority[
                "normal_analyzer_sha256"
            ],
            "analyzer_sha256": matched_authority["analyzer_sha256"],
            "image_ref": matched_authority["image_ref"],
            "image_pypto_commit": matched_authority[
                "image_pypto_commit"
            ],
        }
    if publication_ready and publication_authority is None:
        raise AssertionError(
            "publication readiness requires a complete matched authority"
        )
    result = {
        "schema": "step3p5.five-layer-moe-dfx-campaign.v1",
        "round": args.round,
        "batches": batches,
        "candidate_analysis_dir": args.candidate_analysis_dir,
        "diagnostic_capture_passed": diagnostic_capture_passed,
        "candidate_analyzer_gate_passed_all_batches": (
            candidate_analyzer_gate_passed
        ),
        "candidate_kernel_release_passed_all_batches": (
            candidate_kernel_release_passed
        ),
        "candidate_route_evidence_ready_all_batches": (
            candidate_route_evidence_ready
        ),
        "source_compatibility_passed": source_compatibility_passed,
        "source_compatibility_required": source_compatibility_required,
        "source_compatibility": source_compatibility,
        "publication_ready": publication_ready,
        "publication_authority": publication_authority,
        "publication_blockers": [
            name
            for name, passed in (
                ("diagnostic_capture", diagnostic_capture_passed),
                ("candidate_analyzer_gate", candidate_analyzer_gate_passed),
                ("candidate_kernel_release", candidate_kernel_release_passed),
                ("candidate_route_evidence", candidate_route_evidence_ready),
                ("source_compatibility", source_compatibility_ready),
                ("publication_authority", publication_authority_ready),
            )
            if not passed
        ],
    }
    if matched_policy is not None:
        result["matched_source_policy"] = {
            "schema": matched_policy["schema"],
            "policy_sha256": matched_policy["policy_sha256"],
            "authority": matched_authority,
            "sources": {
                source: {
                    key: matched_policy["sources"][source][key]
                    for key in (
                        "profile",
                        "policy_id",
                        "source_kind",
                        "source_role",
                        "source_manifest_sha256",
                        "decode_fwd_sha256",
                        "release_enforced",
                    )
                }
                for source in SOURCES
            },
        }
    json_path = out / f"{args.report_prefix}.json"
    md_path = out / f"{args.report_prefix}.md"
    _stable_write(
        json_path,
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    _stable_write(md_path, _markdown(result))
    print(
        json.dumps(
            {
                "diagnostic_capture_passed": diagnostic_capture_passed,
                "publication_ready": publication_ready,
                "report": str(json_path),
            },
            sort_keys=True,
        )
    )
    if not diagnostic_capture_passed:
        return 1
    if args.require_publication_ready and not publication_ready:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
