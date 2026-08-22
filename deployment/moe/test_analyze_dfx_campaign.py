from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch


MOE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MOE_DIR))

import analyze_dfx_campaign as analyzer  # noqa: E402
from validate_five_layer_case import (  # noqa: E402
    _run_evidence_sha256,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path) -> str:
    lines = [
        f"{_sha256(path)}  {path.relative_to(root)}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SOURCE_SHA256SUMS"
    ]
    manifest = root / "SOURCE_SHA256SUMS"
    _write_text(manifest, "\n".join(lines) + "\n")
    return _sha256(manifest)


def _workload(batch: int) -> dict:
    scheduler_blocks = batch * 512
    physical_blocks = scheduler_blocks + 15
    return {
        "active_batch": batch,
        "context_len_per_sequence": 65536,
        "blocks_per_sequence": 512,
        "active_total_context_tokens": batch * 65536,
        "allocated_scheduler_blocks": scheduler_blocks,
        "allocated_physical_blocks": physical_blocks,
        "kv_num_layers": 5,
        "allocated_kv_rows_per_rank": 5 * physical_blocks * 128,
        "allocated_kv_pool_bytes_per_rank": (
            2 * 5 * physical_blocks * 128 * 128 * 2
        ),
        "input_tokens": list(range(batch)),
    }


def _write_hidden_pair(
    out: Path,
    batch: int,
    *,
    golden_dir: Path | None = None,
) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    if golden_dir is None:
        torch.save(
            torch.zeros((8, batch, 4096), dtype=torch.bfloat16),
            out / "hidden_l3.pt",
        )
        torch.save(
            torch.ones((8, batch, 4096), dtype=torch.bfloat16),
            out / "hidden_l4.pt",
        )
    else:
        for name in ("hidden_l3.pt", "hidden_l4.pt"):
            shutil.copy2(golden_dir / name, out / name)
    return {
        name: _sha256(out / name)
        for name in ("hidden_l3.pt", "hidden_l4.pt")
    }


def _write_kv_maps(
    runtime: Path,
    batch: int,
    nonce: str,
) -> dict[str, str]:
    scheduler_blocks = batch * 512
    physical_blocks = scheduler_blocks + 15
    slots_per_layer = physical_blocks * 128
    bytes_per_layer = slots_per_layer * 128 * 2
    pool_bytes = 2 * 5 * bytes_per_layer
    hashes = {}
    for rank in range(8):
        path = runtime / f"pypto_kvpool_map.json.rank{rank}"
        map_name = path.name
        done_name = f"{map_name}.done"
        map_path = f"/out/runtime/{map_name}"
        done_path = f"/out/runtime/{done_name}"
        entries = {}
        for layer in range(5):
            for section_index, which in enumerate(("K", "V")):
                entries[f"L{layer}.{which}"] = {
                    "layer_idx": layer,
                    "which": which,
                    "group_id": 0,
                    "num_blocks": physical_blocks,
                    "block_size": 128,
                    "num_slots": slots_per_layer,
                    "num_kv_heads": 1,
                    "head_dim": 128,
                    "dtype": "bfloat16",
                    "nbytes": bytes_per_layer,
                    "shape": [physical_blocks, 128, 1, 128],
                    "flat_shape": [slots_per_layer, 128],
                    "offset": (
                        section_index * 5 * bytes_per_layer
                        + layer * bytes_per_layer
                    ),
                }
        ipc_session = {
            "session_nonce": nonce,
            "producer_rank": rank,
        }
        map_value = {
            "version": 3,
            "rank": rank,
            "tp_world_size": 8,
            "num_layers": 5,
            "scheduler_num_blocks": scheduler_blocks,
            "physical_num_blocks": physical_blocks,
            "padding_block_count": 15,
            "reserve_start": scheduler_blocks,
            "block_size": 128,
            "num_kv_heads": 1,
            "head_dim": 128,
            "dtype": "bfloat16",
            "pool_bytes": pool_bytes,
            "padding_block_ids": list(
                range(scheduler_blocks, physical_blocks)
            ),
            "groups": [
                {"group_id": 0, "layer_indices": list(range(5))}
            ],
            "ipc_session": ipc_session,
            "map": entries,
            "sections": {
                "K": {"offset": 0, "nbytes": 5 * bytes_per_layer},
                "V": {
                    "offset": 5 * bytes_per_layer,
                    "nbytes": 5 * bytes_per_layer,
                },
            }
        }
        _write_json(path, map_value)
        _write_json(
            runtime / done_name,
            {
                "ipc_session": ipc_session,
                "key_path": f"/out/runtime/pypto_kvpool.key.rank{rank}",
                "map_path": map_path,
                "ready_at": 1.0,
                "ready_schema_version": 1,
            },
        )
        _write_json(
            runtime / f"ready.rank{rank}",
            {
                "rank": rank,
                "kv": {
                    "ok": True,
                    "rank": rank,
                    "num_layers": 5,
                    "scheduler_num_blocks": scheduler_blocks,
                    "physical_num_blocks": physical_blocks,
                    "pool_bytes": pool_bytes,
                    "map_path": map_path,
                    "ready_path": done_path,
                },
            },
        )
        _write_text(
            runtime / f"pypto_kvpool.key.rank{rank}",
            f"fixture-kv-key-rank-{rank}\n",
        )
        hashes[str(rank)] = _sha256(path)
    return hashes


def _runtime_artifact_hashes(
    runtime: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    marker_hashes = {}
    key_hashes = {}
    for rank in range(8):
        map_name = f"pypto_kvpool_map.json.rank{rank}"
        marker_hashes[f"{map_name}.done"] = _sha256(
            runtime / f"{map_name}.done"
        )
        marker_hashes[f"ready.rank{rank}"] = _sha256(
            runtime / f"ready.rank{rank}"
        )
        key_hashes[str(rank)] = _sha256(
            runtime / f"pypto_kvpool.key.rank{rank}"
        )
    return marker_hashes, key_hashes


def _write_dfx_raw_evidence(
    run: Path,
    *,
    batch: int,
    image_ref: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Create the minimal complete raw tree used by publication fixtures."""
    _write_text(
        run / "image_audit.log",
        "\n".join(
            (
                "[audit] pin pypto      "
                "8e92b46808f9f7c09b6431ad4691503f09c12ee5 clean",
                "[audit] pin pypto-lib  "
                "491267c45875e9b1e0071eed224e2e73526799e2 clean",
                "[audit] git credential scrub: PASS",
                "[audit] attention profile: a2a3",
                "[audit] prepared swimlane reuse capability: "
                "{'available': True, 'constructed': True, 'required': '1'}",
                "[audit] build jobs: 2 (resource only)",
                "IMAGE_IMMUTABLE_AUDIT=PASS",
            )
        )
        + "\n",
    )
    audit_sha = _sha256(run / "image_audit.log")
    _write_json(
        run / "image_audit_invocation.json",
        {
            "audit_log_sha256": audit_sha,
            "image_ref": image_ref,
            "passed": True,
            "phase": "pre-source-mount",
            "schema": "step3p5.moe.pre-mount-image-audit.v1",
            "source_mount": False,
        },
    )
    _write_json(
        run / "capability_report.json",
        {
            "attention_profile": "a2a3",
            "image_commits": {
                "pypto": "8e92b46808f9f7c09b6431ad4691503f09c12ee5",
                "pypto_lib": (
                    "491267c45875e9b1e0071eed224e2e73526799e2"
                ),
            },
            "image_ref": image_ref,
            "pypto_git_head": (
                "8e92b46808f9f7c09b6431ad4691503f09c12ee5"
            ),
            "reuse_capability": {
                "environment_present": True,
                "environment_value": "1",
                "fields_available": True,
                "required": True,
                "required_fields": [
                    "enable_dep_gen",
                    "enable_chip_swimlane",
                    "l2_swimlane_reuse_dep_gen",
                ],
                "reuse_config_constructed": True,
            },
            "schema": "step3p5.moe-image-capability.v1",
        },
    )
    raw = run / "runtime" / f"bs{batch}" / "dfx_raw"
    dep_hashes: dict[str, str] = {}
    swim_hashes: dict[str, str] = {}
    for rank in range(8):
        dispatch = raw / f"rank{rank}" / "d0"
        _write_json(dispatch / "deps.json", {"rank": rank, "tasks": []})
        _write_json(
            dispatch / "l2_swimlane_records.json",
            {"metadata": {"num_cores": 72, "core_types": []}, "tasks": []},
        )
        _write_json(dispatch / "name_map.json", {"callable_id_to_name": {}})
        _write_text(dispatch / "critical_path_report.md", "# fixture\n")
        _write_json(
            dispatch / "merged_swimlane_fixture.json",
            {
                "traceEvents": [
                    {
                        "dur": 1.0,
                        "name": f"rank{rank}-task",
                        "ph": "X",
                        "ts": 0.0,
                    }
                ]
            },
        )
        dep_relative = f"dfx_outputs/rank{rank}/d0/deps.json"
        swim_relative = (
            f"dfx_outputs/rank{rank}/d0/l2_swimlane_records.json"
        )
        dep_hashes[dep_relative] = _sha256(dispatch / "deps.json")
        swim_hashes[swim_relative] = _sha256(
            dispatch / "l2_swimlane_records.json"
        )
    return dep_hashes, swim_hashes


def _write_normal_run(
    campaign: Path,
    *,
    source_kind: str,
    round_id: int,
    batch: int,
    image_ref: str,
    policy: dict,
    golden_dir: Path,
) -> None:
    run_name = f"{source_kind}-r{round_id}-normal-bs{batch}-64k"
    run = campaign / "runs" / run_name
    runtime = run / "runtime"
    batch_dir = runtime / f"bs{batch}"
    workload = _workload(batch)
    hidden_hashes = _write_hidden_pair(
        batch_dir,
        batch,
        golden_dir=golden_dir,
    )
    nonce = hashlib.sha256(run_name.encode()).hexdigest()
    kv_hashes = _write_kv_maps(runtime, batch, nonce)
    marker_hashes, key_hashes = _runtime_artifact_hashes(runtime)
    _write_text(run / "container.rc", "0\n")
    _write_text(run / "image_ref.txt", image_ref + "\n")
    _write_text(run / "run_nonce.txt", nonce + "\n")
    timing = {
        "iters": 30,
        "warmup": 5,
        "min_ms": 1.0 + batch / 1000,
        "mean_ms": 1.1 + batch / 1000,
        "p50_ms": 1.1 + batch / 1000,
        "p99_ms": 1.2 + batch / 1000,
        "max_ms": 1.3 + batch / 1000,
    }
    report = {
        "schema": "step3p5.five-layer-moe-matrix-case.v1",
        "source_kind": source_kind,
        "round": round_id,
        "mode": "normal",
        "image_ref": image_ref,
        "image_pypto_commit": (
            "8e92b46808f9f7c09b6431ad4691503f09c12ee5"
        ),
        "source": {
            "source_manifest_sha256": policy["source_manifest_sha256"],
            "decode_fwd_sha256": policy["decode_fwd_sha256"],
            "five_layer_program_sha256": "1" * 64,
            "five_layer_holder_sha256": "2" * 64,
        },
        "workload": workload,
        "timing": timing,
        "comparisons": {},
        "dfx": {},
        "files": hidden_hashes,
    }
    _write_json(batch_dir / "report.json", report)
    _write_json(
        runtime / "matrix_report.json",
        {
            "schema": "step3p5.five-layer-moe-64k-matrix.v1",
            "source_kind": source_kind,
            "round": round_id,
            "mode": "normal",
            "image_ref": image_ref,
            "image_pypto_commit": (
                "8e92b46808f9f7c09b6431ad4691503f09c12ee5"
            ),
            "batches": [batch],
            "context_len_per_sequence": 65536,
            "blocks_per_sequence": 512,
            "reports": {str(batch): f"bs{batch}/report.json"},
            "report": report,
        },
    )
    _write_json(
        run / "artifact_validation.json",
        {
            "schema": "step3p5.five-layer-moe-case-validation.v1",
            "passed": True,
            "run": run_name,
            "source_kind": source_kind,
            "round": round_id,
            "mode": "normal",
            "active_batch": batch,
            "context_len_per_sequence": 65536,
            "active_total_context_tokens": batch * 65536,
            "image_ref": image_ref,
            "run_nonce": nonce,
            "workload": {
                key: workload[key]
                for key in workload
                if key != "input_tokens"
            },
            "hidden_sha256": hidden_hashes,
            "kv_map_sha256_by_rank": kv_hashes,
            "runtime_marker_sha256": marker_hashes,
            "kv_key_sha256_by_rank": key_hashes,
        },
    )


def _source_trees(root: Path) -> tuple[Path, Path]:
    dfx = root / "dfx-source"
    route = root / "route-source"
    if dfx.exists() and route.exists():
        return dfx, route
    shared = {
        "COMMIT": "commit\n",
        analyzer.CRITICAL_DECODE: "decode\n",
        analyzer.CRITICAL_PROGRAM: "program\n",
        "tests/step3p5/harnesses/_stage_five_layer_moe.py": "stage\n",
        analyzer.CRITICAL_HOLDER: "holder\n",
        "tools/step3p5/analyze_five_layer_moe_dfx.py": "analyzer\n",
        "shared.txt": "shared\n",
    }
    for relative, content in shared.items():
        _write_text(dfx / relative, content)
        _write_text(route / relative, content)
    _write_text(dfx / "DFX_SOURCE_PROVENANCE.txt", "dfx\n")
    _write_text(route / "DFX_SOURCE_PROVENANCE.txt", "route\n")
    _write_text(dfx / "PARENT_SOURCE_SHA256SUMS", "dfx-parent\n")
    _write_text(route / "PARENT_SOURCE_SHA256SUMS", "route-parent\n")
    for relative in analyzer.ROUTE_ADDITIONS:
        _write_text(route / relative, relative + "\n")
    _manifest(dfx)
    _manifest(route)
    return dfx, route


def _campaign(
    root: Path,
    *,
    route_ready: bool,
    candidate_gate_pass: bool = True,
    matched_policy: dict | None = None,
) -> Path:
    campaign = root / "campaign"
    for batch in analyzer.BATCHES:
        for source in analyzer.SOURCES:
            run_name = f"{source}-r1-dfx-bs{batch}-64k"
            run = campaign / "runs" / run_name
            runtime = run / "runtime"
            batch_dir = runtime / f"bs{batch}"
            policy = (
                matched_policy["sources"][source]
                if matched_policy is not None
                else analyzer.DFX_POLICIES[source]
            )
            profile = policy.get("profile", source)
            source_manifest_sha256 = policy.get(
                "source_manifest_sha256",
                "a" * 64,
            )
            image_ref = (
                matched_policy["authority"]["image_ref"]
                if matched_policy is not None
                else "image@sha256:" + "9" * 64
            )
            image_pypto_commit = (
                matched_policy["authority"]["image_pypto_commit"]
                if matched_policy is not None
                else "8e92b46808f9f7c09b6431ad4691503f09c12ee5"
            )
            nonce = hashlib.sha256(run_name.encode()).hexdigest()
            workload = _workload(batch)
            golden_dir = (
                Path(
                    matched_policy["authority"]["golden_manifests"][
                        str(batch)
                    ]["path"]
                ).parent
                if matched_policy is not None
                else None
            )
            hidden_hashes = _write_hidden_pair(
                batch_dir,
                batch,
                golden_dir=golden_dir,
            )
            kv_hashes = _write_kv_maps(runtime, batch, nonce)
            marker_hashes, key_hashes = _runtime_artifact_hashes(runtime)
            dep_hashes, swim_hashes = _write_dfx_raw_evidence(
                run,
                batch=batch,
                image_ref=image_ref,
            )
            validation: dict = {
                "schema": "step3p5.five-layer-moe-case-validation.v1",
                "passed": True,
                "run": run_name,
                "source_kind": source,
                "round": 1,
                "mode": "dfx",
                "active_batch": batch,
                "context_len_per_sequence": 65536,
                "active_total_context_tokens": batch * 65536,
                "image_ref": image_ref,
                "run_nonce": nonce,
                "workload": {
                    key: workload[key]
                    for key in workload
                    if key != "input_tokens"
                },
                "hidden_sha256": hidden_hashes,
                "kv_map_sha256_by_rank": kv_hashes,
                "runtime_marker_sha256": marker_hashes,
                "kv_key_sha256_by_rank": key_hashes,
            }
            if matched_policy is not None:
                validation["dfx_gate"] = {
                    "matched_policy": {
                        "schema": matched_policy["schema"],
                        "policy_sha256": matched_policy[
                            "policy_sha256"
                        ],
                        "policy_id": policy["policy_id"],
                        "profile": profile,
                        "source_kind": policy["source_kind"],
                        "source_role": policy["source_role"],
                        "source_manifest_sha256": (
                            source_manifest_sha256
                        ),
                        "decode_fwd_sha256": policy[
                            "decode_fwd_sha256"
                        ],
                        "release_enforced": policy[
                            "release_enforced"
                        ],
                    }
                }
            _write_json(
                run / "artifact_validation.json",
                validation,
            )
            case_report = {
                    "schema": "step3p5.five-layer-moe-matrix-case.v1",
                    "source_kind": source,
                    "mode": "dfx",
                    "round": 1,
                    "image_ref": image_ref,
                    "image_pypto_commit": image_pypto_commit,
                    "comparisons": {
                        "hidden_l3": {"exact": True},
                        "hidden_l4": {"exact": True},
                    },
                    "workload": workload,
                    "timing": {"iters": 1, "warmup": 2},
                    "dfx": {
                        "dep_gen_artifacts": dep_hashes,
                        "dep_gen_preserved_after_swim": True,
                        "swimlane_artifacts": swim_hashes,
                    },
                    "source": {
                        "source_manifest_sha256": (
                            source_manifest_sha256
                        ),
                        "decode_fwd_sha256": policy[
                            "decode_fwd_sha256"
                        ],
                        "five_layer_program_sha256": (
                            _sha256(
                                root
                                / "dfx-source"
                                / analyzer.CRITICAL_PROGRAM
                            )
                            if matched_policy is not None
                            and source == "candidate"
                            else "1" * 64
                        ),
                        "five_layer_holder_sha256": (
                            _sha256(
                                root
                                / "dfx-source"
                                / analyzer.CRITICAL_HOLDER
                            )
                            if matched_policy is not None
                            and source == "candidate"
                            else "2" * 64
                        ),
                    },
                    "files": hidden_hashes,
                }
            _write_json(batch_dir / "report.json", case_report)
            _write_json(
                run / "runtime" / "matrix_report.json",
                {
                    "schema": "step3p5.five-layer-moe-64k-matrix.v1",
                    "source_kind": source,
                    "round": 1,
                    "mode": "dfx",
                    "image_ref": image_ref,
                    "image_pypto_commit": image_pypto_commit,
                    "batches": [batch],
                    "context_len_per_sequence": 65536,
                    "blocks_per_sequence": 512,
                    "reports": {str(batch): f"bs{batch}/report.json"},
                    "report": case_report,
                },
            )
            _write_text(run / "container.rc", "0\n")
            _write_text(run / "image_ref.txt", image_ref + "\n")
            _write_text(run / "run_nonce.txt", nonce + "\n")
            enforced = policy["release_enforced"]
            gate_pass = (
                candidate_gate_pass if enforced else None
            )
            if not enforced:
                status = "DIAGNOSTIC_ONLY"
                analyzer_gate = True
                blockers = []
            elif not candidate_gate_pass:
                status = "BLOCKED"
                analyzer_gate = False
                blockers = [
                    {
                        "code": "expert_aic_duration_release_failed",
                    }
                ]
            elif route_ready:
                status = "PENDING_EXTERNAL_GATE"
                analyzer_gate = True
                blockers = []
            else:
                status = "NOT_EVALUABLE"
                analyzer_gate = True
                blockers = []
            source_policy = {
                "policy_id": policy["policy_id"],
            }
            if matched_policy is not None:
                source_policy.update(
                    {
                        "source_role": policy["source_role"],
                        "enforce_candidate_release_gate": enforced,
                        "decode_sha256_prefix": policy[
                            "decode_fwd_sha256"
                        ][:8],
                    }
                )
            expert_release = {
                "release_enforced": enforced,
                "release_gate_pass": gate_pass,
                "release_gate_status": (
                    "PASS"
                    if gate_pass is True
                    else "BLOCKED"
                    if gate_pass is False
                    else "NOT_APPLICABLE"
                ),
            }
            admission = {
                "analyzer_gate_pass": analyzer_gate,
                "blockers": blockers,
                "release_readiness": {
                    "status": status,
                    "publication_allowed": False,
                    "recv_meta_publication_evidence_ready": (
                        route_ready if enforced else False
                    ),
                },
            }
            if matched_policy is not None:
                expert_release.update(
                    {
                        "profile": profile,
                        "source_policy": dict(source_policy),
                    }
                )
                admission.update(
                    {
                        "profile": profile,
                        "source_policy": dict(source_policy),
                        "expert_release_enforced": enforced,
                    }
                )
            dfx_report = {
                "profile": profile,
                "source_policy": source_policy,
                "rank_contract": {
                    "expected": [f"rank{rank}/d0" for rank in range(8)],
                    "actual": [f"rank{rank}/d0" for rank in range(8)],
                    "exact": True,
                },
                "structural_contracts": {"pass": True},
                "slice_contract": {
                    "expected_equals_observed": True,
                },
                "routed_slice_profiles": {"pass": True},
                "expert_kernel_release": expert_release,
                "admission": admission,
            }
            _write_json(
                batch_dir / "dfx_analysis" / "moe_dfx_report.json",
                dfx_report,
            )
            _write_text(
                batch_dir
                / "dfx_analysis"
                / "moe_critical_path_report.md",
                f"# {run_name}\n",
            )
            pre_mount = analyzer._validate_pre_mount_image_audit(
                run,
                image_ref=image_ref,
            )
            capability = analyzer._validate_capability_report(
                run,
                image_ref=image_ref,
                mode="dfx",
            )
            validation["pre_mount_image_audit"] = pre_mount
            validation["capability_report"] = capability
            validation["dfx_raw_evidence"] = (
                analyzer._validate_dfx_raw_evidence(
                    batch_dir,
                    dfx_artifacts=case_report["dfx"],
                )
            )
            _write_json(run / "artifact_validation.json", validation)
    if matched_policy is not None:
        _write_text(
            campaign / "dfx_campaign_spec.txt",
            "\n".join(
                (
                    f"image={matched_policy['authority']['image_ref']}",
                    "batches=1,2,4,7,8,16",
                    "context_len_per_sequence=65536",
                    "rounds=1",
                    "warmup=2",
                    "measured_iters=1",
                    "capture=separate-warm-dep-gen-then-l2-swimlane",
                    "l2_swimlane_reuse_dep_gen=required",
                    (
                        "matched_policy_sha256="
                        f"{matched_policy['policy_sha256']}"
                    ),
                    "publication_ready=required",
                )
            )
            + "\n",
        )
    return campaign


def _matched_policy(root: Path) -> dict:
    dfx, _route = _source_trees(root)
    baseline = root / "baseline-source"
    _write_text(baseline / analyzer.CRITICAL_DECODE, "baseline-decode\n")
    _manifest(baseline)
    sources = {
        "baseline": {
            "profile": "fused-56b3d477",
            "source_kind": "baseline",
            "source_role": "baseline",
            "policy_id": "moe-baseline-fused-v1",
            "source_manifest_sha256": _sha256(
                baseline / "SOURCE_SHA256SUMS"
            ),
            "decode_fwd_sha256": _sha256(
                baseline / analyzer.CRITICAL_DECODE
            ),
            "release_enforced": False,
        },
        "candidate": {
            "profile": "moe-winner",
            "source_kind": "candidate",
            "source_role": "candidate",
            "policy_id": "moe-winner-selected-tile-v1",
            "source_manifest_sha256": _sha256(
                dfx / "SOURCE_SHA256SUMS"
            ),
            "decode_fwd_sha256": _sha256(
                dfx / analyzer.CRITICAL_DECODE
            ),
            "release_enforced": True,
        },
    }
    evidence_dir = root / "authority"
    source_audit = evidence_dir / "matched_source_audit.json"
    image_ref = "image@sha256:" + "9" * 64
    raw_manifest = "e" * 64
    _write_json(
        source_audit,
        {
            "schema": analyzer.MATCHED_SOURCE_AUDIT_SCHEMA,
            "passed": True,
            "image_ref": image_ref,
            "selected_variant": sources["candidate"]["profile"],
            "selected_raw_source_manifest_sha256": raw_manifest,
            "sources": {
                source: {
                    "root": str(
                        baseline if source == "baseline" else dfx
                    ),
                    "profile": policy["profile"],
                    "source_kind": source,
                    "source_role": policy["source_role"],
                    "source_manifest_sha256": policy[
                        "source_manifest_sha256"
                    ],
                    "decode_fwd_sha256": policy["decode_fwd_sha256"],
                    "exact_tree": True,
                }
                for source, policy in sources.items()
            },
        },
    )
    selection = evidence_dir / "final_selection_report.json"
    _write_json(
        selection,
        {
            "schema": analyzer.FINAL_SELECTION_SCHEMA,
            "passed": True,
            "image_ref": image_ref,
            "selected_variant": sources["candidate"]["profile"],
            "selected_decode_fwd_sha256": sources["candidate"][
                "decode_fwd_sha256"
            ],
            "selected_raw_source_manifest_sha256": raw_manifest,
            "source_audit_sha256": _sha256(source_audit),
            "tile_decision": "KEEP_CONTROL",
            "tile_selected_variant": "mm-n64-r16",
            "activation_decision": "KEEP_ACT_N64",
            "activation_selected_variant": sources["candidate"]["profile"],
            "rejected_variants": [
                "mm-n64-r32",
                "lineage-only",
                "full-execution-patch",
            ],
        },
    )
    scripts_dir = evidence_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    for name in (
        "analyze_dfx_campaign.py",
        "analyze_matrix_correctness.py",
        "analyze_matrix_performance.py",
        "container_five_layer_matrix.sh",
        "five_layer_matrix.py",
        "run_0162_five_layer_matrix.sh",
        "run_0162_normal_campaign.sh",
        "run_0162_normal_counterbalance.sh",
        "create_normal_seal_authority.py",
        "run_0162_normal_evidence_seal.sh",
        "validate_five_layer_case.py",
        "validate_five_layer_route_case.py",
        "verify_exact_tree_manifest.py",
    ):
        shutil.copy2(Path(analyzer.__file__).with_name(name), scripts_dir / name)
    scripts_manifest = scripts_dir / "SCRIPTS_SHA256SUMS"
    _write_text(
        scripts_manifest,
        "\n".join(
            f"{_sha256(path)}  {path.name}"
            for path in sorted(scripts_dir.iterdir())
            if path.is_file() and path != scripts_manifest
        )
        + "\n",
    )
    normal_correctness = evidence_dir / "matrix_correctness_report.json"
    _write_json(
        normal_correctness,
        {
            "schema": "step3p5.five-layer-moe-64k-correctness.v1",
            "passed": True,
            "batches": list(analyzer.BATCHES),
            "context_len_per_sequence": 65536,
            "gates": {
                "health": True,
                "fresh_process_raw_exact": True,
                "baseline_equals_candidate": True,
                "batch_extension_invariance": True,
            },
        },
    )
    golden_manifests = {}
    for batch in analyzer.BATCHES:
        golden_dir = evidence_dir / "golden" / f"bs{batch}"
        files = _write_hidden_pair(golden_dir, batch)
        manifest = golden_dir / "manifest.json"
        _write_json(
            manifest,
            {
                "schema": "step3p5.five-layer-moe-golden.v3",
                "source_kind": "baseline",
                "active_batch": batch,
                "context_len_per_sequence": 65536,
                "image_ref": image_ref,
                "files": files,
            },
        )
        golden_manifests[str(batch)] = {
            "path": str(manifest),
            "sha256": _sha256(manifest),
        }
        for round_id in (1, 2, 3):
            for source_kind, policy in sources.items():
                _write_normal_run(
                    evidence_dir,
                    source_kind=source_kind,
                    round_id=round_id,
                    batch=batch,
                    image_ref=image_ref,
                    policy=policy,
                    golden_dir=golden_dir,
                )

    normal_performance = evidence_dir / "matrix_performance_report.json"
    _write_json(
        normal_performance,
        {
            "schema": "step3p5.five-layer-moe-64k-performance.v2",
            "passed": True,
            "measurement_integrity_passed": True,
            "correctness_report_passed": True,
            "hidden_hash_exact_across_selected_rounds": True,
            "image_ref": image_ref,
            "rounds": [1, 2, 3],
            "performance_non_regression_all_batches": True,
            "batches": {
                str(batch): {
                    "context_len_per_sequence": 65536,
                    "active_total_context_tokens": batch * 65536,
                    "paired_rounds": {
                        str(round_id): {}
                        for round_id in (1, 2, 3)
                    },
                    "candidate_p50_non_regression": True,
                    **{
                        source_kind: {
                            "rounds": [
                                analyzer.validate_normal_run(
                                    evidence_dir,
                                    source_kind,
                                    round_id,
                                    batch,
                                )
                                for round_id in (1, 2, 3)
                            ]
                        }
                        for source_kind in sources
                    },
                }
                for batch in analyzer.BATCHES
            },
        },
    )
    normal_seal_authority = evidence_dir / "normal_seal_authority.json"
    authority_runs = {}
    for batch in analyzer.BATCHES:
        for round_id in (1, 2, 3):
            for source_kind in sources:
                run_name = (
                    f"{source_kind}-r{round_id}-normal-bs{batch}-64k"
                )
                run = evidence_dir / "runs" / run_name
                validation_path = run / "artifact_validation.json"
                validation = json.loads(
                    validation_path.read_text(encoding="utf-8")
                )
                legacy = dict(validation)
                legacy.pop("runtime_marker_sha256", None)
                legacy.pop("kv_key_sha256_by_rank", None)
                authority_runs[run_name] = {
                    "legacy_validation_sha256": hashlib.sha256(
                        json.dumps(
                            legacy,
                            indent=2,
                            sort_keys=True,
                        ).encode("utf-8")
                    ).hexdigest(),
                    "sealed_validation_sha256": _sha256(validation_path),
                    "evidence_sha256": _run_evidence_sha256(run),
                }
    _write_json(
        normal_seal_authority,
        {
            "schema": (
                "step3p5.five-layer-moe-normal-seal-authority.v1"
            ),
            "campaign_root": str(evidence_dir.resolve()),
            "runs": authority_runs,
        },
    )
    normal_spec = evidence_dir / "normal_campaign_spec.txt"
    _write_text(
        normal_spec,
        "\n".join(
            (
                f"image={image_ref}",
                (
                    "campaign_scripts_manifest_sha256="
                    f"{_sha256(scripts_manifest)}"
                ),
                "batches=1,2,4,7,8,16",
                "context_len_per_sequence=65536",
                "warmup=5",
                "measured_iters=30",
                (
                    "order=per-bs:baseline-r1,candidate-r1,"
                    "baseline-r2,candidate-r2"
                ),
            )
        )
        + "\n",
    )
    counterbalance_spec = evidence_dir / "normal_counterbalance_spec.txt"
    _write_text(
        counterbalance_spec,
        "\n".join(
            (
                f"image={image_ref}",
                (
                    "campaign_scripts_manifest_sha256="
                    f"{_sha256(scripts_manifest)}"
                ),
                "batches=1,2,4,7,8,16",
                "context_len_per_sequence=65536",
                "round=3",
                "warmup=5",
                "measured_iters=30",
                "order=per-bs:candidate-r3,baseline-r3",
            )
        )
        + "\n",
    )
    path = root / "matched_policy.json"
    _write_json(
        path,
        {
            "schema": "step3p5.moe.matched-dfx-policy.v1",
            "authority": {
                "image_ref": image_ref,
                "image_pypto_commit": (
                    "8e92b46808f9f7c09b6431ad4691503f09c12ee5"
                ),
                "selection_report": {
                    "path": str(selection),
                    "sha256": _sha256(selection),
                },
                "source_audit": {
                    "path": str(source_audit),
                    "sha256": _sha256(source_audit),
                },
                "scripts_manifest": {
                    "path": str(scripts_manifest),
                    "sha256": _sha256(scripts_manifest),
                },
                "normal_capture_scripts_manifest": {
                    "path": str(scripts_manifest),
                    "sha256": _sha256(scripts_manifest),
                },
                "normal_seal_authority": {
                    "path": str(normal_seal_authority),
                    "sha256": _sha256(normal_seal_authority),
                },
                "normal_correctness_report": {
                    "path": str(normal_correctness),
                    "sha256": _sha256(normal_correctness),
                },
                "normal_performance_report": {
                    "path": str(normal_performance),
                    "sha256": _sha256(normal_performance),
                },
                "normal_campaign_spec": {
                    "path": str(normal_spec),
                    "sha256": _sha256(normal_spec),
                },
                "normal_counterbalance_spec": {
                    "path": str(counterbalance_spec),
                    "sha256": _sha256(counterbalance_spec),
                },
                "analyzer_sha256": _sha256(
                    dfx
                    / "tools/step3p5/analyze_five_layer_moe_dfx.py"
                ),
                "validator_sha256": _sha256(
                    Path(analyzer.__file__).with_name(
                        "validate_five_layer_case.py"
                    )
                ),
                "golden_manifests": golden_manifests,
                "workload": {
                    "layers": [0, 1, 2, 3, 4],
                    "batches": [1, 2, 4, 7, 8, 16],
                    "context_len_per_sequence": 65536,
                    "blocks_per_sequence": 512,
                    "hidden_outputs": ["hidden_l3", "hidden_l4"],
                    "hidden_exact": True,
                },
                "capture": {
                    "normal_rounds": 3,
                    "normal_iters": 30,
                    "normal_warmup": 5,
                    "counterbalanced": True,
                    "dfx_rounds": 1,
                    "dfx_iters": 1,
                    "dfx_warmup": 2,
                    "l2_swimlane_reuse_dep_gen": True,
                },
            },
            "sources": sources,
        },
    )
    return analyzer.load_matched_dfx_policy(path)


def _compatibility(
    campaign: Path,
    *,
    profile: str = "moe-winner",
    source_role: str = "candidate",
    candidate_analysis_dir: str = "dfx_analysis",
) -> Path:
    path = campaign / "source_compatibility.json"
    dfx, route = _source_trees(campaign.parent)
    dfx_entries = analyzer.verify_exact_tree(dfx)
    route_entries = analyzer.verify_exact_tree(route)
    critical_hashes = {
        relative: dfx_entries[relative]
        for relative in analyzer.CRITICAL_FILES
    }
    route_hashes = {
        relative: route_entries[relative]
        for relative in analyzer.ROUTE_ADDITIONS
    }
    first_case = json.loads(
        (
            campaign
            / "runs"
            / "candidate-r1-dfx-bs1-64k"
            / "runtime"
            / "bs1"
            / "report.json"
        ).read_text(encoding="utf-8")
    )
    batches = {}
    for batch in analyzer.BATCHES:
        case = json.loads(
            (
                campaign
                / "runs"
                / f"candidate-r1-dfx-bs{batch}-64k"
                / "runtime"
                / f"bs{batch}"
                / "report.json"
            ).read_text(encoding="utf-8")
        )
        route_runtime = (
            campaign
            / "route-sidecar"
            / f"candidate-route-bs{batch}-64k"
            / "runtime"
        )
        route_runtime.mkdir(parents=True, exist_ok=True)
        _write_hidden_pair(
            route_runtime,
            batch,
            golden_dir=campaign.parent
            / "authority"
            / "golden"
            / f"bs{batch}",
        )
        recv_meta = torch.zeros((8, 2, 8, 40), dtype=torch.int32)
        for layer in range(2):
            for source_rank in range(8):
                recv_meta[source_rank, layer, source_rank, 0] = batch * 8
        local_count = recv_meta[:, :, :, :36].sum(
            dim=2,
            dtype=torch.int64,
        ).to(torch.int32)
        torch.save(recv_meta, route_runtime / "recv_meta.pt")
        torch.save(
            local_count,
            route_runtime / "local_expert_count.pt",
        )
        torch.save(
            {
                "schema": "step3p5.five-layer-moe-recv-meta.v1",
                "recv_meta": recv_meta.permute(1, 0, 2, 3).contiguous(),
                "local_expert_count": local_count.permute(
                    1, 0, 2
                ).contiguous(),
                "window_provenance": [
                    {"layer": "L3", "window_id": f"l3-{batch}"},
                    {"layer": "L4", "window_id": f"l4-{batch}"},
                ],
            },
            route_runtime / "recv_meta_sidecar.pt",
        )
        route_artifacts = {
            name: _sha256(route_runtime / name)
            for name in analyzer.ROUTE_ARTIFACT_NAMES
        }
        sidecar_sha = route_artifacts["recv_meta_sidecar.pt"]
        route_report = route_runtime / "five_layer_moe_route_report.json"
        _write_json(
            route_report,
            {
                "provenance": {
                    "image_digest": first_case["image_ref"],
                    "source": {
                        "source_tree_manifest_sha256": _sha256(
                            route / "SOURCE_SHA256SUMS"
                        ),
                        "decode_fwd_sha256": critical_hashes[
                            analyzer.CRITICAL_DECODE
                        ],
                        "formal_program_sha256": critical_hashes[
                            analyzer.CRITICAL_PROGRAM
                        ],
                        "route_program_sha256": route_hashes[
                            "tests/step3p5/harnesses/"
                            "_five_layer_moe_route_program.py"
                        ],
                        "route_stage_sha256": route_hashes[
                            "tests/step3p5/harnesses/"
                            "_stage_five_layer_moe_route.py"
                        ],
                        "route_holder_sha256": route_hashes[
                            "tools/step3p5/five_layer_moe_route_holder.py"
                        ],
                    },
                    "input_contract": {
                        "input_tokens": case["workload"]["input_tokens"],
                        "workload": {
                            "active_batch": batch,
                            "context_len": 65536,
                        },
                    },
                },
                "artifacts": route_artifacts,
            },
        )
        validation = route_runtime / "route_artifact_validation.json"
        _write_json(
            validation,
            {
                "schema": "step3p5.five-layer-moe-route-validation.v1",
                "passed": True,
                "profile": profile,
                "source_role": source_role,
                "active_batch": batch,
                "context_len_per_sequence": 65536,
                "input_tokens": case["workload"]["input_tokens"],
                "image_ref": first_case["image_ref"],
                "source_manifest_sha256": _sha256(
                    route / "SOURCE_SHA256SUMS"
                ),
                "decode_fwd_sha256": critical_hashes[
                    analyzer.CRITICAL_DECODE
                ],
                "hidden_bit_exact": True,
                "padding_zero": True,
                "local_count_exact": True,
                "expected_routes_per_source_per_layer": batch * 8,
                "global_routes_per_layer": [batch * 64, batch * 64],
                "expected_global_routes_per_layer": batch * 64,
                "window_independence_validated": True,
                "artifacts": route_artifacts,
            },
        )
        batches[str(batch)] = {
            "dfx_run": f"candidate-r1-dfx-bs{batch}-64k",
            "route_run": f"candidate-route-bs{batch}-64k",
            "input_tokens": case["workload"]["input_tokens"],
            "sidecar_sha256": sidecar_sha,
            "route_validation_sha256": _sha256(validation),
            "route_report_sha256": _sha256(route_report),
            "profile": profile,
            "source_role": source_role,
        }
    _write_json(
        path,
        {
            "schema": analyzer.SOURCE_COMPATIBILITY_SCHEMA,
            "passed": True,
            "round": 1,
            "profile": profile,
            "source_role": source_role,
            "image_ref": first_case["image_ref"],
            "context_len_per_sequence": 65536,
            "sidecar_dir": "route-sidecar",
            "dfx_source_root": str(dfx),
            "route_source_root": str(route),
            "dfx_source_manifest_sha256": _sha256(
                dfx / "SOURCE_SHA256SUMS"
            ),
            "route_source_manifest_sha256": _sha256(
                route / "SOURCE_SHA256SUMS"
            ),
            "critical_hashes": critical_hashes,
            "route_addition_hashes": route_hashes,
            "batches": batches,
        },
    )
    compatibility_sha = analyzer._sha256(path)
    for batch in analyzer.BATCHES:
        analysis_dir = (
            campaign
            / "runs"
            / f"candidate-r1-dfx-bs{batch}-64k"
            / "runtime"
            / f"bs{batch}"
            / candidate_analysis_dir
        )
        lineage_dir = (
            analysis_dir.parent
            if analysis_dir.name == "analysis"
            else analysis_dir
        )
        (lineage_dir / "source_compatibility_sha256.txt").write_text(
            compatibility_sha + "\n",
            encoding="utf-8",
        )
        (lineage_dir / "route_validation_sha256.txt").write_text(
            batches[str(batch)]["route_validation_sha256"] + "\n",
            encoding="utf-8",
        )
    return path


def test_diagnostic_campaign_passes_before_route_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign(tmp_path, route_ready=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
        ],
    )
    assert analyzer.main() == 0
    report = json.loads(
        (campaign / "dfx_campaign.json").read_text(encoding="utf-8")
    )
    assert report["diagnostic_capture_passed"]
    assert report["candidate_kernel_release_passed_all_batches"]
    assert not report["candidate_route_evidence_ready_all_batches"]
    assert not report["publication_ready"]
    assert report["publication_blockers"] == [
        "candidate_route_evidence",
        "publication_authority",
    ]
    assert "matched_source_policy" not in report


def test_publication_gate_passes_with_route_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _matched_policy(tmp_path)
    campaign = _campaign(
        tmp_path,
        route_ready=True,
        matched_policy=policy,
    )
    compatibility = _compatibility(
        campaign,
        profile=policy["sources"]["candidate"]["profile"],
        source_role=policy["sources"]["candidate"]["source_role"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
            "--require-publication-ready",
            "--source-compatibility",
            str(compatibility),
        ],
    )
    assert analyzer.main() == 0
    report = json.loads(
        (campaign / "dfx_campaign.json").read_text(encoding="utf-8")
    )
    assert report["publication_ready"]
    assert report["source_compatibility_passed"]
    assert report["publication_blockers"] == []


def test_require_publication_rejects_candidate_gate_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _matched_policy(tmp_path)
    campaign = _campaign(
        tmp_path,
        route_ready=True,
        candidate_gate_pass=False,
        matched_policy=policy,
    )
    compatibility = _compatibility(
        campaign,
        profile=policy["sources"]["candidate"]["profile"],
        source_role=policy["sources"]["candidate"]["source_role"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
            "--require-publication-ready",
            "--source-compatibility",
            str(compatibility),
        ],
    )
    assert analyzer.main() == 2
    report = json.loads(
        (campaign / "dfx_campaign.json").read_text(encoding="utf-8")
    )
    assert not report["candidate_analyzer_gate_passed_all_batches"]
    assert not report["candidate_kernel_release_passed_all_batches"]
    assert report["publication_blockers"] == [
        "candidate_analyzer_gate",
        "candidate_kernel_release",
    ]


def test_route_ready_without_source_compatibility_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign(tmp_path, route_ready=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--require-publication-ready",
        ],
    )
    assert analyzer.main() == 2
    report = json.loads(
        (campaign / "dfx_campaign.json").read_text(encoding="utf-8")
    )
    assert not report["source_compatibility_passed"]
    assert report["publication_blockers"] == [
        "source_compatibility",
        "publication_authority",
    ]


def test_source_policy_mismatch_is_rejected(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, route_ready=False)
    path = (
        campaign
        / "runs"
        / "candidate-r1-dfx-bs1-64k"
        / "runtime"
        / "bs1"
        / "dfx_analysis"
        / "moe_dfx_report.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    report["source_policy"]["policy_id"] = "wrong"
    _write_json(path, report)
    with pytest.raises(AssertionError, match="source policy"):
        analyzer._record(
            campaign,
            source="candidate",
            round_id=1,
            batch=1,
            candidate_analysis_dir="dfx_analysis",
        )


def test_missing_hidden_comparison_is_rejected(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, route_ready=False)
    path = (
        campaign
        / "runs"
        / "candidate-r1-dfx-bs1-64k"
        / "runtime"
        / "bs1"
        / "report.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    report["comparisons"] = {}
    _write_json(path, report)
    with pytest.raises(AssertionError, match="hidden outputs"):
        analyzer._record(
            campaign,
            source="candidate",
            round_id=1,
            batch=1,
            candidate_analysis_dir="dfx_analysis",
        )


def test_generic_dual_role_matched_policy_passes_all_12_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _matched_policy(tmp_path)
    campaign = _campaign(
        tmp_path,
        route_ready=False,
        matched_policy=policy,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
        ],
    )
    assert analyzer.main() == 0
    report = json.loads(
        (campaign / "dfx_campaign.json").read_text(encoding="utf-8")
    )
    matched = report["matched_source_policy"]
    assert matched["policy_sha256"] == policy["policy_sha256"]
    assert matched["sources"]["baseline"]["profile"] == "fused-56b3d477"
    assert matched["sources"]["candidate"]["profile"] == "moe-winner"
    for batch in analyzer.BATCHES:
        for source in analyzer.SOURCES:
            record = report["batches"][str(batch)][source]
            assert (
                record["matched_policy_sha256"]
                == policy["policy_sha256"]
            )
            assert record["profile"] == policy["sources"][source][
                "profile"
            ]


def test_generic_matched_policy_publication_gate_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _matched_policy(tmp_path)
    campaign = _campaign(
        tmp_path,
        route_ready=True,
        matched_policy=policy,
    )
    compatibility = _compatibility(
        campaign,
        profile=policy["sources"]["candidate"]["profile"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
            "--source-compatibility",
            str(compatibility),
            "--require-publication-ready",
        ],
    )
    assert analyzer.main() == 0
    report = json.loads(
        (campaign / "dfx_campaign.json").read_text(encoding="utf-8")
    )
    assert report["publication_ready"]
    assert report["source_compatibility"]["profile"] == "moe-winner"


def test_matched_publication_rejects_baseline_image_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _matched_policy(tmp_path)
    campaign = _campaign(
        tmp_path,
        route_ready=False,
        matched_policy=policy,
    )
    path = (
        campaign
        / "runs"
        / "baseline-r1-dfx-bs4-64k"
        / "runtime"
        / "bs4"
        / "report.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["image_ref"] = "image@sha256:" + "f" * 64
    _write_json(path, value)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
        ],
    )
    with pytest.raises(AssertionError, match="host/case image"):
        analyzer.main()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_validation", "route validation artifact"),
        ("sidecar_tamper", "sidecar artifact hash"),
        ("route_report_tamper", "route report artifact hash"),
        ("critical_hash_missing", "critical hashes are incomplete"),
        ("extra_source_file", "exact-tree manifest mismatch"),
    ),
)
def test_matched_publication_revalidates_route_and_source_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    policy = _matched_policy(tmp_path)
    campaign = _campaign(
        tmp_path,
        route_ready=True,
        matched_policy=policy,
    )
    compatibility = _compatibility(
        campaign,
        profile=policy["sources"]["candidate"]["profile"],
        source_role=policy["sources"]["candidate"]["source_role"],
    )
    if mutation == "missing_validation":
        (
            campaign
            / "route-sidecar"
            / "candidate-route-bs1-64k"
            / "runtime"
            / "route_artifact_validation.json"
        ).unlink()
    elif mutation == "sidecar_tamper":
        (
            campaign
            / "route-sidecar"
            / "candidate-route-bs2-64k"
            / "runtime"
            / "recv_meta_sidecar.pt"
        ).write_bytes(b"tampered")
    elif mutation == "route_report_tamper":
        path = (
            campaign
            / "route-sidecar"
            / "candidate-route-bs4-64k"
            / "runtime"
            / "five_layer_moe_route_report.json"
        )
        path.write_text(path.read_text(encoding="utf-8") + "\n")
    elif mutation == "critical_hash_missing":
        value = json.loads(compatibility.read_text(encoding="utf-8"))
        del value["critical_hashes"]["COMMIT"]
        _write_json(compatibility, value)
    else:
        dfx_root = Path(
            json.loads(
                compatibility.read_text(encoding="utf-8")
            )["dfx_source_root"]
        )
        _write_text(dfx_root / "unexpected.py", "unexpected\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
            "--source-compatibility",
            str(compatibility),
            "--require-publication-ready",
        ],
    )
    with pytest.raises(AssertionError, match=message):
        analyzer.main()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("profile", "profile does not match"),
        ("manifest", "exact source tree"),
        ("batches", "exactly the six-BS matrix"),
    ),
)
def test_matched_compatibility_contract_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    policy = _matched_policy(tmp_path)
    campaign = _campaign(
        tmp_path,
        route_ready=True,
        matched_policy=policy,
    )
    compatibility = _compatibility(
        campaign,
        profile=policy["sources"]["candidate"]["profile"],
    )
    value = json.loads(compatibility.read_text(encoding="utf-8"))
    if mutation == "profile":
        value["profile"] = "wrong-profile"
    elif mutation == "manifest":
        value["dfx_source_manifest_sha256"] = "f" * 64
    else:
        del value["batches"]["16"]
    _write_json(compatibility, value)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
            "--source-compatibility",
            str(compatibility),
            "--require-publication-ready",
        ],
    )
    with pytest.raises(AssertionError, match=message):
        analyzer.main()


def test_one_run_with_wrong_matched_policy_sha_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _matched_policy(tmp_path)
    campaign = _campaign(
        tmp_path,
        route_ready=False,
        matched_policy=policy,
    )
    validation_path = (
        campaign
        / "runs"
        / "candidate-r1-dfx-bs7-64k"
        / "artifact_validation.json"
    )
    validation = json.loads(
        validation_path.read_text(encoding="utf-8")
    )
    validation["dfx_gate"]["matched_policy"]["policy_sha256"] = "f" * 64
    _write_json(validation_path, validation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
        ],
    )
    with pytest.raises(
        AssertionError,
        match="matched validation policy_sha256 mismatch",
    ):
        analyzer.main()


@pytest.mark.parametrize("source", analyzer.SOURCES)
def test_role_source_manifest_mismatch_is_rejected(
    tmp_path: Path,
    source: str,
) -> None:
    policy = _matched_policy(tmp_path)
    campaign = _campaign(
        tmp_path,
        route_ready=False,
        matched_policy=policy,
    )
    report_path = (
        campaign
        / "runs"
        / f"{source}-r1-dfx-bs1-64k"
        / "runtime"
        / "bs1"
        / "report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source"]["source_manifest_sha256"] = "f" * 64
    _write_json(report_path, report)
    with pytest.raises(AssertionError, match="source manifest policy"):
        analyzer._record(
            campaign,
            source=source,
            round_id=1,
            batch=1,
            candidate_analysis_dir="dfx_analysis",
            matched_policy=policy,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("profile", "source policy"),
        ("policy_id", "source policy"),
        ("release_enforced", "release enforcement"),
    ),
)
def test_matched_report_profile_policy_and_release_mismatch_are_rejected(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    policy = _matched_policy(tmp_path)
    campaign = _campaign(
        tmp_path,
        route_ready=False,
        matched_policy=policy,
    )
    report_path = (
        campaign
        / "runs"
        / "candidate-r1-dfx-bs1-64k"
        / "runtime"
        / "bs1"
        / "dfx_analysis"
        / "moe_dfx_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if field == "profile":
        report["profile"] = "wrong-profile"
    elif field == "policy_id":
        report["source_policy"]["policy_id"] = "wrong-policy"
    else:
        report["expert_kernel_release"]["release_enforced"] = False
    _write_json(report_path, report)
    with pytest.raises(AssertionError, match=message):
        analyzer._record(
            campaign,
            source="candidate",
            round_id=1,
            batch=1,
            candidate_analysis_dir="dfx_analysis",
            matched_policy=policy,
        )
