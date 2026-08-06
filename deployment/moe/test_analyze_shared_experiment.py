from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


MOE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MOE_DIR))

import analyze_shared_experiment as analyzer  # noqa: E402


IMAGE = "hub/image@sha256:" + "c" * 64
SOURCE_MANIFESTS = {
    "baseline": "a" * 64,
    "candidate": "b" * 64,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")


def _save_hidden(
    batch_dir: Path,
    hidden_l3: torch.Tensor,
    hidden_l4: torch.Tensor,
) -> dict[str, str]:
    batch_dir.mkdir(parents=True, exist_ok=True)
    torch.save(hidden_l3, batch_dir / "hidden_l3.pt")
    torch.save(hidden_l4, batch_dir / "hidden_l4.pt")
    return {
        "hidden_l3.pt": _sha256(batch_dir / "hidden_l3.pt"),
        "hidden_l4.pt": _sha256(batch_dir / "hidden_l4.pt"),
    }


def _golden(root: Path) -> tuple[Path, torch.Tensor, torch.Tensor]:
    golden = root / "golden" / "bs1"
    golden.mkdir(parents=True)
    hidden_l3 = (
        torch.arange(
            analyzer.TP * 4096,
            dtype=torch.float32,
        )
        .reshape(analyzer.TP, 1, 4096)
        .to(torch.bfloat16)
    )
    hidden_l4 = (hidden_l3.float() * 0.5).to(torch.bfloat16)
    files = _save_hidden(golden, hidden_l3, hidden_l4)
    _write_json(
        golden / "manifest.json",
        {
            "schema": "step3p5.five-layer-moe-golden.v3",
            "source_kind": "baseline",
            "active_batch": 1,
            "context_len_per_sequence": 65536,
            "image_ref": IMAGE,
            "files": files,
        },
    )
    return golden.parent, hidden_l3, hidden_l4


def _case(
    campaign: Path,
    *,
    source: str,
    round_id: int,
    mode: str,
    hidden_l3: torch.Tensor,
    hidden_l4: torch.Tensor,
    p50_ms: float = 10.0,
) -> Path:
    name = f"{source}-r{round_id}-{mode}-bs1-64k"
    run = campaign / "runs" / name
    batch_dir = run / "runtime" / "bs1"
    order = analyzer.EXPECTED_NORMAL_ORDER[round_id]
    order_index = order.index(source)
    stamp_index = (round_id - 1) * 2 + order_index
    _write_text(run / "container.rc", "0")
    _write_text(
        run / "run_nonce.txt",
        hashlib.sha256(name.encode()).hexdigest(),
    )
    _write_text(
        run / "started_at.txt",
        f"2026-08-05T00:{stamp_index:02d}:00+00:00",
    )
    _write_text(
        run / "finished_at.txt",
        f"2026-08-05T00:{stamp_index:02d}:30+00:00",
    )
    files = _save_hidden(batch_dir, hidden_l3, hidden_l4)
    _write_json(run / "artifact_validation.json", {"passed": True})
    _write_json(
        batch_dir / "report.json",
        {
            "schema": "step3p5.five-layer-moe-matrix-case.v1",
            "source_kind": source,
            "round": round_id,
            "mode": mode,
            "image_ref": IMAGE,
            "source": {
                "source_manifest_sha256": SOURCE_MANIFESTS[source],
                "decode_fwd_sha256": (analyzer.EXPECTED_DECODE_SHA256[source]),
            },
            "workload": {
                "active_batch": 1,
                "context_len_per_sequence": 65536,
                "blocks_per_sequence": 512,
                "active_total_context_tokens": 65536,
            },
            "timing": {
                "p50_ms": p50_ms,
                "iters": 30 if mode == "normal" else 1,
                "warmup": 5 if mode == "normal" else 2,
            },
            "comparisons": (
                {
                    "hidden_l3": {"exact": True},
                    "hidden_l4": {"exact": True},
                }
                if mode == "dfx"
                else {}
            ),
            "files": files,
        },
    )
    return run


def _resource(
    count: int,
    *,
    duration_us: float,
    peak: int,
    resource_name: str,
    task_id: str,
    start: int,
    end: int,
) -> dict:
    return {
        "available": count > 0,
        "expected_slices": count,
        "observed_slices": count,
        "peak_concurrency": peak,
        "physical_slices": [
            {
                "task_id": task_id,
                "core_id": index,
                "resource": resource_name,
                "start_tick": start,
                "end_tick": end,
                "service_span_us": duration_us,
            }
            for index in range(count)
        ],
    }


def _stage(
    task_id: str,
    *,
    order: int,
    blocks: int,
    start: int,
    end: int,
    active_resource: str,
    duration_us: float,
    peak: int,
    predecessors: list[str] | None = None,
) -> dict:
    resources = {}
    for resource_name in ("aic", "aiv"):
        active = resource_name == active_resource
        resources[resource_name] = _resource(
            blocks if active else 0,
            duration_us=duration_us,
            peak=peak if active else 0,
            resource_name=resource_name,
            task_id=task_id,
            start=start,
            end=end,
        )
    return {
        "task_ids": [task_id],
        "task_instances": 1,
        "logical_blocks": blocks,
        "blocks_per_task": [blocks],
        "start_tick": start,
        "end_tick": end,
        "stage_span_us": float(end - start),
        "resources": resources,
        "task_instance_details": [
            {
                "task_id": task_id,
                "name": task_id,
                "order": order,
                "block_num": blocks,
                "start_tick": start,
                "end_tick": end,
                "timing_evidence": {
                    "queue_delay": {
                        "predecessor_task_ids": predecessors or [],
                    }
                },
            }
        ],
    }


def _simple_stage(start: int, end: int) -> dict:
    return {
        "start_tick": start,
        "end_tick": end,
        "stage_span_us": float(end - start),
    }


def _baseline_layers(rank: int) -> dict[str, dict]:
    result = {}
    for layer_index, layer in enumerate(analyzer.LAYERS):
        prefix = f"b{rank}-{layer_index}"
        shared = _stage(
            f"{prefix}-shared",
            order=10,
            blocks=1,
            start=100,
            end=200,
            active_resource="aic",
            duration_us=100.0,
            peak=1,
        )
        allreduce = _stage(
            f"{prefix}-ar",
            order=11,
            blocks=1,
            start=200,
            end=240,
            active_resource="aiv",
            duration_us=40.0,
            peak=1,
            predecessors=[f"{prefix}-shared"],
        )
        result[layer] = {
            "shared_mlp": shared,
            "shared_all_reduce": allreduce,
            "expert_gate": _simple_stage(250, 290),
            "combine_scatter": _simple_stage(290, 350),
            "combine_wait": _simple_stage(350, 450),
        }
    return result


def _candidate_layers(rank: int, down_us: float) -> dict[str, dict]:
    result = {}
    for layer_index, layer in enumerate(analyzer.LAYERS):
        prefix = f"c{rank}-{layer_index}"
        mm_id = f"{prefix}-mm"
        act_id = f"{prefix}-act"
        down_id = f"{prefix}-down"
        ar_id = f"{prefix}-ar"
        result[layer] = {
            "shared_gate_up": _stage(
                mm_id,
                order=10,
                blocks=5,
                start=100,
                end=120,
                active_resource="aic",
                duration_us=20.0,
                peak=5,
            ),
            "shared_gate_up_act": _stage(
                act_id,
                order=11,
                blocks=5,
                start=120,
                end=130,
                active_resource="aiv",
                duration_us=2.0,
                peak=5,
                predecessors=[mm_id],
            ),
            "shared_down": _stage(
                down_id,
                order=12,
                blocks=16,
                start=130,
                end=170,
                active_resource="aic",
                duration_us=down_us,
                peak=16,
                predecessors=[act_id],
            ),
            "shared_split": {
                "task_ids": [mm_id, act_id, down_id],
                "task_instances": 3,
                "logical_blocks": 26,
                "blocks_per_task": [5, 5, 16],
                "start_tick": 100,
                "end_tick": 170,
                "stage_span_us": 70.0,
            },
            "shared_all_reduce": _stage(
                ar_id,
                order=13,
                blocks=1,
                start=170,
                end=210,
                active_resource="aiv",
                duration_us=40.0,
                peak=1,
                predecessors=[down_id],
            ),
            "expert_gate": _simple_stage(220, 260),
            "combine_scatter": _simple_stage(260, 315),
            "combine_wait": _simple_stage(315, 417),
        }
    return result


def _dfx_report(source: str, *, down_us: float) -> dict:
    ranks = {}
    for rank in range(analyzer.TP):
        ranks[f"rank{rank}/d0"] = {
            "layers": (
                _baseline_layers(rank)
                if source == "baseline"
                else _candidate_layers(rank, down_us)
            )
        }
    return {
        "schema": "step3p5.five-layer-moe-dfx.v6",
        "profile": analyzer.EXPECTED_DFX_PROFILE[source],
        "source_policy": {
            "policy_id": analyzer.EXPECTED_DFX_POLICY[source],
            "decode_sha256_prefix": analyzer.EXPECTED_DECODE_SHA256[source][:8],
            "source_role": analyzer.EXPECTED_DFX_ROLE[source],
            "enforce_candidate_release_gate": True,
            "experimental": source == "candidate",
        },
        "rank_contract": {
            "exact": True,
            "expected": list(analyzer.RANKS),
            "actual": list(analyzer.RANKS),
        },
        "structural_contracts": {"pass": True},
        "slice_contract": {
            "expected_equals_observed": True,
            "errors": [],
        },
        "routed_slice_profiles": {"pass": True},
        "expert_kernel_release": {
            "pass": True,
            "diagnostic_pass": True,
            "release_gate_pass": True,
            "release_gate_status": "PASS",
            "coverage_pass": True,
            "duration_pass": True,
            "activation_pass": True,
            "profile": analyzer.EXPECTED_DFX_PROFILE[source],
            "release_enforced": True,
            "coverage_errors": [],
            "duration_errors": [],
            "activation_errors": [],
            "duration_limits_us": {
                "p50_min": 10.0,
                "p50_max": 30.0,
                "p90_max": 30.0,
                "p99_max": 60.0,
                "max": 100.0,
            },
            "source_policy": {
                "policy_id": analyzer.EXPECTED_DFX_POLICY[source],
            },
        },
        "admission": {
            "pass": True,
            "analyzer_gate_pass": True,
            "profile": analyzer.EXPECTED_DFX_PROFILE[source],
            "expert_release_enforced": True,
            "blockers": [],
            "source_policy": {
                "policy_id": analyzer.EXPECTED_DFX_POLICY[source],
            },
        },
        "ranks": ranks,
    }


def _full_artifacts(
    campaign: Path,
    *,
    hidden_l3: torch.Tensor,
    hidden_l4: torch.Tensor,
    down_us: float,
) -> None:
    for source in analyzer.SOURCES:
        run = _case(
            campaign,
            source=source,
            round_id=1,
            mode="dfx",
            hidden_l3=hidden_l3,
            hidden_l4=hidden_l4,
        )
        _write_json(
            run / "runtime" / "bs1" / "dfx_analysis" / "moe_dfx_report.json",
            _dfx_report(source, down_us=down_us),
        )
    route_runtime = (
        campaign
        / "route-sidecar"
        / "candidate-shared-route-bs1-64k"
        / "runtime"
    )
    route_runtime.mkdir(parents=True, exist_ok=True)
    sidecar_path = route_runtime / "recv_meta_sidecar.pt"
    torch.save({"schema": "step3p5.five-layer-moe-recv-meta.v1"}, sidecar_path)
    _write_json(
        route_runtime / "route_artifact_validation.json",
        {
            "schema": "step3p5.five-layer-moe-route-validation.v1",
            "passed": True,
            "active_batch": 1,
            "context_len_per_sequence": 65536,
            "image_ref": IMAGE,
            "profile": "shared-split",
            "source_role": "candidate",
            "source_manifest_sha256": SOURCE_MANIFESTS["candidate"],
            "decode_fwd_sha256": (analyzer.EXPECTED_DECODE_SHA256["candidate"]),
            "hidden_bit_exact": True,
            "padding_zero": True,
            "local_count_exact": True,
            "global_routes_per_layer": [64, 64],
            "expected_global_routes_per_layer": 64,
            "window_independence_validated": True,
            "artifacts": {
                "recv_meta_sidecar.pt": _sha256(sidecar_path),
            },
        },
    )


def _fixture(
    root: Path,
    *,
    include_full: bool,
    down_us: float = 12.0,
) -> tuple[Path, Path]:
    campaign = root / "campaign"
    golden, hidden_l3, hidden_l4 = _golden(root)
    p50_values = {
        "baseline": [10.0, 10.2, 9.8],
        "candidate": [9.4, 9.5, 9.3],
    }
    for round_id in analyzer.ROUNDS:
        for source in analyzer.SOURCES:
            _case(
                campaign,
                source=source,
                round_id=round_id,
                mode="normal",
                hidden_l3=hidden_l3,
                hidden_l4=hidden_l4,
                p50_ms=p50_values[source][round_id - 1],
            )
    _write_json(
        campaign / "shared_experiment_spec.json",
        {
            "schema": "step3p5.bs1-shared-expert-experiment.v1",
            "authoritative_date": "2026-08-05",
            "image_ref": IMAGE,
            "workload": {
                "active_batch": 1,
                "context_len_per_sequence": 65536,
                "blocks_per_sequence": 512,
                "active_total_context_tokens": 65536,
            },
            "normal": {
                "iters": 30,
                "warmup": 5,
                "round_order": [
                    {
                        "round": round_id,
                        "order": list(analyzer.EXPECTED_NORMAL_ORDER[round_id]),
                    }
                    for round_id in analyzer.ROUNDS
                ],
            },
            "dfx": {
                "round": 1,
                "order": ["baseline", "candidate"],
                "iters": 1,
                "warmup": 2,
                "profiles": analyzer.EXPECTED_DFX_PROFILE,
            },
            "route": {"profile": "shared-split"},
            "sources": {
                source: {
                    "decode_fwd_sha256": analyzer.EXPECTED_DECODE_SHA256[source],
                    "source_manifest_sha256": SOURCE_MANIFESTS[source],
                }
                for source in analyzer.SOURCES
            },
        },
    )
    if include_full:
        _full_artifacts(
            campaign,
            hidden_l3=hidden_l3,
            hidden_l4=hidden_l4,
            down_us=down_us,
        )
    return campaign, golden


def test_full_shared_experiment_acceptance_passes(tmp_path: Path) -> None:
    campaign, golden = _fixture(tmp_path, include_full=True)
    assert (
        analyzer.main(
            [
                "--campaign",
                str(campaign),
                "--golden",
                str(golden),
            ]
        )
        == 0
    )
    report = json.loads(
        (campaign / "shared_experiment_acceptance.json").read_text(encoding="utf-8")
    )
    assert report["passed"]
    assert report["normal"]["wall"]["median_wall_p50_gain_pct"] >= 2.0
    assert report["dfx"]["shared_chain"]["chain_passed"]
    assert report["dfx"]["shared_chain"]["activation_aiv_only"]
    assert report["route_sidecar"]["passed"]


def test_shared_down_too_fine_writes_complete_failure(
    tmp_path: Path,
) -> None:
    campaign, golden = _fixture(
        tmp_path,
        include_full=True,
        down_us=8.0,
    )
    assert (
        analyzer.main(
            [
                "--campaign",
                str(campaign),
                "--golden",
                str(golden),
            ]
        )
        == 1
    )
    report = json.loads(
        (campaign / "shared_experiment_acceptance.json").read_text(encoding="utf-8")
    )
    assert not report["passed"]
    down = report["dfx"]["kernel_grain"]["shared_down_aic"]
    assert down["requires_coarser_down_block_comparison"]
    assert down["required_comparisons"] == ["8-block", "4-block"]
    assert report["route_sidecar"]["passed"]
    assert set(report["dfx"]["comparative_metrics"]) == {"L3", "L4"}
    assert any(item["code"] == "dfx_shared_down_grain" for item in report["blockers"])
    markdown = (campaign / "shared_experiment_acceptance.md").read_text(
        encoding="utf-8"
    )
    assert "8-block and 4-block" in markdown


def test_normal_only_requires_no_dfx_or_route_artifacts(
    tmp_path: Path,
) -> None:
    campaign, golden = _fixture(tmp_path, include_full=False)
    assert (
        analyzer.main(
            [
                "--campaign",
                str(campaign),
                "--golden",
                str(golden),
                "--normal-only",
            ]
        )
        == 0
    )
    report = json.loads(
        (campaign / "shared_experiment_normal_only.json").read_text(encoding="utf-8")
    )
    assert report["passed"]
    assert report["dfx"]["status"] == "SKIPPED_NORMAL_ONLY"
    assert report["route_sidecar"]["status"] == "SKIPPED_NORMAL_ONLY"


def test_normal_metadata_is_fail_closed(tmp_path: Path) -> None:
    campaign, golden = _fixture(tmp_path, include_full=False)
    report_path = (
        campaign
        / "runs"
        / "baseline-r1-normal-bs1-64k"
        / "runtime"
        / "bs1"
        / "report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["timing"]["iters"] = 1
    _write_json(report_path, report)
    assert (
        analyzer.main(
            ["--campaign", str(campaign), "--golden", str(golden), "--normal-only"]
        )
        == 1
    )


def test_release_policy_contradiction_is_fail_closed(tmp_path: Path) -> None:
    campaign, golden = _fixture(tmp_path, include_full=True)
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
    report["expert_kernel_release"]["release_enforced"] = False
    report["admission"]["expert_release_enforced"] = False
    _write_json(report_path, report)
    assert analyzer.main(["--campaign", str(campaign), "--golden", str(golden)]) == 1


def test_forged_peak_concurrency_is_fail_closed(tmp_path: Path) -> None:
    campaign, golden = _fixture(tmp_path, include_full=True)
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
    stage = report["ranks"]["rank0/d0"]["layers"]["L3"]["shared_gate_up"]
    for index, item in enumerate(
        stage["resources"]["aic"]["physical_slices"]
    ):
        item["start_tick"] = 100 + index * 2
        item["end_tick"] = 102 + index * 2
    _write_json(report_path, report)
    assert analyzer.main(["--campaign", str(campaign), "--golden", str(golden)]) == 1


def test_stale_shared_split_timeline_is_fail_closed(tmp_path: Path) -> None:
    campaign, golden = _fixture(tmp_path, include_full=True)
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
    report["ranks"]["rank0/d0"]["layers"]["L3"]["shared_split"]["end_tick"] = 100
    _write_json(report_path, report)
    assert analyzer.main(["--campaign", str(campaign), "--golden", str(golden)]) == 1
