#!/usr/bin/env python3
"""Accept or reject the BS1 five-layer shared-expert experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


TP = 8
BATCH = 1
CONTEXT_LEN = 65536
BLOCKS_PER_SEQUENCE = 512
LAYERS = ("L3", "L4")
RANKS = tuple(f"rank{rank}/d0" for rank in range(TP))
ROUNDS = (1, 2, 3)
SOURCES = ("baseline", "candidate")
HIDDEN_NAMES = ("hidden_l3", "hidden_l4")
NORMAL_ITERS = 30
NORMAL_WARMUP = 5
DFX_ITERS = 1
DFX_WARMUP = 2
EXPECTED_NORMAL_ORDER = {
    1: ("baseline", "candidate"),
    2: ("candidate", "baseline"),
    3: ("baseline", "candidate"),
}
EXPECTED_DECODE_SHA256 = {
    "baseline": ("65b0b8bf139aa40a5cf67317148dc16193ff22a81b394fbfe86e31ea05623e08"),
    "candidate": ("572ea2a2b0ceab8952cbd8d7c1f383351fb877e11b01b4d68a60697f4508576b"),
}
EXPECTED_DFX_PROFILE = {
    "baseline": "row16",
    "candidate": "shared-split",
}
EXPECTED_DFX_POLICY = {
    "baseline": "shared-experiment-reference-row16-v1",
    "candidate": "shared-experiment-5-5-16-v1",
}
EXPECTED_DFX_ROLE = {
    "baseline": "reference",
    "candidate": "candidate",
}
SHARED_BLOCKS = {
    "shared_gate_up": 5,
    "shared_gate_up_act": 5,
    "shared_down": 16,
}
ROUTED_ENVELOPE_STAGES = (
    "expert_gate_up",
    "expert_gate",
    "expert_up",
    "expert_gate_up_act",
    "routed_h_quant",
    "expert_down",
    "combine_scatter",
)
WALL_GAIN_MIN_PCT = 2.0
SHARED_SPAN_GAIN_MIN_PCT = 20.0
REGRESSION_LIMIT_PCT = 5.0
REGRESSION_LIMIT_US = 10.0
GATE_UP_LIMITS_US = {
    "p50_min": 10.0,
    "p50_max": 30.0,
    "p90_max": 30.0,
    "p99_max": 60.0,
    "max": 100.0,
}


@dataclass
class CheckBook:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        code: str,
        passed: bool | None,
        message: str,
        *,
        phase: str,
        scope: str = "",
        evidence: Any = None,
    ) -> None:
        item = {
            "code": code,
            "phase": phase,
            "scope": scope,
            "status": ("SKIP" if passed is None else "PASS" if passed else "FAIL"),
            "message": message,
        }
        if evidence is not None:
            item["evidence"] = evidence
        self.checks.append(item)

    @property
    def passed(self) -> bool:
        return not any(item["status"] == "FAIL" for item in self.checks)

    @property
    def blockers(self) -> list[dict[str, Any]]:
        return [
            {
                "code": item["code"],
                "phase": item["phase"],
                "scope": item["scope"],
                "message": item["message"],
            }
            for item in self.checks
            if item["status"] == "FAIL"
        ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--golden", required=True)
    parser.add_argument(
        "--normal-only",
        action="store_true",
        help="validate only the six BS1 normal runs before DFX capture",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_digest_ref(value: object) -> bool:
    if not isinstance(value, str) or "@sha256:" not in value:
        return False
    return _is_sha256(value.rsplit("@sha256:", 1)[1])


def _read_json(
    path: Path,
    book: CheckBook,
    *,
    phase: str,
    scope: str,
) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("expected a JSON object")
    except Exception as exc:
        book.add(
            "artifact_json_read",
            False,
            f"cannot read {path}: {exc}",
            phase=phase,
            scope=scope,
        )
        return None
    return value


def _read_text(
    path: Path,
    book: CheckBook,
    *,
    phase: str,
    scope: str,
) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        book.add(
            "artifact_text_read",
            False,
            f"cannot read {path}: {exc}",
            phase=phase,
            scope=scope,
        )
        return None


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _read_tensor(
    path: Path,
    book: CheckBook,
    *,
    phase: str,
    scope: str,
) -> torch.Tensor | None:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor):
            raise TypeError("expected a tensor")
    except Exception as exc:
        book.add(
            "artifact_tensor_read",
            False,
            f"cannot read {path}: {exc}",
            phase=phase,
            scope=scope,
        )
        return None
    return value


def _nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile without samples")
    ordered = sorted(float(value) for value in values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(quantile * len(ordered)) - 1),
    )
    return ordered[index]


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "p50_us": None,
            "p90_us": None,
            "p99_us": None,
            "max_us": None,
        }
    return {
        "count": len(values),
        "p50_us": round(_nearest_rank(values, 0.50), 4),
        "p90_us": round(_nearest_rank(values, 0.90), 4),
        "p99_us": round(_nearest_rank(values, 0.99), 4),
        "max_us": round(max(values), 4),
    }


def _resolve_golden_dir(root: Path) -> Path:
    if (root / "bs1" / "manifest.json").is_file():
        return root / "bs1"
    return root


def _load_golden(
    golden_root: Path,
    book: CheckBook,
) -> dict[str, Any]:
    golden_dir = _resolve_golden_dir(golden_root)
    manifest = _read_json(
        golden_dir / "manifest.json",
        book,
        phase="normal",
        scope="golden",
    )
    issues: list[str] = []
    tensors: dict[str, torch.Tensor] = {}
    hashes: dict[str, str] = {}
    if manifest is None:
        issues.append("manifest is unavailable")
    else:
        if manifest.get("schema") != "step3p5.five-layer-moe-golden.v3":
            issues.append("unsupported golden schema")
        if manifest.get("source_kind") != "baseline":
            issues.append("golden source_kind is not baseline")
        if manifest.get("active_batch") != BATCH:
            issues.append("golden active_batch is not 1")
        if manifest.get("context_len_per_sequence") != CONTEXT_LEN:
            issues.append("golden context_len_per_sequence is not 65536")

    hidden: dict[str, Any] = {}
    for name in HIDDEN_NAMES:
        path = golden_dir / f"{name}.pt"
        tensor = _read_tensor(
            path,
            book,
            phase="normal",
            scope=f"golden/{name}",
        )
        item_issues: list[str] = []
        digest = None
        if path.is_file():
            digest = _sha256(path)
            hashes[path.name] = digest
        if tensor is None:
            item_issues.append("tensor is unavailable")
        else:
            tensors[name] = tensor
            if tensor.dtype != torch.bfloat16:
                item_issues.append(f"dtype is {tensor.dtype}, expected bfloat16")
            if tuple(tensor.shape) != (TP, BATCH, 4096):
                item_issues.append(
                    f"shape is {tuple(tensor.shape)}, expected {(TP, BATCH, 4096)}"
                )
        expected_hash = (
            manifest.get("files", {}).get(path.name)
            if isinstance(manifest, dict)
            else None
        )
        if digest is None or expected_hash != digest:
            item_issues.append("manifest file hash does not match")
        book.add(
            "golden_hidden_contract",
            not item_issues,
            "; ".join(item_issues) if item_issues else "golden tensor is frozen",
            phase="normal",
            scope=name,
            evidence={"sha256": digest},
        )
        hidden[name] = {
            "path": str(path),
            "sha256": digest,
            "issues": item_issues,
        }
        issues.extend(f"{name}: {issue}" for issue in item_issues)

    book.add(
        "golden_manifest_contract",
        not issues,
        "; ".join(issues) if issues else "frozen BS1 golden is valid",
        phase="normal",
        scope="golden",
    )
    return {
        "directory": str(golden_dir),
        "manifest": manifest or {},
        "tensors": tensors,
        "hashes": hashes,
        "hidden": hidden,
        "valid": not issues,
    }


def _load_experiment_spec(
    campaign: Path,
    book: CheckBook,
) -> dict[str, Any]:
    path = campaign / "shared_experiment_spec.json"
    value = _read_json(
        path,
        book,
        phase="normal",
        scope="experiment-spec",
    )
    issues: list[str] = []
    if value is None:
        value = {}
        issues.append("experiment spec is unavailable")
    if value.get("schema") != "step3p5.bs1-shared-expert-experiment.v1":
        issues.append("unsupported experiment spec schema")
    if value.get("authoritative_date") != "2026-08-05":
        issues.append("authoritative_date is not 2026-08-05")
    if not _is_digest_ref(value.get("image_ref")):
        issues.append("image_ref is not digest-pinned")

    workload = value.get("workload")
    expected_workload = {
        "active_batch": BATCH,
        "context_len_per_sequence": CONTEXT_LEN,
        "blocks_per_sequence": BLOCKS_PER_SEQUENCE,
        "active_total_context_tokens": CONTEXT_LEN,
    }
    if not isinstance(workload, dict):
        issues.append("workload is unavailable")
        workload = {}
    for key, expected in expected_workload.items():
        if workload.get(key) != expected:
            issues.append(
                f"workload.{key}={workload.get(key)!r}, expected {expected!r}"
            )

    normal = value.get("normal")
    expected_round_order = [
        {
            "round": round_id,
            "order": list(EXPECTED_NORMAL_ORDER[round_id]),
        }
        for round_id in ROUNDS
    ]
    if not isinstance(normal, dict):
        issues.append("normal spec is unavailable")
        normal = {}
    if normal.get("iters") != NORMAL_ITERS:
        issues.append(f"normal.iters is not {NORMAL_ITERS}")
    if normal.get("warmup") != NORMAL_WARMUP:
        issues.append(f"normal.warmup is not {NORMAL_WARMUP}")
    if normal.get("round_order") != expected_round_order:
        issues.append("normal.round_order is not the frozen counterbalance")

    dfx = value.get("dfx")
    expected_dfx = {
        "round": 1,
        "order": ["baseline", "candidate"],
        "iters": DFX_ITERS,
        "warmup": DFX_WARMUP,
        "profiles": EXPECTED_DFX_PROFILE,
    }
    if not isinstance(dfx, dict):
        issues.append("DFX spec is unavailable")
        dfx = {}
    for key, expected in expected_dfx.items():
        if dfx.get(key) != expected:
            issues.append(f"dfx.{key} does not match the frozen contract")

    route = value.get("route")
    if not isinstance(route, dict):
        issues.append("route spec is unavailable")
        route = {}
    if route.get("profile") != "shared-split":
        issues.append("route.profile is not shared-split")

    sources = value.get("sources")
    if not isinstance(sources, dict):
        issues.append("source spec is unavailable")
        sources = {}
    for source in SOURCES:
        source_info = sources.get(source)
        if not isinstance(source_info, dict):
            issues.append(f"sources.{source} is unavailable")
            continue
        if source_info.get("decode_fwd_sha256") != EXPECTED_DECODE_SHA256[source]:
            issues.append(f"sources.{source}.decode_fwd_sha256 mismatch")
        if not _is_sha256(source_info.get("source_manifest_sha256")):
            issues.append(f"sources.{source}.source_manifest_sha256 is invalid")

    book.add(
        "experiment_spec_contract",
        not issues,
        "; ".join(issues) if issues else "experiment spec is frozen",
        phase="normal",
        scope="experiment-spec",
        evidence={"path": str(path)},
    )
    return {
        "path": str(path),
        "payload": value,
        "valid": not issues,
        "issues": issues,
    }


def _run_name(
    source: str,
    round_id: int,
    mode: str,
) -> str:
    return f"{source}-r{round_id}-{mode}-bs1-64k"


def _hidden_contract(
    batch_dir: Path,
    report: dict[str, Any] | None,
    golden: dict[str, Any],
    book: CheckBook,
    *,
    phase: str,
    scope: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in HIDDEN_NAMES:
        path = batch_dir / f"{name}.pt"
        actual = _read_tensor(
            path,
            book,
            phase=phase,
            scope=f"{scope}/{name}",
        )
        expected = golden["tensors"].get(name)
        issues: list[str] = []
        bad_count = None
        digest = _sha256(path) if path.is_file() else None
        if actual is None:
            issues.append("actual tensor is unavailable")
        else:
            if actual.dtype != torch.bfloat16:
                issues.append(f"dtype is {actual.dtype}, expected bfloat16")
            if tuple(actual.shape) != (TP, BATCH, 4096):
                issues.append(
                    f"shape is {tuple(actual.shape)}, expected {(TP, BATCH, 4096)}"
                )
        if expected is None:
            issues.append("golden tensor is unavailable")
        elif (
            actual is not None
            and actual.dtype == expected.dtype == torch.bfloat16
            and tuple(actual.shape) == tuple(expected.shape)
        ):
            actual_bits = actual.contiguous().view(torch.int16)
            expected_bits = expected.contiguous().view(torch.int16)
            bad_count = int(torch.count_nonzero(actual_bits != expected_bits).item())
            if bad_count:
                issues.append(f"{bad_count} BF16 values differ at the bit level")
        elif actual is not None:
            issues.append("tensor cannot be compared bitwise")

        reported_hash = (
            report.get("files", {}).get(path.name) if isinstance(report, dict) else None
        )
        if digest is None or reported_hash != digest:
            issues.append("case report file hash does not match")
        book.add(
            "hidden_bf16_bit_exact",
            not issues,
            "; ".join(issues) if issues else "BF16 raw values match frozen golden",
            phase=phase,
            scope=f"{scope}/{name}",
            evidence={
                "path": str(path),
                "sha256": digest,
                "bad_count": bad_count,
            },
        )
        result[name] = {
            "path": str(path),
            "sha256": digest,
            "bad_count": bad_count,
            "passed": not issues,
            "issues": issues,
        }
    return result


def _case_record(
    campaign: Path,
    golden: dict[str, Any],
    book: CheckBook,
    *,
    source: str,
    round_id: int,
    mode: str,
) -> dict[str, Any]:
    name = _run_name(source, round_id, mode)
    run = campaign / "runs" / name
    batch_dir = run / "runtime" / "bs1"
    phase = "normal" if mode == "normal" else "dfx"
    validation = _read_json(
        run / "artifact_validation.json",
        book,
        phase=phase,
        scope=name,
    )
    report = _read_json(
        batch_dir / "report.json",
        book,
        phase=phase,
        scope=name,
    )
    issues: list[str] = []
    container_rc = _read_text(
        run / "container.rc",
        book,
        phase=phase,
        scope=name,
    )
    if container_rc != "0":
        issues.append(f"container.rc is {container_rc!r}, expected '0'")
    run_nonce = _read_text(
        run / "run_nonce.txt",
        book,
        phase=phase,
        scope=name,
    )
    if not _is_sha256(run_nonce):
        issues.append("run_nonce is not a lowercase SHA256")
    started_at_text = _read_text(
        run / "started_at.txt",
        book,
        phase=phase,
        scope=name,
    )
    finished_at_text = _read_text(
        run / "finished_at.txt",
        book,
        phase=phase,
        scope=name,
    )
    started_at = _parse_timestamp(started_at_text)
    finished_at = _parse_timestamp(finished_at_text)
    if started_at is None or finished_at is None:
        issues.append("run start/finish timestamps are unavailable or invalid")
    elif finished_at < started_at:
        issues.append("run finished before it started")
    if validation is None or validation.get("passed") is not True:
        issues.append("artifact_validation.passed is not true")
    if report is None:
        issues.append("case report is unavailable")
        report = {}
    expected_identity = {
        "source_kind": source,
        "round": round_id,
        "mode": mode,
    }
    for key, expected in expected_identity.items():
        if report.get(key) != expected:
            issues.append(f"{key}={report.get(key)!r}, expected {expected!r}")
    workload = report.get("workload")
    if not isinstance(workload, dict):
        issues.append("workload is unavailable")
        workload = {}
    expected_workload = {
        "active_batch": BATCH,
        "context_len_per_sequence": CONTEXT_LEN,
        "blocks_per_sequence": BLOCKS_PER_SEQUENCE,
        "active_total_context_tokens": CONTEXT_LEN,
    }
    for key, expected in expected_workload.items():
        if workload.get(key) != expected:
            issues.append(
                f"workload.{key}={workload.get(key)!r}, expected {expected!r}"
            )
    source_info = report.get("source")
    if not isinstance(source_info, dict):
        issues.append("source provenance is unavailable")
        source_info = {}
    decode_sha = source_info.get("decode_fwd_sha256")
    if decode_sha != EXPECTED_DECODE_SHA256[source]:
        issues.append(
            f"decode SHA is {decode_sha!r}, expected {EXPECTED_DECODE_SHA256[source]}"
        )
    source_manifest = source_info.get("source_manifest_sha256")
    if not _is_sha256(source_manifest):
        issues.append("source manifest SHA is not a lowercase SHA256")
    image_ref = report.get("image_ref")
    if not isinstance(image_ref, str) or not image_ref:
        issues.append("image_ref is missing")
    if mode == "dfx":
        comparisons = report.get("comparisons")
        if not (
            isinstance(comparisons, dict)
            and set(comparisons) == set(HIDDEN_NAMES)
            and all(
                isinstance(item, dict) and item.get("exact") is True
                for item in comparisons.values()
            )
        ):
            issues.append("DFX report comparisons are not exact")
    timing = report.get("timing")
    p50_ms = None
    expected_iters = NORMAL_ITERS if mode == "normal" else DFX_ITERS
    expected_warmup = NORMAL_WARMUP if mode == "normal" else DFX_WARMUP
    if not isinstance(timing, dict):
        issues.append("timing is unavailable")
    else:
        if timing.get("iters") != expected_iters:
            issues.append(
                f"timing.iters={timing.get('iters')!r}, "
                f"expected {expected_iters}"
            )
        if timing.get("warmup") != expected_warmup:
            issues.append(
                f"timing.warmup={timing.get('warmup')!r}, "
                f"expected {expected_warmup}"
            )
    if mode == "normal":
        if isinstance(timing, dict):
            try:
                p50_ms = float(timing["p50_ms"])
                if not math.isfinite(p50_ms) or p50_ms <= 0:
                    raise ValueError("p50 must be finite and positive")
            except Exception as exc:
                issues.append(f"invalid timing.p50_ms: {exc}")
                p50_ms = None

    book.add(
        "case_source_workload",
        not issues,
        "; ".join(issues) if issues else "source and BS1 workload are frozen",
        phase=phase,
        scope=name,
        evidence={
            "decode_fwd_sha256": decode_sha,
            "source_manifest_sha256": source_manifest,
            "image_ref": image_ref,
            "p50_ms": p50_ms,
            "iters": timing.get("iters") if isinstance(timing, dict) else None,
            "warmup": timing.get("warmup") if isinstance(timing, dict) else None,
        },
    )
    hidden = _hidden_contract(
        batch_dir,
        report,
        golden,
        book,
        phase=phase,
        scope=name,
    )
    return {
        "run": name,
        "path": str(run),
        "source": source,
        "round": round_id,
        "mode": mode,
        "source_workload_passed": not issues,
        "issues": issues,
        "decode_fwd_sha256": decode_sha,
        "source_manifest_sha256": source_manifest,
        "image_ref": image_ref,
        "p50_ms": p50_ms,
        "iters": timing.get("iters") if isinstance(timing, dict) else None,
        "warmup": timing.get("warmup") if isinstance(timing, dict) else None,
        "run_nonce": run_nonce,
        "started_at": started_at_text,
        "finished_at": finished_at_text,
        "hidden": hidden,
        "batch_dir": str(batch_dir),
    }


def _normal_analysis(
    campaign: Path,
    golden: dict[str, Any],
    spec: dict[str, Any],
    book: CheckBook,
) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {source: {} for source in SOURCES}
    for round_id in ROUNDS:
        for source in SOURCES:
            records[source][str(round_id)] = _case_record(
                campaign,
                golden,
                book,
                source=source,
                round_id=round_id,
                mode="normal",
            )

    consistency_issues: list[str] = []
    source_manifests: dict[str, list[str]] = {}
    for source in SOURCES:
        values = {
            item["source_manifest_sha256"]
            for item in records[source].values()
            if _is_sha256(item["source_manifest_sha256"])
        }
        source_manifests[source] = sorted(values)
        if len(values) != 1:
            consistency_issues.append(
                f"{source} source manifest changed or is unavailable"
            )
    image_refs = {
        item["image_ref"]
        for source_records in records.values()
        for item in source_records.values()
        if item["image_ref"]
    }
    golden_image = golden["manifest"].get("image_ref")
    if len(image_refs) != 1:
        consistency_issues.append("normal runs do not use one image")
    elif golden_image and next(iter(image_refs)) != golden_image:
        consistency_issues.append("normal image differs from frozen golden")
    book.add(
        "normal_source_image_consistency",
        not consistency_issues,
        (
            "; ".join(consistency_issues)
            if consistency_issues
            else "normal source manifests and image are stable"
        ),
        phase="normal",
        scope="six-runs",
        evidence={
            "source_manifests": source_manifests,
            "image_refs": sorted(image_refs),
            "golden_image_ref": golden_image,
        },
    )

    counterbalance_issues: list[str] = []
    seen_nonces: set[str] = set()
    observed_order: list[dict[str, Any]] = []
    for round_id in ROUNDS:
        expected_order = EXPECTED_NORMAL_ORDER[round_id]
        round_records = [
            records[source][str(round_id)] for source in expected_order
        ]
        if any(item["run_nonce"] in seen_nonces for item in round_records):
            counterbalance_issues.append(
                f"round {round_id} reuses a run nonce"
            )
        seen_nonces.update(
            item["run_nonce"]
            for item in round_records
            if _is_sha256(item["run_nonce"])
        )
        timestamps = [
            (
                _parse_timestamp(item["started_at"]),
                _parse_timestamp(item["finished_at"]),
            )
            for item in round_records
        ]
        if any(start is None or finish is None for start, finish in timestamps):
            counterbalance_issues.append(
                f"round {round_id} has incomplete timestamps"
            )
        else:
            first_finish = timestamps[0][1]
            second_start = timestamps[1][0]
            if first_finish > second_start:
                counterbalance_issues.append(
                    f"round {round_id} runs overlap or are out of order"
                )
        observed_order.append(
            {
                "round": round_id,
                "expected": list(expected_order),
                "observed": [
                    item["source"] for item in round_records
                ],
                "started_at": [item["started_at"] for item in round_records],
                "finished_at": [item["finished_at"] for item in round_records],
            }
        )
    if len(seen_nonces) != len(ROUNDS) * len(SOURCES):
        counterbalance_issues.append("normal runs do not have six unique nonces")
    spec_payload = spec.get("payload", {})
    spec_sources = spec_payload.get("sources", {})
    for source in SOURCES:
        expected_manifest = (
            spec_sources.get(source, {}).get("source_manifest_sha256")
            if isinstance(spec_sources, dict)
            else None
        )
        if source_manifests.get(source) != [expected_manifest]:
            counterbalance_issues.append(
                f"{source} manifest does not match experiment spec"
            )
    book.add(
        "normal_counterbalance_contract",
        not counterbalance_issues,
        (
            "; ".join(counterbalance_issues)
            if counterbalance_issues
            else "normal runs satisfy frozen AB/BA/AB order and metadata"
        ),
        phase="normal",
        scope="six-runs",
        evidence={"observed_order": observed_order},
    )

    wall: dict[str, Any] = {}
    p50_by_source: dict[str, list[float]] = {}
    for source in SOURCES:
        p50_by_source[source] = [
            item["p50_ms"]
            for item in records[source].values()
            if item["p50_ms"] is not None
        ]
    wall_issues: list[str] = []
    if any(len(values) != len(ROUNDS) for values in p50_by_source.values()):
        wall_issues.append("all six wall p50 values are required")
    else:
        baseline_median = statistics.median(p50_by_source["baseline"])
        candidate_median = statistics.median(p50_by_source["candidate"])
        gain_pct = (baseline_median - candidate_median) / baseline_median * 100.0
        paired = {}
        for round_id in ROUNDS:
            base = records["baseline"][str(round_id)]["p50_ms"]
            candidate = records["candidate"][str(round_id)]["p50_ms"]
            paired[str(round_id)] = {
                "baseline_p50_ms": base,
                "candidate_p50_ms": candidate,
                "gain_pct": round(
                    (base - candidate) / base * 100.0,
                    4,
                ),
            }
        wall = {
            "baseline_round_p50_ms": p50_by_source["baseline"],
            "candidate_round_p50_ms": p50_by_source["candidate"],
            "baseline_median_p50_ms": round(baseline_median, 6),
            "candidate_median_p50_ms": round(candidate_median, 6),
            "median_wall_p50_gain_pct": round(gain_pct, 4),
            "minimum_gain_pct": WALL_GAIN_MIN_PCT,
            "paired_rounds": paired,
        }
        if gain_pct < WALL_GAIN_MIN_PCT:
            wall_issues.append(f"median wall p50 gain {gain_pct:.4f}% is below 2%")
    book.add(
        "normal_wall_p50_gain",
        not wall_issues,
        "; ".join(wall_issues) if wall_issues else "median wall p50 gain passes",
        phase="normal",
        scope="three-round-median",
        evidence=wall,
    )
    return {
        "required_runs": [
            _run_name(source, round_id, "normal")
            for round_id in ROUNDS
            for source in SOURCES
        ],
        "records": records,
        "source_manifests": source_manifests,
        "image_refs": sorted(image_refs),
        "wall": wall,
        "counterbalance": {
            "passed": not counterbalance_issues,
            "observed_order": observed_order,
            "unique_nonces": len(seen_nonces),
        },
    }


def _dfx_report_path(record: dict[str, Any]) -> Path:
    return Path(record["batch_dir"]) / "dfx_analysis" / "moe_dfx_report.json"


def _dfx_header_contract(
    dfx: dict[str, Any] | None,
    source: str,
    book: CheckBook,
    *,
    scope: str,
) -> dict[str, Any]:
    issues: list[str] = []
    if dfx is None:
        issues.append("DFX report is unavailable")
        dfx = {}
    if not str(dfx.get("schema", "")).startswith("step3p5.five-layer-moe-dfx.v6"):
        issues.append("DFX schema is not v6")
    expected_profile = EXPECTED_DFX_PROFILE[source]
    if dfx.get("profile") != expected_profile:
        issues.append(
            f"profile is {dfx.get('profile')!r}, expected "
            f"{expected_profile!r}"
        )
    policy = dfx.get("source_policy")
    if not isinstance(policy, dict):
        issues.append("source_policy is unavailable")
        policy = {}
    expected_policy_id = EXPECTED_DFX_POLICY[source]
    expected_decode_prefix = EXPECTED_DECODE_SHA256[source][:8]
    expected_role = EXPECTED_DFX_ROLE[source]
    if policy.get("policy_id") != expected_policy_id:
        issues.append("source policy ID mismatch")
    if policy.get("decode_sha256_prefix") != expected_decode_prefix:
        issues.append("source policy decode prefix mismatch")
    if policy.get("source_role") != expected_role:
        issues.append("source policy role mismatch")
    if policy.get("enforce_candidate_release_gate") is not True:
        issues.append("shared experiment release enforcement is not enabled")
    if policy.get("experimental") is not (source == "candidate"):
        issues.append("source policy experimental flag mismatch")
    ranks = dfx.get("ranks")
    if not isinstance(ranks, dict) or set(ranks) != set(RANKS):
        issues.append("DFX rank set is not exactly rank0/d0..rank7/d0")
    rank_contract = dfx.get("rank_contract")
    if not (
        isinstance(rank_contract, dict)
        and rank_contract.get("exact") is True
        and rank_contract.get("expected") == rank_contract.get("actual") == list(RANKS)
    ):
        issues.append("rank_contract is not exact")
    structural = dfx.get("structural_contracts")
    if not isinstance(structural, dict) or structural.get("pass") is not True:
        issues.append("structural_contracts.pass is not true")
    slice_contract = dfx.get("slice_contract")
    if not (
        isinstance(slice_contract, dict)
        and slice_contract.get("expected_equals_observed") is True
        and slice_contract.get("errors") == []
    ):
        issues.append("top-level slice contract failed")
    routed_profiles = dfx.get("routed_slice_profiles")
    if not isinstance(routed_profiles, dict) or routed_profiles.get("pass") is not True:
        issues.append("routed slice profile contract failed")
    expert = dfx.get("expert_kernel_release")
    if not isinstance(expert, dict):
        issues.append("expert kernel release contract is unavailable")
        expert = {}
    admission = dfx.get("admission")
    if not isinstance(admission, dict):
        issues.append("admission contract is unavailable")
        admission = {}
    if expert.get("profile") != expected_profile:
        issues.append("expert release profile mismatch")
    if expert.get("release_enforced") is not True:
        issues.append("expert release gate is not enforced")
    for key in (
        "release_gate_pass",
        "diagnostic_pass",
        "coverage_pass",
        "duration_pass",
        "activation_pass",
        "pass",
    ):
        if expert.get(key) is not True:
            issues.append(f"expert release {key} is not true")
    if expert.get("release_gate_status") != "PASS":
        issues.append("expert release gate status is not PASS")
    if expert.get("coverage_errors") != []:
        issues.append("expert coverage errors are not empty")
    if expert.get("duration_errors") != []:
        issues.append("expert duration errors are not empty")
    if expert.get("activation_errors") != []:
        issues.append("expert activation errors are not empty")
    expert_policy = expert.get("source_policy")
    if not isinstance(expert_policy, dict) or expert_policy.get("policy_id") != expected_policy_id:
        issues.append("expert release source policy mismatch")
    if admission.get("profile") != expected_profile:
        issues.append("admission profile mismatch")
    if admission.get("pass") is not True or admission.get("analyzer_gate_pass") is not True:
        issues.append("DFX analyzer gate did not pass")
    if admission.get("expert_release_enforced") is not True:
        issues.append("admission release enforcement is not true")
    if admission.get("blockers") != []:
        issues.append("DFX admission blockers are not empty")
    admission_policy = admission.get("source_policy")
    if not isinstance(admission_policy, dict) or admission_policy.get("policy_id") != expected_policy_id:
        issues.append("admission source policy mismatch")
    limits = expert.get("duration_limits_us")
    if limits != {
        "p50_min": GATE_UP_LIMITS_US["p50_min"],
        "p50_max": GATE_UP_LIMITS_US["p50_max"],
        "p90_max": GATE_UP_LIMITS_US["p90_max"],
        "p99_max": GATE_UP_LIMITS_US["p99_max"],
        "max": GATE_UP_LIMITS_US["max"],
    }:
        issues.append("expert duration limits differ from frozen limits")
    book.add(
        "dfx_profile_and_integrity",
        not issues,
        "; ".join(issues) if issues else "DFX profile and integrity pass",
        phase="dfx",
        scope=scope,
        evidence={
            "profile": dfx.get("profile"),
            "policy_id": policy.get("policy_id"),
        },
    )
    return dfx


def _stage_task(
    stage: Any,
    *,
    expected_blocks: int | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    issues: list[str] = []
    if not isinstance(stage, dict):
        return None, ["stage is unavailable"]
    task_ids = stage.get("task_ids")
    details = stage.get("task_instance_details")
    if stage.get("task_instances") != 1:
        issues.append("task_instances is not 1")
    if not isinstance(task_ids, list) or len(task_ids) != 1:
        issues.append("stage does not contain exactly one task ID")
    if not isinstance(details, list) or len(details) != 1:
        issues.append("stage does not contain exactly one task detail")
        detail = None
    else:
        detail = details[0]
        if not isinstance(detail, dict):
            issues.append("task detail is not an object")
            detail = None
    if detail is not None and task_ids and detail.get("task_id") != task_ids[0]:
        issues.append("task detail ID differs from stage task ID")
    if expected_blocks is not None:
        if stage.get("logical_blocks") != expected_blocks:
            issues.append(f"logical_blocks is not {expected_blocks}")
        if stage.get("blocks_per_task") != [expected_blocks]:
            issues.append(f"blocks_per_task is not [{expected_blocks}]")
        if detail is None or detail.get("block_num") != expected_blocks:
            issues.append(f"task block_num is not {expected_blocks}")
    for item, label in ((stage, "stage"), (detail, "task detail")):
        if item is None:
            continue
        start = item.get("start_tick")
        end = item.get("end_tick")
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            issues.append(f"{label} timing envelope is invalid")
    if detail is not None and isinstance(stage.get("start_tick"), int):
        if detail.get("start_tick") != stage.get("start_tick"):
            issues.append("task detail start differs from stage start")
        if detail.get("end_tick") != stage.get("end_tick"):
            issues.append("task detail end differs from stage end")
    return detail, issues


def _resource_audit(
    stage: dict[str, Any],
    stage_name: str,
    resource_name: str,
) -> dict[str, Any]:
    issues: list[str] = []
    resources = stage.get("resources")
    resource = resources.get(resource_name) if isinstance(resources, dict) else None
    if not isinstance(resource, dict):
        return {
            "issues": [f"{stage_name}.{resource_name} is unavailable"],
            "recomputed_peak": None,
        }
    expected = resource.get("expected_slices")
    observed = resource.get("observed_slices")
    if not isinstance(expected, int) or not isinstance(observed, int):
        issues.append(f"{stage_name}.{resource_name} slice counts are invalid")
        expected = observed = -1
    elif expected != observed:
        issues.append(
            f"{stage_name}.{resource_name} expected={expected}, observed={observed}"
        )
    physical = resource.get("physical_slices")
    if not isinstance(physical, list) or len(physical) != max(observed, 0):
        issues.append(
            f"{stage_name}.{resource_name} physical slice records "
            "do not match observed_slices"
        )
        physical = physical if isinstance(physical, list) else []

    task_ids = stage.get("task_ids")
    expected_task_id = task_ids[0] if isinstance(task_ids, list) and task_ids else None
    intervals: list[tuple[int, int]] = []
    by_core: dict[int, list[tuple[int, int]]] = {}
    events: list[tuple[int, int]] = []
    for index, item in enumerate(physical):
        if not isinstance(item, dict):
            issues.append(f"{stage_name}.{resource_name}[{index}] is not an object")
            continue
        if item.get("task_id") != expected_task_id:
            issues.append(
                f"{stage_name}.{resource_name}[{index}] task ID does not match stage"
            )
        if item.get("resource") != resource_name:
            issues.append(
                f"{stage_name}.{resource_name}[{index}] resource tag mismatch"
            )
        core = item.get("core_id")
        start = item.get("start_tick")
        end = item.get("end_tick")
        duration = item.get("service_span_us")
        if not isinstance(core, int) or not isinstance(start, int) or not isinstance(end, int):
            issues.append(
                f"{stage_name}.{resource_name}[{index}] timing/core fields are invalid"
            )
            continue
        if end <= start:
            issues.append(
                f"{stage_name}.{resource_name}[{index}] has non-positive interval"
            )
            continue
        if not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or float(duration) <= 0:
            issues.append(
                f"{stage_name}.{resource_name}[{index}] service duration is invalid"
            )
        interval = (start, end)
        intervals.append(interval)
        by_core.setdefault(core, []).append(interval)
        events.extend(((start, 1), (end, -1)))

    for core, core_intervals in by_core.items():
        ordered = sorted(core_intervals)
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] < previous[1]:
                issues.append(
                    f"{stage_name}.{resource_name} core {core} has overlapping slices"
                )

    active = 0
    recomputed_peak = 0
    for tick, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        recomputed_peak = max(recomputed_peak, active)
    reported_peak = resource.get("peak_concurrency")
    if reported_peak != recomputed_peak:
        issues.append(
            f"{stage_name}.{resource_name} peak_concurrency={reported_peak!r}, "
            f"recomputed={recomputed_peak}"
        )
    if observed == 0:
        if resource.get("available") is not False:
            issues.append(f"{stage_name}.{resource_name} empty resource is available")
        if reported_peak != 0:
            issues.append(f"{stage_name}.{resource_name} empty resource peak is not 0")
    elif resource.get("available") is not True:
        issues.append(f"{stage_name}.{resource_name} non-empty resource is unavailable")
    return {
        "issues": issues,
        "recomputed_peak": recomputed_peak,
        "observed": observed,
        "interval_count": len(intervals),
    }


def _resource_issues(stage: dict[str, Any], stage_name: str) -> list[str]:
    issues: list[str] = []
    for resource_name in ("aic", "aiv"):
        issues.extend(
            _resource_audit(stage, stage_name, resource_name)["issues"]
        )
    return issues


def _candidate_chain_contract(
    candidate: dict[str, Any],
    book: CheckBook,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    chain_failures: list[str] = []
    shape_failures: list[str] = []
    parallel_failures: list[str] = []
    activation_failures: list[str] = []
    ranks = candidate.get("ranks", {})
    for rank in RANKS:
        for layer in LAYERS:
            scope = f"{rank}/{layer}"
            stages = (
                ranks.get(rank, {}).get("layers", {}).get(layer, {})
                if isinstance(ranks, dict)
                else {}
            )
            details: dict[str, dict[str, Any] | None] = {}
            ids: dict[str, str | None] = {}
            local_shape_issues: list[str] = []
            local_chain_issues: list[str] = []
            local_parallel_issues: list[str] = []
            local_activation_issues: list[str] = []
            if stages.get("shared_mlp") is not None:
                local_shape_issues.append("candidate still exposes shared_mlp")
            for stage_name, blocks in SHARED_BLOCKS.items():
                stage = stages.get(stage_name)
                detail, issues = _stage_task(
                    stage,
                    expected_blocks=blocks,
                )
                details[stage_name] = detail
                ids[stage_name] = detail.get("task_id") if detail is not None else None
                local_shape_issues.extend(f"{stage_name}: {issue}" for issue in issues)
                if isinstance(stage, dict):
                    local_shape_issues.extend(_resource_issues(stage, stage_name))
                    active_resource = {
                        "shared_gate_up": "aic",
                        "shared_gate_up_act": "aiv",
                        "shared_down": "aic",
                    }[stage_name]
                    for resource_name in ("aic", "aiv"):
                        observed = (
                            stage.get("resources", {})
                            .get(resource_name, {})
                            .get("observed_slices")
                        )
                        expected_slices = (
                            blocks if resource_name == active_resource else 0
                        )
                        if observed != expected_slices:
                            local_shape_issues.append(
                                f"{stage_name}.{resource_name} observed_slices="
                                f"{observed!r}, expected {expected_slices}"
                            )
            allreduce = stages.get("shared_all_reduce")
            allreduce_detail, issues = _stage_task(allreduce)
            details["shared_all_reduce"] = allreduce_detail
            ids["shared_all_reduce"] = (
                allreduce_detail.get("task_id")
                if allreduce_detail is not None
                else None
            )
            local_shape_issues.extend(f"shared_all_reduce: {issue}" for issue in issues)
            if isinstance(allreduce, dict):
                local_shape_issues.extend(
                    _resource_issues(allreduce, "shared_all_reduce")
                )

            split = stages.get("shared_split")
            expected_split_ids = [
                ids["shared_gate_up"],
                ids["shared_gate_up_act"],
                ids["shared_down"],
            ]
            if not isinstance(split, dict):
                local_shape_issues.append("shared_split stage is unavailable")
            else:
                if split.get("task_ids") != expected_split_ids:
                    local_shape_issues.append(
                        "shared_split task_ids do not exactly match 5/5/16 stages"
                    )
                if split.get("task_instances") != 3:
                    local_shape_issues.append("shared_split task_instances is not 3")
                if split.get("blocks_per_task") != [
                    SHARED_BLOCKS["shared_gate_up"],
                    SHARED_BLOCKS["shared_gate_up_act"],
                    SHARED_BLOCKS["shared_down"],
                ]:
                    local_shape_issues.append(
                        "shared_split blocks_per_task is not [5, 5, 16]"
                    )
                if split.get("logical_blocks") != sum(SHARED_BLOCKS.values()):
                    local_shape_issues.append(
                        "shared_split logical_blocks is not 26"
                    )
                stage_values = [
                    stages.get(name) for name in SHARED_BLOCKS
                ]
                if all(isinstance(item, dict) for item in stage_values):
                    expected_start = min(
                        int(item["start_tick"]) for item in stage_values
                    )
                    expected_end = max(
                        int(item["end_tick"]) for item in stage_values
                    )
                    if split.get("start_tick") != expected_start:
                        local_shape_issues.append(
                            "shared_split start_tick is not derived from child stages"
                        )
                    if split.get("end_tick") != expected_end:
                        local_shape_issues.append(
                            "shared_split end_tick is not derived from child stages"
                        )

            chain = (
                "shared_gate_up",
                "shared_gate_up_act",
                "shared_down",
                "shared_all_reduce",
            )
            for previous, current in zip(chain, chain[1:]):
                previous_id = ids[previous]
                current_detail = details[current]
                predecessors = (
                    current_detail.get("timing_evidence", {})
                    .get("queue_delay", {})
                    .get("predecessor_task_ids", [])
                    if current_detail is not None
                    else []
                )
                if previous_id is None or previous_id not in predecessors:
                    local_chain_issues.append(
                        f"missing direct dependency {previous}->{current}"
                    )
                previous_stage = stages.get(previous)
                current_stage = stages.get(current)
                if isinstance(previous_stage, dict) and isinstance(current_stage, dict):
                    if previous_stage.get("end_tick", 0) > current_stage.get(
                        "start_tick", 0
                    ):
                        local_chain_issues.append(
                            f"execution envelopes overlap: {previous}->{current}"
                        )
            orders = [
                details[name].get("order") if details[name] is not None else None
                for name in chain
            ]
            if not (
                all(isinstance(value, int) for value in orders)
                and orders == sorted(orders)
                and len(set(orders)) == len(orders)
            ):
                local_chain_issues.append(
                    f"task orders are not strictly increasing: {orders}"
                )

            active_resources = {
                "shared_gate_up": "aic",
                "shared_gate_up_act": "aiv",
                "shared_down": "aic",
            }
            for stage_name, resource_name in active_resources.items():
                stage = stages.get(stage_name)
                audit = (
                    _resource_audit(stage, stage_name, resource_name)
                    if isinstance(stage, dict)
                    else {"recomputed_peak": None}
                )
                peak = audit.get("recomputed_peak")
                if not isinstance(peak, int) or peak <= 1:
                    local_parallel_issues.append(
                        f"{stage_name}.{resource_name} recomputed_peak={peak!r}"
                    )
            activation_resources = stages.get("shared_gate_up_act", {}).get(
                "resources", {}
            )
            aic_observed = (
                activation_resources.get("aic", {}).get("observed_slices")
                if isinstance(activation_resources, dict)
                else None
            )
            aiv_observed = (
                activation_resources.get("aiv", {}).get("observed_slices")
                if isinstance(activation_resources, dict)
                else None
            )
            if (
                aic_observed != 0
                or not isinstance(aiv_observed, int)
                or aiv_observed <= 0
            ):
                local_activation_issues.append(
                    f"activation observed AIC/AIV={aic_observed}/{aiv_observed}"
                )

            chain_failures.extend(f"{scope}: {issue}" for issue in local_chain_issues)
            shape_failures.extend(f"{scope}: {issue}" for issue in local_shape_issues)
            parallel_failures.extend(
                f"{scope}: {issue}" for issue in local_parallel_issues
            )
            activation_failures.extend(
                f"{scope}: {issue}" for issue in local_activation_issues
            )
            records.append(
                {
                    "rank": rank,
                    "layer": layer,
                    "task_ids": ids,
                    "chain_passed": not local_chain_issues,
                    "shape_and_slice_passed": not local_shape_issues,
                    "parallelism_passed": not local_parallel_issues,
                    "activation_aiv_only": not local_activation_issues,
                    "issues": {
                        "chain": local_chain_issues,
                        "shape_and_slice": local_shape_issues,
                        "parallelism": local_parallel_issues,
                        "activation": local_activation_issues,
                    },
                }
            )

    book.add(
        "dfx_shared_task_chain",
        not chain_failures,
        (
            "; ".join(chain_failures)
            if chain_failures
            else "all 16 rank/layer chains are mm->act->down->allreduce"
        ),
        phase="dfx",
        scope="candidate",
    )
    book.add(
        "dfx_shared_block_and_slice_contract",
        not shape_failures,
        (
            "; ".join(shape_failures)
            if shape_failures
            else "5/5/16 block and expected=observed contracts pass"
        ),
        phase="dfx",
        scope="candidate",
    )
    book.add(
        "dfx_shared_stage_parallelism",
        not parallel_failures,
        (
            "; ".join(parallel_failures)
            if parallel_failures
            else "active shared stages have peak concurrency greater than 1"
        ),
        phase="dfx",
        scope="candidate",
    )
    book.add(
        "dfx_shared_activation_aiv_only",
        not activation_failures,
        (
            "; ".join(activation_failures)
            if activation_failures
            else "shared activation is AIV-only on all ranks and layers"
        ),
        phase="dfx",
        scope="candidate",
    )
    return {
        "expected_rank_layer_count": TP * len(LAYERS),
        "records": records,
        "chain_passed": not chain_failures,
        "shape_and_slice_passed": not shape_failures,
        "parallelism_passed": not parallel_failures,
        "activation_aiv_only": not activation_failures,
    }


def _stage_scale(stage: dict[str, Any]) -> float | None:
    try:
        tick_span = int(stage["end_tick"]) - int(stage["start_tick"])
        span_us = float(stage["stage_span_us"])
    except (KeyError, TypeError, ValueError):
        return None
    if tick_span <= 0 or span_us <= 0:
        return None
    return span_us / tick_span


def _shared_to_allreduce_span(
    stages: dict[str, Any],
    shared_stage: str,
) -> float | None:
    shared = stages.get(shared_stage)
    allreduce = stages.get("shared_all_reduce")
    if not isinstance(shared, dict) or not isinstance(allreduce, dict):
        return None
    scale = _stage_scale(shared)
    if scale is None:
        return None
    try:
        ticks = int(allreduce["end_tick"]) - int(shared["start_tick"])
    except (KeyError, TypeError, ValueError):
        return None
    return ticks * scale if ticks > 0 else None


def _routed_envelope(stages: dict[str, Any]) -> float | None:
    present = [
        stages[name]
        for name in ROUTED_ENVELOPE_STAGES
        if isinstance(stages.get(name), dict)
    ]
    scatter = stages.get("combine_scatter")
    if not present or not isinstance(scatter, dict):
        return None
    anchor = next((stage for stage in present if _stage_scale(stage)), None)
    if anchor is None:
        return None
    scale = _stage_scale(anchor)
    try:
        start = min(int(stage["start_tick"]) for stage in present)
        end = int(scatter["end_tick"])
    except (KeyError, TypeError, ValueError):
        return None
    return (end - start) * scale if end > start else None


def _regression_result(
    baseline: float | None,
    candidate: float | None,
) -> dict[str, Any]:
    if baseline is None or candidate is None or baseline <= 0:
        return {
            "passed": False,
            "baseline_us": baseline,
            "candidate_us": candidate,
            "reason": "metric is unavailable",
        }
    allowance = max(
        baseline * REGRESSION_LIMIT_PCT / 100.0,
        REGRESSION_LIMIT_US,
    )
    delta = candidate - baseline
    return {
        "passed": delta <= allowance,
        "baseline_us": round(baseline, 4),
        "candidate_us": round(candidate, 4),
        "delta_us": round(delta, 4),
        "allowance_us": round(allowance, 4),
        "allowance_rule": "max(5% of baseline, 10us)",
    }


def _comparative_dfx_metrics(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    book: CheckBook,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer in LAYERS:
        baseline_local: list[float] = []
        candidate_local: list[float] = []
        baseline_ready: list[float] = []
        candidate_ready: list[float] = []
        baseline_routed: list[float] = []
        candidate_routed: list[float] = []
        baseline_wait: list[float] = []
        candidate_wait: list[float] = []
        for rank in RANKS:
            base_stages = (
                baseline.get("ranks", {}).get(rank, {}).get("layers", {}).get(layer, {})
            )
            cand_stages = (
                candidate.get("ranks", {})
                .get(rank, {})
                .get("layers", {})
                .get(layer, {})
            )
            base_shared = base_stages.get("shared_mlp")
            cand_shared = cand_stages.get("shared_split")
            if isinstance(base_shared, dict):
                try:
                    baseline_local.append(float(base_shared["stage_span_us"]))
                except (KeyError, TypeError, ValueError):
                    pass
            if isinstance(cand_shared, dict):
                try:
                    candidate_local.append(float(cand_shared["stage_span_us"]))
                except (KeyError, TypeError, ValueError):
                    pass
            base_ready = _shared_to_allreduce_span(
                base_stages,
                "shared_mlp",
            )
            cand_ready = _shared_to_allreduce_span(
                cand_stages,
                "shared_split",
            )
            if base_ready is not None:
                baseline_ready.append(base_ready)
            if cand_ready is not None:
                candidate_ready.append(cand_ready)
            base_routed = _routed_envelope(base_stages)
            cand_routed = _routed_envelope(cand_stages)
            if base_routed is not None:
                baseline_routed.append(base_routed)
            if cand_routed is not None:
                candidate_routed.append(cand_routed)
            for stages, values in (
                (base_stages, baseline_wait),
                (cand_stages, candidate_wait),
            ):
                wait = stages.get("combine_wait")
                if isinstance(wait, dict):
                    try:
                        values.append(float(wait["stage_span_us"]))
                    except (KeyError, TypeError, ValueError):
                        pass

        local: dict[str, Any]
        if len(baseline_local) == len(candidate_local) == TP:
            base_median = statistics.median(baseline_local)
            cand_median = statistics.median(candidate_local)
            gain = (base_median - cand_median) / base_median * 100.0
            local = {
                "passed": gain >= SHARED_SPAN_GAIN_MIN_PCT,
                "baseline_median_us": round(base_median, 4),
                "candidate_median_us": round(cand_median, 4),
                "reduction_pct": round(gain, 4),
                "minimum_reduction_pct": SHARED_SPAN_GAIN_MIN_PCT,
                "baseline_by_rank_us": baseline_local,
                "candidate_by_rank_us": candidate_local,
            }
        else:
            local = {
                "passed": False,
                "reason": "shared local spans are incomplete",
                "baseline_count": len(baseline_local),
                "candidate_count": len(candidate_local),
            }
        book.add(
            "dfx_shared_local_span_reduction",
            local["passed"],
            (
                "shared local median span reduction passes"
                if local["passed"]
                else "shared local median span reduction is below 20% or unavailable"
            ),
            phase="dfx",
            scope=layer,
            evidence=local,
        )

        if len(baseline_ready) == len(candidate_ready) == TP:
            base_ready_median = statistics.median(baseline_ready)
            cand_ready_median = statistics.median(candidate_ready)
            ready = {
                "passed": cand_ready_median < base_ready_median,
                "baseline_median_us": round(base_ready_median, 4),
                "candidate_median_us": round(cand_ready_median, 4),
                "advance_us": round(
                    base_ready_median - cand_ready_median,
                    4,
                ),
                "baseline_by_rank_us": [round(value, 4) for value in baseline_ready],
                "candidate_by_rank_us": [round(value, 4) for value in candidate_ready],
            }
        else:
            ready = {
                "passed": False,
                "reason": "shared-to-allreduce spans are incomplete",
                "baseline_count": len(baseline_ready),
                "candidate_count": len(candidate_ready),
            }
        book.add(
            "dfx_shared_to_allreduce_advance",
            ready["passed"],
            (
                "shared start to allreduce end is earlier"
                if ready["passed"]
                else "shared start to allreduce end did not advance"
            ),
            phase="dfx",
            scope=layer,
            evidence=ready,
        )

        routed = _regression_result(
            max(baseline_routed) if len(baseline_routed) == TP else None,
            max(candidate_routed) if len(candidate_routed) == TP else None,
        )
        book.add(
            "dfx_routed_producer_envelope_regression",
            routed["passed"],
            (
                "routed producer envelope max regression is within tolerance"
                if routed["passed"]
                else "routed producer envelope max regression exceeds tolerance"
            ),
            phase="dfx",
            scope=layer,
            evidence=routed,
        )
        combine_wait = _regression_result(
            max(baseline_wait) if len(baseline_wait) == TP else None,
            max(candidate_wait) if len(candidate_wait) == TP else None,
        )
        book.add(
            "dfx_combine_wait_regression",
            combine_wait["passed"],
            (
                "combine_wait max regression is within tolerance"
                if combine_wait["passed"]
                else "combine_wait max regression exceeds tolerance"
            ),
            phase="dfx",
            scope=layer,
            evidence=combine_wait,
        )
        result[layer] = {
            "shared_local_span": local,
            "shared_start_to_allreduce_end": ready,
            "routed_producer_envelope_max": routed,
            "combine_wait_max": combine_wait,
            "routed_envelope_definition": (
                "Earliest local routed-compute stage start through local "
                "combine_scatter end; compared as a per-rank envelope, then max."
            ),
        }
    return result


def _physical_durations(
    stage: dict[str, Any],
    resource_name: str,
) -> list[float]:
    resource = stage.get("resources", {}).get(resource_name, {})
    physical = resource.get("physical_slices")
    if not isinstance(physical, list):
        return []
    values: list[float] = []
    for item in physical:
        if not isinstance(item, dict):
            return []
        try:
            value = float(item["service_span_us"])
        except (KeyError, TypeError, ValueError):
            return []
        if not math.isfinite(value) or value <= 0:
            return []
        values.append(value)
    return values


def _gate_up_pass(distribution: dict[str, Any]) -> bool:
    return bool(
        distribution["count"]
        and GATE_UP_LIMITS_US["p50_min"]
        <= distribution["p50_us"]
        <= GATE_UP_LIMITS_US["p50_max"]
        and distribution["p90_us"] <= GATE_UP_LIMITS_US["p90_max"]
        and distribution["p99_us"] <= GATE_UP_LIMITS_US["p99_max"]
        and distribution["max_us"] <= GATE_UP_LIMITS_US["max"]
    )


def _shared_kernel_grain(
    candidate: dict[str, Any],
    book: CheckBook,
) -> dict[str, Any]:
    gate_values: list[float] = []
    down_values: list[float] = []
    gate_scopes: dict[str, Any] = {}
    down_scopes: dict[str, Any] = {}
    ranks = candidate.get("ranks", {})
    for rank in RANKS:
        for layer in LAYERS:
            scope = f"{rank}/{layer}"
            stages = (
                ranks.get(rank, {}).get("layers", {}).get(layer, {})
                if isinstance(ranks, dict)
                else {}
            )
            gate = _physical_durations(
                stages.get("shared_gate_up", {}),
                "aic",
            )
            down = _physical_durations(
                stages.get("shared_down", {}),
                "aic",
            )
            gate_dist = _distribution(gate)
            down_dist = _distribution(down)
            gate_dist["passed"] = _gate_up_pass(gate_dist)
            down_dist["passed"] = bool(
                down_dist["count"] and down_dist["p50_us"] >= 10.0
            )
            gate_scopes[scope] = gate_dist
            down_scopes[scope] = down_dist
            gate_values.extend(gate)
            down_values.extend(down)

    gate_global = _distribution(gate_values)
    gate_global["passed"] = _gate_up_pass(gate_global)
    gate_all_scopes = all(item["passed"] for item in gate_scopes.values())
    gate_passed = gate_global["passed"] and gate_all_scopes
    book.add(
        "dfx_shared_gate_up_aic_grain",
        gate_passed,
        (
            "shared gate/up AIC duration gates pass"
            if gate_passed
            else "shared gate/up AIC duration gates failed"
        ),
        phase="dfx",
        scope="candidate/all-ranks-layers",
        evidence={
            "limits_us": GATE_UP_LIMITS_US,
            "global": gate_global,
            "per_rank_layer": gate_scopes,
        },
    )

    down_global = _distribution(down_values)
    too_fine_scopes = [
        scope
        for scope, item in down_scopes.items()
        if item["count"] and item["p50_us"] < 10.0
    ]
    requires_comparison = bool(
        not down_global["count"] or down_global["p50_us"] < 10.0 or too_fine_scopes
    )
    down_passed = bool(down_global["count"]) and not requires_comparison
    book.add(
        "dfx_shared_down_grain",
        down_passed,
        (
            "shared down AIC p50 is at least 10us"
            if down_passed
            else (
                "shared down is too fine or unavailable; an 8-block and "
                "4-block coarse-grain comparison is required"
            )
        ),
        phase="dfx",
        scope="candidate/all-ranks-layers",
        evidence={
            "global": down_global,
            "per_rank_layer": down_scopes,
            "too_fine_scopes": too_fine_scopes,
            "requires_coarser_down_block_comparison": requires_comparison,
            "required_comparisons": (
                ["8-block", "4-block"] if requires_comparison else []
            ),
        },
    )
    return {
        "shared_gate_up_aic": {
            "passed": gate_passed,
            "limits_us": GATE_UP_LIMITS_US,
            "global": gate_global,
            "per_rank_layer": gate_scopes,
        },
        "shared_down_aic": {
            "passed": down_passed,
            "global": down_global,
            "per_rank_layer": down_scopes,
            "too_fine_scopes": too_fine_scopes,
            "requires_coarser_down_block_comparison": requires_comparison,
            "required_comparisons": (
                ["8-block", "4-block"] if requires_comparison else []
            ),
        },
    }


def _dfx_analysis(
    campaign: Path,
    golden: dict[str, Any],
    normal: dict[str, Any],
    book: CheckBook,
) -> dict[str, Any]:
    case_records: dict[str, Any] = {}
    dfx_reports: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        record = _case_record(
            campaign,
            golden,
            book,
            source=source,
            round_id=1,
            mode="dfx",
        )
        case_records[source] = record
        raw = _read_json(
            _dfx_report_path(record),
            book,
            phase="dfx",
            scope=record["run"],
        )
        dfx_reports[source] = _dfx_header_contract(
            raw,
            source,
            book,
            scope=record["run"],
        )

    consistency_issues: list[str] = []
    normal_images = set(normal.get("image_refs", []))
    for source in SOURCES:
        normal_manifests = set(normal.get("source_manifests", {}).get(source, []))
        if case_records[source]["source_manifest_sha256"] not in normal_manifests:
            consistency_issues.append(
                f"{source} DFX source manifest differs from normal runs"
            )
        if case_records[source]["image_ref"] not in normal_images:
            consistency_issues.append(f"{source} DFX image differs from normal runs")
    book.add(
        "dfx_source_image_consistency",
        not consistency_issues,
        (
            "; ".join(consistency_issues)
            if consistency_issues
            else "DFX and normal source/image provenance match"
        ),
        phase="dfx",
        scope="baseline-and-candidate",
    )

    chain = _candidate_chain_contract(dfx_reports["candidate"], book)
    comparison = _comparative_dfx_metrics(
        dfx_reports["baseline"],
        dfx_reports["candidate"],
        book,
    )
    grain = _shared_kernel_grain(dfx_reports["candidate"], book)
    return {
        "status": "ANALYZED",
        "case_records": case_records,
        "profiles": {
            source: {
                "expected": EXPECTED_DFX_PROFILE[source],
                "actual": dfx_reports[source].get("profile"),
                "policy_id": (
                    dfx_reports[source].get("source_policy", {}).get("policy_id")
                ),
                "report": str(_dfx_report_path(case_records[source])),
            }
            for source in SOURCES
        },
        "shared_chain": chain,
        "comparative_metrics": comparison,
        "kernel_grain": grain,
    }


def _route_sidecar_gate(
    campaign: Path,
    normal: dict[str, Any],
    book: CheckBook,
) -> dict[str, Any]:
    path = (
        campaign
        / "route-sidecar"
        / "candidate-shared-route-bs1-64k"
        / "runtime"
        / "route_artifact_validation.json"
    )
    value = _read_json(
        path,
        book,
        phase="route",
        scope="candidate-shared-route-bs1-64k",
    )
    issues: list[str] = []
    if value is None:
        issues.append("route artifact validation is unavailable")
        value = {}
    sidecar_path = path.parent / "recv_meta_sidecar.pt"
    expected = {
        "schema": "step3p5.five-layer-moe-route-validation.v1",
        "passed": True,
        "active_batch": BATCH,
        "context_len_per_sequence": CONTEXT_LEN,
        "profile": "shared-split",
        "source_role": "candidate",
        "decode_fwd_sha256": EXPECTED_DECODE_SHA256["candidate"],
        "hidden_bit_exact": True,
        "padding_zero": True,
        "local_count_exact": True,
        "window_independence_validated": True,
        "global_routes_per_layer": [64, 64],
        "expected_global_routes_per_layer": 64,
    }
    checks: dict[str, bool] = {}
    for key, expected_value in expected.items():
        checks[key] = value.get(key) == expected_value
        if not checks[key]:
            issues.append(f"{key}={value.get(key)!r}, expected {expected_value!r}")
    checks["sidecar_present"] = sidecar_path.is_file()
    checks["sidecar_hash"] = (
        checks["sidecar_present"]
        and value.get("artifacts", {}).get("recv_meta_sidecar.pt")
        == _sha256(sidecar_path)
    )
    if not checks["sidecar_present"]:
        issues.append("recv_meta_sidecar.pt is unavailable")
    elif not checks["sidecar_hash"]:
        issues.append("route validation sidecar hash does not match")
    candidate_manifests = set(normal.get("source_manifests", {}).get("candidate", []))
    checks["source_manifest_matches_normal"] = (
        value.get("source_manifest_sha256") in candidate_manifests
    )
    if not checks["source_manifest_matches_normal"]:
        issues.append("route source manifest differs from candidate normal runs")
    normal_images = set(normal.get("image_refs", []))
    checks["image_matches_normal"] = value.get("image_ref") in normal_images
    if not checks["image_matches_normal"]:
        issues.append("route image differs from normal runs")
    book.add(
        "route_sidecar_validator_gate",
        not issues,
        (
            "; ".join(issues)
            if issues
            else (
                "route validator passed hidden, padding, local-count, "
                "64-route, and independent-window gates"
            )
        ),
        phase="route",
        scope="candidate-shared-route-bs1-64k",
        evidence=checks,
    )
    return {
        "status": "ANALYZED",
        "path": str(path),
        "passed": not issues,
        "checks": checks,
        "payload": value,
        "issues": issues,
    }


def _markdown(report: dict[str, Any]) -> str:
    passed = report["passed"]
    lines = [
        "# BS1 shared-expert experiment acceptance",
        "",
        f"- Mode: `{report['mode']}`",
        f"- Result: `{'PASS' if passed else 'FAIL'}`",
        "- Workload: `BS1`, one sequence with `context_len=65536`",
        "",
        "## Normal runs",
        "",
        "| Round | Baseline p50 ms | Candidate p50 ms | Gain |",
        "|---:|---:|---:|---:|",
    ]
    wall = report.get("normal", {}).get("wall", {})
    paired = wall.get("paired_rounds", {})
    for round_id in ROUNDS:
        item = paired.get(str(round_id), {})
        if item:
            lines.append(
                f"| {round_id} | {item['baseline_p50_ms']:.4f} | "
                f"{item['candidate_p50_ms']:.4f} | "
                f"{item['gain_pct']:+.2f}% |"
            )
        else:
            lines.append(f"| {round_id} | - | - | - |")
    lines.extend(
        [
            "",
            "Three-round median wall p50 gain: "
            + (
                f"`{wall['median_wall_p50_gain_pct']:+.2f}%` "
                f"(required `>= {WALL_GAIN_MIN_PCT:.0f}%`)"
                if "median_wall_p50_gain_pct" in wall
                else "`unavailable`"
            ),
            "",
        ]
    )

    dfx = report.get("dfx", {})
    if dfx.get("status") == "SKIPPED_NORMAL_ONLY":
        lines.extend(
            [
                "## DFX",
                "",
                "Skipped by `--normal-only`.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## DFX",
                "",
                "| Source | Required profile | Actual profile |",
                "|:---|:---|:---|",
            ]
        )
        for source in SOURCES:
            profile = dfx.get("profiles", {}).get(source, {})
            lines.append(
                f"| {source} | {profile.get('expected', '-')} | "
                f"{profile.get('actual', '-')} |"
            )
        lines.extend(
            [
                "",
                "| Layer | Shared local reduction | Shared ready advance us | "
                "Routed max delta/allowance us | Wait max delta/allowance us |",
                "|:---|---:|---:|---:|---:|",
            ]
        )
        metrics = dfx.get("comparative_metrics", {})
        for layer in LAYERS:
            item = metrics.get(layer, {})
            local = item.get("shared_local_span", {})
            ready = item.get("shared_start_to_allreduce_end", {})
            routed = item.get("routed_producer_envelope_max", {})
            wait = item.get("combine_wait_max", {})
            lines.append(
                f"| {layer} | {local.get('reduction_pct', 0):.2f}% | "
                f"{ready.get('advance_us', 0):.2f} | "
                f"{routed.get('delta_us', 0):.2f}/"
                f"{routed.get('allowance_us', 0):.2f} | "
                f"{wait.get('delta_us', 0):.2f}/"
                f"{wait.get('allowance_us', 0):.2f} |"
            )
        grain = dfx.get("kernel_grain", {})
        gate = grain.get("shared_gate_up_aic", {}).get("global", {})
        down = grain.get("shared_down_aic", {})
        down_global = down.get("global", {})
        lines.extend(
            [
                "",
                "Shared gate/up AIC global distribution: "
                f"p50=`{gate.get('p50_us')}`, p90=`{gate.get('p90_us')}`, "
                f"p99=`{gate.get('p99_us')}`, max=`{gate.get('max_us')}` us.",
                "",
                f"Shared down AIC global p50: `{down_global.get('p50_us')}` us.",
                "",
            ]
        )
        if down.get("requires_coarser_down_block_comparison"):
            lines.extend(
                [
                    "**Required follow-up:** shared down is too fine; run "
                    "the 8-block and 4-block coarse-grain comparisons.",
                    "",
                ]
            )

    route = report.get("route_sidecar", {})
    lines.extend(
        [
            "## Route sidecar",
            "",
            (
                "Skipped by `--normal-only`."
                if route.get("status") == "SKIPPED_NORMAL_ONLY"
                else (
                    "Validator gate: " + ("`PASS`" if route.get("passed") else "`FAIL`")
                )
            ),
            "",
            "## Blockers",
            "",
        ]
    )
    blockers = report.get("blockers", [])
    if blockers:
        lines.extend(
            f"- `{item['code']}` ({item['phase']}/{item['scope']}): {item['message']}"
            for item in blockers
        )
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def _write_reports(
    campaign: Path,
    report: dict[str, Any],
    *,
    normal_only: bool,
) -> tuple[Path, Path]:
    campaign.mkdir(parents=True, exist_ok=True)
    prefix = (
        "shared_experiment_normal_only"
        if normal_only
        else "shared_experiment_acceptance"
    )
    json_path = campaign / f"{prefix}.json"
    markdown_path = campaign / f"{prefix}.md"
    report["report_paths"] = {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    campaign = Path(args.campaign).resolve()
    golden_root = Path(args.golden).resolve()
    book = CheckBook()
    report: dict[str, Any] = {
        "schema": "step3p5.bs1-shared-expert-acceptance.v1",
        "mode": "normal-only" if args.normal_only else "full",
        "campaign": str(campaign),
        "golden": str(golden_root),
        "workload": {
            "active_batch": BATCH,
            "context_len_per_sequence": CONTEXT_LEN,
            "active_total_context_tokens": CONTEXT_LEN,
        },
        "expected_decode_sha256": EXPECTED_DECODE_SHA256,
        "thresholds": {
            "median_wall_p50_gain_min_pct": WALL_GAIN_MIN_PCT,
            "shared_local_span_reduction_min_pct": (SHARED_SPAN_GAIN_MIN_PCT),
            "routed_and_wait_max_regression": "max(5%, 10us)",
            "shared_gate_up_aic_us": GATE_UP_LIMITS_US,
            "shared_down_aic_p50_min_us": 10.0,
        },
    }
    try:
        golden = _load_golden(golden_root, book)
        report["golden_contract"] = {
            key: value for key, value in golden.items() if key != "tensors"
        }
        spec = _load_experiment_spec(campaign, book)
        report["experiment_spec"] = {
            key: value for key, value in spec.items() if key != "payload"
        }
        normal = _normal_analysis(campaign, golden, spec, book)
        report["normal"] = normal
        if args.normal_only:
            report["dfx"] = {"status": "SKIPPED_NORMAL_ONLY"}
            report["route_sidecar"] = {"status": "SKIPPED_NORMAL_ONLY"}
            book.add(
                "dfx_full_acceptance",
                None,
                "DFX is intentionally skipped in normal-only mode",
                phase="dfx",
                scope="normal-only",
            )
            book.add(
                "route_sidecar_validator_gate",
                None,
                "route sidecar is intentionally skipped in normal-only mode",
                phase="route",
                scope="normal-only",
            )
        else:
            report["dfx"] = _dfx_analysis(
                campaign,
                golden,
                normal,
                book,
            )
            report["route_sidecar"] = _route_sidecar_gate(
                campaign,
                normal,
                book,
            )
    except Exception as exc:
        book.add(
            "internal_analyzer_error",
            False,
            f"{type(exc).__name__}: {exc}",
            phase="internal",
            scope="analyzer",
        )
        report["internal_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    report["checks"] = book.checks
    report["passed"] = book.passed
    report["blockers"] = book.blockers
    json_path, markdown_path = _write_reports(
        campaign,
        report,
        normal_only=args.normal_only,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "mode": report["mode"],
                "json": str(json_path),
                "markdown": str(markdown_path),
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
