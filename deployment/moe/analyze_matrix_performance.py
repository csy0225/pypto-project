#!/usr/bin/env python3
"""Summarize the fixed five-layer 64K-per-sequence normal campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

from validate_five_layer_case import (
    _validate_map as validate_kv_map,
    _validate_runtime_markers as validate_runtime_markers,
)


BATCHES = (1, 2, 4, 7, 8, 16)
SOURCES = ("baseline", "candidate")
CONTEXT_LEN = 65536
WARMUP = 5
ITERS = 30
IMAGE_PYPTO_COMMIT = "8e92b46808f9f7c09b6431ad4691503f09c12ee5"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--rounds",
        default="1,2,3",
        help="comma-separated fresh-process rounds to compare",
    )
    return parser.parse_args()


def _run_name(source: str, round_id: int, batch: int) -> str:
    return f"{source}-r{round_id}-normal-bs{batch}-64k"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise AssertionError(f"missing text artifact: {path}")
    return path.read_text(encoding="utf-8").strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _round_ids(text: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid rounds: {text!r}")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate rounds: {text!r}")
    return values


def _stable_write(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"refusing to replace different report: {path}")
    path.write_text(content, encoding="utf-8")


def validate_normal_run(
    campaign: Path,
    source: str,
    round_id: int,
    batch: int,
) -> dict[str, Any]:
    run = _run_name(source, round_id, batch)
    run_root = campaign / "runs" / run
    validation = _load_json(run_root / "artifact_validation.json")
    runtime = run_root / "runtime"
    batch_dir = runtime / f"bs{batch}"
    report_path = batch_dir / "report.json"
    matrix_path = runtime / "matrix_report.json"
    report = _load_json(report_path)
    matrix = _load_json(matrix_path)
    if validation.get("passed") is not True:
        raise AssertionError(f"{run}: artifact validation did not pass")
    if _read_text(run_root / "container.rc") != "0":
        raise AssertionError(f"{run}: container did not exit cleanly")
    image_ref = _read_text(run_root / "image_ref.txt")
    nonce = _read_text(run_root / "run_nonce.txt")
    if re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        raise AssertionError(f"{run}: invalid run nonce")
    expected_identity = {
        "source_kind": source,
        "round": round_id,
        "mode": "normal",
        "image_ref": image_ref,
        "image_pypto_commit": IMAGE_PYPTO_COMMIT,
    }
    for artifact_name, artifact in (
        ("report", report),
        ("matrix", matrix),
    ):
        for key, expected in expected_identity.items():
            if artifact.get(key) != expected:
                raise AssertionError(
                    f"{run}: {artifact_name}.{key}="
                    f"{artifact.get(key)!r}, expected={expected!r}"
                )
    workload = report["workload"]
    if workload.get("active_batch") != batch:
        raise AssertionError(f"{run}: active batch mismatch")
    if workload.get("context_len_per_sequence") != CONTEXT_LEN:
        raise AssertionError(f"{run}: context contract mismatch")
    if workload.get("active_total_context_tokens") != batch * CONTEXT_LEN:
        raise AssertionError(f"{run}: total context contract mismatch")
    if workload.get("blocks_per_sequence") != 512:
        raise AssertionError(f"{run}: block contract mismatch")
    timing = report["timing"]
    if timing.get("warmup") != WARMUP or timing.get("iters") != ITERS:
        raise AssertionError(f"{run}: timing protocol mismatch: {timing}")
    for key in ("p50_ms", "mean_ms", "p99_ms", "min_ms", "max_ms"):
        value = timing.get(key)
        if not isinstance(value, (int, float)) or float(value) <= 0:
            raise AssertionError(f"{run}: invalid timing value {key}={value!r}")

    expected_validation = {
        "schema": "step3p5.five-layer-moe-case-validation.v1",
        "passed": True,
        "run": run,
        "source_kind": source,
        "round": round_id,
        "mode": "normal",
        "active_batch": batch,
        "context_len_per_sequence": CONTEXT_LEN,
        "active_total_context_tokens": batch * CONTEXT_LEN,
        "image_ref": image_ref,
        "run_nonce": nonce,
    }
    for key, expected in expected_validation.items():
        if validation.get(key) != expected:
            raise AssertionError(
                f"{run}: artifact validation {key} mismatch"
            )
    if validation.get("workload") != {
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
    }:
        raise AssertionError(f"{run}: validated workload differs from report")

    hidden_sha256 = {
        f"{name}.pt": _sha256(batch_dir / f"{name}.pt")
        for name in ("hidden_l3", "hidden_l4")
    }
    if validation.get("hidden_sha256") != hidden_sha256:
        raise AssertionError(f"{run}: validated hidden hashes differ")
    for name, digest in hidden_sha256.items():
        if report.get("files", {}).get(name) != digest:
            raise AssertionError(f"{run}: report hidden hash differs: {name}")

    kv_hashes = {}
    runtime_hashes = {}
    for rank in range(8):
        kv_hashes[str(rank)] = validate_kv_map(
            runtime / f"pypto_kvpool_map.json.rank{rank}",
            rank=rank,
            batch=batch,
            nonce=nonce,
        )
        runtime_hashes.update(
            validate_runtime_markers(
                runtime,
                rank=rank,
                batch=batch,
                nonce=nonce,
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
        raise AssertionError(f"{run}: validated KV map hashes differ")
    if validation.get("runtime_marker_sha256") != marker_hashes:
        raise AssertionError(f"{run}: validated runtime marker hashes differ")
    if validation.get("kv_key_sha256_by_rank") != key_hashes:
        raise AssertionError(f"{run}: validated KV key hashes differ")

    expected_matrix = {
        "schema": "step3p5.five-layer-moe-64k-matrix.v1",
        **expected_identity,
        "batches": [batch],
        "context_len_per_sequence": CONTEXT_LEN,
        "blocks_per_sequence": 512,
        "reports": {str(batch): f"bs{batch}/report.json"},
        "report": report,
    }
    if matrix != expected_matrix:
        raise AssertionError(f"{run}: matrix report identity mismatch")

    evidence_sha256 = {
        "container.rc": _sha256(run_root / "container.rc"),
        "image_ref.txt": _sha256(run_root / "image_ref.txt"),
        "run_nonce.txt": _sha256(run_root / "run_nonce.txt"),
        "artifact_validation.json": _sha256(
            run_root / "artifact_validation.json"
        ),
        "runtime/matrix_report.json": _sha256(matrix_path),
        f"runtime/bs{batch}/report.json": _sha256(report_path),
        **{
            f"runtime/bs{batch}/{name}": digest
            for name, digest in hidden_sha256.items()
        },
        **{
            f"runtime/pypto_kvpool_map.json.rank{rank}": digest
            for rank, digest in kv_hashes.items()
        },
        **{
            f"runtime/{relative}": digest
            for relative, digest in runtime_hashes.items()
        },
    }
    return {
        "run": run,
        "source_kind": source,
        "round": round_id,
        "active_batch": batch,
        "image_ref": image_ref,
        "image_pypto_commit": IMAGE_PYPTO_COMMIT,
        "run_nonce": nonce,
        "source": report["source"],
        "workload": {
            "context_len_per_sequence": workload[
                "context_len_per_sequence"
            ],
            "blocks_per_sequence": workload["blocks_per_sequence"],
            "active_total_context_tokens": workload[
                "active_total_context_tokens"
            ],
        },
        "timing_protocol": {
            "warmup": timing["warmup"],
            "iters": timing["iters"],
        },
        "p50_ms": float(timing["p50_ms"]),
        "mean_ms": float(timing["mean_ms"]),
        "p99_ms": float(timing["p99_ms"]),
        "min_ms": float(timing["min_ms"]),
        "max_ms": float(timing["max_ms"]),
        "hidden_sha256": {
            name: report["files"][f"{name}.pt"]
            for name in ("hidden_l3", "hidden_l4")
        },
        "runtime_marker_sha256": marker_hashes,
        "kv_key_sha256_by_rank": key_hashes,
        "evidence_sha256": evidence_sha256,
    }


def _source_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    p50_values = [item["p50_ms"] for item in records]
    mean_values = [item["mean_ms"] for item in records]
    return {
        "rounds": records,
        "median_round_p50_ms": _round(statistics.median(p50_values)),
        "median_round_mean_ms": _round(statistics.median(mean_values)),
        "round_p50_span_ms": _round(max(p50_values) - min(p50_values)),
        "round_mean_span_ms": _round(max(mean_values) - min(mean_values)),
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Five-layer MoE normal campaign performance",
        "",
        (
            "Each batch size is reported independently. Every sequence has "
            "context_len=65536; no cross-BS aggregate is used."
        ),
        "",
        "| BS | baseline p50 ms | candidate p50 ms | reduction | paired rounds |",
        "|---:|---:|---:|---:|:---|",
    ]
    for batch in BATCHES:
        item = result["batches"][str(batch)]
        base = item["baseline"]["median_round_p50_ms"]
        candidate = item["candidate"]["median_round_p50_ms"]
        reduction = item["aggregate_p50_reduction_pct"]
        paired = ", ".join(
            f"r{round_id}:{data['p50_reduction_pct']:+.2f}%"
            for round_id, data in item["paired_rounds"].items()
        )
        lines.append(
            f"| {batch} | {base:.4f} | {candidate:.4f} | "
            f"{reduction:+.2f}% | {paired} |"
        )
    lines.extend(
        [
            "",
            "Measurement integrity: "
            + ("PASS" if result["measurement_integrity_passed"] else "FAIL"),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    campaign = Path(args.campaign).resolve()
    round_ids = _round_ids(args.rounds)
    correctness = _load_json(
        campaign / "matrix_correctness_report.json"
    )
    if correctness.get("passed") is not True:
        raise AssertionError("matrix correctness gate has not passed")

    records: dict[int, dict[str, dict[int, dict[str, Any]]]] = {}
    image_refs: set[str] = set()
    run_nonces: set[str] = set()
    source_hashes: dict[str, set[tuple[str, str]]] = {
        source: set() for source in SOURCES
    }
    for batch in BATCHES:
        records[batch] = {source: {} for source in SOURCES}
        for round_id in round_ids:
            for source in SOURCES:
                item = validate_normal_run(
                    campaign,
                    source,
                    round_id,
                    batch,
                )
                records[batch][source][round_id] = item
                image_refs.add(item["image_ref"])
                if item["run_nonce"] in run_nonces:
                    raise AssertionError(
                        f"duplicate normal run nonce: {item['run_nonce']}"
                    )
                run_nonces.add(item["run_nonce"])
                source_hashes[source].add(
                    (
                        item["source"]["source_manifest_sha256"],
                        item["source"]["decode_fwd_sha256"],
                    )
                )
    if len(image_refs) != 1:
        raise AssertionError(f"campaign used multiple images: {image_refs}")
    expected_run_count = len(BATCHES) * len(SOURCES) * len(round_ids)
    if len(run_nonces) != expected_run_count:
        raise AssertionError(
            "normal campaign run nonces are not unique across all runs"
        )
    if any(len(values) != 1 for values in source_hashes.values()):
        raise AssertionError(
            f"source hashes changed within a source kind: {source_hashes}"
        )

    batches: dict[str, Any] = {}
    non_regression = True
    hidden_exact = True
    for batch in BATCHES:
        reference_hashes = records[batch]["baseline"][
            round_ids[0]
        ]["hidden_sha256"]
        for source in SOURCES:
            for round_id in round_ids:
                hidden_exact = (
                    hidden_exact
                    and records[batch][source][round_id][
                        "hidden_sha256"
                    ]
                    == reference_hashes
                )
        baseline = _source_summary(
            [
                records[batch]["baseline"][round_id]
                for round_id in round_ids
            ]
        )
        candidate = _source_summary(
            [
                records[batch]["candidate"][round_id]
                for round_id in round_ids
            ]
        )
        base_p50 = baseline["median_round_p50_ms"]
        candidate_p50 = candidate["median_round_p50_ms"]
        paired_rounds: dict[str, Any] = {}
        for round_id in round_ids:
            base = records[batch]["baseline"][round_id]
            cand = records[batch]["candidate"][round_id]
            paired_rounds[str(round_id)] = {
                "baseline_p50_ms": _round(base["p50_ms"]),
                "candidate_p50_ms": _round(cand["p50_ms"]),
                "p50_delta_ms": _round(
                    cand["p50_ms"] - base["p50_ms"]
                ),
                "p50_reduction_pct": _round(
                    (base["p50_ms"] - cand["p50_ms"])
                    / base["p50_ms"]
                    * 100.0,
                    3,
                ),
                "baseline_mean_ms": _round(base["mean_ms"]),
                "candidate_mean_ms": _round(cand["mean_ms"]),
            }
        batch_non_regression = candidate_p50 <= base_p50
        non_regression = non_regression and batch_non_regression
        batches[str(batch)] = {
            "context_len_per_sequence": CONTEXT_LEN,
            "active_total_context_tokens": batch * CONTEXT_LEN,
            "baseline": baseline,
            "candidate": candidate,
            "paired_rounds": paired_rounds,
            "aggregate_p50_delta_ms": _round(
                candidate_p50 - base_p50
            ),
            "aggregate_p50_reduction_pct": _round(
                (base_p50 - candidate_p50) / base_p50 * 100.0,
                3,
            ),
            "candidate_p50_non_regression": batch_non_regression,
        }

    result = {
        "schema": "step3p5.five-layer-moe-64k-performance.v2",
        "measurement_integrity_passed": hidden_exact,
        "correctness_report_passed": True,
        "hidden_hash_exact_across_selected_rounds": hidden_exact,
        "image_ref": next(iter(image_refs)),
        "rounds": list(round_ids),
        "batches": batches,
        "performance_non_regression_all_batches": non_regression,
        "interpretation": (
            "No latency is aggregated across batch sizes. Same-round pairs "
            "and the median across selected fresh-process rounds are both "
            "reported; launch-order provenance remains in campaign specs."
        ),
    }
    out = Path(args.out).resolve() if args.out else campaign
    out.mkdir(parents=True, exist_ok=True)
    json_content = json.dumps(result, indent=2, sort_keys=True) + "\n"
    _stable_write(out / "matrix_performance_report.json", json_content)
    _stable_write(
        out / "matrix_performance_report.md",
        _markdown(result),
    )
    print(json.dumps(
        {
            "measurement_integrity_passed": hidden_exact,
            "performance_non_regression_all_batches": non_regression,
            "report": str(out / "matrix_performance_report.json"),
        },
        sort_keys=True,
    ))
    return 0 if hidden_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
