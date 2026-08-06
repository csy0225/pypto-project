#!/usr/bin/env python3
"""Validate fresh-process 64K matrix exactness and freeze baseline goldens."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


BATCHES = (1, 2, 4, 7, 8, 16)
RUN_SPECS = (
    ("baseline", 1),
    ("candidate", 1),
    ("baseline", 2),
    ("candidate", 2),
)
OUTPUTS = ("hidden_l3", "hidden_l4")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, object]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "exact": bool(torch.equal(actual, expected)),
        "bad_count": int(torch.count_nonzero(actual != expected).item()),
        "max_abs": float(delta.amax().item()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
    }


def _run_name(source_kind: str, round_id: int, batch: int) -> str:
    return f"{source_kind}-r{round_id}-normal-bs{batch}-64k"


def main() -> int:
    args = _parse_args()
    campaign = Path(args.campaign)
    tensors: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    reports: dict[str, dict[int, dict[str, object]]] = {}
    run_names = [
        _run_name(source_kind, round_id, batch)
        for source_kind, round_id in RUN_SPECS
        for batch in BATCHES
    ]
    for source_kind, round_id in RUN_SPECS:
        for batch in BATCHES:
            run = _run_name(source_kind, round_id, batch)
            runtime = campaign / "runs" / run / "runtime"
            batch_dir = runtime / f"bs{batch}"
            tensors[run] = {}
            reports[run] = {}
            reports[run][batch] = json.loads(
                (batch_dir / "report.json").read_text(encoding="utf-8")
            )
            report = reports[run][batch]
            if (
                report.get("source_kind") != source_kind
                or int(report.get("round", -1)) != round_id
                or report.get("mode") != "normal"
            ):
                raise AssertionError((run, "run identity", report))
            tensors[run][batch] = {}
            for output in OUTPUTS:
                tensor_path = batch_dir / f"{output}.pt"
                tensor = torch.load(
                    tensor_path,
                    map_location="cpu",
                    weights_only=True,
                )
                expected_shape = (8, batch, 4096)
                if tuple(tensor.shape) != expected_shape:
                    raise AssertionError(
                        (run, batch, output, tensor.shape, expected_shape)
                    )
                if tensor.dtype != torch.bfloat16:
                    raise AssertionError((run, batch, output, tensor.dtype))
                if report["files"].get(f"{output}.pt") != _sha256(
                    tensor_path
                ):
                    raise AssertionError((run, batch, output, "hash"))
                tensors[run][batch][output] = tensor

            workload = report["workload"]
            expected_workload = {
                "active_batch": batch,
                "context_len_per_sequence": 65536,
                "blocks_per_sequence": 512,
                "active_total_context_tokens": batch * 65536,
                "allocated_scheduler_blocks": batch * 512,
                "allocated_physical_blocks": batch * 512 + 15,
                "kv_num_layers": 5,
                "allocated_kv_rows_per_rank": (
                    5 * (batch * 512 + 15) * 128
                ),
                "allocated_kv_pool_bytes_per_rank": (
                    2 * 5 * (batch * 512 + 15) * 128 * 128 * 2
                ),
            }
            for key, expected in expected_workload.items():
                if workload.get(key) != expected:
                    raise AssertionError(
                        (run, key, workload.get(key), expected)
                    )

    reference_runs = {
        batch: _run_name("baseline", 1, batch)
        for batch in BATCHES
    }
    comparisons: dict[str, object] = {}
    for run in run_names:
        comparisons[run] = {}
        batch = next(
            item for item in BATCHES if f"-bs{item}-64k" in run
        )
        comparisons[run][str(batch)] = {
            output: _compare(
                tensors[run][batch][output],
                tensors[reference_runs[batch]][batch][output],
            )
            for output in OUTPUTS
        }
    health_ok = all(
        output_health["finite"]
        and output_health["tp_spread_max"] == 0.0
        and output_health["nonzero_rank_rows"]
        == output_health["expected_nonzero_rank_rows"]
        for run in run_names
        for batch in reports[run]
        for output_health in reports[run][batch]["health"].values()
    )
    exact_ok = all(
        item["exact"]
        for run in comparisons.values()
        for batch in run.values()
        for item in batch.values()
    )
    batch_extension: dict[str, object] = {}
    extension_ok = True
    for source_kind, round_id in RUN_SPECS:
        spec = f"{source_kind}-r{round_id}"
        batch_extension[spec] = {}
        for smaller, larger in zip(BATCHES, BATCHES[1:]):
            small_run = _run_name(source_kind, round_id, smaller)
            large_run = _run_name(source_kind, round_id, larger)
            batch_extension[spec][f"{smaller}->{larger}"] = {}
            for output in OUTPUTS:
                item = _compare(
                    tensors[large_run][larger][output][:, :smaller],
                    tensors[small_run][smaller][output],
                )
                batch_extension[spec][f"{smaller}->{larger}"][output] = item
                extension_ok = extension_ok and bool(item["exact"])
    passed = health_ok and exact_ok and extension_ok
    result = {
        "schema": "step3p5.five-layer-moe-64k-correctness.v1",
        "reference_runs": reference_runs,
        "runs": run_names,
        "batches": list(BATCHES),
        "context_len_per_sequence": 65536,
        "comparisons": comparisons,
        "batch_extension_invariance": batch_extension,
        "gates": {
            "health": health_ok,
            "fresh_process_raw_exact": exact_ok,
            "baseline_equals_candidate": exact_ok,
            "batch_extension_invariance": extension_ok,
        },
        "passed": passed,
    }
    report_path = campaign / "matrix_correctness_report.json"
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        print(json.dumps(result["gates"], sort_keys=True))
        return 1

    golden_root = campaign / "golden" / "heterogeneous-64k"
    if golden_root.exists():
        raise FileExistsError(f"refusing to overwrite {golden_root}")
    for batch in BATCHES:
        batch_golden = golden_root / f"bs{batch}"
        batch_golden.mkdir(parents=True)
        source_run = reference_runs[batch]
        source = (
            campaign
            / "runs"
            / source_run
            / "runtime"
            / f"bs{batch}"
        )
        run_root = campaign / "runs" / source_run
        source_report = reports[source_run][batch]["source"]
        files = {}
        for output in OUTPUTS:
            destination = batch_golden / f"{output}.pt"
            shutil.copy2(source / f"{output}.pt", destination)
            files[destination.name] = _sha256(destination)
        (batch_golden / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "step3p5.five-layer-moe-golden.v3",
                    "source_run": source_run,
                    "source_kind": "baseline",
                    "source_decode_fwd_sha256": source_report[
                        "decode_fwd_sha256"
                    ],
                    "source_manifest_sha256": source_report[
                        "source_manifest_sha256"
                    ],
                    "active_batch": batch,
                    "context_len_per_sequence": 65536,
                    "image_ref": (
                        run_root / "image_ref.txt"
                    ).read_text(encoding="utf-8").strip(),
                    "files": files,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {"passed": True, "golden_root": str(golden_root)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
