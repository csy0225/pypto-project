#!/usr/bin/env python3
"""Run the fixed Step3p5 L0-L4 MoE 64K-per-sequence BS matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import time
from pathlib import Path

import torch


TP = 8
STORAGE_BATCH = 16
HIDDEN = 4096
HEAD_DIM = 128
BLOCK_SIZE = 128
BF16_BYTES = 2
CONTEXT_LEN = 65536
BLOCKS_PER_SEQUENCE = CONTEXT_LEN // BLOCK_SIZE
REQUIRED_BATCHES = (1, 2, 4, 7, 8, 16)
TOKENS = (
    6127,
    303,
    1207,
    19384,
    872,
    428,
    4231,
    2636,
    6178,
    410,
    1,
    2,
    3,
    4,
    5,
    6,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-kind", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--mode", choices=("normal", "dfx"), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--golden-root", default="")
    parser.add_argument("--dfx-profile", default="")
    return parser.parse_args()


def _devices(text: str) -> list[int]:
    devices = [int(item) for item in text.split(",") if item.strip()]
    if len(devices) != TP or len(set(devices)) != TP:
        raise ValueError(f"expected {TP} distinct devices, got {devices}")
    return devices


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allocation(active_batch: int) -> tuple[int, int]:
    scheduler_blocks = active_batch * BLOCKS_PER_SEQUENCE
    return scheduler_blocks, scheduler_blocks + (STORAGE_BATCH - 1)


def _configure(out: Path, active_batch: int) -> None:
    _scheduler_blocks, physical_blocks = _allocation(active_batch)
    os.environ["PYPTO_STEP3P5_MAX_SEQ"] = str(CONTEXT_LEN)
    os.environ["PYPTO_STEP3P5_BLOCK_TABLE_FLAT"] = str(
        STORAGE_BATCH * BLOCKS_PER_SEQUENCE
    )
    os.environ["PYPTO_STEP3P5_KV_NUM_LAYERS"] = "5"
    os.environ["PYPTO_STEP3P5_KV_CACHE_ROWS"] = str(
        5 * physical_blocks * BLOCK_SIZE
    )
    os.environ["PYPTO_STEP3P5_ROPE_SEQ"] = str(CONTEXT_LEN)
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(out / "build_output")
    if os.environ.get("PYPTO_STEP3P5_ATTN_TASK_PROFILE") != "a2a3":
        raise ValueError("matrix requires PYPTO_STEP3P5_ATTN_TASK_PROFILE=a2a3")


def _metadata(
    active_batch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from tools.step3p5.kv_padding import (
        make_padding_reserve,
        validate_fixed_batch_metadata,
    )

    scheduler_blocks, physical_blocks = _allocation(active_batch)
    reserve = make_padding_reserve(
        scheduler_blocks,
        physical_blocks,
        storage_capacity=STORAGE_BATCH,
    )
    seq = torch.ones(STORAGE_BATCH, dtype=torch.int32)
    pos = torch.zeros(STORAGE_BATCH, dtype=torch.int32)
    table = torch.zeros(
        STORAGE_BATCH,
        BLOCKS_PER_SEQUENCE,
        dtype=torch.int32,
    )
    slot = torch.zeros(STORAGE_BATCH, dtype=torch.int32)
    step = CONTEXT_LEN - 1
    for row in range(active_batch):
        first = row * BLOCKS_PER_SEQUENCE
        block_ids = first + torch.arange(
            BLOCKS_PER_SEQUENCE,
            dtype=torch.int32,
        )
        table[row] = block_ids
        seq[row] = CONTEXT_LEN
        pos[row] = step
        slot[row] = (
            int(block_ids[-1]) * BLOCK_SIZE + (step % BLOCK_SIZE)
        )
    for row, block_id in enumerate(
        reserve.padding_block_ids[: STORAGE_BATCH - active_batch],
        start=active_batch,
    ):
        table[row, 0] = int(block_id)
        slot[row] = int(block_id) * BLOCK_SIZE

    validate_fixed_batch_metadata(
        seq_lens=seq,
        positions=pos,
        block_table=table,
        slot_mapping=slot,
        valid_rows=active_batch,
        reserve=reserve,
        where=f"five-layer matrix BS{active_batch} 64K-per-sequence",
    )
    return seq, pos, table, slot


def _comparison(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, object]:
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            f"shape mismatch: actual={tuple(actual.shape)} "
            f"expected={tuple(expected.shape)}"
        )
    delta = (actual.float() - expected.float()).abs()
    return {
        "exact": bool(torch.equal(actual, expected)),
        "bad_count": int(torch.count_nonzero(actual != expected).item()),
        "max_abs": float(delta.amax().item()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
    }


def _golden_tensor(
    golden_root: Path,
    *,
    active_batch: int,
    name: str,
    out: Path,
) -> torch.Tensor:
    resolved_golden = golden_root.resolve()
    resolved_out = out.resolve()
    if (
        resolved_golden == resolved_out
        or resolved_golden in resolved_out.parents
        or resolved_out in resolved_golden.parents
    ):
        raise ValueError(
            "golden_root and out must be disjoint paths: "
            f"golden={resolved_golden}, out={resolved_out}"
        )
    batch_golden = resolved_golden / f"bs{active_batch}"
    manifest_path = batch_golden / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "step3p5.five-layer-moe-golden.v3":
        raise ValueError(f"{manifest_path}: unsupported golden schema")
    if manifest.get("source_kind") != "baseline":
        raise ValueError(f"{manifest_path}: golden is not from baseline")
    for field in (
        "source_run",
        "source_decode_fwd_sha256",
        "source_manifest_sha256",
    ):
        if not manifest.get(field):
            raise ValueError(f"{manifest_path}: missing {field}")
    if int(manifest.get("active_batch", -1)) != active_batch:
        raise ValueError(f"{manifest_path}: active_batch mismatch")
    if int(manifest.get("context_len_per_sequence", -1)) != CONTEXT_LEN:
        raise ValueError(f"{manifest_path}: context mismatch")
    image_ref = os.environ.get("MATRIX_IMAGE_REF")
    if image_ref and manifest.get("image_ref") != image_ref:
        raise ValueError(f"{manifest_path}: image_ref mismatch")
    path = batch_golden / f"{name}.pt"
    expected_sha = manifest.get("files", {}).get(path.name)
    if not expected_sha or _sha256(path) != expected_sha:
        raise ValueError(f"{manifest_path}: {path.name} hash mismatch")
    return torch.load(path, map_location="cpu", weights_only=True)


def _health(tensor: torch.Tensor, active_batch: int) -> dict[str, object]:
    active = tensor[:, :active_batch].float()
    row_abs_max = active.abs().amax(dim=-1)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(active).all()),
        "nonzero_rank_rows": int(
            torch.count_nonzero(row_abs_max > 0).item()
        ),
        "expected_nonzero_rank_rows": TP * active_batch,
        "tp_spread_max": float(
            (active - active[0:1]).abs().amax().item()
        ),
        "abs_max": float(active.abs().amax().item()),
        "abs_mean": float(active.abs().mean().item()),
    }


def _timing(samples_ms: list[float], warmup: int) -> dict[str, object]:
    ordered = sorted(samples_ms)
    return {
        "iters": len(ordered),
        "warmup": warmup,
        "min_ms": ordered[0],
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": ordered[len(ordered) // 2],
        "p99_ms": ordered[
            min(len(ordered) - 1, int(len(ordered) * 0.99))
        ],
        "max_ms": ordered[-1],
    }


def _run_one_batch(
    *,
    holder,
    main_stage,
    focused_stage,
    args: argparse.Namespace,
    active_batch: int,
    out: Path,
) -> dict[str, object]:
    batch_out = out / f"bs{active_batch}"
    batch_out.mkdir(parents=True, exist_ok=False)
    active_hidden = torch.stack(
        [
            main_stage._load_embedding_row(args.ckpt, token)
            for token in TOKENS[:active_batch]
        ],
        dim=0,
    ).contiguous()
    seq, pos, table, slot = _metadata(active_batch)
    set_kwargs = {
        "seq_lens": seq,
        "positions": pos,
        "block_table": table,
        "slot_mapping": slot,
    }

    for _ in range(args.warmup):
        holder.set_live_step(active_hidden, **set_kwargs)
        holder.run()

    samples_ms: list[float] = []
    last_result = None
    for _ in range(args.iters):
        holder.set_live_step(active_hidden, **set_kwargs)
        started = time.time()
        last_result = holder.run()
        samples_ms.append((time.time() - started) * 1000.0)
    if last_result is None:
        raise RuntimeError(f"BS{active_batch}: no measured iteration ran")

    outputs = {
        name: (
            last_result[name][:, :active_batch]
            .to(torch.bfloat16)
            .clone()
            .cpu()
        )
        for name in ("hidden_l3", "hidden_l4")
    }
    for name, tensor in outputs.items():
        torch.save(tensor, batch_out / f"{name}.pt")
    health = {
        name: _health(tensor, active_batch)
        for name, tensor in outputs.items()
    }
    if not all(
        item["finite"]
        and item["tp_spread_max"] == 0.0
        and item["nonzero_rank_rows"]
        == item["expected_nonzero_rank_rows"]
        for item in health.values()
    ):
        raise AssertionError(f"BS{active_batch}: hidden health failed {health}")

    comparisons: dict[str, object] = {}
    if args.golden_root:
        golden_root = Path(args.golden_root)
        for name, tensor in outputs.items():
            comparisons[name] = _comparison(
                tensor,
                _golden_tensor(
                    golden_root,
                    active_batch=active_batch,
                    name=name,
                    out=out,
                ),
            )
        if not all(item["exact"] for item in comparisons.values()):
            raise AssertionError(
                f"BS{active_batch}: hidden is not bit-exact to golden "
                f"{comparisons}"
            )

    dfx: dict[str, object] = {}
    if args.mode == "dfx":
        dfx_profile = (
            args.dfx_profile
            or os.environ.get("PYPTO_MOE_DFX_PROFILE")
            or args.source_kind
        )
        for _ in range(2):
            holder.set_live_step(active_hidden, **set_kwargs)
            holder.run()
        holder.set_live_step(active_hidden, **set_kwargs)
        holder.run(dfx="dep")
        build_dir = Path(holder.compiled.output_dir)
        dep_hashes = focused_stage._wait_for_artifacts(
            build_dir,
            "deps.json",
        )
        holder.set_live_step(active_hidden, **set_kwargs)
        holder.run()
        holder.set_live_step(active_hidden, **set_kwargs)
        holder.run(dfx="swim")
        dep_after = focused_stage._wait_for_artifacts(
            build_dir,
            "deps.json",
        )
        if dep_after != dep_hashes:
            raise RuntimeError(
                f"BS{active_batch}: swim capture changed dep-gen artifacts"
            )
        swim_hashes = focused_stage._wait_for_artifacts(
            build_dir,
            "l2_swimlane_records.json",
        )
        from tools.step3p5.analyze_five_layer_moe_dfx import analyze

        report = analyze(
            build_dir,
            batch_out / "dfx_analysis",
            profile=dfx_profile,
        )
        shutil.copytree(
            build_dir / "dfx_outputs",
            batch_out / "dfx_raw",
        )
        dfx = {
            "dep_gen_artifacts": dep_hashes,
            "dep_gen_preserved_after_swim": True,
            "swimlane_artifacts": swim_hashes,
            "report": str(
                batch_out / "dfx_analysis" / "moe_dfx_report.json"
            ),
            "rank_contract": report["rank_contract"],
            "structural_contracts": report["structural_contracts"],
            "slice_contract": report["slice_contract"],
            "routed_slice_profiles": report["routed_slice_profiles"],
            "source_policy": report["source_policy"],
            "expert_kernel_release": report["expert_kernel_release"],
            "admission": report["admission"],
            "profile": dfx_profile,
        }

    scheduler_blocks, physical_blocks = _allocation(active_batch)
    report = {
        "schema": "step3p5.five-layer-moe-matrix-case.v1",
        "source_kind": args.source_kind,
        "round": args.round,
        "mode": args.mode,
        "image_ref": os.environ.get("MATRIX_IMAGE_REF", ""),
        "image_pypto_commit": os.environ.get(
            "PYPTO_IMAGE_PYPTO_COMMIT",
            "",
        ),
        "source": {
            "source_manifest_sha256": _sha256(
                Path("SOURCE_SHA256SUMS")
            ),
            "decode_fwd_sha256": _sha256(
                Path("models/step3p5/decode_fwd.py")
            ),
            "five_layer_program_sha256": _sha256(
                Path(
                    "tests/step3p5/harnesses/"
                    "_five_layer_moe_program.py"
                )
            ),
            "five_layer_holder_sha256": _sha256(
                Path("tools/step3p5/five_layer_moe_holder.py")
            ),
        },
        "workload": {
            "active_batch": active_batch,
            "context_len_per_sequence": CONTEXT_LEN,
            "blocks_per_sequence": BLOCKS_PER_SEQUENCE,
            "active_total_context_tokens": active_batch * CONTEXT_LEN,
            "allocated_scheduler_blocks": scheduler_blocks,
            "allocated_physical_blocks": physical_blocks,
            "kv_num_layers": 5,
            "allocated_kv_rows_per_rank": (
                5 * physical_blocks * BLOCK_SIZE
            ),
            "allocated_kv_pool_bytes_per_rank": (
                2
                * 5
                * physical_blocks
                * BLOCK_SIZE
                * HEAD_DIM
                * BF16_BYTES
            ),
            "input_tokens": list(TOKENS[:active_batch]),
        },
        "timing": _timing(samples_ms, args.warmup),
        "health": health,
        "comparisons": comparisons,
        "dfx": dfx,
        "files": {
            f"{name}.pt": _sha256(batch_out / f"{name}.pt")
            for name in outputs
        },
    }
    (batch_out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    args = _parse_args()
    if args.iters <= 0 or args.warmup < 0:
        raise ValueError("--iters must be positive and --warmup non-negative")
    if args.batch not in REQUIRED_BATCHES:
        raise ValueError(
            f"--batch must be one of {REQUIRED_BATCHES}, got {args.batch}"
        )
    devices = _devices(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    _configure(out, args.batch)

    from tests.step3p5.harnesses import (
        _stage_five_layer_moe as focused_stage,
    )
    from tests.step3p5.harnesses import (
        _stage_main_hidden_only as main_stage,
    )
    from tools.step3p5.five_layer_moe_holder import FiveLayerMoeHolder

    scheduler_blocks, _physical_blocks = _allocation(args.batch)
    exporter_args = argparse.Namespace(
        out=str(out),
        ckpt=args.ckpt,
        num_blocks=scheduler_blocks,
        kv_num_layers=5,
        kv_probe=False,
    )
    procs = main_stage._start_exporters(exporter_args, devices)
    try:
        holder = FiveLayerMoeHolder(
            devices,
            str(out),
            args.ckpt,
            platform="a2a3",
            kv_ipc=True,
        ).build()
        with holder:
            report = _run_one_batch(
                holder=holder,
                main_stage=main_stage,
                focused_stage=focused_stage,
                args=args,
                active_batch=args.batch,
                out=out,
            )
    finally:
        main_stage._stop_exporters(out, procs)

    matrix = {
        "schema": "step3p5.five-layer-moe-64k-matrix.v1",
        "source_kind": args.source_kind,
        "round": args.round,
        "mode": args.mode,
        "image_ref": os.environ.get("MATRIX_IMAGE_REF", ""),
        "image_pypto_commit": os.environ.get(
            "PYPTO_IMAGE_PYPTO_COMMIT",
            "",
        ),
        "batches": [args.batch],
        "context_len_per_sequence": CONTEXT_LEN,
        "blocks_per_sequence": BLOCKS_PER_SEQUENCE,
        "reports": {
            str(args.batch): str(
                Path(f"bs{args.batch}") / "report.json"
            )
        },
        "report": report,
    }
    (out / "matrix_report.json").write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "matrix_report": str(out / "matrix_report.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
