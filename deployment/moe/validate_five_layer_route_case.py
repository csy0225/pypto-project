#!/usr/bin/env python3
"""Independently validate one L0-L4 MoE route sidecar capture."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch


TP = 8
TOPK = 8
N_LOCAL_EXPERTS = 36
N_LOCAL_EXPERTS_PAD = 40
CONTEXT_LEN = 65536
BLOCK_SIZE = 128
BLOCKS_PER_SEQUENCE = CONTEXT_LEN // BLOCK_SIZE
ROUTE_VALIDATION_SCHEMA = "step3p5.five-layer-moe-route-validation.v1"
ROUTE_ARTIFACT_NAMES = (
    "hidden_l3.pt",
    "hidden_l4.pt",
    "recv_meta.pt",
    "local_expert_count.pt",
    "recv_meta_sidecar.pt",
)
SOURCE_PROFILES = {
    "row16": {
        "decode_fwd_sha256": (
            "65b0b8bf139aa40a5cf67317148dc16193ff22a81b394fbfe86e31ea05623e08"
        ),
        "source_role": "reference",
    },
    "shared-split": {
        "decode_fwd_sha256": (
            "572ea2a2b0ceab8952cbd8d7c1f383351fb877e11b01b4d68a60697f4508576b"
        ),
        "source_role": "candidate",
    },
}
CANDIDATE_DECODE_SHA256 = SOURCE_PROFILES["row16"]["decode_fwd_sha256"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--golden-dir", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--active-batch", type=int, required=True)
    parser.add_argument("--input-tokens", required=True)
    parser.add_argument(
        "--profile",
        default="row16",
    )
    parser.add_argument("--expected-decode-sha256", default="")
    parser.add_argument("--source-role", default="")
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _sha256(path: Path) -> str:
    _require(path.is_file(), f"missing artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def _tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(value, torch.Tensor), f"{path}: expected tensor")
    return value


def _tokens(text: str, active_batch: int) -> list[int]:
    values = [int(item) for item in text.split(",") if item.strip()]
    _require(
        len(values) == active_batch,
        "input token count does not match active batch",
    )
    return values


def _stable_write(path: Path, value: dict[str, Any]) -> None:
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        _require(
            path.read_text(encoding="utf-8") == content,
            f"refusing to replace different validation evidence: {path}",
        )
        return
    path.write_text(content, encoding="utf-8")


def validate_route_validation_record(
    value: dict[str, Any],
    *,
    active_batch: int,
    profile: str,
    source_role: str,
    decode_fwd_sha256: str,
    image_ref: str,
    source_manifest_sha256: str,
    input_tokens: list[int] | None = None,
) -> None:
    """Fail closed when a persisted route-validation record is consumed."""
    expected_keys = {
        "schema",
        "passed",
        "active_batch",
        "context_len_per_sequence",
        "input_tokens",
        "image_ref",
        "source_manifest_sha256",
        "profile",
        "source_role",
        "decode_fwd_sha256",
        "hidden_bit_exact",
        "padding_zero",
        "local_count_exact",
        "expected_routes_per_source_per_layer",
        "global_routes_per_layer",
        "expected_global_routes_per_layer",
        "window_independence_validated",
        "artifacts",
    }
    _require(
        set(value) == expected_keys,
        "route validation keys are incomplete or unexpected",
    )
    _require(value.get("schema") == ROUTE_VALIDATION_SCHEMA, "route schema")
    _require(value.get("passed") is True, "route validation did not pass")
    _require(value.get("active_batch") == active_batch, "route batch mismatch")
    _require(
        value.get("context_len_per_sequence") == CONTEXT_LEN,
        "route context mismatch",
    )
    tokens = value.get("input_tokens")
    _require(
        isinstance(tokens, list)
        and len(tokens) == active_batch
        and all(isinstance(item, int) for item in tokens),
        "route input token contract mismatch",
    )
    if input_tokens is not None:
        _require(tokens == input_tokens, "route input tokens differ")
    expected_scalars = {
        "profile": profile,
        "source_role": source_role,
        "decode_fwd_sha256": decode_fwd_sha256,
        "image_ref": image_ref,
        "source_manifest_sha256": source_manifest_sha256,
    }
    for key, expected in expected_scalars.items():
        _require(value.get(key) == expected, f"route {key} mismatch")
    for key in (
        "hidden_bit_exact",
        "padding_zero",
        "local_count_exact",
        "window_independence_validated",
    ):
        _require(value.get(key) is True, f"route {key} is not true")
    expected_per_source = active_batch * TOPK
    expected_global = active_batch * TP * TOPK
    _require(
        value.get("expected_routes_per_source_per_layer")
        == expected_per_source,
        "route per-source total mismatch",
    )
    _require(
        value.get("expected_global_routes_per_layer") == expected_global,
        "route expected global total mismatch",
    )
    _require(
        value.get("global_routes_per_layer")
        == [expected_global, expected_global],
        "route observed global totals mismatch",
    )
    artifacts = value.get("artifacts")
    _require(
        isinstance(artifacts, dict)
        and set(artifacts) == set(ROUTE_ARTIFACT_NAMES)
        and all(
            isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in artifacts.values()
        ),
        "route artifact hash contract mismatch",
    )


def _source_profile(args: argparse.Namespace) -> dict[str, str]:
    _require(
        bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.profile)),
        "invalid route profile",
    )
    explicit = bool(args.expected_decode_sha256 or args.source_role)
    if explicit:
        _require(
            bool(re.fullmatch(r"[0-9a-f]{64}", args.expected_decode_sha256)),
            "invalid expected decode hash",
        )
        _require(
            bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.source_role)),
            "invalid route source role",
        )
        return {
            "decode_fwd_sha256": args.expected_decode_sha256,
            "source_role": args.source_role,
        }
    _require(
        args.profile in SOURCE_PROFILES,
        "generic route profile requires explicit decode hash and source role",
    )
    return SOURCE_PROFILES[args.profile]


def recompute_route_tensor_contracts(
    *,
    runtime: Path,
    golden_dir: Path,
    active_batch: int,
    image_ref: str,
) -> dict[str, Any]:
    """Recompute route correctness from the persisted tensor entities."""
    batch = active_batch
    _require(batch in {1, 2, 4, 7, 8, 16}, "unsupported active batch")
    runtime = runtime.resolve()
    golden_dir = golden_dir.resolve()
    report = _json(runtime / "five_layer_moe_route_report.json")
    golden_manifest = _json(golden_dir / "manifest.json")

    _require(
        golden_manifest.get("schema") == "step3p5.five-layer-moe-golden.v3",
        "unsupported golden schema",
    )
    _require(
        golden_manifest.get("source_kind") == "baseline",
        "golden is not from the frozen baseline",
    )
    _require(
        golden_manifest.get("active_batch") == batch,
        "golden active batch mismatch",
    )
    _require(
        golden_manifest.get("context_len_per_sequence") == CONTEXT_LEN,
        "golden context mismatch",
    )
    _require(
        golden_manifest.get("image_ref") == image_ref,
        "golden image mismatch",
    )

    hidden_l3 = _tensor(runtime / "hidden_l3.pt")
    hidden_l4 = _tensor(runtime / "hidden_l4.pt")
    golden_l3 = _tensor(golden_dir / "hidden_l3.pt")
    golden_l4 = _tensor(golden_dir / "hidden_l4.pt")
    for name, actual, expected in (
        ("hidden_l3", hidden_l3, golden_l3),
        ("hidden_l4", hidden_l4, golden_l4),
    ):
        _require(
            tuple(actual.shape) == (TP, batch, 4096),
            f"{name}: unexpected shape {tuple(actual.shape)}",
        )
        _require(torch.equal(actual, expected), f"{name}: not bit-exact")
        expected_sha = golden_manifest.get("files", {}).get(f"{name}.pt")
        _require(
            expected_sha == _sha256(golden_dir / f"{name}.pt"),
            f"{name}: golden hash mismatch",
        )

    recv_meta = _tensor(runtime / "recv_meta.pt")
    local_count = _tensor(runtime / "local_expert_count.pt")
    _require(
        tuple(recv_meta.shape) == (TP, 2, TP, N_LOCAL_EXPERTS_PAD),
        f"recv_meta shape mismatch: {tuple(recv_meta.shape)}",
    )
    _require(
        tuple(local_count.shape) == (TP, 2, N_LOCAL_EXPERTS),
        f"local count shape mismatch: {tuple(local_count.shape)}",
    )
    _require(recv_meta.dtype == torch.int32, "recv_meta dtype mismatch")
    _require(local_count.dtype == torch.int32, "local count dtype mismatch")
    _require(not bool(torch.any(recv_meta < 0)), "negative route count")
    _require(
        not bool(torch.any(recv_meta[:, :, :, N_LOCAL_EXPERTS:] != 0)),
        "padded experts 36:40 are non-zero",
    )

    routed = recv_meta[:, :, :, :N_LOCAL_EXPERTS].to(torch.int64)
    derived_count = routed.sum(dim=2).to(torch.int32)
    _require(
        torch.equal(derived_count, local_count),
        "local count is not exact sum over source rank",
    )
    per_layer_source = routed.sum(dim=(0, 3))
    expected_per_source = batch * TOPK
    _require(
        torch.equal(
            per_layer_source,
            torch.full(
                (2, TP),
                expected_per_source,
                dtype=torch.int64,
            ),
        ),
        "per-layer source route totals mismatch",
    )
    global_per_layer = per_layer_source.sum(dim=1)
    expected_global = batch * TP * TOPK
    _require(
        torch.equal(
            global_per_layer,
            torch.full((2,), expected_global, dtype=torch.int64),
        ),
        "global route totals mismatch",
    )

    sidecar_path = runtime / "recv_meta_sidecar.pt"
    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=True)
    _require(isinstance(sidecar, dict), "sidecar root is not a mapping")
    _require(
        sidecar.get("schema") == "step3p5.five-layer-moe-recv-meta.v1",
        "sidecar schema mismatch",
    )
    _require(
        torch.equal(
            sidecar["recv_meta"],
            recv_meta.permute(1, 0, 2, 3).contiguous(),
        ),
        "sidecar recv_meta differs from device-order artifact",
    )
    _require(
        torch.equal(
            sidecar["local_expert_count"],
            local_count.permute(1, 0, 2).contiguous(),
        ),
        "sidecar local counts differ from device-order artifact",
    )
    windows = sidecar.get("window_provenance")
    _require(
        isinstance(windows, list) and len(windows) == 2,
        "sidecar window provenance is incomplete",
    )
    _require(
        len({str(item.get("window_id")) for item in windows}) == 2,
        "L3/L4 sidecar windows are not independent",
    )

    artifact_hashes = {
        name: _sha256(runtime / name) for name in ROUTE_ARTIFACT_NAMES
    }
    _require(
        report.get("artifacts") == artifact_hashes,
        "route report artifact hashes mismatch",
    )
    return {
        "hidden_bit_exact": True,
        "padding_zero": True,
        "local_count_exact": True,
        "expected_routes_per_source_per_layer": expected_per_source,
        "global_routes_per_layer": global_per_layer.tolist(),
        "expected_global_routes_per_layer": expected_global,
        "window_independence_validated": True,
        "artifacts": artifact_hashes,
    }


def main() -> int:
    args = _parse_args()
    batch = args.active_batch
    _require(batch in {1, 2, 4, 7, 8, 16}, "unsupported active batch")
    tokens = _tokens(args.input_tokens, batch)
    runtime = Path(args.runtime).resolve()
    golden_dir = Path(args.golden_dir).resolve()
    source_root = Path(args.source_root).resolve()
    source_profile = _source_profile(args)

    report = _json(runtime / "five_layer_moe_route_report.json")
    provenance = report.get("provenance")
    _require(isinstance(provenance, dict), "missing route provenance")
    input_contract = provenance.get("input_contract")
    _require(isinstance(input_contract, dict), "missing input contract")
    workload = input_contract.get("workload")
    _require(isinstance(workload, dict), "missing route workload")

    expected_workload = {
        "active_batch": batch,
        "context_len": CONTEXT_LEN,
        "num_blocks_per_sequence": BLOCKS_PER_SEQUENCE,
        "context_semantics": "per_active_sequence",
        "blocks_per_sequence": BLOCKS_PER_SEQUENCE,
        "block_table_blocks_per_row_capacity": BLOCKS_PER_SEQUENCE,
        "scheduler_num_blocks": batch * BLOCKS_PER_SEQUENCE,
        "physical_num_blocks": batch * BLOCKS_PER_SEQUENCE + 15,
        "max_sequence_tokens": CONTEXT_LEN,
    }
    for key, expected in expected_workload.items():
        _require(
            workload.get(key) == expected,
            f"workload {key}={workload.get(key)!r}, expected={expected!r}",
        )
    _require(input_contract.get("input_tokens") == tokens, "token mismatch")
    _require(
        provenance.get("image_digest") == args.image_ref,
        "route image digest mismatch",
    )

    source = provenance.get("source")
    _require(isinstance(source, dict), "missing route source provenance")
    source_manifest = source_root / "SOURCE_SHA256SUMS"
    _require(
        source.get("source_tree_manifest_sha256")
        == _sha256(source_manifest),
        "route source manifest hash mismatch",
    )
    _require(
        source.get("decode_fwd_sha256")
        == source_profile["decode_fwd_sha256"],
        f"route source is not the frozen {args.profile} source",
    )

    tensor_contract = recompute_route_tensor_contracts(
        runtime=runtime,
        golden_dir=golden_dir,
        active_batch=batch,
        image_ref=args.image_ref,
    )
    result = {
        "schema": ROUTE_VALIDATION_SCHEMA,
        "passed": True,
        "active_batch": batch,
        "context_len_per_sequence": CONTEXT_LEN,
        "input_tokens": tokens,
        "image_ref": args.image_ref,
        "source_manifest_sha256": _sha256(source_manifest),
        "profile": args.profile,
        "source_role": source_profile["source_role"],
        "decode_fwd_sha256": source_profile["decode_fwd_sha256"],
        **tensor_contract,
    }
    validate_route_validation_record(
        result,
        active_batch=batch,
        profile=args.profile,
        source_role=source_profile["source_role"],
        decode_fwd_sha256=source_profile["decode_fwd_sha256"],
        image_ref=args.image_ref,
        source_manifest_sha256=_sha256(source_manifest),
        input_tokens=tokens,
    )
    output = runtime / "route_artifact_validation.json"
    _stable_write(output, result)
    print(json.dumps({"passed": True, "report": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
