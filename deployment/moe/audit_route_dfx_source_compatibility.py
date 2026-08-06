#!/usr/bin/env python3
"""Bind route sidecars to DFX captures through a fail-closed source audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from validate_five_layer_route_case import (
    ROUTE_ARTIFACT_NAMES,
    validate_route_validation_record,
)
from verify_exact_tree_manifest import verify_exact_tree

BATCHES = (1, 2, 4, 7, 8, 16)
CONTEXT_LEN = 65536
CRITICAL_FILES = (
    "COMMIT",
    "models/step3p5/decode_fwd.py",
    "tests/step3p5/harnesses/_five_layer_moe_program.py",
    "tests/step3p5/harnesses/_stage_five_layer_moe.py",
    "tools/step3p5/five_layer_moe_holder.py",
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
    parser.add_argument("--dfx-source", required=True)
    parser.add_argument("--route-source", required=True)
    parser.add_argument("--sidecar-dir", required=True)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--profile", default="row16")
    parser.add_argument("--source-role", default="reference")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _sha256(path: Path) -> str:
    _require(path.is_file(), f"missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def _manifest_entries(root: Path) -> dict[str, str]:
    return verify_exact_tree(root)


def _stable_write(path: Path, value: dict[str, Any]) -> None:
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        _require(
            path.read_text(encoding="utf-8") == content,
            f"refusing to replace different compatibility report: {path}",
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _relative_sidecar_dir(text: str) -> Path:
    path = Path(text)
    _require(
        not path.is_absolute() and path.parts and ".." not in path.parts,
        "--sidecar-dir must be a relative campaign path",
    )
    return path


def _audit_sources(
    dfx_source: Path,
    route_source: Path,
) -> dict[str, Any]:
    dfx_entries = _manifest_entries(dfx_source)
    route_entries = _manifest_entries(route_source)
    dfx_paths = set(dfx_entries)
    route_paths = set(route_entries)
    additions = sorted(route_paths - dfx_paths)
    removals = sorted(dfx_paths - route_paths)
    changed = sorted(
        path
        for path in dfx_paths & route_paths
        if dfx_entries[path] != route_entries[path]
    )
    _require(not removals, f"route source removed DFX files: {removals}")
    _require(
        additions == sorted(ROUTE_ADDITIONS),
        f"unexpected route source additions: {additions}",
    )
    _require(
        changed == sorted(EXPECTED_CHANGED_PROVENANCE),
        f"unexpected common-file drift: {changed}",
    )

    critical_hashes: dict[str, str] = {}
    for relative in CRITICAL_FILES:
        _require(relative in dfx_entries, f"DFX source missing {relative}")
        _require(relative in route_entries, f"route source missing {relative}")
        _require(
            dfx_entries[relative] == route_entries[relative],
            f"critical source drift: {relative}",
        )
        critical_hashes[relative] = dfx_entries[relative]

    route_hashes = {
        relative: route_entries[relative] for relative in ROUTE_ADDITIONS
    }
    return {
        "dfx_source_manifest_sha256": _sha256(
            dfx_source / "SOURCE_SHA256SUMS"
        ),
        "route_source_manifest_sha256": _sha256(
            route_source / "SOURCE_SHA256SUMS"
        ),
        "critical_hashes": critical_hashes,
        "route_addition_hashes": route_hashes,
        "changed_provenance_files": changed,
    }


def _audit_batch(
    campaign: Path,
    sidecar_dir: Path,
    *,
    batch: int,
    round_id: int,
    image_ref: str,
    source_audit: dict[str, Any],
    profile: str,
    source_role: str,
) -> dict[str, Any]:
    dfx_run = (
        campaign
        / "runs"
        / f"candidate-r{round_id}-dfx-bs{batch}-64k"
    )
    dfx_case = _json(
        dfx_run / "runtime" / f"bs{batch}" / "report.json"
    )
    route_runtime = (
        campaign
        / sidecar_dir
        / f"candidate-route-bs{batch}-64k"
        / "runtime"
    )
    validation_path = route_runtime / "route_artifact_validation.json"
    route_report_path = route_runtime / "five_layer_moe_route_report.json"
    validation = _json(validation_path)
    route_report = _json(route_report_path)
    route_provenance = route_report.get("provenance")
    _require(
        isinstance(route_provenance, dict),
        f"BS{batch}: missing route provenance",
    )
    route_source = route_provenance.get("source")
    _require(
        isinstance(route_source, dict),
        f"BS{batch}: missing route source provenance",
    )
    input_contract = route_provenance.get("input_contract")
    _require(
        isinstance(input_contract, dict),
        f"BS{batch}: missing route input contract",
    )
    route_workload = input_contract.get("workload")
    _require(
        isinstance(route_workload, dict),
        f"BS{batch}: missing route workload",
    )

    critical = source_audit["critical_hashes"]
    route_additions = source_audit["route_addition_hashes"]
    dfx_source = dfx_case.get("source")
    dfx_workload = dfx_case.get("workload")
    _require(isinstance(dfx_source, dict), f"BS{batch}: missing DFX source")
    _require(
        isinstance(dfx_workload, dict),
        f"BS{batch}: missing DFX workload",
    )
    _require(dfx_case.get("source_kind") == "candidate", "DFX source role")
    _require(dfx_case.get("mode") == "dfx", "DFX mode mismatch")
    _require(dfx_case.get("round") == round_id, "DFX round mismatch")
    _require(dfx_case.get("image_ref") == image_ref, "DFX image mismatch")
    _require(
        dfx_workload.get("active_batch") == batch,
        f"BS{batch}: DFX active batch mismatch",
    )
    _require(
        dfx_workload.get("context_len_per_sequence") == CONTEXT_LEN,
        f"BS{batch}: DFX context mismatch",
    )
    _require(
        dfx_source.get("source_manifest_sha256")
        == source_audit["dfx_source_manifest_sha256"],
        f"BS{batch}: DFX source manifest mismatch",
    )
    _require(
        dfx_source.get("decode_fwd_sha256")
        == critical["models/step3p5/decode_fwd.py"],
        f"BS{batch}: DFX decode hash mismatch",
    )
    _require(
        dfx_source.get("five_layer_program_sha256")
        == critical[
            "tests/step3p5/harnesses/_five_layer_moe_program.py"
        ],
        f"BS{batch}: DFX formal program hash mismatch",
    )
    _require(
        dfx_source.get("five_layer_holder_sha256")
        == critical["tools/step3p5/five_layer_moe_holder.py"],
        f"BS{batch}: DFX holder hash mismatch",
    )

    _require(validation.get("passed") is True, "route validation failed")
    _require(validation.get("profile") == profile, "route profile mismatch")
    _require(
        validation.get("source_role") == source_role,
        "route source role mismatch",
    )
    _require(
        validation.get("active_batch") == batch,
        f"BS{batch}: route validation batch mismatch",
    )
    _require(
        validation.get("context_len_per_sequence") == CONTEXT_LEN,
        f"BS{batch}: route validation context mismatch",
    )
    _require(
        validation.get("image_ref") == image_ref,
        f"BS{batch}: route validation image mismatch",
    )
    _require(
        validation.get("source_manifest_sha256")
        == source_audit["route_source_manifest_sha256"],
        f"BS{batch}: route source manifest mismatch",
    )
    _require(
        validation.get("decode_fwd_sha256")
        == critical["models/step3p5/decode_fwd.py"],
        f"BS{batch}: route validation decode mismatch",
    )
    validate_route_validation_record(
        validation,
        active_batch=batch,
        profile=profile,
        source_role=source_role,
        decode_fwd_sha256=critical["models/step3p5/decode_fwd.py"],
        image_ref=image_ref,
        source_manifest_sha256=source_audit[
            "route_source_manifest_sha256"
        ],
        input_tokens=dfx_workload["input_tokens"],
    )
    for artifact_name in ROUTE_ARTIFACT_NAMES:
        artifact_path = route_runtime / artifact_name
        _require(
            _sha256(artifact_path)
            == validation["artifacts"][artifact_name],
            f"BS{batch}: route artifact hash mismatch: {artifact_name}",
        )
    sidecar_path = route_runtime / "recv_meta_sidecar.pt"
    sidecar_sha = _sha256(sidecar_path)
    _require(
        validation.get("artifacts", {}).get("recv_meta_sidecar.pt")
        == sidecar_sha,
        f"BS{batch}: route sidecar hash mismatch",
    )

    _require(
        route_provenance.get("image_digest") == image_ref,
        f"BS{batch}: route report image mismatch",
    )
    _require(
        route_source.get("source_tree_manifest_sha256")
        == source_audit["route_source_manifest_sha256"],
        f"BS{batch}: route report manifest mismatch",
    )
    expected_route_source = {
        "decode_fwd_sha256": critical[
            "models/step3p5/decode_fwd.py"
        ],
        "formal_program_sha256": critical[
            "tests/step3p5/harnesses/_five_layer_moe_program.py"
        ],
        "route_program_sha256": route_additions[
            "tests/step3p5/harnesses/_five_layer_moe_route_program.py"
        ],
        "route_stage_sha256": route_additions[
            "tests/step3p5/harnesses/_stage_five_layer_moe_route.py"
        ],
        "route_holder_sha256": route_additions[
            "tools/step3p5/five_layer_moe_route_holder.py"
        ],
    }
    for key, expected in expected_route_source.items():
        _require(
            route_source.get(key) == expected,
            f"BS{batch}: route report {key} mismatch",
        )
    _require(
        route_workload.get("active_batch") == batch,
        f"BS{batch}: route report batch mismatch",
    )
    _require(
        route_workload.get("context_len") == CONTEXT_LEN,
        f"BS{batch}: route report context mismatch",
    )
    _require(
        dfx_workload.get("input_tokens") == input_contract.get("input_tokens"),
        f"BS{batch}: DFX/route input tokens differ",
    )

    return {
        "dfx_run": dfx_run.name,
        "route_run": route_runtime.parent.name,
        "input_tokens": dfx_workload["input_tokens"],
        "sidecar_sha256": sidecar_sha,
        "route_validation_sha256": _sha256(validation_path),
        "route_report_sha256": _sha256(route_report_path),
        "profile": profile,
        "source_role": source_role,
    }


def main() -> int:
    args = _parse_args()
    _require(args.round > 0, "--round must be positive")
    _require(
        bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.profile)),
        "invalid route profile",
    )
    _require(
        bool(
            re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]*",
                args.source_role,
            )
        ),
        "invalid route source role",
    )
    campaign = Path(args.campaign).resolve()
    dfx_source = Path(args.dfx_source).resolve()
    route_source = Path(args.route_source).resolve()
    sidecar_dir = _relative_sidecar_dir(args.sidecar_dir)
    source_audit = _audit_sources(dfx_source, route_source)
    batches = {
        str(batch): _audit_batch(
            campaign,
            sidecar_dir,
            batch=batch,
            round_id=args.round,
            image_ref=args.image_ref,
            source_audit=source_audit,
            profile=args.profile,
            source_role=args.source_role,
        )
        for batch in BATCHES
    }
    result = {
        "schema": "step3p5.five-layer-moe-route-dfx-source-compatibility.v1",
        "passed": True,
        "round": args.round,
        "profile": args.profile,
        "source_role": args.source_role,
        "image_ref": args.image_ref,
        "context_len_per_sequence": CONTEXT_LEN,
        "sidecar_dir": str(sidecar_dir),
        "dfx_source_root": str(dfx_source),
        "route_source_root": str(route_source),
        **source_audit,
        "batches": batches,
    }
    output = Path(args.out).resolve()
    _stable_write(output, result)
    print(json.dumps({"passed": True, "report": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
