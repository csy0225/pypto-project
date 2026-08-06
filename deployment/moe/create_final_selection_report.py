#!/usr/bin/env python3
"""Bind the single-axis tile and activation decisions to the final candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "step3p5.moe.final-selection.v1"
BATCHES = (1, 2, 4, 7, 8, 16)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--selected-variant", required=True)
    parser.add_argument("--selected-decode-fwd-sha256", required=True)
    parser.add_argument("--selected-raw-source-manifest-sha256", required=True)
    parser.add_argument("--source-audit", required=True)
    parser.add_argument("--tile-selection-report", required=True)
    parser.add_argument("--activation-sweep-report", required=True)
    parser.add_argument("--activation-source-audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source_audit_path = Path(args.source_audit).resolve()
    source_audit = _json(source_audit_path)
    tile_path = Path(args.tile_selection_report).resolve()
    tile = _json(tile_path)
    activation_path = Path(args.activation_sweep_report).resolve()
    activation = _json(activation_path)
    activation_audit_path = Path(args.activation_source_audit).resolve()
    activation_audit = _json(activation_audit_path)

    if not (
        source_audit.get("passed") is True
        and source_audit.get("image_ref") == args.image_ref
        and source_audit.get("selected_variant") == args.selected_variant
        and source_audit.get("selected_raw_source_manifest_sha256")
        == args.selected_raw_source_manifest_sha256
    ):
        raise AssertionError("matched source audit does not select candidate")
    if not (
        tile.get("passed") is True
        and tile.get("image") == args.image_ref
        and tile.get("decision") == "KEEP_CONTROL"
        and tile.get("selected_variant") == "mm-n64-r16"
    ):
        raise AssertionError("tile selection is not KEEP_CONTROL")
    if not (
        activation.get("correctness_passed") is True
        and activation.get("image") == args.image_ref
        and activation_audit.get("passed") is True
    ):
        raise AssertionError("activation sweep evidence failed")
    selected_rows = [
        row
        for row in activation.get("rows", [])
        if row.get("variant") == args.selected_variant
    ]
    if (
        {row.get("batch") for row in selected_rows} != set(BATCHES)
        or any(
            row.get("decode_fwd_sha256")
            != args.selected_decode_fwd_sha256
            or row.get("source_manifest_sha256")
            != args.selected_raw_source_manifest_sha256
            or row.get("hidden_l3_exact") is not True
            or row.get("hidden_l4_exact") is not True
            for row in selected_rows
        )
    ):
        raise AssertionError("selected activation rows are incomplete")
    audited = {
        item.get("variant"): item
        for item in activation_audit.get("variants", [])
    }
    selected_audit = audited.get(args.selected_variant)
    if not (
        isinstance(selected_audit, dict)
        and selected_audit.get("decode_fwd_sha256")
        == args.selected_decode_fwd_sha256
        and selected_audit.get("source_manifest_sha256")
        == args.selected_raw_source_manifest_sha256
    ):
        raise AssertionError("activation source audit does not bind candidate")

    result = {
        "schema": SCHEMA,
        "passed": True,
        "image_ref": args.image_ref,
        "selected_variant": args.selected_variant,
        "selected_decode_fwd_sha256": args.selected_decode_fwd_sha256,
        "selected_raw_source_manifest_sha256": (
            args.selected_raw_source_manifest_sha256
        ),
        "source_audit_sha256": _sha256(source_audit_path),
        "tile_decision": "KEEP_CONTROL",
        "tile_selected_variant": "mm-n64-r16",
        "activation_decision": "KEEP_ACT_N64",
        "activation_selected_variant": args.selected_variant,
        "rejected_variants": [
            "mm-n64-r32",
            "lineage-only",
            "full-execution-patch",
        ],
        "evidence": {
            "tile_selection_report": {
                "path": str(tile_path),
                "sha256": _sha256(tile_path),
            },
            "activation_sweep_report": {
                "path": str(activation_path),
                "sha256": _sha256(activation_path),
            },
            "activation_source_audit": {
                "path": str(activation_audit_path),
                "sha256": _sha256(activation_audit_path),
            },
        },
    }
    output = Path(args.out)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"FINAL_SELECTION=PASS out={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
