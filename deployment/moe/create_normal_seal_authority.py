#!/usr/bin/env python3
"""Freeze normal-campaign evidence before adding runtime seal fields."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from validate_five_layer_case import (
    NORMAL_SEAL_AUTHORITY_SCHEMA,
    RUN_RE,
    _canonical_json_bytes,
    _json,
    _require,
    _run_evidence_sha256,
    _sha256,
    _validate_runtime_markers,
)


BATCHES = (1, 2, 4, 7, 8, 16)
SOURCES = ("baseline", "candidate")
ROUNDS = (1, 2, 3)
SEAL_FIELDS = ("runtime_marker_sha256", "kv_key_sha256_by_rank")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _legacy_and_sealed(
    run: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _json(run / "artifact_validation.json")
    legacy = dict(value)
    for field in SEAL_FIELDS:
        legacy.pop(field, None)
    nonce = (run / "run_nonce.txt").read_text(encoding="utf-8").strip()
    match = RUN_RE.fullmatch(run.name)
    _require(match is not None, f"invalid normal run name: {run.name}")
    _, _, mode, batch_text = match.groups()
    _require(mode == "normal", f"not a normal run: {run.name}")
    batch = int(batch_text)
    marker_hashes: dict[str, str] = {}
    key_hashes: dict[str, str] = {}
    for rank in range(8):
        hashes = _validate_runtime_markers(
            run / "runtime",
            rank=rank,
            batch=batch,
            nonce=nonce,
        )
        key_name = f"pypto_kvpool.key.rank{rank}"
        key_hashes[str(rank)] = hashes[key_name]
        marker_hashes.update(
            {
                name: digest
                for name, digest in hashes.items()
                if name != key_name
            }
        )
    sealed = dict(legacy)
    sealed["runtime_marker_sha256"] = marker_hashes
    sealed["kv_key_sha256_by_rank"] = key_hashes
    return legacy, sealed


def _record(run: Path) -> dict[str, Any]:
    legacy, sealed = _legacy_and_sealed(run)
    return {
        "legacy_validation_sha256": hashlib.sha256(
            _canonical_json_bytes(legacy)
        ).hexdigest(),
        "sealed_validation_sha256": hashlib.sha256(
            _canonical_json_bytes(sealed)
        ).hexdigest(),
        "evidence_sha256": _run_evidence_sha256(run),
    }


def main() -> int:
    args = _parse_args()
    campaign = Path(args.campaign).resolve()
    _require(campaign.is_dir(), f"missing campaign: {campaign}")
    runs_root = campaign / "runs"
    expected = {
        f"{source}-r{round_id}-normal-bs{batch}-64k"
        for source in SOURCES
        for round_id in ROUNDS
        for batch in BATCHES
    }
    actual = {
        path.name
        for path in runs_root.iterdir()
        if path.is_dir()
    }
    _require(
        actual == expected,
        "normal seal authority must cover exactly the 36 normal runs",
    )
    result = {
        "schema": NORMAL_SEAL_AUTHORITY_SCHEMA,
        "campaign_root": str(campaign),
        "runs": {
            name: _record(runs_root / name)
            for name in sorted(expected)
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical_json_bytes(result).decode("utf-8")
    if output.exists():
        _require(
            output.is_file() and not output.is_symlink(),
            f"authority output is not a regular file: {output}",
        )
        _require(
            output.read_text(encoding="utf-8") == content,
            f"refusing to replace different authority: {output}",
        )
    else:
        output.write_text(content, encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": NORMAL_SEAL_AUTHORITY_SCHEMA,
                "manifest": str(output.resolve()),
                "sha256": _sha256(output),
                "runs": len(expected),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
