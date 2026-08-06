#!/usr/bin/env python3
"""Freeze the exact baseline/candidate source identities for formal MoE A/B."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from verify_exact_tree_manifest import verify_exact_tree


SCHEMA = "step3p5.moe.matched-source-audit.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(root: Path, *, kind: str, profile: str) -> dict[str, object]:
    root = root.resolve()
    verify_exact_tree(root)
    return {
        "root": str(root),
        "profile": profile,
        "source_kind": kind,
        "source_role": kind,
        "source_manifest_sha256": _sha256(root / "SOURCE_SHA256SUMS"),
        "decode_fwd_sha256": _sha256(
            root / "models/step3p5/decode_fwd.py"
        ),
        "analyzer_sha256": _sha256(
            root / "tools/step3p5/analyze_five_layer_moe_dfx.py"
        ),
        "exact_tree": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--baseline-source", required=True)
    parser.add_argument("--baseline-profile", required=True)
    parser.add_argument("--candidate-source", required=True)
    parser.add_argument("--candidate-profile", required=True)
    parser.add_argument("--selected-raw-source-manifest-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sources = {
        "baseline": _source(
            Path(args.baseline_source),
            kind="baseline",
            profile=args.baseline_profile,
        ),
        "candidate": _source(
            Path(args.candidate_source),
            kind="candidate",
            profile=args.candidate_profile,
        ),
    }
    if (
        sources["baseline"]["analyzer_sha256"]
        != sources["candidate"]["analyzer_sha256"]
    ):
        raise AssertionError("baseline/candidate analyzers differ")
    result = {
        "schema": SCHEMA,
        "passed": True,
        "image_ref": args.image_ref,
        "selected_variant": args.candidate_profile,
        "selected_raw_source_manifest_sha256": (
            args.selected_raw_source_manifest_sha256
        ),
        "sources": sources,
    }
    output = Path(args.out)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"MATCHED_SOURCE_AUDIT=PASS out={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
