#!/usr/bin/env python3
"""Create the immutable authority policy for formal matched-source DFX."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "step3p5.moe.matched-dfx-policy.v1"
BATCHES = (1, 2, 4, 7, 8, 16)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": _sha256(path)}


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--image-pypto-commit", required=True)
    parser.add_argument("--selection-report", required=True)
    parser.add_argument("--source-audit", required=True)
    parser.add_argument("--scripts-manifest", required=True)
    parser.add_argument("--normal-capture-scripts-manifest", required=True)
    parser.add_argument("--normal-seal-authority", required=True)
    parser.add_argument("--normal-correctness-report", required=True)
    parser.add_argument("--normal-performance-report", required=True)
    parser.add_argument("--normal-campaign-spec", required=True)
    parser.add_argument("--normal-counterbalance-spec", required=True)
    parser.add_argument("--golden-root", required=True)
    parser.add_argument("--analyzer", required=True)
    parser.add_argument("--validator", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    selection_path = Path(args.selection_report)
    source_audit_path = Path(args.source_audit)
    selection = _json(selection_path)
    source_audit = _json(source_audit_path)
    if selection.get("passed") is not True or source_audit.get("passed") is not True:
        raise AssertionError("selection/source audit did not pass")
    sources = source_audit.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"baseline", "candidate"}:
        raise AssertionError("source audit must cover baseline/candidate")
    normalized_sources = {}
    for kind in ("baseline", "candidate"):
        item = sources[kind]
        normalized_sources[kind] = {
            "profile": item["profile"],
            "source_kind": kind,
            "source_role": kind,
            "policy_id": (
                f"formal-{kind}-{item['profile']}-20260806-v1"
            ),
            "source_manifest_sha256": item["source_manifest_sha256"],
            "decode_fwd_sha256": item["decode_fwd_sha256"],
            "release_enforced": kind == "candidate",
        }
    golden_root = Path(args.golden_root).resolve()
    golden = {
        str(batch): _reference(golden_root / f"bs{batch}" / "manifest.json")
        for batch in BATCHES
    }
    result = {
        "schema": SCHEMA,
        "authority": {
            "image_ref": args.image_ref,
            "image_pypto_commit": args.image_pypto_commit,
            "selection_report": _reference(selection_path),
            "source_audit": _reference(source_audit_path),
            "scripts_manifest": _reference(Path(args.scripts_manifest)),
            "normal_capture_scripts_manifest": _reference(
                Path(args.normal_capture_scripts_manifest)
            ),
            "normal_seal_authority": _reference(
                Path(args.normal_seal_authority)
            ),
            "normal_correctness_report": _reference(
                Path(args.normal_correctness_report)
            ),
            "normal_performance_report": _reference(
                Path(args.normal_performance_report)
            ),
            "normal_campaign_spec": _reference(
                Path(args.normal_campaign_spec)
            ),
            "normal_counterbalance_spec": _reference(
                Path(args.normal_counterbalance_spec)
            ),
            "analyzer_sha256": _sha256(Path(args.analyzer)),
            "validator_sha256": _sha256(Path(args.validator)),
            "golden_manifests": golden,
            "workload": {
                "layers": [0, 1, 2, 3, 4],
                "batches": list(BATCHES),
                "context_len_per_sequence": 65536,
                "blocks_per_sequence": 512,
                "hidden_outputs": ["hidden_l3", "hidden_l4"],
                "hidden_exact": True,
            },
            "capture": {
                "normal_rounds": 3,
                "normal_iters": 30,
                "normal_warmup": 5,
                "counterbalanced": True,
                "dfx_rounds": 1,
                "dfx_iters": 1,
                "dfx_warmup": 2,
                "l2_swimlane_reuse_dep_gen": True,
            },
        },
        "sources": normalized_sources,
    }
    output = Path(args.out)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"MATCHED_DFX_POLICY=PASS out={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
