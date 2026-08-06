#!/usr/bin/env python3
"""Aggregate independently validated L0-L4 MoE route sidecars."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from validate_five_layer_route_case import (
    ROUTE_ARTIFACT_NAMES,
    recompute_route_tensor_contracts,
    validate_route_validation_record,
)


BATCHES = (1, 2, 4, 7, 8, 16)
CONTEXT_LEN = 65536
PROFILE_DECODE_SHA256 = {
    "row16": (
        "65b0b8bf139aa40a5cf67317148dc16193ff22a81b394fbfe86e31ea05623e08"
    ),
    "shared-split": (
        "572ea2a2b0ceab8952cbd8d7c1f383351fb877e11b01b4d68a60697f4508576b"
    ),
}
PROFILE_SOURCE_ROLE = {
    "row16": "reference",
    "shared-split": "candidate",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--sidecar-dir", default="route-sidecar")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--expected-profile",
        default="row16",
    )
    parser.add_argument("--expected-decode-sha256", default="")
    parser.add_argument("--expected-source-role", default="")
    parser.add_argument("--expected-image-ref", required=True)
    parser.add_argument(
        "--expected-source-manifest-sha256",
        required=True,
    )
    parser.add_argument(
        "--golden-root",
        default="",
        help=(
            "six-BS golden root; defaults to "
            "<campaign>/golden/heterogeneous-64k"
        ),
    )
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected JSON object")
    return value


def _stable_write(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"refusing to replace different report: {path}")
    path.write_text(content, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = _parse_args()
    campaign = Path(args.campaign).resolve()
    sidecar_dir = Path(args.sidecar_dir)
    if (
        sidecar_dir.is_absolute()
        or not sidecar_dir.parts
        or ".." in sidecar_dir.parts
    ):
        raise ValueError("--sidecar-dir must be a relative campaign path")
    out = Path(args.out).resolve() if args.out else campaign
    out.mkdir(parents=True, exist_ok=True)
    golden_root = (
        Path(args.golden_root).resolve()
        if args.golden_root
        else campaign / "golden" / "heterogeneous-64k"
    )
    if not golden_root.is_dir():
        raise AssertionError(f"golden root is missing: {golden_root}")
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*",
        args.expected_profile,
    ):
        raise ValueError("invalid expected profile")
    if not re.fullmatch(
        r".+@sha256:[0-9a-f]{64}",
        args.expected_image_ref,
    ):
        raise ValueError("invalid expected image ref")
    if not re.fullmatch(
        r"[0-9a-f]{64}",
        args.expected_source_manifest_sha256,
    ):
        raise ValueError("invalid expected source manifest hash")
    explicit = bool(
        args.expected_decode_sha256 or args.expected_source_role
    )
    if explicit:
        if not re.fullmatch(r"[0-9a-f]{64}", args.expected_decode_sha256):
            raise ValueError("invalid expected decode hash")
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*",
            args.expected_source_role,
        ):
            raise ValueError("invalid expected source role")
        expected_decode = args.expected_decode_sha256
        expected_role = args.expected_source_role
    else:
        if args.expected_profile not in PROFILE_DECODE_SHA256:
            raise ValueError(
                "generic profile requires explicit decode hash and source role"
            )
        expected_decode = PROFILE_DECODE_SHA256[args.expected_profile]
        expected_role = PROFILE_SOURCE_ROLE[args.expected_profile]
    records: dict[str, Any] = {}
    source_manifests: set[str] = set()
    image_refs: set[str] = set()
    passed = True
    for batch in BATCHES:
        run = campaign / sidecar_dir / f"candidate-route-bs{batch}-64k"
        validation = _json(
            run / "runtime" / "route_artifact_validation.json"
        )
        if validation.get("passed") is not True:
            raise AssertionError(f"{run.name}: validation did not pass")
        sidecar = run / "runtime" / "recv_meta_sidecar.pt"
        if not sidecar.is_file():
            raise AssertionError(f"{run.name}: route sidecar is missing")
        validate_route_validation_record(
            validation,
            active_batch=batch,
            profile=args.expected_profile,
            source_role=expected_role,
            decode_fwd_sha256=expected_decode,
            image_ref=args.expected_image_ref,
            source_manifest_sha256=args.expected_source_manifest_sha256,
        )
        runtime = run / "runtime"
        recomputed = recompute_route_tensor_contracts(
            runtime=runtime,
            golden_dir=golden_root / f"bs{batch}",
            active_batch=batch,
            image_ref=args.expected_image_ref,
        )
        for key, expected in recomputed.items():
            if validation.get(key) != expected:
                raise AssertionError(
                    f"{run.name}: recomputed route {key} mismatch"
                )
        for name in ROUTE_ARTIFACT_NAMES:
            path = runtime / name
            if (
                not path.is_file()
                or _sha256(path)
                != validation["artifacts"][name]
            ):
                raise AssertionError(
                    f"{run.name}: route artifact hash mismatch: {name}"
                )
        expected_global = batch * 64
        checks = {
            "schema": (
                validation.get("schema")
                == "step3p5.five-layer-moe-route-validation.v1"
            ),
            "profile": validation.get("profile") == args.expected_profile,
            "source_role": validation.get("source_role") == expected_role,
            "decode": validation.get("decode_fwd_sha256") == expected_decode,
            "active_batch": validation.get("active_batch") == batch,
            "context": (
                validation.get("context_len_per_sequence") == CONTEXT_LEN
            ),
            "hidden": validation.get("hidden_bit_exact") is True,
            "padding": validation.get("padding_zero") is True,
            "local_count": validation.get("local_count_exact") is True,
            "window": (
                validation.get("window_independence_validated") is True
            ),
            "global_routes": (
                validation.get("global_routes_per_layer")
                == [expected_global, expected_global]
                and validation.get("expected_global_routes_per_layer")
                == expected_global
            ),
            "sidecar_hash": (
                validation.get("artifacts", {}).get("recv_meta_sidecar.pt")
                == _sha256(sidecar)
            ),
        }
        record_passed = all(checks.values())
        passed = passed and record_passed
        source_manifests.add(validation["source_manifest_sha256"])
        image_refs.add(validation["image_ref"])
        records[str(batch)] = {
            "run": run.name,
            "passed": record_passed,
            "checks": checks,
            "expected_global_routes_per_layer": expected_global,
            "sidecar": str(run / "runtime" / "recv_meta_sidecar.pt"),
            "profile": validation.get("profile"),
            "evidence_sha256": {
                "runtime/route_artifact_validation.json": _sha256(
                    runtime / "route_artifact_validation.json"
                ),
                "runtime/five_layer_moe_route_report.json": _sha256(
                    runtime / "five_layer_moe_route_report.json"
                ),
                **{
                    f"runtime/{name}": recomputed["artifacts"][name]
                    for name in ROUTE_ARTIFACT_NAMES
                },
            },
        }
    if len(source_manifests) != 1:
        raise AssertionError("route source manifest changed across BS")
    if source_manifests != {args.expected_source_manifest_sha256}:
        raise AssertionError("route source manifest differs from authority")
    if len(image_refs) != 1:
        raise AssertionError("route image changed across BS")
    if image_refs != {args.expected_image_ref}:
        raise AssertionError("route image differs from authority")

    result = {
        "schema": "step3p5.five-layer-moe-route-campaign.v1",
        "passed": passed,
        "context_len_per_sequence": CONTEXT_LEN,
        "sidecar_dir": str(sidecar_dir),
        "profile": args.expected_profile,
        "source_role": expected_role,
        "decode_fwd_sha256": expected_decode,
        "batches": records,
        "source_manifest_sha256": next(iter(source_manifests)),
        "image_ref": next(iter(image_refs)),
    }
    lines = [
        "# Five-layer MoE route sidecar campaign",
        "",
        "| BS | hidden exact | local counts | global routes/layer | result |",
        "|---:|:---:|:---:|---:|:---:|",
    ]
    for batch in BATCHES:
        record = records[str(batch)]
        checks = record["checks"]
        lines.append(
            f"| {batch} | {checks['hidden']} | {checks['local_count']} | "
            f"{record['expected_global_routes_per_layer']} | "
            f"{'PASS' if record['passed'] else 'FAIL'} |"
        )
    lines.extend(["", f"Campaign: {'PASS' if passed else 'FAIL'}", ""])
    _stable_write(
        out / "route_campaign_report.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    _stable_write(out / "route_campaign_report.md", "\n".join(lines))
    print(
        json.dumps(
            {
                "passed": passed,
                "report": str(out / "route_campaign_report.json"),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
