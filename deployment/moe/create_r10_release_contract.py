#!/usr/bin/env python3
"""Build a fail-closed immutable-image r10 release contract.

The generator is intentionally read-only with respect to campaign evidence.  It
only writes ``--out`` when every required gate is present and passes.  Missing
or inconsistent evidence is rendered to stdout and returns non-zero; use
``--write-fail`` only when a machine-readable blocked verdict is desired.

The expected r10 identity is kept here as a guard against accidentally
combining evidence from another image or campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "step3p5.r10-release-admission.v2"
R10_TAG = "hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260825-r10"
R10_REPOSITORY = "hub.i.basemind.com/stepcast/vllm-pypto"
R10_MANIFEST = (
    "sha256:"
    "8510f30e1f2a2f2edcaa834c831165b349a4aca1212b655ca2a02ed6b3e9907b"
)
R10_CONFIG = (
    "sha256:"
    "38ebba41d6aa0c49940c03e2e7c6fa42d85b61d631c143d38944683d0c657b5f"
)
R10_IMAGE = f"{R10_REPOSITORY}@{R10_MANIFEST}"
R9_MANIFEST = (
    "sha256:"
    "b637f00c66d4dc976c053c617d2e19e6d6d66f68f4bef30250984da7a71690f6"
)
R9_CONFIG = (
    "sha256:"
    "f6c8f72eecad0a9d40d0c4ea55afaab09dd4e2f5fe54d6a091e332465e421dae"
)
R9_IMAGE = f"{R10_REPOSITORY}@{R9_MANIFEST}"
R10_PYPTO_LIB = "fe641929dbf959d887ad111f3bd7cac0b73fa34b"
R10_PYPTO_LIB_TREE = "5d8f7e647cab301ee5bb2f0175fec4d91bfa71e8"
R10_PINS = {
    "pypto": "519b588a7a6461cac0e443e853accf29479c1d15",
    "pypto-lib": R10_PYPTO_LIB,
    "pto-isa": "cd4a3d3f7a1a27fcfe536f617e9bca3008929664",
    "PTOAS": "307d0484a9e7d5e36f01b253d2bebe4d2f45fe81",
    "simpler": "85a82c454074c069315ed6485033c3c2b136e562",
    "ptoas-bin": "v0.57",
}
REPO_ORDER = ["pto-isa", "PTOAS", "simpler", "pypto", "pypto-lib"]
BATCHES = [1, 2, 4, 7, 8, 16]
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _path(value: str | Path) -> Path:
    """Accept the documentation's ``0162:/absolute/path`` spelling."""
    text = str(value)
    if re.match(r"^[^/:\s]+:/", text):
        text = text.split(":", 1)[1]
    return Path(text).expanduser().resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"{label}: missing file: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"{label}: invalid JSON {path}: {error}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label}: expected object: {path}")
        return {}
    return value


def _kv(path: Path, errors: list[str], label: str) -> dict[str, str]:
    if not path.is_file():
        errors.append(f"{label}: missing contract: {path}")
        return {}
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        errors.append(f"{label}: cannot read {path}: {error}")
        return {}
    for line in lines:
        if "=" not in line or not line.strip():
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _env(path: Path, errors: list[str], label: str) -> dict[str, str]:
    """Parse simple KEY=value build specs without executing shell code."""
    values: dict[str, str] = {}
    if not path.is_file():
        errors.append(f"{label}: missing env file: {path}")
        return values
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        errors.append(f"{label}: cannot read {path}: {error}")
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _int(value: Any) -> int | None:
    try:
        if isinstance(value, str):
            match = re.match(r"^[+-]?\d+", value.strip())
            if match:
                return int(match.group(0))
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool(value: Any) -> bool:
    return value is True


def _manifest_from_ref(value: Any) -> str | None:
    if not isinstance(value, str) or "@sha256:" not in value:
        return None
    return "sha256:" + value.rsplit("@sha256:", 1)[1]


def _config_from_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if value.startswith("sha256:"):
        return value
    if HEX64.fullmatch(value):
        return f"sha256:{value}"
    return None


def _find_one(
    directory: Path,
    names: tuple[str, ...],
    patterns: tuple[str, ...],
    errors: list[str],
    label: str,
) -> Path | None:
    if not directory.is_dir():
        errors.append(f"{label}: missing directory: {directory}")
        return None
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    candidates: set[Path] = set()
    for pattern in patterns:
        candidates.update(
            path for path in directory.rglob(pattern) if path.is_file()
        )
    if len(candidates) == 1:
        return next(iter(candidates))
    if not candidates:
        errors.append(
            f"{label}: no matching evidence under {directory}; "
            f"names={names}, patterns={patterns}"
        )
    else:
        errors.append(
            f"{label}: ambiguous evidence under {directory}: "
            + ", ".join(str(path) for path in sorted(candidates))
        )
    return None


def _find_without_errors(
    directory: Path,
    names: tuple[str, ...],
    patterns: tuple[str, ...],
) -> Path | None:
    if not directory.is_dir():
        return None
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    candidates: set[Path] = set()
    for pattern in patterns:
        candidates.update(
            path for path in directory.rglob(pattern) if path.is_file()
        )
    return next(iter(candidates)) if len(candidates) == 1 else None


def _marker_dir(
    root: Path,
    explicit: str,
    marker: str,
    fallback: tuple[str, ...],
    errors: list[str],
    label: str,
) -> Path | None:
    if explicit:
        path = _path(explicit)
        if not path.is_dir():
            errors.append(f"{label}: missing directory: {path}")
            return None
        return path
    marker_path = root / marker
    if marker_path.is_file():
        try:
            path = _path(marker_path.read_text(encoding="utf-8").strip())
        except OSError as error:
            errors.append(f"{label}: cannot read marker {marker_path}: {error}")
            return None
        if path.is_dir():
            return path
        errors.append(f"{label}: marker points to missing directory: {path}")
        return None
    candidates = sorted(
        path
        for pattern in fallback
        for path in root.glob(pattern)
        if path.is_dir()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        errors.append(
            f"{label}: no directory supplied and no marker/candidate found"
        )
    else:
        errors.append(
            f"{label}: multiple candidates; pass an explicit directory: "
            + ", ".join(str(path) for path in candidates)
        )
    return None


def _manifest_info(
    path: Path | None, errors: list[str], label: str
) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        errors.append(f"{label}: missing artifact manifest: {path}")
        return None
    entries = 0
    malformed: list[int] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        errors.append(f"{label}: cannot read artifact manifest: {error}")
        return None
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not HEX64.fullmatch(parts[0]):
            malformed.append(line_number)
        else:
            entries += 1
    if malformed:
        errors.append(
            f"{label}: malformed SHA256 manifest lines {malformed}: {path}"
        )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "entries": entries,
        "malformed_lines": malformed,
    }


def _json_manifest_info(
    path: Path | None, errors: list[str], label: str
) -> dict[str, Any] | None:
    """Record a JSON evidence manifest (used by the six-BS runner)."""
    if path is None:
        return None
    value = _json(path, errors, label)
    if not value:
        return None
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "entries": len(value),
        "kind": "json",
    }


def _artifact(
    root: Path, path: Path | None, errors: list[str], label: str
) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        errors.append(f"{label}: missing artifact: {path}")
        return None
    try:
        relative = str(path.relative_to(root))
    except ValueError:
        relative = str(path)
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "path": relative,
    }


def _all_true(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(
        item is True for item in value.values()
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a fail-closed r10 release_contract.json"
    )
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--route-dir", required=True)
    parser.add_argument("--outer-dir", required=True)
    parser.add_argument("--bs-dir", required=True)
    parser.add_argument("--git-sync-dir", required=True)
    parser.add_argument("--aba-dir", required=True)
    parser.add_argument("--compile-dir")
    parser.add_argument("--liveness-dir")
    parser.add_argument("--precision-verdict")
    parser.add_argument("--itl-dir")
    parser.add_argument("--publication-dir")
    parser.add_argument("--image-audit-dir")
    parser.add_argument("--source-unit-dir")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--write-fail",
        action="store_true",
        help="write a blocked verdict to --out when validation fails",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = _path(args.campaign_root)
    errors: list[str] = []
    checks: dict[str, bool] = {}
    if not root.is_dir():
        errors.append(f"campaign-root: missing directory: {root}")

    def require(name: str, condition: bool, detail: str = "") -> None:
        checks[name] = bool(condition)
        if not condition:
            errors.append(f"{name}: {detail or 'check failed'}")

    route_dir = _path(args.route_dir)
    outer_dir = _path(args.outer_dir)
    bs_dir = _path(args.bs_dir)
    sync_dir = _path(args.git_sync_dir)
    aba_dir = _path(args.aba_dir)
    compile_dir = _marker_dir(
        root,
        args.compile_dir or "",
        ".latest-compile",
        ("compile-r10-*",),
        errors,
        "compile",
    )
    liveness_dir = _marker_dir(
        root,
        args.liveness_dir or "",
        ".r10-liveness-latest",
        ("liveness-r10-*",),
        errors,
        "liveness",
    )
    itl_dir = _marker_dir(
        root,
        args.itl_dir or "",
        ".latest-itl",
        ("itl-*",),
        errors,
        "itl",
    )
    publication_dir = _marker_dir(
        root,
        args.publication_dir or "",
        ".latest-publication",
        ("publication-*",),
        errors,
        "publication",
    )
    image_audit_dir = _path(args.image_audit_dir) if args.image_audit_dir else root / "image-audit"
    source_unit_dir = _marker_dir(
        root,
        args.source_unit_dir or "",
        ".latest-source-unit-gate",
        ("source-gates/*",),
        errors,
        "source-unit",
    )

    source_unit_contract: dict[str, str] = {}
    source_unit_log: Path | None = None
    if source_unit_dir:
        source_unit_contract = _kv(
            source_unit_dir / "run_contract.txt",
            errors,
            "source-unit contract",
        )
        source_unit_log = source_unit_dir / "unit.log"
        unit_text = (
            source_unit_log.read_text(encoding="utf-8")
            if source_unit_log.is_file()
            else ""
        )
        require(
            "source_unit.rc",
            _int(source_unit_contract.get("rc")) == 0,
            "source-unit rc is not zero",
        )
        require(
            "source_unit.tests",
            re.search(r"\b162 passed\b", unit_text) is not None,
            "source-unit log does not report 162 passed",
        )

    identity_path = (
        image_audit_dir / "identity.txt"
        if image_audit_dir.is_dir()
        else None
    )
    pins_path = (
        image_audit_dir / "src-pins.shas"
        if image_audit_dir.is_dir()
        else None
    )
    identity = _kv(identity_path, errors, "image identity") if identity_path else {}
    pin_text = (
        pins_path.read_text(encoding="utf-8") if pins_path and pins_path.is_file() else ""
    )
    if pins_path and not pins_path.is_file():
        errors.append(f"image pins: missing file: {pins_path}")
    parsed_pins: dict[str, str] = {}
    for field, value in re.findall(
        r"([A-Za-z0-9_-]+)=([0-9A-Za-zA-Z._-]+)", pin_text
    ):
        parsed_pins[field] = value.rstrip("\r\n")
    build_spec_path = _find_without_errors(
        root / "image-build" / "builds",
        ("stepfun-upgrade-20260825-r10.env",),
        ("*r10*.env",),
    )
    if build_spec_path is None:
        # Local dry-run/audit checkout keeps the same spec under deployment/docker.
        build_spec_path = _find_without_errors(
            Path.cwd() / "deployment" / "docker" / "builds",
            ("stepfun-upgrade-20260825-r10.env",),
            ("*r10*.env",),
        )
    build_spec = _env(
        build_spec_path,
        errors,
        "image build spec",
    ) if build_spec_path else {}
    if not build_spec_path:
        errors.append("image build spec: r10 env file not found")
    parsed_pins.setdefault("ptoas-bin", build_spec.get("PTOAS_BIN_VER", ""))
    require("image.tag", identity.get("image_tag") == R10_TAG, "r10 tag mismatch")
    require("image.ref", identity.get("image_ref") == R10_IMAGE, "r10 manifest ref mismatch")
    require("image.manifest", identity.get("manifest") == R10_MANIFEST, "manifest mismatch")
    require("image.config", identity.get("config") == R10_CONFIG, "config mismatch")
    require(
        "image.pypto_lib",
        identity.get("pypto_lib_commit") == R10_PYPTO_LIB,
        "pypto-lib commit mismatch",
    )
    for name, expected in R10_PINS.items():
        require(
            f"pins.{name}",
            parsed_pins.get(name) == expected,
            f"expected {expected!r}, got {parsed_pins.get(name)!r}",
        )
    require(
        "pins.ptoas-bin.sha256",
        build_spec.get("PTOAS_BIN_SHA256")
        == "2183e4cf00fd019403825290233c32b84c1b9904474ec4614ab976dac143aaae",
        "ptoas binary SHA mismatch",
    )

    publication_verdict = (
        _json(
            publication_dir / "remote_registry_verdict.json",
            errors,
            "publication remote verdict",
        )
        if publication_dir
        else {}
    )
    require(
        "publication.pass",
        publication_verdict.get("pass") is True,
        "remote registry verdict did not pass",
    )
    require(
        "publication.manifest",
        publication_verdict.get("actual_manifest") == R10_MANIFEST
        and publication_verdict.get("expected_manifest") == R10_MANIFEST,
        "remote manifest mismatch",
    )
    require(
        "publication.config",
        publication_verdict.get("actual_config") == R10_CONFIG
        and publication_verdict.get("expected_config") == R10_CONFIG,
        "remote config mismatch",
    )
    publication_artifact_manifest = _manifest_info(
        publication_dir / "remote_registry_artifacts.sha256"
        if publication_dir
        else None,
        errors,
        "publication artifacts",
    )
    require(
        "publication.artifacts",
        publication_artifact_manifest is not None
        and not publication_artifact_manifest["malformed_lines"],
        "publication artifact manifest missing or malformed",
    )
    image_input_manifest = _manifest_info(
        image_audit_dir / "build_inputs.sha256",
        errors,
        "image build-input artifacts",
    )
    require(
        "image.audit",
        (image_audit_dir / "static_and_smoke.rc").is_file()
        and _int((image_audit_dir / "static_and_smoke.rc").read_text().strip())
        == 0
        and (image_audit_dir / "static_and_smoke.log").is_file()
        and "R10_STATIC_EXTENSION_AUDIT=PASS"
        in (image_audit_dir / "static_and_smoke.log").read_text(
            encoding="utf-8"
        ),
        "image static/smoke audit did not pass",
    )

    compile_contract: dict[str, str] = {}
    compile_rc = None
    compile_artifact_manifest = None
    if compile_dir:
        compile_contract = _kv(
            compile_dir / "run_contract.txt", errors, "compile contract"
        )
        compile_rc = _int(compile_contract.get("compile_rc"))
        compile_rc_path = compile_dir / "compile.rc"
        if compile_rc_path.is_file():
            compile_rc_values = [
                _int(line.strip())
                for line in compile_rc_path.read_text().splitlines()
                if line.strip()
            ]
            require(
                "compile.rc",
                bool(compile_rc_values) and all(value == 0 for value in compile_rc_values),
                "compile.rc is not zero",
            )
        else:
            errors.append(f"compile.rc: missing file: {compile_rc_path}")
        compile_artifact_manifest = _manifest_info(
            compile_dir / "artifacts.sha256", errors, "compile artifacts"
        )
        require(
            "compile.image",
            compile_contract.get("image_ref") == R10_IMAGE,
            "compile image ref mismatch",
        )
        require(
            "compile.config",
            compile_contract.get("config") == R10_CONFIG,
            "compile config mismatch",
        )
        require(
            "compile.rc_contract",
            compile_rc == 0,
            f"compile_rc={compile_contract.get('compile_rc')!r}",
        )
        require(
            "compile.no_overlay",
            compile_contract.get("source_overlay") == "false"
            and compile_contract.get("runtime_overlay") == "false",
            "source/runtime overlay was enabled",
        )
        require(
            "compile.artifacts",
            compile_artifact_manifest is not None
            and not compile_artifact_manifest["malformed_lines"],
            "compile artifact manifest missing or malformed",
        )

    liveness_contract: dict[str, str] = {}
    liveness_report: dict[str, Any] = {}
    liveness_artifact_manifest = None
    if liveness_dir:
        liveness_contract = _kv(
            liveness_dir / "run_contract.txt", errors, "liveness contract"
        )
        liveness_report = _json(
            liveness_dir / "whole_network_report.json",
            errors,
            "liveness report",
        )
        stages = liveness_report.get("stages")
        stage_pass = (
            isinstance(stages, list)
            and len(stages) >= 3
            and all(
                isinstance(stage, dict)
                and stage.get("passed") is True
                and stage.get("returncode") == 0
                and stage.get("timed_out") is False
                for stage in stages
            )
        )
        liveness_manifest_path = liveness_dir / "artifacts.sha256"
        liveness_artifact_manifest = (
            _manifest_info(liveness_manifest_path, errors, "liveness artifacts")
            if liveness_manifest_path.is_file()
            else None
        )
        require(
            "liveness.ok",
            liveness_report.get("ok") is True and stage_pass,
            "whole-network liveness report is not fully green",
        )
        liveness_rc_path = liveness_dir / "container.rc"
        liveness_rc_values = (
            [
                _int(line.strip())
                for line in liveness_rc_path.read_text().splitlines()
                if line.strip()
            ]
            if liveness_rc_path.is_file()
            else []
        )
        require(
            "liveness.rc",
            _int(liveness_contract.get("rc")) == 0
            and bool(liveness_rc_values)
            and all(value == 0 for value in liveness_rc_values),
            "liveness rc is not zero",
        )
        require(
            "liveness.image",
            liveness_contract.get("image_ref") == R10_IMAGE
            and liveness_contract.get("image_runtime") == R10_IMAGE,
            "liveness image ref mismatch",
        )
        require(
            "liveness.config",
            liveness_contract.get("config_ref") == R10_CONFIG,
            "liveness config mismatch",
        )
        require(
            "liveness.h4",
            liveness_contract.get("h4_resident") == "all",
            "liveness was not run with H4 resident=all",
        )
        source = liveness_report.get("source") or {}
        require(
            "liveness.source",
            source.get("sha") == R10_PYPTO_LIB
            and source.get("tree") == R10_PYPTO_LIB_TREE
            and source.get("dirty") is False,
            "liveness source identity mismatch",
        )
        require(
            "liveness.artifacts",
            liveness_artifact_manifest is None
            or not liveness_artifact_manifest["malformed_lines"],
            "liveness artifact manifest malformed",
        )

    precision_path = (
        _path(args.precision_verdict)
        if args.precision_verdict
        else root / "precision-r10-h4-all-none-parity-verdict.json"
    )
    precision = _json(precision_path, errors, "precision parity")
    precision_checks = precision.get("checks")
    precision_summary = precision.get("summary") or {}
    require("precision.pass", precision.get("pass") is True, "precision verdict failed")
    require(
        "precision.checks",
        _all_true(precision_checks),
        "precision contains a false or missing check",
    )
    require(
        "precision.image",
        (precision.get("inputs") or {}).get("image_ref") == R10_IMAGE
        and (precision.get("inputs") or {}).get("image_runtime") == R10_IMAGE,
        "precision image ref mismatch",
    )
    require(
        "precision.config",
        (precision.get("inputs") or {}).get("config_ref") == R10_CONFIG,
        "precision config mismatch",
    )
    require(
        "precision.strict_counts",
        precision_summary.get("output_token_pairs_exact") == 128
        and precision_summary.get("tensor_pair_count") == 256
        and precision_summary.get("tensor_pairs_torch_equal") == 256
        and precision_summary.get("tensor_pairs_file_byte_equal") == 256
        and precision_summary.get("tensor_files_finite") == 512,
        "strict parity counts are incomplete",
    )
    precision_artifacts = _find_one(
        root,
        ("precision-r10-h4-all-none-parity-artifacts.sha256",),
        ("*parity-artifacts.sha256",),
        errors,
        "precision artifacts",
    )
    precision_artifact_manifest = _manifest_info(
        precision_artifacts, errors, "precision artifacts"
    )

    itl_admission: dict[str, Any] = {}
    itl_contract: dict[str, str] = {}
    itl_artifact_manifest = None
    if itl_dir:
        itl_admission = _json(
            itl_dir / "itl_admission.json", errors, "ITL admission"
        )
        itl_contract = _kv(
            itl_dir / "run_contract.txt", errors, "ITL contract"
        )
        itl_manifest_path = itl_dir / "artifacts.sha256"
        itl_artifact_manifest = (
            _manifest_info(itl_manifest_path, errors, "ITL artifacts")
            if itl_manifest_path.is_file()
            else None
        )
        long_result = itl_admission.get("long_64k_1000") or {}
        curve = itl_admission.get("curve")
        require(
            "itl.pass",
            itl_admission.get("pass") is True
            and itl_admission.get("performance_pass") is True
            and itl_admission.get("structural_pass") is True,
            "ITL admission did not pass",
        )
        require(
            "itl.image",
            itl_admission.get("image_manifest") == R10_IMAGE
            and itl_contract.get("bound_image_runtime") == R10_IMAGE,
            "ITL image ref mismatch",
        )
        require(
            "itl.config",
            itl_admission.get("image_config") == R10_CONFIG
            and itl_contract.get("bound_config") == R10_CONFIG,
            "ITL config mismatch",
        )
        require(
            "itl.h4",
            itl_admission.get("h4_resident") == "all",
            "ITL was not run with H4 resident=all",
        )
        require(
            "itl.long",
            long_result.get("context_len") == 65536
            and long_result.get("iters") == 1000
            and _float(long_result.get("itl_ms_p50")) is not None,
            "64K/1000 ITL result missing",
        )
        require(
            "itl.curve",
            isinstance(curve, list)
            and {
                row.get("context_len")
                for row in curve
                if isinstance(row, dict)
            }
            == {1024, 8192, 32768, 65536},
            "context curve is incomplete",
        )
        require(
            "itl.artifacts",
            itl_artifact_manifest is None
            or not itl_artifact_manifest["malformed_lines"],
            "ITL artifact manifest malformed",
        )

    route_gate_path = _find_one(
        route_dir,
        ("route_gate.json",),
        ("*route*gate*.json",),
        errors,
        "route gate",
    )
    route_gate = _json(route_gate_path, errors, "route gate") if route_gate_path else {}
    route_validation_path = route_dir / "route-runtime" / "route_artifact_validation.json"
    route_validation = _json(route_validation_path, errors, "route validation")
    route_contract = _kv(route_dir / "run_contract.txt", errors, "route contract")
    route_artifact_manifest = _manifest_info(
        route_dir / "artifacts.sha256", errors, "route artifacts"
    )
    route_checks = route_gate.get("checks")
    require("route.pass", route_gate.get("pass") is True, "route gate failed")
    require("route.checks", _all_true(route_checks), "route gate has a false check")
    require(
        "route.image",
        route_gate.get("image_ref") == R10_IMAGE
        and route_contract.get("image_ref") == R10_IMAGE
        and route_validation.get("image_ref") == R10_IMAGE,
        "route image ref mismatch",
    )
    require(
        "route.config",
        route_contract.get("config") == R10_CONFIG,
        "route config mismatch",
    )
    require(
        "route.validation",
        route_validation.get("passed") is True
        and route_validation.get("profile") == "packed-nz"
        and route_validation.get("hidden_bit_exact") is True
        and route_validation.get("local_count_exact") is True
        and route_validation.get("padding_zero") is True,
        "route validation is incomplete",
    )
    require(
        "route.artifacts",
        route_artifact_manifest is not None
        and not route_artifact_manifest["malformed_lines"],
        "route artifact manifest missing or malformed",
    )

    outer_gate_path = _find_one(
        outer_dir,
        ("outer_admission.json",),
        ("*outer*admission*.json",),
        errors,
        "outer gate",
    )
    outer_gate = _json(outer_gate_path, errors, "outer gate") if outer_gate_path else {}
    outer_contract = _json(outer_dir / "run_contract.json", errors, "outer contract")
    outer_artifact_manifest = _manifest_info(
        outer_dir / "artifacts.sha256", errors, "outer artifacts"
    )
    require("outer.pass", outer_gate.get("pass") is True, "outer gate failed")
    require(
        "outer.image",
        outer_gate.get("image_ref") == R10_IMAGE
        and outer_contract.get("image_ref") == R10_IMAGE,
        "outer image ref mismatch",
    )
    require(
        "outer.config",
        outer_contract.get("config_ref") == R10_CONFIG,
        "outer config mismatch",
    )
    require(
        "outer.counts",
        outer_gate.get("chip_swimlane_records") == 8
        and outer_gate.get("deps_records") == 8
        and outer_gate.get("name_maps") == 8
        and outer_gate.get("critical_path_reports") == 8
        and outer_gate.get("merged_swimlanes") == 8
        and outer_gate.get("dfx_protocol_rank_count") == 8,
        "outer 8-rank evidence is incomplete",
    )
    require(
        "outer.hidden",
        outer_gate.get("hidden_l3_exact") is True
        and outer_gate.get("hidden_l4_exact") is True
        and outer_gate.get("hidden_l3_sha_matches_golden") is True
        and outer_gate.get("hidden_l4_sha_matches_golden") is True,
        "outer hidden exactness failed",
    )
    require(
        "outer.dfx",
        outer_gate.get("analyzer_pass") is True
        and outer_gate.get("analyzer_gate_pass") is True
        and outer_gate.get("analyzer_blockers") == []
        and outer_gate.get("resource_grid_exact") is True,
        "outer DFX analyzer/resource gate failed",
    )
    require(
        "outer.artifacts",
        outer_artifact_manifest is not None
        and not outer_artifact_manifest["malformed_lines"],
        "outer artifact manifest missing or malformed",
    )

    bs_verdict_path = _find_one(
        bs_dir,
        ("six_batch_r9_r10_verdict.json", "bs_admission.json"),
        ("*six*batch*verdict*.json", "*bs*admission*.json"),
        errors,
        "six-BS verdict",
    )
    bs_verdict = _json(bs_verdict_path, errors, "six-BS verdict") if bs_verdict_path else {}
    bs_summary = bs_verdict.get("summary") or {}
    bs_workload = bs_verdict.get("workload") or {}
    bs_cases = bs_verdict.get("cases")
    require("bs.pass", bs_verdict.get("pass") is True, "six-BS verdict failed")
    require(
        "bs.counts",
        bs_summary.get("exact_batches") == 6
        and bs_summary.get("healthy_arm_batches") == 12
        and isinstance(bs_cases, list)
        and len(bs_cases) == 6
        and {case.get("active_batch") for case in bs_cases if isinstance(case, dict)}
        == set(BATCHES),
        "six-BS matrix is incomplete",
    )
    require(
        "bs.workload",
        bs_workload.get("active_batches") == BATCHES
        or bs_verdict.get("active_batches") == BATCHES,
        "six-BS active batch contract mismatch",
    )
    require(
        "bs.case_pass",
        isinstance(bs_cases, list)
        and all(case.get("passed") is True for case in bs_cases if isinstance(case, dict)),
        "one or more six-BS cases failed",
    )
    bs_sha_manifest_path = _find_without_errors(
        bs_dir,
        ("artifacts.sha256",),
        ("*artifacts.sha256",),
    )
    bs_artifact_manifest = _manifest_info(
        bs_sha_manifest_path,
        errors,
        "six-BS artifacts",
    )
    if bs_artifact_manifest is None:
        bs_json_manifest_path = _find_one(
            bs_dir,
            ("evidence_manifest.json",),
            ("*evidence*manifest*.json",),
            errors,
            "six-BS evidence manifest",
        )
        bs_artifact_manifest = _json_manifest_info(
            bs_json_manifest_path,
            errors,
            "six-BS evidence manifest",
        )
        bs_seal_path = _find_one(
            bs_dir,
            ("evidence_seal.sha256",),
            ("*evidence*seal*.sha256",),
            errors,
            "six-BS evidence seal",
        )
        if bs_seal_path is None:
            errors.append("six-BS evidence seal: missing")
        else:
            seal = bs_seal_path.read_text(encoding="utf-8").strip()
            if not HEX64.fullmatch(seal):
                errors.append(f"six-BS evidence seal: malformed: {bs_seal_path}")
    require(
        "bs.artifacts",
        bs_artifact_manifest is not None,
        "six-BS artifact manifest/evidence seal missing",
    )

    aba_path = _find_one(
        aba_dir,
        ("aba_admission.json",),
        ("*aba*admission*.json",),
        errors,
        "immutable A/B/A verdict",
    )
    aba = _json(aba_path, errors, "immutable A/B/A verdict") if aba_path else {}
    arms = aba.get("arms") or {}
    require(
        "aba.schema",
        aba.get("schema") == "step3p5.r10-immutable-image-aba-admission.v1",
        "unexpected A/B/A schema",
    )
    require("aba.pass", aba.get("pass") is True, "immutable A/B/A failed")
    require(
        "aba.checks",
        _all_true(aba.get("checks")),
        "A/B/A has a false or missing check",
    )
    require(
        "aba.arms",
        set(arms) == {"A1", "B", "A2"}
        and all(isinstance(arms[name], dict) for name in arms)
        and arms["A1"].get("image_manifest") in {R9_IMAGE, R9_MANIFEST}
        and arms["A1"].get("image_config") == R9_CONFIG
        and arms["B"].get("image_manifest") in {R10_IMAGE, R10_MANIFEST}
        and arms["B"].get("image_config") == R10_CONFIG
        and arms["A2"].get("image_manifest") in {R9_IMAGE, R9_MANIFEST}
        and arms["A2"].get("image_config") == R9_CONFIG,
        "A/B/A arm image identities are incomplete",
    )
    require(
        "aba.metrics",
        _float(aba.get("baseline_midpoint_p50_ms")) is not None
        and _float(aba.get("baseline_bracket_p50_ms")) is not None
        and _float(aba.get("candidate_minus_midpoint_p50_ms")) is not None
        and _float(aba.get("candidate_minus_midpoint_p50_pct")) is not None,
        "A/B/A comparison metrics are missing",
    )
    aba_artifact_manifest = _manifest_info(
        _find_one(
            aba_dir,
            ("artifacts.sha256", "evidence_manifest.sha256"),
            ("*artifacts.sha256", "*evidence*manifest*.sha256"),
            errors,
            "A/B/A artifacts",
        ),
        errors,
        "A/B/A artifacts",
    )

    sync_verdict_path = _find_one(
        sync_dir,
        ("verdict.json",),
        ("*verdict*.json",),
        errors,
        "develop sync verdict",
    )
    sync = _json(sync_verdict_path, errors, "develop sync verdict") if sync_verdict_path else {}
    repositories = sync.get("repositories")
    if isinstance(repositories, dict):
        rows = repositories.get("repositories")
        branch = repositories.get("branch")
        sync_pass = repositories.get("pass")
    else:
        rows = repositories
        branch = sync.get("branch")
        sync_pass = sync.get("pass")
    rows = rows if isinstance(rows, list) else []
    expected_sha = {
        "pto-isa": R10_PINS["pto-isa"],
        "PTOAS": R10_PINS["PTOAS"],
        "simpler": R10_PINS["simpler"],
        "pypto": R10_PINS["pypto"],
        "pypto-lib": R10_PYPTO_LIB,
    }
    sync_rows_ok = (
        [row.get("repository") for row in rows] == REPO_ORDER
        and all(
            isinstance(row, dict)
            and row.get("new_stepfun_develop") == row.get("verified_remote_sha")
            and row.get("new_stepfun_develop") == expected_sha.get(row.get("repository"))
            for row in rows
        )
    )
    require(
        "sync.schema",
        sync.get("schema") == "step3p5.r10-five-repo-sync.v1"
        and sync.get("mode") == "run"
        and sync.get("executed") is True,
        "preflight or unexpected sync verdict cannot admit a release",
    )
    require(
        "sync.pass",
        sync_pass is True and sync.get("pass") is True,
        "sync verdict failed",
    )
    require("sync.branch", branch == "refs/heads/stepfun/develop", "wrong sync branch")
    require("sync.rows", sync_rows_ok, "develop rows do not match r10 pins")
    sync_artifact_manifest = _manifest_info(
        _find_one(
            sync_dir,
            ("artifacts.sha256",),
            ("*artifacts.sha256",),
            errors,
            "develop sync artifacts",
        ),
        errors,
        "develop sync artifacts",
    )

    artifact_paths: list[tuple[str, Path | None, str]] = [
        ("image.identity", identity_path, "image identity"),
        ("image.pins", pins_path, "image pins"),
        ("image.build_spec", build_spec_path, "image build spec"),
        (
            "image.build_inputs",
            image_audit_dir / "build_inputs.sha256",
            "image build-input artifacts",
        ),
        ("publication.remote_verdict", publication_dir / "remote_registry_verdict.json" if publication_dir else None, "publication"),
        (
            "publication.artifacts",
            publication_dir / "remote_registry_artifacts.sha256"
            if publication_dir
            else None,
            "publication artifacts",
        ),
        ("source_unit.contract", source_unit_dir / "run_contract.txt" if source_unit_dir else None, "source-unit"),
        ("source_unit.log", source_unit_log, "source-unit"),
        ("compile.contract", compile_dir / "run_contract.txt" if compile_dir else None, "compile"),
        ("compile.rc", compile_dir / "compile.rc" if compile_dir else None, "compile"),
        ("liveness.contract", liveness_dir / "run_contract.txt" if liveness_dir else None, "liveness"),
        ("liveness.report", liveness_dir / "whole_network_report.json" if liveness_dir else None, "liveness"),
        ("precision.verdict", precision_path, "precision"),
        ("itl.admission", itl_dir / "itl_admission.json" if itl_dir else None, "itl"),
        ("itl.contract", itl_dir / "run_contract.txt" if itl_dir else None, "itl"),
        ("route.gate", route_gate_path, "route"),
        ("route.validation", route_validation_path, "route"),
        ("route.contract", route_dir / "run_contract.txt", "route"),
        ("outer.gate", outer_gate_path, "outer"),
        ("outer.contract", outer_dir / "run_contract.json", "outer"),
        ("bs.verdict", bs_verdict_path, "six-BS"),
        ("aba.verdict", aba_path, "A/B/A"),
        ("sync.verdict", sync_verdict_path, "develop sync"),
    ]
    artifacts: dict[str, Any] = {}
    for key, path, label in artifact_paths:
        item = _artifact(root, path, errors, label)
        if item is not None:
            artifacts[key] = item

    artifact_manifests = {
        "image_build_inputs": image_input_manifest,
        "publication": publication_artifact_manifest,
        "compile": compile_artifact_manifest,
        "liveness": liveness_artifact_manifest,
        "precision": precision_artifact_manifest,
        "itl": itl_artifact_manifest,
        "route": route_artifact_manifest,
        "outer": outer_artifact_manifest,
        "six_bs": bs_artifact_manifest,
        "aba": aba_artifact_manifest,
        "develop_sync": sync_artifact_manifest,
    }

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "release-admitted" if not errors else "blocked",
        "pass": not errors,
        "created": datetime.now(timezone.utc).isoformat(),
        "host": socket.getfqdn(),
        "campaign_root": str(root),
        "checks": checks,
        "errors": errors,
        "image": {
            "tag": R10_TAG,
            "manifest": R10_MANIFEST,
            "config": R10_CONFIG,
            "registry_verified": publication_verdict.get("pass") is True,
            "base_manifest": (
                R9_MANIFEST
            ),
        },
        "pins": R10_PINS,
        "source_unit": {
            "contract": source_unit_contract,
            "passed": checks.get("source_unit.rc", False)
            and checks.get("source_unit.tests", False),
        },
        "compile": {
            "contract": compile_contract,
            "artifact_manifest": compile_artifact_manifest,
        },
        "liveness": {
            "contract": liveness_contract,
            "report": {
                "ok": liveness_report.get("ok"),
                "source": liveness_report.get("source"),
                "stages": [
                    {
                        "name": stage.get("name"),
                        "passed": stage.get("passed"),
                        "returncode": stage.get("returncode"),
                        "timed_out": stage.get("timed_out"),
                    }
                    for stage in (liveness_report.get("stages") or [])
                    if isinstance(stage, dict)
                ],
            },
            "artifact_manifest": liveness_artifact_manifest,
        },
        "precision": {
            "schema": precision.get("schema"),
            "pass": precision.get("pass"),
            "summary": precision_summary,
            "artifact_manifest": precision_artifact_manifest,
        },
        "performance": {
            "itl": {
                "schema": itl_admission.get("schema"),
                "pass": itl_admission.get("pass"),
                "h4_resident": itl_admission.get("h4_resident"),
                "long_64k_1000": itl_admission.get("long_64k_1000"),
                "curve": itl_admission.get("curve"),
                "delta_vs_baseline_midpoint_ms": itl_admission.get(
                    "delta_vs_baseline_midpoint_ms"
                ),
                "delta_vs_r9_published_ms": itl_admission.get(
                    "delta_vs_r9_published_ms"
                ),
                "artifact_manifest": itl_artifact_manifest,
            },
            "immutable_aba": aba,
        },
        "route": {
            "gate": route_gate,
            "validation": route_validation,
            "artifact_manifest": route_artifact_manifest,
        },
        "outer_hidden_and_swimlane": {
            "gate": outer_gate,
            "artifact_manifest": outer_artifact_manifest,
        },
        "six_batch": {
            "verdict": bs_verdict,
            "artifact_manifest": bs_artifact_manifest,
        },
        "repositories": sync,
        "artifacts": artifacts,
        "artifact_manifests": artifact_manifests,
    }
    output = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(output, end="")
    if errors:
        if args.write_fail:
            out = _path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(output, encoding="utf-8")
        return 1
    out = _path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(output, encoding="utf-8")
    print(f"R10_RELEASE_CONTRACT_PASS out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
