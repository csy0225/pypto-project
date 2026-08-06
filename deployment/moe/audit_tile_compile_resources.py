#!/usr/bin/env python3
"""Fail-closed resource audit for five-layer MoE tile candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any


SPACE_LIMITS = {
    "Mat": 524288,
    "Left": 65536,
    "Right": 65536,
    "Acc": 131072,
    "Vec": 188416,
}
REQUIRED_FUNCTIONS = {
    "expert_gate_mm_aic": ("Mat", "Left", "Right", "Acc"),
    "expert_gate_mm_aiv": ("Vec",),
    "expert_up_mm_aic": ("Mat", "Left", "Right", "Acc"),
    "expert_up_mm_aiv": ("Vec",),
    "expert_gate_up_act": ("Vec",),
    "routed_h_quant": ("Vec",),
    "expert_down_aic": ("Mat", "Left", "Right", "Acc"),
    "expert_down_aiv": ("Vec",),
    "swa_moe_chip_orch_expert_gate_mm_aic": (
        "Mat",
        "Left",
        "Right",
        "Acc",
    ),
    "swa_moe_chip_orch_expert_gate_mm_aiv": ("Vec",),
    "swa_moe_chip_orch_expert_up_mm_aic": (
        "Mat",
        "Left",
        "Right",
        "Acc",
    ),
    "swa_moe_chip_orch_expert_up_mm_aiv": ("Vec",),
    "swa_moe_chip_orch_expert_gate_up_act": ("Vec",),
    "swa_moe_chip_orch_routed_h_quant": ("Vec",),
    "swa_moe_chip_orch_expert_down_aic": (
        "Mat",
        "Left",
        "Right",
        "Acc",
    ),
    "swa_moe_chip_orch_expert_down_aiv": ("Vec",),
}
VARIANT_RE = re.compile(r"^mm-n(32|64)-r(8|16|32)$")
SECTION_RE = re.compile(r"^--- ([A-Za-z0-9_]+) ---$")
SUMMARY_RE = re.compile(
    r"^\s*(Mat|Left|Right|Acc|Vec)\s+\|\s+"
    r"([0-9]+(?:\.[0-9]+)?)\s*(B|KB|MB)\s+\|\s+"
    r"([0-9]+(?:\.[0-9]+)?)\s*(B|KB|MB)\s+\|"
)
BUFFER_HEADER_RE = re.compile(
    r"^\s*Buffers \((Mat|Left|Right|Acc|Vec)\)"
)
ADDRESS_RE = re.compile(r"\[\s*([0-9]+),\s*([0-9]+)\)")
FATAL_HINT_RE = re.compile(
    r"(?i)(overflow|"
    r"exceed(?:s|ed|ing)?\s+(?:the\s+)?(?:limit|capacity)|"
    r"out\s+of\s+(?:memory|resource)|"
    r"allocat(?:e|ion)\S*\s+(?:failed|failure)|"
    r"resource\s+(?:exhausted|failure|failed))"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes(value: str, unit: str) -> int:
    factor = {"B": 1, "KB": 1024, "MB": 1024 * 1024}[unit]
    # The report rounds displayed usage to one decimal KB.  Exact capacity
    # checks use address ranges; retain a deterministic integer only for the
    # human-facing displayed value and reported limit.
    result = (Decimal(value) * factor).to_integral_value(
        rounding=ROUND_HALF_UP
    )
    return int(result)


def _profile(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise AssertionError(f"missing tile profile: {path}")
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise AssertionError(f"{path}: malformed profile line {line!r}")
        if key in result:
            raise AssertionError(f"{path}: duplicate profile key {key}")
        result[key] = value
    variant = result.get("variant", "")
    match = VARIANT_RE.fullmatch(variant)
    if match is None:
        raise AssertionError(f"{path}: invalid variant {variant!r}")
    expected = {
        "schema": "step3p5.moe.tile-sweep.v1",
        "ROUTED_GATE_MM_K_CHUNK": "512",
        "ROUTED_GATE_MM_N_CHUNK": match.group(1),
        "ROUTED_GATE_ACT_N_CHUNK": "64",
        "ROUTED_H_QUANT_N_CHUNK": "64",
        "ROUTED_DOWN_N_CHUNK": "256",
        "RECV_TILE": match.group(2),
        "RECV_SPECIAL_TILE": "32",
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise AssertionError(
                f"{path}: {key}={result.get(key)!r}, expected {value!r}"
            )
    return result


def _report_artifacts(root: Path) -> tuple[Path, Path]:
    memory_paths = sorted(
        path
        for path in root.rglob("memory_after_AllocateMemoryAddr.txt")
        if path.is_file()
    )
    pairs = [
        (memory, memory.parent / "perf_hints.log")
        for memory in memory_paths
        if (memory.parent / "perf_hints.log").is_file()
    ]
    if len(pairs) != 1:
        raise AssertionError(
            "expected exactly one paired AllocateMemoryAddr/perf_hints "
            f"report below {root}, found {len(pairs)}"
        )
    return pairs[0]


def _parse_memory_report(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if "=== Memory Usage Report ===" not in lines:
        raise AssertionError(f"{path}: not an AllocateMemoryAddr report")
    if "Pass: AllocateMemoryAddr" not in lines:
        raise AssertionError(f"{path}: wrong allocation pass")
    if "Backend: 910B" not in lines:
        raise AssertionError(f"{path}: wrong backend")

    sections: dict[str, dict[str, Any]] = {}
    current_name: str | None = None
    current_space: str | None = None

    for line in lines:
        section_match = SECTION_RE.fullmatch(line)
        if section_match:
            current_name = section_match.group(1)
            if current_name in sections:
                raise AssertionError(f"{path}: duplicate section {current_name}")
            sections[current_name] = {
                "summary": {},
                "address_ends": {},
            }
            current_space = None
            continue
        if current_name is None:
            continue

        summary_match = SUMMARY_RE.match(line)
        if summary_match:
            space = summary_match.group(1)
            if space in sections[current_name]["summary"]:
                raise AssertionError(
                    f"{path}: duplicate {current_name}.{space} summary"
                )
            sections[current_name]["summary"][space] = {
                "displayed_used_bytes": _bytes(
                    summary_match.group(2),
                    summary_match.group(3),
                ),
                "reported_limit_bytes": _bytes(
                    summary_match.group(4),
                    summary_match.group(5),
                ),
            }
            continue

        buffer_match = BUFFER_HEADER_RE.match(line)
        if buffer_match:
            current_space = buffer_match.group(1)
            continue

        address_match = ADDRESS_RE.search(line)
        if address_match and current_space is not None:
            start = int(address_match.group(1))
            end = int(address_match.group(2))
            if start < 0 or end <= start:
                raise AssertionError(
                    f"{path}: invalid address range [{start}, {end})"
                )
            sections[current_name]["address_ends"].setdefault(
                current_space,
                [],
            ).append(end)

    if not sections:
        raise AssertionError(f"{path}: no function sections found")
    return sections


def _audit_target_section(
    *,
    name: str,
    expected_spaces: tuple[str, ...],
    parsed: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if name not in parsed:
        raise AssertionError(f"missing required MoE resource section: {name}")
    section = parsed[name]
    summary = section["summary"]
    address_ends = section["address_ends"]
    missing = [space for space in expected_spaces if space not in summary]
    if missing:
        raise AssertionError(f"{name}: missing resource summaries {missing}")

    spaces: dict[str, Any] = {}
    for space, item in summary.items():
        if space not in SPACE_LIMITS:
            raise AssertionError(f"{name}: unknown resource space {space}")
        expected_limit = SPACE_LIMITS[space]
        reported_limit = item["reported_limit_bytes"]
        if reported_limit != expected_limit:
            raise AssertionError(
                f"{name}.{space}: reported limit={reported_limit}, "
                f"expected={expected_limit}"
            )
        if item["displayed_used_bytes"] > expected_limit:
            raise AssertionError(
                f"{name}.{space}: displayed used="
                f"{item['displayed_used_bytes']} exceeds "
                f"limit={expected_limit}"
            )
        ends = address_ends.get(space, [])
        if not ends:
            raise AssertionError(
                f"{name}.{space}: exact buffer address evidence is missing"
            )
        exact_used = max(ends)
        if exact_used > expected_limit:
            raise AssertionError(
                f"{name}.{space}: exact used={exact_used} exceeds "
                f"limit={expected_limit}"
            )
        spaces[space] = {
            **item,
            "exact_used_bytes": exact_used,
            "headroom_bytes": expected_limit - exact_used,
        }
    return {
        "function": name,
        "expected_spaces": list(expected_spaces),
        "spaces": spaces,
    }


def audit_compile_resources(
    *,
    build_root: Path,
    profile_path: Path,
    source_manifest_path: Path,
) -> dict[str, Any]:
    build_root = build_root.resolve()
    if not build_root.is_dir():
        raise AssertionError(f"missing build root: {build_root}")
    if not source_manifest_path.is_file():
        raise AssertionError(f"missing source manifest: {source_manifest_path}")

    profile = _profile(profile_path)
    memory_report, perf_hints = _report_artifacts(build_root)
    hint_lines = perf_hints.read_text(encoding="utf-8").splitlines()
    fatal_hints = [line for line in hint_lines if FATAL_HINT_RE.search(line)]
    if fatal_hints:
        raise AssertionError(
            "fatal resource diagnostics in perf_hints.log: "
            + " | ".join(fatal_hints[:5])
        )

    parsed = _parse_memory_report(memory_report)
    targets: dict[str, Any] = {}
    for name, expected_spaces in REQUIRED_FUNCTIONS.items():
        targets[name] = _audit_target_section(
            name=name,
            expected_spaces=expected_spaces,
            parsed=parsed,
        )

    maxima = {}
    for space in SPACE_LIMITS:
        values = [
            item["exact_used_bytes"]
            for target in targets.values()
            for name, item in target["spaces"].items()
            if name == space
        ]
        if not values:
            raise AssertionError(f"no exact usage evidence for {space}")
        maxima[space] = max(values)
    return {
        "schema": "step3p5.moe.tile-compile-resource-audit.v1",
        "variant": profile["variant"],
        "build_root": str(build_root),
        "profile": profile,
        "profile_sha256": _sha256(profile_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "memory_report": {
            "path": str(memory_report),
            "sha256": _sha256(memory_report),
        },
        "perf_hints": {
            "path": str(perf_hints),
            "sha256": _sha256(perf_hints),
            "line_count": len(hint_lines),
            "fatal_resource_diagnostics": [],
        },
        "limits_bytes": SPACE_LIMITS,
        "maxima_bytes": maxima,
        "required_function_count": len(REQUIRED_FUNCTIONS),
        "functions": targets,
        "passed": True,
    }


def _write_stable(path: Path, value: dict[str, Any]) -> None:
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise AssertionError(
                f"refusing to replace different resource audit: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = Path(args.out)
    try:
        result = audit_compile_resources(
            build_root=Path(args.build_root),
            profile_path=Path(args.profile),
            source_manifest_path=Path(args.source_manifest),
        )
    except Exception as error:
        failure = {
            "schema": "step3p5.moe.tile-compile-resource-audit.v1",
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
        }
        _write_stable(output, failure)
        print(f"MOE_TILE_COMPILE_RESOURCE_AUDIT=FAIL: {error}")
        return 1
    _write_stable(output, result)
    print("MOE_TILE_COMPILE_RESOURCE_AUDIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
