#!/usr/bin/env python3
"""Audit immutable routed gate/up tile-sweep source variants."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


EXPECTED = {
    "ROUTED_GATE_MM_K_CHUNK": 512,
    "ROUTED_GATE_ACT_N_CHUNK": 64,
    "ROUTED_H_QUANT_N_CHUNK": 64,
    "ROUTED_DOWN_N_CHUNK": 256,
    "RECV_SPECIAL_TILE": 32,
}
ALLOWED_MM_N = (32, 64)
ALLOWED_RECV_TILE = (8, 16, 32)
VARIANT_RE = re.compile(r"^mm-n(32|64)-r(8|16|32)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assignments(tree: ast.Module) -> dict[str, int]:
    values: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, int):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    return values


def _method(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one {name}, found {len(matches)}")
    return matches[0]


def _segment(source: str, node: ast.AST) -> str:
    value = ast.get_source_segment(source, node)
    if value is None:
        raise AssertionError("source segment is unavailable")
    return value


def _profile(path: Path) -> dict[str, str]:
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
    return result


def audit_source(source_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    name = source_dir.name
    match = VARIANT_RE.fullmatch(name)
    if match is None:
        raise AssertionError(f"unsupported tile variant: {name}")
    expected_mm_n = int(match.group(1))
    expected_recv_tile = int(match.group(2))

    decode = source_dir / "models/step3p5/decode_fwd.py"
    manifest = source_dir / "SOURCE_SHA256SUMS"
    profile_path = source_dir / "MOE_TILE_SWEEP_PROFILE.txt"
    for path in (decode, manifest, profile_path):
        if not path.is_file():
            raise AssertionError(f"{source_dir}: missing {path.name}")

    subprocess.run(
        ["sha256sum", "-c", "SOURCE_SHA256SUMS"],
        cwd=source_dir,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    source = decode.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(decode))
    assignments = _assignments(tree)

    expected = dict(EXPECTED)
    expected.update(
        {
            "ROUTED_GATE_MM_N_CHUNK": expected_mm_n,
            "RECV_TILE": expected_recv_tile,
        }
    )
    for key, value in expected.items():
        if assignments.get(key) != value:
            raise AssertionError(
                f"{name}: {key}={assignments.get(key)!r}, expected {value}"
            )

    # The sweep must not alter the L43/L44 specialization or the numerical
    # activation/quant/down portions of the ordinary routed expert.
    specialized = _segment(source, _method(tree, "_expert_routed_swiglu7"))
    if "ROUTED_GATE_MM_N_CHUNK" in specialized:
        raise AssertionError(f"{name}: MM-N sweep leaked into specialization")
    if "RECV_TILE" in specialized:
        raise AssertionError(f"{name}: receive tile sweep leaked into specialization")

    routed = _segment(source, _method(tree, "_expert_routed_consume"))
    if "ROUTED_GATE_ACT_N_CHUNK" not in routed:
        raise AssertionError(
            f"{name}: activation chunk is missing from routed consume"
        )
    if "ROUTED_DOWN_N_CHUNK" not in routed:
        raise AssertionError(f"{name}: down chunk is missing")
    quant_begin = routed.index('name_hint="routed_h_quant"')
    down_begin = routed.index('name_hint="expert_down"', quant_begin)
    quant_segment = routed[quant_begin:down_begin]
    if "ROUTED_H_QUANT_N_CHUNK" not in quant_segment:
        raise AssertionError(f"{name}: quant chunk is not independently frozen")
    if "ROUTED_GATE_ACT_N_CHUNK" in quant_segment:
        raise AssertionError(f"{name}: activation chunk leaked into quant")

    producer_l3 = _segment(
        source,
        _method(tree, "_expert_routed_gate_up_produce_l3"),
    )
    producer_l4 = _segment(
        source,
        _method(tree, "_expert_routed_gate_up_produce_l4"),
    )
    for label, body in (("L3", producer_l3), ("L4", producer_l4)):
        if body.count('name_hint="expert_gate_mm"') != 1:
            raise AssertionError(f"{name}: {label} gate task contract changed")
        if body.count('name_hint="expert_up_mm"') != 1:
            raise AssertionError(f"{name}: {label} up task contract changed")
        if "inter // ROUTED_GATE_MM_N_CHUNK" not in body:
            raise AssertionError(f"{name}: {label} MM-N is not task partition")
        if "(n_rows + RECV_TILE - 1) // RECV_TILE" not in body:
            raise AssertionError(f"{name}: {label} receive tile is not row partition")

    profile = _profile(profile_path)
    expected_axis = (
        "control"
        if name == "mm-n64-r16"
        else "mm_n"
        if expected_recv_tile == 16
        else "recv_tile"
        if expected_mm_n == 64
        else "cross_reserved"
    )
    expected_profile = {
        "schema": "step3p5.moe.tile-sweep.v1",
        "variant": name,
        "axis": expected_axis,
        "ROUTED_GATE_MM_K_CHUNK": "512",
        "ROUTED_GATE_MM_N_CHUNK": str(expected_mm_n),
        "ROUTED_GATE_ACT_N_CHUNK": "64",
        "ROUTED_H_QUANT_N_CHUNK": "64",
        "ROUTED_DOWN_N_CHUNK": "256",
        "RECV_TILE": str(expected_recv_tile),
        "RECV_SPECIAL_TILE": "32",
    }
    for key, value in expected_profile.items():
        if profile.get(key) != value:
            raise AssertionError(
                f"{name}: profile {key}={profile.get(key)!r}, expected {value!r}"
            )
    if not profile.get("base_source_manifest_sha256"):
        raise AssertionError(f"{name}: base source manifest is missing")

    return {
        "variant": name,
        "axis": expected_axis,
        "source_dir": str(source_dir),
        "decode_fwd_sha256": _sha256(decode),
        "source_manifest_sha256": _sha256(manifest),
        "mm_n_chunk": expected_mm_n,
        "recv_tile": expected_recv_tile,
        "activation_n_chunk": 64,
        "quant_n_chunk": 64,
        "quant_chunk_is_activation_alias": False,
        "down_n_chunk": 256,
        "specialized_helper_unchanged": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(args.source_root)
    names = (
        "mm-n32-r8",
        "mm-n32-r16",
        "mm-n32-r32",
        "mm-n64-r8",
        "mm-n64-r16",
        "mm-n64-r32",
    )
    records = [audit_source(root / name) for name in names]
    if len({item["decode_fwd_sha256"] for item in records}) != len(records):
        raise AssertionError("tile variants do not have distinct source hashes")
    result = {
        "schema": "step3p5.moe.tile-sweep-source-audit.v1",
        "source_root": str(root.resolve()),
        "variants": records,
        "passed": True,
    }
    output = Path(args.out)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("MOE_TILE_SWEEP_SOURCE_AUDIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
