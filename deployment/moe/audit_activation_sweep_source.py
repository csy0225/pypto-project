#!/usr/bin/env python3
"""Audit immutable source variants for the MoE activation sweep."""
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
    "ROUTED_GATE_MM_N_CHUNK": 64,
    "ROUTED_H_QUANT_N_CHUNK": 64,
    "ROUTED_DOWN_N_CHUNK": 256,
    "RECV_TILE": 16,
}
ALLOWED_ACTIVATION = (64, 128, 256)
VARIANT_RE = re.compile(r"^act-n(64|128|256)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_assignments(tree: ast.Module) -> dict[str, int]:
    values: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
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
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise AssertionError("source segment is unavailable")
    return segment


def _profile(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise AssertionError(f"{path}: malformed profile line: {line!r}")
        result[key] = value
    return result


def audit_source(source_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    name = source_dir.name
    if not VARIANT_RE.fullmatch(name):
        raise AssertionError(f"unsupported activation variant: {name}")
    decode = source_dir / "models/step3p5/decode_fwd.py"
    manifest = source_dir / "SOURCE_SHA256SUMS"
    profile_path = source_dir / "ACTIVATION_SWEEP_PROFILE.txt"
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
    assignments = _module_assignments(tree)
    for key, expected in EXPECTED.items():
        if assignments.get(key) != expected:
            raise AssertionError(
                f"{name}: {key}={assignments.get(key)!r}, expected={expected}"
            )
    activation = assignments.get("ROUTED_GATE_ACT_N_CHUNK")
    if activation not in ALLOWED_ACTIVATION:
        raise AssertionError(f"{name}: invalid activation chunk {activation}")
    expected_activation = int(name.removeprefix("act-n"))
    if activation != expected_activation:
        raise AssertionError(
            f"{name}: profile name says {expected_activation}, "
            f"source says {activation}"
        )

    routed = _segment(source, _method(tree, "_expert_routed"))
    quant_begin = routed.index('name_hint="routed_h_quant"')
    down_begin = routed.index('name_hint="expert_down"', quant_begin)
    quant_segment = routed[quant_begin:down_begin]
    if "ROUTED_H_QUANT_N_CHUNK" not in quant_segment:
        raise AssertionError(f"{name}: quant stage is not independently chunked")
    if "ROUTED_GATE_ACT_N_CHUNK" in quant_segment:
        raise AssertionError(
            f"{name}: activation chunk leaked into the quant stage"
        )

    specialized = _segment(
        source,
        _method(tree, "_expert_routed_swiglu7"),
    )
    if "ROUTED_GATE_ACT_N_CHUNK" in specialized:
        raise AssertionError(
            f"{name}: specialized L43/L44 helper changed with activation sweep"
        )

    profile = _profile(profile_path)
    expected_profile = {
        "ROUTED_GATE_MM_K_CHUNK": str(EXPECTED["ROUTED_GATE_MM_K_CHUNK"]),
        "ROUTED_GATE_MM_N_CHUNK": str(EXPECTED["ROUTED_GATE_MM_N_CHUNK"]),
        "ROUTED_GATE_ACT_N_CHUNK": str(activation),
        "ROUTED_H_QUANT_N_CHUNK": str(EXPECTED["ROUTED_H_QUANT_N_CHUNK"]),
        "ROUTED_DOWN_N_CHUNK": str(EXPECTED["ROUTED_DOWN_N_CHUNK"]),
        "RECV_TILE": str(EXPECTED["RECV_TILE"]),
    }
    for key, expected in expected_profile.items():
        if profile.get(key) != expected:
            raise AssertionError(
                f"{name}: profile {key}={profile.get(key)!r}, "
                f"expected={expected!r}"
            )

    return {
        "variant": name,
        "source_dir": str(source_dir),
        "decode_fwd_sha256": _sha256(decode),
        "source_manifest_sha256": _sha256(manifest),
        "activation_n_chunk": activation,
        "quant_n_chunk": EXPECTED["ROUTED_H_QUANT_N_CHUNK"],
        "specialized_helper_unchanged_by_sweep": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(args.source_root)
    records = [audit_source(root / name) for name in ("act-n64", "act-n128", "act-n256")]
    if len({item["decode_fwd_sha256"] for item in records}) != len(records):
        raise AssertionError("activation variants do not have distinct source hashes")
    result = {
        "schema": "step3p5.moe.activation-sweep-source-audit.v1",
        "source_root": str(root.resolve()),
        "variants": records,
        "passed": True,
    }
    output = Path(args.out)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("MOE_ACTIVATION_SOURCE_AUDIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
