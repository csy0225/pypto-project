from __future__ import annotations

import json
from pathlib import Path

import pytest

import audit_activation_sweep_source as audit


def _variant(tmp_path: Path, name: str, activation: int) -> Path:
    source = tmp_path / name
    decode = source / "models" / "step3p5"
    decode.mkdir(parents=True)
    (source / "models" / "step3p5" / "decode_fwd.py").write_text(
        f"""
ROUTED_GATE_MM_K_CHUNK = 512
ROUTED_GATE_MM_N_CHUNK = 64
ROUTED_GATE_ACT_N_CHUNK = {activation}
ROUTED_H_QUANT_N_CHUNK = 64
ROUTED_DOWN_N_CHUNK = 256
RECV_TILE = 16
def _expert_routed(self):
    with pl.at(name_hint="routed_h_quant"):
        for _ in pl.range(1280 // ROUTED_H_QUANT_N_CHUNK):
            pass
    for _ in pl.spmd(16, name_hint="expert_down"):
        pass
def _expert_routed_swiglu7(self):
    with pl.at(name_hint="expert_gate_up"):
        pass
""",
        encoding="utf-8",
    )
    (source / "ACTIVATION_SWEEP_PROFILE.txt").write_text(
        "\n".join(
            [
                "schema=step3p5.moe.activation-sweep.v1",
                "base_candidate_commit=db8fa7aaac69d1597590cb0ec14fe79ac9c3316f",
                "ROUTED_GATE_MM_K_CHUNK=512",
                "ROUTED_GATE_MM_N_CHUNK=64",
                f"ROUTED_GATE_ACT_N_CHUNK={activation}",
                "ROUTED_H_QUANT_N_CHUNK=64",
                "ROUTED_DOWN_N_CHUNK=256",
                "RECV_TILE=16",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "COMMIT").write_text("test\n", encoding="utf-8")
    import subprocess

    subprocess.run(
        "find . -type f ! -name SOURCE_SHA256SUMS -print0 | "
        "sort -z | xargs -0 sha256sum > SOURCE_SHA256SUMS",
        cwd=source,
        shell=True,
        check=True,
    )
    return source


def test_audit_accepts_three_variants(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    for value in (64, 128, 256):
        _variant(root, f"act-n{value}", value)
    output = tmp_path / "audit.json"
    records = [
        audit.audit_source(root / f"act-n{value}")
        for value in (64, 128, 256)
    ]
    assert [item["activation_n_chunk"] for item in records] == [64, 128, 256]
    output.write_text(json.dumps(records), encoding="utf-8")


def test_audit_rejects_activation_leaking_into_quant(tmp_path: Path) -> None:
    source = _variant(tmp_path, "act-n64", 64)
    path = source / "models" / "step3p5" / "decode_fwd.py"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "ROUTED_H_QUANT_N_CHUNK", "ROUTED_GATE_ACT_N_CHUNK"
        ),
        encoding="utf-8",
    )
    import subprocess

    subprocess.run(
        "find . -type f ! -name SOURCE_SHA256SUMS -print0 | "
        "sort -z | xargs -0 sha256sum > SOURCE_SHA256SUMS",
        cwd=source,
        shell=True,
        check=True,
    )
    with pytest.raises(AssertionError, match="ROUTED_H_QUANT_N_CHUNK"):
        audit.audit_source(source)
