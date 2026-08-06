from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import audit_tile_sweep_source as audit


def _variant(tmp_path: Path, name: str, mm_n: int, recv_tile: int) -> Path:
    source = tmp_path / name
    decode = source / "models" / "step3p5"
    decode.mkdir(parents=True)
    (decode / "decode_fwd.py").write_text(
        f"""
ROUTED_GATE_MM_K_CHUNK = 512
ROUTED_GATE_MM_N_CHUNK = {mm_n}
ROUTED_GATE_ACT_N_CHUNK = 64
ROUTED_H_QUANT_N_CHUNK = 64
ROUTED_DOWN_N_CHUNK = 256
RECV_SPECIAL_TILE = 32
RECV_TILE = {recv_tile}
def _expert_routed_gate_up_produce_l3(self):
    for _ in pl.parallel(n_local_experts):
        n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE
        with pl.spmd(inter // ROUTED_GATE_MM_N_CHUNK,
                     name_hint="expert_gate_mm"):
            pass
        with pl.spmd(inter // ROUTED_GATE_MM_N_CHUNK,
                     name_hint="expert_up_mm"):
            pass
def _expert_routed_gate_up_produce_l4(self):
    for _ in pl.parallel(n_local_experts):
        n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE
        with pl.spmd(inter // ROUTED_GATE_MM_N_CHUNK,
                     name_hint="expert_gate_mm"):
            pass
        with pl.spmd(inter // ROUTED_GATE_MM_N_CHUNK,
                     name_hint="expert_up_mm"):
            pass
def _expert_routed_consume(self):
    ROUTED_GATE_ACT_N_CHUNK
    with pl.at(name_hint="routed_h_quant"):
        ROUTED_H_QUANT_N_CHUNK
    with pl.spmd(name_hint="expert_down"):
        ROUTED_DOWN_N_CHUNK
def _expert_routed_swiglu7(self):
    pass
""",
        encoding="utf-8",
    )
    (source / "MOE_TILE_SWEEP_PROFILE.txt").write_text(
        "\n".join(
            [
                "schema=step3p5.moe.tile-sweep.v1",
                f"variant={name}",
                (
                    "axis=control"
                    if name == "mm-n64-r16"
                    else "axis=mm_n"
                    if recv_tile == 16
                    else "axis=recv_tile"
                    if mm_n == 64
                    else "axis=cross_reserved"
                ),
                "base_source_manifest_sha256=test-base",
                "ROUTED_GATE_MM_K_CHUNK=512",
                f"ROUTED_GATE_MM_N_CHUNK={mm_n}",
                "ROUTED_GATE_ACT_N_CHUNK=64",
                "ROUTED_H_QUANT_N_CHUNK=64",
                "ROUTED_DOWN_N_CHUNK=256",
                f"RECV_TILE={recv_tile}",
                "RECV_SPECIAL_TILE=32",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "COMMIT").write_text("test\n", encoding="utf-8")
    subprocess.run(
        "find . -type f ! -name SOURCE_SHA256SUMS -print0 | "
        "sort -z | xargs -0 sha256sum > SOURCE_SHA256SUMS",
        cwd=source,
        shell=True,
        check=True,
    )
    return source


def test_audit_accepts_all_tile_variants(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    for mm_n in (32, 64):
        for recv_tile in (8, 16, 32):
            _variant(
                tmp_path / "sources",
                f"mm-n{mm_n}-r{recv_tile}",
                mm_n,
                recv_tile,
            )
    records = [
        audit.audit_source(root / f"mm-n{mm_n}-r{recv_tile}")
        for mm_n in (32, 64)
        for recv_tile in (8, 16, 32)
    ]
    assert len(records) == 6
    assert {(item["mm_n_chunk"], item["recv_tile"]) for item in records} == {
        (32, 8),
        (32, 16),
        (32, 32),
        (64, 8),
        (64, 16),
        (64, 32),
    }


def test_audit_rejects_specialization_drift(tmp_path: Path) -> None:
    source = _variant(tmp_path, "mm-n32-r16", 32, 16)
    path = source / "models" / "step3p5" / "decode_fwd.py"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "def _expert_routed_swiglu7(self):\n    pass",
            "def _expert_routed_swiglu7(self):\n"
            "    ROUTED_GATE_MM_N_CHUNK\n",
        ),
        encoding="utf-8",
    )
    subprocess.run(
        "find . -type f ! -name SOURCE_SHA256SUMS -print0 | "
        "sort -z | xargs -0 sha256sum > SOURCE_SHA256SUMS",
        cwd=source,
        shell=True,
        check=True,
    )
    with pytest.raises(AssertionError, match="specialization"):
        audit.audit_source(source)
