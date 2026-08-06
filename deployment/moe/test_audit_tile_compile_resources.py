from __future__ import annotations

from pathlib import Path

import pytest

import audit_tile_compile_resources as audit


EXPECTED_FUNCTIONS = {
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


def _section(name: str, spaces: tuple[str, ...]) -> str:
    lines = [f"--- {name} ---", ""]
    for space in spaces:
        limit = audit.SPACE_LIMITS[space]
        lines.append(
            f"  {space:<6} |     1.0 KB  |"
            f"   {limit / 1024:.1f} KB  |    1.0%  |  1"
        )
    for space in spaces:
        lines.extend(
            [
                "",
                f"  Buffers ({space})  -- base allocations",
                "    Name | Size | Address range | Live range",
                "    mem_0 | 1.0 KB | [0, 1024) | [1, 2]",
            ]
        )
    return "\n".join(lines)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    build = tmp_path / "build"
    report = build / "FiveLayerMoe_test" / "report"
    report.mkdir(parents=True)
    sections = []
    for name, spaces in EXPECTED_FUNCTIONS.items():
        sections.append(_section(name, spaces))
    (report / "memory_after_AllocateMemoryAddr.txt").write_text(
        "=== Memory Usage Report ===\n"
        "Pass: AllocateMemoryAddr\n"
        "Backend: 910B\n\n"
        + "\n\n".join(sections)
        + "\n",
        encoding="utf-8",
    )
    (report / "perf_hints.log").write_text(
        "[perf_hint PH001] recommended >= 512B\n",
        encoding="utf-8",
    )
    profile = tmp_path / "MOE_TILE_SWEEP_PROFILE.txt"
    profile.write_text(
        "\n".join(
            [
                "schema=step3p5.moe.tile-sweep.v1",
                "variant=mm-n64-r16",
                "axis=control",
                "base_source_manifest_sha256=test",
                "ROUTED_GATE_MM_K_CHUNK=512",
                "ROUTED_GATE_MM_N_CHUNK=64",
                "ROUTED_GATE_ACT_N_CHUNK=64",
                "ROUTED_H_QUANT_N_CHUNK=64",
                "ROUTED_DOWN_N_CHUNK=256",
                "RECV_TILE=16",
                "RECV_SPECIAL_TILE=32",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "SOURCE_SHA256SUMS"
    manifest.write_text("test manifest\n", encoding="utf-8")
    return build, profile, manifest


def test_resource_audit_accepts_all_required_functions(
    tmp_path: Path,
) -> None:
    build, profile, manifest = _fixture(tmp_path)
    result = audit.audit_compile_resources(
        build_root=build,
        profile_path=profile,
        source_manifest_path=manifest,
    )
    assert result["passed"] is True
    assert result["required_function_count"] == len(EXPECTED_FUNCTIONS)
    assert set(result["maxima_bytes"]) == set(audit.SPACE_LIMITS)


def test_resource_audit_rejects_missing_swa_function(
    tmp_path: Path,
) -> None:
    build, profile, manifest = _fixture(tmp_path)
    path = next(build.rglob("memory_after_AllocateMemoryAddr.txt"))
    section = _section(
        "swa_moe_chip_orch_expert_gate_up_act",
        ("Vec",),
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace(section + "\n\n", ""),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="missing required"):
        audit.audit_compile_resources(
            build_root=build,
            profile_path=profile,
            source_manifest_path=manifest,
        )


def test_resource_audit_uses_exact_address_for_overflow(
    tmp_path: Path,
) -> None:
    build, profile, manifest = _fixture(tmp_path)
    path = next(build.rglob("memory_after_AllocateMemoryAddr.txt"))
    text = path.read_text(encoding="utf-8")
    marker = "--- expert_gate_mm_aic ---"
    begin = text.index(marker)
    end = text.index("--- expert_gate_mm_aiv ---")
    block = text[begin:end].replace(
        "[0, 1024)",
        "[0, 524289)",
        1,
    )
    path.write_text(text[:begin] + block + text[end:], encoding="utf-8")
    with pytest.raises(AssertionError, match="exact used=524289"):
        audit.audit_compile_resources(
            build_root=build,
            profile_path=profile,
            source_manifest_path=manifest,
        )


def test_resource_audit_rejects_fatal_perf_hint(tmp_path: Path) -> None:
    build, profile, manifest = _fixture(tmp_path)
    next(build.rglob("perf_hints.log")).write_text(
        "Vec resource overflow: used 200000, limit 188416\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="fatal resource"):
        audit.audit_compile_resources(
            build_root=build,
            profile_path=profile,
            source_manifest_path=manifest,
        )


def test_resource_audit_rejects_duplicate_summary(tmp_path: Path) -> None:
    build, profile, manifest = _fixture(tmp_path)
    path = next(build.rglob("memory_after_AllocateMemoryAddr.txt"))
    text = path.read_text(encoding="utf-8")
    marker = "  Mat    |"
    first = text.index(marker)
    duplicate = text[first : text.index("\n", first)] + "\n"
    path.write_text(text[:first] + duplicate + text[first:], encoding="utf-8")
    with pytest.raises(AssertionError, match="duplicate"):
        audit.audit_compile_resources(
            build_root=build,
            profile_path=profile,
            source_manifest_path=manifest,
        )


def test_resource_audit_rejects_displayed_summary_overflow(
    tmp_path: Path,
) -> None:
    build, profile, manifest = _fixture(tmp_path)
    path = next(build.rglob("memory_after_AllocateMemoryAddr.txt"))
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "  Mat    |     1.0 KB  |   512.0 KB",
        "  Mat    |   600.0 KB  |   512.0 KB",
        1,
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(AssertionError, match="displayed used"):
        audit.audit_compile_resources(
            build_root=build,
            profile_path=profile,
            source_manifest_path=manifest,
        )
