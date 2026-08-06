from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


MOE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MOE_DIR))

import audit_route_dfx_source_compatibility as audit  # noqa: E402
from validate_five_layer_route_case import (  # noqa: E402
    ROUTE_ARTIFACT_NAMES,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, value: dict) -> None:
    _write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path) -> None:
    lines = [
        f"{_sha(path)}  {path.relative_to(root)}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SOURCE_SHA256SUMS"
    ]
    _write(root / "SOURCE_SHA256SUMS", "\n".join(lines) + "\n")


def _sources(root: Path) -> tuple[Path, Path]:
    dfx = root / "dfx"
    route = root / "route"
    shared = {
        "COMMIT": "commit\n",
        "models/step3p5/decode_fwd.py": "decode\n",
        "tests/step3p5/harnesses/_five_layer_moe_program.py": "program\n",
        "tests/step3p5/harnesses/_stage_five_layer_moe.py": "stage\n",
        "tools/step3p5/five_layer_moe_holder.py": "holder\n",
        "tools/step3p5/analyze_five_layer_moe_dfx.py": "analyzer\n",
        "shared.txt": "same\n",
    }
    for relative, content in shared.items():
        _write(dfx / relative, content)
        _write(route / relative, content)
    _write(dfx / "DFX_SOURCE_PROVENANCE.txt", "dfx\n")
    _write(route / "DFX_SOURCE_PROVENANCE.txt", "route\n")
    _write(dfx / "PARENT_SOURCE_SHA256SUMS", "dfx-parent\n")
    _write(route / "PARENT_SOURCE_SHA256SUMS", "route-parent\n")
    for relative in audit.ROUTE_ADDITIONS:
        _write(route / relative, relative + "\n")
    _manifest(dfx)
    _manifest(route)
    return dfx, route


def _campaign(
    root: Path,
    dfx: Path,
    route: Path,
    *,
    image_ref: str,
    profile: str = "row16",
    source_role: str = "reference",
) -> Path:
    campaign = root / "campaign"
    critical = {
        relative: _sha(dfx / relative) for relative in audit.CRITICAL_FILES
    }
    route_hashes = {
        relative: _sha(route / relative) for relative in audit.ROUTE_ADDITIONS
    }
    for batch in audit.BATCHES:
        tokens = list(range(batch))
        dfx_run = (
            campaign
            / "runs"
            / f"candidate-r1-dfx-bs{batch}-64k"
            / "runtime"
            / f"bs{batch}"
        )
        _write_json(
            dfx_run / "report.json",
            {
                "source_kind": "candidate",
                "mode": "dfx",
                "round": 1,
                "image_ref": image_ref,
                "workload": {
                    "active_batch": batch,
                    "context_len_per_sequence": audit.CONTEXT_LEN,
                    "input_tokens": tokens,
                },
                "source": {
                    "source_manifest_sha256": _sha(
                        dfx / "SOURCE_SHA256SUMS"
                    ),
                    "decode_fwd_sha256": critical[
                        "models/step3p5/decode_fwd.py"
                    ],
                    "five_layer_program_sha256": critical[
                        "tests/step3p5/harnesses/_five_layer_moe_program.py"
                    ],
                    "five_layer_holder_sha256": critical[
                        "tools/step3p5/five_layer_moe_holder.py"
                    ],
                },
            },
        )
        runtime = (
            campaign
            / "route-sidecar-v4"
            / f"candidate-route-bs{batch}-64k"
            / "runtime"
        )
        runtime.mkdir(parents=True, exist_ok=True)
        for name in ROUTE_ARTIFACT_NAMES:
            (runtime / name).write_bytes(f"{name}-{batch}".encode())
        artifacts = {
            name: _sha(runtime / name) for name in ROUTE_ARTIFACT_NAMES
        }
        source = {
            "source_tree_manifest_sha256": _sha(
                route / "SOURCE_SHA256SUMS"
            ),
            "decode_fwd_sha256": critical[
                "models/step3p5/decode_fwd.py"
            ],
            "formal_program_sha256": critical[
                "tests/step3p5/harnesses/_five_layer_moe_program.py"
            ],
            "route_program_sha256": route_hashes[
                "tests/step3p5/harnesses/_five_layer_moe_route_program.py"
            ],
            "route_stage_sha256": route_hashes[
                "tests/step3p5/harnesses/_stage_five_layer_moe_route.py"
            ],
            "route_holder_sha256": route_hashes[
                "tools/step3p5/five_layer_moe_route_holder.py"
            ],
        }
        _write_json(
            runtime / "five_layer_moe_route_report.json",
            {
                "provenance": {
                    "image_digest": image_ref,
                    "source": source,
                    "input_contract": {
                        "input_tokens": tokens,
                        "workload": {
                            "active_batch": batch,
                            "context_len": audit.CONTEXT_LEN,
                        },
                    },
                },
                "artifacts": artifacts,
            },
        )
        _write_json(
            runtime / "route_artifact_validation.json",
            {
                "schema": "step3p5.five-layer-moe-route-validation.v1",
                "passed": True,
                "profile": profile,
                "source_role": source_role,
                "active_batch": batch,
                "context_len_per_sequence": audit.CONTEXT_LEN,
                "input_tokens": tokens,
                "image_ref": image_ref,
                "source_manifest_sha256": _sha(
                    route / "SOURCE_SHA256SUMS"
                ),
                "decode_fwd_sha256": critical[
                    "models/step3p5/decode_fwd.py"
                ],
                "hidden_bit_exact": True,
                "padding_zero": True,
                "local_count_exact": True,
                "expected_routes_per_source_per_layer": batch * 8,
                "global_routes_per_layer": [batch * 64, batch * 64],
                "expected_global_routes_per_layer": batch * 64,
                "window_independence_validated": True,
                "artifacts": artifacts,
            },
        )
    return campaign


def _argv(
    tmp_path: Path,
    dfx: Path,
    route: Path,
    campaign: Path,
    image_ref: str,
    *,
    profile: str = "row16",
    source_role: str = "reference",
) -> list[str]:
    return [
        "audit_route_dfx_source_compatibility.py",
        "--campaign",
        str(campaign),
        "--dfx-source",
        str(dfx),
        "--route-source",
        str(route),
        "--sidecar-dir",
        "route-sidecar-v4",
        "--image-ref",
        image_ref,
        "--profile",
        profile,
        "--source-role",
        source_role,
        "--out",
        str(tmp_path / "compatibility.json"),
    ]


def test_additive_route_overlay_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_ref = "image@sha256:" + "a" * 64
    dfx, route = _sources(tmp_path)
    campaign = _campaign(tmp_path, dfx, route, image_ref=image_ref)
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(tmp_path, dfx, route, campaign, image_ref),
    )
    assert audit.main() == 0
    report = json.loads(
        (tmp_path / "compatibility.json").read_text(encoding="utf-8")
    )
    assert report["passed"]
    assert sorted(report["route_addition_hashes"]) == sorted(
        audit.ROUTE_ADDITIONS
    )
    assert report["batches"]["16"]["input_tokens"] == list(range(16))


def test_unapproved_route_file_is_rejected(tmp_path: Path) -> None:
    dfx, route = _sources(tmp_path)
    _write(route / "models/step3p5/unapproved.py", "drift\n")
    _manifest(route)
    with pytest.raises(AssertionError, match="unexpected route source additions"):
        audit._audit_sources(dfx, route)


def test_generic_profile_and_candidate_role_are_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_ref = "image@sha256:" + "a" * 64
    dfx, route = _sources(tmp_path)
    campaign = _campaign(
        tmp_path,
        dfx,
        route,
        image_ref=image_ref,
        profile="act-n64",
        source_role="candidate",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            tmp_path,
            dfx,
            route,
            campaign,
            image_ref,
            profile="act-n64",
            source_role="candidate",
        ),
    )
    assert audit.main() == 0
    report = json.loads(
        (tmp_path / "compatibility.json").read_text(encoding="utf-8")
    )
    assert report["profile"] == "act-n64"
    assert report["source_role"] == "candidate"


def test_common_compute_drift_is_rejected(tmp_path: Path) -> None:
    dfx, route = _sources(tmp_path)
    _write(route / "models/step3p5/decode_fwd.py", "changed\n")
    _manifest(route)
    with pytest.raises(AssertionError, match="common-file drift"):
        audit._audit_sources(dfx, route)


def test_route_report_formal_program_drift_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_ref = "image@sha256:" + "a" * 64
    dfx, route = _sources(tmp_path)
    campaign = _campaign(tmp_path, dfx, route, image_ref=image_ref)
    path = (
        campaign
        / "route-sidecar-v4"
        / "candidate-route-bs1-64k"
        / "runtime"
        / "five_layer_moe_route_report.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["provenance"]["source"]["formal_program_sha256"] = "0" * 64
    _write_json(path, value)
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(tmp_path, dfx, route, campaign, image_ref),
    )
    with pytest.raises(AssertionError, match="formal_program_sha256"):
        audit.main()
