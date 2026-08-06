from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pytest
import torch


MOE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MOE_DIR))

import analyze_route_campaign as analyzer  # noqa: E402

IMAGE_REF = "image@sha256:" + "b" * 64
SOURCE_MANIFEST_SHA256 = "a" * 64


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _campaign(
    root: Path,
    *,
    source_suffix: str = "",
    sidecar_dir: str = "route-sidecar",
    profile: str = "row16",
    source_role: str = "reference",
    decode_fwd_sha256: str = "",
) -> Path:
    campaign = root / "campaign"
    golden_root = campaign / "golden" / "heterogeneous-64k"
    for batch in analyzer.BATCHES:
        run = (
            campaign
            / sidecar_dir
            / f"candidate-route-bs{batch}-64k"
            / "runtime"
        )
        run.mkdir(parents=True, exist_ok=True)
        golden = golden_root / f"bs{batch}"
        golden.mkdir(parents=True, exist_ok=True)
        hidden_l3 = torch.zeros((8, batch, 4096), dtype=torch.bfloat16)
        hidden_l4 = torch.ones((8, batch, 4096), dtype=torch.bfloat16)
        torch.save(hidden_l3, run / "hidden_l3.pt")
        torch.save(hidden_l4, run / "hidden_l4.pt")
        torch.save(hidden_l3, golden / "hidden_l3.pt")
        torch.save(hidden_l4, golden / "hidden_l4.pt")
        _write_json(
            golden / "manifest.json",
            {
                "schema": "step3p5.five-layer-moe-golden.v3",
                "source_kind": "baseline",
                "active_batch": batch,
                "context_len_per_sequence": analyzer.CONTEXT_LEN,
                "image_ref": IMAGE_REF,
                "files": {
                    name: hashlib.sha256(
                        (golden / name).read_bytes()
                    ).hexdigest()
                    for name in ("hidden_l3.pt", "hidden_l4.pt")
                },
            },
        )
        recv_meta = torch.zeros((8, 2, 8, 40), dtype=torch.int32)
        for layer in range(2):
            for source_rank in range(8):
                recv_meta[source_rank, layer, source_rank, 0] = batch * 8
        local_count = recv_meta[:, :, :, :36].sum(
            dim=2,
            dtype=torch.int64,
        ).to(torch.int32)
        torch.save(recv_meta, run / "recv_meta.pt")
        torch.save(local_count, run / "local_expert_count.pt")
        torch.save(
            {
                "schema": "step3p5.five-layer-moe-recv-meta.v1",
                "recv_meta": recv_meta.permute(1, 0, 2, 3).contiguous(),
                "local_expert_count": local_count.permute(
                    1, 0, 2
                ).contiguous(),
                "window_provenance": [
                    {"layer": "L3", "window_id": f"l3-{batch}"},
                    {"layer": "L4", "window_id": f"l4-{batch}"},
                ],
            },
            run / "recv_meta_sidecar.pt",
        )
        artifacts = {
            name: hashlib.sha256((run / name).read_bytes()).hexdigest()
            for name in analyzer.ROUTE_ARTIFACT_NAMES
        }
        _write_json(
            run / "five_layer_moe_route_report.json",
            {"artifacts": artifacts},
        )
        _write_json(
            run / "route_artifact_validation.json",
            {
                "passed": True,
                "schema": "step3p5.five-layer-moe-route-validation.v1",
                "active_batch": batch,
                "context_len_per_sequence": analyzer.CONTEXT_LEN,
                "input_tokens": list(range(batch)),
                "profile": profile,
                "source_role": source_role,
                "decode_fwd_sha256": (
                    decode_fwd_sha256
                    or analyzer.PROFILE_DECODE_SHA256["row16"]
                ),
                "hidden_bit_exact": True,
                "padding_zero": True,
                "local_count_exact": True,
                "window_independence_validated": True,
                "expected_routes_per_source_per_layer": batch * 8,
                "global_routes_per_layer": [batch * 64, batch * 64],
                "expected_global_routes_per_layer": batch * 64,
                "source_manifest_sha256": (
                    "a" * 63 + source_suffix
                    if source_suffix and batch == 16
                    else "a" * 64
                ),
                "image_ref": IMAGE_REF,
                "artifacts": artifacts,
            },
        )
    return campaign


def _authority_args(source_manifest: str = SOURCE_MANIFEST_SHA256) -> list[str]:
    return [
        "--expected-image-ref",
        IMAGE_REF,
        "--expected-source-manifest-sha256",
        source_manifest,
    ]


def test_route_campaign_passes_all_six_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_route_campaign.py",
            "--campaign",
            str(campaign),
            *_authority_args(),
        ],
    )
    assert analyzer.main() == 0
    report = json.loads(
        (campaign / "route_campaign_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["passed"]
    assert report["batches"]["16"]["expected_global_routes_per_layer"] == 1024


def test_route_campaign_supports_isolated_sidecar_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar_dir = "route-sidecar-v2"
    campaign = _campaign(tmp_path, sidecar_dir=sidecar_dir)
    out = campaign / sidecar_dir
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_route_campaign.py",
            "--campaign",
            str(campaign),
            "--sidecar-dir",
            sidecar_dir,
            "--out",
            str(out),
            *_authority_args(),
        ],
    )
    assert analyzer.main() == 0
    report = json.loads(
        (out / "route_campaign_report.json").read_text(encoding="utf-8")
    )
    assert report["passed"]
    assert report["sidecar_dir"] == sidecar_dir


def test_route_campaign_accepts_generic_matched_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decode = "c" * 64
    campaign = _campaign(
        tmp_path,
        profile="act-n64",
        source_role="candidate",
        decode_fwd_sha256=decode,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_route_campaign.py",
            "--campaign",
            str(campaign),
            "--expected-profile",
            "act-n64",
            "--expected-decode-sha256",
            decode,
            "--expected-source-role",
            "candidate",
            *_authority_args(),
        ],
    )
    assert analyzer.main() == 0
    report = json.loads(
        (campaign / "route_campaign_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["profile"] == "act-n64"
    assert report["source_role"] == "candidate"


def test_route_campaign_rejects_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign(tmp_path, source_suffix="c")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_route_campaign.py",
            "--campaign",
            str(campaign),
            *_authority_args(),
        ],
    )
    with pytest.raises(AssertionError, match="source_manifest_sha256"):
        analyzer.main()


def test_route_campaign_rejects_profile_or_sidecar_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign(tmp_path)
    run = (
        campaign
        / "route-sidecar"
        / "candidate-route-bs1-64k"
        / "runtime"
    )
    value = json.loads(
        (run / "route_artifact_validation.json").read_text(encoding="utf-8")
    )
    value["profile"] = "shared-split"
    _write_json(run / "route_artifact_validation.json", value)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_route_campaign.py",
            "--campaign",
            str(campaign),
            *_authority_args(),
        ],
    )
    with pytest.raises(AssertionError, match="route profile mismatch"):
        analyzer.main()
