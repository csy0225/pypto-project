from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch


MOE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MOE_DIR))

import validate_five_layer_route_case as validator  # noqa: E402


IMAGE = "hub/image@sha256:" + "b" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(
    root: Path,
    *,
    bad_padding: bool = False,
    profile: str = "row16",
    decode_fwd_sha256: str = "",
) -> tuple[Path, Path, Path]:
    runtime = root / "runtime"
    golden = root / "golden"
    source = root / "source"
    runtime.mkdir()
    golden.mkdir()
    source.mkdir()
    (source / "SOURCE_SHA256SUMS").write_text("fixture\n", encoding="utf-8")

    hidden_l3 = torch.zeros((8, 1, 4096), dtype=torch.bfloat16)
    hidden_l4 = torch.ones((8, 1, 4096), dtype=torch.bfloat16)
    torch.save(hidden_l3, runtime / "hidden_l3.pt")
    torch.save(hidden_l4, runtime / "hidden_l4.pt")
    torch.save(hidden_l3, golden / "hidden_l3.pt")
    torch.save(hidden_l4, golden / "hidden_l4.pt")
    _write_json(
        golden / "manifest.json",
        {
            "schema": "step3p5.five-layer-moe-golden.v3",
            "source_kind": "baseline",
            "active_batch": 1,
            "context_len_per_sequence": 65536,
            "image_ref": IMAGE,
            "files": {
                "hidden_l3.pt": _sha256(golden / "hidden_l3.pt"),
                "hidden_l4.pt": _sha256(golden / "hidden_l4.pt"),
            },
        },
    )

    recv_meta = torch.zeros((8, 2, 8, 40), dtype=torch.int32)
    for layer in range(2):
        for source_rank in range(8):
            recv_meta[source_rank, layer, source_rank, 0] = 8
    if bad_padding:
        recv_meta[0, 0, 0, 36] = 1
    local_count = recv_meta[:, :, :, :36].sum(
        dim=2,
        dtype=torch.int64,
    ).to(torch.int32)
    sidecar = {
        "schema": "step3p5.five-layer-moe-recv-meta.v1",
        "recv_meta": recv_meta.permute(1, 0, 2, 3).contiguous(),
        "local_expert_count": local_count.permute(1, 0, 2).contiguous(),
        "window_provenance": [
            {"layer": "L3", "window_id": "l3"},
            {"layer": "L4", "window_id": "l4"},
        ],
    }
    torch.save(recv_meta, runtime / "recv_meta.pt")
    torch.save(local_count, runtime / "local_expert_count.pt")
    torch.save(sidecar, runtime / "recv_meta_sidecar.pt")
    artifacts = {
        name: _sha256(runtime / name)
        for name in (
            "hidden_l3.pt",
            "hidden_l4.pt",
            "recv_meta.pt",
            "local_expert_count.pt",
            "recv_meta_sidecar.pt",
        )
    }
    _write_json(
        runtime / "five_layer_moe_route_report.json",
        {
            "provenance": {
                "image_digest": IMAGE,
                "source": {
                    "source_tree_manifest_sha256": _sha256(
                        source / "SOURCE_SHA256SUMS"
                    ),
                    "decode_fwd_sha256": (
                        decode_fwd_sha256
                        or validator.SOURCE_PROFILES[profile][
                            "decode_fwd_sha256"
                        ]
                    ),
                },
                "input_contract": {
                    "input_tokens": [6127],
                    "workload": {
                        "active_batch": 1,
                        "context_len": 65536,
                        "num_blocks_per_sequence": 512,
                        "context_semantics": "per_active_sequence",
                        "blocks_per_sequence": 512,
                        "block_table_blocks_per_row_capacity": 512,
                        "scheduler_num_blocks": 512,
                        "physical_num_blocks": 527,
                        "max_sequence_tokens": 65536,
                    },
                },
            },
            "artifacts": artifacts,
        },
    )
    return runtime, golden, source


def _argv(
    runtime: Path,
    golden: Path,
    source: Path,
    *,
    profile: str = "row16",
    expected_decode_sha256: str = "",
    source_role: str = "",
) -> list[str]:
    args = [
        "validate_five_layer_route_case.py",
        "--runtime",
        str(runtime),
        "--golden-dir",
        str(golden),
        "--source-root",
        str(source),
        "--image-ref",
        IMAGE,
        "--active-batch",
        "1",
        "--input-tokens",
        "6127",
        "--profile",
        profile,
    ]
    if expected_decode_sha256:
        args.extend(
            [
                "--expected-decode-sha256",
                expected_decode_sha256,
                "--source-role",
                source_role,
            ]
        )
    return args


def test_route_validator_recomputes_tensor_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, golden, source = _fixture(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(runtime, golden, source))
    assert validator.main() == 0
    result = json.loads(
        (runtime / "route_artifact_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["passed"]
    assert result["global_routes_per_layer"] == [64, 64]
    assert result["profile"] == "row16"


def test_route_validator_accepts_shared_split_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, golden, source = _fixture(
        tmp_path,
        profile="shared-split",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(runtime, golden, source, profile="shared-split"),
    )
    assert validator.main() == 0
    result = json.loads(
        (runtime / "route_artifact_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["profile"] == "shared-split"
    assert result["source_role"] == "candidate"


def test_route_validator_accepts_generic_matched_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decode = "a" * 64
    runtime, golden, source = _fixture(
        tmp_path,
        profile="row16",
        decode_fwd_sha256=decode,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            runtime,
            golden,
            source,
            profile="act-n64",
            expected_decode_sha256=decode,
            source_role="candidate",
        ),
    )
    assert validator.main() == 0
    result = json.loads(
        (runtime / "route_artifact_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["profile"] == "act-n64"
    assert result["source_role"] == "candidate"


def test_route_validator_rejects_nonzero_expert_padding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, golden, source = _fixture(tmp_path, bad_padding=True)
    monkeypatch.setattr(sys, "argv", _argv(runtime, golden, source))
    with pytest.raises(AssertionError, match="padded experts"):
        validator.main()


@pytest.mark.parametrize(
    "missing",
    (
        "hidden_bit_exact",
        "padding_zero",
        "global_routes_per_layer",
        "window_independence_validated",
    ),
)
def test_persisted_route_validation_rejects_missing_contract_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    runtime, golden, source = _fixture(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(runtime, golden, source))
    assert validator.main() == 0
    path = runtime / "route_artifact_validation.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    del value[missing]
    with pytest.raises(AssertionError, match="keys are incomplete"):
        validator.validate_route_validation_record(
            value,
            active_batch=1,
            profile="row16",
            source_role="reference",
            decode_fwd_sha256=validator.SOURCE_PROFILES["row16"][
                "decode_fwd_sha256"
            ],
            image_ref=IMAGE,
            source_manifest_sha256=_sha256(
                source / "SOURCE_SHA256SUMS"
            ),
            input_tokens=[6127],
        )
