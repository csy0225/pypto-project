from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

import pytest
import torch


MOE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MOE_DIR))

import analyze_dfx_campaign as analyzer  # noqa: E402
import test_analyze_dfx_campaign as fixtures  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _expect_publication_rejected(call: Callable[[], int]) -> None:
    """Accept either a fail-closed exception or a non-zero CLI return."""
    try:
        return_code = call()
    except Exception:
        return
    assert return_code != 0, "adversarial publication was accepted"


def _policy_with_normal_records(
    root: Path,
    *,
    mutation: str | None = None,
) -> dict:
    """Give every normal record explicit capture identity, then mutate one."""
    initial = fixtures._matched_policy(root)
    policy_path = Path(initial["policy_path"])
    policy_value = json.loads(policy_path.read_text(encoding="utf-8"))
    authority = policy_value["authority"]
    performance_path = Path(
        authority["normal_performance_report"]["path"]
    )
    performance = json.loads(
        performance_path.read_text(encoding="utf-8")
    )
    target = performance["batches"]["4"]["candidate"]["rounds"][1]
    if mutation == "iters":
        target["timing_protocol"]["iters"] = 29
    elif mutation == "warmup":
        target["timing_protocol"]["warmup"] = 4
    elif mutation == "run_identity":
        target["run"] = "candidate-r9-normal-bs4-64k"

    _write_json(performance_path, performance)
    authority["normal_performance_report"]["sha256"] = _sha256(
        performance_path
    )
    _write_json(policy_path, policy_value)
    return analyzer.load_matched_dfx_policy(policy_path)


def _forge_hash_consistent_invalid_route(
    campaign: Path,
    compatibility_path: Path,
    *,
    batch: int,
) -> None:
    runtime = (
        campaign
        / "route-sidecar"
        / f"candidate-route-bs{batch}-64k"
        / "runtime"
    )
    recv_path = runtime / "recv_meta.pt"
    recv_meta = torch.load(recv_path, map_location="cpu", weights_only=True)
    recv_meta[0, 0, 0, 36] = 1
    torch.save(recv_meta, recv_path)

    artifacts = {
        name: _sha256(runtime / name)
        for name in analyzer.ROUTE_ARTIFACT_NAMES
    }
    report_path = runtime / "five_layer_moe_route_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifacts"] = artifacts
    _write_json(report_path, report)
    validation_path = runtime / "route_artifact_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["artifacts"] = artifacts
    _write_json(validation_path, validation)

    compatibility = json.loads(
        compatibility_path.read_text(encoding="utf-8")
    )
    item = compatibility["batches"][str(batch)]
    item["route_validation_sha256"] = _sha256(validation_path)
    item["route_report_sha256"] = _sha256(report_path)
    _write_json(compatibility_path, compatibility)
    compatibility_sha = _sha256(compatibility_path)
    for current_batch in analyzer.BATCHES:
        lineage = (
            campaign
            / "runs"
            / f"candidate-r1-dfx-bs{current_batch}-64k"
            / "runtime"
            / f"bs{current_batch}"
            / "dfx_analysis"
        )
        (lineage / "source_compatibility_sha256.txt").write_text(
            compatibility_sha + "\n",
            encoding="utf-8",
        )
    target_lineage = (
        campaign
        / "runs"
        / f"candidate-r1-dfx-bs{batch}-64k"
        / "runtime"
        / f"bs{batch}"
        / "dfx_analysis"
    )
    (target_lineage / "route_validation_sha256.txt").write_text(
        item["route_validation_sha256"] + "\n",
        encoding="utf-8",
    )


def test_publication_rejects_dfx_tensor_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy_with_normal_records(tmp_path)
    campaign = fixtures._campaign(
        tmp_path,
        route_ready=False,
        matched_policy=policy,
    )
    tampered = (
        campaign
        / "runs"
        / "candidate-r1-dfx-bs4-64k"
        / "runtime"
        / "bs4"
        / "hidden_l4.pt"
    )
    tampered.write_bytes(tampered.read_bytes() + b"-tampered")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
        ],
    )
    _expect_publication_rejected(analyzer.main)


def test_publication_recomputes_route_tensor_correctness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy_with_normal_records(tmp_path)
    campaign = fixtures._campaign(
        tmp_path,
        route_ready=True,
        matched_policy=policy,
    )
    compatibility = fixtures._compatibility(
        campaign,
        profile=policy["sources"]["candidate"]["profile"],
        source_role=policy["sources"]["candidate"]["source_role"],
    )
    _forge_hash_consistent_invalid_route(
        campaign,
        compatibility,
        batch=4,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
            "--source-compatibility",
            str(compatibility),
            "--require-publication-ready",
        ],
    )
    _expect_publication_rejected(analyzer.main)


@pytest.mark.parametrize("mutation", ("iters", "warmup", "run_identity"))
def test_publication_rejects_invalid_normal_capture_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    policy = _policy_with_normal_records(tmp_path, mutation=mutation)
    campaign = fixtures._campaign(
        tmp_path,
        route_ready=False,
        matched_policy=policy,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
        ],
    )
    _expect_publication_rejected(analyzer.main)


def test_require_publication_ready_requires_matched_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = fixtures._campaign(tmp_path, route_ready=True)
    compatibility = tmp_path / "legacy_source_compatibility.json"
    _write_json(compatibility, {"passed": True})

    def _legacy_compatibility(**_kwargs: object) -> dict:
        return {
            "report": str(compatibility),
            "report_sha256": _sha256(compatibility),
            "profile": "candidate",
            "source_role": "candidate",
            "round": 1,
            "image_ref": "image@sha256:" + "9" * 64,
            "batches": list(analyzer.BATCHES),
        }

    monkeypatch.setattr(
        analyzer,
        "_validate_source_compatibility",
        _legacy_compatibility,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--source-compatibility",
            str(compatibility),
            "--require-publication-ready",
        ],
    )
    _expect_publication_rejected(analyzer.main)


def test_publication_rejects_duplicate_dfx_run_nonce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy_with_normal_records(tmp_path)
    campaign = fixtures._campaign(
        tmp_path,
        route_ready=False,
        matched_policy=policy,
    )
    source_run = campaign / "runs" / "candidate-r1-dfx-bs1-64k"
    target_run = campaign / "runs" / "candidate-r1-dfx-bs2-64k"
    nonce = (source_run / "run_nonce.txt").read_text(encoding="utf-8")
    (target_run / "run_nonce.txt").write_text(nonce, encoding="utf-8")
    validation_path = target_run / "artifact_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["run_nonce"] = nonce.strip()
    _write_json(validation_path, validation)
    _replay_kv_session_nonce(
        target_run,
        replayed_nonce=nonce.strip(),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
        ],
    )
    with pytest.raises(AssertionError, match="DFX run nonces"):
        analyzer.main()


def test_publication_rejects_duplicate_normal_run_nonce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = fixtures._matched_policy(tmp_path)
    policy_path = Path(initial["policy_path"])
    policy_value = json.loads(policy_path.read_text(encoding="utf-8"))
    performance_path = Path(
        policy_value["authority"]["normal_performance_report"]["path"]
    )
    normal_campaign = performance_path.parent
    source_run = normal_campaign / "runs" / "baseline-r1-normal-bs1-64k"
    target_run = normal_campaign / "runs" / "candidate-r2-normal-bs2-64k"
    nonce = (source_run / "run_nonce.txt").read_text(encoding="utf-8")
    (target_run / "run_nonce.txt").write_text(nonce, encoding="utf-8")
    validation_path = target_run / "artifact_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["run_nonce"] = nonce.strip()
    _write_json(validation_path, validation)
    _replay_kv_session_nonce(
        target_run,
        replayed_nonce=nonce.strip(),
    )

    performance = json.loads(
        performance_path.read_text(encoding="utf-8")
    )
    actual = analyzer.validate_normal_run(
        normal_campaign,
        "candidate",
        2,
        2,
    )
    performance["batches"]["2"]["candidate"]["rounds"][1] = actual
    _write_json(performance_path, performance)
    policy_value["authority"]["normal_performance_report"]["sha256"] = (
        _sha256(performance_path)
    )
    _write_json(policy_path, policy_value)
    policy = analyzer.load_matched_dfx_policy(policy_path)
    campaign = fixtures._campaign(
        tmp_path,
        route_ready=False,
        matched_policy=policy,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
        ],
    )
    with pytest.raises(
        AssertionError,
        match="normal run nonce|evidence_sha256.run_nonce",
    ):
        analyzer.main()


def _replay_kv_session_nonce(
    run: Path,
    *,
    replayed_nonce: str,
) -> None:
    validation_path = run / "artifact_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    kv_hashes = {}
    for rank in range(8):
        path = run / "runtime" / f"pypto_kvpool_map.json.rank{rank}"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["ipc_session"]["session_nonce"] = replayed_nonce
        _write_json(path, value)
        done_path = run / "runtime" / f"{path.name}.done"
        done = json.loads(done_path.read_text(encoding="utf-8"))
        done["ipc_session"]["session_nonce"] = replayed_nonce
        _write_json(done_path, done)
        kv_hashes[str(rank)] = _sha256(path)
    validation["kv_map_sha256_by_rank"] = kv_hashes
    marker_hashes, key_hashes = fixtures._runtime_artifact_hashes(
        run / "runtime"
    )
    validation["runtime_marker_sha256"] = marker_hashes
    validation["kv_key_sha256_by_rank"] = key_hashes
    _write_json(validation_path, validation)


def test_publication_rejects_replayed_dfx_kv_session_nonce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy_with_normal_records(tmp_path)
    campaign = fixtures._campaign(
        tmp_path,
        route_ready=False,
        matched_policy=policy,
    )
    source_run = campaign / "runs" / "candidate-r1-dfx-bs1-64k"
    target_run = campaign / "runs" / "candidate-r1-dfx-bs2-64k"
    replayed_nonce = (
        source_run / "run_nonce.txt"
    ).read_text(encoding="utf-8").strip()
    _replay_kv_session_nonce(
        target_run,
        replayed_nonce=replayed_nonce,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
        ],
    )
    with pytest.raises(AssertionError, match="IPC nonce mismatch"):
        analyzer.main()


def test_publication_rejects_replayed_normal_kv_session_nonce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = fixtures._matched_policy(tmp_path)
    policy_path = Path(initial["policy_path"])
    policy_value = json.loads(policy_path.read_text(encoding="utf-8"))
    performance_path = Path(
        policy_value["authority"]["normal_performance_report"]["path"]
    )
    normal_campaign = performance_path.parent
    source_run = normal_campaign / "runs" / "baseline-r1-normal-bs1-64k"
    target_run = normal_campaign / "runs" / "candidate-r2-normal-bs2-64k"
    replayed_nonce = (
        source_run / "run_nonce.txt"
    ).read_text(encoding="utf-8").strip()
    _replay_kv_session_nonce(
        target_run,
        replayed_nonce=replayed_nonce,
    )

    performance = json.loads(
        performance_path.read_text(encoding="utf-8")
    )
    target_record = performance["batches"]["2"]["candidate"]["rounds"][1]
    validation_path = target_run / "artifact_validation.json"
    target_record["evidence_sha256"]["artifact_validation.json"] = _sha256(
        validation_path
    )
    for rank in range(8):
        relative = f"runtime/pypto_kvpool_map.json.rank{rank}"
        target_record["evidence_sha256"][relative] = _sha256(
            target_run / relative
        )
    _write_json(performance_path, performance)
    policy_value["authority"]["normal_performance_report"]["sha256"] = (
        _sha256(performance_path)
    )
    _write_json(policy_path, policy_value)
    policy = analyzer.load_matched_dfx_policy(policy_path)
    campaign = fixtures._campaign(
        tmp_path,
        route_ready=False,
        matched_policy=policy,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
        ],
    )
    with pytest.raises(AssertionError, match="IPC nonce mismatch"):
        analyzer.main()


@pytest.mark.parametrize(
    "marker",
    (
        "runtime/ready.rank0",
        "runtime/pypto_kvpool_map.json.rank0.done",
    ),
)
def test_publication_rejects_missing_dfx_runtime_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
) -> None:
    policy = _policy_with_normal_records(tmp_path)
    campaign = fixtures._campaign(
        tmp_path,
        route_ready=False,
        matched_policy=policy,
    )
    run = campaign / "runs" / "candidate-r1-dfx-bs1-64k"
    (run / marker).unlink()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
        ],
    )
    _expect_publication_rejected(analyzer.main)


@pytest.mark.parametrize(
    "marker",
    (
        "runtime/ready.rank0",
        "runtime/pypto_kvpool_map.json.rank0.done",
    ),
)
def test_publication_rejects_missing_normal_runtime_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
) -> None:
    policy = _policy_with_normal_records(tmp_path)
    policy_value = json.loads(
        Path(policy["policy_path"]).read_text(encoding="utf-8")
    )
    performance_path = Path(
        policy_value["authority"]["normal_performance_report"]["path"]
    )
    run = (
        performance_path.parent
        / "runs"
        / "candidate-r2-normal-bs2-64k"
    )
    (run / marker).unlink()
    campaign = fixtures._campaign(
        tmp_path,
        route_ready=False,
        matched_policy=policy,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
        ],
    )
    _expect_publication_rejected(analyzer.main)


def test_publication_rejects_forged_dfx_key_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy_with_normal_records(tmp_path)
    campaign = fixtures._campaign(
        tmp_path,
        route_ready=False,
        matched_policy=policy,
    )
    done_path = (
        campaign
        / "runs"
        / "candidate-r1-dfx-bs1-64k"
        / "runtime"
        / "pypto_kvpool_map.json.rank0.done"
    )
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["key_path"] = "/forged/not-the-captured-key.rank0"
    _write_json(done_path, done)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
        ],
    )
    _expect_publication_rejected(analyzer.main)


def test_publication_rejects_forged_normal_key_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = fixtures._matched_policy(tmp_path)
    policy = analyzer.load_matched_dfx_policy(initial["policy_path"])
    performance_path = Path(
        policy["authority"]["normal_performance_report"]["path"]
    )
    done_path = (
        performance_path.parent
        / "runs"
        / "candidate-r2-normal-bs2-64k"
        / "runtime"
        / "pypto_kvpool_map.json.rank0.done"
    )
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["key_path"] = "/forged/not-the-captured-key.rank0"
    _write_json(done_path, done)
    campaign = fixtures._campaign(
        tmp_path,
        route_ready=False,
        matched_policy=policy,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
        ],
    )
    _expect_publication_rejected(analyzer.main)


@pytest.mark.parametrize("source", ("dfx", "normal"))
def test_publication_rejects_missing_runtime_marker_hash_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    policy = _policy_with_normal_records(tmp_path)
    if source == "dfx":
        campaign = fixtures._campaign(
            tmp_path,
            route_ready=False,
            matched_policy=policy,
        )
        validation_path = (
            campaign
            / "runs"
            / "candidate-r1-dfx-bs1-64k"
            / "artifact_validation.json"
        )
    else:
        policy_value = json.loads(
            Path(policy["policy_path"]).read_text(encoding="utf-8")
        )
        performance_path = Path(
            policy_value["authority"]["normal_performance_report"]["path"]
        )
        campaign = fixtures._campaign(
            tmp_path,
            route_ready=False,
            matched_policy=policy,
        )
        validation_path = (
            performance_path.parent
            / "runs"
            / "candidate-r2-normal-bs2-64k"
            / "artifact_validation.json"
        )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation.pop("runtime_marker_sha256", None)
    _write_json(validation_path, validation)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_dfx_campaign.py",
            "--campaign",
            str(campaign),
            "--matched-policy",
            policy["policy_path"],
        ],
    )
    _expect_publication_rejected(analyzer.main)
