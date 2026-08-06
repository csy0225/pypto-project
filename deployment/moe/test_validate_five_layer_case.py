from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


MOE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MOE_DIR))

import validate_five_layer_case as validator  # noqa: E402


def _authority_value() -> dict:
    evidence = {"path": "/tmp/evidence.json", "sha256": "a" * 64}
    return {
        "image_ref": "image@sha256:" + "9" * 64,
        "image_pypto_commit": validator.IMAGE_PYPTO_COMMIT,
        "selection_report": dict(evidence),
        "source_audit": dict(evidence),
        "scripts_manifest": dict(evidence),
        "normal_capture_scripts_manifest": dict(evidence),
        "normal_seal_authority": dict(evidence),
        "normal_correctness_report": dict(evidence),
        "normal_performance_report": dict(evidence),
        "normal_campaign_spec": dict(evidence),
        "normal_counterbalance_spec": dict(evidence),
        "analyzer_sha256": "1" * 64,
        "validator_sha256": "2" * 64,
        "golden_manifests": {
            str(batch): {
                "path": f"/tmp/golden/bs{batch}/manifest.json",
                "sha256": "3" * 64,
            }
            for batch in (1, 2, 4, 7, 8, 16)
        },
        "workload": {
            "layers": [0, 1, 2, 3, 4],
            "batches": [1, 2, 4, 7, 8, 16],
            "context_len_per_sequence": 65536,
            "blocks_per_sequence": 512,
            "hidden_outputs": ["hidden_l3", "hidden_l4"],
            "hidden_exact": True,
        },
        "capture": {
            "normal_rounds": 3,
            "normal_iters": 30,
            "normal_warmup": 5,
            "counterbalanced": True,
            "dfx_rounds": 1,
            "dfx_iters": 1,
            "dfx_warmup": 2,
            "l2_swimlane_reuse_dep_gen": True,
        },
    }


def _policy_value() -> dict:
    return {
        "schema": validator.MATCHED_DFX_POLICY_SCHEMA,
        "authority": _authority_value(),
        "sources": {
            "baseline": {
                "profile": "act-n64",
                "source_kind": "baseline",
                "source_role": "baseline",
                "policy_id": "moe-baseline-act-n64-v1",
                "source_manifest_sha256": "b" * 64,
                "decode_fwd_sha256": "d" * 64,
                "release_enforced": False,
            },
            "candidate": {
                "profile": "moe-winner",
                "source_kind": "candidate",
                "source_role": "candidate",
                "policy_id": "moe-winner-selected-tile-v1",
                "source_manifest_sha256": "c" * 64,
                "decode_fwd_sha256": "e" * 64,
                "release_enforced": True,
            },
        },
    }


def _write_policy(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_runtime_evidence_seal_is_strictly_additive(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact_validation.json"
    complete = {
        "passed": True,
        "runtime_marker_sha256": {"ready.rank0": "a" * 64},
        "kv_key_sha256_by_rank": {"0": "b" * 64},
    }
    legacy = {"passed": True}
    legacy_sha = validator._sha256_bytes(
        validator._canonical_json_bytes(legacy)
    )
    complete_sha = validator._sha256_bytes(
        validator._canonical_json_bytes(complete)
    )
    path.write_text(
        json.dumps(legacy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validator._write_stable(
        path,
        complete,
        seal_runtime_evidence=True,
        seal_authority_record={
            "legacy_validation_sha256": legacy_sha,
            "sealed_validation_sha256": complete_sha,
        },
    )
    assert json.loads(path.read_text(encoding="utf-8")) == complete

    path.write_text(
        json.dumps({"passed": False}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="non-additive"):
        validator._write_stable(
            path,
            complete,
            seal_runtime_evidence=True,
            seal_authority_record={
                "legacy_validation_sha256": legacy_sha,
                "sealed_validation_sha256": complete_sha,
            },
        )


def test_runtime_evidence_seal_requires_existing_regular_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.json"
    value = {
        "passed": True,
        "runtime_marker_sha256": {"ready.rank0": "a" * 64},
        "kv_key_sha256_by_rank": {"0": "b" * 64},
    }
    legacy = dict(value)
    legacy.pop("runtime_marker_sha256")
    legacy.pop("kv_key_sha256_by_rank")
    with pytest.raises(AssertionError, match="existing regular file"):
        validator._write_stable(
            path,
            value,
            seal_runtime_evidence=True,
            seal_authority_record={
                "legacy_validation_sha256": validator._sha256_bytes(
                    validator._canonical_json_bytes(legacy)
                ),
                "sealed_validation_sha256": validator._sha256_bytes(
                    validator._canonical_json_bytes(value)
                ),
            },
        )


def test_runtime_evidence_seal_rejects_bool_int_type_confusion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact_validation.json"
    complete = {
        "passed": True,
        "active_batch": 1,
        "runtime_marker_sha256": {"ready.rank0": "a" * 64},
        "kv_key_sha256_by_rank": {"0": "b" * 64},
    }
    legacy = dict(complete)
    legacy.pop("runtime_marker_sha256")
    legacy.pop("kv_key_sha256_by_rank")
    path.write_text(
        json.dumps({**legacy, "active_batch": True}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="type mismatch"):
        validator._write_stable(
            path,
            complete,
            seal_runtime_evidence=True,
            seal_authority_record={
                "legacy_validation_sha256": validator._sha256_bytes(
                    validator._canonical_json_bytes(legacy)
                ),
                "sealed_validation_sha256": validator._sha256_bytes(
                    validator._canonical_json_bytes(complete)
                ),
            },
        )


@pytest.mark.parametrize("tampered_file", ("timing.json", "run_nonce.txt"))
def test_seal_authority_rejects_synchronized_old_evidence_tamper(
    tmp_path: Path,
    tampered_file: str,
) -> None:
    campaign = tmp_path / "campaign"
    run = campaign / "runs" / "candidate-r1-normal-bs1-64k"
    run.mkdir(parents=True)
    (run / "artifact_validation.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (run / "timing.json").write_text(
        '{"p50_ms": 1.0}\n',
        encoding="utf-8",
    )
    (run / "run_nonce.txt").write_text("nonce\n", encoding="utf-8")
    evidence = validator._run_evidence_sha256(run)
    record = {
        "legacy_validation_sha256": "a" * 64,
        "sealed_validation_sha256": "b" * 64,
        "evidence_sha256": evidence,
    }
    authority = tmp_path / "authority.json"
    authority.write_text(
        json.dumps(
            {
                "schema": validator.NORMAL_SEAL_AUTHORITY_SCHEMA,
                "campaign_root": str(campaign.resolve()),
                "runs": {run.name: record},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    trusted_sha = validator._sha256(authority)
    if tampered_file == "timing.json":
        (run / tampered_file).write_text(
            '{"p50_ms": 0.001}\n',
            encoding="utf-8",
        )
    else:
        (run / tampered_file).write_text("replayed\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="evidence_sha256"):
        validator._seal_authority_record(
            run=run,
            authority_path=str(authority),
            authority_sha256=trusted_sha,
        )


def test_load_matched_policy_accepts_generic_per_role_profiles(
    tmp_path: Path,
) -> None:
    path = tmp_path / "policy.json"
    _write_policy(path, _policy_value())
    policy = validator.load_matched_dfx_policy(path)
    assert policy["sources"]["baseline"]["profile"] == "act-n64"
    assert policy["sources"]["candidate"]["profile"] == "moe-winner"
    assert policy["authority"]["image_pypto_commit"] == (
        validator.IMAGE_PYPTO_COMMIT
    )
    assert len(policy["policy_sha256"]) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_candidate", "exactly baseline and candidate"),
        ("invalid_profile", "invalid sources.baseline.profile"),
        ("malformed_manifest", "source_manifest_sha256"),
        ("candidate_release_disabled", "release gate"),
        ("missing_authority", "authority"),
    ),
)
def test_load_matched_policy_rejects_incomplete_contract(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    value = copy.deepcopy(_policy_value())
    if mutation == "missing_candidate":
        del value["sources"]["candidate"]
    elif mutation == "invalid_profile":
        value["sources"]["baseline"]["profile"] = "bad profile"
    elif mutation == "malformed_manifest":
        value["sources"]["baseline"]["source_manifest_sha256"] = "bad"
    elif mutation == "candidate_release_disabled":
        value["sources"]["candidate"]["release_enforced"] = False
    else:
        del value["authority"]
    path = tmp_path / f"{mutation}.json"
    _write_policy(path, value)
    with pytest.raises(AssertionError, match=message):
        validator.load_matched_dfx_policy(path)


def _binding() -> tuple[
    dict,
    dict,
    dict,
    dict,
    dict,
]:
    policy = _policy_value()["sources"]["candidate"]
    source = {
        "source_manifest_sha256": policy["source_manifest_sha256"],
        "decode_fwd_sha256": policy["decode_fwd_sha256"],
    }
    source_policy = {
        "policy_id": policy["policy_id"],
        "source_role": policy["source_role"],
        "enforce_candidate_release_gate": policy["release_enforced"],
        "decode_sha256_prefix": policy["decode_fwd_sha256"][:8],
    }
    expert_release = {
        "profile": policy["profile"],
        "source_policy": dict(source_policy),
        "release_enforced": policy["release_enforced"],
    }
    admission = {
        "profile": policy["profile"],
        "source_policy": dict(source_policy),
        "expert_release_enforced": policy["release_enforced"],
    }
    return policy, source, source_policy, expert_release, admission


def test_validate_matched_policy_binding_accepts_complete_contract() -> None:
    policy, source, source_policy, expert_release, admission = _binding()
    validator.validate_matched_policy_binding(
        source_kind="candidate",
        profile="moe-winner",
        policy=policy,
        source=source,
        source_policy=source_policy,
        expert_release=expert_release,
        admission=admission,
    )


@pytest.mark.parametrize(
    ("target", "message"),
    (
        ("manifest", "source manifest"),
        ("decode", "decode hash"),
        ("nested_policy", "admission source policy"),
        ("admission_release", "admission release enforcement"),
    ),
)
def test_validate_matched_policy_binding_fails_closed(
    target: str,
    message: str,
) -> None:
    policy, source, source_policy, expert_release, admission = _binding()
    if target == "manifest":
        source["source_manifest_sha256"] = "f" * 64
    elif target == "decode":
        source["decode_fwd_sha256"] = "f" * 64
    elif target == "nested_policy":
        admission["source_policy"]["policy_id"] = "wrong-policy"
    else:
        admission["expert_release_enforced"] = False
    with pytest.raises(AssertionError, match=message):
        validator.validate_matched_policy_binding(
            source_kind="candidate",
            profile="moe-winner",
            policy=policy,
            source=source,
            source_policy=source_policy,
            expert_release=expert_release,
            admission=admission,
        )


def _write_image_identity_fixture(
    run: Path,
    *,
    image_ref: str,
    mode: str,
) -> None:
    run.mkdir(parents=True, exist_ok=True)
    audit = "\n".join(
        (
            "[audit] pin pypto      "
            "8e92b46808f9f7c09b6431ad4691503f09c12ee5 clean",
            "[audit] pin pypto-lib  "
            "c9af5790d5fe450e14fd43c88099b87539089d17 clean",
            "[audit] git credential scrub: PASS",
            "[audit] attention profile: a2a3",
            "[audit] prepared swimlane reuse capability: "
            "{'available': True, 'constructed': True, 'required': '1'}",
            "[audit] build jobs: 2 (resource only)",
            "IMAGE_IMMUTABLE_AUDIT=PASS",
        )
    ) + "\n"
    (run / "image_audit.log").write_text(audit, encoding="utf-8")
    audit_sha = validator._sha256(run / "image_audit.log")
    (run / "image_audit_invocation.json").write_text(
        json.dumps(
            {
                "audit_log_sha256": audit_sha,
                "image_ref": image_ref,
                "passed": True,
                "phase": "pre-source-mount",
                "schema": validator.IMAGE_AUDIT_SCHEMA,
                "source_mount": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "capability_report.json").write_text(
        json.dumps(
            {
                "attention_profile": "a2a3",
                "image_commits": {
                    "pypto": validator.IMAGE_PYPTO_COMMIT,
                    "pypto_lib": validator.IMAGE_PYPTO_LIB_COMMIT,
                },
                "image_ref": image_ref,
                "pypto_git_head": validator.IMAGE_PYPTO_COMMIT,
                "reuse_capability": {
                    "environment_present": mode == "dfx",
                    "environment_value": "1" if mode == "dfx" else None,
                    "fields_available": True,
                    "required": mode == "dfx",
                    "reuse_config_constructed": True,
                },
                "schema": validator.CAPABILITY_SCHEMA,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_pre_mount_audit_and_capability_are_identity_bound(
    tmp_path: Path,
) -> None:
    run = tmp_path / "candidate-r1-dfx-bs1-64k"
    image_ref = "hub.i.basemind.com/stepcast/vllm-pypto@sha256:" + "a" * 64
    _write_image_identity_fixture(run, image_ref=image_ref, mode="dfx")
    audit = validator._validate_pre_mount_image_audit(
        run,
        image_ref=image_ref,
    )
    capability = validator._validate_capability_report(
        run,
        image_ref=image_ref,
        mode="dfx",
    )
    assert len(audit["image_audit_log_sha256"]) == 64
    assert len(capability["capability_report_sha256"]) == 64

    invocation = json.loads(
        (run / "image_audit_invocation.json").read_text(encoding="utf-8")
    )
    invocation["source_mount"] = True
    (run / "image_audit_invocation.json").write_text(
        json.dumps(invocation) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="source_mount"):
        validator._validate_pre_mount_image_audit(
            run,
            image_ref=image_ref,
        )


def test_dfx_raw_evidence_requires_eight_merged_rank_traces(
    tmp_path: Path,
) -> None:
    batch_dir = tmp_path / "runtime" / "bs1"
    raw = batch_dir / "dfx_raw"
    dep_hashes = {}
    swim_hashes = {}
    for rank in range(8):
        dispatch = raw / f"rank{rank}" / "d0"
        dispatch.mkdir(parents=True)
        for name, content in (
            ("deps.json", "{}\n"),
            ("l2_swimlane_records.json", "{}\n"),
            ("name_map.json", "{}\n"),
            ("critical_path_report.md", "# trace\n"),
        ):
            (dispatch / name).write_text(content, encoding="utf-8")
        (dispatch / "merged_swimlane_fixture.json").write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {
                            "dur": 1.0,
                            "name": f"rank{rank}-task",
                            "ph": "X",
                            "ts": 0.0,
                        }
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        dep_hashes[
            f"dfx_outputs/rank{rank}/d0/deps.json"
        ] = validator._sha256(dispatch / "deps.json")
        swim_hashes[
            f"dfx_outputs/rank{rank}/d0/l2_swimlane_records.json"
        ] = validator._sha256(dispatch / "l2_swimlane_records.json")

    artifacts = {
        "dep_gen_artifacts": dep_hashes,
        "dep_gen_preserved_after_swim": True,
        "swimlane_artifacts": swim_hashes,
    }
    evidence = validator._validate_dfx_raw_evidence(
        batch_dir,
        dfx_artifacts=artifacts,
    )
    assert evidence["rank_count"] == 8
    assert evidence["merged_swimlane_count"] == 8
    assert set(evidence["sha256_by_rank"]) == {
        str(rank) for rank in range(8)
    }

    (raw / "rank7" / "d0" / "merged_swimlane_fixture.json").unlink()
    with pytest.raises(AssertionError, match="merged swimlane"):
        validator._validate_dfx_raw_evidence(
            batch_dir,
            dfx_artifacts=artifacts,
        )
