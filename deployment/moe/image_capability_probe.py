#!/usr/bin/env python3
"""Fail-closed image identity and prepared-swimlane capability probe."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import fields
from pathlib import Path
from typing import Any


def git_head(path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"],
        text=True,
    ).strip()


def inspect_reuse_capability(
    run_config_cls: type,
    *,
    require_reuse: bool,
    requirement_env: str | None,
) -> dict[str, Any]:
    if requirement_env not in (None, "0", "1"):
        raise AssertionError(
            "PYPTO_REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN must be unset, 0, or 1"
        )
    if require_reuse and requirement_env not in (None, "1"):
        raise AssertionError(
            "formal DFX forbids an explicit disabled reuse requirement"
        )

    field_names = {field.name for field in fields(run_config_cls)}
    required_fields = {
        "enable_chip_swimlane",
        "enable_dep_gen",
        "l2_swimlane_reuse_dep_gen",
    }
    fields_available = required_fields <= field_names
    constructed = False
    if fields_available:
        config = run_config_cls(
            platform="a2a3",
            enable_chip_swimlane=True,
            enable_dep_gen=False,
            l2_swimlane_reuse_dep_gen=True,
        )
        constructed = config.l2_swimlane_reuse_dep_gen is True

    if require_reuse and not (fields_available and constructed):
        raise AssertionError(
            "formal DFX requires prepared l2_swimlane dep-gen reuse"
        )

    return {
        "environment_present": requirement_env is not None,
        "environment_value": requirement_env,
        "fields_available": fields_available,
        "required": require_reuse,
        "required_fields": sorted(required_fields),
        "reuse_config_constructed": constructed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--expected-pypto", required=True)
    parser.add_argument("--expected-pypto-lib", required=True)
    parser.add_argument("--expected-attn-profile", required=True)
    parser.add_argument("--require-l2-swimlane-reuse", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from models.step3p5 import config as step3p5_config
    from pypto.runtime.runner import RunConfig

    actual_pypto = git_head("/workspace/pypto")
    image_pypto = os.environ.get("PYPTO_IMAGE_PYPTO_COMMIT")
    image_pypto_lib = os.environ.get("PYPTO_IMAGE_PYPTO_LIB_COMMIT")
    profile_env = os.environ.get("PYPTO_STEP3P5_ATTN_TASK_PROFILE")

    assert actual_pypto == args.expected_pypto, (
        actual_pypto,
        args.expected_pypto,
    )
    assert image_pypto == args.expected_pypto, (
        image_pypto,
        args.expected_pypto,
    )
    assert image_pypto_lib == args.expected_pypto_lib, (
        image_pypto_lib,
        args.expected_pypto_lib,
    )
    assert profile_env == args.expected_attn_profile, (
        profile_env,
        args.expected_attn_profile,
    )
    assert step3p5_config.ATTN_TASK_PROFILE == args.expected_attn_profile, (
        step3p5_config.ATTN_TASK_PROFILE,
        args.expected_attn_profile,
    )

    reuse = inspect_reuse_capability(
        RunConfig,
        require_reuse=args.require_l2_swimlane_reuse,
        requirement_env=os.environ.get(
            "PYPTO_REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN"
        ),
    )
    report = {
        "attention_profile": args.expected_attn_profile,
        "image_commits": {
            "pypto": image_pypto,
            "pypto_lib": image_pypto_lib,
        },
        "image_ref": args.image_ref,
        "pypto_git_head": actual_pypto,
        "reuse_capability": reuse,
        "schema": "step3p5.moe-image-capability.v1",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
