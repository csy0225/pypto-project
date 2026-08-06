from dataclasses import dataclass

import pytest

from image_capability_probe import inspect_reuse_capability


@dataclass
class ReuseCapableRunConfig:
    platform: str
    enable_l2_swimlane: bool
    enable_dep_gen: bool
    l2_swimlane_reuse_dep_gen: bool


@dataclass
class LegacyRunConfig:
    platform: str
    enable_l2_swimlane: bool
    enable_dep_gen: bool


@pytest.mark.parametrize("requirement_env", [None, "1"])
def test_formal_probe_accepts_capability(requirement_env):
    report = inspect_reuse_capability(
        ReuseCapableRunConfig,
        require_reuse=True,
        requirement_env=requirement_env,
    )
    assert report["fields_available"] is True
    assert report["reuse_config_constructed"] is True


@pytest.mark.parametrize("requirement_env", ["0", "invalid"])
def test_formal_probe_rejects_disabled_or_invalid_environment(
    requirement_env,
):
    with pytest.raises(AssertionError):
        inspect_reuse_capability(
            ReuseCapableRunConfig,
            require_reuse=True,
            requirement_env=requirement_env,
        )


def test_formal_probe_rejects_missing_capability():
    with pytest.raises(AssertionError):
        inspect_reuse_capability(
            LegacyRunConfig,
            require_reuse=True,
            requirement_env=None,
        )


@pytest.mark.parametrize("requirement_env", [None, "0", "1"])
def test_normal_probe_does_not_require_capability(requirement_env):
    report = inspect_reuse_capability(
        LegacyRunConfig,
        require_reuse=False,
        requirement_env=requirement_env,
    )
    assert report["fields_available"] is False
    assert report["reuse_config_constructed"] is False


def test_normal_probe_rejects_invalid_environment():
    with pytest.raises(AssertionError):
        inspect_reuse_capability(
            LegacyRunConfig,
            require_reuse=False,
            requirement_env="invalid",
        )
