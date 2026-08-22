#!/usr/bin/env python3
"""Fail closed when an image config/history contains credential material."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# 0162 has no docker -- it runs containerd + nerdctl. Same subcommands, so the
# caller only has to name the CLI.
_CLI = shlex.split(os.environ.get("CONTAINER_CLI", "docker"))


def docker_output(*args: str) -> bytes:
    return subprocess.check_output(
        [*_CLI, *args], stderr=subprocess.STDOUT
    )


def fingerprint(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()[:12]


def is_literal_secret(value: bytes) -> bool:
    value = value.strip(b"'\"")
    if len(value) < 12:
        return False
    # BuildKit keeps Dockerfile variable references in history. They are not
    # secret values and must not be confused with a literal credential.
    if any(marker in value for marker in (b"$", b"{", b"}", b"<", b">", b"*")):
        return False
    lowered = value.lower()
    return lowered not in {
        b"none",
        b"null",
        b"unset",
        b"redacted",
        b"changeme",
        b"placeholder",
    }


if len(sys.argv) < 3:
    raise SystemExit(
        f"usage: {sys.argv[0]} IMAGE CREDENTIAL_FILE [CREDENTIAL_FILE ...]"
    )

image = sys.argv[1]
credential_paths = [Path(arg) for arg in sys.argv[2:]]
history = docker_output(
    "history", "--no-trunc", "--format", "{{json .CreatedBy}}", image
)
inspect = docker_output("image", "inspect", image)
payloads = [("history", history), ("inspect", inspect)]

findings: set[tuple[str, str, int, str]] = set()
candidate_count = 0
for path in credential_paths:
    value = path.read_bytes().strip()
    if len(value) < 12:
        continue
    candidate_count += 1
    for payload_name, payload in payloads:
        count = payload.count(value)
        if count:
            findings.add(
                (
                    payload_name,
                    str(path),
                    count,
                    fingerprint(value),
                )
            )

# Also catch stale literal credentials that are no longer equal to today's
# credential files. Only fingerprints and lengths are reported.
patterns = {
    "credential_assignment": re.compile(
        rb"(?:ACCESS_TOKEN|AUTH_TOKEN|PASSWORD|SECRET|PRIVATE_KEY)="
        rb"([^\s\"']+)"
    ),
    "url_password": re.compile(rb"https?://[^/\s:@]+:([^@/\s]+)@"),
    "github_url_token": re.compile(rb"https?://([^/@\s]+)@github\.com/"),
}
for payload_name, payload in payloads:
    for kind, pattern in patterns.items():
        for match in pattern.finditer(payload):
            value = match.group(1).strip(b"'\"")
            if is_literal_secret(value):
                findings.add(
                    (
                        payload_name,
                        f"pattern:{kind}",
                        len(value),
                        fingerprint(value),
                    )
                )

if findings:
    for payload_name, source, count_or_length, digest_prefix in sorted(findings):
        print(
            "CREDENTIAL_AUDIT_FAIL"
            f" image={image} payload={payload_name} source={source}"
            f" count_or_length={count_or_length}"
            f" sha256_prefix={digest_prefix}"
        )
    raise SystemExit(2)

print(
    f"CREDENTIAL_AUDIT_PASS image={image}"
    f" credential_files_checked={candidate_count}"
)
