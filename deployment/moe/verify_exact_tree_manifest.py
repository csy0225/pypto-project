#!/usr/bin/env python3
"""Verify that a SHA256 manifest describes the complete regular-file tree."""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path


HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
SOURCE_MANIFEST = "SOURCE_SHA256SUMS"
SOURCE_SYMLINK_MANIFEST = "SOURCE_SYMLINKS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_entries(root: Path, manifest_name: str) -> dict[str, str]:
    manifest = root / manifest_name
    if not manifest.is_file():
        raise AssertionError(f"missing manifest: {manifest}")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise AssertionError(
                f"{manifest}:{line_number}: malformed manifest entry"
            )
        digest, relative = parts
        relative = relative.lstrip("*")
        if relative.startswith("./"):
            relative = relative[2:]
        path = Path(relative)
        if (
            not HEX_SHA256.fullmatch(digest)
            or path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or relative in entries
            or relative == manifest_name
        ):
            raise AssertionError(
                f"{manifest}:{line_number}: invalid manifest entry"
            )
        entries[relative] = digest
    return entries


def verify_exact_tree(
    root: Path,
    *,
    manifest_name: str = SOURCE_MANIFEST,
    symlink_manifest_name: str | None = SOURCE_SYMLINK_MANIFEST,
) -> dict[str, str]:
    root = root.resolve()
    manifest_path = root / manifest_name
    entries = _manifest_entries(root, manifest_name)
    actual_symlinks = {
        str(path.relative_to(root)): path.readlink().as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_symlink()
    }
    symlink_manifest = (
        root / symlink_manifest_name
        if symlink_manifest_name is not None
        else None
    )
    declared_symlinks: dict[str, str] = {}
    if symlink_manifest is not None and symlink_manifest.is_file():
        for line_number, line in enumerate(
            symlink_manifest.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                raise AssertionError(
                    f"{symlink_manifest}:{line_number}: malformed entry"
                )
            relative, target = parts
            relative_path = Path(relative)
            target_path = Path(target)
            if (
                relative_path.is_absolute()
                or not relative_path.parts
                or ".." in relative_path.parts
                or target_path.is_absolute()
                or relative in declared_symlinks
            ):
                raise AssertionError(
                    f"{symlink_manifest}:{line_number}: invalid entry"
                )
            resolved_target = (
                root / relative_path.parent / target_path
            ).resolve()
            try:
                resolved_target.relative_to(root)
            except ValueError as error:
                raise AssertionError(
                    f"{symlink_manifest}:{line_number}: target escapes root"
                ) from error
            declared_symlinks[relative] = target
    if declared_symlinks != actual_symlinks:
        raise AssertionError(
            f"{root}: exact-tree symlink mismatch; "
            f"declared={declared_symlinks}, actual={actual_symlinks}"
        )
    actual = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path != manifest_path
    }
    if entries != actual:
        missing = sorted(set(entries) - set(actual))
        extra = sorted(set(actual) - set(entries))
        changed = sorted(
            relative
            for relative in set(entries) & set(actual)
            if entries[relative] != actual[relative]
        )
        raise AssertionError(
            f"{root}: exact-tree manifest mismatch; "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return entries


def materialize_manifest_projection(
    source: Path,
    destination: Path,
    *,
    manifest_name: str = SOURCE_MANIFEST,
) -> dict[str, str]:
    """Copy only regular files explicitly covered by the source manifest."""
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    entries = _manifest_entries(source, manifest_name)
    destination.mkdir(parents=True)
    for relative, expected_digest in entries.items():
        source_path = source / relative
        if not source_path.is_file() or source_path.is_symlink():
            raise AssertionError(
                f"{source}: manifest entry is not a regular file: {relative}"
            )
        actual_digest = _sha256(source_path)
        if actual_digest != expected_digest:
            raise AssertionError(
                f"{source}: manifest hash mismatch: {relative}"
            )
        destination_path = destination / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    shutil.copy2(source / manifest_name, destination / manifest_name)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", default=SOURCE_MANIFEST)
    parser.add_argument(
        "--symlink-manifest",
        default=SOURCE_SYMLINK_MANIFEST,
        help="set to '-' to reject every symlink without a declaration file",
    )
    parser.add_argument(
        "--materialize",
        default="",
        help="copy the manifest-covered regular-file projection to this path",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    symlink_manifest = (
        None if args.symlink_manifest == "-" else args.symlink_manifest
    )
    if args.materialize:
        entries = materialize_manifest_projection(
            root,
            Path(args.materialize),
            manifest_name=args.manifest,
        )
        print(
            f"MANIFEST_PROJECTION=PASS files={len(entries)} "
            f"root={root} out={Path(args.materialize).resolve()}"
        )
        return 0
    entries = verify_exact_tree(
        root,
        manifest_name=args.manifest,
        symlink_manifest_name=symlink_manifest,
    )
    print(f"EXACT_TREE_MANIFEST=PASS files={len(entries)} root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
