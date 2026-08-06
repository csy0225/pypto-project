from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import verify_exact_tree_manifest as verifier


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> None:
    (root / "nested").mkdir()
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    (root / "nested" / "b.txt").write_text("b\n", encoding="utf-8")
    lines = [
        f"{_sha256(path)}  {path.relative_to(root)}"
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    (root / "SOURCE_SHA256SUMS").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def test_exact_tree_accepts_complete_manifest(tmp_path: Path) -> None:
    _fixture(tmp_path)
    entries = verifier.verify_exact_tree(tmp_path)
    assert set(entries) == {"a.txt", "nested/b.txt"}


def test_exact_tree_rejects_extra_file(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "extra.py").write_text("extra\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="extra=.*extra.py"):
        verifier.verify_exact_tree(tmp_path)


def test_exact_tree_rejects_changed_file(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="changed=.*a.txt"):
        verifier.verify_exact_tree(tmp_path)


def test_exact_tree_rejects_symlink(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "link").symlink_to(tmp_path / "a.txt")
    with pytest.raises(AssertionError, match="symlink mismatch"):
        verifier.verify_exact_tree(tmp_path)


def test_exact_tree_accepts_declared_internal_symlink(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    (tmp_path / "link").symlink_to("a.txt")
    (tmp_path / verifier.SOURCE_SYMLINK_MANIFEST).write_text(
        "link\ta.txt\n",
        encoding="utf-8",
    )
    lines = [
        f"{_sha256(path)}  {path.relative_to(tmp_path)}"
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.name != "SOURCE_SHA256SUMS"
    ]
    (tmp_path / "SOURCE_SHA256SUMS").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    verifier.verify_exact_tree(tmp_path)


def test_exact_tree_accepts_named_manifest(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('ok')\n", encoding="utf-8")
    manifest = tmp_path / "SCRIPTS_SHA256SUMS"
    manifest.write_text(
        f"{_sha256(tmp_path / 'a.py')}  a.py\n",
        encoding="utf-8",
    )
    entries = verifier.verify_exact_tree(
        tmp_path,
        manifest_name=manifest.name,
        symlink_manifest_name=None,
    )
    assert entries == {"a.py": _sha256(tmp_path / "a.py")}


def test_manifest_projection_drops_unmanifested_files_and_symlinks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "kept.txt").write_text("kept\n", encoding="utf-8")
    (source / "extra.pyc").write_bytes(b"stale")
    (source / "link").symlink_to("kept.txt")
    (source / verifier.SOURCE_MANIFEST).write_text(
        f"{_sha256(source / 'kept.txt')}  kept.txt\n",
        encoding="utf-8",
    )
    destination = tmp_path / "destination"
    verifier.materialize_manifest_projection(source, destination)
    assert (destination / "kept.txt").read_text(encoding="utf-8") == "kept\n"
    assert not (destination / "extra.pyc").exists()
    assert not (destination / "link").exists()
