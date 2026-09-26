"""Machine-readable provenance manifests for publication experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from age_gap.common.io import PROJECT_ROOT

SCHEMA_VERSION = 1


def sha256_file(path: Path | str) -> str:
    """Return a streaming SHA-256 digest for an existing file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def file_record(path: Path | str) -> dict[str, Any]:
    """Describe an input/output file without embedding its contents."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": _display_path(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for package in ("numpy", "torch", "torchvision", "scikit-learn", "opencv-python-headless"):
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = None
    return out


def write_experiment_manifest(
    path: Path | str,
    *,
    experiment: str,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    inputs: list[Path | str],
    outputs: list[Path | str],
    command: list[str] | None = None,
) -> Path:
    """Write an atomic provenance manifest for one completed experiment."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": command if command is not None else sys.argv,
        "parameters": parameters,
        "metrics": metrics,
        "inputs": [file_record(item) for item in inputs],
        "outputs": [file_record(item) for item in outputs],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "git": _git_state(),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
