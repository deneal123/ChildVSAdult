"""Working-master layout assets remain native-bound and scientifically scoped.

These are local byte/source linkage checks, not image rights, scientific-result
validation, final-bundle acceptance or portal-proof inspection.
"""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT = ROOT / "latex/papers/journal-1-tbiom/en/supplement.tex"


def check_record(root, record):
    path = Path(record["path"])
    path = path if path.is_absolute() else root / path
    assert type(record["bytes"]) is int
    assert record["bytes"] == path.stat().st_size
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    return path.resolve()


@pytest.mark.parametrize("kind", ["external", "scaling"])
def test_working_master_uses_exact_bound_layout_asset_without_science_clearance(kind):
    directory = ROOT / f"metrics/{kind}_layout_v2_20261008"
    manifest = json.loads((directory / "summary.manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert manifest["experiment"] == f"{kind}-layout-only-v2"
    assert manifest["metrics"] == summary
    assert summary["publication_ready"] is False
    assert summary["scientific_evidence_validated"] is False
    assert "historical" in summary["scope"]
    assert len(manifest["inputs"]) == len(manifest["outputs"]) == 3
    inputs = {check_record(ROOT, record) for record in manifest["inputs"]}
    outputs = {check_record(ROOT, record) for record in manifest["outputs"]}
    assert inputs == {
        (ROOT / f"scripts/render_{kind}_layout_v2.py").resolve(),
        (ROOT / "scripts/make_figures.py").resolve(),
        (ROOT / f"latex/shared/figures/fig_{kind}.pdf").resolve(),
    }
    assert outputs == {
        (directory / "summary.json").resolve(),
        (ROOT / f"latex/shared/figures/fig_{kind}_layout_v2.pdf").resolve(),
        (ROOT / f"latex/shared/figures/fig_{kind}_layout_v2.png").resolve(),
    }
    source = SUPPLEMENT.read_text(encoding="utf-8")
    assert source.count("{" + f"fig_{kind}_layout_v2.pdf" + "}") == 1
    assert "{" + f"fig_{kind}.pdf" + "}" not in source


@pytest.mark.parametrize("corruption", ["size", "same_size_bytes", "boolean_size"])
def test_layout_byte_guard_rejects_corrupted_native_record(tmp_path, corruption):
    path = tmp_path / "layout.pdf"
    path.write_bytes(b"synthetic")
    record = dict(path="layout.pdf", bytes=9,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if corruption == "size":
        record["bytes"] = 10
    elif corruption == "boolean_size":
        record["bytes"] = True
    else:
        path.write_bytes(b"SYNTHETIC")
    with pytest.raises(AssertionError):
        check_record(tmp_path, record)
