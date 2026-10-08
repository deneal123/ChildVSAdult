"""Synthetic scalar-schema tests for the v2 explicit-cell absolute age presenter.

Every fixture here is generated in-memory or in a tiny temporary directory. There
is no GPU, no real faces/weights/model inference and no training, so a passing
suite is NOT native execution evidence and does not rehash any real artifact.
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROPOSAL = Path(__file__).resolve().parents[1]

from scripts import render_age_baseline_points_v2 as render  # noqa: E402

DECLARED = [
    "random_head_lr1e-06_s42",
    "random_head_lr1e-06_s1",
    "random_head_lr1e-06_s2",
    "random_tail_lr1e-06_s42",
    "random_tail_lr1e-06_s1",
    "random_tail_lr1e-06_s2",
    "random_head_lr1e-05_s42",
    "random_head_lr1e-05_s1",
]

REFERENCE = dict(mean=dict(mae_image=8., mae_person=9.),
                 median=dict(mae_image=7., mae_person=8.))


def native(declared=DECLARED):
    probes = {
        f"synthetic/{side}": dict(probe_mae_image=5., probe_mae_person=6.,
                                   baselines=copy.deepcopy(REFERENCE))
        for side in ("strong/frozen", "weak/frozen")
    }
    for seed in (42, 1, 2):
        probes[f"synthetic/weak/tuned_seed{seed}"] = dict(
            probe_mae_image=5., probe_mae_person=6., baselines=copy.deepcopy(REFERENCE))
    for cell in declared:
        probes[f"synthetic/{cell}/frozen_age_probe"] = dict(
            probe_mae_image=5., probe_mae_person=6., baselines=copy.deepcopy(REFERENCE))
        probes[f"synthetic/{cell}/tuned_age_probe"] = dict(
            probe_mae_image=5., probe_mae_person=6., baselines=copy.deepcopy(REFERENCE))
    return dict(experiment=render.EXPERIMENT,
                parameters=dict(device="cpu", constants="equal-person fit-only mean/median"),
                metrics=dict(execution_complete=True, publication_ready=False, probes=probes))


def test_explicit_cells_render_with_denominators_labels_and_disclaimers():
    payload = render.collect([native(), native()], DECLARED)
    assert payload["native_probe_records"] == 2 * (5 + 2 * len(DECLARED))
    assert payload["unique_model_points"] == 5 + len(DECLARED)
    assert payload["expected_strong_cells"] == 8 and payload["observed_strong_cells"] == 8
    assert payload["canonical_matrix_cells"] == 36 and payload["matrix_complete"] is False
    assert payload["models"]["weak/frozen"] == dict(mae_image=5., mae_person=6.)
    text = render.table(payload)
    assert "5.0000 & 6.0000" in text and "8.0000 & 9.0000" in text
    assert "Image-weighted" in text and "Equal-person" in text
    assert "Strong random head, LR $10^{-6}$, seed 42" in text
    assert "Strong random head, LR $10^{-5}$, seed 1" in text
    assert "incomplete" in text and "not a controlled causal" in text
    assert "no new interval" in text and "$10^{-6}$/$10^{-5}$" in text
    assert payload["publication_ready"] is False and payload["mechanism_complete"] is False


def test_declared_cells_are_exact_membership():
    other = [cell for cell in sorted(render.CANONICAL_CELLS) if cell not in set(DECLARED)][:2]
    payload = render.collect([native(other)], other)
    assert payload["expected_strong_cells"] == 2 and payload["observed_strong_cells"] == 2
    assert payload["matrix_complete"] is False


@pytest.mark.parametrize("change", ["nan", "negative", "boolean", "missing_denominator",
                                   "wrong_reference", "incomplete", "missing_model", "injection",
                                   "unexpected_cell", "missing_cell"])
def test_invalid_or_incompatible_points_rejected(change):
    data = native()
    probes = data["metrics"]["probes"]
    key = "synthetic/strong/frozen"
    if change in ("nan", "negative", "boolean"):
        probes[key]["probe_mae_person"] = {"nan": float("nan"), "negative": -1., "boolean": True}[change]
    elif change == "missing_denominator":
        del probes[key]["baselines"]["mean"]["mae_person"]
    elif change == "wrong_reference":
        probes[key]["baselines"]["mean"]["mae_person"] = 3.
    elif change == "incomplete":
        data["metrics"]["execution_complete"] = False
    elif change == "missing_model":
        probes.pop("synthetic/random_head_lr1e-06_s42/tuned_age_probe")
    elif change == "injection":
        probes["synthetic/weak/untrusted\\input{secret}"] = probes.pop(key)
    elif change == "unexpected_cell":
        cell = "lookalike_full_lr1e-05_s2"
        probes[f"synthetic/{cell}/tuned_age_probe"] = copy.deepcopy(probes[key])
        probes[f"synthetic/{cell}/frozen_age_probe"] = copy.deepcopy(probes[key])
    else:
        cell = "random_head_lr1e-06_s42"
        probes.pop(f"synthetic/{cell}/tuned_age_probe")
        probes.pop(f"synthetic/{cell}/frozen_age_probe")
    with pytest.raises((ValueError, KeyError)):
        render.collect([data], DECLARED)


@pytest.mark.parametrize("values,message", [
    ([], "at least one"),
    (["random_head_lr1e-06_s9"], "malformed"),
    (["random_all_lr1e-06_s42"], "malformed"),
    (["random_head_lr1e-06_s42", "random_head_lr1e-06_s42"], "duplicate"),
])
def test_expected_cell_declarations_are_validated(values, message):
    with pytest.raises(ValueError, match=message):
        render.expected_cells(values)


def test_declaration_mismatch_reports_unexpected_and_missing():
    data = native(["random_tail_lr1e-06_s42"])
    with pytest.raises(ValueError, match="unexpected strong cells.*missing strong cells"):
        render.collect([data], DECLARED)


def test_duplicate_model_points_cannot_disagree():
    first, second = native(), native()
    second["metrics"]["probes"]["synthetic/strong/frozen"]["probe_mae_person"] = 6.1
    with pytest.raises(ValueError, match="duplicate"):
        render.collect([first, second], DECLARED)


def test_individual_probe_aliases_are_not_new_models():
    data = native()
    point = copy.deepcopy(data["metrics"]["probes"]["synthetic/random_tail_lr1e-06_s2/tuned_age_probe"])
    data["metrics"]["probes"]["individual/random_tail_lr1e-06_s2/tuned_age_probe"] = point
    frozen = copy.deepcopy(data["metrics"]["probes"]["synthetic/strong/frozen"])
    data["metrics"]["probes"]["individual/random_tail_lr1e-06_s2/frozen_age_probe"] = frozen
    payload = render.collect([data], DECLARED)
    assert payload["observed_strong_cells"] == len(DECLARED)
    assert payload["unique_model_points"] == 5 + len(DECLARED)


def test_verify_records_rejects_missing_and_changed(tmp_path):
    from age_gap.common.manifest import file_record

    source = tmp_path / "source.json"
    source.write_text("fixture only", encoding="utf-8")
    record = file_record(source)
    render.verify_records([record], root=tmp_path)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        render.verify_records([record], root=tmp_path)
    source.unlink()
    with pytest.raises(ValueError, match="unavailable"):
        render.verify_records([record], root=tmp_path)
    with pytest.raises(ValueError, match="nonempty"):
        render.verify_records([], root=tmp_path)


def fixture_tree(tmp_path, monkeypatch):
    from age_gap.common.manifest import file_record

    source = tmp_path / "source.py"
    source.write_text("fixture only", encoding="utf-8")
    for name in ("scripts/common_pair_linkage_light_v1.py", "scripts/render_age_baseline_points_v1.py",
                 "scripts/run_oriented_campaign.py", "scripts/run_restricted_matched_campaign.py",
                 "scripts/evaluate_oriented_cuda_v1.py", "src/age_gap/common/manifest.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture dependency", encoding="utf-8")
    payload = native()
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(payload["metrics"]), encoding="utf-8")
    payload.update(inputs=[file_record(source)], outputs=[file_record(summary)])
    binding = tmp_path / "native.manifest.json"
    binding.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(render, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(render, "__file__", str(source))
    return source, binding


@pytest.mark.parametrize("mutation", ["before_writer", "after_writer", "deleted_after_writer"])
def test_new_manifest_removed_if_source_changes_at_publication(tmp_path, monkeypatch, mutation):
    source, binding = fixture_tree(tmp_path, monkeypatch)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "fixture", "--bindings", str(binding), "--out", str(out),
        "--expected-cells", *DECLARED])
    writer = render.write_experiment_manifest

    def mutated(*args, **kwargs):
        if mutation == "before_writer":
            source.write_text("changed", encoding="utf-8")
        result = writer(*args, **kwargs)
        if mutation == "after_writer":
            source.write_text("changed", encoding="utf-8")
        if mutation == "deleted_after_writer":
            source.unlink()
        return result

    monkeypatch.setattr(render, "write_experiment_manifest", mutated)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        render.main()
    assert not (out / "presentation.manifest.json").exists()
    assert binding.exists()


def test_tampered_completion_manifest_inputs_are_removed(tmp_path, monkeypatch):
    _source, binding = fixture_tree(tmp_path, monkeypatch)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "fixture", "--bindings", str(binding), "--out", str(out),
        "--expected-cells", *DECLARED])

    def tampered(target, **kwargs):
        kwargs["inputs"] = [*kwargs["inputs"], tmp_path / "src/age_gap/common/manifest.py"]
        return render.write_experiment_manifest(target, **kwargs)

    monkeypatch.setattr(render, "write_experiment_manifest", tampered)
    with pytest.raises(RuntimeError):
        render.main()
    assert not (out / "presentation.manifest.json").exists()


def test_fresh_output_root_is_required(tmp_path, monkeypatch):
    _source, binding = fixture_tree(tmp_path, monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileExistsError):
        render.run([binding], out, DECLARED)


def test_import_never_pulls_torch_or_opencv():
    code = (
        "import sys;"
        f"sys.path.insert(0, {str(PROPOSAL)!r});"
        "import scripts.render_age_baseline_points_v2 as r;"
        "assert 'torch' not in sys.modules and 'cv2' not in sys.modules;"
        "assert r.collect.__module__ == 'scripts.render_age_baseline_points_v2';"
        "print('lightweight')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "lightweight" in result.stdout


@pytest.mark.parametrize("declared", [[], DECLARED + [DECLARED[0]]])
def test_root_collect_validates_direct_declarations(declared):
    with pytest.raises(ValueError, match="at least one|duplicate"):
        render.collect([native()], declared)


def test_root_complete_coverage_does_not_claim_incomplete_matrix_or_mechanism():
    declared = sorted(render.CANONICAL_CELLS)
    payload = render.collect([native(declared)], declared)
    assert payload["matrix_complete"] is True
    assert payload["mechanism_complete"] is False
    assert "matrix stays incomplete" not in render.table(payload)
    assert "not a mechanism conclusion" in render.table(payload)


def test_root_bound_summary_must_equal_manifest_metrics(tmp_path, monkeypatch):
    from age_gap.common.manifest import file_record

    _, binding = fixture_tree(tmp_path, monkeypatch)
    payload = json.loads(binding.read_text(encoding="utf-8"))
    payload["metrics"]["publication_ready"] = True
    binding.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees"):
        render.verified(binding, render.EXPERIMENT)
    payload["outputs"] = []
    binding.write_text(json.dumps(payload), encoding="utf-8")
    assert file_record(tmp_path / "summary.json") not in payload["outputs"]
    with pytest.raises(ValueError, match="must be bound"):
        render.verified(binding, render.EXPERIMENT)
