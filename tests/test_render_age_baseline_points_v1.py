"""Synthetic scalar-schema tests; not real probe or native-hash evidence."""

import copy
import json
import sys

import pytest

from scripts import render_age_baseline_points_v1 as render


def native():
    probes = {}
    for key in render.EXPECTED:
        probes["synthetic/" + key] = dict(probe_mae_image=5., probe_mae_person=6.,
                                       baselines=dict(mean=dict(mae_image=8., mae_person=9.),
                                                      median=dict(mae_image=7., mae_person=8.)))
    return dict(experiment=render.EXPERIMENT,
                parameters=dict(device="cpu", constants="equal-person fit-only mean/median"),
                metrics=dict(execution_complete=True, publication_ready=False, probes=probes))


def test_deduplicated_models_keep_denominators_and_scope():
    payload = render.collect([native(), native()])
    assert payload["native_probe_records"] == 22 and payload["unique_model_points"] == 11
    assert payload["models"]["weak/frozen"] == dict(mae_image=5., mae_person=6.)
    text = render.table(payload)
    assert "5.0000 & 6.0000" in text and "8.0000 & 9.0000" in text
    assert "Image-weighted" in text and "Equal-person" in text
    assert "no new interval" in text and "not a controlled causal" in text
    assert payload["publication_ready"] is False and payload["mechanism_complete"] is False


@pytest.mark.parametrize("change", ["nan", "negative", "boolean", "missing_denominator",
                                   "wrong_reference", "incomplete", "missing_model", "injection"])
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
        probes.pop(key)
    else:
        probes["synthetic/weak/untrusted\\input{secret}"] = probes.pop(key)
    with pytest.raises((ValueError, KeyError)):
        render.collect([data])


def test_duplicate_model_points_cannot_disagree():
    first, second = native(), native()
    second["metrics"]["probes"]["synthetic/strong/frozen"]["probe_mae_person"] = 6.1
    with pytest.raises(ValueError, match="duplicate"):
        render.collect([first, second])


def test_individual_probe_aliases_are_not_new_models():
    first = native()
    point = copy.deepcopy(first["metrics"]["probes"]["synthetic/strong/random_tail_lr1e-06_s2"])
    first["metrics"]["probes"]["individual/random_tail_lr1e-06_s2/tuned_age_probe"] = point
    frozen = copy.deepcopy(first["metrics"]["probes"]["synthetic/strong/frozen"])
    first["metrics"]["probes"]["individual/random_tail_lr1e-06_s2/frozen_age_probe"] = frozen
    payload = render.collect([first])
    assert payload["native_probe_records"] == 13 and payload["unique_model_points"] == 11


@pytest.mark.parametrize("mutation", ["before_writer", "after_writer", "deleted_after_writer"])
def test_new_manifest_removed_if_source_changes_at_publication(tmp_path, monkeypatch, mutation):
    from age_gap.common.manifest import file_record

    source = tmp_path / "source.py"
    source.write_text("fixture only", encoding="utf-8")
    for name in ("scripts/run_oriented_campaign.py", "scripts/run_restricted_matched_campaign.py",
                 "scripts/evaluate_oriented_cuda_v1.py", "src/age_gap/common/manifest.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture dependency", encoding="utf-8")
    payload = native()
    payload.update(inputs=[file_record(source)], outputs=[])
    binding = tmp_path / "native.manifest.json"
    binding.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(render, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(render, "__file__", str(source))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["fixture", "--bindings", str(binding), "--out", str(out)])
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
