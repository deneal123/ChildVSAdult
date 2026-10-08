"""Synthetic-only checks for the common-mechanism presentation draft.

CPU-only, in-memory dictionaries and tiny temp files: no real binding is executed, no crop
tree is hashed, no GPU probe, no training.  The draft module is loaded by path so the
proposal stays inside the artifact directory and no canonical source is touched.  Every
fixture below is derived from the documented native schema, never from real metrics.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from age_gap.common.manifest import file_record
from scripts import render_common_mechanism_v2 as render


def test_renderer_uses_canonical_matrix_linkage_auditor(monkeypatch):
    monkeypatch.undo()  # Undo the synthetic-only linkage stub for this identity check.
    from scripts.audit_common_pair_protocol_v3 import audit

    assert render.audit_shared_pairs is audit


def test_eight_completed_cells_include_high_lr_without_pooling():
    names = [f"random_{scope}_lr1e-06_s{seed}"
             for scope in ("head", "tail") for seed in (42, 1, 2)]
    names += ["random_head_lr1e-05_s42", "random_head_lr1e-05_s1"]
    payload = render.collect([native(name) for name in names])
    assert payload["observed_strong_cells"] == 8
    assert payload["mechanism_complete"] is False
    text = render.strong_table(payload)
    assert text.count("random, ") == 8
    assert "LR $10^{-5}$, seed 1" in text
    assert "8 of 36" in text
    assert "no mechanism decision" in text


@pytest.fixture(autouse=True)
def synthetic_pair_linkage(monkeypatch):
    """Tiny renderer tests stub pair checking; separate auditor tests exercise real arrays."""
    def fake(bindings):
        cells = [next(iter(json.loads(path.read_text())["metrics"]["results"]["strong"]["diagnostics"]))
                 for path in bindings]
        return dict(exact_pair_arrays_equal=True, n_images=650, n_pairs=5308,
                    checked_cells=sorted(cells)), set()
    monkeypatch.setattr(render, "audit_shared_pairs", fake)


def sep_model(smd=3.34, delta=0.0, ci=(3.14, 3.58), delta_ci=(0.0, 0.0)):
    return dict(negative_mean=.046, positive_mean=.487, negative_variance=.0054,
                positive_variance=.029, standardized_mean_difference=smd, ci95=list(ci),
                delta_vs_frozen=delta, delta_ci95=list(delta_ci))


def diagnostics(seed=0):
    return dict(
        representation=dict(mean_cosine_drift=.057, linear_cka=.947, n_images=650,
                            limitation="descriptive"),
        paired_age_error=dict(delta_mae_person=-.137, ci95=[-.64, .358], n_persons=82,
                              resamples=2000, seed=seed, limitation="conditional"),
    )


def tuned(name, *, smd=3.207, delta=-.135, delta_ci=(-.2456, -.0121)):
    """A tuned strong checkpoint whose delta is consistent with the frozen reference."""
    return sep_model(smd=smd, delta=delta, ci=(2.98, 3.47), delta_ci=delta_ci)


def weak_family(scale=1.0):
    models = {"frozen": sep_model(smd=1.791 * scale, ci=(1.62, 1.99))}
    for name, smd, frozen_smd in (("tuned_seed42", 2.063, 1.791), ("tuned_seed1", 2.068, 1.791),
                                  ("tuned_seed2", 2.076, 1.791)):
        models[name] = sep_model(smd=smd * scale,
                                 delta=(smd - frozen_smd) * scale,
                                 ci=(1.91, 2.24),
                                 delta_ci=(.11, .42))
    return dict(
        separability=dict(status="ok", version=render.SEPARATION_VERSION, models=models,
                          n_pairs=5308, n_persons=82, requested_draws=2000, valid_draws=2000,
                          seed=0, positive_weight="owner multiplicity once",
                          negative_weight="endpoint multiplicity product",
                          limitation="fixed image-pair cosine separation"),
        diagnostics={name: diagnostics() for name in render.WEAK_TUNED},
    )


def native(name="random_head_lr1e-06_s42"):
    frozen = sep_model(smd=3.342)
    strong = dict(
        separability=dict(status="ok", version=render.SEPARATION_VERSION,
                          models={"frozen": frozen, name: tuned(name)},
                          n_pairs=5308, n_persons=82, requested_draws=2000,
                          valid_draws=2000, seed=0,
                          positive_weight="owner multiplicity once",
                          negative_weight="endpoint multiplicity product",
                          limitation="fixed image-pair cosine separation"),
        diagnostics={name: diagnostics()},
    )
    return dict(
        experiment=render.EXPERIMENT,
        parameters=dict(device="cpu", blas_threads=1, age_alpha=1, age_folds=5,
                        age_fold_seed=42, bootstrap_seed=0, resamples=2000,
                        protocol="completed strong bound image-index pairs"),
        metrics=dict(execution_complete=True, publication_ready=False, results=dict(
            strong=strong, weak=weak_family()),
            n_images=650, n_pairs=5308,
            limitation="common fixed image pairs, conditional fixed checkpoints/folds; "
                       "incomplete strong matrix; no mechanism decision"),
    )


# --- collect(): derivation, dedupe and honest completeness flags -----------------------


def test_collect_derives_bound_fields_and_dedupes_weak_family():
    payload = render.collect([native()])
    assert payload["observed_strong_cells"] == 1
    assert payload["expected_strong_matrix_cells"] == 36
    assert payload["evaluated_strong_cells"] == 1
    assert payload["mechanism_complete"] is False
    assert payload["publication_ready"] is False
    assert payload["n_images"] == 650 and payload["n_pairs"] == 5308
    assert payload["resamples"] == 2000 and payload["bootstrap_seed"] == 0
    assert set(payload["strong"]["cells"]) == {"random_head_lr1e-06_s42"}
    # weak is bound once and is not duplicated per binding
    assert set(payload["weak"]["models"]) == {"frozen", *render.WEAK_TUNED}
    assert len(payload["weak"]["diagnostics"]) == 3


def test_six_distinct_strong_cells_are_all_retained_without_averaging():
    cells = ["random_head_lr1e-06_s42", "random_head_lr1e-06_s1", "random_head_lr1e-06_s2",
             "random_tail_lr1e-06_s42", "random_tail_lr1e-06_s1", "random_tail_lr1e-06_s2"]
    payload = render.collect([native(name) for name in cells])
    assert payload["observed_strong_cells"] == 6
    assert payload["expected_strong_matrix_cells"] == 36
    assert set(payload["strong"]["cells"]) == set(cells)
    assert len(payload["strong"]["diagnostics"]) == 6
    text = render.strong_table(payload)
    assert text.count("random, ") == 6  # one row per checkpoint, no pooled/mean row
    assert "mean" not in text.split(r"\midrule")[1]


# --- label discipline: canonical human labels, no raw CLI strings ----------------------


def test_tables_use_canonical_labels_and_carry_all_limits():
    payload = render.collect([native()])
    strong = render.strong_table(payload)
    weak = render.weak_table(payload)
    assert "random, head, LR $10^{-6}$, seed 42" in strong
    assert "_lr1e-06_" not in strong and "lr1e-06" not in strong
    assert "random_tail_lr1e-06_s2" not in strong
    assert "tuned, seed 42" in weak
    for text in (strong, weak):
        assert "not a seed mean" in text and "not an ensemble" in text
        assert "not age removal" in text
        assert "no training-seed population inference" in text
        assert "no mechanism decision" in text
        assert "not deployment-calibrated" in text or "no identity-independence clearance" in text
    assert r"Checkpoint & CKA" in strong and strong.count("&") == strong.count(r"\\") * 7


def test_weak_table_is_rendered_once_not_per_binding():
    payload = render.collect([native(), native("random_tail_lr1e-06_s1")])
    text = render.weak_table(payload)
    assert text.count(r"\begin{table*}") == 1
    assert text.count("tuned, seed ") == len(render.WEAK_TUNED)


# --- rejections: duplicates, changed references, protocol, shape, types ---------------


def test_duplicate_binding_and_duplicate_cell_rejected():
    with pytest.raises(ValueError, match="unique"):
        render.collect([native(), native()])


def test_changed_frozen_reference_rejected():
    second = native("random_tail_lr1e-06_s1")
    second["metrics"]["results"]["strong"]["separability"]["models"]["frozen"]["ci95"] = [.6, .9]
    with pytest.raises(ValueError, match="frozen reference"):
        render.collect([native(), second])


def test_changed_weak_family_rejected():
    second = native("random_tail_lr1e-06_s1")
    second["metrics"]["results"]["weak"]["separability"]["models"]["tuned_seed1"]["ci95"] = [2.0, 2.5]
    with pytest.raises(ValueError, match="weak reference"):
        render.collect([native(), second])


@pytest.mark.parametrize("field,value", [
    ("device", "cuda"), ("blas_threads", 4), ("age_folds", 3), ("age_fold_seed", 1),
    ("bootstrap_seed", 1), ("resamples", 1000), ("protocol", "something else"),
])
def test_changed_protocol_rejected(field, value):
    data = native()
    data["parameters"][field] = value
    with pytest.raises(ValueError, match="protocol"):
        render.collect([data])


@pytest.mark.parametrize("field,value", [("execution_complete", False),
                                         ("publication_ready", True),
                                         ("n_images", 649), ("n_pairs", 5307),
                                         ("limitation", "mechanism established")])
def test_incomplete_or_overclaimed_binding_rejected(field, value):
    data = native()
    data["metrics"][field] = value
    with pytest.raises(ValueError):
        render.collect([data])


@pytest.mark.parametrize("mutate", ["missing_tuned", "extra_model", "missing_diagnostics",
                                    "extra_diagnostics"])
def test_wrong_strong_shape_rejected(mutate):
    data = native()
    strong = data["metrics"]["results"]["strong"]
    if mutate == "missing_tuned":
        strong["separability"]["models"].pop("random_head_lr1e-06_s42")
    elif mutate == "extra_model":
        strong["separability"]["models"]["random_tail_lr1e-06_s1"] = tuned("x")
    elif mutate == "missing_diagnostics":
        strong["diagnostics"].pop("random_head_lr1e-06_s42")
    else:
        strong["diagnostics"]["random_tail_lr1e-06_s1"] = diagnostics()
    with pytest.raises(ValueError):
        render.collect([data])


@pytest.mark.parametrize("field,value", [
    ("linear_cka", float("nan")), ("mean_cosine_drift", float("inf")),
    ("mean_cosine_drift", True), ("n_images", 650.0), ("n_images", True),
])
def test_nonfinite_or_bool_diagnostics_rejected(field, value):
    data = native()
    data["metrics"]["results"]["strong"]["diagnostics"]["random_head_lr1e-06_s42"][
        "representation"][field] = value
    with pytest.raises(ValueError):
        render.collect([data])


def test_nan_bool_and_backwards_separation_rejected():
    for mutate in ("nan_smd", "bool_mean", "backwards_ci", "zero_frozen_delta"):
        data = native()
        models = data["metrics"]["results"]["strong"]["separability"]["models"]
        if mutate == "nan_smd":
            models["random_head_lr1e-06_s42"]["standardized_mean_difference"] = float("nan")
        elif mutate == "bool_mean":
            models["random_head_lr1e-06_s42"]["positive_mean"] = True
        elif mutate == "backwards_ci":
            models["random_head_lr1e-06_s42"]["delta_ci95"] = [-.01, -.2]
        else:
            models["frozen"]["delta_vs_frozen"] = .5
        with pytest.raises(ValueError):
            render.collect([data])


def test_inconsistent_delta_and_missing_field_rejected():
    data = native()
    data["metrics"]["results"]["strong"]["separability"]["models"]["random_head_lr1e-06_s42"][
        "delta_vs_frozen"] = .3
    with pytest.raises(ValueError, match="delta"):
        render.collect([data])
    data = native()
    del data["metrics"]["results"]["strong"]["separability"]["models"]["random_head_lr1e-06_s42"][
        "ci95"]
    with pytest.raises((KeyError, ValueError)):
        render.collect([data])


def test_wrong_separability_protocol_rejected():
    for field, value in (("version", "fixed-pair-subject-separability-v2"),
                         ("valid_draws", 1999), ("seed", 1),
                         ("positive_weight", "uniform")):
        data = native()
        data["metrics"]["results"]["strong"]["separability"][field] = value
        with pytest.raises(ValueError):
            render.collect([data])


def test_empty_collection_fails_closed():
    with pytest.raises(ValueError, match="at least one"):
        render.collect([])


# --- run(): real file IO on tiny synthetic manifests only -----------------------------


def binding_file(tmp_path, name="random_head_lr1e-06_s42", *, covered=None):
    """Write one synthetic binding manifest with declared, checksummed inputs."""
    payload = native(name)
    declared = covered or []
    if not declared:
        source = tmp_path / "dep.txt"
        source.write_text("frozen-dependency\n", encoding="utf-8")
        declared = [file_record(source)]
    payload["inputs"] = declared
    payload["outputs"] = []
    path = tmp_path / f"{name}.manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_missing_and_existing_output_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        render.run([tmp_path / "absent.manifest.json"], tmp_path / "out")
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileExistsError):
        render.run([binding_file(tmp_path)], out)


def test_tampered_declared_input_rejected(tmp_path):
    source = tmp_path / "dep.txt"
    source.write_text("frozen-dependency\n", encoding="utf-8")
    binding = binding_file(tmp_path, covered=[file_record(source)])
    source.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        render.run([binding], tmp_path / "out")


def test_final_source_mutation_refused(tmp_path, monkeypatch):
    binding = binding_file(tmp_path)
    original = render.strong_table

    def mutating(payload):
        binding.write_text(binding.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return original(payload)

    monkeypatch.setattr(render, "strong_table", mutating)
    with pytest.raises(RuntimeError, match="ancestry changed"):
        render.run([binding], tmp_path / "out")


def test_run_publishes_bound_manifest(tmp_path):
    binding = binding_file(tmp_path)
    out = tmp_path / "out"
    payload = render.run([binding], out)
    assert payload["observed_strong_cells"] == 1
    assert (out / "strong_table.tex").is_file() and (out / "weak_table.tex").is_file()
    manifest = json.loads((out / "presentation.manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment"] == "common-mechanism-per-checkpoint-presentation-v2"
    metrics = manifest["metrics"]
    assert metrics["mechanism_complete"] is False
    assert metrics["publication_ready"] is False
    assert metrics["expected_strong_matrix_cells"] == 36
    assert metrics["observed_strong_cells"] == 1
    names = {record["path"] for record in manifest["outputs"]}
    assert any(name.endswith("summary.json") for name in names)
    assert any(name.endswith("strong_table.tex") for name in names)
    assert any(name.endswith("weak_table.tex") for name in names)
    declared = {Path(record["path"]).name for record in manifest["inputs"]}
    assert binding.name in declared
    assert all(record["sha256"] and record["bytes"] > 0 for record in manifest["inputs"])


def test_source_is_not_mutated_by_a_normal_render(tmp_path):
    """A successful run must leave every declared input byte-identical."""
    binding = binding_file(tmp_path)
    payload = json.loads(binding.read_text(encoding="utf-8"))
    declared_before = copy.deepcopy(payload["inputs"])
    render.run([binding], tmp_path / "out")
    after = json.loads(binding.read_text(encoding="utf-8"))["inputs"]
    assert after == declared_before


@pytest.mark.parametrize("field", ["blas_threads", "age_alpha"])
def test_boolean_is_not_an_integer_protocol_parameter(field):
    data = native()
    data["parameters"][field] = True
    with pytest.raises(ValueError, match="protocol"):
        render.collect([data])


def test_final_publication_mutation_deletes_only_new_manifest(tmp_path, monkeypatch):
    binding = binding_file(tmp_path)
    writer = render.write_experiment_manifest

    def mutated(*args, **kwargs):
        result = writer(*args, **kwargs)
        binding.write_text(binding.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return result

    monkeypatch.setattr(render, "write_experiment_manifest", mutated)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="after publication"):
        render.run([binding], out)
    assert not (out / "presentation.manifest.json").exists()
    assert binding.exists()


def test_binding_mutation_during_verification_rejected(tmp_path, monkeypatch):
    binding = binding_file(tmp_path)
    verifier = render.verified

    def mutated(path, experiment):
        value = verifier(path, experiment)
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return value

    monkeypatch.setattr(render, "verified", mutated)
    with pytest.raises(RuntimeError, match="during native verification"):
        render.run([binding], tmp_path / "out")


def test_exact_pair_audit_must_cover_rendered_cells(tmp_path, monkeypatch):
    binding = binding_file(tmp_path)
    monkeypatch.setattr(render, "audit_shared_pairs", lambda paths: (
        dict(exact_pair_arrays_equal=True, n_images=650, n_pairs=5308, checked_cells=[]), set()))
    with pytest.raises(ValueError, match="cover every rendered"):
        render.run([binding], tmp_path / "out")


@pytest.mark.parametrize("error", [OSError("write failed"), KeyboardInterrupt()])
def test_writer_failure_after_marker_creation_removes_marker(tmp_path, monkeypatch, error):
    binding = binding_file(tmp_path)
    writer = render.write_experiment_manifest

    def failing(*args, **kwargs):
        writer(*args, **kwargs)
        raise error

    monkeypatch.setattr(render, "write_experiment_manifest", failing)
    out = tmp_path / "out"
    with pytest.raises(type(error)):
        render.run([binding], out)
    assert not (out / "presentation.manifest.json").exists()
    assert binding.exists() and (out / "summary.json").exists()


def test_input_disappearance_after_publication_removes_marker(tmp_path, monkeypatch):
    binding = binding_file(tmp_path)
    writer = render.write_experiment_manifest

    def disappearing(*args, **kwargs):
        writer(*args, **kwargs)
        binding.unlink()

    monkeypatch.setattr(render, "write_experiment_manifest", disappearing)
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        render.run([binding], out)
    assert not (out / "presentation.manifest.json").exists()


def test_unreadable_published_manifest_removes_marker(tmp_path, monkeypatch):
    binding = binding_file(tmp_path)
    writer = render.write_experiment_manifest

    def broken(target, **kwargs):
        writer(target, **kwargs)
        target.write_text("{broken-json", encoding="utf-8")

    monkeypatch.setattr(render, "write_experiment_manifest", broken)
    out = tmp_path / "out"
    with pytest.raises(json.JSONDecodeError):
        render.run([binding], out)
    assert not (out / "presentation.manifest.json").exists()
