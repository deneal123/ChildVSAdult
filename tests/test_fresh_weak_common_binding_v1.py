"""Generated metadata/vectors only; no native weak output or neural execution."""

import copy
from pathlib import Path

import numpy as np
import pytest

from scripts import fresh_weak_common_binding_v1 as helper


def native():
    return dict(
        experiment=helper.WEAK_EXPERIMENT,
        parameters=dict(
            native_experiment="facenet-fixed8-validation-loss-v2",
            device="cuda",
            image_inference=True,
            evaluator_version="weak-fixed8-evaluator-v3-root-e421",
            protocol_seed=42,
            tolerance=2,
            n_boot=2000,
            bootstrap_seed=0,
            actual_cells=list(helper.WEAK_KEYS[1:]),
        ),
        metrics=dict(
            execution_complete=True,
            publication_ready=False,
            seeds=[1, 2, 42],
            actual_cells=list(helper.WEAK_KEYS[1:]),
            n_crops=650,
            bindings=["native1", "native2", "native42"],
        ),
    )


def test_one_canonical_strong_cell():
    value = dict(
        experiment="adaface-fixed8-cuda-full-fgnet-roc-v2",
        parameters=dict(actual_cells=["random_head_lr1e-06_s42"], protocol_seed=42, tolerance=2),
        metrics=dict(execution_complete=True, publication_ready=False),
    )
    assert helper.strong_cell(value) == "random_head_lr1e-06_s42"


@pytest.mark.parametrize("change", ["escape", "multiple", "wrong_lr", "boolean_seed", "unfinished"])
def test_bad_strong_source_or_cell_refused(change):
    value = dict(
        experiment="adaface-fixed8-cuda-full-fgnet-roc-v2",
        parameters=dict(actual_cells=["random_head_lr1e-06_s42"], protocol_seed=42, tolerance=2),
        metrics=dict(execution_complete=True, publication_ready=False),
    )
    if change == "escape":
        value["parameters"]["actual_cells"] = ["../../unsafe"]
    elif change == "multiple":
        value["parameters"]["actual_cells"] *= 2
    elif change == "wrong_lr":
        value["parameters"]["actual_cells"] = ["random_head_lr3e-05_s42"]
    elif change == "boolean_seed":
        value["parameters"]["protocol_seed"] = True
    else:
        value["metrics"]["execution_complete"] = False
    with pytest.raises(ValueError):
        helper.strong_cell(value)


def test_mapping_by_key_not_old_order_or_old_cache_name():
    files = helper.weak_files(native(), Path("fresh/summary.manifest.json"))
    assert list(files) == ["frozen", "tuned_s1", "tuned_s2", "tuned_s42"]
    assert files["tuned_s42"] == Path("fresh/private/embeddings_tuned_s42.npz")


@pytest.mark.parametrize(
    "bindings", ["abc", [1, 2, 42], ["", "b", "c"], ["native1", "./native1", "native2"]]
)
def test_binding_types_and_aliases_refused(bindings):
    value = native()
    value["metrics"]["bindings"] = bindings
    with pytest.raises(ValueError, match="unique upstream"):
        helper.weak_files(value, Path("fresh/summary.manifest.json"))


@pytest.mark.parametrize(
    "change",
    [
        "legacy",
        "partial",
        "old_order",
        "boolean_seed",
        "cells",
        "boolean_crops",
        "bootstrap",
        "complete",
        "publication",
        "bindings",
    ],
)
def test_wrong_lineage_or_seed_family_refused(change):
    value = native()
    if change == "legacy":
        value["experiment"] = "fgnet-endpoint-age-matched-error-breakdown"
    elif change == "partial":
        value["metrics"]["seeds"] = [42]
    elif change == "old_order":
        value["metrics"]["seeds"] = [42, 1, 2]
    elif change == "boolean_seed":
        value["metrics"]["seeds"] = [True, 2, 42]
    elif change == "cells":
        value["parameters"]["actual_cells"] = ["random_head_lr1e-06_s42"]
    elif change == "boolean_crops":
        value["metrics"]["n_crops"] = True
    elif change == "bootstrap":
        value["parameters"]["bootstrap_seed"] = False
    elif change == "complete":
        value["metrics"]["execution_complete"] = False
    elif change == "publication":
        value["metrics"]["publication_ready"] = True
    else:
        value["metrics"]["bindings"] = ["one"] * 3
    with pytest.raises(ValueError):
        helper.weak_files(value, Path("fresh/summary.manifest.json"))


def pairs():
    people = np.arange(650) % 82
    ages = np.arange(650) % 64
    left = np.arange(5308) % 650
    right = (left + np.where(np.arange(5308) % 2, 1, 82)) % 650
    source = dict(
        left=left,
        right=right,
        labels=(people[left] == people[right]).astype(int),
        subject_a=people[left],
        subject_b=people[right],
        source_gap=np.abs(ages[left] - ages[right]),
    )
    return source, people, ages


def test_exact_pairs_and_unsigned_chronological_age():
    source, people, ages = pairs()
    helper.exact_pairs(source, copy.deepcopy(source), people, ages)
    helper.exact_pairs(source, copy.deepcopy(source), people, ages.astype(np.uint64))


@pytest.mark.parametrize(
    "change", ["float_index", "negative", "metadata", "label", "gap", "order", "coverage"]
)
def test_pair_type_linkage_or_order_refused(change):
    source, people, ages = pairs()
    weak = copy.deepcopy(source)
    if change == "float_index":
        weak["left"] = weak["left"].astype(float)
    elif change == "negative":
        weak["right"][0] = -1
    elif change == "metadata":
        weak["subject_a"][0] += 1
    elif change == "label":
        weak["labels"][0] = 1 - weak["labels"][0]
    elif change == "gap":
        weak["source_gap"][0] += 1
    elif change == "order":
        weak = {key: value[::-1].copy() for key, value in weak.items()}
    else:
        weak["left"] = weak["left"][:-1]
    with pytest.raises(ValueError):
        helper.exact_pairs(source, weak, people, ages)


def test_full_typed_unit_vectors(tmp_path):
    vectors = np.zeros((650, 512), dtype=np.float32)
    vectors[:, 0] = 1
    path = tmp_path / "vectors.npz"
    np.savez_compressed(path, indices=np.arange(650), embeddings=vectors)
    assert np.array_equal(helper.read_vectors(path), vectors)
    np.savez_compressed(path, indices=np.arange(650, dtype=float), embeddings=vectors)
    with pytest.raises(ValueError):
        helper.read_vectors(path)


@pytest.mark.parametrize("change", ["nan", "norm", "dimension", "old_key"])
def test_bad_vectors_refused(tmp_path, change):
    vectors = np.zeros((650, 512), dtype=np.float32)
    vectors[:, 0] = 1
    values = dict(indices=np.arange(650), embeddings=vectors)
    if change == "nan":
        vectors[0, 0] = np.nan
    elif change == "norm":
        vectors[0, 0] = 2
    elif change == "dimension":
        values["embeddings"] = vectors[:, :511]
    else:
        values["key"] = np.array("legacy")
    path = tmp_path / "vectors.npz"
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError):
        helper.read_vectors(path)


def test_literal_transitive_imports_and_cycles(tmp_path):
    folder = tmp_path / "scripts"
    folder.mkdir()
    a, b = folder / "a.py", folder / "b.py"
    a.write_text("from scripts.b import item\n", encoding="utf-8")
    b.write_text("def f():\n    import scripts.a\nimport external_package\n", encoding="utf-8")
    assert helper.script_ancestry(a, tmp_path) == sorted([a, b])


def test_missing_literal_import_refused(tmp_path):
    folder = tmp_path / "scripts"
    folder.mkdir()
    source = folder / "a.py"
    source.write_text("from scripts.missing import item\n", encoding="utf-8")
    with pytest.raises(ValueError):
        helper.script_ancestry(source, tmp_path)
