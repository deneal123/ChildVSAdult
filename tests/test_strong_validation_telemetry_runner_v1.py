"""Synthetic contracts and static ownership; no native model/training execution."""

import ast
import copy
from pathlib import Path

import pytest

from scripts import run_strong_validation_telemetry_cuda_v1 as runner


def records():
    values = [
        dict(
            epoch=i,
            validation_loss=0.25,
            validation_numerator=1.0,
            validation_denominator=4.0,
            validation_pairs=4,
            validation_auc=0.75,
        )
        for i in range(1, 9)
    ]
    history = [dict(epoch=i, validation_auc=0.75) for i in range(1, 9)]
    return values, history


def test_complete_records():
    value, history = records()
    before = copy.deepcopy(history)
    runner.validation_contract(value, history, 4)
    assert history == before


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "epoch",
        "boolean_epoch",
        "count",
        "boolean_count",
        "mass",
        "ratio",
        "auc",
        "nan",
        "boolean_metric",
        "old_epoch",
    ],
)
def test_bad_records_refused(change):
    value, history = records()
    if change == "missing":
        value.pop()
    elif change == "extra":
        value[0]["extra"] = 1
    elif change == "epoch":
        value[0]["epoch"] = 2
    elif change == "boolean_epoch":
        value[0]["epoch"] = True
    elif change == "count":
        value[0]["validation_pairs"] = 3
    elif change == "boolean_count":
        value[0]["validation_pairs"] = True
    elif change == "mass":
        value[0]["validation_denominator"] = 5.0
    elif change == "ratio":
        value[0]["validation_loss"] = 0.3
    elif change == "auc":
        value[0]["validation_auc"] = 0.8
    elif change == "nan":
        value[0]["validation_numerator"] = float("nan")
    elif change == "boolean_metric":
        value[0]["validation_loss"] = True
    else:
        history[0]["epoch"] = 2
    with pytest.raises(ValueError):
        runner.validation_contract(value, history, 4)


def test_static_transitive_literal_ancestry_only(tmp_path):
    folder = tmp_path / "scripts"
    folder.mkdir()
    a, b, c = [folder / (letter + ".py") for letter in "abc"]
    a.write_text("from scripts.b import thing\n", encoding="utf-8")
    b.write_text("def f():\n    import scripts.c\n", encoding="utf-8")
    c.write_text("from scripts.a import x\nimport external_package\n", encoding="utf-8")
    assert runner.local_script_ancestry(a, tmp_path) == sorted([a, b, c])


def test_missing_local_dependency_refused(tmp_path):
    folder = tmp_path / "scripts"
    folder.mkdir()
    source = folder / "a.py"
    source.write_text("from scripts.missing import x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unavailable"):
        runner.local_script_ancestry(source, tmp_path)


def test_native_cli_cuda_only_separate_schema_and_pinned_eight_epochs():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    train = next(
        n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "finetune"
    )
    args = {
        k.arg: ast.literal_eval(k.value)
        for k in train.keywords
        if isinstance(k.value, ast.Constant)
    }
    assert args["epochs"] == args["patience"] == 8
    assert args["batch_size"] == 16 and args["gap_weight"] == 0.0
    assert args["checkpoint_selection"] == "last_epoch"
    assert args["batchnorm_policy"] == "frozen_all"
    assert '"adaface-fixed8-native-cuda-validation-telemetry-v1"' in source
    assert "no CPU fallback" in source
    assert "with installed_validation_collector(trainer, collector):" in source
    assert "validation_history=collector.records" in source
    assert "native_bitwise_equivalence_verified=False" in source
    assert "*local_script_ancestry(Path(__file__), root)" in source


def test_cli_redacts_private_exception_body(monkeypatch, capsys):
    def fail():
        raise FileNotFoundError("PRIVATE_FACE_IDENTIFIER_NEVER_ECHO")

    monkeypatch.setattr(runner, "main", fail)
    assert runner.cli() == 1
    captured = capsys.readouterr()
    assert "PRIVATE_FACE_IDENTIFIER" not in captured.out + captured.err
    assert "native-producer-refusal-no-private-details" in captured.out


def test_cli_success(monkeypatch, capsys):
    monkeypatch.setattr(runner, "main", lambda: None)
    assert runner.cli() == 0
    assert capsys.readouterr().out == ""
