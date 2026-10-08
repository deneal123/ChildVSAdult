import copy
import json

import pytest

from age_gap.common.manifest import file_record
from scripts.validate_annotation_returns_v1 import (
    check_template_binding,
    read_rows,
    validate_return,
)


def files(tmp_path, *, age=False):
    template = dict(task_id="synthetic", audit_type="age_extraction" if age else "cross_split_group_candidate",
                    images=["images/synthetic.png"], allowed=["same_identity", "different_identity", "uncertain"],
                    response=None, notes="", target_image_index=0)
    answer = copy.deepcopy(template)
    answer["response"] = 17 if age else "same_identity"
    first, second = tmp_path / "template.jsonl", tmp_path / "answer.jsonl"
    first.write_text(json.dumps(template), encoding="utf-8")
    second.write_text(json.dumps(answer), encoding="utf-8")
    return first, second, answer


def test_complete_separate_copy_is_integrity_not_human_proof(tmp_path):
    template, answers, row = files(tmp_path)
    row["notes"] = "synthetic fixture, not a human response"
    answers.write_text(json.dumps(row), encoding="utf-8")
    result = validate_return(template, answers)
    assert result["response_integrity_verified"]
    assert not result["human_authorship_verified"] and not result["annotator_independence_verified"]


@pytest.mark.parametrize("field,value", [("images", []), ("task_id", "changed"),
                                       ("allowed", ["fake"]), ("audit_type", "changed"),
                                       ("notes", 123), ("response", None), ("response", "invalid"),
                                       ("target_image_index", False), ("target_image_index", 0.0)])
def test_changed_or_invalid_answers_refused(tmp_path, field, value):
    template, answers, row = files(tmp_path)
    row[field] = value
    answers.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_return(template, answers)


@pytest.mark.parametrize("value", [True, 12.0, "12", -1, 121, None])
def test_ages_are_strict_integers(tmp_path, value):
    template, answers, row = files(tmp_path, age=True)
    row["response"] = value
    answers.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_return(template, answers)


def test_same_path_and_machine_key_extra_fields_refused(tmp_path):
    template, answers, row = files(tmp_path)
    with pytest.raises(ValueError, match="separate copy"):
        validate_return(template, template)
    row["auto_label"] = "same_identity"
    answers.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        validate_return(template, answers)


def test_duplicate_json_keys_refused(tmp_path):
    path = tmp_path / "malformed.jsonl"
    path.write_text('{"response":"same_identity","response":"different_identity"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON"):
        read_rows(path)


def test_missing_task_refused(tmp_path):
    template, answers, _ = files(tmp_path)
    answers.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="coverage"):
        validate_return(template, answers)


def test_duplicate_template_task_ids_refused(tmp_path):
    template, answers, _ = files(tmp_path)
    template.write_text((template.read_text() + "\n") * 2, encoding="utf-8")
    answers.write_text((answers.read_text() + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        validate_return(template, answers)


def bound_fixture(tmp_path):
    template, _, _ = files(tmp_path)
    payload = dict(schema_version=1, experiment="isolated-blinded-human-audit-handoff",
                   outputs=[file_record(template)])
    manifest = tmp_path / "pack.manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, template, payload


def test_exact_template_output_is_not_full_pack_proof(tmp_path):
    manifest, template, _ = bound_fixture(tmp_path)
    result = check_template_binding(manifest, template)
    assert result == dict(template_output_binding_verified=True, entire_pack_verified=False)
    template.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="output binding"):
        check_template_binding(manifest, template)


@pytest.mark.parametrize("kind", ["input_only", "duplicate", "bad_record", "wrong_experiment", "wrong_schema"])
def test_unbound_templates_refused(tmp_path, kind):
    manifest, template, payload = bound_fixture(tmp_path)
    if kind == "input_only":
        payload["inputs"] = payload.pop("outputs")
    elif kind == "duplicate":
        payload["outputs"] *= 2
    elif kind == "bad_record":
        payload["outputs"][0]["extra"] = "not-native"
    elif kind == "wrong_experiment":
        payload["experiment"] = "fake-human-gold"
    else:
        payload["schema_version"] = 99
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        check_template_binding(manifest, template)
