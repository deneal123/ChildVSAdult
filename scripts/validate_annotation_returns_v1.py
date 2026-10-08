"""Integrity of returned annotation files, not proof of human identity or independence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest

PACK_EXPERIMENTS = {
    "isolated-blinded-human-audit-handoff", "blinded-human-supervision-audit-pack",
    "blinded-train-benchmark-overlap-pack", "blinded-cross-split-missed-merge-candidate-pack",
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def read_rows(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line, object_pairs_hook=_unique_object)
            if not isinstance(row, dict):
                raise ValueError("JSONL rows must be objects")
            rows.append(row)
    return rows


def check_template_binding(pack_path, template_path):
    """Verify one output binding, not every image or producer authenticity."""
    pack_path, template_path = Path(pack_path).resolve(), Path(template_path).resolve()
    before = [file_record(pack_path), file_record(template_path)]
    pack = json.loads(pack_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if pack.get("schema_version") != 1 or pack.get("experiment") not in PACK_EXPERIMENTS:
        raise ValueError("supported native annotation-pack manifest required")
    matching = []
    for record in pack.get("outputs", []):
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise ValueError("exact native output records required")
        path = Path(record["path"])
        path = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        if path == template_path:
            matching.append(record)
    if matching != [before[1]]:
        raise ValueError("template must match exactly one native output binding")
    if [file_record(pack_path), file_record(template_path)] != before:
        raise RuntimeError("template binding changed during validation")
    return dict(template_output_binding_verified=True, entire_pack_verified=False)


def validate_return(template_path, answers_path):
    template_path, answers_path = Path(template_path).resolve(), Path(answers_path).resolve()
    if template_path == answers_path:
        raise ValueError("answers must be a separate copy, not the immutable template")
    before = [file_record(template_path), file_record(answers_path)]
    templates, answers = read_rows(template_path), read_rows(answers_path)
    if not templates or len(templates) != len(answers):
        raise ValueError("complete nonempty task coverage required")
    seen, uncertain = set(), 0
    for template, answer in zip(templates, answers, strict=True):
        if template.get("response") is not None:
            raise ValueError("template must remain blank")
        task_id = template.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise ValueError("unique nonempty template task IDs required")
        seen.add(task_id)
        immutable_template = {key: value for key, value in template.items() if key not in {"response", "notes"}}
        immutable_answer = {key: value for key, value in answer.items() if key not in {"response", "notes"}}
        if set(template) != set(answer) or json.dumps(
            immutable_template, sort_keys=True, allow_nan=False,
        ) != json.dumps(immutable_answer, sort_keys=True, allow_nan=False):
            raise ValueError("immutable task fields or row order changed")
        if "notes" in answer and not isinstance(answer["notes"], str):
            raise ValueError("notes must be text")
        value = answer.get("response")
        if template.get("audit_type") == "age_extraction":
            if value != "uncertain" and (type(value) is not int or not 0 <= value <= 120):
                raise ValueError("integer age 0..120 or uncertain required")
        else:
            allowed = template.get("allowed")
            if not isinstance(allowed, list) or not allowed or not isinstance(value, str) or value not in allowed:
                raise ValueError("non-null response from original allowed labels required")
        uncertain += value == "uncertain"
    if [file_record(template_path), file_record(answers_path)] != before:
        raise RuntimeError("annotation inputs changed during validation")
    return dict(tasks=len(answers), uncertain=uncertain, response_integrity_verified=True,
                human_authorship_verified=False, annotator_independence_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--pack-manifest", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh receipt directory required")
    inputs = [args.pack_manifest, args.template, args.answers, Path(__file__).resolve(),
              PROJECT_ROOT / "src/age_gap/common/io.py", PROJECT_ROOT / "src/age_gap/common/manifest.py"]
    before = [file_record(path) for path in inputs]
    binding = check_template_binding(args.pack_manifest, args.template)
    metrics = validate_return(args.template, args.answers)
    metrics.update(binding)
    if [file_record(path) for path in inputs] != before:
        raise RuntimeError("receipt inputs changed")
    args.out.mkdir(parents=True)
    summary = args.out / "summary.json"
    summary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(args.out / "summary.manifest.json", experiment="annotation-return-integrity",
                              parameters=dict(edits_allowed=["response", "notes"], row_order="immutable",
                                              scope="returned-file integrity; not human validation"),
                              metrics=metrics, inputs=inputs, outputs=[summary])
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
