"""Prepare isolated local reviewer directories; no external transfer or human labels."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT, read_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.validate_blinded_audit import OPAQUE_IMAGE, validate


def prepare(base: Path, out: Path):
    base, out = base.resolve(), out.resolve()
    if out.exists() or base == out or base in out.parents:
        raise ValueError("fresh output outside immutable source pack required")
    plans, task_counts = {}, {}
    for name in ("annotator_a", "annotator_b"):
        rows = list(read_jsonl(base / f"{name}.jsonl"))
        task_counts[name] = len(rows)
        images = sorted({relative for row in rows for relative in row.get("images", [])})
        for relative in images:
            source = (base / relative).resolve()
            if not OPAQUE_IMAGE.fullmatch(relative) or source.parent != (base / "images").resolve():
                raise ValueError("unsafe source image reference")
        plans[name] = [f"{name}.jsonl", "README.md", *images]
    source_paths = sorted({base / rel for plan in plans.values() for rel in plan})
    # Private inputs remain coordinator-only, never copied to reviewer folders.
    source_paths += [base / name for name in (
        "audit_key.jsonl", "adjudicated_gold.template.jsonl", "images.manifest.json",
        "age_baseline_predictions.jsonl", "age_baseline_predictions.manifest.json",
    )]
    source_paths += [Path(__file__).resolve(), PROJECT_ROOT / "scripts/validate_blinded_audit.py"]
    before = {str(path): file_record(path) for path in source_paths}
    validation = validate(base, 400)
    out.mkdir(parents=True)
    outputs = []
    for name, plan in plans.items():
        for relative in plan:
            source, target = base / relative, out / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            record = file_record(target)
            expected = before[str(source)]
            if record["sha256"] != expected["sha256"] or record["bytes"] != expected["bytes"]:
                raise RuntimeError("reviewer copy differs from bound source")
            outputs.append(target)
        actual = {p.relative_to(out / name).as_posix() for p in (out / name).rglob("*") if p.is_file()}
        if actual != set(plan):
            raise RuntimeError("unexpected reviewer file; blinding not established")
    for source in source_paths:
        if file_record(source) != before[str(source)]:
            raise RuntimeError("source pack changed while preparing handoff")
    metrics = dict(handoff_complete=True, human_annotation_complete=False,
                   external_transfer_performed=False, reviewers=2,
                   task_counts=task_counts,
                   reviewer_files={name: len(plan) for name, plan in plans.items()},
                   validation=validation)
    summary = out / "coordinator-summary.json"
    summary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    outputs.append(summary)
    write_experiment_manifest(
        out / "coordinator.manifest.json", experiment="isolated-blinded-human-audit-handoff",
        parameters=dict(source_pack=str(base), distribution="only one reviewer directory per annotator",
                        safety="private biometrics/captions; not public or approved for external release",
                        excluded="key, predictions, adjudication template, other reviewer answers, coordinator manifest"),
        metrics=metrics, inputs=source_paths, outputs=outputs,
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "validation"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=PROJECT_ROOT / "data/interim/human_audit")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.base, args.out)


if __name__ == "__main__":
    main()
